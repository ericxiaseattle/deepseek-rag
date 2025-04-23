#!/usr/bin/env python3
import argparse
import subprocess
import re
import csv
import os
import psutil
import numpy as np
import pandas as pd
import random
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from evaluate import load as load_metric


def parse_llama_output(output: str):
    metrics = {
        "load_time": None,
        "prompt_eval_time": None,
        "eval_time": None,
        "total_time": None,
        "sampling_time": None,
    }
    for line in output.splitlines():
        if "load time" in line:
            match = re.search(r"load time =\s+([\d.]+)", line)
            if match:
                metrics["load_time"] = float(match.group(1))
        elif "prompt eval time" in line:
            match = re.search(r"prompt eval time =\s+([\d.]+)", line)
            if match:
                metrics["prompt_eval_time"] = float(match.group(1))
        elif "eval time" in line and "prompt eval time" not in line:
            match = re.search(r"eval time =\s+([\d.]+)", line)
            if match:
                metrics["eval_time"] = float(match.group(1))
        elif "total time" in line:
            match = re.search(r"total time =\s+([\d.]+)", line)
            if match:
                metrics["total_time"] = float(match.group(1))
        elif "sampling time" in line:
            match = re.search(r"sampling time =\s+([\d.]+)", line)
            if match:
                metrics["sampling_time"] = float(match.group(1))
    return metrics


def extract_generated_answer(output: str) -> str:
    lines = output.strip().splitlines()
    answer_lines = []
    found_perf = False

    for line in reversed(lines):
        if line.startswith("llama_perf_"):
            found_perf = True
            continue
        if found_perf:
            if line.strip() == "":
                if answer_lines:
                    break
                else:
                    continue
            answer_lines.append(line)

    if not answer_lines:
        return "[Could not extract answer]"

    answer_lines.reverse()
    answer = "\n".join(answer_lines).strip()
    if answer.startswith("A:"):
        answer = answer[2:].strip()
    return answer


def run_llama_cli(prompt: str, model_path: str, n_gpu_layers: int) -> tuple[str, dict, str]:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process(os.getpid())
    peak_rss = 0
    peak_cpu = 0.0
    peak_vram = 0

    proc = subprocess.Popen(
        [
            "/home/bubbles/github/llama.cpp/./llama-cli",
            "--model", model_path,
            "--cache-type-k", "q4_0",
            "--threads", "12",
            "--no-conversation",
            "--prio", "2",
            "--n-gpu-layers", str(n_gpu_layers),
            "--temp", "0.6",
            "--ctx-size", "8192",
            "--seed", "3407",
            "--prompt", prompt
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    output = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode("utf-8", errors="replace")
        print(decoded_line, end="")
        output += decoded_line
        peak_rss = max(peak_rss, process.memory_info().rss)
        peak_cpu = max(peak_cpu, process.cpu_percent(interval=0.05))
        peak_vram = max(peak_vram, nvmlDeviceGetMemoryInfo(handle).used)

    proc.wait()
    llama_perf = parse_llama_output(output)
    answer = extract_generated_answer(output)
    return output, {
        "peak_cpu": peak_cpu,
        "peak_mem": peak_rss / 1024**2,
        "peak_vram": peak_vram / 1024**2,
        **llama_perf
    }, answer


def load_retriever():
    class CustomMedEmbed(Embeddings):
        def __init__(self):
            self.model = SentenceTransformer("abhinand/MedEmbed-small-v0.1")
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        def embed_query(self, text: str) -> list[float]:
            return self.model.encode(text, convert_to_numpy=True).tolist()

    embeddings = CustomMedEmbed()
    faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def build_prompt(context_docs, question):
    instruction = (
        "You are a biomedical research assistant. "
        "Answer the question based only on the provided context. "
        "Be concise, and if the answer is not in the context, say so clearly.\n\n"
    )
    context = "\n".join([doc.page_content.strip() for doc in context_docs])
    return f"<｜User｜>{instruction}{context}\n\nQ: {question}\nA:<｜Assistant｜>"


def compute_retrieval_metrics(retrieved_docs, gold_ids, k_values=[3, 5]):
    retrieved_ids = [int(doc.metadata.get("source_id", -1)) for doc in retrieved_docs]
    metrics = {}
    for k in k_values:
        top_k = retrieved_ids[:k]
        relevance = [1 if doc_id in gold_ids else 0 for doc_id in top_k]
        ideal = sorted(relevance, reverse=True)
        hits = sum(relevance)
        recall = hits / len(gold_ids) if gold_ids else 0.0
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = ndcg
    return metrics


def evaluate_generation(prediction: str, reference: str, retrieved_docs, gold_passage_ids):
    rouge = load_metric("rouge")
    bertscore = load_metric("bertscore")
    rouge_score = rouge.compute(predictions=[prediction], references=[reference])
    bert_score = bertscore.compute(predictions=[prediction], references=[reference], lang="en")
    retrieval_metrics = compute_retrieval_metrics(retrieved_docs, gold_passage_ids)
    return {
        "rougeL": rouge_score["rougeL"].mid.fmeasure if hasattr(rouge_score["rougeL"], "mid") else rouge_score["rougeL"],
        "bertscore_f1": sum(bert_score["f1"]) / len(bert_score["f1"]),
        **retrieval_metrics
    }


def log_to_tsv(row: dict, results_tsv: str):
    write_header = not os.path.exists(results_tsv)
    with open(results_tsv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
            delimiter='\t',
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            escapechar='\\'
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation with specified quantization and output TSV file.")
    parser.add_argument("-f", "--file", required=True, help="Path to output TSV file")
    parser.add_argument("-q", "--quant", required=True, choices=["1", "2", "3"], help="Quantization level: 1=1.73-bit,2=2.22-bit,3=2.51-bit")
    parser.add_argument("-L", "--n-gpu-layers", type=int, default=4, help="Number of GPU layers to offload")
    parser.add_argument("-n", "--n-samples", type=int, default=50, help="Number of samples to process")
    args = parser.parse_args()

    quant_map = {
        "1": "DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf",
        "2": "DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf",
        "3": "DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf"
    }
    model_path = quant_map[args.quant]
    results_tsv = args.file
    n_gpu_layers = args.n_gpu_layers
    n_samples = args.n_samples

    random.seed(42)
    dataset = load_dataset("enelpol/rag-mini-bioasq", name="question-answer-passages", split="test")
    retriever = load_retriever()

    sample_indices = random.sample(range(len(dataset)), n_samples)
    existing_indices = set()
    if os.path.exists(results_tsv):
        df = pd.read_csv(results_tsv, sep="\t")
        if "sample_index" in df.columns:
            existing_indices = set(df["sample_index"].tolist())

    for idx in sample_indices:
        if idx in existing_indices:
            print(f"[SKIPPING] Sample {idx} already exists in {results_tsv}")
            continue

        sample = dataset[idx]
        question = sample["question"]
        reference_answer = sample["answer"]
        gold_passage_ids = sample.get("relevant_passage_ids", [])

        top_docs = retriever.invoke(question)
        prompt = build_prompt(top_docs, question)

        print(f"\n=== SAMPLE {idx} ===")
        print("[PROMPT]\n" + prompt)
        print("\n[LLAMA RESPONSE]")
        raw_output, perf, generated_answer = run_llama_cli(prompt, model_path, n_gpu_layers)

        print("\n[EVALUATION]")
        metrics = evaluate_generation(generated_answer, reference_answer, top_docs, gold_passage_ids)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        row = {
            "sample_index": idx,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            **metrics,
            **perf
        }
        log_to_tsv(row, results_tsv)


if __name__ == "__main__":
    main()

