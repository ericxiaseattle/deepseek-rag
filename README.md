# Quantized DeepSeek-R1 for Low-Resource RAG

This project implements a simple, low-resource **Retrieval-Augmented Generation (RAG)** pipeline for the medical domain, using **pre-quantized DeepSeek-R1 language models**, **FAISS vector indexing**, and **MedEmbed sentence embeddings**.  
It includes benchmarking scripts to evaluate generation quality (ROUGE-L, BERTScore-F1) and system resource utilization (VRAM, CPU), demonstrating efficiency gains in low-resource environments.

## Features
- ⚡️ **Lightweight RAG loop:** Built with [LangChain](https://docs.langchain.com/) and [FAISS](https://faiss.ai/), supporting fast retrieval.
- 🧠 **Medical domain support:** Uses `MedEmbed` embeddings for better medical text representation.
- 🏎️ **Efficiency improvements:** Achieved 27% faster inference and 20% lower VRAM usage with negligible (<0.2%) drop in ROUGE-L/BERTScore-F1 (Benchmarked on 1.73b, 2.22b, 2.51b quant levels)
- 📊 **Benchmarking:** Includes scripts to measure inference quality and system resource consumption.

## Project structure
```
.
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── src
    ├── build_faiss.py  # Script for building FAISS index with MedEmbed embeddings
    └── rag.py          # Simple RAG loop + benchmarking for speed, VRAM, and generation quality
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/deepseek-rag.git
   cd deepseek-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download the quantized DeepSeek-R1 models using `setup.py`:
   ```bash
   python setup.py
   ```

## Usage

### 1️⃣ Build FAISS index
```bash
python src/build_faiss.py
```

This script:
- Loads a medical dataset (e.g., from Hugging Face)
- Splits documents using `RecursiveCharacterTextSplitter`
- Embeds text using MedEmbed
- Saves FAISS index for retrieval

### 2️⃣ Run RAG + Benchmarking
```bash
python src/rag.py
```

This script:
- Loads FAISS index
- Runs inference with quantized DeepSeek-R1 model
- Benchmarks ROUGE-L, BERTScore-F1
- Measures VRAM, CPU usage and inference latency

## Benchmark results
📝 **Summary:**
- **27% faster inference**
- **20% reduction in VRAM usage**
- **<0.2% drop in generation quality (ROUGE-L, BERTScore-F1)**

## Notes
- The dynamically-quantized DeepSeek-R1 models are downloaded from [`unsloth/DeepSeek-R1-GGUF`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF).
- This project does **not** include model quantization code itself. It focuses on applying pre-quantized models in a lightweight RAG pipeline.

## License
This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
- [DeepSeek](https://huggingface.co/deepseek-ai)
- [Unsloth Group](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [MedEmbed](https://huggingface.co/abhinand/MedEmbed-small-v0.1)
