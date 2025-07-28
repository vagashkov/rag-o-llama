# rag-o-llama
Simple RAG system built with ollama and LangChain

## Installation

### LLM
- install ollama framework:
curl -fsSL https://ollama.com/install.sh | sh
- after installation use "ollama pull <model_name>" to download models for text embedding and generation.

### Script settings and execution
- consider using virtual environment for your project (project was tested using Python 3.11 on Debian 12)
- install dependencies using pip (pip install -r requirements.txt)


### RAG
Create .env file using .env.example and check it's settings:
- EMBEDDING_MODEL: model used to convert text into embeddings
- GENERATING_MODEL: model used for answers generaition
- VECTOR_DB_STORAGE: ChromaDB storage location
- CONTEXT_TEMPLATE: session initialization prompt

### Usage:
- Use "python main.py" to launch an assistant and follow it's instructions