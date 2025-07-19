# LLM Question Answering Application 🤖

This project is a **Streamlit-based application** that uses **Large Language Models (LLMs)** for document-based question answering. It leverages the **LangChain** framework to load, process, and query documents, employing embeddings and vector stores for efficient information retrieval.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Use](#how-to-use)

## Features
- **Upload support for PDF, DOCX, and TXT files**.
- **Embeddings with `OllamaEmbeddings`** for document chunk representations.
- **Efficient vector search with `Chroma`** for similarity matching.
- **Multi-query retriever** to enhance the accuracy of document retrieval.
- **Streamlit UI** for an interactive question-answering experience.
- **GPU support** using `torch` for accelerated model inference.
- **Uses GPT** for:
  - Streamlit function generation
  - Debugging and optimization
  - Code snippet generation during development

## Requirements
Ensure you have the following dependencies installed:

- Python 3.8+
- Streamlit
- LangChain
- PyTorch
- Ollama (For embeddings and chat models)
- Chroma (Vector database)
- UnstructuredPDFLoader (For PDF reading)
- Docx2txtLoader (For DOCX reading)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/llm-question-answering-app.git
    cd llm-question-answering-app
    ```

2. **Set up a virtual environment** (recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## How to Use

1. **Upload a Document**:
   - Use the **sidebar** to upload a PDF, DOCX, or TXT file.
   - Adjust the **chunk size** if needed.
   - Click the **"Add Data"** button to process the file.

2. **Ask a Question**:
   - Enter your question in the text box.
   - The application will search the document and provide an answer based on the content.

3. **View History**:
   - All previous questions and answers are saved and displayed in the **history** section.

## Note:
Make sure you download model from ollama initially

