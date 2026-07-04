# 🌧️ Hydro-Meteorological AI Research Assistant

This repository contains a Retrieval-Augmented Generation (RAG) based AI assistant specifically designed for hydro-meteorological research, focusing on moisture-driven landslides, spatial interpolation methods, and complex terrain modeling. 

The project allows users to query a large corpus of scientific literature and receive highly technical, citation-backed answers without AI hallucination.

## 🏗️ Project Architecture

The project is divided into two main phases:

### 1. Data Processing & Knowledge Base Construction (Google Colab)
The backend processing was performed in a cloud environment (Google Colab) to handle heavy PDF parsing and embedding generation.
* **Document Ingestion:** Processed over 1300+ pages of high-impact scientific literature (PDFs) related to climate modeling and hydro-meteorology.
* **Text Chunking:** Utilized `RecursiveCharacterTextSplitter` from LangChain to break down large documents into manageable chunks (1000 characters with 200 overlap) to preserve scientific context.
* **Vector Embeddings:** Converted text chunks into dense vector representations using OpenAI's `text-embedding-3-small` model.
* **Vector Database:** Built a highly efficient FAISS (Facebook AI Similarity Search) index to store the embeddings. This database was then permanently exported for local usage.

### 2. Frontend Chat Interface (Local Streamlit App)
The frontend is a lightweight, secure web application built with Streamlit, running locally to ensure data privacy and fast iteration.
* **Local Database Loading:** The exported FAISS database (`index.faiss` and `index.pkl`) is loaded locally using `@st.cache_resource` for optimized performance.
* **Retrieval System:** Implements Maximum Marginal Relevance (MMR) search to fetch the most relevant and diverse context chunks from the scientific literature.
* **Generative Engine:** Uses OpenAI's `gpt-3.5-turbo` via LangChain's LCEL (LangChain Expression Language) pipeline.
* **Strict Prompting:** The AI is strictly instructed to act as a Hydro-Meteorological Data Scientist. It is constrained to answer *only* based on the retrieved context and must append the specific "Source Paper" citation at the end of relevant sentences.

## 🚀 How to Run Locally

### Prerequisites
Ensure you have Python installed and your OpenAI API key ready.

### Installation
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install streamlit langchain langchain-openai langchain-community faiss-cpu tiktoken