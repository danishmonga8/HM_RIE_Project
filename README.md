# 🌧️ Hydro-Meteorological AI Research Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

This repository contains a Retrieval-Augmented Generation (RAG) based AI assistant specifically designed for hydro-meteorological research, focusing on moisture-driven landslides, spatial interpolation methods, and complex terrain modeling. 

The project allows users to query a large corpus of scientific literature and receive highly technical, citation-backed answers without AI hallucination.

> **Security note:** The included FAISS document store uses Python pickle
> deserialization. Run it only from a trusted, unmodified checkout and never
> replace the index with files from an untrusted source. See the security notes
> below before starting the application.

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

## Security and responsible use

- Treat `faiss_index/index.pkl` as executable content because loading a pickle
  can run code. Use only the version committed in a trusted checkout.
- Enter API keys only in the local Streamlit password field. Never commit keys,
  place them in screenshots, or share them in issues.
- Generated answers are research assistance, not a substitute for checking the
  cited paper. Verify quotations, values, and methodological claims at source.
- Confirm that you have the necessary rights to store and redistribute any
  literature added to the index.
