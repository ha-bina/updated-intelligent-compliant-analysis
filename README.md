# updated intelligient copliant analysis
This project implements an AI-powered complaint analysis system for CrediTrust Financial, designed to help internal stakeholders quickly identify and understand customer pain points across multiple financial products. The system uses semantic search and large language models to provide evidence-backed answers to natural language questions about customer complaints.
# Features
Natural Language Querying: Ask plain-English questions about customer complaints

Multi-Product Analysis: Support for Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers

Semantic Search: FAISS-based vector similarity search for relevant complaint retrieval

Evidence-Based Answers: LLM-generated responses grounded in actual customer complaints

Transparent Source Display: Shows retrieved complaint excerpts for verification

User-Friendly Interface: Gradio-based web interface for non-technical users

# Project Structure
text
updated intelligent compliant analysis/
├── config/              # Configuration settings
├── data/               # Raw and processed data files
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── src/                # Source code modules
│   ├── data_processing.py    # Data loading and cleaning
│   ├── embedding.py          # Text chunking and embedding
│   ├── retrieval.py          # Semantic search implementation
│   ├── generation.py         # LLM response generation
│   ├── evaluation.py         # System evaluation utilities
│   └── app.py               # Web interface
├── vector_store/       # Persisted vector database
├── requirements.txt    # Python dependencies
└── README.md          # This file
# Installation
Clone the repository:
git clone <htttps://github.com/ha-bina/updated-intelligent-compliant-analysis.git>

# Install dependencies:

pip install -r requirements.txt
Download the CFPB complaint dataset and place it in the data/ directory.

# Usage
# Data Processing
# To preprocess the complaint data:
python src/main.py --mode process-data
# Create Vector Store
To generate embeddings and create the vector store:
python src/main.py --mode create-vector-store
# Run Application
To launch the web interface:
python src/main.py --mode run-app

# System Components
1. Data Processing
The system processes CFPB complaint data, filtering for the five relevant product categories and cleaning the complaint narratives. Key steps include:

Filtering by product category

Removing empty narratives

Text cleaning (lowercasing, special character removal, boilerplate removal)

2. Text Chunking and Embedding
Long complaint narratives are split into chunks using a recursive text splitter. Each chunk is embedded using the all-MiniLM-L6-v2 sentence transformer model, chosen for its balance of performance and efficiency.

3. Vector Store
Embeddings are stored in a FAISS index with associated metadata including: Complaint ID, Product category, Original narrative text and Chunk text

4. Retrieval and Generation

Embeds the user's question, Retrieves the top-k most relevant complaint chunks and Formats the context with retrieved excerpts


5. User Interface
The Gradio interface provides: Question input field, Response display, Source excerpts with similarity scores, Clear functionality

6. Evaluation
The system includes evaluation utilities to assess performance on representative questions. Evaluation metrics focus on: Answer relevance and accuracy,Source quality and relevance, Response usefulness for business stakeholders
# Business Impact
This system addresses CrediTrust's key challenges: Reduced Analysis Time: From days to minutes for identifying complaint trends, Empowered Non-Technical Teams: Support and compliance can get answers without data analysts, Proactive Problem Solving: Real-time insights enable proactive issue resolution.

# Technical Details
Embedding Model: sentence-transformers
Vector Store: FAISS with cosine similarity
LLM: Mistral-7B-Instruct
Chunk Size: 512 tokens with 50 token overlap
Top-k Retrieval: 5 most relevant chunks
