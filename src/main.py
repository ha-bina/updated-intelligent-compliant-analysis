import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from src.data_processing import load_data, filter_data, preprocess_complaints, save_processed_data
from src.embedding import TextChunker, EmbeddingModel, VectorStore
from src.app import ComplaintAnalysisApp

def process_data():
    """Process raw data and create vector store."""
    print("Loading and processing data...")
    df = load_data("cfpb_complaints.csv")
    filtered_df = filter_data(df)
    processed_df = preprocess_complaints(filtered_df)
    save_processed_data(processed_df, "processed_complaints.csv")
    return processed_df

def create_vector_store(df):
    """Create and save vector store from processed data."""
    print("Creating vector store...")
    
    # Chunk the narratives
    chunker = TextChunker()
    all_chunks = []
    metadata = []
    
    for _, row in df.iterrows():
        chunks = chunker.chunk_text(row['cleaned_narrative'])
        all_chunks.extend(chunks)
        
        for chunk in chunks:
            metadata.append({
                "complaint_id": row['Complaint ID'],
                "product": row['Product'],
                "original_narrative": row['Consumer complaint narrative'],
                "text_chunk": chunk
            })
    
    # Generate embeddings
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_batch(all_chunks)
    
    # Create and save vector store
    vector_store = VectorStore()
    vector_store.create_store(embeddings, metadata)
    vector_store.save_store("cfpb_complaints")
    
    print("Vector store created and saved.")

def run_app():
    """Run the Gradio application."""
    app = ComplaintAnalysisApp()
    app.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrediTrust Financial Complaint Analysis System")
    parser.add_argument(
        "--mode",
        choices=["process-data", "create-vector-store", "run-app"],
        required=True,
        help="Select operation mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "process-data":
        process_data()
    elif args.mode == "create-vector-store":
        df = process_data()
        create_vector_store(df)
    elif args.mode == "run-app":
        run_app()