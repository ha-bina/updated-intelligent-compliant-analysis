import gradio as gr
from src.retrieval import Retriever
from src.generation import ResponseGenerator
from src.embedding import EmbeddingModel, VectorStore
from config.settings import VECTOR_STORE_DIR

class ComplaintAnalysisApp:
    def __init__(self):
        # Load components
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore.load_store("cfpb_complaints")
        self.retriever = Retriever(self.vector_store, self.embedding_model)
        self.generator = ResponseGenerator()
    
    def respond(self, question: str):
        """Process user question and generate response."""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question)
        
        # Generate response
        response = self.generator.generate_response(question, retrieved_docs)
        
        # Format sources for display
        sources_html = "<h3>Sources Used:</h3>"
        for doc in response['sources']:
            sources_html += f"""
            <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                <p><strong>Product:</strong> {doc['source']['product']}</p>
                <p><strong>Similarity Score:</strong> {doc['similarity']:.2f}</p>
                <p><strong>Excerpt:</strong> {doc['text']}</p>
                <p><strong>Complaint ID:</strong> {doc['source']['complaint_id']}</p>
            </div>
            """
        
        return response['answer'], sources_html
    
    def launch(self):
        """Launch Gradio interface."""
        with gr.Blocks(title="CrediTrust Complaint Analysis") as demo:
            gr.Markdown("# CrediTrust Financial Complaint Analysis")
            gr.Markdown("Ask questions about customer complaints across our financial products.")
            
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Enter your question",
                        placeholder="e.g., Why are people unhappy with BNPL?",
                        lines=3
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Analysis",
                        interactive=False,
                        lines=10
                    )
                    sources_output = gr.HTML(label="Retrieved Sources")
            
            submit_btn.click(
                fn=self.respond,
                inputs=question_input,
                outputs=[answer_output, sources_output]
            )
            
            clear_btn.click(
                fn=lambda: ["", ""],
                inputs=[],
                outputs=[answer_output, sources_output]
            )
        
        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    app = ComplaintAnalysisApp()
    app.launch()