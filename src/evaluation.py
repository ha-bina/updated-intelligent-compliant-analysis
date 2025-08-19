from typing import List, Dict, Any
import pandas as pd

class RAGEvaluator:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def evaluate_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Evaluate RAG system on a set of questions."""
        results = []
        
        for question in questions:
            retrieved_docs = self.retriever.retrieve(question)
            response = self.generator.generate_response(question, retrieved_docs)
            
            results.append({
                "question": question,
                "answer": response['answer'],
                "retrieved_sources": [
                    {
                        "product": doc['source']['product'],
                        "similarity": doc['similarity'],
                        "excerpt": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    }
                    for doc in response['sources'][:2]  # Show top 2 sources
                ]
            })
        
        return results
    
    def create_evaluation_table(self, evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create evaluation table from results."""
        eval_data = []
        
        for result in evaluation_results:
            eval_data.append({
                "Question": result['question'],
                "Generated Answer": result['answer'],
                "Retrieved Sources": "\n\n".join(
                    f"Product: {src['product']}\n"
                    f"Similarity: {src['similarity']:.2f}\n"
                    f"Excerpt: {src['excerpt']}"
                    for src in result['retrieved_sources']
                ),
                "Quality Score": "",  # To be filled manually
                "Comments/Analysis": ""  # To be filled manually
            })
        
        return pd.DataFrame(eval_data)