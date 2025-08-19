from transformers import pipeline
from config.settings import (
    LLM_MODEL_NAME,
    PROMPT_TEMPLATE,
    MAX_NEW_TOKENS,
    TEMPERATURE
)
from typing import List, Dict, Any

class ResponseGenerator:
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model=LLM_MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(
                f"Complaint ID: {doc['source']['complaint_id']}\n"
                f"Product: {doc['source']['product']}\n"
                f"Excerpt: {doc['text']}\n"
                f"Similarity Score: {doc['similarity']:.2f}\n"
            )
        return "\n".join(context_parts)
    
    def generate_response(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using LLM."""
        context = self.format_context(retrieved_docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        response = self.llm(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=self.llm.tokenizer.eos_token_id
        )
        
        return {
            "answer": response[0]['generated_text'][len(prompt):].strip(),
            "sources": retrieved_docs
        }