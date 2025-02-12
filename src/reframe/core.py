from typing import Optional
from openai import OpenAI
from .classifiers import drug_classifier, stigma_classifier, rewriter

class Reframe:
    """Main class for the reframe package."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key) # we might need to generalize this for other clients

    def classify_drug_content(self, text: str) -> str:
        """Classify if text contains drug-related content."""
        return drug_classifier.classify_if_drug_post(text, self.client)
        
    def analyze_stigma(self, text: str) -> str:
        """Analyze text for stigmatizing language."""
        return stigma_classifier.determine_if_contains_stigma(text, self.client)
        
    def rewrite_text(self, text: str, explanation: str, 
                    style: Optional[dict] = None) -> str:
        """Rewrite text to remove stigmatizing language."""
        
        return rewriter.rewrite_to_destigma(text, explanation, style, self.client)