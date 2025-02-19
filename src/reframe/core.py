from typing import Optional
from openai import OpenAI
from .classifiers import drug_classifier, stigma_classifier, rewriter, get_style

class Reframe:
    """Main class for the reframe package."""
    # we need one single script to keep all functions
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key) # we might need to generalize this for other clients

    def classify_drug_content(self, text: str) -> str:
        """Classify if text contains drug-related content."""
        return drug_classifier.classify_if_drug(text, self.client)
        
    def analyze_stigma(self, text: str) -> str:
        """Analyze text for stigmatizing language. Returns the label and explanation."""
        return stigma_classifier.classify_if_stigma(text, self.client)
    
    def retrive_style_instruction(self, text: str) -> dict:
        """Retrieve style instructions for rewrit_text."""
        return get_style.analyze_text_llm(text, self.client)
    
    def rewrite_text(self, text: str, explanation: str, 
                    style: Optional[dict] = None) -> str:
        """Rewrite text to remove stigmatizing language."""
        return rewriter.rewrite_to_destigma(text, explanation, style, self.client)