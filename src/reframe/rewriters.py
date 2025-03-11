"""Text rewriters for destigmatizing content."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .utils import get_model_mapping

from .clients import LLMClient, detect_client_type


class TextRewriter(ABC):
    """Abstract base class for text rewriters."""
    
    @abstractmethod
    def rewrite(self, text: str, **kwargs) -> str:
        """Rewrite the provided text.
        
        Args:
            text: Text to rewrite
            **kwargs: Additional parameters for rewriting
            
        Returns:
            str: Rewritten text
        """
        pass


class DestigmatizingRewriter(TextRewriter):
    """Rewriter that removes stigmatizing language."""
    
    def __init__(self, client: Any):
        """Initialize with an LLM client.
        
        Args:
            client: LLM client instance
        """
        self.client = client
        self.retry_wait_time = 5  # seconds between retries
        
    def _parse_explanation(self, explanation: str) -> Dict[str, str]:
        """Parse stigma explanation into components.
        
        Args:
            explanation: Explanation text from stigma classifier
            
        Returns:
            dict: Extracted components
        """
        explanation_lower = explanation.lower()
        components = {}
        
        # Extract labeling
        if "labeling:" in explanation_lower:
            labeling_start = explanation_lower.find("labeling:") + len("labeling:")
            labeling_end = explanation_lower.find(",", labeling_start)
            if labeling_end == -1:  # If no comma found, take the rest of the string
                labeling_end = len(explanation_lower)
            components["labeling"] = explanation_lower[labeling_start:labeling_end].strip()
        
        # Extract stereotyping
        if "stereotyping:" in explanation_lower:
            stereo_start = explanation_lower.find("stereotyping:") + len("stereotyping:")
            stereo_end = explanation_lower.find(",", stereo_start)
            if stereo_end == -1:
                stereo_end = len(explanation_lower)
            components["stereotyping"] = explanation_lower[stereo_start:stereo_end].strip()
        
        # Extract separation
        if "separation:" in explanation_lower:
            sep_start = explanation_lower.find("separation:") + len("separation:")
            sep_end = explanation_lower.find(",", sep_start)
            if sep_end == -1:
                sep_end = len(explanation_lower)
            components["separation"] = explanation_lower[sep_start:sep_end].strip()
        
        # Extract discrimination
        if "discrimination:" in explanation_lower:
            disc_start = explanation_lower.find("discrimination:") + len("discrimination:")
            disc_end = explanation_lower.find(",", disc_start)
            if disc_end == -1:
                disc_end = len(explanation_lower)
            components["discrimination"] = explanation_lower[disc_start:disc_end].strip()
        
        return components
    
    def rewrite(self, text: str, explanation: str, style_instruct: str, step: int = 1, 
               model: Optional[str] = None, retries: int = 2) -> str:
        """Rewrite text to remove stigmatizing language.
        
        Args:
            text: Text to rewrite
            explanation: Explanation of stigma from classifier
            style_instruct: Style instructions to maintain
            step: Rewriting step (1=remove labeling, 2=remove stereotyping/separation/discrimination)
            model: Model to use for rewriting
            retries: Number of retries on failure
            
        Returns:
            str: Rewritten text
        """
        # Determine client type and map model if needed
        client_type = detect_client_type(self.client)
        mapped_model = get_model_mapping(model, client_type)
        components = self._parse_explanation(explanation)
        
        if step == 1:
            instruction = "Rewrite this post to remove any and all labeling."
            definition = "Labeling includes the use of derogatory or othering language related to drug use/addiction."
            explanation_part = components.get("labeling", explanation.lower())
        else:
            instruction = "Rewrite this post to remove any and all instances of stereotyping, insinuations of separation, and/or discriminatory language."
            definition = "Stereotyping reinforces negative generalizations about people who use drugs. Separation creates a divide between people who use drugs and those who don't. Discrimination implies or suggests unfair treatment based on drug use."
            
            # Combine the non-labeling components for step 2
            component_parts = []
            if "stereotyping" in components:
                component_parts.append(f"Stereotyping: {components['stereotyping']}")
            if "separation" in components:
                component_parts.append(f"Separation: {components['separation']}")
            if "discrimination" in components:
                component_parts.append(f"Discrimination: {components['discrimination']}")
            
            explanation_part = "; ".join(component_parts) if component_parts else explanation.lower()

        prompt = f"""
        {instruction}; 
        {definition};
        Only rewrite the relevant parts of the post, do not rewrite the whole post. Do not change the meaning of the post or add any new information.
        Also, match the output to the given stylistic profile.
        Example:
        post: "My mom is an addict."; This post uses the term 'addict'; [('tone': 'negative'),('punctuation_usage': 'moderate, with . being most frequent'),('passive_voice_usage': 'none'),('sentence_length_variation': 'ranging from short (5 words) to long (5 words) with an average of 5.0 words per sentence'),('lexical_diversity': 'moderately high')]
        rewrite: "My mom has a substance use disorder."

        Do not include "Here is the rewritten post:" in your response. Just return the rewritten post. Nothing more.
        """
        ex = f"This post uses {explanation_part}"

        while retries > 0:
            try:
                rewritten = self.client.create_completion(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text + ";" + ex + ";" + style_instruct}
                    ],
                    model=mapped_model
                )
                return rewritten.lower().strip()
                
            except Exception as e:
                print(f"An error occurred: {e}. Retrying...")
                retries -= 1
                time.sleep(self.retry_wait_time)
                
        return "Error rewriting text"
