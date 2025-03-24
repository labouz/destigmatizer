"""
Filter functions for text processing pipelines.
"""
from typing import Dict, Any, Optional

def classify_drug_filter(text: str, client: Any, model: Optional[str] = None, 
                       pipeline_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Classify if text contains drug-related content.
    
    Args:
        text: Input text
        client: LLM client
        model: Model to use
        pipeline_data: Current pipeline data
        
    Returns:
        dict: Result with classification metadata
    """
    from .classifiers import DrugClassifier
    
    classifier = DrugClassifier(client)
    result = classifier.classify(text, model=model)
    
    return {
        "text": text,
        "metadata": {
            "is_drug_related": result.lower() == 'd'
        }
    }

def classify_stigma_filter(text: str, client: Any, model: Optional[str] = None, 
                        pipeline_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Classify if text contains stigmatizing language.
    
    Args:
        text: Input text
        client: LLM client
        model: Model to use
        pipeline_data: Current pipeline data
        
    Returns:
        dict: Result with classification metadata
    """
    from .classifiers import StigmaClassifier
    
    classifier = StigmaClassifier(client)
    result = classifier.classify(text, model=model)
    
    is_stigmatizing = result.startswith('s')
    explanation = result.split(', ', 1)[1] if is_stigmatizing and ', ' in result else result
    
    return {
        "text": text,
        "metadata": {
            "is_stigmatizing": is_stigmatizing,
            "stigma_explanation": explanation
        }
    }

def analyze_style_filter(text: str, client: Any, model: Optional[str] = None, 
                      pipeline_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Analyze text style and emotion.
    
    Args:
        text: Input text
        client: LLM client
        model: Model to use
        pipeline_data: Current pipeline data
        
    Returns:
        dict: Result with style analysis metadata
    """
    from .analyzers import StyleAnalyzer, EmotionAnalyzer, LLMBasedAnalyzer
    
    style_analyzer = StyleAnalyzer()
    emotion_analyzer = EmotionAnalyzer(client)
    analyzer = LLMBasedAnalyzer(client, emotion_analyzer, style_analyzer)
    
    result = analyzer.analyze(text, model=model)
    
    return {
        "text": text,
        "metadata": {
            "style_analysis": result
        }
    }

def rewrite_text_filter(text: str, client: Any, model: Optional[str] = None, 
                     pipeline_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Rewrite stigmatizing text to be more inclusive.
    
    Args:
        text: Input text
        client: LLM client
        model: Model to use
        pipeline_data: Current pipeline data
        
    Returns:
        dict: Result with rewritten text
    """
    from .rewriters import DestigmatizingRewriter
    
    # Skip if not stigmatizing or metadata is missing
    if not pipeline_data or not pipeline_data.get("metadata", {}).get("is_stigmatizing", False):
        return {"text": text}
    
    explanation = pipeline_data.get("metadata", {}).get("stigma_explanation", "")
    style_analysis = pipeline_data.get("metadata", {}).get("style_analysis", {})
    style_instruct = str(style_analysis)
    
    rewriter = DestigmatizingRewriter(client)
    rewritten_text = rewriter.rewrite(
        text=text,
        explanation=explanation,
        style_instruct=style_instruct,
        model=model
    )
    
    return {
        "text": rewritten_text,
        "metadata": {
            "was_rewritten": True
        }
    }