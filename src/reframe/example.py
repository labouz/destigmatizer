"""
Example usage for the Reframe package.

This module demonstrates how to use the Reframe package to:
1. Classify drug-related content
2. Identify stigmatizing language
3. Analyze text style
4. Rewrite stigmatizing content
5. Detect emotions in text

Usage:
  # Run all examples
  python -m reframe.example all
  
  # Run specific example
  python -m reframe.example classify_drug
  
  # Run with specific client type
  python -m reframe.example classify_drug --client openai
  
  # Run with specific model
  python -m reframe.example classify_drug --model gpt-4o
  
  # Run with specific client and model
  python -m reframe.example classify_drug --client openai --model gpt-4o
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any
import reframe

# Default models to use for examples
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-haiku-20240307"

def get_client(client_type: str = None, api_key: str = None, model: str = None):
    """
    Initialize and return a client based on client_type.
    
    Args:
        client_type: The type of client to initialize ("openai", "together", "claude")
        api_key: API key to use. If None, will look for environment variables
        model: Specific model to use. If None, will use the default for the client
        
    Returns:
        tuple: (client, client_type, model)
    """
    if client_type is None:
        # Try to determine from environment variables
        if os.getenv("OPENAI_API_KEY"):
            client_type = "openai"
        elif os.getenv("TOGETHER_API_KEY"):
            client_type = "together"
        elif os.getenv("ANTHROPIC_API_KEY"):
            client_type = "claude"
        else:
            raise ValueError(
                "No API keys found in environment. Please set OPENAI_API_KEY, "
                "TOGETHER_API_KEY, or ANTHROPIC_API_KEY."
            )
    
    # Get API key from environment if not provided
    if api_key is None:
        if client_type.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif client_type.lower() == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
        elif client_type.lower() == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            
    # Select appropriate model based on client_type if model not specified
    if model is None:
        if client_type.lower() == "openai":
            model = OPENAI_MODEL
        elif client_type.lower() == "together":
            model = LLAMA_MODEL
        elif client_type.lower() == "claude":
            model = CLAUDE_MODEL
    
    # Initialize client
    if client_type.lower() == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    elif client_type.lower() == "together":
        from together import Together
        client = Together(api_key=api_key)
    elif client_type.lower() == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")
        
    return client, client_type, model


def example_classify_drug(client=None, client_type=None, model=None):
    """
    Example showing how to classify drug-related content.
    """
    if client is None:
        client, client_type, model = get_client(client_type=client_type, model=model)
        
    print("\n=== Drug Classification Example ===")
    print(f"Using client: {client_type}, model: {model}")
    
    # Test posts
    posts = [
        "I'm so high right now, smoking the best weed ever",
        "I need to get clean, this addiction is ruining my life",
        "I'm feeling really down today, need someone to talk to",
        "Just got my prescription filled for my anxiety meds"
    ]
    
    for i, post in enumerate(posts):
        print(f"\nPost {i+1}: {post}")
        result = reframe.classify_if_drug(
            post,
            client=client,
            client_type=client_type,
            model=model
        )
        print(f"Classification: {result}")
    
    return "Drug classification example completed"


def example_classify_stigma(client=None, client_type=None, model=None):
    """
    Example showing how to identify stigmatizing language.
    """
    if client is None:
        client, client_type, model = get_client(client_type=client_type, model=model)
        
    print("\n=== Stigma Identification Example ===")
    print(f"Using client: {client_type}, model: {model}")
    
    # Test posts
    posts = [
        "All these junkies should be locked up, they're ruining our city",
        "People struggling with addiction need support and understanding",
        "She's just a crackhead who can't be trusted",
        "My friend is in recovery from substance use disorder and doing well"
    ]
    
    for i, post in enumerate(posts):
        print(f"\nPost {i+1}: {post}")
        result = reframe.classify_if_stigma(
            post,
            client=client,
            client_type=client_type,
            model=model
        )
        print(f"Classification: {result}")
    
    return "Stigma classification example completed"


def example_analyze_text(client=None, client_type=None, model=None):
    """
    Example showing how to analyze text style.
    """
    if client is None:
        client, client_type, model = get_client(client_type=client_type, model=model)
        
    print("\n=== Text Style Analysis Example ===")
    print(f"Using client: {client_type}, model: {model}")
    
    sample_texts = [
        "This is a test sentence. It contains multiple parts. We want to analyze its style.",
        "I'm really angry about the drug users in our neighborhood! They make everything unsafe!"
    ]
    
    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text}")
        style_result = reframe.analyze_text_llm(
            text, 
            client, 
            model=model,
            client_type=client_type
        )
        print(f"Style analysis result:")
        for key, value in style_result.items():
            print(f"  {key}: {value}")
    
    return "Text analysis example completed"


def example_rewrite_content(client=None, client_type=None, model=None):
    """
    Example showing how to rewrite stigmatizing content.
    """
    if client is None:
        client, client_type, model = get_client(client_type=client_type, model=model)
        
    print("\n=== Content Rewriting Example ===")
    print(f"Using client: {client_type}, model: {model}")
    
    # Example post with stigmatizing language
    post = "Junkies are ruining our neighborhood"
    print(f"Original text: {post}")
    
    # Get stigma classification with explanation
    explanation = """Labeling: Uses the term 'junkies,' a derogatory label, 
    Stereotyping: Portrays people who use drugs as causing neighborhood problems, 
    Separation: Creates an us-vs-them mentality, 
    Discrimination: Implies people who use drugs don't belong in the neighborhood"""
    
    # Analyze original text style
    print("Analyzing text style...")
    style_result = reframe.analyze_text_llm(
        post, 
        client, 
        model=model,
        client_type=client_type
    )
    style_instruct = str(style_result)
    
    # Rewrite step 1 - Remove labeling
    print("\nStep 1: Removing stigmatizing labels...")
    rewrite_step1 = reframe.rewrite_to_destigma(
        post,
        explanation,
        style_instruct,
        1,
        model=model,
        client=client,
        client_type=client_type
    )
    print(f"After step 1: {rewrite_step1}")
    
    # Rewrite step 2 - Remove stereotyping, separation, discrimination
    print("\nStep 2: Removing stereotyping, separation, discrimination...")
    rewrite_step2 = reframe.rewrite_to_destigma(
        rewrite_step1,
        explanation,
        style_instruct,
        2,
        model=model,
        client=client,
        client_type=client_type
    )
    print(f"Final rewrite: {rewrite_step2}")
    
    return "Content rewriting example completed"


def example_emotion_detection(client=None, client_type=None, model=None):
    """
    Example showing how to detect emotions in text.
    """
    if client is None:
        client, client_type, model = get_client(client_type=client_type, model=model)
        
    print("\n=== Emotion Detection Example ===")
    print(f"Using client: {client_type}, model: {model}")
    
    texts = [
        "I'm so angry about the drug problem in our city!",
        "I feel hopeless about my brother's addiction",
        "I'm proud of my friend who's been sober for a year now",
        "I'm worried about the increasing drug use in schools"
    ]
    
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text}")
        emotion = reframe.get_emotion(
            text,
            client,
            model=model,
            client_type=client_type
        )
        print(f"Detected emotion: {emotion}")
    
    return "Emotion detection example completed"


def run_all_examples(client_type=None, model=None):
    """Run all examples with the same client."""
    try:
        client, client_type, model = get_client(client_type=client_type, model=model)
        print(f"Using client type: {client_type} with model: {model}")
        
        example_classify_drug(client, client_type, model)
        example_classify_stigma(client, client_type, model)
        example_analyze_text(client, client_type, model)
        example_rewrite_content(client, client_type, model)
        example_emotion_detection(client, client_type, model)
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run Reframe examples')
    
    # Add arguments
    parser.add_argument('example', choices=['all', 'classify_drug', 'classify_stigma', 
                                           'analyze_text', 'rewrite_content', 'emotion_detection'],
                        help='Which example to run')
    parser.add_argument('--client', choices=['openai', 'together', 'claude'],
                        help='Client type to use (openai, together, claude)')
    parser.add_argument('--model', help='Specific model to use')
    
    # Parse arguments from sys.argv if given, otherwise show help
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    # Run the specified example
    if args.example == "all":
        run_all_examples(client_type=args.client, model=args.model)
    elif args.example == "classify_drug":
        example_classify_drug(client_type=args.client, model=args.model)
    elif args.example == "classify_stigma":
        example_classify_stigma(client_type=args.client, model=args.model)
    elif args.example == "analyze_text":
        example_analyze_text(client_type=args.client, model=args.model)
    elif args.example == "rewrite_content":
        example_rewrite_content(client_type=args.client, model=args.model)
    elif args.example == "emotion_detection":
        example_emotion_detection(client_type=args.client, model=args.model)