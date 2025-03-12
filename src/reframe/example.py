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
  python -m reframe.example test1
  
  # Run with specific client type
  python -m reframe.example test1 --client openai
  
  # Run with specific model
  python -m reframe.example test1 --model gpt-4o
  
  # Run with specific client and model
  python -m reframe.example test1 --client openai --model gpt-4o
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any
from reframe import core
from reframe.clients import get_client
from reframe.utils import get_model_mapping, get_default_model


# Default models to use for examples
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-haiku-20240307"

def get_example_client(client_type: str = None, api_key: str = None, model: str = None):
    """
    Initialize and return a client based on client_type.
    
    Args:
        client_type: The type of client to initialize ("openai", "together", "claude")
        api_key: API key to use. If None, will look for environment variables
        model: Specific model to use. If None, will use the default for the client
        
    Returns:
        tuple: (client, client_type, model)
    """
    # Select appropriate model based on client_type if model not specified
    if model is None:
        if client_type and client_type.lower() == "openai":
            model = OPENAI_MODEL
        elif client_type and client_type.lower() == "together":
            model = LLAMA_MODEL
        elif client_type and client_type.lower() == "claude":
            model = CLAUDE_MODEL
    
    # Initialize client using reframe's client factory
    try:
        client = get_client(client_type, api_key)
        # If client_type wasn't specified, detect it from the created client
        if client_type is None:
            from reframe.clients import detect_client_type
            client_type = detect_client_type(client)
            
            # Use get_model_mapping for model selection
        if model is None:
            model = get_default_model(client_type)
        else:
            model = get_model_mapping(model, client_type)
    except ValueError as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)
        
    return client, client_type, model

# test 1
def example_classify_drug(post=None, client=None, client_type=None, model=None):
    """
    Example showing how to classify drug-related content.
    
    Args:
        post: Text to classify, if None a default will be used
        client: Initialized client object
        client_type: Type of client to use if client is None
        model: Model to use
        
    Returns:
        str: Classification result ('D', 'ND', or 'skipped')
    """
    if client is None:
        client, client_type, model = get_example_client(client_type=client_type, model=model)
        
    print("\n━━━ Drug Classification ━━━")
    
    print(f"Input text:\n{post}")
    
    drug_result = core.classify_if_drug(
        text=post,
        client=client,
        model=model,
        retries=2
    )
    print(f"Result: {drug_result}")
    
    return drug_result

# test 2
def example_classify_stigma(post=None, client=None, client_type=None, model=None):
    """
    Example showing how to identify stigmatizing language.
    
    Args:
        post: Text to classify, if None a default will be used
        client: Initialized client object
        client_type: Type of client to use if client is None
        model: Model to use
        
    Returns:
        str: Stigma classification result
    """
    if client is None:
        client, client_type, model = get_example_client(client_type=client_type, model=model)
        
    print("\n━━━ Stigmatizing Language Identification ━━━")
    
    print(f"Input text:\n{post}")
    
    stigma_result = core.classify_if_stigma(
        text=post,
        client=client,
        model=model,
        retries=2
    )
    print(f"Result: {stigma_result}")
    
    return stigma_result

# test 3
def example_classify_drug_and_stigma(post=None, client=None, client_type=None, model=None):
    """
    Example showing how to classify for both drugs and stigma.
    
    Args:
        post: Text to classify, if None a default will be used
        client: Initialized client object
        client_type: Type of client to use if client is None
        model: Model to use
        
    Returns:
        tuple: (drug_result, stigma_result)
    """
    if client is None:
        client, client_type, model = get_example_client(client_type=client_type, model=model)
        
    print("\n━━━ Drug and Stigmatizing Language Analysis ━━━")
    
    # Use default post if none provided
    if post is None:
        post = "Addicts really need to get control of themselves. Just stop doing drugs. It seems like these people are just lacking willpower."
    
    print(f"Input text:\n{post}")
    
    drug_result = core.classify_if_drug(
        text=post,
        client=client,
        model=model,
        retries=2
    )
    stigma_result = core.classify_if_stigma(
        text=post,
        client=client,
        model=model,
        retries=2
    )
    
    print(f"Drug classification: {drug_result}")
    print(f"Stigma classification: {stigma_result}")
    
    return drug_result, stigma_result


# test 4
def example_rewrite_content(post=None, stigma_result=None, client=None, client_type=None, model=None):
    """
    Example showing how to rewrite stigmatizing content.
    
    Args:
        post: Text to rewrite, if None a default will be used
        stigma_result: Stigma classification result containing explanation, if None a default will be used
        client: Initialized client object
        client_type: Type of client to use if client is None
        model: Model to use
        
    Returns:
        str: Rewritten content
    """
    if client is None:
        client, client_type, model = get_example_client(client_type=client_type, model=model)
        
    print("\n━━━ Content Rewriting ━━━")
    
    # Use default post if none provided
    if post is None:
        post = "Addicts really need to get control of themselves. Just stop doing drugs. It seems like these people are just lacking willpower."
    
    explanation = stigma_result
    
    print(f"Original text:\n{post}")
    print(f"Stigma explanation: {explanation}")
    
    # Analyze original text style
    style_result = core.analyze_text_llm(
        text=post, 
        client=client,
        model=model
    )
    style_instruct = str(style_result)
    
    # Rewrite step 1 - Remove labeling
    print("\n■ Step 1: Removing stigmatizing labels")
    rewrite_step1 = core.rewrite_to_destigma(
        text=post,
        explanation=explanation,
        style_instruct=style_instruct,
        step=1,
        model=model,
        client=client,
        retries=2
    )
    print(f"{rewrite_step1}")
    
    # Rewrite step 2 - Remove stereotyping, separation, discrimination
    print("\n■ Step 2: Removing stereotyping, separation, discrimination")
    rewrite_step2 = core.rewrite_to_destigma(
        text=rewrite_step1,
        explanation=explanation,
        style_instruct=style_instruct,
        step=2,
        model=model,
        client=client,
        retries=2
    )
    print(f"{rewrite_step2}")
    
    return rewrite_step2


def run_all_examples(client_type=None, model=None):
    """Run all examples with the same client, passing results between them."""
    try:
        client, client_type, model = get_example_client(client_type=client_type, model=model)
        print(f"\n▶ Running all examples using {client_type} with model {model}")
        
        # Define test posts
        non_drug_post = "I think we should really work on the housing crisis in urban areas. The homeless are getting scary."
        drug_and_stigma_post = "Addicts really need to get control of themselves. Just stop doing drugs. It seems like these people are just lacking willpower."
        
        # Run all test cases in sequence
        run_test_case("test1", client, client_type, model)
        run_test_case("test2", client, client_type, model)
        run_test_case("test3", client, client_type, model)
        run_test_case("test4", client, client_type, model)
        
        print("\n✓ All examples completed successfully!")
    except Exception as e:
        print(f"✗ Error running examples: {e}")


def run_test_case(test_case, client=None, client_type=None, model=None):
    """Run a specific test case with predefined test posts."""
    try:
        if client is None:
            client, client_type, model = get_example_client(client_type=client_type, model=model)
        
        print(f"\n▶ Running Test Case {test_case} [{client_type} - {model}]")
        
        # Define test posts
        non_drug_post = "I think we should really work on the housing crisis in urban areas. The homeless are getting scary."
        drug_non_stigma_post = "I really feel for people who suffer from substance use disorder, being unable to control an impulse due to dependency sounds scary."
        drug_and_stigma_post = "Addicts really need to get control of themselves. Just stop doing drugs. It seems like these people are just lacking willpower."
        
        if test_case == "test1":
            # Test case 1: Classify non-drug post
            drug_result = example_classify_drug(post=non_drug_post, client=client, client_type=client_type, model=model)
        elif test_case == "test2":
            # Test case 2: Classify drug post for stigma
            stigma_result = example_classify_stigma(post=drug_non_stigma_post, client=client, client_type=client_type, model=model)
        elif test_case == "test3":
            # Test case 3: Classify drug post for both drug and stigma
            drug_result, stigma_result = example_classify_drug_and_stigma(post=drug_and_stigma_post, client=client, client_type=client_type, model=model)
        elif test_case == "test4":
            # Test case 4: Classify drug post for stigma and then rewrite
            # First classify to get stigma explanation
            drug_result, stigma_result = example_classify_drug_and_stigma(post=drug_and_stigma_post, client=client, client_type=client_type, model=model)
            # Then rewrite based on classification
            rewritten_text = example_rewrite_content(
                post=drug_and_stigma_post, 
                stigma_result=stigma_result,
                client=client, 
                client_type=client_type, 
                model=model
            )
        else:
            print(f"Unknown test case: {test_case}")
            return
    except Exception as e:
        print(f"✗ Error running test case {test_case}: {e}")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run Reframe examples')
    
    # Define available examples
    examples = {
        'all': run_all_examples,
        'test1': lambda **kwargs: run_test_case('test1', **kwargs),
        'test2': lambda **kwargs: run_test_case('test2', **kwargs),
        'test3': lambda **kwargs: run_test_case('test3', **kwargs),
        'test4': lambda **kwargs: run_test_case('test4', **kwargs)
    }
    
    # Add arguments
    parser.add_argument('example', choices=list(examples.keys()),
                        help='Example to run (all or specific test case)')
    parser.add_argument('--client', choices=['openai', 'together', 'claude'],
                        help='Client type to use (openai, together, claude)')
    parser.add_argument('--model', help='Specific model to use')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the selected example
    examples[args.example](client_type=args.client, model=args.model)