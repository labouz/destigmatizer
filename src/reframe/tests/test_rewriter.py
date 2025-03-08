import sys
import os
import json
import argparse
import reframe

from reframe.tests.utils import setup_test_argument_parser, parse_test_args

def test_rewriter(api_key=None, model=None, client_type=None):
    """
    Test the text rewriting functionality.
    
    Args:
        api_key (str, optional): API key for the LLM service
        model (str, optional): Model name to use for testing
        client_type (str, optional): Type of client ("openai", "together", "claude", etc.)
    """
    # Initialize client
    try:
        client, client_type = reframe.initialize(api_key=api_key, client_type=client_type)
        print("✓ Client initialization successful")
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    # Test texts and explanations
    test_cases = [
        {
            "text": "Junkies are ruining our neighborhood",
            "explanation": "Labeling: Uses the term 'junkies,' a derogatory label, Stereotyping: Portrays people who use drugs as causing neighborhood problems, Separation: Creates an us-vs-them mentality, Discrimination: Implies people who use drugs don't belong in the neighborhood",
            "description": "Stigmatizing language about drug users"
        },
        {
            "text": "All these addicts need to be locked up",
            "explanation": "Labeling: Uses the term 'addicts' as a dehumanizing label, Stereotyping: Implies all people with addiction are criminals, Separation: Creates a clear divide between 'these addicts' and others, Discrimination: Advocates for punitive measures rather than healthcare",
            "description": "Advocating for punitive measures"
        }
    ]
    
    # Print model information
    print(f"Using model: {model or 'default'} with client type: {client_type}")
    
    # First get style analysis for one of the texts to use in rewriting
    print("\nGetting style analysis for rewriting...")
    style_result = reframe.analyze_text_llm(
        test_cases[0]["text"],
        client,
        model=model,
        client_type=client_type
    )
    print(f"Style analysis result: {style_result}")
    style_instruct = str(style_result)
    
    # Test rewriting
    print("\nTesting rewriting functionality...")
    for case in test_cases:
        print(f"\nOriginal text ({case['description']}): {case['text']}")
        print(f"Explanation: {case['explanation']}")
        
        # Step 1: Remove labeling
        print("\nStep 1: Remove labeling...")
        rewrite_step1 = reframe.rewrite_to_destigma(
            case["text"],
            case["explanation"],
            style_instruct,
            step=1,
            model=model,
            client=client,
            client_type=client_type
        )
        print(f"Rewrite step 1 result: {rewrite_step1}")
        
        # Step 2: Remove stereotyping, separation, and discrimination
        print("\nStep 2: Remove stereotyping, separation, and discrimination...")
        rewrite_step2 = reframe.rewrite_to_destigma(
            rewrite_step1,
            case["explanation"],
            style_instruct,
            step=2,
            model=model,
            client=client,
            client_type=client_type
        )
        print(f"Rewrite step 2 result: {rewrite_step2}")
        
        print(f"\nComparison:")
        print(f"Original: {case['text']}")
        print(f"Final: {rewrite_step2}")

if __name__ == "__main__":
    # Set up argument parser
    parser = setup_test_argument_parser('Test text rewriting functionality')
    
    # Parse arguments and get API key, model, and client type
    api_key, model, client_type = parse_test_args(parser)
    
    # Run the test
    test_rewriter(api_key=api_key, model=model, client_type=client_type)
