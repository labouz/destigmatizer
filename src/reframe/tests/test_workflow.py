import sys
import os
import json
import argparse
import reframe

from reframe.tests.utils import setup_test_argument_parser, parse_test_args

def test_workflow(api_key=None, model=None, client_type=None):
    """
    Test the reframe workflow.
    
    Args:
        api_key (str, optional): API key for the LLM service
        model (str, optional): Model name to use for testing
        client_type (str, optional): Type of client ("openai", "together" or "Claude")
    """
    # Initialize client
    try:
        client = reframe.initialize(api_key=api_key, client_type=client_type)
        print("✓ Client initialization successful")
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    # Print model information
    print(f"Using model: {model or 'default'} with client type: {client_type}")

    # Test post
    test_post = "junkies are causing problems in our neighborhood"
    print(f"\nTesting workflow on post: \"{test_post}\"")
    
    # Test drug classification
    print("\nTesting drug classification...")
    drug_result = reframe.classify_if_drug(
        test_post,
        client=client,
        model=model
    )
    print(f"Drug classification result: {drug_result}")
    
    # Step 1: Classify if stigma and get explanation
    print("\nStep 1: Stigma classification and explanation...")
    stigma_result = reframe.classify_if_stigma(
        test_post,
        client=client,
        model=model
    )
    print(f"Stigma classification result: {stigma_result}")
    
    # Extract label and explanation from stigma classification result
    if ', ' in stigma_result:
        label, explanation = stigma_result.split(', ', 1)
    else:
        label = stigma_result
        explanation = ""
    
    print(f"Extracted label: {label}")
    print(f"Extracted explanation: {explanation}")

    # Step 2: Analyze text style
    print("\nStep 2: Text style analysis...")
    style_result = reframe.analyze_text_llm(
        test_post, 
        client, 
        model=model
    )
    print(f"Style analysis result: {style_result}")

    # Step 3: Emotion detection
    print("\nStep 3: Emotion detection...")
    emotion = reframe.get_emotion(
        test_post,
        client,
        model=model
    )
    print(f"Detected emotion: {emotion}")

    # Step 4: Rewriting with actual explanation and style
    print("\nStep 4: Rewriting process...")
    # First rewrite
    rewrite_step1 = reframe.rewrite_to_destigma(
        test_post,
        explanation,
        str(style_result),
        1,
        model=model,
        client=client
    )
    print(f"Rewrite step 1 result: {rewrite_step1}")

    # Second rewrite using the output from first rewrite
    rewrite_step2 = reframe.rewrite_to_destigma(
        rewrite_step1,
        explanation,
        str(style_result),
        2,
        model=model,
        client=client
    )
    print(f"Rewrite step 2 result: {rewrite_step2}")

if __name__ == "__main__":
    # Set up argument parser
    parser = setup_test_argument_parser('Test reframe workflow functionality')
    
    # Parse arguments and get API key, model, and client type
    api_key, model, client_type = parse_test_args(parser)
    
    # Run the test
    test_workflow(api_key=api_key, model=model, client_type=client_type)