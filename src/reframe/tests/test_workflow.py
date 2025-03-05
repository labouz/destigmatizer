import sys
import os
import json
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.reframe import reframe

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
        client, client_type = reframe.initialize(api_key=api_key, client_type=client_type)
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
        client_type=client_type,
        model=model
    )
    print(f"Drug classification result: {drug_result}")
    
    # Step 1: Classify if stigma and get explanation
    print("\nStep 1: Stigma classification and explanation...")
    stigma_result = reframe.classify_if_stigma(
        test_post,
        client=client,
        client_type=client_type,
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
        model=model,
        client_type=client_type
    )
    print(f"Style analysis result: {style_result}")

    # Step 3: Emotion detection
    print("\nStep 3: Emotion detection...")
    emotion = reframe.get_emotion(
        test_post,
        client,
        model=model,
        client_type=client_type
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
        client=client,
        client_type=client_type
    )
    print(f"Rewrite step 1 result: {rewrite_step1}")

    # Second rewrite using the output from first rewrite
    rewrite_step2 = reframe.rewrite_to_destigma(
        rewrite_step1,
        explanation,
        str(style_result),
        2,
        model=model,
        client=client,
        client_type=client_type
    )
    print(f"Rewrite step 2 result: {rewrite_step2}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test reframe workflow functionality')
    parser.add_argument('--api_key', help='API key for LLM service')
    parser.add_argument('--model', help='Model name to use for testing')
    parser.add_argument('--client_type', default='together', choices=['openai', 'together'], 
                        help='Client type (openai or together)')
    
    args = parser.parse_args()
    
    # Get API key either from parameter or secrets file
    api_key = args.api_key
    if api_key is None:
        try:
            with open("secrets.json") as f:
                secrets = json.load(f)
                api_key = secrets.get("OPENAI_API_KEY") if args.client_type == "openai" else secrets.get("TOGETHER_API_KEY")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading API key from secrets.json: {e}")
            print("Please provide an API key using --api_key")
            sys.exit(1)
    
    test_workflow(api_key=api_key, model=args.model, client_type=args.client_type)