import sys
import os
import json
import argparse
import reframe

def test_stigma_classifier(api_key=None, model=None, client_type=None):
    """
    Test the stigma classification functionality.
    
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

    # Test posts for classification
    test_posts = {
        "stigma_post_1": "All these junkies should be locked up, they're ruining our city",
        "stigma_post_2": "Once an addict, always an addict. They'll never change.",
        "non_stigma_post_1": "People struggling with addiction need support and understanding",
        "non_stigma_post_2": "My friend is in recovery from opioid use disorder and is doing incredibly well."
    }
    
    # Print model information
    print(f"Using model: {model or 'default'} with client type: {client_type}")
    
    # Test stigma classification
    print("\nTesting stigma classification...")
    for post_type, post in test_posts.items():
        print(f"\nTesting on: {post}")
        result = reframe.classify_if_stigma(
            post,
            client=client,
            client_type=client_type,
            model=model
        )
        print(f"{post_type}: {result}")
        
        # If stigmatizing, show the explanation (which follows after the "S, " prefix)
        if result.startswith("s,"):
            print(f"Explanation: {result[2:].strip()}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test stigma classification functionality')
    parser.add_argument('--api_key', help='API key for LLM service')
    parser.add_argument('--model', help='Model name to use for testing')
    parser.add_argument('--client_type', default='openai', 
                        help='Client type (e.g., openai, together, claude)')
    
    args = parser.parse_args()
    
    # Get API key either from parameter, environment variables, or secrets file
    api_key = args.api_key
    if api_key is None:
        # First try to get from environment variables
        if args.client_type.lower() == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif args.client_type.lower() == "together":
            api_key = os.environ.get("TOGETHER_API_KEY")
        elif args.client_type.lower() == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # If not found in env vars, try secrets file
        if api_key is None:
            try:
                with open("secrets.json") as f:
                    secrets = json.load(f)
                    if args.client_type.lower() == "openai":
                        api_key = secrets.get("OPENAI_API_KEY")
                    elif args.client_type.lower() == "together":
                        api_key = secrets.get("TOGETHER_API_KEY")
                    elif args.client_type.lower() == "claude":
                        api_key = secrets.get("ANTHROPIC_API_KEY")
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"Error loading API key from secrets.json: {e}")
                print("Please provide an API key using --api_key or set the appropriate environment variable")
                sys.exit(1)
                
        if api_key is None:
            print(f"No API key found for {args.client_type}. Please provide an API key.")
            sys.exit(1)
    
    test_stigma_classifier(api_key=api_key, model=args.model, client_type=args.client_type)
