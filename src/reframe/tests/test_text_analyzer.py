import sys
import os
import json
import argparse
import reframe

def test_text_analyzer(api_key=None, model=None, client_type=None):
    """
    Test the text style analysis functionality.
    
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
    
    # Test texts for analysis
    test_texts = {
        "simple_text": "This is a test sentence. It contains multiple parts.",
        "complex_text": "The complexity of language analysis cannot be overstated; various factors contribute to the nuanced understanding of written communication. For instance, sentence length, vocabulary diversity, and punctuation usage all play crucial roles in determining text style.",
        "mixed_text": "I hate this! Why can't people understand? It's not that complicated, is it? Sometimes I wonder if I'm the problem."
    }
    
    # Print model information
    print(f"Using model: {model or 'default'} with client type: {client_type}")
    
    # Test text analysis
    print("\nTesting text analysis...")
    for text_type, text in test_texts.items():
        print(f"\nAnalyzing: {text_type}")
        print(f"Text: {text}")
        result = reframe.analyze_text_llm(
            text,
            client,
            model=model,
            client_type=client_type
        )
        print(f"Style analysis result: {result}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test text style analysis functionality')
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
    
    test_text_analyzer(api_key=api_key, model=args.model, client_type=args.client_type)
