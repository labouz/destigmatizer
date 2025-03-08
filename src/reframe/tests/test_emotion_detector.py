import sys
import os
import json
import argparse

import reframe

def test_emotion_detector(api_key=None, model=None, client_type=None):
    """
    Test the emotion detection functionality.
    
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
    
    # Test texts for emotion detection
    test_texts = {
        "angry_text": "I'm so furious about this situation! Everyone keeps ignoring the problem and it's driving me crazy!",
        "sad_text": "I feel so alone today. Nothing seems to make me happy anymore.",
        "happy_text": "I'm thrilled about the news! This is the best thing that's happened to me all year.",
        "fearful_text": "I'm terrified about what might happen next. I can't stop thinking about all the possibilities.",
        "stigmatizing_text": "Those junkies are ruining our neighborhood. They need to be removed."
    }
    
    # Print model information
    print(f"Using model: {model or 'default'} with client type: {client_type}")
    
    # Test emotion detection
    print("\nTesting emotion detection...")
    for text_type, text in test_texts.items():
        print(f"\nText ({text_type}): {text}")
        emotion = reframe.get_emotion(
            text,
            client,
            model=model,
            client_type=client_type
        )
        print(f"Detected emotion: {emotion}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test emotion detection functionality')
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
    
    test_emotion_detector(api_key=api_key, model=args.model, client_type=args.client_type)
