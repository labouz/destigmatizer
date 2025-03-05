import sys
import os
import json
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reframe import reframe

def test_emotion_detector(api_key=None, model=None, client_type="openai"):
    """
    Test the emotion detection functionality.
    
    Args:
        api_key (str, optional): API key for the LLM service
        model (str, optional): Model name to use for testing
        client_type (str, optional): Type of client ("openai" or "together")
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
    parser.add_argument('--client_type', default='openai', choices=['openai', 'together'], 
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
    
    test_emotion_detector(api_key=api_key, model=args.model, client_type=args.client_type)
