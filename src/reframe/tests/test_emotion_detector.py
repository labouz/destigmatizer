import sys
import os
import json
import argparse
import reframe

from reframe.tests.utils import setup_test_argument_parser, parse_test_args

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
    parser = setup_test_argument_parser('Test emotion detection functionality')
    
    # Parse arguments and get API key, model, and client type
    api_key, model, client_type = parse_test_args(parser)
    
    # Run the test
    test_emotion_detector(api_key=api_key, model=model, client_type=client_type)
