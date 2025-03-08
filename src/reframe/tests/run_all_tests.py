import sys
import os
import argparse

# Import test modules correctly
from reframe.tests import (
    test_drug_classifier,
    test_stigma_classifier, 
    test_text_analyzer,
    test_rewriter,
    test_emotion_detector,
    test_workflow
)

from reframe.tests.utils import get_api_key_for_testing, setup_test_argument_parser, parse_test_args


def run_all_tests(api_key=None, model=None, client_type=None):
    """
    Run all tests in sequence.
    
    Args:
        api_key (str, optional): API key for the LLM service
        model (str, optional): Model name to use for testing
        client_type (str, optional): Type of client ("openai", "together", "claude", etc.)
    """
    print("=" * 80)
    print("RUNNING ALL REFRAME TESTS")
    print(f"Model: {model or 'default'}")
    print(f"Client type: {client_type}")
    print("=" * 80)
    
    print("\n1. Drug Classification Test")
    print("-" * 40)
    test_drug_classifier(api_key, model, client_type)
    
    print("\n2. Stigma Classification Test")
    print("-" * 40)
    test_stigma_classifier(api_key, model, client_type)
    
    print("\n3. Text Analysis Test")
    print("-" * 40)
    test_text_analyzer(api_key, model, client_type)
    
    print("\n4. Text Rewriting Test")
    print("-" * 40)
    test_rewriter(api_key, model, client_type)
    
    print("\n5. Emotion Detection Test")
    print("-" * 40)
    test_emotion_detector(api_key, model, client_type)
    
    print("\n6. Complete Workflow Test")
    print("-" * 40)
    test_workflow(api_key, model, client_type)
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

def main():
    # Set up argument parser
    parser = setup_test_argument_parser('Run reframe tests')
    parser.add_argument('test', nargs='?', choices=['all', 'drug', 'stigma', 'analysis', 'rewriter', 'emotion', 'workflow'], 
                       default='all', help='Specific test to run')
    
    # Parse arguments and get API key, model, and client type
    args = parser.parse_args()
    api_key = get_api_key_for_testing(args.api_key, args.client_type)
    
    # Run in script mode
    if args.test == 'all':
        run_all_tests(api_key=api_key, model=args.model, client_type=args.client_type)
    elif args.test == 'drug':
        test_drug_classifier(api_key, args.model, args.client_type)
    elif args.test == 'stigma':
        test_stigma_classifier(api_key, args.model, args.client_type)
    elif args.test == 'analysis':
        test_text_analyzer(api_key, args.model, args.client_type)
    elif args.test == 'rewriter':
        test_rewriter(api_key, args.model, args.client_type)
    elif args.test == 'emotion':
        test_emotion_detector(api_key, args.model, args.client_type)
    elif args.test == 'workflow':
        test_workflow(api_key, args.model, args.client_type)

if __name__ == "__main__":
    main()