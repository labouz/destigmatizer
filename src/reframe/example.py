import reframe
import json

# Default model to use for all tests
Llama_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
OpenAi_MODEL = "gpt-4o"

def test_all_functions(api_key=None, model=None):
    """
    Test all reframe functions.
    
    Args:
        api_key (str, optional): API key for LLM service. Falls back to secrets.json if not provided.
        model (str, optional): Model name to use for testing. Defaults to global MODEL.
    """
    if not api_key:
        print("API key is required. Please provide it as a parameter")
        return

    # Test initialization
    print("\nTesting initialization...")
    client, client_type = reframe.initialize(api_key=api_key, client_type="together")
    print("✓ Initialization successful")

    # Test posts for classification
    test_posts = {
        "drug_post": "I'm so high right now, smoking the best weed ever",
        "non_drug_post": "I'm feeling really down today, need someone to talk to",
        "stigma_post": "All these junkies should be locked up, they're ruining our city",
        "non_stigma_post": "People struggling with addiction need support and understanding"
    }

    # Test drug classification
    print("\nTesting drug classification...")
    for post_type, post in test_posts.items():
        print(f"Testing on: {post}")
        result = reframe.classify_if_drug(
            post,
            client=client,
            client_type=client_type,
            model=model
        )
        print(f"{post_type}: {result}")

    # Test stigma classification
    print("\nTesting stigma classification...")
    for post_type, post in test_posts.items():
        print(f"Testing on: {post}")
        result = reframe.classify_if_stigma(
            post,
            client=client,
            client_type=client_type,
            model=model
        )
        print(f"{post_type}: {result}")

    # Test text analysis
    print("\nTesting text analysis...")
    sample_text = "This is a test sentence. It contains multiple parts. We want to analyze its style."
    print(f"Sample text: {sample_text}")
    style_result = reframe.analyze_text_llm(
        sample_text, 
        client, 
        model=model,
        client_type=client_type
    )
    print(f"Style analysis result: {style_result}")

    # Test rewriting
    print("\nTesting rewriting...")
    test_text = "Junkies are ruining our neighborhood"
    print(f"Original text: {test_text}")
    sample_explanation = """Labeling: Uses the term 'junkies,' a derogatory label, 
    Stereotyping: Portrays people who use drugs as causing neighborhood problems, 
    Separation: Creates an us-vs-them mentality, 
    Discrimination: Implies people who use drugs don't belong in the neighborhood"""
    print(f"Sample explanation: {sample_explanation}")
    style_instruct = str(style_result)
    print(f"Style instruction: {style_instruct}")

    # Test rewriting step 1
    rewrite_step1 = reframe.rewrite_to_destigma(
        test_text,
        sample_explanation,
        style_instruct,
        1,
        model=model,
        client=client,
        client_type=client_type
    )
    print(f"Rewrite step 1 result: {rewrite_step1}")

    # Test rewriting step 2
    rewrite_step2 = reframe.rewrite_to_destigma(
        rewrite_step1,
        sample_explanation,
        style_instruct,
        2,
        model=model,
        client=client,
        client_type=client_type
    )
    print(f"Rewrite step 2 result: {rewrite_step2}")

    # Test emotion detection
    print("\nTesting emotion detection...")
    emotion = reframe.get_emotion(
        test_text,
        client,
        model=model,
        client_type=client_type
    )
    print(f"Detected emotion: {emotion}")

if __name__ == "__main__":
    try:
        with open("secrets.json") as f:
            secrets = json.load(f)
            api_key = secrets.get("TOGETHER_API_KEY")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading API key from secrets.json: {e}")
        print("Please provide an API key as a parameter")
        exit(1)
    test_all_functions(api_key=api_key, model=Llama_MODEL)