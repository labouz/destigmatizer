import reframe
import json

# Default model to use for all tests
Llama_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
OpenAi_MODEL = "gpt-4o"
MODEL = Llama_MODEL

def test_workflow():
    # Load API key from secrets file
    with open("secrets.json") as f:
        secrets = json.load(f)
        api_key = secrets["TOGETHER_API_KEY"]

    # Test initialization
    print("\nTesting initialization...")
    client, client_type = reframe.initialize(api_key=api_key, client_type="together")
    print("✓ Initialization successful")

    # Test post with workflow
    test_post = "Junkies are ruining our neighborhood"
    
    # Step 1: Classify if stigma and get explanation
    print("\nStep 1: Stigma classification and explanation...")
    stigma_result = reframe.classify_if_stigma(
        test_post,
        client=client,
        client_type=client_type,
        model=MODEL
    )
    print(f"Stigma classification result: {stigma_result}")
    
    # Extract explanation from stigma classification result
    # Assuming the result contains explanation after the comma
    explanation = stigma_result.split(', ', 1)[1] if ', ' in stigma_result else ""
    print(f"Extracted explanation: {explanation}")

    # Step 2: Analyze text style
    print("\nStep 2: Text style analysis...")
    style_result = reframe.analyze_text_llm(
        test_post, 
        client, 
        model=MODEL,
        client_type=client_type
    )
    print(f"Style analysis result: {style_result}")

    # Step 3: Emotion detection
    print("\nStep 3: Emotion detection...")
    emotion = reframe.get_emotion(
        test_post,
        client,
        model=MODEL,
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
        model=MODEL,
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
        model=MODEL,
        client=client,
        client_type=client_type
    )
    print(f"Rewrite step 2 result: {rewrite_step2}")

if __name__ == "__main__":
    test_workflow()