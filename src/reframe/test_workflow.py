import reframe
import json

# Default model to use for all tests
Llama_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
OpenAi_MODEL = "gpt-4o"
MODEL = Llama_MODEL

def test_workflow(api_key=None, model=None):
    """
    Test the reframe workflow.
    
    Args:
        api_key (str, optional): API key for LLM service. Falls back to secrets.json if not provided.
        model (str, optional): Model name to use for testing. Defaults to global MODEL.
    """
    # Use provided model or fall back to default
    use_model = model or MODEL
    
    # Get API key either from parameter or secrets file
    if api_key is None:
        try:
            with open("secrets.json") as f:
                secrets = json.load(f)
                api_key = secrets.get("TOGETHER_API_KEY")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading API key from secrets.json: {e}")
            print("Please provide an API key as a parameter")
            return
    
    if not api_key:
        print("API key is required. Please provide it as a parameter")
        return

    # Test initialization
    print("\nTesting initialization...")
    client, client_type = reframe.initialize(api_key=api_key, client_type="together")
    print("✓ Initialization successful")

    # Test post with workflow
    test_post = "junkies are causing problems in our neighborhood"
    
    # Test drug classification
    print("\nTesting drug classification...")
    drug_result = reframe.classify_if_drug(
        test_post,
        client=client,
        client_type=client_type,
        model=use_model
    )
    print(f"Drug classification result: {drug_result}")
    
    # Step 1: Classify if stigma and get explanation
    print("\nStep 1: Stigma classification and explanation...")
    stigma_result = reframe.classify_if_stigma(
        test_post,
        client=client,
        client_type=client_type,
        model=use_model
    )
    print(f"Stigma classification result: {stigma_result}")
    
    # Extract label and explanation from stigma classification result
    # Assuming the result format is "label, explanation"
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
        model=use_model,
        client_type=client_type
    )
    print(f"Style analysis result: {style_result}")

    # Step 3: Emotion detection
    print("\nStep 3: Emotion detection...")
    emotion = reframe.get_emotion(
        test_post,
        client,
        model=use_model,
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
        model=use_model,
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
        model=use_model,
        client=client,
        client_type=client_type
    )
    print(f"Rewrite step 2 result: {rewrite_step2}")

if __name__ == "__main__":
    test_workflow()