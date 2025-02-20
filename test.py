from src.reframe.core import Reframe
import json

# Load API key from secrets file
with open("secrets.json") as f:
    secrets = json.load(f)
    api_key = secrets["OPENAI_API_KEY_SR"]  # Use the specific API key

# Initialize Reframe
reframe = Reframe(api_key=api_key)

# # Test drug content classification
# test_text = "this is for testing"
# result = reframe.classify_drug_content(test_text)
# print(f"Drug classification result: {result}")

# # Test stigma analysis
# stigma_result = reframe.analyze_stigma(test_text)
# print(f"Stigma analysis result: {stigma_result}")

# # Test style analysis
# style_result = reframe.retrive_style_instruction(test_text)
# print(f"Style analysis result: {style_result}")

# Test text rewriting with sample explanation and style
test_text = "Junkies are ruining our neighborhood"
sample_explanation = """Labeling: Uses the term 'junkies,' a derogatory label, 
Stereotyping: Portrays people who use drugs as causing neighborhood problems, 
Separation: Creates an us-vs-them mentality, 
Discrimination: Implies people who use drugs don't belong in the neighborhood"""

style_result = {'top emotions': 'neutral', 'punctuation_usage': 'moderate, with  being most frequent', 'passive_voice_usage': 'none', 'sentence_length_variation': 'ranging from short (4 words) to long (4 words) with an average of 4.0 words per sentence', 'lexical_diversity': '4.00 (MTLD)'}
style_instruct = str(style_result)  # Use the actual style analysis result
rewrite_result = reframe.rewrite_text(test_text, sample_explanation, 1, "gpt-4o", 2, style_instruct)
print(f"Rewrite result: {rewrite_result}")