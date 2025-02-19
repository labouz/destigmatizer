from reframe import Reframe

# Initialize with OpenAI
reframe = Reframe(api_key="your-api-key")

# Analyze text
result = reframe.classify_drug_content("this is for testing")