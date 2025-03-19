from destigmatizer import create_pipeline
from destigmatizer.filters import classify_drug_filter, classify_stigma_filter, rewrite_text_filter
from destigmatizer.clients import get_client
import json
# Initialize client

with open('./secrets.json') as f:
    secrets = json.load(f)
api_key = secrets.get('OPENAI_API_KEY')
client = get_client("openai", api_key=api_key)

# Create pipeline
pipeline = create_pipeline(client=client, model="gpt-4o")

# Configure pipeline
pipeline.add_filter(classify_drug_filter)
pipeline.add_filter(classify_stigma_filter)
pipeline.add_filter(rewrite_text_filter)

# Process text
text = "Addicts really need to get control of themselves."
result = pipeline.process(text)

print(f"Original: {text}")
print(f"Rewritten: {result['processed_text']}")
print(f"Metadata: {result['metadata']}")