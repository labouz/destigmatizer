# Destigmatizer (Reframe)
Python library that leverages LLMs to remove harmful language from texts on drug use.

## Testing
You can explore Destigmatizer's functionality interactively using Jupyter notebooks. Below is an example of how to classify and rewrite text using the library.

### Using Destigmatizer in a Jupyter Notebook

**Installing and Set Up**
```python
!pip install destigmatizer

# Import the destigmatizer package
import destigmatizer
from destigmatizer import core
from destigmatizer.clients import get_client
from destigmatizer.utils import get_default_model, get_model_mapping
import json
import os

# Define example texts
NON_DRUG_POST = "I think we should really work on the housing crisis in urban areas. The homeless are getting scary."
DRUG_NON_STIGMA_POST = "I really feel for people who suffer from substance use disorder, being unable to control an impulse due to dependency sounds scary."
DRUG_AND_STIGMA_POST = "Addicts really need to get control of themselves. Just stop doing drugs. It seems like these people are just lacking willpower."

# Set up API client
# for example OpenAI

client = get_client('openai', api_key=api_key)
```

**Classifying Drug-Related and Stigmatizing Content**

```
# Classify whether a text is drug-related and whether it contains stigma
drug_classification = core.classify_drug_content([NON_DRUG_POST, DRUG_NON_STIGMA_POST, DRUG_AND_STIGMA_POST], client, model)
stigma_classification = core.classify_stigma([NON_DRUG_POST, DRUG_NON_STIGMA_POST, DRUG_AND_STIGMA_POST], client, model)

# Print results
for text, drug_label, stigma_label in zip([NON_DRUG_POST, DRUG_NON_STIGMA_POST, DRUG_AND_STIGMA_POST], drug_classification, stigma_classification):
    print(f"Text: {text}\nDrug-related: {drug_label}\nStigmatizing: {stigma_label}\n")

```

**Rewwriting Stigmatizing Content**

```
# Rewrite a stigmatizing post to be more empathetic
rewritten_text = core.rewrite_stigmatizing_text(DRUG_AND_STIGMA_POST, client, model)

print(f"Original: {DRUG_AND_STIGMA_POST}")
print(f"Rewritten: {rewritten_text}")

```

