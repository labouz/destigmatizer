# reframe
Python library that leverages LLMs to remove harmful language from texts on drug use.

## Testing

Reframe includes integrated tests for all major functions. You can run tests in two ways:

### Using the CLI command (recommended)

```bash
# Test all functions
reframe-test

# Test a specific function
reframe-test drug --api_key YOUR_API_KEY

# Use a specific model with Together AI
reframe-test --client_type together --model "meta-llama/Meta-Llama-3.1" 

# Use Claude
reframe-test --client_type claude --model "claude-3-haiku-20240307"

# Using environment variables (recommended)
# First set the environment variables:
export OPENAI_API_KEY="your-openai-key"
export TOGETHER_API_KEY="your-together-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Then run tests without specifying the API key
reframe-test --client_type openai
reframe-test --client_type together
reframe-test --client_type claude
```

### Running test scripts directly
```bash
# From the root directory
python3 -m reframe.tests.run_all_tests