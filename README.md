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
```

### Running test scripts directly
```bash
# From the root directory
python3 -m reframe.tests.run_all_tests