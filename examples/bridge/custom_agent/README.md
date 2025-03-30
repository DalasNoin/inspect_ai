# How to replicate the CoT jailbreaking results

## Setup

Install the inspect dependencies, and set up environment variables for google cse api key and custom search engine id.
Also set up keys for openrouter, deepseek and openai.

## Run the evaluation

Navigate to the examples directory and run the following command:

```bash
cd examples
```

## Run the inspect evaluation

```bash
inspect eval bridge/custom_agent/task_async.py --model-base-url https://api.deepseek.com --model openai/deepseek-reasoner -T scoring_model=openai/gpt-4o
```

## Test individual tasks

```bash
python bridge/custom_agent/test_async_improved.py --jailbreak
```