# How to replicate the CoT jailbreaking results

## Setup

Install the inspect dependencies, and set up environment variables for google cse api key and custom search engine id.
Also set up keys for openrouter, deepseek and openai.

## Run the evaluation

Navigate to the examples directory and run the following command:

```bash
$ cd examples
```

## Run the inspect evaluation

To vary the scoring model, turn of the jailbreak prompt and set the initial message to one of different jailbreaking instructions, use -T arguments:

```bash
$ inspect eval bridge/custom_agent/task_async.py --model-base-url https://api.deepseek.com --model openai/deepseek-reasoner -T scoring_model=openai/gpt-4o
```

Our task supports these parameters:

```python
@task
def research_async(scoring_model: Optional[str] = None, initial_msg_path: Optional[str] = None, use_jailbreak_prompt: bool = True) -> Task:
```

## Test individual tasks

```bash
$ python bridge/custom_agent/test_async_improved.py --jailbreak
```

## View the results and evaluations

```bash
$ inspect view