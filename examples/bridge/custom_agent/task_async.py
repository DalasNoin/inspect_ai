from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact, model_graded_qa
from inspect_ai.solver import bridge
from inspect_ai.solver import Solver
from deepseek_agent_async import AsyncDeepSeekAgent
from typing import Any, Optional


def web_research_agent_async(
    use_jailbreak_prompt: bool = True,
    initial_msg_path: Optional[str] = None,
    output_cot: bool = True,
) -> Solver:
    tools = [
        "internet_search",
        "browser",
    ]
    if use_jailbreak_prompt:
        initial_conversation = initial_msg_path
        agent = AsyncDeepSeekAgent(
            tools=tools, initial_conversation=initial_conversation
        )
    else:
        agent = AsyncDeepSeekAgent(tools=tools)

    async def wrap_agent(input: dict[str, Any]) -> str:
        # Extract the query from the input
        query = input["input"]

        # If input is a list of messages, extract the content from the first message
        if (
            isinstance(query, list)
            and len(query) > 0
            and isinstance(query[0], dict)
            and "content" in query[0]
        ):
            query = query[0]["content"]

        # Call the agent asynchronously and get the response
        final_response, conversation = await agent(query)
        cot = ""
        step = 0
        for i, msg in enumerate(conversation):
            if msg["role"] == "assistant":
                # get the cot portion of the message enclosed in <think> and </think>, if present

                new_cot = (
                    msg["content"].split("<think>")[1].split("</think>")[0]
                    if "<think>" in msg["content"] and "</think>" in msg["content"]
                    else ""
                )
                cot += f"Step {step + 1}: {new_cot}\n"
                step += 1

        if not cot:
            cot = "No chain of thought provided"

        # Return just the final response string
        if output_cot:
            return dict(
                output=f"Chain of thought: {cot}\n\nFinal response: {final_response}"
            )
        else:
            return dict(output=f"Final response: {final_response}")

    return wrap_agent


@task
def research_async(
    scoring_model: Optional[str | list[str]] = None,
    initial_msg_path: Optional[str] = None,
    use_jailbreak_prompt: bool = True,
) -> Task:
    if initial_msg_path is None:
        initial_msg_path = "bridge/custom_agent/initial_msg_inverse.json"

    return Task(
        dataset=json_dataset("dataset.json"),
        solver=bridge(
            web_research_agent_async(
                use_jailbreak_prompt=use_jailbreak_prompt,
                initial_msg_path=initial_msg_path,
                output_cot=True,
            )
        ),
        # scorer=model_graded_qa(instructions=GRADING_PROMPT, model="openai/gpt-4o-mini"),
        scorer=model_graded_qa(model=scoring_model)
        if not isinstance(scoring_model, list)
        else [model_graded_qa(model=m) for m in scoring_model],
    )


@task
def research_async_no_cot(
    scoring_model: Optional[str | list[str]] = None,
    initial_msg_path: Optional[str] = None,
    use_jailbreak_prompt: bool = True,
) -> Task:
    if initial_msg_path is None:
        initial_msg_path = "bridge/custom_agent/initial_msg_inverse.json"
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=bridge(
            web_research_agent_async(
                use_jailbreak_prompt=use_jailbreak_prompt,
                initial_msg_path=initial_msg_path,
                output_cot=False,
            )
        ),
        scorer=model_graded_qa(model=scoring_model)
        if not isinstance(scoring_model, list)
        else [model_graded_qa(model=m) for m in scoring_model],
    )
