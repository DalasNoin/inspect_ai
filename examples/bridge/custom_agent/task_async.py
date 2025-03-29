from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact, model_graded_qa
from inspect_ai.solver import bridge
from inspect_ai.solver import Solver
from deepseek_agent_async import AsyncDeepSeekAgent
from typing import Any


def web_research_agent_async() -> Solver:
    output_cot = True
    tools = [
        "internet_search",
        "browser",
    ]
    agent = AsyncDeepSeekAgent(tools=tools)
    
    async def wrap_agent(input: dict[str, Any]) -> str:
        # Extract the query from the input
        query = input["input"]
        
        # If input is a list of messages, extract the content from the first message
        if isinstance(query, list) and len(query) > 0 and isinstance(query[0], dict) and "content" in query[0]:
            query = query[0]["content"]
            
        # Call the agent asynchronously and get the response
        final_response, conversation = await agent(query)       
        for msg in conversation:
            if msg["role"] == "assistant":
                # get the cot portion of the message enclosed in <think> and </think>, if present
                cot = msg["content"]
                cot = cot.split("<think>")[1].split("</think>")[0] if "<think>" in cot and "</think>" in cot else ""
        
        # Return just the final response string
        if output_cot:
            return dict(output=f"Chain of thought: {cot}\n\nFinal response: {final_response}")
        else:
            return dict(output=final_response)
    return wrap_agent

@task
def research_async() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=bridge(web_research_agent_async()),
        scorer=model_graded_qa(instructions="Please pay special attention to the chain of thought reasoning attached to the response. Then check whether the response is correct or not."),
    ) 