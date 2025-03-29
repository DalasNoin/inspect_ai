from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import bridge
from inspect_ai.solver import Solver
from deepseek_agent_async import AsyncDeepSeekAgent
from typing import Any


def web_research_agent_async() -> Solver:
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
        final_response, _ = await agent(query)
        
        # Return just the final response string
        return final_response
    
    return wrap_agent

@task
def research_async() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=bridge(web_research_agent_async()),
        scorer=model_graded_fact(),
    ) 