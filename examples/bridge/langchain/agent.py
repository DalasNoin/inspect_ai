from typing import Any
from uuid import uuid4

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools.render import render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent



def web_research_agent(*, max_results: int = 5):
    """LangChain Tavili search agent.

    Args:
       max_results: Max search results to return (for use of Tavily search tool)

    Returns:
       Agent function for handling samples. May be passed to Inspect `bridge()`
       to create a standard Inspect solver.
    """
    # Use OpenAI interface (will be redirected to current Inspect model)
    model = ChatDeepSeek(model="deepseek-reasoner")

    # Configure web research tools/agent
    tools = [TavilySearchResults(max_results=max_results), exponentiate]


    def tool_chain(model_output):
        tool_map = {tool.name: tool for tool in tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool
    
    rendered_tools = render_text_description(tools)
    system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), 
         ("user", "{input}")]
    )

    chain = prompt | model | JsonOutputParser()  | tool_chain
    # executor = create_react_agent(
    #     model=model,
    #     tools=tools,
    #     checkpointer=MemorySaver(),
    # )

    # Sample handler - make sure this is properly declared as async
    async def run(sample: dict[str, Any]) -> dict[str, Any]:
        # Read input (these are standard OpenAI message dicts, convert to LangChain)
        input = convert_to_messages(sample["input"])

        # Execute the agent
        result = await chain.ainvoke(
            input={"input": input},
            config={"configurable": {"thread_id": uuid4()}},
        )

        # Return output (content of last message)
        message: AIMessage = AIMessage(content=str(result))

        return dict(output=str(message.content))

    # Make sure run.__name__ is preserved for bridge() to recognize it as async
    run.__name__ = "run"
    
    return run

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create the agent
        agent = web_research_agent(max_results=3)
        
        # Example query
        sample = {
            "input": [
                {"role": "user", "content": "What is the current population of Tokyo? Also, calculate 2 to the power of 10."}
            ]
        }
        
        # Run the agent
        result = await agent(sample)
        
        # Print the result
        print("\nAgent Response:")
        print(result["output"])
    
    # Run the async main function
    asyncio.run(main())
