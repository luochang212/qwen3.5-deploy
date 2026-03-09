import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


# 配置 LLM 模型
BASE_URL = "http://localhost:8001"
MODEL_NAME = "unsloth/Qwen3.5-4B"


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"


async def agent_invoke(prompt: str):
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=os.getenv("BASE_URL", BASE_URL),
        api_key="sk-no-key-required",
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather],
    )

    async for token, metadata in agent.astream(  
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="messages",
    ):
        node = metadata['langgraph_node']
        if node == 'model' and token.content:
            content = token.content
            print(content, end="", flush=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(agent_invoke("What is the weather in SF?"))
    print()
