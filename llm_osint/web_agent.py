from pathlib import Path
from typing import Callable, List, Optional

from langchain.agents import Tool, create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from llm_osint.tools.read_link import get_read_link_tool
from llm_osint.tools.search import get_search_tool


def build_web_agent(
    name: str,
    scraper_func: Callable,
    llm: BaseChatModel,
    web_reader_llm: BaseChatModel) -> AgentExecutor:
    """
    Build a web agent that searches the web for information about a person.
    The agent has access to a web search tool and a web reader/scraper tool.
    Args:
        name: The name of the person.
        scraper_func: The function that scrapes the web page.
        scraper_prompt: The prompt for the scraper.
        llm: The LLM model to use.
        web_reader_llm: The web reader LLM model.
    Returns:
        The web agent.
    """
    tools = [get_search_tool(),
             get_read_link_tool(
                name=name, scrapper_func=scraper_func,
                map_llm=web_reader_llm, reduce_llm=llm)]
    return build_web_agent_from_tools(tools, llm)


def build_web_agent_from_tools(
    tools: List[Tool],
    llm: BaseChatModel,
    memory: Optional[BaseChatMemory] = None) -> AgentExecutor:
    """
    Build a web agent that searches the web for information about a person.
    Args:
        tools: The tools to use.
        llm: The LLM model to use.
        memory: The memory to use.
    Returns:
        The web agent.
    """

    system_prompt = PromptTemplate.from_file(Path(__file__).parent / "prompts" / "web_agent.txt")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.template),
        ("human", "{input}\n\n{agent_scratchpad}")
    ])
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt)
    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        early_stopping_method="generate",
        verbose=True,
    )
    return agent_chain
