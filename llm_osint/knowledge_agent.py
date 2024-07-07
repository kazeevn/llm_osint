from typing import List, Callable, Optional
import logging
from pathlib import Path
from omegaconf import OmegaConf
from langchain_core.messages import HumanMessage
from langchain.agents.agent import AgentExecutor
from langchain.chat_models.base import BaseChatModel

logger = logging.getLogger(__name__)

def run_chain_with_retries(agent_chain: AgentExecutor, retries: int, **agent_run_kwargs) -> str:
    """
    Run an agent chain with retries.
    Args:
        agent_chain: The agent chain to run.
        retries: The number of retries.
        agent_run_kwargs: The keyword arguments to pass to the agent chain.
    Returns:
        The output of the agent chain.
    """
    exception = None
    for _ in range(retries):
        try:
            output = agent_chain.invoke(agent_run_kwargs)
            return output['output']
        except Exception as e:
            exception = e
    raise exception


def run_knowledge_agent(
    build_web_agent_func: Callable[[], AgentExecutor],
    deep_dive_topics: int,
    deep_dive_rounds: int,
    model: BaseChatModel = None,
    retries: Optional[int] = 3,
    **prompt_args) -> List[str]:
    """
    Search the internet for information about a person and return the gathered information.
    Uses web search and web page reading tools to gather information in ReAct paradigm. Prompts are
    defined in the config file and heavily influence the information gathered.
    Args:
        build_web_agent_func: A function that builds a web agent.
        deep_dive_topics: The number of topics to deep dive into.
        deep_dive_rounds: The number of rounds to deep dive.
        model: The LLM model to use.
        retries: The number of retries.
        prompt_args: Additional prompt arguments.
    Returns:
        The gathered information in the form of list of chunks corresponding to topics.
    """
    prompts = OmegaConf.load(Path(__file__).parent / "prompts" / "knowledge_agent.yaml")
    gather_prompt = prompts.gather.format(**prompt_args)
    initial_agent_chain = build_web_agent_func()
    initial_info_chunk = run_chain_with_retries(
        initial_agent_chain,
        retries,
        input=prompts.initial_web_agent.format(gather_prompt=gather_prompt),
    )
    knowledge_chunks = [initial_info_chunk]

    for _ in range(deep_dive_rounds):
        round_knowledge = "\n\n".join(knowledge_chunks)
        deep_dive_area_prompt = prompts.deep_dive_list.format(
            num_topics=deep_dive_topics, gather_prompt=gather_prompt,
            current_knowledge=round_knowledge, **prompt_args)
        deep_dive_list = model.invoke([HumanMessage(deep_dive_area_prompt)]).content
        try:
            deep_dive_areas = [v.split(". ", 1)[1] for v in deep_dive_list.strip().split("\n")]
        except IndexError:
            print("Failed to parse topics", deep_dive_list)
            break
        for deep_dive_topic in deep_dive_areas:
            deep_dive_web_agent_prompt = prompts.deep_dive_web_agent.format(
                gather_prompt=gather_prompt,
                current_knowledge=round_knowledge,
                deep_dive_topic=deep_dive_topic)
            agent_chain = build_web_agent_func()
            try:
                info_chunk = run_chain_with_retries(
                    agent_chain,
                    retries,
                    input=deep_dive_web_agent_prompt,
                )
                knowledge_chunks.append(info_chunk)
            except Exception as e:
                print("Agent failed", e)
    return knowledge_chunks
