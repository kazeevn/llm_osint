from typing import List, Callable, Optional
from langchain_core.messages import HumanMessage
from langchain.agents.agent import AgentExecutor
from llm_osint import knowledge_agent_constants, llm
import logging

logger = logging.getLogger(__name__)

def run_chain_with_retries(agent_chain: AgentExecutor, retries: int, **agent_run_kwargs) -> str:
    if len(agent_run_kwargs['input']) > 16000:
        raise ValueError(f"Input too long\n{agent_run_kwargs['input'][:1000]}\n...\n{agent_run_kwargs['input'][-1000:]}")
    exception = None
    for _ in range(retries):
        try:
            output = agent_chain.invoke(agent_run_kwargs)
            return output['output']
        except Exception as e:
            exception = e
    raise exception


def run_knowledge_agent(
    gather_prompt: str,
    build_web_agent_func: Callable[[], AgentExecutor],
    deep_dive_topics: int,
    deep_dive_rounds: int,
    retries: Optional[int] = 3,
    model: Optional[llm.LLMModel] = None,
    **prompt_args
) -> List[str]:
    if model is None:
        model = llm.get_default_llm()
    initial_agent_chain = build_web_agent_func()
    initial_info_chunk = run_chain_with_retries(
        initial_agent_chain,
        retries,
        input=knowledge_agent_constants.INITIAL_WEB_AGENT_PROMPT.format(gather_prompt=gather_prompt),
    )
    knowledge_chunks = [initial_info_chunk]

    for _ in range(deep_dive_rounds):
        round_knowledge = "\n\n".join(knowledge_chunks)
        deep_dive_area_prompt = knowledge_agent_constants.DEEP_DIVE_LIST_PROMPT.format(
            num_topics=deep_dive_topics, gather_prompt=gather_prompt, current_knowledge=round_knowledge, **prompt_args
        )
        deep_dive_list = model.invoke([HumanMessage(deep_dive_area_prompt)]).content
        try:
            deep_dive_areas = [v.split(". ", 1)[1] for v in deep_dive_list.strip().split("\n")]
        except IndexError:
            print("Failed to parse topics", deep_dive_list)
            break
        for deep_dive_topic in deep_dive_areas:
            deep_dive_web_agent_prompt = knowledge_agent_constants.DEEP_DIVE_WEB_AGENT_PROMPT.format(
                gather_prompt=gather_prompt,
                current_knowledge=round_knowledge,
                deep_dive_topic=deep_dive_topic,
            )
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
