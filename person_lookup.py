import re
from functools import partial
import importlib
import argparse
from pathlib import Path
import logging
from omegaconf import OmegaConf
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from llm_osint import knowledge_agent, web_agent, cache_utils


@cache_utils.cache_func
def fetch_internet_content(name: str, config: OmegaConf) -> str:
    """
    Search the internet for information about a person and return the gathered information.
    Uses web search and web page reading tools to gather information in ReAct paradigm. Prompts are
    defined in the config file and heavily influence the information gathered.

    Args:
        name: The name of the person
        config: The configuration
    Returns:
        The gathered information.
    """
    scraper_func = getattr(importlib.import_module("llm_osint.link_scraping"), config.web_agent.scraper)
    model = ChatOpenAI(**config.llm)
    web_reader_llm = ChatOpenAI(**config.web_agent.web_reader_llm)
    knowledge_chunks = knowledge_agent.run_knowledge_agent(
        build_web_agent_func=partial(
            web_agent.build_web_agent,
            name=name,
            scraper_func=scraper_func,
            llm=model,
            web_reader_llm=web_reader_llm),
        deep_dive_topics=config.web_agent.deep_dive_topics,
        deep_dive_rounds=config.web_agent.deep_dive_rounds,
        model=model,
        name=name
    )
    return "\n\n".join(knowledge_chunks)

ASK_PROMPT = r"""Given these details about {name}.

  ---
  {internet_content}
  ---

  {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--ask", type=str)
    parser.add_argument("--log-level", type=str, default=logging.WARNING)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    config = OmegaConf.load(Path(__file__).parent / "config.yaml")
   
    content = fetch_internet_content(args.name, config)
    file_name = re.sub(r"[^\w]", "", args.name).lower() + ".txt"
    with open(Path("internet_content", file_name), "wt", encoding="utf-8") as f:
        f.write(content)

    if args.ask:
        model = ChatOpenAI(**config.llm)
        print(model.invoke([HumanMessage(
            ASK_PROMPT.format(
                name=args.name, internet_content=content, question=args.ask))]).content)


if __name__ == "__main__":
    main()
