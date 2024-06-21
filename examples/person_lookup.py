import re
from functools import partial
import importlib
import argparse
from llm_osint.tools.search import get_search_tool
from llm_osint.tools.read_link import get_read_link_tool
from llm_osint import knowledge_agent, web_agent, cache_utils, llm

SCRAPING_INSTRUCTIONS = """
For example
- name:
- location:
- age:
- job title:
- projects:
- hobbies:
- people they know:
- common activities:
- interesting personal links:
- key dates:
- coworkers:
- friends:
"""

GATHER_PROMPT = """
Learn as much as possible about {name} like their job, hobbies, common daily activities, friends, social media, interests, personality traits, preferences, and aspirations.
"""

ASK_PROMPT = """
Given these details about {name}.

---
{internet_content}
---

{question}
"""


def build_web_agent(name, scraper_func):
    tools = [get_search_tool(),
             get_read_link_tool(name=name, example_instructions=SCRAPING_INSTRUCTIONS, scrapper_func=scraper_func)]
    return web_agent.build_web_agent(tools)


@cache_utils.cache_func
def fetch_internet_content(name, scraper_func) -> str:
    knowledge_chunks = knowledge_agent.run_knowledge_agent(
        GATHER_PROMPT.format(name=name),
        build_web_agent_func=partial(build_web_agent, name=name, scraper_func=scraper_func),
        deep_dive_topics=1,
        deep_dive_rounds=1,
        name=name,
    )
    return "\n\n".join(knowledge_chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--ask", type=str)
    parser.add_argument("--scraper", type=str, default="scrape_text")
    args = parser.parse_args()
    scraper_func = getattr(importlib.import_module("llm_osint.link_scraping"), args.scraper)
    fn = re.sub(r"[^\w]", "", args.name).lower() + ".txt"

    content = fetch_internet_content(args.name, scraper_func)
    with open(fn, "w", encoding="utf-8") as f:
        f.write(content)

    if args.ask:
        model = llm.get_default_llm()
        print(model.call_as_llm(ASK_PROMPT.format(name=args.name, internet_content=content, question=args.ask)))


if __name__ == "__main__":
    main()
