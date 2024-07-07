from typing import Optional, Callable
from omegaconf import OmegaConf
from pathlib import Path
from langchain.agents import Tool
from langchain.chat_models.base import BaseChatModel

from llm_osint.link_scraping import scrape_naive, chunk_and_strip_html
from llm_osint.llm_map_reduce import map_reduce_texts

class ReadLinkWrapper:
    def __init__(
        self,
        scrapper_func: Optional[Callable] = scrape_naive,
        map_llm: BaseChatModel = None,
        reduce_llm: BaseChatModel = None,
        reduce_chunks: int = 100,
        **format_kwargs):
        """
        Args:
            scrapper_func: The function to scrape the link.
            map_llm: The LLM model to use for mapping.
            reduce_llm: The LLM model to use for reducing.
            reduce_chunks: The number of chunks to reduce at a time.
            format_kwargs: The format kwargs.
        """
        self.prompts = OmegaConf.load(Path(__file__).parent / "read_link.yaml")
        self.format_kwargs = format_kwargs
        self.scrapper_func = scrapper_func
        self.map_llm = map_llm
        self.reduce_llm = reduce_llm
        self.reduce_chunks = reduce_chunks

    def run(self, url: str) -> str:
        """
        Read the contents of a link, and extract the information with map/reduce.
        """
        if url.endswith(".pdf"):
            return "Cannot read links that end in pdf"
        chunks = chunk_and_strip_html(self.scrapper_func(url), 4000)
        format_args = {**self.format_kwargs, "link": url, "example_instructions": self.prompts.example_instructions}
        return map_reduce_texts(
            chunks,
            map_prompt=self.prompts.map.format(**format_args),
            reduce_prompt=self.prompts.reduce.format(**format_args),
            reduce_chunks=self.reduce_chunks,
            map_llm=self.map_llm,
            reduce_llm=self.reduce_llm
        )


def get_read_link_tool(**kwargs) -> Tool:
    read_link = ReadLinkWrapper(**kwargs)
    return Tool(
        name="Read Link",
        func=read_link.run,
        description="Useful to read and extract the contents of any link. "
        "The input should be a valid url starting with http or https.",
    )
