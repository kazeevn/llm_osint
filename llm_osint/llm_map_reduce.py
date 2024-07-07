from typing import List, Union
import asyncio
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage
from llm_osint import cache_utils


@cache_utils.cache_func
def reduce_(prompt: str, texts: List[str], model: BaseChatModel) -> str:
    return model.invoke([HumanMessage(prompt.format(texts="\n\n".join(texts)))]).content

async def map_to_texts(
    texts: List[str],
    map_prompt: str,
    model: BaseChatModel) -> List[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(model.ainvoke([HumanMessage(map_prompt.format(text=text))])) for text in texts]
    return [x.result().content for x in tasks]

def map_reduce_texts(
    texts: List[str],
    map_prompt: Union[str, None],
    reduce_prompt: str,
    reduce_chunks: int,
    map_llm: BaseChatModel,
    reduce_llm: BaseChatModel) -> str:
    if map_prompt is None:
        mapped_texts = texts
    else:
        mapped_texts = asyncio.run(map_to_texts(texts, map_prompt, map_llm))
    while len(mapped_texts) > 1:
        reduced_chunks = []
        while len(mapped_texts) > 0 and len(reduced_chunks) < reduce_chunks:
            reduced_chunks.append(mapped_texts.pop(0))
        mapped_texts.append(reduce_(reduce_prompt, reduced_chunks, reduce_llm))

    return mapped_texts[0]
