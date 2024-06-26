from typing import List, Optional, Any, Dict
from pathlib import Path
from langchain.agents import Tool
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain import hub
from langchain.callbacks.stdout import StdOutCallbackHandler
from llm_osint import llm

from langchain_core.utils import print_text
from langchain_core.agents import AgentAction, AgentFinish

class PrettyPrinter(StdOutCallbackHandler):
    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print_text('\n' + action.log, color=color or self.color)


def build_web_agent(tools: List[Tool], memory: Optional[BaseChatMemory] = None) -> AgentExecutor:    
    system_prompt = PromptTemplate.from_file(Path(__file__).parent / "web_agent_prompt.txt")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.template),
        ("human", "{input}\n\n{agent_scratchpad}")
    ])
    #prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm.get_default_llm(),
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
        #callbacks=[PrettyPrinter()]
    )
    return agent_chain
