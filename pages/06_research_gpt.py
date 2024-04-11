import streamlit as st
import os
from typing import Any, Type
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema import SystemMessage


st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔍🔎",
)

st.markdown(
    """
        # ResearchGPT

        Welcome to ResearchGPT.

        Write down the name of a theme and our Agent will do the resarch for you.
    """
)

llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-0125")


class TopicWikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search in Wikipedia for.")


class TopicWikipediaSearchTool(BaseTool):
    name = "TopicWikipediaSearchTool"
    description = """
        Use this tool to research for a theme.
        It takes a query as an argument.
        Example query: Research about the XZ backdoor
    """

    args_schema: Type[TopicWikipediaSearchToolArgsSchema] = (
        TopicWikipediaSearchToolArgsSchema
    )

    def _run(self, query):
        wpd = WikipediaAPIWrapper()
        return wpd.run(query)


class TopicDuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search in DuckDuckGo for.")


class TopicDuckDuckGoSearchTool(BaseTool):
    name = "TopicDuckDuckGoSearchTool"
    description = """
        Use this tool to research for a theme.
        It takes a query as an argument.
        Example query: Research about the XZ backdoor
    """

    args_schema: Type[TopicDuckDuckGoSearchToolArgsSchema] = (
        TopicDuckDuckGoSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        TopicWikipediaSearchTool(),
        TopicDuckDuckGoSearchTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a web surfer.
            You want to research a theme in Wikipedia or DuckDuckGo.
            If there is a website, go to the website, and write down the content in .txt file.          
        """
        )
    },
)


content = st.text_input("Write the name of the content you are interested on.")

if content:
    result = agent.invoke(content)

    with open("research.txt", "wb") as file:
        encoded = result["output"].encode("utf-8")
        file.write(encoded)

    st.write(result)
