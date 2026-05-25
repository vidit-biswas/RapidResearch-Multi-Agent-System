from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, AgentType

from tools import web_search, scrape_url



import streamlit as st

# Model setup
llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0,
    api_key=st.secrets["MISTRAL_API_KEY"]
)
# First agent
def build_search_agent():

    agent = initialize_agent(
        tools=[web_search],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent

# Second agent
def build_reader_agent():

    agent = initialize_agent(
        tools=[scrape_url],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent

# Writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful reports."
    ),

    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings
- Conclusion
- Sources

Be detailed, factual and professional."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic."
    ),

    (
        "human",
        """Review the research report below.

Report:
{report}

Respond in this format:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

Verdict:
..."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()