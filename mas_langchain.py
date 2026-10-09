
import os
from getpass import getpass

os.environ["GEMINI_API_KEY"]=getpass("Enter your Gemini API Key:")
os.environ["SERP_API_KEY"]=getpass("Enter your Serp API Key:")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper

llm=ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

search_tool=SerpAPIWrapper(
    serpapi_api_key=os.environ["SERP_API_KEY"],
    params={
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
    }
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Agent 1 - Researcher Agent
research_prompt=ChatPromptTemplate.from_messages([
    ("system","""
    You are a Research Analyst and you are supposed to
    research the user request using supplied web search tool.
    Include useful facts, source URLs etc.
    Never invent facts or citations"""),
    ("human","""
    user request:{request}
    web search results:{search_results}"""),
])

research_agent=research_prompt|llm|StrOutputParser()

query = "Compare and research about these two companies - Trello and Monday.com"
search_results = search_tool.run(query)

research_brief = research_agent.invoke({
    "request": query,
    "search_results": search_results
})

print(search_results)

#Agent 2 - Analysis Agent
analysis_prompt=ChatPromptTemplate.from_messages([
    ("system","""
    You are a business operations analyst and you are supposed to
    analyze the research brief against the user's objectives
    Identify key findings, potential risks, opportunities,
    missing information etc.
    Do not invent facts"""),
    ("human","""
    user request:{request}
    research brief:{research_brief}"""),
])

analysis_agent=analysis_prompt|llm|StrOutputParser()

#Agent 3 - Operational Actions Agent
operations_prompt=ChatPromptTemplate.from_messages([
    ("system","""
    You are a business operations manager and you are supposed to
    turn the analysis into a concise and practical action brief.

    In the brief, we can have the following:
    - Executive Summary
    - Key Findings
    - Risks
    - Recommendations and next steps

    Do not invent facts, deadlines, prices etc."""),
    ("human","""
    user request:{request}
    analysis:{analysis}"""),
])

operations_agent=operations_prompt|llm|StrOutputParser()

def run_workflow(request):
  #agent 1 - research using SerpAPI
  search_results = search_tool.run(request)
  research_brief = research_agent.invoke({
    "request": request,
    "search_results": search_results
  })

  #agent 2 - analyze research brief
  analysis = analysis_agent.invoke({
    "request": request,
    "research_brief": research_brief
  })

  #agent 3 - report
  report = operations_agent.invoke({
    "request": request,
    "analysis": analysis
  })

  print(report)

  return report

request="""Research demand for AI Literacy and AI Training among Indian Enterprises.
Find relevant industry reports, identify common enterprise usecases and recommend
three service offerings worth validating with potential customers"""

report=run_workflow(request)

