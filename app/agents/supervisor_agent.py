import datetime
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from app.tools.handoff_tool import make_handoff_tool

supervisor_tools = [
    make_handoff_tool(agent_name="flights_advisor"),
    make_handoff_tool(agent_name="hotel_advisor"),
]
model = ChatOpenAI(model="gpt-4o-2024-08-06")

supervisor = create_react_agent(
    model=model.bind_tools(supervisor_tools, parallel_tool_calls=False, strict=False),
    tools=supervisor_tools,
    prompt=(
        "You are a team supervisor for a travel agency managing a hotel and flights advisor."
        f"Today is {datetime.datetime.now().strftime('%Y-%m-%d')}. "
        "Whenever you receive request from human for first time, Greet them and provide them with options you can help with like hotel, flight booking and suggest iteinery."
        "For finding hotels, use hotel_advisor. "
        "For finding flights, use flights_advisor."
        "Transfer to only one agent (or tool) at a time, nothing more than one. Sending requests to multiple agents at a time is NOT supported"
        "Be very friendly and helpful to the user. Make sure to provide human-readable response before transferring to another agent. Do NOT transfer to another agent without asking human"
    ),
    name="supervisor",
)


def call_supervisor(
    state: MessagesState,
) -> Command[Literal["hotel_advisor", "human", "flights_advisor"]]:
    response = supervisor.invoke(state)
    return Command(update=response, goto="human")
