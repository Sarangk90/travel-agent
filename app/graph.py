from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import Command, interrupt

from app.agents.supervisor_agent import call_supervisor
from app.agents.flights_advisor_agent import call_flights_advisor
from app.agents.hotel_advisor_agent import call_hotel_advisor

_ = load_dotenv()


def human_node(
    state: MessagesState, config
) -> Command[Literal["hotel_advisor", "flights_advisor", "human"]]:
    """A node for collecting user input."""

    message = state["messages"][-1].content
    user_input = interrupt(value=message)

    # identify the last active agent
    # (the last active node before returning to human)
    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")

    active_agent = langgraph_triggers[0].split(":")[1]

    return Command(
        update={
            "messages": [
                HumanMessage(content=user_input),
            ]
        },
        goto=active_agent,
    )


builder = StateGraph(MessagesState)
builder.add_node("supervisor", call_supervisor)
builder.add_node("hotel_advisor", call_hotel_advisor)
builder.add_node("flights_advisor", call_flights_advisor)

builder.add_node("human", human_node)

builder.add_edge(START, "supervisor")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
