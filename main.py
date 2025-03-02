import uuid

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from langgraph.types import Command

from app.graph import graph


def invoke_graph(user_input: str, thread_id: str):
    thread_config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    input_data = Command(resume=user_input) if len(state.next) > 0 and state.next[0] == 'human' else {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    for update in graph.stream(input_data, config=thread_config, stream_mode="updates"):
        if '__interrupt__' in update:
            break
        else:
            for node_id, value in update.items():
                if "messages" in value and value["messages"]:
                    last_message = value["messages"][-1]
                    if last_message.type == "ai":
                        print(f"{node_id}: {last_message.content}")


if __name__ == '__main__':
    thread_id = uuid.uuid4()
    while True:
        user_input = input("Enter your message: ")
        result = invoke_graph(user_input, thread_id)
