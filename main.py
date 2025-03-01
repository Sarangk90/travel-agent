import uuid

from langchain_core.messages import HumanMessage
from app.graph import Agent

if __name__ == '__main__':
    user_input = input('Enter your travel query: ')
    agent = Agent()
    if user_input:
        try:
            thread_id = str(uuid.uuid4())

            messages = [HumanMessage(content=user_input)]
            config = {'configurable': {'thread_id': thread_id}}

            result = agent.graph.invoke({'messages': messages}, config=config)

            print('Travel Information:')
            print(result['messages'][-1].content)

        except Exception as e:
            print(f'Error: {e}')
    else:
        print('Please enter a travel query.')
