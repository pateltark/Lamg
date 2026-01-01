from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

llm = init_chat_model("google_genai:gemini-2.0-flash")

llm.invoke("who are you?")

class chatstate(TypedDict):
    massage : Annotated[list[BaseMessage], add_messages]

def chatbot_n(state:chatstate):
    msg = state['massage']
    response = llm.invoke(msg).content
    return {"messages": [response]}

checkpointer = InMemorySaver()

build = StateGraph(chatstate)

build.add_node("chatbo_n", chatbot_n)

build.add_edge(START, "chatbot")
build.add_edge("chatbot", END)

chatbot = build.compile(checkpointer=checkpointer)
