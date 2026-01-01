from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Load model
llm = init_chat_model("google_genai:gemini-2.0-flash")

# Test basic call
# print(llm.invoke("who are you?").content)

# Define State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node function
def chatbot_node(state: ChatState):
    msgs = state["messages"]
    response = llm.invoke(msgs).content
    return {"messages": [AIMessage(content=response)]}

# Checkpointing
checkpointer = InMemorySaver()

# Build graph
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

chatbot = builder.compile(checkpointer=checkpointer)