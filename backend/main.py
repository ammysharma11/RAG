# main.py (FastAPI Backend) with Docstrings
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

PERSIST_DIR = "vectorstore_data"
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

vectorstores = {}

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

@tool
def retriever_tool(pdf_id: str, query: str) -> str:
    """
    Searches and returns relevant information from a specific PDF document
    stored in a Chroma vectorstore.

    Args:
        pdf_id (str): The ID of the PDF document (used to retrieve the correct vectorstore).
        query (str): The natural language query to search for within the document.

    Returns:
        str: A string containing the relevant document chunks or a message
             indicating no information was found.
    """
    if pdf_id not in vectorstores:
        return "No document found for the given ID."
    
    retriever = vectorstores[pdf_id].as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the document."
            
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
            
    return "\n\n".join(results)

tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    """
    Represents the state of the agent in the LangGraph.
    
    Attributes:
        messages (Annotated[Sequence[BaseMessage], add_messages]): A list of messages
                                                                  representing the conversation history.
        pdf_id (str): The identifier for the currently active PDF document.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_id: str

def should_continue(state: AgentState):
    """
    Determines whether the agent should continue by checking for tool calls in the last message.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        bool: True if the last message contains tool calls, False otherwise.
    """
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions based on the PDF document provided.
Use the retriever tool available to answer questions about the document's content. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """
    Invokes the Large Language Model (LLM) with the current conversation state.

    Args:
        state (AgentState): The current state of the agent, including messages.

    Returns:
        AgentState: The updated state with the LLM's response appended to the messages.
    """
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm_with_tools.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    """
    Executes tool calls specified in the LLM's response.

    Args:
        state (AgentState): The current state of the agent, including the LLM's response
                            with potential tool calls.

    Returns:
        AgentState: The updated state with the results of the tool calls appended as ToolMessages.
    """
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if t['name'] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            # --- START OF MODIFICATION ---
            # Create a dictionary of arguments for the tool
            # The 'query' comes from the LLM's tool_calls args
            # The 'pdf_id' comes from the agent's state
            tool_args = t['args'].copy() # Start with args provided by LLM
            tool_args['pdf_id'] = state['pdf_id'] # Add pdf_id from state

            result = tools_dict[t['name']].invoke(tool_args)
            # --- END OF MODIFICATION ---
            
            print(f"Result length: {len(str(result))}")
            
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}
def create_rag_graph():
    """
    Creates and compiles the LangGraph StateGraph for the RAG agent.

    Returns:
        CompiledGraph: The compiled LangGraph object ready for invocation.
    """
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    return graph.compile()

rag_agent = create_rag_graph()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file, processes it into chunks, and stores it in a Chroma vectorstore.

    Args:
        file (UploadFile): The PDF file to be uploaded.

    Raises:
        HTTPException: If the file is not a PDF or if an error occurs during processing.

    Returns:
        JSONResponse: A confirmation message and the unique ID assigned to the PDF.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        pdf_loader = PyPDFLoader(file_path)
        pages = pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        pages_split = text_splitter.split_documents(pages)

        pdf_id = file.filename.replace(".pdf", "")
        collection_name = f"pdf_collection_{pdf_id}"
        persist_directory_for_pdf = os.path.join(PERSIST_DIR, pdf_id)

        if not os.path.exists(persist_directory_for_pdf):
            os.makedirs(persist_directory_for_pdf)

        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory_for_pdf,
            collection_name=collection_name
        )
        vectorstores[pdf_id] = vectorstore
        return JSONResponse(content={"message": "PDF processed successfully", "pdf_id": pdf_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

@app.post("/chat/")
async def chat_with_pdf(data: dict):
    """
    Handles chat messages from the user, processes them with the RAG agent,
    and returns an AI response based on the uploaded PDF.

    Args:
        data (dict): A dictionary containing 'message' (user's query) and 'pdf_id'.

    Raises:
        HTTPException: If message or pdf_id is missing, or if the PDF is not processed,
                       or if an error occurs during chat processing.

    Returns:
        JSONResponse: A dictionary containing the AI's response.
    """
    user_input = data.get("message")
    pdf_id = data.get("pdf_id")

    if not user_input or not pdf_id:
        raise HTTPException(status_code=400, detail="Message and PDF ID are required.")
    
    if pdf_id not in vectorstores:
        raise HTTPException(status_code=404, detail="PDF not found or not processed. Please upload the PDF first.")

    messages = [HumanMessage(content=user_input)]
    
    try:
        result = rag_agent.invoke({"messages": messages, "pdf_id": pdf_id})
        return JSONResponse(content={"response": result['messages'][-1].content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {e}")
