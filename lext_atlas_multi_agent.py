# =========================================
# DATA PROCESSING - FILE ORGANIZER
# =========================================

"""
from langchain_community.document_loaders import PyPDFLoader
# state
def classify_by_state(text: str) -> str:
    estados = ["New York", "Florida", "California"]
    for e in estados:
        if e.lower() in text.lower():
            return e
    return "Unknown"


# read all pdf docuemnts and detect whish US state
# reciebe 2 rutas: input = recibe sin clasifica, output: se crea subcarpetas po estado
def classify_documents(input_folder: str, output_folder: str):
    for file in os.listdir(input_folder):  # recorre archivos dentro de carpeta
        if file.endswith(".pdf"):
            file_path = os.path.join(input_folder, file)
            # PyPDFloader
            loader = PyPDFLoader(file_path)
            docs = loader.load()  # carga documento
            # extraer texto principal del primer documento
            if len(docs) > 0:
                text = " ".join([doc.page_content for doc in docs])

            # LLAMAR FUNCION EXTERNA
            estado = classify_by_state(text)
            dest_folder = os.path.join(
                output_folder, estado
            )  # crear la ruta de destino
            os.makedirs(dest_folder, exist_ok=True)  # si ya existe no la crea
            shutil.move(os.path.join(input_folder, file), dest_folder)
"""


# input_folder = r"C:\Users\grupo\OneDrive\Escritorio\HACKATHON\data_insurance"
# output_folder = r"C:\Users\grupo\OneDrive\Escritorio\HACKATHON\data_insurance\classified_data"
# sample_docs = classify_documents(input_folder, output_folder)

# =========================================
# IMPORT DATA
# =========================================
# Direction
import os


def get_all_pdf(folder_path):
    """Iterates through all subfolders of 'folder_path' and returns a list of the paths to all found PDF files."""
    all_pdfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                all_pdfs.append(file_path)

    return all_pdfs


output_folder = (
    r"C:\Users\grupo\OneDrive\Escritorio\HACKATHON\data_insurance\classified_data"
)
pdf_file = get_all_pdf(output_folder)


# Import
from langchain_community.document_loaders import PyPDFLoader


def load_documents_from_pdf(pdf_paths) -> list:
    documents = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        documents.extend(docs)
    return documents


documents = load_documents_from_pdf(pdf_file)  # <----------------------

# ============================================================================================================================================

# =========================================
# MODELS AZURE + EMBEDDINGS
# =========================================
import os, shutil
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
# Modelo
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    model_name="gpt-4",
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Embedding
embedding = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    model="text-embedding-3-large",
    api_version=os.getenv("OPENAI_EMBEDDING_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

memory = MemorySaver()


# =========================================
# VECTORE STORE
# =========================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# chunk - split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

# vectore store - embedding
vectore_store = FAISS.from_documents(all_splits, embedding)  # search FAISS
retriever = vectore_store.as_retriever()  # converts index into a retriever
print(retriever)


# =========================================
# GRAPH STATE - INPUT
# =========================================
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Optional, Literal, List
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)


# define graph state
class GraphState(TypedDict):
    messages: Optional[Annotated[List[BaseMessage], add_messages]]
    language: Optional[str]
    query: str
    states: str
    docs: Optional[list[str]]
    next: Optional[str]


def input_initial(state: GraphState) -> GraphState:
    return state


# NODE input collector - tranforma la pregutna a messges
def node_input_collector(state: GraphState) -> GraphState:
    try:
        query_str = state["query"].lower()
        if query_str in ["exit", "quit"]:
            return END
        else:
            input_content = (
                f"{state['query']} returns status information only {state['states']}"
            )
            new_user_message = HumanMessage(content=input_content)
            existing_messages = state.get("messages", [])
            updated_messages = existing_messages + [new_user_message]
            return {**state, "messages": updated_messages}
    except Exception as e:
        print(f"Error in node_input_collector: {e}")
        raise


# -----------------------------
# NODE input validation: FUNCTION SELECTOR
# nodo validator
def node_input_validation(
    state: GraphState,
) -> Command[Literal["rag", "researcher", END]]:
    try:
        # evaluate states
        if state["states"] in ["Florida", "New York", "California"]:
            goto = "rag"
            updated_state = {
                **state,
                "messages": [HumanMessage(content=state["query"])],
                "next": goto,
            }
            return Command(
                goto=goto,
                update=updated_state,
            )
        else:
            goto = "researcher"
            updated_state = {
                **state,
                "messages": [HumanMessage(content=state["query"])],
                "next": goto,
            }
            return Command(
                goto=goto,
                update=updated_state,
            )
    except Exception as e:
        print(f"Error in node_input_validation: {e}")


# =========================================
# PROMPT
# =========================================


def make_system_prompt(sufrix: str) -> str:
    return (
        f"{sufrix}"
        """\nYou are an agent specializing in retrieving state legal regulations in the area of ​​vehicle insurance using a Generation-Assisted Retrieval (GAR) system or web search. Follow the instructions below to accurately retrieve and deliver the information.

# Instructions

1. **Identify the state (`states`) specified by the user:**
- Ensure the value for `states` is one of the following: `New York`, `California`, or `Florida`.
- If the state is different from those mentioned or is specified as `Others`, use the alternative web search node, `node_researcher`.
- if use the alternative web search node, `node_researcher` it is mandatory that when you give your answer, you mention this text just as I wrote it at the beginning: A web search was performed to provide this information since we did not find any information for it in our database
2. **Search process:**
- If `states` is a valid state:
1. Access the indexed legal corpus using the GAR node.
2. Retrieve the most relevant state regulations related to the key values ​​in the provided documents.
- If `states` is different or equal to `Other`:
- Use the `node_researcher` web search node to find alternative regulations.

3. **Data Visualization:**
- If the user explicitly requests data visualization:
1. Complete the search process first through either the RAG or `node_researcher` node, depending on the specified state.
2. Generate charts or visualizations using the `node_chart_generator` node.

4. **Additional Validations:**
- Make sure to process any requests in the logical order of the nodes: first retrieve relevant information and then proceed with the visualization if requested.

# Output Format

Response Format:
Subject: [Query Topic]
Source Document: [Manual Name]
Response: [Clear, simple, and summarized technical explanation]
Legal Reference or Procedure (if applicable): [Section or article number cited]

If warranted, provide an optional summary of the results in the following **Markdown table format**:

| Title | Citation | Abstract |
|-----------------------|------------------------|----------------------------------------|
| [Regulation Title] | [Regulation Citation] | [2-3 sentence summary of regulation] |

If no regulations are found for the specified `state` and `project_type`, return an empty table with the following note **outside the table**:

`No relevant regulations were found for the specified state and project type`.

```
# Notes

- Perform a regulation search or web search first before proceeding with any requested visualization.
- In case of errors or missing data, explicitly state that no regulations or available graphics were found and provide alternative steps if applicable.
- Review that the nodes used align with state specifications and user requirements.
- Ensure the summary is concise and provides sufficient context about the regulation's relevance to the query.
- If there are additional instructions (e.g., timing, file structures, required formats), include them clearly.
- Never fabricate information. If you can't find the answer, reply: "According to the available manuals, that information is not directly specified. I recommend contacting via the web."
"""
    )


# =========================================
# Tools
# =========================================
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from matplotlib import pyplot as plt
import uuid


# 1.Tool RAG
def retrieve_rag(state: GraphState) -> GraphState:  # quick search
    """Search for relevant documents from a query using embeddings and FAISS-based retrieval."""
    query = (
        state["messages"][-1].content
        if isinstance(state["messages"][-1], BaseMessage)
        else state["messages"][-1]
    )
    docs = retriever.invoke(query)
    prompt = f"You will only provide exclusive information about the state of {state['states']}"
    prompt_human = HumanMessage(content=prompt)
    update_messages = state["messages"] + [prompt_human]
    return {
        **state,
        "messages": update_messages,
        "docs": [doc.page_content for doc in docs],
    }


# 2. Tool Search
search = TavilySearchResults()
# 3. Tool coder
repl = PythonREPL()
save_chart = r"C:\Users\grupo\OneDrive\Escritorio\HACKATHON\test_chart"


@tool
def python_repl_tool(code: Annotated[str, "The python code"]):
    """Dynamically executes Python code and returns the result or an error message."""
    try:
        image_path = None
        # Si contiene plt.show(), agregamos el guardado antes
        if "plt.show()" in code:
            filename = f"chart_{uuid.uuid4().hex[:8]}.png"
            full_path = os.path.join(save_chart, filename)
            code = code.replace(
                "plt.show()", f"plt.savefig(r'{full_path}')\nplt.show()"
            )
            image_path = full_path
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute: Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout:\n{result}"
    # if "plt.savefig" in code:
    # result_str += f"\n✅ Chart saved to: `{full_path}`"
    if image_path:
        result_str += f"\n✅ Chart saved to: `{image_path}`"

    # return result_str + "\nIf you have completed all tasks, respond with FINAL ANSWER"
    return {"text": result_str, "image": image_path}


# =========================================
# AGENT + NODE
# =========================================
from langgraph.prebuilt import create_react_agent


# function get next
def get_next_node(last_message: BaseMessage, goto: str):
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        if "FINAL ANSWER" in last_message.content.upper():
            return END
    return goto


# 1. Agent: RAG
# ----------------------------
# NODE rag
def node_retriever_rag(state: GraphState) -> Command[Literal["chart_generator", END]]:
    try:
        call_retrieve = retrieve_rag(state)

        def ensure_messages(messages):
            safe_messages = []
            for m in messages:
                if isinstance(m, BaseMessage):
                    safe_messages.append(m)
                else:
                    safe_messages.append(HumanMessage(content=m))
            return safe_messages

        human = ensure_messages(call_retrieve["messages"])
        prompt = f"Respond only with information that is in the document{call_retrieve['docs']}"
        result = llm.invoke(human + [HumanMessage(content=prompt)])

        # result = agent_retriever_rag(state)
        # result["messages"][-1] = HumanMessage(content=call_retrieve["messages"][-1].content, name="rag")
        updated_messages = call_retrieve["messages"] + [result]
        goto = get_next_node(result, "chart_generator")
        update_state = {**state, "messages": updated_messages, "next": goto}
        return Command(goto=goto, update=update_state)
    except Exception as e:
        print(f"Error in node_retriever_rag: {e}")
        raise


# 2. Agent: SEARCH
# ----------------------------
agent_searcher_web = create_react_agent(
    llm,
    tools=[search],
    prompt=make_system_prompt(
        "You can only do research, You can only investigate about the regularization of vehicle insurance in the state of the USA that they specify to you. At the beginning of the response, it mentions this: A web search was performed to provide this information since we did not find information for that status in our database."
    ),
    checkpointer=memory,
)


# NODE researcher
def node_researcher(state: GraphState) -> Command[Literal["chart_generator", END]]:
    """
    This node handles the research step:
    - It sends the current messages to the web search agent.
    - Updates the last message to reflect the researcher's findings.
    - Decides the next step based on whether a 'FINAL ANSWER' was detected.
    """
    try:
        result_messages = agent_searcher_web.invoke(state)

        messages = result_messages["messages"]
        last_message = (
            messages[-1] if messages else HumanMessage(content="Sin respuesta")
        )

        last_message.name = "researcher"
        goto = get_next_node(last_message, "chart_generator")
        update_state = {**state, "messages": messages, "next": goto}

        return Command(goto=goto, update=update_state)
    except Exception as e:
        print(f"Error in node_researcher: {e}")


# 3. Agent: CHART GENERATOR
# ----------------------------

agent_chart_generator = create_react_agent(
    llm,
    tools=[python_repl_tool],
    prompt=make_system_prompt(
        "You can only generate charts. You are working with a researcher colleague and retriever docs"
    ),
    checkpointer=memory,
)


# NODE chart generator
def node_chart_generator(
    state: GraphState,
) -> dict:
    try:
        result = agent_chart_generator.invoke(state)
        return result["messages"][-1]

    except Exception as e:
        print(f"Error in node_chart_generator: {e}")
        raise


# =========================================
# ORCHESTRATION
# =========================================
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda


graph = StateGraph(GraphState)
graph.add_node("input", RunnableLambda(input_initial))
graph.add_node("input_collector", RunnableLambda(node_input_collector))
graph.add_node("input_validation", RunnableLambda(node_input_validation))
graph.add_node("rag", RunnableLambda(node_retriever_rag))
graph.add_node("researcher", node_researcher)
graph.add_node("chart_generator", node_chart_generator)
# initial flow
graph.set_entry_point("input")
graph.add_edge("input", "input_collector")
graph.add_edge("input_collector", "input_validation")
# conditional
graph.add_conditional_edges(
    "input_validation",
    lambda state: state["next"],  # usa el campo next que ya asignaste
    {"rag": "rag", "researcher": "researcher"},
)
# nodes
graph.add_edge("rag", "chart_generator")
graph.add_edge("researcher", "chart_generator")
graph.add_edge("rag", "__end__")
graph.add_edge("researcher", "__end__")
graph.add_edge("chart_generator", "__end__")

# Compilacion
app = graph.compile()

"""
# test
test = {
    "query": "Give me a summary table on the regularizations of washigtong",
    "states": "Other",
    "messages": [],
}
question_test = app.invoke(test)
print(question_test["messages"][-1].content)
"""

# path = r"C:\Users\grupo\OneDrive\Escritorio\HACKATHON\graph\mermaid_graph.png"
"""
# =========================================
# GRAPH VISUALIZATION
# =========================================
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from IPython.display import Image, display

image_bytes = app.get_graph().draw_mermaid_png()
# saved image
with open(path, "wb") as f:
    f.write(image_bytes)
print("saved image'")
"""

# =====================================================================================================================================

# =========================================
# DEPLOYMENT
# =========================================
import gradio as gr
from gtts import gTTS
import tempfile
from gradio.themes.base import Base
import whisper
import uuid

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load Whisper model
whisper_model = whisper.load_model("base")


# Main chatbot function
def run_chatbot_with_audio(state, query, audio_path):
    try:
        if audio_path:
            # Transcribe using Whisper
            transcription = whisper_model.transcribe(audio_path)
            query_text = transcription["text"].strip()
        elif query:
            # use text query if no audio is provided
            query_text = query.strip()
        else:
            return "⚠️ Please provide either voice or text input.", None, None

        # Input for the agent
        input_state = {
            "states": state,
            "query": query_text,
            "messages": [],
            "docs": [],
        }

        result = app.invoke(input_state)

        # Handle both plain text and dict output (text + optional image)
        final_message = result["messages"][-1].content
        if isinstance(final_message, dict):
            reply = final_message.get("text", "")
            image_path = final_message.get("image", None)
        else:
            reply = final_message
            image_path = None

        # Convert reply to speech using gTTS
        tts = gTTS(reply)
        tts_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        tts.save(tts_file)

        return f'🗣️ You said: "{query_text}"\n🤖 Chatbot: {reply}', tts_file, image_path

    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}", None, None


# UI with Gradio
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center; font-size: 3em;'>🎙️ <strong>Multi Agent LexAtlas</strong></h1>
        <p style='text-align: center;'>A specialized agent for retrieving state-level legal regulations on auto insurance using AI.</p>
        """,
        elem_id="title-centered",
    )

    with gr.Row():
        with gr.Column():
            state = gr.Radio(
                choices=["Florida", "California", "New York", "Other"],
                label="🌎 Select State",
            )
            query = gr.Textbox(lines=2, label="💬 Optional Additional Query")
            audio = gr.Audio(
                label="🎤 Voice Input (.wav)", type="filepath", format="wav"
            )
            btn = gr.Button("Submit")

        with gr.Column():
            output_text = gr.Textbox(label="📜 Transcript and Response")
            output_audio = gr.Audio(label="🔊 Voice Response")
            output_image = gr.Image(label="📊 Generated Chart")

    btn.click(
        fn=run_chatbot_with_audio,
        inputs=[state, query, audio],
        outputs=[output_text, output_audio, output_image],
    )

demo.launch(share=True)
