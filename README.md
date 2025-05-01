# **MULTI AGENT LEXATLAS LANGCHAIN**

With the main version built using the Azure AI Agent SDK and the semantic kernel, we developed a prototype with **LangChain**. This allowed us to evaluate differences in orchestration, agent chaining behavior, and immediate effectiveness across frameworks, providing valuable insights into performance and modularity trade-offs.


---


## 🚀 Deployment

**(Attach interface image)**


---


## 🧠 Graph

**(ATTACH IMAGE)**

The multi-agent **LexAtlas** LangChain version has the following structure:

1. `input`: receives information via the input.
2. `input_collector`: collects data into a dictionary, as it receives the user's message along with the selected state option.
3. `input_validation`: validates responses based on the provided conditions.
4. **RAG Agent**: if the selected option is within (`Florida`, `New York`, or `California`), this agent provides an answer or, if suggested by the prompt, generates a chart by directing to the `graph_generator` node.
5. **Researcher Agent**: if the selected option is "other", it performs a web search and returns output. It will only generate a chart using the `graph_generator` node if prompted.
6. **graph_generator Agent**: generates charts only if prompted by `prompt` or `researcher`.


---


## 🛠️ Frameworks, Libraries, and Tools

| Category                  | Tool/Library                                                    |
|---------------------------|------------------------------------------------------------------|
| Document Processing        | `langchain_community.document_loaders.PyPDFLoader`              |
| LLMs and Embeddings        | `AzureChatOpenAI`, `AzureOpenAIEmbeddings`                      |
| Vector Store               | `FAISS` (via LangChain)                                         |
| State Graph                | `LangGraph`                                                     |
| Embeddings                 | `text-embedding-3-large` (Azure OpenAI)                         |
| Web Retrieval              | `TavilySearchResults`                                           |
| Optional Visualization     | `matplotlib`, `PythonREPL`                                     |
| Utilities                  | `dotenv`, `os`, `shutil`, `uuid`                                |
| Deployment                 | `gradio                                                         |
