from typing import TypedDict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph 

# --- Setup Models ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="gemma3:4b", temperature=0.3)

classifier_prompt = PromptTemplate.from_template("""
You are a topic classifier.
Classify the user's query into one of: deforestation, population, ww2, payslip, general_chat
Respond with only the label.
Query: {query}
Label:
""")
classifier_chain = classifier_prompt | llm

# --- Define State ---
class RAGState(TypedDict, total=False):
    query: str
    label: str
    result: str

graph = StateGraph(RAGState)

# --- Nodes ---
def classify_node(state: RAGState) -> RAGState:
    query = state["query"]
    label = classifier_chain.invoke({"query": query}).strip().lower()
    state["label"] = label
    return state

def retrieve_node(state: RAGState) -> RAGState:
    label = state["label"]
    retriever = FAISS.load_local(
        f"vectorstores/{label}_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    ).as_retriever()
    response = RetrievalQA.from_chain_type(llm=llm, retriever=retriever).invoke({"query": state["query"]})
    state["result"] = f"📚 Answer from {label.capitalize()} Vector DB: {response['result']}"
    return state

def chat_node(state: RAGState) -> RAGState:
    query = state["query"]
    state["result"] = f"💬 General Chat Answer: {llm.invoke(f'You are a helpful assistant. Answer this: {query}')}"
    return state

def error_node(state: RAGState) -> RAGState:
    query = state["query"]
    state["result"] = f"❌ Error: Invalid label '{state.get('label')}'.\n🧪 Fallback Answer: {llm.invoke(f'Something went wrong. Try answering anyway: {query}')}"
    return state

# --- Graph setup ---
graph.add_node("classify", classify_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("chat", chat_node)
graph.add_node("error", error_node)

graph.set_entry_point("classify")

def branch_edge(state: RAGState) -> str:
    label = state["label"]
    if label in ["deforestation", "population", "ww2", "payslip"]:
        return "retrieve"
    elif label == "general_chat":
        return "chat"
    else:
        return "error"

graph.add_conditional_edges("classify", branch_edge, {"retrieve": "retrieve", "chat": "chat", "error": "error"})
graph.set_finish_point("retrieve")
graph.set_finish_point("chat")
graph.set_finish_point("error")

app = graph.compile()

def hybrid_qa_pipeline_langgraph(user_query: str) -> str:
    state = {"query": user_query}
    final_state = app.invoke(state)
    return final_state["result"]

if __name__ == "__main__":
    for q in ["What is the deforestation", "What is the global warming", "end of the world war?"]:
        print(hybrid_qa_pipeline_langgraph(q))