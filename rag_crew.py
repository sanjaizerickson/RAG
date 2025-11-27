from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Agents ---
class ClassifierAgent:
    def __init__(self, llm, prompt_template: str):
        self.llm = llm
        self.prompt_template = prompt_template

    def classify(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        return self.llm.invoke(prompt).strip().lower()

class RetrieverAgent:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model

    def retrieve(self, query: str, label: str) -> str:
        retriever = FAISS.load_local(
            f"vectorstores/{label}_faiss",
            self.embedding_model,
            allow_dangerous_deserialization=True
        ).as_retriever()
        response = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever).invoke({"query": query})
        return f"📚 Answer from {label.capitalize()} Vector DB: {response['result']}"

class ChatAgent:
    def __init__(self, llm):
        self.llm = llm

    def chat(self, query: str) -> str:
        chat_prompt = f"You are a helpful assistant. Answer this: {query}"
        return f"💬 General Chat Answer: {self.llm.invoke(chat_prompt)}"

class ErrorAgent:
    def __init__(self, llm):
        self.llm = llm

    def handle_error(self, query: str, label: str) -> str:
        fallback_prompt = f"Something went wrong. Try answering anyway: {query}"
        return f"❌ Error: Invalid label '{label}'.\n🧪 Fallback Answer: {self.llm.invoke(fallback_prompt)}"

# --- Crew Manager ---
class CrewManager:
    def __init__(self, classifier_agent, retriever_agent, chat_agent, error_agent):
        self.classifier_agent = classifier_agent
        self.retriever_agent = retriever_agent
        self.chat_agent = chat_agent
        self.error_agent = error_agent

    def handle_query(self, user_query: str) -> str:
        label = self.classifier_agent.classify(user_query)
        if label in ["deforestation", "population", "ww2", "payslip"]:
            return self.retriever_agent.retrieve(user_query, label)
        elif label == "general_chat":
            return self.chat_agent.chat(user_query)
        else:
            return self.error_agent.handle_error(user_query, label)

# --- Setup Models ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="gemma3:4b", temperature=0.3)

classifier_prompt = """
You are a topic classifier.
Classify the user's query into one of: deforestation, population, ww2, payslip, general_chat
Respond with only the label.
Query: {query}
Label:
"""

# --- Instantiate Agents ---
classifier_agent = ClassifierAgent(llm, classifier_prompt)
retriever_agent = RetrieverAgent(llm, embedding_model)
chat_agent = ChatAgent(llm)
error_agent = ErrorAgent(llm)

crew_manager = CrewManager(classifier_agent, retriever_agent, chat_agent, error_agent)

if __name__ == "__main__":
    queries = ["What is the deforestation", "What is the global warming", "end of the world war?"]
    for q in queries:
        print(crew_manager.handle_query(q))