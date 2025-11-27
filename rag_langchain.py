from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

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

# New pipeline style: prompt | llm
classifier_chain = classifier_prompt | llm


def hybrid_qa_pipeline(user_query: str) -> str:
    try:
        label = classifier_chain.invoke({"query": user_query}).strip().lower()

        if label in ["deforestation", "population", "ww2", "payslip"]:
            retriever = FAISS.load_local(
                f"vectorstores/{label}_faiss",
                embedding_model,
                allow_dangerous_deserialization=True
            ).as_retriever()
            response = RetrievalQA.from_chain_type(llm=llm, retriever=retriever).invoke({"query": user_query})
            return f"📚 Answer from {label.capitalize()} Vector DB: {response['result']}"

        elif label == "general_chat":
            chat_prompt = f"You are a helpful assistant. Answer this: {user_query}"
            return f"💬 General Chat Answer: {llm.invoke(chat_prompt)}"

        else:
            raise ValueError(f"Invalid label from classifier: {label}")

    except Exception as e:
        fallback_prompt = f"Something went wrong. Try answering anyway: {user_query}"
        return f"❌ Error: {e}\n🧪 Fallback Answer: {llm.invoke(fallback_prompt)}"


if __name__ == "__main__":
    for q in ["What is the deforestation", "What is the global warming", "end of the world war?"]:
        print(hybrid_qa_pipeline(q))