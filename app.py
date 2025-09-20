import os
from flask import Flask,render_template,request,session,url_for,redirect,abort

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
# from langchain import hub
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import httpx

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
app.secret_key = '_5#y2L"F4Q8z'

if not os.getenv("GEMINI_API_KEY"):
    raise Exception("GEMINI_API_KEY environment variable not set")

VALID_USERNAME = os.getenv("VALID_USERNAME", "ashu")
VALID_PASSWORD = os.getenv("VALID_PASSWORD", "a")

chat_history = []

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)

DATA_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
csv_file_name = "faq.csv"
faq_csv_path = os.path.join(DATA_FOLDER, csv_file_name)
try:
    loader = CSVLoader(file_path=faq_csv_path)
    docs = loader.load()
except Exception as e:
    print("Problem while loading file")
    docs=[]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  
    chunk_overlap=80,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)

template = """You are a helpful and professional support desk assistant.
Your main goal is to provide accurate answers based *only* on the provided context.
If the answer is NOT available in the provided context, state clearly and politely that you don't have enough information from the provided context to answer the question, and offer to assist with other queries or suggest contacting human support.
Do NOT make up information or use your general knowledge.
Keep your answers concise and directly address the user's question.
context:
{context}

question: {question}
answer:"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content}
    ).to_messages()
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()



@app.route("/")
def index():
    return render_template("login.html",status = True)

@app.route("/logout")
def logout():
    session.clear()
    return render_template("login.html",status = True)

@app.route("/auth/",methods = ["GET","POST"])
def auth():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            return render_template("login.html",status = False)

        if username.lower() == VALID_USERNAME.lower() and password == VALID_PASSWORD:
            session["logged_in"] = True
            session["username"] = username.lower()
            return redirect(url_for("bot"))
        else:
            return render_template("login.html",status = False)

@app.route("/bot",methods = ["GET","POST"])
def bot():
    if "username" not in session:
        abort(401)
    if not os.getenv("GEMINI_API_KEY"):
        return f"Problem while accessing api key"
    if request.method == 'POST':
        try:
            prompt = request.form["prompt"]
            if prompt :
                chat_history.append({"role": "user", "content": prompt})
                result = graph.invoke({"question": prompt})

                chat_history.append({"role": "bot", "content": result["answer"]})

                return redirect(url_for('bot'))

        except httpx.ConnectError as e:
            return f"Problem while accessing app {e}"

        if session["username"] is not None and session["logged_in"] == True:
            return render_template("chat.html",chat_history = chat_history)

        return redirect(url_for("/"))

    else:
        return render_template("chat.html",chat_history = chat_history)

if __name__ == "__main__":
    app.run(debug = True)
