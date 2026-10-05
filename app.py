import streamlit as st
import os
from hashlib import sha256
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


TRUSTED_INDEX_DIGESTS = {
    "index.faiss": "efd1495cd5c40bd13c5cee048b811f15fe0534ab54d995d8fa48b9c37c378b77",
    "index.pkl": "9e6877a6db976345802f138e8ed17eb795f997722d1ffb5d8b78c61fac0cccc1",
}


def verify_knowledge_base(index_dir: Path) -> None:
    """Refuse to deserialize a missing or modified bundled knowledge base."""
    for filename, expected_digest in TRUSTED_INDEX_DIGESTS.items():
        path = index_dir / filename
        if not path.is_file():
            raise ValueError(f"Required knowledge-base file is missing: {filename}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(
                f"Knowledge-base integrity check failed for {filename}. "
                "Restore it from a trusted checkout before continuing."
            )

# 1. Page Configuration
st.set_page_config(page_title="Hydro-Met AI Assistant", page_icon="🌧️", layout="wide")
st.title("🌧️ Hydro-Meteorological Research Assistant")
st.caption("Powered by FAISS Vector Database & OpenAI Engine")

# 2. Sidebar Configuration (Secure API Key Entry)
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    model_name = st.text_input(
        "OpenAI model",
        value=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        help="Change this if your account uses a different available chat model.",
    )
    st.markdown("---")
    st.markdown("This local AI assistant is trained on your specific academic knowledge base for complex terrain interpolation and moisture modelling.")

# Halt execution if no API key is provided
if not api_key:
    st.warning("👈 Please enter your OpenAI API Key in the sidebar to start.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# 3. Load Database (Cached so it runs fast)
@st.cache_resource
def load_knowledge_base():
    index_dir = Path("faiss_index")
    verify_knowledge_base(index_dir)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Pointing to the local faiss_index folder
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

try:
    vector_db = load_knowledge_base()
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
except Exception as e:
    st.error(f"❌ Database not found. Please ensure the 'faiss_index' folder is in the same directory as this app. Error: {e}")
    st.stop()

# 4. Initialize AI Engine
llm = ChatOpenAI(model=model_name, temperature=0)
system_prompt = (
    "You are an elite Hydro-Meteorological Data Scientist and AI Architect. "
    "Use the following retrieved context from high-impact scientific literature to answer the user's question. "
    "If the answer is not present in the context, clearly state that you do not have enough information. Do not hallucinate. "
    "Your response must be highly technical, structured, and you MUST cite your sources by mentioning the 'Source Paper' names provided in the context at the end of relevant sentences.\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

def format_docs(docs):
    return "\n\n".join([f"{doc.page_content}\n[Source Paper: {doc.metadata.get('source', 'Unknown').split('/')[-1]}]" for doc in docs])

rag_chain = {"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 5. Chat Interface setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Field
if user_query := st.chat_input("Ask a technical question about your research papers..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching FAISS Brain & Synthesizing Response..."):
            try:
                response = rag_chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"⚠️ Engine Error: {e}")





                
