import os
from pathlib import Path

import streamlit as st

from langchain_chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Same extension-dispatched loading as ingest.py, so dropping a PDF or an
# HTML export into data/ works here too, not just .md/.txt files.
LOADERS_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}

# What changed vs. the original:
#   - Hybrid retrieval: BM25 (keyword) + vector search, merged with an
#     EnsembleRetriever — same idea as the FastAPI app.py variant.
#   - Optional cross-encoder reranking, toggleable in the sidebar so a
#     slow/limited machine can skip the extra model download.
#   - Multi-turn conversation: a follow-up question is rewritten into a
#     standalone one using the chat history before retrieval runs.
#   - Answers stream token-by-token via st.write_stream instead of
#     appearing all at once.

# Page configuration
st.set_page_config(page_title="Nimbus Docs AI Assistant", page_icon="🤖")
st.title("🤖 Nimbus Docs Q&A Assistant")
st.caption("A grounded RAG demo powered by Google Gemini & LangChain")

with st.sidebar:
    st.header("⚙️ Configuration")

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    api_key = st.text_input(
        "Enter Google Gemini API Key:",
        type="password",
        key="gemini_api_key",
        help="Get a free key from Google AI Studio",
    )

    st.markdown(
        "[Get a free Gemini API key](https://aistudio.google.com/)"
    )

    rerank_enabled = st.checkbox(
        "Rerank retrieved chunks",
        value=True,
        help="Uses a small local cross-encoder to re-score retrieved chunks "
        "by relevance before they go to the model.",
    )

    if st.button("🚀 Start Assistant", type="primary"):
        if not st.session_state.gemini_api_key.strip():
            st.error("Please enter your Gemini API key.")
        else:
            st.session_state.api_key_ready = True
            st.rerun()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner="Indexing documentation into vector store...")
def get_retriever(api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", google_api_key=api_key
    )

    # Load every supported file in the data directory (previously .md only)
    docs = []
    data_path = Path(__file__).parent / "data"
    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue
        loader_cls = LOADERS_BY_SUFFIX.get(file_path.suffix.lower())
        if loader_cls is None:
            continue
        loader = (
            loader_cls(str(file_path), encoding="utf-8")
            if loader_cls is TextLoader
            else loader_cls(str(file_path))
        )
        for doc in loader.load():
            doc.metadata["source"] = file_path.name
            docs.append(doc)

    # Split documents into overlapping semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )
    splits = text_splitter.split_documents(docs)

    vector_retriever = Chroma.from_documents(
        documents=splits, embedding=embeddings
    ).as_retriever(search_kwargs={"k": 6})

    if not splits:
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 6

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
    )


@st.cache_resource(show_spinner=False)
def get_reranker():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as exc:  # missing package, no internet to fetch weights, etc.
        st.warning(f"Reranking unavailable, continuing without it ({exc}).")
        return None


def rerank(reranker, question: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    if not reranker or not docs:
        return docs[:top_n]
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def format_history(messages: list[dict]) -> str:
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    if not turns:
        return "(no previous turns)"
    lines = []
    for msg in turns:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


# Halt execution until the user provides an API key
if not st.session_state.get("api_key_ready", False):
    st.info(
        "👈 Enter your Gemini API key and click "
        "**Start Assistant**.",
        icon="🔑",
    )
    st.stop()

api_key = st.session_state.gemini_api_key.strip()

# Initialize retriever, reranker, and generation/condense chains
retriever = get_retriever(api_key)
reranker = get_reranker() if rerank_enabled else None

llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.6-flash", google_api_key=api_key, temperature=0
)

system_prompt = """You answer questions about Nimbus using ONLY the context below.
If the answer isn't in the context, say so plainly — never guess or use outside knowledge.
Be concise and factual.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{question}")]
)
rag_chain = prompt | llm | StrOutputParser()

condense_prompt = ChatPromptTemplate.from_template(
    """Given the conversation so far and a new follow-up question, rewrite the
follow-up as a standalone question that makes sense without the conversation.
If the follow-up is already standalone, return it unchanged. Return ONLY the
rewritten question, nothing else.

Conversation so far:
{chat_history}

Follow-up question: {question}
Standalone question:"""
)
condense_chain = condense_prompt | llm | StrOutputParser()

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📚 Sources: {', '.join(msg['sources'])}")

# Handle new user input
if user_question := st.chat_input("Ask a question about Nimbus..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            standalone_question = (
                condense_chain.invoke(
                    {
                        "chat_history": format_history(st.session_state.messages[:-1]),
                        "question": user_question,
                    }
                ).strip()
                if len(st.session_state.messages) > 1
                else user_question
            )

            retrieved_docs = retriever.invoke(standalone_question)
            retrieved_docs = rerank(reranker, standalone_question, retrieved_docs)

            context = "\n\n".join(
                f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
                for doc in retrieved_docs
            )
            sources = sorted(
                {doc.metadata.get("source", "unknown") for doc in retrieved_docs}
            )

        response = st.write_stream(
            rag_chain.stream({"context": context, "question": standalone_question})
        )
        if sources:
            st.caption(f"📚 Sources: {', '.join(sources)}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )
