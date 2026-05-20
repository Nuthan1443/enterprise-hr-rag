import os
import logging
from pathlib import Path

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# ── Prompt Template ────────────────────────────────────────────────────────────
ORDER_PROMPT_TEMPLATE = """You are an expert Order Management Assistant for an e-commerce operations team.
Your knowledge comes from internal SOPs, carrier SLA documents, and customer support runbooks.

Rules:
- Answer strictly from the provided context. Do not invent policies or SLAs.
- If the answer involves a specific number (SLA hours, refund days, penalty amount), state it explicitly.
- If the context does not contain enough information, say: "This information is not available in the current documentation. Please check with the operations team directly."
- Be concise and actionable — support agents need fast, clear answers.
- If a runbook step is relevant, summarize the key steps clearly.

Context:
{context}

Question: {question}

Answer:"""

ORDER_PROMPT = PromptTemplate(
    template=ORDER_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


class RAGPipeline:
    def __init__(self):
        self.data_dir = Path("data")
        self.vectorstore_dir = Path("vectorstore")
        self.vectorstore = None
        self.retriever = None
        self.embeddings = None
        self.llm = None
        self.qa_chain = None
        self.pipeline_ready = False

        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L3-v2"
        )
        self.groq_model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # LangSmith tracing (optional)
        if os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv(
                "LANGCHAIN_PROJECT", "order-intelligence-rag"
            )
            logger.info("LangSmith tracing enabled.")
        else:
            logger.info("LangSmith tracing disabled.")

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required.")

    def _init_embeddings(self):
        if self.embeddings is not None:
            return

        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded.")

    def _init_llm(self):
        if self.llm is not None:
            return

        from langchain_groq import ChatGroq

        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model_name=self.groq_model_name,
            temperature=0.0,
            max_tokens=1024,
        )
        logger.info("Groq LLM initialized: %s", self.groq_model_name)

    def ingest_documents(self):
        """Load, chunk, embed, and persist documents into ChromaDB."""
        self._init_embeddings()

        chroma_path = str(self.vectorstore_dir / "chroma_db")

        from langchain_community.document_loaders import TextLoader, DirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        if Path(chroma_path).exists():
            logger.info("Existing vectorstore found — loading...")
            self.vectorstore = Chroma(
                persist_directory=chroma_path,
                embedding_function=self.embeddings
            )
            logger.info("Vectorstore loaded from disk.")
        else:
            logger.info("No vectorstore found — building from documents...")
            loader = DirectoryLoader(
                str(self.data_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} document(s) from {self.data_dir}.")

            # Chunking strategy: smaller chunks for precise SLA/policy retrieval
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=256,
                chunk_overlap=48,
                separators=["\n\n", "\n", ". ", " "]
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks.")

            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=chroma_path
            )
            logger.info(f"Vectorstore built and persisted at {chroma_path}.")

        self._build_chain()

    def _build_chain(self):
        """Assemble the RetrievalQA chain with MMR retrieval."""
        # Configure retriever (MMR) — we'll call it from `query()` directly
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={
                "k": 5,  # Return top 5 chunks
                "fetch_k": 15,  # Fetch 15, then diversify to 5
                "lambda_mult": 0.7,  # 0 = max diversity, 1 = max relevance
            },
        )

        # `RetrievalQA` was removed in LangChain v1; instead we perform
        # retrieval + direct LLM invocation in `query()` to keep compatibility.
        self.qa_chain = None
        logger.info("Retriever configured for manual QA flow.")

    def query(self, question: str) -> dict:
        """Run a question through the RAG pipeline and return answer + sources."""
        if not self.pipeline_ready:
            self.ingest_documents()
            self.pipeline_ready = True

        logger.info(f"Query received: {question}")

        # 1) Retrieve top documents (try retriever API, fall back to vectorstore search)
        try:
            docs = self.retriever.get_relevant_documents(question)
        except Exception:
            docs = self.vectorstore.similarity_search(question, k=5)

        # 2) Build context from retrieved docs
        context = "\n\n".join([d.page_content for d in docs[:5]])

        # 3) Format prompt and invoke LLM directly (avoids deprecated RetrievalQA)
        prompt_text = ORDER_PROMPT_TEMPLATE.format(context=context, question=question)

        self._init_llm()
        ai_msg = self.llm.invoke([("human", prompt_text)])

        # Extract text content from the returned AIMessage-like object
        answer_text = getattr(ai_msg, "content", str(ai_msg))

        # Return truncated source chunks for auditability
        source_chunks = [doc.page_content[:200] + "..." for doc in docs]

        return {
            "answer": answer_text,
            "source_chunks": source_chunks,
            "model_used": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        }
