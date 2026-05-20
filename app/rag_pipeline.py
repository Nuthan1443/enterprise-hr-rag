import os
import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
        self.qa_chain = None

        # LangSmith tracing (optional)
        if os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv(
                "LANGCHAIN_PROJECT", "order-intelligence-rag"
            )
            logger.info("LangSmith tracing enabled.")
        else:
            logger.info("LangSmith tracing disabled.")

        # Embeddings — local, no API cost
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded.")

        # Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required.")

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            temperature=0.0,   # Zero temp for factual SOP answers
            max_tokens=1024,
        )
        logger.info(f"Groq LLM initialized: {os.getenv('GROQ_MODEL', 'llama3-8b-8192')}")

    def ingest_documents(self):
        """Load, chunk, embed, and persist documents into ChromaDB."""
        chroma_path = str(self.vectorstore_dir / "chroma_db")

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
                chunk_size=400,
                chunk_overlap=80,
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
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",          # Maximal Marginal Relevance
            search_kwargs={
                "k": 5,                 # Return top 5 chunks
                "fetch_k": 15,          # Fetch 15, then diversify to 5
                "lambda_mult": 0.7      # 0 = max diversity, 1 = max relevance
            }
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": ORDER_PROMPT}
        )
        logger.info("RetrievalQA chain assembled.")

    def query(self, question: str) -> dict:
        """Run a question through the RAG pipeline and return answer + sources."""
        if not self.qa_chain:
            raise RuntimeError("Pipeline not ready. Call ingest_documents() first.")

        logger.info(f"Query received: {question}")
        result = self.qa_chain.invoke({"query": question})

        # Return first 200 chars of each source chunk for auditability
        source_chunks = [
            doc.page_content[:200] + "..."
            for doc in result.get("source_documents", [])
        ]

        return {
            "answer": result["result"],
            "source_chunks": source_chunks,
            "model_used": os.getenv("GROQ_MODEL", "llama3-8b-8192")
        }
