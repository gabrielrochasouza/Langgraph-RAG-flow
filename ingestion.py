from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions

load_dotenv()

urls = [
    "http://lilianweng.github.io/posts/2023-06-23-agent/",
    "http://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "http://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag_chroma",
    embedding=gemini_embeddings,
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag_chroma",
    persist_directory="./.chroma",
    embedding_function=gemini_embeddings,
).as_retriever()
