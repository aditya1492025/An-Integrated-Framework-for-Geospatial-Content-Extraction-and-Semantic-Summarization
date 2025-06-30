import streamlit as st
import os
import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever
from embeddings import get_embedding_function

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_resource
def initialize_embedding_and_db():
    embed_model = get_embedding_function()
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("pdf_chunks")
        client.delete_collection("web_chunks")
        pdf_collection = Chroma(client=client, collection_name="pdf_chunks", embedding_function=embed_model)
        web_collection = Chroma(client=client, collection_name="web_chunks", embedding_function=embed_model)
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB collections: {e}")
        return embed_model, None, None
    return embed_model, pdf_collection, web_collection

@st.cache_resource
def initialize_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_path = os.path.join(MODEL_DIR, "mistral-7b-instruct-v0.2")
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} to {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_DIR)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
    else:
        print(f"Loading {model_name} from {model_path}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1000,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False
    )
    
    return HuggingFacePipeline(pipeline=text_pipeline)

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Custom retriever class to combine PDF and web data
class CombinedRetriever(BaseRetriever):
    def __init__(self, pdf_retriever, web_retriever=None):
        super().__init__()
        self._pdf_retriever = pdf_retriever
        self._web_retriever = web_retriever
        print("CombinedRetriever initialized with:", {"pdf_retriever": bool(self._pdf_retriever), "web_retriever": bool(self._web_retriever)})

    def _get_relevant_documents(self, query: str) -> list:
        if not hasattr(self, '_pdf_retriever'):
            st.error("PDF retriever attribute missing.")
            raise AttributeError("CombinedRetriever is missing required pdf_retriever attribute.")
        
        pdf_docs = self._pdf_retriever.get_relevant_documents(query)
        for doc in pdf_docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = "pdf"
        
        if self._web_retriever:
            web_docs = self._web_retriever.get_relevant_documents(query)
            for doc in web_docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = "web"
            return pdf_docs + web_docs
        return pdf_docs

# LangChain RAG Setup
PROMPT_TEMPLATE = """
Using the provided context, identify and list exactly two urgent issues for the project, each with its own solution, in this format:

1. [Describe the most urgent issue based on the context] [Ref: chunk_x]
   Solution: [Provide a practical solution for issue 1]
2. [Describe the second most urgent issue based on the context] [Ref: chunk_x]
   Solution: [Provide a practical solution for issue 2]

- Extract issues and solutions directly from the context.
- The context includes chunks labeled with their source (e.g., [pdf] or [web]).
- If the context is empty or lacks issues, state "No urgent issues identified" and provide no solutions.
- Ensure the two issues are distinct and avoid repetition.

Context:
{context}

Question:
{question}
"""

PPT_PROMPT_TEMPLATE = """
Using the provided context, generate exactly two self-explanatory insights for a PowerPoint presentation in this format:

1. [A concise insight derived from the document context]
2. [Another concise insight derived from the document context]

- Base insights solely on the context.
- The context includes chunks labeled with their source (e.g., [pdf] or [web]).
- If the context is empty, state "No insights available due to lack of document content".
- Ensure insights are unique, relevant to the document, and suitable for a presentation.

Context:
{context}

Question: What are key insights from the document for a presentation?
"""

def setup_rag_chain(llm, pdf_collection, web_collection, prompt_template):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    pdf_retriever = pdf_collection.as_retriever(search_kwargs={"k": 5})  # 5 from PDF
    web_retriever = web_collection.as_retriever(search_kwargs={"k": 2}) if web_collection else None  # 2 from web if available
    
    combined_retriever = CombinedRetriever(pdf_retriever=pdf_retriever, web_retriever=web_retriever)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=combined_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

def query_rag(query_text: str, pdf_collection, web_collection, llm, project_title: str, prompt_template=PROMPT_TEMPLATE) -> str:
    with st.spinner("Retrieving relevant documents from PDF and web data..."):
        chain = setup_rag_chain(llm, pdf_collection, web_collection, prompt_template)
        result = chain({"query": query_text})
    
    context = "\n\n---\n\n".join([f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}" for doc in result["source_documents"]])
    if not context.strip():
        st.warning("No relevant context retrieved from PDF or web data.")
    else:
        print("Retrieved context includes data from:", [doc.metadata.get("source", "unknown") for doc in result["source_documents"]])
    
    response = result["result"].strip()
    return response