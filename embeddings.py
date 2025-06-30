import streamlit as st
import torch
import os
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the local model path and model name
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "local_models", "nomic-embed-text-v1")
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"

@st.cache_resource
def get_embedding_function():
    """Initialize and return the embedding function, downloading the model if not present locally."""
    # Check if the local model directory exists and has required files
    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
    model_exists = os.path.exists(LOCAL_MODEL_PATH) and all(
        os.path.exists(os.path.join(LOCAL_MODEL_PATH, f)) for f in required_files
    )

    # If the model doesn't exist locally, download it
    if not model_exists:
        st.info(f"Local model not found at '{LOCAL_MODEL_PATH}'. Downloading {MODEL_NAME}...")
        try:
            # Download the model and tokenizer
            model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Create the directory if it doesn't exist
            os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
            
            # Save the model and tokenizer to the local path
            model.save_pretrained(LOCAL_MODEL_PATH)
            tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            st.success(f"Downloaded and saved {MODEL_NAME} to {LOCAL_MODEL_PATH}")
        except Exception as e:
            st.error(f"Failed to download {MODEL_NAME}: {e}")
            return None

    # Load the embedding model from the local path
    try:
        embed_model = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,  # Use local path
            model_kwargs={"device": device, "trust_remote_code": True}
        )
        #st.success(f"Loaded local embedding model from {LOCAL_MODEL_PATH}")
        return embed_model
    except Exception as e:
        st.error(f"Failed to load local embedding model from {LOCAL_MODEL_PATH}: {e}")
        return None