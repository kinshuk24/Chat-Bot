import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request, jsonify, send_file
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from concurrent.futures import ThreadPoolExecutor
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173",
    # "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv('.env')
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
executor = ThreadPoolExecutor(max_workers=2)
groq_api_key = os.getenv("GROQ_API_KEY")
llama_api_key = os.getenv("OPENAI_API_KEY")
llm = Groq(model="llama3-8b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 20
docs_digital_transformation = SimpleDirectoryReader(input_files=["Data/Digital_Transformation.pdf"]).load_data()
index = VectorStoreIndex.from_documents(docs_digital_transformation)
query_engine = index.as_query_engine(similarity_top_k=3)
try:
    response = query_engine.query("Who are you?")
    print(response)
except Exception as e:
    print(f"An error occurred while querying: {e}")
