import os
from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv('.env')

# Load translation model and tokenizer
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# API keys
groq_api_key = os.getenv("GROQ_API_KEY")
llama_api_key = os.getenv("OPENAI_API_KEY")

# Set up LLM and embedding model
llm = Groq(model="llama3-8b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 20

# Pre-load documents and create an index
docs_digital_transformation = SimpleDirectoryReader(input_files=["Data/Digital_Transformation.pdf"]).load_data()
index = VectorStoreIndex.from_documents(docs_digital_transformation)
query_engine = index.as_query_engine(similarity_top_k=3)

def translate_text(text: str, target_lang_code: str) -> str:
    """Translates the given text to the specified target language."""
    tokenizer_input = translation_tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **tokenizer_input,
        forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code)
    )
    translated_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

class QueryRequest(BaseModel):
    query: str
    language: str 

@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        query_text = request.query
        lang = request.language  # Get the language code from the request
        
        # Print the request for debugging
        print(request)

        # Get the response from the query engine
        response = query_engine.query(query_text)
        
        # Ensure the response is a string or extract the relevant part
        response_text = str(response)  # Modify this as necessary based on the actual structure of response

        # Translate the response
        translated_response = translate_text(response_text, lang)
        
        # Print the translated response for debugging
        print(f"Translated Response: {translated_response}")  
        
        # Return the translated response
        return {"response": translated_response}  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying the model: {str(e)}")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
