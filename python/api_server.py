from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import json
import torch
from models.genshin_assistant import GenshinAssistant
from test_model import load_model, generate_response

app = FastAPI(title="Katheryne Assistant API")

class Query(BaseModel):
    text: str

class ModelResponse(BaseModel):
    response: dict

# Global model and vocabulary
model = None
vocab = None

@app.on_event("startup")
async def startup_event():
    """Load model and vocabulary on startup"""
    global model, vocab
    
    model_path = Path("models/assistant_best.pt")
    vocab_path = Path("models/assistant_vocab.json")
    
    if not model_path.exists():
        raise RuntimeError("Model file not found")
    if not vocab_path.exists():
        raise RuntimeError("Vocabulary file not found")
    
    model, vocab = load_model(model_path, vocab_path)

@app.post("/query", response_model=ModelResponse)
async def process_query(query: Query):
    """Process a query and return the response"""
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        response = generate_response(model, vocab, query.text)
        return ModelResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

def main():
    """Run the API server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()