from fastapi import FastAPI
from pydantic import BaseModel  
from transformers import pipeline
import uvicorn

app = FastAPI(title="Análise de Sentimento API", version="1.0.0")

# Inicializar o modelo (pode demorar na primeira execução)
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    sentiment_analyzer = None

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "API de Análise de Sentimento", "status": "online"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": sentiment_analyzer is not None
    }

@app.post("/sentiment")
async def analyze_sentiment(input: TextInput):
    if sentiment_analyzer is None:
        return {"error": "Modelo não carregado"}
    
    try:
        result = sentiment_analyzer(input.text)
        return {
            "text": input.text, 
            "sentiment": result[0]["label"], 
            "score": round(result[0]["score"], 4)
        }
    except Exception as e:
        return {"error": f"Erro na análise: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)