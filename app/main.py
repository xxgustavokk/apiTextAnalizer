from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

#carrega as variáveis de ambiente do arquivo .env
load_dotenv()

#inicializa o cliente do Gemini com a chave de API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#configura o banco de dados usando SQLAlchemy, onde a URL do banco é obtida da variável de ambiente
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

#define o modelo de dados para armazenar as análises feitas pelo modelo Gemini, com campos para o texto original, o resultado da análise e a data de criação
class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text)
    result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

#cria a aplicação e define como fastapi para criar a API
app = FastAPI()

#define o formato da entrada de dados para que seja validada automaticamente pelo FastAPI
class TextInput(BaseModel):
    text: str

#valida se a API está funcionando
@app.get("/")
def root():
    return {"message": "API funcionando!"}

#define a rota para analisar o texto, onde recebe um JSON com o campo "text" e retorna a análise feita pelo modelo Gemini, e salva no banco de dados.
@app.post("/analyze")
def analyze_text(input: TextInput):
    prompt = f"""
    Analyze the following text and provide:
    1) A brief summary
    2) Three main points
    3) Overall sentiment (positive, negative, or neutral)

    Text: {input.text}
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    result = response.text

    # Salva no banco
    db = SessionLocal()
    analysis = Analysis(original_text=input.text, result=result)
    db.add(analysis)
    db.commit()
    db.close()

    return {"analysis": result}

#aponta pra onde esta as analizes feitas e retorna em uma lista.
@app.get("/history")
def get_history():
    db = SessionLocal()
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    db.close()
    return [
        {
            "id": a.id,
            "original_text": a.original_text[:100] + "...",
            "result": a.result,
            "created_at": a.created_at
        }
        for a in analyses
    ]