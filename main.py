from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Summarization API",
    description="Using t5-small with automatic download",
    version="1.0"
)

summarizer = pipeline(
    "summarization",
    model="t5-small"
)

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize(data: TextInput):
    result = summarizer(
        data.text,
        max_length=120,
        min_length=20,
        do_sample=False
    )
    return {"summary": result[0]["summary_text"]}

@app.get("/")
def root():
    return {"message": "API is running"}
