# app.py - FastAPI Backend
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os
from typing import List, Optional
import asyncio
import logging
import datetime
from motor.motor_asyncio import AsyncIOMotorClient

# Import custom modules
from modules.paper_retrieval import PubMedRetriever, GoogleScholarRetriever
from modules.ai_processing import DeepSeekProcessor
from modules.database import Database

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lung Cancer Research Assistant")

# Environment variables (should be in .env file)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Database connection
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URI)
    app.db = app.mongodb_client.research_assistant
    app.database = Database(app.db)
    logger.info("Connected to the MongoDB database!")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    logger.info("MongoDB connection closed")

# Initialize retrievers and AI processor
pubmed_retriever = PubMedRetriever()
scholar_retriever = GoogleScholarRetriever()
ai_processor = DeepSeekProcessor(api_key=DEEPSEEK_API_KEY)

# Data models
class Question(BaseModel):
    query: str
    max_results: int = 10

class PaperSummary(BaseModel):
    title: str
    authors: List[str]
    publication_date: str
    journal: str
    abstract: str
    summary: str
    url: str
    source: str  # "PubMed" or "Google Scholar"

class QuestionResponse(BaseModel):
    query: str
    summaries: List[PaperSummary]
    timestamp: str

# Routes
@app.get("/")
async def root():
    return {"message": "Lung Cancer Research Assistant API"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(question: Question, background_tasks: BackgroundTasks):
    """
    Process a user question and return relevant paper summaries
    """
    logger.info(f"Received question: {question.query}")
    
    # Check cache first
    cached_response = await app.database.get_cached_response(question.query)
    if cached_response:
        logger.info("Returning cached response")
        return cached_response
    
    # Start processing in the background if not cached
    # For immediate response, return a job ID and use WebSockets for real-time updates
    background_tasks.add_task(process_question, question)
    
    # Return an immediate response
    return QuestionResponse(
        query=question.query,
        summaries=[],
        timestamp=datetime.datetime.now().isoformat()
    )

async def process_question(question: Question):
    """
    Background task to process a question
    """
    # Retrieve papers from PubMed
    pubmed_papers = await pubmed_retriever.search(
        query=f"lung cancer {question.query}",
        max_results=question.max_results // 2
    )
    
    # Retrieve papers from Google Scholar
    scholar_papers = await scholar_retriever.search(
        query=f"lung cancer {question.query}",
        max_results=question.max_results // 2
    )
    
    # Combine results
    all_papers = pubmed_papers + scholar_papers
    
    # Process each paper with DeepSeek
    summaries = []
    for paper in all_papers:
        summary = await ai_processor.summarize_paper(paper)
        summaries.append(summary)
        # Store in database incrementally
        await app.database.store_paper_summary(summary)
    
    # Create final response
    response = QuestionResponse(
        query=question.query,
        summaries=summaries,
        timestamp=datetime.datetime.now().isoformat()
    )
    
    # Cache the response
    await app.database.cache_response(question.query, response)
    
    # In a real application, you'd notify the client via WebSockets here
    
    return response

@app.get("/history", response_model=List[QuestionResponse])
async def get_history(limit: int = 10):
    """
    Get recent question history
    """
    history = await app.database.get_recent_questions(limit)
    return history

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)