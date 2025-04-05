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
from modules.sql_database import SQLDatabase
from modules.vector_database import VectorDatabase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lung Cancer Research Assistant")

# Environment variables (should be in .env file)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
SQL_DB_PATH = os.getenv("SQL_DB_PATH", "./data/papers.db")
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./data/vector_index")
VECTOR_METADATA_PATH = os.getenv("VECTOR_METADATA_PATH", "./data/vector_metadata")

# Database connection
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URI)
    app.db = app.mongodb_client.research_assistant
    logger.info("Connected to the MongoDB database!")

    # SQL Database connection
    app.sql_db = SQLDatabase(f"sqlite:///{SQL_DB_PATH}")
    logger.info("Connected to the SQL database!")
    
    # Vector Database connection
    app.vector_db = VectorDatabase(
        index_path=VECTOR_INDEX_PATH,
        metadata_path=VECTOR_METADATA_PATH
    )
    logger.info("Connected to the Vector database!")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    logger.info("MongoDB connection closed")

# Initialize retrievers and AI processor
pubmed_retriever = PubMedRetriever()
scholar_retriever = GoogleScholarRetriever()
ai_processor = DeepSeekProcessor()

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

# Helper functions for database operations
async def store_paper_in_databases(paper, summary_text):
    """Store paper in both SQL and vector databases"""
    try:
        # Convert paper to the format expected by SQL DB
        paper_data = {
            "title": paper["title"],
            "authors": paper["authors"],
            "journal": paper["journal"],
            "publication_date": paper["publication_date"],
            "url": paper["url"],
            "source": paper["source"],
            "abstract": paper["abstract"],
            "summary": summary_text
        }
        
        # Store abstract in vector database
        abstract_vector_id = await app.vector_db.add_document(
            doc_id=paper["url"],
            text=paper["abstract"],
            doc_type="abstract",
            metadata={"title": paper["title"], "type": "abstract"}
        )
        
        # Store summary in vector database
        summary_vector_id = await app.vector_db.add_document(
            doc_id=f"{paper['url']}_summary",
            text=summary_text,
            doc_type="summary",
            metadata={"title": paper["title"], "type": "summary"}
        )
        
        # Store in SQL database with vector IDs
        paper_id = await app.sql_db.add_paper(
            paper_data, 
            str(abstract_vector_id), 
            str(summary_vector_id)
        )
        
        logger.info(f"Stored paper '{paper['title']}' in databases with ID {paper_id}")
        return paper_id
        
    except Exception as e:
        logger.error(f"Error storing paper in databases: {e}")
        return None

async def search_papers(query_text, limit=10):
    """Search for papers using vector database and retrieve from SQL DB"""
    try:
        # Check if query is cached in SQL DB
        cached_results = await app.sql_db.get_cached_query(query_text)
        if cached_results:
            logger.info(f"Found cached results for query: {query_text}")
            return cached_results
        
        # Search vector database for relevant papers
        vector_results = await app.vector_db.search(
            query=query_text,
            limit=limit
        )
        
        if not vector_results:
            logger.info(f"No vector search results for: {query_text}")
            # Fallback to SQL-based search
            return await app.sql_db.search_papers(query_text, limit)
        
        # Get full paper details from SQL database
        vector_ids = [result["id"] for result in vector_results]
        papers = await app.sql_db.get_papers_by_vector_ids(vector_ids)
        
        # Cache the results for future use
        paper_ids = [paper["id"] for paper in papers]
        await app.sql_db.cache_query(query_text, paper_ids)
        
        logger.info(f"Found {len(papers)} results for query: {query_text}")
        return papers
        
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        return []
    
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
    
    # First check SQL database cache
    cached_results = await app.sql_db.get_cached_query(question.query)
    if cached_results:
        logger.info("Returning cached SQL response")
        summaries = []
        for paper in cached_results:
            summary = PaperSummary(
                title=paper["title"],
                authors=paper["authors"],
                publication_date=paper["publication_date"] if paper["publication_date"] else "",
                journal=paper["journal"],
                abstract=paper["abstract"],
                summary=paper["summary"],
                url=paper["url"],
                source=paper["source"]
            )
            summaries.append(summary)
        
        return QuestionResponse(
            query=question.query,
            summaries=summaries,
            timestamp=datetime.datetime.now().isoformat()
        )
    
    # Then check MongoDB cache (original functionality)
    try:
        cached_response = await app.db.cached_responses.find_one({"query": question.query})
        if cached_response:
            logger.info("Returning cached MongoDB response")
            cached_response.pop("_id", None)  # Remove MongoDB ID
            return QuestionResponse(**cached_response)
    except Exception as e:
        logger.error(f"Error checking MongoDB cache: {e}")
    
    # Start processing in the background if not cached
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
    # First check if we already have relevant papers in our database
    existing_papers = await search_papers(question.query, question.max_results)
    
    if existing_papers and len(existing_papers) >= question.max_results // 2:
        logger.info(f"Found {len(existing_papers)} existing papers in database")
        # We have enough existing papers, no need to fetch new ones
        summaries = []
        for paper in existing_papers[:question.max_results]:
            summary = PaperSummary(
                title=paper["title"],
                authors=paper["authors"],
                publication_date=paper["publication_date"] if paper["publication_date"] else "",
                journal=paper["journal"],
                abstract=paper["abstract"],
                summary=paper["summary"],
                url=paper["url"],
                source=paper["source"]
            )
            summaries.append(summary)
    else:
        # Not enough existing papers, fetch new ones
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
            summary_text = await ai_processor.summarize_paper(paper)
            
            # Store in both databases
            paper_id = await store_paper_in_databases(paper, summary_text)
            
            # Create summary object for response
            summary = PaperSummary(
                title=paper["title"],
                authors=paper["authors"],
                publication_date=paper["publication_date"],
                journal=paper["journal"],
                abstract=paper["abstract"],
                summary=summary_text,
                url=paper["url"],
                source=paper["source"]
            )
            summaries.append(summary)
            
            # Store in MongoDB incrementally (original functionality)
            try:
                await app.db.paper_summaries.insert_one(summary.dict())
            except Exception as e:
                logger.error(f"Error storing in MongoDB: {e}")
    
    # Create final response
    response = QuestionResponse(
        query=question.query,
        summaries=summaries,
        timestamp=datetime.datetime.now().isoformat()
    )
    
    # Cache the response in MongoDB (original functionality)
    try:
        await app.db.cached_responses.insert_one(response.dict())
    except Exception as e:
        logger.error(f"Error caching in MongoDB: {e}")
    
    # Cache paper IDs in SQL database
    if summaries:
        paper_ids = []
        for summary in summaries:
            paper = await app.sql_db.get_paper_by_url(summary.url)
            if paper:
                paper_ids.append(paper["id"])
        
        if paper_ids:
            await app.sql_db.cache_query(question.query, paper_ids)
    
    # In a real application, you'd notify the client via WebSockets here
    
    return response

@app.get("/history", response_model=List[QuestionResponse])
async def get_history(limit: int = 10):
    """
    Get recent question history
    """
    try:
        # Get from MongoDB (original functionality)
        cursor = app.db.cached_responses.find().sort("timestamp", -1).limit(limit)
        history = []
        async for doc in cursor:
            doc.pop("_id", None)  # Remove MongoDB ID
            history.append(QuestionResponse(**doc))
        return history
    except Exception as e:
        logger.error(f"Error retrieving history from MongoDB: {e}")
        
        # Fallback to SQL database
        try:
            recent_queries = await app.sql_db.get_recent_queries(limit)
            history = []
            for query_item in recent_queries:
                papers = await app.sql_db.get_cached_query(query_item["query_text"])
                if papers:
                    summaries = []
                    for paper in papers:
                        summary = PaperSummary(
                            title=paper["title"],
                            authors=paper["authors"],
                            publication_date=paper["publication_date"] if paper["publication_date"] else "",
                            journal=paper["journal"],
                            abstract=paper["abstract"],
                            summary=paper["summary"],
                            url=paper["url"],
                            source=paper["source"]
                        )
                        summaries.append(summary)
                    
                    history.append(QuestionResponse(
                        query=query_item["query_text"],
                        summaries=summaries,
                        timestamp=query_item["created_at"]
                    ))
            return history
        except Exception as e2:
            logger.error(f"Error retrieving history from SQL DB: {e2}")
            return []

@app.get("/papers/recent", response_model=List[PaperSummary])
async def get_recent_papers(limit: int = 10, source: Optional[str] = None):
    """
    Get recently added papers
    """
    try:
        papers = await app.sql_db.get_recent_papers(limit, source)
        summaries = []
        for paper in papers:
            summary = PaperSummary(
                title=paper["title"],
                authors=paper["authors"],
                publication_date=paper["publication_date"] if paper["publication_date"] else "",
                journal=paper["journal"],
                abstract=paper["abstract"],
                summary=paper["summary"],
                url=paper["url"],
                source=paper["source"]
            )
            summaries.append(summary)
        return summaries
    except Exception as e:
        logger.error(f"Error retrieving recent papers: {e}")
        return []

@app.get("/stats")
async def get_stats():
    """
    Get database statistics
    """
    try:
        stats = await app.sql_db.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        return {"error": str(e)}

@app.post("/papers/search")
async def search_papers_api(query: str, limit: int = 10):
    """
    Search for papers
    """
    papers = await search_papers(query, limit)
    summaries = []
    for paper in papers:
        summary = PaperSummary(
            title=paper["title"],
            authors=paper["authors"],
            publication_date=paper["publication_date"] if paper["publication_date"] else "",
            journal=paper["journal"],
            abstract=paper["abstract"],
            summary=paper["summary"],
            url=paper["url"],
            source=paper["source"]
        )
        summaries.append(summary)
    return summaries

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)