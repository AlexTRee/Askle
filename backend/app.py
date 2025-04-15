# app.py - FastAPI Backend
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import asyncio
import logging
from datetime import datetime, timezone
import traceback # For detailed error logging

# --- Import Custom Modules ---
# Ensure these paths are correct relative to app.py or adjust PYTHONPATH
try:
    from modules.paper_retrieval import PubMedRetriever, GoogleScholarRetriever, PaperInfo
    from modules.ai_processing import DeepSeekProcessor
    from modules.sql_database import SQLDatabase, QueryStatus
    from modules.vector_database import VectorDatabase
except ImportError as e:
    print(f"Error importing modules: {e}. Make sure modules are in the correct path.")
    # Depending on setup, might need:
    # import sys
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # from modules...
    raise

# --- Setup Logging ---
# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Get logger for this module
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI(
    title="Askle Evidence Synthesis Assistant",
    description="API for retrieving, summarizing, and synthesizing biomedical research papers.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allow all origins for development. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider changing to specific origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Router ---
api_router = APIRouter(prefix="/api")

# --- Configuration & Global Variables ---
# Use environment variables or defaults
SQL_DB_PATH = os.getenv("SQL_DB_PATH", "./backend/data/papers.db") # Ensure ./data exists
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./backend/data/vector_index.faiss")
VECTOR_METADATA_PATH = os.getenv("VECTOR_METADATA_PATH", "./backend/data/vector_metadata.pkl")
DEEPSEEK_MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH", None) # Let DeepSeekProcessor handle default
# Max papers to fetch from each source
PAPERS_PER_SOURCE = 5

# --- Database and Processor Initialization ---
# These will be initialized in the startup event
sql_db: Optional[SQLDatabase] = None
vector_db: Optional[VectorDatabase] = None
ai_processor: Optional[DeepSeekProcessor] = None
pubmed_retriever: Optional[PubMedRetriever] = None
scholar_retriever: Optional[GoogleScholarRetriever] = None

# --- Application Lifecycle Events (Startup/Shutdown) ---
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global sql_db, vector_db, ai_processor, pubmed_retriever, scholar_retriever
    logger.info("Application startup: Initializing resources...")
    try:
        # Initialize SQL Database
        # Ensure the directory exists before creating the connection string
        db_dir = os.path.dirname(SQL_DB_PATH)
        if db_dir and not os.path.exists(db_dir):
             os.makedirs(db_dir)
             logger.info(f"Created database directory: {db_dir}")
        sql_db = SQLDatabase(connection_string=f"sqlite:///{SQL_DB_PATH}")
        logger.info(f"Connected to SQL database: sqlite:///{SQL_DB_PATH}")

        # Initialize Vector Database
        vector_db = VectorDatabase(
            index_path=VECTOR_INDEX_PATH,
            metadata_path=VECTOR_METADATA_PATH
            # Add other params like embedding_model_name if needed
        )
        logger.info("Initialized Vector database.")

        # Initialize AI Processor (can take time to load model)  
        ai_processor = DeepSeekProcessor(model_path=DEEPSEEK_MODEL_PATH)  # Pass model path if needed
        logger.info("Initialized DeepSeek AI Processor.")

        # Initialize Paper Retrievers
         # TODO 
        # Check for environment variable, use API key for better request rates
        # 3 requests/second without an API key, 10 requests/second with API key.
        # pubmed_api_key = os.getenv("PUBMED_API_KEY")
        # if pubmed_api_key:
        #     logger.info("PubMed Retriever initialized with API key.")
        # else:
        #     logger.info("PubMed Retriever initialized without API key (rate limits apply).")
        pubmed_retriever = PubMedRetriever()
        scholar_retriever = GoogleScholarRetriever()
        logger.info("Initialized Paper Retrievers.")

        logger.info("Resource initialization complete.")

    except Exception as e:
        logger.exception(f"FATAL: Error during application startup: {e}")
        # Depending on severity, might want to exit or prevent app from starting fully
        raise RuntimeError(f"Application startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Application shutdown: Cleaning up resources...")
    if vector_db:
        vector_db.close() # Ensures final save of vector index/metadata
        logger.info("Vector database closed.")
    # SQL DB connection pool is typically managed by SQLAlchemy engine, explicit close not always needed
    logger.info("Resource cleanup finished.")

# --- Pydantic Models for API Request/Response ---
class AskQuestionRequest(BaseModel):
    query: str = Field(..., description="The user's question or research topic.")
    # max_results is now determined by PAPERS_PER_SOURCE * 2

class PaperSummaryResponse(BaseModel):
    """Defines the structure for returning processed paper information."""
    sql_id: Optional[int] = None # Include SQL ID if available
    title: str
    authors: List[str]
    publication_date: Optional[str] = None # Handle potential null dates
    journal: str
    abstract: str
    summary: str # Generated summary
    url: str
    source: str # e.g., "PubMed", "Google Scholar"

class AskQuestionResponse(BaseModel):
    """Response for the /ask endpoint (immediate)."""
    query: str
    status: str = Field(default="processing", description="Indicates that processing has started.")
    message: str = Field(default="Request received. Processing in background.", description="User-friendly message.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class StatusResponse(BaseModel):
    """Response for the /status endpoint."""
    query: str
    # Use the QueryStatus enum for the status field
    status: QueryStatus = Field(..., description="Current status: 'processing', 'completed', or 'error'.")
    summaries: Optional[List[PaperSummaryResponse]] = Field(default=None, description="List of summaries if status is 'completed'.")
    error_message: Optional[str] = Field(default=None, description="Error details if status is 'error'.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class HistoryItem(BaseModel):
    query: str
    timestamp: str # ISO format timestamp of the query
    # Maybe add number of results?

# --- Dependency Injection for DB/Processors ---
# Use dependencies to ensure resources are initialized before routes use them
async def get_sql_db() -> SQLDatabase:
    if sql_db is None:
        raise HTTPException(status_code=503, detail="SQL Database not initialized")
    return sql_db

async def get_vector_db() -> VectorDatabase:
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector Database not initialized")
    return vector_db

async def get_ai_processor() -> DeepSeekProcessor:
    if ai_processor is None:
        raise HTTPException(status_code=503, detail="AI Processor not initialized")
    return ai_processor

async def get_pubmed_retriever() -> PubMedRetriever:
    if pubmed_retriever is None:
        raise HTTPException(status_code=503, detail="PubMed Retriever not initialized")
    return pubmed_retriever

async def get_scholar_retriever() -> GoogleScholarRetriever:
    if scholar_retriever is None:
        raise HTTPException(status_code=503, detail="Google Scholar Retriever not initialized")
    return scholar_retriever


# --- Background Task Function ---
async def process_question_background(
    original_query: str,
    sql_db_dep: SQLDatabase,
    vector_db_dep: VectorDatabase,
    ai_processor_dep: DeepSeekProcessor,
    pubmed_retriever_dep: PubMedRetriever,
    scholar_retriever_dep: GoogleScholarRetriever
):
    """
    The core background task. Includes error handling to update cache status.
    Fetches papers based on keywords, stores abstracts, summarizes, updates storage, and caches results.
    """
    logger.info(f"Background task started for query: '{original_query[:100]}...'")
    task_start_time = datetime.now(timezone.utc)
    processed_paper_sql_ids = [] # Keep track of IDs for caching
    # Ensure cache entry exists with 'processing' status initially
    await sql_db_dep.cache_query(original_query, status=QueryStatus.PROCESSING)

    try:
        # 1. Extract Keywords
        logger.info("Step 1: Extracting keywords...")
        keyword_result = await ai_processor_dep.extract_keywords(original_query)
        keywords = keyword_result.get("keywords", [])
        if not keywords:
            logger.warning("No keywords extracted. Using original query for retrieval.")
            search_query = ai_processor_dep.sanitize_input(original_query) # Sanitize original query as fallback
        else:
            search_query = " ".join(keywords) # Combine keywords for searching
            logger.info(f"Extracted keywords: {keywords}. Using search query: '{search_query}'")

        # 2. Retrieve Papers (Concurrently)
        logger.info(f"Step 2: Retrieving papers from PubMed and Google Scholar (max {PAPERS_PER_SOURCE} each)...")
        pubmed_task = pubmed_retriever_dep.search(query=search_query, max_results=PAPERS_PER_SOURCE)
        scholar_task = scholar_retriever_dep.search(query=search_query, max_results=PAPERS_PER_SOURCE)
        retrieved_papers_lists = await asyncio.gather(pubmed_task, scholar_task, return_exceptions=True)

        all_retrieved_paper_infos: List[PaperInfo] = []
        for i, result in enumerate(retrieved_papers_lists):
            source_name = "PubMed" if i == 0 else "Google Scholar"
            if isinstance(result, Exception):
                logger.error(f"Error retrieving papers from {source_name}: {result}")
            elif result:
                logger.info(f"Retrieved {len(result)} papers from {source_name}.")
                all_retrieved_paper_infos.extend(result)
            else:
                 logger.info(f"No papers retrieved from {source_name}.")

        if not all_retrieved_paper_infos:
            logger.warning(f"No papers found from any source for query: '{search_query}'. Caching empty result.")
            # Cache COMPLETED status with empty list
            await sql_db_dep.cache_query(original_query, [], status=QueryStatus.COMPLETED)
            return # Stop processing

        logger.info(f"Total papers retrieved: {len(all_retrieved_paper_infos)}")

        # 3. Initial Storage (Abstracts)
        logger.info("Step 3: Storing initial paper data (abstracts) in SQL DB...")
        initial_paper_data_for_summary: List[Dict[str, Any]] = [] # Store data needed for next step
        processed_urls = set() # Avoid processing exact same URL twice if retrieved from both sources

        # Use a list to store tasks for adding abstract vectors (optional)
        vector_abstract_tasks = []

        for paper_info in all_retrieved_paper_infos:
            if paper_info.url in processed_urls:
                 logger.debug(f"Skipping duplicate URL: {paper_info.url}")
                 continue
            processed_urls.add(paper_info.url)

            paper_dict = paper_info.to_dict() # Convert PaperInfo object to dict

            # --- Optional: Add Abstract to Vector DB ---
            # Create a unique doc ID for the abstract
            abstract_text = paper_dict.get('abstract', '')
            paper_url = paper_dict.get('url', '')

            # Add abstract vector sequentially BEFORE adding to SQL
            abstract_vector_id: Optional[str] = None # Initialize
            if abstract_text and paper_url:
                    abstract_doc_id = f"{paper_url}_abstract"
                    vector_meta = {"url": paper_url, 
                                   "title": paper_dict.get('title',''), 
                                   "source": paper_dict.get('source','')}
                                    # Add sql_paper_id later if possible, or store URL here and link later
                    try:
                        # Await the vector addition here to get the ID
                        abstract_vector_id = await vector_db_dep.add_document(
                            doc_id=abstract_doc_id,
                            text=abstract_text,
                            doc_type="abstract",
                            metadata=vector_meta
                        )
                        if abstract_vector_id:
                            logger.info(f"Stored abstract vector with ID: {abstract_vector_id}")
                        else:
                            logger.warning(f"Failed to store abstract vector for URL {paper_url}")
                    except Exception as vec_e:
                        logger.error(f"Error adding abstract vector for {paper_url}: {vec_e}")

            # --- Add Paper to SQL DB ---
            # Pass the obtained abstract_vector_id
            sql_paper_id = await sql_db_dep.add_paper(paper_dict, abstract_vector_id=abstract_vector_id)

            if sql_paper_id is not None:
                logger.info(f"Stored paper '{paper_dict['title'][:30]}...' in SQL DB with ID: {sql_paper_id}")
                # Add SQL ID to the dict for the next step
                paper_dict['sql_id'] = sql_paper_id
                initial_paper_data_for_summary.append(paper_dict)
                processed_paper_sql_ids.append(sql_paper_id)
            else:
                logger.error(f"Failed to store paper '{paper_dict['title'][:30]}...' in SQL DB.")

        # --- Await Abstract Vector Additions (Optional) ---
        # If you need the abstract vector IDs to store in SQL immediately, you'd need a more complex flow.
        # For simplicity, we add them async here and don't store the returned ID back in SQL for now.
        # If storing abstract_vector_id in SQL is critical, modify sql_db.add_paper and the flow here.
        # logger.info(f"Waiting for {len(vector_abstract_tasks)} abstract vector additions (if any)...")
        # vector_abstract_results = await asyncio.gather(*vector_abstract_tasks, return_exceptions=True)
        # for i, result in enumerate(vector_abstract_results):
        #      if isinstance(result, Exception):
        #           logger.error(f"Error adding abstract vector (task {i}): {result}")
        #      else: logger.debug(f"Abstract vector added at index: {result}")

        if not initial_paper_data_for_summary:
            logger.error("No papers were successfully stored in the SQL database. Aborting summarization.")
            # Cache ERROR Status
            await sql_db_dep.cache_query(original_query, status=QueryStatus.ERROR, error_message="Failed to store retrieved papers in database.")
            return

        # 4. Batch Summarize Papers
        logger.info(f"Step 4: Starting batch summarization for {len(initial_paper_data_for_summary)} papers...")
        # Pass the list of dictionaries (which now include 'sql_id')
        papers_with_summaries = await ai_processor_dep.batch_summarize_papers(initial_paper_data_for_summary)
        logger.info("Batch summarization finished.")

        # 5. Update DB with Summaries
        logger.info("Step 5: Updating SQL DB and adding summaries to Vector DB...")
        # vector_summary_tasks = []
        # Store tasks for SQL updates, process one paper at a time for vector/SQL link
        sql_update_tasks = []

        for paper_summary_dict in papers_with_summaries:
            sql_id = paper_summary_dict.get('sql_id')
            summary_text = paper_summary_dict.get('summary', '')
            paper_url = paper_summary_dict.get('url', '') # Needed for vector doc ID

            if not sql_id or not summary_text or summary_text.startswith("Error generating summary:"):
                logger.warning(f"Skipping summary update for paper SQL ID {sql_id} due to missing ID or summary error.")
                continue
            
            # Add summary vector sequentially BEFORE updating SQL
            summary_vector_id: Optional[str] = None # Initialize
            if summary_text and paper_url:
                 summary_doc_id = f"{paper_url}_summary"
                 vector_meta = {
                     "sql_paper_id": sql_id,
                     "url": paper_url,
                     "title": paper_summary_dict.get('title', ''),
                     "source": paper_summary_dict.get('source', ''),
                 }
                 try:
                    # Await the vector addition here to get the ID
                    summary_vector_id = await vector_db_dep.add_document(
                         doc_id=summary_doc_id,
                         text=summary_text,
                         doc_type="summary",
                         metadata=vector_meta
                    )
                    if summary_vector_id:
                         logger.info(f"Stored summary vector with ID: {summary_vector_id} for SQL ID {sql_id}")
                    else:
                         logger.warning(f"Failed to store summary vector for SQL ID {sql_id}")
                 except Exception as vec_e:
                      logger.error(f"Error adding summary vector for SQL ID {sql_id}: {vec_e}")

            # Update SQL DB (Add task to list), pass summary_vector_id
            sql_update_tasks.append(
                sql_db_dep.update_paper_summary(
                    paper_id=sql_id,
                    summary=summary_text,
                    summary_vector_id=summary_vector_id # Pass the obtained ID
                )
            )

        # --- Await SQL Updates Concurrently ---
        logger.info(f"Waiting for {len(sql_update_tasks)} SQL summary updates...")
        sql_update_results = await asyncio.gather(*sql_update_tasks, return_exceptions=True)
        successful_updates = 0
        for i, result in enumerate(sql_update_results):
             if isinstance(result, Exception): logger.error(f"Error updating SQL summary (task {i}): {result}")
             elif result is True: successful_updates += 1
        logger.info(f"Successfully updated {successful_updates} summaries in SQL DB.")

        # 6. Cache Final Results (Success)
        # Cache the list of successfully processed SQL paper IDs against the original query
        logger.info(f"Step 6: Caching results for original query: '{original_query[:100]}...'")
        if processed_paper_sql_ids:
             # Cache COMPLETED status with paper IDs
            await sql_db_dep.cache_query(original_query, processed_paper_sql_ids, status=QueryStatus.COMPLETED)
            logger.info(f"Cached {len(processed_paper_sql_ids)} paper IDs for the query.")
        else:
            logger.warning("No paper IDs to cache, but process completed. Caching empty completed state.")
             # Cache COMPLETED status with empty list
            await sql_db_dep.cache_query(original_query, [], status=QueryStatus.COMPLETED)

        task_end_time = datetime.now(timezone.utc)
        duration = task_end_time - task_start_time
        logger.info(f"Background task for query '{original_query[:100]}...' completed successfully in {duration}.")

    # Catch specific exceptions if needed, otherwise broad Exception
    except Exception as e:
        # Cache ERROR status on any failure
        error_details = f"Error type: {type(e).__name__}. Details: {str(e)}"
        full_traceback = traceback.format_exc() # Get full traceback for logs
        logger.exception(f"Critical error during background processing for query '{original_query[:100]}...': {error_details}\nTraceback:\n{full_traceback}")
        await sql_db_dep.cache_query(original_query, status=QueryStatus.ERROR, error_message=error_details)


# --- API Routes ---
@api_router.get("/", summary="API Root")
async def api_root():
    """Provides a simple message indicating the API is running."""
    return {"message": "Askle Evidence Synthesis Assistant API"}

@api_router.post("/ask",
                 response_model=AskQuestionResponse,
                 status_code=202, # Accepted
                 summary="Submit a question for processing")
async def ask_question(
    request: AskQuestionRequest,
    background_tasks: BackgroundTasks,
    # Use Depends to get initialized instances
    sql_db_dep: SQLDatabase = Depends(get_sql_db),
    vector_db_dep: VectorDatabase = Depends(get_vector_db),
    ai_processor_dep: DeepSeekProcessor = Depends(get_ai_processor),
    pubmed_retriever_dep: PubMedRetriever = Depends(get_pubmed_retriever),
    scholar_retriever_dep: GoogleScholarRetriever = Depends(get_scholar_retriever)
):
    """
    Accepts a user's research question.
    Checks cache; if completed or error, returns status immediately.
    If processing or not found, ensures background task is running (or starts it).
    """
    original_query = request.query
    logger.info(f"Received /ask request for query: '{original_query[:100]}...'")

    # 1. Check Cache First
    try:
        cache_data = await sql_db_dep.get_cached_query(original_query)
        
        if cache_data and cache_data["status"] in [QueryStatus.COMPLETED, QueryStatus.ERROR]:
            logger.info(f"Query '{original_query[:100]}...' already processed (Status: {cache_data['status'].value}). Informing user.")
            # Return 202 Accepted, but indicate the status is already known
            return AskQuestionResponse(
                query=original_query,
                status=cache_data["status"].value, # Return the actual final status
                message=f"Query already processed. Status: {cache_data['status'].value}. Use /api/status to retrieve details."
            )
        elif cache_data and cache_data["status"] == QueryStatus.PROCESSING:
             logger.info(f"Query '{original_query[:100]}...' is already processing.")
             # Already processing, just return Accepted
             return AskQuestionResponse(query=original_query, status="processing", message="Request is already being processed.")
        
        # Cache miss (cache_data is None) or expired - Proceed to start background task
        logger.info(f"Query '{original_query[:100]}...' not found in cache or requires reprocessing. Starting background task.")
        background_tasks.add_task(
            process_question_background,
            original_query,
            sql_db_dep,
            vector_db_dep,
            ai_processor_dep,
            pubmed_retriever_dep,
            scholar_retriever_dep
        )
        # Return 'Accepted' Response
        return AskQuestionResponse(query=original_query, status="processing", message="Request received. Processing initiated.")
    
    except Exception as e:
        logger.exception(f"Error during /ask endpoint for query '{original_query[:100]}...': {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing ask request.")

    # 2. If Not Cached, Start Background Task   
    logger.info(f"Cache miss for query: '{original_query[:100]}...'. Starting background processing.")
    background_tasks.add_task(
        process_question_background,
        original_query,
        sql_db_dep,
        vector_db_dep,
        ai_processor_dep,
        pubmed_retriever_dep,
        scholar_retriever_dep
    )

    # 3. Return 'Accepted' Response
    return AskQuestionResponse(query=original_query) # Default status/message are set


@api_router.get("/status",
                response_model=StatusResponse,
                summary="Check processing status and get results")
async def check_status(
    query: str,
    sql_db_dep: SQLDatabase = Depends(get_sql_db)
):
    """
    Checks the status of a previously submitted query using the updated cache logic.
    """
    logger.info(f"Received /status request for query: '{query[:100]}...'")
    try:
        # Use the updated SQL method
        cache_data = await sql_db_dep.get_cached_query(query)

        if cache_data is None:
             # Query never submitted or issue finding it
             logger.warning(f"Status check: Query '{query[:100]}...' not found in cache.")
             # Return a specific status or error? Let's use ERROR for now.
             return StatusResponse(query=query, status=QueryStatus.ERROR, error_message="Query not found or never submitted.")

        # Prepare response based on cache_data
        response_status = cache_data["status"]
        response_error = cache_data["error_message"]
        response_summaries = None

        if response_status == QueryStatus.COMPLETED:
             response_summaries = []
             for paper in cache_data.get("papers", []):
                 response_summaries.append(PaperSummaryResponse(
                     sql_id=paper.get("id"),
                     title=paper.get("title", "N/A"),
                     authors=paper.get("authors", []),
                     publication_date=paper.get("publication_date"),
                     journal=paper.get("journal", "N/A"),
                     abstract=paper.get("abstract", ""),
                     summary=paper.get("summary", ""),
                     url=paper.get("url", ""),
                     source=paper.get("source", "N/A")
                 ))
        return StatusResponse(
            query=query,
            status=response_status,
            summaries=response_summaries,
            error_message=response_error
        )

    except Exception as e:
        logger.exception(f"Error checking status for query '{query[:100]}...': {e}")
        # Return generic error status
        return StatusResponse(query=query, status=QueryStatus.ERROR, error_message=f"Internal server error checking status: {str(e)}")


@api_router.get("/history",
                response_model=List[HistoryItem],
                summary="Get recent query history")
async def get_history(
    limit: int = 20,
    sql_db_dep: SQLDatabase = Depends(get_sql_db)
):
    """Retrieves a list of recently processed queries from the SQL database."""
    logger.info(f"Received /history request (limit: {limit}).")
    try:
        recent_queries = await sql_db_dep.get_recent_queries(limit) # Fetches {'id', 'query_text', 'created_at'}
        # Convert to HistoryItem Pydantic model
        history = [
            HistoryItem(query=q["query_text"], timestamp=q["created_at"])
            for q in recent_queries
        ]
        return history
    except Exception as e:
        logger.exception(f"Error retrieving history: {e}")
        # Return empty list on error
        return []


@api_router.get("/papers/recent",
                response_model=List[PaperSummaryResponse],
                summary="Get recently added papers")
async def get_recent_papers(
    limit: int = 10,
    source: Optional[str] = None,
    sql_db_dep: SQLDatabase = Depends(get_sql_db)
):
    """Retrieves recently added papers from the SQL database, optionally filtered by source."""
    logger.info(f"Received /papers/recent request (limit: {limit}, source: {source}).")
    try:
        papers = await sql_db_dep.get_recent_papers(limit, source)
        # Map DB dicts to Pydantic response model
        summaries = [
             PaperSummaryResponse(
                 sql_id=p.get("id"),
                 title=p.get("title", "N/A"),
                 authors=p.get("authors", []),
                 publication_date=p.get("publication_date"),
                 journal=p.get("journal", "N/A"),
                 abstract=p.get("abstract", ""),
                 summary=p.get("summary", ""),
                 url=p.get("url", ""),
                 source=p.get("source", "N/A")
             ) for p in papers
        ]
        return summaries
    except Exception as e:
        logger.exception(f"Error retrieving recent papers: {e}")
        return []


@api_router.get("/stats", summary="Get database statistics")
async def get_stats(
    sql_db_dep: SQLDatabase = Depends(get_sql_db),
    vector_db_dep: VectorDatabase = Depends(get_vector_db)
):
    """Retrieves statistics from both the SQL and Vector databases."""
    logger.info("Received /stats request.")
    try:
        # Run stats retrieval concurrently
        sql_stats_task = sql_db_dep.get_statistics()
        vector_stats_task = asyncio.to_thread(vector_db_dep.get_stats) # get_stats is sync

        sql_stats, vector_stats = await asyncio.gather(sql_stats_task, vector_stats_task, return_exceptions=True)

        response = {}
        if isinstance(sql_stats, Exception):
             logger.error(f"Error getting SQL stats: {sql_stats}")
             response["sql_database_error"] = str(sql_stats)
        else:
             response["sql_database"] = sql_stats

        if isinstance(vector_stats, Exception):
             logger.error(f"Error getting Vector DB stats: {vector_stats}")
             response["vector_database_error"] = str(vector_stats)
        else:
             response["vector_database"] = vector_stats

        return response

    except Exception as e:
        logger.exception(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


# --- Include API router in the main app ---
app.include_router(api_router)

# --- Static Files and Root Endpoint ---
# Serve React frontend build if it exists
STATIC_DIR = "build" # Assuming React build output is in 'build' directory
if os.path.exists(STATIC_DIR) and os.path.isdir(STATIC_DIR):
    logger.info(f"Serving static files from directory: {STATIC_DIR}")
    app.mount("/static", StaticFiles(directory=os.path.join(STATIC_DIR, "static")), name="static")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_react_app(request: Request, full_path: str):
        """Serves the React index.html for any path not matching API routes or static files."""
        # Check if path starts with API prefix, if so, let 404 happen
        api_prefix = api_router.prefix
        if full_path.startswith(api_prefix.strip('/')):
             # This should ideally be handled by FastAPI routing order,
             # but double-check to prevent serving index.html for API paths.
             # Let FastAPI handle the 404 for undefined API routes.
             pass # Fall through to allow potential API route matching

        # Serve index.html for all other paths
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
             return FileResponse(index_path)
        else:
             logger.error(f"Frontend entry point not found: {index_path}")
             return JSONResponse(status_code=404, content={"detail": "Frontend not found."})

else:
    logger.warning(f"Static files directory '{STATIC_DIR}' not found. Frontend will not be served.")
    # Define a simple root if no frontend build exists
    @app.get("/", include_in_schema=False)
    async def root_fallback():
        return {"message": "Welcome to Askle API. Frontend build not found."}


# --- Main Execution ---
if __name__ == "__main__":
    # Get host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true" # Enable reload by default for dev

    logger.info(f"Starting Uvicorn server on {host}:{port} (Reload: {reload})")
    uvicorn.run("app:app", host=host, port=port, reload=reload)

