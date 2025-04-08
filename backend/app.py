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
import datetime
import traceback # For detailed error logging

# --- Import Custom Modules ---
# Ensure these paths are correct relative to app.py or adjust PYTHONPATH
try:
    from modules.paper_retrieval import PubMedRetriever, GoogleScholarRetriever, PaperInfo
    from modules.ai_processing import DeepSeekProcessor
    from modules.sql_database import SQLDatabase
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
SQL_DB_PATH = os.getenv("SQL_DB_PATH", "./data/papers.db") # Ensure ./data exists
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./data/vector_index.faiss")
VECTOR_METADATA_PATH = os.getenv("VECTOR_METADATA_PATH", "./data/vector_metadata.pkl")
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
        ai_processor = DeepSeekProcessor(model_path=DEEPSEEK_MODEL_PATH) # Pass model path if needed
        logger.info("Initialized DeepSeek AI Processor.")

        # Initialize Paper Retrievers
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
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

class StatusResponse(BaseModel):
    """Response for the /status endpoint."""
    query: str
    status: str = Field(..., description="Current status: 'processing', 'completed', or 'error'.")
    summaries: Optional[List[PaperSummaryResponse]] = Field(default=None, description="List of summaries if status is 'completed'.")
    error_message: Optional[str] = Field(default=None, description="Error details if status is 'error'.")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

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
    The core background task to handle the new workflow.
    Fetches papers based on keywords, stores abstracts, summarizes, updates storage, and caches results.
    """
    logger.info(f"Background task started for query: '{original_query[:100]}...'")
    task_start_time = datetime.datetime.now(datetime.timezone.utc)
    processed_paper_sql_ids = [] # Keep track of IDs for caching

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
            await sql_db_dep.cache_query(original_query, []) # Cache empty result
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
            abstract_doc_id = f"{paper_dict['url']}_abstract"
            # Prepare metadata for vector DB
            vector_meta = {
                "url": paper_dict['url'],
                "title": paper_dict['title'],
                "source": paper_dict['source'],
                # Add sql_paper_id later if possible, or store URL here and link later
            }
            # Add task to vector add list (non-blocking)
            vector_abstract_tasks.append(
                vector_db_dep.add_document(
                    doc_id=abstract_doc_id,
                    text=paper_dict['abstract'],
                    doc_type="abstract",
                    metadata=vector_meta
                )
            )
            # We don't await here, gather results later. Get the vector ID *after* SQL insert.

            # --- Add Paper to SQL DB ---
            # Initially add without vector IDs, or pass None
            sql_paper_id = await sql_db_dep.add_paper(paper_dict, abstract_vector_id=None) # Pass None initially

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
        logger.info(f"Waiting for {len(vector_abstract_tasks)} abstract vector additions (if any)...")
        vector_abstract_results = await asyncio.gather(*vector_abstract_tasks, return_exceptions=True)
        for i, result in enumerate(vector_abstract_results):
             if isinstance(result, Exception):
                  logger.error(f"Error adding abstract vector (task {i}): {result}")
             # else: logger.debug(f"Abstract vector added at index: {result}")


        if not initial_paper_data_for_summary:
            logger.error("No papers were successfully stored in the SQL database. Aborting summarization.")
            await sql_db_dep.cache_query(original_query, []) # Cache empty result
            return

        # 4. Batch Summarize Papers
        logger.info(f"Step 4: Starting batch summarization for {len(initial_paper_data_for_summary)} papers...")
        # Pass the list of dictionaries (which now include 'sql_id')
        papers_with_summaries = await ai_processor_dep.batch_summarize_papers(initial_paper_data_for_summary)
        logger.info("Batch summarization finished.")

        # 5. Update DB with Summaries
        logger.info("Step 5: Updating SQL DB and adding summaries to Vector DB...")
        vector_summary_tasks = []
        sql_update_tasks = []

        for paper_summary_dict in papers_with_summaries:
            sql_id = paper_summary_dict.get('sql_id')
            summary_text = paper_summary_dict.get('summary', '')
            paper_url = paper_summary_dict.get('url', '') # Needed for vector doc ID

            if not sql_id or not summary_text or summary_text.startswith("Error generating summary:"):
                logger.warning(f"Skipping summary update for paper SQL ID {sql_id} due to missing ID or summary error.")
                continue

            # --- Optional: Add Summary to Vector DB ---
            summary_doc_id = f"{paper_url}_summary"
            vector_meta = {
                "sql_paper_id": sql_id, # Now we have the SQL ID
                "url": paper_url,
                "title": paper_summary_dict.get('title', ''),
                "source": paper_summary_dict.get('source', ''),
            }
            vector_summary_tasks.append(
                 vector_db_dep.add_document(
                    doc_id=summary_doc_id,
                    text=summary_text,
                    doc_type="summary",
                    metadata=vector_meta
                )
            )
            # Again, we don't store the summary_vector_id back to SQL for simplicity here.
            # If needed, await the result and pass it to update_paper_summary.

            # --- Update SQL DB with Summary ---
            sql_update_tasks.append(
                sql_db_dep.update_paper_summary(
                    paper_id=sql_id,
                    summary=summary_text,
                    summary_vector_id=None # Pass None for now
                )
            )

        # --- Await Vector Summary Additions & SQL Updates ---
        logger.info(f"Waiting for {len(vector_summary_tasks)} summary vector additions...")
        vector_summary_results = await asyncio.gather(*vector_summary_tasks, return_exceptions=True)
        for i, result in enumerate(vector_summary_results):
             if isinstance(result, Exception): logger.error(f"Error adding summary vector (task {i}): {result}")

        logger.info(f"Waiting for {len(sql_update_tasks)} SQL summary updates...")
        sql_update_results = await asyncio.gather(*sql_update_tasks, return_exceptions=True)
        successful_updates = 0
        for i, result in enumerate(sql_update_results):
             if isinstance(result, Exception): logger.error(f"Error updating SQL summary (task {i}): {result}")
             elif result is True: successful_updates += 1
             # else: logger.warning(f"SQL summary update task {i} returned False.")
        logger.info(f"Successfully updated {successful_updates} summaries in SQL DB.")


        # 6. Cache Final Results
        # Cache the list of successfully processed SQL paper IDs against the original query
        logger.info(f"Step 6: Caching results for original query: '{original_query[:100]}...'")
        if processed_paper_sql_ids:
            await sql_db_dep.cache_query(original_query, processed_paper_sql_ids)
            logger.info(f"Cached {len(processed_paper_sql_ids)} paper IDs for the query.")
        else:
            logger.warning("No paper IDs to cache for the query.")

        task_end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = task_end_time - task_start_time
        logger.info(f"Background task for query '{original_query[:100]}...' completed successfully in {duration}.")

    except Exception as e:
        logger.exception(f"Error during background processing for query '{original_query[:100]}...': {e}")
        # Optionally, cache an error state? For now, just log.
        # await sql_db_dep.cache_query(original_query, [], status="error", message=str(e)) # Needs DB schema change


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
    Checks cache; if not found, starts background processing and returns an 'Accepted' response.
    """
    original_query = request.query
    logger.info(f"Received /ask request for query: '{original_query[:100]}...'")

    # 1. Check Cache First
    try:
        cached_results = await sql_db_dep.get_cached_query(original_query)
        if cached_results:
            logger.info(f"Cache hit for query: '{original_query[:100]}...'. Returning cached results immediately.")
            # If found in cache, return 200 OK with results directly (or redirect to status?)
            # For consistency with background processing, maybe still return 202 and let status handle it?
            # Let's return 202 but indicate cache hit in message for clarity.
            return AskQuestionResponse(
                query=original_query,
                status="completed_cached",
                message="Query results found in cache. Use /api/status to retrieve."
            )
    except Exception as e:
        logger.exception(f"Error checking cache for query '{original_query[:100]}...': {e}")
        # Proceed as if not cached, but log the error

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
    Checks the status of a previously submitted query.
    Returns 'processing', 'completed' with summaries, or 'error'.
    """
    logger.info(f"Received /status request for query: '{query[:100]}...'")
    try:
        # Check SQL cache for results associated with the original query
        cached_papers = await sql_db_dep.get_cached_query(query)

        if cached_papers is None:
             # This case might indicate an error during caching or query never submitted
             # Let's assume 'processing' for now, or maybe 'not_found'?
             logger.warning(f"Status check: Query '{query[:100]}...' not found in cache (returned None). Assuming processing or error.")
             # Check if the query exists in the 'queries' table at all? Requires new DB method.
             # For now, return processing.
             return StatusResponse(query=query, status="processing") # Or potentially error/not_found

        elif isinstance(cached_papers, list) and not cached_papers:
             # Empty list means processing finished but found no relevant papers
             logger.info(f"Status check: Query '{query[:100]}...' completed with no results found.")
             return StatusResponse(query=query, status="completed", summaries=[])

        elif isinstance(cached_papers, list) and cached_papers:
             # Found results
             logger.info(f"Status check: Query '{query[:100]}...' completed with {len(cached_papers)} results.")
             summaries = []
             for paper in cached_papers:
                 # Map DB dictionary to Pydantic response model
                 summaries.append(PaperSummaryResponse(
                     sql_id=paper.get("id"),
                     title=paper.get("title", "N/A"),
                     authors=paper.get("authors", []),
                     publication_date=paper.get("publication_date"), # Already ISO string or None from DB method
                     journal=paper.get("journal", "N/A"),
                     abstract=paper.get("abstract", ""),
                     summary=paper.get("summary", ""), # This should be the generated summary
                     url=paper.get("url", ""),
                     source=paper.get("source", "N/A")
                 ))
             return StatusResponse(query=query, status="completed", summaries=summaries)

        else:
             # Should not happen if get_cached_query returns list or None, but handle defensively
             logger.error(f"Status check: Unexpected result type from get_cached_query for '{query[:100]}...': {type(cached_papers)}")
             return StatusResponse(query=query, status="error", error_message="Internal error checking status.")

    except Exception as e:
        logger.exception(f"Error checking status for query '{query[:100]}...': {e}")
        return StatusResponse(query=query, status="error", error_message=str(e))


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

