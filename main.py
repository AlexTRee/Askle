import asyncio
import logging
from datetime import datetime, timedelta
from modules.sql_database import SQLDatabase
# Import other modules
# from modules.vector_database import VectorDatabase
# from modules.paper_scraper import PaperScraper
# from modules.ai_summarizer import DeepSeekSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def add_paper_workflow(paper_data, sql_db, vector_db, summarizer):
    """Process a single paper through the entire pipeline"""
    try:
        # 1. Extract abstract text from paper_data
        abstract = paper_data.get("abstract", "")
        
        # 2. Store abstract in vector database
        abstract_vector_id = await vector_db.add_text(
            text=abstract,
            metadata={"title": paper_data["title"], "type": "abstract"}
        )
        
        # 3. Generate summary with DeepSeek
        summary = await summarizer.generate_summary(abstract)
        
        # 4. Store summary in vector database
        summary_vector_id = await vector_db.add_text(
            text=summary,
            metadata={"title": paper_data["title"], "type": "summary"}
        )
        
        # 5. Add paper with both abstract and summary to SQL database
        paper_data["abstract"] = abstract
        paper_data["summary"] = summary
        paper_id = await sql_db.add_paper(paper_data, abstract_vector_id, summary_vector_id)
        
        logger.info(f"Added paper ID {paper_id}: {paper_data['title']}")
        return paper_id
        
    except Exception as e:
        logger.error(f"Error processing paper {paper_data['title']}: {e}")
        return None

async def query_workflow(query_text, sql_db, vector_db):
    """Process a user query and return relevant papers"""
    try:
        # 1. Check if query is cached
        cached_results = await sql_db.get_cached_query(query_text)
        if cached_results:
            logger.info(f"Found cached results for query: {query_text}")
            return cached_results
        
        # 2. Perform vector search on summaries
        vector_results = await vector_db.search(
            query=query_text,
            limit=10,
            filter_metadata={"type": "summary"}  # Focus on summaries for better relevance
        )
        
        # 3. Get full paper details from SQL database
        vector_ids = [result["id"] for result in vector_results]
        papers = await sql_db.get_papers_by_vector_ids(vector_ids)
        
        # 4. Cache the results
        paper_ids = [paper["id"] for paper in papers]
        await sql_db.cache_query(query_text, paper_ids)
        
        logger.info(f"Found {len(papers)} results for query: {query_text}")
        return papers
        
    except Exception as e:
        logger.error(f"Error processing query {query_text}: {e}")
        return []

async def daily_paper_update(sql_db, vector_db, scraper, summarizer):
    """Daily job to fetch new lung cancer papers"""
    try:
        # Get today's date and yesterday's date
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        date_str = yesterday.strftime("%Y/%m/%d")
        
        # 1. Fetch new papers from PubMed
        pubmed_papers = await scraper.fetch_pubmed_papers(
            query="lung cancer", 
            date_filter=date_str
        )
        
        # 2. Fetch new papers from Google Scholar
        scholar_papers = await scraper.fetch_scholar_papers(
            query="lung cancer", 
            date_filter=1  # Last day
        )
        
        # 3. Process all papers
        all_papers = pubmed_papers + scholar_papers
        logger.info(f"Found {len(all_papers)} new lung cancer papers to process")
        
        # 4. Process each paper
        added_count = 0
        for paper_data in all_papers:
            paper_id = await add_paper_workflow(paper_data, sql_db, vector_db, summarizer)
            if paper_id:
                added_count += 1
        
        # 5. Clean up expired cache
        cleared = await sql_db.clear_expired_cache()
        
        logger.info(f"Daily update complete: {added_count} papers added, {cleared} expired queries cleared")
        return added_count
        
    except Exception as e:
        logger.error(f"Error in daily paper update: {e}")
        return 0

async def main():
    """Main application entry point"""
    # Initialize database connections
    sql_db = SQLDatabase()
    
    # Initialize other components (you'll need to implement these)
    # vector_db = VectorDatabase()
    # scraper = PaperScraper()
    # summarizer = DeepSeekSummarizer()
    
    # For testing: Add a sample paper
    sample_paper = {
        "title": "Recent advances in immunotherapy for non-small cell lung cancer",
        "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
        "journal": "Journal of Clinical Oncology",
        "publication_date": "2025 Apr 1",
        "url": "https://example.com/paper123",
        "source": "PubMed",
        "abstract": "Lung cancer remains one of the leading causes of cancer-related deaths worldwide..."
    }
    
    # Uncomment when other components are implemented
    # paper_id = await add_paper_workflow(sample_paper, sql_db, vector_db, summarizer)
    
    # Get statistics about the database
    stats = await sql_db.get_statistics()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Sample query
    # Uncomment when other components are implemented
    # results = await query_workflow("immunotherapy resistance mechanisms in lung cancer", sql_db, vector_db)
    # print(f"Found {len(results)} relevant papers")
    # for i, paper in enumerate(results[:3], 1):
    #     print(f"\n{i}. {paper['title']}")
    #     print(f"   Authors: {', '.join(paper['authors'])}")
    #     print(f"   Source: {paper['source']}")
    #     print(f"   URL: {paper['url']}")
    #     print(f"   Summary: {paper['summary'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())