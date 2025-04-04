# modules/sql_database.py
import logging
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.sql import func
from typing import List, Dict, Any, Optional
import datetime
import os

logger = logging.getLogger(__name__)

# Create base class for declarative models
Base = declarative_base()

# Association table for paper-author many-to-many relationship
paper_authors = Table(
    'paper_authors', 
    Base.metadata,
    Column('paper_id', Integer, ForeignKey('papers.id')),
    Column('author_id', Integer, ForeignKey('authors.id'))
)

class Author(Base):
    """Author model"""
    __tablename__ = 'authors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    
    # Relationships
    papers = relationship("Paper", secondary=paper_authors, back_populates="authors")
    
    def __repr__(self):
        return f"<Author(id={self.id}, name='{self.name}')>"

class Journal(Base):
    """Journal model"""
    __tablename__ = 'journals'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    
    # Relationships
    papers = relationship("Paper", back_populates="journal")
    
    def __repr__(self):
        return f"<Journal(id={self.id}, name='{self.name}')>"

class Source(Base):
    """Source model (PubMed, Google Scholar, etc.)"""
    __tablename__ = 'sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    
    # Relationships
    papers = relationship("Paper", back_populates="source")
    
    def __repr__(self):
        return f"<Source(id={self.id}, name='{self.name}')>"

class Paper(Base):
    """Paper model"""
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    publication_date = Column(DateTime)
    url = Column(String(500), unique=True)
    abstract_vector_id = Column(String(255))  # Reference to vector DB
    summary_vector_id = Column(String(255))   # Reference to vector DB
    journal_id = Column(Integer, ForeignKey('journals.id'))
    source_id = Column(Integer, ForeignKey('sources.id'))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    abstract = Column(Text)  # Added to store original abstract text
    summary = Column(Text)   # Added to store generated summary text
    
    # Relationships
    journal = relationship("Journal", back_populates="papers")
    authors = relationship("Author", secondary=paper_authors, back_populates="papers")
    source = relationship("Source", back_populates="papers")
    
    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title[:30]}...', journal='{self.journal.name if self.journal else 'None'}')>"

class Query(Base):
    """User query model"""
    __tablename__ = 'queries'
    
    id = Column(Integer, primary_key=True)
    query_text = Column(String(500), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)
    
    # Relationships
    results = relationship("QueryResult", back_populates="query")
    
    def __repr__(self):
        return f"<Query(id={self.id}, query_text='{self.query_text[:30]}...')>"

class QueryResult(Base):
    """Query result model (links queries to papers)"""
    __tablename__ = 'query_results'
    
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey('queries.id'))
    paper_id = Column(Integer, ForeignKey('papers.id'))
    rank = Column(Integer)  # Order in results
    
    # Relationships
    query = relationship("Query", back_populates="results")
    paper = relationship("Paper")
    
    def __repr__(self):
        return f"<QueryResult(query_id={self.query_id}, paper_id={self.paper_id}, rank={self.rank})>"

class SQLDatabase:
    """Class to handle SQL database operations"""
    
    def __init__(self, connection_string=None):
        """Initialize database connection"""
        if connection_string is None:
            # Default to SQLite for simplicity
            db_path = os.environ.get("SQL_DB_PATH", "./data/papers.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            connection_string = f"sqlite:///{db_path}"
        
        try:
            self.engine = create_engine(connection_string)
            self.Session = scoped_session(sessionmaker(bind=self.engine))
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            logger.info(f"Connected to SQL database at {connection_string}")
            
        except Exception as e:
            logger.error(f"Error connecting to SQL database: {e}")
            raise
    
    async def add_paper(self, paper_data: Dict[str, Any], abstract_vector_id: str, summary_vector_id: str) -> int:
        """
        Add a paper and its metadata to the SQL database
        
        Args:
            paper_data: Dictionary with paper metadata
            abstract_vector_id: ID of abstract in vector database
            summary_vector_id: ID of summary in vector database
            
        Returns:
            ID of the added paper
        """
        session = self.Session()
        try:
            # Check if paper already exists
            existing_paper = session.query(Paper).filter_by(url=paper_data["url"]).first()
            if existing_paper:
                # Update existing paper
                existing_paper.abstract_vector_id = abstract_vector_id
                existing_paper.summary_vector_id = summary_vector_id
                existing_paper.updated_at = datetime.datetime.now()
                # Update abstract and summary if they exist in paper_data
                if paper_data.get("abstract"):
                    existing_paper.abstract = paper_data["abstract"]
                if paper_data.get("summary"):
                    existing_paper.summary = paper_data["summary"]
                session.commit()
                return existing_paper.id
            
            # Get or create source
            source = session.query(Source).filter_by(name=paper_data["source"]).first()
            if not source:
                source = Source(name=paper_data["source"])
                session.add(source)
                session.flush()  # Get ID without committing transaction
            
            # Get or create journal
            journal = session.query(Journal).filter_by(name=paper_data["journal"]).first()
            if not journal:
                journal = Journal(name=paper_data["journal"])
                session.add(journal)
                session.flush()
            
            # Parse publication date
            try:
                if isinstance(paper_data["publication_date"], str):
                    pub_date = datetime.datetime.strptime(paper_data["publication_date"], "%Y %b %d")
                else:
                    pub_date = None
            except ValueError:
                pub_date = None
            
            # Create new paper
            new_paper = Paper(
                title=paper_data["title"],
                publication_date=pub_date,
                url=paper_data["url"],
                abstract_vector_id=abstract_vector_id,
                summary_vector_id=summary_vector_id,
                journal_id=journal.id,
                source_id=source.id,
                abstract=paper_data.get("abstract"),
                summary=paper_data.get("summary")
            )
            session.add(new_paper)
            session.flush()
            
            # Add authors
            for author_name in paper_data["authors"]:
                author = session.query(Author).filter_by(name=author_name).first()
                if not author:
                    author = Author(name=author_name)
                    session.add(author)
                    session.flush()
                new_paper.authors.append(author)
            
            # Commit transaction
            session.commit()
            return new_paper.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding paper to SQL database: {e}")
            raise
        finally:
            session.close()
    
    async def update_paper_summary(self, paper_id: int, summary: str, summary_vector_id: str) -> bool:
        """
        Update a paper's summary and its vector ID
        
        Args:
            paper_id: ID of the paper
            summary: Generated summary text
            summary_vector_id: ID of summary in vector database
            
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            paper = session.query(Paper).filter_by(id=paper_id).first()
            if not paper:
                logger.warning(f"Paper with ID {paper_id} not found")
                return False
            
            paper.summary = summary
            paper.summary_vector_id = summary_vector_id
            paper.updated_at = datetime.datetime.now()
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating paper summary: {e}")
            return False
        finally:
            session.close()
    
    async def get_paper_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get paper details by URL"""
        session = self.Session()
        try:
            paper = session.query(Paper).filter_by(url=url).first()
            if not paper:
                return None
            
            return {
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal.name if paper.journal else None,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id,
                "abstract": paper.abstract,
                "summary": paper.summary
            }
        except Exception as e:
            logger.error(f"Error getting paper by URL: {e}")
            return None
        finally:
            session.close()
    
    async def get_paper_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get paper details by ID"""
        session = self.Session()
        try:
            paper = session.query(Paper).filter_by(id=paper_id).first()
            if not paper:
                return None
            
            return {
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal.name if paper.journal else None,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id,
                "abstract": paper.abstract,
                "summary": paper.summary
            }
        except Exception as e:
            logger.error(f"Error getting paper by ID: {e}")
            return None
        finally:
            session.close()
    
    async def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Basic search of papers by title and abstract
        Note: This is a simple SQL-based search. Use vector search for better results.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of paper details
        """
        session = self.Session()
        try:
            # Creating a search pattern for SQL LIKE
            search_pattern = f"%{query}%"
            
            papers = session.query(Paper).filter(
                sqlalchemy.or_(
                    Paper.title.like(search_pattern),
                    Paper.abstract.like(search_pattern)
                )
            ).order_by(Paper.publication_date.desc()).limit(limit).all()
            
            return [{
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal.name if paper.journal else None,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id
            } for paper in papers]
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
        finally:
            session.close()
    
    async def get_papers_by_vector_ids(self, vector_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get papers by their vector IDs (for use after vector search)
        
        Args:
            vector_ids: List of vector IDs (could be abstract_vector_id or summary_vector_id)
            
        Returns:
            List of paper details
        """
        session = self.Session()
        try:
            papers = session.query(Paper).filter(
                sqlalchemy.or_(
                    Paper.abstract_vector_id.in_(vector_ids),
                    Paper.summary_vector_id.in_(vector_ids)
                )
            ).all()
            
            # Sort the results to match the order of vector_ids
            result = []
            id_to_paper = {
                p.abstract_vector_id: p for p in papers if p.abstract_vector_id in vector_ids
            }
            id_to_paper.update({
                p.summary_vector_id: p for p in papers if p.summary_vector_id in vector_ids
            })
            
            for vid in vector_ids:
                if vid in id_to_paper:
                    paper = id_to_paper[vid]
                    result.append({
                        "id": paper.id,
                        "title": paper.title,
                        "url": paper.url,
                        "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                        "journal": paper.journal.name if paper.journal else None,
                        "authors": [author.name for author in paper.authors],
                        "source": paper.source.name if paper.source else None,
                        "abstract": paper.abstract,
                        "summary": paper.summary,
                        "abstract_vector_id": paper.abstract_vector_id,
                        "summary_vector_id": paper.summary_vector_id
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting papers by vector IDs: {e}")
            return []
        finally:
            session.close()
    
    async def get_recent_papers(self, limit: int = 10, source: str = None) -> List[Dict[str, Any]]:
        """
        Get most recently added papers
        
        Args:
            limit: Maximum number of papers to return
            source: Optional filter by source (PubMed, Google Scholar)
            
        Returns:
            List of paper details
        """
        session = self.Session()
        try:
            query = session.query(Paper).order_by(Paper.publication_date.desc())
            
            if source:
                source_obj = session.query(Source).filter_by(name=source).first()
                if source_obj:
                    query = query.filter_by(source_id=source_obj.id)
            
            recent_papers = query.limit(limit).all()
            
            return [{
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal.name if paper.journal else None,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id
            } for paper in recent_papers]
            
        except Exception as e:
            logger.error(f"Error getting recent papers: {e}")
            return []
        finally:
            session.close()
    
    async def cache_query(self, query_text: str, paper_ids: List[int]) -> int:
        """
        Cache a query and its results
        
        Args:
            query_text: Text of the query
            paper_ids: List of paper IDs in result order
            
        Returns:
            ID of the cached query
        """
        session = self.Session()
        try:
            # Create query with 24-hour expiration
            new_query = Query(
                query_text=query_text,
                expires_at=datetime.datetime.now() + datetime.timedelta(days=1)
            )
            session.add(new_query)
            session.flush()
            
            # Add query results
            for rank, paper_id in enumerate(paper_ids):
                result = QueryResult(
                    query_id=new_query.id,
                    paper_id=paper_id,
                    rank=rank
                )
                session.add(result)
            
            session.commit()
            return new_query.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error caching query: {e}")
            raise
        finally:
            session.close()
    
    async def get_cached_query(self, query_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached query results if available and not expired
        
        Args:
            query_text: Text of the query
            
        Returns:
            List of paper details or None if not cached
        """
        session = self.Session()
        try:
            # Find non-expired cached query
            query = session.query(Query).filter(
                Query.query_text == query_text,
                Query.expires_at > datetime.datetime.now()
            ).first()
            
            if not query:
                return None
            
            # Get results in rank order
            results = session.query(QueryResult).filter_by(query_id=query.id).order_by(QueryResult.rank).all()
            
            # Get paper details for each result
            papers = []
            for result in results:
                paper = result.paper
                papers.append({
                    "id": paper.id,
                    "title": paper.title,
                    "url": paper.url,
                    "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                    "journal": paper.journal.name if paper.journal else None,
                    "authors": [author.name for author in paper.authors],
                    "source": paper.source.name if paper.source else None,
                    "abstract": paper.abstract,
                    "summary": paper.summary,
                    "abstract_vector_id": paper.abstract_vector_id,
                    "summary_vector_id": paper.summary_vector_id
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"Error getting cached query: {e}")
            return None
        finally:
            session.close()
    
    async def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent queries"""
        session = self.Session()
        try:
            recent_queries = session.query(Query).order_by(Query.created_at.desc()).limit(limit).all()
            return [{
                "id": query.id,
                "query_text": query.query_text,
                "created_at": query.created_at.isoformat()
            } for query in recent_queries]
        except Exception as e:
            logger.error(f"Error getting recent queries: {e}")
            return []
        finally:
            session.close()
    
    async def get_papers_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search papers by keyword in title or abstract
        
        Args:
            keyword: Keyword to search for
            limit: Maximum number of papers to return
            
        Returns:
            List of paper details
        """
        session = self.Session()
        try:
            search_pattern = f"%{keyword}%"
            papers = session.query(Paper).filter(
                sqlalchemy.or_(
                    Paper.title.like(search_pattern),
                    Paper.abstract.like(search_pattern)
                )
            ).order_by(Paper.publication_date.desc()).limit(limit).all()
            
            return [{
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal.name if paper.journal else None,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id
            } for paper in papers]
            
        except Exception as e:
            logger.error(f"Error searching papers by keyword: {e}")
            return []
        finally:
            session.close()
    
    async def get_papers_by_date_range(self, start_date: datetime.datetime, end_date: datetime.datetime, 
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get papers published within a date range
        
        Args:
            start_date: Start date of range
            end_date: End date of range
            limit: Maximum number of papers to return
            
        Returns:
            List of paper details
        """
        session = self.Session()
        try:
            papers = session.query(Paper).filter(
                Paper.publication_date >= start_date,
                Paper.publication_date <= end_date
            ).order_by(Paper.publication_date.desc()).limit(limit).all()
            
            return [{
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal.name if paper.journal else None,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id
            } for paper in papers]
            
        except Exception as e:
            logger.error(f"Error getting papers by date range: {e}")
            return []
        finally:
            session.close()
    
    async def get_papers_by_author(self, author_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get papers by a specific author
        
        Args:
            author_name: Name of the author
            limit: Maximum number of papers to return
            
        Returns:
            List of paper details
        """
        session = self.Session()
        try:
            author = session.query(Author).filter_by(name=author_name).first()
            if not author:
                return []
            
            papers = author.papers[:limit]
            
            return [{
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": paper.journal.name if paper.journal else None,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id
            } for paper in papers]
            
        except Exception as e:
            logger.error(f"Error getting papers by author: {e}")
            return []
        finally:
            session.close()
    
    async def get_papers_by_journal(self, journal_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get papers published in a specific journal
        
        Args:
            journal_name: Name of the journal
            limit: Maximum number of papers to return
            
        Returns:
            List of paper details
        """
        session = self.Session()
        try:
            journal = session.query(Journal).filter_by(name=journal_name).first()
            if not journal:
                return []
            
            papers = session.query(Paper).filter_by(journal_id=journal.id)\
                .order_by(Paper.publication_date.desc()).limit(limit).all()
            
            return [{
                "id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": journal.name,
                "authors": [author.name for author in paper.authors],
                "source": paper.source.name if paper.source else None,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "abstract_vector_id": paper.abstract_vector_id,
                "summary_vector_id": paper.summary_vector_id
            } for paper in papers]
            
        except Exception as e:
            logger.error(f"Error getting papers by journal: {e}")
            return []
        finally:
            session.close()
    
    async def clear_expired_cache(self) -> int:
        """
        Clear expired cached queries
        
        Returns:
            Number of cleared entries
        """
        session = self.Session()
        try:
            # Delete query results first (due to foreign key constraint)
            expired_query_ids = [q.id for q in session.query(Query).filter(
                Query.expires_at < datetime.datetime.now()
            ).all()]
            
            if not expired_query_ids:
                return 0
            
            # Delete related query results
            deleted_results = session.query(QueryResult).filter(
                QueryResult.query_id.in_(expired_query_ids)
            ).delete(synchronize_session=False)
            
            # Delete expired queries
            deleted_queries = session.query(Query).filter(
                Query.expires_at < datetime.datetime.now()
            ).delete(synchronize_session=False)
            
            session.commit()
            return deleted_queries
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing expired cache: {e}")
            return 0
        finally:
            session.close()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the database contents
        
        Returns:
            Dictionary with statistics
        """
        session = self.Session()
        try:
            paper_count = session.query(Paper).count()
            author_count = session.query(Author).count()
            journal_count = session.query(Journal).count()
            query_count = session.query(Query).count()
            
            # Get paper count by source
            source_stats = {}
            sources = session.query(Source).all()
            for source in sources:
                source_stats[source.name] = session.query(Paper).filter_by(source_id=source.id).count()
            
            # Get papers per month (last 6 months)
            today = datetime.datetime.now()
            six_months_ago = today - datetime.timedelta(days=180)
            
            monthly_papers = {}
            for i in range(6):
                month_start = today - datetime.timedelta(days=30 * i)
                month_end = today - datetime.timedelta(days=30 * (i-1)) if i > 0 else today
                month_name = month_start.strftime("%Y-%m")
                
                count = session.query(Paper).filter(
                    Paper.publication_date >= month_start,
                    Paper.publication_date < month_end
                ).count()
                
                monthly_papers[month_name] = count
            
            return {
                "total_papers": paper_count,
                "total_authors": author_count,
                "total_journals": journal_count,
                "total_queries": query_count,
                "papers_by_source": source_stats,
                "papers_by_month": monthly_papers,
                "last_updated": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {
                "error": str(e),
                "last_updated": datetime.datetime.now().isoformat()
            }
        finally:
            session.close()