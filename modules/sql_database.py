# modules/sql_database.py
import logging
import sqlalchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    ForeignKey, Text, Table, or_, func, and_, desc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from typing import List, Dict, Any, Optional
import datetime
import os

logger = logging.getLogger(__name__)

# Base class for models
Base = declarative_base()

# Association table
paper_authors = Table(
    'paper_authors', Base.metadata,
    Column('paper_id', Integer, ForeignKey('papers.id', ondelete='CASCADE')),
    Column('author_id', Integer, ForeignKey('authors.id', ondelete='CASCADE'))
)

class Author(Base):
    __tablename__ = 'authors'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    papers = relationship(
        "Paper", secondary=paper_authors,
        back_populates="authors", cascade="all, delete"
    )

    def __repr__(self):
        return f"<Author(id={self.id}, name='{self.name}')>"

class Journal(Base):
    __tablename__ = 'journals'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    papers = relationship("Paper", back_populates="journal", cascade="all, delete")

    def __repr__(self):
        return f"<Journal(id={self.id}, name='{self.name}')>"

class Source(Base):
    __tablename__ = 'sources'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    papers = relationship("Paper", back_populates="source", cascade="all, delete")

    def __repr__(self):
        return f"<Source(id={self.id}, name='{self.name}')>"

class Paper(Base):
    __tablename__ = 'papers'
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    publication_date = Column(DateTime)
    url = Column(String(500), unique=True, nullable=False)
    abstract_vector_id = Column(String(255), index=True)
    summary_vector_id = Column(String(255), index=True)
    journal_id = Column(Integer, ForeignKey('journals.id', ondelete='SET NULL'))
    source_id = Column(Integer, ForeignKey('sources.id', ondelete='SET NULL'))
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    abstract = Column(Text)
    summary = Column(Text)

    journal = relationship("Journal", back_populates="papers")
    authors = relationship("Author", secondary=paper_authors, back_populates="papers")
    source = relationship("Source", back_populates="papers")

    def __repr__(self):
        journal_name = self.journal.name if self.journal else None
        return f"<Paper(id={self.id}, title='{self.title[:30]}...', journal='{journal_name}')>"

class Query(Base):
    __tablename__ = 'queries'
    id = Column(Integer, primary_key=True)
    query_text = Column(String(500), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    results = relationship("QueryResult", back_populates="query", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Query(id={self.id}, text='{self.query_text[:30]}...')>"

class QueryResult(Base):
    __tablename__ = 'query_results'
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey('queries.id', ondelete='CASCADE'), nullable=False)
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), nullable=False)
    rank = Column(Integer, nullable=False)
    query = relationship("Query", back_populates="results")
    paper = relationship("Paper")

    def __repr__(self):
        return f"<QueryResult(query_id={self.query_id}, paper_id={self.paper_id}, rank={self.rank})>"

class SQLDatabase:
    def __init__(self, connection_string: Optional[str] = None):
        # Setup DB path
        if connection_string is None:
            db_path = os.environ.get("SQL_DB_PATH", "./data/papers.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            connection_string = f"sqlite:///{db_path}"
        # Engine & session
        self.engine = create_engine(
            connection_string,
            connect_args={"check_same_thread": False} if connection_string.startswith("sqlite") else {}
        )
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        Base.metadata.create_all(self.engine)
        logger.info(f"Connected to DB: {connection_string}")

    def _session_scope(self):
        from contextlib import contextmanager
        @contextmanager
        def session_scope():
            session = self.Session()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        return session_scope()

    async def add_paper(self, paper_data: Dict[str, Any], abstract_vector_id: str, summary_vector_id: str) -> int:
        """Add a paper to the database or update if it already exists"""
        with self._session_scope() as session:
            # Check existing
            paper = session.query(Paper).filter_by(url=paper_data["url"]).first()
            now = datetime.datetime.utcnow()
            if paper:
                paper.abstract_vector_id = abstract_vector_id
                paper.summary_vector_id = summary_vector_id
                paper.updated_at = now
                if paper_data.get("abstract"): paper.abstract = paper_data["abstract"]
                if paper_data.get("summary"): paper.summary = paper_data["summary"]
                return paper.id
            # get-or-create helper
            def get_or_create(model, **kwargs):
                instance = session.query(model).filter_by(**kwargs).first()
                if instance:
                    return instance
                instance = model(**kwargs)
                session.add(instance)
                session.flush()
                return instance
            source = get_or_create(Source, name=paper_data["source"])
            journal = get_or_create(Journal, name=paper_data["journal"])
            # parse date
            pub_date = None
            pd = paper_data.get("publication_date")
            if isinstance(pd, str):
                for fmt in ("%Y-%m-%d", "%Y %b %d", "%Y-%m-%dT%H:%M:%S"):  # multiple formats
                    try:
                        pub_date = datetime.datetime.strptime(pd, fmt)
                        break
                    except ValueError:
                        continue
            new_paper = Paper(
                title=paper_data.get("title"),
                publication_date=pub_date,
                url=paper_data.get("url"),
                abstract_vector_id=abstract_vector_id,
                summary_vector_id=summary_vector_id,
                journal=journal,
                source=source,
                abstract=paper_data.get("abstract"),
                summary=paper_data.get("summary"),
                created_at=now,
                updated_at=now
            )
            session.add(new_paper)
            session.flush()
            # authors
            for name in paper_data.get("authors", []):
                author = get_or_create(Author, name=name)
                new_paper.authors.append(author)
            return new_paper.id

    async def get_cached_query(self, query_text: str) -> List[Dict[str, Any]]:
        """Get cached results for a query if it exists and is not expired"""
        with self._session_scope() as session:
            now = datetime.datetime.utcnow()
            query = session.query(Query).filter(
                and_(
                    Query.query_text == query_text,
                    Query.expires_at > now
                )
            ).first()
            
            if not query:
                return []
            
            # Query exists and is not expired
            results = []
            for qr in sorted(query.results, key=lambda x: x.rank):
                paper = qr.paper
                if not paper:
                    continue
                    
                # Format the paper data
                paper_dict = {
                    "id": paper.id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "summary": paper.summary,
                    "url": paper.url,
                    "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                    "source": paper.source.name if paper.source else "Unknown",
                    "journal": paper.journal.name if paper.journal else "Unknown",
                    "authors": [author.name for author in paper.authors]
                }
                results.append(paper_dict)
                
            return results

    async def get_paper_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get a paper by its URL"""
        with self._session_scope() as session:
            paper = session.query(Paper).filter_by(url=url).first()
            if not paper:
                return None
                
            return {
                "id": paper.id,
                "title": paper.title,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "url": paper.url,
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "source": paper.source.name if paper.source else "Unknown",
                "journal": paper.journal.name if paper.journal else "Unknown",
                "authors": [author.name for author in paper.authors]
            }

    async def cache_query(self, query_text: str, paper_ids: List[int], expires_days: int = 7):
        """Cache a query result with a list of paper IDs"""
        with self._session_scope() as session:
            # Delete existing query if exists
            existing_query = session.query(Query).filter_by(query_text=query_text).first()
            if existing_query:
                session.delete(existing_query)
                session.flush()
            
            # Create new query
            now = datetime.datetime.utcnow()
            expires_at = now + datetime.timedelta(days=expires_days)
            new_query = Query(
                query_text=query_text,
                created_at=now,
                expires_at=expires_at
            )
            session.add(new_query)
            session.flush()
            
            # Add results
            for rank, paper_id in enumerate(paper_ids):
                result = QueryResult(
                    query_id=new_query.id,
                    paper_id=paper_id,
                    rank=rank
                )
                session.add(result)
                
            return new_query.id

    async def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent queries with their creation timestamps"""
        with self._session_scope() as session:
            queries = session.query(Query)\
                .order_by(desc(Query.created_at))\
                .limit(limit)\
                .all()
                
            results = []
            for query in queries:
                results.append({
                    "id": query.id,
                    "query_text": query.query_text,
                    "created_at": query.created_at.isoformat()
                })
                
            return results

    async def get_recent_papers(self, limit: int = 10, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recently added papers, optionally filtered by source"""
        with self._session_scope() as session:
            query = session.query(Paper)\
                .join(Source, Paper.source_id == Source.id, isouter=True)\
                .order_by(desc(Paper.created_at))
                
            if source:
                query = query.filter(Source.name == source)
                
            papers = query.limit(limit).all()
            
            results = []
            for paper in papers:
                results.append({
                    "id": paper.id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "summary": paper.summary,
                    "url": paper.url,
                    "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                    "source": paper.source.name if paper.source else "Unknown",
                    "journal": paper.journal.name if paper.journal else "Unknown",
                    "authors": [author.name for author in paper.authors]
                })
                
            return results

    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self._session_scope() as session:
            paper_count = session.query(func.count(Paper.id)).scalar()
            author_count = session.query(func.count(Author.id)).scalar()
            journal_count = session.query(func.count(Journal.id)).scalar()
            query_count = session.query(func.count(Query.id)).scalar()
            
            # Source distribution
            source_counts = {}
            sources = session.query(Source.name, func.count(Paper.id))\
                .join(Paper, Source.id == Paper.source_id)\
                .group_by(Source.name)\
                .all()
                
            for source_name, count in sources:
                source_counts[source_name] = count
                
            # Recent activity
            now = datetime.datetime.utcnow()
            week_ago = now - datetime.timedelta(days=7)
            month_ago = now - datetime.timedelta(days=30)
            
            papers_last_week = session.query(func.count(Paper.id))\
                .filter(Paper.created_at >= week_ago)\
                .scalar()
                
            papers_last_month = session.query(func.count(Paper.id))\
                .filter(Paper.created_at >= month_ago)\
                .scalar()
                
            return {
                "total_papers": paper_count,
                "total_authors": author_count,
                "total_journals": journal_count,
                "total_queries": query_count,
                "source_distribution": source_counts,
                "papers_last_week": papers_last_week,
                "papers_last_month": papers_last_month,
                "updated_at": now.isoformat()
            }

    async def get_papers_by_vector_ids(self, vector_ids: List[str]) -> List[Dict[str, Any]]:
        """Get papers by their abstract_vector_id or summary_vector_id"""
        with self._session_scope() as session:
            papers = session.query(Paper).filter(
                or_(
                    Paper.abstract_vector_id.in_(vector_ids),
                    Paper.summary_vector_id.in_(vector_ids)
                )
            ).all()
            
            results = []
            for paper in papers:
                results.append({
                    "id": paper.id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "summary": paper.summary,
                    "url": paper.url,
                    "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                    "source": paper.source.name if paper.source else "Unknown",
                    "journal": paper.journal.name if paper.journal else "Unknown",
                    "authors": [author.name for author in paper.authors]
                })
                
            return results

    async def search_papers(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search papers by text match on title, abstract, and summary"""
        with self._session_scope() as session:
            # Create search pattern for SQL LIKE
            search_pattern = f"%{query_text}%"
            
            papers = session.query(Paper).filter(
                or_(
                    Paper.title.ilike(search_pattern),
                    Paper.abstract.ilike(search_pattern),
                    Paper.summary.ilike(search_pattern)
                )
            ).order_by(desc(Paper.publication_date)).limit(limit).all()
            
            results = []
            for paper in papers:
                results.append({
                    "id": paper.id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "summary": paper.summary,
                    "url": paper.url,
                    "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                    "source": paper.source.name if paper.source else "Unknown",
                    "journal": paper.journal.name if paper.journal else "Unknown",
                    "authors": [author.name for author in paper.authors]
                })
                
            return results