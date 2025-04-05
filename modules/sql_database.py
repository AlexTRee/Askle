# modules/sql_database.py
import logging
import sqlalchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    ForeignKey, Text, Table, or_, func
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
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def add_paper(self, paper_data: Dict[str, Any], abstract_vector_id: str, summary_vector_id: str) -> int:
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

        with session_scope() as session:
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

    # Other methods (update, get, search) should similarly use context-managed sessions,
    # parameterize queries to avoid SQL injection, and handle None cases gracefully.

# End of revised module
