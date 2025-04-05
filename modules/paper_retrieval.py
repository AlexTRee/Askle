# modules/paper_retrieval.py
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import scholarly
import datetime
import time
import backoff

logger = logging.getLogger(__name__)

class PaperInfo:
    """Data class for paper information"""
    def __init__(self, title: str, authors: List[str], publication_date: str, 
                journal: str, abstract: str, url: str, source: str):
        self.title = title
        self.authors = authors
        self.publication_date = publication_date
        self.journal = journal
        self.abstract = abstract
        self.url = url
        self.source = source
    
    def to_dict(self):
        """Convert the paper info to a dictionary"""
        return {
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "journal": self.journal,
            "abstract": self.abstract,
            "url": self.url,
            "source": self.source
        }

class PubMedRetriever:
    """Class to retrieve papers from PubMed"""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = f"{self.base_url}esearch.fcgi"
        self.fetch_url = f"{self.base_url}efetch.fcgi"
        
    async def search(self, query: str, max_results: int = 5) -> List[PaperInfo]:
        """
        Search PubMed for papers matching the query
        
        Args:
            query: Search terms
            max_results: Maximum number of results to return
            
        Returns:
            List of PaperInfo objects
        """
        try:
            # Filter for recent papers (last 30 days)
            thirty_days_ago = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y/%m/%d")
            query_with_date = f"({query}) AND {thirty_days_ago}[PDAT]"
            
            # First get the IDs of matching papers
            async with aiohttp.ClientSession() as session:
                search_params = {
                    "db": "pubmed",
                    "term": query_with_date,
                    "retmode": "json",
                    "retmax": max_results,
                    "sort": "date"  # Sort by date to get the most recent
                }
                
                try:
                    async with session.get(self.search_url, params=search_params, timeout=30) as response:
                        if response.status != 200:
                            logger.error(f"PubMed search failed with status {response.status}: {await response.text()}")
                            return []
                        
                        search_results = await response.json()
                        ids = search_results.get("esearchresult", {}).get("idlist", [])
                        
                        if not ids:
                            logger.info(f"No PubMed results found for query: {query}")
                            return []
                except aiohttp.ClientError as e:
                    logger.error(f"Network error during PubMed search: {e}")
                    return []
                except asyncio.TimeoutError:
                    logger.error("PubMed search request timed out")
                    return []
                
                # Now fetch details for these IDs
                try:
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "xml",
                        "rettype": "abstract"
                    }
                    
                    async with session.get(self.fetch_url, params=fetch_params, timeout=30) as response:
                        if response.status != 200:
                            logger.error(f"PubMed fetch failed with status {response.status}: {await response.text()}")
                            return []
                        
                        xml_content = await response.text()
                        
                        # Parse XML and extract paper information
                        return self._parse_pubmed_xml(xml_content)
                except aiohttp.ClientError as e:
                    logger.error(f"Network error during PubMed fetch: {e}")
                    return []
                except asyncio.TimeoutError:
                    logger.error("PubMed fetch request timed out")
                    return []
        except Exception as e:
            logger.error(f"Unexpected error in PubMed search: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[PaperInfo]:
        """
        Parse PubMed XML response and extract paper information
        
        Args:
            xml_content: XML content returned from PubMed
            
        Returns:
            List of PaperInfo objects
        """
        papers = []
        try:
            root = ET.fromstring(xml_content)
            
            for article_element in root.findall(".//PubmedArticle"):
                try:
                    # Extract article metadata
                    medline_citation = article_element.find(".//MedlineCitation")
                    if medline_citation is None:
                        continue
                        
                    article = medline_citation.find(".//Article")
                    if article is None:
                        continue
                    
                    # Title
                    title_element = article.find(".//ArticleTitle")
                    title = title_element.text if title_element is not None and title_element.text else "Unknown Title"
                    
                    # Authors
                    authors = []
                    author_list = article.find(".//AuthorList")
                    if author_list is not None:
                        for author in author_list.findall(".//Author"):
                            last_name = author.find(".//LastName")
                            fore_name = author.find(".//ForeName")
                            if last_name is not None and fore_name is not None:
                                author_name = f"{fore_name.text if fore_name.text else ''} {last_name.text if last_name.text else ''}".strip()
                                if author_name:
                                    authors.append(author_name)
                            elif last_name is not None and last_name.text:
                                authors.append(last_name.text)
                    
                    # Journal info
                    journal_element = article.find(".//Journal")
                    journal_title = "Unknown Journal"
                    if journal_element is not None:
                        title_elem = journal_element.find(".//Title")
                        if title_elem is not None and title_elem.text:
                            journal_title = title_elem.text
                    
                    # Publication date
                    publication_date = "Unknown Date"
                    if journal_element is not None:
                        pub_date_element = journal_element.find(".//PubDate")
                        if pub_date_element is not None:
                            year_elem = pub_date_element.find(".//Year")
                            month_elem = pub_date_element.find(".//Month")
                            day_elem = pub_date_element.find(".//Day")
                            
                            year = year_elem.text if year_elem is not None and year_elem.text else ""
                            month = month_elem.text if month_elem is not None and month_elem.text else ""
                            day = day_elem.text if day_elem is not None and day_elem.text else ""
                            
                            date_parts = [part for part in [year, month, day] if part]
                            if date_parts:
                                publication_date = " ".join(date_parts)
                    
                    # Abstract
                    abstract = "Abstract not available"
                    abstract_element = article.find(".//Abstract")
                    if abstract_element is not None:
                        abstract_texts = abstract_element.findall(".//AbstractText")
                        abstract_parts = []
                        for text in abstract_texts:
                            if text.text:
                                label = text.get("Label")
                                if label:
                                    abstract_parts.append(f"{label}: {text.text}")
                                else:
                                    abstract_parts.append(text.text)
                        if abstract_parts:
                            abstract = "\n".join(abstract_parts)
                    
                    # URL
                    url = "https://pubmed.ncbi.nlm.nih.gov/"
                    pmid_element = medline_citation.find(".//PMID")
                    if pmid_element is not None and pmid_element.text:
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_element.text}/"
                    
                    paper = PaperInfo(
                        title=title,
                        authors=authors,
                        publication_date=publication_date,
                        journal=journal_title,
                        abstract=abstract.strip(),
                        url=url,
                        source="PubMed"
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error parsing PubMed article: {e}")
                    continue
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing PubMed XML: {e}")
                    
        return papers

class GoogleScholarRetriever:
    """Class to retrieve papers from Google Scholar"""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    async def search(self, query: str, max_results: int = 5) -> List[PaperInfo]:
        """
        Search Google Scholar for papers matching the query
        
        Args:
            query: Search terms
            max_results: Maximum number of results to return
            
        Returns:
            List of PaperInfo objects
        """
        # scholarly is synchronous, so we'll run it in a thread pool
        loop = asyncio.get_event_loop()
        try:
            papers = await loop.run_in_executor(
                None, self._scholarly_search, query, max_results
            )
            return papers
        except Exception as e:
            logger.error(f"Error in Google Scholar search thread: {e}")
            return []
    
    @backoff.on_exception(backoff.expo, 
                         (scholarly.scholarly.SearchError, Exception),
                         max_tries=3,
                         giveup=lambda e: isinstance(e, scholarly.scholarly.MaxTriesExceededException))
    def _scholarly_search(self, query: str, max_results: int) -> List[PaperInfo]:
        """
        Execute scholarly search in a thread pool
        
        Args:
            query: Search terms
            max_results: Maximum number of results to return
            
        Returns:
            List of PaperInfo objects
        """
        papers = []
        try:
            # Add date filter to get recent papers (Since previous year)
            current_year = datetime.datetime.now().year
            search_query = scholarly.search_pubs(f"{query} since:{current_year-1}")
            
            for i, result in enumerate(search_query):
                if i >= max_results:
                    break
                    
                # Get complete publication data with retry logic
                pub = None
                for attempt in range(self.max_retries):
                    try:
                        pub = scholarly.fill(result)
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            logger.error(f"Error filling Google Scholar publication after {self.max_retries} attempts: {e}")
                            continue
                        logger.warning(f"Retrying scholarly.fill after error: {e}")
                        time.sleep(self.retry_delay)
                
                if not pub:
                    continue
                
                # Extract fields with None checks
                title = pub.get('title', 'Unknown Title')
                
                # Extract authors
                authors = []
                raw_authors = pub.get('bib', {}).get('author', [])
                if isinstance(raw_authors, str):
                    authors = [raw_authors]
                elif isinstance(raw_authors, list):
                    authors = raw_authors
                
                # Extract journal/conference
                bib = pub.get('bib', {})
                journal = bib.get('journal')
                if not journal:
                    journal = bib.get('venue')  
                if not journal:
                    journal = 'Unknown Journal'
                
                # Extract year
                year = str(bib.get('pub_year', 'Unknown Year'))
                
                # Extract URL with fallbacks
                url = None
                for url_key in ['pub_url', 'eprint_url', 'url']:
                    url = pub.get(url_key)
                    if url:
                        break
                        
                if not url and 'cluster_id' in pub:
                    url = f"https://scholar.google.com/scholar?cluster={pub.get('cluster_id')}"
                elif not url:
                    url = "https://scholar.google.com"
                
                # Extract abstract
                abstract = bib.get('abstract', 'Abstract not available')
                
                paper = PaperInfo(
                    title=title,
                    authors=authors,
                    publication_date=year,
                    journal=journal,
                    abstract=abstract,
                    url=url,
                    source="Google Scholar"
                )
                
                papers.append(paper)
                
        except scholarly.scholarly.SearchError as e:
            logger.error(f"Google Scholar search error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Google Scholar search: {e}")
            
        return papers