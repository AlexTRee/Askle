# modules/paper_retrieval.py
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import scholarly
import datetime

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
        """Search PubMed for papers matching the query"""
        # Filter for recent papers (last 30 days)
        thirty_days_ago = (datetime.datetime.now() - datetime.datetime.timedelta(days=30)).strftime("%Y/%m/%d")
        query = f"({query}) AND {thirty_days_ago}[PDAT]"
        
        # First get the IDs of matching papers
        async with aiohttp.ClientSession() as session:
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "date"  # Sort by date to get the most recent
            }
            
            async with session.get(self.search_url, params=search_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search failed: {response.status}")
                    return []
                
                search_results = await response.json()
                ids = search_results.get("esearchresult", {}).get("idlist", [])
                
                if not ids:
                    logger.info(f"No PubMed results found for query: {query}")
                    return []
            
            # Now fetch details for these IDs
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            async with session.get(self.fetch_url, params=fetch_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed fetch failed: {response.status}")
                    return []
                
                xml_content = await response.text()
                
                # Parse XML and extract paper information
                return self._parse_pubmed_xml(xml_content)
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[PaperInfo]:
        """Parse PubMed XML response and extract paper information"""
        papers = []
        root = ET.fromstring(xml_content)
        
        for article_element in root.findall(".//PubmedArticle"):
            try:
                # Extract article metadata
                medline_citation = article_element.find(".//MedlineCitation")
                article = medline_citation.find(".//Article")
                
                # Title
                title = article.find(".//ArticleTitle").text
                
                # Authors
                authors = []
                author_list = article.find(".//AuthorList")
                if author_list is not None:
                    for author in author_list.findall(".//Author"):
                        last_name = author.find(".//LastName")
                        fore_name = author.find(".//ForeName")
                        if last_name is not None and fore_name is not None:
                            authors.append(f"{fore_name.text} {last_name.text}")
                        elif last_name is not None:
                            authors.append(last_name.text)
                
                # Journal info
                journal_element = article.find(".//Journal")
                journal_title = journal_element.find(".//Title").text if journal_element.find(".//Title") is not None else "Unknown Journal"
                
                # Publication date
                pub_date_element = journal_element.find(".//PubDate")
                year = pub_date_element.find(".//Year").text if pub_date_element.find(".//Year") is not None else ""
                month = pub_date_element.find(".//Month").text if pub_date_element.find(".//Month") is not None else ""
                day = pub_date_element.find(".//Day").text if pub_date_element.find(".//Day") is not None else ""
                publication_date = f"{year} {month} {day}".strip()
                
                # Abstract
                abstract_element = article.find(".//Abstract")
                abstract = ""
                if abstract_element is not None:
                    abstract_texts = abstract_element.findall(".//AbstractText")
                    for text in abstract_texts:
                        label = text.get("Label")
                        if label:
                            abstract += f"{label}: {text.text}\n"
                        else:
                            abstract += f"{text.text}\n"
                
                # URL
                pmid = medline_citation.find(".//PMID").text
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
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
                
        return papers

class GoogleScholarRetriever:
    """Class to retrieve papers from Google Scholar"""
    
    def __init__(self):
        # Configure scholarly to use a proxy if needed
        pass
        
    async def search(self, query: str, max_results: int = 5) -> List[PaperInfo]:
        """Search Google Scholar for papers matching the query"""
        # Note: scholarly is synchronous, so we'll run it in a thread pool
        loop = asyncio.get_event_loop()
        papers = await loop.run_in_executor(
            None, self._scholarly_search, query, max_results
        )
        return papers
    
    def _scholarly_search(self, query: str, max_results: int) -> List[PaperInfo]:
        """Execute scholarly search in a thread pool"""
        papers = []
        try:
            # Add date filter to get recent papers
            search_query = scholarly.search_pubs(query + " since:2023")
            
            for i, result in enumerate(search_query):
                if i >= max_results:
                    break
                    
                # Get complete publication data
                try:
                    pub = scholarly.fill(result)
                except Exception as e:
                    logger.error(f"Error filling Google Scholar publication: {e}")
                    continue
                
                # Extract fields
                title = pub.get('title', 'Unknown Title')
                
                # Extract authors
                authors = pub.get('bib', {}).get('author', [])
                if isinstance(authors, str):
                    authors = [authors]
                
                # Extract journal/conference
                journal = pub.get('bib', {}).get('journal', 
                          pub.get('bib', {}).get('venue', 'Unknown Journal'))
                
                # Extract year
                year = pub.get('bib', {}).get('pub_year', 'Unknown Year')
                
                # Extract URL
                url = pub.get('pub_url')
                if not url and 'eprint_url' in pub:
                    url = pub['eprint_url']
                if not url:
                    url = f"https://scholar.google.com/scholar?cluster={pub.get('cluster_id', '')}"
                
                # Extract abstract
                abstract = pub.get('bib', {}).get('abstract', 'Abstract not available')
                
                paper = PaperInfo(
                    title=title,
                    authors=authors,
                    publication_date=str(year),
                    journal=journal,
                    abstract=abstract,
                    url=url,
                    source="Google Scholar"
                )
                
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error in Google Scholar search: {e}")
            
        return papers