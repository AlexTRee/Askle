# modules/paper_retrieval.py
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from scholarly import scholarly
import datetime
import time
import backoff
import re

# Configure logging
# It's often better to configure logging in the main application entry point (e.g., app.py)
# But if this module is run standalone or needs specific logging, configure here.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class PaperInfo:
    """
    Data class to hold structured information about a retrieved paper.
    """
    def __init__(self, title: str, authors: List[str], publication_date: str,
                 journal: str, abstract: str, url: str, source: str):
        self.title = title if title else "Unknown Title"
        self.authors = authors if authors else ["Unknown Authors"]
        self.publication_date = publication_date if publication_date else "Unknown Date"
        self.journal = journal if journal else "Unknown Journal"
        self.abstract = abstract if abstract else "Abstract not available"
        self.url = url if url else "No URL provided"
        self.source = source # e.g., "PubMed", "Google Scholar"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the paper info to a dictionary for easier serialization (e.g., JSON)."""
        return {
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "journal": self.journal,
            # The abstract will be used for summarization later in the workflow
            "abstract": self.abstract,
            "url": self.url,
            "source": self.source
        }

class PubMedRetriever:
    """
    Class responsible for retrieving paper information from PubMed via the E-utilities API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the PubMedRetriever.

        Args:
            api_key: An optional NCBI API key for potentially higher request rates.
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = f"{self.base_url}esearch.fcgi"
        self.fetch_url = f"{self.base_url}efetch.fcgi"
        self.api_key = api_key
        # Consider using a shared aiohttp.ClientSession if making many calls across the app
        # For simplicity here, a new session is created per search call.

    async def search(self, query: str, max_results: int = 5) -> List[PaperInfo]:
        """
        Searches PubMed for papers matching the provided query terms.
        This query should ideally be the keywords extracted by the AI model.

        Args:
            query: The search query string (e.g., "mRNA vaccines AND efficacy").
            max_results: The maximum number of paper details to retrieve.

        Returns:
            A list of PaperInfo objects containing details of the found papers.
            Returns an empty list if no results are found or an error occurs.
        """
        logger.info(f"Starting PubMed search for query: '{query}' with max_results={max_results}")
        paper_infos = []
        try:
            # Filter for recent papers (last 30 days) - Adjust timeframe if needed
            # thirty_days_ago = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y/%m/%d")
            # query_with_date = f"({query}) AND ({thirty_days_ago}[Date - Publication] : 3000[Date - Publication])" # More robust date range
            # Removing date filter for now as it might be too restrictive, can be added back if needed.
            query_term = query

            async with aiohttp.ClientSession() as session:
                # 1. Search for PubMed IDs (PMIDs)
                search_params = {
                    "db": "pubmed",
                    "term": query_term,
                    "retmode": "json",
                    "retmax": max_results,
                    "sort": "relevance", # Or 'pub+date' for most recent first
                    "usehistory": "y" # Use history for efficient fetching
                }
                if self.api_key:
                    search_params["api_key"] = self.api_key

                search_response_json = None
                try:
                    logger.debug(f"PubMed search request params: {search_params}")
                    async with session.get(self.search_url, params=search_params, timeout=30) as response:
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        search_response_json = await response.json()
                        logger.debug(f"PubMed search response JSON: {search_response_json}")

                except aiohttp.ClientResponseError as e:
                    logger.error(f"PubMed search HTTP error: {e.status} {e.message} for query '{query}'")
                    return []
                except aiohttp.ClientError as e:
                    logger.error(f"PubMed search connection error: {e} for query '{query}'")
                    return []
                except asyncio.TimeoutError:
                    logger.error(f"PubMed search request timed out for query '{query}'")
                    return []

                # Extract IDs and WebEnv/QueryKey for fetching
                esearchresult = search_response_json.get("esearchresult", {})
                ids = esearchresult.get("idlist", [])
                webenv = esearchresult.get("webenv")
                query_key = esearchresult.get("querykey")

                if not ids:
                    logger.info(f"No PubMed results found for query: '{query}'")
                    return []

                logger.info(f"Found {len(ids)} PubMed IDs. Fetching details...")

                # 2. Fetch detailed information for the found PMIDs using WebEnv
                fetch_params = {
                    "db": "pubmed",
                    "retmode": "xml",
                    "rettype": "abstract",
                    "retmax": max_results # Fetch details only for the number requested
                    # Use WebEnv and QueryKey for efficient fetching from history server
                    # "id": ",".join(ids), # Less efficient than WebEnv
                }
                if webenv and query_key:
                     fetch_params["WebEnv"] = webenv
                     fetch_params["query_key"] = query_key
                else: # Fallback if history is not available
                    fetch_params["id"] = ",".join(ids)

                if self.api_key:
                    fetch_params["api_key"] = self.api_key

                xml_content = None
                try:
                    logger.debug(f"PubMed fetch request params: {fetch_params}")
                    async with session.get(self.fetch_url, params=fetch_params, timeout=45) as response: # Increased timeout for fetch
                        response.raise_for_status()
                        xml_content = await response.text()
                        logger.debug("PubMed fetch successful, received XML content.")

                except aiohttp.ClientResponseError as e:
                    logger.error(f"PubMed fetch HTTP error: {e.status} {e.message}")
                    return []
                except aiohttp.ClientError as e:
                    logger.error(f"PubMed fetch connection error: {e}")
                    return []
                except asyncio.TimeoutError:
                    logger.error("PubMed fetch request timed out")
                    return []

                # 3. Parse the XML content
                if xml_content:
                    paper_infos = self._parse_pubmed_xml(xml_content)
                    logger.info(f"Successfully parsed {len(paper_infos)} papers from PubMed XML.")

        except Exception as e:
            # Catch-all for unexpected errors during the process
            logger.exception(f"Unexpected error during PubMed search for query '{query}': {e}")
            return [] # Return empty list on failure

        # Ensure we don't return more results than requested, although fetch_params['retmax'] should handle this
        return paper_infos[:max_results]

    def _parse_pubmed_xml(self, xml_content: str) -> List[PaperInfo]:
        """
        Parses the XML response from PubMed eFetch into a list of PaperInfo objects.

        Args:
            xml_content: The XML string received from PubMed.

        Returns:
            A list of PaperInfo objects extracted from the XML.
        """
        papers = []
        try:
            root = ET.fromstring(xml_content)

            for article_element in root.findall(".//PubmedArticle"):
                try:
                    # --- Extract PMID ---
                    pmid_element = article_element.find(".//MedlineCitation/PMID")
                    pmid = pmid_element.text if pmid_element is not None and pmid_element.text else None
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "URL not available"

                    # --- Extract Article Metadata ---
                    article = article_element.find(".//MedlineCitation/Article")
                    if article is None:
                        logger.warning(f"Skipping article with PMID {pmid} due to missing Article element.")
                        continue

                    # --- Title ---
                    title_element = article.find(".//ArticleTitle")
                    # Handle potential XML tags within title (e.g., <i>, <b>)
                    title = "".join(title_element.itertext()).strip() if title_element is not None else "Unknown Title"

                    # --- Authors ---
                    authors = []
                    author_list = article.find(".//AuthorList")
                    if author_list is not None:
                        for author in author_list.findall(".//Author"):
                            last_name = author.find(".//LastName")
                            fore_name = author.find(".//ForeName")
                            initials = author.find(".//Initials") # Sometimes ForeName is missing
                            collective_name = author.find(".//CollectiveName") # Handle group authors

                            if collective_name is not None and collective_name.text:
                                authors.append(collective_name.text.strip())
                            elif last_name is not None and last_name.text:
                                name_parts = []
                                if fore_name is not None and fore_name.text:
                                    name_parts.append(fore_name.text.strip())
                                elif initials is not None and initials.text: # Use initials if forename missing
                                     name_parts.append(initials.text.strip())
                                name_parts.append(last_name.text.strip())
                                authors.append(" ".join(name_parts))
                            elif last_name is not None and last_name.text is None and initials is not None and initials.text:
                                # Handle cases where only initials might be present with empty LastName tag
                                authors.append(initials.text.strip())


                    # --- Journal Info ---
                    journal_element = article.find(".//Journal")
                    journal_title = "Unknown Journal"
                    pub_date_str = "Unknown Date"

                    if journal_element is not None:
                        title_elem = journal_element.find(".//Title")
                        if title_elem is not None and title_elem.text:
                            journal_title = title_elem.text.strip()

                        # --- Publication Date (more robust parsing) ---
                        pub_date_element = journal_element.find(".//JournalIssue/PubDate")
                        if pub_date_element is not None:
                            year = pub_date_element.findtext(".//Year")
                            month = pub_date_element.findtext(".//Month")
                            day = pub_date_element.findtext(".//Day")
                            medline_date = pub_date_element.findtext(".//MedlineDate") # Alternative format

                            if year:
                                date_parts = [year]
                                if month: date_parts.append(month)
                                if day: date_parts.append(day)
                                pub_date_str = " ".join(date_parts)
                            elif medline_date:
                                pub_date_str = medline_date # e.g., "2023 Spring" or "2023 Jan-Feb"

                    # --- Abstract ---
                    abstract = "Abstract not available"
                    abstract_element = article.find(".//Abstract")
                    if abstract_element is not None:
                        # Collect text from all AbstractText elements, handling labels
                        abstract_parts = []
                        for text_node in abstract_element.findall(".//AbstractText"):
                            node_text = "".join(text_node.itertext()).strip() if text_node is not None else ""
                            if node_text:
                                label = text_node.get("Label")
                                if label:
                                    abstract_parts.append(f"**{label.strip()}**: {node_text}")
                                else:
                                    abstract_parts.append(node_text)
                        if abstract_parts:
                            abstract = "\n\n".join(abstract_parts) # Join sections with double newline

                    # --- Create PaperInfo Object ---
                    if title != "Unknown Title": # Basic quality check
                        paper = PaperInfo(
                            title=title,
                            authors=authors if authors else ["Unknown Authors"],
                            publication_date=pub_date_str,
                            journal=journal_title,
                            abstract=abstract.strip(),
                            url=url,
                            source="PubMed"
                        )
                        papers.append(paper)
                    else:
                        logger.warning(f"Skipping article with PMID {pmid} due to missing title.")

                except Exception as e:
                    # Log error for specific article but continue parsing others
                    logger.error(f"Error parsing PubMed article (PMID: {pmid if pmid else 'Unknown'}): {e}", exc_info=True)
                    continue # Skip to the next article

        except ET.ParseError as e:
            logger.error(f"XML parsing failed for PubMed response: {e}")
        except Exception as e:
            # Catch-all for unexpected parsing errors
            logger.exception(f"Unexpected error parsing PubMed XML: {e}")

        return papers


class GoogleScholarRetriever:
    """
    Class responsible for retrieving paper information from Google Scholar using the 'scholarly' library.
    Note: Google Scholar can be sensitive to automated scraping. Use responsibly and expect potential blocks.
    """

    def __init__(self, max_retries: int = 3, retry_delay_secs: int = 5):
        """
        Initializes the GoogleScholarRetriever.

        Args:
            max_retries: Maximum number of attempts to fetch full paper details if initial attempts fail.
            retry_delay_secs: Seconds to wait between retries.
        """
        self.max_retries = max_retries
        self.retry_delay_secs = retry_delay_secs
        # Configure scholarly settings if needed (e.g., proxy)
        # scholarly.use_proxy(...)

    async def search(self, query: str, max_results: int = 5) -> List[PaperInfo]:
        """
        Searches Google Scholar for papers matching the provided query terms.
        This query should ideally be the keywords extracted by the AI model.

        Args:
            query: The search query string.
            max_results: The maximum number of paper details to retrieve.

        Returns:
            A list of PaperInfo objects containing details of the found papers.
            Returns an empty list if no results are found or an error occurs.
        """
        logger.info(f"Starting Google Scholar search for query: '{query}' with max_results={max_results}")
        try:
            # scholarly is synchronous, run it in a thread pool executor to avoid blocking asyncio event loop
            loop = asyncio.get_event_loop()
            papers = await loop.run_in_executor(
                None,  # Uses the default ThreadPoolExecutor
                self._scholarly_search_wrapper,
                query,
                max_results
            )
            logger.info(f"Google Scholar search completed, found {len(papers)} papers.")
            return papers
        except Exception as e:
            # Catch errors occurring during the threaded execution setup or retrieval
            logger.exception(f"Error running Google Scholar search in executor for query '{query}': {e}")
            return []

    # Define the backoff decorator outside the method for clarity
    # Use a more specific exception if possible, but Exception catches broader network/parsing issues
    # common with scraping. The giveup condition prevents retrying on fatal scholarly errors.
    @backoff.on_exception(backoff.expo,
                          Exception, # Catch broader errors during fill
                          max_tries=3, # Matches self.max_retries conceptually
                          max_time=60, # Add a max time limit for backoff
                          giveup=lambda e: isinstance(e, (scholarly.scholarly.ScholarlyError, scholarly.scholarly.MaxTriesExceededException)),
                          logger=logger, # Log backoff attempts
                          )
    def _fill_with_retry(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempts to fetch the full publication details using scholarly.fill with retry logic.
        This method is decorated with @backoff.

        Args:
            result: A single search result dictionary from scholarly.search_pubs_generator.

        Returns:
            The filled publication dictionary, or None if filling fails after retries.
        """
        try:
            # The 'fill' operation fetches more details and can fail due to network issues or blocks
            pub = scholarly.fill(result)
            # Perform a basic check to see if filling actually added useful data (like abstract)
            if pub and 'bib' in pub and 'abstract' in pub['bib']:
                 logger.debug(f"Successfully filled details for publication: {pub.get('bib', {}).get('title', 'N/A')}")
                 return pub
            else:
                 logger.warning(f"Filling publication seemed incomplete (missing abstract?): {pub.get('bib', {}).get('title', 'N/A')}")
                 # Decide if incomplete fill is acceptable or should be treated as failure
                 # Returning pub here, but could return None if abstract is mandatory
                 return pub
        except Exception as e:
            logger.warning(f"scholarly.fill failed: {e}. Retrying if possible...")
            raise # Re-raise the exception for backoff to catch and handle retries

    def _scholarly_search_wrapper(self, query: str, max_results: int) -> List[PaperInfo]:
        """
        Internal synchronous wrapper that performs the Google Scholar search.
        This method is executed in a separate thread by run_in_executor.

        Args:
            query: The search query string.
            max_results: Maximum number of results to retrieve.

        Returns:
            List of PaperInfo objects.
        """
        papers = []
        try:
            logger.info(f"[Thread Pool] Executing scholarly search for: '{query}'")
            # Add date filter: papers published since the start of the previous year. Adjust as needed.
            current_year = datetime.datetime.now().year
            # Note: Google Scholar date filtering can sometimes be approximate.
            search_query_gen = scholarly.search_pubs(f"{query}", year_low=current_year - 1) # Search since last year

            retrieved_count = 0
            for result in search_query_gen:
                if retrieved_count >= max_results:
                    logger.info(f"[Thread Pool] Reached max_results ({max_results}). Stopping Google Scholar search.")
                    break

                logger.debug(f"[Thread Pool] Processing raw result: {result.get('bib', {}).get('title', 'N/A')}")

                # Attempt to fill the publication details with retry logic
                try:
                    # Call the method decorated with @backoff
                    pub = self._fill_with_retry(result)
                except Exception as e:
                    # This catches errors if backoff itself fails or gives up
                    logger.error(f"[Thread Pool] Failed to fill publication details after retries for result {result.get('bib', {}).get('title', 'N/A')}: {e}")
                    pub = None # Ensure pub is None if filling ultimately fails

                if not pub:
                    logger.warning(f"[Thread Pool] Skipping result due to fill failure: {result.get('bib', {}).get('title', 'N/A')}")
                    continue # Skip this paper if details couldn't be fetched

                # --- Extract fields safely using .get() ---
                bib = pub.get('bib', {})
                title = bib.get('title', 'Unknown Title')

                # Authors can be a string or list, normalize to list
                raw_authors = bib.get('author', 'Unknown Authors')
                if isinstance(raw_authors, str) and raw_authors:
                    authors = [auth.strip() for auth in raw_authors.split(' and ')]
                elif isinstance(raw_authors, list):
                    authors = raw_authors
                else:
                    authors = ['Unknown Authors']

                # Journal/Venue/Publisher info
                journal = bib.get('journal', bib.get('venue', bib.get('publisher', 'Unknown Journal')))

                # Publication Year
                year = str(bib.get('pub_year', 'Unknown Date')) # Ensure it's a string

                # Abstract
                abstract = bib.get('abstract', 'Abstract not available')

                # URL - Prioritize official pub URL, then eprint, then general URL
                url = pub.get('pub_url', pub.get('eprint_url', pub.get('url', 'No URL found')))
                # Fallback to cluster URL if others are missing
                if not url and pub.get('cluster_id'):
                    url = f"https://scholar.google.com/scholar?cluster={pub.get('cluster_id')}"
                elif not url:
                     url = "https://scholar.google.com" # Generic fallback

                # --- Create PaperInfo Object ---
                if title != "Unknown Title" and abstract != "Abstract not available": # Basic quality check
                    paper = PaperInfo(
                        title=title,
                        authors=authors,
                        publication_date=year,
                        journal=journal,
                        abstract=abstract.strip(),
                        url=url,
                        source="Google Scholar"
                    )
                    papers.append(paper)
                    retrieved_count += 1
                    logger.debug(f"[Thread Pool] Successfully processed paper: {title}")
                else:
                    logger.warning(f"[Thread Pool] Skipping paper due to missing title or abstract: {title}")

        # Catch specific scholarly errors if needed, otherwise broad Exception
        except scholarly.scholarly.ScholarlyError as e:
            logger.error(f"[Thread Pool] Google Scholar API/access error: {e}")
        except Exception as e:
            # Catch unexpected errors during the search loop
            logger.exception(f"[Thread Pool] Unexpected error during Google Scholar search execution for query '{query}': {e}")

        return papers

# --- Example Usage (Optional - for testing this module directly) ---
async def main_test(test_query: str):
    """Example of how to use the retrievers."""
    print(f"Testing retrievers with query: '{test_query}'")

    pubmed_retriever = PubMedRetriever()
    google_retriever = GoogleScholarRetriever()

    # Run searches concurrently
    pubmed_task = asyncio.create_task(pubmed_retriever.search(test_query, max_results=5))
    google_task = asyncio.create_task(google_retriever.search(test_query, max_results=5))

    pubmed_results, google_results = await asyncio.gather(pubmed_task, google_task)

    print("\n--- PubMed Results ---")
    if pubmed_results:
        for i, paper in enumerate(pubmed_results):
            print(f"{i+1}. Title: {paper.title}")
            print(f"   Authors: {', '.join(paper.authors)}")
            print(f"   Journal: {paper.journal} ({paper.publication_date})")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:150]}...") # Print snippet
            print("-" * 10)
    else:
        print("No results found or error occurred.")

    print("\n--- Google Scholar Results ---")
    if google_results:
        for i, paper in enumerate(google_results):
            print(f"{i+1}. Title: {paper.title}")
            print(f"   Authors: {', '.join(paper.authors)}")
            print(f"   Journal: {paper.journal} ({paper.publication_date})")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:150]}...") # Print snippet
            print("-" * 10)
    else:
        print("No results found or error occurred.")

    all_results = pubmed_results + google_results
    print(f"\nTotal papers retrieved: {len(all_results)}")

if __name__ == "__main__":
    # Example of running the test function
    # Replace with keywords relevant to your domain
    # example_keywords = "mRNA vaccine efficacy COVID-19 clinical trial"
    example_keywords = "immune checkpoint inhibitors combination therapy melanoma"

    # Use asyncio.run() for Python 3.7+
    asyncio.run(main_test(example_keywords))
