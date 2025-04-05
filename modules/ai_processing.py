# modules/ai_processing.py
import logging
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

class DeepSeekProcessor:
    """Class to process papers using locally hosted DeepSeek model"""
    
    def __init__(self, model_path=None, device=None, max_token_limit=4096):
        # Default to environment variable or standard path if not provided
        if model_path is None:
            model_path = os.environ.get("DEEPSEEK_MODEL_PATH", "/deepseek-r1-distill-qwen-1.5b")
        
        self.max_token_limit = max_token_limit
        logger.info(f"Loading DeepSeek model from {model_path}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Determine device if not specified
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            # Configure model loading with reduced precision for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move model to device if not using device_map="auto"
            if self.device == "cpu" or (device is not None and device != "auto"):
                self.model = self.model.to(self.device)
            
            logger.info(f"DeepSeek model loaded on {self.device}")
            
            if self.device == "cpu":
                logger.warning("Running on CPU. Inference will be slow. Consider using a GPU for better performance.")
            
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {e}")
            raise RuntimeError(f"Failed to load DeepSeek model: {e}")
    
    async def summarize_paper(self, paper) -> Dict[str, Any]:
        """
        Summarize a paper using DeepSeek model
        
        Args:
            paper: Paper object containing title, authors, abstract, etc.
            
        Returns:
            Dictionary containing paper info and summary
        """
        try:
            # Prepare the prompt for DeepSeek
            prompt = self._create_summary_prompt(paper)
            
            # Check if prompt is too long and truncate if necessary
            token_count = len(self.tokenizer.encode(prompt))
            if token_count > self.max_token_limit - 500:  # Reserve 500 tokens for generation
                logger.warning(f"Prompt too long ({token_count} tokens). Truncating abstract.")
                # Recalculate with truncated abstract
                truncated_abstract = self._truncate_text(paper.abstract, self.max_token_limit - 1000)
                paper_copy = type('', (), {})()  # Create a simple object to hold truncated data
                for attr in ['title', 'authors', 'publication_date', 'journal', 'url', 'source']:
                    setattr(paper_copy, attr, getattr(paper, attr))
                paper_copy.abstract = truncated_abstract
                prompt = self._create_summary_prompt(paper_copy)
            
            # Generate summary using local model
            summary = await self._generate_summary(prompt)
            
            # Create and return the result
            result = {
                "title": paper.title,
                "authors": paper.authors,
                "publication_date": paper.publication_date,
                "journal": paper.journal,
                "abstract": paper.abstract,
                "summary": summary,
                "url": paper.url,
                "source": paper.source
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error summarizing paper '{paper.title}': {e}")
            # Return paper info with error message as summary
            return {
                "title": paper.title,
                "authors": paper.authors if hasattr(paper, 'authors') else [],
                "publication_date": paper.publication_date if hasattr(paper, 'publication_date') else "",
                "journal": paper.journal if hasattr(paper, 'journal') else "",
                "abstract": paper.abstract if hasattr(paper, 'abstract') else "",
                "summary": f"Error generating summary: {str(e)}",
                "url": paper.url if hasattr(paper, 'url') else "",
                "source": paper.source if hasattr(paper, 'source') else ""
            }
    
    async def batch_summarize_papers(self, papers: List) -> List[Dict[str, Any]]:
        """
        Summarize multiple papers in parallel
        
        Args:
            papers: List of paper objects
            
        Returns:
            List of dictionaries containing paper info and summaries
        """
        tasks = [self.summarize_paper(paper) for paper in papers]
        return await asyncio.gather(*tasks)
    
    def _create_summary_prompt(self, paper) -> str:
        """
        Create a prompt for DeepSeek to summarize a paper
        
        Args:
            paper: Paper object containing paper information
            
        Returns:
            Formatted prompt string
        """
        # Safely handle paper attributes
        title = getattr(paper, 'title', 'Unknown Title')
        authors = getattr(paper, 'authors', [])
        authors_str = ', '.join(authors) if authors else 'Unknown'
        journal = getattr(paper, 'journal', 'Unknown Journal')
        pub_date = getattr(paper, 'publication_date', 'Unknown Date')
        abstract = getattr(paper, 'abstract', 'No abstract available')
        
        return f"""<|im_start|>system
You are an AI assistant specializing in lung cancer research. Provide a concise summary of research papers.<|im_end|>

<|im_start|>user
Please provide a concise summary of the following research paper. Focus on key findings, methodology, and clinical implications.

Title: {title}
Authors: {authors_str}
Journal: {journal}
Publication Date: {pub_date}

Abstract:
{abstract}

Your summary should:
1. Highlight the main findings
2. Explain the methodology briefly
3. Discuss clinical implications
4. Be concise (150-200 words)
5. Use plain language accessible to medical professionals<|im_end|>

<|im_start|>assistant
"""
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated text
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        return truncated_text + "... [truncated due to length]"
    
    async def _generate_summary(self, prompt: str) -> str:
        """
        Generate summary using locally hosted DeepSeek model
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated summary text
        """
        try:
            # Run CPU/GPU-bound task in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, self._generate_text, prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error: Unable to generate summary with local DeepSeek model."
    
    @lru_cache(maxsize=100)  # Cache recent generations to improve performance
    def _generate_text(self, prompt: str) -> str:
        """
        Core text generation function that runs on the executor
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated text response
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=500,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract just the assistant's response using more robust pattern matching
            response_text = ""
            
            # Look for the assistant's response section
            if "<|im_start|>assistant" in generated_text:
                # Split by assistant marker and take the last part
                parts = generated_text.split("<|im_start|>assistant")
                response_part = parts[-1]
                
                # Remove any trailing markers
                for marker in ["<|im_end|>", "<|im_start|>"]:
                    if marker in response_part:
                        response_part = response_part.split(marker)[0]
                
                response_text = response_part.strip()
            else:
                # Fallback: return everything after the last line of the prompt
                prompt_last_line = "<|im_start|>assistant"
                response_text = generated_text[generated_text.find(prompt_last_line) + len(prompt_last_line):].strip()
            
            # Clean up any remaining special tokens
            special_tokens = ["<|endoftext|>", "<|im_end|>"]
            for token in special_tokens:
                response_text = response_text.replace(token, "")
                
            return response_text.strip()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory error: {e}")
            raise RuntimeError("GPU out of memory. Try reducing batch size or model size.")
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            raise