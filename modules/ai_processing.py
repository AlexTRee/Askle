# modules/ai_processing.py
import logging
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

logger = logging.getLogger(__name__)

class DeepSeekProcessor:
    """Class to process papers using locally hosted DeepSeek R1 14b model"""
    
    def __init__(self, model_path=None):
        # Default to environment variable or standard path if not provided
        if model_path is None:
            model_path = os.environ.get("DEEPSEEK_MODEL_PATH", "deepseek-ai/deepseek-llm-7b-chat")
        
        logger.info(f"Loading DeepSeek model from {model_path}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Configure model loading with reduced precision for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use half precision
                device_map="auto"  # Automatically distribute across available GPUs/devices
            )
            
            # Check if we have GPU acceleration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"DeepSeek model loaded on {self.device}")
            
            if self.device == "cpu":
                logger.warning("Running on CPU. Inference will be slow. Consider using a GPU for better performance.")
            
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {e}")
            raise RuntimeError(f"Failed to load DeepSeek model: {e}")
    
    async def summarize_paper(self, paper) -> Dict[str, Any]:
        """Summarize a paper using DeepSeek R1 14b"""
        try:
            # Prepare the prompt for DeepSeek
            prompt = self._create_summary_prompt(paper)
            
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
            logger.error(f"Error summarizing paper: {e}")
            # Return paper info with error message as summary
            return {
                "title": paper.title,
                "authors": paper.authors,
                "publication_date": paper.publication_date,
                "journal": paper.journal,
                "abstract": paper.abstract,
                "summary": f"Error generating summary: {str(e)}",
                "url": paper.url,
                "source": paper.source
            }
    
    def _create_summary_prompt(self, paper) -> str:
        """Create a prompt for DeepSeek to summarize a paper"""
        return f"""<|im_start|>system
You are an AI assistant specializing in lung cancer research. Provide a concise summary of research papers.<|im_end|>

<|im_start|>user
Please provide a concise summary of the following research paper. Focus on key findings, methodology, and clinical implications.

Title: {paper.title}
Authors: {', '.join(paper.authors)}
Journal: {paper.journal}
Publication Date: {paper.publication_date}

Abstract:
{paper.abstract}

Your summary should:
1. Highlight the main findings
2. Explain the methodology briefly
3. Discuss clinical implications
4. Be concise (150-200 words)
5. Use plain language accessible to medical professionals<|im_end|>

<|im_start|>assistant
"""
    
    async def _generate_summary(self, prompt: str) -> str:
        """Generate summary using locally hosted DeepSeek model"""
        try:
            # Convert to async/await pattern for consistency with the rest of the application
            # For CPU-bound tasks, we'd typically use run_in_executor
            import asyncio
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, self._generate_text, prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error: Unable to generate summary with local DeepSeek model."
    
    def _generate_text(self, prompt: str) -> str:
        """Core text generation function that runs on the executor"""
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
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|im_start|>assistant" in generated_text:
                response_text = generated_text.split("<|im_start|>assistant")[-1]
                # Remove any trailing system messages if present
                if "<|im_start|>" in response_text:
                    response_text = response_text.split("<|im_start|>")[0]
                return response_text.strip()
            else:
                # If we can't find the assistant marker, return everything after the prompt
                return generated_text[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            raise