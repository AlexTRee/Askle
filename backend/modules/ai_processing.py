# modules/ai_processing.py
import logging
import torch
import json
import re
from typing import Dict, Any, List, Optional
# Ensure transformers library is installed: pip install transformers torch accelerate
# 'accelerate' is recommended for efficient model loading with device_map='auto'
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Added BitsAndBytesConfig for potential quantization
import os
import asyncio
# Removed functools.lru_cache import
import html
import gc # Garbage collector for explicit cleanup if needed

# Configure logging (ideally configured at application entry point)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekProcessor:
    """
    Handles interactions with a locally hosted DeepSeek model for tasks like
    summarization, keyword extraction, and input sanitization.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 max_token_limit: int = 4096, # Max tokens for the model's context window
                 generation_max_new_tokens: int = 500, # Max tokens to generate for response
                 use_quantization: bool = False # Option to enable 4-bit quantization
                ):
        """
        Initializes the DeepSeekProcessor, loading the model and tokenizer.

        Args:
            model_path: Path to the locally stored Hugging Face model. Defaults to
                        env var DEEPSEEK_MODEL_PATH or a relative path.
            device: Device to run the model on ('cuda', 'cpu', None for auto-detect).
            max_token_limit: The maximum context window size of the model.
            generation_max_new_tokens: Max tokens to generate in responses.
            use_quantization: If True, load the model with 4-bit quantization (requires bitsandbytes).
        """
        # Default to environment variable or standard path if not provided
        if model_path is None:
            # Adjust the default path based on where this script is relative to the data folder
            default_model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'hub', 'models--deepseek-ai--deepseek-r1-distill-qwen-1.5b')
            model_path = os.environ.get("DEEPSEEK_MODEL_PATH", default_model_dir)

        if not os.path.isdir(model_path):
             logger.error(f"Model directory not found: {model_path}")
             raise FileNotFoundError(f"DeepSeek model directory not found at {model_path}")

        self.max_token_limit = max_token_limit
        self.generation_max_new_tokens = generation_max_new_tokens
        logger.info(f"Attempting to load DeepSeek model from: {model_path}")

        try:
            # --- Tokenizer ---
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # trust_remote_code might be needed for some models
            # Ensure pad token is set if missing (common issue)
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
                 logger.warning("Tokenizer missing pad_token, setting to eos_token.")


            # --- Device Selection ---
            if device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info("CUDA available, selecting GPU.")
                # elif torch.backends.mps.is_available(): # Uncomment for MacOS Metal support
                #    self.device = "mps"
                #    logger.info("MPS available, selecting Apple Metal.")
                else:
                    self.device = "cpu"
                    logger.warning("CUDA (or MPS) not available, selecting CPU. Inference will be slow.")
            else:
                self.device = device
            logger.info(f"Selected device: {self.device}")


            # --- Model Loading ---
            model_kwargs = {}
            if self.device == "cuda":
                 model_kwargs['torch_dtype'] = torch.float16 # Use float16 on GPU for efficiency
                 model_kwargs['device_map'] = "auto" # Let accelerate handle device mapping
                 if use_quantization:
                      try:
                           # Requires bitsandbytes library: pip install bitsandbytes
                           quantization_config = BitsAndBytesConfig(
                               load_in_4bit=True,
                               bnb_4bit_compute_dtype=torch.float16, # Or bfloat16 if supported
                               bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                           )
                           model_kwargs['quantization_config'] = quantization_config
                           logger.info("Enabled 4-bit quantization using bitsandbytes.")
                      except ImportError:
                           logger.warning("BitsAndBytes library not found. Cannot use 4-bit quantization. Install with 'pip install bitsandbytes'.")
                      except Exception as q_err:
                           logger.error(f"Failed to configure quantization: {q_err}. Loading without quantization.")


            elif self.device == "cpu":
                 model_kwargs['torch_dtype'] = torch.float32 # CPU generally uses float32
                 # No device_map for CPU

            # elif self.device == "mps": # Uncomment for MacOS Metal support
            #      model_kwargs['torch_dtype'] = torch.float16 # MPS often works better with float16
            #      # No device_map for MPS, needs manual .to(device) later

            logger.info(f"Loading model with arguments: {model_kwargs}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True, # May be needed
                **model_kwargs
            )

            # Manually move model to device if not using device_map='auto'
            if self.device != "cuda" or 'device_map' not in model_kwargs:
                 if hasattr(self.model, 'to'):
                      self.model = self.model.to(self.device)
                 else:
                      logger.error("Model object does not have a '.to()' method for device placement.")


            # Set model to evaluation mode
            self.model.eval()

            logger.info(f"DeepSeek model '{model_path}' loaded successfully onto device '{self.device}'.")

        except ImportError as ie:
             logger.exception(f"ImportError loading model/tokenizer. Ensure 'transformers', 'torch', and potentially 'accelerate' or 'bitsandbytes' are installed: {ie}")
             raise RuntimeError(f"Missing dependency: {ie}")
        except Exception as e:
            logger.exception(f"Error loading DeepSeek model from {model_path}: {e}")
            raise RuntimeError(f"Failed to load DeepSeek model: {e}")


    def sanitize_input(self, user_input: str) -> str:
        """
        Sanitizes user input to remove potentially harmful characters/sequences
        before including it in prompts.

        Args:
            user_input: Raw user input string.

        Returns:
            Sanitized input string.
        """
        if not user_input:
            return ""

        # Ensure it's a string
        sanitized = str(user_input)

        # Basic stripping
        sanitized = sanitized.strip()

        # Escape HTML characters (<, >, &)
        sanitized = html.escape(sanitized)

        # Remove control characters (except common whitespace like \n, \t)
        # Keeps newline, tab, carriage return; removes others from 0x00-0x1F and 0x7F
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)

        # Remove specific model control tokens (adjust list based on the specific DeepSeek model)
        # Common examples for chat models:
        special_tokens_to_remove = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]
        for token in special_tokens_to_remove:
            sanitized = sanitized.replace(token, "")

        # Optional: Normalize whitespace (replace multiple spaces/newlines with single)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        # Limit overall length to prevent excessively long inputs in prompts
        # Adjust max_length as needed, considering typical query lengths
        max_length = 500
        if len(sanitized) > max_length:
            logger.warning(f"Sanitized input truncated from {len(sanitized)} to {max_length} characters.")
            sanitized = sanitized[:max_length] + "..." # Indicate truncation

        # logger.debug(f"Sanitized input: '{user_input}' -> '{sanitized}'") # Debug level might be too verbose
        return sanitized
    

    def _create_keyword_extraction_prompt(self, sanitized_input: str) -> str:
        """
        Creates the prompt for extracting search keywords from user input.

        Args:
            sanitized_input: The sanitized user query.

        Returns:
            Formatted prompt string using DeepSeek chat template, instructing JSON output.
        """
        # Use f-string with DeepSeek chat template structure
        prompt = f"""<|im_start|>system
        You are an AI assistant specialized in processing biomedical research queries. Your task is to extract the most relevant keywords and concepts from a user's question that can be used to search scientific databases like PubMed or Google Scholar. Respond *only* with a valid JSON object containing a single key "keywords", which holds an array of the extracted strings.<|im_end|>
        <|im_start|>user
        Analyze the following user query and extract the core biomedical entities, concepts, conditions, treatments, and research intent. Format the output as a JSON object with a "keywords" array. Focus on terms that will yield relevant scientific literature.
        
        User Query: "{sanitized_input}"<|im_end|>
        <|im_start|>assistant
        """
        # The final "<|im_start|>assistant\n" signals the model to start its response.
        return prompt
    

    async def _generate_text_async(self, prompt: str) -> str:
        """
        Asynchronously wraps the synchronous text generation method using run_in_executor.

        Args:
            prompt: The input prompt string.

        Returns:
            The generated text response from the model.
        """
        # Use the default asyncio event loop and default ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        try:
            generated_text = await loop.run_in_executor(None, self._generate_text, prompt)
            return generated_text
        except Exception as e:
             # Log the error from the executor context
             logger.error(f"Error during executor task for text generation: {e}", exc_info=True)
             # Re-raise or return an error message
             # raise # Option 1: Re-raise the exception
             return f"Error: Text generation failed. Details: {str(e)}" # Option 2: Return error message
        

    def _generate_text(self, prompt: str) -> str:
        """
        Synchronous core text generation function using the loaded model.
        This method is intended to be run in a separate thread via run_in_executor.

        Args:
            prompt: The input prompt string.

        Returns:
            The extracted assistant's response text.
        """
        logger.debug(f"Generating text for prompt: {prompt[:100]}...")
        response_text = "" # Initialize response_text
        try:
            # Tokenize the prompt, ensuring tensors are on the correct device
            # Using padding=True, truncation=True might be safer if prompts could exceed max_length,
            # but truncation should ideally happen before calling generate.
            # Let's rely on the earlier truncation logic for now.
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.device)
            input_ids = inputs["input_ids"]

            # Check input length against model max length (safer)
            # model.config.max_position_embeddings might be the attribute name
            model_max_len = getattr(self.model.config, 'max_position_embeddings', self.max_token_limit)
            if input_ids.shape[1] >= model_max_len:
                 logger.error(f"Input prompt length ({input_ids.shape[1]}) exceeds model maximum ({model_max_len}) even after potential truncation. Cannot generate.")
                 # Truncate input_ids directly as a last resort? Risky.
                 # input_ids = input_ids[:, :model_max_len - self.generation_max_new_tokens] # Ensure space for generation
                 # Or just return error
                 raise ValueError(f"Input prompt too long ({input_ids.shape[1]} tokens) for model max length ({model_max_len}).")


            # Generate text using the model
            # Use torch.inference_mode() for potentially better performance than torch.no_grad() in newer PyTorch versions
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.generation_max_new_tokens, # Use configured value
                    temperature=0.3, # Lower temperature for more focused summaries/keywords
                    top_p=0.9,
                    do_sample=True, # Sample for slight variability, set False for deterministic output
                    pad_token_id=self.tokenizer.pad_token_id, # Ensure pad token is set
                    eos_token_id=self.tokenizer.eos_token_id # Ensure EOS token is recognized
                )

            # Decode the entire generated sequence
            # outputs[0] contains the full sequence (prompt + generation)
            full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            logger.debug(f"Full generated output (incl. prompt): {full_generated_text}")

            # --- Extract only the newly generated part (assistant's response) ---
            # Method 1: Decode only the generated tokens
            # generated_ids = outputs[0][input_ids.shape[-1]:] # Get only tokens after prompt
            # response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Method 2: Split based on the prompt structure (more robust if special tokens are needed in output)
            assistant_marker = "<|im_start|>assistant"
            if assistant_marker in full_generated_text:
                 # Find the last occurrence of the assistant marker
                 response_part = full_generated_text.rsplit(assistant_marker, 1)[-1]

                 # Remove known end markers from the response part
                 end_markers = ["<|im_end|>", "<|endoftext|>"]
                 for marker in end_markers:
                      if marker in response_part:
                           response_part = response_part.split(marker)[0]

                 response_text = response_part.strip()
            else:
                 # Fallback if the exact marker isn't found (shouldn't happen with controlled prompts)
                 logger.warning("Could not find '<|im_start|>assistant' marker in generated text. Attempting fallback extraction.")
                 # Return text after the original prompt length (less reliable)
                 prompt_length_in_decoded_output = len(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))
                 response_text = full_generated_text[prompt_length_in_decoded_output:].strip()
                 # Clean remaining special tokens just in case
                 special_tokens = ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
                 for token in special_tokens:
                      response_text = response_text.replace(token, "")


            logger.debug(f"Extracted response: {response_text}")
            return response_text.strip()

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during text generation: {e}. Try reducing generation_max_new_tokens or using quantization.")
            # Attempt to clear cache and retry? Or just raise.
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(f"GPU out of memory: {e}") # Re-raise as RuntimeError
        except Exception as e:
            logger.exception(f"Core text generation error: {e}") # Log full traceback
            raise # Re-raise the original exception

    
    async def extract_keywords(self, user_input: str) -> Dict[str, List[str]]:
        """
        Extracts keywords/entities from user input using the DeepSeek model.
        Designed to produce terms suitable for searching PubMed/Google Scholar.

        Args:
            user_input: The raw user question or query text.

        Returns:
            A dictionary containing a 'keywords' list. Returns basic tokenized
            keywords as a fallback on error.
        """
        logger.info(f"Extracting keywords for input: '{user_input[:100]}...'")
        sanitized_input = "" # Ensure defined in outer scope
        try:
            # 1. Sanitize the input
            sanitized_input = self.sanitize_input(user_input)
            if not sanitized_input:
                 logger.warning("Keyword extraction failed: Sanitized input is empty.")
                 return {"keywords": []}

            # 2. Create the prompt
            prompt = self._create_keyword_extraction_prompt(sanitized_input)

            # 3. Generate response from the model
            logger.debug("Generating keyword extraction response...")
            raw_response = await self._generate_text_async(prompt)
            logger.debug(f"Raw keyword extraction response: {raw_response}")


            # 4. Parse the JSON response
            try:
                # Attempt to find JSON object within the response (more robust)
                # Handles cases like ```json\n{...}\n``` or surrounding text
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL | re.MULTILINE)
                if json_match:
                    json_string = json_match.group(0)
                    # Further clean potential markdown code block markers
                    json_string = re.sub(r'^```json\s*', '', json_string, flags=re.MULTILINE)
                    json_string = re.sub(r'\s*```$', '', json_string, flags=re.MULTILINE)
                    keywords_json = json.loads(json_string)
                    logger.debug(f"Parsed JSON: {keywords_json}")
                else:
                    # If no clear JSON block found, try parsing the whole response
                    logger.warning("Could not find clear JSON block in response, attempting to parse entire response.")
                    keywords_json = json.loads(raw_response)

                # Validate structure and return
                if isinstance(keywords_json, dict) and "keywords" in keywords_json and isinstance(keywords_json["keywords"], list):
                    # Optional: Further sanitize extracted keywords? (e.g., strip whitespace)
                    keywords_json["keywords"] = [kw.strip() for kw in keywords_json["keywords"] if isinstance(kw, str) and kw.strip()]
                    logger.info(f"Successfully extracted keywords: {keywords_json['keywords']}")
                    return keywords_json
                else:
                     logger.error(f"Parsed JSON has unexpected structure: {keywords_json}")
                     raise ValueError("Parsed JSON missing 'keywords' list.")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse keywords JSON: {e}. Raw response: '{raw_response}'")
                # Fallback: extract quoted strings as potential keywords
                keywords = re.findall(r'"([^"]+)"', raw_response)
                if keywords:
                     logger.warning(f"Falling back to regex extraction, found: {keywords}")
                     return {"keywords": [kw.strip() for kw in keywords if kw.strip()]}
                else:
                     # Final fallback: simple tokenization of original sanitized input
                     logger.warning("Falling back to simple tokenization.")
                     words = sanitized_input.split()
                     basic_keywords = [word.strip() for word in words if len(word.strip()) > 2] # Min length 3
                     # If tokenization yields nothing, return the whole sanitized input as one keyword
                     return {"keywords": basic_keywords or ([sanitized_input] if sanitized_input else [])}

        except Exception as e:
            logger.exception(f"Unexpected error during keyword extraction for input '{user_input[:100]}...': {e}")
            # Final fallback in case of unexpected errors
            words = sanitized_input.split() if sanitized_input else user_input.split()
            basic_keywords = [word.strip() for word in words if len(word.strip()) > 2]
            return {"keywords": basic_keywords or ([sanitized_input] if sanitized_input else [])}
        
    
    def _create_summary_prompt(self, paper_dict: Dict[str, Any]) -> str:
        """
        Creates the prompt for summarizing a paper, using data from a dictionary.

        Args:
            paper_dict: Dictionary containing paper information.

        Returns:
            Formatted prompt string using DeepSeek chat template.
        """
        # Safely get attributes from the dictionary using .get() with defaults
        title = paper_dict.get('title', 'Unknown Title')
        authors = paper_dict.get('authors', [])
        authors_str = ', '.join(authors) if isinstance(authors, list) and authors else 'Unknown Authors'
        journal = paper_dict.get('journal', 'Unknown Journal')
        pub_date = paper_dict.get('publication_date', 'Unknown Date') # Assumes date is already string/formatted
        abstract = paper_dict.get('abstract', 'No abstract available.')

        # Ensure abstract is a string
        if not isinstance(abstract, str):
             abstract = str(abstract)

        # Use f-string with DeepSeek chat template structure
        # Note: Ensure this template matches the fine-tuning/expected format of your specific DeepSeek model
        prompt = f"""<|im_start|>system
        You are an expert AI assistant specializing in summarizing biomedical research papers. Your summaries should be clear, concise, and accurate, focusing on the key aspects relevant to researchers and clinicians.<|im_end|>
        <|im_start|>user
        Please generate a concise summary (around 150-200 words) of the following research paper. Focus on:
        1.  **Background & Objective**: Briefly state the context and purpose.
        2.  **Methods**: Summarize the key methods used.
        3.  **Key Findings**: Highlight the most important results.
        4.  **Conclusions & Implications**: State the main conclusions and their potential significance or clinical relevance.
        
        Use clear and professional language. Avoid jargon where possible, or explain it briefly.
        
        **Paper Details:**
        * **Title:** {title}
        * **Authors:** {authors_str}
        * **Journal:** {journal} ({pub_date})
        * **Abstract:**
        {abstract}<|im_end|>
        <|im_start|>assistant
        """
        return prompt
    

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncates text to approximately the specified maximum number of tokens.

        Args:
            text: The text to truncate.
            max_tokens: The target maximum number of tokens.

        Returns:
            The truncated text, potentially with an indicator.
        """
        if not text or max_tokens <= 0:
            return ""

        # Encode the text to get tokens
        # Use add_special_tokens=False to avoid counting special tokens if they aren't part of the text limit logic
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

        if len(tokens) <= max_tokens:
            return text # No truncation needed

        # Truncate the token list
        truncated_tokens = tokens[:max_tokens]

        # Decode back to text
        # Use clean_up_tokenization_spaces=True for better readability
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Add an indicator that truncation occurred
        return truncated_text + "... [truncated]"
    

    async def summarize_paper(self, paper_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarizes a paper using the DeepSeek model. Accepts paper data as a dictionary.

        Args:
            paper_dict: Dictionary containing paper details, expected keys:
                        'title', 'authors', 'abstract', 'publication_date',
                        'journal', 'url', 'source'.

        Returns:
            A dictionary containing the original paper info plus the 'summary'.
            If summarization fails, the 'summary' field will contain an error message.
        """
        paper_title = paper_dict.get('title', 'Unknown Title') # Use .get for safety
        logger.info(f"Starting summarization for paper: '{paper_title[:50]}...'")

        # Create a copy to potentially modify (e.g., truncate abstract)
        result_dict = paper_dict.copy()

        try:
            # Prepare the prompt using the dictionary
            prompt = self._create_summary_prompt(paper_dict)

            # --- Token Length Check and Truncation ---
            # Calculate space needed for prompt + desired output length + buffer
            prompt_allowance = self.max_token_limit - self.generation_max_new_tokens - 50 # 50 token buffer

            # Tokenize the prompt to check its length
            # Use truncation=False initially to get the true length
            prompt_tokens = self.tokenizer.encode(prompt, truncation=False)
            prompt_token_count = len(prompt_tokens)

            if prompt_token_count > prompt_allowance:
                logger.warning(f"Prompt for '{paper_title[:50]}...' is too long ({prompt_token_count} tokens, allowance {prompt_allowance}). Truncating abstract.")

                # Calculate how many tokens to keep for the abstract
                # Estimate non-abstract prompt length (crude but often sufficient)
                temp_prompt_no_abstract = self._create_summary_prompt({**paper_dict, 'abstract': ''})
                non_abstract_tokens = len(self.tokenizer.encode(temp_prompt_no_abstract))
                abstract_allowance = prompt_allowance - non_abstract_tokens

                if abstract_allowance <= 0:
                     logger.error(f"Cannot summarize '{paper_title[:50]}...': Prompt without abstract already exceeds token limit.")
                     raise ValueError("Prompt structure exceeds token limit even without abstract.")

                original_abstract = paper_dict.get('abstract', '')
                truncated_abstract = self._truncate_text(original_abstract, abstract_allowance)

                # Recreate the prompt with the truncated abstract
                temp_paper_dict = paper_dict.copy()
                temp_paper_dict['abstract'] = truncated_abstract
                prompt = self._create_summary_prompt(temp_paper_dict)
                logger.info(f"Recreated prompt with truncated abstract (new length: {len(self.tokenizer.encode(prompt))} tokens).")


            # --- Generate Summary ---
            logger.debug(f"Generating summary for '{paper_title[:50]}...'")
            summary = await self._generate_text_async(prompt)
            logger.info(f"Successfully generated summary for '{paper_title[:50]}...'")

            # Add summary to the result dictionary
            result_dict['summary'] = summary.strip()

            return result_dict

        except Exception as e:
            logger.exception(f"Error summarizing paper '{paper_title[:50]}...': {e}")
            # Return original info with error message in summary field
            result_dict['summary'] = f"Error generating summary: {str(e)}"
            return result_dict


    async def batch_summarize_papers(self, paper_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Summarizes multiple papers concurrently using asyncio.gather.

        Args:
            paper_dicts: A list of dictionaries, where each dictionary represents a paper.

        Returns:
            A list of dictionaries, each containing paper info and its generated summary.
        """
        if not paper_dicts:
            return []
        logger.info(f"Starting batch summarization for {len(paper_dicts)} papers.")
        tasks = [self.summarize_paper(paper_dict) for paper_dict in paper_dicts]
        summary_results = await asyncio.gather(*tasks)
        logger.info(f"Finished batch summarization for {len(paper_dicts)} papers.")
        return summary_results
        

    def _create_final_summary_prompt(self, summaries: List[str], user_query: str) -> str:
        """
        Creates the prompt for generating a final summary from multiple paper summaries.

        Args:
            summaries: A list of paper summaries.
            user_query: The original user query.

        Returns:
            Formatted prompt string.
        """
        summaries_text = "\n\n".join([f"Summary {i+1}:\n{summary}" for i, summary in enumerate(summaries)])
        prompt = f"""<|im_start|>system
        You are an expert AI assistant tasked with synthesizing a comprehensive answer to a user's question based on multiple research paper summaries. Read the following summaries and generate a concise (around 250-350 words) answer that addresses the user's query, highlighting key findings and potential implications across the different papers.
        User Query: "{user_query}"
        Paper Summaries:
        {summaries_text}<|im_end|>
        <|im_start|>assistant
        """
        return prompt
    

    async def summarize_multiple_summaries(self, summaries: List[str], user_query: str) -> str:
        """
        Generates a final, consolidated summary from a list of individual paper summaries.

        Args:
            summaries: A list of strings, where each string is a summary of a paper.
            user_query: The original user query that led to these papers.

        Returns:
            A string containing the final, consolidated summary.
        """
        logger.info(f"Starting final summarization for query: '{user_query[:50]}...'")
        try:
            prompt = self._create_final_summary_prompt(summaries, user_query)
            final_summary = await self._generate_text_async(prompt)
            logger.info(f"Successfully generated final summary for query: '{user_query[:50]}...'")
            return final_summary.strip()
        except Exception as e:
            logger.exception(f"Error generating final summary for query '{user_query[:50]}...': {e}")
            return f"Error generating final summary: {str(e)}"


# Example Usage (Optional - for testing this module directly)
async def example_ai_usage():
    print("--- AI Processor Example Usage ---")
    processor = None # Ensure defined in outer scope
    try:
        # Initialize processor (adjust model path if needed)
        # Set use_quantization=True if you have bitsandbytes and want to test 4-bit loading
        processor = DeepSeekProcessor(use_quantization=False)

        # --- Test Keyword Extraction ---
        print("\n--- Testing Keyword Extraction ---")
        user_query = "What is the efficacy of combining immunotherapy (like PD-1 inhibitors) with chemotherapy for metastatic non-small cell lung cancer (NSCLC)?"
        keywords_result = await processor.extract_keywords(user_query)
        print(f"Query: {user_query}")
        print(f"Extracted Keywords: {keywords_result}")

        user_query_simple = "CAR-T therapy side effects"
        keywords_result_simple = await processor.extract_keywords(user_query_simple)
        print(f"\nQuery: {user_query_simple}")
        print(f"Extracted Keywords: {keywords_result_simple}")


        # --- Test Summarization ---
        print("\n--- Testing Summarization ---")
        # Example paper data (as dictionary)
        paper_example = {
            "title": "Pembrolizumab plus Chemotherapy in Metastatic Non–Small-Cell Lung Cancer",
            "authors": ["Leena Gandhi, M.D., Ph.D.", "Delvys Rodriguez-Abreu, M.D.", "et al."],
            "publication_date": "2018-05-31", # Example date format
            "journal": "N Engl J Med",
            "abstract": "BACKGROUND: Pembrolizumab monotherapy is a standard first-line treatment for patients with metastatic non–small-cell lung cancer (NSCLC) with a programmed death ligand 1 (PD-L1) tumor proportion score (TPS) of 50% or more. KEYNOTE-189 evaluated pembrolizumab plus pemetrexed and a platinum-based drug (pemetrexed–platinum) in patients with nonsquamous NSCLC without sensitizing EGFR or ALK mutations. METHODS: In this double-blind, phase 3 trial, we randomly assigned 616 patients, regardless of PD-L1 TPS, in a 2:1 ratio to receive pemetrexed and a platinum-based drug plus either 200 mg of pembrolizumab or placebo every 3 weeks for 4 cycles, followed by pembrolizumab or placebo plus pemetrexed maintenance therapy. The primary end points were overall survival and progression-free survival. RESULTS: At the first interim analysis (median follow-up, 10.5 months), overall survival was significantly longer in the pembrolizumab-combination group than in the placebo-combination group (estimated 12-month survival rate, 69.2% vs. 49.4%; hazard ratio for death, 0.49; 95% confidence interval [CI], 0.38 to 0.64; P<0.001). Progression-free survival was also significantly longer in the pembrolizumab-combination group (median, 8.8 months vs. 4.9 months; hazard ratio for disease progression or death, 0.52; 95% CI, 0.43 to 0.64; P<0.001). The benefit of pembrolizumab–chemotherapy occurred regardless of PD-L1 TPS. The incidence of grade 3 or higher adverse events was 67.2% in the pembrolizumab-combination group and 65.8% in the placebo-combination group. CONCLUSIONS: In patients with previously untreated metastatic nonsquamous NSCLC without EGFR or ALK mutations, the addition of pembrolizumab to standard chemotherapy with pemetrexed and platinum resulted in significantly longer overall survival and progression-free survival than chemotherapy alone.",
            "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa1801005",
            "source": "PubMed" # Example source
        }

        summary_result = await processor.summarize_paper(paper_example)

        print(f"\nPaper Title: {summary_result.get('title')}")
        print(f"Generated Summary:\n{summary_result.get('summary')}")

        # --- Test Batch Summarization ---
        print("\n--- Testing Batch Summarization ---")
        paper_example_2 = paper_example.copy() # Create a second dummy paper
        paper_example_2['title'] = "A Second Example Paper on Cancer Research"
        paper_example_2['abstract'] = "This second abstract discusses different aspects of cancer treatment using novel methods. The study involved multiple phases and showed promising results in preclinical models."
        paper_example_2['url'] = "http://example.com/paper2"

        batch_results = await processor.batch_summarize_papers([paper_example, paper_example_2])
        print(f"Batch summarized {len(batch_results)} papers.")
        for i, res in enumerate(batch_results):
             print(f"\n--- Summary for Paper {i+1} ---")
             print(f"Title: {res.get('title')}")
             print(f"Summary: {res.get('summary')}")


    except FileNotFoundError as fnf:
        print(f"ERROR: Model files not found. Please check the path or set DEEPSEEK_MODEL_PATH. {fnf}")
    except RuntimeError as rte:
        print(f"ERROR: A runtime error occurred, possibly during model loading or generation. {rte}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.exception("Unexpected error during example usage.")
    finally:
        # Clean up model resources if processor was initialized
        if processor and hasattr(processor, 'model'):
            del processor.model
            del processor.tokenizer
            if processor.device == 'cuda':
                 torch.cuda.empty_cache()
            gc.collect()
            print("\nCleaned up model resources.")
        print("\n--- AI Processor Example Usage Complete ---")


if __name__ == "__main__":
    # Requires asyncio to run the async example function
    import asyncio
    asyncio.run(example_ai_usage())
