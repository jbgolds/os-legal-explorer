import json
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, cast

import openai
from openai import OpenAI
import pandas as pd
import requests
from tqdm import tqdm

# Try to import ollama, but don't fail if it's not available
try:
    import ollama
    from ollama import Client as OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.llm_extraction.models import CitationAnalysis
from src.llm_extraction.prompts import system_prompt
from src.llm_extraction.rate_limited_gemini import RateLimiter, TextChunker


class LLMClientType:
    """Enumeration of supported LLM client types."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class OpenAICompatibleClient:
    """A rate-limited client that works with OpenAI API and compatible services like Ollama.

    This client supports:
    1. OpenAI's native API
    2. Ollama's API through OpenAI compatibility mode
    3. Any other OpenAI-compatible API
    
    Features:
    - Text chunking for long inputs
    - Rate limiting with exponential backoff
    - Retry logic for failed API calls
    - Concurrent processing via thread pool
    - Progress tracking
    """
    DEFAULT_MODEL = "gpt-3.5-turbo"  # Default for OpenAI
    WORD_COUNT_THRESHOLD = 8000  # threshold to decide when to chunk text
    
    # Keep worker counts for logging
    _worker_counter = 0
    
    def __init__(
        self, 
        api_key: str,
        client_type: str = LLMClientType.OPENAI,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        rate_limit: int = 60, 
        max_concurrent: int = 10,
        use_global_rate_limiter: bool = True
    ):
        """
        Initialize the OpenAI-compatible client.
        
        Args:
            api_key: API key for authentication
            client_type: Type of client ("openai", "ollama", "openai_compatible")
            base_url: Base URL for API (required for Ollama, e.g., "http://localhost:11434")
            model: Model to use (defaults vary by client_type)
            custom_system_prompt: Override the default system prompt
            rate_limit: Requests per minute limit
            max_concurrent: Maximum concurrent requests
            use_global_rate_limiter: Whether to use global rate limiter
        """
        self.api_key = api_key
        self.client_type = client_type
        self.base_url = base_url
        
        # Set appropriate defaults based on client type
        if client_type == LLMClientType.OLLAMA:
            self.model = model or "llama3"
            if not base_url:
                self.base_url = "http://localhost:11434"
            
            # Check if Ollama is available
            if not OLLAMA_AVAILABLE and client_type == LLMClientType.OLLAMA:
                raise ImportError("Ollama package is required but not installed. Please install it with 'pip install ollama'.")
        else:
            self.model = model or OpenAICompatibleClient.DEFAULT_MODEL
            
        # Use the appropriate system prompt
        self.system_prompt = custom_system_prompt or system_prompt

        # Set up rate limiter 
        self.rate_limiter = RateLimiter(
            rpm_limit=rate_limit, max_concurrent=max_concurrent
        )
        
        # Set up chunker and assign worker ID
        self.chunker = TextChunker()
        self._worker_id = OpenAICompatibleClient._worker_counter
        OpenAICompatibleClient._worker_counter += 1
        
        # Configure client based on type
        if client_type in [LLMClientType.OPENAI, LLMClientType.OPENAI_COMPATIBLE]:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif client_type == LLMClientType.OLLAMA and OLLAMA_AVAILABLE:
            self.client = OllamaClient(host=base_url)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized {client_type} client (worker {self._worker_id}) with model: {self.model}, "
            f"rate limit: {rate_limit} RPM, max concurrent: {max_concurrent}"
        )

    def get_worker_id(self) -> int:
        """Get the worker ID for this client instance."""
        return self._worker_id
    
    def _process_single_chunk(
        self, content: str, max_retries: int = 3
    ) -> Optional[CitationAnalysis]:
        """Process a single text chunk by calling the API with retries and rate limiting."""
        worker_id = self.get_worker_id()
        
        for attempt in range(max_retries):
            # Acquire rate limiter permission
            self.rate_limiter.acquire()
            
            try:
                response_text = None
                
                if self.client_type in [LLMClientType.OPENAI, LLMClientType.OPENAI_COMPATIBLE]:
                    # OpenAI API
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content.strip()}
                    ]
                    
                    # Use beta.completions.parse to directly parse into CitationAnalysis
                    result = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        temperature=0.0, # TODO: Make configurable.
                        response_format=CitationAnalysis
                    )
                    self.rate_limiter.release()
                    self.rate_limiter.report_success()
                    return result
                   
                
                    # Ollama API - use the generate method, not chat directly
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content.strip()}
                ]
                
                
                response = self.client.generate(
                    model=self.model,
                    prompt=content.strip(),
                    system=self.system_prompt,
                    format=CitationAnalysis.model_json_schema()
                )
                response_text = response.get('response', '')

                # Release the rate limiter
                self.rate_limiter.release()
                
                # Attempt to parse the output into a CitationAnalysis model
                if not response_text:
                    self.logger.warning(f"Worker {worker_id}: Empty response from API")
                    continue
                
                try:
                    # First try direct JSON parsing
                    result = CitationAnalysis.model_validate_json(response_text)
                    if result:
                        self.rate_limiter.report_success()
                        return result
                except Exception as parse_error:
                    self.logger.warning(
                        f"Worker {worker_id}: Failed direct JSON parsing: {str(parse_error)}"
                    )
                    
                    # Try to extract JSON from text with markdown code blocks
                    try:
                        if "```json" in response_text:
                            self.logger.info(f"Worker {worker_id}: Attempting to extract JSON from markdown code block")
                            json_content = response_text.split("```json")[1].split("```")[0].strip()
                            result = CitationAnalysis.model_validate_json(json_content)
                            if result:
                                self.rate_limiter.report_success()
                                return result
                    except Exception as markdown_error:
                        self.logger.warning(
                            f"Worker {worker_id}: Failed to extract JSON from markdown: {str(markdown_error)}"
                        )
                    
                    # Try to extract JSON from text with just code blocks
                    try:
                        if "```" in response_text:
                            self.logger.info(f"Worker {worker_id}: Attempting to extract JSON from generic code block")
                            parts = response_text.split("```")
                            if len(parts) >= 3:  # At least one code block
                                for part in parts[1::2]:  # Get every other part starting from index 1
                                    clean_part = part.strip()
                                    if clean_part.startswith("json"):
                                        clean_part = clean_part[4:].strip()
                                    try:
                                        result = CitationAnalysis.model_validate_json(clean_part)
                                        if result:
                                            self.rate_limiter.report_success()
                                            return result
                                    except:
                                        continue
                    except Exception as block_error:
                        self.logger.warning(
                            f"Worker {worker_id}: Failed to extract JSON from code blocks: {str(block_error)}"
                        )
                    
                    self.logger.error(
                        f"Worker {worker_id}: All parsing attempts failed. Response text: {response_text[:200]}..."
                    )
                        
            except Exception as e:
                # Always release the rate limiter on exception
                self.rate_limiter.release()
                
                error_msg = str(e)
                self.logger.error(f"Worker {worker_id}: API call failed: {error_msg}")
                
                # Check if this is a rate limit error
                is_rate_limit_error = any(
                    term in error_msg.lower() 
                    for term in ["rate limit", "ratelimit", "too many requests", "429"]
                )
                
                if is_rate_limit_error:
                    # Report rate limit error
                    self.rate_limiter.report_error(is_rate_limit_error=True)
                    backoff_time = 5.0 * (2 ** attempt) * (0.5 + random.random())
                    self.logger.warning(
                        f"Worker {worker_id}: Rate limit error detected. "
                        f"Backing off for {backoff_time:.2f} seconds"
                    )
                else:
                    # Report other error
                    self.rate_limiter.report_error(is_rate_limit_error=False)
                    backoff_time = 1.0 * (2 ** attempt) * (0.5 + random.random())
                
                time.sleep(backoff_time)
            
            # If we're here, the attempt failed but we can retry
            if attempt < max_retries - 1:
                self.logger.info(
                    f"Worker {worker_id}: Retrying chunk after failure "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            else:
                self.logger.error(
                    f"Worker {worker_id}: All {max_retries} attempts failed"
                )
        
        # If we get here, all retries failed
        return None

    def generate_content_with_chat(
        self, text: str, cluster_id: int, max_retries: int = 3
    ) -> Optional[CitationAnalysis]:
        """
        Generate content using chat-based interaction. 
        
        For long text, this method splits it into chunks and combines the responses.
        
        Args:
            text: Input text to process
            cluster_id: Identifier for the document cluster 
            max_retries: Maximum number of retry attempts for failed API calls
            
        Returns:
            CitationAnalysis object or None if processing failed
        """
        worker_id = self.get_worker_id()
        
        if not text or not text.strip():
            raise ValueError("Empty or invalid text input")
            
        word_count = self.chunker.count_words(text)
        
        # For short text, process in a single chunk
        if word_count <= OpenAICompatibleClient.WORD_COUNT_THRESHOLD:
            self.logger.info(
                f"Worker {worker_id}: Processing cluster {cluster_id} "
                f"with {word_count} words as a single chunk"
            )
            return self._process_single_chunk(text, max_retries)
            
        # For long text, split into chunks and process each one
        chunks = self.chunker.simple_split_into_chunks(text)
        self.logger.info(
            f"Worker {worker_id}: Text for cluster {cluster_id} is long "
            f"({word_count} words). Splitting into {len(chunks)} chunks."
        )
        
        responses: List[CitationAnalysis] = []
        chunk_failures = 0
        
        # Process each chunk
        for idx, chunk in enumerate(chunks, 1):
            chunk_words = self.chunker.count_words(chunk)
            self.logger.info(
                f"Worker {worker_id}: Processing chunk {idx}/{len(chunks)} "
                f"for cluster {cluster_id} ({chunk_words} words)"
            )
            
            result = self._process_single_chunk(chunk, max_retries)
            if result:
                responses.append(result)
                self.logger.info(
                    f"Worker {worker_id}: Successfully processed chunk {idx}/{len(chunks)}"
                )
            else:
                chunk_failures += 1
                self.logger.warning(
                    f"Worker {worker_id}: Failed to process chunk {idx}/{len(chunks)} "
                    f"after {max_retries} attempts"
                )
        
        # Combine all successful responses
        if responses:
            try:
                # Log chunk success rate
                self.logger.info(
                    f"Worker {worker_id}: Successfully processed {len(responses)}/{len(chunks)} "
                    f"chunks ({len(responses)/len(chunks)*100:.1f}% success rate)"
                )
                
                # Use the CitationAnalysis class method to combine responses
                return CitationAnalysis.combine_analyses(responses)
            except Exception as e:
                self.logger.error(
                    f"Worker {worker_id}: Failed to combine chunk responses for "
                    f"cluster {cluster_id}: {str(e)}"
                )
                return None
        else:
            self.logger.warning(
                f"Worker {worker_id}: No valid responses collected for cluster {cluster_id} "
                f"(all {len(chunks)} chunks failed)"
            )
            return None

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        max_workers: Optional[int] = None,
        output_file: Optional[str] = None,
        batch_size: int = 10,
    ) -> Dict[int, Optional[CitationAnalysis]]:
        """
        Process a DataFrame of texts using a thread pool with progress tracking.
        
        Args:
            df: DataFrame containing texts to process
            text_column: Name of the column containing the text
            max_workers: Maximum number of worker threads
            output_file: Path to save intermediate results
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary mapping cluster IDs to CitationAnalysis objects
        """
        # Determine appropriate number of workers
        if max_workers is None:
            suggested_workers = min(
                max(1, int(self.rate_limiter.rpm_limit * 0.7)),
                getattr(self.rate_limiter, 'max_concurrent', 10),
                len(df),
            )
            max_workers = suggested_workers
            self.logger.info(f"Auto-adjusted max_workers to {max_workers}")
        
        # Adjust batch_size to be no larger than the DataFrame
        batch_size = min(max(batch_size, max_workers), len(df))
        
        results: Dict[int, Optional[CitationAnalysis]] = {}
        errors = []
        
        def process_row(row) -> Tuple[int, Optional[CitationAnalysis], Optional[str]]:
            """Process a single row from the DataFrame."""
            cluster_id = int(row["cluster_id"])
            try:
                self.logger.info(f"Starting processing cluster_id {cluster_id}")
                result = self.generate_content_with_chat(row[text_column], cluster_id)
                self.logger.info(f"Finished processing cluster_id {cluster_id}")
                return cluster_id, result, None
            except Exception as e:
                self.logger.error(f"Error processing cluster_id {cluster_id}: {str(e)}")
                return cluster_id, None, str(e)
        
        self.logger.info(
            f"Processing {len(df)} rows with {max_workers} workers in batches of {batch_size}"
        )
        
        # Calculate total number of batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        self.logger.info(f"Total number of batches: {total_batches}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]
                current_batch = batch_start // batch_size + 1
                
                self.logger.info(
                    f"Processing batch {current_batch}/{total_batches}, "
                    f"rows {batch_start} to {batch_end-1}"
                )
                
                # Submit all rows in the batch
                futures = [
                    executor.submit(process_row, row) 
                    for _, row in batch_df.iterrows()
                ]
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc=f"Batch {current_batch}/{total_batches}"):
                    try:
                        cluster_id, result, error = future.result()
                        results[cluster_id] = result
                        
                        if error:
                            errors.append({"cluster_id": cluster_id, "error": error})
                        
                        # Write intermediate results if output file is specified
                        if output_file:
                            # Use atomic write to prevent corruption
                            tmp_file = f"{output_file}.tmp"
                            with open(tmp_file, "w") as f:
                                # Only convert to dict when writing to file
                                serialized_results = {
                                    str(cid): (item.model_dump() if item else None)
                                    for cid, item in results.items()
                                }
                                json.dump(serialized_results, f, indent=2)
                            os.replace(tmp_file, output_file)
                    
                    except Exception as e:
                        self.logger.error(f"Error processing future: {str(e)}")
        
        # Save debug info
        if errors:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = f"openai_errors_{timestamp}.json"
            debug_path = os.path.join("/tmp", debug_file)
            
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            
            self.logger.warning(
                f"Encountered {len(errors)} errors during processing. "
                f"Details saved to {debug_path}"
            )
        
        return results


# Add backward compatibility alias
OpenAIClient = OpenAICompatibleClient 