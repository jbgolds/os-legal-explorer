import time
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from google.genai.chats import Chat
from json_repair import repair_json
import re
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore
from tqdm import tqdm
from datetime import datetime
from threading import local

from src.llm_extraction.models import (
    Citation,
    CitationAnalysis,
    CombinedResolvedCitationAnalysis,
)
from src.llm_extraction.prompts import system_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextChunker:
    """Handles text chunking and response combining."""

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text, ignoring empty lines and extra whitespace."""
        return len([word for word in text.strip().split() if word])

    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs, handling various newline formats."""
        # Handle different types of paragraph separators
        text = text.replace("\r\n", "\n")  # Normalize line endings

        # Split on double newlines (common paragraph separator)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # If no paragraphs found with double newlines, try single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        if not paragraphs:
            logging.warning(
                f"No paragraphs found in text, returning text as single chunk: {text[0:50]}..."
            )
            return [text]

        return paragraphs

    @staticmethod
    def simple_split_into_chunks(text: str, max_words: int = 10000) -> List[str]:
        """Split text into chunks using paragraph boundaries so that each chunk is under max_words."""
        paragraphs = TextChunker.split_paragraphs(text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for paragraph in paragraphs:
            paragraph_word_count = TextChunker.count_words(paragraph)

            # Handle paragraphs that are themselves too long
            if paragraph_word_count > max_words:
                # First flush any current chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0

                # Split the long paragraph on word boundaries
                words = paragraph.split()
                temp_words = []
                temp_count = 0

                for word in words:
                    if temp_count + 1 > max_words:
                        chunks.append(" ".join(temp_words))
                        temp_words = [word]
                        temp_count = 1
                    else:
                        temp_words.append(word)
                        temp_count += 1

                if temp_words:
                    current_chunk = [" ".join(temp_words)]
                    current_word_count = temp_count
                continue

            # Normal case: If adding the paragraph would exceed max_words, start new chunk
            if current_word_count + paragraph_word_count > max_words:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_word_count = paragraph_word_count
            else:
                current_chunk.append(paragraph)
                current_word_count += paragraph_word_count

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


class TokenBucket:
    """Token bucket algorithm for rate limiting."""

    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # tokens per minute
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed_minutes = (now - self.last_update) / 60.0  # Convert to minutes
        new_tokens = elapsed_minutes * self.rate  # Rate is already in tokens per minute
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking."""
        with self.lock:
            self._add_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        while True:
            with self.lock:
                self._add_tokens()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.1)  # Sleep outside the lock


class RateLimiter:
    """Rate limiter using token bucket algorithm with concurrent request limiting."""

    def __init__(self, rpm_limit: int = 15, max_concurrent: int = 10):
        self.token_bucket = TokenBucket(rate=rpm_limit, capacity=rpm_limit)
        self.concurrent_semaphore = Semaphore(max_concurrent)

    def acquire(self) -> None:
        """Acquire both rate limit and concurrency permits."""
        self.token_bucket.acquire()
        self.concurrent_semaphore.acquire()

    def release(self) -> None:
        """Release concurrency permit."""
        self.concurrent_semaphore.release()


def repair_json_string(json_str: str) -> Optional[str]:
    """
    Simple JSON repair function that tries to fix common issues.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string if successful, otherwise None. Note: Although json.loads returns a dictionary, we use it here only for validation and still return the JSON string.
    """
    try:
        # First try standard repair
        repaired = repair_json(json_str)
        # Test if it's valid JSON
        json.loads(repaired)
        return repaired
    except Exception as e:
        logging.error(f"JSON repair failed: {str(e)}")
        return None


class ResponseSerializer:
    """Handles serialization of Gemini API responses with direct CitationAnalysis parsing."""

    @staticmethod
    def serialize(
        response: GenerateContentResponse | None,
    ) -> Optional[CitationAnalysis]:
        """
        Serialize a response directly to CitationAnalysis with enhanced error handling.

        Args:
            response: Raw Gemini API response

        Returns:
            CitationAnalysis or None if parsing fails
        """
        if response is None:
            logging.error("Response is None")
            return None

        # Save raw response and initialize tracking variables
        raw_response, validation_errors = ResponseSerializer._initialize_debug_tracking(
            response
        )

        # Try each parsing strategy in order until one succeeds

        # Strategy 1: Direct parsing from response.parsed
        result = ResponseSerializer._try_parse_from_parsed(response, validation_errors)
        if result:
            ResponseSerializer._save_debug_info(raw_response, validation_errors)
            return result

        # Strategy 2: Parse from response.text as JSON
        result = ResponseSerializer._try_parse_from_text(response, validation_errors)
        if result:
            ResponseSerializer._save_debug_info(raw_response, validation_errors)
            return result

        # Strategy 3: Attempt to create fallback from collected data
        result = ResponseSerializer._try_create_fallback(validation_errors)

        # Save debug info regardless of outcome
        ResponseSerializer._save_debug_info(raw_response, validation_errors)

        if not result:
            logging.warning("All parsing attempts failed, returning None")

        return result

    @staticmethod
    def _initialize_debug_tracking(
        response: GenerateContentResponse,
    ) -> Tuple[Dict, List]:
        """Initialize raw response and validation error tracking."""
        raw_response = {
            "text": response.text if hasattr(response, "text") else None,
            "parsed": response.parsed if hasattr(response, "parsed") else None,
        }
        validation_errors = []
        return raw_response, validation_errors

    @staticmethod
    def _try_parse_from_parsed(
        response: GenerateContentResponse, validation_errors: List
    ) -> Optional[CitationAnalysis]:
        """Attempt to parse from response.parsed field."""
        if not response.parsed:
            return None

        # If already correct type, return immediately
        if isinstance(response.parsed, CitationAnalysis):
            return response.parsed

        try:
            if isinstance(response.parsed, str):
                return CitationAnalysis.model_validate_json(response.parsed)
            elif isinstance(response.parsed, dict):
                return CitationAnalysis.model_validate(response.parsed)
        except Exception as e:
            validation_errors.append(
                {
                    "stage": "direct_parsing",
                    "error": str(e),
                    "input": response.parsed,
                }
            )
            logging.warning(f"Failed direct parsing of response.parsed: {str(e)}")

        return None

    @staticmethod
    def _try_parse_from_text(
        response: GenerateContentResponse, validation_errors: List
    ) -> Optional[CitationAnalysis]:
        """Attempt to parse from response.text field."""
        if not hasattr(response, "text") or not response.text:
            return None

        # First, try to parse the text as JSON directly
        try:
            json_data = json.loads(response.text)
            result = ResponseSerializer._validate_json_data(
                json_data, validation_errors
            )
            if result:
                return result
        except json.JSONDecodeError:
            # If direct parsing fails, try repair
            repaired_json = repair_json_string(response.text)
            if repaired_json:
                try:
                    json_data = json.loads(repaired_json)
                    result = ResponseSerializer._validate_json_data(
                        json_data, validation_errors, repaired=True
                    )
                    if result:
                        return result
                except json.JSONDecodeError:
                    validation_errors.append(
                        {
                            "stage": "json_repair",
                            "error": "Failed to parse repaired JSON",
                            "input": repaired_json,
                        }
                    )
                    logging.warning("Failed to parse repaired JSON")
        except Exception as e:
            validation_errors.append(
                {"stage": "text_parsing", "error": str(e), "input": response.text}
            )
            logging.warning(f"Failed parsing response.text: {str(e)}")

        return None

    @staticmethod
    def _validate_json_data(
        json_data: Union[Dict, List], validation_errors: List, repaired: bool = False
    ) -> Optional[CitationAnalysis]:
        """Validate and convert JSON data (dict or list) to CitationAnalysis."""
        prefix = "repaired_" if repaired else ""

        # Handle list responses (common from LLM)
        if (
            isinstance(json_data, list)
            and len(json_data) > 0
            and isinstance(json_data[0], dict)
        ):
            try:
                return CitationAnalysis.model_validate(json_data[0])
            except Exception as e:
                validation_errors.append(
                    {
                        "stage": f"{prefix}list_validation",
                        "error": str(e),
                        "input": json_data[0],
                    }
                )
                logging.warning(
                    f"Failed to validate first item in {prefix}list: {str(e)}"
                )

        # Handle dict responses
        elif isinstance(json_data, dict):
            try:
                return CitationAnalysis.model_validate(json_data)
            except Exception as e:
                validation_errors.append(
                    {
                        "stage": f"{prefix}dict_validation",
                        "error": str(e),
                        "input": json_data,
                    }
                )
                logging.warning(f"Failed to validate {prefix}dict: {str(e)}")

        # Store for potential fallback
        ResponseSerializer._thread_local.input_data = json_data
        return None

    @staticmethod
    def _try_create_fallback(validation_errors: List) -> Optional[CitationAnalysis]:
        """Attempt to create a minimal valid CitationAnalysis as fallback."""
        if not hasattr(ResponseSerializer, "_thread_local") or not hasattr(
            ResponseSerializer._thread_local, "input_data"
        ):
            return None

        input_data = ResponseSerializer._thread_local.input_data
        if not input_data:
            return None

        try:
            # Try to construct a minimal valid CitationAnalysis from whatever we have
            if isinstance(input_data, dict):
                return ResponseSerializer._create_minimal_citation_analysis(input_data)
            elif (
                isinstance(input_data, list)
                and len(input_data) > 0
                and isinstance(input_data[0], dict)
            ):
                return ResponseSerializer._create_minimal_citation_analysis(
                    input_data[0]
                )
        except Exception as e:
            validation_errors.append(
                {"stage": "fallback_creation", "error": str(e), "input": input_data}
            )
            logging.warning(f"Failed to create fallback CitationAnalysis: {str(e)}")

        return None

    @staticmethod
    def _create_minimal_citation_analysis(data: Dict) -> CitationAnalysis:
        """Create a minimal valid CitationAnalysis from a dictionary."""
        date = data.get("date", datetime.now().date().isoformat())
        brief_summary = data.get("brief_summary", "No summary available")

        return CitationAnalysis(
            date=date,
            brief_summary=brief_summary,
            majority_opinion_citations=[],
            concurring_opinion_citations=[],
            dissenting_citations=[],
        )

    @staticmethod
    def _save_debug_info(raw_response: Dict, validation_errors: List) -> None:
        """Save debug information to thread-local storage."""
        if not hasattr(ResponseSerializer, "_thread_local"):
            ResponseSerializer._thread_local = local()

        ResponseSerializer._thread_local.last_raw_response = raw_response
        ResponseSerializer._thread_local.last_validation_errors = validation_errors

    @staticmethod
    def get_last_debug_info():
        """Get the debug information from the last serialization attempt."""
        if not hasattr(ResponseSerializer, "_thread_local"):
            return None, None

        return (
            getattr(ResponseSerializer._thread_local, "last_raw_response", None),
            getattr(ResponseSerializer._thread_local, "last_validation_errors", None),
        )


class GeminiClient:
    DEFAULT_MODEL = "gemini-2.0-flash-001"

    def __init__(
        self,
        api_key: str,
        rpm_limit: int = 15,
        max_concurrent: int = 10,
        config: Optional[GenerateContentConfig] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.client = genai.Client(api_key=api_key)
        # Ensure we have a valid config

        self.config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CitationAnalysis,
            system_instruction=system_prompt,
        )

        self.rate_limiter = RateLimiter(
            rpm_limit=rpm_limit, max_concurrent=max_concurrent
        )
        self.chunker = TextChunker()
        self.serializer = ResponseSerializer()
        self.model = model

        # Initialize thread-local storage for worker IDs
        self.worker_data = local()
        self.worker_counter = 0
        self.worker_counter_lock = Lock()

        logging.info(
            f"Initialized client with RPM limit: {rpm_limit}, AFC concurrent limit: {max_concurrent}, model: {model}"
        )

    def get_worker_id(self) -> int:
        """Get or create worker ID for current thread."""
        # Make the entire check-and-assign operation atomic
        with self.worker_counter_lock:
            if not hasattr(self.worker_data, "worker_id"):
                self.worker_counter += 1
                self.worker_data.worker_id = self.worker_counter
        return self.worker_data.worker_id

    def combine_chunk_responses(
        self, responses: List[CitationAnalysis], cluster_id: int
    ) -> Optional[CombinedResolvedCitationAnalysis]:
        """
        Combine multiple chunk responses into a single citation analysis result.

        Args:
            responses: List of CitationAnalysis objects from processing chunks
            cluster_id: The cluster ID to associate with the combined response

        Returns:
            CombinedResolvedCitationAnalysis or None if no valid responses

        Raises:
            ValueError: If no valid CitationAnalysis objects are provided
        """
        # Filter out None responses and ensure we have valid ones
        valid_responses = [r for r in responses if r is not None]
        if not valid_responses:
            logging.warning("No valid responses to combine")
            return None

        try:
            # Create combined analysis using from_citations
            combined = CombinedResolvedCitationAnalysis.from_citations(
                valid_responses, cluster_id
            )
            return combined
        except Exception as e:
            logging.error(f"Error combining responses: {str(e)}")
            return None

    def generate_content_with_chat(
        self,
        text: str,
        cluster_id: int,
        model: str = "gemini-2.0-flash-001",
        max_retries: int = 3,
    ) -> Optional[CombinedResolvedCitationAnalysis]:
        """Generate content using chat history for better context preservation."""
        model = model or self.model
        worker_id = self.get_worker_id()

        if not text or not text.strip():
            raise ValueError("Empty or invalid text input")

        word_count = self.chunker.count_words(text)

        # Ensure we have a valid config before proceeding
        if not self.config:
            raise ValueError("Configuration is required but not provided")

        # Create a copy of the config with chunking instructions if needed
        config_chunking = GenerateContentConfig(
            response_mime_type=self.config.response_mime_type or "application/json",
            response_schema=self.config.response_schema,
            system_instruction=(
                (self.config.system_instruction or system_prompt)
                + (
                    "\n\n The document will be sent in multiple parts. "
                    "For each part, analyze the citations and legal arguments while maintaining context "
                    "from previous parts. Please provide your analysis in the same structured format "
                    "filling in the lists of citation analysis for each response."
                    if word_count > 10000
                    else ""
                )
            ),
        )

        def process_single_chunk(
            content: str, chunk_config: GenerateContentConfig
        ) -> Optional[CitationAnalysis]:
            """Process a single chunk with retries when validation fails"""
            for attempt in range(max_retries):
                # Acquire rate limiter for each attempt
                self.rate_limiter.acquire()

                try:
                    # Make the API call
                    response: GenerateContentResponse = (
                        self.client.models.generate_content(
                            model=model,
                            config=chunk_config,
                            contents=content.strip(),
                        )
                    )

                    # Try to validate/serialize the response
                    result = self.serializer.serialize(response)
                    if result:
                        # Success! Release the rate limiter and return the result
                        self.rate_limiter.release()
                        return result

                    # Validation failed
                    self.rate_limiter.release()

                    # Only retry if we haven't exceeded max attempts
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Worker {worker_id}: Validation failed, retrying (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(1)  # Short delay before retry
                    else:
                        logging.warning(
                            f"Worker {worker_id}: All {max_retries} validation attempts failed"
                        )
                        return None

                except Exception as e:
                    # Always release on exception
                    self.rate_limiter.release()

                    logging.error(f"Worker {worker_id}: API call failed: {str(e)}")

                    # Only retry if we haven't exceeded max attempts
                    if attempt < max_retries - 1:
                        logging.info(
                            f"Worker {worker_id}: Retrying after error (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(1)  # Short delay before retry
                    else:
                        logging.error(
                            f"Worker {worker_id}: All {max_retries} attempts failed with errors"
                        )
                        return None

            # Should never reach here, but just in case
            return None

        # Handle small texts (single chunk processing)
        if word_count <= 10000:
            try:
                result = process_single_chunk(text, self.config)
                # Always run through combine_chunk_responses for consistent typing
                if result:
                    return self.combine_chunk_responses([result], cluster_id)
                return None
            except Exception as e:
                logging.error(
                    f"Worker {worker_id}: Processing failed for text of length {word_count}: {str(e)}"
                )
                raise

        # Handle large texts with chunking
        logging.info(
            f"Worker {worker_id}: Text length ({word_count} words) exceeds limit. Using chat-based chunking..."
        )
        chunks = self.chunker.simple_split_into_chunks(text)
        chat = None

        try:
            # Create a chat session with chunked config
            chat: Chat = self.client.chats.create(model=model, config=config_chunking)
            responses: List[CitationAnalysis] = []
            chunk_failures = 0  # Track how many chunks fail

            for i, chunk in enumerate(chunks, 1):
                chunk_words = self.chunker.count_words(chunk)
                logging.info(
                    f"Worker {worker_id}: Processing chunk {i}/{len(chunks)} ({chunk_words} words)"
                )

                # Process each chunk with retries
                success = False
                for attempt in range(max_retries):
                    # Acquire rate limiter for each attempt
                    self.rate_limiter.acquire()

                    try:
                        # Send the message
                        response: GenerateContentResponse = chat.send_message(
                            chunk.strip()
                        )

                        # Check if response has the expected text attribute
                        if not hasattr(response, "text"):
                            logging.warning(
                                f"Worker {worker_id}: Response missing 'text' attribute from chunk {i} (attempt {attempt + 1}/{max_retries})"
                            )
                            self.rate_limiter.release()

                            # Only retry if we haven't exceeded max attempts
                            if attempt < max_retries - 1:
                                time.sleep(1)  # Short delay before retry
                                continue
                            else:
                                chunk_failures += 1
                                break

                        # Try to validate/serialize the response
                        result = self.serializer.serialize(response)
                        if result:
                            # Success! Release the rate limiter and store the result
                            self.rate_limiter.release()
                            responses.append(result)
                            success = True
                            break

                        # Validation failed
                        self.rate_limiter.release()

                        # Only retry if we haven't exceeded max attempts
                        if attempt < max_retries - 1:
                            logging.warning(
                                f"Worker {worker_id}: Validation failed for chunk {i}, retrying (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(1)  # Short delay before retry
                        else:
                            logging.warning(
                                f"Worker {worker_id}: All {max_retries} validation attempts failed for chunk {i}"
                            )
                            chunk_failures += 1

                    except Exception as e:
                        # Always release on exception
                        self.rate_limiter.release()

                        logging.error(
                            f"Worker {worker_id}: Error processing chunk {i}: {str(e)}"
                        )

                        # Only retry if we haven't exceeded max attempts
                        if attempt < max_retries - 1:
                            logging.warning(
                                f"Worker {worker_id}: Retrying chunk {i} after error (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(1)  # Short delay before retry
                        else:
                            logging.error(
                                f"Worker {worker_id}: All {max_retries} attempts failed for chunk {i}"
                            )
                            chunk_failures += 1
                            break

                # Log outcome for this chunk
                if success:
                    logging.info(
                        f"Worker {worker_id}: Successfully processed chunk {i}/{len(chunks)}"
                    )
                else:
                    logging.warning(
                        f"Worker {worker_id}: Failed to process chunk {i}/{len(chunks)} after {max_retries} attempts"
                    )

            if not responses:
                logging.warning(
                    f"Worker {worker_id}: No valid responses were collected during processing (all {len(chunks)} chunks failed)"
                )
                # Return None instead of an empty list
                return None

            # Log chunk success rate
            logging.info(
                f"Worker {worker_id}: Successfully processed {len(responses)}/{len(chunks)} chunks "
                f"({len(responses)/len(chunks)*100:.1f}% success rate)"
            )

            # Merge all responses into one consolidated result
            try:
                return self.combine_chunk_responses(responses, cluster_id)
            except Exception as e:
                logging.error(
                    f"Worker {worker_id}: Failed to combine chunk responses: {str(e)}"
                )
                # Don't try to salvage partial responses, just return None so caller can retry
                return None

        except Exception as e:
            logging.error(
                f"Worker {worker_id}: Chat-based processing failed for text of length {word_count}: {str(e)}"
            )
            raise
        finally:
            if chat:
                try:
                    del chat
                except Exception as e:
                    logging.warning(
                        f"Worker {worker_id}: Failed to cleanup chat session: {str(e)}"
                    )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        model: str = "gemini-2.0-flash-001",
        max_workers: Optional[int] = None,
        output_file: Optional[str] = None,
        batch_size: int = 10,
    ) -> Dict[int, Optional[CombinedResolvedCitationAnalysis]]:
        """Process a DataFrame of content generation requests using thread pool."""
        # Adjust max_workers based on DataFrame size and rate limit
        if max_workers is None:
            max_workers = min(
                10,  # Default max
                self.rate_limiter.token_bucket.rate,  # Rate limit
                len(df),  # Don't use more workers than rows
            )

        # Adjust batch_size to be no larger than the DataFrame
        batch_size = min(batch_size, len(df))

        results: Dict[int, Optional[CombinedResolvedCitationAnalysis]] = {}
        errors = []
        total_processed = 0

        # Add debug collection
        debug_info = {
            "raw_responses": {},
            "validation_errors": {},
        }

        def process_row(
            row,
        ) -> tuple[
            int, Optional[CombinedResolvedCitationAnalysis], Optional[str], dict
        ]:
            worker_id = self.get_worker_id()
            try:
                logging.info(
                    f"Worker {worker_id}: Starting processing cluster_id {row['cluster_id']}"
                )
                result = self.generate_content_with_chat(
                    row[text_column], row["cluster_id"], model
                )

                # Collect debug info
                raw_response, validation_errors = (
                    ResponseSerializer.get_last_debug_info()
                )
                row_debug = {
                    "raw_response": raw_response,
                    "validation_errors": validation_errors,
                }

                logging.info(
                    f"Worker {worker_id}: Finished processing cluster_id {row['cluster_id']}"
                )
                return int(row["cluster_id"]), result, None, row_debug
            except Exception as e:
                logging.error(
                    f"Worker {worker_id}: Error processing cluster_id {row['cluster_id']}: {str(e)}"
                )
                return int(row["cluster_id"]), None, str(e), {}

        logging.info(
            f"Processing {len(df)} rows with {max_workers} workers in batches of {batch_size}"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process in batches
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]

                logging.info(
                    f"Processing batch {batch_start//batch_size + 1}, rows {batch_start} to {batch_end}"
                )

                # Create futures for this batch only
                futures = [
                    executor.submit(process_row, row) for _, row in batch_df.iterrows()
                ]

                # Process futures for this batch
                for future in tqdm(
                    futures,
                    total=len(futures),
                    desc=f"Processing batch {batch_start//batch_size + 1}",
                    position=0,
                    leave=True,
                ):
                    try:
                        cluster_id, result, error, row_debug = future.result()
                        results[cluster_id] = result
                        total_processed += 1

                        # Store debug info
                        if row_debug:
                            debug_info["raw_responses"][cluster_id] = row_debug.get(
                                "raw_response"
                            )
                            debug_info["validation_errors"][cluster_id] = row_debug.get(
                                "validation_errors"
                            )

                        if error:
                            errors.append({"cluster_id": cluster_id, "error": error})

                        if output_file:
                            # Use atomic write to prevent corruption
                            tmp_file = f"{output_file}.tmp"
                            with open(tmp_file, "w") as f:
                                # Only convert to dict when writing to file
                                serialized_results = {
                                    str(cid): (item.model_dump() if item else None)
                                    for cid, item in results.items()
                                }
                                json.dump(serialized_results, f)
                            os.replace(tmp_file, output_file)

                    except Exception as e:
                        worker_id = self.get_worker_id()
                        logging.error(
                            f"Worker {worker_id}: Failed to process future in batch {batch_start//batch_size + 1}: {str(e)}"
                        )

                # Clear the futures list after batch is done
                futures.clear()

        # Save debug info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = f"gemini_debug_{timestamp}.json"
        debug_path = os.path.join("/tmp", debug_file)

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)

        logging.info(f"Debug information saved to {debug_path}")

        if errors:
            logging.warning(f"Encountered {len(errors)} errors during processing")

        return results


# Example usage:
# def main():
#     config = GenerateContentConfig(
#         response_mime_type="application/json",
#         response_schema=CitationAnalysis,
#         system_instruction=system_prompt,
#     )

#     client = GeminiClient(
#         api_key=os.getenv("GEMINI_API_KEY"),
#         rpm_limit=10,  # Conservative limits for testing
#         max_concurrent=10,  # Respect AFC limit
#         config=config,
#     )

#     df = pd.read_csv("data_final/supreme_court_1950_some_processing.csv")
#     df = df.sample(250)  # Take first 25 rows for testing

#     # Process DataFrame with max_workers respecting AFC limit
#     results = client.process_dataframe(
#         df,
#         text_column="text",
#         output_file=f"responses_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#         max_workers=10,  # Match AFC limit
#     )

#     print(f"Processed {len(results)} items")


# if __name__ == "__main__":
#     main()
