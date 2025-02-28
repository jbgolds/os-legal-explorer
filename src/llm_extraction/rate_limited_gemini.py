import time
import json
from typing import Any, Dict, List, Optional, Union
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


def repair_json_string(json_str: str) -> str:
    """
    Simple JSON repair function that tries to fix common issues.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string or None if repair fails
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
        Serialize a response directly to CitationAnalysis with simple error handling.

        Args:
            response: Raw Gemini API response

        Returns:
            CitationAnalysis or None if parsing fails
        """
        if response is None:
            logging.error("Response is None")
            return None

        # Save raw response for debugging
        raw_response = {
            "text": response.text if hasattr(response, "text") else None,
            "parsed": response.parsed if hasattr(response, "parsed") else None,
        }

        validation_errors = []

        # First try: Direct parsing if response.parsed exists
        if hasattr(response, "parsed") and response.parsed:
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

        # Second try: Parse from response.text
        if hasattr(response, "text") and response.text:
            try:
                # Try to parse as JSON
                try:
                    json_data = json.loads(response.text)

                    # Handle list responses (common from LLM)
                    if isinstance(json_data, list) and len(json_data) > 0:
                        if isinstance(json_data[0], dict):
                            try:
                                return CitationAnalysis.model_validate(json_data[0])
                            except Exception as e:
                                validation_errors.append(
                                    {
                                        "stage": "list_validation",
                                        "error": str(e),
                                        "input": json_data[0],
                                    }
                                )
                                logging.warning(
                                    f"Failed to validate first item in list: {str(e)}"
                                )

                    # Handle dict responses
                    if isinstance(json_data, dict):
                        try:
                            return CitationAnalysis.model_validate(json_data)
                        except Exception as e:
                            validation_errors.append(
                                {
                                    "stage": "dict_validation",
                                    "error": str(e),
                                    "input": json_data,
                                }
                            )
                            logging.warning(f"Failed to validate dict: {str(e)}")

                except json.JSONDecodeError:
                    # If can't parse as JSON, try repair
                    repaired_json = repair_json_string(response.text)
                    if repaired_json:
                        try:
                            json_data = json.loads(repaired_json)

                            # Handle list responses
                            if isinstance(json_data, list) and len(json_data) > 0:
                                if isinstance(json_data[0], dict):
                                    try:
                                        return CitationAnalysis.model_validate(
                                            json_data[0]
                                        )
                                    except Exception as e:
                                        validation_errors.append(
                                            {
                                                "stage": "repaired_list_validation",
                                                "error": str(e),
                                                "input": json_data[0],
                                            }
                                        )
                                        logging.warning(
                                            f"Failed to validate first item in repaired list: {str(e)}"
                                        )

                            # Handle dict responses
                            if isinstance(json_data, dict):
                                try:
                                    return CitationAnalysis.model_validate(json_data)
                                except Exception as e:
                                    validation_errors.append(
                                        {
                                            "stage": "repaired_dict_validation",
                                            "error": str(e),
                                            "input": json_data,
                                        }
                                    )
                                    logging.warning(
                                        f"Failed to validate repaired dict: {str(e)}"
                                    )
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

        # Store debug info in thread-local storage
        if not hasattr(ResponseSerializer, "_thread_local"):
            ResponseSerializer._thread_local = local()

        ResponseSerializer._thread_local.last_raw_response = raw_response
        ResponseSerializer._thread_local.last_validation_errors = validation_errors

        # If all attempts fail, return None
        logging.warning("All parsing attempts failed, returning None")
        return None

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
        self, responses: List[CitationAnalysis]
    ) -> Optional[Dict]:
        """
        Combine multiple chunk responses into a single citation analysis result.

        Args:
            responses: List of CitationAnalysis objects from processing chunks

        Returns:
            Optional[Dict]: Combined citation analysis in dictionary format, or None if no valid responses

        Raises:
            ValueError: If no valid CitationAnalysis objects are provided
        """
        # Filter out None responses and ensure we have valid ones
        valid_responses = [r for r in responses if r is not None]
        if not valid_responses:
            logging.warning("No valid responses to combine")
            return None

        # Use the cluster_id from the first response if available
        cluster_id = getattr(valid_responses[0], "cluster_id", 0)

        try:
            # Create combined analysis using from_citations
            combined = CombinedResolvedCitationAnalysis.from_citations(
                valid_responses, cluster_id
            )
            return combined.model_dump()
        except Exception as e:
            logging.error(f"Error combining responses: {str(e)}")
            return None

    def generate_content_with_chat(
        self, text: str, model: str = "gemini-2.0-flash-001"
    ) -> Union[List[Dict], Dict]:
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

        if word_count <= 10000:
            try:
                self.rate_limiter.acquire()
                response: GenerateContentResponse | None = (
                    self.client.models.generate_content(
                        model=model,
                        config=self.config,
                        contents=text.strip(),
                    )
                )
                if not response:
                    raise ValueError(
                        f"Received empty response from API for text of length {word_count}"
                    )
                # Use the serializer to convert response to a proper dict
                result = self.serializer.serialize(response)
                if result:
                    return [result]
                else:
                    logging.warning(
                        "Failed to serialize response, returning empty list"
                    )
                    return []
            except Exception as e:
                logging.error(
                    f"Worker {worker_id}: API call failed for text of length {word_count}: {str(e)}"
                )
                raise
            finally:
                self.rate_limiter.release()

        logging.info(
            f"Worker {worker_id}: Text length ({word_count} words) exceeds limit. Using chat-based chunking..."
        )
        chunks = self.chunker.simple_split_into_chunks(text)
        chat = None

        try:
            # Create a chat session with chunked config
            chat: Chat = self.client.chats.create(model=model, config=config_chunking)
            responses: List[CitationAnalysis] = []
            for i, chunk in enumerate(chunks, 1):
                chunk_words = self.chunker.count_words(chunk)
                logging.info(
                    f"Worker {worker_id}: Processing chunk {i}/{len(chunks)} ({chunk_words} words)"
                )

                self.rate_limiter.acquire()
                try:
                    response: GenerateContentResponse = chat.send_message(chunk.strip())
                    if not response or not hasattr(response, "text"):
                        logging.warning(
                            f"Invalid or empty response from chunk {i} (length: {chunk_words})"
                        )
                        continue

                    result = self.serializer.serialize(response)
                    if result:
                        responses.append(result)
                except Exception as e:
                    logging.error(
                        f"Worker {worker_id}: Failed to process chunk {i}/{len(chunks)} "
                        f"(length: {chunk_words}): {str(e)}"
                    )
                finally:
                    self.rate_limiter.release()

            if not responses:
                logging.warning("No valid responses were collected during processing")
                return []

            # If multiple responses were collected, merge them into one consolidated result
            if len(responses) > 1:
                try:
                    merged = self.combine_chunk_responses(responses)
                    return [merged]
                except Exception as e:
                    logging.error(f"Failed to combine chunk responses: {str(e)}")
                    # Return the first valid response if combining fails
                    return [responses[0]]
            return responses

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
                    logging.warning(f"Failed to cleanup chat session: {str(e)}")

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        model: str = "gemini-2.0-flash-001",
        max_workers: Optional[int] = None,
        output_file: Optional[str] = None,
        batch_size: int = 10,
    ) -> Dict[Any, Any]:
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

        results = {}
        errors = []
        total_processed = 0

        # Add debug collection
        debug_info = {
            "raw_responses": {},
            "validation_errors": {},
        }

        def process_row(row) -> tuple[str, List[Dict], Optional[str], dict]:
            worker_id = self.get_worker_id()
            try:
                logging.info(
                    f"Worker {worker_id}: Starting processing cluster_id {row['cluster_id']}"
                )
                result = self.generate_content_with_chat(row[text_column], model)

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
                return row["cluster_id"], result, None, row_debug
            except Exception as e:
                logging.error(
                    f"Worker {worker_id}: Error processing cluster_id {row['cluster_id']}: {str(e)}"
                )
                return row["cluster_id"], None, str(e), {}

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
                            errors.append(
                                {"cluster_id": str(cluster_id), "error": error}
                            )

                        if output_file:
                            # Use atomic write to prevent corruption
                            tmp_file = f"{output_file}.tmp"
                            with open(tmp_file, "w") as f:
                                json.dump(
                                    {str(cid): resp for cid, resp in results.items()},
                                    f,
                                )
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

        with open(debug_path, "w") as f:
            json.dump(debug_info, f, indent=2)

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
