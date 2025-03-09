import time
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from google.genai.chats import Chat

from json_repair import repair_json
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from tqdm import tqdm
from datetime import datetime
from threading import local
import random
import threading

from src.llm_extraction.models import (
    Citation,
    CitationAnalysis,
    CombinedResolvedCitationAnalysis,
)
from src.llm_extraction.prompts import system_prompt, chunking_instructions

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextChunker:
    """Handles text chunking and response combining."""

    # Word count threshold for chunking text
    WORD_COUNT_THRESHOLD = 8000

    # Add a delay between processing chunks to avoid rate limit issues
    CHUNK_PROCESSING_DELAY = 2.0  # seconds

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
    def simple_split_into_chunks(
        text: str, max_words: Optional[int] = None
    ) -> List[str]:
        """Split text into chunks using paragraph boundaries so that each chunk is under max_words."""
        if max_words is None:
            max_words = TextChunker.WORD_COUNT_THRESHOLD
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
        # Add counter for tracking token usage
        self.tokens_used = 0
        self.last_minute = int(self.last_update / 60)
        logging.info(f"TokenBucket initialized with rate={rate}, capacity={capacity}")

    def update_rate(self, new_rate: float) -> None:
        """Update the token generation rate."""
        with self.lock:
            old_rate = self.rate
            self.rate = new_rate
            self.capacity = new_rate  # Update capacity to match rate
            logging.info(f"TokenBucket rate updated: {old_rate:.2f} -> {new_rate:.2f}")

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed_minutes = (now - self.last_update) / 60.0  # Convert to minutes
        new_tokens = elapsed_minutes * self.rate  # Rate is already in tokens per minute

        # Reset counter if we've moved to a new minute
        current_minute = int(now / 60)
        if current_minute > self.last_minute:
            logging.info(
                f"New minute: {self.tokens_used} tokens used in previous minute"
            )
            self.tokens_used = 0
            self.last_minute = current_minute

        # Be more conservative - only add 90% of calculated tokens to account for timing issues
        conservative_new_tokens = new_tokens * 0.9
        old_tokens = self.tokens
        self.tokens = min(self.capacity, self.tokens + conservative_new_tokens)

        # Log token replenishment if significant
        if conservative_new_tokens > 0.1:
            logging.debug(
                f"Added {conservative_new_tokens:.2f} tokens. Before: {old_tokens:.2f}, After: {self.tokens:.2f}"
            )

        self.last_update = now

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking."""
        with self.lock:
            self._add_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                self.tokens_used += 1
                return True
            return False

    def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        attempts = 0
        while True:
            with self.lock:
                self._add_tokens()
                if self.tokens >= 1:
                    self.tokens -= 1
                    self.tokens_used += 1
                    if attempts > 0:
                        logging.info(f"Token acquired after {attempts} attempts")
                    return

            attempts += 1
            if attempts == 1:
                logging.info(
                    f"Waiting for token. Current tokens: {self.tokens:.2f}, Used in this minute: {self.tokens_used}"
                )
            elif attempts % 10 == 0:  # Log every 10 attempts
                logging.info(
                    f"Still waiting for token after {attempts} attempts. Current tokens: {self.tokens:.2f}"
                )

            # Exponential backoff with jitter to prevent thundering herd
            sleep_time = min(0.1 * (1.5 ** min(attempts, 10)), 5.0)  # Cap at 5 seconds
            sleep_time = sleep_time * (0.5 + random.random())  # Add jitter
            time.sleep(sleep_time)  # Sleep outside the lock


class RateLimiter:
    """Rate limiter using token bucket algorithm with concurrent request limiting."""

    def __init__(self, rpm_limit: int = 50, max_concurrent: int = 10):
        # Be more conservative with the rate limit
        conservative_rpm = int(rpm_limit * 0.9)  # Use 90% of the limit
        self.token_bucket = TokenBucket(
            rate=conservative_rpm, capacity=conservative_rpm
        )
        self.concurrent_semaphore = Semaphore(max_concurrent)
        self.acquire_count = 0
        self.lock = Lock()
        self.rpm_limit = rpm_limit
        self.max_concurrent = max_concurrent
        self.error_count = 0
        self.success_count = 0
        self.last_adjustment = time.time()
        self.rate_limit_errors = 0
        self.active_requests = 0  # Track currently active requests
        self.creation_time = time.time()
        self.last_stats_log = time.time()
        self.stats_log_interval = 60  # Log stats every minute

        # Create a unique ID for this rate limiter instance for logging
        self.limiter_id = f"RL-{int(self.creation_time % 10000):04d}"

        logging.info(
            f"[{self.limiter_id}] RateLimiter initialized with rpm_limit={conservative_rpm} (90% of {rpm_limit}), max_concurrent={max_concurrent}"
        )

    def log_stats(self, force: bool = False):
        """Log current statistics about the rate limiter usage."""
        now = time.time()
        if force or (now - self.last_stats_log) >= self.stats_log_interval:
            uptime = now - self.creation_time
            requests_per_minute = (
                self.acquire_count / (uptime / 60) if uptime > 0 else 0
            )
            success_rate = (
                (self.success_count / self.acquire_count * 100)
                if self.acquire_count > 0
                else 0
            )

            logging.info(
                f"[{self.limiter_id}] Stats: "
                f"Uptime={uptime:.1f}s, "
                f"Requests={self.acquire_count}, "
                f"RPM={requests_per_minute:.1f}, "
                f"Active={self.active_requests}, "
                f"Success={self.success_count} ({success_rate:.1f}%), "
                f"Errors={self.error_count}, "
                f"RateLimit={self.rate_limit_errors}"
            )
            self.last_stats_log = now

    def acquire(self) -> None:
        """Acquire permission to make a request, blocking if necessary."""
        with self.lock:
            self.acquire_count += 1

        # First acquire a token from the token bucket (rate limit)
        self.token_bucket.acquire()

        # Then acquire the semaphore (concurrent limit)
        self.concurrent_semaphore.acquire()

        # Track active requests
        with self.lock:
            self.active_requests += 1
            # Log stats periodically
            self.log_stats()

        logging.debug(
            f"[{self.limiter_id}] Acquired permission (active: {self.active_requests})"
        )

    def release(self) -> None:
        """Release a permit, allowing another request to proceed."""
        # Update active requests count
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)

        # Release the semaphore
        self.concurrent_semaphore.release()

        logging.debug(
            f"[{self.limiter_id}] Released permission (active: {self.active_requests})"
        )

    def report_success(self) -> None:
        """Report a successful API call, potentially adjusting rate limit upward."""
        with self.lock:
            self.success_count += 1

            # Only consider increasing rate if we've had a good number of successes
            now = time.time()
            if (
                self.success_count >= 20  # At least 20 successful requests
                and self.error_count == 0  # No errors
                and now - self.last_adjustment
                > 300  # At least 5 minutes since last adjustment
            ):
                current_rate = self.token_bucket.rate
                # Don't exceed the original RPM limit
                new_rate = min(self.rpm_limit, current_rate * 1.1)  # Increase by 10%

                # Only log if there's an actual change
                if new_rate > current_rate:
                    self.token_bucket.update_rate(new_rate)
                    logging.info(
                        f"[{self.limiter_id}] Increasing rate limit from {current_rate:.2f} to {new_rate:.2f} RPM after {self.success_count} successful requests"
                    )
                    self.last_adjustment = now
                    self.success_count = 0  # Reset counter

                    # Force log stats after adjustment
                    self.log_stats(force=True)

    def report_error(self, is_rate_limit_error: bool = False) -> None:
        """Report an error, potentially adjusting the rate limit."""
        with self.lock:
            self.error_count += 1
            if is_rate_limit_error:
                self.rate_limit_errors += 1
                logging.warning(
                    f"[{self.limiter_id}] Rate limit error detected (total: {self.rate_limit_errors})"
                )

            # Only adjust if we've seen enough errors and it's been a while since last adjustment
            now = time.time()
            if (
                is_rate_limit_error
                and now - self.last_adjustment
                > 60  # At least 1 minute since last adjustment
            ):
                # Calculate error rate over the last minute
                error_rate = self.error_count / max(1, self.acquire_count)

                # If error rate is high, reduce the rate limit
                if error_rate > 0.1:  # More than 10% errors
                    current_rate = self.token_bucket.rate
                    new_rate = max(1, int(current_rate * 0.8))  # Reduce by 20%
                    self.token_bucket.update_rate(new_rate)
                    logging.warning(
                        f"[{self.limiter_id}] Reducing rate limit from {current_rate} to {new_rate} due to high error rate ({error_rate:.2f})"
                    )
                    self.last_adjustment = now

                    # Force log stats after adjustment
                    self.log_stats(force=True)


def repair_json_string(json_str: str) -> Optional[dict]:
    """
    Simple JSON repair function that tries to fix common issues.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON as dict if successful, otherwise None.
    """
    try:
        # First try standard repair
        repaired = repair_json(json_str, return_objects=True)
        # Test if it's valid JSON
        if isinstance(repaired, dict):
            return repaired
        elif (
            isinstance(repaired, tuple)
            and len(repaired) > 0
            and isinstance(repaired[0], dict)
        ):
            return repaired[0]
        else:
            logging.warning(f"Repaired JSON is not a dict: {type(repaired)}")
            return None
    except Exception as e:
        logging.error(f"JSON repair failed: {str(e)}")
        return None


class GlobalRateLimiter:
    """Global singleton rate limiter that can be shared across all clients.

    This class implements the Singleton pattern to ensure only one rate limiter
    instance is created and shared across all threads and processes.

    Usage:
        limiter = GlobalRateLimiter.get_instance(rpm_limit=25, max_concurrent=10)
    """

    _instance = None
    _lock = Lock()
    _initialized_params = None

    def __new__(cls, *args, **kwargs):
        """Prevent direct instantiation of this class.

        Users should use get_instance() instead.
        """
        raise TypeError(
            "GlobalRateLimiter cannot be instantiated directly. Use GlobalRateLimiter.get_instance() instead."
        )

    @classmethod
    def get_instance(cls, rpm_limit: int = 50, max_concurrent: int = 10) -> RateLimiter:
        """Get or create the global rate limiter instance.

        This method ensures a single rate limiter is shared across all threads and processes.
        If an instance already exists, it will be returned regardless of the parameters provided.

        Args:
            rpm_limit: Requests per minute limit
            max_concurrent: Maximum number of concurrent requests

        Returns:
            The global RateLimiter instance
        """
        with cls._lock:
            if cls._instance is None:
                logging.info(
                    f"Creating global rate limiter with RPM={rpm_limit}, concurrent={max_concurrent}"
                )
                cls._instance = RateLimiter(
                    rpm_limit=rpm_limit, max_concurrent=max_concurrent
                )
                cls._initialized_params = {
                    "rpm_limit": rpm_limit,
                    "max_concurrent": max_concurrent,
                }
            else:
                # Log if parameters are different from what was used to initialize
                if cls._initialized_params is not None:
                    if (
                        cls._initialized_params["rpm_limit"] != rpm_limit
                        or cls._initialized_params["max_concurrent"] != max_concurrent
                    ):
                        logging.warning(
                            f"Requested rate limiter with RPM={rpm_limit}, concurrent={max_concurrent}, "
                            f"but using existing instance with RPM={cls._initialized_params['rpm_limit']}, "
                            f"concurrent={cls._initialized_params['max_concurrent']}"
                        )

                logging.debug(f"Reusing existing global rate limiter instance")

            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the global rate limiter instance.

        This is primarily useful for testing purposes.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized_params = None
            logging.info("Global rate limiter instance has been reset")


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
                    result = ResponseSerializer._validate_json_data(
                        repaired_json, validation_errors, repaired=True
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
            if len(json_data) > 1:
                logging.warning(
                    f"Found {len(json_data)} items in list, using first item, but here is preview of second item: {json_data[1]}"
                )

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
                # Try cleaning the data before giving up
                try:
                    cleaned_data = ResponseSerializer._try_clean_citation_lists(
                        json_data[0], validation_errors, prefix
                    )
                    if cleaned_data:
                        try:
                            return CitationAnalysis.model_validate(cleaned_data)
                        except Exception as e2:
                            logging.warning(
                                f"Failed to validate cleaned list item: {str(e2)}"
                            )
                except Exception as clean_error:
                    logging.warning(
                        f"Error while trying to clean list item: {str(clean_error)}"
                    )

        # Handle dict responses
        elif isinstance(json_data, dict):
            try:
                # Try direct validation first
                return CitationAnalysis.model_validate(json_data)
            except Exception as e:
                # If it fails, maybe we have invalid items in citation lists
                # Try to clean up the data by filtering out invalid items
                cleaned_data = ResponseSerializer._try_clean_citation_lists(
                    json_data, validation_errors, prefix
                )

                if cleaned_data:
                    try:
                        # Try validation with cleaned data
                        return CitationAnalysis.model_validate(cleaned_data)
                    except Exception as e2:
                        validation_errors.append(
                            {
                                "stage": f"{prefix}dict_validation_after_cleaning",
                                "error": str(e2),
                                "input": cleaned_data,
                            }
                        )
                        logging.warning(
                            f"Failed to validate {prefix}dict after cleaning: {str(e2)}"
                        )
                else:
                    validation_errors.append(
                        {
                            "stage": f"{prefix}dict_validation",
                            "error": str(e),
                            "input": json_data,
                        }
                    )
                    logging.warning(f"Failed to validate {prefix}dict: {str(e)}")

        # Store for potential fallback
        if not hasattr(ResponseSerializer, "_thread_local"):
            ResponseSerializer._thread_local = local()
        ResponseSerializer._thread_local.input_data = json_data
        return None

    @staticmethod
    def _try_clean_citation_lists(
        data: Dict, validation_errors: List, prefix: str = ""
    ) -> Optional[Dict]:
        """
        Attempts to clean citation lists by filtering out invalid items.

        Args:
            data: Dictionary containing citation lists
            validation_errors: List to append validation errors to
            prefix: Optional prefix for error logging

        Returns:
            Cleaned dictionary or None if cleaning failed
        """
        try:
            # Create a copy to avoid modifying the original
            cleaned_data = data.copy()

            # Use the correct field names from the CitationAnalysis model
            citation_fields = [
                "majority_opinion_citations",
                "concurring_opinion_citations",
                "dissenting_citations",
            ]

            for field in citation_fields:
                if field in cleaned_data and isinstance(cleaned_data[field], list):
                    original_count = len(cleaned_data[field])
                    valid_items = []

                    for i, item in enumerate(cleaned_data[field]):
                        try:
                            # Directly validate against Citation model instead of field checking
                            Citation.model_validate(item)
                            valid_items.append(item)
                        except Exception as e:
                            logging.info(
                                f"Filtering out invalid item {i} in {field}: {str(e)}"
                            )

                    # Update the list with only valid items
                    cleaned_data[field] = valid_items

                    if len(valid_items) < original_count:
                        logging.warning(
                            f"Filtered {original_count - len(valid_items)} invalid items from {field}"
                        )

            return cleaned_data

        except Exception as e:
            validation_errors.append(
                {
                    "stage": f"{prefix}citation_list_cleaning",
                    "error": str(e),
                    "input": data,
                }
            )
            logging.warning(f"Failed to clean citation lists: {str(e)}")
            return None

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
            ResponseSerializer._thread_local = local()
            return None, None

        return (
            getattr(ResponseSerializer._thread_local, "last_raw_response", None),
            getattr(ResponseSerializer._thread_local, "last_validation_errors", None),
        )


class GeminiClient:
    DEFAULT_MODEL = "gemini-2.0-flash-001"
    _worker_counter = 0
    _worker_counter_lock = Lock()

    def __init__(
        self,
        api_key: str,
        rpm_limit: int = 50,
        max_concurrent: int = 10,
        config: Optional[GenerateContentConfig] = None,
        model: str = DEFAULT_MODEL,
        use_global_rate_limiter: bool = True,
    ):
        self.api_key = api_key  # Store the API key directly
        self.client = genai.Client(api_key=api_key)
        # Ensure we have a valid config
        self.config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CitationAnalysis,
            system_instruction=system_prompt,
        )

        # Create chunking config
        self.config_chunking = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CitationAnalysis,
            system_instruction=f"{system_prompt}\n\n{chunking_instructions}",
        )
        self.rpm_limit = rpm_limit
        self.max_concurrent = max_concurrent

        # Use either global or local rate limiter based on parameter
        if use_global_rate_limiter:
            # Get the global singleton rate limiter
            self.rate_limiter = GlobalRateLimiter.get_instance(
                rpm_limit=rpm_limit, max_concurrent=max_concurrent
            )
            logging.info(f"Using global rate limiter for GeminiClient instance")
        else:
            # Create a new instance-specific rate limiter
            self.rate_limiter = RateLimiter(
                rpm_limit=rpm_limit, max_concurrent=max_concurrent
            )
            logging.info(f"Using local rate limiter for GeminiClient instance")

        # For tracking worker IDs in multi-threaded environments
        with self._worker_counter_lock:
            self._worker_id = self._worker_counter
            GeminiClient._worker_counter += 1

        self.chunker = TextChunker()
        self.serializer = ResponseSerializer()
        self.model = model

        logging.info(
            f"Initialized client with RPM limit: {rpm_limit}, AFC concurrent limit: {max_concurrent}, model: {model}"
        )

    @classmethod
    def create_shared_clients(
        cls,
        api_key: str,
        num_clients: int,
        rpm_limit: int = 50,
        max_concurrent: int = 10,
        model: str = DEFAULT_MODEL,
    ) -> List["GeminiClient"]:
        """Create multiple GeminiClient instances that share the same rate limiter.

        This is useful for creating a pool of clients that will be used in a multi-threaded
        environment, ensuring they all respect the same rate limits.

        This method leverages the GlobalRateLimiter singleton pattern to ensure all clients
        share exactly the same rate limiter instance, preventing race conditions and ensuring
        proper rate limiting across all threads.

        Args:
            api_key: The API key to use for all clients
            num_clients: The number of clients to create
            rpm_limit: The requests per minute limit to use
            max_concurrent: The maximum number of concurrent requests
            model: The model to use

        Returns:
            A list of GeminiClient instances that share the same rate limiter
        """
        # First, ensure the global rate limiter is initialized with the desired parameters
        # This will create the singleton if it doesn't exist yet
        GlobalRateLimiter.get_instance(
            rpm_limit=rpm_limit, max_concurrent=max_concurrent
        )

        # Then create the clients, all using the same global rate limiter
        clients = []
        for _ in range(num_clients):
            clients.append(
                cls(
                    api_key=api_key,
                    rpm_limit=rpm_limit,
                    max_concurrent=max_concurrent,
                    model=model,
                    use_global_rate_limiter=True,  # Always use global rate limiter for shared clients
                )
            )

        logging.info(
            f"Created {num_clients} GeminiClient instances with shared rate limiter"
        )
        return clients

    def get_worker_id(self) -> int:
        """Get worker ID for current thread/client instance.

        Each GeminiClient instance has a unique worker ID assigned at creation time.
        This is useful for tracking which client is handling which request in logs.

        Returns:
            The worker ID for this client instance
        """
        return self._worker_id

    def generate_content_with_chat(
        self,
        text: str,
        cluster_id: int,
        model: str = "gemini-2.0-flash-001",
        max_retries: int = 3,
    ) -> Optional[CitationAnalysis]:
        """Generate content using chat history for better context preservation."""
        model = model or self.model
        worker_id = self.get_worker_id()

        if not text or not text.strip():
            raise ValueError("Empty or invalid text input")

        word_count = self.chunker.count_words(text)

        # Ensure we have a valid config before proceeding
        if not self.config:
            raise ValueError("Configuration is required but not provided")

        # Use appropriate config based on word count
        config_to_use = (
            self.config_chunking
            if word_count > TextChunker.WORD_COUNT_THRESHOLD
            else self.config
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
                        # Report success to the rate limiter
                        self.rate_limiter.report_success()
                        return result

                    # Validation failed
                    self.rate_limiter.release()

                    # Only retry if we haven't exceeded max attempts
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Worker {worker_id}: Validation failed, retrying (attempt {attempt + 1}/{max_retries})"
                        )
                        # Use exponential backoff with jitter for retries
                        backoff_time = 1.0 * (2**attempt) * (0.5 + random.random())
                        logging.info(
                            f"Worker {worker_id}: Backing off for {backoff_time:.2f} seconds before retry"
                        )
                        time.sleep(
                            backoff_time
                        )  # Longer delay before retry with exponential backoff
                    else:
                        logging.warning(
                            f"Worker {worker_id}: All {max_retries} validation attempts failed"
                        )
                        return None

                except Exception as e:
                    # Always release on exception
                    self.rate_limiter.release()

                    error_msg = str(e)
                    logging.error(f"Worker {worker_id}: API call failed: {error_msg}")

                    # Check if this is a rate limit error
                    is_rate_limit_error = (
                        "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg
                    )
                    if is_rate_limit_error:
                        # Report rate limit error to the rate limiter
                        self.rate_limiter.report_error(is_rate_limit_error=True)
                        # Use longer backoff for rate limit errors
                        backoff_time = 5.0 * (2**attempt) * (0.5 + random.random())
                        logging.warning(
                            f"Worker {worker_id}: Rate limit error detected. Backing off for {backoff_time:.2f} seconds"
                        )
                        time.sleep(backoff_time)
                    else:
                        # Report other error to the rate limiter
                        self.rate_limiter.report_error(is_rate_limit_error=False)
                        # Standard backoff for other errors
                        backoff_time = 1.0 * (2**attempt) * (0.5 + random.random())
                        logging.info(
                            f"Worker {worker_id}: Backing off for {backoff_time:.2f} seconds before retry"
                        )
                        time.sleep(backoff_time)

                    # Only retry if we haven't exceeded max attempts
                    if attempt < max_retries - 1:
                        logging.info(
                            f"Worker {worker_id}: Retrying after error (attempt {attempt + 1}/{max_retries})"
                        )
                    else:
                        logging.error(
                            f"Worker {worker_id}: All {max_retries} attempts failed with errors"
                        )
                        return None

            # Should never reach here, but just in case
            return None

        # Handle small texts (single chunk processing)
        if word_count <= TextChunker.WORD_COUNT_THRESHOLD:
            try:
                result = process_single_chunk(text, self.config)
                # Return the raw CitationAnalysis
                return result
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
        chat: Optional[Chat] = None

        try:
            # Create a chat session with chunked config
            chat = self.client.chats.create(model=model, config=config_to_use)
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
                                # Use exponential backoff with jitter for retries
                                backoff_time = (
                                    1.0 * (2**attempt) * (0.5 + random.random())
                                )
                                logging.info(
                                    f"Worker {worker_id}: Backing off for {backoff_time:.2f} seconds before retry"
                                )
                                time.sleep(backoff_time)  # Longer delay before retry
                                continue
                            else:
                                chunk_failures += 1
                                break

                        # Try to validate/serialize the response
                        result = self.serializer.serialize(response)
                        if result:
                            # Success! Release the rate limiter and store the result
                            self.rate_limiter.release()
                            # Report success to the rate limiter
                            self.rate_limiter.report_success()
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
                            # Use exponential backoff with jitter for retries
                            backoff_time = 1.0 * (2**attempt) * (0.5 + random.random())
                            logging.info(
                                f"Worker {worker_id}: Backing off for {backoff_time:.2f} seconds before retry"
                            )
                            time.sleep(backoff_time)  # Longer delay before retry
                        else:
                            logging.warning(
                                f"Worker {worker_id}: All {max_retries} validation attempts failed for chunk {i}"
                            )
                            chunk_failures += 1

                    except Exception as e:
                        # Always release on exception
                        self.rate_limiter.release()

                        error_msg = str(e)
                        logging.error(
                            f"Worker {worker_id}: Error processing chunk {i}: {error_msg}"
                        )

                        # Check if this is a rate limit error
                        is_rate_limit_error = (
                            "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg
                        )
                        if is_rate_limit_error:
                            # Report rate limit error to the rate limiter
                            self.rate_limiter.report_error(is_rate_limit_error=True)
                            # Use longer backoff for rate limit errors
                            backoff_time = 5.0 * (2**attempt) * (0.5 + random.random())
                            logging.warning(
                                f"Worker {worker_id}: Rate limit error detected. Backing off for {backoff_time:.2f} seconds"
                            )
                            time.sleep(backoff_time)
                        else:
                            # Report other error to the rate limiter
                            self.rate_limiter.report_error(is_rate_limit_error=False)
                            # Standard backoff for other errors
                            backoff_time = 1.0 * (2**attempt) * (0.5 + random.random())
                            logging.info(
                                f"Worker {worker_id}: Backing off for {backoff_time:.2f} seconds before retry"
                            )
                            time.sleep(backoff_time)

                        # Only retry if we haven't exceeded max attempts
                        if attempt < max_retries - 1:
                            logging.warning(
                                f"Worker {worker_id}: Retrying chunk {i} after error (attempt {attempt + 1}/{max_retries})"
                            )

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

                # Add delay between chunks to avoid rate limit issues
                if i < len(chunks):
                    delay = TextChunker.CHUNK_PROCESSING_DELAY * (
                        0.5 + random.random()
                    )  # Add jitter
                    logging.info(
                        f"Worker {worker_id}: Waiting {delay:.2f}s before processing next chunk"
                    )
                    time.sleep(delay)

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

            # Use the CitationAnalysis class method to combine responses
            try:
                return CitationAnalysis.combine_analyses(responses)
            except Exception as e:
                logging.error(
                    f"Worker {worker_id}: Failed to combine chunk responses: {str(e)}"
                )
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
    ) -> Dict[int, Optional[CitationAnalysis]]:
        """Process a DataFrame of content generation requests using thread pool."""
        # Adjust max_workers based on DataFrame size and rate limit
        if max_workers is None:
            # Be more conservative with max_workers to avoid overwhelming the rate limiter
            suggested_workers = min(
                max(
                    1, int(self.rpm_limit * 0.7)
                ),  # Use at most 70% of RPM as worker count
                self.max_concurrent,
                len(df),
            )
            max_workers = suggested_workers
            logging.info(
                f"Auto-adjusted max_workers to {max_workers} (70% of RPM limit)"
            )

        # Adjust batch_size to be no larger than the DataFrame
        batch_size = min(
            max(batch_size, max_workers),  # Ensure batch size >= max_workers
            len(df),
        )

        # Create a pool of clients that share the same rate limiter
        # This ensures all workers respect the same global rate limits
        # Use the stored API key
        api_key = self.api_key

        # Use the class method to create shared clients
        shared_clients = self.create_shared_clients(
            api_key=api_key,
            num_clients=max_workers,
            rpm_limit=self.rpm_limit,
            max_concurrent=self.max_concurrent,
            model=model,
        )

        # Create a thread-local storage to assign clients to worker threads
        thread_local = threading.local()

        # Function to get a client for the current thread
        def get_thread_client():
            if not hasattr(thread_local, "client_index"):
                # Assign a client to this thread
                with threading.Lock():
                    if not hasattr(thread_local, "client_index"):
                        thread_local.client_index = random.randint(
                            0, len(shared_clients) - 1
                        )
            return shared_clients[thread_local.client_index]

        results: Dict[int, Optional[CitationAnalysis]] = {}
        errors = []
        total_processed = 0

        # Add debug collection
        debug_info = {
            "raw_responses": {},
            "validation_errors": {},
        }

        def process_row(
            row,
        ) -> tuple[int, Optional[CitationAnalysis], Optional[str], dict]:
            # Get the client assigned to this thread
            thread_client = get_thread_client()
            worker_id = thread_client.get_worker_id()

            try:
                logging.info(
                    f"Worker {worker_id}: Starting processing cluster_id {row['cluster_id']}"
                )
                # Use the thread's client to generate content
                result = thread_client.generate_content_with_chat(
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

        # Calculate total number of batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        logging.info(f"Total number of batches: {total_batches}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process in batches
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]
                current_batch = batch_start // batch_size + 1

                logging.info(
                    f"Processing batch {current_batch}/{total_batches}, rows {batch_start} to {batch_end}"
                )

                # Create futures for this batch only
                futures = [
                    executor.submit(process_row, row) for _, row in batch_df.iterrows()
                ]

                # Process futures for this batch
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing batch {current_batch}/{total_batches}",
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
                                json.dump(serialized_results, f, indent=2)
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

        # Helper function to make objects JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, (CitationAnalysis, CombinedResolvedCitationAnalysis)):
                return obj.model_dump()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj

        # Make debug_info JSON serializable before saving
        serializable_debug_info = make_json_serializable(debug_info)

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(serializable_debug_info, f, indent=2, ensure_ascii=False)

        logging.info(f"Debug information saved to {debug_path}")

        if errors:
            logging.warning(f"Encountered {len(errors)} errors during processing")

        return results


