import time
import json
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from google.genai.chats import Chat
from json_repair import repair_json
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore
from tqdm import tqdm
from datetime import datetime
from threading import local

from src.llm_extraction.models import CitationAnalysis
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


class ResponseSerializer:
    """Handles serialization of Gemini API responses."""

    @staticmethod
    def serialize(response: GenerateContentResponse | None) -> Dict[str, Any]:
        """Serialize a response to a consistent dictionary format."""
        if response is None:
            raise ValueError("Invalid input type")

        if response.parsed:
            # doubt this will be the case, think OpenAI client does this casting automatically.
            if isinstance(response.parsed, CitationAnalysis):
                return response.parsed.model_dump()

            if isinstance(response.parsed, str):
                try:
                    return json.loads(response.parsed)
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse response as JSON")
            if isinstance(response.parsed, dict):
                return response.parsed

            raise ValueError(
                f"Unexpected response type for response.parsed: {type(response.parsed)}"
            )

        # try and decode from `text` field, which may be invalid json
        if response.text is not None:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                try:
                    return repair_json(response.text)
                except Exception as e:
                    logging.error(f"Failed to parse response as JSON: {str(e)}")
                    return {"error": str(e)}

        return {"error": "No response data available"}


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
        self.config = config
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

    def generate_content_with_chat(
        self, text: str, model: str = "gemini-2.0-flash-001"
    ) -> Union[List[Dict]]:
        """Generate content using chat history for better context preservation."""
        model = model or self.model
        worker_id = self.get_worker_id()

        word_count = self.chunker.count_words(text)

        # Create a copy of the config with chunking instructions if needed
        if word_count > 10000:
            config = GenerateContentConfig(
                response_mime_type=self.config.response_mime_type,
                response_schema=self.config.response_schema,
                system_instruction=(
                    self.config.system_instruction
                    + "\n\n The document will be sent in multiple parts. "
                    "For each part, analyze the citations and legal arguments while maintaining context "
                    "from previous parts. Please provide your analysis in the same structured format "
                    "filling in the lists of citation analysis for each response."
                ),
            )
        else:
            config = self.config

        if word_count <= 10000:
            try:
                self.rate_limiter.acquire()
                response: GenerateContentResponse | None = (
                    self.client.models.generate_content(
                        model=model,
                        config=config,
                        contents=text.strip(),
                    )
                )
                if not response:
                    raise ValueError(
                        f"Received empty response from API for text of length {word_count}"
                    )
                # Use the serializer to convert response to a proper dict
                return [self.serializer.serialize(response)]
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
            chat: Chat = self.client.chats.create(model=model, config=config)
            responses: List[Dict] = []
            for i, chunk in enumerate(chunks, 1):
                chunk_words = self.chunker.count_words(chunk)
                logging.info(
                    f"Worker {worker_id}: Processing chunk {i}/{len(chunks)} ({chunk_words} words)"
                )

                self.rate_limiter.acquire()
                try:
                    response: GenerateContentResponse = chat.send_message(chunk.strip())
                    if not response or not hasattr(response, "text"):
                        raise ValueError(
                            f"Invalid or empty response from chunk {i} (length: {chunk_words})"
                        )
                    responses.append(self.serializer.serialize(response))
                except Exception as e:
                    logging.error(
                        f"Worker {worker_id}: Failed to process chunk {i}/{len(chunks)} "
                        f"(length: {chunk_words}): {str(e)}"
                    )
                    raise
                finally:
                    self.rate_limiter.release()

            if not responses:
                raise ValueError("No valid responses were collected during processing")

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
        if max_workers is None:
            max_workers = min(
                10, self.rate_limiter.token_bucket.rate
            )  # Conservative default

        results = {}
        errors = []
        total_processed = 0

        def process_row(row) -> tuple[str, List[Dict], Optional[str]]:
            worker_id = self.get_worker_id()
            try:
                logging.info(
                    f"Worker {worker_id}: Starting processing cluster_id {row['cluster_id']}"
                )
                result = self.generate_content_with_chat(row[text_column], model)
                logging.info(
                    f"Worker {worker_id}: Finished processing cluster_id {row['cluster_id']}"
                )
                return row["cluster_id"], result, None
            except Exception as e:
                logging.error(
                    f"Worker {worker_id}: Error processing cluster_id {row['cluster_id']}: {str(e)}"
                )
                return row["cluster_id"], None, str(e)

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
                        cluster_id, result, error = future.result()
                        results[cluster_id] = result
                        total_processed += 1

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

        if errors:
            logging.warning(f"Encountered {len(errors)} errors during processing")

        return results


# Example usage:
def main():
    config = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CitationAnalysis,
        system_instruction=system_prompt,
    )

    client = GeminiClient(
        api_key=os.getenv("GEMINI_API_KEY"),
        rpm_limit=10,  # Conservative limits for testing
        max_concurrent=10,  # Respect AFC limit
        config=config,
    )

    df = pd.read_csv("data_final/supreme_court_1950_some_processing.csv")
    df = df.sample(250)  # Take first 25 rows for testing

    # Process DataFrame with max_workers respecting AFC limit
    results = client.process_dataframe(
        df,
        text_column="text",
        output_file=f"responses_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        max_workers=10,  # Match AFC limit
    )

    print(f"Processed {len(results)} items")


if __name__ == "__main__":
    main()
