
from src.llm_extraction.rate_limited_gemini import GeminiClient, TextChunker
import os
from google.genai.types import GenerateContentConfig, CreateBatchJobConfig, BatchJob
from google import genai

from google.cloud import storage
import fsspec
import logging

from src.llm_extraction.models import CitationAnalysis
import pandas as pd
import json

from src.llm_extraction.prompts import system_prompt, chunking_instructions

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
class BatchGeminiClient:
    """
    GeminiBatchClient for batch job submission.
    This client uses the same text chunking logic, message formatting, and configuration as GeminiClient,
    but produces a JSONL file suitable for submission as a batch job.
    """
    def __init__(
            self, 
                 chunking_config=None,
                   model=GeminiClient.DEFAULT_MODEL, 
                   word_threshold=TextChunker.WORD_COUNT_THRESHOLD,
                   results_bucket=os.environ["GEMINI_RESULTS_BUCKET"],
                   input_bucket=os.environ["GEMINI_INPUT_BUCKET"]

                   ):
        self.results_bucket = results_bucket
        self.input_bucket = input_bucket

        self.genai_client = genai.Client(vertexai=True, project=os.environ["GEMINI_PROJECT"], location=os.environ["GEMINI_LOCATION"])
        self.config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CitationAnalysis,
            system_instruction=system_prompt
        )
        self.config_chunking = chunking_config or GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CitationAnalysis,
            system_instruction=f"{system_prompt}\n\n{chunking_instructions}",
            temperature=0.4
        )
        self.model = model
        self.word_threshold = word_threshold
        self.chunker = TextChunker()

    def prepare_request_for_row(self, row, text_column="text"):
        import json
        text = row.get(text_column, "")
        if not text or not text.strip():
            raise ValueError("Empty or invalid text input")

        word_count = self.chunker.count_words(text)
        if word_count > self.word_threshold:
            chunks = self.chunker.simple_split_into_chunks(text, max_words=self.word_threshold)
        else:
            chunks = [text.strip()]

        messages = []
        for chunk in chunks:
            message = {"role": "user", "parts": [{"text": chunk}]}
            messages.append(message)

        generation_config = self.config_chunking if word_count > self.word_threshold else self.config
        # Convert generation_config to dict; use model_dump if available
        if hasattr(generation_config, "model_dump"):
            gen_config_dict = generation_config.model_dump()
        else:
            gen_config_dict = generation_config.__dict__

        return {"request": {"contents": messages, "generationConfig": gen_config_dict}}


    def write_requests_to_file(self, df: pd.DataFrame, output_file_name: str, text_column: str = "text"):
        lines = []
        for i, row in df.iterrows():
            try:
                payload = self.prepare_request_for_row(row, text_column=text_column)
                lines.append(json.dumps(payload))
            except Exception as e:
                # Optionally log or handle errors; skipping rows with errors.
                logging.error(f"Error preparing request for row {i} {row}: {e}")
                continue

        # use tmp dir
        output_file_name = os.path.join("/tmp", output_file_name)
        with open(output_file_name, "w", encoding="utf-8") as f:
            for r in lines:
                f.write(r + "\n")
        
        return output_file_name

    def upload_file_to_bucket(self, file_path: str, bucket_name: str, file_name: str):
        """
        Upload a file to a bucket.
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)

        return blob.public_url


    def submit_batch_job(self, df, jsonl_file_name: str, text_column="text"):
        """
        Write the JSONL file from the dataframe and submit a batch job
        using the provided genai client and batch job configuration.
        """
        jsonl_file = self.write_requests_to_file(df, jsonl_file_name, text_column=text_column)
        jsonl_file_url = self.upload_file_to_bucket(jsonl_file, self.input_bucket, jsonl_file_name)


        batch_job = self.genai_client.batches.create(
            model=self.model,
            src=jsonl_file_url,
            config=CreateBatchJobConfig(
                dest=f"gs://{self.results_bucket}"
            )
        )
        return batch_job
    
    def check_batch_job_status(self, batch_job: BatchJob):
        if not batch_job.name:
            raise ValueError("Batch job name somehow is not set")
        
        batch_job = self.genai_client.batches.get(name=batch_job.name)
        logging.info(f"Batch job {batch_job.name} status: {batch_job.state}")
        return batch_job
    
    def retrieve_batch_job_results(self, batch_job: BatchJob):
        


        fs = fsspec.filesystem("gcs")

        file_paths = fs.glob(f"{batch_job.dest.gcs_uri}/*/predictions.jsonl") # type: ignore

        if batch_job.state == "JOB_STATE_SUCCEEDED":
            # Load the JSONL file into a DataFrame
            df = pd.read_json(f"gs://{file_paths[0]}", lines=True)

            df = df.join(pd.json_normalize(df["response"], "candidates")) # type: ignore
            return df