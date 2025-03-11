import json
import logging
import os

import fsspec
import pandas as pd
from google import genai
from google.cloud import storage
from google.genai.types import (BatchJob, CreateBatchJobConfig,
                                GenerateContentConfig, GenerateContentResponse)

from src.api.services.pipeline.pipeline_model import ExtractionConfig
from src.api.services.pipeline.pipeline_service import check_node_status
from src.llm_extraction.models import CitationAnalysis
from src.llm_extraction.prompts import chunking_instructions, system_prompt
from src.llm_extraction.rate_limited_gemini import GeminiClient, TextChunker

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

        # Get project ID and location from environment
        project_id = os.environ["GEMINI_PROJECT"]
        location = os.environ["GEMINI_LOCATION"]
        
        # Check if credentials file is available
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path and os.path.exists(credentials_path):
            # Use credentials from the JSON file
            logging.info(f"Using Google credentials from: {credentials_path}")
            # The genai.Client will automatically use GOOGLE_APPLICATION_CREDENTIALS
            # when vertexai=True
        
        self.genai_client = genai.Client(vertexai=True, project=project_id, location=location)
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
        # Convert row to a plain dictionary if it's not already
        if hasattr(row, "to_dict"):
            row_dict = row.to_dict()
        else:
            # Make a copy to avoid modifying the original
            row_dict = dict(row)
        
        text = row_dict.get(text_column, "")
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
        # if hasattr(generation_config, "model_dump"):
        #     gen_config_dict = generation_config.model_dump()
        # else:
        #     gen_config_dict = generation_config.__dict__
            
        # # Ensure only JSON serializable data is included
        # gen_config_dict = {k: v for k, v in gen_config_dict.items() 
        #                   if isinstance(v, (str, int, float, bool, list, dict)) or v is None}

        return {"request": {"contents": messages, "generationConfig": {"systemPrompt": generation_config.system_instruction, "temperature": generation_config.temperature, "responseMimeType": generation_config.response_mime_type, "responseSchema": CitationAnalysis.model_json_schema()}}}

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
        
        Uses credentials from:
        1. GOOGLE_APPLICATION_CREDENTIALS environment variable (path to mounted JSON file)
        2. Falls back to GEMINI_PROJECT environment variable
        """
        # First check if credentials file is available
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.environ.get("GEMINI_PROJECT")
        
        if credentials_path and os.path.exists(credentials_path):
            # Use credentials from the JSON file
            logging.info(f"Using Google credentials from: {credentials_path}")
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            storage_client = storage.Client(credentials=credentials, project=project_id)
        elif project_id:
            # Fall back to project ID only
            logging.info(f"Using default credentials with project: {project_id}")
            storage_client = storage.Client(project=project_id)
        else:
            raise ValueError("Neither GOOGLE_APPLICATION_CREDENTIALS nor GEMINI_PROJECT environment variables are set")
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)

        # Return the gs:// URL format required by the genai client
        return f"gs://{bucket_name}/{file_name}"

    def extract_opinions(self, config: ExtractionConfig, db_connection) -> pd.DataFrame:
        """
        Extract opinions from the database using the same logic as pipeline_service.
        
        Args:
            config: Extraction configuration
            db_connection: Database connection
            
        Returns:
            DataFrame containing the extracted opinions
        """
        logging.info(f"Extracting opinions with config: {config}")
        
        # Build SQL query based on configuration - same logic as in pipeline_service
        filters = []
        params = {}

        # Check if we're processing a single cluster ID
        if config.single_cluster_id:
            filters.append("soc.id = %(single_cluster_id)s")
            params["single_cluster_id"] = config.single_cluster_id
            logging.info(f"Extracting single cluster ID: {config.single_cluster_id}")

        if config.court_id:
            filters.append("sd.court_id = %(court_id)s")
            params["court_id"] = config.court_id

        if config.start_date:
            filters.append("soc.date_filed >= %(start_date)s")
            params["start_date"] = config.start_date

        if config.end_date:
            filters.append("soc.date_filed <= %(end_date)s")
            params["end_date"] = config.end_date

        filter_clause = " AND ".join(filters)
        if filter_clause:
            filter_clause = (
                f"WHERE {filter_clause} AND soc.precedential_status = 'Published'"
            )
        else:
            filter_clause = "WHERE soc.precedential_status = 'Published'"

        # Build the query; adding html fields for cleaning
        query = f"""
        SELECT  
            so.cluster_id as cluster_id, 
            so.type as so_type, 
            so.id as so_id, 
            so.page_count as so_page_count, 
            so.html_with_citations as so_html_with_citations, 
            so.html as so_html, 
            so.plain_text as so_plain_text, 
            soc.case_name as cluster_case_name,
            soc.date_filed as soc_date_filed,
            soc.citation_count as soc_citation_count,
            sd.court_id as court_id,
            sd.docket_number as sd_docket_number,
            sc.full_name as court_name
        FROM search_opinion so 
        LEFT JOIN search_opinioncluster soc ON so.cluster_id = soc.id
        LEFT JOIN search_docket sd ON soc.docket_id = sd.id
        LEFT JOIN search_court sc ON sd.court_id = sc.id
        {filter_clause}
        ORDER BY soc.date_filed DESC
        """

        if config.limit:
            query += f" LIMIT {config.limit}"

        if config.offset:
            query += f" OFFSET {config.offset}"

        logging.info(f"Executing query: {query} with params: {params}")
        
        # Execute the query
        df = pd.read_sql(query, db_connection, params=params)
        logging.info(f"Extracted {len(df)} opinions")
        
        return df

    def clean_opinions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the extracted opinions using the same logic as pipeline_service.
        
        Args:
            df: DataFrame containing the extracted opinions
            
        Returns:
            Cleaned DataFrame
        """
        from src.api.services.pipeline.pipeline_service import \
            clean_extracted_opinions
        
        logging.info(f"Cleaning {len(df)} opinions")
        cleaned_df = clean_extracted_opinions(df)
        logging.info(f"Cleaned to {len(cleaned_df)} opinions")
        
        return cleaned_df

    def submit_batch_job_from_config(self, config: ExtractionConfig, db_connection, jsonl_file_name: str) -> BatchJob:
        """
        Extract opinions based on config, clean them, and submit a batch job.
        
        Args:
            config: Extraction configuration
            db_connection: Database connection
            jsonl_file_name: Name for the JSONL file
            
        Returns:
            The submitted batch job
        """
        # Extract opinions from the database
        df = self.extract_opinions(config, db_connection)
        
        # Clean the opinions
        cleaned_df = self.clean_opinions(df)
        
        # Save the cleaned opinions to a CSV file for reference
        cleaned_csv_path = os.path.join("/tmp", f"cleaned_{jsonl_file_name.replace('.jsonl', '.csv')}")
        cleaned_df.to_csv(cleaned_csv_path, index=False)
        logging.info(f"Saved cleaned opinions to {cleaned_csv_path}")
        
        # Import check_node_status from pipeline_service
        
        # Initialize already_processed_df with the same columns as cleaned_df
        already_processed_df = pd.DataFrame(columns=cleaned_df.columns)
        
        # Check if each opinion already exists in Neo4j with ai_summary
        for index, row in cleaned_df.iterrows():
            cluster_id = row["cluster_id"]
            node_status = check_node_status(str(cluster_id))
            if node_status.exists and node_status.has_ai_summary:
                logging.info(
                    f"Cluster {cluster_id} already exists in Neo4j with ai_summary. Skipping LLM processing to save API requests."
                )
                already_processed_df = pd.concat(
                    [already_processed_df, pd.DataFrame([row])], ignore_index=True
                )
        
        # Drop the already processed rows from the cleaned df
        cleaned_df = cleaned_df[
            ~cleaned_df["cluster_id"].isin(already_processed_df["cluster_id"])
        ]
        logging.info(
            f"Remaining opinions to process: {len(cleaned_df)}. Dropped {len(already_processed_df)} opinions that already exist in Neo4j"
        )
        
        # If no opinions left to process, return early with empty batch
        if len(cleaned_df) == 0:
            logging.warning("No opinions to process after filtering out existing ones in Neo4j")
            # Create a dummy batch job to indicate no processing is needed
            # We'll use a special flag in the job store to indicate this
            logging.info("Returning empty batch job")
            # Return a placeholder batch job with a special name to indicate no processing needed
            return BatchJob(name="NO_PROCESSING_NEEDED")
        # Prepare the text column for batch processing
        # The pipeline expects a 'text' column, so we'll create one if it doesn't exist
        if 'text' not in cleaned_df.columns:
            cleaned_df['text'] = cleaned_df['so_plain_text']
            logging.info("Created 'text' column from 'so_plain_text'")
        
        # Submit the batch job
        return self.submit_batch_job(cleaned_df, jsonl_file_name, text_column="text")

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
        """
        Retrieve the results of a completed batch job.
        
        Args:
            batch_job: The batch job object
            
        Returns:
            DataFrame containing the results
        """
        # Special case for when no processing was needed
        if hasattr(batch_job, 'name') and batch_job.name == "NO_PROCESSING_NEEDED":
            logging.info("No processing was needed - returning empty DataFrame")
            return pd.DataFrame()  # Return empty DataFrame

        fs = fsspec.filesystem("gcs")
        
        # Check if batch_job.dest exists and has gcs_uri attribute
        if not hasattr(batch_job, 'dest') or not batch_job.dest or not hasattr(batch_job.dest, 'gcs_uri'):
            logging.error(f"Batch job destination not properly defined: {batch_job}")
            return pd.DataFrame()  # Return empty DataFrame
            
        file_paths = fs.glob(f"{batch_job.dest.gcs_uri}/*/predictions.jsonl")
        
        if not file_paths:
            logging.warning(f"No prediction files found at {batch_job.dest.gcs_uri}")
            return pd.DataFrame()  # Return empty DataFrame if no files found

        if batch_job.state == "JOB_STATE_SUCCEEDED":
            # Load the JSONL file into a DataFrame
            file_path = f"gs://{file_paths[0]}"
            logging.info(f"Loading results from {file_path}")
            df = pd.read_json(file_path, lines=True)
            
            # Debug: Print the columns and a sample row
            logging.info(f"DataFrame columns: {df.columns.tolist()}")
            if not df.empty:
                logging.info(f"Sample row: {df.iloc[0].to_dict()}")
                
                # Check if 'response' column exists
                if 'response' not in df.columns:
                    logging.error("'response' column not found in results DataFrame")
                    return df  # Return what we have
                
                # Process each response
                try:
                    # Create a new column for the processed responses
                    processed_responses = []
                    error_responses = []
                    
                    for idx, row in df.iterrows():
                        response_data = row['response']
                        logging.debug(f"Processing response {idx}, type: {type(response_data)}")
                        
                        # Check if the response is an error message
                        is_error = False
                        error_message = None
                        
                        if isinstance(response_data, dict) and 'status' in response_data:
                            if 'error' in response_data:
                                is_error = True
                                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                                logging.warning(f"Response {idx} contains an error: {error_message}")
                        
                        if isinstance(response_data, str):
                            try:
                                json_data = json.loads(response_data)
                                if isinstance(json_data, dict) and 'status' in json_data and 'error' in json_data:
                                    is_error = True
                                    error_message = json_data.get('error', {}).get('message', 'Unknown error')
                                    logging.warning(f"Response {idx} contains an error: {error_message}")
                            except json.JSONDecodeError:
                                pass  # Not a JSON string, continue processing
                        
                        if is_error:
                            # Add to error responses
                            error_responses.append({
                                'original_index': idx,
                                'error_message': error_message,
                                'raw_response': str(response_data)[:500]  # Truncate for logging
                            })
                            # Add an empty dict to processed responses to maintain alignment
                            processed_responses.append({})
                        else:
                            # Process as normal response
                            try:
                                # For now, just convert to a dictionary if possible
                                if isinstance(response_data, dict):
                                    processed_responses.append(response_data)
                                elif isinstance(response_data, str):
                                    try:
                                        processed_responses.append(json.loads(response_data))
                                    except json.JSONDecodeError:
                                        # If it's not valid JSON, just use it as is
                                        processed_responses.append({'text': response_data})
                                else:
                                    processed_responses.append({'data': str(response_data)})
                            except Exception as e:
                                logging.exception(f"Error processing response {idx}: {str(e)}")
                                processed_responses.append({})
                    
                    # Log error statistics
                    if error_responses:
                        logging.warning(f"Found {len(error_responses)} error responses out of {len(df)} total")
                        error_df = pd.DataFrame(error_responses)
                        # Save error responses for debugging
                        error_df.to_csv('/tmp/gemini_batch_errors.csv', index=False)
                        logging.info(f"Saved error responses to /tmp/gemini_batch_errors.csv")
                    
                    # Create a DataFrame from the processed responses
                    if processed_responses:
                        # First, normalize the processed responses
                        try:
                            # Try to normalize the responses
                            normalized_df = pd.json_normalize(processed_responses)
                            
                            # Combine with the original DataFrame
                            result_df = pd.concat([
                                df.drop('response', axis=1).reset_index(drop=True),
                                normalized_df.reset_index(drop=True)
                            ], axis=1)
                            
                            return result_df
                        except Exception as e:
                            logging.exception(f"Error normalizing processed responses: {str(e)}")
                            # Fall back to returning the original DataFrame with a new column
                            df['processed_response'] = processed_responses
                            return df
                
                except Exception as e:
                    logging.exception(f"Error processing responses: {str(e)}")
                    # Return the original DataFrame without modifications
            
            return df