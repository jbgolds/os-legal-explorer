#!/usr/bin/env python3
"""
Script to upload files to a Cloudflare R2 bucket.

This script uses boto3 to interact with Cloudflare R2 storage (which is S3-compatible).
It supports uploading a single file or all files in a directory.

Usage:
    python upload_to_cloudflare.py --file path/to/file.txt --bucket my-bucket --new-name object_name
    python upload_to_cloudflare.py --directory path/to/dir --bucket my-bucket

Environment variables:
    R2_ACCESS_KEY: Your Cloudflare R2 access key ID
    R2_SECRET_ACCESS_KEY: Your Cloudflare R2 secret access key
    R2_ENDPOINT_URL: Your Cloudflare R2 endpoint URL (e.g., https://accountid.r2.cloudflarestorage.com)
    R2_REGION: Your Cloudflare R2 region
"""

import argparse
import logging
import mimetypes
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()




def upload_file(file_path, bucket, uploaded_file_name: str):
    """
    Upload a file to a Cloudflare R2 bucket.

    Args:
        file_path (str): Path to the file to upload
        bucket (str): Bucket name
        uploaded_file_name (str): S3 object name for the uploaded file
        
    Returns:
        bool: True if file was uploaded, False otherwise
    """
    # Get credentials from environment variables
    access_key = os.getenv("R2_ACCESS_KEY")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("R2_ENDPOINT_URL")
    
    if not all([access_key, secret_key, endpoint_url]):
        logger.error(
            "Missing required environment variables. Please set R2_ACCESS_KEY, "
            "R2_SECRET_ACCESS_KEY, and R2_ENDPOINT_URL."
        )
        sys.exit(1)

  
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="enam",
    )
    
    logger.info(f"Uploading {file_path} to bucket {bucket} as {uploaded_file_name}")
    with open(file_path, "rb") as f:
        client.upload_fileobj(f, bucket, uploaded_file_name)
    
    logger.info(f"Successfully uploaded {file_path} to {bucket}/{uploaded_file_name}")
    return True
    




def main():
    """Main function to parse arguments and upload files."""
    parser = argparse.ArgumentParser(description="Upload files to Cloudflare R2 bucket")
    parser.add_argument("--file", help="Path to the file to upload")
    parser.add_argument("--new-name", help="Name of the uploaded file")
    parser.add_argument("--bucket", help="R2 bucket name")
    args = parser.parse_args()

    # Check if file exists
    if not os.path.isfile(args.file):
        logger.error(f"File not found: {args.file}")
        return 1
        
    success = upload_file(args.file, args.bucket, args.new_name)
    return 0 if success else 1
    
if __name__ == "__main__":
    sys.exit(main()) 