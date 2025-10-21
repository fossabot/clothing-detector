#!/usr/bin/env python3
"""
Minimal S3ModelDownloader class
"""

import boto3
import os
import yaml
from datetime import datetime

class S3ModelDownloader:
    def __init__(self, config_file=None):
        """Initialize with config file"""
        if config_file is None:
            # Get the directory where this file is located
            detector_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(detector_dir, 'configs', 'config.yaml')
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup AWS credentials (config file > env vars)
        aws_config = self.config.get('aws', {})
        self.aws_config = {
            'access_key_id': aws_config.get('access_key_id') or os.getenv('AWS_ACCESS_KEY_ID'),
            'secret_access_key': aws_config.get('secret_access_key') or os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region': aws_config.get('region', 'us-east-1')
        }
        
        # Create S3 client
        if self.aws_config['access_key_id'] and self.aws_config['secret_access_key']:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_config['access_key_id'],
                aws_secret_access_key=self.aws_config['secret_access_key'],
                region_name=self.aws_config['region']
            )
        else:
            self.s3_client = boto3.client('s3', region_name=self.aws_config['region'])
    
    def _should_skip_file(self, local_path, bucket_name, s3_path):
        """Check if file should be skipped (compare with S3)"""
        if not os.path.exists(local_path):
            return False
        
        try:
            # Get S3 object metadata
            response = self.s3_client.head_object(Bucket=bucket_name, Key=s3_path)
            s3_last_modified = response['LastModified']
            
            # Get local file modification time
            local_mtime = os.path.getmtime(local_path)
            local_last_modified = datetime.fromtimestamp(local_mtime, tz=s3_last_modified.tzinfo)
            
            # Skip if local file is newer or same age
            if local_last_modified >= s3_last_modified:
                print(f"File is up to date, skipping: {local_path}")
                return True
            else:
                print(f"Newer version available on S3, will download: {local_path}")
                return False
                
        except Exception as e:
            print(f"Could not check S3 metadata, skipping: {e}")
            return True  # Skip on error to be safe
    
    def download_file(self, bucket_name, s3_path, local_path):
        """Download single file"""
        if self._should_skip_file(local_path, bucket_name, s3_path):
            return
        
        print(f"Downloading s3://{bucket_name}/{s3_path} to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3_client.download_file(bucket_name, s3_path, local_path)
        print("Download complete")
    
    def download_all(self, force=False):
        """Download all files from config"""
        for download in self.config.get('downloads', []):
            bucket = download['bucket']
            s3_path = download['s3_path']
            local_path = download['local_path']
            
            if s3_path.endswith('/'):
                # Directory download
                os.makedirs(local_path, exist_ok=True)
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket, Prefix=s3_path):
                    for obj in page.get('Contents', []):
                        if not obj['Key'].endswith('/'):
                            key = obj['Key']
                            rel_path = key[len(s3_path):].lstrip('/')
                            file_path = os.path.join(local_path, rel_path)
                            
                            if force or not self._should_skip_file(file_path, bucket, key):
                                self.s3_client.download_file(bucket, key, file_path)
            else:
                # Single file download
                self.download_file(bucket, s3_path, local_path)
