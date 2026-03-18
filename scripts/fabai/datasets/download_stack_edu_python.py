import gzip

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from datasets import load_dataset

load_dotenv(override=True)

num_proc = 30
s3 = boto3.client('s3')
bucket_name = "softwareheritage"

def download_contents(blob_id):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise

# For Python
ds = load_dataset("HuggingFaceTB/stack-edu", "Python", split="train", num_proc=num_proc)
ds = ds.map(download_contents, input_columns="blob_id", num_proc=num_proc)

# Filter out failed downloads
ds = ds.filter(lambda x: x['download_success'])

# Optionally, print the first example to verify the data
print(ds[0])
