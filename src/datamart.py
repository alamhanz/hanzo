from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()


def download_file_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a file from Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the file in the bucket.
        destination_file_name (str): The local destination path where the file will be saved.
    """
    try:
        # Initialize the GCS client
        storage_client = storage.Client()

        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Get the blob (file object)
        blob = bucket.blob(source_blob_name)

        # Download the file
        blob.download_to_filename(destination_file_name)

        print(f"File {source_blob_name} downloaded to {destination_file_name}.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    # Replace these variables with your values
    bucket_name = "public-artifact"
    source_blob_name = (
        "kaggle-data/child-mind-sleep-state/base-feat_eng-v0_test.parquet"
    )
    destination_file_name = "temp/test.parquet"

    download_file_from_gcs(bucket_name, source_blob_name, destination_file_name)

# public-artifact/kaggle-data/child-mind-sleep-state/base-feat_eng-v0_test.parquet
