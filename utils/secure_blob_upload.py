from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# === Configuration ===
storage_account_name = "aq0344ffd6f6f95419189ec0a"
container_name = "starlightfree"  
blob_name = "datasets/labeled_logs.csv"
local_file_path = "../datasets/logs/labeled_logs.csv"

# === Generate Secure BlobServiceClient ===
account_url = f"https://{storage_account_name}.blob.core.windows.net"
credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

# === Upload File to Blob ===
with open(local_file_path, "rb") as data:
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(data, overwrite=True)

print(f"✅ Securely uploaded {blob_name} to {container_name} in {storage_account_name}")
