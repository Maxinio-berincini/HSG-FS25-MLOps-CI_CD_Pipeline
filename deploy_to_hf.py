from huggingface_hub import HfApi
import os

HF_REPO_ID = "Maxinio-Berincini/HSG-FS25-MLOps-CI_CD_Pipeline"
MODEL_FILE = "model_artifacts/model.pt"
HF_TOKEN = os.environ["HF_TOKEN"]

api = HfApi()

files_to_upload = {
    "app.py": "app.py",
    "model.py": "model.py",
    "model_artifacts/model.pt": "model.pt"
}

for local_path, repo_path in files_to_upload.items():
    api.upload_file(
         path_or_fileobj=local_path,
         path_in_repo=repo_path,
         repo_id=HF_REPO_ID,
         repo_type="space",
         token=HF_TOKEN,
    )
print("Model deployed to Hugging Face")
