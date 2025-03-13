from huggingface_hub import HfApi
import os

HF_REPO_ID = "Maxinio-Berincini/HSG-FS25-MLOps-CI_CD_Pipeline"
MODEL_FILE = "model_artifacts/model.pt"
HF_TOKEN = os.environ["HF_TOKEN"]

api = HfApi()
api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo="model.pt",
    repo_id=HF_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
)
print("Model deployed to Hugging Face")
