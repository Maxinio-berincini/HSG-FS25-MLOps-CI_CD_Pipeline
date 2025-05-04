import os
import mlflow
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi

from scripts.utils.config import (
    REGISTERED_MODEL_NAME,
    PRODUCTION_ALIAS,
    HF_REPO_ID,
    ARTIFACT_DIR,
    PROJECT_ROOT
)

HF_TOKEN = os.environ["HF_TOKEN"]

# initialize MLflow and Hugging Face API
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()
api = HfApi()

# get production model URI
prod_mv = client.get_model_version_by_alias(
    name=REGISTERED_MODEL_NAME,
    alias=PRODUCTION_ALIAS
)
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
print(f"Production model: '{REGISTERED_MODEL_NAME}' v{prod_mv.version}")

# download production model
local_dir = os.path.join(ARTIFACT_DIR, "production")
os.makedirs(local_dir, exist_ok=True)
mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path=local_dir
)
print(f"Downloaded artifacts to '{local_dir}'")

# replace placeholders in app.py
version = prod_mv.version
description = prod_mv.description or ""

with open(PROJECT_ROOT / "app" / "app.py", "r") as f:
    tpl = f.read()

app_py = (
    tpl
    .replace("{{MODEL_VERSION}}", str(version))
    .replace("{{MODEL_DESCRIPTION}}", description.replace("\"", "\\\"").replace("±", "\\u00b1"))
)

with open(PROJECT_ROOT / "app" / "app.py", "w") as f:
    f.write(app_py)

# prepare files for upload
files_to_upload = {
    PROJECT_ROOT / "app" / "app.py": "app/app.py",
    PROJECT_ROOT / "scripts" / "utils" / "model.py": "scripts/utils/model.py",
    os.path.join(local_dir, "data", "model.pth"): "model_artifacts/model.pth",
}

# upload files to repo
for local_path, repo_path in files_to_upload.items():
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")
    print(f"Uploading {local_path} to repo at {repo_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=HF_REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
    )
print("Deployment complete")
