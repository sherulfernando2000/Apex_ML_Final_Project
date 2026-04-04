# 🚀 AuraCart Deployment Guide

## 1. Create GCS Bucket

### Option A: `gcloud` CLI

```bash
gcloud auth login
gcloud config set project ml-final-project-apex

gcloud storage buckets create gs://auracart-ml-models \
    --project=ml-final-project-apex \
    --location=asia-southeast1 \
    --uniform-bucket-level-access

gcloud storage ls gs://auracart-ml-models
```

> **💡** Bucket names are globally unique. If taken, try `auracart-ml-models-<initials>`.

### Option B: Google Cloud Console

1. Go to [Cloud Storage](https://console.cloud.google.com/storage/browser)
2. Click **CREATE** → Name: `auracart-ml-models` → Region: `asia-southeast1` → Standard → Uniform → **CREATE**

### Option C: Python (in Notebook)

```python
from google.cloud import storage

PROJECT_ID = 'ml-final-project-apex'
BUCKET_NAME = 'auracart-ml-models'
LOCATION = 'asia-southeast1'

client = storage.Client(project=PROJECT_ID)
try:
    bucket = client.get_bucket(BUCKET_NAME)
    print(f'✅ Bucket already exists: gs://{BUCKET_NAME}')
except Exception:
    bucket = client.bucket(BUCKET_NAME)
    bucket.storage_class = 'STANDARD'
    bucket = client.create_bucket(bucket, location=LOCATION)
    print(f'✅ Bucket created: gs://{BUCKET_NAME} in {LOCATION}')
```

---

## 2. Fix Container Version Clash

### ❌ The Problem

Your notebook uses this container:
```python
serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest'
```

But your model was trained with **scikit-learn 1.5.2** (see `requirements.txt`).

The `sklearn-cpu.1-3` container bundles sklearn **1.3.x** — when it tries to load a model saved with 1.5.2, it crashes:
```
FailedPrecondition: 400 Model server exited unexpectedly.
```

### ✅ The Fix

Change **one line** in the Step 6 cell — replace the container URI:

```diff
- serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest',
+ serving_container_image_uri='asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest',
```

| Container Tag     | sklearn Version | Match? |
|-------------------|-----------------|--------|
| `sklearn-cpu.1-3` | ~1.3.x          | ❌     |
| `sklearn-cpu.1-4` | ~1.4.x          | ❌     |
| `sklearn-cpu.1-5` | ~1.5.x          | ✅     |
| `sklearn-cpu.1-6` | ~1.6.x          | ❌     |

> **Rule:** The container's major.minor must match the scikit-learn version used to save the model.  
> Using `asia-docker` instead of `us-docker` reduces latency for Singapore region.

---

## 3. Updated Step 6 Cell (Copy-Paste)

Replace the entire Step 6 code cell in `4_mlops_deployment.ipynb` with:

```python
# === Vertex AI Deployment ===
import google.cloud.aiplatform as aiplatform

aiplatform.init(
    project=PROJECT_ID,
    location='asia-southeast1'
)

# Container must match sklearn version in requirements.txt (1.5.x)
SKLEARN_CONTAINER = 'asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest'

# Step 1: Import the model
model = aiplatform.Model.upload(
    display_name='auracart-customer-segment-classifier',
    artifact_uri=f'gs://{BUCKET_NAME}/{MODEL_DIR}',
    serving_container_image_uri=SKLEARN_CONTAINER,
)
print(f'Model uploaded: {model.display_name}')
print(f'Model resource name: {model.resource_name}')

# Step 2: Deploy to an endpoint
endpoint = model.deploy(
    deployed_model_display_name='auracart-segment-v1',
    machine_type='n1-standard-2',
    min_replica_count=1,
    max_replica_count=1,
)
print(f'\n✅ Endpoint created: {endpoint.display_name}')
print(f'Endpoint resource name: {endpoint.resource_name}')
```

---

## 4. Before Re-running

Before you re-run Step 6, clean up the failed resources in GCP Console:

1. **Delete the failed endpoint:** [Vertex AI → Endpoints](https://console.cloud.google.com/vertex-ai/endpoints) → delete `auracart-segment-v1`
2. **Delete the old model:** [Vertex AI → Model Registry](https://console.cloud.google.com/vertex-ai/models) → delete `auracart-customer-segment-classifier`
3. Then re-run the updated Step 6 cell above
