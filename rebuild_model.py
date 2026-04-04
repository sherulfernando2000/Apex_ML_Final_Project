"""Rebuild model.joblib with sklearn 1.3.2 for Vertex AI compatibility."""
import subprocess
import sys
import codecs

# Change standard output to handle utf-8 to avoid charmap errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Step 1: Downgrade sklearn exactly to 1.3.2
print("Installing scikit-learn 1.3.2 for Vertex AI compatibility...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.2", "numpy==1.26.4", "pandas==2.2.3", "joblib==1.4.2", "imbalanced-learn==0.12.4", "-q"])

# Force reload
import importlib
import sklearn
importlib.reload(sklearn)
print(f"scikit-learn version: {sklearn.__version__}")

# Step 2: Rebuild the pipeline
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = 'artifacts'

# Load processed data
df = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'processed_data.csv'))

# Define features and target
target = 'customer_segment'
exclude = ['price', 'delivery_status', 'customer_segment']
feature_cols = [c for c in df.columns if c not in exclude]

numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()

X = df[feature_cols]
y = df[target]

# Encode target
le_segment = LabelEncoder()
y_enc = le_segment.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)

# Train classifier
classifier = LogisticRegression(
    solver='lbfgs', C=1.0, max_iter=5000,
    class_weight='balanced', random_state=42
)

# Create unified pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

final_pipeline.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, f1_score
y_pred = final_pipeline.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")

# Save
model_path = os.path.join(ARTIFACTS_DIR, 'model.joblib')
joblib.dump(final_pipeline, model_path)
print(f"\n[OK] model.joblib rebuilt with sklearn {sklearn.__version__}")
print(f"   File size: {os.path.getsize(model_path) / 1024:.1f} KB")

# Save label encoder too
joblib.dump(le_segment, os.path.join(ARTIFACTS_DIR, 'le_segment_deployment.pkl'))
print("[OK] Label encoder saved")

# Verify it loads correctly
loaded = joblib.load(model_path)
test_pred = loaded.predict(X_test[:3])
print(f"\nVerification predictions: {le_segment.inverse_transform(test_pred)}")
print("\n[OK] Ready to upload to GCS and deploy!")
