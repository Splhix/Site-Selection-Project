"""
Example script to load a saved risk model.
Usage: python load_model_example.py <model_filename.pkl>
"""
import sys
from pathlib import Path
import joblib
import json

if len(sys.argv) < 2:
    print("Usage: python load_model_example.py <model_filename.pkl>")
    sys.exit(1)

script_dir = Path(__file__).parent
models_dir = script_dir / "models"
model_path = models_dir / sys.argv[1]

if not model_path.exists():
    print(f"❌ Model file not found: {model_path}")
    sys.exit(1)

# Load the model
print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Try to load corresponding metadata
metadata_path = model_path.with_suffix('.json').with_name(
    model_path.stem.replace('risk_model_', 'risk_model_') + '_metadata.json'
)
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"\n📊 Model Metadata:")
    print(f"  - Type: {metadata.get('model_type')}")
    print(f"  - CV Accuracy: {metadata.get('cv_accuracy_mean'):.4f} ± {metadata.get('cv_accuracy_std'):.4f}")
    print(f"  - Samples: {metadata.get('n_samples')}")
    print(f"  - Features: {metadata.get('features_numeric')} + {metadata.get('features_categorical')}")

print("\nModel is ready to use for predictions!")

