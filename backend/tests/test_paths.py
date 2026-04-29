from pathlib import Path

# Simulate the paths as they would be in train_classifier.py
script_path = Path(__file__).resolve()
ROOT_PATH = script_path.parent  # This would be backend/ when script is in backend/
BACKEND_PATH = ROOT_PATH  # backend/

# But train_classifier.py is in backend/ml/, so:
ml_script_path = ROOT_PATH / "ml" / "train_classifier.py"
ROOT_PATH_ACTUAL = ml_script_path.resolve().parent  # backend/ml/
BACKEND_PATH_ACTUAL = ROOT_PATH_ACTUAL.parent  # backend/

print("=" * 60)
print("Path Resolution Test")
print("=" * 60)
print(f"ML script path: {ml_script_path}")
print(f"ROOT_PATH (ml dir): {ROOT_PATH_ACTUAL}")
print(f"BACKEND_PATH: {BACKEND_PATH_ACTUAL}")
print()
print(f"Data path should be: {ROOT_PATH_ACTUAL / 'data' / 'processed' / 'travel_data.csv'}")
print(f"Data exists: {(ROOT_PATH_ACTUAL / 'data' / 'processed').exists()}")
print()
print(f"Models dir should be: {BACKEND_PATH_ACTUAL / 'models'}")
print(f"Models dir exists: {(BACKEND_PATH_ACTUAL / 'models').exists()}")
print()
print(f"Outputs dir will be: {ROOT_PATH_ACTUAL / 'outputs'}")
print()
print(f"project_root for metadata: {BACKEND_PATH_ACTUAL}")
