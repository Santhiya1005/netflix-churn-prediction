import joblib
import pickle

def load_features():
    try:
        cols = joblib.load("feature_columns.pkl")
    except Exception:
        with open("feature_columns.pkl", "rb") as f:
            cols = pickle.load(f)
    return cols

cols = load_features()

print("Total feature columns:", len(cols))
print("-" * 50)
for i, col in enumerate(cols, start=1):
    print(f"{i}. {col}")