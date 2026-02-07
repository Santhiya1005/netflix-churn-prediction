import joblib

feature_columns = joblib.load("feature_columns.pkl")

print("Total features:", len(feature_columns))
print("Some features:", feature_columns[:50])  # first 50 columns

city_columns = [col for col in feature_columns if col.startswith("City_")]
print("Number of city features:", len(city_columns))
print("Example city features:", city_columns[:10])