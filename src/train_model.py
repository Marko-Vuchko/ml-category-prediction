# Biblioteke
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

# Učitavanje podataka
df = pd.read_csv("data/products.csv")

# Uklanjanje redova sa nedostajućim vrednostima
df = df.dropna()

# Standardizacija imena kolona
df.columns = (df.columns.str.lower().str.strip().str.replace("_", " ").str.replace(" ", "_").str.lstrip("_"))

# Konvertovanje kolone 'category_label' u string, pretvaranje u mala slova i uklanjanje razmaka
df["category_label"] = df["category_label"].astype(str).str.lower().str.strip()

# Mapiranje nepravilnih vrednosti na ispravne
df["category_label"] = df["category_label"].replace({"fridges": "fridge", "mobile phones": "mobile phone", "fridge freezers": "freezers","cpus": "cpu"})

# Konvertovanje kolone 'category_label' u kategorijski tip podataka
df["category_label"] = df["category_label"].astype("category")

# Uklanjanje kolona koje nisu potrebne za treniranje modela
df = df.drop(columns=["product_id", "merchant_id", "product_code", "merchant_rating", "listing_date"])

# Karakteristike i oznake
X = df[["product_title", "number_of_views"]]
y = df["category_label"]

# Preprocesiranje podataka
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "product_title"),
        ("views", MinMaxScaler(), ["number_of_views"])
    ]
)

# Definisanje pipeline-a za najbolji model
pipeline = Pipeline([
    ["preprocessing", preprocessor],
    ["classifier", LinearSVC()]
])

# Treniranje modela
pipeline.fit(X, y)

# Čuvanje modela
joblib.dump(pipeline, "model/category_prediction_model.pkl") 
print("Model trained and saved as 'model/category_prediction_model.pkl'")