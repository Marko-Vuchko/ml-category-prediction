# Biblioteke
import joblib
import pandas as pd

# Učitavanje modela
model = joblib.load("model/category_prediction_model.pkl")

# Obaveštenje o uspešnom učitavanju modela i instrukcije za izlaz
print("Model loaded successfully!")
print("Type 'q' at any point to stop.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "q":
        print("Exiting...")
        break
 
    # Kreiranje DataFrame-a za predikciju
    user_input = pd.DataFrame([{
        "review_title": title,
        "product_title": title,
        "number_of_views": 0 
    }])
    
    # Kreiranje predikcije i prikaz rezultata
    prediction = model.predict(user_input)
    clean_prediction = prediction[0].strip("[]'").title()
    print(f"\nPredicted category: {clean_prediction}\n" + "-" * 50)