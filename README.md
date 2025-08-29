# Model predviđanja kategorija proizvoda
Ovaj projekat implementira model mašinskog učenja koji predviđa kategorije proizvoda na osnovu naziva proizvoda. Model je dizajniran da automatski klasifikuje proizvode u odgovarajuće kategorije, što ga čini korisnim za platforme za elektronsku trgovinu i sisteme za upravljanje proizvodima.

## Karakteristike
- Predviđanje kategorija proizvoda u realnom vremenu;
- Jednostavan interfejs;
- Sistem predviđanja zasnovan na nazivu proizvoda;
- Čist i formatiran prikaz rezultata.

## Korišćene tehnologije
- Python
- Google Colab
- scikit-learn
- numpy
- pandas
- joblib

## Struktura projekta
```
ml-category-prediction/
├── data/
│   └── products.csv
├── model/
│   └── category_prediction_model.pkl
├── notebook/
│   └── ML_Category_Prediction.ipynb
├── src/
│   └── predict_category.py
│   └── train_model.py
├── .gitignore
└── README.md
```

## Instalacija
1. Klonirajte repozitorijum:
```bash
git clone https://github.com/Marko-Vuchko/ml-category-prediction.git
```
2. Instalirajte potrebne biblioteke:
```bash
pip install scikit-learn pandas numpy joblib
```

## Korišćenje:
1. Pronadjite direktorijum projekta:
```bash
cd ml-category-prediction
```
2. Pokrenite skriptu za predikciju:
```bash
python src/predict_category.py
```
3. Unesite naziv proizvoda kada se to od vas zatraži. Model će predvideti njegovu kategoriju.
4. Unesite "q" da biste izašli iz programa.

## Primer
```
Model loaded successfully!
Type 'q' at any point to stop.

Enter product title: iPhone 7 32GB Gold
Predicted category: Mobile Phones
----------------------------------------
```

## Informacije o modelu
Model je obučen da klasifikuje proizvode u različite kategorije na osnovu njihovog naziva. Koristi proces mašinskog učenja koji uključuje:
- Prethodnu obradu teksta;
- Ekstrakciju karakteristika;
- Algoritam klasifikacije.

## Autor
***Marko Vučković***
