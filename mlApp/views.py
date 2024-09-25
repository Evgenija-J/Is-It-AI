from django.shortcuts import render

from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

logistic_regression_model = load('./savedModels/logistic_regression_model.joblib')
ensemble_model = load('./savedModels/ensemble_model.joblib')
grid_search_model = load('./savedModels/grid_model.joblib')
knn_classifier_model = load('./savedModels/kNeighbors_classifier_model.joblib')
mlp_classifier_model = load('./savedModels/mlp_model.joblib')
multinomial_nb_model = load('./savedModels/multinomialNB_model.joblib')
random_forest_classifier_model = load('./savedModels/random_forest_model.joblib')
svc_model = load('./savedModels/svc_model.joblib')
# xgb_classifier_model = load('./savedModels/xgb_model.joblib')
# vectorizer = load('./savedModels/tfidf_vectorizer.joblib')

def predictor(request):
    return render(request, 'main.html')


def fitVectorizer():
    df = pd.read_csv('./data/data.csv')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

    new_vectorizer = TfidfVectorizer(max_features=5000)

    X_train = new_vectorizer.fit_transform(train_df["text"])
    X_valid = new_vectorizer.transform(valid_df["text"])
    X_test = new_vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_valid = valid_df["label"]
    y_test = test_df["label"]

    return new_vectorizer


vectorizer = fitVectorizer()


def transform_text(new_text):
    new_text_tfidf = vectorizer.transform([new_text])  
    return new_text_tfidf


def formInfo(request):
    text = request.GET['essay'] 
    new_text_tfidf = transform_text(text)

    predictions = {
        'logistic_regression': logistic_regression_model.predict(new_text_tfidf),
        'ensemble': ensemble_model.predict(new_text_tfidf),
        'grid': grid_search_model.predict(new_text_tfidf),
        'kneighbors': knn_classifier_model.predict(new_text_tfidf),
        'mlp': mlp_classifier_model.predict(new_text_tfidf),
        'multinomial': multinomial_nb_model.predict(new_text_tfidf),
        'random_forest': random_forest_classifier_model.predict(new_text_tfidf),
        'svc': svc_model.predict(new_text_tfidf),
        # xgb_prediction = xgb_classifier_model.predict(new_text_tfidf)
    }

    total_predictions = len(predictions)
    ai_predictions = sum(1 for pred in predictions.values() if pred == 1)
    ai_percentage = (ai_predictions / total_predictions) * 100
    

    return render(request, 'result.html', {
        'logistic_regression_prediction': "real" if predictions['logistic_regression'] == 0 else "AI",
        'ensemble_prediction': "real" if predictions['ensemble'] == 0 else "AI",
        'grid_prediction': "real" if predictions['grid'] == 0 else "AI",
        'kneighbors_prediction': "real" if predictions['kneighbors'] == 0 else "AI",
        'mlp_prediction': "real" if predictions['mlp'] == 0 else "AI",
        'multinomial_prediction': "real" if predictions['multinomial'] == 0 else "AI",
        'random_forest_prediction': "real" if predictions['random_forest'] == 0 else "AI",
        'svc_prediction': "real" if predictions['svc'] == 0 else "AI",
        # 'xgb_prediction': "real" if xgb_prediction == 0 else "AI"
        'ai_percentage': ai_percentage,
        })
