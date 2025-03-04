import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_rf(X_train, y_train):
    """Trains a Random Forest model and saves it."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    pickle.dump(rf_model, open("rf_model.pkl", "wb"))

def predict_rf(X_test, y_test):
    """Loads trained RF model and makes predictions."""
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    predictions = rf_model.predict(X_test)
    return accuracy_score(y_test, predictions), f1_score(y_test, predictions, average="weighted"), predictions
