import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_xgb(X_train, y_train):
    """Trains an XGBoost model and saves it."""
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))

def predict_xgb(X_test, y_test):
    """Loads trained XGBoost model and makes predictions."""
    xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
    predictions = xgb_model.predict(X_test)
    return accuracy_score(y_test, predictions), f1_score(y_test, predictions, average="weighted"), predictions
