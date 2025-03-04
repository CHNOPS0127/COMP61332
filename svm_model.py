import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def train_svm(X_train, y_train):
    """Trains an SVM model and saves it."""
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    pickle.dump(svm_model, open("svm_model.pkl", "wb"))

def predict_svm(X_test, y_test):
    """Loads trained SVM model and makes predictions."""
    svm_model = pickle.load(open("svm_model.pkl", "rb"))
    predictions = svm_model.predict(X_test)
    return accuracy_score(y_test, predictions), f1_score(y_test, predictions, average="weighted"), predictions
