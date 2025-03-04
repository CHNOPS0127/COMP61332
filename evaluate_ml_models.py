import pandas as pd
from preprocess import *
from svm_model import predict_svm
from rf_model import predict_rf
from xgb_model import predict_xgb

# Load processed data
X_test = pd.read_csv("ML_features_test.csv")
y_test = pd.read_csv("ML_labels_test.csv").values.ravel()

# Evaluate models
svm_acc, svm_f1, svm_preds = predict_svm(X_test, y_test)
rf_acc, rf_f1, rf_preds = predict_rf(X_test, y_test)
xgb_acc, xgb_f1, xgb_preds = predict_xgb(X_test, y_test)

# Display results
print(f"SVM - Accuracy: {svm_acc}, F1 Score: {svm_f1}")
print(f"Random Forest - Accuracy: {rf_acc}, F1 Score: {rf_f1}")
print(f"XGBoost - Accuracy: {xgb_acc}, F1 Score: {xgb_f1}")

# Save predictions
predictions_df = pd.DataFrame({
    "True Label": y_test,
    "SVM Prediction": svm_preds,
    "Random Forest Prediction": rf_preds,
    "XGBoost Prediction": xgb_preds
})
predictions_df.to_csv("model_predictions.csv", index=False)
print("\nSample Predictions:\n", predictions_df.head())
