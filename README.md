# Relation Extraction for Scientific Texts

This repository contains code for extracting relationships between entities in scientific texts using two approaches:
1. **Traditional Machine Learning-Based Approach** (SVM, Random Forest, XGBoost)
2. **Deep Learning-Based Approach** (BERT Transformer Model)

## **Project Overview**
Relation extraction is a fundamental task in NLP that involves identifying and classifying semantic relationships between entities in text. This project applies **traditional machine learning models** and **deep learning transformer models** to extract relations from scientific literature.

## Requirements
* Python 3.7+
* Required Libraries:
  ```bash
  pip install pandas spacy numpy scikit-learn xgboost matplotlib seaborn
  python -m spacy download en_core_web_md
  ```

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   ```
2. Navigate to the project directory:
   ```bash
   cd yourproject
   ```
3. Install dependencies as mentioned above.

## **Methods**
We implement **two different approaches** for relation extraction:

### **1. Traditional Machine Learning-Based Approach**
We train and evaluate **three supervised models**:
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **XGBoost (XGB)**

These models use hand-crafted linguistic features such as:
- Named Entity Recognition (NER) tags
- Part-of-Speech (POS) tags
- Dependency parsing relations
- TF-IDF vectorized text features

📌 **Scripts:**  
- [Train SVM](train_svm.py)  
- [Train Random Forest](train_rf.py)  
- [Train XGBoost](train_xgb.py)  

### **2. Deep Learning-Based Approach (BERT Transformer Model)**
We implement a **BERT-based relation extraction model**, leveraging a transformer encoder to learn contextualized representations of entity pairs.

- **Pre-trained BERT embeddings** are fine-tuned on our dataset.
- We use **Hugging Face's `transformers` library** for model implementation.
- The BERT model predicts relation types given an input sentence.

📌 **Script:**  
- [Train BERT Model](train_bert.py)  

## **Installation**
Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Additionally, install the **spaCy model** and **Hugging Face Transformers**:
```bash
python -m spacy download en_core_web_md
pip install transformers
```

## **Data Preprocessing**
We extract entity pairs and linguistic features using **spaCy** NLP processing. The `preprocess.py` script performs:
- Tokenization
- Lemmatization
- Dependency parsing
- Feature extraction for machine learning models

📌 **Run data preprocessing:**
```bash
python preprocess.py
```

## **Model Training**
Train each model using the respective scripts:

### **Traditional ML Models**
```bash
python train_svm.py
python train_rf.py
python train_xgb.py
```

### **BERT Transformer Model**
```bash
python train_bert.py
```

## **Evaluation**
Evaluate the models using precision, recall, F1-score, and confusion matrices.

### **Traditional ML Models Evaluation**
```bash
python evaluate_ml_models.py
```

### **BERT Model Evaluation**
```bash
python evaluate_bert.py
```

## **Usage**
Once trained, models can be used to make predictions.

Example for **SVM**:
```python
import joblib
svm_model = joblib.load("svm_model.pkl")
prediction = svm_model.predict([input_features])
print("Predicted Relation:", prediction)
```

Example for **BERT**:
```python
from transformers import pipeline
nlp_model = pipeline("text-classification", model="bert_model")
result = nlp_model("The enzyme catalyzes the reaction.")
print(result)
```

## **Results**
| Model     | Precision | Recall | F1-Score |
|-----------|-----------|-----------|-----------|
| SVM       | 0.78      | 0.75      | 0.76      |
| Random Forest | 0.82 | 0.79 | 0.80 |
| XGBoost   | 0.84      | 0.81      | 0.82      |
| BERT      | **0.91**  | **0.88**  | **0.89**  |

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.

---

### **Contact**
For any queries, please reach out to the project contributors.
