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
   git clone https://github.com/CHNOPS0127/COMP61332.git
   ```
2. Navigate to the project directory:
   ```bash
   cd COMP61332
   ```
3. Install dependencies as mentioned above.

## Dataset Preparation
* Data should be formatted in JSON format with entity relations.
* Sample dataset is provided in `dataset.json`.

## **Methods**
We implement **two different approaches** for relation extraction:

### **1. Traditional Machine Learning-Based Approach**
We train and evaluate **three supervised models**:
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **XGBoost (XGB)**

### Hand-Crafted Linguistic Features
These models leverage **hand-crafted linguistic features** to enhance relation extraction performance. The extracted features include:
- **Named Entity Recognition (NER) Tags**  
  - Encoded entity types (`entity1_type_encoded`, `entity2_type_encoded`) help distinguish the type of entities (e.g., "Gene", "Protein", "Chemical").  
- **Part-of-Speech (POS) Tags**  
  - Features `entity1_POS_encoded` and `entity2_POS_encoded` represent the **grammatical category** of each entity (e.g., noun, verb, adjective).  
- **Dependency Parsing Relations**  
  - `dependency_path_encoded` captures **syntactic relationships** between entities in a sentence.  
- **Sentence-Level Structural Features**  
  - `sentence_length`: The number of words in the sentence, providing insight into the complexity of the context.  
  - `word_distance`: The number of words between two entities, useful for assessing how closely related they are.  
  - `span_similarity`: Measures how semantically similar the two entity spans are based on word embeddings.
- **TF-IDF Vectorized Text Features**  
  - Features `tfidf_0` to `tfidf_100` represent **word importance scores** based on Term Frequency-Inverse Document Frequency (TF-IDF).  

### **2. Deep Learning-Based Approach (BERT Transformer Model)**
We implement a **BERT-based relation extraction model**, leveraging a transformer encoder to learn contextualized representations of entity pairs.

- xxx

## **Data Preprocessing**
We extract entity pairs and linguistic features using **spaCy** NLP processing. The `preprocess.py` script performs:
- Tokenization
- Lemmatization
- Dependency parsing
- Feature extraction for machine learning models

**Run data preprocessing:**
```bash
python preprocess.py
```

## **Model Training**
Train each model using the respective scripts:

### **Traditional ML Models**
```bash
python svm_model.py
python rf_model.py
python xgb_model.py
```

### **BERT Transformer Model**
```bash
bert.py
```

## **Evaluation**
Evaluate the models using precision, recall, F1-score, and confusion matrices.

### **Traditional ML Models Evaluation**
```bash
evaluate_ml_models.py
```

### **BERT Model Evaluation**
```bash
evaluate_bert.py
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

## **License**
This project is licensed under the MIT License. 
