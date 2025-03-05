# Relation Extraction for Scientific Texts

This repository contains code for extracting relationships between entities in scientific texts using two approaches:
1. **Traditional Machine Learning-Based Approach** (SVM, Random Forest, XGBoost)
2. **Deep Learning-Based Approach** (BERT Transformer Model)

## **Project Overview**
Relation extraction is a fundamental task in NLP that involves identifying and classifying semantic relationships between entities in text. This project applies **traditional machine learning models** and **deep learning transformer models** to extract relations from scientific literature.

## Requirements
* Python 3.10+
* Required Packages:
  - pandas
  - numpy
  - spacy
  - sklearn
  - torch
  - transformers (HuggingFace)

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
* Example Format
```json
[
  {
    "filename": "E85-1004",
    "text": "This paper reports a completed stage of ongoing research at the University of York...",
    "entities": [
      {
        "entity_id": "T1",
        "label": "Method",
        "start": 112,
        "end": 131,
        "text": "analytical inverses"
      },
      {
        "entity_id": "T2",
        "label": "OtherScientificTerm",
        "start": 138,
        "end": 164,
        "text": "compositional syntax rules"
      }
    ],
    "relations": [
      {
        "relation_id": "R1",
        "type": "USED-FOR",
        "arg1": "T1",
        "arg2": "T2"
      }
    ]
  }
]
```
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
  - Encoded entity types (`entity1_type_encoded`, `entity2_type_encoded`) help distinguish the type of entities (e.g., "Method", "OtherScientificTerm", "Generic").  
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
The code implements a pretrained transformer approach (using BERT-base) for relation extraction. It fine-tunes BERT with an enhanced classifier head, incorporating techniques such as focal loss, class weighting, and stratified sampling to tackle class imbalance. Hyperparameters like a low learning rate, dropout, weight decay, and gradient accumulation are carefully tuned to ensure stable training and effective generalization. Evaluation metrics such as macro F1, precision, and recall are tracked during training to monitor performance and trigger early stopping.

## **Data Preprocessing**
First, We extract sentences with entity tags from the .json file. Then we tokenize the sentences using BertTokenizer.

## **Model Training**

### **Traditional ML Models**
_____ ""

### **BERT Transformer Model**
dl_bert_training.ipynb

### **Traditional ML Models Evaluation**
Evaluate the models using precision, recall, F1-score, and confusion matrices.

### **BERT Model Evaluation**
For evaluating the training of the model, we created a custom compute function which calculates various F1 scores(macro, weighted) and F1 scores for each relation type. This was chosen as an option since the relation classes are highly imbalanced and accuracy did not demonstrate improvements in learning.

## **Usage**
Once trained, models can be used to make predictions as shown in the notebooks.

## **License**
This project is licensed under the MIT License. 
