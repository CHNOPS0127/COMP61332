import json
import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load spaCy model
nlp = spacy.load("en_core_web_md")
vectorizer = TfidfVectorizer(max_features=100)
label_encoders = {}
scaler = StandardScaler()

def feature_extraction(data): 
    """Extracts features from the input dataset."""
    structured_data = []
    
    for entry in data:
        text = entry["text"]
        entities = {e["entity_id"]: e for e in entry["entities"]}
        relations = entry["relations"]
    
        for relation in relations:
            entity1 = entities[relation["arg1"]]
            entity2 = entities[relation["arg2"]]
    
            entity1_type, entity2_type = entity1["label"], entity2["label"]
            between_text = text[entity1["end"]:entity2["start"]].strip() if entity1["end"] < entity2["start"] else text[entity2["end"]:entity1["start"]].strip()

            doc = nlp(text)
            entity1_root, entity2_root = None, None
            for token in doc:
                if token.text in entity1["text"].split():
                    entity1_root = token
                if token.text in entity2["text"].split():
                    entity2_root = token

            entity1_head_emb = entity1_root.vector if entity1_root else np.zeros(300)
            entity2_head_emb = entity2_root.vector if entity2_root else np.zeros(300)
    
            span_similarity = 1 - cosine(entity1_head_emb, entity2_head_emb) if np.linalg.norm(entity1_head_emb) != 0 and np.linalg.norm(entity2_head_emb) != 0 else 0
            dependency_path = f"{entity1_root.dep_} → {entity2_root.dep_}" if entity1_root and entity2_root else "None"
    
            entity1_POS = entity1_root.pos_ if entity1_root else "UNKNOWN"
            entity2_POS = entity2_root.pos_ if entity2_root else "UNKNOWN"

            structured_data.append({
                "sentence": text, 
                "entity1": entity1["text"], 
                "entity1_type": entity1_type,
                "entity2": entity2["text"], 
                "entity2_type": entity2_type,
                "text_between_entities": between_text, 
                "relation": relation["type"],
                "sentence_length": len(text.split()), 
                "word_distance": len(between_text.split()),
                "dependency_path": dependency_path, 
                "entity1_POS": entity1_POS, 
                "entity2_POS": entity2_POS,
                "span_similarity": span_similarity
            })
    
    return pd.DataFrame(structured_data)

def lemmatize_text(text):
    """Applies lemmatization to the input text."""
    return " ".join([token.lemma_ for token in nlp(text)])

def tf_idf(text, isTrain=False):
    """Computes TF-IDF features for input text."""
    text = text.apply(lemmatize_text)
    return pd.DataFrame(vectorizer.fit_transform(text).toarray() if isTrain else vectorizer.transform(text).toarray(), columns=[f"tfidf_{i}" for i in range(100)])

def label_encode(df, columns_to_encode, isTrain=False):
    """Encodes categorical features into numerical values."""
    for column in columns_to_encode:
        if isTrain:
            label_encoders[column] = LabelEncoder()
            df[column + "_encoded"] = label_encoders[column].fit_transform(df[column])
        else:
            df[column + "_encoded"] = df[column].apply(lambda x: label_encoders[column].transform([x])[0] if x in label_encoders[column].classes_ else -1)
    return df

def normalize_features(df, columns_to_normalise, isTrain=False):
    """Applies standard scaling to numerical features."""
    df[columns_to_normalise] = scaler.fit_transform(df[columns_to_normalise]) if isTrain else scaler.transform(df[columns_to_normalise])
    return df
