import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

from utils import NUM_CLASSES, EMOTIONS_ID2LABEL

MNB_MODEL_PATH = "trained_mnb_model.joblib"
LR_MODEL_PATH = "trained_lr_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

def train_and_evaluate_sklearn_model(
    model_type,
    X_train, y_train_raw,
    X_test, y_test_raw,
    force_retrain=False
):
    model_save_path = MNB_MODEL_PATH if model_type == 'mnb' else LR_MODEL_PATH
    model_name = "Multinomial Naive Bayes" if model_type == 'mnb' else "Logistic Regression"

    print(f"\n--- {model_name} ---")

    mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))
    y_train_bin = mlb.fit_transform(y_train_raw)
    y_test_bin = mlb.transform(y_test_raw)
    print(f"Label shapes: Train={y_train_bin.shape}, Test={y_test_bin.shape}")

    if os.path.exists(VECTORIZER_PATH) and not force_retrain:
        vectorizer = joblib.load(VECTORIZER_PATH)
        X_train_tfidf = vectorizer.transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
    else:
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        print(f"Saving TF-IDF vectorizer to {VECTORIZER_PATH}")
        joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"TF-IDF shapes: Train={X_train_tfidf.shape}, Test={X_test_tfidf.shape}")

    if os.path.exists(model_save_path) and not force_retrain:
        pipeline = joblib.load(model_save_path)
    else:
        if model_type == 'mnb':
            base_classifier = MultinomialNB()
        elif model_type == 'lr':
            base_classifier = LogisticRegression(solver='liblinear',
                                                 max_iter=1000, class_weight='balanced',
                                                 random_state=42)

        multi_label_classifier = OneVsRestClassifier(base_classifier)
        pipeline = multi_label_classifier

        pipeline.fit(X_train_tfidf, y_train_bin)
        print(f"Saving {model_name} model to {model_save_path}")
        joblib.dump(pipeline, model_save_path)
        print("Model training complete and saved.")

    print(f"Evaluating model...")
    y_pred_bin = pipeline.predict(X_test_tfidf)
    y_pred_proba = pipeline.predict_proba(X_test_tfidf)
    print(f"Probabilities shape: {y_pred_proba.shape}")

    accuracy = accuracy_score(y_test_bin, y_pred_bin)

    precision_micro = precision_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)
    recall_micro = recall_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)
    f1_micro = f1_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)

    precision_macro = precision_score(y_test_bin, y_pred_bin, average='macro', zero_division=0)
    recall_macro = recall_score(y_test_bin, y_pred_bin, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_bin, y_pred_bin, average='macro', zero_division=0)

    roc_auc_micro = roc_auc_score(y_test_bin, y_pred_proba, average='micro', multi_class='ovr')
    roc_auc_macro = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')


    print(f"\nTest Set Evaluation Results ({model_name}):")
    print(f"- Accuracy (Exact Match): {accuracy:.4f}")
    print(f"- Precision (Micro): {precision_micro:.4f}")
    print(f"- Recall (Micro):    {recall_micro:.4f}")
    print(f"- F1-Score (Micro):  {f1_micro:.4f}")
    print(f"- Precision (Macro): {precision_macro:.4f}")
    print(f"- Recall (Macro):    {recall_macro:.4f}")
    print(f"- F1-Score (Macro):  {f1_macro:.4f}")
    print(f"- ROC AUC (Micro):   {roc_auc_micro:.4f}")
    print(f"- ROC AUC (Macro):   {roc_auc_macro:.4f}")

    metric_results = {
        'accuracy': accuracy,
        'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro,
        'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro,
        'roc_auc_micro': roc_auc_micro, 'roc_auc_macro': roc_auc_macro
    }

    return pipeline, vectorizer, mlb, y_test_bin, y_pred_proba, metric_results

def predict_emotions_sklearn(text_list, model_pipeline, vectorizer, label_binarizer, threshold=0.5):
    X_tfidf = vectorizer.transform(text_list)

    try:
        predictions_proba = model_pipeline.predict_proba(X_tfidf)
        predictions_bin = (predictions_proba >= threshold).astype(int)
    except AttributeError:
        predictions_bin = model_pipeline.predict(X_tfidf)
        predictions_proba = predictions_bin.astype(float)

    results = []
    for i in range(len(text_list)):
        text = text_list[i]
        pred_indices = np.where(predictions_bin[i] == 1)[0]

        predicted_emotions = []
        confidences = []

        if len(pred_indices) > 0:
            predicted_emotions = [EMOTIONS_ID2LABEL[label_binarizer.classes_[idx]] for idx in pred_indices]
            # handle cases where predict_proba failed
            try:
                confidences = [float(predictions_proba[i, idx]) for idx in pred_indices]
            except IndexError:
                 # assign dummy
                 confidences = [1.0] * len(predicted_emotions)
        else:
            if hasattr(model_pipeline, 'predict_proba') and predictions_proba.shape == predictions_bin.shape: # Check if proba exists and is valid
                highest_prob_index = np.argmax(predictions_proba[i])
                predicted_emotions.append(EMOTIONS_ID2LABEL[label_binarizer.classes_[highest_prob_index]])
                confidences.append(float(predictions_proba[i, highest_prob_index]))
            else:
                predicted_emotions.append("unknown")
                confidences.append(0.0)

        results.append({
            'text': text,
            'emotions': predicted_emotions,
            'confidences': confidences
        })
    return results