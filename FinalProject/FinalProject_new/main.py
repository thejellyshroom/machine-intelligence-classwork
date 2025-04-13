import time
from data_preprocessing import preprocess_for_bert
from train_bert import train_and_evaluate_bert, predict_emotions_bert
from sklearn_models import train_and_evaluate_sklearn_model, predict_emotions_sklearn
from evaluate import plot_roc_curves

# Default configuration values
DEFAULT_TRAIN_SIZE = 10000
DEFAULT_TEST_SIZE = 1000
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_LR = 2e-5

def run_example_predictions(bert_model, bert_tok, mnb_model, mnb_vec, mnb_bin, lr_model, lr_vec, lr_bin):
    print("\n--- Running Example Predictions ---")
    test_sentences = [
        "I'm so happy today!",
        "This makes me really angry.",
        "I'm feeling very sad and disappointed.",
        "That's really interesting, tell me more.",
        "I am both excited and nervous about the presentation.",
    ]

    print("\nExample Predictions (BERT):")
    bert_preds = predict_emotions_bert(test_sentences, bert_model, bert_tok)
    for p in bert_preds:
        print(f"  Text: {p['text']}")
        print(f"    Predicted: {p['emotions']} ({[f'{c:.2f}' for c in p['confidences']]})")

    print("\nExample Predictions (MNB):")
    mnb_preds = predict_emotions_sklearn(test_sentences, mnb_model, mnb_vec, mnb_bin, threshold=0.1)
    for p in mnb_preds:
        print(f"  Text: {p['text']}")
        print(f"    Predicted: {p['emotions']} ({[f'{c:.2f}' for c in p['confidences']]})")

    print("\nExample Predictions (LR):")
    lr_preds = predict_emotions_sklearn(test_sentences, lr_model, lr_vec, lr_bin, threshold=0.2)
    for p in lr_preds:
        print(f"  Text: {p['text']}")
        print(f"    Predicted: {p['emotions']} ({[f'{c:.2f}' for c in p['confidences']]})")

def main():
    start_time = time.time()
    print("\n Load and preprocess")
    _, tf_test_ds, train_texts, test_texts, train_labels_raw, test_labels_raw, test_labels_bert_np, bert_tokenizer = preprocess_for_bert(
        train_size=DEFAULT_TRAIN_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        model_name="bert-base-uncased"
    )
    print("Data preparation complete.")

    print("\n BERT model")
    bert_classifier, _, _, y_pred_proba_bert, bert_metrics = train_and_evaluate_bert(
        train_size=DEFAULT_TRAIN_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR,
        force_retrain=False 
    )
    bert_roc_auc = plot_roc_curves(test_labels_bert_np, y_pred_proba_bert, "BERT")

    print("\n Scikit models")
    mnb_pipeline, mnb_vectorizer, mnb_mlb, y_test_bin_mnb, y_pred_proba_mnb, mnb_metrics = train_and_evaluate_sklearn_model(
        model_type='mnb',
        X_train=train_texts,
        y_train_raw=train_labels_raw,
        X_test=test_texts,
        y_test_raw=test_labels_raw,
        force_retrain=False 
    )
    mnb_roc_auc = plot_roc_curves(y_test_bin_mnb, y_pred_proba_mnb, "MultinomialNB")
    print("MNB processing complete.")

    lr_pipeline, lr_vectorizer, lr_mlb, y_test_bin_lr, y_pred_proba_lr, lr_metrics = train_and_evaluate_sklearn_model(
        model_type='lr',
        X_train=train_texts,
        y_train_raw=train_labels_raw,
        X_test=test_texts,
        y_test_raw=test_labels_raw,
        force_retrain=False  # Do not activate force retrain
    )
    lr_roc_auc = plot_roc_curves(y_test_bin_lr, y_pred_proba_lr, "LogisticRegression")
    print("LR processing complete.")

    print("\n Summary")
    print(f"BERT Micro AUC: {bert_roc_auc.get('micro', 'N/A'):.4f}, Macro AUC: {bert_roc_auc.get('macro', 'N/A'):.4f}")
    print(f"MNB  Micro AUC: {mnb_roc_auc.get('micro', 'N/A'):.4f}, Macro AUC: {mnb_roc_auc.get('macro', 'N/A'):.4f}")
    print(f"LR   Micro AUC: {lr_roc_auc.get('micro', 'N/A'):.4f}, Macro AUC: {lr_roc_auc.get('macro', 'N/A'):.4f}")

    # Run example predictions
    run_example_predictions(
        bert_classifier, bert_tokenizer,
        mnb_pipeline, mnb_vectorizer, mnb_mlb,
        lr_pipeline, lr_vectorizer, lr_mlb
    )

if __name__ == "__main__":
    main() 