import tensorflow as tf
import os
import keras
from data_preprocessing import preprocess_for_bert
from bert_model import build_bert_model, BERTForClassification
from utils import EMOTIONS_ID2LABEL, BERT_MODEL_NAME

TRAIN_SIZE = 10000
TEST_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_SAVE_PATH = "trained_bert_model.keras"
PREDICTION_THRESHOLD = 0.1

def train_and_evaluate_bert(
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    model_name=BERT_MODEL_NAME,
    save_path=MODEL_SAVE_PATH,
    force_retrain=False
):
    if os.path.exists(save_path) and not force_retrain:
        print(f"Loading existing trained model from {save_path}")
        try:
            classifier = keras.models.load_model(save_path, compile=False)
            classifier.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.AUC(multi_label=True, name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )
            print("Model loaded and recompiled.")
            _, _, _, _, _, _, test_labels_np, tokenizer = preprocess_for_bert(
                train_size, test_size, batch_size, model_name
            )
            _, tf_test_dataset, _, _, _, _, _, _ = preprocess_for_bert(
                train_size, test_size, batch_size, model_name
            )

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will force retraining...")
            force_retrain = True

    if not os.path.exists(save_path) or force_retrain:
        print("Preparing data and training a new BERT model...")
        tf_train_dataset, tf_test_dataset, _, _, _, _, test_labels_np, tokenizer = preprocess_for_bert(
            train_size, test_size, batch_size, model_name
        )

        classifier = build_bert_model(model_name, learning_rate=learning_rate)

        print(f"Starting training for {epochs} epochs...")
        history = classifier.fit(
            tf_train_dataset,
            epochs=epochs,
            validation_data=tf_test_dataset
        )
        print("Training finished.")
        print(f"Saving model to {save_path}")
        classifier.save(save_path)
        print("Model saved.")

    print("\nEvaluating model on test set...")
    results = classifier.evaluate(tf_test_dataset, verbose=1)
    print("\nTest Set Evaluation Results (BERT):")
    metric_results = {}
    for name, value in zip(classifier.metrics_names, results):
        print(f"- {name}: {value:.4f}")
        metric_results[name] = value

    print("\nGetting predictions for test set...")
    y_pred_proba = classifier.predict(tf_test_dataset)
    print(f"Predictions shape: {y_pred_proba.shape}")

    return classifier, tokenizer, test_labels_np, y_pred_proba, metric_results

def predict_emotions_bert(text_list, model, tokenizer, threshold=PREDICTION_THRESHOLD):
    # Tokenize sentences
    inputs = tokenizer(text_list, return_tensors="tf", padding=True, truncation=True, max_length=128)

    # Get predictions
    predictions = model(inputs)  # Use model call directly instead of predict
    probabilities = tf.sigmoid(predictions.logits).numpy()

    # Decode predictions
    results = []
    for i in range(len(text_list)):
        text = text_list[i]
        probas = probabilities[i]
        predicted_indices = tf.where(probas > threshold).numpy().flatten()

        predicted_emotions = []
        confidences = []

        if len(predicted_indices) > 0:
            for index in predicted_indices:
                predicted_emotions.append(EMOTIONS_ID2LABEL[index])
                confidences.append(float(probas[index]))
        else:
            highest_prob_index = tf.argmax(probas).numpy()
            predicted_emotions.append(EMOTIONS_ID2LABEL[highest_prob_index])
            confidences.append(float(probas[highest_prob_index]))

        results.append({
            'text': text,
            'emotions': predicted_emotions,
            'confidences': confidences
        })
    return results