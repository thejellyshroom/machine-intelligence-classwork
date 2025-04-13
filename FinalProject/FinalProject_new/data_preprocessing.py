import tensorflow as tf
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from utils import NUM_CLASSES, FEATURE_COLS, LABEL_COL, BERT_MODEL_NAME

DEFAULT_TRAIN_SIZE = 10000
DEFAULT_TEST_SIZE = 1000
DEFAULT_BATCH_SIZE = 32


def load_and_prepare_data(train_size=DEFAULT_TRAIN_SIZE, test_size=DEFAULT_TEST_SIZE):
    emotion_dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    print(f"Using dataset. Selecting {train_size} training examples and {test_size} test examples...")
    small_train_dataset = emotion_dataset['train'].select(range(train_size))
    small_test_dataset = emotion_dataset['test'].select(range(test_size))

    return small_train_dataset, small_test_dataset

def get_tokenizer(model_name=BERT_MODEL_NAME):
    print(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_data(dataset, tokenizer):
    print("Tokenizing dataset...")
    def tokenize_batch(batch): #oom issues
        return tokenizer(batch["text"], padding=True, truncation=True, return_tensors='tf')

    return dataset.map(tokenize_batch, batched=True, batch_size=None)

def create_multi_hot_labels_column(example):
    multi_hot_label = np.zeros(NUM_CLASSES, dtype=np.float32)
    for label_id in example['labels']:
        if isinstance(label_id, int) and 0 <= label_id < NUM_CLASSES:
                multi_hot_label[label_id] = 1.0
    example['multi_hot_labels'] = multi_hot_label
    return example

def process_labels(dataset):
    print("Processing labels...")
    dataset = dataset.map(create_multi_hot_labels_column)

    columns_to_remove = [col for col in ['label_int', 'labels'] if col in dataset.features]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    if 'multi_hot_labels' in dataset.features:
        dataset = dataset.rename_column('multi_hot_labels', LABEL_COL)
    else:
        print("Error, no label detected.")

    return dataset


def calculate_sample_weights(labels_np):
    print("Calculating sample weights...")
    num_samples = len(labels_np)
    label_counts = np.sum(labels_np, axis=0)

    class_weights_calc = {}
    for i in range(NUM_CLASSES):
        count = label_counts[i] if label_counts[i] > 0 else 1
        class_weights_calc[i] = num_samples / (NUM_CLASSES * count)

    sample_weights_np = np.zeros(num_samples, dtype=np.float32)
    for i in range(num_samples):
        sample_label_indices = np.where(labels_np[i] == 1.0)[0]
        if len(sample_label_indices) > 0:
            sample_weights_np[i] = max(class_weights_calc[idx] for idx in sample_label_indices)
        else:
            sample_weights_np[i] = 1.0
    print("Sample weights calculated.")
    return sample_weights_np

def create_tf_datasets(encoded_train_data, encoded_test_data, batch_size=DEFAULT_BATCH_SIZE):
    print("Creating TensorFlow datasets...")

    train_features_np = {col: np.array(encoded_train_data[col]) for col in FEATURE_COLS}
    train_labels_np = np.array(encoded_train_data[LABEL_COL])
    sample_weights_np = calculate_sample_weights(train_labels_np)

    test_features_np = {col: np.array(encoded_test_data[col]) for col in FEATURE_COLS}
    test_labels_np = np.array(encoded_test_data[LABEL_COL])

    required_cols = FEATURE_COLS + [LABEL_COL]
    if not all(col in encoded_train_data.features for col in required_cols):
         print(f"Missing columns in training data. Need: {required_cols}")
    if not all(col in encoded_test_data.features for col in required_cols):
         print(f"Missing columns in test data. Need: {required_cols}")


    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features_np, train_labels_np, sample_weights_np)
    )
    #shuffle = better for gradient
    train_dataset = train_dataset.shuffle(len(sample_weights_np)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_features_np, test_labels_np)
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("TensorFlow datasets created successfully.")
    return train_dataset, test_dataset, test_labels_np

def preprocess_for_bert(train_size=DEFAULT_TRAIN_SIZE, test_size=DEFAULT_TEST_SIZE, batch_size=DEFAULT_BATCH_SIZE, model_name=BERT_MODEL_NAME):
    raw_train_data, raw_test_data = load_and_prepare_data(train_size, test_size)
    tokenizer = get_tokenizer(model_name)

    train_texts = [item['text'] for item in raw_train_data]
    test_texts = [item['text'] for item in raw_test_data]

    encoded_train = tokenize_data(raw_train_data, tokenizer)
    encoded_test = tokenize_data(raw_test_data, tokenizer)

    processed_train = process_labels(encoded_train)
    processed_test = process_labels(encoded_test)

    tf_train_dataset, tf_test_dataset, test_labels_np = create_tf_datasets(
        processed_train, processed_test, batch_size
    )

    train_labels_raw = [item['labels'] for item in raw_train_data]
    test_labels_raw = [item['labels'] for item in raw_test_data]


    return tf_train_dataset, tf_test_dataset, train_texts, test_texts, train_labels_raw, test_labels_raw, test_labels_np, tokenizer