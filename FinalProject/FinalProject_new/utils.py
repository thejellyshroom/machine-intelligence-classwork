NUM_CLASSES = 28

# Use the Hugging Face identifier for the pre-trained model
BERT_MODEL_NAME = "bert-base-uncased"

EMOTIONS_ID2LABEL = {
    0: 'admiration',
    1: 'amusement',
    2: 'anger',
    3: 'annoyance',
    4: 'approval',
    5: 'caring',
    6: 'confusion',
    7: 'curiosity',
    8: 'desire',
    9: 'disappointment',
    10: 'disapproval',
    11: 'disgust',
    12: 'embarrassment',
    13: 'excitement',
    14: 'fear',
    15: 'gratitude',
    16: 'grief',
    17: 'joy',
    18: 'love',
    19: 'nervousness',
    20: 'optimism',
    21: 'pride',
    22: 'realization',
    23: 'relief',
    24: 'remorse',
    25: 'sadness',
    26: 'surprise',
    27: 'neutral'
}

EMOTIONS_LABEL2ID = {v: k for k, v in EMOTIONS_ID2LABEL.items()}

# need to separate labels into another column because the data stored is a list
FEATURE_COLS = ["input_ids", "token_type_ids", "attention_mask"]
LABEL_COL = "labels" 