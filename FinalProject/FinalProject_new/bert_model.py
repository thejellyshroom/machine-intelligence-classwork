import tensorflow as tf
from transformers import TFAutoModel
import keras # Import Keras
from utils import NUM_CLASSES, BERT_MODEL_NAME

# Register the custom model class for saving/loading
@keras.saving.register_keras_serializable()
class BERTForClassification(keras.Model):
    def __init__(self, bert_model_name=BERT_MODEL_NAME, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.num_classes = num_classes
        print(f"Initializing BERT base model ({self.bert_model_name})...")
        self.bert = TFAutoModel.from_pretrained(self.bert_model_name)
        self.fc = keras.layers.Dense(self.num_classes, activation='sigmoid', name='classifier')

    def call(self, inputs):
        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "bert_model_name": self.bert_model_name,
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        #if custom objects are needed??
        return cls(**config)

def build_bert_model(model_name=BERT_MODEL_NAME, num_classes=NUM_CLASSES, learning_rate=2e-5):
    print("Building BERT classifier...")
    classifier = BERTForClassification(bert_model_name=model_name, num_classes=num_classes)

    classifier.compile(
        # optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        
        #needed for class imbalances (multip label)
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(multi_label=True, name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    print("BERT model built and compiled.")
    return classifier