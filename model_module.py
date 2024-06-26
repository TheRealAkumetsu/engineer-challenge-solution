import tensorflow as tf
from transformers import RobertaTokenizer, RobertaConfig, TFRobertaModel
import numpy as np


class RobertaModelWrapper:
    def __init__(self, model_path, tokenizer_name):
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.MAX_LEN = 96  # Define MAX_LEN in the initializer
        self.model = self._load_model(model_path)
        self.sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

    def _load_model(self, model_path):
        model = self._build_model()
        model.load_weights(model_path)
        return model

    def _build_model(self):
        ids = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)

        config = RobertaConfig.from_pretrained('roberta-base')
        bert_model = TFRobertaModel.from_pretrained('roberta-base', config=config)
        x = bert_model(ids, attention_mask=att)[0]

        x1 = tf.keras.layers.Dropout(0.1)(x)
        x1 = tf.keras.layers.Conv1D(1, 1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x)
        x2 = tf.keras.layers.Conv1D(1, 1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        model = tf.keras.models.Model(inputs=[ids, att], outputs=[x1, x2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

        return model

    def predict(self, text, sentiment):
        print(f"Debug: Original Text - {text}")
        print(f"Debug: Sentiment - {sentiment}")

        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True)
        input_ids = np.ones((1, self.MAX_LEN), dtype='int32')
        attention_mask = np.zeros((1, self.MAX_LEN), dtype='int32')

        input_ids[0, :len(encoded['input_ids'])] = encoded['input_ids']
        attention_mask[0, :len(encoded['attention_mask'])] = encoded['attention_mask']

        print(f"Debug: Input IDs - {input_ids}")
        print(f"Debug: Attention Mask - {attention_mask}")

        preds = self.model.predict([input_ids, attention_mask])
        start_idx = np.argmax(preds[0], axis=1)[0]
        end_idx = np.argmax(preds[1], axis=1)[0]

        print(f"Debug: Start Index - {start_idx}")
        print(f"Debug: End Index - {end_idx}")

        if start_idx > end_idx:
            print("Debug: start_idx > end_idx")
            return text
        else:
            selected_text = self.tokenizer.decode(input_ids[0][start_idx:end_idx + 1])
            print(f"Debug: Selected Text - {selected_text}")
            return selected_text