import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras


class BertClassifier:

    MODEL_FILE_PATH = './model/bert_model.pkl'
    EPOCHS = 100

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertModel.from_pretrained('bert-base-uncased')

    def pre_process_text(self, input):
        input_ids = tf.constant(self.tokenizer.encode(input, add_special_tokens=True, max_length=50, pad_to_max_length=True))[None, :]
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]
        reshaped = np.array(last_hidden_states).flatten()
        return reshaped

    def train(self, inputs, labels):
        self.nn_model = keras.Sequential([
            keras.layers.Flatten(input_shape=(38400, )),
            keras.layers.Dropout(0.7),
            keras.layers.Dense(2),
            keras.layers.Softmax()
        ])
        self.nn_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        self.nn_model.fit(inputs, labels, epochs=self.EPOCHS)
        keras.models.save_model(self.nn_model, self.MODEL_FILE_PATH)

    def evaluate(self, inputs, labels):
        prepared_model = keras.models.load_model(self.MODEL_FILE_PATH)
        if prepared_model:
            self.nn_model = prepared_model
        test_loss, test_acc = self.nn_model.evaluate(inputs, labels, verbose=2)
        return test_acc

    def predict(self, inputs):
        prepared_model = keras.models.load_model(self.MODEL_FILE_PATH)
        if prepared_model:
            self.nn_model = prepared_model
        predictions = self.nn_model.predict(inputs)
        return predictions
