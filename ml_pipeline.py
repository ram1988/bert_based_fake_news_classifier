import pandas as pd
import numpy as np
import pickle
from BertClassifier import BertClassifier

classifier = BertClassifier()

def load_sentences(file_path, load_train_data=True):
    df = pd.read_csv(file_path)
    print(df)
    return df["text"], (df["target"] if "target" in df else None), (df["id"] if not load_train_data and "id" in df else None)

def prepare_train_dataset():
    train_text, train_labels, id = load_sentences("./data/train.csv")
    print("the total recs-->")
    print(len(train_text))
    encoded_train_dataset = []

    for i, text in enumerate(train_text):
        print(i)
        encoded = classifier.pre_process_text(text)
        encoded_train_dataset.append(encoded)

    encoded_train_dataset = np.array(encoded_train_dataset)
    train_labels = np.array(train_labels)

    prepared_train_dataset = [encoded_train_dataset, train_labels]
    pickle.dump(prepared_train_dataset, open("prepared_train_dataset.pkl", "wb"))
    return encoded_train_dataset, train_labels

def train_and_evaluate():
    # prepare_train_dataset()
    prepared_train_dataset = pickle.load(open("prepared_train_dataset.pkl","rb"))
    train_data = prepared_train_dataset[0]
    train_labels = prepared_train_dataset[1]
    total_len = len(train_data)
    # 80-20 Train/Validation dataset Split
    total_eval_len = int(0.2 * total_len)

    train_records = train_data[0:total_len]
    train_record_labels = train_labels[0:total_len]

    eval_records = train_data[total_eval_len:]
    eval_record_labels = train_labels[total_eval_len:]

    classifier.train(train_records, train_record_labels)
    accuracy = classifier.evaluate(eval_records, eval_record_labels)
    print(accuracy)

def predict_outcome(text):
    encoded = classifier.pre_process_text(text)
    encoded = np.array([encoded])
    result = classifier.predict(encoded)
    result = result[0]
    result_0 = result[0]
    result_1 = result[1]
    predicted = 1 if result_1 > result_0 else 0
    return predicted

def predict_outcomes():
    test_texts, test_labels, test_id = load_sentences("./data/test.csv", False)
    with open('outcome.csv', 'a') as outcome_file:
        outcome_file.write("id,target")
        for i, test_text in enumerate(test_texts):
            print(test_id[i])
            predicted = predict_outcome(test_text)
            outcome_file.write(str(test_id[i]) + "," + str(predicted) + "\n")


# Label Meaning: 1=original news, 0=fake news
# prepare the dataset compatible to the Model training
prepare_train_dataset()
# train and evaluate the model
train_and_evaluate()

# verification
op = predict_outcome("this is a fake text...")
print(op)
op = predict_outcome('"Kansas City, MO",Gunmen kill four in El Salvador bus attack: Suspected Salvadoran gang members killed four people and wounded s... http://t.co/LZWRONbTmi"')
print(op)

# generate the predictions for the test set
#predict_outcomes()
