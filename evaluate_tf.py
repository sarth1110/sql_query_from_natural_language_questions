import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import gzip
import json
import re
import copy
from ast import literal_eval
from stanfordcorenlp import StanfordCoreNLP
print("Modules imported.\n")

# nlp = StanfordCoreNLP('http://localhost', port=9000)

fptr = open('../data/WikiSQL/data/train.jsonl', 'r')
queries = fptr.readlines()

dev_fptr = open('../data/WikiSQL/data/dev.jsonl', 'r')
dev_queries = dev_fptr.readlines()

test_fptr = open('../data/WikiSQL/data/test.jsonl', 'r')
test_queries = test_fptr.readlines()

print("Extracting Data...")


trainTabId = []
trainQuestion = []
trainSQL = []
trainSel = []
trainCon = []
trainAgg = []
for i, query in enumerate(queries):
    q = json.loads(query)
    trainTabId.append(q["table_id"])
    trainQuestion.append(q["question"])
    trainSQL.append(q["sql"])
    trainSel.append(q["sql"]["sel"])
    trainCon.append(q["sql"]["conds"])
    trainAgg.append(q["sql"]["agg"])

devTabId = []
devQuestion = []
devSQL = []
devSel = []
devCon = []
devAgg = []
for i, query in enumerate(dev_queries):
    q = json.loads(query)
    devTabId.append(q["table_id"])
    devQuestion.append(q["question"])
    devSQL.append(q["sql"])
    devSel.append(q["sql"]["sel"])
    devCon.append(q["sql"]["conds"])
    devAgg.append(q["sql"]["agg"])

testTabId = []
testQuestion = []
testSQL = []
testSel = []
testCon = []
testAgg = []
for i, query in enumerate(test_queries):
    q = json.loads(query)
    testTabId.append(q["table_id"])
    testQuestion.append(q["question"])
    testSQL.append(q["sql"])
    testSel.append(q["sql"]["sel"])
    testCon.append(q["sql"]["conds"])
    testAgg.append(q["sql"]["agg"])

print("Data Extracted.")

def remove_puctuations(text):
    text = re.sub(r'[?]', '', text)
    return text

trainQuestion = np.array([j.lower() for j in trainQuestion])
devQuestion = np.array([j.lower() for j in devQuestion])
testQuestion = np.array([j.lower() for j in testQuestion])

ohe = sklearn.preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(trainAgg).reshape(-1, 1)).toarray()
# print(y_train)
y_val = ohe.transform(np.array(devAgg).reshape(-1, 1)).toarray()
y_test = ohe.transform(np.array(testAgg).reshape(-1, 1)).toarray()
# print(len(trainAgg))
# print(len(trainQuestion))

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="unk")
tokenizer.fit_on_texts(trainQuestion)
X_train = tokenizer.texts_to_sequences(trainQuestion)
X_val = tokenizer.texts_to_sequences(devQuestion)
X_test = tokenizer.texts_to_sequences(testQuestion)

max_len = max([len(x) for x in X_train])
print(max_len)
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train,maxlen=max_len,padding="post", truncating="post")
X_val_pad = tf.keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_len, padding="post", truncating="post")
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len, padding="post", truncating="post")
print(len(X_train_pad))

model = tf.keras.models.load_model("modelsAgg/agg_tf.h5")
print(model.summary())

y_pred_one_hot = model.predict(X_test_pad)
# print(y_pred_one_hot)
# Convert one-hot encoded predictions to class labels
y_pred = np.argmax(y_pred_one_hot, axis=1)
# print(y_pred)
# print(list(y_pred))
# for i in y_pred:
#     print(i)
print("Accuracy score for Aggregation function is : ",accuracy_score(testAgg, list(y_pred)))
