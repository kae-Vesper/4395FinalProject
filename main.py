from time import perf_counter
import torch
import torch.nn as nn
import json
import csv
import math
import string
from collections import defaultdict
from SVM import SupportVectorClassifier
from tdifd import LogisticRegressionModel
from BERT import BERTModel

unk = '<UNK>'

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

# After that we will load the data
def get_data():
    label_map = {'ham': 0, 'spam': 1}
    datasets = {}

    for name in ["training.json", "validation.json", "test.json"]:
        try:
            with open(f'./{name}', 'r') as file:
                datasets[name] = json.load(file)
        except FileNotFoundError:
            print(f"Error: {name} not found.")
            exit(1)

    tra, val, test = [], [], []

    for dataset_name, dataset in datasets.items():
        for elt in dataset:
            label = elt.get("label", "").strip().lower()
            if label in label_map:
                processed_message = preprocess(elt.get("message", ""))
                if processed_message:
                    entry = (processed_message, label_map[label])
                    if dataset_name == "training.json":
                        tra.append(entry)
                    elif dataset_name == "validation.json":
                        val.append(entry)
                    else:
                        test.append(entry)

    return tra, val, test

# Computing the TD-IDF
def compute_tf_idf(data):
    doc_count = len(data)
    word_doc_freq = defaultdict(int)

    for document, _ in data:
        unique_words = set(document)
        for word in unique_words:
            word_doc_freq[word] += 1

    vectors = []
    for document, y in data:
        vector = {}
        total_words = len(document)
        for word in document:
            tf = document.count(word) / total_words
            idf = math.log((1 + doc_count) / (1 + word_doc_freq[word])) + 1
            vector[word] = tf * idf
        vectors.append((vector, y))

    return vectors

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word

def convert_to_vector(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def convert_to_tf_vector(data, word2index, vocab):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(vocab))
        for word, score in document.items():
            if word in word2index:
                vector[word2index[word]] = score
        vectorized_data.append((vector, y))
    return vectorized_data

def test_model(name, model, test_data, test_data_vector, word2index):
    count = 0
    accurate = 0
    with open(f"./{name}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["message", "pred_label", "label", "confidence", "impactful words"])

        i=0
        for message, label in test_data_vector:
            pred_label, confidence, words = model.test(message,word2index)
            if pred_label == label:
                accurate += 1
            writer.writerow([test_data[count][0], pred_label, label, confidence, words])
            count += 1
    return accurate/count

if __name__ == '__main__':
    #hyperparameters
    hidden_dim = 256
    epochs = 2

    #load data
    print("========== Loading data ==========\n")
    train_data, valid_data, test_data = get_data()
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    """"#Compute TF-IDF
    print("========== Computing TF-TDF ==========")
    start_time = perf_counter()
    train_data_vector = compute_tf_idf(train_data)
    valid_data_vector = compute_tf_idf(valid_data)
    test_data_vector = compute_tf_idf(test_data)
    end_time = perf_counter()
    print(f"TF-IDF computation time: {end_time - start_time:.4f} seconds\n")

    #Vectorize data
    print("========== Vectorizing data for TF-IDF ==========")
    start_time = perf_counter()
    train_data_vector = convert_to_tf_vector(train_data_vector, word2index, vocab)
    valid_data_vector = convert_to_tf_vector(valid_data_vector, word2index, vocab)
    test_data_vector = convert_to_tf_vector(test_data_vector, word2index, vocab)
    end_time = perf_counter()
    print(f"Vectorization time: {end_time - start_time: .4f} seconds \n")

    #logical regression model
    print("========== Building logical regression model ==========\n")
    #initialize model
    lr_model = LogisticRegressionModel(len(vocab))

    #train
    print("---------- Training model ----------")
    start_time = perf_counter()
    print(f"Training data size: {len(train_data_vector)}")
    lr_model.train_model(train_data_vector, valid_data_vector, epochs)
    end_time = perf_counter()
    print(f"Training time:  {end_time-start_time:.4f} seconds\n")

    #test
    print("---------- Testing model -----------")
    start_time = perf_counter()
    test_acc = test_model("Logical_Regression_Model", lr_model, test_data, test_data_vector, word2index)
    end_time = perf_counter()
    print(f"Test accuracy: {test_acc:.2f}")
    print(f"Testing time:  {end_time-start_time:.4f} seconds\n")"""

    #SVM model
    print("========== Building SVM model ==========\n")
    # initialize model
    svm_model = SupportVectorClassifier()

    print("---------- Vectorizing data for SVM ----------")
    start_time = perf_counter()
    train_data_vector = convert_to_vector(train_data, word2index)
    valid_data_vector = convert_to_vector(valid_data, word2index)
    test_data_vector = convert_to_vector(test_data, word2index)
    end_time = perf_counter()
    print(f"Vectorization time: {end_time - start_time: .4f} seconds \n")

    # train
    print("---------- Training model ----------")
    start_time = perf_counter()
    train_acc, valid_acc = svm_model.fit(train_data_vector, valid_data_vector, epochs)
    end_time = perf_counter()
    print(f"Training time:  {end_time - start_time:.4f} seconds\n")

    # test
    print("---------- Testing model -----------")
    start_time = perf_counter()

    prediction, confidence, keywords = svm_model.predict(test_data_vector)
    count = 0
    accurate = 0
    with open("./svm.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["message", "pred_label", "label", "confidence"])
        for i in range(len(test_data_vector)):
            writer.writerow([test_data[i][0], prediction[i], test_data[i][1], confidence[i], keywords[i]])
            if prediction[i] == test_data[i][1]:
                accurate += 1
            count += 1
    end_time = perf_counter()
    print("Test accuracy: ", accurate / count)
    print(f"Testing time:  {end_time - start_time:.4f} seconds\n")

    #BERT model
    print("========== Building BERT model ==========\n")
    # initialize model
    bert_model = BERTModel()

    #vectorizing data for bert model
    print("---------- Vectorizing data for BERT ----------")
    start_time = perf_counter()
    train_data_vector = []
    valid_data_vector = []
    test_data_vector = []
    for document, y in train_data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        train_data_vector.append((vector, y))
    for document, y in valid_data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        valid_data_vector.append((vector, y))
    for document, y in test_data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        test_data_vector.append((vector, y))
    end_time = perf_counter()
    print(f"Vectorization time: {end_time - start_time:.4f} seconds\n")

    # train
    print("---------- Training model ----------")
    start_time = perf_counter()
    bert_model.train(train_data_vector, valid_data_vector, epochs)
    end_time = perf_counter()
    print(f"Training time:  {end_time - start_time:.4f} seconds\n")

    # test
    print("---------- Testing model -----------")
    start_time = perf_counter()
    test_acc = bert_model.test(test_data_vector)
    end_time = perf_counter()
    print("Test accuracy: ", test_acc)
    print(f"Testing time:  {end_time - start_time:.4f} seconds")