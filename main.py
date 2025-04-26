import numpy as np
import torch

unk = '<UNK>'

if __name__ == '__main__':
    #hyperparameters
    hidden_dim = 256
    epochs = 10

    #load data
    print("========== Loading data ==========\n")
    train_data, valid_data, test_data = get_data()
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    #Vectorize data
    print("========== Vectorizing data ==========\n")
    train_data = convert_to_vector(train_data, word2index)
    valid_data = convert_to_vector(valid_data, word2index)
    test_data = convert_to_vector(test_data, word2index)

    #logical regression model
    print("========== Building logical regression model ==========\n")
    #initialize model
    lr_model = LogisticRegressionModel()

    #train
    print("---------- Training model ----------")
    train_acc, valid_acc = lr_model.train(train_data, valid_data, epochs, hidden_dim)
    print("Train accuracy: ", train_acc)
    print("Validation accuracy: ", valid_acc)

    #test
    print("\n---------- Testing model -----------\n")
    test_acc, test_loss = lr_model.test(test_data, hidden_dim)
    print("Test accuracy: ", test_acc)

    #Naive Bayes model
    ##TODO: write code for training and testing a logical regression model with TF-IDF
    ##let me know what functions are needed to be included in the class for the model



    #BERT model
    ##TODO: write code for training and testing a logical regression model with TF-IDF
    ##let me know what functions are needed to be included in the class for the model
