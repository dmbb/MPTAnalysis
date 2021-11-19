import socket
import dpkt
import os
import tensorflow as tf
import csv
import numpy as np
import random
import math
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn import preprocessing
import time

from copy import deepcopy
from scipy import interp

np.random.seed(1)
graph_level_seed = 1
operation_level_seed = 1
tf.set_random_seed(graph_level_seed)
random.seed(1)

plt.rcParams['font.family'] = 'Helvetica'

def gatherDataset_january(data_folder, cfg, SPLIT_FACTOR):
    random.seed(1)
    #Load Datasets
    f = open(data_folder + cfg[0] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    reg = list(reader)

    f = open(data_folder + cfg[1] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    fac = list(reader)
    print("###########################################")
    print("Configuration " + cfg[1])
    print("###########################################")

    #Convert data to floats (and labels to integers)
    reg_data = []
    for i in reg[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(1)
        reg_data.append(int_array)

    fac_data = []
    for i in fac[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(0)
        fac_data.append(int_array)


    #Shuffle both datasets
    shuffled_reg_data = random.sample(reg_data, len(reg_data))
    shuffled_fac_data = random.sample(fac_data, len(fac_data))

    #Build label tensors
    reg_labels = []
    for i in shuffled_reg_data:
        reg_labels.append(int(i[len(reg_data[0])-1]))

    fac_labels = []
    for i in shuffled_fac_data:
        fac_labels.append(int(i[len(reg_data[0])-1]))

    #Take label out of data tensors
    for i in range(0, len(shuffled_reg_data)):
        shuffled_reg_data[i].pop()

    for i in range(0, len(shuffled_fac_data)):
        shuffled_fac_data[i].pop()


    #Build training and testing datasets
    #Split each class data in the appropriate proportion for training
    reg_proportion_index = int(len(reg_labels)* SPLIT_FACTOR)
    reg_train_x = shuffled_reg_data[:reg_proportion_index]
    reg_train_y = reg_labels[:reg_proportion_index]

    fac_proportion_index = int(len(fac_labels)*SPLIT_FACTOR)
    fac_train_x = shuffled_fac_data[:fac_proportion_index]
    fac_train_y = fac_labels[:fac_proportion_index]

    #Create training sets by simply using normal samples
    train_x = reg_train_x #+ fac_train_x
    train_y = reg_train_y #+ fac_train_y

    #Make the split for the testing data
    reg_test_x = shuffled_reg_data[reg_proportion_index:]
    reg_test_y = reg_labels[reg_proportion_index:]

    fac_test_x = shuffled_fac_data[fac_proportion_index:]
    fac_test_y = fac_labels[fac_proportion_index:]

    #Create testing set by combining the holdout samples
    test_x = reg_test_x + fac_test_x
    test_y = reg_test_y + fac_test_y

    return train_x, train_y, test_x, test_y, len(reg_data[0])

def gatherDataset_10times(data_folder, cfg, split_factor):
    random.seed(1)
    SPLIT_FACTOR = split_factor
    #Load Datasets
    f = open(data_folder + cfg[0] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    reg = list(reader)

    f = open(data_folder + cfg[1] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    fac = list(reader)
    print("###########################################")
    print("Configuration " + cfg[1])
    print("###########################################")


    #Convert data to floats (and labels to integers)
    reg_data = []
    for i in reg[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(0) #0, inliers
        reg_data.append(int_array)

    fac_data = []
    for i in fac[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(1) #1, outliers
        fac_data.append(int_array)

    train_x_t = []
    train_y_t = []
    test_x_t = []
    test_y_t = []

    for k in range(0,10):
        reg_data2 = deepcopy(reg_data)
        fac_data2 = deepcopy(fac_data)


        #Shuffle both datasets
        shuffled_reg_data = random.sample(reg_data2, len(reg_data2))
        shuffled_fac_data = random.sample(fac_data2, len(fac_data2))

        #Build label tensors
        reg_labels = []
        for i in shuffled_reg_data:
            reg_labels.append(int(i[len(reg_data2[0])-1]))

        fac_labels = []
        for i in shuffled_fac_data:
            fac_labels.append(int(i[len(reg_data2[0])-1]))

        #Take label out of data tensors
        for i in range(0, len(shuffled_reg_data)):
            shuffled_reg_data[i].pop()

        for i in range(0, len(shuffled_fac_data)):
            shuffled_fac_data[i].pop()


        #Build training and testing datasets
        #Split each class data in the appropriate proportion for training
        reg_proportion_index = int(len(reg_labels)* SPLIT_FACTOR)
        reg_train_x = shuffled_reg_data[:reg_proportion_index]
        reg_train_y = reg_labels[:reg_proportion_index]

        fac_proportion_index = int(len(fac_labels)*SPLIT_FACTOR)
        fac_train_x = shuffled_fac_data[:fac_proportion_index]
        fac_train_y = fac_labels[:fac_proportion_index]

        #Create training sets by combining the randomly selected samples from each class
        train_x = reg_train_x
        train_y = reg_train_y

        #Make the split for the testing data
        reg_test_x = shuffled_reg_data[reg_proportion_index:]
        reg_test_y = reg_labels[reg_proportion_index:]
        fac_test_x = shuffled_fac_data[fac_proportion_index:]
        fac_test_y = fac_labels[fac_proportion_index:]

        #Create testing set by combining the holdout samples
        test_x = reg_test_x + fac_test_x
        test_y = reg_test_y + fac_test_y

        train_x_t.append(train_x)
        train_y_t.append(train_y)
        test_x_t.append(test_x)
        test_y_t.append(test_y)

    return train_x_t, train_y_t, test_x_t, test_y_t, len(reg_data2[0])

class Encoder(object):
    def __init__(self, inp, n_features, n_hidden, drop_input, drop_hidden, repr_size):
        # inp is the placeholder for the input, n_features is the number of features our data has (21 in this example)
        # n_hidden is the size of the first hidden layer and repr_size is the dimensionality of the representation
        self.inp = inp
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W1 = tf.Variable(tf.random_normal([n_features, self.n_hidden], stddev=0.35))
        self.W2 = tf.Variable(tf.random_normal([self.n_hidden, repr_size], stddev=0.35))


        self.b1 = tf.Variable(tf.random_normal([self.n_hidden], stddev=0.35))
        self.b2 = tf.Variable(tf.random_normal([repr_size], stddev=0.35))

        self.layer_0 = tf.nn.dropout(self.inp, drop_input)
        self.layer_1 = tf.nn.relu(tf.matmul(self.layer_0, self.W1) + self.b1)
        self.layer_1 = tf.nn.dropout(self.layer_1, drop_hidden)
        self.encoder_out = tf.matmul(self.layer_1, self.W2) + self.b2


class Decoder(object):
    def __init__(self, inp, n_features, n_hidden, drop_input, drop_hidden, repr_size):
        self.inp = inp
        self.n_hidden = n_hidden
        self.W1 = tf.Variable(tf.random_normal([repr_size, self.n_hidden], stddev=0.35))
        self.W2 = tf.Variable(tf.random_normal([self.n_hidden, n_features], stddev=0.35))
        self.b1 = tf.Variable(tf.random_normal([self.n_hidden], stddev=0.35))
        self.b2 = tf.Variable(tf.random_normal([n_features], stddev=0.35))

        self.layer_0 = tf.nn.dropout(self.inp, drop_input)
        self.layer_1 = tf.nn.relu(tf.matmul(self.layer_0, self.W1) + self.b1)
        self.layer_1 = tf.nn.dropout(self.layer_1, drop_hidden)
        self.decoder_out = tf.matmul(self.layer_1, self.W2) + self.b2

class Autoencoder(object):
    def __init__(self, n_features, batch_size, n_hidden, drop_input, drop_hidden, repr_size, learning_rate):
        # n_features is the number of features our data has (21 in this example)
        # repr_size the dimensionality of our representation
        # n_hidden_1 is the size of the layers closest to the in and output
        # n_hidden_2 is the size of the layers closest to the embedding layer
        # batch_size number of samples to run per batch

        self.n_features = n_features
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.drop_input = drop_input
        self.hidden = drop_hidden
        self.repr_size = repr_size

        # Start session, placeholder has None in shape for batches
        self.sess = tf.Session()
        self.inp = tf.placeholder(tf.float32, [None, n_features])

        # Make the encoder and the decoder
        self.encoder = Encoder(self.inp, n_features, n_hidden, drop_input, drop_hidden, repr_size)
        self.decoder = Decoder(self.encoder.encoder_out, n_features, n_hidden, drop_input, drop_hidden, repr_size)

        # Loss function mean squared error and AdamOptimizer
        self.loss = tf.reduce_mean(tf.square(self.decoder.decoder_out - self.inp), -1)
        self.mean_loss = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.mean_loss)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

    def run_epoch(self, data_list):
        # Train once over the passed data_list and return the mean reconstruction loss after the epoch
        for index in range(len(data_list) // self.batch_size):
            self.sess.run(self.train_op, feed_dict={self.inp: data_list[index * self.batch_size : (index+1) * self.batch_size]})
        return self.sess.run(self.mean_loss, feed_dict={self.inp: data_list})

    def representations(self, data_list):
        # Return a list of representations for the given list of samples
        return self.sess.run(self.encoder.encoder_out, feed_dict={self.inp: data_list})

    def reconstruction_errors(self, data_list):
        # Get mean squared reconstruction errors of passed data_list
        return self.sess.run(self.loss, feed_dict={self.inp: data_list})


def runANN(data_folder,cfg):
    epochs = 1000
    #Gather the dataset
    #train_x, train_y are just regular samples
    train_x, train_y, test_x, test_y, num_input = gatherDataset_january(data_folder, cfg, 0.7)

    #std_scale = preprocessing.StandardScaler().fit(train_x)
    #train_x = std_scale.transform(train_x)
    #test_x = std_scale.transform(test_x)

    #n_features, batch_size, n_hidden, drop_input, drop_hidden, repr_size
    ae = Autoencoder(num_input, 128, 128, 0.8, 0.5, 32)
    for i in range(epochs):
        if(i%50==0):
            print("Epoch: " + str(i))
        ae.run_epoch(train_x)

    """
    #Show compressed representation of samples (valid for repr_size=2,3)
    anomaly_repr = ae.representations(test_x[len(test_x)/2:])
    normal_repr = ae.representations(test_x[:len(test_x)/2])
    anom_x, anom_y = zip(*anomaly_repr)
    norm_x, norm_y = zip(*normal_repr)
    plt.scatter(anom_x, anom_y, color='red', alpha=0.7)
    plt.scatter(norm_x, norm_y, alpha=0.7)
    plt.show()
    """

    #Reconstruct samples
    anomaly_errors = ae.reconstruction_errors(test_x[len(test_x)/2:])
    normal_val_errors = ae.reconstruction_errors(test_x[:len(test_x)/2])

    roc_y = [1 for _ in range(len(anomaly_errors))] + [0 for _ in range(len(normal_val_errors))]
    roc_score = np.concatenate([anomaly_errors, normal_val_errors])


    # Compute ROC curve and ROC area for each class
    #number of thresholds = number of data samples - default drop_intermediate
    # does not show some low performing configs for creating smoother ROCs

    fpr, tpr, thresholds = roc_curve(roc_y, roc_score, drop_intermediate=True)
    roc_auc = auc(fpr, tpr)


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def runANNSearch(data_folder,cfg):
    epochs = 100
    #Gather the dataset
    #train_x, train_y are just regular samples
    train_x_t, train_y_t, test_x_t, test_y_t, num_input = gatherDataset_10times(data_folder, cfg, 0.9)

    #std_scale = preprocessing.StandardScaler().fit(train_x)
    #train_x = std_scale.transform(train_x)
    #test_x = std_scale.transform(test_x)

    max_auc = 0
    max_batch_size = 0
    max_hidden = 0
    max_repr_size = 0

    auc_report = []
    n_hidden_report = []
    repr_size_report = []
    batch_sizes_report = []

    best_config = []
    max_auc = 0

    learning_rates = [0.001]  # [0.01, 0.001] # default is 0.001
    batch_sizes = [32]#[8, 16, 32, 64, 128, 256]
    n_hiddens = [8, 16, 32, 64, 128, 256]#np.logspace(2, 10, base=2, num=12)
    #drop_inputs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    #drop_hiddens = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    repr_sizes = [4, 8, 16, 32, 64, 128, 256] #np.logspace(2, 10, base=2, num=12) #num 20

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for n_hidden in n_hiddens:
                for repr_size in repr_sizes:
                    if(repr_size <= n_hidden):
                        #start = time.time()
                        np.random.seed(1)
                        graph_level_seed = 1
                        operation_level_seed = 1
                        tf.set_random_seed(graph_level_seed)
                        random.seed(1)

                        step_auc = []
                        mean_fpr = np.linspace(0, 1, 100)
                        tprs = []
                        for n in range(0,10):
                            #n_features, batch_size, n_hidden, drop_input, drop_hidden, repr_size
                            ae = Autoencoder(num_input, batch_size, int(n_hidden), 1, 1, int(repr_size), learning_rate)

                            train_x = train_x_t[n]
                            train_y = train_y_t[n]
                            test_x = test_x_t[n]
                            test_y = test_y_t[n]

                            for i in range(epochs):
                                ae.run_epoch(train_x)

                            #Reconstruct samples
                            anomaly_errors = ae.reconstruction_errors(test_x[len(test_x)/2:])
                            normal_val_errors = ae.reconstruction_errors(test_x[:len(test_x)/2])

                            roc_y = [1 for _ in range(len(anomaly_errors))] + [0 for _ in range(len(normal_val_errors))]
                            roc_score = np.concatenate([anomaly_errors, normal_val_errors])


                            # Compute ROC curve and ROC area for each class
                            fpr, tpr, thresholds = roc_curve(roc_y, roc_score, drop_intermediate=True)
                            tprs.append(interp(mean_fpr, fpr, tpr))
                            tprs[-1][0] = 0.0
                            roc_auc = auc(fpr, tpr)
                            #print "Fold %i auc: %f" % (n, roc_auc)
                            step_auc.append(roc_auc)

                        avg_auc = sum(step_auc)/float(len(step_auc))

                        auc_report.append(avg_auc)
                        """
                        n_hidden_report.append(int(n_hidden))
                        repr_size_report.append(int(repr_size))
                        batch_sizes_report.append(batch_size)
                        """
                        mean_tpr = np.mean(tprs, axis=0)
                        mean_tpr[-1] = 1.0
                        mean_auc = auc(mean_fpr, mean_tpr)

                        if(mean_auc > max_auc):
                            max_auc = mean_auc
                            best_config = [mean_fpr, mean_tpr, n_hidden, repr_size]

                        #end = time.time()
                        #print(end - start)
                        print(("%f - Batch Size:%i, Learning Rate:%f, n_hidden:%i, repr_size:%i" % (avg_auc, batch_size, learning_rate, int(n_hidden), int(repr_size))))


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)

    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='orange', label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)
    plt.plot(best_config[0], best_config[1], color='b', label=r'ROC (AUC = %0.2f)' % (max_auc), lw=2, alpha=.8)
    plt.legend(loc='lower right', fontsize='x-large')

    fig.savefig('Autoencoder/' + "Facet_Autoencoder_" + cfg[1] + ".pdf")   # save the figure to file
    plt.close(fig)

    print("################\n# Summary")
    print("Max. AUC: %f, N_hidden: %i, Repr_Size: %i" % (max_auc, best_config[2],best_config[3]))
    print("Avg. AUC %f: " % (np.mean(auc_report,axis=0)))
    """
    full_report = zip(auc_report, batch_sizes_report, n_hidden_report, repr_size_report)
    full_report.sort(key = lambda t: t[0])

    f = open(cfg[1] + '_report.txt', 'w')

    for item in full_report:
        f.write("%f - Batch Size:%i, n_hidden:%i, repr_size:%i\n" % (item[0], item[1], item[2], item[3]))
    np.save(cfg[1] + '_report', np.array(full_report))
    """


if __name__ == "__main__":

    cfgs = [
    ["RegularTraffic_Christmas",
    "FacetTraffic_12.5_Christmas"],
    ["RegularTraffic_Christmas",
    "FacetTraffic_25_Christmas"],
    ["RegularTraffic_Christmas",
    "FacetTraffic_50_Christmas"]]


    print("Autoencoder - Packet Length Features - Set2")
    feature_set = 'PL_60'
    data_folder = 'FeatureSets/' + feature_set + '/'

    for cfg in cfgs:
        runANNSearch(data_folder,cfg)
