import socket
import dpkt
import os
import csv
import numpy as np
import random
import math

import matplotlib.pyplot as plt

from copy import deepcopy

from scipy import interp
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


plt.rcParams['font.family'] = 'Helvetica'


def gatherHoldoutData(data_folder, cfg, split_factor):
    random.seed(1)
    SPLIT_FACTOR = split_factor
    #Load Datasets
    f = open(data_folder + cfg[0] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    reg = list(reader)

    f = open(data_folder + cfg[1] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    fac = list(reader)
    print "###########################################"
    print "Configuration " + cfg[1]
    print "###########################################"


    #Convert data to floats (and labels to integers)
    reg_data = []
    for i in reg[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(-1) #0, inliers
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
        train_x = reg_train_x + fac_train_x
        train_y = reg_train_y + fac_train_y

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

    return train_x_t, train_y_t, test_x_t, test_y_t

def gatherAllData(data_folder, cfg):
    #Load Datasets
    f = open(data_folder + cfg[0] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    reg = list(reader)

    f = open(data_folder + cfg[1] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    fac = list(reader)
    print "###########################################"
    print "Configuration " + cfg[1]
    print "###########################################"

    #Convert data to floats (and labels to integers)
    reg_data = []
    for i in reg[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(0)
        reg_data.append(int_array)

    fac_data = []
    for i in fac[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(1)
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

    #Create training sets by combining the randomly selected samples from each class
    train_x = shuffled_reg_data + shuffled_fac_data
    train_y = reg_labels + fac_labels

    #Shuffle positive/negative samples for CV purposes
    x_shuf = []
    y_shuf = []
    index_shuf = range(len(train_x))
    shuffle(index_shuf)
    for i in index_shuf:
        x_shuf.append(train_x[i])
        y_shuf.append(train_y[i])

    return x_shuf, y_shuf

def preprocessData(train_x, test_x, scaling):
    """
    pca = PCA(n_components=10)
    train_x = pca.fit_transform(train_x[:len(train_x)/2])
    train_x = list(train_x) + list(train_x)
    test_x = pca.transform(test_x)
    #print pca.variances_
    """
    if("soft_scaling" in scaling):
        # Soft scaling - each feature's Gaussian distribution is centered around 0
        #and it's standard deviation is 1. Subtracts the mean of the values and
        #divides by twice the standard deviation

        # PreProcess training samples
        std_scale = preprocessing.StandardScaler().fit(train_x[:len(train_x)/2])
        train_x = std_scale.transform(train_x[:len(train_x)/2])

        # PreProcess test samples
        test_x = std_scale.transform(test_x)

    elif("minmax_scaling" in scaling):
        # MinMax scaling - maps the min and max values of a given feature to 0 and 1

        # PreProcess training samples
        minmax_scale = preprocessing.MinMaxScaler().fit(train_x[:len(train_x)/2])
        train_x = minmax_scale.transform(train_x[:len(train_x)/2])

        # PreProcess test samples
        test_x = minmax_scale.transform(test_x)
    elif("default" in scaling):
        train_x = train_x[:len(train_x)/2]

    return train_x, test_x

def runWoodOCSVM(data_folder, cfg):
    #Gather the dataset
    #train_x[:len(train_x)/2] = (training) 70% of reg samples
    #train_x[len(train_x)/2:] = (training) 70% of facet samples
    #test_x[:len(test_x)/2] = (testing) 30% of reg samples
    #test_x[len(test_x)/2:] = (testing) 30% of facet samples
    train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg)

    # PreProcess data
    train_x, test_x = preprocessData(train_x, test_x, "minmax_scaling")


    # fit the model
    clf = OneClassSVM(nu=0.1, gamma='auto', kernel="rbf")

    #fit the model to labeled normal data
    clf.fit(train_x)

    #make predictions on testing data
    pred_reg_test = clf.predict(test_x[:len(test_x)/2])
    pred_anomaly_test = clf.predict(test_x[len(test_x)/2:])

    pred_reg_test_outliers = sum(1 for i in pred_reg_test if i < 0)
    pred_anomaly_test_outliers = sum(1 for i in pred_anomaly_test if i > 0)

    print pred_reg_test
    print pred_anomaly_test
    print "Error rate (Normal Data): " + str(pred_reg_test_outliers/(len(test_x)/2.0))
    print "Error rate (Anomalous Data): " + str(pred_anomaly_test_outliers/(len(test_x)/2.0))

#OCSVM grid search has a problem: Parameters are selected under some metric
# This metric is interestingly defined for 1+ classes, not so for just one.
# For instance, maximizing recall will bring inside the decision boundary the majority
#  of normal samples, but also include a large part of anomalies (already tested)
def runOCSVMGridSearch(data_folder, cfg):
    #Gather the dataset
    #train_x[:len(train_x)/2] = (training) 70% of reg samples
    #train_x[len(train_x)/2:] = (training) 70% of facet samples
    #test_x[:len(test_x)/2] = (testing) 30% of reg samples
    #test_x[len(test_x)/2:] = (testing) 30% of facet samples
    train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg)

    # PreProcess data
    train_x, test_x = preprocessData(train_x, test_x, "soft_scaling")

    parameters = {'nu' : np.linspace(0.01, 1, 100)}


    # fit the model
    #clf = OneClassSVM(nu=0.95 * 0.5 + 0.05, kernel="rbf")
    clf = GridSearchCV(OneClassSVM(kernel = "rbf"), parameters, cv=5, scoring='recall')
    clf.fit(train_x, train_y[:len(train_y)/2]) #fit to normal samples only

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on training set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Classification results on test set:")
    y_true, y_pred = test_y, clf.predict(test_x)
    print y_true
    print y_pred
    print(classification_report(y_true, y_pred))
    print accuracy_score(y_true, y_pred)
    print("accuracy: ", accuracy_score(y_true, y_pred))
    print("precision: ", precision_score(y_true, y_pred))
    print("recall: ", recall_score(y_true, y_pred))
    print("area under curve (auc): ", roc_auc_score(y_true, y_pred))

#OCSVM search over the nu parameter. Since the optimal choice will depend on the
# results of classification with a test set, it can no longer be deemed semi-supervised.
#Still here for possible utility purposes.
def runOCSVMSearch(data_folder, cfg):
    #Gather the dataset
    #train_x[:len(train_x)/2] = (training) 70% of reg samples
    #train_x[len(train_x)/2:] = (training) 70% of facet samples
    #test_x[:len(test_x)/2] = (testing) 30% of reg samples
    #test_x[len(test_x)/2:] = (testing) 30% of facet samples
    train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg, 0.5)

    # PreProcess data
    train_x, test_x = preprocessData(train_x, test_x, "soft_scaling")

    nus = np.linspace(0.01, 1, 100)

    max_acc = 0
    max_nu_acc = 0
    max_roc = 0
    max_nu_roc = 0

    # fit to models
    for n, nu in enumerate(nus):

        clf = OneClassSVM(nu=nu, gamma='auto', kernel="rbf")

        #fit the model to labeled normal data
        clf.fit(train_x)

        #make predictions on testing data
        #print("Classification results on test set:")
        y_true, y_pred = test_y, clf.predict(test_x)
        if(False):
            #print(classification_report(y_true, y_pred))
            print("accuracy: ", accuracy_score(y_true, y_pred))
            print("precision: ", precision_score(y_true, y_pred))
            print("recall: ", recall_score(y_true, y_pred))
            print("area under curve (auc): ", roc_auc_score(y_true, y_pred))

        acc = accuracy_score(y_true, y_pred)
        if(acc > max_acc):
            max_acc = acc
            max_nu_acc = nu

        #roc_auc = roc_auc_score(y_true, y_pred)
        #if(roc_auc > max_roc):
        #    max_roc = roc_auc
        #    max_nu_roc = nu

        print "Iter " + str(n) + ", Nu = " + str(nu)

    print "Max acc: " + str(max_acc) + ", Nu = " + str(max_nu_acc)
    #print "Max ROC: " + str(max_roc) + ", Nu = " + str(max_nu_roc)

    return max_nu_acc

#OCSVM running with a particular nu value. Fits a model to training data and
# reports classification results for testing data.
def runOptimizedOCSVM(data_folder, cfg, max_nu_roc):
    train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg, 0.5)

    # PreProcess data
    train_x, test_x = preprocessData(train_x, test_x, "soft_scaling")

    clf = OneClassSVM(nu=max_nu_roc, gamma='auto', kernel="rbf")

    #fit the model to labeled normal data
    clf.fit(train_x)

    #make predictions on testing data
    y_true, y_pred = test_y, clf.predict(test_x)

    FPositives = 0
    FNegatives = 0
    TPositives = 0
    TNegatives = 0

    for n, lbl in enumerate(y_pred):
        if(lbl == -1 and y_true[n] == -1):
            TNegatives += 1
        elif(lbl == 1 and y_true[n] == -1):
            FPositives += 1
        elif(lbl == -1 and y_true[n] == 1):
            FNegatives += 1
        elif(lbl == 1 and y_true[n] == 1):
            TPositives += 1

    accuracy = (TPositives + TNegatives)/float(len(test_x))
    TNR = TNegatives/(TNegatives+float(FPositives))
    FNR = FNegatives/(TPositives+float(FNegatives))
    TPR = TPositives/(TPositives+float(FNegatives))
    FPR = FPositives/(FPositives+float(TNegatives))
    PPV = TPositives/(TPositives+float(FPositives))
    NPV = TNegatives/(TNegatives+float(FNegatives))

    print "Acc = " + str(accuracy)
    print "TPR = " + str(TPR)
    print "TNR = " + str(TNR)
    print "FPR = " + str(FPR)
    print "FNR = " + str(FNR)
    print "PPV = " + str(PPV)
    print "NPV = " + str(NPV)

    #print(classification_report(y_true, y_pred))
    labels=[-1,1]
    print "Confusion Matrix\n"
    print(confusion_matrix(y_true, y_pred, labels))
    #print("accuracy: ", accuracy_score(y_true, y_pred))
    #print("precision: ", precision_score(y_true, y_pred))
    #print("recall: ", recall_score(y_true, y_pred))


    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    print thresholds

    roc_auc = auc(fpr, tpr)
    print roc_auc

    #Figure properties
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize='x-large')
    plt.legend(loc='lower right', fontsize='large')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='orange', label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)

    plt.plot(fpr, tpr, color='b', label=r'ROC curve (AUC = %0.2f)' % (roc_auc), lw=2, alpha=.8)
    plt.show()
    """


def runOptimizedOCSVM_CV(data_folder, cfg):
    train_X, train_Y, test_X, test_Y = gatherHoldoutData(data_folder, cfg, 0.9)

    nus = np.linspace(0.1, 1, 10)
    gammas = np.linspace(0.001, 1, 100)


    auc_report = []
    best_config = []
    max_auc = 0
    for nu in nus:
        for gamma in gammas:
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            for n in range(0,10):
                train_x = train_X[n]
                train_y = train_Y[n]
                test_x = test_X[n]
                test_y = test_Y[n]

                # PreProcess data
                train_x, test_x = preprocessData(train_x, test_x, "default")

                clf = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf")
                #fit the model to labeled normal data
                clf.fit(train_x)

                #make predictions on testing data
                y_true, y_pred = test_y, clf.predict(test_x) # For percentage of outliers in testing set [:(len(test_x)/2)+int((len(test_x)/2)*0.1)]

                for n ,l in enumerate(y_pred):
                    if(l==1):
                        y_pred[n] = -1
                    elif(l==-1):
                        y_pred[n] = 1

                fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=True,pos_label=1)
                #print y_true
                #print y_pred
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

                roc_auc = auc(fpr, tpr)
                #print "Fold %i auc: %f" % (n, roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            auc_report.append(mean_auc)

            if(mean_auc > max_auc):
                max_auc = mean_auc
                best_config = [mean_fpr, mean_tpr, nu, gamma]
            print ("%f - nu:%f, gamma:%f" % (mean_auc, nu, gamma))

    """
    best_config = []
    max_auc = 0

    for nu in nus:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for n in range(0,10):
            train_x = train_X[n]
            train_y = train_Y[n]
            test_x = test_X[n]
            test_y = test_Y[n]

            # PreProcess data
            train_x, test_x = preprocessData(train_x, test_x, "default")

            clf = OneClassSVM(nu=nu, kernel="linear")
            #fit the model to labeled normal data
            clf.fit(train_x)

            #make predictions on testing data
            y_true, y_pred = test_y, clf.predict(test_x) # For percentage of outliers in testing set [:(len(test_x)/2)+int((len(test_x)/2)*0.1)]

            for n ,l in enumerate(y_pred):
                if(l==1):
                    y_pred[n] = -1
                elif(l==-1):
                    y_pred[n] = 1

            fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=True,pos_label=1)
            #print y_true
            #print y_pred
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

            roc_auc = auc(fpr, tpr)
            #print "Fold %i auc: %f" % (n, roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print mean_auc

        if(mean_auc > max_auc):
            max_auc = mean_auc
            best_config = [mean_fpr, mean_tpr]
        print ("%f - nu:%f" % (mean_auc, nu))
    """

    print "################\n# Summary"
    print "Max. AUC: %f, Nu: %f, Gamma: %f" % (max_auc, best_config[2],best_config[3])
    print "Avg. AUC: %f " % (np.mean(auc_report,axis=0))
    #Figure properties

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

    fig.savefig('OCSVM/' + "Facet_OCSVM_" + cfg[1] + ".pdf")   # save the figure to file
    plt.close(fig)


if __name__ == "__main__":
    
    cfgs = [
        ["RegularTraffic_Christmas",
        "FacetTraffic_12.5_Christmas"],
        ["RegularTraffic_Christmas",
        "FacetTraffic_25_Christmas"],
        ["RegularTraffic_Christmas",
        "FacetTraffic_50_Christmas"]]


    feature_set = 'PL_60'
    data_folder = 'FeatureSets/' + feature_set + '/'

    print "\n#####################################"
    print "One-class SVM - Packet Length Features - Set2"
    for cfg in cfgs:
        print "One-class SVM"
        runOptimizedOCSVM_CV(data_folder, cfg)