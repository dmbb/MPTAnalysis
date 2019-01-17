import socket
import dpkt
import os
import csv
import numpy as np
import random
import math
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from copy import deepcopy
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

plt.rcParams['font.family'] = 'Helvetica'

random.seed(42)
rng = np.random.RandomState(42)

def gatherHoldoutData(data_folder, cfg):

    SPLIT_FACTOR = 0.7
    #Load Datasets
    f = open(data_folder + cfg[0] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    reg = list(reader)

    f = open(data_folder + cfg[1] + "_dataset.csv", 'r')
    reader = csv.reader(f, delimiter=',')
    fac = list(reader)


    #Convert data to floats (and labels to integers)
    reg_data = []
    for i in reg[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(1) #0, inliers
        reg_data.append(int_array)

    fac_data = []
    for i in fac[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(-1) #1, outliers
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

    return train_x, train_y, test_x, test_y

def gatherHoldoutData_10times(data_folder, cfg, split_factor):
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

def runIsolationSearch(data_folder, cfg, cnt_factor):


    max_acc = 0
    max_tree = 0

    for n, t in enumerate(range(10,500,10)):
        print t
        acc = 0
        tnr = 0
        fnr = 0
        tpr = 0
        fpr = 0
        ppv = 0
        npv = 0
        for i in range(0,3):
            #Gather the dataset
            train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg)

            clf = IsolationForest(n_estimators=int(t), random_state=rng, max_features=1.0, contamination=cnt_factor)

            # fit the model
            cnt_train = int(math.ceil(cnt_factor * (len(train_x)/2)))
            clf.fit(train_x[:(len(train_x)/2) + cnt_train])

            #make predictions on testing data
            cnt_test = int(math.ceil(cnt_factor * (len(test_x)/2)))
            #y_true, y_pred = test_y[:(len(test_x)/2) + cnt_test], clf.predict(test_x[:(len(test_x)/2) + cnt_test])

            y_true, y_pred = test_y, clf.predict(test_x)

            #print(roc_auc_score(y_true, -clf.decision_function(test_x[:(len(test_x)/2) + cnt_test])))

            eps = 0.0000000001
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

            accuracy = (TPositives + TNegatives)/float((len(test_x)))
            TNR = TNegatives/(TNegatives+float(FPositives)+eps)
            FNR = FNegatives/(TPositives+float(FNegatives))
            TPR = TPositives/(TPositives+float(FNegatives))
            FPR = FPositives/(FPositives+float(TNegatives)+eps)
            PPV = TPositives/(TPositives+float(FPositives))
            NPV = TNegatives/(TNegatives+float(FNegatives)+eps)

            acc+=accuracy
            tnr+=TNR
            fnr+=FNR
            tpr+=TPR
            fpr+=FPR
            ppv+=PPV
            npv+=NPV

        ac = acc/3
        if(int(t)%100 == 0):
            print "100 trees = " + str(ac)
        if(ac > max_acc):
            max_acc = ac
            max_tree = int(t)
    print max_acc
    print max_tree

def runIsolationRounds(data_folder, cfg, cnt_factor):

    acc = 0
    tnr = 0
    fnr = 0
    tpr = 0
    fpr = 0
    ppv = 0
    npv = 0

    for i in range(0,10):
        #Gather the dataset
        train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg)

        clf = IsolationForest(n_estimators=100, random_state=rng, max_features=1.0, contamination=cnt_factor)

        # fit the model
        cnt_train = int(math.ceil(cnt_factor * (len(train_x)/2)))
        clf.fit(train_x[:(len(train_x)/2) + cnt_train])

        #make predictions on testing data
        cnt_test = int(math.ceil(cnt_factor * (len(test_x)/2)))
        #y_true, y_pred = test_y[:(len(test_x)/2) + cnt_test], clf.predict(test_x[:(len(test_x)/2) + cnt_test])

        y_true, y_pred = test_y, clf.predict(test_x)

        #print(roc_auc_score(y_true, -clf.decision_function(test_x[:(len(test_x)/2) + cnt_test])))

        eps = 0.0000000001
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

        accuracy = (TPositives + TNegatives)/float((len(test_x)))
        TNR = TNegatives/(TNegatives+float(FPositives)+eps)
        FNR = FNegatives/(TPositives+float(FNegatives))
        TPR = TPositives/(TPositives+float(FNegatives))
        FPR = FPositives/(FPositives+float(TNegatives)+eps)
        PPV = TPositives/(TPositives+float(FPositives))
        NPV = TNegatives/(TNegatives+float(FNegatives)+eps)

        acc+=accuracy
        tnr+=TNR
        fnr+=FNR
        tpr+=TPR
        fpr+=FPR
        ppv+=PPV
        npv+=NPV


    print "Acc = " + str(acc/10)
    print "TPR = " + str(tpr/10)
    print "TNR = " + str(tnr/10)
    print "FPR = " + str(fpr/10)
    print "FNR = " + str(fnr/10)
    print "PPV = " + str(ppv/10)
    print "NPV = " + str(npv/10)

def runIsolation(data_folder, cfg, cnt_factor):
    rng = np.random.RandomState(42)
    #Gather the dataset
    train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg)

    clf = IsolationForest(n_estimators=100,random_state=rng, bootstrap=True, max_features=1.0, contamination=cnt_factor)

    # fit the model
    cnt_train = int(math.ceil(cnt_factor * (len(train_x)/2)))
    clf.fit(train_x[:(len(train_x)/2) + cnt_train])

    #make predictions on testing data
    cnt_test = int(math.ceil(cnt_factor * (len(test_x)/2)))
    y_true, y_pred = test_y[:(len(test_x)/2) + cnt_test], clf.predict(test_x[:(len(test_x)/2) + cnt_test])

    #y_true, y_pred = test_y, clf.predict(test_x)

    #print(roc_auc_score(y_true, -clf.decision_function(test_x[:(len(test_x)/2) + cnt_test])))

    eps = 0.0000000001
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

    accuracy = (TPositives + TNegatives)/float((len(test_x)/2) + cnt_test)
    TNR = TNegatives/(TNegatives+float(FPositives)+eps)
    FNR = FNegatives/(TPositives+float(FNegatives))
    TPR = TPositives/(TPositives+float(FNegatives))
    FPR = FPositives/(FPositives+float(TNegatives)+eps)
    PPV = TPositives/(TPositives+float(FPositives))
    NPV = TNegatives/(TNegatives+float(FNegatives)+eps)
    print "Acc = " + str(accuracy)
    print "TPR = " + str(TPR)
    print "TNR = " + str(TNR)
    print "FPR = " + str(FPR)
    print "FNR = " + str(FNR)
    print "PPV = " + str(PPV)
    print "NPV = " + str(NPV)

def runOptimizedIso_CV(data_folder, cfg):
    train_X, train_Y, test_X, test_Y = gatherHoldoutData_10times(data_folder, cfg, 0.9)

    estimators = [50, 100, 200] #np.linspace(0.1, 1, 10)
    samples=[64, 128, 256, 512]
    cnt_factors = [0]

    auc_report = []
    best_config = []
    max_auc = 0
    for estimator in estimators:
        for s in samples:
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                for n in range(0,10):
                    train_x = train_X[n]
                    train_y = train_Y[n]
                    test_x = test_X[n]
                    test_y = test_Y[n]

                    rng = np.random.RandomState(2)
                    clf = IsolationForest(n_estimators=estimator, max_samples=s, random_state=rng, bootstrap=True, max_features=1.0, contamination=0.5)
                    clf.fit(train_x)

                    #make predictions on testing data
                    y_true, y_pred = test_y, clf.predict(test_x)
                    #print y_pred
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
                    best_config = [mean_fpr, mean_tpr, estimator,s]
                print ("%f - estimator:%i, max-samples: %i" % (mean_auc, estimator, s))

    print "################\n# Summary"
    print "Max. AUC: %f, Estimator: %i, Samples: %i" % (max_auc, best_config[2],best_config[3])
    print "Avg. AUC: %f, " % (np.mean(auc_report,axis=0))
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

    fig.savefig('Isolation/' + "DeltaShaper_Isolation_" + cfg[1] + ".pdf")   # save the figure to file
    plt.close(fig)


if __name__ == "__main__":

    cfgs = [
    ["RegularTraffic",
    "DeltaShaperTraffic_320"],
    ["RegularTraffic",
    "DeltaShaperTraffic_160"]]

    if not os.path.exists('Isolation'):
            os.makedirs('Isolation')

    print "Isolation Forest - Summary Statistic Features - Set1"
    feature_set = 'Stats_60' #'Stats_60' / 'PL_60'
    data_folder = 'FeatureSets/' + feature_set + '/'
    if not os.path.exists('Isolation/' + feature_set):
                os.makedirs('Isolation/' + feature_set)


    for cfg in cfgs:
        runOptimizedIso_CV(data_folder,cfg)
    print "#####################################\n"

    print "Isolation Forest - Packet Length Features - Set2"
    feature_set = 'PL_60' #'Stats_60' / 'PL_60'
    data_folder = 'FeatureSets/' + feature_set + '/'
    if not os.path.exists('Isolation/' + feature_set):
                os.makedirs('Isolation/' + feature_set)

    for cfg in cfgs:
        runOptimizedIso_CV(data_folder,cfg)

