import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from scipy import interp
import random
from random import shuffle
import math
#Classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#Eval Metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score

np.random.seed(1)
random.seed(1)


def gatherHoldoutData(data_folder, cfg):
    SPLIT_FACTOR = 0.7
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


def runXGBoost(data_folder, cfg):
    #Gather the dataset
    print "Gather dataset"
    train_x, train_y, test_x, test_y = gatherHoldoutData(data_folder, cfg)


    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'


    model = XGBClassifier()
    model.fit(np.asarray(train_x), np.asarray(train_y))

    y_pred = model.predict(np.asarray(test_x))
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(np.asarray(test_y), predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    y_pred = model.predict_proba(np.asarray(test_x))[:,1]
    print 'Area under ROC:', roc_auc_score(np.asarray(test_y),y_pred)


def runClassification_CV(data_folder,cfg,classifier):
    print "Gather dataset"
    train_x, train_y= gatherAllData(data_folder, cfg)

    model = classifier[0]
    clf_name = classifier[1]

    #Report Cross-Validation Accuracy
    scores = cross_val_score(model, np.asarray(train_x), np.asarray(train_y), cv=10)
    print clf_name
    print "Avg. Accuracy: " + str(sum(scores)/float(len(scores)))

    cv = KFold(n_splits=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)


    #Split the data in k-folds, perform classification, and report ROC
    i = 0
    for train, test in cv.split(train_x, train_y):
        probas_ = model.fit(np.asarray(train_x)[train], np.asarray(train_y)[train]).predict_proba(np.asarray(train_x)[test])

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(np.asarray(train_y)[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    unblock70 = True
    unblock80 = True
    unblock90 = True
    unblock95 = True
    for n, i in enumerate(mean_tpr):
        if(i >= 0.7 and unblock70):
            print '70%  TPR  = ' + str(mean_fpr[n])
            unblock70 = False
        if(i >= 0.8 and unblock80):
            print '80%  TPR  = ' + str(mean_fpr[n])
            unblock80 = False
        if(i >= 0.9 and unblock90):
            print '90%  TPR  = ' + str(mean_fpr[n])
            unblock90 = False
        if(i >= 0.95 and unblock95):
            print '95%  TPR  = ' + str(mean_fpr[n])
            unblock95 = False

    #Figure properties
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    #Compute Standard Deviation between folds
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'$\pm$ ROC Std. Dev.')



    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='orange', label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')

    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize='x-large')
    plt.legend(loc='lower right', fontsize='large')

    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)

    fig.savefig('xgBoost/' + "ROC_" + clf_name + "_" + cfg[1] + ".pdf")   # save the figure to file
    plt.close(fig)

if __name__ == "__main__":
    data_folder = 'TrafficCaptures/'

    cfgs = [
    ["YouTube_home_world_live",
    "CovertCast_home_world"]]


    classifiers = [
    [RandomForestClassifier(n_estimators=100, max_features=None), "RandomForest"],
    [DecisionTreeClassifier(), "Decision Tree"],
    [XGBClassifier(),"XGBoost"]
    ]


    if not os.path.exists('xgBoost'):
                os.makedirs('xgBoost')

    for cfg in cfgs:
        for classifier in classifiers:
            print "Running classifiers for " + cfg[0] + " and " + cfg[1]
            runClassification_CV(data_folder, cfg, classifier)
