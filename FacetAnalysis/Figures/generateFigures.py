#!/usr/bin/env python
import os
import math
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import collections
import random


threshold_folder = '../EMD/TrafficCaptures/240Resolution/'
xclassifier_folder = '../X2/TrafficCaptures/240Resolution/'
xgboost_folder = '../xgBoost/'

facet_samples_folder = '../auxFolder/TrafficCaptures/240Resolution/'

colors = ["0.7", "0.3", "0.0"]
linestyle= ['k-', 'k.-','k*-']
plt.rcParams['font.family'] = 'Helvetica'

def genSimilarityGraph():
    cfgs = [12.5,25,50]
    for n, cfg in enumerate(cfgs):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(right=0.96, top=0.98)
        ################################
        # Plot EMD + X classifier ROC curve
        ################################
        EMD_sensitivity = np.load(threshold_folder + 'FacetTraffic_' + str(cfg) + '_Christmas/Rate_50_Sensitivity.npy')
        EMD_specificity = np.load(threshold_folder + 'FacetTraffic_' + str(cfg) + '_Christmas/Rate_50_Specificity.npy')

        EMD_auc = np.trapz(np.array(EMD_sensitivity)[::-1], (np.array(EMD_specificity))[::-1])
        print("EMD AUC" + str(EMD_auc))

        ax1.plot(1 - EMD_specificity, EMD_sensitivity, linestyle[0], lw=2, color=colors[0], label = 'EMD ROC (AUC = %0.2f)' % (EMD_auc))


        X_sensitivity = np.load(xclassifier_folder + 'ROC_True_FacetTraffic_'+ str(cfg) + '_Christmas_20_Sensitivity.npy')
        X_specificity = np.load(xclassifier_folder + 'ROC_True_FacetTraffic_'+ str(cfg) + '_Christmas_20_Specificity.npy')

        X_auc = np.trapz(np.array(X_sensitivity)[::-1], (1-np.array(X_specificity))[::-1])
        print("X AUC" + str(X_auc))

        ax1.plot(1 - X_specificity, X_sensitivity, linestyle[0], lw=2, color=colors[1], label = 'Chi-Square ROC (AUC = %0.2f)' % (X_auc))


        ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
        ax1.grid(color='black', linestyle='dotted')
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        plt.legend(loc='lower right', fontsize=18)

        plt.setp(ax1.get_xticklabels(), fontsize=16)
        plt.setp(ax1.get_yticklabels(), fontsize=16)

        fig.savefig('Facet_SimilarityGraph_'+ str(cfg) +'.pdf')   # save the figure to file
        plt.close(fig)


def genCompareStatsGraph():
    cfgs = [12.5,25,50]

    ################################
    # Plot DecisionTrees ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96, top=0.98)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'Stats_60/ROC_DecisionTree_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'Stats_60/ROC_DecisionTree_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'Sum. Stats., s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc='lower right', fontsize=18)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    fig.savefig('Facet_DecisionTree_CompareStats.pdf')   # save the figure to file
    plt.close(fig)


    ################################
    # Plot RandomForest ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96, top=0.98)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'Stats_60/ROC_RandomForest_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'Stats_60/ROC_RandomForest_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'Sum. Stats., s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc='lower right', fontsize=18)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    fig.savefig('Facet_RandomForest_CompareStats.pdf')   # save the figure to file
    plt.close(fig)

    ################################
    # Plot XGBoost ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96, top=0.98)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'Stats_60/ROC_XGBoost_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'Stats_60/ROC_XGBoost_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'Sum. Stats., s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc='lower right', fontsize=18)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    fig.savefig('Facet_XGBoost_CompareStats.pdf')   # save the figure to file
    plt.close(fig)

def genCompareHistGraph():
    cfgs = [12.5,25,50]

    ################################
    # Plot DecisionTrees ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96, top=0.98)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'PL_60/ROC_DecisionTree_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'PL_60/ROC_DecisionTree_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'K=5, s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc='lower right', fontsize=18)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    fig.savefig('Facet_DecisionTree_CompareHist.pdf')   # save the figure to file
    plt.close(fig)


    ################################
    # Plot RandomForest ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96, top=0.98)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'PL_60/ROC_RandomForest_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'PL_60/ROC_RandomForest_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'K=5, s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc='lower right', fontsize=18)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    fig.savefig('Facet_RandomForest_CompareHist.pdf')   # save the figure to file
    plt.close(fig)

    ################################
    # Plot XGBoost ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96, top=0.98)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'PL_60/ROC_XGBoost_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'PL_60/ROC_XGBoost_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'K=5, s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc='lower right', fontsize=18)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    fig.savefig('Facet_XGBoost_CompareHist.pdf')   # save the figure to file
    plt.close(fig)


def genFeatureImportance():
    max_features = 20
    cfgs = [12.5, 25, 50]
    classifiers = ["DecisionTree", "RandomForest", "xgBoost"]
    feature_sets = ["PL_60", "Stats_60"]

    for feature in feature_sets:

        for classifier in classifiers:

            top_f = []
            for n, cfg in enumerate(cfgs):
                f_imp = np.load(xgboost_folder + feature + '/FeatureImportance_' + classifier + '_FacetTraffic_' + str(cfg) + '_Christmas.npy')

                num, imp, name = np.hsplit(f_imp, 3)
                num = [int(item) for sublist in num for item in sublist]
                if("bigrams20" is feature):
                    num = np.asarray(num)
                else:
                    num = np.asarray(num) * 5 # Corresponding packet length

                imp = [float(item) for sublist in imp for item in sublist]
                name = [item for sublist in name for item in sublist]

                fig = plt.figure(figsize=(7,8))
                ax1 = fig.add_subplot(111)
                plt.subplots_adjust(right=0.96, top=0.98)
                ax1.barh(list(range(0,len(imp[:max_features]))), imp[:max_features], align='center', height=0.5, color="gray")

                plt.xlabel('Feature Importance Score', fontsize=25)
                if(feature is "Stats_60"):
                    plt.ylabel('Feature', fontsize=25)
                else:
                    plt.ylabel('Feature (Packet Length Bin)', fontsize=25)


                y_pos = np.arange(len(imp[:max_features]))
                ax1.set_yticks(y_pos)
                ax1.invert_yaxis()
                plt.margins(0.02)

                #Adjust number margin
                plt.xlim(xmin=0,xmax=0.12)

                #Adjust tick sizes
                plt.setp(ax1.get_xticklabels(), fontsize=18)
                plt.setp(ax1.get_yticklabels(), fontsize=18)


                if("PL_60" is feature):
                    plt.subplots_adjust(left=0.14)
                    ax1.set_yticklabels(num[:max_features])
                    #Gather statistics
                    print("Feature %s, Configuration %s, Classifier %s" % (feature, cfg, classifier))
                    print("Useful bins %s" % (sum(i > 0 for i in imp)))
                    plt.xlim(xmin=0,xmax=0.10)
                    top_f.append(num[:max_features])

                elif("Stats_60" is feature):
                    selected_names = name[:max_features]
                    print(selected_names)

                    proper_names = []
                    for k, sn in enumerate(selected_names):
                        if(sn.endswith("Out")):
                            proper_names.append(selected_names[k][:-3] + " (Out)")
                        elif(sn.endswith("In")):
                            proper_names.append(selected_names[k][:-2] + " (In)")
                        elif(sn.startswith(" in_")):
                            proper_names.append(selected_names[k][4:] + " (In)")
                        elif(sn.startswith(" out_")):
                            proper_names.append(selected_names[k][5:] + " (Out)")
                        else:
                            proper_names.append(sn)

                    ax1.set_yticklabels(proper_names)
                    start, end = ax1.get_xlim()
                    ax1.xaxis.set_ticks(np.arange(start, end+0.001, 0.02))

                    plt.xlim(xmin=0,xmax=0.10)
                    #Adjust space on the left
                    plt.subplots_adjust(left=0.465)
                    #Gather statistics
                    top_f.append(proper_names)
                    print("Feature %s, Configuration %s, Classifier %s" % (feature, cfg, classifier))
                    print("Useful stats %s" % (sum(i > 0 for i in imp)))
                elif("2sidedPL_60" is feature):
                    selected_features = num[:max_features]
                    selected_names = name[:max_features]

                    proper_names = []
                    for k, sn in enumerate(selected_names):
                        if("Bin2" in sn):
                            proper_names.append(str(selected_features[k]) + " (In)")
                        else:
                            proper_names.append(str(selected_features[k]) + " (Out)")
                    ax1.set_yticklabels(proper_names)
                    #Gather statistics
                    print("Feature %s, Configuration %s, Classifier %s" % (feature, cfg, classifier))
                    print("Useful bins %s" % (sum(i > 0 for i in imp)))

                elif("bigrams20" is feature):
                    ax1.set_yticklabels(name[:max_features])
                    #Gather statistics
                    print("Feature %s, Configuration %s, Classifier %s" % (feature, cfg, classifier))
                    print("Useful Bigrams %s" % (sum(i > 0 for i in imp)))

                    top_f.append(num[:max_features])


                if not os.path.exists('FeatureImportance'):
                    os.makedirs('FeatureImportance')
                if not os.path.exists('FeatureImportance/' + feature):
                    os.makedirs('FeatureImportance/' + feature)


                fig.savefig('FeatureImportance/' + feature + '/Facet_' + classifier + '_' + str(cfg) + '_FeatureImportance.pdf')   # save the figure to file
                plt.close(fig)



def ObtainFrequencyDistribution(sample, binWidth):
    f = open(sample + '/packetCount_'+str(binWidth), 'r')

    bin_dict = {}
    bins=[]
    #Generate the set of all possible bins
    for i in range(0,1500, binWidth):
        bin_dict[str(i).replace(" ", "")] = 0


    lines = f.readlines()
    for line in lines:
        try:
            bins.append(line.rstrip('\n'))
        except IndexError:
            break #Reached last index, stop processing
    f.close()

    #Account for each bin elem
    for i in bins:
        bin_dict[str(i)]+=1

    #Order bin_key : num_packets
    od_dict = collections.OrderedDict(sorted(list(bin_dict.items()), key=lambda t: float(t[0])))
    bin_list = []
    for i in od_dict:
        bin_list.append(int(od_dict[i]))

    return bin_list



def genCompareHistGraphPoster():
    cfgs = [12.5,25,50]

    ################################
    # Plot DecisionTrees ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'PL_60/ROC_Decision Tree_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'PL_60/ROC_Decision Tree_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'K=5, s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.legend(loc='lower right', fontsize=20)

    plt.setp(ax1.get_xticklabels(), fontsize=18)
    plt.setp(ax1.get_yticklabels(), fontsize=18)

    fig.savefig('Facet_DecisionTree_CompareHist_Poster.pdf')   # save the figure to file
    plt.close(fig)


    ################################
    # Plot RandomForest ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'PL_60/ROC_RandomForest_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'PL_60/ROC_RandomForest_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'K=5, s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.legend(loc='lower right', fontsize=20)

    plt.setp(ax1.get_xticklabels(), fontsize=18)
    plt.setp(ax1.get_yticklabels(), fontsize=18)

    fig.savefig('Facet_RandomForest_CompareHist_Poster.pdf')   # save the figure to file
    plt.close(fig)

    ################################
    # Plot XGBoost ROC curve
    ################################
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(right=0.96)
    for n, cfg in enumerate(cfgs):
        stats_dec_tree_sensitivity = np.load(xgboost_folder + 'PL_60/ROC_XGBoost_FacetTraffic_' + str(cfg) + '_Christmas_Sensitivity.npy')
        stats_dec_tree_fpr = np.load(xgboost_folder + 'PL_60/ROC_XGBoost_FacetTraffic_' + str(cfg) + '_Christmas_Specificity.npy')

        stats_dec_tree_auc = np.trapz(stats_dec_tree_sensitivity, stats_dec_tree_fpr)
        print("stats AUC " + str(cfg) + ": " + str(stats_dec_tree_auc))
        ax1.plot(stats_dec_tree_fpr, stats_dec_tree_sensitivity, linestyle[0], lw=2, color=colors[n], label = 'K=5, s=' + str(cfg) + '%% ROC (AUC = %0.2f)' % (stats_dec_tree_auc))


    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.legend(loc='lower right', fontsize=20)

    plt.setp(ax1.get_xticklabels(), fontsize=18)
    plt.setp(ax1.get_yticklabels(), fontsize=18)

    fig.savefig('Facet_XGBoost_CompareHist_Poster.pdf')   # save the figure to file
    plt.close(fig)


def genSimilarityGraphPoster():
    cfgs = [50]
    for n, cfg in enumerate(cfgs):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(right=0.96)
        ################################
        # Plot EMD + X classifier ROC curve
        ################################
        EMD_sensitivity = np.load(threshold_folder + 'FacetTraffic_' + str(cfg) + '_Christmas/Rate_50_Sensitivity.npy')
        EMD_specificity = np.load(threshold_folder + 'FacetTraffic_' + str(cfg) + '_Christmas/Rate_50_Specificity.npy')

        EMD_auc = np.trapz(np.array(EMD_sensitivity)[::-1], (np.array(EMD_specificity))[::-1])
        print("EMD AUC" + str(EMD_auc))

        ax1.plot(1 - EMD_specificity, EMD_sensitivity, linestyle[0], lw=2, color=colors[0], label = 'EMD ROC (AUC = %0.2f)' % (EMD_auc))


        X_sensitivity = np.load(xclassifier_folder + 'ROC_True_FacetTraffic_'+ str(cfg) + '_Christmas_20_Sensitivity.npy')
        X_specificity = np.load(xclassifier_folder + 'ROC_True_FacetTraffic_'+ str(cfg) + '_Christmas_20_Specificity.npy')

        X_auc = np.trapz(np.array(X_sensitivity)[::-1], (1-np.array(X_specificity))[::-1])
        print("X AUC" + str(X_auc))

        ax1.plot(1 - X_specificity, X_sensitivity, linestyle[0], lw=2, color=colors[1], label = 'Chi-Square ROC (AUC = %0.2f)' % (X_auc))


        ax1.plot([0, 1], [0, 1], 'k--', lw=2, color="0.0", label = 'Random Guess')
        ax1.grid(color='black', linestyle='dotted')
        plt.xlabel('False Positive Rate', fontsize=24)
        plt.ylabel('True Positive Rate', fontsize=24)
        plt.legend(loc='lower right', fontsize=20)

        plt.setp(ax1.get_xticklabels(), fontsize=18)
        plt.setp(ax1.get_yticklabels(), fontsize=18)

        fig.savefig('Facet_SimilarityGraph_'+ str(cfg) +'Poster.pdf')   # save the figure to file
        plt.close(fig)




if __name__ == "__main__":
    #genSimilarityGraph()
    genCompareStatsGraph()
    genCompareHistGraph()
    genFeatureImportance()
        
    #Figures for Poster
    #genCompareHistGraphPoster()
    #genSimilarityGraphPoster()