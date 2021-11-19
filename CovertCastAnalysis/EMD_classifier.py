#!/usr/bin/env python
import dpkt
import subprocess
import socket
import os
from random import randint
import math
from itertools import product
import datetime
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from pyemd import emd
import collections

BIN_WIDTH = [50]
folder = "auxFolder/"


cfgs = [
["YouTube_home_world_live",
"CovertCast_home_world"]
]




def GatherChatSamples(sampleFolder, baselines, binWidth):
    Samples = []
    for baseline in baselines:
        for cap in os.listdir(folder + sampleFolder + baseline):
            Samples.append(folder + sampleFolder + baseline + "/" + cap + "/" + 'packetCount_' + str(binWidth))
    return Samples

def ComputeRate(sampleFolder, emdResults, num_irregular_samples, num_regular_samples, binWidth):
    deltas = np.arange(0.001, 1, 0.001)

    Sensitivity = []
    Specificity = []


    max_acc = 0
    max_delta = 0
    max_tpr = 0
    max_tnr = 0
    max_fpr = 0

    accuracy = 0
    for delta in deltas:
        FPositives = 0
        FNegatives = 0
        TPositives = 0
        TNegatives = 0


        #Positives are Facet samples classified as Facet
        for i, capEMD in enumerate(emdResults):
            if(capEMD > delta and i < num_regular_samples): # Regular baselines
                FPositives += 1
            if (capEMD < delta and i < num_regular_samples):
                TNegatives += 1
            if(capEMD <= delta and i >= num_regular_samples): #Irregular baseline
                FNegatives += 1
            if(capEMD > delta and i >= num_regular_samples):
                TPositives += 1
        """
        #NEGATED
        for i, capEMD in enumerate(emdResults):
            if(capEMD > delta and i < num_regular_samples): # Regular baselines
                TNegatives += 1
            if (capEMD < delta and i < num_regular_samples):
                FPositives += 1
            if(capEMD <= delta and i >= num_regular_samples): #Irregular baseline
                TPositives += 1
            if(capEMD > delta and i >= num_regular_samples):
                FNegatives += 1
        """
        Sensitivity.append(TPositives/(TPositives+float(FNegatives)))
        Specificity.append(TNegatives/(TNegatives+float(FPositives)))

        accuracy = (TPositives + TNegatives)/float(num_irregular_samples + num_regular_samples)
        if(accuracy > max_acc):
            max_acc = accuracy
            max_delta = delta
            max_tpr = TPositives/(TPositives+float(FNegatives))
            max_tnr = TNegatives/(TNegatives+float(FPositives))
            max_fpr = 1 - max_tnr

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    print("AUC")
    auc = np.trapz(np.array(Sensitivity), 1 - np.array(Specificity))
    print(auc)
    #ROC Curve
    ax1.plot(1 - np.array(Specificity), np.array(Sensitivity), 'k.-', color='black', label = 'ROC (AUC = %0.2f)' % (auc))
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='orange', label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')

    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize='x-large')
    plt.legend(loc='lower right', fontsize='large')

    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)

    max_stats = "Max acc: " + str(max_acc) + " Max TPR:" + str(max_tpr) + " Max TNR:" + str(max_tnr) + " Max FPR:" + str(max_fpr) + " delta:" + str(max_delta)

    fig.savefig('EMD/' + sampleFolder + baselines[1] + '/Rate_' + str(binWidth) + '.pdf')   # save the figure to file
    plt.close(fig)

    return max_stats


def GenerateDists(samples, binWidth):
    dists = []
    print("Building distributions")

    for sample in samples:
        #print sample
        f = open(sample, 'r')

        Gk = {}
        bins=[]
        #Generate the set of all possible bins
        for i in range(0,1500, binWidth):
            Gk[str(i).replace(" ", "")] = 0


        lines = f.readlines()
        for line in lines:
            try:
                bins.append(line.rstrip('\n'))
            except IndexError:
                break #Reached last index, stop processing

        #Account for each bin elem
        for i in bins:
            Gk[str(i)]+=1

        od = collections.OrderedDict(sorted(Gk.items()))
        Gklist = []
        for i in od:
            Gklist.append(float(od[i]))
        Gklist = np.array(Gklist)

        dists.append(Gklist)
        f.close()
    print("End - Building distributions")

    #Build distance matrix
    Gk = {}
    bins=[]
    #Generate the set of all possible bins
    for i in range(0,1500, binWidth):
        Gk[str(i).replace(" ", "")] = 0

    #Generate distance matrix
    distance_matrix = []
    for i in range(0,len(Gk)):
        line =[]
        for j in range(0,len(Gk)):
            if(i==j):
                line.append(0.0)
            else:
                line.append(1.0)

        distance_matrix.append(np.array(line))
    distance_matrix = np.array(distance_matrix)

    return dists, distance_matrix


def Classifier(toClassify, allSamples, baseSamples, distance_matrix, binWidth):
    emdResults = []
    emdSum = 0
    ##################################
    #Read first element in combination
    ##################################
    Gk_corelist = toClassify

    for n, sample in enumerate(baseSamples):

        Gklist = sample
        ############################
        ###### NORMALIZATION #######
        ############################
        ground1 = max(Gk_corelist)
        ground2 = max(Gklist)
        if(ground1 > ground2):
            MAX = ground1
        else:
            MAX = ground2

        if(max(np.cumsum(Gk_corelist)) > max(np.cumsum(Gklist))):
            cSum = max(np.cumsum(Gk_corelist))
        else:
            cSum =  max(np.cumsum(Gklist))

        dtm = distance_matrix/cSum

        emdR = float(emd(Gk_corelist, Gklist, dtm))
        emdSum += emdR
        emdResults.append(emdR)

    avgEMD = emdSum / len(emdResults)
    #print str(avgEMD)
    return avgEMD

def plotEMD(sampleFolder, baselines, binWidth):
    regularSamples = GatherChatSamples(sampleFolder,baselines[:-1], binWidth)
    allSamples = GatherChatSamples(sampleFolder, baselines, binWidth)

    allSamplesDists, distance_matrix = GenerateDists(allSamples, binWidth)

    emdResults = []
    for n, bs in enumerate(allSamplesDists):
        #print allSamples[n]
        emdResults.append(Classifier(bs, allSamples, allSamplesDists[:len(regularSamples)], distance_matrix, binWidth))

    acc = float(0)
    for i in range(0,len(regularSamples)):
        acc += emdResults[i]
    acc = acc/len(regularSamples)
    print("AVG Regular " + str(acc))

    max_stat = ComputeRate(sampleFolder, emdResults, len(allSamples) - len(regularSamples), len(regularSamples), binWidth)
    print(max_stat)


if __name__ == "__main__":

    sampleFolders = ["TrafficCaptures/"]

    if not os.path.exists('EMD'):
                os.makedirs('EMD')

    for sampleFolder in sampleFolders:
        for baselines in cfgs:
            print("===========================================")
            print("Analyzing " + baselines[0] + " - " + baselines[1])
            for binWidth in BIN_WIDTH:
                print("##############")
                print("BinWidth: " + str(binWidth))
                if not os.path.exists('EMD/' + sampleFolder + baselines[1]):
                    os.makedirs('EMD/' + sampleFolder + baselines[1])
                plotEMD(sampleFolder, baselines, binWidth)
