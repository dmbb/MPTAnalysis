import dpkt
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import socket
import time
import collections
from scipy.stats import entropy

auxFolder = 'auxFolder/'

cfgs = [
["RegularTraffic_Christmas",
"FacetTraffic_12.5_Christmas"],
["RegularTraffic_Christmas",
"FacetTraffic_25_Christmas"],
["RegularTraffic_Christmas",
"FacetTraffic_50_Christmas"]
]


BIN_WIDTH = [15]

def ComputeFrequencyDistributions(sampleFolder, cfg, binWidth):
    freq_dists = []

    for mode in cfg:
    #Compute frequency distribution for A and B
        freq_dist = []
        for sample in os.listdir(sampleFolder + mode):

            f = open(auxFolder + os.path.dirname(sampleFolder) + "/" + mode + "/" + sample + '/packetCount_' + str(binWidth), 'r')

            bin_dict = {}
            bins=[]
            #Generate the set of all possible bins
            for i in range(0,1500, binWidth):
                bin_dict[str(i).replace(" ", "")] = 1


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
                bin_list.append(float(od_dict[i]))

            #Build up the list of a distribution samples freq dist
            freq_dist.append(bin_list)
        #Build up the list of all freq dists for different sample folders
        freq_dists.append(freq_dist)

    return freq_dists

def KL_Classify(freq_dists):

    #Time measurement - avg single KL
    """times = []
    for j in range(0, len(freq_dists[1])):
        start_time = time.time()
        d = entropy(freq_dists[0][0],freq_dists[1][j])
        end_time = time.time()
        times.append(end_time - start_time)
    print "Avg KL: " + "{0:.5f}".format(np.mean(times,axis=0))"""


    #time measurement - avg classification
    times = []
    start_time = time.time()
    for j in range(0, len(freq_dists[1])):
        d = entropy(freq_dists[0][0],freq_dists[1][j])
    for j in range(0, len(freq_dists[1])):
        d = entropy(freq_dists[0][1],freq_dists[1][j])

        
    end_time = time.time()
    times.append(end_time - start_time)
    #print "Avg sample classification time: " + "{0:.5f}".format(end_time - start_time)


    ###############################
    #Model Building
    ###############################
    start_time = time.time()
    # A vs A
    AvsA_matrix = []
    for i in range(0, len(freq_dists[0])):
        AxVsAy = []
        for j in range(0, len(freq_dists[0])):
            d = entropy(freq_dists[0][i],freq_dists[0][j])
            AxVsAy.append(d)
        AvsA_matrix.append(AxVsAy)




    # A vs B
    AvsB_matrix = []
    for i in range(0,len(freq_dists[0])):
        AxVsBy = []
        start_time = time.time()
        for j in range(0, len(freq_dists[1])):
            d = entropy(freq_dists[0][i],freq_dists[1][j])
            AxVsBy.append(d)
        AvsB_matrix.append(AxVsBy)



    # B vs B
    BvsB_matrix = []
    for i in range(0, len(freq_dists[1])):
        BxVsBy = []
        for j in range(0, len(freq_dists[1])):
            d = entropy(freq_dists[1][i],freq_dists[1][j])
            BxVsBy.append(d)
        BvsB_matrix.append(BxVsBy)

    # B vs A
    BvsA_matrix = []
    for i in range(0,len(freq_dists[1])):
        BxVsAy = []
        for j in range(0, len(freq_dists[0])):
            d = entropy(freq_dists[1][i],freq_dists[0][j])
            BxVsAy.append(d)
        BvsA_matrix.append(BxVsAy)

    end_time = time.time()
    print("Model Building Time: " + "{0:.5f}".format(end_time - start_time))
    ##########################
    #Compute success metric
    #Set A - YouTube
    #Set B - CovertCast
    #TP = Correctly identify CovertCast
    #TN = Correctly identify YouTube
    ##########################

    total_KL_distances = 0
    success = 0
    TrueNegatives = 0
    TruePositives = 0

    #A - B
    for i in range(0,len(freq_dists[0])):
        for j in range(0, len(AvsA_matrix[i])):
            for k in range(0, len(AvsB_matrix[i])):
                total_KL_distances+=1
                if(AvsA_matrix[i][j] < AvsB_matrix[i][k]):
                    success += 1
                    TrueNegatives += 1
    # B - A
    for i in range(0,len(freq_dists[1])):
        for j in range(0, len(BvsB_matrix[i])):
            for k in range(0, len(BvsA_matrix[i])):
                total_KL_distances +=1
                if(BvsB_matrix[i][j] < BvsA_matrix[i][k]):
                    success += 1
                    TruePositives += 1


    print("Total Accuracy: " + str(success / float(total_KL_distances)))
    print("TruePositives: " + str(TruePositives / float(total_KL_distances/2.0)))
    print("TrueNegatives: " + str(TrueNegatives / float(total_KL_distances/2.0)))


if __name__ == "__main__":

    sampleFolders = ['TrafficCaptures/240Resolution/']

    for sampleFolder in sampleFolders:
        print("###########################")
        print(os.path.dirname(sampleFolder))
        print("###########################")
        for cfg in cfgs:
            print("KL classifier - " + cfg[0] + " vs " + cfg[1])
            for binWidth in BIN_WIDTH:
                print("Bin Width: " + str(binWidth))
                KL_Classify(ComputeFrequencyDistributions(sampleFolder, cfg, binWidth))
