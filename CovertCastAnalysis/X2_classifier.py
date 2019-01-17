import dpkt
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import socket
import collections
from itertools import product
from scipy.stats import entropy, chisquare, norm, rv_continuous
import random

random.seed(a=1)

auxFolder = 'auxFolder/'

cfgs = [
["YouTube_home_world_live",
"CovertCast_home_world"]
]


BIN_WIDTH = [20]
#BIN_WIDTH = [50]

def ComputeBiGramDistributions(sampleFolder, cfg, binWidth):
    freq_dists = []

    for mode in cfg:
    #Compute frequency distribution for A and B
        freq_dist = []
        for sample in os.listdir(sampleFolder + mode):

            f = open(auxFolder + os.path.dirname(sampleFolder) + "/" + mode + "/" + sample + '/bigrams_' + str(binWidth), 'r')

            bin_dict = {}
            bigrams=[]
            #Generate the set of all possible bigrams
            for i in product(range(0,1500, binWidth), repeat=2):
                bin_dict[str(i).replace(" ", "")] = 1


            lines = f.readlines()
            for line in lines:
                try:
                    bigrams.append(line.rstrip('\n'))
                except IndexError:
                    break #Reached last index, stop processing
            f.close()

            #Account for each bin elem
            for i in bigrams:
                bin_dict['('+str(i)+')']+=1

            #Order bin_key : num_packets
            od_dict = collections.OrderedDict(sorted(bin_dict.items()))
            bin_list = []
            for i in od_dict:
                bin_list.append(float(od_dict[i]))

            #Build up the list of a distribution samples freq dist
            freq_dist.append(bin_list)
        #Build up the list of all freq dists for different sample folders
        freq_dists.append(freq_dist)

    return freq_dists


def computeIntraVariance(freq_dists):
    varIntra = np.zeros(len(freq_dists[0][0]))

    for i in range(0, len(freq_dists[0][0])):
        somatory = 0

        for m in freq_dists:
            term = 0
            #Compute total n_grams in model
            total_ngrams_model = 0
            for v in m:
                total_ngrams_model += sum(v)

            #Compute probability of a given n_gram in model
            prob_ngram_model = 0
            for v in m:
                prob_ngram_model += v[i]
            prob_ngram_model = prob_ngram_model / float(total_ngrams_model)

            for v in m:
                n_gram_prob_v = v[i]/sum(v)
                term += (float(n_gram_prob_v) - prob_ngram_model)**2

            somatory += 1/float(len(m)) * term

        varIntra[i] = 1/2.0 * somatory

    return varIntra


def computeInterVariance(freq_dists):
    varInter = np.zeros(len(freq_dists[0][0]))

    total_videos = len(freq_dists[0]) + len(freq_dists[1])

    for i in range(0, len(freq_dists[0][0])):
        somatory = 0

        ###For each model
        for n, m in enumerate(freq_dists):
            #Compute total n_grams in model
            total_ngrams_model = 0
            for v in m:
                total_ngrams_model += sum(v)

            #Compute total n_grams in other model
            total_ngrams_other_model = 0
            for v in freq_dists[(n+1)%2]:
                total_ngrams_other_model += sum(v)

            #Compute probability of a given n_gram in model
            prob_ngram_model = 0
            for v in m:
                prob_ngram_model += v[i]
            prob_ngram_model = prob_ngram_model / float(total_ngrams_model)

            #Compute probability of a given n_gram in the other model
            prob_ngram_other_model = 0
            for v in freq_dists[(n+1)%2]:
                prob_ngram_other_model += v[i]
            prob_ngram_other_model = prob_ngram_other_model / float(total_ngrams_other_model)

            ###For each video in model
            for v in m:
                n_gram_prob_v = v[i]/sum(v)
                somatory += (float(n_gram_prob_v) - prob_ngram_model)**2

        varInter[i] = 1.0/total_videos * somatory

    return varInter


def optimizeBigrams(freq_dists):

    varIntra = computeIntraVariance(freq_dists)
    varInter = computeInterVariance(freq_dists)

    DIS = np.zeros(len(varIntra))
    DIS = varInter/varIntra

    indexes_to_remove = []

    for n, i in enumerate(DIS):
        if(i < 1):
            indexes_to_remove.append(n)

    return indexes_to_remove


def buildModels(freq_dists):
    #####################################
    # Build models
    #####################################
    model_chat = np.zeros(len(freq_dists[0][0]))
    model_censored = np.zeros(len(freq_dists[0][0]))

    total_ngrams_chat_set = 0
    for dist in freq_dists[0]:
        total_ngrams_chat_set += sum(dist)

    total_ngrams_censored_set = 0
    for dist in freq_dists[1]:
        total_ngrams_censored_set += sum(dist)


    for i in range(0, len(model_chat)):
        somatory = 0
        for v in freq_dists[0]:
            n_gram_prob = v[i]/sum(v)
            v_total_grams = sum(v)
            somatory += (v_total_grams * n_gram_prob)
        model_chat[i] = (1/total_ngrams_chat_set) * somatory


    for i in range(0, len(model_censored)):
        somatory = 0
        for v in freq_dists[1]:
            n_gram_prob = v[i]/float(sum(v))
            v_total_grams = sum(v)
            somatory += (v_total_grams * n_gram_prob)
        model_censored[i] = (1/float(total_ngrams_censored_set)) * somatory

    return model_chat, model_censored

#Reproduces Facet Fixed threshold evalution
def Prepare_X_Fixed(fig_folder, cfg,binWidth,freq_dists):
    optimization = True

    #Transform original freq_dists to include only the better bi-grams
    chat_samples = freq_dists[0]
    censored_samples = freq_dists[1]

    filtered_freq_dists = []
    filtered_chat_samples = []
    filtered_censored_samples = []

    if(optimization):
        #Optimize bigram choice, build updated frequency distributions
        indexes_to_remove = optimizeBigrams(freq_dists)

        for sample in chat_samples:
            filtered_chat_samples.append(np.delete(sample, indexes_to_remove))

        for sample in censored_samples:
            filtered_censored_samples.append(np.delete(sample, indexes_to_remove))
    else:
        #Ignore optimization procedure, carry on with original frequency distributions
        filtered_chat_samples = chat_samples
        filtered_censored_samples = censored_samples

    #2x Cross validation
    filtered_freq_dists1 = []
    filtered_freq_dists2 = []

    #To Remove
    #x = random.sample(filtered_chat_samples, len(filtered_chat_samples))
    #x2 = random.sample(filtered_censored_samples, len(filtered_censored_samples))

    filtered_freq_dists1.append(filtered_chat_samples[:len(filtered_chat_samples)/2])
    filtered_freq_dists1.append(filtered_censored_samples[:len(filtered_censored_samples)/2])

    filtered_freq_dists2.append(filtered_chat_samples[len(filtered_chat_samples)/2:])
    filtered_freq_dists2.append(filtered_censored_samples[len(filtered_censored_samples)/2:])

    model_chat1, model_censored1 = buildModels(filtered_freq_dists1)
    acc1, tnr1, fnr1, tpr1, fpr1, ppv1, npv1 = X_Classify_Fixed(cfg,binWidth,filtered_freq_dists2, model_chat1, model_censored1)
    print "1st Fold"
    print "Acc = " + str(acc1)
    print "TPR = " + str(tpr1)
    print "TNR = " + str(tnr1)
    print "FPR = " + str(fpr1)
    print "FNR = " + str(fnr1)
    print "PPV = " + str(ppv1)
    print "NPV = " + str(npv1)

    model_chat2, model_censored2 = buildModels(filtered_freq_dists2)
    acc2, tnr2, fnr2, tpr2, fpr2, ppv2, npv2 = X_Classify_Fixed(cfg,binWidth,filtered_freq_dists1, model_chat2, model_censored2)
    print "\n2nd Fold"
    print "Acc = " + str(acc2)
    print "TPR = " + str(tpr2)
    print "TNR = " + str(tnr2)
    print "FPR = " + str(fpr2)
    print "FNR = " + str(fnr2)
    print "PPV = " + str(ppv2)
    print "NPV = " + str(npv2)

    print "\n###################"
    print "Average"
    print "Acc = " + str((acc1 + acc2)/2.0)
    print "TPR = " + str((tpr1 + tpr2)/2.0)
    print "TNR = " + str((tnr1 + tnr2)/2.0)
    print "FPR = " + str((fpr1 + fpr2)/2.0)
    print "FNR = " + str((fnr1 + fnr2)/2.0)
    print "PPV = " + str((ppv1 + ppv2)/2.0)
    print "NPV = " + str((npv1 + npv2)/2.0)


######################################################################################
def X_Classify_Fixed(cfg, binWidth, freq_dists, model_chat, model_censored):
    ##########################
    #Classify samples
    ##########################
    FPositives = 0
    FNegatives = 0
    TPositives = 0
    TNegatives = 0

    #True negative is being classified as facet when it is facet
    for v in freq_dists[0]:
        chat_score = chisquare(v, model_chat)
        censored_score = chisquare(v, model_censored)

        if(chat_score < censored_score):
            TPositives += 1
        elif(censored_score < chat_score):
            FNegatives += 1

    for v in freq_dists[1]:
        chat_score = chisquare(v, model_chat)
        censored_score = chisquare(v, model_censored)

        if(censored_score < chat_score):
            TNegatives += 1
        elif(chat_score < censored_score):
            FPositives += 1


    accuracy = (TPositives + TNegatives)/float(len(freq_dists[0]) + len(freq_dists[1]))
    TNR = TNegatives/(TNegatives+float(FPositives))
    FNR = FNegatives/(TPositives+float(FNegatives))
    TPR = TPositives/(TPositives+float(FNegatives))
    FPR = FPositives/(FPositives+float(TNegatives))
    PPV = TPositives/(TPositives+float(FPositives))
    NPV = TNegatives/(TNegatives+float(FNegatives))

    return accuracy, TNR, FNR, TPR, FPR, PPV, NPV


#Reproduces Facet Changing deltas evaluation
def Prepare_X_RatioReproduction(fig_folder, cfg,binWidth,freq_dists):
    optimization = True


    #Transform original freq_dists to include only the better bi-grams
    chat_samples = freq_dists[0]
    censored_samples = freq_dists[1]

    filtered_freq_dists = []
    filtered_chat_samples = []
    filtered_censored_samples = []

    if(optimization):
        #Optimize bigram choice, build updated frequency distributions
        indexes_to_remove = optimizeBigrams(freq_dists)

        for sample in chat_samples:
            filtered_chat_samples.append(np.delete(sample, indexes_to_remove))

        for sample in censored_samples:
            filtered_censored_samples.append(np.delete(sample, indexes_to_remove))
    else:
        #Ignore optimization procedure, carry on with original frequency distributions
        filtered_chat_samples = chat_samples
        filtered_censored_samples = censored_samples

    #2x Cross validation
    filtered_freq_dists1 = []
    filtered_freq_dists2 = []

    #To remove
    #x = random.sample(filtered_chat_samples, len(filtered_chat_samples))
    #x2 = random.sample(filtered_censored_samples, len(filtered_censored_samples))

    filtered_freq_dists1.append(filtered_chat_samples[:len(filtered_chat_samples)/2])
    filtered_freq_dists1.append(filtered_censored_samples[:len(filtered_censored_samples)/2])

    filtered_freq_dists2.append(filtered_chat_samples[len(filtered_chat_samples)/2:])
    filtered_freq_dists2.append(filtered_censored_samples[len(filtered_censored_samples)/2:])

    model_chat1, model_censored1 = buildModels(filtered_freq_dists1)
    max_acc, max_delta, max_tpr, max_fpr, val90, val80, val70, specificity, sensitivity = X_Classify_RatioReproduction(cfg,binWidth,filtered_freq_dists2, model_chat1, model_censored1)
    print "1st Fold"
    print "TPR90 = " + str(val90)
    print "TPR80 = " + str(val80)
    print "TPR70 = " + str(val70)
    print "Max acc: " + str(max_acc) + " Max TPR:" + str(max_tpr) + " Max FPR:" + str(max_fpr) + " delta:" + str(max_delta)

    model_chat2, model_censored2 = buildModels(filtered_freq_dists2)
    max_acc2, max_delta2, max_tpr2, max_fpr2, val902, val802, val702, specificity2, sensitivity2 = X_Classify_RatioReproduction(cfg,binWidth,filtered_freq_dists1, model_chat2, model_censored2)
    print "2nd Fold"
    print "TPR90 = " + str(val902)
    print "TPR80 = " + str(val802)
    print "TPR70 = " + str(val702)
    print "Max acc: " + str(max_acc2) + " Max TPR:" + str(max_tpr2) + " Max FPR:" + str(max_fpr2) + " delta:" + str(max_delta2)

    print "###################"
    print "Average FPR"
    print "TPR90 = " + str((val902+val90)/2.0)
    print "TPR80 = " + str((val802+val80)/2.0)
    print "TPR70 = " + str((val702+val70)/2.0)
    print "Max acc: " + str((max_acc+max_acc2)/2.0) + " Max TPR:" + str((max_tpr+max_tpr2)/2.0) + " Max FPR:" + str((max_fpr+max_fpr2)/2.0) + " delta:" + str((max_delta + max_delta2)/2.0)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    Specificity = (specificity + specificity2)/2.0
    Sensitivity = (sensitivity + sensitivity2)/2.0

    """
    np.set_printoptions(threshold=np.inf)
    print specificity
    print specificity2
    """

    #ROC Curve
    ax1.plot(1 - specificity, sensitivity, color='red', lw=2, alpha=0.7, label = 'k-Fold ROC')
    ax1.plot(1 - specificity2, sensitivity2, color='red', lw=2, alpha=0.7)
    ax1.plot(1 - Specificity, Sensitivity, 'k.-', color='black', label = 'Mean ROC')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='orange', label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')

    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize='x-large')
    plt.legend(loc='lower right', fontsize='large')

    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)

    fig.savefig(fig_folder + "ROC_" + str(optimization) + "_" + cfg[1] + "_" + str(binWidth)+".pdf")   # save the figure to file
    plt.close(fig)

def X_Classify_RatioReproduction(cfg, binWidth,freq_dists, model_chat, model_censored):
    ##########################
    #Classify samples
    ##########################
    deltas = np.arange(0.001, 5, 0.001)
    FalsePositives = []
    FalseNegatives = []
    TruePositives = []
    TrueNegatives = []

    Sensitivity = []
    Specificity = []
    FalsePositiveRate = []
    FalseNegativeRate =[]

    holding90 = True
    holding80 = True
    holding70 = True

    thresh90 = 0
    thresh80 = 0
    thresh70 = 0

    val90 = 0
    val80 = 0
    val70 = 0

    max_acc = 0
    max_delta = 0
    max_tpr = 0
    max_fpr = 0

    for delta in deltas:
        FPositives = 0
        FNegatives = 0
        TPositives = 0
        TNegatives = 0

        chat_ratios = []
        censored_ratios = []

        #Positive example is chat
        #True positive is being classified as facet when it is facet
        for v in freq_dists[0]:
            chat_score, p_value = chisquare(v, model_chat)
            censored_score, p_value2 = chisquare(v, model_censored)


            ratio = chat_score / float(censored_score)
            chat_ratios.append(ratio)
            if(ratio < delta):
                TNegatives += 1
            elif(ratio > delta):
                FPositives += 1

        for v in freq_dists[1]:
            chat_score, p_value = chisquare(v, model_chat)
            censored_score, p_value2 = chisquare(v, model_censored)

            ratio = chat_score / float(censored_score)
            censored_ratios.append(ratio)
            if(ratio > delta):
                TPositives += 1
            elif(ratio < delta):
                FNegatives += 1


        accuracy = (TPositives + TNegatives)/float(len(freq_dists[0]) + len(freq_dists[1]))
        TNR = TNegatives/(TNegatives+float(FPositives))
        FNR = FNegatives/(TPositives+float(FNegatives))
        TPR = TPositives/(TPositives+float(FNegatives))
        FPR = FPositives/(FPositives+float(TNegatives))

        if(accuracy > max_acc):
            max_acc = accuracy
            max_tpr = TPR
            max_fpr = FPR
            max_delta = delta

        FalsePositives.append(FPositives)
        FalseNegatives.append(FNegatives)
        TruePositives.append(TPositives)
        TrueNegatives.append(TNegatives)
        Sensitivity.append(TPositives/(TPositives+float(FNegatives)))
        Specificity.append(TNegatives/(TNegatives+float(FPositives)))
        FalsePositiveRate.append(FPR)
        FalseNegativeRate.append(FNR)

        if(holding90):
            if(FNR >= 0.1):
                holding90 = False
                thresh90 = delta
                val90 = FPR

        if(holding80):
            if(FNR >= 0.2):
                holding80 = False
                thresh80 = delta
                val80 = FPR

        if(holding70):
            if(FNR >= 0.3):
                holding70 = False
                thresh70 = delta
                val70 = FPR

    return max_acc, max_delta, max_tpr, max_fpr, val90, val80, val70, np.array(Specificity), np.array(Sensitivity)





if __name__ == "__main__":

    sampleFolder = "TrafficCaptures/"

    if not os.path.exists('X2'):
                os.makedirs('X2')
    if not os.path.exists('X2/' + os.path.dirname(sampleFolder)):
                os.makedirs('X2/' + os.path.dirname(sampleFolder))

    fig_folder = 'X2/' + os.path.dirname(sampleFolder) + '/'


    print "###########################"
    print os.path.dirname(sampleFolder)
    print "###########################"
    for cfg in cfgs:
        random.seed(a=1) # re-seed
        print "====================================="
        print "X classifier - " + cfg[0] + " vs " + cfg[1]
        for binWidth in BIN_WIDTH:
            print "---------------------"
            print "Bin Width: " + str(binWidth)
            print "---------------------"
            #Compute bigram distributions and shuffle the samples
            freq_dists = ComputeBiGramDistributions(sampleFolder, cfg, binWidth)
            x = random.sample(freq_dists[0], len(freq_dists[0]))
            x2 = random.sample(freq_dists[1], len(freq_dists[1]))
            freqs = []
            freqs.append(x)
            freqs.append(x2)

            #For reproducing results of Facet paper (70%,80%,90% blockage)
            #Prepare_X_RatioReproduction(fig_folder, cfg,binWidth, freqs)

            #For getting fixed classification rates to compare with classifiers without a notion of internal thereshold
            Prepare_X_Fixed(fig_folder, cfg,binWidth, freqs)
