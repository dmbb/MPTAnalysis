#!/usr/bin/env python
import collections
import dpkt
import subprocess
import socket
import os
import math
import csv
import numpy as np
from itertools import product
from scipy.stats import kurtosis, skew

from dataset_gen import GenerateDatasets

import time
#DEST_IP = '172.31.0.21' # For Skypev4.3 captures
DEST_IP = '172.31.0.2'
SOURCE_IP = '172.31.0.19'

def RoundToNearest(n, m):
        r = n % m
        return n + m - r if r + r >= m else n - r

def ExtractFeatures(sampleFolder):

    arff = open(sampleFolder + '_dataset.csv', 'wb')
    written_header = False

    for sample in os.listdir(sampleFolder):
        f = open(sampleFolder + "/" + sample + "/" + sample)
        print(sample)
        pcap = dpkt.pcap.Reader(f)

        #Analyse packets transmited
        totalPackets = 0
        totalPacketsIn = 0
        totalPacketsOut = 0

        #Analyse bytes transmitted
        totalBytes = 0
        totalBytesIn = 0
        totalBytesOut = 0

        #Analyse packet sizes
        packetSizes = []
        packetSizesIn = []
        packetSizesOut = []

        #Analyse inter packet timing
        packetTimes = []
        packetTimesIn = []
        packetTimesOut = []

        #Analyse outcoming bursts
        out_bursts_packets = []
        out_burst_sizes = []
        out_burst_times = []
        out_burst_start = 0
        out_current_burst = 0
        out_current_burst_start = 0
        out_current_burst_size = 0
        out_current_burst_time = 0

        #Analyse incoming bursts
        in_bursts_packets = []
        in_burst_sizes = []
        in_burst_times = []
        in_burst_start = 0
        in_current_burst = 0
        in_current_burst_start = 0
        in_current_burst_size = 0
        in_current_burst_time = 0

        prev_ts = 0
        absTimesOut = []
        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            ip_hdr = eth.data
            try:
                src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                dst_ip_addr_str = socket.inet_ntoa(ip_hdr.dst)
                #Target UDP communication between both cluster machines
                if (ip_hdr.p == 17 and ((dst_ip_addr_str == SOURCE_IP) or (src_ip_addr_str == SOURCE_IP))):
                    #General packet statistics
                    totalPackets += 1
                    #if(src_ip_addr_str == DEST_IP):
                    if(src_ip_addr_str != SOURCE_IP):
                        totalPacketsIn += 1
                        packetSizesIn.append(len(buf))

                        if(prev_ts != 0):
                            ts_difference = ts - prev_ts
                            packetTimesIn.append(ts_difference)

                        if(out_current_burst != 0):
                            if(out_current_burst > 1):
                                out_bursts_packets.append(out_current_burst) #packets on burst
                                out_burst_sizes.append(out_current_burst_size) #total bytes on burst
                                out_burst_times.append(ts - out_current_burst_start)
                            out_current_burst = 0
                            out_current_burst_size = 0
                            out_current_burst_start = 0
                        if(in_current_burst == 0):
                            in_current_burst_start = ts
                        in_current_burst += 1
                        in_current_burst_size += len(buf)
                    else:
                        totalPacketsOut += 1
                        absTimesOut.append(ts)
                        packetSizesOut.append(len(buf))
                        if(prev_ts != 0):
                            ts_difference = ts - prev_ts
                            packetTimesOut.append(ts_difference)
                        if(out_current_burst == 0):
                            out_current_burst_start = ts
                        out_current_burst += 1
                        out_current_burst_size += len(buf)

                        if(in_current_burst != 0):
                            if(in_current_burst > 1):
                                in_bursts_packets.append(out_current_burst) #packets on burst
                                in_burst_sizes.append(out_current_burst_size) #total bytes on burst
                                in_burst_times.append(ts - out_current_burst_start)
                            in_current_burst = 0
                            in_current_burst_size = 0
                            in_current_burst_start = 0




                    #Bytes transmitted statistics
                    totalBytes += len(buf)
                    #if(src_ip_addr_str == DEST_IP):
                    if(src_ip_addr_str != SOURCE_IP):
                        totalBytesIn += len(buf)
                    else:
                        totalBytesOut += len(buf)


                    #Packet Size statistics
                    packetSizes.append(len(buf))

                    #Packet Times statistics
                    if(prev_ts != 0):
                        #print "{0:.6f}".format(ts)
                        ts_difference = ts - prev_ts
                        packetTimes.append(ts_difference)

                    prev_ts = ts
            except:
                pass
        f.close()

        """
        #################################################################
        # Compute BytesPerSecond & PacketsPerSecond Timeseries
        ##################################################################
        t = absTimesOut[0] + 0.25
        last_index = 0
        BytesPerSecond = []
        AvgBytesPerSecond = []
        PacketsPerSecond = []

        while t < (absTimesOut[0] + 58): #! Adjust conforming to cap len.
        #Cannot be max(absTimesOut) because some samples are slightly bigger than others
            indices = [i for i,v in enumerate(absTimesOut) if v <= t]

            bps = 0
            for i in indices[last_index:]:
                bps += packetSizesOut[i]

            BytesPerSecond.append(bps)
            if(len(indices[last_index:]) > 0):
                AvgBytesPerSecond.append(bps/len(indices[last_index:]))
            else:
                AvgBytesPerSecond.append(0)
            PacketsPerSecond.append(len(indices[last_index:]))
            last_index = len(indices)
            t += 0.25
        """


        ################################################################
        #Bin packet sizes
        ################################################################

        bin_dict = {}
        binWidth = 5
        #Generate the set of all possible bins
        for i in range(0,1500, binWidth):
            bin_dict[str(i).replace(" ", "")] = 0

        for i in packetSizesOut:
            binned = RoundToNearest(i,binWidth)
            bin_dict[str(binned)]+=1

        od_dict = collections.OrderedDict(sorted(list(bin_dict.items()), key=lambda t: float(t[0])))
        bin_list = []
        for i in od_dict:
            bin_list.append(od_dict[i]) #Fraction of packets inside a given bin


        bin_dict2 = {}
        binWidth = 5
        #Generate the set of all possible bins
        for i in range(0,1500, binWidth):
            bin_dict2[str(i).replace(" ", "")] = 0

        for i in packetSizesIn:
            binned = RoundToNearest(i,binWidth)
            bin_dict2[str(binned)]+=1

        od_dict2 = collections.OrderedDict(sorted(list(bin_dict2.items()), key=lambda t: float(t[0])))
        bin_list2 = []
        for i in od_dict2:
            bin_list2.append(od_dict2[i])#/float(len(packetSizesIn))) #Fraction of packets inside a given bin

        """
        ################################################################
        #Bigram packet sizes
        ################################################################
        bi_gram_dict = {}
        binWidth = 20

        bi_grams=[]
        #Generate the set of all possible bi-grams
        for i in product(range(0,1500, binWidth), repeat=2):
            bi_gram_dict[str(i).replace(" ", "")] = 0

        counter = 0
        for n, i in enumerate(packetSizesOut[:-1]):
            binned_n = RoundToNearest(packetSizesOut[n],binWidth)
            binned_n_next = RoundToNearest(packetSizesOut[n+1],binWidth)
            bi_gram_dict['('+str(binned_n) + ','+ str(binned_n_next) + ')'] += 1
            counter += 1


        od = collections.OrderedDict(sorted(bi_gram_dict.items()))

        bigram_list = []
        for i in od:
            bigram_list.append(float(od[i]/float(counter)))
        """


        ################################################################
        ####################Compute statistics#####################
        ################################################################

        ##########################################################
        #Statistical indicators for packet sizes (total)
        meanPacketSizes = np.mean(packetSizes)
        medianPacketSizes = np.median(packetSizes)
        stdevPacketSizes = np.std(packetSizes)
        variancePacketSizes = np.var(packetSizes)
        kurtosisPacketSizes = kurtosis(packetSizes)
        skewPacketSizes = skew(packetSizes)
        maxPacketSize = np.amax(packetSizes)
        minPacketSize = np.amin(packetSizes)
        p10PacketSizes = np.percentile(packetSizes,10)
        p20PacketSizes = np.percentile(packetSizes,20)
        p30PacketSizes = np.percentile(packetSizes,30)
        p40PacketSizes = np.percentile(packetSizes,40)
        p50PacketSizes = np.percentile(packetSizes,50)
        p60PacketSizes = np.percentile(packetSizes,60)
        p70PacketSizes = np.percentile(packetSizes,70)
        p80PacketSizes = np.percentile(packetSizes,80)
        p90PacketSizes = np.percentile(packetSizes,90)


        ##########################################################
        #Statistical indicators for packet sizes (in)
        meanPacketSizesIn = np.mean(packetSizesIn)
        medianPacketSizesIn = np.median(packetSizesIn)
        stdevPacketSizesIn = np.std(packetSizesIn)
        variancePacketSizesIn = np.var(packetSizesIn)
        kurtosisPacketSizesIn = kurtosis(packetSizesIn)
        skewPacketSizesIn = skew(packetSizesIn)
        maxPacketSizeIn = np.amax(packetSizesIn)
        minPacketSizeIn = np.amin(packetSizesIn)
        p10PacketSizesIn = np.percentile(packetSizesIn,10)
        p20PacketSizesIn = np.percentile(packetSizesIn,20)
        p30PacketSizesIn = np.percentile(packetSizesIn,30)
        p40PacketSizesIn = np.percentile(packetSizesIn,40)
        p50PacketSizesIn = np.percentile(packetSizesIn,50)
        p60PacketSizesIn = np.percentile(packetSizesIn,60)
        p70PacketSizesIn = np.percentile(packetSizesIn,70)
        p80PacketSizesIn = np.percentile(packetSizesIn,80)
        p90PacketSizesIn = np.percentile(packetSizesIn,90)


        ##########################################################
        #Statistical indicators for packet sizes (out)
        meanPacketSizesOut = np.mean(packetSizesOut)
        medianPacketSizesOut = np.median(packetSizesOut)
        stdevPacketSizesOut = np.std(packetSizesOut)
        variancePacketSizesOut = np.var(packetSizesOut)
        kurtosisPacketSizesOut = kurtosis(packetSizesOut)
        skewPacketSizesOut = skew(packetSizesOut)
        maxPacketSizeOut = np.amax(packetSizesOut)
        minPacketSizeOut = np.amin(packetSizesOut)
        p10PacketSizesOut = np.percentile(packetSizesOut,10)
        p20PacketSizesOut = np.percentile(packetSizesOut,20)
        p30PacketSizesOut = np.percentile(packetSizesOut,30)
        p40PacketSizesOut = np.percentile(packetSizesOut,40)
        p50PacketSizesOut = np.percentile(packetSizesOut,50)
        p60PacketSizesOut = np.percentile(packetSizesOut,60)
        p70PacketSizesOut = np.percentile(packetSizesOut,70)
        p80PacketSizesOut = np.percentile(packetSizesOut,80)
        p90PacketSizesOut = np.percentile(packetSizesOut,90)

        """
        #############################################################
        # Statistical indicators for derivative of packet sizes (out)
        derivative = np.gradient(packetSizesOut)
        meanDer = np.mean(derivative)
        medianDer = np.median(derivative)
        stdevDer = np.std(derivative)
        varianceDer = np.var(derivative)
        kurtosisDer = kurtosis(derivative)
        skewDer = skew(derivative)
        maxDer = np.amax(derivative)
        minDer = np.amin(derivative)
        p10Der = np.percentile(derivative,10)
        p20Der = np.percentile(derivative,20)
        p30Der = np.percentile(derivative,30)
        p40Der = np.percentile(derivative,40)
        p50Der = np.percentile(derivative,50)
        p60Der = np.percentile(derivative,60)
        p70Der = np.percentile(derivative,70)
        p80Der = np.percentile(derivative,80)
        p90Der = np.percentile(derivative,90)


        #################################################################
        # Statistical indicators for 2nd derivative of packet sizes (out)
        derivative_2 = np.gradient(derivative)
        meanDer2 = np.mean(derivative_2)
        medianDer2 = np.median(derivative_2)
        stdevDer2 = np.std(derivative_2)
        varianceDer2 = np.var(derivative_2)
        kurtosisDer2 = kurtosis(derivative_2)
        skewDer2 = skew(derivative_2)
        maxDer2 = np.amax(derivative_2)
        minDer2 = np.amin(derivative_2)
        p10Der2 = np.percentile(derivative_2,10)
        p20Der2 = np.percentile(derivative_2,20)
        p30Der2 = np.percentile(derivative_2,30)
        p40Der2 = np.percentile(derivative_2,40)
        p50Der2 = np.percentile(derivative_2,50)
        p60Der2 = np.percentile(derivative_2,60)
        p70Der2 = np.percentile(derivative_2,70)
        p80Der2 = np.percentile(derivative_2,80)
        p90Der2 = np.percentile(derivative_2,90)

        #################################################################
        # Statistical indicators for 3rd derivative of packet sizes (out)
        derivative_3 = np.gradient(derivative_2)
        meanDer3 = np.mean(derivative_3)
        medianDer3 = np.median(derivative_3)
        stdevDer3 = np.std(derivative_3)
        varianceDer3 = np.var(derivative_3)
        kurtosisDer3 = kurtosis(derivative_3)
        skewDer3 = skew(derivative_3)
        maxDer3 = np.amax(derivative_3)
        minDer3 = np.amin(derivative_3)
        p10Der3 = np.percentile(derivative_3,10)
        p20Der3 = np.percentile(derivative_3,20)
        p30Der3 = np.percentile(derivative_3,30)
        p40Der3 = np.percentile(derivative_3,40)
        p50Der3 = np.percentile(derivative_3,50)
        p60Der3 = np.percentile(derivative_3,60)
        p70Der3 = np.percentile(derivative_3,70)
        p80Der3 = np.percentile(derivative_3,80)
        p90Der3 = np.percentile(derivative_3,90)
        """
        ##################################################################
        #Statistical indicators for Inter-Packet Times (total)

        meanPacketTimes = np.mean(packetTimes)
        medianPacketTimes = np.median(packetTimes)
        stdevPacketTimes = np.std(packetTimes)
        variancePacketTimes = np.var(packetTimes)
        kurtosisPacketTimes = kurtosis(packetTimes)
        skewPacketTimes = skew(packetTimes)
        maxIPT = np.amax(packetTimes)
        minIPT = np.amin(packetTimes)
        p10PacketTimes = np.percentile(packetTimes,10)
        p20PacketTimes = np.percentile(packetTimes,20)
        p30PacketTimes = np.percentile(packetTimes,30)
        p40PacketTimes = np.percentile(packetTimes,40)
        p50PacketTimes = np.percentile(packetTimes,50)
        p60PacketTimes = np.percentile(packetTimes,60)
        p70PacketTimes = np.percentile(packetTimes,70)
        p80PacketTimes = np.percentile(packetTimes,80)
        p90PacketTimes = np.percentile(packetTimes,90)


        ##################################################################
        #Statistical indicators for Inter-Packet Times (in)
        meanPacketTimesIn = np.mean(packetTimesIn)
        medianPacketTimesIn = np.median(packetTimesIn)
        stdevPacketTimesIn = np.std(packetTimesIn)
        variancePacketTimesIn = np.var(packetTimesIn)
        kurtosisPacketTimesIn = kurtosis(packetTimesIn)
        skewPacketTimesIn = skew(packetTimesIn)
        maxPacketTimesIn = np.amax(packetTimesIn)
        minPacketTimesIn = np.amin(packetTimesIn)
        p10PacketTimesIn = np.percentile(packetTimesIn,10)
        p20PacketTimesIn = np.percentile(packetTimesIn,20)
        p30PacketTimesIn = np.percentile(packetTimesIn,30)
        p40PacketTimesIn = np.percentile(packetTimesIn,40)
        p50PacketTimesIn = np.percentile(packetTimesIn,50)
        p60PacketTimesIn = np.percentile(packetTimesIn,60)
        p70PacketTimesIn = np.percentile(packetTimesIn,70)
        p80PacketTimesIn = np.percentile(packetTimesIn,80)
        p90PacketTimesIn = np.percentile(packetTimesIn,90)


        ##################################################################
        #Statistical indicators for Inter-Packet Times (out)
        meanPacketTimesOut = np.mean(packetTimesOut)
        medianPacketTimesOut = np.median(packetTimesOut)
        stdevPacketTimesOut = np.std(packetTimesOut)
        variancePacketTimesOut = np.var(packetTimesOut)
        kurtosisPacketTimesOut = kurtosis(packetTimesOut)
        skewPacketTimesOut = skew(packetTimesOut)
        maxPacketTimesOut = np.amax(packetTimesOut)
        minPacketTimesOut = np.amin(packetTimesOut)
        p10PacketTimesOut = np.percentile(packetTimesOut,10)
        p20PacketTimesOut = np.percentile(packetTimesOut,20)
        p30PacketTimesOut = np.percentile(packetTimesOut,30)
        p40PacketTimesOut = np.percentile(packetTimesOut,40)
        p50PacketTimesOut = np.percentile(packetTimesOut,50)
        p60PacketTimesOut = np.percentile(packetTimesOut,60)
        p70PacketTimesOut = np.percentile(packetTimesOut,70)
        p80PacketTimesOut = np.percentile(packetTimesOut,80)
        p90PacketTimesOut = np.percentile(packetTimesOut,90)


        ########################################################################
        #Statistical indicators for Outgoing bursts

        out_totalBursts = len(out_bursts_packets)
        out_meanBurst = np.mean(out_bursts_packets)
        out_medianBurst = np.median(out_bursts_packets)
        out_stdevBurst = np.std(out_bursts_packets)
        out_varianceBurst = np.var(out_bursts_packets)
        out_maxBurst = np.amax(out_bursts_packets)
        out_kurtosisBurst = kurtosis(out_bursts_packets)
        out_skewBurst = skew(out_bursts_packets)
        out_p10Burst = np.percentile(out_bursts_packets,10)
        out_p20Burst = np.percentile(out_bursts_packets,20)
        out_p30Burst = np.percentile(out_bursts_packets,30)
        out_p40Burst = np.percentile(out_bursts_packets,40)
        out_p50Burst = np.percentile(out_bursts_packets,50)
        out_p60Burst = np.percentile(out_bursts_packets,60)
        out_p70Burst = np.percentile(out_bursts_packets,70)
        out_p80Burst = np.percentile(out_bursts_packets,80)
        out_p90Burst = np.percentile(out_bursts_packets,90)


        ########################################################################
        # Statistical indicators for Outgoing bytes (sliced intervals)
        out_meanBurstBytes = np.mean(out_burst_sizes)
        out_medianBurstBytes = np.median(out_burst_sizes)
        out_stdevBurstBytes = np.std(out_burst_sizes)
        out_varianceBurstBytes = np.var(out_burst_sizes)
        out_kurtosisBurstBytes = kurtosis(out_burst_sizes)
        out_skewBurstBytes = skew(out_burst_sizes)
        out_maxBurstBytes = np.amax(out_burst_sizes)
        out_minBurstBytes = np.amin(out_burst_sizes)
        out_p10BurstBytes = np.percentile(out_burst_sizes,10)
        out_p20BurstBytes = np.percentile(out_burst_sizes,20)
        out_p30BurstBytes = np.percentile(out_burst_sizes,30)
        out_p40BurstBytes = np.percentile(out_burst_sizes,40)
        out_p50BurstBytes = np.percentile(out_burst_sizes,50)
        out_p60BurstBytes = np.percentile(out_burst_sizes,60)
        out_p70BurstBytes = np.percentile(out_burst_sizes,70)
        out_p80BurstBytes = np.percentile(out_burst_sizes,80)
        out_p90BurstBytes = np.percentile(out_burst_sizes,90)

        ########################################################################
        #Statistical indicators for Incoming bursts

        in_totalBursts = len(in_bursts_packets)
        in_meanBurst = np.mean(in_bursts_packets)
        in_medianBurst = np.median(in_bursts_packets)
        in_stdevBurst = np.std(in_bursts_packets)
        in_varianceBurst = np.var(in_bursts_packets)
        in_maxBurst = np.amax(in_bursts_packets)
        in_kurtosisBurst = kurtosis(in_bursts_packets)
        in_skewBurst = skew(in_bursts_packets)
        in_p10Burst = np.percentile(in_bursts_packets,10)
        in_p20Burst = np.percentile(in_bursts_packets,20)
        in_p30Burst = np.percentile(in_bursts_packets,30)
        in_p40Burst = np.percentile(in_bursts_packets,40)
        in_p50Burst = np.percentile(in_bursts_packets,50)
        in_p60Burst = np.percentile(in_bursts_packets,60)
        in_p70Burst = np.percentile(in_bursts_packets,70)
        in_p80Burst = np.percentile(in_bursts_packets,80)
        in_p90Burst = np.percentile(in_bursts_packets,90)


        ########################################################################
        # Statistical indicators for Incoming burst bytes (sliced intervals)
        in_meanBurstBytes = np.mean(in_burst_sizes)
        in_medianBurstBytes = np.median(in_burst_sizes)
        in_stdevBurstBytes = np.std(in_burst_sizes)
        in_varianceBurstBytes = np.var(in_burst_sizes)
        in_kurtosisBurstBytes = kurtosis(in_burst_sizes)
        in_skewBurstBytes = skew(in_burst_sizes)
        in_maxBurstBytes = np.amax(in_burst_sizes)
        in_minBurstBytes = np.amin(in_burst_sizes)
        in_p10BurstBytes = np.percentile(in_burst_sizes,10)
        in_p20BurstBytes = np.percentile(in_burst_sizes,20)
        in_p30BurstBytes = np.percentile(in_burst_sizes,30)
        in_p40BurstBytes = np.percentile(in_burst_sizes,40)
        in_p50BurstBytes = np.percentile(in_burst_sizes,50)
        in_p60BurstBytes = np.percentile(in_burst_sizes,60)
        in_p70BurstBytes = np.percentile(in_burst_sizes,70)
        in_p80BurstBytes = np.percentile(in_burst_sizes,80)
        in_p90BurstBytes = np.percentile(in_burst_sizes,90)


        label = os.path.basename(sampleFolder)
        if('Regular' in sampleFolder):
            label = 'Regular'

        #Write sample features to the csv file
        f_names = []
        f_values = []


        for i, b in enumerate(bin_list):
            f_names.append('packetLengthBin_' + str(i))
            f_values.append(b)


        for i, b in enumerate(bin_list2):
            f_names.append('packetLengthBin2_' + str(i))
            f_values.append(b)

        """
        for i, b in enumerate(bigram_list):
            f_names.append('packetLengthBiGram_' + str(i))
            f_values.append(b)
        """


        """
        f_names.append('AvgBytesPerSecond')
        f_values.append(np.mean(BytesPerSecond))
        f_names.append('maxBytesPerSecond')
        f_values.append(np.amax(BytesPerSecond))
        f_names.append('minBytesPerSecond')
        f_values.append(np.amin(BytesPerSecond))
        f_names.append('stdDevBytesPerSecond')
        f_values.append(np.std(BytesPerSecond))
        f_names.append('medianBytesPerSecond')
        f_values.append(np.median(BytesPerSecond))
        f_names.append('firstQuartileBytesPerSecond')
        f_values.append(np.percentile(BytesPerSecond,25))
        f_names.append('thirdQuartileBytesPerSecond')
        f_values.append(np.percentile(BytesPerSecond,75))
        f_names.append('fivePercentileBytesPerSecond')
        f_values.append(np.percentile(BytesPerSecond,5))
        f_names.append('tenPercentileBytesPerSecond')
        f_values.append(np.percentile(BytesPerSecond,10))
        f_names.append('ninetyPercentileBytesPerSecond')
        f_values.append(np.percentile(BytesPerSecond,90))
        f_names.append('ninetyFivePercentileBytesPerSecond')
        f_values.append(np.percentile(BytesPerSecond,95))

        f_names.append('AvgPacketsPerSecond')
        f_values.append(np.mean(PacketsPerSecond))
        f_names.append('maxPacketsPerSecond')
        f_values.append(np.amax(PacketsPerSecond))
        f_names.append('minPacketsPerSecond')
        f_values.append(np.amin(PacketsPerSecond))
        f_names.append('stdDevPacketsPerSecond')
        f_values.append(np.std(PacketsPerSecond))
        f_names.append('medianPacketsPerSecond')
        f_values.append(np.median(PacketsPerSecond))
        f_names.append('firstQuartilePacketsPerSecond')
        f_values.append(np.percentile(PacketsPerSecond,25))
        f_names.append('thirdQuartilePacketsPerSecond')
        f_values.append(np.percentile(PacketsPerSecond,75))
        f_names.append('fivePercentilePacketsPerSecond')
        f_values.append(np.percentile(PacketsPerSecond,5))
        f_names.append('tenPercentilePacketsPerSecond')
        f_values.append(np.percentile(PacketsPerSecond,10))
        f_names.append('ninetyPercentilePacketsPerSecond')
        f_values.append(np.percentile(PacketsPerSecond,90))
        f_names.append('ninetyFivePercentilePacketsPerSecond')
        f_values.append(np.percentile(PacketsPerSecond,95))
        """

        """
        ###################################################################
        #Global Packet Features
        f_names.append('TotalPackets')
        f_values.append(totalPackets)
        f_names.append('totalPacketsIn')
        f_values.append(totalPacketsIn)
        f_names.append('totalPacketsOut')
        f_values.append(totalPacketsOut)
        f_names.append('totalBytes')
        f_values.append(totalBytes)
        f_names.append('totalBytesIn')
        f_values.append(totalBytesIn)
        f_names.append('totalBytesOut')
        f_values.append(totalBytesOut)

        ###################################################################
        #Packet Length Features
        f_names.append('minPacketSize')
        f_values.append(minPacketSize)
        f_names.append('maxPacketSize')
        f_values.append(maxPacketSize)
        #f_names.append('medianPacketSizes')
        #f_values.append(medianPacketSizes)
        f_names.append('meanPacketSizes')
        f_values.append(meanPacketSizes)
        f_names.append('stdevPacketSizes')
        f_values.append(stdevPacketSizes)
        f_names.append('variancePacketSizes')
        f_values.append(variancePacketSizes)
        f_names.append('kurtosisPacketSizes')
        f_values.append(kurtosisPacketSizes)
        f_names.append('skewPacketSizes')
        f_values.append(skewPacketSizes)

        f_names.append('p10PacketSizes')
        f_values.append(p10PacketSizes)
        f_names.append('p20PacketSizes')
        f_values.append(p20PacketSizes)
        f_names.append('p30PacketSizes')
        f_values.append(p30PacketSizes)
        f_names.append('p40PacketSizes')
        f_values.append(p40PacketSizes)
        f_names.append('p50PacketSizes')
        f_values.append(p50PacketSizes)
        f_names.append('p60PacketSizes')
        f_values.append(p60PacketSizes)
        f_names.append('p70PacketSizes')
        f_values.append(p70PacketSizes)
        f_names.append('p80PacketSizes')
        f_values.append(p80PacketSizes)
        f_names.append('p90PacketSizes')
        f_values.append(p90PacketSizes)


        ###################################################################
        #Packet Length Features (in)
        f_names.append('minPacketSizeIn')
        f_values.append(minPacketSizeIn)
        f_names.append('maxPacketSizeIn')
        f_values.append(maxPacketSizeIn)
        #f_names.append('medianPacketSizesIn')
        #f_values.append(medianPacketSizesIn)
        f_names.append('meanPacketSizesIn')
        f_values.append(meanPacketSizesIn)
        f_names.append('stdevPacketSizesIn')
        f_values.append(stdevPacketSizesIn)
        f_names.append('variancePacketSizesIn')
        f_values.append(variancePacketSizesIn)
        f_names.append('skewPacketSizesIn')
        f_values.append(skewPacketSizesIn)
        f_names.append('kurtosisPacketSizesIn')
        f_values.append(kurtosisPacketSizesIn)

        f_names.append('p10PacketSizesIn')
        f_values.append(p10PacketSizesIn)
        f_names.append('p20PacketSizesIn')
        f_values.append(p20PacketSizesIn)
        f_names.append('p30PacketSizesIn')
        f_values.append(p30PacketSizesIn)
        f_names.append('p40PacketSizesIn')
        f_values.append(p40PacketSizesIn)
        f_names.append('p50PacketSizesIn')
        f_values.append(p50PacketSizesIn)
        f_names.append('p60PacketSizesIn')
        f_values.append(p60PacketSizesIn)
        f_names.append('p70PacketSizesIn')
        f_values.append(p70PacketSizesIn)
        f_names.append('p80PacketSizesIn')
        f_values.append(p80PacketSizesIn)
        f_names.append('p90PacketSizesIn')
        f_values.append(p90PacketSizesIn)

        ###################################################################
        #Packet Length Features (out)
        f_names.append('minPacketSizeOut')
        f_values.append(minPacketSizeOut)
        f_names.append('maxPacketSizeOut')
        f_values.append(maxPacketSizeOut)
        #f_names.append('medianPacketSizesOut')
        #f_values.append(medianPacketSizesOut)
        f_names.append('meanPacketSizesOut')
        f_values.append(meanPacketSizesOut)
        f_names.append('stdevPacketSizesOut')
        f_values.append(stdevPacketSizesOut)
        f_names.append('variancePacketSizesOut')
        f_values.append(variancePacketSizesOut)
        f_names.append('skewPacketSizesOut')
        f_values.append(skewPacketSizesOut)
        f_names.append('kurtosisPacketSizesOut')
        f_values.append(kurtosisPacketSizesOut)

        f_names.append('p10PacketSizesOut')
        f_values.append(p10PacketSizesOut)
        f_names.append('p20PacketSizesOut')
        f_values.append(p20PacketSizesOut)
        f_names.append('p30PacketSizesOut')
        f_values.append(p30PacketSizesOut)
        f_names.append('p40PacketSizesOut')
        f_values.append(p40PacketSizesOut)
        f_names.append('p50PacketSizesOut')
        f_values.append(p50PacketSizesOut)
        f_names.append('p60PacketSizesOut')
        f_values.append(p60PacketSizesOut)
        f_names.append('p70PacketSizesOut')
        f_values.append(p70PacketSizesOut)
        f_names.append('p80PacketSizesOut')
        f_values.append(p80PacketSizesOut)
        f_names.append('p90PacketSizesOut')
        f_values.append(p90PacketSizesOut)


        ###################################################################
        #Packet Timing Features
        f_names.append('maxIPT')
        f_values.append(maxIPT)
        f_names.append('minIPT')
        f_values.append(minIPT)
        #f_names.append('medianPacketTimes')
        #f_values.append(medianPacketTimes)
        f_names.append('meanPacketTimes')
        f_values.append(meanPacketTimes)
        f_names.append('stdevPacketTimes')
        f_values.append(stdevPacketTimes)
        f_names.append('variancePacketTimes')
        f_values.append(variancePacketTimes)
        f_names.append('kurtosisPacketTimes')
        f_values.append(kurtosisPacketTimes)
        f_names.append('skewPacketTimes')
        f_values.append(skewPacketTimes)

        f_names.append('p10PacketTimes')
        f_values.append(p10PacketTimes)
        f_names.append('p20PacketTimes')
        f_values.append(p20PacketTimes)
        f_names.append('p30PacketTimes')
        f_values.append(p30PacketTimes)
        f_names.append('p40PacketTimes')
        f_values.append(p40PacketTimes)
        f_names.append('p50PacketTimes')
        f_values.append(p50PacketTimes)
        f_names.append('p60PacketTimes')
        f_values.append(p60PacketTimes)
        f_names.append('p70PacketTimes')
        f_values.append(p70PacketTimes)
        f_names.append('p80PacketTimes')
        f_values.append(p80PacketTimes)
        f_names.append('p90PacketTimes')
        f_values.append(p90PacketTimes)


        ###################################################################
        #Packet Timing Features (in)
        f_names.append('minPacketTimesIn')
        f_values.append(minPacketTimesIn)
        f_names.append('maxPacketTimesIn')
        f_values.append(maxPacketTimesIn)
        #f_names.append('medianPacketTimesIn')
        #f_values.append(medianPacketTimesIn)
        f_names.append('meanPacketTimesIn')
        f_values.append(meanPacketTimesIn)
        f_names.append('stdevPacketTimesIn')
        f_values.append(stdevPacketTimesIn)
        f_names.append('variancePacketTimesIn')
        f_values.append(variancePacketTimesIn)
        f_names.append('skewPacketTimesIn')
        f_values.append(skewPacketTimesIn)
        f_names.append('kurtosisPacketTimesIn')
        f_values.append(kurtosisPacketTimesIn)

        f_names.append('p10PacketTimesIn')
        f_values.append(p10PacketTimesIn)
        f_names.append('p20PacketTimesIn')
        f_values.append(p20PacketTimesIn)
        f_names.append('p30PacketTimesIn')
        f_values.append(p30PacketTimesIn)
        f_names.append('p40PacketTimesIn')
        f_values.append(p40PacketTimesIn)
        f_names.append('p50PacketTimesIn')
        f_values.append(p50PacketTimesIn)
        f_names.append('p60PacketTimesIn')
        f_values.append(p60PacketTimesIn)
        f_names.append('p70PacketTimesIn')
        f_values.append(p70PacketTimesIn)
        f_names.append('p80PacketTimesIn')
        f_values.append(p80PacketTimesIn)
        f_names.append('p90PacketTimesIn')
        f_values.append(p90PacketTimesIn)


        ###################################################################
        #Packet Timing Features (out)
        f_names.append('minPacketTimesOut')
        f_values.append(minPacketTimesOut)
        f_names.append('maxPacketTimesOut')
        f_values.append(maxPacketTimesOut)
        #f_names.append('medianPacketTimesOut')
        #f_values.append(medianPacketTimesOut)
        f_names.append('meanPacketTimesOut')
        f_values.append(meanPacketTimesOut)
        f_names.append('stdevPacketTimesOut')
        f_values.append(stdevPacketTimesOut)
        f_names.append('variancePacketTimesOut')
        f_values.append(variancePacketTimesOut)
        f_names.append('skewPacketTimesOut')
        f_values.append(skewPacketTimesOut)
        f_names.append('kurtosisPacketTimesOut')
        f_values.append(kurtosisPacketTimesOut)

        f_names.append('p10PacketTimesOut')
        f_values.append(p10PacketTimesOut)
        f_names.append('p20PacketTimesOut')
        f_values.append(p20PacketTimesOut)
        f_names.append('p30PacketTimesOut')
        f_values.append(p30PacketTimesOut)
        f_names.append('p40PacketTimesOut')
        f_values.append(p40PacketTimesOut)
        f_names.append('p50PacketTimesOut')
        f_values.append(p50PacketTimesOut)
        f_names.append('p60PacketTimesOut')
        f_values.append(p60PacketTimesOut)
        f_names.append('p70PacketTimesOut')
        f_values.append(p70PacketTimesOut)
        f_names.append('p80PacketTimesOut')
        f_values.append(p80PacketTimesOut)
        f_names.append('p90PacketTimesOut')
        f_values.append(p90PacketTimesOut)


        #################################################################
        #Outgoing Packet number of Bursts features
        f_names.append('out_totalBursts')
        f_values.append(out_totalBursts)
        f_names.append('out_maxBurst')
        f_values.append(out_maxBurst)
        f_names.append('out_meanBurst')
        f_values.append(out_meanBurst)
        #f_names.append('out_medianBurst')
        #f_values.append(out_medianBurst)
        f_names.append('out_stdevBurst')
        f_values.append(out_stdevBurst)
        f_names.append('out_varianceBurst')
        f_values.append(out_varianceBurst)
        f_names.append('out_kurtosisBurst')
        f_values.append(out_kurtosisBurst)
        f_names.append('out_skewBurst')
        f_values.append(out_skewBurst)

        f_names.append('out_p10Burst')
        f_values.append(out_p10Burst)
        f_names.append('out_p20Burst')
        f_values.append(out_p20Burst)
        f_names.append('out_p30Burst')
        f_values.append(out_p30Burst)
        f_names.append('out_p40Burst')
        f_values.append(out_p40Burst)
        f_names.append('out_p50Burst')
        f_values.append(out_p50Burst)
        f_names.append('out_p60Burst')
        f_values.append(out_p60Burst)
        f_names.append('out_p70Burst')
        f_values.append(out_p70Burst)
        f_names.append('out_p80Burst')
        f_values.append(out_p80Burst)
        f_names.append('out_p90Burst')
        f_values.append(out_p90Burst)


        #################################################################
        #Outgoing Packet Bursts data size features
        f_names.append('out_maxBurstBytes')
        f_values.append(out_maxBurstBytes)
        f_names.append('out_minBurstBytes')
        f_values.append(out_minBurstBytes)
        f_names.append('out_meanBurstBytes')
        f_values.append(out_meanBurstBytes)
        #f_names.append('out_medianBurstBytes')
        #f_values.append(out_medianBurstBytes)
        f_names.append('out_stdevBurstBytes')
        f_values.append(out_stdevBurstBytes)
        f_names.append('out_varianceBurstBytes')
        f_values.append(out_varianceBurstBytes)
        f_names.append('out_kurtosisBurstBytes')
        f_values.append(out_kurtosisBurstBytes)
        f_names.append('out_skewBurstBytes')
        f_values.append(out_skewBurstBytes)

        f_names.append('out_p10BurstBytes')
        f_values.append(out_p10BurstBytes)
        f_names.append('out_p20BurstBytes')
        f_values.append(out_p20BurstBytes)
        f_names.append('out_p30BurstBytes')
        f_values.append(out_p30BurstBytes)
        f_names.append('out_p40BurstBytes')
        f_values.append(out_p40BurstBytes)
        f_names.append('out_p50BurstBytes')
        f_values.append(out_p50BurstBytes)
        f_names.append('out_p60BurstBytes')
        f_values.append(out_p60BurstBytes)
        f_names.append('out_p70BurstBytes')
        f_values.append(out_p70BurstBytes)
        f_names.append('out_p80BurstBytes')
        f_values.append(out_p80BurstBytes)
        f_names.append('out_p90BurstBytes')
        f_values.append(out_p90BurstBytes)

        #################################################################
        #Incoming Packet number of Bursts features
        f_names.append('in_totalBursts')
        f_values.append(in_totalBursts)
        f_names.append('in_maxBurst')
        f_values.append(in_maxBurst)
        f_names.append('in_meanBurst')
        f_values.append(in_meanBurst)
        f_names.append('in_stdevBurst')
        f_values.append(in_stdevBurst)
        f_names.append('in_varianceBurst')
        f_values.append(in_varianceBurst)
        f_names.append('in_kurtosisBurst')
        f_values.append(in_kurtosisBurst)
        f_names.append('in_skewBurst')
        f_values.append(in_skewBurst)

        f_names.append('in_p10Burst')
        f_values.append(in_p10Burst)
        f_names.append('in_p20Burst')
        f_values.append(in_p20Burst)
        f_names.append('in_p30Burst')
        f_values.append(in_p30Burst)
        f_names.append('in_p40Burst')
        f_values.append(in_p40Burst)
        f_names.append('in_p50Burst')
        f_values.append(in_p50Burst)
        f_names.append('in_p60Burst')
        f_values.append(in_p60Burst)
        f_names.append('in_p70Burst')
        f_values.append(in_p70Burst)
        f_names.append('in_p80Burst')
        f_values.append(in_p80Burst)
        f_names.append('in_p90Burst')
        f_values.append(in_p90Burst)


        #################################################################
        #Incoming Packet Bursts data size features
        f_names.append('in_maxBurstBytes')
        f_values.append(in_maxBurstBytes)
        f_names.append('in_minBurstBytes')
        f_values.append(in_minBurstBytes)
        f_names.append('in_meanBurstBytes')
        f_values.append(in_meanBurstBytes)
        #f_names.append('in_medianBurstBytes')
        #f_values.append(in_medianBurstBytes)
        f_names.append('in_stdevBurstBytes')
        f_values.append(in_stdevBurstBytes)
        f_names.append('in_varianceBurstBytes')
        f_values.append(in_varianceBurstBytes)
        f_names.append('in_kurtosisBurstBytes')
        f_values.append(in_kurtosisBurstBytes)
        f_names.append('in_skewBurstBytes')
        f_values.append(in_skewBurstBytes)

        f_names.append('in_p10BurstBytes')
        f_values.append(in_p10BurstBytes)
        f_names.append('in_p20BurstBytes')
        f_values.append(in_p20BurstBytes)
        f_names.append('in_p30BurstBytes')
        f_values.append(in_p30BurstBytes)
        f_names.append('in_p40BurstBytes')
        f_values.append(in_p40BurstBytes)
        f_names.append('in_p50BurstBytes')
        f_values.append(in_p50BurstBytes)
        f_names.append('in_p60BurstBytes')
        f_values.append(in_p60BurstBytes)
        f_names.append('in_p70BurstBytes')
        f_values.append(in_p70BurstBytes)
        f_names.append('in_p80BurstBytes')
        f_values.append(in_p80BurstBytes)
        f_names.append('in_p90BurstBytes')
        f_values.append(in_p90BurstBytes)
        """

        print(len(f_names))
        f_names.append('Class')
        f_values.append(label)

        if(not written_header):
            arff.write(', '.join(f_names))
            arff.write('\n')
            print("Writing header")
            written_header = True

        l = []
        for v in f_values:
            l.append(str(v))
        arff.write(', '.join(l))
        arff.write('\n')
    arff.close()

def FeatureExtractionStatsBenchmark(sampleFolder):
    traceInterval = 60 #Amount of time in packet trace to consider for feature extraction

    feature_set_folder = 'FeatureSets/Stats_' + str(traceInterval)

    if not os.path.exists(feature_set_folder):
                os.makedirs(feature_set_folder)
    arff_path = feature_set_folder + '/' + os.path.basename(sampleFolder) + '_dataset.csv'
    arff = open(arff_path, 'wb')
    written_header = False

    start_time = time.time()
    sample_times = []

    for sample in os.listdir(sampleFolder):
        start_sample_time = time.time()
        f = open(sampleFolder + "/" + sample + "/" + sample)
        pcap = dpkt.pcap.Reader(f)

        #Analyse packets transmited
        totalPackets = 0
        totalPacketsIn = 0
        totalPacketsOut = 0

        #Analyse bytes transmitted
        totalBytes = 0
        totalBytesIn = 0
        totalBytesOut = 0

        #Analyse packet sizes
        packetSizes = []
        packetSizesIn = []
        packetSizesOut = []

        #Analyse inter packet timing
        packetTimes = []
        packetTimesIn = []
        packetTimesOut = []

        #Analyse outcoming bursts
        out_bursts_packets = []
        out_burst_sizes = []
        out_burst_times = []
        out_burst_start = 0
        out_current_burst = 0
        out_current_burst_start = 0
        out_current_burst_size = 0
        out_current_burst_time = 0

        #Analyse incoming bursts
        in_bursts_packets = []
        in_burst_sizes = []
        in_burst_times = []
        in_burst_start = 0
        in_current_burst = 0
        in_current_burst_start = 0
        in_current_burst_size = 0
        in_current_burst_time = 0

        prev_ts = 0
        absTimesOut = []
        firstTime = 0.0
        setFirst = False
        for ts, buf in pcap:
            if(not(setFirst)):
                firstTime = ts
                setFirst = True

            if(ts < (firstTime + traceInterval)):

                eth = dpkt.ethernet.Ethernet(buf)
                ip_hdr = eth.data
                try:
                    src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                    dst_ip_addr_str = socket.inet_ntoa(ip_hdr.dst)
                    #Target UDP communication between both cluster machines
                    if (ip_hdr.p == 17 and ((dst_ip_addr_str == SOURCE_IP) or (src_ip_addr_str == SOURCE_IP))):
                        #General packet statistics
                        totalPackets += 1
                        #if(src_ip_addr_str == DEST_IP):
                        if(src_ip_addr_str != SOURCE_IP):
                            totalPacketsIn += 1
                            packetSizesIn.append(len(buf))

                            if(prev_ts != 0):
                                ts_difference = ts - prev_ts
                                packetTimesIn.append(ts_difference)

                            if(out_current_burst != 0):
                                if(out_current_burst > 1):
                                    out_bursts_packets.append(out_current_burst) #packets on burst
                                    out_burst_sizes.append(out_current_burst_size) #total bytes on burst
                                    out_burst_times.append(ts - out_current_burst_start)
                                out_current_burst = 0
                                out_current_burst_size = 0
                                out_current_burst_start = 0
                            if(in_current_burst == 0):
                                in_current_burst_start = ts
                            in_current_burst += 1
                            in_current_burst_size += len(buf)
                        else:
                            totalPacketsOut += 1
                            absTimesOut.append(ts)
                            packetSizesOut.append(len(buf))
                            if(prev_ts != 0):
                                ts_difference = ts - prev_ts
                                packetTimesOut.append(ts_difference)
                            if(out_current_burst == 0):
                                out_current_burst_start = ts
                            out_current_burst += 1
                            out_current_burst_size += len(buf)

                            if(in_current_burst != 0):
                                if(in_current_burst > 1):
                                    in_bursts_packets.append(out_current_burst) #packets on burst
                                    in_burst_sizes.append(out_current_burst_size) #total bytes on burst
                                    in_burst_times.append(ts - out_current_burst_start)
                                in_current_burst = 0
                                in_current_burst_size = 0
                                in_current_burst_start = 0


                        #Bytes transmitted statistics
                        totalBytes += len(buf)
                        #if(src_ip_addr_str == DEST_IP):
                        if(src_ip_addr_str != SOURCE_IP):
                            totalBytesIn += len(buf)
                        else:
                            totalBytesOut += len(buf)

                        #Packet Size statistics
                        packetSizes.append(len(buf))

                        #Packet Times statistics
                        if(prev_ts != 0):
                            #print "{0:.6f}".format(ts)
                            ts_difference = ts - prev_ts
                            packetTimes.append(ts_difference)

                        prev_ts = ts
                except:
                    pass
        f.close()


        ################################################################
        ####################Compute statistics#####################
        ################################################################

        ##########################################################
        #Statistical indicators for packet sizes (total)
        meanPacketSizes = np.mean(packetSizes)
        medianPacketSizes = np.median(packetSizes)
        stdevPacketSizes = np.std(packetSizes)
        variancePacketSizes = np.var(packetSizes)
        kurtosisPacketSizes = kurtosis(packetSizes)
        skewPacketSizes = skew(packetSizes)
        maxPacketSize = np.amax(packetSizes)
        minPacketSize = np.amin(packetSizes)
        p10PacketSizes = np.percentile(packetSizes,10)
        p20PacketSizes = np.percentile(packetSizes,20)
        p30PacketSizes = np.percentile(packetSizes,30)
        p40PacketSizes = np.percentile(packetSizes,40)
        p50PacketSizes = np.percentile(packetSizes,50)
        p60PacketSizes = np.percentile(packetSizes,60)
        p70PacketSizes = np.percentile(packetSizes,70)
        p80PacketSizes = np.percentile(packetSizes,80)
        p90PacketSizes = np.percentile(packetSizes,90)


        ##########################################################
        #Statistical indicators for packet sizes (in)
        meanPacketSizesIn = np.mean(packetSizesIn)
        medianPacketSizesIn = np.median(packetSizesIn)
        stdevPacketSizesIn = np.std(packetSizesIn)
        variancePacketSizesIn = np.var(packetSizesIn)
        kurtosisPacketSizesIn = kurtosis(packetSizesIn)
        skewPacketSizesIn = skew(packetSizesIn)
        maxPacketSizeIn = np.amax(packetSizesIn)
        minPacketSizeIn = np.amin(packetSizesIn)
        p10PacketSizesIn = np.percentile(packetSizesIn,10)
        p20PacketSizesIn = np.percentile(packetSizesIn,20)
        p30PacketSizesIn = np.percentile(packetSizesIn,30)
        p40PacketSizesIn = np.percentile(packetSizesIn,40)
        p50PacketSizesIn = np.percentile(packetSizesIn,50)
        p60PacketSizesIn = np.percentile(packetSizesIn,60)
        p70PacketSizesIn = np.percentile(packetSizesIn,70)
        p80PacketSizesIn = np.percentile(packetSizesIn,80)
        p90PacketSizesIn = np.percentile(packetSizesIn,90)


        ##########################################################
        #Statistical indicators for packet sizes (out)
        meanPacketSizesOut = np.mean(packetSizesOut)
        medianPacketSizesOut = np.median(packetSizesOut)
        stdevPacketSizesOut = np.std(packetSizesOut)
        variancePacketSizesOut = np.var(packetSizesOut)
        kurtosisPacketSizesOut = kurtosis(packetSizesOut)
        skewPacketSizesOut = skew(packetSizesOut)
        maxPacketSizeOut = np.amax(packetSizesOut)
        minPacketSizeOut = np.amin(packetSizesOut)
        p10PacketSizesOut = np.percentile(packetSizesOut,10)
        p20PacketSizesOut = np.percentile(packetSizesOut,20)
        p30PacketSizesOut = np.percentile(packetSizesOut,30)
        p40PacketSizesOut = np.percentile(packetSizesOut,40)
        p50PacketSizesOut = np.percentile(packetSizesOut,50)
        p60PacketSizesOut = np.percentile(packetSizesOut,60)
        p70PacketSizesOut = np.percentile(packetSizesOut,70)
        p80PacketSizesOut = np.percentile(packetSizesOut,80)
        p90PacketSizesOut = np.percentile(packetSizesOut,90)

        ##################################################################
        #Statistical indicators for Inter-Packet Times (total)

        meanPacketTimes = np.mean(packetTimes)
        medianPacketTimes = np.median(packetTimes)
        stdevPacketTimes = np.std(packetTimes)
        variancePacketTimes = np.var(packetTimes)
        kurtosisPacketTimes = kurtosis(packetTimes)
        skewPacketTimes = skew(packetTimes)
        maxIPT = np.amax(packetTimes)
        minIPT = np.amin(packetTimes)
        p10PacketTimes = np.percentile(packetTimes,10)
        p20PacketTimes = np.percentile(packetTimes,20)
        p30PacketTimes = np.percentile(packetTimes,30)
        p40PacketTimes = np.percentile(packetTimes,40)
        p50PacketTimes = np.percentile(packetTimes,50)
        p60PacketTimes = np.percentile(packetTimes,60)
        p70PacketTimes = np.percentile(packetTimes,70)
        p80PacketTimes = np.percentile(packetTimes,80)
        p90PacketTimes = np.percentile(packetTimes,90)


        ##################################################################
        #Statistical indicators for Inter-Packet Times (in)
        meanPacketTimesIn = np.mean(packetTimesIn)
        medianPacketTimesIn = np.median(packetTimesIn)
        stdevPacketTimesIn = np.std(packetTimesIn)
        variancePacketTimesIn = np.var(packetTimesIn)
        kurtosisPacketTimesIn = kurtosis(packetTimesIn)
        skewPacketTimesIn = skew(packetTimesIn)
        maxPacketTimesIn = np.amax(packetTimesIn)
        minPacketTimesIn = np.amin(packetTimesIn)
        p10PacketTimesIn = np.percentile(packetTimesIn,10)
        p20PacketTimesIn = np.percentile(packetTimesIn,20)
        p30PacketTimesIn = np.percentile(packetTimesIn,30)
        p40PacketTimesIn = np.percentile(packetTimesIn,40)
        p50PacketTimesIn = np.percentile(packetTimesIn,50)
        p60PacketTimesIn = np.percentile(packetTimesIn,60)
        p70PacketTimesIn = np.percentile(packetTimesIn,70)
        p80PacketTimesIn = np.percentile(packetTimesIn,80)
        p90PacketTimesIn = np.percentile(packetTimesIn,90)


        ##################################################################
        #Statistical indicators for Inter-Packet Times (out)
        meanPacketTimesOut = np.mean(packetTimesOut)
        medianPacketTimesOut = np.median(packetTimesOut)
        stdevPacketTimesOut = np.std(packetTimesOut)
        variancePacketTimesOut = np.var(packetTimesOut)
        kurtosisPacketTimesOut = kurtosis(packetTimesOut)
        skewPacketTimesOut = skew(packetTimesOut)
        maxPacketTimesOut = np.amax(packetTimesOut)
        minPacketTimesOut = np.amin(packetTimesOut)
        p10PacketTimesOut = np.percentile(packetTimesOut,10)
        p20PacketTimesOut = np.percentile(packetTimesOut,20)
        p30PacketTimesOut = np.percentile(packetTimesOut,30)
        p40PacketTimesOut = np.percentile(packetTimesOut,40)
        p50PacketTimesOut = np.percentile(packetTimesOut,50)
        p60PacketTimesOut = np.percentile(packetTimesOut,60)
        p70PacketTimesOut = np.percentile(packetTimesOut,70)
        p80PacketTimesOut = np.percentile(packetTimesOut,80)
        p90PacketTimesOut = np.percentile(packetTimesOut,90)


        ########################################################################
        #Statistical indicators for Outgoing bursts

        out_totalBursts = len(out_bursts_packets)
        out_meanBurst = np.mean(out_bursts_packets)
        out_medianBurst = np.median(out_bursts_packets)
        out_stdevBurst = np.std(out_bursts_packets)
        out_varianceBurst = np.var(out_bursts_packets)
        out_maxBurst = np.amax(out_bursts_packets)
        out_kurtosisBurst = kurtosis(out_bursts_packets)
        out_skewBurst = skew(out_bursts_packets)
        out_p10Burst = np.percentile(out_bursts_packets,10)
        out_p20Burst = np.percentile(out_bursts_packets,20)
        out_p30Burst = np.percentile(out_bursts_packets,30)
        out_p40Burst = np.percentile(out_bursts_packets,40)
        out_p50Burst = np.percentile(out_bursts_packets,50)
        out_p60Burst = np.percentile(out_bursts_packets,60)
        out_p70Burst = np.percentile(out_bursts_packets,70)
        out_p80Burst = np.percentile(out_bursts_packets,80)
        out_p90Burst = np.percentile(out_bursts_packets,90)


        ########################################################################
        # Statistical indicators for Outgoing bytes (sliced intervals)
        out_meanBurstBytes = np.mean(out_burst_sizes)
        out_medianBurstBytes = np.median(out_burst_sizes)
        out_stdevBurstBytes = np.std(out_burst_sizes)
        out_varianceBurstBytes = np.var(out_burst_sizes)
        out_kurtosisBurstBytes = kurtosis(out_burst_sizes)
        out_skewBurstBytes = skew(out_burst_sizes)
        out_maxBurstBytes = np.amax(out_burst_sizes)
        out_minBurstBytes = np.amin(out_burst_sizes)
        out_p10BurstBytes = np.percentile(out_burst_sizes,10)
        out_p20BurstBytes = np.percentile(out_burst_sizes,20)
        out_p30BurstBytes = np.percentile(out_burst_sizes,30)
        out_p40BurstBytes = np.percentile(out_burst_sizes,40)
        out_p50BurstBytes = np.percentile(out_burst_sizes,50)
        out_p60BurstBytes = np.percentile(out_burst_sizes,60)
        out_p70BurstBytes = np.percentile(out_burst_sizes,70)
        out_p80BurstBytes = np.percentile(out_burst_sizes,80)
        out_p90BurstBytes = np.percentile(out_burst_sizes,90)

        ########################################################################
        #Statistical indicators for Incoming bursts

        in_totalBursts = len(in_bursts_packets)
        in_meanBurst = np.mean(in_bursts_packets)
        in_medianBurst = np.median(in_bursts_packets)
        in_stdevBurst = np.std(in_bursts_packets)
        in_varianceBurst = np.var(in_bursts_packets)
        in_maxBurst = np.amax(in_bursts_packets)
        in_kurtosisBurst = kurtosis(in_bursts_packets)
        in_skewBurst = skew(in_bursts_packets)
        in_p10Burst = np.percentile(in_bursts_packets,10)
        in_p20Burst = np.percentile(in_bursts_packets,20)
        in_p30Burst = np.percentile(in_bursts_packets,30)
        in_p40Burst = np.percentile(in_bursts_packets,40)
        in_p50Burst = np.percentile(in_bursts_packets,50)
        in_p60Burst = np.percentile(in_bursts_packets,60)
        in_p70Burst = np.percentile(in_bursts_packets,70)
        in_p80Burst = np.percentile(in_bursts_packets,80)
        in_p90Burst = np.percentile(in_bursts_packets,90)


        ########################################################################
        # Statistical indicators for Incoming burst bytes (sliced intervals)
        in_meanBurstBytes = np.mean(in_burst_sizes)
        in_medianBurstBytes = np.median(in_burst_sizes)
        in_stdevBurstBytes = np.std(in_burst_sizes)
        in_varianceBurstBytes = np.var(in_burst_sizes)
        in_kurtosisBurstBytes = kurtosis(in_burst_sizes)
        in_skewBurstBytes = skew(in_burst_sizes)
        in_maxBurstBytes = np.amax(in_burst_sizes)
        in_minBurstBytes = np.amin(in_burst_sizes)
        in_p10BurstBytes = np.percentile(in_burst_sizes,10)
        in_p20BurstBytes = np.percentile(in_burst_sizes,20)
        in_p30BurstBytes = np.percentile(in_burst_sizes,30)
        in_p40BurstBytes = np.percentile(in_burst_sizes,40)
        in_p50BurstBytes = np.percentile(in_burst_sizes,50)
        in_p60BurstBytes = np.percentile(in_burst_sizes,60)
        in_p70BurstBytes = np.percentile(in_burst_sizes,70)
        in_p80BurstBytes = np.percentile(in_burst_sizes,80)
        in_p90BurstBytes = np.percentile(in_burst_sizes,90)


        label = os.path.basename(sampleFolder)
        if('Regular' in sampleFolder):
            label = 'Regular'

        #Write sample features to the csv file
        f_names = []
        f_values = []

        ###################################################################
        #Global Packet Features
        f_names.append('TotalPackets')
        f_values.append(totalPackets)
        f_names.append('totalPacketsIn')
        f_values.append(totalPacketsIn)
        f_names.append('totalPacketsOut')
        f_values.append(totalPacketsOut)
        f_names.append('totalBytes')
        f_values.append(totalBytes)
        f_names.append('totalBytesIn')
        f_values.append(totalBytesIn)
        f_names.append('totalBytesOut')
        f_values.append(totalBytesOut)

        ###################################################################
        #Packet Length Features
        f_names.append('minPacketSize')
        f_values.append(minPacketSize)
        f_names.append('maxPacketSize')
        f_values.append(maxPacketSize)
        #f_names.append('medianPacketSizes')
        #f_values.append(medianPacketSizes)
        f_names.append('meanPacketSizes')
        f_values.append(meanPacketSizes)
        f_names.append('stdevPacketSizes')
        f_values.append(stdevPacketSizes)
        f_names.append('variancePacketSizes')
        f_values.append(variancePacketSizes)
        f_names.append('kurtosisPacketSizes')
        f_values.append(kurtosisPacketSizes)
        f_names.append('skewPacketSizes')
        f_values.append(skewPacketSizes)

        f_names.append('p10PacketSizes')
        f_values.append(p10PacketSizes)
        f_names.append('p20PacketSizes')
        f_values.append(p20PacketSizes)
        f_names.append('p30PacketSizes')
        f_values.append(p30PacketSizes)
        f_names.append('p40PacketSizes')
        f_values.append(p40PacketSizes)
        f_names.append('p50PacketSizes')
        f_values.append(p50PacketSizes)
        f_names.append('p60PacketSizes')
        f_values.append(p60PacketSizes)
        f_names.append('p70PacketSizes')
        f_values.append(p70PacketSizes)
        f_names.append('p80PacketSizes')
        f_values.append(p80PacketSizes)
        f_names.append('p90PacketSizes')
        f_values.append(p90PacketSizes)


        ###################################################################
        #Packet Length Features (in)
        f_names.append('minPacketSizeIn')
        f_values.append(minPacketSizeIn)
        f_names.append('maxPacketSizeIn')
        f_values.append(maxPacketSizeIn)
        #f_names.append('medianPacketSizesIn')
        #f_values.append(medianPacketSizesIn)
        f_names.append('meanPacketSizesIn')
        f_values.append(meanPacketSizesIn)
        f_names.append('stdevPacketSizesIn')
        f_values.append(stdevPacketSizesIn)
        f_names.append('variancePacketSizesIn')
        f_values.append(variancePacketSizesIn)
        f_names.append('skewPacketSizesIn')
        f_values.append(skewPacketSizesIn)
        f_names.append('kurtosisPacketSizesIn')
        f_values.append(kurtosisPacketSizesIn)

        f_names.append('p10PacketSizesIn')
        f_values.append(p10PacketSizesIn)
        f_names.append('p20PacketSizesIn')
        f_values.append(p20PacketSizesIn)
        f_names.append('p30PacketSizesIn')
        f_values.append(p30PacketSizesIn)
        f_names.append('p40PacketSizesIn')
        f_values.append(p40PacketSizesIn)
        f_names.append('p50PacketSizesIn')
        f_values.append(p50PacketSizesIn)
        f_names.append('p60PacketSizesIn')
        f_values.append(p60PacketSizesIn)
        f_names.append('p70PacketSizesIn')
        f_values.append(p70PacketSizesIn)
        f_names.append('p80PacketSizesIn')
        f_values.append(p80PacketSizesIn)
        f_names.append('p90PacketSizesIn')
        f_values.append(p90PacketSizesIn)

        ###################################################################
        #Packet Length Features (out)
        f_names.append('minPacketSizeOut')
        f_values.append(minPacketSizeOut)
        f_names.append('maxPacketSizeOut')
        f_values.append(maxPacketSizeOut)
        #f_names.append('medianPacketSizesOut')
        #f_values.append(medianPacketSizesOut)
        f_names.append('meanPacketSizesOut')
        f_values.append(meanPacketSizesOut)
        f_names.append('stdevPacketSizesOut')
        f_values.append(stdevPacketSizesOut)
        f_names.append('variancePacketSizesOut')
        f_values.append(variancePacketSizesOut)
        f_names.append('skewPacketSizesOut')
        f_values.append(skewPacketSizesOut)
        f_names.append('kurtosisPacketSizesOut')
        f_values.append(kurtosisPacketSizesOut)

        f_names.append('p10PacketSizesOut')
        f_values.append(p10PacketSizesOut)
        f_names.append('p20PacketSizesOut')
        f_values.append(p20PacketSizesOut)
        f_names.append('p30PacketSizesOut')
        f_values.append(p30PacketSizesOut)
        f_names.append('p40PacketSizesOut')
        f_values.append(p40PacketSizesOut)
        f_names.append('p50PacketSizesOut')
        f_values.append(p50PacketSizesOut)
        f_names.append('p60PacketSizesOut')
        f_values.append(p60PacketSizesOut)
        f_names.append('p70PacketSizesOut')
        f_values.append(p70PacketSizesOut)
        f_names.append('p80PacketSizesOut')
        f_values.append(p80PacketSizesOut)
        f_names.append('p90PacketSizesOut')
        f_values.append(p90PacketSizesOut)


        ###################################################################
        #Packet Timing Features
        f_names.append('maxIPT')
        f_values.append(maxIPT)
        f_names.append('minIPT')
        f_values.append(minIPT)
        #f_names.append('medianPacketTimes')
        #f_values.append(medianPacketTimes)
        f_names.append('meanPacketTimes')
        f_values.append(meanPacketTimes)
        f_names.append('stdevPacketTimes')
        f_values.append(stdevPacketTimes)
        f_names.append('variancePacketTimes')
        f_values.append(variancePacketTimes)
        f_names.append('kurtosisPacketTimes')
        f_values.append(kurtosisPacketTimes)
        f_names.append('skewPacketTimes')
        f_values.append(skewPacketTimes)

        f_names.append('p10PacketTimes')
        f_values.append(p10PacketTimes)
        f_names.append('p20PacketTimes')
        f_values.append(p20PacketTimes)
        f_names.append('p30PacketTimes')
        f_values.append(p30PacketTimes)
        f_names.append('p40PacketTimes')
        f_values.append(p40PacketTimes)
        f_names.append('p50PacketTimes')
        f_values.append(p50PacketTimes)
        f_names.append('p60PacketTimes')
        f_values.append(p60PacketTimes)
        f_names.append('p70PacketTimes')
        f_values.append(p70PacketTimes)
        f_names.append('p80PacketTimes')
        f_values.append(p80PacketTimes)
        f_names.append('p90PacketTimes')
        f_values.append(p90PacketTimes)


        ###################################################################
        #Packet Timing Features (in)
        f_names.append('minPacketTimesIn')
        f_values.append(minPacketTimesIn)
        f_names.append('maxPacketTimesIn')
        f_values.append(maxPacketTimesIn)
        #f_names.append('medianPacketTimesIn')
        #f_values.append(medianPacketTimesIn)
        f_names.append('meanPacketTimesIn')
        f_values.append(meanPacketTimesIn)
        f_names.append('stdevPacketTimesIn')
        f_values.append(stdevPacketTimesIn)
        f_names.append('variancePacketTimesIn')
        f_values.append(variancePacketTimesIn)
        f_names.append('skewPacketTimesIn')
        f_values.append(skewPacketTimesIn)
        f_names.append('kurtosisPacketTimesIn')
        f_values.append(kurtosisPacketTimesIn)

        f_names.append('p10PacketTimesIn')
        f_values.append(p10PacketTimesIn)
        f_names.append('p20PacketTimesIn')
        f_values.append(p20PacketTimesIn)
        f_names.append('p30PacketTimesIn')
        f_values.append(p30PacketTimesIn)
        f_names.append('p40PacketTimesIn')
        f_values.append(p40PacketTimesIn)
        f_names.append('p50PacketTimesIn')
        f_values.append(p50PacketTimesIn)
        f_names.append('p60PacketTimesIn')
        f_values.append(p60PacketTimesIn)
        f_names.append('p70PacketTimesIn')
        f_values.append(p70PacketTimesIn)
        f_names.append('p80PacketTimesIn')
        f_values.append(p80PacketTimesIn)
        f_names.append('p90PacketTimesIn')
        f_values.append(p90PacketTimesIn)


        ###################################################################
        #Packet Timing Features (out)
        f_names.append('minPacketTimesOut')
        f_values.append(minPacketTimesOut)
        f_names.append('maxPacketTimesOut')
        f_values.append(maxPacketTimesOut)
        #f_names.append('medianPacketTimesOut')
        #f_values.append(medianPacketTimesOut)
        f_names.append('meanPacketTimesOut')
        f_values.append(meanPacketTimesOut)
        f_names.append('stdevPacketTimesOut')
        f_values.append(stdevPacketTimesOut)
        f_names.append('variancePacketTimesOut')
        f_values.append(variancePacketTimesOut)
        f_names.append('skewPacketTimesOut')
        f_values.append(skewPacketTimesOut)
        f_names.append('kurtosisPacketTimesOut')
        f_values.append(kurtosisPacketTimesOut)

        f_names.append('p10PacketTimesOut')
        f_values.append(p10PacketTimesOut)
        f_names.append('p20PacketTimesOut')
        f_values.append(p20PacketTimesOut)
        f_names.append('p30PacketTimesOut')
        f_values.append(p30PacketTimesOut)
        f_names.append('p40PacketTimesOut')
        f_values.append(p40PacketTimesOut)
        f_names.append('p50PacketTimesOut')
        f_values.append(p50PacketTimesOut)
        f_names.append('p60PacketTimesOut')
        f_values.append(p60PacketTimesOut)
        f_names.append('p70PacketTimesOut')
        f_values.append(p70PacketTimesOut)
        f_names.append('p80PacketTimesOut')
        f_values.append(p80PacketTimesOut)
        f_names.append('p90PacketTimesOut')
        f_values.append(p90PacketTimesOut)


        #################################################################
        #Outgoing Packet number of Bursts features
        f_names.append('out_totalBursts')
        f_values.append(out_totalBursts)
        f_names.append('out_maxBurst')
        f_values.append(out_maxBurst)
        f_names.append('out_meanBurst')
        f_values.append(out_meanBurst)
        #f_names.append('out_medianBurst')
        #f_values.append(out_medianBurst)
        f_names.append('out_stdevBurst')
        f_values.append(out_stdevBurst)
        f_names.append('out_varianceBurst')
        f_values.append(out_varianceBurst)
        f_names.append('out_kurtosisBurst')
        f_values.append(out_kurtosisBurst)
        f_names.append('out_skewBurst')
        f_values.append(out_skewBurst)

        f_names.append('out_p10Burst')
        f_values.append(out_p10Burst)
        f_names.append('out_p20Burst')
        f_values.append(out_p20Burst)
        f_names.append('out_p30Burst')
        f_values.append(out_p30Burst)
        f_names.append('out_p40Burst')
        f_values.append(out_p40Burst)
        f_names.append('out_p50Burst')
        f_values.append(out_p50Burst)
        f_names.append('out_p60Burst')
        f_values.append(out_p60Burst)
        f_names.append('out_p70Burst')
        f_values.append(out_p70Burst)
        f_names.append('out_p80Burst')
        f_values.append(out_p80Burst)
        f_names.append('out_p90Burst')
        f_values.append(out_p90Burst)


        #################################################################
        #Outgoing Packet Bursts data size features
        f_names.append('out_maxBurstBytes')
        f_values.append(out_maxBurstBytes)
        f_names.append('out_minBurstBytes')
        f_values.append(out_minBurstBytes)
        f_names.append('out_meanBurstBytes')
        f_values.append(out_meanBurstBytes)
        #f_names.append('out_medianBurstBytes')
        #f_values.append(out_medianBurstBytes)
        f_names.append('out_stdevBurstBytes')
        f_values.append(out_stdevBurstBytes)
        f_names.append('out_varianceBurstBytes')
        f_values.append(out_varianceBurstBytes)
        f_names.append('out_kurtosisBurstBytes')
        f_values.append(out_kurtosisBurstBytes)
        f_names.append('out_skewBurstBytes')
        f_values.append(out_skewBurstBytes)

        f_names.append('out_p10BurstBytes')
        f_values.append(out_p10BurstBytes)
        f_names.append('out_p20BurstBytes')
        f_values.append(out_p20BurstBytes)
        f_names.append('out_p30BurstBytes')
        f_values.append(out_p30BurstBytes)
        f_names.append('out_p40BurstBytes')
        f_values.append(out_p40BurstBytes)
        f_names.append('out_p50BurstBytes')
        f_values.append(out_p50BurstBytes)
        f_names.append('out_p60BurstBytes')
        f_values.append(out_p60BurstBytes)
        f_names.append('out_p70BurstBytes')
        f_values.append(out_p70BurstBytes)
        f_names.append('out_p80BurstBytes')
        f_values.append(out_p80BurstBytes)
        f_names.append('out_p90BurstBytes')
        f_values.append(out_p90BurstBytes)

        #################################################################
        #Incoming Packet number of Bursts features
        f_names.append('in_totalBursts')
        f_values.append(in_totalBursts)
        f_names.append('in_maxBurst')
        f_values.append(in_maxBurst)
        f_names.append('in_meanBurst')
        f_values.append(in_meanBurst)
        f_names.append('in_stdevBurst')
        f_values.append(in_stdevBurst)
        f_names.append('in_varianceBurst')
        f_values.append(in_varianceBurst)
        f_names.append('in_kurtosisBurst')
        f_values.append(in_kurtosisBurst)
        f_names.append('in_skewBurst')
        f_values.append(in_skewBurst)

        f_names.append('in_p10Burst')
        f_values.append(in_p10Burst)
        f_names.append('in_p20Burst')
        f_values.append(in_p20Burst)
        f_names.append('in_p30Burst')
        f_values.append(in_p30Burst)
        f_names.append('in_p40Burst')
        f_values.append(in_p40Burst)
        f_names.append('in_p50Burst')
        f_values.append(in_p50Burst)
        f_names.append('in_p60Burst')
        f_values.append(in_p60Burst)
        f_names.append('in_p70Burst')
        f_values.append(in_p70Burst)
        f_names.append('in_p80Burst')
        f_values.append(in_p80Burst)
        f_names.append('in_p90Burst')
        f_values.append(in_p90Burst)


        #################################################################
        #Incoming Packet Bursts data size features
        f_names.append('in_maxBurstBytes')
        f_values.append(in_maxBurstBytes)
        f_names.append('in_minBurstBytes')
        f_values.append(in_minBurstBytes)
        f_names.append('in_meanBurstBytes')
        f_values.append(in_meanBurstBytes)
        #f_names.append('in_medianBurstBytes')
        #f_values.append(in_medianBurstBytes)
        f_names.append('in_stdevBurstBytes')
        f_values.append(in_stdevBurstBytes)
        f_names.append('in_varianceBurstBytes')
        f_values.append(in_varianceBurstBytes)
        f_names.append('in_kurtosisBurstBytes')
        f_values.append(in_kurtosisBurstBytes)
        f_names.append('in_skewBurstBytes')
        f_values.append(in_skewBurstBytes)

        f_names.append('in_p10BurstBytes')
        f_values.append(in_p10BurstBytes)
        f_names.append('in_p20BurstBytes')
        f_values.append(in_p20BurstBytes)
        f_names.append('in_p30BurstBytes')
        f_values.append(in_p30BurstBytes)
        f_names.append('in_p40BurstBytes')
        f_values.append(in_p40BurstBytes)
        f_names.append('in_p50BurstBytes')
        f_values.append(in_p50BurstBytes)
        f_names.append('in_p60BurstBytes')
        f_values.append(in_p60BurstBytes)
        f_names.append('in_p70BurstBytes')
        f_values.append(in_p70BurstBytes)
        f_names.append('in_p80BurstBytes')
        f_values.append(in_p80BurstBytes)
        f_names.append('in_p90BurstBytes')
        f_values.append(in_p90BurstBytes)

        #print np.array(f_values).nbytes
        #print np.array(f_values).itemsize
        #print len(f_names)
        
        f_names.append('Class')
        f_values.append(label)

        end_sample_time = time.time()
        sample_times.append(end_sample_time - start_sample_time)

        if(not written_header):
            arff.write(', '.join(f_names))
            arff.write('\n')
            print("Writing header")
            written_header = True

        l = []
        for v in f_values:
            l.append(str(v))
        arff.write(', '.join(l))
        arff.write('\n')
    arff.close()
    end_time = time.time()
    print("Total time elapsed: " + "{0:.5f}".format(end_time - start_time))
    print("Average sample time: "+ "{0:.5f}".format(np.mean(sample_times)))
    return feature_set_folder

def FeatureExtractionPLBenchmark(sampleFolder):
    
    traceInterval = 60 #Amount of time in packet trace to consider for feature extraction

    feature_set_folder = 'FeatureSets/PL_' + str(traceInterval)

    if not os.path.exists(feature_set_folder):
                os.makedirs(feature_set_folder)
    arff_path = feature_set_folder + '/' + os.path.basename(sampleFolder) + '_dataset.csv'
    arff = open(arff_path, 'wb')
    written_header = False

    start_time = time.time()
    sample_times = []

    for sample in os.listdir(sampleFolder):
        start_sample_time = time.time()
        f = open(sampleFolder + "/" + sample + "/" + sample)
        pcap = dpkt.pcap.Reader(f)

        #Analyse packets transmited
        packetSizesIn = []
        packetSizesOut = []
        bin_dict = {}
        bin_dict2 = {}
        binWidth = 5
        #Generate the set of all possible bins
        for i in range(0,1500, binWidth):
            bin_dict[i] = 0
            bin_dict2[i] = 0

        firstTime = 0.0
        setFirst = False
        for ts, buf in pcap:
            if(not(setFirst)):
                firstTime = ts
                setFirst = True

            if(ts < (firstTime + traceInterval)):

                eth = dpkt.ethernet.Ethernet(buf)
                ip_hdr = eth.data
                try:
                    src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                    dst_ip_addr_str = socket.inet_ntoa(ip_hdr.dst)
                    #Target UDP communication between both cluster machines
                    if (ip_hdr.p == 17 and src_ip_addr_str == SOURCE_IP):
                        binned = RoundToNearest(len(buf),binWidth)
                        bin_dict[binned]+=1
                    elif(ip_hdr.p == 17 and src_ip_addr_str != SOURCE_IP):
                        binned = RoundToNearest(len(buf),binWidth)
                        bin_dict2[binned]+=1
                except:
                    pass
        f.close()

        od_dict = collections.OrderedDict(sorted(list(bin_dict.items()), key=lambda t: float(t[0])))
        bin_list = []
        for i in od_dict:
            bin_list.append(od_dict[i])

        od_dict2 = collections.OrderedDict(sorted(list(bin_dict2.items()), key=lambda t: float(t[0])))
        bin_list2 = []
        for i in od_dict2:
            bin_list2.append(od_dict2[i])

        end_sample_time = time.time()
        #print "Mark intermediate processing time: " + "{0:.5f}".format(end_sample_time - start_sample_time)
        sample_times.append(end_sample_time - start_sample_time)
        #np.save("DeltaShaperTraffic_Web_300_Double/480Resolution/records/" + sample+ "_list", np.array(bin_list))
        #print np.array(bin_list).nbytes + np.array(bin_list2).nbytes
        label = os.path.basename(sampleFolder)
        if('Regular' in sampleFolder):
            label = 'Regular'

        #Write sample features to the csv file
        f_names = []
        f_values = []

        for i, b in enumerate(bin_list):
            f_names.append('packetLengthBin_' + str(i))
            f_values.append(b)

        for i, b in enumerate(bin_list2):
            f_names.append('packetLengthBin2_' + str(i))
            f_values.append(b)

        #print len(f_names)
        f_names.append('Class')
        f_values.append(label)

        if(not written_header):
            arff.write(', '.join(f_names))
            arff.write('\n')
            print("Writing header")
            written_header = True

        l = []
        for v in f_values:
            l.append(str(v))
        arff.write(', '.join(l))
        arff.write('\n')
        end_sample_time = time.time()
        #print "Sample processing time: " + "{0:.5f}".format(end_sample_time - start_sample_time)
    arff.close()
    end_time = time.time()
    print("Total time elapsed: " + "{0:.5f}".format(end_time - start_time))
    print("Average sample time: "+ "{0:.5f}".format(np.mean(sample_times)))
    return feature_set_folder

def FeatureExtractionPLBenchmark_reb(sampleFolder):
    start_time = time.time()
    arff = open(sampleFolder + '_dataset_reb.csv', 'wb')
    written_header = False
    sample_times = []

    for sample in os.listdir(sampleFolder):
        start_sample_time = time.time()
        f = open(sampleFolder + "/" + sample + "/" + sample)
        pcap = dpkt.pcap.Reader(f)

        #Analyse packets transmited
        packetSizesIn = []
        packetSizesOut = []
        bin_dict = {}
        bin_dict2 = {}
        binWidth = 5
        #Generate the set of all possible bins
        for i in range(0,1500, binWidth):
            bin_dict[i] = 0
            bin_dict2[i] = 0

        firstTime = 0.0
        setFirst = False
        for ts, buf in pcap:
            if(not(setFirst)):
                firstTime = ts
                setFirst = True

            if(ts < (firstTime + 1)):

                eth = dpkt.ethernet.Ethernet(buf)
                ip_hdr = eth.data
                try:
                    src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                    dst_ip_addr_str = socket.inet_ntoa(ip_hdr.dst)
                    #Target UDP communication between both cluster machines
                    if (ip_hdr.p == 17 and src_ip_addr_str == SOURCE_IP):
                        binned = RoundToNearest(len(buf),binWidth)
                        bin_dict[binned]+=1
                    elif(ip_hdr.p == 17 and src_ip_addr_str != SOURCE_IP):
                        binned = RoundToNearest(len(buf),binWidth)
                        bin_dict2[binned]+=1
                except:
                    pass
        f.close()

        od_dict = collections.OrderedDict(sorted(list(bin_dict.items()), key=lambda t: float(t[0])))
        bin_list = []
        for i in od_dict:
            bin_list.append(od_dict[i])

        od_dict2 = collections.OrderedDict(sorted(list(bin_dict2.items()), key=lambda t: float(t[0])))
        bin_list2 = []
        for i in od_dict2:
            bin_list2.append(od_dict2[i])

        end_sample_time = time.time()
        #print "Mark intermediate processing time: " + "{0:.5f}".format(end_sample_time - start_sample_time)
        sample_times.append(end_sample_time - start_sample_time)
        #np.save("DeltaShaperTraffic_Web_300_Double/480Resolution/records/" + sample+ "_list", np.array(bin_list))
        #print np.array(bin_list).nbytes + np.array(bin_list2).nbytes
        label = os.path.basename(sampleFolder)
        if('Regular' in sampleFolder):
            label = 'Regular'

        #Write sample features to the csv file
        f_names = []
        f_values = []

        for i, b in enumerate(bin_list):
            f_names.append('packetLengthBin_' + str(i))
            f_values.append(b)

        for i, b in enumerate(bin_list2):
            f_names.append('packetLengthBin2_' + str(i))
            f_values.append(b)

        #print len(f_names)
        f_names.append('Class')
        f_values.append(label)

        if(not written_header):
            arff.write(', '.join(f_names))
            arff.write('\n')
            print("Writing header")
            written_header = True

        l = []
        for v in f_values:
            l.append(str(v))
        arff.write(', '.join(l))
        arff.write('\n')
        end_sample_time = time.time()
        #print "Sample processing time: " + "{0:.5f}".format(end_sample_time - start_sample_time)
    arff.close()
    end_time = time.time()
    print("Total time elapsed: " + "{0:.5f}".format(end_time - start_time))
    print("Average sample time: "+ "{0:.5f}".format(np.mean(sample_times)))



if __name__ == "__main__":
    sampleFolders = [
    "TrafficCaptures/480Resolution/DeltaShaperTraffic_320",
    "TrafficCaptures/480Resolution/DeltaShaperTraffic_160",
    "TrafficCaptures/480Resolution/RegularTraffic",
    ]
    

    if not os.path.exists('FeatureSets'):
                os.makedirs('FeatureSets')
    
    print("Generating Dataset based on Summary Statistic Features")
    for sampleFolder in sampleFolders:
        print("\n#############################")
        print("Parsing " + sampleFolder)
        print("#############################")
        feature_set_folder = FeatureExtractionStatsBenchmark(sampleFolder)
    GenerateDatasets(feature_set_folder + '/')
    

    print("Generating Dataset based on Binned Packet Length Features")
    for sampleFolder in sampleFolders:
        print("\n#############################")
        print("Parsing " + sampleFolder)
        print("#############################")
        feature_set_folder = FeatureExtractionPLBenchmark(sampleFolder)
    GenerateDatasets(feature_set_folder + '/')