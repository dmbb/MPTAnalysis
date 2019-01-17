import dpkt
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import socket

BIN_WIDTH = [15,20,50]

InterPacketBins = [5000,2500,1000]


auxFolder = 'auxFolder/'

def RoundToNearest(n, m):
        r = n % m
        return n + m - r if r + r >= m else n - r

def CreateBigrams(capsFolder, sampleFolder):
    for sample in os.listdir(capsFolder + sampleFolder):
        for binWidth in BIN_WIDTH:
            faux = open(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample + '/bigrams_' + str(binWidth), 'w')
            f = open(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample + "/packetCount_" + str(binWidth), 'r')

            lines = f.readlines()
            for index, line in enumerate(lines):
                try:
                    faux.write(line.rstrip('\n') + "," + lines[index+1])
                except IndexError:
                    break #Reached last index, stop processing
            faux.close()
            f.close()


def ComputeDelta(capsFolder, sampleFolder):
    for sample in os.listdir(capsFolder + sampleFolder):
        for binWidth in InterPacketBins:
            faux = open(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample + '/deltaT_' + str(binWidth), 'w')
            f = open(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample + "/timestamps", 'r')

            lines = f.readlines()
            for index, line in enumerate(lines):
                try:
                    delta = "%0.6f" % (float(lines[index+1]) - (float(line.rstrip('\n'))))
                    delta = float(delta) * 1000000
                    faux.write("%s\n" % RoundToNearest(int(delta), int(binWidth)))
                except IndexError:
                    break #Reached last index, stop processing

            faux.close()
            f.close()

def ParseCapture(capsFolder, sampleFolder):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for sample in os.listdir(capsFolder + sampleFolder):
        descriptors = []
        f = open(capsFolder + sampleFolder + "/" + sample + "/" + sample)
        print sample
        if not os.path.exists(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample):
        	os.makedirs(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample)
        packet_count = open(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample + '/packetCount_0', 'w')
        timestamps = open(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample + "/timestamps", 'w')
        for binWidth in BIN_WIDTH:
            descriptors.append(open(auxFolder + os.path.dirname(capsFolder) + "/" + sampleFolder + "/" + sample + '/packetCount_' + str(binWidth), 'w'))


        pcap = dpkt.pcap.Reader(f)

        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            ip_hdr = eth.data
            try:
                if (eth.type != dpkt.ethernet.ETH_TYPE_IP and eth.type != dpkt.ethernet.ETH_TYPE_IP6):
                    continue
                if eth.type != dpkt.ethernet.ETH_TYPE_IP6:
                    src_ip_addr_str = socket.inet_ntoa(ip_hdr.src)
                else:
                    src_ip_addr_str = socket.inet_ntop(socket.AF_INET6, ip_hdr.src)

                if (ip_hdr.p == 17 and src_ip_addr_str == '172.31.0.19'):
                    for i, descript in enumerate(BIN_WIDTH):
                        descriptors[i].write("%s\n" % RoundToNearest(len(buf), BIN_WIDTH[i]))
                    timestamps.write("{0:.6f}".format(ts) + "\n")
                    packet_count.write("%s\n" % len(buf))

            except Exception as e:
                print "[Exception]" + str(e)
        packet_count.close()
        timestamps.close()
        for i, descript in enumerate(BIN_WIDTH):
            descriptors[i].close()
        f.close()


if __name__ == "__main__":
    sampleFolders = ["TrafficCaptures/480Resolution/"]
    modeFolders = ["RegularTraffic","DeltaShaperTraffic_320", "DeltaShaperTraffic_160"]

    for sampleFolder in sampleFolders:
        for modeFolder in modeFolders:
            if not os.path.exists(auxFolder + sampleFolder + modeFolder):
            	os.makedirs(auxFolder + sampleFolder + modeFolder)
            ParseCapture(sampleFolder, modeFolder)
            CreateBigrams(sampleFolder, modeFolder)
            #ComputeDelta(sampleFolder, modeFolder)
