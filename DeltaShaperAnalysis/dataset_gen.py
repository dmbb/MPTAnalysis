#!/usr/bin/env python
import sys
import csv
import glob
import os


def MergeDatasets(data_folder):
    if(os.path.exists(data_folder + '/full_dataset.csv')):
        os.remove(data_folder + '/full_dataset.csv')

    features_files = [data_folder + "deltashaper_dataset.csv", data_folder + "RegularTraffic_dataset.csv"]

    print "Merging full dataset..."
    header_saved = False
    with open(data_folder + '/full_dataset.csv','wb') as fout:
        for filename in features_files:
            print "merging " + filename
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
    print "Dataset merged!"


def CombinedMerging(data_folder):
    if(os.path.exists(data_folder + '/regular_320_dataset.csv')):
        os.remove(data_folder + '/regular_320_dataset.csv')
    if(os.path.exists(data_folder + '/regular_160_dataset.csv')):
        os.remove(data_folder + '/regular_160_dataset.csv')

    features_files = [data_folder + "DeltaShaperTraffic_320_dataset.csv", data_folder + "RegularTraffic_dataset.csv"]

    print "Merging dataset..."
    header_saved = False
    with open(data_folder + '/regular_320_dataset.csv','wb') as fout:
        for filename in features_files:
            print "merging " + filename
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
    print "Dataset merged!"

    features_files = [data_folder + "DeltaShaperTraffic_160_dataset.csv", data_folder + "RegularTraffic_dataset.csv"]

    print "Merging dataset..."
    header_saved = False
    with open(data_folder + '/regular_160_dataset.csv','wb') as fout:
        for filename in features_files:
            print "merging " + filename
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
    print "Dataset merged!"



def MergeSamples(data_folder):
    #Generate training dataset
    deltashaper_files = glob.glob(data_folder + "/DeltaShaperTraffic_*.csv")

    header_saved = False
    with open(data_folder + 'deltashaper_dataset.csv','wb') as fout:
        for filename in deltashaper_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)


def GenerateDatasets(data_folder):
    MergeSamples(data_folder)
    CombinedMerging(data_folder)
    MergeDatasets(data_folder)
