#!/usr/bin/env python
import sys
import csv
import glob
import os

def MergeDatasets(data_folder):
    if(os.path.exists(data_folder + '/full_dataset.csv')):
        os.remove(data_folder + '/full_dataset.csv')

    features_files = glob.glob(data_folder + "/*_dataset.csv")

    print("Merging full dataset...")
    header_saved = False
    with open(data_folder + '/full_dataset.csv','wb') as fout:
        for filename in features_files:
            print("merging " + filename)
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)
    print("Dataset merged!")


def MergeSamples(data_folder):
    #Generate training dataset
    youtube_files = glob.glob(data_folder + "/YouTubeTraffic_*.csv")

    header_saved = False
    with open(data_folder + '/youtube_dataset.csv','wb') as fout:
        for filename in youtube_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)

    covertcast_files = glob.glob(data_folder + "/CovertCastTraffic_*.csv")

    header_saved = False
    with open(data_folder + '/covertcast_dataset.csv','wb') as fout:
        for filename in covertcast_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)
                    header_saved = True
                for line in fin:
                    fout.write(line)

