#!/usr/bin/env python
import sys
import csv
import glob
import os


def MergeDatasets(data_folder):
    if(os.path.exists(data_folder + '/full_dataset.csv')):
        os.remove(data_folder + '/full_dataset.csv')

    features_files = [data_folder + "facet_dataset.csv", data_folder + "RegularTraffic_Christmas_dataset.csv"]

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



def CombinedMerging(data_folder):
    if(os.path.exists(data_folder + '/regular_12.5_dataset.csv')):
        os.remove(data_folder + '/regular_12.5_dataset.csv')
    if(os.path.exists(data_folder + '/regular_25_dataset.csv')):
        os.remove(data_folder + '/regular_25_dataset.csv')
    if(os.path.exists(data_folder + '/regular_50_dataset.csv')):
        os.remove(data_folder + '/regular_50_dataset.csv')

    features_files = [data_folder + "FacetTraffic_12.5_Christmas_dataset.csv", data_folder + "RegularTraffic_Christmas_dataset.csv"]

    print("Merging dataset...")
    header_saved = False
    with open(data_folder + '/regular_12.5_dataset.csv','wb') as fout:
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

    features_files = [data_folder + "FacetTraffic_25_Christmas_dataset.csv", data_folder + "RegularTraffic_Christmas_dataset.csv"]

    print("Merging dataset...")
    header_saved = False
    with open(data_folder + '/regular_25_dataset.csv','wb') as fout:
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

    features_files = [data_folder + "FacetTraffic_50_Christmas_dataset.csv", data_folder + "RegularTraffic_Christmas_dataset.csv"]

    print("Merging dataset...")
    header_saved = False
    with open(data_folder + '/regular_50_dataset.csv','wb') as fout:
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
    facet_files = glob.glob(data_folder + "/FacetTraffic_*.csv")

    header_saved = False
    with open(data_folder + '/facet_dataset.csv','wb') as fout:
        for filename in facet_files:
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
    #MergeDatasets(data_folder)

