## Index

1 - Classifiers description
2- Extracting features for feeding classifiers
3 - Running the classifiers
4 - Watching some more figures
5 - Full example for Facet analysis

#### Python 2.7 package requirements
Install the required Python packages by running:
    `pip install -r requirements.txt`
    
[**Note**] Not using virtualenvs, packages will be installed system-wide.

#### Traffic Captures Location
Copy each `TrafficCaptures` folder into the respective path in **MPTAnalysis** repo:

    MPTAnalysis/FacetAnalysis/TrafficCaptures
    MPTAnalysis/DeltaShaperAnalysis/TrafficCaptures
    MPTAnalysis/CovertCastAnalysis/TrafficCaptures

## 1- Classifiers Description
### Similarity-based classifiers
*EMD classifier:*
`EMD_classifier.py` -- This file includes the threshold-based EMD classifier as proposed in DeltaShaper.

*Chi-Square classifier:*
`X2_classifier.py`-- This file includes the Chi-Square test-based classifier as proposed in Facet.

*Kullback-Leibler classifier:*
`KL_classifier.py`-- This file includes the Kullback-Leibler-divergence  classifier as proposed in CovertCast. 

### Decision Tree-based classifiers
*Decision Tree, Random Forest, and XGBoost:*
`xgboost_classifier.py` -- This file includes the three decision tree-based classifiers used in our paper.

### Semi-Supervised / Unsupervised
*Autoencoder:*
`autoencoder.py`-- This file contains the TensorFlow code required to run our semi-supervised autoencoder.

*One-Class SVM:*
`OCSVM.py`-- This file includes the One-Class SVM classifier.

*IsolationForests:*
`IsolationForests.py`-- This file includes the Isolation Forests classifier.

## 2 - Extracting features for feeding classifiers
### Similarity-based classifiers
For using our similarity-based classifiers, raw packet captures must be mangled in order to extract binned packet sizes / bi-grams of packet sizes. `ParseCaptures.py`includes the code for parsing the raw packet captures into packet length bins of size [15, 20, 50], which will be respectively used by the [KL, X2, EMD] classifiers. Extracted features will be located in a newly generated folder called `auxFolder`.

Albeit `ParseCaptures.py` is also prepared to extract inter-packet timing features, we will not be using these with our similarity-based classifiers.

[**Disclaimer**] Extraction can take a while, I did not parallelize this code as it would be just a one-time execution.

### Remaining classifiers
For using the remaining classifiers, we will extract features and build datasets to be stored in `.csv` files.

File `extractFeatures.py`contains the required code for extracting our two different sets of features (Binned packet lengths / Summary statistics) from existing packet captures. This file defines two functions for each set of features, respectively: `FeatureExtractionPLBenchmark` and `FeatureExtractionStatsBenchmark`. Each can be called in the main code. `GenerateDatasets` will take the job of combining the extracted sets of features and build the datasets.

Feature datasets will be stored in the `FeatureSets` folder. For instance `PL_60` stores the datasets pertaining to the extraction of binned Packet Lengths collected in an interval of 60 seconds of the whole packet trace.

## 3 - Running the classifiers
### Similarity-based classifiers
`X2_classifier.py` provides two main functions for analysis, which can be selected to be used in its `main` interchangeably. `Prepare_X_RatioReproduction` reproduces the results of Facet's paper, outputting the results of a classifier with changing deltas (and enabling us to plot a ROC curve). `Prepare_X_Fixed` allows for obtaining fixed classification results for comparison with the Kullback-Leibler classifier which only outputs fixed classification rates.

Creates a folder called `X2` for holding AUC plots and serialized TPR/FPR rates for later producing the figures included in the paper.

[**Warning**] This code is not parallelized. Building the models for the classifier is an overnight effort (at least for Facet data).

----
`EMD_classifier.py`can just be executed in order to output the classifier's results with changing deltas. The script prints the delta threshold where maximum accuracy for the classifier is reached, in order to compare with the Kullback-Leibler classifier which only outputs fixed classification rates.

Creates a folder called `EMD` for holding AUC plots and serialized TPR/FPR rates for later producing the figures included in the paper.

----
`KL_classifier.py` outputs fixed classification results.

### Decision Tree-based classifiers
`xgboost_classifier.py`outputs the classification results of our three different decision tree-based classifiers, for different True Positive / False Positive rates ratios. For training these classifiers, data is assumed to be fully labeled.

The script creates a folder called `xgBoost` for storing ROC AUC figures of each classification effort, along with serialized data for building our paper ROC figures + feature importance data.

[**Note**] In the `main` function, variable `data_folder` must point the folder containing the dataset extracted with the desired feature set. For our paper results, `FeatureSets/Stats_60` or `FeatureSets/PL_60` correspond to the either of our feature sets using Summary Statistics or binned Packet Lengths.

### Semi-Supervised / Unsupervised
`OCSVM.py` runs a grid search on the parameter space of (nu,gamma) for OCSVM. It outputs the average and maximum AUC obtained after attempting to classify data points from a learned representation of legitimate video transmissions-only. In its `main` function, variable `data_folder` must point the folder containing the dataset extracted with the desired feature set.

`autoencoder.py` runs a grid search on the parameter space of (neurons in the hidden layer, size of the compressed representation layer) for our Autoencoder. It outputs the average and maximum AUC obtained after attempting to classify data points from a learned representation of legitimate video transmissions-only. In its `main` function, variable `data_folder` must point the folder containing the dataset extracted with the desired feature set.

`IsolationForests.py`runs a grid search on the parameter space of (number of trees, samples per tree) for our Isolation Forest. It outputs the average and maximum AUC obtained after attempting to classify unlabeled data points. In its `main` function, variable `data_folder` must point the folder containing the dataset extracted with the desired feature set.

The script creates a folder called `Isolation` for storing ROC AUC figures.

[**Note**] In the `main` function, variable `data_folder` must point the folder containing the dataset extracted with the desired feature set. For our paper results, `FeatureSets/Stats_60` or `FeatureSets/PL_60` correspond to the either of our feature sets using Summary Statistics or binned Packet Lengths.

## 4 - Watching some more figures
In the case of Facet / DeltaShaper analysis, there is a folder called `Figures`. This folder includes `generateFigures.py` which generates the figures used in our paper + some more detail about feature analysis.

## 5 - Full example for Facet analysis

    #Parse raw .pcap files for generating features for similarity-based classifiers
    $ cd FacetAnalysis
    $ python ParseCaptures.py
    
    #Run any similarity-based classifier
    $ python [EMD_classifier.py, KL_classifier.py, X2_classifier.py]
    
    #Parse raw .pcap files for generating features for state-of-the-art ML algorithms
    $ python extractFeatures.py
    
    #Run any ML classifier
    $ python [xgboost_classifier.py, OCSVM.py, autoencoder.py, IsolationForests.py]
    
    #Generate paper figures
    $ cd Figures
    $ python generateFigures.py


