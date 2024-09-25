#################################################
# CONSTANTS
#################################################

colon_data = "/projectnb/mlresearch/data/binary/Colon_Cancer_COAD/TCGA-COAD-Copy1.htseq_fpkm.tsv"
colon_phenotype = "/projectnb/mlresearch/data/binary/Colon_Cancer_COAD/TCGA-COAD-Copy1.GDC_phenotype.tsv"
colon_enhanced_genes = "/projectnb/mlresearch/test/Junze/cancer/diff_exp_csvs/Colon_DE.csv"

lung_data = "/projectnb/mlresearch/data/binary/Lung_Adenocarcinoma_LUAD/TCGA-LUAD.htseq_fpkm.tsv"
lung_phenotype = "/projectnb/mlresearch/data/binary/Lung_Adenocarcinoma_LUAD/TCGA-LUAD.GDC_phenotype.tsv"
lung_enhanced_genes = "/projectnb/mlresearch/test/Junze/cancer/diff_exp_csvs/Lung_DE.csv"

prostate_data = "/projectnb/mlresearch/test/Junze/cancer/data/binary/TCGA-COAD.htseq_fpkm.tsv"
prostate_phenotype = "/projectnb/mlresearch/test/Junze/cancer/data/binary/TCGA-COAD.GDC_phenotype.tsv"
prostate_enhanced_genes = "/projectnb/mlresearch/test/Junze/cancer/diff_exp_csvs/Prostate_DE.csv"


binary_imported_normal = "/projectnb/mlresearch/data/binary/Imported/Imported_Benign"
binary_imported_scc = "/projectnb/mlresearch/data/binary/Imported/Imported_Malignant"


oral_normal = "/projectnb/mlresearch/test/Junze/cancer/data/Image/oral_normal"
oral_scc = "/projectnb/mlresearch/test/Junze/cancer/data/Image/oral_scc"


all_normal= "/projectnb/mlresearch/data/image/ALL/ALL_Benign"
all_scc= "/projectnb/mlresearch/data/image/ALL/ALL_Pre_B"

breast_normal= "/projectnb/mlresearch/data/image/Breast/Breast_Benign"
breast_scc= "/projectnb/mlresearch/data/image/Breast/Breast_Malignant"

chest_normal= "/projectnb/mlresearch/data/image/Chest_X_Ray/0"
chest_scc= "/projectnb/mlresearch/data/image/Chest_X_Ray/1"

image_imported_normal = "/projectnb/mlresearch/data/image/Imported/Imported_Benign"
image_imported_scc = "/projectnb/mlresearch/data/image/Imported/Imported_Malignant"

breast_ultra_normal= "/projectnb/mlresearch/data/image/Breast_Ultra_Sound/0"
breast_ultra_scc= "/projectnb/mlresearch/data/image/Breast_Ultra_Sound/1"

colon_normal= "/projectnb/mlresearch/data/image/Colon/Colon_Benign_Tissue"
colon_scc= "/projectnb/mlresearch/data/image/Colon/Colon_Adenocarcinoma"

lung_normal= "/projectnb/mlresearch/data/image/Lung/Lung_Benign_Tissue"
lung_scc= "/projectnb/mlresearch/data/image/Lung/Lung_Adenocarcinoma"


#################################################
# IMPORTS
#################################################

import pandas as pd
import gzip
import requests
import numpy as np 
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers,Input
from data_augmentation.tm_function import *
from machines.Machines import BinaryModels
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

#################################################
# PIPELINE FUNCTIONS
#################################################

# Get data into a dataframe (Done)
# Augment data using TM function / linear interpolation (Do after getting machines set up)
# Input into Binary machines
# Return metrics

def binary_pipeline(data, phenotype, model_type, TM_use = False, value = None, enhanced_genes = None, lin_int = False, 
                    noise = False, noise_type = None, noise_level = None):
                    
                    
    if TM_use == True:
        # if enhanced_genes == None or value == None:
        #       raise ValueError("enhanced_genes and value must not be None")
        data = TM(data, enhanced_genes, value)
    
    else:
        data = sequencing_dataframe(data)

    #Linear interpolation if necessary 
    if lin_int == True:
        data = linear_interpolation(data)
    
    # Noise injection if necessary
    if noise == True:
        if noise_type == None:
            raise ValueError("noise_type and noise_level must not be None")

        data = noise_injection(noise_type, data, noise_level)

    # First separate the cancer from healthy, add indicators, and shuffle them back together
    cancer, healthy = separate_cancer(phenotype, data)
    
    # Ensure that the number of cancer samples equals the number of healthy samples
    cancer = cancer.sample(n=len(healthy), random_state=42)
    
    # Combine the cancer and healthy samples and shuffle them
    combined_samples = pd.concat([cancer, healthy], ignore_index=True)
    combined_samples = combined_samples.sample(frac=1).reset_index(drop=True)
    
    # Extract features and labels
    combined_samples = combined_samples.iloc[:, 1:]  # Remove the first column (assumed to be irrelevant)
    X = combined_samples.drop(columns=['sample_type.samples', 'id', 'is_cancer']).values.astype(np.float32)
    y = combined_samples['is_cancer']
    
    # Perform train-test split first
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    

    # if KL:
    #   edit_matlab_matrices(X_train, y_train, X_val, y_val)
    #   run_matlab code
    #   extract parameters.training.Tumor.C / normal.C
    #   Columns are the new samples except for the last one which is a label that tells you what level of the multilevel filter that feature belongs to
    #   Rows are the features themselves, almost same as the beginning features, -m features
    
    #Use string given in function
    binary_model = BinaryModels(model_type, X_train=X_train, y_train=y_train)
    model = binary_model.get_model()

    y_pred = model.predict(X_val)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    #compile parameters
    params = {'TM': TM_use, 'value': value, 'lin_int': lin_int, 'noise': noise, 'noise_type': noise_type}
    return params, metrics

BINARY_PARAM_GRID = {
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'noise_type': ["uniform", "salt and pepper",  "poisson", "exponential"],
    }

import random 
def select_random_binary_params():
    value = random.choice(BINARY_PARAM_GRID['value'])
    noise_type = random.choice(BINARY_PARAM_GRID['noise_type'])
    return value, noise_type

# This is the function that takes in the enhanced genes, the TM value, and returns a csv of all of the gene names and their pathways
def return_enhanced_genes(enhanced_genes, value):
    # KEGG_Data retrieved using KEGG api in R
    kegg = pd.read_csv("KEGG_data.csv")
    df, df_gene_names, df_gene_list = enhanced_gene_dataframe(enhanced_genes)
    gene_list = filter_genes(df, value)
    print(gene_list)
    data = kegg[kegg["symbol"].isin(gene_list)]
    return data
#################################################
# FULL PIPELINE
#################################################
# print("Colon")
# params, metrics = binary_pipeline(colon_data, colon_phenotype, 'svm')
# print("Params: ")
# print(params)
# print("Metrics: ")
# print(metrics)

# print("Colon TM")
# params, metrics = binary_pipeline(colon_data, colon_phenotype, 'svm', TM_use = True, value = 4, enhanced_genes = colon_enhanced_genes)
# print("Params: ")
# print(params)
# print("Metrics: ")
# print(metrics)

# print("Colon lin int")
# params, metrics = binary_pipeline(lung_data, lung_phenotype, 'svm', lin_int = True)
# print("Params: ")
# print(params)
# print("Metrics: ")
# print(metrics)

# print("Colon noise")
# params, metrics = binary_pipeline(lung_data, lung_phenotype, 'svm', noise = True, noise_type = "poisson")
# print("Params: ")
# print(params)
# print("Metrics: ")
# print(metrics)

# print("Colon all")
# params, metrics = binary_pipeline(lung_data, lung_phenotype, 'svm',  TM_use = True, value = 4, enhanced_genes = colon_enhanced_genes, lin_int = True,  noise = True, noise_type = "poisson")
# print("Params: ")
# print(params)
# print("Metrics: ")
# print(metrics)

# print("Colon all")
# params, metrics = binary_pipeline(lung_data, lung_phenotype, 'svm',  TM_use = True, value = 4, enhanced_genes = colon_enhanced_genes, noise = True, noise_type = "poisson")
# print("Params: ")
# print(params)
# print("Metrics: ")
# print(metrics)