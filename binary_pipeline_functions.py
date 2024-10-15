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
# from tensorflow.keras import models,layers,losses,metrics,optimizers,Input
from data_augmentation.tm_function import *
from machines.Machines import BinaryModels
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import subprocess
import scipy.io
# from Matlab.edit_matlab_matrices import *
# from Matlab.extract_coeffs import *
# from Matlab.run_matlab import *
#################################################
# PIPELINE FUNCTIONS
#################################################

# Get data into a dataframe (Done)
# Augment data using TM function / linear interpolation (Do after getting machines set up)
# Input into Binary machines
# Return metrics

def kl_module_function(data, truncation_parameter, highest_coeff, phenotype = None):
    # 1. Edit the MATLAB matrices so the data is in the correct format
    # data = "/projectnb/mlresearch/data/binary/Colon_Cancer_COAD/TCGA-COAD-Copy1.htseq_fpkm.tsv"
    # phenotype = "/projectnb/mlresearch/data/binary/Colon_Cancer_COAD/TCGA-COAD-Copy1.GDC_phenotype.tsv"
    csv_file = '/projectnb/mlresearch/Matlab/data/Tan_data-2/colon.txt'
    test_csv_file = '/projectnb/mlresearch/Matlab/source/colonTestFile.txt'
    
    # Function to split data into train and test, and save into CSV
    def separate_cancer(phenotypes, df):
        if type(df) == str:
            data = sequencing_dataframe(df)
        else:
            data = df
        
        if data == colon_data or data == lung_data or data == prostate_data:
            phenotypes = sequencing_dataframe(phenotypes)
            df = data.T
            df['id'] = df.index
            ph = phenotypes[['submitter_id.samples', 'sample_type.samples']]
            merged_df = ph.merge(df, left_on='submitter_id.samples', right_on='id', how='inner')
            primary_tumor_ids = merged_df.loc[merged_df['sample_type.samples'] == 'Primary Tumor', 'submitter_id.samples']
            cancer = merged_df[merged_df['submitter_id.samples'].isin(primary_tumor_ids)]
            healthy = merged_df[~merged_df['submitter_id.samples'].isin(primary_tumor_ids)]
            cancer['is_cancer'] = 1
            healthy['is_cancer'] = 0
            cancer_transposed = cancer.drop(columns=['submitter_id.samples', 'sample_type.samples', 'id', 'is_cancer']).T
            healthy_transposed = healthy.drop(columns=['submitter_id.samples', 'sample_type.samples', 'id', 'is_cancer']).T
            cancer_transposed.columns = ['Tumor'] * cancer_transposed.shape[1]
            healthy_transposed.columns = ['Normal'] * healthy_transposed.shape[1]
            combined_data = pd.concat([cancer_transposed, healthy_transposed], axis=1)
            indices = range(combined_data.shape[1])
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
            
            train_data = combined_data.iloc[:, train_indices]
            test_data = combined_data.iloc[:, test_indices]
        elif binary_imported_normal is not None and binary_imported_scc is not None:
            # Handle binary imported data
            # 'data' is already split into healthy and cancer datasets for imported data
            # Combine normal and SCC data into one DataFrame
            combined_data = pd.concat([binary_imported_normal, binary_imported_scc], axis=1)
            
            # Assuming normal data is on the left and SCC on the right, create phenotype labels
            normal_labels = [0] * binary_imported_normal.shape[1]
            cancer_labels = [1] * binary_imported_scc.shape[1]
            
            # Combine labels into a phenotype Series
            phenotype = pd.Series(normal_labels + cancer_labels)
            
            # Split combined data into training and testing sets
            indices = range(combined_data.shape[1])
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
            
            train_data = combined_data.iloc[:, train_indices]
            test_data = combined_data.iloc[:, test_indices]
            
            # Split phenotype into training and testing sets
            y_train = phenotype.iloc[train_indices]
            y_test = phenotype.iloc[test_indices]

    
    def sequencing_dataframe(sequencing):
        with open(sequencing, 'rt') as file:
            df = pd.read_csv(file, sep='\t')
        return df
    
    train_data, test_data = separate_cancer(phenotype, data)
    train_data.to_csv(csv_file, index=False)
    test_data.to_csv(test_csv_file, index=False)
    
    # 2. Run the MATLAB script (TestFileFull.m)
    def run_matlab(command):
        process = subprocess.run(command, shell=True, check=True)
        if process.returncode == 0:
            print("MATLAB script ran successfully.")
        else:
            raise RuntimeError("Error running MATLAB script.")
    print("Running MATLAB paths")
    run_matlab("""matlab -nosplash -nodisplay -r "try, run('/projectnb/mlresearch/Matlab/source/paths.m'); run('/projectnb/mlresearch/Matlab/source/TestfileFull.m'); save('test_parameters.mat', 'parameters'); exit; catch e, disp(getReport(e,'extended')), exit(1), end" """)
    
    # 3. Extract the coefficients from the .mat file and filter based on truncation_parameter and highest_coeff
    pd.set_option('display.max_rows', None)        # Display all rows
    pd.set_option('display.max_columns', None)     # Display all columns
    pd.set_option('display.max_colwidth', None)    # Display full column width (no truncation)
    pd.set_option('display.expand_frame_repr', False)  # Prevent line breaks in wide DataFrames

    def extract_and_filter_dataframes(truncation_parameter, highest_coeff, training_tumor, training_normal, testing_tumor, testing_normal):
        # Load the .mat file
        mat_data = scipy.io.loadmat('/projectnb/mlresearch/Matlab/source/test_parameters.mat')
        parameters = mat_data['parameters']
        
        # Extract the coefficient arrays for each category
        training_tumor_levelcoeff = parameters['Training'][0][0]['tumor'][0][0]['C']
        training_normal_levelcoeff = parameters['Training'][0][0]['normal'][0][0]['C']
        testing_tumor_levelcoeff = parameters['Testing'][0][0]['tumor'][0][0]['C']
        testing_normal_levelcoeff = parameters['Testing'][0][0]['normal'][0][0]['C']
        
        # Convert the coefficient arrays to Pandas DataFrames
        training_tumor_df = pd.DataFrame(training_tumor_levelcoeff)
        training_normal_df = pd.DataFrame(training_normal_levelcoeff)
        testing_tumor_df = pd.DataFrame(testing_tumor_levelcoeff)
        testing_normal_df = pd.DataFrame(testing_normal_levelcoeff)
        
        # Helper function to filter based on the last column value
        def filter_by_last_column(outer_df, highest_coeff):
            # Extract the NumPy array from the pandas DataFrame
            if outer_df.shape == (1, 1):
                df = outer_df.iloc[0, 0]  # Extract the NumPy array from the single cell
            else:
                raise ValueError("Expected the DataFrame to have shape (1,1), but got shape {}".format(outer_df.shape))
            
            # Ensure the extracted data is a NumPy array
            if not isinstance(df, np.ndarray):
                raise TypeError("The extracted object is not a NumPy array")
            
            # Initialize an empty array to store indices
            indices = []

            # Iterate over the rows of the NumPy array
            for idx in range(df.shape[0]):
                # Get the value in the last column for the current row
                last_column_value = df[idx, -1]  # Access the last element of the current row
                
                print(f"Row {idx}, Last Column Value: {last_column_value}")

                # Check if the value is not -1 and is less than or equal to highest_coeff
                if last_column_value != -1 and last_column_value <= highest_coeff:
                    indices.append(idx)  # Add the index to the array if conditions are met

            return indices
        
        # Filter indices for each DataFrame
        training_tumor_indices = filter_by_last_column(training_tumor_df, highest_coeff)
        training_normal_indices = filter_by_last_column(training_normal_df, highest_coeff)
        testing_tumor_indices = filter_by_last_column(testing_tumor_df, highest_coeff)
        testing_normal_indices = filter_by_last_column(testing_normal_df, highest_coeff)
        
        # Now, we will use these indices to filter the original train_data and test_data
        training_tumor_filtered = training_tumor.iloc[training_tumor_indices].reset_index(drop=True)
        training_normal_filtered = training_normal.iloc[training_normal_indices].reset_index(drop=True)
        testing_tumor_filtered = testing_tumor.iloc[testing_tumor_indices].reset_index(drop=True)
        testing_normal_filtered = testing_normal.iloc[testing_normal_indices].reset_index(drop=True)

       # Create labels: 1 for tumor and 0 for normal
        y_train_tumor = [1] * training_tumor_filtered.shape[1]  # Number of columns corresponds to samples after transpose
        y_train_normal = [0] * training_normal_filtered.shape[1]
        y_val_tumor = [1] * testing_tumor_filtered.shape[1]
        y_val_normal = [0] * testing_normal_filtered.shape[1]

        # Combine feature data and transpose them to get samples as rows and features as columns
        X_train = pd.concat([training_tumor_filtered, training_normal_filtered], axis=1).T.reset_index(drop=True)
        X_val = pd.concat([testing_tumor_filtered, testing_normal_filtered], axis=1).T.reset_index(drop=True)

        # Merge labels
        y_train = pd.Series(y_train_tumor + y_train_normal).reset_index(drop=True)
        y_val = pd.Series(y_val_tumor + y_val_normal).reset_index(drop=True)

        # Shuffle the training data and labels
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Shuffle the validation data and labels
        X_val, y_val = shuffle(X_val, y_val, random_state=42)

        # Return the train and validation splits
        return X_train, y_train, X_val, y_val
    
    def split_tumor_normal(train_data, test_data):
        # Separate tumor and normal from train_data
        training_tumor = train_data.loc[:, train_data.columns == 'Tumor']
        training_normal = train_data.loc[:, train_data.columns == 'Normal']
        
        # Separate tumor and normal from test_data
        testing_tumor = test_data.loc[:, test_data.columns == 'Tumor']
        testing_normal = test_data.loc[:, test_data.columns == 'Normal']
        
        return training_tumor, training_normal, testing_tumor, testing_normal

    # Split the train_data and test_data into tumor and normal components
    training_tumor, training_normal, testing_tumor, testing_normal = split_tumor_normal(train_data, test_data)

    X_train, y_train, X_val, y_val = extract_and_filter_dataframes(truncation_parameter, highest_coeff, training_tumor, training_normal, testing_tumor, testing_normal)

    return X_train, y_train, X_val, y_val


def binary_pipeline(data, phenotype, model_type, TM_use = False, value = None, enhanced_genes = None, lin_int = 0, 
                    noise = False, noise_type = None, noise_level = None, kl_module = False, truncation_parameter = 10, highest_coeff = 3):

    if kl_module:
        X_train, y_train, X_val, y_val = kl_module_function(data=data, phenotype=phenotype, truncation_parameter=truncation_parameter, highest_coeff=highest_coeff)
    #TM Function if necessary (use values from before)
    if TM_use == True:
        # if enhanced_genes == None or value == None:
        #       raise ValueError("enhanced_genes and value must not be None")
        data = TM(data, enhanced_genes, value)
    
    else:
        data = sequencing_dataframe(data)
    
    cancer, healthy = separate_cancer(phenotype, data)
    cancer = cancer.sample(n=len(healthy), random_state=42)
    combined_samples = pd.concat([cancer, healthy], ignore_index=True)
    combined_samples = combined_samples.sample(frac=1).reset_index(drop=True)
    combined_samples = combined_samples.iloc[:, 1:]
    X = combined_samples.drop(columns=['sample_type.samples','id', 'is_cancer']).values.astype(np.float32)
    y = combined_samples['is_cancer']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



    #Linear interpolation if necessary 
    if lin_int != 0:
        X_train, y_train = linear_interpolation(X_train, y_train, lin_int)
    
    # Noise injection if necessary
    if noise == True:
        if noise_type == None:
            raise ValueError("noise_type and noise_level must not be None")

        X_train = noise_injection(noise_type, X_train, noise_level)

    #First seperate the cancer from healthy, add indicators, and shuffle them back together


    #Pipe resulting dataframes into svm using split and stuff
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


# Debugging main function 

# def main():
#     X_train, y_train, X_val, y_val = kl_module(data=colon_data, phenotype = colon_phenotype, truncation_parameter=10, highest_coeff=3)
#     print(X_train)
#     print(X_val)
#     print(y_train)
#     print(y_val)

# if __name__ == "__main__":
#     main()