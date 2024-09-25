#################################################
# IMPORTS
#################################################
import sys
from run_function import *
import csv
# main file
# Our main file will be the control mechanism and part of creating more strategies according to the accuracy of the model.
# we will test the whole pillars using this main function.

#################################################
# MAIN FUNCTIONS
#################################################
# Here we take a line of the text file provided GUI, which is a dictionary, and run it through run_function to train the model according to the params



def main(parameter_dictionary, output_filepath):
    function_parameters = list(parameter_dictionary.values())
    if parameter_dictionary['TM_use'] == False:
        params, metrics = run_function(*function_parameters)
    else:
        params, metrics, Kegg_csv = run_function(*function_parameters)
        Kegg_csv.to_csv('Temp_DE2.csv', index=False)
    params = str(params)
    print(params)
    metrics = str(metrics)
    with open(output_filepath, "a") as file:
        file.write(params)
        file.write(metrics)
        file.write('\n')

#################################################
# CALL MAIN
#################################################

# this retrieves the filename for the txt file with the 
if __name__ == "__main__":
    
    if len(sys.argv) < 1:
        print("Usage: python main.py filepath.txt")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        output_filepath = sys.argv[2]
    else:
        output_filepath = "output.txt"

    with open(output_filepath, "w") as file:
        pass

    filepath = sys.argv[1]
    with open(filepath, 'r') as file:
        for line in file:
            if line == "\n":
                pass
            else:
                print(line)
                parameter_dictionary = {}
                parameter_dictionary = eval(line)
                main(parameter_dictionary, output_filepath)
            