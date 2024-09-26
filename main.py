#################################################
# IMPORTS
#################################################
import sys
from run_function import *
import csv
import concurrent
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

    input_filepath = sys.argv[1]
                
    processes = []

    # Using executor to master processes
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Reading test_100_epochs.txt and putting the dictionaries into the run function, and then putting the metrics from that into the output.txt
        with open(input_filepath, 'r') as file:
            for line in file:
                if line.strip() == "":
                    continue
                else:
                    # print(line.strip())
                    parameter_dictionary = json.loads(line.strip())
                    # self.helper(parameter_dictionary, output_filepath)
                    processes.append(executor.submit(main, parameter_dictionary, output_filepath))
                    
        
        # conclude the processes as they complete
        for p in concurrent.futures.as_completed(processes):

            try:
                p.result()
        