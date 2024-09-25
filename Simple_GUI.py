#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import ttk, messagebox
import json
from itertools import product
import re
import run_function
import concurrent

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Analysis Tool")
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        
        # Tab 1: Dataset
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Dataset")
        self.create_tab1_widgets()
        
        # Tab 2: Machines
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Machines")
        self.create_tab2_widgets()
        
        # Tab 3: Analysis
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Analysis")
        self.create_tab3_widgets()
        
        self.notebook.pack(expand=1, fill='both')
        
    def create_tab1_widgets(self):
        self.datasets = {
            "Binary Colon": tk.BooleanVar(),
            "Binary Lung": tk.BooleanVar(),
            "Binary Prostate": tk.BooleanVar()
        }

        ttk.Label(self.tab1, text="Select Datasets:").pack(anchor='w')
        for dataset, var in self.datasets.items():
            ttk.Checkbutton(self.tab1, text=dataset, variable=var).pack(anchor='w')
    
    def create_tab2_widgets(self):
        self.machines = {
            "Binary SVM": tk.BooleanVar(),
            "Binary KNN": tk.BooleanVar(),
            "Binary Deepnet": tk.BooleanVar()
        }
        
        ttk.Label(self.tab2, text="Select Machines:").pack(anchor='w')
        for machine, var in self.machines.items():
            ttk.Checkbutton(self.tab2, text=machine, variable=var).pack(anchor='w')
    
    def create_tab3_widgets(self):
        ttk.Label(self.tab3, text="Summary of Selections:").pack(anchor='w')
        self.summary_label = ttk.Label(self.tab3, text="", anchor='w', justify='left')
        self.summary_label.pack(anchor='w', padx=10, pady=10)
        
        self.run_button = ttk.Button(self.tab3, text="Run", command=self.run_analysis)
        self.run_button.pack(anchor='w', padx=10, pady=10)
        
        # Update the summary when the tab is selected
        self.notebook.bind("<<NotebookTabChanged>>", self.update_summary)
    
    def update_summary(self, event=None):
        selected_datasets = [name for name, var in self.datasets.items() if var.get()]
        selected_machines = [name for name, var in self.machines.items() if var.get()]
        
        summary_text = f"Selected Datasets:\n{', '.join(selected_datasets) if selected_datasets else 'None'}\n\n"
        summary_text += f"Selected Machines:\n{', '.join(selected_machines) if selected_machines else 'None'}"
        
        self.summary_label.config(text=summary_text)

    def run_analysis(self):
        selected_datasets = [name for name, var in self.datasets.items() if var.get()]
        selected_machines = [name for name, var in self.machines.items() if var.get()]
        
        if not selected_datasets:
            messagebox.showwarning("No Dataset Selected", "Please select at least one dataset.")
            return
        
        if not selected_machines:
            messagebox.showwarning("No Machine Selected", "Please select at least one machine.")
            return

        configurations = []
        for dataset, machine in product(selected_datasets, selected_machines):
            config = {
                "Dataset": dataset, "Binary": "True", "Image": "False", "TM_use": False, "value": "None", "Interpolation_value": "None", "Noise": False, "noise_type": False, "augmentation": False, "shear_factor": "None", "shear_prop": 0, "crop_scale_factor": "None", "crop_scale_prop": 0, "flip_code": "None", "flip_prop": 0, "rotation_angle": "None", "rotate_prop": 0, "color_number": "None", "color_prop": 0, "blur_param": "None", "blur_prop": 0, "features": "False", "glcm_distance": "None", "glcm_angle": "None", "glcm_prop": 0, "lbp_radius": "None", "lbp_prop": 0, "haralick_prop": 0, "Machine": machine, "lr": "None", "epochs": "None"
            }
            configurations.append(config)

        # Declaring ther variables for the input into run function
        output_filepath = "results/evaluation_scores.txt"
        filepath = "results/models-parameters_list.txt"

        open(output_filepath, 'w').close()
        
        # Writing the dictionaries into the test_100_epochs.txt
        with open(filepath, 'w') as f:
            for config in configurations:
                print(config)
                f.write(json.dumps(config))
                f.write("\n\n")

        # Process pool
        processes = []

        # Using executor to master processes
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Reading test_100_epochs.txt and putting the dictionaries into the run function, and then putting the metrics from that into the output.txt
            with open(filepath, 'r') as file:
                for line in file:
                    if line.strip() == "":
                        continue
                    else:
                        # print(line.strip())
                        parameter_dictionary = json.loads(line.strip())
                        # self.helper(parameter_dictionary, output_filepath)
                        processes.append(executor.submit(helper, parameter_dictionary, output_filepath))
                        
            
            # conclude the processes as they complete
            for p in concurrent.futures.as_completed(processes):

                try:
                    p.result()
    
                except Exception as e:
                    print(f"Error in execution: {e}")
        

        # Step 1: Read the content of the text file
        with open(filepath, 'r') as file:
            content = file.read()

        # Step 2: Use a regular expression to find and remove double quotes around "False"
        modified_content = re.sub(r'\"(False|None)\"', r'\1', content)

        # Step 3: Write the modified content back to the text file
        with open(filepath, 'w') as file:
            file.write(modified_content)
        
        messagebox.showinfo("Analysis Complete", f"Configuration file '{filepath}' has been created.")
        print(f"Configuration file '{filepath}' has been created.")

def helper(parameter_dictionary, output_filepath):
        function_parameters = list(parameter_dictionary.values())
        if parameter_dictionary['TM_use'] == False:
            params, metrics = run_function.run_function(*function_parameters)
        else:
            params, metrics, Kegg_csv = run_function.run_function(*function_parameters)
            Kegg_csv.to_csv('Temp_DE2.csv', index=False)
        params = str(params)
        print(params)
        metrics = str(metrics)
        with open(output_filepath, "a") as file:
            file.write(params)
            file.write(metrics)
            file.write('\n')      



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

