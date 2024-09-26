import gradio as gr
import json
from itertools import product
# import run_function
import concurrent
import re

# Helper function to run the analysis
def run_analysis(datasets, machines):
    selected_datasets = [dataset for dataset, selected in datasets.items() if selected]
    selected_machines = [machine for machine, selected in machines.items() if selected]

    if not selected_datasets:
        return "Please select at least one dataset."
    
    if not selected_machines:
        return "Please select at least one machine."

    configurations = []
    for dataset, machine in product(selected_datasets, selected_machines):
        config = {
            "Dataset": dataset, "Binary": "True", "Image": "False", "TM_use": False, "value": "None", "Interpolation_value": "None", "Noise": False, "noise_type": False, "augmentation": False, "shear_factor": "None", "shear_prop": 0, "crop_scale_factor": "None", "crop_scale_prop": 0, "flip_code": "None", "flip_prop": 0, "rotation_angle": "None", "rotate_prop": 0, "color_number": "None", "color_prop": 0, "blur_param": "None", "blur_prop": 0, "features": "False", "glcm_distance": "None", "glcm_angle": "None", "glcm_prop": 0, "lbp_radius": "None", "lbp_prop": 0, "haralick_prop": 0, "Machine": machine, "lr": "None", "epochs": "None"
        }
        configurations.append(config)

    output_filepath = "results/evaluation_scores.txt"
    filepath = "results/models-parameters_list.txt"

    with open(filepath, 'w') as f:
        for config in configurations:
            f.write(json.dumps(config))
            f.write("\n\n")
            
    # VISH CODE GOES HERE
    # Should run bash script to move file with name filepath to SCC before 
    # calling "main.py filepath output_filepath" and ending by return output_filepath
    # to the users directory

    

# Gradio Interface
def update_summary(datasets, machines):
    selected_datasets = [dataset for dataset, selected in datasets.items() if selected]
    selected_machines = [machine for machine, selected in machines.items() if selected]

    summary_text = f"Selected Datasets:\n{', '.join(selected_datasets) if selected_datasets else 'None'}\n\n"
    summary_text += f"Selected Machines:\n{', '.join(selected_machines) if selected_machines else 'None'}"
    return summary_text

# Gradio components
with gr.Blocks() as app:
    gr.Markdown("# Cancer Analysis Tool")

    # Tab 1: Dataset Selection
    with gr.Tab("Dataset"):
        datasets = gr.CheckboxGroup(
            ["Binary Colon", "Binary Lung", "Binary Prostate"],
            label="Select Datasets:"
        )

    # Tab 2: Machine Selection
    with gr.Tab("Machines"):
        machines = gr.CheckboxGroup(
            ["Binary SVM", "Binary KNN", "Binary Deepnet"],
            label="Select Machines:"
        )

    # Tab 3: Summary and Analysis
    with gr.Tab("Analysis"):
        summary = gr.Textbox(label="Summary of Selections", interactive=False)
        run_button = gr.Button("Run Analysis")
        result = gr.Textbox(label="Result", interactive=False)

        run_button.click(
            run_analysis,
            inputs=[datasets, machines],
            outputs=result
        )
        datasets.change(update_summary, [datasets, machines], summary)
        machines.change(update_summary, [datasets, machines], summary)

# Launch the Gradio interface
app.launch()
