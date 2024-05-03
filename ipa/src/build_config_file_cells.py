import os
import questionary
import yaml

CONFIG_NAME = 'config_file.yml'


def main() -> None:
    """
    Create preprocessing_config.yaml from user inputs.
    """

    print("Please answer the following questions to build the config file. \n Press enter to use the default values.")

    name_of_experiment = questionary.path("Name of run:").ask()

    folder_path = questionary.path("Path to the folder containing the images:").ask()
    
    save_path = questionary.path("Save path:").ask()
    if save_path == "": save_path = '/tungstenfs/scratch/ggiorget/nessim/2_color_imaging/localization_precision_estimation/runs/'
    
    thresh= questionary.text("Threshold for h max:").ask()
    if thresh == '': thresh = 0.5
    
    threads = questionary.text("Number of threads:").ask()
    if threads == '': threads = 5
    
    cutoff = questionary.text("Cutoff for matching:").ask()
    if cutoff == '': cutoff = 0.3


    confi = os.path.join(save_path,name_of_experiment ,CONFIG_NAME)

    config = {
        "name_of_run": name_of_experiment,
        "folder_path": folder_path,
        "save_path": os.path.join(save_path,name_of_experiment),
        "thresh": float(thresh),
        "threads": int(threads),
        "cutoff": float(cutoff)
    }

    os.makedirs(save_path+'/'+name_of_experiment, exist_ok=True)

    with open(os.path.join(save_path,name_of_experiment ,CONFIG_NAME), "w") as f:
        yaml.safe_dump(config, f)

    return confi

if __name__ == "__main__":
    config = main()