import json
import pkg_resources

# Function to load master parameters
def load_params():
    try:
        # Get the full path to master_params.json within the installed package
        file_path = pkg_resources.resource_filename(
            'vea', 'master/master_params.json'
        )
        # Load and return the JSON content
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise RuntimeError("master_params.json not found in the package. Please ensure it is installed correctly.")

# Automatically load parameters when importing header
params = load_params()