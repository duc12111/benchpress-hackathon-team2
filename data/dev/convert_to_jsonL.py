import os
import json

def merge_json_to_jsonl(folder_path, output_file):
    # Ensure the folder path is valid
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
    
    try:
        # Open the output JSONL file
        with open(output_file, "w") as jsonl_file:
            # Iterate through all JSON files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".json"):
                    json_file_path = os.path.join(folder_path, file_name)
                    
                    try:
                        # Read the JSON file
                        with open(json_file_path, "r") as json_file:
                            data = json.load(json_file)
                        
                        # Write data to the JSONL file
                        if isinstance(data, list):
                            # If the JSON is a list, write each object as a line
                            for item in data:
                                jsonl_file.write(json.dumps(item) + "\n")
                        elif isinstance(data, dict):
                            # If the JSON is a single object, write it as a single line
                            jsonl_file.write(json.dumps(data) + "\n")
                        else:
                            print(f"Skipping {file_name}: Not a valid JSON object or array.")
                    
                        print(f"Processed {file_name}")
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

# Usage Example
folder_path = "data/dev/"
output_file = "output.jsonl"
merge_json_to_jsonl(folder_path, output_file)
