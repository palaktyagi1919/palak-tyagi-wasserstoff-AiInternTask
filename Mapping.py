import json
import os

def create_data_mapping(object_descriptions, object_texts, object_summaries, output_dir="segmented_objects"):
    data_mapping = {}

    try:
        # Iterate over files in the specified directory
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                object_id = filename.split(".")[0]
                # Populate the data mapping dictionary
                data_mapping[object_id] = {
                    "description": object_descriptions.get(filename, ""),
                    "text": object_texts.get(filename, ""),
                    "summary": object_summaries.get(filename, "")
                }

        # Save the data mapping to a JSON file
        with open("data_mapping.json", "w") as f:
            json.dump(data_mapping, f, indent=4)
        print("Data mapping saved successfully.")

    except FileNotFoundError as e:
        print(f"Error: Directory not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
object_descriptions = {"object1.png": "description1", "object2.png": "description2"}
object_texts = {"object1.png": "text1", "object2.png": "text2"}
object_summaries = {"object1.png": "summary1", "object2.png": "summary2"}

create_data_mapping(object_descriptions, object_texts, object_summaries)