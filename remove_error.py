# load from path, the path is a json file
import json
import os
import argparse

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# remove the error from the json file
def remove_error(json_dict):
    delete_count = 0
    valid_count = 0
    keys = list(json_dict.keys())
    for element in keys:
        if ' error' in json_dict[element] or ' Error' in json_dict[element] or " fail" in json_dict[element].lower():
            del json_dict[element]
            delete_count += 1
        else:
            valid_count += 1
    print(">>>>> Delete: ", delete_count)
    print(">>>>> Valid: ", valid_count)
    return json_dict

def remove_error_inference(json_dict):
    delete_count = 0
    valid_count = 0
    for types in json_dict:
        elements = json_dict[types]
        keys = list(elements.keys())
        for element in keys:
            if ' error' in elements[element] or ' Error' in elements[element] or " fail" in elements[element].lower():
                del elements[element]
                delete_count += 1
            else:
                valid_count += 1
        json_dict[types] = elements
    print(">>>>> Delete: ", delete_count)
    print(">>>>> Valid: ", valid_count)
    return json_dict

# save the json file
def save_json(json_dict, path):
    with open(path, 'w') as f:
        json.dump(json_dict, f, indent=4)   

def main(input_path, output_path):
    json_dict = load_json(input_path)
    if "inference" in input_path:
        json_dict = remove_error_inference(json_dict)
    else:
        json_dict = remove_error(json_dict)
    save_json(json_dict, output_path)

def check_overlapping(input_path1, input_path2):
    json_dict1 = load_json(input_path1)
    json_dict2 = load_json(input_path2)

    exist = 0
    not_exist = 0
    for element in json_dict1:
        if element in json_dict2:
            exist += 1
        else:
            not_exist += 1
    print(">>>>> Exist: ", exist)
    print(">>>>> Not Exist: ", not_exist)

def merge(input_path1, input_path2):
    json_dict1 = load_json(input_path1)
    json_dict2 = load_json(input_path2)
    for element in json_dict1:
        if element not in json_dict2:
            json_dict2[element] = json_dict1[element]
    save_json(json_dict2, input_path2)

def count_files(input_path):
    json_dict1 = load_json(input_path)
    print(len(json_dict1))

if __name__ == "__main__":
    old_file = f"/export/home/becky/verl-tool-mm-deepsearch/web_search_extracted_old.json"
    filename = "image_search_extracted.json"
    input_path = f"/export/home/becky/verl-tool-mm-deepsearch/{filename}"
    output_path = f"/export/home/becky/verl-tool-mm-deepsearch/{filename}"
    main(input_path, output_path)
    # check_overlapping(old_file, input_path)
    # merge(old_file, input_path)
    # count_files(input_path)
    # count_files(old_file)