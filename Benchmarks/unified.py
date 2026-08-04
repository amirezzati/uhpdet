import os
import json


def unified_amber(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    for item in data:
        item['image'] = f'Benchmark_Images/amber/image/{item["image"]}'
        item.pop("affirmative", None)   # removes key if present, ignores if missing
        item.pop("contradictory", None) 
        item.pop("converted_affirmative_contradictory", None) 

    with open(f'{os.path.splitext(dataset_path)[0]}_unified.json', 'w') as f:
        json.dump(data, f, indent=4)



def find_phd_images(parent_path, image_id):
    image_path_base_train = f"{parent_path}/Benchmark_Images/phd/phd_images/train2014"
    image_path_base_val = f"{parent_path}/Benchmark_Images/phd/phd_images/val2014"
    
    img_path_train = f"{image_path_base_train}/COCO_train2014_{image_id}.jpg"
    img_path_val = f"{image_path_base_val}/COCO_val2014_{image_id}.jpg"

    if os.path.exists(img_path_train):
        relative = os.path.relpath(img_path_train, parent_path)
        return relative
    elif os.path.exists(img_path_val):
        relative = os.path.relpath(img_path_val, parent_path)
        return relative
    else:
        print(f"Image not found for ID: {image_id}. Skipping.")
        return None
        
def unified_phd(dataset_path, type='yes'):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    k = 1
    for item in data:
        image_path = find_phd_images(parent_dir, item["image_id"])
        if image_path is not None:
            item['image'] = image_path
        else:
            item['image'] = image_path
            print(f"Image not found for ID: {item['image_id']}. Skipping.")
        item["id"] = k
        k = k + 1


        item.pop("affirmative", None)   # removes key if present, ignores if missing
        item.pop("contradictory", None) 
        item.pop("converted_affirmative_contradictory", None) 
        item.pop("unique_id", None)
        item.pop("context", None)
        item.pop("hitem", None)
        item.pop("subject", None)
        item.pop("gt", None)
        item.pop("image_id", None)
        item['type'] = item.pop("task", None)

        item['truth'] = type
        if type == 'yes':
            item.pop("no_question", None)
            item['query'] = item['yes_question']
            item.pop("yes_question", None)
        elif type == 'no':
            item.pop("yes_question", None)
            item['query'] = item['no_question']
            item.pop("no_question", None)


    with open(f'{os.path.splitext(dataset_path)[0]}_unified.json', 'w') as f:
        json.dump(data, f, indent=4)
        

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

unified_phd(f'{current_dir}/PHD_yes_questions/data_with_outputs.json', type='yes')
unified_phd(f'{current_dir}/PHD_no_questions/data_with_outputs.json', type='no')
# unified_amber(f'{current_dir}/AMBER_no_questions/data_with_outputs.json')
# unified_amber(f'{current_dir}/AMBER_yes_questions/data_with_outputs.json')