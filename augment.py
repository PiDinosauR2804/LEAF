import os
import json
from loguru import logger
import copy
from utils.convert import sent2ids

input_path = 'Advanced-HANet\\output\\des4'
output_path = 'Advanced-HANet\\output\\data_augment\\des4'
datasets = ['MAVEN', 'ACE']
NUM_PERM = 5

def augment_data(line, num_descriptions=10)->list:
    augment_data_list = {}
    for key, value in line.items():
        key_id = int(key)
        augment_data_list[key] = []
        for data in value:
            if 'events' in data:
                events = data["events"]
                for i, event in enumerate(events):
                    if data['label'][i] == key_id:
                        if 'description' in event:
                            description = event['description']
                            des_range = min(len(description), num_descriptions)
                            for j in range(des_range):
                            # Tạo dữ liệu mới bằng cách kết hợp text và description
                            # Tạo dữ liệu mới bằng cách kết hợp text và description
                            #new_data = copy.deepcopy(data)
                            #new_data['text'] = new_data['text'] + ' ' + description
                                new_data = {}
                                new_data['text'] = data['text'] +' '+ description[j]
                                new_data['event_words'] = [data['event_words'][i]]
                                new_data['label'] = [key_id]
                                new_data['events'] = [copy.deepcopy(event)]
                                new_data = sent2ids(new_data)
                                augment_data_list[key].append(new_data)

    return augment_data_list

def augment_dataset(input_path, output_path, dataset):
    os.path.exists(os.path.join(input_path))
    os.makedirs(output_path, exist_ok=True)
    for dataset in datasets:
        if os.path.exists(os.path.join(input_path, dataset)):
            for i in range(NUM_PERM):
                input_folder = os.path.join(input_path, dataset, 'perm'+str(i))
                if not os.path.exists(input_folder):
                    print(f"Input folder {input_folder} does not exist.")
                    continue
                output_folder = os.path.join(output_path, dataset, 'perm'+str(i))
                os.makedirs(output_folder, exist_ok=True)
                for file_name in os.listdir(input_folder):

                    if not file_name.endswith('.jsonl'):
                        continue
                    input_file = os.path.join(input_folder, file_name)
                    output_file = os.path.join(output_folder, file_name)
                    print(f"[START] Processing {file_name} in {output_file}")
                    new_data = []
                    with open(input_file, 'r') as f:
                        for line in f:
                            line = json.loads(line)
                            #Gọi hàm augment_data để tạo ra dữ liệu mới
                            aug_data = augment_data(line)
                            new_data.append(aug_data)
                    # Ghi dữ liệu mới vào file đầu ra
                    if new_data is None:
                        print(f"Error: No data in {file_name}")
                        continue
                    with open(output_file, 'w') as f:
                        for line in new_data:
                            f.write(json.dumps(line) + '\n')
                    logger.info(f"Augmented {file_name} and saved to {output_file}")
                    logger.info(f"[FINISHED] Processing {file_name} in {output_file}")

if __name__ == "__main__":
    # Check if input path exists
    os.makedirs(output_path, exist_ok=True)
    augment_dataset(input_path, output_path, datasets)                       