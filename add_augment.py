import os
import json
from loguru import logger

augment_path = 'Advanced-HANet//output//data_augment//des4'
train_path = 'Advanced-HANet//data//data_ids'
output_path = 'Advanced-HANet//data//data_ids_aug'

os.makedirs(output_path, exist_ok=True)

datasets = ['MAVEN', 'ACE']
NUM_PERM = 5


def add_augment_data(input_path, original_path, output_path, dataset):
    for dataset in datasets:
        if not os.path.exists(os.path.join(train_path, dataset)):
            logger.info(f"Train folder {os.path.join(train_path, dataset)} does not exist.")
            continue
        if not os.path.exists(os.path.join(augment_path, dataset)):
            logger.info(f"Augment folder {os.path.join(augment_path, dataset)} does not exist.")
            continue
        for i in range(NUM_PERM):
            input_folder = os.path.join(augment_path, dataset, 'perm'+str(i))
            if not os.path.exists(input_folder):
                logger.info(f"Input folder {input_folder} does not exist.")
                continue

            original_folder = os.path.join(train_path, dataset, 'perm'+str(i))
            if not os.path.exists(original_folder):
                logger.info(f"Original folder {original_folder} does not exist.")
                continue

            output_folder = os.path.join(output_path, dataset, 'perm'+str(i))
            os.makedirs(output_folder, exist_ok=True)

            for file_name in os.listdir(input_folder):
                if not file_name.endswith('.jsonl'):
                    continue
                input_file = os.path.join(input_folder, file_name)
                original_file = os.path.join(original_folder, file_name)
                output_file = os.path.join(output_folder, file_name)
                logger.info(f"[START] Processing {file_name} in {output_file}")
                aug_data = []
                original_data = []
                
                with open(input_file, 'r') as f:
                    for line in f:
                        line = json.loads(line)
                        aug_data.append(line)

                with open(original_file, 'r') as f:
                    for line in f:
                        line = json.loads(line)
                        original_data.append(line)
                
                with open(output_file, 'w') as f:
                    for i in range(len(original_data)):
                        new_data = {}
                        for key, value in original_data[i].items():
                            new_data[key] = value
                            if key in aug_data[i]:
                                new_data[key].extend(aug_data[i][key])
                        f.write(json.dumps(new_data) + '\n')
                logger.info(f"[END] Processing {file_name} in {output_file}")

if __name__ == "__main__":
    add_augment_data(augment_path, train_path, output_path, datasets)
    logger.info("Add augment data done.")
