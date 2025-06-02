from description_gen import gen_description_llm
from extract_event import list2ids
import copy
import json
import os

input_path = "test"
output_path = "data_augment_incremental_by_llm"
datasets = ['MAVEN']
NUM_TRY = 4

def augment_data(item):
    # Tạo description cho trigger word
    text = item['text']
    events = item['events']
    for event in events:
        trigger_word = event['trigger_word']
        # Tạo description cho trigger word
        description = gen_description_llm(text, trigger_word, model="gemini-2.0-flash", candidate=1)
        if description is None:
            print(f"Error generating description for trigger word: {trigger_word}")
            continue
        # Thêm trigger_description vào event
        event['trigger_description'] = description

    # Copy và augment data
    new_data_list = []
    new_data_list.append(item)
    for event in item['events']:
        # Tạo một bản sao của item và thêm trigger_description vào text
        new_item = copy.deepcopy(item)
        new_item['text'] = new_item['text'] + ' ' + event['trigger_description']
    new_data_list.append(new_item)
    return new_data_list

def augment_dataset(input_path:str, output_path:str):
    # Tạo thư mục chứa data augment
    os.makedirs(output_path, exist_ok=True)
    for dataset in datasets:
        os.makedirs(os.path.join(output_path, dataset),exist_ok=True)
        # Duyệt qua các perm 
        for i in range(5):
            # Kiểm tra xem thư mục có tồn tại không
            input_folder = os.path.join(input_path, dataset, "perm"+str(i))
            if not os.path.exists(input_folder, exist_ok=True):
                print(f'Folder not found: {input_folder}')
                continue
            # Tạo thư mục chứa data augment cho từng perm
            output_folder = os.path.join(output_path, dataset, "perm"+str(i))
            os.makedirs(output_folder, exist_ok=True)
            # List chứa augmented data
            augmented_data_list = []
            # Duyệt qua các file trong perm
            for file_name in os.listdir(input_folder):
                if file_name.endswith('.jsonl'):
                    input_file = os.path.join(input_folder, file_name)
                    output_file = os.path.join(output_folder, file_name)
                    with open(input_file, 'r') as f:
                        for line in f:
                            augment_line = {}
                            # Chuyển đổi từng dòng JSON thành dict
                            json_line = json.loads(line)
                            for key, value in json_line.items():
                                # Augment data cho từng dòng
                                augmented_data = augment_data(value)
                                # Chuyển đổi thành ids
                                ids_augmented_data = list2ids(augmented_data, found_trigger = True)
                                augment_line[key] = ids_augmented_data
                            augment_data_list.append(augment_line)
                    # Ghi dữ liệu đã augment vào file mới                   
                    with open(output_file, 'w') as f:
                        for data in augment_data_list:
                            f.write(json.dumps(data) + '\n')
                    print(f"Augmented data saved to {output_file}")

if main() == "__main__":
    augment_dataset(input_path, output_path)
    print("Data augmentation completed.")



        
    






