from utils.convert import sent2ids_expand_batch, sent2ids_batch
import os
import json


input_root = './output/des4'
output_root = './output/ids2'

datasets = ['MAVEN', 'ACE']
perms = [0, 1, 2, 3, 4]
num_des = 10
num_neg = 10

for dataset in datasets:
    os.makedirs(os.path.join(output_root, dataset), exist_ok=True)

    for i in perms:
        input_folder = os.path.join(input_root, dataset, "perm"+str(i))
        
        if not os.path.exists(input_folder):
            print(f"[ERROR] Folder {input_folder} is not exist")
            continue

        output_folder = os.path.join(output_root, dataset, "perm"+str(i))
        os.makedirs(output_folder, exist_ok=True)

        for file_name in os.listdir(input_folder):
            if not file_name.endswith(".jsonl"):
                continue
            
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
        
            with open(input_file, 'r') as f:
                input_lines = [json.loads(line) for line in f]
            
            
            original_num_item = 0
            new_num_item = 0
            output_lines = []
            
            for line_idx, line in enumerate(input_lines):
                output_lines.append({})
                for key in line:
                    original_num_item += len(line[key])
                    new_items = []
                    key_int = int(key)
                    
                    for idx, item in enumerate(line[key]):
                        new_items.append(item)
                        
                        for event, label in zip(item['events'], item['label']):
                            if label == key_int:
                                trigger_word = event['trigger_word']
                                for des in event['description'][:num_des]:
                                    if trigger_word in des:
                                        new_items.append({
                                            'text': des,
                                            'events': [{
                                                'trigger_word': trigger_word,
                                            }],
                                            'label': [label],
                                        })
                                        new_num_item += 1
                    converted_items = sent2ids_expand_batch(new_items, neg_size=num_neg)
                    for item in converted_items:
                        piece_ids = item['piece_ids']
                        if len(piece_ids) == 0:
                            raise ValueError(f"piece_ids is empty for item: {item}")

                    output_lines[line_idx][key] = converted_items
                    
            with open(output_file, 'w') as f:
                for line in output_lines:
                    f.write(json.dumps(line) + '\n')
                    
            print(f"Converted {input_file} to {output_file} with {original_num_item} original items and {new_num_item} new items")
