import json
import os

with open(r'..\raw_text\MAVEN\perm0\MAVEN_10task_10way_10shot.train.jsonl', 'r', encoding='utf-8') as f:
    my_data = [json.loads(line) for line in f]

trigger_dict = {}

for pack_task in my_data:
    for key, value in pack_task.items():
        key = int(key)
        temp = []
        for sample in value:
            for idx, label in enumerate(sample['label']):
                if label == key:
                    temp.append(sample["event_words"][idx])
        
        trigger_dict[key] = temp
        
output_path = 'description_data/MAVEN/trigger_dict.json'

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(trigger_dict, f, ensure_ascii=False, indent=4)

print(f'✅ Đã lưu trigger_dict vào: {output_path}')