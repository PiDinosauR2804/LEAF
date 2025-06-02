import json
import pandas as pd
from tqdm.auto import tqdm
import os
from transformers import BertTokenizerFast

# Khởi tạo tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Đường dẫn đến các folder chứa dữ liệux
input_path = "data/data_ids"
output_path = "raw_text"
datasets = ["ACE", "MAVEN"]
# datasets = ["MAVEN"]

def ids2list(list_data: list) -> list:
    res = []
    for item in list_data:
        piece_ids = item['piece_ids']
        # Lấy list token từ piece_ids
        tokens = tokenizer.convert_ids_to_tokens(piece_ids)
        reconstructed_text = ""
        token_offsets = []  # lưu offset của từng token dưới dạng (start, end)

        # Xây dựng văn bản và tính toán offset của từng token
        for tok in tokens:
            if tok.startswith("##"):
                # Nếu token bắt đầu bằng "##", nối trực tiếp token không có khoảng trắng
                token_clean = tok[2:]
                start = len(reconstructed_text)
                reconstructed_text += token_clean
                end = len(reconstructed_text) - 1
            else:
                # Thêm khoảng trắng nếu không phải token đầu tiên
                if reconstructed_text != "":
                    reconstructed_text += " "
                start = len(reconstructed_text)
                reconstructed_text += tok
                end = len(reconstructed_text) - 1
            token_offsets.append((start, end))

        reconstructed_text = reconstructed_text.replace(" ' ", "'")
        reconstructed_text = reconstructed_text.replace(" - ", "-")
        reconstructed_text = reconstructed_text.replace("[CLS] ", "")
        reconstructed_text = reconstructed_text.replace(" [SEP]", "")
        reconstructed_text = reconstructed_text.replace(" n't", "n't")

        # Tính offset dựa theo span trong item
        offsets = []
        
        for idx, sp in enumerate(item['span']):
            event = tokenizer.decode(item['piece_ids'][sp[0]: sp[1]+1], skip_special_tokens=True)
            # Tìm vị trí event trong tokens
            
            if event.startswith("##"):
                event = event[2:]
                
            # Thêm offset vào danh sách
            offsets.append(event)
        
        res.append({
            "piece_ids": item["piece_ids"],
            "span": item["span"],
            "label": item["label"],
            "text": reconstructed_text,
            # "tokens": reconstructed_text.split(),  # hoặc có thể giữ danh sách tokens gốc
            "event_words": offsets
        })
    return res

def convert(input_path:str, output_path:str, datasets:list)->None:
    os.makedirs(output_path, exist_ok=True)
    # Duyệt qua từng dataset
    for dataset in datasets:
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(os.path.join(output_path, dataset), exist_ok=True)
        
        # Convert for train data
        for i in range(5):
            input_folder = os.path.join(input_path, dataset, "perm"+str(i))
            # Kiểm tra xem thư mục có tồn tại không
            if not os.path.exists(input_folder):
                print(f"Folder {input_folder} does not exist!")
                continue
            
            # Tạo thư mục đầu ra cho mỗi perm
            output_folder = os.path.join(output_path, dataset, "perm"+str(i))
            os.makedirs(output_folder, exist_ok=True)
            
            for file_name in os.listdir(input_folder):
                if file_name.endswith(".jsonl"):
                    input_file = os.path.join(input_folder, file_name)
                    output_file = os.path.join(output_path, dataset, "perm"+str(i), file_name)
                    new_data = []
                    
                    with open(input_file, 'r') as f:
                        for line in f:
                            new_line = {}
                            # Chuyển đổi từng dòng JSON thành dict
                            json_line = json.loads(line)
                            for key, value in json_line.items():
                                new_line[key] = ids2list(value)
                            new_data.append(new_line)
                    
                    if new_data:
                        with open(output_file, 'w') as f:
                            for new_line in new_data:
                                f.write(json.dumps(new_line) + '\n')
                        print(f"Converted {input_file} to {output_file}")
                    else:
                        print(f"Error: No data found in {input_file}")
                    
        # Convert for test data
        input_file = os.path.join(input_path, dataset, f"{dataset}.test.jsonl")
        output_file = os.path.join(output_path, dataset, f"{dataset}.test.jsonl")
        new_data = None
        
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
            new_data = ids2list(data)
        
        if new_data:
            with open(output_file, 'w') as f:
                for item in new_data:
                    f.write(json.dumps(item) + '\n')
            print(f"Converted {input_file} to {output_file}")
        else:
            print(f"Error: No data found in {input_file}")
            
if __name__ == "__main__":
    convert(input_path, output_path, datasets)
    # convert(input_path, output_path, datasets)
        