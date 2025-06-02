from collections import defaultdict
import hashlib
import os
import json

# Chuẩn hóa văn bản để tạo khóa duy nhất
def get_text_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# Load trạng thái đã xử lý từ trước
def load_status(file_path):
    # Xóa file phục vụ debug
    if os.path.exists(file_path):
        os.remove(file_path)
    
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r') as f:
        return set(json.loads(line)['hash'] for line in f)

# Ghi trạng thái sau khi xử lý
def save_status(file_path, text):
    with open(file_path, 'a') as f:
        f.write(json.dumps({'hash': get_text_hash(text)}) + '\n')

def is_duplicate_hashes(list_data):
    hash_map = defaultdict(list)
    
    for idx, item in enumerate(list_data):
        text_hash = get_text_hash(item['text'])
        hash_map[text_hash].append(idx)

    duplicates = {h: idxs for h, idxs in hash_map.items() if len(idxs) > 1}
    
    if duplicates:
        return True
    else:
        return False
