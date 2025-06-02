from transformers import BertTokenizerFast
import random

# Khởi tạo tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def ids2sent(item:dict)->dict:
    # Chuyển đổi các ID thành văn bản
    text = tokenizer.decode(item['piece_ids'], skip_special_tokens=True)
    # Tạo một danh sách rỗng để lưu trữ các offset
    offsets = []
    for sp in item['span']:
        event = tokenizer.decode(item['piece_ids'][sp[0]: sp[1]+1], skip_special_tokens=True)
        # Tìm vị trí event trong tokens
        start = text.find(event)
        end = start + len(event) - 1
        
        if start == -1 or end == -1:
            print(f"Error: {event} not found in {text}")
            continue
        # Thêm offset vào danh sách
        offsets.append([start, end])
        
    return{
        "text": text, 
        "tokens": text.split(), 
        "offsets": offsets, 
        "label": item['label']
    }
    
def ids2sent_batch(list_sent:list)->list:
    # Chuyển đổi danh sách các ID thành văn bản
    sent_data = []
    for item in list_sent:
        sent_data.append(ids2sent(item))
    return sent_data

def sent2ids(item: dict) -> dict:
    opt = tokenizer(item['text'], return_offsets_mapping=True, truncation=True, max_length=512, padding="max_length")
    piece_ids = [piece_id for piece_id in opt['input_ids'] if piece_id != tokenizer.pad_token_id]
    offsets_mp = opt['offset_mapping']
    input_tokens = tokenizer.convert_ids_to_tokens(opt['input_ids'])

    # Lọc ra các offset hợp lệ (bỏ qua [CLS], [SEP], padding)
    valid_offsets = [(i, span) for i, span in enumerate(offsets_mp) if span != (0, 0)]
    
    span = []
    for event in item['events']:
        trigger_word = event['trigger_word']
        start_char = item['text'].find(trigger_word)
        if start_char == -1:
            raise ValueError(f"Trigger word '{trigger_word}' not found in text: {item['text']}")

        end_char = start_char + len(trigger_word)

        # Tìm token span mà offset nằm trong khoảng trigger word
        start_token_idx = end_token_idx = None
        for i, (s, e) in valid_offsets:
            if s <= start_char < e:
                start_token_idx = i
            if s < end_char <= e:
                end_token_idx = i
            if start_token_idx is not None and end_token_idx is not None:
                break

        if start_token_idx is not None and end_token_idx is not None:
            span.append((start_token_idx, end_token_idx))
        else:
            raise ValueError(f"Span for trigger word '{trigger_word}' not found in offsets mapping | text: {item['text']}")

    final_item = item | {
        'piece_ids': piece_ids,
        'span': span
    }
    return final_item


def sent2ids_batch(list_sent:list)->list:
    ids_data = []
    for item in list_sent:
        ids_data.append(sent2ids(item))
    return ids_data


def sent2ids_expand(item:dict, neg_size=5)->dict:
    new_item = item.copy()
    
    # augment negative event
    trigger_words = [event['trigger_word'] for event in item['events']]
    text = item['text']
    neg_words = [word for word in text.replace('.', ' . ').split() if word not in trigger_words]
    neg_words = list(set(neg_words))
    neg_words = [word for word in neg_words if len(word) > 1]
    neg_words = random.sample(neg_words, min(neg_size, len(neg_words)))
    
    for neg_word in neg_words:
        new_item['events'].append({
            'trigger_word': neg_word,
        })
        
        new_item['label'].append(0)
        
    final_item = sent2ids(new_item)
    
    return final_item

def sent2ids_expand_batch(list_sent:list, neg_size=5)->list:
    ids_data = []
    for item in list_sent:
        ids_data.append(sent2ids_expand(item, neg_size))
    return ids_data