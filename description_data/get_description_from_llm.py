import json
import os
import google.generativeai as genai
import base64
import os
import time
import grpc
import json
import re

with open(r'description_data\trigger_dict.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
def get_list_trigger(idx):
    return ", ".join(map(str, data[idx]))

NUM = 5

PROMPT = """
Tôi có một danh sách các cụm từ gọi là "Event". Hãy viết một mô tả thật đầy đủ, rõ ràng và súc tích để giải thích ý nghĩa, hoặc hàm ý chung các cụm từ đó chỉ.


⚠️ Không nên nhắc lại các từ trong danh sách event.
⚠️ Câu trả lời đầu ra phải theo định dạng:
Description: ...

✅ Mỗi lần thực hiện, hãy cố gắng thay đổi cách diễn đạt, cấu trúc câu hoặc lựa chọn từ ngữ để tạo ra phiên bản khác biệt nhưng đúng nghĩa.

❌ Không cần thêm tiêu đề, giải thích, hay câu dẫn nào khác trong đầu ra.
"""

def generative_model(api_key):
    # Cấu hình API với khóa
    genai.configure(api_key=api_key)

def send_request_with_retry(model, content, max_retries=5, retry_delay=30):
    """
    Gửi yêu cầu đến API với cơ chế thử lại nếu gặp lỗi API limit (429).
    :param content: Nội dung gửi đến API.
    :param max_retries: Số lần thử lại tối đa.
    :param retry_delay: Thời gian chờ giữa các lần thử lại (giây).
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(contents=content)
            time.sleep(retry_delay)
            text = response.text
            
            match = re.search(r'Description:\s*(.*)', text)
            if match:
                description = match.group(1).strip()
                return description  # Thành công, trả về ngay
            else:
                print(f"[Cảnh báo] Không tìm thấy 'Description:' trong phản hồi. "
                    f"Thử lại... (Lần thử {attempt + 1}/{max_retries})")
        except grpc.RpcError as rpc_error:
            # Kiểm tra mã lỗi
            if rpc_error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                print(f"[Lỗi {rpc_error.code()}] Đã đạt giới hạn API. "
                      f"Thử lại sau {retry_delay} giây... (Lần thử {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)  # Chờ trước khi thử lại
            else:
                print(f"[Lỗi {rpc_error.code()}] Chi tiết: {rpc_error.details()}")
                break  # Lỗi không liên quan đến giới hạn API, dừng thử lại
        except Exception as e:
            print(f"Lỗi không xác định: {e}")
            break
    return None  # Trả về None nếu vượt quá số lần thử lại

def convert_to_format(filename, raw_text, text):
    idx = filename.split('.')[0]
    sample = {
        "id": idx,
        "num_text": len(raw_text),
    }
    return sample

def main():
    # Khởi tạo API Key và cấu hình
    
    api_keys = [
 
    ]
    
    API_KEY = api_keys[0]
    
    genai.configure(api_key=API_KEY)
    
    generation_config = {
            "temperature": 2,
            "top_p": 0.9,
            "top_k": 50,
            "max_output_tokens": 512,
            "response_mime_type": "text/plain",
        }

    model = genai.GenerativeModel(
          model_name="gemini-2.0-flash-exp",
          generation_config=generation_config,
        )
    
    
    description_dict= {}
    
    for key, value in data.items():
        temp = []
        for i in range(NUM):    
            QUESTION = f"Các cụm từ: {get_list_trigger(key)}"
        
            content = [
                {
                    "role": "user",
                    "parts": [
                        PROMPT,
                        QUESTION,
                    ],
                }
            ]

            result = send_request_with_retry(model, content)

            temp.append(result)

            if result:
                print("Đáp án:", result)
            else:
                print("Không thể hoàn thành yêu cầu sau nhiều lần thử lại.")
        description_dict[key] = temp
        
    output_path = 'description_data/description_trigger_dict.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(description_dict, f, ensure_ascii=False, indent=4)

    print(f'✅ Đã lưu description_trigger_dict vào: {output_path}')
            

if __name__ == "__main__":
    main()
