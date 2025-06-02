from google import genai
from google.genai import types
import re

client = genai.Client(api_key='AIzaSyAwHAkDEc2rMrMj_1Ga5dHFzBw1___5n3Y')
# Prompt dùng để gen description cho trigger
prompt = """You are an event extraction expert. Given a text and a trigger word, you should generate a description for the trigger with the format:
The description is: [...].

For example:
The text is "John and Mary met at the park on Monday" and the trigger word is "met", the output should be:
The description is: [The trigger word "met" refers to the event where two or more parties encountered each other, marking the occurrence of a meeting or interaction.]

Now, please generate a description for the following text and trigger word:
The text is: '{content}' and the trigger word is: '{trigger_word}'
"""
# Chuyển các description thành string list
def description_to_string(response_list: list) -> list:
    if len(response_list) == 0:
        print("No response candidates found.")
        return None
    output = [] # List các description

    for des in response_list:
        for part in des.content.parts:
            if part.text is not None:
                output.append(part.text)
    return output

# Extract description từ response
def extract_description(response_list:list) -> list:
    des_list = []
    for des in response_list:
        match = re.search(r'The description is: \[\s*(.*)\]', des, re.DOTALL)
        if match:
            des_str = match.group(1)
            try:
                des_list.append(des_str)
            except ValueError as e:
                print(f'Error parsing description: {e}')
        else:
            print('No description found in the response.')
    if des_list is not None:
        return des_list
    else:
        print("No description found.")
        return None        

def gen_description_llm(text:str, trigger_word:str, model="gemini-2.0-flash", candidate=1):
    description = client.models.generate_content(
        model=model,
        contents=prompt.format(content=text, trigger_word=trigger_word),
        config=types.GenerateContentConfig(
            response_modalities=['TEXT'],
            candidate_count=candidate
        )
    )
    description_list = description_to_string(description.candidates)
    description_list = extract_description(description_list)
    return description_list