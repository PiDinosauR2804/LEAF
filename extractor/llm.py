from google import genai
from google.genai import types
import time
import re
import ast
import random
from loguru import logger
import json

class Extractor():
    def __init__(self, api_key:str):
        """
        Initialize the Extractor with the provided API key and model.
        """
        self.api_key = api_key
        self.error_waiting_time = 0.1
    
    def extract_event(self, text:str, **args):
        """
        Extract events from text using Google Gemini API.
        """
        ran = random.random()
        if ran < 0.7:
            return [{"text": text, 
                    "event_type": "meeting", 
                    "trigger_word": random.choice(text.split()),
                    "event_time": None, 
                    "event_location": None, 
                    "event_participants": []}], random.randint(0, 2), random.randint(0, 2)  # Dummy response for the base class
        elif ran < 0.8:
            return [{"text": text, 
                    "event_type": "meeting", 
                    "trigger_word": "abc", 
                    "event_time": None, 
                    "event_location": None, 
                    "event_participants": []}], 0, 0  # Dummy response for the base class
        elif ran < 0.9:
            return [], 0, 0
        else:
            raise Exception("RESOURCE_EXHAUSTED")  # Simulate quota exhaustion
        
class Extractor_Gemini(Extractor):
    def __init__(self, api_key:str):
        """
        Initialize the Extractor with the provided API key and model.
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.error_waiting_time = 15
        self.prompt = """You are an event extraction expert. Given a text, your task is to extract event triggers 
and generate structured information for each event.

Each extracted event should be represented as a dictionary with the following keys:
- "event_type": a brief label describing the type of event,
- "trigger_word": the verb or noun in the text that indicates the occurrence of the event,
- "event_time": the time expression mentioned in the text for the event (or None if not present),
- "event_location": the location where the event occurred (or None if not present),
- "event_participants": a list of entities involved in the event (or None if not present),
- "description": a concise explanation of the event based on the trigger word and context.

If any key information is missing from the text, assign it a value of `None`.

You must extract **at least one** event from the text.

At the end of your response, output the extracted events in the following format:
The events are: [{{...}}, {{...}}, ...]

### Examples:

1. Input: "John and Mary met at the park on Monday"
   Output:  
   The events are: [
       {{
           "event_type": "meeting",
           "trigger_word": "met",
           "event_time": "Monday",
           "event_location": "park",
           "event_participants": ["John", "Mary"],
           "description": "The trigger word 'met' refers to a meeting between John and Mary that took place at the park on Monday."
       }}
   ]

2. Input: "The July 2006 earthquake was also centered in the Indian Ocean, from the coast of Java, and had a duration of more than three minutes."
   Output:  
   The events are: [
       {{
           "event_type": "catastrophe",
           "trigger_word": "earthquake",
           "event_time": "July 2006",
           "event_location": "Indian Ocean",
           "event_participants": None,
           "description": "The trigger word 'earthquake' refers to a natural disaster event that occurred in July 2006 in the Indian Ocean."
       }},
       {{
           "event_type": "placing",
           "trigger_word": "centered",
           "event_time": "July 2006",
           "event_location": "Indian Ocean",
           "event_participants": None,
           "description": "The trigger word 'centered' indicates the location focus of the earthquake in the Indian Ocean."
       }}
   ]

Pay close attention to identifying suitable trigger words—these are typically verbs or nouns that signal an action or occurrence.

Now, extract the list of event dictionaries from the following text (write your answer, do not code):
{content}
"""

        self.prompt2 = """You are an event extraction expert. Given a sentence and a list of trigger word in that sentence, 
for each trigger word, you extract arguments related to that event and generate {k} rich, context-rich descriptions of that trigger word. 
Your answer should be a dictionary and you must write in the following format:
{{
    "trigger_word": str,
    "event_type": str,
    "event_time": str,
    "event_location": str,
    "event_participants": list[str],
    "description": list[str]
}}
Assign None if the value is not present in the text.
For example, given the text "John and Mary met and signed the contract at a cafe on Monday", the trigger word list is ["met", "signed"], you should write the answer in the following format:
The events are:
[
    {{
        "trigger_word": "met",
        "event_type": "meeting",
        "event_time": "Monday",
        "event_location": "cafe",
        "event_participants": ["John", "Mary"],
        "description": [
            "The trigger word 'met' refers to a meeting between two or more people.",
            "The event 'met' means that someone met someone else at a specific time and place.",
            ... (k descriptions)
        ]
    }},
    {{
        "trigger_word": "signed",
        "event_type": "signing",
        "event_time": "Monday",
        "event_location": "cafe",
        "event_participants": ["John", "Mary"],
        "description": [
            "Signing is that a person signs a document or agreement.",
            "The event 'signed' indicates that an agreement was formalized at a specific time and place.",
            ... (k descriptions)
        ]
    }}
]
Now, given the text: {content};
and trigger word lists: {trigger_words}.
please extract (write your answer, do not code) the event arguments and generate {k} rich, context-rich descriptions of each event in order.
"""

    def response_to_string(self, response, idx=0):
        if idx > len(response.candidates):
            idx = 0
        output = []
        
        for part in response.candidates[idx].content.parts:
            if part.text is not None:
                output.append(part.text)
            if part.executable_code is not None:
                output.append(f"```python\n{part.executable_code.code}\n```")  # Định dạng mã code
            if part.code_execution_result is not None:
                output.append(f"Output:\n{part.code_execution_result.output}")
            if part.inline_data is not None:
                output.append("[Hình ảnh được nhúng]")  # Không thể hiển thị trực tiếp hình ảnh trong chuỗi

        return "\n".join(output)

    def extract_response(self, text:str):
        match = re.search(r"\[\s*{.*?}\s*](?=\s*$|\s*[,.\n])", text, re.DOTALL)

        if match:
            events_str = match.group()
            try:
                events_list = ast.literal_eval(events_str.replace("null", "None").replace("true", "True").replace("false", "False"))
                # events_list = json.loads(events_str.replace("None", "null").replace("True", "true").replace("False", "false"))
                return events_list
            
            except ValueError as e:
                logger.error(f"[EXTRACT RESPONSE] Error parsing events: {e} | Text: {text}")
                return None
        else:
            logger.error(f"[EXTRACT RESPONSE] No events found in the response | Text: {text}")
            return None
        
    def validate_event_list(self, event_list:list)->list:
        """
        Validate the event list to ensure it contains the required keys.
        """
        valid_events = []
        for event in event_list:
            if not isinstance(event, dict):
                logger.error(f"[VALIDATE EVENT LIST] Event is not a dict: {event}")
            else:
                for key, value in event.items():
                    if isinstance(value, str):
                        if len(value) == 0:
                            event[key] = None
                        else:
                            event[key] = value.lower().strip()
                            
                    elif isinstance(value, list):
                        if len(value) == 0:
                            event[key] = None
                        else:
                            event[key] = [v.lower().strip() for v in value if isinstance(v, str)]
                    else:
                        event[key] = None
                        
            if event.get("trigger_word") is not None:
                if event.get("trigger_word"):
                    valid_events.append(event)
                else:
                    logger.error(f"[VALIDATE EVENT LIST] Trigger word not in text: {event.get('trigger_word')} | Text: {event.get('text')}")
            else:
                logger.error(f"[VALIDATE EVENT LIST] Event not have trigger_word attribute: {event} | Text: {event.get('text')}")

        return valid_events
    
    def merge_event_list(self, event_lists: list[list[dict]]) -> tuple[list[dict], int, int]:
        """
        Merge multiple event lists into one, grouped by trigger_word.
        Avoids duplicate events and merges attributes from similar events.
        """
        trigger2event = {}
        num_added_event = 0
        num_added_attribute = 0

        for event_list in event_lists:
            for event in event_list:
                trigger = event['trigger_word']
                if trigger not in trigger2event:
                    trigger2event[trigger] = event.copy()
                    num_added_event += 1
                else:
                    existing_event = trigger2event[trigger]
                    for key, value in event.items():
                        if not value:
                            continue
                        old_value = existing_event.get(key)
                        if not old_value:
                            existing_event[key] = value
                            num_added_attribute += 1
                        elif isinstance(old_value, list) and isinstance(value, list):
                            merged = list(set(old_value + value))
                            if len(merged) > len(old_value):
                                existing_event[key] = merged
                                num_added_attribute += 1
        num_added_event -= len(event_lists[0])
        merged_list = list(trigger2event.values())
        return merged_list, num_added_event, num_added_attribute


    def extract_event(self, text:str, model="gemini-2.0-flash", candidate=1):
        """
        Extract events from text using Google Gemini API.
        """
        # Gen answer
        response = self.client.models.generate_content(
            model=model,
            contents=self.prompt.format(content=text),
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                candidate_count=candidate
            )
        )

        res = []
        for idx in range(len(response.candidates)):
            response_string = self.response_to_string(response, idx)
            event_list = self.extract_response(response_string)
            if event_list:
                valid_event_list = self.validate_event_list(event_list)
                if valid_event_list:
                    res.append(valid_event_list)
                else:
                    logger.error(f"[EXTRACT EVENT] Failed to validate event list: {event_list} | Text: {text}")
            else:
                logger.error(f"[EXTRACT EVENT] Failed to extract event list from response: {response_string} | Text: {text}")

        if res:
            merged_list, num_added_event, num_added_attribute = self.merge_event_list(res)
            return merged_list, num_added_event, num_added_attribute

        return [], 0, 0
    
    def rearrange_event_list(self, event_list:list, trigger_words:list)->list:

        """
        Rearrange the event list to match the order of trigger words.
        """
        trigger2event = {event['trigger_word']: event for event in event_list}
        rearranged_list = []
        
        for trigger in trigger_words:
            if trigger in trigger2event:
                rearranged_list.append(trigger2event[trigger])
            else:
                logger.error(f"[REARRANGE EVENT LIST] Trigger word not found in event list: {trigger} | Event List: {event_list}")
                # rearranged_list.append({"trigger_word": trigger, "event_type": None, "event_time": None, "event_location": None, "event_participants": None, "description": None})
        
        return rearranged_list
        
        
    def extract_event2(self, text:str, trigger_words:str, k=3, model="gemini-2.0-flash", candidate=1):
        """
        Extract events from text using Google Gemini API.
        """
        # Gen answer
        response = self.client.models.generate_content(
            model=model,
            contents=self.prompt2.format(content=text, trigger_words=trigger_words, k=k),
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                candidate_count=candidate
            )
        )

        res = []
        for idx in range(len(response.candidates)):
            response_string = self.response_to_string(response, idx)
            event_list = self.extract_response(response_string)
            
            if event_list:
                valid_event_list = self.validate_event_list(event_list)
                if valid_event_list:
                    res.append(valid_event_list)
                else:
                    logger.error(f"[EXTRACT EVENT 2] Failed to validate event list: {event_list} | Text: {text}")
            else:
                logger.error(f"[EXTRACT EVENT 2] Failed to extract event list from response: {response_string} | Text: {text}")

        if res:
            merged_list, num_added_event, num_added_attribute = self.merge_event_list(res)
            final_list = self.rearrange_event_list(merged_list, trigger_words)
            if len(final_list) == len(trigger_words):
                return final_list, num_added_event, num_added_attribute
            else:
                logger.error(f"[EXTRACT EVENT 2] Number of events extracted ({len(merged_list)}) does not match number of trigger words ({len(trigger_words)}) | Text: {text}")

        return [], 0, 0
            
    
    
def is_quota_exhausted_error(e: Exception):
    return "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e) or "503" in str(e)

def is_valid_extractor(extractor, text="australia won the tournament, beating pakistan in the final by 25 runs.", max_try=2, timeout=5):
    for _ in range(max_try):
        try:
            _ = extractor.extract_event(text, model="gemini-2.0-flash", candidate=1)
        except Exception as e:
            if is_quota_exhausted_error(e):
                return False
            time.sleep(timeout)
    return True