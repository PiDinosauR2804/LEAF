from extractor.llm import Extractor, Extractor_Gemini, is_quota_exhausted_error, is_valid_extractor
from utils.convert import sent2ids
from extractor.api_key import GEMINI_KEY
import time
import threading
import queue
from loguru import logger
from tqdm.auto import tqdm

class Consumer:
    def __init__(self, task_queue:queue.Queue, num_try=3, max_consecutive_429_error=3, model='gemini-2.0-flash', 
                 candidate=1, max_num_threads=10, gen_des:bool=False)->None:
        # pram for consumer
        self.task_queue = task_queue
        self.num_try = num_try
        self.max_consecutive_429_error = max_consecutive_429_error
        self.model = model
        self.candidate = candidate
        self.max_num_threads = max_num_threads
        self.queue_waiting_time = 1
        self.extractors = []
        self.gen_des = gen_des
        # param for each consumtion time
        self.pbar = None
        self.results = []
        self.processed_item = 0
        self.remained_item = 0
        self.num_added_event = []
        self.num_added_attribute = []
        self.num_added_description = []
        # Attribute for threading
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.lock = threading.Lock()

        # Init extractors
        for i in range(len(GEMINI_KEY)):
            extractor = Extractor_Gemini(api_key=GEMINI_KEY[i])
            
            if is_valid_extractor(extractor):
                self.extractors.append({
                    'extractor': extractor,
                    'consecutive_429_error': 0,
                })
                
            if len(self.extractors) >= max_num_threads:
                break
        logger.info(f"[INFO] Found {len(self.extractors)} valid extractors")   
        self.pause_threads()
        self.threads = []
        for i in range(len(self.extractors)):
            # Create a thread for each extractor
            self.threads.append(threading.Thread(target=self.worker, args=(i,)))
            self.threads[i].start()
        logger.info(f"[START] Start {len(self.threads)} threads")
            
    def stop_threads(self):
        self.stop_event.set()
        self.resume_threads()
        for thread in self.threads:
            thread.join()
            
        logger.info(f"[STOP] All threads stopped")
            
    def pause_threads(self):
        self.pause_event.clear()
        time.sleep(self.queue_waiting_time + 1)  # Wait for a while before stopping the threads
        
        logger.info(f"[PAUSE] All threads paused")
        
    def resume_threads(self):
        self.pause_event.set()
        
        logger.info(f"[RESUME] All threads resumed")
    
    def clear_results(self):
        self.results = []
        self.processed_item = 0
        self.remained_item = 0
        self.num_added_event = []
        self.num_added_attribute = []
        
        logger.info(f"[CLEAR] All results cleared")
        
    def append_processed_item(self, line_idx, key, idx, item):
        with self.lock:
            self.results.append((line_idx, key, idx, item))
            self.processed_item += 1
            self.task_queue.task_done()
            self.pbar.update(1)
        
    def append_remained_item(self, line_idx, key, idx, item):
        with self.lock:
            self.results.append((line_idx, key, idx, item))
            self.remained_item += 1
            self.task_queue.task_done()
            self.pbar.update(1)
        
    def consume_left_items(self):
        while not self.task_queue.empty():
            line_idx, key, idx, item = self.task_queue.get_nowait()
            self.append_remained_item(line_idx, key, idx, item)

    def worker(self, worker_id):
        extractor = self.extractors[worker_id]
        
        while not self.stop_event.is_set():
            if extractor['consecutive_429_error'] >= self.max_consecutive_429_error:
                logger.critical(f"[FATAL] Worker {worker_id} got error 429 more than {self.max_consecutive_429_error:} times and stoping. {sum(thread.is_alive() for thread in self.threads) - 1} threads are still running.")
                # Stop the thread
                break
            
            self.pause_event.wait()  # Wait until the pause event is set
            
            try:
                line_idx, key, idx, item = self.task_queue.get(timeout=self.queue_waiting_time)
            except queue.Empty:
                continue
            
            # Check if the item is already processed
            if 'events' in item:
                self.append_processed_item(line_idx, key, idx, item)
                continue
            
            # Try to process the item
            event_list = None
            for i in range(self.num_try):
                try:
                    new_item = None
                    if self.gen_des:
                        event_list, num_added_event, num_added_attribute = \
                            extractor['extractor'].extract_event2(item['text'], item['event_words'], model=self.model, candidate=self.candidate)
                    
                        if event_list:
                            new_item = {
                                'text': item['text'],
                                'event_words': item['event_words'],
                                'label': item['label'],
                                'events': event_list,
                            }
    
                    else:
                        event_list, num_added_event, num_added_attribute = \
                            extractor['extractor'].extract_event(item['text'], model=self.model, candidate=self.candidate)
                            
                        if event_list:
                            new_item = {
                                'text': item['text'],
                                'events': event_list,
                            }
                            
                    extractor['consecutive_429_error'] = 0
                    if new_item:
                        new_item = sent2ids(new_item) # Add piece_ids, span and offsets
                        self.append_processed_item(line_idx, key, idx, new_item)
                        with self.lock:
                            self.num_added_event.append(num_added_event)
                            self.num_added_attribute.append(num_added_attribute)
                        break
                    
                except Exception as e:
                    if is_quota_exhausted_error(e):
                        extractor['consecutive_429_error'] += 1
                        time.sleep(extractor['extractor'].error_waiting_time)  # Wait for 15 seconds before retrying
                    else:
                        extractor['consecutive_429_error'] = 0
                        logger.error(f"[ERROR at ATTEMPT {i+1}/{self.num_try}] Worker {worker_id} got error: {e} | Text: {item['text']} | Trigger: {item.get('event_words')}")
                        # raise e
            else:
                if not event_list:
                    logger.error(f"[FAIL] Worker {worker_id} cannot extract event list after {self.num_try} attempts. | Text: {item['text']} | Trigger: {item.get('event_words')}")
                else:
                    logger.error(f"[FAIL] Worker {worker_id} fail to process event list: {event_list} | Text: {item['text']} | Trigger: {item.get('event_words')}")
                self.append_remained_item(line_idx, key, idx, item)

    
    def consume(self, pbar_des:str='Consuming item', model='gemini-2.0-flash', candidate=1):
        self.candidate = candidate
        self.model = model
        self.pbar = tqdm(total=self.task_queue.qsize(), desc=pbar_des, unit="item")
        self.clear_results()
        logger.info(f"[CONSUMING] Start consuming.")
        self.resume_threads()
        
        try:
            while self.task_queue.unfinished_tasks > 0:
                if not any(thread.is_alive() for thread in self.threads):
                    logger.critical(f"[FATAL] No threads are running. Consuming remained {self.task_queue.qsize()} items ...")
                    self.consume_left_items()
                    break
                
                time.sleep(1)
            
            self.pause_threads()
            
        except KeyboardInterrupt:
            logger.critical(f"[FALTAL] Consuming interrupted by user. Stopping safely...")
            self.stop_threads()
            self.consume_left_items()
            
        finally:
            self.pbar.close()
            logger.info(f"[FINISHED] Finished consuming {self.processed_item}/{len(self.results)} items with {self.remained_item}/{len(self.results)} remained items.")
            # Print describetion of the number of added events and attributes
            real_processed_item = len(self.num_added_event)
            if real_processed_item > 0:
                real_num_added_event = [num_added_event for num_added_event in self.num_added_event if num_added_event > 0]
                real_num_added_attribute = [num_added_attribute for num_added_attribute in self.num_added_attribute if num_added_attribute > 0]
                percent_is_added_event = len(real_num_added_event) / real_processed_item * 100
                # Calculate the average number of added events and attributes
                avg_added_event = sum(real_num_added_event) / real_processed_item
                avg_added_attribute = sum(real_num_added_attribute) / real_processed_item
                
                logger.info(f"[FINISHED] Really processed {real_processed_item}/{len(self.results)} items with {percent_is_added_event:.2f}% event list are added \
| AVG added events: {avg_added_event:.2f} | AVG added attributes: {avg_added_attribute:.2f}")
            
            return self.results, self.processed_item, self.remained_item