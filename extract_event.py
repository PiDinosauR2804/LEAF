from extractor.producer import Producer
from extractor.consumer import Consumer
from extractor.eval_extractor import eval
from configs import parse_arguments
# from extractor.extractor_config import extractor_parse_arguments as parse_arguments
import json
import os
import time
import queue
from loguru import logger
from tqdm.auto import tqdm

def get_lines_from_results(results:list)->list:
    # Convert results to a list of lines
    output_lines = []
    for line_idx, key, idx, item in results:
        if line_idx >= len(output_lines):
            output_lines.append({})
        if key is not None and idx is not None:
            if key not in output_lines[line_idx]:
                output_lines[line_idx][key] = []
            output_lines[line_idx][key].append(item)
        else:
            output_lines[line_idx] = item
    return output_lines

def run(origin_input_path:str, output_path:str, datasets:list, perms:list, model:str, candidate:int, num_try:int, 
        max_consecutive_429_error:int, max_num_threads:int, resume:bool=False, convert_test:bool=False, gen_des:bool=False)->None:
    # Resume from output_path if resume is True
    if resume:
        input_path = output_path
    else:
        input_path = origin_input_path
        os.makedirs(output_path, exist_ok=True)

    # Create task queue, producer and consumer
    task_queue = queue.Queue()
    producer = Producer(task_queue)
    consumer = Consumer(task_queue, num_try=num_try, max_consecutive_429_error=max_consecutive_429_error, 
                        model=model, candidate=candidate, max_num_threads=max_num_threads, gen_des=gen_des)
    
    # Convert dataset
    for dataset in datasets:
        os.makedirs(os.path.join(output_path, dataset), exist_ok=True)

        for i in perms:
            input_folder = os.path.join(input_path, dataset, "perm"+str(i))
            origin_input_folder = os.path.join(origin_input_path, dataset, "perm"+str(i))
            if not os.path.exists(input_folder):
                logger.error(f"[ERROR] Folder {input_folder} is not exist", mode="ERROR")
                continue

            output_folder = os.path.join(output_path, dataset, "perm"+str(i))
            os.makedirs(output_folder, exist_ok=True)

            for file_name in os.listdir(input_folder):
                if not file_name.endswith(".jsonl"):
                    continue

                input_file = os.path.join(input_folder, file_name)
                origin_input_file = os.path.join(origin_input_folder, file_name)
                output_file = os.path.join(output_folder, file_name)
                
                start_time = time.time()
                # Start producing
                logger.info("="*100)
                logger.info(f"Start processing {input_file} to {output_file}")
                logger.info("="*100)
                gt_list = producer.produce(input_file, origin_input_file, is_train=True)
                # Start consuming
                try:
                    results, processed_item, remained_item = consumer.consume(pbar_des=f'{dataset}/perm{i}/{file_name}', model=model, candidate=candidate) # results is a list of tuples (line_idx, key, idx, item)
                except KeyboardInterrupt:
                    logger.error(f"[ERROR] KeyboardInterrupt, stop consuming safely...")
                    consumer.stop_threads()
                    
                
                # Save results to output file
                # Sort results by line_idx, key and idx
                results.sort(key=lambda x: (x[0], x[1], x[2])) # results is a list of tuples (line_idx, key, idx, item) 
                output_lines = get_lines_from_results(results) # output_lines is a list of lines
                
                with open(output_file, 'w') as f:   
                    for line in output_lines:                       
                            f.write(json.dumps(line) + '\n')
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                avg_bleu, avg_precision, avg_recall, avg_f1 = eval(gt_list, results)
                logger.info(f"[FINISHED] Processing {processed_item}/{len(results)} item, remaning {remained_item}/{len(results)} in {elapsed_time:.2f} seconds")
                logger.info(f"[FINISHED] BLEU: {avg_bleu:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")
                logger.info(f"[SAVE FILE] All item saved to {output_file}")            

        if convert_test:
            # Convert test
            input_file = os.path.join(input_path, dataset, f"{dataset}.test.jsonl")
            origin_input_file = os.path.join(origin_input_path, dataset, f"{dataset}.test.jsonl")
            output_file = os.path.join(output_path, dataset, f"{dataset}.test.jsonl")

            if not os.path.exists(input_file):
                logger.error(f"[ERROR] File {input_file} is not exist", mode="ERROR")
                continue
            
            start_time = time.time()
            # Start producing
            logger.info("="*100)
            logger.info("="*100)
            producer.produce(input_file, origin_input_file, is_train=False)
            # Start consuming
            try:
                results, processed_item, remained_item = consumer.consume(pbar_des=f'{dataset}/{dataset}/.test.jsonl', model=model, candidate=candidate) # results is a list of tuples (line_idx, key, idx, item)
            except KeyboardInterrupt:
                logger.error(f"[ERROR] KeyboardInterrupt, stop consuming safely...")
                consumer.stop_threads()
            
            # Save results to output file
            # Sort results by line_idx, key and idx
            results.sort(key=lambda x: (x[0])) # results is a list of tuples (line_idx, None, None, item)
            output_lines = get_lines_from_results(results) # output_lines is a list of lines
            
            with open(output_file, 'w') as f:
                for line in output_lines:
                    if line is not None:                        
                        f.write(json.dumps(line) + '\n')
                        
            end_time = time.time()
            elapsed_time = end_time - start_time
            # logger.info(f"[FINISHED] Processing {processed_item}/{len(results)} item, remaning {remained_item}/{len(results)} in {elapsed_time:.2f} seconds")
            logger.info(f"[SAVE FILE] All item saved to {output_file}")
        
    consumer.stop_threads()
    logger.info(f"[FINISHED] All items processed and saved to {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    input_root = args.input_root
    output_root = args.output_root
    datasets = args.datasets
    perms = args.perms
    model = args.model
    candidate = args.candidate
    num_try = args.num_try
    max_consecutive_429_error = args.max_consecutive_429_error
    max_num_threads = args.max_num_threads
    resume = args.eresume
    convert_test = args.convert_test
    logs_dir = args.logs_dir
    gen_des = args.gen_des
    
    # Configure logging
    os.makedirs(logs_dir, exist_ok=True)
    # --- Xoá handler mặc định ---
    logger.remove()
    # --- Thêm handler ghi log ra file ---
    date_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger.add(os.path.join(logs_dir, f"{date_str}.log"), rotation="1 MB", retention="10 days", enqueue=True, level="INFO")
    # --- Thêm handler ghi log qua tqdm.write ---
    # Ghi log ra console qua tqdm.write + có màu
    logger.level("CRITICAL", color="<bg red><white>")
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file: >18}: {line: <4}</cyan> - <level>{message}</level>",
    )
    
    # Log arguments
    logger.info(f"[INFO] Arguments:")
    logger.info(f"Input root: {input_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Perms: {perms}")
    logger.info(f"Model: {model}")
    logger.info(f"Candidate: {candidate}")
    logger.info(f"Num try: {num_try}")
    logger.info(f"Max consecutive 429 error: {max_consecutive_429_error}")
    logger.info(f"Max num threads: {max_num_threads}")
    logger.info(f"Resume: {resume}")
    logger.info(f"Convert test: {convert_test}")
    logger.info(f"Logs dir: {logs_dir}")
    logger.info(f"Gen des: {gen_des}")
    logger.info(f"[INFO] Start processing...")

    run(input_root, output_root, datasets, perms, model, candidate, num_try, max_consecutive_429_error, max_num_threads, resume, convert_test, gen_des)