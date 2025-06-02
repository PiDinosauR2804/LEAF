python extract_event.py \
    --input_root data/data_text \
    --output_root output \
    --datasets MAVEN \
    --model gemini-2.0-flash-lite \
    --candidate 1 \
    --num_try 3 \
    --max_consecutive_429_error 3 \
    --max_num_threads 10 \
    --logs_dir logs/extractor