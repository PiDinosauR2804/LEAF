import argparse

def extractor_parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', default='./data/data_text', type=str)
    parser.add_argument('--output_root', default='./output', type=str)
    parser.add_argument('--datasets', default=['MAVEN'], type=list, nargs='+')
    parser.add_argument('--model', default='gemini-2.0-flash', type=str)
    parser.add_argument('--candidate', default=1, type=int)
    parser.add_argument('--num_try', default=3, type=int)
    parser.add_argument('--max_consecutive_429_error', default=3, type=int)
    parser.add_argument('--max_num_threads', default=10, type=int)
    parser.add_argument('--logs_dir', default='./logs/extractor', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    
    args, _ = parser.parse_known_args()

    return args