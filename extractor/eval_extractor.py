import os
import json
from loguru import logger

def bleu_score(prediction:list[int], reference:list[int]):
    """
    Calculate BLEU score for a single prediction and reference.
    """
    # Initialize the BLEU score
    bleu = 0.0
    # Calculate the BLEU score for each n-gram (1-gram, 2-gram, etc.)
    for n in range(1,5):
        # Create n-grams for prediction and reference
        pred_ngrams = [tuple(prediction[i:i+n]) for i in range(len(prediction)-n+1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
        # Count the number of matches between prediction and reference n-grams
        matches = sum(1 for ngram in pred_ngrams if ngram in ref_ngrams)
        # Calculate precision
        precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
        # Calculate BLEU score using geometric mean of precisions
        bleu += precision / 4.0
    return bleu

def eval(gt_list:list, pr_list:list):
    bleu_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Sort the lists by line_idx, key and idx
    for gt, pr in zip(gt_list, pr_list):
        try:
            # gt is a list of tuples (line_idx, key, idx, item)
            # pr is a list of tuples (line_idx, key, idx, item)
            gt_line_idx, gt_key, gt_idx, gt_item = gt
            pr_line_idx, pr_key, pr_idx, pr_item = pr
            if gt_line_idx != pr_line_idx or gt_key != pr_key or gt_idx != pr_idx:
                logger.error(f"[ERROR] Line index or key or idx do not match: {gt}\nvs\n{pr}")
                continue
            
            if pr_item.get('span') is None:
                # logger.error(f"[ERROR] Span is None in prediction: {pr_item}")
                continue
            
            # Calculate BLEU score for each item
            bleu = bleu_score(gt_item['text'], pr_item['text'])
            # get span at the position that corresponding label > 0
            gt_span = [span for label, span in zip(gt_item['label'], gt_item['span']) if label > 0]
            pr_span = pr_item['span']
            gt_set = set([tuple(span) for span in gt_span])
            pr_set = set([tuple(span) for span in pr_span])#
            # print false positive samples
            false_positive = pr_set - gt_set
            if len(false_positive) > 0:
                # logger.error(f"[ERROR] False positive samples: {false_positive} | Text: {pr_item['text']}")
                pass
            # print false negative samples
            false_negative = gt_set - pr_set
            if len(false_negative) > 0:
                # logger.error(f"[ERROR] False negative samples: {false_negative} | Text: {gt_item['text']}")
                pass
            true_positive = len(gt_set & pr_set)

            precision = true_positive / len(pr_span) if len(pr_span) > 0 else 0.0
            recall = true_positive / len(gt_span) if len(gt_span) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            bleu_scores.append(bleu)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        except Exception as e:
            logger.error(f"[ERROR] Error in calculating scores: {e}")
            logger.error(f"[ERROR] GT: {gt_item} | PR: {pr_item}")
            raise e
        
    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) > 0 else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if len(precision_scores) > 0 else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if len(recall_scores) > 0 else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0
    
    return avg_bleu, avg_precision, avg_recall, avg_f1
        