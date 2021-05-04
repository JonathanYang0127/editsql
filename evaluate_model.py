import argparse
import numpy as np
from functools import partial
import os
from model_util import evaluate_utterance_sample
from model.torch_utils import per_token_accuracy
import json

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str)

args = parser.parse_args()

SQL_KEYWORDS = [
    'select',
    'group',
    'where', 
    'order',
    'by',
    'join', 
    'on',
    'as',
    'desc',
    'asc'
]

###########################################################
#                       Metrics                           #
###########################################################
def accumulate(metric_fn, data):
    accuracies = []
    for d in data:
        value = metric_fn(d)
        if value is not None:
            print(value)
            accuracies.append(value)
  
    return accuracies


def compute_token_accuracy_question(data):
    gold_seq = data['gold_query']
    pred_seq = data['flat_prediction']

    return per_token_accuracy(gold_seq, pred_seq)

def compute_string_accuracy_question(data):
    gold_seq = data['gold_query']
    pred_seq = data['flat_prediction']

    if len(gold_seq) != len(pred_seq):
        print(gold_seq, pred_seq, '\n')
        return 0.0

    for i, gold_token in enumerate(gold_seq):
        if gold_token != pred_seq[i]:
            print(gold_seq, pred_seq, '\n')
            return 0.0

    return 1.0 

def compute_table_accuracy_question(data):
    gold_seq = data['gold_query']
    pred_seq = data['flat_prediction']

    table_keyword = 'from'
    if 'from' not in pred_seq:
        table_keyword = 'select'

    gold_table_idx, pred_table_idx = gold_seq.index(table_keyword), pred_seq.index(table_keyword)
    gold_end_idx, pred_end_idx = len(gold_seq), len(pred_seq)

    for j in range(gold_table_idx + 1, len(gold_seq)):
        if gold_seq[j] in SQL_KEYWORDS and not (gold_seq[j] in ['join', 'as']):
            gold_end_idx = j
            break

    for j in range(pred_table_idx + 1, len(pred_seq)):
        if pred_seq[j] in SQL_KEYWORDS and not (pred_seq[j] in ['join', 'as']):
            pred_end_idx = j
            break

    gold_subseq = gold_seq[gold_table_idx + 1: gold_end_idx]
    pred_subseq = pred_seq[pred_table_idx + 1: pred_end_idx]
    gold_tables, pred_tables = set(), set()

    for element in gold_subseq:
        if table_keyword == 'from' and not (element in [',', 'as', 'join']):
            if not (len(element) == 2 and element[0] == 't' and element[1].isdigit()):
                gold_tables.add(element)
        if table_keyword == 'select' and '.' in element:
            gold_tables.add(element)

    for element in pred_subseq:
        if table_keyword == 'from' and not (element in [',', 'as', 'join']):
            if not (len(element) == 2 and element[0] == 't' and element[1].isdigit()):
                pred_tables.add(element)
        if table_keyword == 'select' and '.' in element:
            pred_tables.add(element)

    print(gold_tables, pred_tables)
    if gold_tables == pred_tables:
        return 1.0
    return 0.0

def compute_interaction_match(data):
    question_accuracies = accumulate(compute_string_accuracy_question, data)

    last_interaction_start = -1
    for i, d in enumerate(data):
        if question_accuracies[i] is None:
            continue
        if d['index_in_interaction'] > 1:
            question_accuracies[last_interaction_start] *= question_accuracies[i]
            question_accuracies[i] = 0
        else:
            last_interaction_start = i

    return question_accuracies

def compute_index_question_accuracy(question_accuracy_fn, index, data):
    question_accuracies = accumulate(question_accuracy_fn, data)
    index_question_accuracies = []

    for i, d in enumerate(data):
        if question_accuracies[i] is None:
            continue
        if d['index_in_interaction'] == index:
            index_question_accuracies.append(question_accuracies[i])
        
    return index_question_accuracies

def compute_last_question_accuracy(question_accuracy_fn, data):
    question_accuracies = accumulate(question_accuracy_fn, data)
    last_question_accuracies = []

    for i, d in enumerate(data):
        if d['index_in_interaction'] == 1 and i != 0:
            last_question_accuracies.append(question_accuracies[i - 1])

    last_question_accuracies.append(question_accuracies[-1])
    return last_question_accuracies

METRIC_DICT = {
    'token_accuracy': partial(accumulate, compute_token_accuracy_question),
    'string_accuracy': partial(accumulate, compute_string_accuracy_question),
    'interaction_accuracy': compute_interaction_match,
    'table_match_accuracy': partial(accumulate, compute_table_accuracy_question),
    'first_question_token_accuracy': partial(compute_index_question_accuracy, compute_token_accuracy_question, 1),
    'first_question_string_accuracy': partial(compute_index_question_accuracy, compute_string_accuracy_question, 1),
    'second_question_token_accuracy': partial(compute_index_question_accuracy, compute_token_accuracy_question, 2),
    'second_question_string_accuracy': partial(compute_index_question_accuracy, compute_string_accuracy_question, 2),
    'last_question_token_accuracy': partial(compute_last_question_accuracy, compute_token_accuracy_question),
    'last_question_string_accuracy': partial(compute_last_question_accuracy, compute_string_accuracy_question)
}



##########################################################
#                      Evaluation                        #
##########################################################
def get_latest_model(log_dir):
    latest_model, latest_version = None, -1

    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if 'save' in f:
                version = int(f[5:])
                if version > latest_version:
                    latest_model, latest_version = os.path.join(root, f), version

    return latest_model

def get_predictions_file(log_dir):
    return os.path.join(log_dir, 'valid_use_predicted_queries_predictions.json')


def evaluate(pred_file, metrics):
    metric_values = {}
    data = []
    with open(pred_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    for metric_str in metrics:
        value = METRIC_DICT[metric_str](data)
        metric_values[metric_str] = value

    return metric_values


pred_file = get_predictions_file(args.log_dir)
metric_values = evaluate(pred_file, ['table_match_accuracy'])#METRIC_DICT.keys())

for key, value in metric_values.items():
    print(key, np.mean(value))

