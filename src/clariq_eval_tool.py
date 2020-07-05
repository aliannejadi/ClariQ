import pandas as pd
import argparse
import pickle
from os import path
import json
from statistics import mean
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def evaluate_clarification_need(experiment_type, data_dir, run_file, out_file):
    if experiment_type in ['train', 'dev']:
        label_file_path = path.join(data_dir, '{}.json'.format(experiment_type))
    else:
        label_file_path = path.join(data_dir, '{}.json'.format(experiment_type))
        raise FileNotFoundError  # TODO: remove when test labels released.
    clarification_labels_dict = pd.read_json(label_file_path).drop_duplicates('topic_id').set_index('topic_id')[
        'clarification_need'].to_dict()
    run_dict = pd.read_csv(run_file, sep=' ', header=None).set_index(0)[1].to_dict()
    y_true = []
    y_pred = []
    for topic_id in clarification_labels_dict:
        y_true.append(clarification_labels_dict[topic_id])
        try:
            y_pred.append(run_dict[topic_id])
        except KeyError:  # no prediction provided in the run file, so we put a dummy label.
            y_pred.append(0)
    print('Precision: ', precision_score(y_true, y_pred, average='micro'))
    print('Recall: ', recall_score(y_true, y_pred, average='micro'))
    print('F1:', f1_score(y_true, y_pred, average='micro'))


def evaluate_document_relevance(experiment_type, data_dir, run_file, out_file, multi_turn):
    eval_file_path, topic_file_path = get_eval_topic_file_paths(data_dir, experiment_type)
    eval_dict = load_eval_dict(eval_file_path, topic_file_path)
    run_dict = load_run_dict_doc_relevance(run_file)
    facet_to_topic_dict = load_facet_to_topic_dict(topic_file_path)
    performance_dict = {}
    for metric in eval_dict:
        performance_dict[metric] = {}
        get_document_relevance_for_metric(eval_dict, facet_to_topic_dict, metric, multi_turn, performance_dict,
                                          run_dict)
    if out_file != '':
        with open(out_file, 'w') as fo:
            json.dump(performance_dict, fo)
    # compute the mean performance per metric and print
    for metric in performance_dict:
        print('{}: {}'.format(metric, mean(performance_dict[metric][k] for k in performance_dict[metric])))


def get_document_relevance_for_metric(eval_dict, facet_to_topic_dict, metric, multi_turn, performance_dict, run_dict):
    for facet_id in eval_dict[metric]:
        try:
            selected_q = get_selected_question(facet_id, facet_to_topic_dict, multi_turn, run_dict)
            try:
                performance_dict[metric][facet_id] = eval_dict[metric][facet_id][selected_q]['with_answer']
            except KeyError:  # if question is not among candidate question, we consider it equal to minimum performance.
                performance_dict[metric][facet_id] = eval_dict[metric][facet_id]['MIN']['with_answer']
        except KeyError:  # if there is no prediction provided for a facet, we consider performance 0.
            performance_dict[metric][facet_id] = 0.


def get_selected_question(facet_id, facet_to_topic_dict, multi_turn, run_dict):
    if multi_turn:
        selected_q = run_dict[facet_id]
    else:
        selected_q = run_dict[facet_to_topic_dict[facet_id]]
    selected_q = 'MIN' if selected_q == 'MAX' else selected_q # to avoid submitting MAX results.
    return selected_q


def get_eval_topic_file_paths(data_dir, experiment_type):
    if experiment_type in ['train', 'dev']:
        eval_file_path = path.join(data_dir, 'single_turn_train_eval.pkl')
        topic_file_path = path.join(data_dir, '{}.tsv'.format(experiment_type))
    else:
        eval_file_path = path.join(data_dir, 'single_turn_test_eval.pkl')
        topic_file_path = path.join(data_dir, 'test_with_labels.tsv')
        # raise FileNotFoundError  # TODO: remove when test eval released.
    return eval_file_path, topic_file_path


def load_facet_to_topic_dict(topic_file_path):
    topic_df = pd.read_csv(topic_file_path, sep='\t')
    facet_to_topic_dict = topic_df.set_index('facet_id')['topic_id'].to_dict()
    return facet_to_topic_dict


def load_eval_dict(eval_file_path, topic_file_path):
    topic_df = pd.read_csv(topic_file_path, sep='\t')
    facet_array = topic_df['facet_id'].values
    with open(eval_file_path, 'rb') as fi:
        eval_dict = pickle.load(fi)
    # we keep only the instances in the topic file.
    new_eval_dict = {}
    for metric in eval_dict:
        new_eval_dict[metric] = {}
        for fid in eval_dict[metric]:
            if fid in facet_array:
                new_eval_dict[metric][fid] = eval_dict[metric][fid]
    return new_eval_dict


def load_run_dict_doc_relevance(run_file):
    run_df = pd.read_csv(run_file, sep=' ', header=None)
    run_df = run_df.sort_values(by=4).drop_duplicates(subset=[0], keep='last')  # we only keep the top ranked question.
    run_dict = run_df.set_index(0)[2].to_dict()  # we convert the run dataframe to dict.
    return run_dict


def evaluate_question_relevance(experiment_type, data_dir, run_file, out_file):
    eval_file_path, topic_file_path = get_eval_topic_file_paths(data_dir, experiment_type)
    topic_df = pd.read_csv(topic_file_path, sep='\t')
    topic_question_set_dict = topic_df.groupby('topic_id')['question_id'].agg(set).to_dict()
    run_df = pd.read_csv(run_file, sep=' ', header=None)
    run_df = run_df.sort_values(by=[0, 4], ascending=False).drop_duplicates(subset=[0, 4], keep='first')
    run_question_set_list = run_df.groupby(0)[2].agg(list).to_dict()

    topk_list = [5, 10, 20, 30]
    recall_score_dict = {}
    for topk in topk_list:
        metric_name = 'Recall{}'.format(topk)
        recall_score_dict[metric_name] = {}
        for tid in topic_question_set_dict:
            try:
                rec = len(set(run_question_set_list[tid][:topk]) & topic_question_set_dict[tid]) / len(
                    topic_question_set_dict[tid])
            except KeyError:  # in case a topic is not included in the predictions
                rec = 0.
            recall_score_dict[metric_name][tid] = rec

    if out_file != '':
        with open(out_file, 'w') as fo:
            json.dump(recall_score_dict, fo)
            
    for metric in recall_score_dict:
        print('{}: {}'.format(metric, mean(recall_score_dict[metric][k] for k in recall_score_dict[metric])))


def main():
    parser = argparse.ArgumentParser(description='Input arguments for ClariQ eval tool.',
                                     add_help=True)
    parser.add_argument('--eval_task',
                        dest='eval_task',
                        type=str,
                        help='Defines the evaluation task. Possible values: '
                             'clarification_need|document_relevance|question_relevance',
                        required=True)
    parser.add_argument('--experiment_type',
                        dest='experiment_type',
                        type=str,
                        help='Defines the experiment type. The run file will be evaluated on the data that you '
                             'specify here. Possible values: train|dev|test. Default value: dev',
                        default='dev')
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        type=str,
                        help='Path to the data directory.',
                        default='../data/',
                        )
    parser.add_argument('--run_file',
                        dest='run_file',
                        type=str,
                        help='Path to the run file.',
                        required=True)
    parser.add_argument('--out_file',
                        dest='out_file',
                        type=str,
                        help='Path to the evaluation output json file.',
                        required=False,
                        default='')
    parser.add_argument('--multi_turn',
                        dest='multi_turn', action='store_true',
                        help='Determines if the results are on multi-turn conversations. Conversation is assumed to '
                             'be single-turn if not specified.',
                        required=False)
    parser.set_defaults(multi_turn=False)

    input_args = parser.parse_args()

    if input_args.eval_task == 'clarification_need':
        evaluate_clarification_need(input_args.experiment_type, input_args.data_dir, input_args.run_file,
                                    input_args.out_file)
    elif input_args.eval_task == 'document_relevance':
        evaluate_document_relevance(input_args.experiment_type, input_args.data_dir, input_args.run_file,
                                    input_args.out_file, input_args.multi_turn)
    elif input_args.eval_task == 'question_relevance':
        evaluate_question_relevance(input_args.experiment_type, input_args.data_dir, input_args.run_file,
                                    input_args.out_file)


if __name__ == '__main__':
    main()
