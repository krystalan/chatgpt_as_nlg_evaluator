
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
from tabulate import tabulate

def sample_level_correlation_summeval(human_metric):
    print(f'Human metric: {human_metric}')

    assert human_metric in ['coherence', 'relevance', 'consistency', 'fluency']

    with open('data/summeval.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    auto_metrics=['rouge1_f', 'rouge2_f', 'rougel_f', 'bert_score_f', 'mover_score', 'prism_src_hypo', 'bart_score_src_hypo', 'bart_score_cnn_src_hypo', 'bart_score_para_src_hypo', 'chatgpt_%s'%human_metric]

    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []

    for metric in auto_metrics:
        correlations = []

        for doc_id in data:
            target_scores = []
            prediction_scores = []

            sys_summs = data[doc_id]['sys_summs']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['scores'][metric])
                target_scores.append(sys_summs[sys_name]['scores'][human_metric])

            if len(set(prediction_scores)) == 1 or len(set(target_scores)) == 1:
                continue

            correlations.append([
                spearmanr(target_scores, prediction_scores)[0],
                pearsonr(target_scores, prediction_scores)[0],
                kendalltau(target_scores, prediction_scores)[0],
            ])
  
        
        corr_mat = np.array(correlations)
        spearman,  pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])

    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))

def dataset_level_correlation_summeval(human_metric):
    print(f'Human metric: {human_metric}')

    assert human_metric in ['coherence', 'relevance', 'consistency', 'fluency']

    with open('data/summeval.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    auto_metrics=['rouge1_f', 'rouge2_f', 'rougel_f', 'bert_score_f', 'mover_score', 'prism_src_hypo', 'bart_score_src_hypo', 'bart_score_cnn_src_hypo', 'bart_score_para_src_hypo', 'chatgpt_%s'%human_metric]

    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []

    for metric in auto_metrics:
        correlations = []

        target_scores = []
        prediction_scores = []
        for doc_id in data:

            sys_summs = data[doc_id]['sys_summs']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['scores'][metric])
                target_scores.append(sys_summs[sys_name]['scores'][human_metric])

        correlations.append([
            spearmanr(target_scores, prediction_scores)[0],
            pearsonr(target_scores, prediction_scores)[0],
            kendalltau(target_scores, prediction_scores)[0],
        ])

        
        corr_mat = np.array(correlations)
        spearman,  pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])

    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))


def sample_level_correlation_openmeva():
    """ Evaluate summaries. Conduct summary-level correlations w.r.t each document """


    auto_metrics = ['rouge1-r', 'rouge2-r', 'rougel-r', 'bertscore-r', 'bartscore', 'bartscore_cnn', 'bartscore_cnn_pa', 'fwppl', 'chatgpt']

    with open('data/openmeva.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']

    metric_with_corr = []
    for metric in auto_metrics:
        correlations = []

        for item in data:

            target_scores = []
            prediction_scores = []

            sys_summs = item['gen']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['score'][metric])
                target_scores.append(sys_summs[sys_name]['score']['human'])

            if len(set(prediction_scores)) == 1 or len(set(target_scores)) == 1:
                continue
            
            correlations.append([
                spearmanr(target_scores, prediction_scores)[0],
                pearsonr(target_scores, prediction_scores)[0],
                kendalltau(target_scores, prediction_scores)[0]
            ])

        corr_mat = np.array(correlations)
        spearman, pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])

    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))

def dataset_level_correlation_openmeva():
    """ Evaluate summaries. Conduct summary-level correlations w.r.t each document """


    auto_metrics = ['rouge1-r', 'rouge2-r', 'rougel-r', 'bertscore-r', 'bartscore', 'bartscore_cnn', 'bartscore_cnn_pa', 'fwppl', 'chatgpt']

    with open('data/openmeva.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']

    metric_with_corr = []
    for metric in auto_metrics:
        correlations = []

        target_scores = []
        prediction_scores = []

        for item in data:

            sys_summs = item['gen']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['score'][metric])
                target_scores.append(sys_summs[sys_name]['score']['human'])

            if len(set(prediction_scores)) == 1 or len(set(target_scores)) == 1:
                continue

        correlations.append([
            spearmanr(target_scores, prediction_scores)[0],
            pearsonr(target_scores, prediction_scores)[0],
            kendalltau(target_scores, prediction_scores)[0]
        ])

        corr_mat = np.array(correlations)
        spearman, pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])

    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))



def dataset_level_correlation_bagel(human_metric): # ['rouge1_r', 'rouge2_r', 'rougel_r', 'bert_score_r', 'mover_score']
    """ Evaluate summaries. Conduct summary-level correlations w.r.t each document """

    print(f'Human metric: {human_metric}')
    assert human_metric in ['informativeness', 'naturalness', 'quality']

    with open('data/bagel.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    auto_metrics=['rouge1_p', 'rouge2_p', 'rougel_p', 'bert_score_p', 'mover_score', 'prism_ref_hypo', 'bart_score_ref_hypo', 'bart_score_cnn_ref_hypo', 'bart_score_para_ref_hypo', 'chatgpt_%s'%human_metric]


    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []

    for metric in auto_metrics:
        correlations = []

        target_scores = []
        prediction_scores = []
        for doc_id in data:

            prediction_scores.append(data[doc_id]['scores'][metric])
            target_scores.append(data[doc_id]['scores'][human_metric])
                
        correlations.append([
            spearmanr(target_scores, prediction_scores)[0],
            pearsonr(target_scores, prediction_scores)[0],
            kendalltau(target_scores, prediction_scores)[0],
        ])


        
        corr_mat = np.array(correlations)
        spearman, pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])

    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))