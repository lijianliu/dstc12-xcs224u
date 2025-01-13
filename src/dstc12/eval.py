# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import tqdm
import numpy as np
from coclust.evaluation.external import accuracy
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from rouge_score.scoring import Score
from langchain_core.runnables import RunnableParallel

from dstc12.utils import DotAllRegexParser
from dstc12.prompts import STYLEGUIDE_SECTION_1_PROMPT, STYLEGUIDE_SECTION_2_PROMPT, STYLEGUIDE_SECTION_3_PROMPT


def acc(references=None, predictions=None):
    assert references and predictions and len(references) == len(predictions)
    return accuracy(references, predictions)


def nmi(references=None, predictions=None):
    assert references and predictions and len(references) == len(predictions)
    return normalized_mutual_info_score(references, predictions)


def rouge(references=None, predictions=None, metrics=['rouge1', 'rouge2', 'rouge3'], average=False):
    assert len(references) == len(predictions)
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    if average:
        scores_aggregated = {metric: {'precision': 0, 'recall': 0, 'fmeasure': 0} for metric in scores}
        for score in scores:
            for metric in score:
                scores_aggregated[metric]['precision'] += score[metric].precision / len(scores)
                scores_aggregated[metric]['recall'] += score[metric].recall / len(scores)
                scores_aggregated[metric]['fmeasure'] += score[metric].fmeasure / len(scores)
        result = {
            metric_name: Score(metric['precision'], metric['recall'], metric['fmeasure'])
            for metric_name, metric in result.items()
        }
    else:
        result = scores
    return result


def rouge_with_multiple_references(references_list, predictions):
    scores = [rouge(refs_i, predictions, aggregate=True) for refs_i in references_list]
    scores_averaged = {
        metric_name: {
            'precision': 0,
            'recall': 0,
            'fmeasure': 0
        } for metric_name in scores[0]
    }

    for score_i in scores:
        for metric_name, score in score_i.items():
            scores_averaged[metric_name]['precision'] += score.precision / len(scores)
            scores_averaged[metric_name]['recall'] += score.recall / len(scores)
            scores_averaged[metric_name]['fmeasure'] += score.fmeasure / len(scores)
    return scores_averaged


def cosine_similarity_with_multiple_references(references_list, predictions):
    scores = [cosine_similarity(refs_i, predictions) for refs_i in references_list]
    scores_averaged = sum(scores) / len(scores)
    return scores_averaged


def process_llm_judge_output(output):
    scores = []
    for  section in ['section_1', 'section_2', 'section_3']:
        assert section in output and 'score' in output[section]
        scores.append(int(output[section]['score']['value'] == 'Good'))
    return np.mean(scores)


def llm_score(predictions, llm):
    chain = (
        RunnableParallel(
            section_1=STYLEGUIDE_SECTION_1_PROMPT
                | llm
                | RunnableParallel(
                    score=DotAllRegexParser(regex=r'<score>\s*(.*?)\s*</score>', output_keys=['value']),
                    explanation=DotAllRegexParser(regex=r'<explanation>\s*(.*?)\s*</explanation>', output_keys=['value']),
            ),
            section_2=STYLEGUIDE_SECTION_2_PROMPT
                | llm
                | RunnableParallel(
                    score=DotAllRegexParser(regex=r'<score>\s*(.*?)\s*</score>', output_keys=['value']),
                    explanation=DotAllRegexParser(regex=r'<explanation>\s*(.*?)\s*</explanation>', output_keys=['value']),
            ),
            section_3=STYLEGUIDE_SECTION_3_PROMPT
                | llm
                | RunnableParallel(
                    score=DotAllRegexParser(regex=r'<score>\s*(.*?)\s*</score>', output_keys=['value']),
                    explanation=DotAllRegexParser(regex=r'<explanation>\s*(.*?)\s*</explanation>', output_keys=['value']),
            ),
        )
    )
    scores = []
    for prediction in tqdm.tqdm(predictions):
        judge_output = chain.invoke({'theme_label': prediction})
        scores.append(process_llm_judge_output(judge_output))
    return np.mean(scores)
