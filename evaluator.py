import json
import os
import numpy as np
import pandas as pd
import re
import string
from typing import Counter
from rouge import Rouge
from evaluate import load
import ast

from ragas_evaluator import ragas_evaluation_without_retrieval


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return (0, 0, 0)
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


rouge_metric = load("rouge", trust_remote_code=True)
squad_v2 = load("squad_v2", trust_remote_code=True)

metrics_list = [
    ("rouge", rouge_metric),
    ("squad_v2", squad_v2),
]


def normalize_text(text):
    """Lowercases, trims, and removes extra spaces to normalize text."""
    return " ".join(text.lower().strip().split())


def calculate_metrics_list(predictions, ground_truths):
    results = []

    for metric_name, metric in metrics_list:
        if metric_name == "rouge":
            # ROUGE metric (handles a list of ground truths for each prediction)
            rouge_results = metric.compute(
                predictions=predictions, references=ground_truths
            )
            results.append(rouge_results)

        elif metric_name == "squad_v2":
            # SQuAD v2 metric (simplified for this context, normalize both predictions and ground truths)
            normalized_predictions = [
                normalize_text(prediction) for prediction in predictions
            ]
            normalized_references = [
                [normalize_text(gt) for gt in ground_truth_list]
                for ground_truth_list in ground_truths
            ]

            squad_results = metric.compute(
                predictions=[
                    {
                        "id": str(i),
                        "prediction_text": prediction,
                        "no_answer_probability": 0.0,
                    }
                    for i, prediction in enumerate(normalized_predictions)
                ],
                references=[
                    {
                        "id": str(i),
                        "answers": {
                            "text": gt_list,
                            "answer_start": [0] * len(gt_list),
                        },
                    }
                    for i, gt_list in enumerate(normalized_references)
                ],
            )
            results.append(
                {
                    "squad_v2": {
                        "exact_match": squad_results["exact"],
                        "f1": squad_results["f1"],
                    }
                }
            )

    return results

def scorer_e(predictions, answers, lengths):
    scores = {
        "0-4k": {"qa_f1_score": [], "rouge_score": [], "qa_precision": [], "qa_recall": []},
        "4-8k": {"qa_f1_score": [], "rouge_score": [], "qa_precision": [], "qa_recall": []},
        "8k+": {"qa_f1_score": [], "rouge_score": [], "qa_precision": [], "qa_recall": []},
    }

    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        qa_f1_score_result = 0.0
        qa_precision_result = 0.0
        qa_recall_result = 0.0
        rouge_score_result = 0.0

        for ground_truth in ground_truths:
            f1, precision, recall = qa_f1_score(prediction, ground_truth)
            
            # Only update precision and recall when a higher F1 score is found
            if f1 > qa_f1_score_result:
                qa_f1_score_result = f1
                qa_precision_result = precision
                qa_recall_result = recall

            rouge_score_result = max(rouge_score_result, rouge_score(prediction, ground_truth))

        if length < 4000:
            scores["0-4k"]["qa_f1_score"].append(qa_f1_score_result)
            scores["0-4k"]["qa_precision"].append(qa_precision_result)
            scores["0-4k"]["qa_recall"].append(qa_recall_result)
            scores["0-4k"]["rouge_score"].append(rouge_score_result)
        elif length < 8000:
            scores["4-8k"]["qa_f1_score"].append(qa_f1_score_result)
            scores["4-8k"]["qa_precision"].append(qa_precision_result)
            scores["4-8k"]["qa_recall"].append(qa_recall_result)
            scores["4-8k"]["rouge_score"].append(rouge_score_result)
        else:
            scores["8k+"]["qa_f1_score"].append(qa_f1_score_result)
            scores["8k+"]["qa_precision"].append(qa_precision_result)
            scores["8k+"]["qa_recall"].append(qa_recall_result)
            scores["8k+"]["rouge_score"].append(rouge_score_result)

    # Calculate mean scores for each metric and each length category
    for key in scores.keys():
        for metric in scores[key].keys():
            if scores[key][metric]:  # Avoid empty list errors
                scores[key][metric] = round(100 * np.mean(scores[key][metric]), 2)
            else:
                scores[key][metric] = 0.0  # Default if no scores

    return scores


def scorer(predictions, answers, df):
    qa_f1_score_total = 0.0
    qa_precision_total = 0.0
    qa_recall_total = 0.0
    rouge_score_total = 0.0

    for idx, (prediction, ground_truths) in enumerate(zip(predictions, answers)):
        qa_f1_score_result = 0.0
        qa_precision_result = 0.0
        qa_recall_result = 0.0
        rouge_score_result = 0.0

        for ground_truth in ground_truths:
            f1, precision, recall = qa_f1_score(prediction, ground_truth)

            # Only update precision and recall when a higher F1 score is found
            if f1 > qa_f1_score_result:
                qa_f1_score_result = f1
                qa_precision_result = precision
                qa_recall_result = recall

            rouge_score_result = max(rouge_score_result, rouge_score(prediction, ground_truth))

        qa_f1_score_total += qa_f1_score_result
        qa_precision_total += qa_precision_result
        qa_recall_total += qa_recall_result
        rouge_score_total += rouge_score_result
        
        df.loc[idx, "qa_f1_score"] = qa_f1_score_result
        # df.loc[idx, "qa_precision"] = qa_precision_result
        # df.loc[idx, "qa_recall"] = qa_recall_result
        

    num_predictions = len(predictions)
    if num_predictions == 0:
        return {
            "qa_f1_score": 0.0,
            "qa_precision": 0.0,
            "qa_recall": 0.0,
            "rouge_score": 0.0,
        }

    return {
        "qa_f1_score": round(100 * qa_f1_score_total / num_predictions, 2),
        "qa_precision": round(100 * qa_precision_total / num_predictions, 2),
        "qa_recall": round(100 * qa_recall_total / num_predictions, 2),
        "rouge_score": round(100 * rouge_score_total / num_predictions, 2),
    }

evaluatorResults = {}


def ai_evaluator(predictions, answers, lengths, questions, df):
    final_scores = {
        "ragas": {}, 
        # "deep-eval": {}
    }
    scores = {
        "0-4k": {
            "ragas": {},
            # "deep-eval": {},
        },
        "4-8k": {
            "ragas": {},
            # "deep-eval": {},
        },
        "8k+": {
            "ragas": {},
            # "deep-eval": {},
        },
    }

    results = []

    for idx, (prediction, ground_truths, length, question) in enumerate(
        zip(predictions, answers, lengths, questions)
    ):
        result = {}
        best_answer_correctness = 0.0
        old_best_answer_correctness = 0.0
        for ground_truth in ground_truths:
            tmp_result = ragas_evaluation_without_retrieval(
                question, prediction, ground_truth
            )

            best_answer_correctness = max(
                best_answer_correctness,
                tmp_result.get("answer_correctness", 0.0),
            )
            
            if best_answer_correctness > old_best_answer_correctness or result == {}:
                result = tmp_result
                old_best_answer_correctness = best_answer_correctness

        results.append(result)

        if length <= 4000:
            length_bin = "0-4k"
        elif length <= 8000:
            length_bin = "4-8k"
        else:
            length_bin = "8k+"

        for metric, value in result.items():
            if final_scores["ragas"].get(metric) is None:
                final_scores["ragas"][metric] = []
            final_scores["ragas"][metric].append(value)

            if scores[length_bin]["ragas"].get(metric) is None:
                scores[length_bin]["ragas"][metric] = []
            scores[length_bin]["ragas"][metric].append(value)

        df.loc[idx, "answer_correctness"] = result.get("answer_correctness", 0.0)

    # Process 'scores' dictionary
    for length_bin in scores.keys():
        for category in scores[length_bin].keys():  # 'ragas' and 'deep-eval'
            for metric in scores[length_bin][category].keys():
                if scores[length_bin][category][metric]:  # Avoid empty list errors
                    scores[length_bin][category][metric] = round(
                        100 * np.mean(scores[length_bin][category][metric]), 2
                    )
                else:
                    scores[length_bin][category][metric] = 0.0  # Default if no scores

    # Process 'final_scores' dictionary
    for category in final_scores.keys():  # 'ragas' and 'deep-eval'
        for metric in final_scores[category].keys():
            if final_scores[category][metric]:
                final_scores[category][metric] = round(
                    100 * np.mean(final_scores[category][metric]), 2
                )
            else:
                final_scores[category][metric] = 0.0

    return scores, final_scores

def evaluate(file_path, dataset_name, is_ai_evaluator=False):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            predictions_list = []
            answers_list = []
            length_list = []
            questions_list = []
            contexts_list = []

            for _, row in df.iterrows():
                predictions_list.append(row["prediction"])
                answers_list.append(ast.literal_eval(row["answers"]))
                length_list.append(row["length"])
                questions_list.append(row["input"])
                contexts_list.append(row["context"])

            rouge_squad = calculate_metrics_list(predictions_list, answers_list)
            scorer_result = scorer(predictions_list, answers_list, df)
            scorer_e_result = scorer_e(predictions_list, answers_list, length_list)

            ai_evaluator_result = None
            if is_ai_evaluator:
                ai_evaluator_result = (
                    ai_evaluator(
                        predictions_list,
                        answers_list,
                        length_list,
                        questions_list,
                        df,
                    ),
                )

            df.to_csv(f"results_long_context_{dataset_name}.csv", index=False)
            evaluatorResults[dataset_name] = {
                "squad_v2": rouge_squad[1]['squad_v2'],
                "scorer_result": scorer_result,
                "scorer_e_result": scorer_e_result,
                "ai_evaluator": ai_evaluator_result,
            }

        except Exception as e:
            print(e)
    else:
        print(f"File {file_path} does not exist")


if __name__ == "__main__":
    is_ai_evaluator = True
    evaluate(
        "./multifieldqa_en_long_context_llm_predictions.csv",
        "multifieldqa_en",
        is_ai_evaluator,
    )
    evaluate(
        "./narrativeqa_long_context_llm_predictions.csv", "narrativeqa", is_ai_evaluator
    )
    evaluate("./qasper_long_context_llm_predictions.csv", "qasper", is_ai_evaluator)

    print(
        json.dumps(
            evaluatorResults,
            indent=4,
        )
    )

    with open("evaluator_long_context_results.json", "w") as json_file:
        json.dump(evaluatorResults, json_file, indent=2)
