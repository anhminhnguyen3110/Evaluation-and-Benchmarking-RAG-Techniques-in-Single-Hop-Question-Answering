import ast
import math
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerSimilarity,
)


from datasets import Dataset

azure_configs = {
    "base_url": "",
    "model_deployment": "gpt-4o",
    "model_name": "gpt-4o",
    "embedding_deployment": "text-embedding-ada-002",
    "embedding_name": "text-embedding-ada-002",
    "openai_api_key": "",
    "openai_api_version": "",
}

azure_model = AzureChatOpenAI(
    openai_api_version=azure_configs["openai_api_version"],
    azure_deployment=azure_configs["model_deployment"],
    azure_endpoint=azure_configs["base_url"],
    api_key=azure_configs["openai_api_key"],
)

azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2022-12-01",
    azure_deployment=azure_configs["embedding_deployment"],
    azure_endpoint=azure_configs["base_url"],
    api_key=azure_configs["openai_api_key"],
)

answer_correctness = AnswerCorrectness(
    weights=[1, 0],
    
)

answer_relevancy = AnswerRelevancy(
    strictness=5,
)

faithfulness = Faithfulness()

context_precision = ContextPrecision()

context_recall = ContextRecall()
answer_similarity = AnswerSimilarity()


def replace_nan_with_zero(value):
    # Check if the value is NaN, replace with 0.0 if true
    return (
        0.0
        if value is None or (isinstance(value, float) and math.isnan(value))
        else value
    )


def ragas_evaluation_with_retrieval(
    question: str,
    prediction: str,
    ground_truth: str,
    context: list[str],
) -> dict:
    if prediction[-1] != ".":
        prediction = prediction + "."
    if ground_truth[-1] != ".":
        ground_truth = ground_truth + "."

    prediction = prediction.lower()
    ground_truth = ground_truth.lower()
        
    dump_data = {
        "question": [question],
        "ground_truth": [ground_truth],
        "answer": [prediction],
        "contexts": [context],
    }
    dump_dataset = Dataset.from_dict(dump_data)

    result = evaluate(
        dump_dataset,
        metrics=[
            answer_correctness,
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
            answer_similarity,
        ],
        llm=azure_model,
        embeddings=azure_embeddings,
    )
    
    print(
        f"""
        Question: {question} \n
        Prediction: {prediction} \n
        Ground Truth: {ground_truth} \n
        Answer Correctness: {result.get("answer_correctness", 0.0)} \n\n
        """
    )

    return {
        "answer_correctness": replace_nan_with_zero(
            result.get("answer_correctness", 0.0)
        ),
        "answer_relevancy": replace_nan_with_zero(result.get("answer_relevancy", 0.0)),
        "faithfulness": replace_nan_with_zero(result.get("faithfulness", 0.0)),
        "context_precision": replace_nan_with_zero(
            result.get("context_precision", 0.0)
        ),
        "context_recall": replace_nan_with_zero(result.get("context_recall", 0.0)),
        "answer_similarity": replace_nan_with_zero(
            result.get("answer_similarity", 0.0)
        ),
    }


def ragas_evaluation_without_retrieval(
    question: str,
    prediction: str,
    ground_truth: str,
) -> dict:
    if prediction[-1] != ".":
        prediction = prediction + "."
    if ground_truth[-1] != ".":
        ground_truth = ground_truth + "."
        
    prediction = prediction.lower()
    ground_truth = ground_truth.lower()
    
    dump_data = {
        "question": [question],
        "ground_truth": [ground_truth],
        "answer": [prediction],
    }

    dump_dataset = Dataset.from_dict(dump_data)

    result = evaluate(
        dump_dataset,
        metrics=[
            answer_correctness,
            answer_similarity,
        ],
        llm=azure_model,
        embeddings=azure_embeddings,
    )

    return {
        "answer_correctness": result.get("answer_correctness", 0.0),
        "answer_similarity": result.get("answer_similarity", 0.0),
    }


if __name__ == "__main__":
    test = pd.read_csv("./multifieldqa_en_long_context_llm_predictions.csv")

    first_row = test.iloc[0]

    question = first_row["input"]
    prediction = first_row["prediction"]
    ground_truth = ast.literal_eval(first_row["answers"])[0]
    context = [first_row["context"]]

    data = {
        "question": 'How big is Augmented LibriSpeech dataset?',
        "ground_truth": 'Unanswerable.',
        "answer": 'unanswerable',
        "contexts": ["Just have the associate sign the back and the cheque can be deposited in your account."]
    }

    # result = ragas_evaluation_without_retrieval(question, prediction, ground_truth)
    result = ragas_evaluation_with_retrieval(
        data["question"], data["answer"], data["ground_truth"], data["contexts"]
    )

    # print(result)
    
    
    # print(
    #     f"""
    #         Question: {question} \n
    #         Prediction: {prediction} \n
    #         Ground Truth: {ground_truth} \n
    #     """
    # )
    data_samples = {
        'question': ['What is the name of the most active fan club?'],
        'answer': ['south west ultras.'],
        'ground_truth': ['south west ultras.'],
    }
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset,metrics=[answer_correctness], llm=azure_model, embeddings=azure_embeddings)
    print(score)
