import json
import os
from datasets import load_dataset
from langchain_openai import AzureChatOpenAI
import pandas as pd


# Azure Chat OpenAI Model (GPT-4o)
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
    temperature=0.0,
)


def prompt_builder(prompt_format, context, input):
    return prompt_format.format(context=context, input=input)


# Dataset preparation
qasper_ds = load_dataset("THUDM/LongBench", "qasper", split="test")
narrativeqa_ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test")
multifieldqa_en_ds = load_dataset("THUDM/LongBench", "multifieldqa_en", split="test")

multifieldqa_en_df = multifieldqa_en_ds.to_pandas()
qasper_df = qasper_ds.to_pandas()
narrativeqa_df = narrativeqa_ds.to_pandas()


def invoke_azure_model(df: pd.DataFrame, output_file, dataset_name, ds):
    if not os.path.exists(output_file):
        with open("dataset2prompt.json", "r") as f:
            dataset2prompt = json.load(f)
        prompt_format = dataset2prompt[dataset_name]

        for index, row in df.iterrows():
            context = row["context"]
            input_question = row["input"]
            prompt = prompt_builder(prompt_format, context, input_question)

            df.at[index, "answers"] = ds[index]["answers"]
            df.at[index, "prediction"] = azure_model.predict(prompt)
            print(f"Process {dataset_name} completed: {index + 1} / {len(df)}")

        df.to_csv(output_file, index=False)


invoke_azure_model(
    multifieldqa_en_df,
    "multifieldqa_en_long_context_llm_predictions.csv",
    "multifieldqa_en",
    multifieldqa_en_ds,
)

invoke_azure_model(
    qasper_df, "qasper_long_context_llm_predictions.csv", "qasper", qasper_ds
)

invoke_azure_model(
    narrativeqa_df,
    "narrativeqa_long_context_llm_predictions.csv",
    "narrativeqa",
    narrativeqa_ds,
)
