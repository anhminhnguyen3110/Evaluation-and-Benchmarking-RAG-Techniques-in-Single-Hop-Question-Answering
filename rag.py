import json
import os
from datasets import load_dataset
from langchain_openai import AzureChatOpenAI
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModel

nltk.download("punkt")

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

query_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
query_encoder = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
context_encoder = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")


def retrieve_context(context, input_question, retriever="BM25", top_k=5):
    # Step 1: Chunk the context using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=0, length_function=len
    )
    context_chunks = splitter.split_text(context)

    # Step 2: Depending on the retriever, choose BM25 or Dragon
    if retriever == "BM25":
        # BM25 retrieval
        tokenized_chunks = [chunk.split() for chunk in context_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        query_tokens = input_question.split()
        scores = bm25.get_scores(query_tokens)

        # Get indices of the top k scores
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        top_k_chunks = [context_chunks[i] for i in top_k_indices]

        return top_k_chunks

    elif retriever == "Dragon":
        # Step 2.2: Tokenize and encode the input query
        query_input = query_tokenizer(input_question, return_tensors="pt")
        query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

        # Step 2.3: Tokenize and encode the context chunks
        ctx_input = query_tokenizer(
            context_chunks, padding=True, truncation=True, return_tensors="pt"
        )
        ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

        # Step 2.4: Compute similarity scores using dot product
        scores = torch.matmul(query_emb, ctx_emb.T).squeeze()

        # Step 2.5: Get indices of the top k scores
        top_k_indices = torch.topk(scores, k=top_k).indices
        top_k_chunks = [context_chunks[i] for i in top_k_indices]

        return top_k_chunks

    else:
        raise ValueError(f"Unknown retriever type: {retriever}")


def prompt_builder(prompt_format, context, input):
    return prompt_format.format(context=context, input=input)


qasper_ds = load_dataset("THUDM/LongBench", "qasper", split="test")
narrativeqa_ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test")
multifieldqa_en_ds = load_dataset("THUDM/LongBench", "multifieldqa_en", split="test")

multifieldqa_en_df = multifieldqa_en_ds.to_pandas()
qasper_df = qasper_ds.to_pandas()
narrativeqa_df = narrativeqa_ds.to_pandas()


def invoke_azure_model(
    df: pd.DataFrame, output_file, dataset_name, ds, retriever="BM25"
):
    if not os.path.exists(output_file):
        with open("dataset2prompt.json", "r") as f:
            dataset2prompt = json.load(f)
        prompt_format = dataset2prompt[dataset_name]

        for index, row in df.iterrows():
            context = retrieve_context(
                row["context"], row["input"], retriever=retriever
            )
            input_question = row["input"]
            prompt = prompt_builder(prompt_format, context, input_question)

            df.at[index, "answers"] = ds[index]["answers"]
            df.at[index, "prediction"] = azure_model.predict(prompt)
            print(f"Process {dataset_name} completed: {index + 1} / {len(df)} with context length: {len(context)} and {[ len(x) for x in context ]}")

        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    retriever = "Dragon"

    invoke_azure_model(
        multifieldqa_en_df,
        f"multifieldqa_en_rag_{retriever}_llm_predictions.csv",
        "multifieldqa_en",
        multifieldqa_en_ds,
        retriever=retriever,
    )

    # invoke_azure_model(
    #     qasper_df,
    #     f"qasper_rag_{retriever}_llm_predictions.csv",
    #     "qasper",
    #     qasper_ds,
    #     retriever=retriever,
    # )

    # invoke_azure_model(
    #     narrativeqa_df,
    #     f"narrativeqa_rag_{retriever}_llm_predictions.csv",
    #     "narrativeqa",
    #     narrativeqa_ds,
    #     retriever=retriever,
    # )
