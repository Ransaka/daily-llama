from typing import Any, List

import numpy as np
from index.indexer import DailyLlamaIndexer
from vector.vectorize import DailyLlamaVectorizer


class DailyLLAMA:
    def __init__(self, source_data_path, source_column, content_column, chat_model, embedding_model="intfloat/e5-small-v2") -> None:
        self.vectorizer = DailyLlamaVectorizer(
            file_path=source_data_path, column_to_embed=source_column, content_column=content_column, model_id=embedding_model)
        self.embeddings = self.vectorizer.retrave_embeddings(
            output_type='numpy')
        self.indexer = DailyLlamaIndexer(self.embeddings)

    def __call__(self, query, k=4) -> Any:
        """
        Call the function.

        Args:
            query (Any): The query to be processed.
            k (int, optional): The number of top documents to retrieve. Defaults to 4.

        Returns:
            Any: The result of the function.
        """
        vector = self.vectorizer.encode_single(text=query)
        topk = self.indexer.topk(vector=vector['embeddings'], k=k)
        docs = self.vectorizer.content[topk]
        docs = np.array(docs).reshape(-1)
        return self.generate_prompt(docs=docs, query=query)

    @staticmethod
    def generate_prompt(docs: np.array, query: str) -> str:
        """
        Generate a prompt for a Q&A bot.

        Args:
            docs (np.array): An array of strings representing the information available to the bot.
            query (str): The user's query.

        Returns:
            str: The generated prompt for the Q&A bot.
        """
        intro = "You are a Q&A bot, and you have the following information. " \
                "Answer user queries based on the below information. " \
                "Start your answer with 'Based on past newspaper contents...'"
        information = "\n".join(docs)
        return f"{intro}\n- {information}\n{query}"
    
    
