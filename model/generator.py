from typing import Any
from index.indexer import DailyLlamaIndexer
from vector.vectorize import DailyLlamaVectorizer


class DailyLLAMA:
    def __init__(self, source_data_path, source_column, model_id="intfloat/e5-small-v2") -> None:
        self.vectorizer = DailyLlamaVectorizer(
            file_path=source_data_path, column_to_embed=source_column, model_id=model_id)
        self.embeddings, self.emb_column_values = self.vectorizer.retrave_embeddings(output_type='numpy')
        self.indexer = DailyLlamaIndexer(self.embeddings)

    def __call__(self, query, k=4) -> Any:
        vector = self.vectorizer.encode_single(text=query)
        topk = self.indexer.topk(vector=vector['embeddings'], k=k)
        return self.emb_column_values[topk]
