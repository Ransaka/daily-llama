import faiss

class DailyLlamaIndexer:
  def __init__(self, embed_vec):
    self.embeddings_vec = embed_vec
    self.build_index()

  def build_index(self):
    """
    Build the index for the embeddings.

    This function initializes the index for the embeddings. It calculates the dimension (self.d)
    of the embeddings vector and creates an IndexFlatL2 object (self.index) for the given dimension.
    It then adds the embeddings vector (self.embeddings_vec) to the index.

    Parameters:
    - None

    Return:
    - None
    """
    self.d = self.embeddings_vec.shape[1]
    self.index = faiss.IndexFlatL2(self.d)
    self.index.add(self.embeddings_vec)

  def topk(self, vector, k = 4):
    """
        A function that takes in a vector and an optional parameter k and returns the indices of the k nearest neighbors in the index.

        Parameters:
            vector: A numpy array representing the input vector.
            k (optional): An integer representing the number of nearest neighbors to retrieve. Defaults to 4 if not specified.

        Returns:
            I: A numpy array containing the indices of the k nearest neighbors in the index.
    """
    # vec = self.retreaver.encode(text)['embeddings'].detach().cpu().numpy()
    _, I = self.index.search(vector, k)
    return I