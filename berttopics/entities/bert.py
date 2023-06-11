from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
# from cuml.manifold import UMAP
# from cuml.cluster import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from .vocab import Vocab

class SentenceTransformerModel:
    def __init__(self, docs: List[str], model_name_or_path: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.docs = docs
        self.model = SentenceTransformer(model_name_or_path)
        self.embeddings = self.model.encode(docs, show_progress_bar=False)
        self.topic_model = None

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings

    def fit_model(self):
        vocab = Vocab(self.docs).get_vocab()

        ## Si usamos UMAP ni HDBSCAN, podemos mejorar el rendimiento reduciendo la dimensionalidad de los embbedings

        # umap_model = UMAP(n_components=5, n_neighbors=50,
        #                 random_state=42, metric="cosine", verbose=True)
        # hdbscan_model = HDBSCAN(min_samples=20, gen_min_span_tree=True,
        #                         prediction_data=False, min_cluster_size=20, verbose=True)
        
        vectorizer_model = CountVectorizer(
            vocabulary=vocab, stop_words="english")

        # reduced_embeddings = umap_model.fit_transform(self.embeddings)

        self.topic_model = BERTopic(
            embedding_model=self.model,
            # umap_model=umap_model,
            # hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True
        ).fit(self.docs, embeddings=self.embeddings)

    def save_model(self, output_dir: str = 'output'):
        self.topic_model.save(path=output_dir)
