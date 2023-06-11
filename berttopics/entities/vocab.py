import collections
from tqdm import tqdm
from typing import List
from sklearn.feature_extraction.text import CountVectorizer

class Vocab:
    def __init__(self, docs: List[str]):
        self.docs = docs

    def get_vocab(self):
        vocab = collections.Counter()
        tokenizer = CountVectorizer().build_tokenizer()

        for doc in tqdm(self.docs):
            vocab.update(tokenizer(doc))

        vocab: List[str] = [word for word, frequency in vocab.items() if frequency >= 15]; len(vocab)

        vocab = sorted(vocab)
        
        return vocab
