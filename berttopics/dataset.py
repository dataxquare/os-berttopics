from sklearn.datasets import fetch_20newsgroups
from typing import List

docs: List[str] = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

docs = [doc for doc in docs if len(doc) > 100]

docs = docs[:1000]