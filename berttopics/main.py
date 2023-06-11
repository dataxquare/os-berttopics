from .dataset import docs
from .entities.vocab import Vocab
from .entities.bert import SentenceTransformerModel

def main():
    print("Running main.py")
    model = SentenceTransformerModel(docs)
    print("Model created")
    model.fit_model()
    print("Model fitted")
    model.save_model()
    print("Model saved")

    print(model.topic_model.get_topic_info())

def test():
    print("Hola")
