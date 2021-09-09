from typing import List
from .abstract_model import Span, Chunker
import gin
from flair.data import Sentence
from flair.models import SequenceTagger


@gin.configurable
class FlairNPDetector(Chunker):
    def __init__(self) -> None:
        self.argss = ["Flair NP Detector"]
        self.tagger = SequenceTagger.load("flair/chunk-english-fast")

    def predict(self, tokens: List[str]) -> List[Span]:
        # make example sentence
        sentence = Sentence(tokens, use_tokenizer=False)
        assert len(tokens) == len(sentence)
        # predict NER tags
        self.tagger.predict(sentence)
        nps = []
        for entity in sentence.get_spans("np"):
            if entity.tag == "NP":
                ids = [t.idx for t in entity.tokens]
                start = min(ids) - 1
                end = max(ids)
                nps.append((start, end))
        return nps
