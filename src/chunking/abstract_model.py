from datasets import DatasetDict
from typing import Dict, List, Tuple
from tqdm import tqdm

Span = Tuple[int, int]


class Chunker:
    def predict(self, tokens: List[str]) -> Span:
        raise NotImplementedError

    def batch_predict(
        self,
        tokens: List[List[str]],
    ) -> List[Span]:
        return [self.predict(token) for token in tqdm(tokens)]


class RandomChunker(Chunker):
    def predict(self, tokens: List[str]) -> Span:
        return super().predict(tokens)
