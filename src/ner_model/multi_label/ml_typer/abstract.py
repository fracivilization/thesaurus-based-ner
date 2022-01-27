from typing import List, Tuple
from dataclasses import dataclass
from omegaconf.omegaconf import MISSING
import numpy as np
from src.ner_model.abstract_model import NERModel
from tqdm import tqdm
from scipy.special import softmax


@dataclass
class MultiLabelTyperConfig:
    multi_label_typer_name: str = MISSING
    label_names: str = "non_initialized: this variable is dinamically decided"


@dataclass
class MultiLabelTyperOutput:
    labels: List[List[str]]
    # max_probs: np.array  # prediction probability for label
    # probs: np.array  # prediction probability for labels
    logits: np.array


class MultiLabelTyper:
    conf = dict()
    label_names: List[str] = []

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> MultiLabelTyperOutput:
        raise NotImplementedError

    def batch_predict(
        self, tokens: List[List[str]], starts: List[List[int]], ends: List[List[int]]
    ) -> List[MultiLabelTyperOutput]:
        assert len(tokens) == len(starts)
        assert len(starts) == len(ends)
        labels = []
        for tok, st, en in zip(tokens, starts, ends):
            labels.append(self.predict(tok, st, en))
        return labels

    def train(self):
        raise NotImplementedError


class MultiLabelTyperNERWrapper(NERModel):
    def __init__(self, span_classifier: MultiLabelTyper, span_length: int) -> None:
        self.span_classifier = span_classifier
        self.span_length = span_length
        self.argss = ["SpanClassifierNERWrapper"] + self.span_classifier.argss

    def outputs2ner_tags(
        self,
        spans: List[Tuple[int, int]],
        outputs: List[MultiLabelTyperOutput],
        snt_len: int,
    ):
        ner_tags = ["O"] * snt_len
        span2output = {span: output for span, output in zip(spans, outputs)}
        span2max_prob = {
            span: softmax(span2output[span].logits).max() for span in span2output
        }
        for (s, e), max_prob in sorted(
            span2max_prob.items(), key=lambda x: x[1], reverse=True
        ):
            predicted_label = span2output[(s, e)].label
            if predicted_label != "O" and all(tag == "O" for tag in ner_tags[s:e]):
                for i in range(s, e):
                    if i == s:
                        ner_tags[i] = "B-%s" % predicted_label
                    else:
                        ner_tags[i] = "I-%s" % predicted_label
        return ner_tags

    def predict(self, tokens: List[str]) -> List[str]:
        # spanのそれぞれに対して、

        spans = [
            (s, e)
            for s in range(len(tokens))
            for e in range(s + 1, len(tokens))
            if e - s <= self.span_length
        ]

        batch_tokens = []
        batch_start = []
        batch_end = []
        for s, e in spans:
            batch_tokens.append(tokens)
            batch_start.append(s)
            batch_end.append(e)
        outputs = self.span_classifier.batch_predict(
            batch_tokens, batch_start, batch_end
        )
        return self.outputs2ner_tags(spans, outputs, len(tokens))

        # 予測をさせる
        # その後その中で最も大きいlogitを持つ要素からうめていく

    def small_batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        batch_tokens = []
        batch_start = []
        batch_end = []
        snt_ids = []
        for snt_id, tok in enumerate(tokens):
            spans = [
                (s, e)
                for s in range(len(tok))
                for e in range(s + 1, len(tok))
                if e - s <= self.span_length
            ]
            for s, e in spans:
                batch_tokens.append(tok)
                batch_start.append(s)
                batch_end.append(e)
                snt_ids.append(snt_id)
        outputs = self.span_classifier.batch_predict(
            batch_tokens, batch_start, batch_end
        )
        from itertools import groupby

        grouped_output = [
            (snt_id, list(o))
            for snt_id, o in groupby(
                zip(snt_ids, batch_start, batch_end, outputs), key=lambda x: x[0]
            )
        ]
        grouped_output = sorted(grouped_output, key=lambda x: x[0])
        # assert [sid for sid, _ in grouped_output] == list(range(len(grouped_output)))
        sntid2output = dict(grouped_output)
        ner_tags = []
        for snt_id, tok in enumerate(tokens):
            if snt_id in sntid2output:
                _, starts, ends, outputs = zip(*sntid2output[snt_id])
                spans = list(zip(starts, ends))
                ner_tags.append(self.outputs2ner_tags(spans, outputs, len(tok)))
            else:
                ner_tags.append(["O"] * len(tok))
        # それぞれの文ごとに分けて出力する
        return ner_tags

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # tokens, starts, ends を作って、 model.batch_predict に投げる
        ner_tags = []
        from more_itertools import chunked
        import torch

        tokenss = list(chunked(tokens, 100))
        torch.cuda.empty_cache()
        # tokens
        for toks in tqdm(tokenss):
            ner_tags += self.small_batch_predict(toks)
        return ner_tags
