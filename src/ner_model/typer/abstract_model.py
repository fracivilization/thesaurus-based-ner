from collections import defaultdict
from datasets import DatasetDict, Dataset, Sequence, Value
from typing import Dict, List, Tuple
from datasets.info import DatasetInfo
from dataclasses import dataclass, field
import numpy as np
from omegaconf.omegaconf import MISSING
from transformers.modeling_outputs import SequenceClassifierOutput
import torch


@dataclass
class SequenceClassifierOutputPlus(SequenceClassifierOutput):
    feature_vecs: torch.Tensor = None


@dataclass
class SpanClassifierOutput:
    label: str
    logits: np.array = None


@dataclass
class SpanClassifierDataTrainingArguments:
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )


@dataclass
class TyperConfig:
    typer_name: str = MISSING


@dataclass
class TyperOutput:
    labels: List[str]
    max_probs: np.array  # prediction probability for label


class Typer:
    conf = dict()

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> TyperOutput:
        raise NotImplementedError

    def batch_predict(
        self, tokens: List[List[str]], starts: List[List[int]], ends: List[List[int]]
    ) -> List[TyperOutput]:
        assert len(tokens) == len(starts)
        assert len(starts) == len(ends)
        labels = []
        for tok, st, en in zip(tokens, starts, ends):
            labels.append(self.predict(tok, st, en))
        return labels

    def train(self):
        raise NotImplementedError


from src.ner_model.abstract_model import NERModel
from tqdm import tqdm
from scipy.special import softmax


class SpanClassifierNERWrapper(NERModel):
    def __init__(self, span_classifier: Typer, span_length: int) -> None:
        self.span_classifier = span_classifier
        self.span_length = span_length
        self.argss = ["SpanClassifierNERWrapper"] + self.span_classifier.argss

    def outputs2ner_tags(
        self,
        spans: List[Tuple[int, int]],
        outputs: List[SpanClassifierOutput],
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


from seqeval.metrics.sequence_labeling import get_entities


class SpanClassifierNERLabelRefinement(NERModel):
    def __init__(self, ner_model: NERModel, span_classifier: Typer) -> None:
        self.ner_model = ner_model
        self.span_classifier = span_classifier
        self.argss = (
            ["SpanClassifierNERWrapper"]
            + self.span_classifier.argss
            + self.ner_model.argss
        )

    def outputs2ner_tags(
        self,
        spans: List[Tuple[int, int]],
        outputs: List[SpanClassifierOutput],
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

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # tokens, starts, ends を作って、 model.batch_predict に投げる
        ner_outputs = self.ner_model.batch_predict(tokens)
        # for
        batch_tokens = []
        batch_start = []
        batch_end = []
        snt_ids = []
        for snt_id, snt in enumerate(ner_outputs):
            ents = get_entities(snt)
            if ents:
                starts, ends = zip(*[(s, e + 1) for l, s, e in ents])
                for s, e in zip(starts, ends):
                    batch_tokens.append(tokens[snt_id])
                    batch_start.append(s)
                    batch_end.append(e)
                    snt_ids.append(snt_id)

        # batch_tokens = []
        # batch_start = []
        # batch_end = []
        # snt_ids = []
        # for snt_id, tok in enumerate(tokens):
        #     spans = [
        #         (s, e)
        #         for s in range(len(tok))
        #         for e in range(s + 1, len(tok))
        #         if e - s <= self.span_length
        #     ]
        #     for s, e in spans:
        #         batch_tokens.append(tok)
        #         batch_start.append(s)
        #         batch_end.append(e)
        #         snt_ids.append(snt_id)
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
