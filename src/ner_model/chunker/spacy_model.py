from typing import List
from .abstract_model import Span, Chunker
import benepar
from benepar.spacy_plugin import BeneparComponent
import spacy
from datasets import DatasetDict
import gin
from spacy.tokens.doc import Doc
from loguru import logger


@gin.configurable
class BeneparNPSpanDetector(Chunker):
    def __init__(self, span_detection_datasets: DatasetDict) -> None:
        self.span_detection_datasets = span_detection_datasets
        self.nlp = spacy.load("en_core_web_sm")
        try:
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        except LookupError:
            import nltk

            benepar.download("benepar_en3")
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))

    def predict(self, tokens: List[str]) -> List[Span]:
        # check input is a sentence
        assert len([w for w in tokens if w == "."]) <= 1

        doc = Doc(self.nlp.vocab, tokens)
        for i, w in enumerate(doc):
            if i == 0:
                doc[i].is_sent_start = True
            else:
                doc[i].is_sent_start = False
        self.nlp.pipeline[-1][1]
        ret = self.nlp.pipeline[-1][1](doc)
        sent = list(ret.sents)[0]
        acceptable_nodes = {"NP", "NML"}
        acceptable_tags = {"NN", "NNS", "NNP", "NNPS"}
        exclude_tags = {
            ":",
            "IN",
            ",",
            "WDT",
            "WP",
            "TO",
            "WP$",
            "WRB",
            "VBZ",
            "RB",
            "-LRB-",
            "-RRB-",
        }
        logger.info(sent._.parse_string)
        if acceptable_nodes & set(sent._.labels) and not any(
            "(%s" % et in sent._.parse_string for et in exclude_tags
        ):
            nps = [sent]
        elif len(sent) == 1 and sent[0].tag_ in acceptable_tags:
            nps = [sent]
        else:
            remained_subtree = [sent]
            nps = []
            while remained_subtree:
                subtree = remained_subtree.pop()
                for c in subtree._.children:
                    if c._.labels:
                        if acceptable_nodes & set(c._.labels):
                            for et in exclude_tags:
                                if "(%s" % et in c._.parse_string:
                                    remained_subtree.append(c)
                                    break
                            else:
                                nps.append(c)
                        else:
                            remained_subtree.append(c)
                    elif len(c) == 1 and c[0].tag_ in acceptable_tags:
                        nps.append(c)
        pass
        spans = []
        for np in nps:
            assert tokens[np.start : np.end] == [w.text for w in np]
            spans.append([np.start, np.end])
        return spans
