from typing import List
from .abstract_model import NERModel
from src.dataset.term2cat.term2cat import Term2Cat
import colt
from seqeval.metrics.sequence_labeling import get_entities
from typing import Dict


@colt.register("RemoveTermAsPostProcessWrapper")
class RemoveTermAsPostProcessWrapper(NERModel):
    def __init__(self, ner_model: NERModel, term2cat: Term2Cat) -> None:
        super().__init__()
        self.ner_model = ner_model
        self.term2cat = term2cat.term2cat
        self.lowered_term2cat = {
            term.lower(): cat for term, cat in self.term2cat.items()
        }

    def predict(self, tokens: List[str]) -> List[str]:
        """predict class

        Args:
            sentence (str): input sentence

        Returns:
            List[str]: BIO tags
        """
        raise NotImplementedError

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """predict class

        Args:
            list of sentence (List[List[str]]): input sentences

        Returns:
            List[List[str]]: BIO tags
        """
        batch_output = self.ner_model.batch_predict(tokens)
        for snt, tags in zip(tokens, batch_output):
            for l, s, e in get_entities(tags):
                # mention = " ".join(snt[s : e + 1])
                for i in range(s, e + 1):
                    end_of_mention = " ".join(snt[i : e + 1]).lower()
                    if end_of_mention in self.lowered_term2cat:
                        if self.lowered_term2cat[end_of_mention].startswith(
                            "fake_cat_"
                        ):
                            fake_label = self.lowered_term2cat[end_of_mention]
                            for i in range(s, e + 1):
                                if i == s:
                                    tags[i] = "B-%s" % fake_label
                                else:
                                    tags[i] = "I-%s" % fake_label
                        else:
                            break
        return batch_output


class KBBasedRemoveWrapper(NERModel):
    def __init__(self, ner_model: NERModel, term2cat: Dict):
        self.ner_model = ner_model
        self.term2cat = term2cat
        self.lowered_term2cat = {term.lower(): cat for term, cat in term2cat.items()}
        self.lowered_term2orig_term = {term.lower(): term for term in term2cat}
        # self.term2cat = {term.lower(): cat for term, cat in term2cat.items()}
        # snts = []
        # for key, split in ner_datasets.items():
        #     if key in {"supervised_validation", "test"}:
        #         for snt in split:
        #             snts.append(" ".join(snt["tokens"]))
        # term2cat = screen_term2cat_by_snts(term2cat, snts)
        # self.keyword_processor = ComplexKeywordProcessor(term2cat)

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        batch_output = self.ner_model.batch_predict(tokens)
        for snt, tags in zip(tokens, batch_output):
            for l, s, e in get_entities(tags):
                mention = " ".join(snt[s : e + 1])
                for i in range(s, e + 1):
                    end_of_mention = " ".join(snt[i : e + 1]).lower()
                    if end_of_mention in self.lowered_term2cat:
                        if self.lowered_term2cat[end_of_mention].startswith(
                            "fake_cat_"
                        ):
                            for i in range(s, e + 1):
                                tags[i] = "O"
                        else:
                            break
        return batch_output
