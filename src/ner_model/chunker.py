from spacy.tokens.doc import Doc
from typing import List, Tuple
from datasets import Dataset
import datasets
import colt

Chunk = Tuple[int, int]


class Chunker:
    def __call__(self, tokens: List[str]) -> List[Chunk]:
        """snt is a space-based concatenate of words"""
        raise NotImplementedError

    def batch_predict(self, tokens: List[List[str]]) -> List[List[Chunk]]:
        raise NotImplementedError


from spacy.tokens.doc import Doc
from typing import List
from dataclasses import dataclass
import click
import spacy


@colt.register("PosChunker")
class PosChunker(Chunker):
    def __init__(self):
        self.args = ["PosChunker", "PTB"]
        pos2abbrev = {
            "CC": "Ccj",  # Coordinating conjunction
            "CD": "Cnm",  # Cardinal number
            "DT": "Det",  # Determiner
            "EX": "Ext",  # Existential there
            "FW": "Frw",  # Foreign word
            "IN": "Psc",  # Preposition or subordinating conjunction
            "JJ": "Ajn",  # Adjective
            "JJR": "Ajc",  # Adjective, comparative
            "JJS": "Ajs",  # Adjective, superlative
            "LS": "Lim",  # List item marker
            "MD": "Mdl",  # Modal
            "NN": "Non",  # Noun, singular or mass
            "NNS": "Nns",  # Noun, plural
            "NNP": "Nnp",  # Proper noun, singular
            "NNPS": "Nps",  # Proper noun, plural
            "PDT": "Pdt",  # Predeterminer
            "POS": "Pos",  # Possessive ending
            "PRP": "Prp",  # Personal pronoun
            "PRP$": "Prs",  # Possessive pronoun
            "RB": "Adv",  # Adverb
            "RBR": "Adc",  # Adverb, comparative
            "RBS": "Ads",  # Adverb, superlative
            "RP": "Pcl",  # Particle
            "SYM": "Sym",  # Symbol
            "TO": "Too",  # to
            "UH": "Itj",  # Interjection
            "VB": "Vbb",  # Verb, base form
            "VBD": "Vbd",  # Verb, past tense
            "VBG": "Vbg",  # Verb, gerund or present participle
            "VBN": "Vbn",  # Verb, past participle
            "VBP": "Vbp",  # Verb, non-3rd person singular present
            "VBZ": "Vbz",  # Verb, 3rd person singular present
            "WDT": "Wdt",  # Wh-determiner
            "WP": "Wpn",  # Wh-pronoun
            "WP$": "Wps",  # Possessive wh-pronoun
            "WRB": "Wrb",  # Wh-adverb
            ",": "Com",
            ".": "Prd",
            ":": "Cln",
            "-LRB-": "Lrb",
            "-RRB-": "Rrb",
            "''": "Dqt",
            "HYPH": "Hph",
            "``": "Gac",
            "_SP": "Spc",
            "XX": "Xxx",
            "NFP": "Nfp",
            "ADD": "Add",
            "$": "Dol",
            "AFX": "Afx",
        }
        self.pos2abbrev = pos2abbrev
        left = [
            "DT",
            "IN",
            "CC",
            ",",
            "RB",
            "RBR",
            "RBS",
            "RP",
            "VB",
            "VBD",
            "VBZ",
            "VBP",
            ".",
            "-LRB-",
            "WDT",
            "TO",
            ":",
            "NNS",
            "-RRB-",
            "PRP$",
        ]
        left = [pos2abbrev[pos] for pos in left]
        inside = ["JJ", "NN", "NNP", "CD", "VBG", "VBN", "''"]
        inside = [pos2abbrev[pos] for pos in inside]
        right = [
            ",",
            ".",
            "IN",
            "VB",
            "VBD",
            "VBZ",
            "VBN",
            "VBP",
            "CC",
            "MD",
            "DT",
            "RB",
            "RBR",
            "RBS",
            "JJ",
            "JJR",
            "JJS",
            "RP",
            "-LRB-",
            "-RRB-",
            "WDT",
            "TO",
            ":",
        ]
        right = [pos2abbrev[pos] for pos in right]

        import re

        pat_sn = re.compile(
            "(?<=%s)(?P<Chunk>(%s)*(Non|Vbg)(Cnm)?)(?=%s)"
            % (
                "|".join(left + ["STT"]),
                "|".join(inside),
                "|".join(right + ["END"]),
            )
        )
        pat_pl = re.compile(
            "(?<=%s)(?P<Chunk>(%s)*(Nns)(Cnm)?)(?=%s)"
            % (
                "|".join(left + ["STT"]),
                "|".join(inside),
                "|".join(right + ["Non", "END"]),
            )
        )
        self.pat_sn = pat_sn
        self.pat_pl = pat_pl
        # self.nlp = spacy.load("en_core_sci_sm")
        # self.tagger = self.nlp.pipeline[1][1]

    def __call__(self, tokens: List[str], poss: List[str]) -> List[Chunk]:
        # doc = self.tagger(spacy.tokens.Doc(self.nlp.vocab, tokens))
        # assert len(doc) == len(tokens)
        pos_seq = "".join(["STT"] + [self.pos2abbrev[pos] for pos in poss] + ["END"])
        predicted_span = [m.span("Chunk") for m in self.pat_sn.finditer(pos_seq)]
        predicted_span += [m.span("Chunk") for m in self.pat_pl.finditer(pos_seq)]
        chunks = [(s // 3 - 1, e // 3 - 1) for s, e in predicted_span]
        return chunks

    def batch_predict(
        self, tokens: List[List[str]], poss: List[List[str]]
    ) -> List[List[Chunk]]:
        return [self.__call__(tok, pos) for tok, pos in zip(tokens, poss)]
