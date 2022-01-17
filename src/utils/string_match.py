from pathlib import Path
from hydra.utils import get_original_cwd
from typing import Dict
from hashlib import md5
import copy
from tqdm import tqdm
from collections import Counter
import dartsclone


class ComplexKeywordTyper:
    def __init__(self, term2cat: Dict[str, str]) -> None:
        buffer_dir = Path(get_original_cwd()).joinpath(
            "data",
            "buffer",
            md5(
                (
                    "ComplexKeywordProcessor from "
                    + str(sorted(term2cat.keys()))
                    + str(sorted(set(term2cat.values())))
                ).encode()
            ).hexdigest(),
        )
        term2cat = copy.copy(term2cat)  # pythonでは参照渡しが行われるため
        self.cat_labels = sorted(set(term2cat.values()))
        if not buffer_dir.exists():
            case_sensitive_terms = set()
            # 小文字化した際に2回以上出現するものを見つける。これらをcase sensitiveとする
            duplicated_lower_terms = set()
            for term, num in tqdm(
                Counter([term.lower() for term in term2cat]).most_common()
            ):
                if num >= 2:
                    duplicated_lower_terms.add(term)
                else:
                    break
            for term, cat in tqdm(term2cat.items()):
                if term.upper() == term:
                    # 略語(大文字に変化させても形状が変化しないもの)をcase_sensitive_term2catとする
                    #  & これらを　term2catから取り除く
                    case_sensitive_terms.add(term)
                elif term.lower() in duplicated_lower_terms:
                    case_sensitive_terms.add(term)
            all_terms = set(term2cat.keys())
            # 残りのものをcase insensitiveとする
            case_insensitive_base_terms = all_terms - case_sensitive_terms
            # for term in tqdm(case_sensitive_terms):
            #     del term2cat[term]

            # self.reversed_case_sensitive_keyword_processor = KeywordProcessor(
            #     case_sensitive=True
            # )
            # self.reversed_case_insensitive_keyword_processor = KeywordProcessor(
            #     case_sensitive=False
            # )

            self.reversed_case_sensitive_darts = dartsclone.DoubleArray()
            self.reversed_case_insensitive_darts = dartsclone.DoubleArray()
            case_sensitive_terms = sorted(
                [t[::-1].encode() for t in case_sensitive_terms]
            )
            case_sensitive_cats = [
                self.cat_labels.index(term2cat[term.decode()[::-1]])
                for term in tqdm(case_sensitive_terms)
            ]

            # for term, cat in tqdm(case_sensitive_terms.items()):
            #     terms.append(term.encode())
            #     cats.append(self.cat_labels.index(self.term2cat[term]))
            # self.reversed_case_sensitive_darts.add_keyword(term[::-1], cat)
            self.reversed_case_sensitive_darts.build(
                case_sensitive_terms, values=case_sensitive_cats
            )
            case_insensitive_terms = []
            case_insensitive_terms_append = case_insensitive_terms.append
            case_insensitive_cats = []
            case_insensitive_cats_append = case_insensitive_cats.append
            for term in tqdm(case_insensitive_base_terms):
                # case insensitiveのものに関しては複数形を追加する
                cat = term2cat[term]
                case_insensitive_terms_append(term.lower()[::-1])
                case_insensitive_cats_append(cat)
                # pluralized_term = pluralize(term)
                # case_insensitive_terms_append(pluralized_term.lower()[::-1])
                # case_insensitive_cats_append(cat)

                # self.reversed_case_insensitive_keyword_processor.add_keyword(
                #     term[::-1], cat
                # )
                # self.reversed_case_insensitive_keyword_processor.add_keyword(
                #     pluralized_term[::-1], cat
                # )
            case_insensitive_term_and_cat = [
                (t.encode(), self.cat_labels.index(c))
                for t, c in zip(case_insensitive_terms, case_insensitive_cats)
            ]
            case_insensitive_term_and_cat = sorted(
                case_insensitive_term_and_cat, key=lambda x: x[0]
            )
            case_insensitive_terms, case_insensitive_cats = zip(
                *case_insensitive_term_and_cat
            )
            self.reversed_case_insensitive_darts.build(
                case_insensitive_terms, values=case_insensitive_cats
            )

            buffer_dir.mkdir()
            self.reversed_case_sensitive_darts.save(
                str(buffer_dir.joinpath("reversed_case_sensitive_darts"))
            )
            self.reversed_case_insensitive_darts.save(
                str(buffer_dir.joinpath("reversed_case_insensitive_darts"))
            )
            # buffer_dir.joinpath("reversed_case_sensitive_darts"),
            # with open(
            #     buffer_dir.joinpath("reversed_case_sensitive_keyword_processor.pkl"),
            #     "wb",
            # ) as f:
            #     pickle.dump(self.reversed_case_sensitive_darts, f)
            # with open(
            #     buffer_dir.joinpath("reversed_case_insensitive_keyword_processor.pkl"),
            #     "wb",
            # ) as f:
            #     pickle.dump(self.reversed_case_insensitive_darts, f)
        self.reversed_case_sensitive_darts = dartsclone.DoubleArray()
        self.reversed_case_sensitive_darts.open(
            str(buffer_dir.joinpath("reversed_case_sensitive_darts"))
        )
        self.reversed_case_insensitive_darts = dartsclone.DoubleArray()
        self.reversed_case_insensitive_darts.open(
            str(buffer_dir.joinpath("reversed_case_insensitive_darts"))
        )
        # with open(
        #     buffer_dir.joinpath("reversed_case_sensitive_keyword_processor.pkl"), "rb"
        # ) as f:
        #     self.reversed_case_sensitive_darts = pickle.load(f)
        # with open(
        #     buffer_dir.joinpath("reversed_case_insensitive_keyword_processor.pkl"), "rb"
        # ) as f:
        #     self.reversed_case_insensitive_darts = pickle.load(f)

    def get_confirmed_common_suffixes(self, chunk: str):
        """Return confirmed common suffixes

        Args:
            chunk (str): chunk for string match

        Returns: confirmed_common_suffixes
            [type]: List of tuple of (type, start)
        """
        reversed_chunk = "".join(reversed(chunk))
        common_suffixes = self.reversed_case_sensitive_darts.common_prefix_search(
            reversed_chunk.lower().encode("utf-8")
        )
        common_suffixes += self.reversed_case_insensitive_darts.common_prefix_search(
            reversed_chunk.lower().encode("utf-8")
        )
        # 単語の途中に出てこないか確認 (e.g. ale: Food -> male or female: Food)
        confirmed_common_suffixes = []
        for cat, start in common_suffixes:
            if start < len(chunk) and reversed_chunk[start] != " ":
                pass
            else:
                confirmed_common_suffixes.append(
                    (self.cat_labels[cat], len(chunk) - start)
                )
        return confirmed_common_suffixes

    def type_chunk(self, chunk: str, **kwargs) -> str:
        common_suffixes = self.get_confirmed_common_suffixes(chunk)
        if common_suffixes:
            cats, starts = zip(*common_suffixes)
            return cats[starts.index(min(starts))]
        else:
            return "nc-O"

    def detect_and_labels(self, snt: str):
        labeled_chunks = []
        for end in range(len(snt)):
            if snt[end] == " ":
                substring = snt[:end]
                reversed_substring = substring[::-1]
                common_suffixes = (
                    self.reversed_case_sensitive_darts.common_prefix_search(
                        reversed_substring.lower().encode("utf-8")
                    )
                )
                common_suffixes += (
                    self.reversed_case_insensitive_darts.common_prefix_search(
                        reversed_substring.lower().encode("utf-8")
                    )
                )

                confirmed_common_suffixes = []
                for cat, start in common_suffixes:
                    if start < len(substring) and reversed_substring[start] != " ":
                        pass
                    if end < start:
                        pass
                    else:
                        confirmed_common_suffixes.append((cat, start))

                common_suffixes = confirmed_common_suffixes
                if common_suffixes:
                    cats, starts = zip(*common_suffixes)
                    start = max(starts)
                    cat = self.cat_labels[cats[starts.index(start)]]
                    assert end - start >= 0
                    labeled_chunks.append((cat, end - start, end))
        return labeled_chunks
