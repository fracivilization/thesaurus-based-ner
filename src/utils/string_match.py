from pathlib import Path
from hydra.utils import get_original_cwd
from typing import Dict, List
from hashlib import md5
import copy
from tqdm import tqdm
from collections import Counter
import dartsclone
from src.utils.utils import WeightedSQliteDict, SQliteJsonDict


class ComplexKeywordTyper:
    def __init__(self, term2value: Dict, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive
        # self.cat_labels = sorted(set(term2value.values()))
        buffer_dir = Path(get_original_cwd()).joinpath(
            "data",
            "buffer",
            self.load_hash_value_for_term2value(term2value, case_sensitive),
        )
        term2value = copy.copy(term2value)  # pythonでは参照渡しが行われるため
        self.term2value = term2value
        if not buffer_dir.exists():
            buffer_dir.mkdir(parents=True, exist_ok=True)
            all_terms = term2value.keys()
            (
                case_sensitive_terms,
                case_insensitive_terms,
            ) = self.load_case_sensitive_and_insensitive_terms(
                all_terms, case_sensitive
            )

            self.reversed_case_sensitive_darts = (
                self.build_reversed_case_sensitive_darts(case_sensitive_terms)
            )
            self.reversed_case_sensitive_darts.save(
                str(buffer_dir.joinpath("reversed_case_sensitive_darts"))
            )
            if not case_sensitive:
                self.case_insentive2case_sensitive = SQliteJsonDict(
                    str(buffer_dir.joinpath("case_insentive2case_sensitive.db")),
                    commit_when_set_item=False,
                )
                for term in case_insensitive_terms:
                    self.case_insentive2case_sensitive[term.lower()] = term
                self.case_insentive2case_sensitive.commit()
                self.reversed_case_insensitive_darts = (
                    self.build_reversed_case_insensitive_darts(case_insensitive_terms)
                )
                self.reversed_case_insensitive_darts.save(
                    str(buffer_dir.joinpath("reversed_case_insensitive_darts"))
                )
        else:
            print("load from ComplexKeywordTyper buffer file: ", buffer_dir)

        self.reversed_case_sensitive_darts = dartsclone.DoubleArray()
        self.reversed_case_sensitive_darts.open(
            str(buffer_dir.joinpath("reversed_case_sensitive_darts"))
        )
        if not case_sensitive:
            self.case_insentive2case_sensitive = SQliteJsonDict(
                str(buffer_dir.joinpath("case_insentive2case_sensitive.db")),
                commit_when_set_item=False,
            )
            self.reversed_case_insensitive_darts = dartsclone.DoubleArray()
            self.reversed_case_insensitive_darts.open(
                str(buffer_dir.joinpath("reversed_case_insensitive_darts"))
            )

    def load_hash_value_for_term2value(self, term2value: Dict, case_sensitive: bool):
        if isinstance(term2value, dict):
            return md5(
                (
                    "ComplexKeywordProcessor from "
                    + str(sorted(term2value.keys()))
                    + str(sorted(set(term2value.values())))
                    + str(case_sensitive)
                ).encode()
            ).hexdigest()
        elif isinstance(term2value, WeightedSQliteDict):
            return md5(
                (
                    "ComplexKeywordProcessor from "
                    + term2value.db_file_path
                    + str(case_sensitive)
                ).encode()
            ).hexdigest()
        else:
            raise NotImplementedError

    def load_case_sensitive_and_insensitive_terms(self, all_terms, case_sensitive):
        all_terms_set = set(all_terms)
        if case_sensitive:
            case_sensitive_terms = all_terms_set
            case_insensitive_terms = []
        else:
            # 小文字化した際に2回以上出現するものを見つける。これらをcase sensitiveとする
            duplicated_lower_terms = self.find_duplicated_lower_terms(all_terms)
            case_sensitive_terms = self.find_case_sensitive_terms(
                all_terms, duplicated_lower_terms
            )
            # 残りのものをcase insensitiveとする
            case_insensitive_terms = all_terms_set - case_sensitive_terms
        return case_sensitive_terms, case_insensitive_terms

    def build_reversed_case_insensitive_darts(self, case_insensitive_terms):
        reversed_case_insensitive_darts = dartsclone.DoubleArray()
        case_insensitive_lowered_terms = []
        case_insensitive_lowered_terms_append = case_insensitive_lowered_terms.append
        for term in tqdm(case_insensitive_terms):
            case_insensitive_lowered_terms_append(term.lower()[::-1].encode())
        case_insensitive_terms = sorted(case_insensitive_lowered_terms)
        reversed_case_insensitive_darts.build(case_insensitive_terms)
        return reversed_case_insensitive_darts

    def build_reversed_case_sensitive_darts(self, case_sensitive_terms):
        reversed_case_sensitive_darts = dartsclone.DoubleArray()
        case_sensitive_terms = sorted([t[::-1].encode() for t in case_sensitive_terms])
        reversed_case_sensitive_darts.build(case_sensitive_terms)
        return reversed_case_sensitive_darts

    def find_case_sensitive_terms(self, all_terms: List[str], duplicated_lower_terms):
        case_sensitive_terms = set()
        for term in tqdm(all_terms):
            if term.upper() == term:
                # 略語(大文字に変化させても形状が変化しないもの)をcase_sensitive_term2catとする
                #  & これらを　term2catから取り除く
                case_sensitive_terms.add(term)
            elif term.lower() in duplicated_lower_terms:
                case_sensitive_terms.add(term)
        return case_sensitive_terms

    def find_duplicated_lower_terms(self, terms: List[str]):
        duplicated_lower_terms = set()
        for term, num in tqdm(Counter([term.lower() for term in terms]).most_common()):
            if num >= 2:
                duplicated_lower_terms.add(term)
            else:
                break
        return duplicated_lower_terms

    def get_confirmed_common_suffixes(self, chunk: str):
        """Return confirmed common suffixes

        Args:
            chunk (str): chunk for string match

        Returns: confirmed_common_suffixes
            [type]: List of tuple of (type, start)
        """
        reversed_chunk = "".join(reversed(chunk))
        common_suffixes = self.reversed_case_sensitive_darts.common_prefix_search(
            reversed_chunk.encode("utf-8")
        )
        if not self.case_sensitive:
            common_suffixes += (
                self.reversed_case_insensitive_darts.common_prefix_search(
                    reversed_chunk.lower().encode("utf-8")
                )
            )
        # 単語の途中に出てこないか確認 (e.g. ale: Food -> male or female: Food)
        confirmed_common_suffixes = []
        # NOTE: マルチバイト文字に対応するためにencodeした時の文字列長を見る
        encoded_chunk = chunk.encode()
        reversed_encoded_chunk = reversed_chunk.encode()
        encoded_chunk_size = len(chunk.encode())
        for _, inv_start in common_suffixes:
            if (
                inv_start < encoded_chunk_size
                and reversed_encoded_chunk[inv_start] != " "
            ):
                pass
            else:
                start = encoded_chunk_size - inv_start
                term = encoded_chunk[start:].decode()
                if term in self.term2value:
                    value = self.term2value[term]
                elif not self.case_sensitive:
                    value = self.term2value[
                        self.case_insentive2case_sensitive[term.lower()]
                    ]
                else:
                    raise NotImplementedError
                confirmed_common_suffixes.append((value, start))
        return confirmed_common_suffixes

    def exact_type_chunk(self, chunk: str) -> str:
        type_of_chunk = "nc-O"
        # double array trie で完全一致するものを探す
        reversed_chunk = "".join(reversed(chunk))
        value_index, _ = self.reversed_case_sensitive_darts.exact_match_search(
            reversed_chunk.encode("utf-8")
        )
        if value_index != -1:
            type_of_chunk = self.cat_labels[value_index]
        if not self.case_sensitive:
            value_index, _ = self.reversed_case_insensitive_darts.exact_match_search(
                reversed_chunk.lower().encode("utf-8")
            )
            # NOTE: ２つのdartsの両方でマッチすることはないはず
            if value_index != -1:
                type_of_chunk = self.cat_labels[value_index]
        return type_of_chunk

    def type_chunk(self, chunk: str, exact=False) -> str:
        if exact:
            return self.exact_type_chunk(chunk)
        else:
            common_suffixes = self.get_confirmed_common_suffixes(chunk)
            if common_suffixes:
                values, starts = zip(*common_suffixes)
                return values[starts.index(min(starts))]
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
                        reversed_substring.encode("utf-8")
                    )
                )
                if not self.case_sensitive:
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
