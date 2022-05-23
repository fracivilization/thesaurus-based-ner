from collections import defaultdict
from dataclasses import dataclass
from datasets import DatasetDict
from datasets import load_dataset
from dataclasses import dataclass
from typing import List
from more_itertools import powerset


@dataclass
class DatasetConfig:
    name_or_path: str = "conll2003"


def load_dataset(config: DatasetConfig) -> DatasetDict:
    if config.name_or_path in {"conll2003"}:
        dataset = load_dataset(config.name_or_path)
    pass


STchild2parent = {
    "Entity": "ROOT",
    "Physical Object": "Entity",
    "Organism": "Physical Object",
    "Plant": "Organism",
    "Fungus": "Organism",
    "Virus": "Organism",
    "Bacterium": "Organism",
    "Archaeon": "Organism",
    "Eukaryote": "Organism",
    "Animal": "Eukaryote",
    "Vertebrate": "Animal",
    "Amphibian": "Vertebrate",
    "Bird": "Vertebrate",
    "Fish": "Vertebrate",
    "Reptile": "Vertebrate",
    "Mammal": "Vertebrate",
    "Human": "Mammal",
    "Anatomical Structure": "Physical Object",
    "Embryonic Structure": "Anatomical Structure",
    "Anatomical Abnormality": "Embryonic Structure",
    "Congenital Abnormality": "Anatomical Abnormality",
    "Acquired Abnormality": "Anatomical Abnormality",
    "Fully Formed Anatomical Structure": "Embryonic Structure",
    "Body Part, Organ, or Organ Component": "Fully Formed Anatomical Structure",
    "Tissue": "Fully Formed Anatomical Structure",
    "Cell": "Fully Formed Anatomical Structure",
    "Cell Component": "Fully Formed Anatomical Structure",
    "Gene or Genome": "Fully Formed Anatomical Structure",
    "Manufactured Object": "Physical Object",
    "Medical Device": "Manufactured Object",
    "Drug Delivery Device": "Medical Device",
    "Research Device": "Manufactured Object",
    "Clinical Drug": "Manufactured Object",
    "Substance": "Physical Object",
    "Chemical": "Substance",
    "Chemical Viewed Functionally": "Chemical",
    "Pharmacologic Substance": "Chemical Viewed Functionally",
    "Antibiotic": "Pharmacologic Substance",
    "Biomedical or Dental Material": "Chemical Viewed Functionally",
    "Biologically Active Substance": "Chemical Viewed Functionally",
    "Hormone": "Biologically Active Substance",
    "Enzyme": "Biologically Active Substance",
    "Vitamin": "Biologically Active Substance",
    "Immunologic Factor": "Biologically Active Substance",
    "Receptor": "Biologically Active Substance",
    "Indicator, Reagent, or Diagnostic Aid": "Chemical Viewed Functionally",
    "Hazardous or Poisonous Substance": "Chemical Viewed Functionally",
    "Chemical Viewed Structurally": "Chemical",
    "Organic Chemical": "Chemical Viewed Structurally",
    "Nucleic Acid, Nucleoside, or Nucleotide": "Organic Chemical",
    "Amino Acid, Peptide, or Protein": "Organic Chemical",
    "Inorganic Chemical": "Chemical Viewed Structurally",
    "Element, Ion, or Isotope": "Chemical Viewed Structurally",
    "Body Substance": "Substance",
    "Food": "Substance",
    "Conceptual Entity": "Entity",
    "Idea or Concept": "Conceptual Entity",
    "Temporal Concept": "Idea or Concept",
    "Qualitative Concept": "Idea or Concept",
    "Quantitative Concept": "Idea or Concept",
    "Functional Concept": "Idea or Concept",
    "Body System": "Functional Concept",
    "Spatial Concept": "Idea or Concept",
    "Body Space or Junction": "Spatial Concept",
    "Body Location or Region": "Spatial Concept",
    "Molecular Sequence": "Spatial Concept",
    "Nucleotide Sequence": "Molecular Sequence",
    "Amino Acid Sequence": "Molecular Sequence",
    "Carbohydrate Sequence": "Molecular Sequence",
    "Geographic Area": "Spatial Concept",
    "Finding": "Conceptual Entity",
    "Laboratory or Test Result": "Finding",
    "Sign or Symptom": "Finding",
    "Organism Attribute": "Conceptual Entity",
    "Clinical Attribute": "Organism Attribute",
    "Intellectual Product": "Conceptual Entity",
    "Classification": "Intellectual Product",
    "Regulation or Law": "Intellectual Product",
    "Language": "Conceptual Entity",
    "Occupation or Discipline": "Conceptual Entity",
    "Biomedical Occupation or Discipline": "Occupation or Discipline",
    "Organization": "Conceptual Entity",
    "Health Care Related Organization": "Organization",
    "Professional Society": "Organization",
    "Self-help or Relief Organization": "Organization",
    "Group Attribute": "Conceptual Entity",
    "Group": "Conceptual Entity",
    "Professional or Occupational Group": "Group",
    "Population Group": "Group",
    "Family Group": "Group",
    "Age Group": "Group",
    "Patient or Disabled Group": "Group",
    "Event": "ROOT",
    "Activity": "Event",
    "Behavior": "Activity",
    "Social Behavior": "Behavior",
    "Individual Behavior": "Behavior",
    "Daily or Recreational Activity": "Activity",
    "Occupational Activity": "Activity",
    "Health Care Activity": "Occupational Activity",
    "Laboratory Procedure": "Health Care Activity",
    "Diagnostic Procedure": "Health Care Activity",
    "Therapeutic or Preventive Procedure": "Health Care Activity",
    "Research Activity": "Occupational Activity",
    "Molecular Biology Research Technique": "Research Activity",
    "Governmental or Regulatory Activity": "Occupational Activity",
    "Educational Activity": "Occupational Activity",
    "Machine Activity": "Activity",
    "Phenomenon or Process": "Event",
    "Human-caused Phenomenon or Process": "Phenomenon or Process",
    "Environmental Effect of Humans": "Human-caused Phenomenon or Process",
    "Natural Phenomenon or Process": "Phenomenon or Process",
    "Biologic Function": "Natural Phenomenon or Process",
    "Physiologic Function": "Biologic Function",
    "Organism Function": "Physiologic Function",
    "Mental Process": "Organism Function",
    "Organ or Tissue Function": "Physiologic Function",
    "Cell Function": "Physiologic Function",
    "Molecular Function": "Physiologic Function",
    "Genetic Function": "Molecular Function",
    "Pathologic Function": "Biologic Function",
    "Disease or Syndrome": "Pathologic Function",
    "Mental or Behavioral Dysfunction": "Disease or Syndrome",
    "Neoplastic Process": "Disease or Syndrome",
    "Cell or Molecular Dysfunction": "Pathologic Function",
    "Experimental Model of Disease": "Pathologic Function",
    "Injury or Poisoning": "Phenomenon or Process",
}
tui2ST = {
    "T000": "ROOT",  # Additional Dummy Semantic Type
    "T116": "Amino Acid, Peptide, or Protein",
    "T020": "Acquired Abnormality",
    "T052": "Activity",
    "T100": "Age Group",
    "T087": "Amino Acid Sequence",
    "T011": "Amphibian",
    "T190": "Anatomical Abnormality",
    "T008": "Animal",
    "T017": "Anatomical Structure",
    "T195": "Antibiotic",
    "T194": "Archaeon",
    "T123": "Biologically Active Substance",
    "T007": "Bacterium",
    "T031": "Body Substance",
    "T022": "Body System",
    "T053": "Behavior",
    "T038": "Biologic Function",
    "T012": "Bird",
    "T029": "Body Location or Region",
    "T091": "Biomedical Occupation or Discipline",
    "T122": "Biomedical or Dental Material",
    "T023": "Body Part, Organ, or Organ Component",
    "T030": "Body Space or Junction",
    "T026": "Cell Component",
    "T043": "Cell Function",
    "T025": "Cell",
    "T019": "Congenital Abnormality",
    "T103": "Chemical",
    "T120": "Chemical Viewed Functionally",
    "T104": "Chemical Viewed Structurally",
    "T185": "Classification",
    "T201": "Clinical Attribute",
    "T200": "Clinical Drug",
    "T077": "Conceptual Entity",
    "T049": "Cell or Molecular Dysfunction",
    "T088": "Carbohydrate Sequence",
    "T060": "Diagnostic Procedure",
    "T056": "Daily or Recreational Activity",
    "T203": "Drug Delivery Device",
    "T047": "Disease or Syndrome",
    "T065": "Educational Activity",
    "T069": "Environmental Effect of Humans",
    "T196": "Element, Ion, or Isotope",
    "T050": "Experimental Model of Disease",
    "T018": "Embryonic Structure",
    "T071": "Entity",
    "T126": "Enzyme",
    "T204": "Eukaryote",
    "T051": "Event",
    "T099": "Family Group",
    "T021": "Fully Formed Anatomical Structure",
    "T013": "Fish",
    "T033": "Finding",
    "T004": "Fungus",
    "T168": "Food",
    "T169": "Functional Concept",
    "T045": "Genetic Function",
    "T083": "Geographic Area",
    "T028": "Gene or Genome",
    "T064": "Governmental or Regulatory Activity",
    "T102": "Group Attribute",
    "T096": "Group",
    "T068": "Human-caused Phenomenon or Process",
    "T093": "Health Care Related Organization",
    "T058": "Health Care Activity",
    "T131": "Hazardous or Poisonous Substance",
    "T125": "Hormone",
    "T016": "Human",
    "T078": "Idea or Concept",
    "T129": "Immunologic Factor",
    "T055": "Individual Behavior",
    "T197": "Inorganic Chemical",
    "T037": "Injury or Poisoning",
    "T170": "Intellectual Product",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
    "T171": "Language",
    "T059": "Laboratory Procedure",
    "T034": "Laboratory or Test Result",
    "T015": "Mammal",
    "T063": "Molecular Biology Research Technique",
    "T066": "Machine Activity",
    "T074": "Medical Device",
    "T041": "Mental Process",
    "T073": "Manufactured Object",
    "T048": "Mental or Behavioral Dysfunction",
    "T044": "Molecular Function",
    "T085": "Molecular Sequence",
    "T191": "Neoplastic Process",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T070": "Natural Phenomenon or Process",
    "T086": "Nucleotide Sequence",
    "T057": "Occupational Activity",
    "T090": "Occupation or Discipline",
    "T109": "Organic Chemical",
    "T032": "Organism Attribute",
    "T040": "Organism Function",
    "T001": "Organism",
    "T092": "Organization",
    "T042": "Organ or Tissue Function",
    "T046": "Pathologic Function",
    "T072": "Physical Object",
    "T067": "Phenomenon or Process",
    "T039": "Physiologic Function",
    "T121": "Pharmacologic Substance",
    "T002": "Plant",
    "T101": "Patient or Disabled Group",
    "T098": "Population Group",
    "T097": "Professional or Occupational Group",
    "T094": "Professional Society",
    "T080": "Qualitative Concept",
    "T081": "Quantitative Concept",
    "T192": "Receptor",
    "T014": "Reptile",
    "T062": "Research Activity",
    "T075": "Research Device",
    "T089": "Regulation or Law",
    "T167": "Substance",
    "T095": "Self-help or Relief Organization",
    "T054": "Social Behavior",
    "T184": "Sign or Symptom",
    "T082": "Spatial Concept",
    "T024": "Tissue",
    "T079": "Temporal Concept",
    "T061": "Therapeutic or Preventive Procedure",
    "T005": "Virus",
    "T127": "Vitamin",
    "T010": "Vertebrate",
}


# tui2ST = {
#     "T000": "ROOT",  # Additional Dummy Semantic Type
#     "T001": "Organism",
#     "T002": "Plant",
#     "T004": "Fungus",
#     "T005": "Virus",
#     "T007": "Bacterium",
#     "T008": "Animal",
#     "T010": "Vertebrate",
#     "T011": "Amphibian",
#     "T012": "Bird",
#     "T013": "Fish",
#     "T014": "Reptile",
#     "T015": "Mammal",
#     "T016": "Human",
#     "T017": "Anatomical Structure",
#     "T018": "Embryonic Structure",
#     "T019": "Congenital Abnormality",
#     "T020": "Acquired Abnormality",
#     "T021": "Fully Formed Anatomical Structure",
#     "T022": "Body System",
#     "T023": "Body Part, Organ, or Organ Component",
#     "T024": "Tissue",
#     "T025": "Cell",
#     "T026": "Cell Component",
#     "T028": "Gene or Genome",
#     "T029": "Body Location or Region",
#     "T030": "Body Space or Junction",
#     "T031": "Body Substance",
#     "T032": "Organism Attribute",
#     "T033": "Finding",
#     "T034": "Laboratory or Test Result",
#     "T037": "Injury or Poisoning",
#     "T038": "Biologic Function",
#     "T039": "Physiologic Function",
#     "T040": "Organism Function",
#     "T041": "Mental Process",
#     "T042": "Organ or Tissue Function",
#     "T043": "Cell Function",
#     "T044": "Molecular Function",
#     "T045": "Genetic Function",
#     "T046": "Pathologic Function",
#     "T047": "Disease or Syndrome",
#     "T048": "Mental or Behavioral Dysfunction",
#     "T049": "Cell or Molecular Dysfunction",
#     "T050": "Experimental Model of Disease",
#     "T051": "Event",
#     "T052": "Activity",
#     "T053": "Behavior",
#     "T054": "Social Behavior",
#     "T055": "Individual Behavior",
#     "T056": "Daily or Recreational Activity",
#     "T057": "Occupational Activity",
#     "T058": "Health Care Activity",
#     "T059": "Laboratory Procedure",
#     "T060": "Diagnostic Procedure",
#     "T061": "Therapeutic or Preventive Procedure",
#     "T062": "Research Activity",
#     "T063": "Molecular Biology Research Technique",
#     "T064": "Governmental or Regulatory Activity",
#     "T065": "Educational Activity",
#     "T066": "Machine Activity",
#     "T067": "Phenomenon or Process",
#     "T068": "Human-caused Phenomenon or Process",
#     "T069": "Environmental Effect of Humans",
#     "T070": "Natural Phenomenon or Process",
#     "T071": "Entity",
#     "T072": "Physical Object",
#     "T073": "Manufactured Object",
#     "T074": "Medical Device",
#     "T075": "Research Device",
#     "T077": "Conceptual Entity",
#     "T078": "Idea or Concept",
#     "T079": "Temporal Concept",
#     "T080": "Qualitative Concept",
#     "T081": "Quantitative Concept",
#     "T082": "Spatial Concept",
#     "T083": "Geographic Area",
#     "T085": "Molecular Sequence",
#     "T086": "Nucleotide Sequence",
#     "T087": "Amino Acid Sequence",
#     "T088": "Carbohydrate Sequence",
#     "T089": "Regulation or Law",
#     "T090": "Occupation or Discipline",
#     "T091": "Biomedical Occupation or Discipline",
#     "T092": "Organization",
#     "T093": "Health Care Related Organization",
#     "T094": "Professional Society",
#     "T095": "Self-help or Relief Organization",
#     "T096": "Group",
#     "T097": "Professional or Occupational Group",
#     "T098": "Population Group",
#     "T099": "Family Group",
#     "T100": "Age Group",
#     "T101": "Patient or Disabled Group",
#     "T102": "Group Attribute",
#     "T103": "Chemical",
#     "T104": "Chemical Viewed Structurally",
#     "T109": "Organic Chemical",
#     "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
#     "T116": "Amino Acid, Peptide, or Protein",
#     "T120": "Chemical Viewed Functionally",
#     "T121": "Pharmacologic Substance",
#     "T122": "Biomedical or Dental Material",
#     "T123": "Biologically Active Substance",
#     "T125": "Hormone",
#     "T126": "Enzyme",
#     "T127": "Vitamin",
#     "T129": "Immunologic Factor",
#     "T130": "Indicator, Reagent, or Diagnostic Aid",
#     "T131": "Hazardous or Poisonous Substance",
#     "T167": "Substance",
#     "T168": "Food",
#     "T169": "Functional Concept",
#     "T170": "Intellectual Product",
#     "T171": "Language",
#     "T184": "Sign or Symptom",
#     "T185": "Classification",
#     "T190": "Anatomical Abnormality",
#     "T191": "Neoplastic Process",
#     "T192": "Receptor",
#     "T194": "Archaeon",
#     "T195": "Antibiotic",
#     "T196": "Element, Ion, or Isotope",
#     "T197": "Inorganic Chemical",
#     "T200": "Clinical Drug",
#     "T201": "Clinical Attribute",
#     "T203": "Drug Delivery Device",
#     "T204": "Eukaryote",
# }


def get_parent2children():
    STparent2children = defaultdict(list)
    for child, parent in STchild2parent.items():
        STparent2children[parent].append(child)
    return dict(STparent2children)


def get_umls_negative_cats(focus_tuis: List[str]):
    concepts = set(STchild2parent.values())
    tui_recorded_concepts = set(tui2ST.values())
    ST2tui = {SemanticType: tui for tui, SemanticType in tui2ST.items()}
    STparent2children = defaultdict(list)
    for child, parent in STchild2parent.items():
        STparent2children[parent].append(child)
    assert not (concepts - tui_recorded_concepts)
    ascendants_concepts = set()
    focus_STs = {tui2ST[tui] for tui in focus_tuis}
    for focusST in focus_STs:
        candidate_concepts = {focusST}
        while candidate_concepts:
            parent_concept = candidate_concepts.pop()
            if (
                parent_concept in STchild2parent
                and parent_concept not in ascendants_concepts
            ):
                candidate_concepts.add(STchild2parent[parent_concept])
            ascendants_concepts.add(parent_concept)
    ascendants_concepts -= focus_STs
    candidate_negative_STs = set()
    for asc_con in ascendants_concepts:
        candidate_negative_STs |= set(STparent2children[asc_con])
    candidate_negative_STs -= ascendants_concepts
    negative_concepts = candidate_negative_STs - focus_STs
    negative_cats = [ST2tui[concept] for concept in negative_concepts]
    return negative_cats


def get_ascendants(tui: str = "T204"):
    pass


def get_tui2ascendants():
    ST2tui = {v: k for k, v in tui2ST.items()}
    tui2ascendants = dict()
    for tui, st in tui2ST.items():
        orig_tui = tui
        ascendants = [tui]
        while tui != "T000":
            st = STchild2parent[st]
            tui = ST2tui[st]
            ascendants.append(tui)
        tui2ascendants[orig_tui] = sorted(ascendants)
    return tui2ascendants


def valid_label_set():
    parent2children = get_parent2children()
    parent2children["UMLS"] = parent2children["ROOT"]
    parent2children["ROOT"] = ["UMLS", "nc-O"]
    ST2tui = {val: key for key, val in tui2ST.items()}
    ST2tui["nc-O"] = "nc-O"
    child2parent = {
        child: parent
        for parent, children in parent2children.items()
        for child in children
    }
    label_names = list(child2parent.keys())

    def ascendant_labels(child_label) -> List[str]:
        ascendants = [child_label]
        while child_label != "ROOT":
            parent = child2parent[child_label]
            ascendants.append(parent)
            child_label = parent
        pass
        return ascendants

    valid_sets = set()
    valid_paths = set(["T000"])
    label2valid_paths = {"T000": "T000"}
    for label in label_names:
        if label in {"UMLS", "ROOT"}:
            continue
        valid_labels = ascendant_labels(label)
        if "UMLS" in valid_labels:
            valid_labels.remove("UMLS")
            valid_labels.append("ROOT")
        if "ROOT" in valid_labels:
            valid_labels.remove("ROOT")
        valid_labels = [ST2tui[vl] for vl in valid_labels]
        valid_path = "_".join(sorted(valid_labels))
        valid_paths.add(valid_path)
        label2valid_paths[ST2tui[label]] = valid_path
        valid_sets |= set(
            ["_".join(sorted(labels)) for labels in powerset(valid_labels) if labels]
        )
    return valid_sets, valid_paths, label2valid_paths


valid_label_set, valid_paths, label2valid_paths = valid_label_set()
label2depth = {label: len(path.split("_")) for label, path in label2valid_paths.items()}


def hierarchical_valid(labels: List[str]):
    """入力されたラベル集合が階層構造として妥当であるかを判断する。

    ここで、階層構造として妥当であるとは、そのラベル集合が("O"ラベルを含んだシソーラスにおける)
    ルートノードから各ノードへのパス、あるいはその部分集合となるようなラベル集合のことである

    Args:
        labels (List[str]): 妥当性を判断するラベル集合
    """
    return "_".join(sorted(labels)) in valid_label_set


def get_complete_path(labels: List[str]):
    # 最も深いレベルに存在するラベルを取得する
    depths = [label2depth[l] for l in labels]
    # そのラベルへのパスとなるラベル集合を取得する
    return label2valid_paths[labels[depths.index(max(depths))]].split("_")


def ranked_label2hierarchical_valid_labels(ranked_labels: List[str]):
    """ランク付けされたラベルのリストから階層構造的に曖昧性のなく妥当なラベルセットを出力する

    Args:
        ranked_labels (List[str]): ラベルが出力されたランク順に並んだリスト
    """
    hierarchical_valid_labels = []
    for label in ranked_labels:
        if not hierarchical_valid(hierarchical_valid_labels + [label]):
            break
        else:
            hierarchical_valid_labels.append(label)
    if "_".join(sorted(hierarchical_valid_labels)) not in valid_paths:
        hierarchical_valid_labels = get_complete_path(hierarchical_valid_labels)
    return hierarchical_valid_labels
