from pathlib import Path


def make_pubmed_xml_to_text(file):
    output_file = Path(file[: file.rfind(".xml")] + ".txt")
    if not output_file.exists():
        import xml.etree.ElementTree as ET

        tree = ET.parse(file)
        root = tree.getroot()
        texts = []
        for abstract_text in root.findall(
            "./PubmedArticle/MedlineCitation/Article/Abstract/AbstractText"
        ):
            if abstract_text.text:
                texts += [abstract_text.text]
        with open(output_file, "w") as f:
            f.write("\n".join(texts))


import sys

if __name__ == "__main__":
    make_pubmed_xml_to_text(sys.argv[1])
