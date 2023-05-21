import click
from datasets import DatasetDict
import os

@click.command()
@click.option('--input-ner-datasetdict', type=str, help='Path to input file')
@click.option('--output-dir', type=str, help='Path to output file')
def main(input_ner_datasetdict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ner_dataset_dict = DatasetDict.load_from_disk(input_ner_datasetdict)
    for key, split in ner_dataset_dict.items():
        label_names = split.features['ner_tags'].feature.names
        conll_dataset = []
        for snt in split:
            tokens = snt['tokens']
            ner_tags = [label_names[tag] for tag in snt['ner_tags']]
            conll_snt = "\n".join([f"{token}\t{ner_tag}" for token, ner_tag in zip(tokens, ner_tags)])
            conll_dataset.append(conll_snt)


        with open(os.path.join(output_dir, f"{key}.txt"), 'w') as f:
            f.write("\n\n".join(conll_dataset))

if __name__ == "__main__":
    main()
