import click
import pdb

# データ種類ごとに統計量を獲得する
@click.command()
@click.option(
    "--msmlc-dataset-path",
    help="Multi Span Multi Label Dataset Path",
    default="data/gold/ccd37922365efc6ae9ff8d65a31c45b0b9a2bd73",
)
@click.option(
    "--show-PNR", help="flag to show Positive to Negative Ratio", is_flag=True
)
def main(msmlc_dataset_path, show_pnr):
    pdb.set_trace()
    pass


if __name__ == "__main__":
    main()
