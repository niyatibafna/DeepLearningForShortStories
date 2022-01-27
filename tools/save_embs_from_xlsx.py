# Saving 4071 sotory embeddings from xlsx
import argparse
import numpy as np
import pandas as pd


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")

    # Model
    parser.add_argument("--xlsx_file", type=str, default="4000-Stories-with-sentiment-analysis.xlsx")
    parser.add_argument("--output_file", type=str, default="embs.npy")

    return parser.parse_args()

def load_numpy_from_xlsx(file):
    """Convert dataframe into numpy."""
    df = pd.read_excel(file)
    col_lst = [str(i) for i in range(300)]
    embs = df.loc[:, col_lst].to_numpy()
    return embs


def main():
    # arguments
    args = get_args()

    # Create np array from xlsx
    embs = load_numpy_from_xlsx(args.xlsx_file)

    # Saving to .npy
    np.save(args.output_file, embs)
    print(f"Saving embeddgins with shape {embs.shape} to: {args.output_file}")

    # Assert the same embeddings
    loaded_embs = np.load(args.output_file)
    isEqual = np.array_equal(embs, loaded_embs, equal_nan=True)
    assert isEqual == True, "Saved embeddings not equal."


if __name__ == "__main__":
    main()


