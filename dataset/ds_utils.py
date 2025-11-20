from datasets import load_dataset
from utils.consts import DATASET_ROOT, DATASET_SPLITS, DATASET_SUBSETS 
import argparse

def ds_use(root: str = DATASET_ROOT, split: str | list[str] = DATASET_SPLITS[0], subset: str | list[str] = DATASET_SUBSETS[0], streaming: bool = False) -> None:
    """
    gets hf dataset for local use. look in utils/consts.py for what the consts contain for librispeech. 
    if streaming enabled, doesn't download the dataset and instead creates an IterableDataset object. Look at
    https://huggingface.co/docs/datasets/en/stream for more info. 
    """

    ds = load_dataset(root, split, subset, streaming=streaming)
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help='root of the dataset on hf', default = DATASET_ROOT, required=False)
    parser.add_argument("--splits", type=str, nargs='+', help='1 or more splits to load/download', default = DATASET_SPLITS, required=True)
    parser.add_argument("--subset", type=str, nargs='+', help = "1 or more subsets of the dataset to load in", default = DATASET_SUBSETS, required = True)
    parser.add_argument('--streaming', type=bool, help='whether to use streaming to load data from the dataset or use disk/download', default = False )
    args = parser.parse_args()
    
    print(f"{"streaming" if args.streaming else "loading"} the split(s) of {args.splits} with subsets {args.subset} from {args.root}...")
    ds_use(args.root, args.splits, args.subset)

