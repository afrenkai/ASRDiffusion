from datasets import load_dataset
from utils.consts import DATASET_ROOT, DATASET_SPLITS, DATASET_SUBSETS 
import argparse

def ds_use(root: str = DATASET_ROOT, split: str | list[str] = DATASET_SPLITS[0], subset: str | list[str] = DATASET_SUBSETS[0], streaming: bool = False, sample: bool = False) -> dict:
    """
    gets hf dataset for local use. look in utils/consts.py for what the consts contain for librispeech. 
    if streaming enabled, doesn't download the dataset and instead creates an IterableDataset object. Look at
    https://huggingface.co/docs/datasets/en/stream for more info. 
    """

    if sample:
        root = "hf-internal-testing/librispeech_asr_demo"
        split = ["validation"]
        subset = [None]
    if isinstance(split, str):
        split = [split]
    if isinstance(subset, str) or subset is None:
        subset = [subset]

    results = {}

    for s in split:
        for su in subset:
            ds = load_dataset(path=root, 
                              name=su,
                              split=s, 
                              streaming=streaming)
            
            results[(s, su)] = ds
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help='root of the dataset on hf', default = DATASET_ROOT, required=False)
    parser.add_argument("--splits", type=str, nargs='+', help='1 or more splits to load/download', default = DATASET_SPLITS, required=False)
    parser.add_argument("--subset", type=str, nargs='+', help = "1 or more subsets of the dataset to load in", default = DATASET_SUBSETS, required = False)
    parser.add_argument('--streaming', type=bool, help='whether to use streaming to load data from the dataset or use disk/download', default = False )
    parser.add_argument("--sample", action="store_true", help="Use the 73-sample hf-internal test ds")

    args = parser.parse_args()

    if args.sample:
        args.root = "hf-internal-testing/librispeech_asr_demo"
        args.splits = ["validation"]
        args.subset = [None]
    
    print(f"{"streaming" if args.streaming else "loading"} the split(s) of {args.splits} with subsets {args.subset} from {args.root}...")
    ds_dict = ds_use(args.root, args.splits, args.subset, args.streaming)
    print("Loaded datasets:", list(ds_dict.keys()))
