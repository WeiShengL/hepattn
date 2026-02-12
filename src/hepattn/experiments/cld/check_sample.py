import os
import traceback

from hepattn.experiments.cld.data import CLDDataModule  # or whatever you actually use
# or: from hepattn.utils.lrsm_dataset import LRSMIterableDataset

SAMPLE_ID = 62633

def main():
    # Create the same dataset/dm you use in training
    dm = CLDDataModule(
        train_dir="/global/cfs/cdirs/m2616/wlai/cld/prepped_with_mask/train",
        val_dir="/global/cfs/cdirs/m2616/wlai/cld/prepped_with_mask/val",
        batch_size=8,
        num_workers=8,
        num_train=100000,
        num_val=1000,
        num_test=100,
    )
    dm.setup("fit")

    ds = dm.train_dataloader().dataset  # might be CombinedLoader; adjust if needed

    # If your dataset is wrapped, you may need to reach the underlying dataset object
    # e.g. ds = ds.datasets[0] or ds.dataset depending on your wrappers.

    try:
        out = ds.load_sample(SAMPLE_ID)
        print("Loaded OK. Keys:", out.keys() if hasattr(out, "keys") else type(out))
    except Exception as e:
        print(f"FAILED sample_id={SAMPLE_ID}")
        traceback.print_exc()

if __name__ == "__main__":
    main()