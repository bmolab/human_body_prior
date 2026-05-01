from dotmap import DotMap
from human_body_prior.data.prepare_data import prepare_vposer_datasets
from human_body_prior.tools.omni_tools import log2file, makepath

amass_dir = r"D:\Git\human_body_prior\AMASS"
out_dir = r"D:\Git\human_body_prior\AMASS\DataSet\SFU_test"

amass_splits = DotMap({
    "train": ["SFU"],
    "vald": ["SFU"],
    "test": ["SFU"],
})

logger = log2file(makepath(out_dir, "prepare.log", isfile=True))
prepare_vposer_datasets(out_dir, amass_splits, amass_dir, logger=logger)
print("Prepared dataset at:", out_dir)
