""" 
获取数据集的api
"""


from pathlib import Path,PosixPath
from utils import utils
from vla_eval.dataset.image_base import ImageBaseDataset
from vla_eval.dataset.text_base import TextaseDataset


DATASET_FOLD = Path(__file__).parent.parent.parent / "data" / "dataset"
DATASET_INDEX_PATH = DATASET_FOLD / "index.json"

def get_dataset_type_map():
    dataset_index = utils.load_json_file(DATASET_INDEX_PATH)
    dataset_map = {}
    for dataset_name,value in dataset_index.items():
        dataset_map[dataset_name] = value["type"]
    return dataset_map

def make(dataset_name:str):
    """得到该验证集的api """
    dataset_map = get_dataset_type_map()
    dataset_type = dataset_map[dataset_name]
    if dataset_type == "text":
        dataset = TextaseDataset(dataset_name)
    elif dataset_type == "image":
        dataset= ImageBaseDataset(dataset_name)
    else:
        raise AssertionError
    return dataset
    