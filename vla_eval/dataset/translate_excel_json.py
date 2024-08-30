from pathlib import Path
from utils import utils

DATASET_FOLD = Path(__file__).parent.parent.parent / "data" / "dataset"


def json_to_excel():
    
    json_stem = "reason"
    json_name = f"{json_stem}.json"
    excel_name = f"{json_stem}.xlsx"
    json_path = DATASET_FOLD / json_name
    excel_path = DATASET_FOLD / excel_name
    json_file = utils.load_json_file(json_path)
    utils.dump_excel_file(json_file,excel_path)
    
def excel_to_json():
    excel_stem = "reason"
    json_name = f"{excel_stem}.json"
    excel_name = f"{excel_stem}.xlsx"
    json_path = DATASET_FOLD / json_name
    excel_path = DATASET_FOLD / excel_name
    excel_file = utils.load_excel_file_to_dict(excel_path)
    utils.dump_json_file(excel_file,json_path)
    

if __name__ == "__main__":
    excel_to_json()