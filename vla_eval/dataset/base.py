import abc
from pathlib import Path,PosixPath
from utils import utils

DATASET_FOLD = Path(__file__).parent.parent.parent / "data" / "dataset"
DATASET_INDEX_PATH = DATASET_FOLD / "index.json"

class BaseDataset(abc.ABC):
    """提供一个对外输出benchmark的接口 """
    database_fold = DATASET_FOLD
    database_index_path = DATASET_INDEX_PATH
    def __init__(self,dataset_name=""):

        self.dataset_index = utils.load_json_file(self.database_index_path)
        self.DATASET_LIST = self.get_dataset_list(self.dataset_index)
        
        if dataset_name not in self.DATASET_LIST:
            print(f"[red]{dataset_name} not found")
            raise FileNotFoundError
        self.dataset_name = dataset_name
        self.dataset_path = self.database_fold / f"{self.dataset_name}.json"
        self.dataset_attribute = self.get_dataset_attribute()
        self.dataset_content = self.get_dataset_content()
        self.questions = None
        
    @classmethod
    def get_dataset_list(cls,dataset_index:dict=None):
        if type(dataset_index)==type(None):
            dataset_index = utils.load_json_file(cls.database_index_path)
        dataset_names = set()
        for dataset_name in dataset_index.keys():
            if dataset_index[dataset_name]["available"]:
                dataset_names.add(dataset_name)
        return dataset_names
    
    def get_dataset_attribute(self):
        """获取当前dataset的属性 """
        return self.dataset_index[self.dataset_name]
    
    def get_dataset_content(self):
        return utils.load_json_file(self.dataset_path)


    @abc.abstractmethod
    def get_questions(self):
        pass
    
    @abc.abstractmethod
    def get_answers(self):
        pass  
    
    def dataset_to_dict(self):
        """用id来索引 """
        dataset_dict = {}
        for label, rows in self.dataset_content.items():
            for row in rows:
                dataset_dict[row["id"]] = row
                dataset_dict[row["id"]]["label"] = [self.dataset_name,label]
        return dataset_dict
        
        
            
if __name__ == "__main__":
    bd = BaseDataset("reason")