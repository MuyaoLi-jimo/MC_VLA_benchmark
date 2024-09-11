from rich import print
from vla_eval.dataset.base import BaseDataset
from vla_eval.model.model import Model

class TextbaseDataset(BaseDataset):
    MODALITY = 'text'
    def __init__(self, dataset_name):
        super().__init__(dataset_name)  
        # self.dataset_name,
        # self.dataset_path
        # self.dataset_attribute
        # self.dataset_content
        if self.dataset_attribute["type"] != self.MODALITY:
            print(f"[red] {dataset_name}没有注册类别")
            raise AssertionError
        
    def get_questions(self,model:Model):
        """输出list格式 """
        if type(self.questions)==type(None):
            self.questions = {}
            for task,rows in self.dataset_content.items():
                self.questions[task] = []
                for row in rows:
                    question = {
                        "id":row["id"],
                        "message":{
                            "role": "user",
                            "content": row["question"],
                        }
                    }
                    self.questions[task].append(question)
        return self.questions
    
    def get_answers(self):
        dataset_dict = self.get_dataset_content_as_dict()
        return dataset_dict
    
    
        
if __name__ == "__main__":
    td = TextbaseDataset("knowledge")
    print(td.get_questions())
        
        
        