from vla_eval.dataset.base import BaseDataset
from utils import utils

class VisualBaseDataset(BaseDataset):
    MODALITY = 'visual'
    def __init__(self, dataset_name):
        super().__init__(dataset_name)  
        # self.dataset_name,
        # self.dataset_path
        # self.dataset_attribute
        # self.dataset_content
        if self.dataset_attribute["type"] != self.MODALITY:
            print(f"[red] {dataset_name}没有注册类别")
            raise AssertionError
      
    def get_questions(self):
        """输出list格式 """
        if type(self.questions)==type(None):
            self.questions = {}
            for task,rows in self.dataset_content.items():
                self.questions[task] = []
                for row in rows:
                    question = row["question"]
                    image_path = row["image_path"]
                    suffix = image_path.split(".")[-1]
                    question = {
                        "id":row["id"],
                        "message":{
                            "role": "user",
                            "content": [
                                    {
                                    "type": "text",
                                    "text": f"<image>{question}\n"
                                    },
                                    {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{suffix};base64,{utils.encode_image_to_base64(image_path)}"
                                    },
                                    },
                                ]
                        }
                    }
                    self.questions[task].append(question)
        return self.questions
    
    def get_answers(self):
        dataset_dict = self.get_dataset_content_as_dict()
        return dataset_dict
    
if __name__ == "__main__":
    td = VisualBaseDataset("visual")
    print(td.get_questions())