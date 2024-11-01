from vla_eval.dataset.base import BaseDataset
from vla_eval.model.model import Model
import cv2
from utils import utils

RESIZE_MAP = {
    "llava_next":[672,336],
}

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
      
    def get_questions(self,model:Model=None):
        """输出list格式 """
        if self.questions:
            return self.questions
        model_name = model.model_name
        self.questions = {}
        for task,rows in self.dataset_content.items():
            self.questions[task] = []
            for row in rows:
                question = row["question"]
                image_path = row["image_path"]
                suffix = image_path.split(".")[-1]
                base64_image = utils.encode_image_to_base64(image_path)
                #if model_attr.get("base","")=="llava_next": 
                    #image = cv2.imread(image_path)
                    #resized_image = cv2.resize(image, RESIZE_MAP[model_attr["base"]]) if image.shape[1]==640 else image
                    ## 将调整后的图像编码为JPEG格式（也可以选择PNG等其他格式）
                    #success, base64_image = cv2.imencode("."+suffix, resized_image)
                    #if not success:
                        #base64_image = utils.encode_image_to_base64(image_path)
                if model_name in { "MiniCPM-V-2_6" ,"llava-1.5-13b-hf", "llava-1.5-7b-hf", "fuyu-8b","llava-v1.6-mistral-7b-hf"} or "molmo" in model_name:
                    input_question = {
                        "id":row["id"],
                        "message":{
                            "role": "user",
                            "content": [
                                {
                                "type": "text",
                                "text": f"{question}\n"
                                },
                                {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{suffix};base64,{base64_image}"
                                },
                                },
                            ]
                        }
                    }
                else:
                    input_question = {
                        "id":row["id"],
                        "message":{
                            "role": "user",
                            "content": [
                                {
                                "type": "text",
                                "text": f"{question}<image>\n"
                                },
                                {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{suffix};base64,{base64_image}"
                                },
                                },
                            ]
                        }
                    }
                self.questions[task].append(input_question)
        return self.questions

    
    def get_answers(self):
        dataset_dict = self.get_dataset_content_as_dict()
        return dataset_dict
    
if __name__ == "__main__":
    model = Model("llama3-llava-next-8b-hf")
    dataset = VisualBaseDataset("visual")
    print(len(dataset.get_questions(model)["info"]))