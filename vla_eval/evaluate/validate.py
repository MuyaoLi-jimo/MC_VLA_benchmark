"""
把得到的结果交给gpt-4o来评价
先验：每一个log文件只会存储一个dataset的数据（方便处理），注意要考虑两个模型一样的情况
"""

from pathlib import Path
from rich import print
from utils import utils
from vla_eval.dataset.base import BaseDataset
from vla_eval.dataset import dataset_wrapper
from vla_eval.model import model
import re
import time

DATA_FOLD = Path(__file__).parent.parent.parent / "data"
LOG_FOLD = DATA_FOLD / "log"
DATASET_FOLD = DATA_FOLD / "dataset"
DATASET_INDEX_PATH = DATASET_FOLD / "index.json"

SYSTEM_PROMPT  = "Assume you are an expert in the field of Minecraft. You have been asked to evaluate answers from two assistents, referred to as A and B, who have both responded to a Minecraft-related question. Your task is to evaluate which response is better.\n"
#"You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "
REQUIREMENT_PROMPT = "Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n"
OPTION_PROMPT = "You can choose only from the following options: A is better, B is better, Tie (if both answers perform similarly).\n"
FORMAT_PROMPT = "Output your final evaluation by strictly following this format: Reason,  [Your Choice].\n" 
DIVIDE_PROMPT = "#############################\n"


def validate(dataset:BaseDataset, test_model_A:model.Model,test_model_B:model.Model,timestamp,judge_name = "gpt-4o",batch_size=10):
    """负责比较两者，同时需要它把所有信息记录下来,并记录所有token使用情况(return即可) """
    modelA_log_path = LOG_FOLD / f"{timestamp}_{test_model_A.model_name}_{dataset.dataset_name}.jsonl"
    modelB_log_path = LOG_FOLD / f"{timestamp}_{test_model_B.model_name}_{dataset.dataset_name}.jsonl"
    modelA_jp = utils.JsonlProcessor(modelA_log_path)
    modelB_jp = utils.JsonlProcessor(modelB_log_path)
    evaluate_log_path = LOG_FOLD / f"{timestamp}_{dataset.dataset_name}.json"  #记录
    judge_model = model.Model(judge_name)
    judge_model.launch()
    dataset_content_dict = dataset.dataset_to_dict() 
    total_id = set(dataset_content_dict.keys())

    visited_datas = {}
    unvisited_datas_A = {}
    unvisited_datas_B = {}
    
    while set(visited_datas.keys())!=total_id: #获取inference得到的数据,每一个batch获取一次
        batch_datas = {}
        while len(batch_datas)<batch_size:
            lineA = modelA_jp.load_lines()
            lineB = modelB_jp.load_lines()
            unvisited_datas_A = add_unvisited_datas(lineA,unvisited_datas_A)
            unvisited_datas_B = add_unvisited_datas(lineB,unvisited_datas_B)
            batch_datas = get_shared_data(unvisited_datas_A,unvisited_datas_B,batch_datas,q_a_datas=dataset_content_dict,content_attrs=dataset.dataset_attribute["attrs"]) #得到重合的部分
        # 接下来，制造传递给judge的数据
        input_datas = []
        for id,data in batch_datas.items():
            input_data = {
                "id":id,
                "messages":create_message(dataset.dataset_attribute, data)
            }
            input_datas.append(input_data)
        print(input_datas)
        batch_results = judge_model.inference(datas = input_datas)
        print(batch_results)
    return

def add_unvisited_datas(lines:list,unvisited_datas:dict):
    for line in lines:
        unvisited_datas[line["id"]] = line
    return unvisited_datas
    
def get_shared_data(datas_A:dict,datas_B:dict,shared_datas:dict,q_a_datas:dict,content_attrs:list):
    shared_datas = {}
    for id in datas_A.keys():
        if id in datas_B:
            shared_datas[id] = {
                "id":id,
                "A":datas_A[id]["a"]["content"],
                "B":datas_B[id]["a"]["content"],
                "token":{
                    "a_in":datas_A[id]["input_tokens"],
                    "a_out":datas_A[id]["output_tokens"],
                    "b_in":datas_B[id]["input_tokens"],
                    "b_out":datas_B[id]["output_tokens"],
                }
            }
            for content_attr in content_attrs:
                #"question,answer,explanation"
                shared_datas[id][content_attr] = q_a_datas[id][content_attr]
    return shared_datas


def create_system_prompt(dataset_attr:dict):
    prompt = ""
    prompt += SYSTEM_PROMPT
    prompt += REQUIREMENT_PROMPT
    prompt += dataset_attr["validate requirement prompt"]
    prompt += OPTION_PROMPT
    prompt += FORMAT_PROMPT
    if "validate example prompt" in dataset_attr:
        prompt += dataset_attr["validate example prompt"]
    return prompt
    
def create_input_prompt(row_input:dict):
    prompt = ""
    prompt += "[question start]\n"
    prompt += row_input["question"]
    prompt += "\n[question end]\n\n"
    if "answer" in row_input:
        prompt += "[reference answer start]\n"
        prompt += row_input["answer"]
        prompt += "\n[reference answer end]\n\n"
    if "explanation" in row_input: 
        prompt += "[explanation start]\n"
        prompt += row_input["explanation"]
        prompt += "\n[explanation end]\n\n"
    prompt += "[answer A start]\n"
    prompt += "A: "
    prompt += row_input["A"]
    prompt += "\n[answer A end]\n\n"
    prompt += "[answer B start]\n"
    prompt += "B: "
    prompt += row_input["B"]
    prompt += "\n[answer B end]\n\n"
    return prompt
    
def create_message(dataset_attr:dict,row_input:dict):
    #"role": "user", "content":""
    messages = []
    messages.append({
        "role": "system",
        "content":create_system_prompt(dataset_attr)
    })
    messages.append({
        "role": "user",
        "content":create_input_prompt(row_input) + DIVIDE_PROMPT + "My evaluation: ",
    })
    return messages
    
def analyze_evaluation(evaluation):
    """解析选项
    
    """
    judgment_map = {
        # 没找到：0
        "A is better":1,
        "B is better":2,
        "Tie":3,
    }
    judgement = ""
    match = re.search(r'\[([^\]]+)\](?:\.)?$', str(evaluation))
    if match:
        judgement =  match.group(1)  # Return the content inside the brackets
    judgement_num = judgment_map.get(judgement,0)
    return judgement_num
        
    
if __name__ == "__main__":
    model_A = model.Model("llama3-llava-next-8b-hf")
    model_B = model.Model("gpt-4o")
    dataset = dataset_wrapper.make(dataset_name="knowledge")
    validate(dataset=dataset,test_model_A=model_A,test_model_B=model_B,timestamp="2024-08-29 21:13:49")