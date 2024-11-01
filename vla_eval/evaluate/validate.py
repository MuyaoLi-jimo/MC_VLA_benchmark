"""
把得到的结果交给gpt-4o来评价
需要进行一个多步推理，保证gpt-4o按照规定格式输出
"""
import numpy as np
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

SYSTEM_PROMPT_2 = "You are an expert in the field of Minecraft. You have been asked to evaluate answers from two assistant, referred to as A and B, who have both responded to a Minecraft-related question. \n"
REQUIREMENT_PROMPT_2 = "Assuming you have just completed an evaluation, the next section involves your analysis of the outcomes for options A and B, along with your decision. Please summarize your judgment. If you previously determined that B is better, respond with [B is better]. If you concluded that A is better, respond with [A is better]. If your assessment resulted in a tie, please answer with [Tie].\n"

def sample_validate_qa(dataset:BaseDataset, test_model_A:model.Model,test_model_B:model.Model):
    """从某数据集中挑出一个问题，进行测评"""
    q_a = dataset.sample()
    uuid = q_a["id"]
    model_A_response = test_model_A.get_dataset_responses(dataset.dataset_name)[uuid]
    model_B_response = test_model_B.get_dataset_responses(dataset.dataset_name)[uuid]
    input_source = get_validate_qa(model_A_response,model_B_response,q_a,content_attrs=dataset.dataset_attribute["attrs"])
    return input_source

def record_validate(score,dataset:BaseDataset, validate_qa:dict, test_model_A:model.Model,test_model_B:model.Model,):
    """将选择的结果转换成固定的格式"""
    output_data = {
        "score":score,
        "dataset":dataset.dataset_name,
        "model_A":test_model_A.model_name,
        "model_B":test_model_B.model_name,
    }
    output_data.update(validate_qa)
    return output_data

def offline_validate(dataset:BaseDataset, test_model_A:model.Model,test_model_B:model.Model,judge_model:model.Model):
    validate_qa = sample_validate_qa(dataset,test_model_A,test_model_B)
    input_data = [{
        "id":validate_qa["id"],
        "messages":create_message(dataset.dataset_attribute, validate_qa)
    }]
    judge_model.launch()
    try:
        half_result = judge_model.inference(input_data)
        re_input_data = regenerate_judgement(half_result)
        final_result = judge_model.inference(datas = re_input_data)[0]
    except Exception as e:
        print(e)
    try:
        judgement = final_result["message"]["content"]
        score = analyze_evaluation(judgement)
    except:
        score = 0
    output_data = record_validate(score,dataset,validate_qa,test_model_A,test_model_B)
    output_data.update({
        "half_judge":half_result[0]["message"]["content"],
        "final_judge":judgement,
        "token":{
            "input":int(half_result[0]["input_tokens"]) + int(final_result["input_tokens"]),
            "output":int(half_result[0]["output_tokens"]) + int(final_result["output_tokens"]),
        },
    })
    return output_data


def online_validate(dataset:BaseDataset, test_model_A:model.Model,test_model_B:model.Model,timestamp,judge_model:model.Model,batch_size=10):
    """负责比较两者，同时需要它把所有信息记录下来,并记录所有token使用情况(return即可) """
    modelA_log_path = LOG_FOLD / f"{timestamp}_{dataset.dataset_name}_{test_model_A.model_name}.jsonl"
    modelB_log_path = LOG_FOLD / f"{timestamp}_{dataset.dataset_name}_{test_model_B.model_name}.jsonl"
    modelA_jp = utils.JsonlProcessor(modelA_log_path)
    modelB_jp = utils.JsonlProcessor(modelB_log_path)
    judgement_log_path = LOG_FOLD / f"{timestamp}_{dataset.dataset_name}.json"  #记录
    dataset_content_dict = dataset.get_dataset_content_as_dict() 
    total_id = set(dataset_content_dict.keys())

    visited_datas = {}
    output = {
        "score":{},
        "token":{"a_in":0,"a_out":0,"b_in":0,"b_out":0,"j_in":0,"j_out":0,},#记录一下总花费
    }  
    unvisited_datas_A = {}
    unvisited_datas_B = {}
    
    time_flag = False
    start_flag = True
    while set(visited_datas.keys())!=total_id: #获取inference得到的数据,每一个batch获取一次
        batch_datas = {}
        start_time = time.time()
        while len(batch_datas)<batch_size:
            lineA = modelA_jp.load_lines()
            lineB = modelB_jp.load_lines()
            unvisited_datas_A = add_unvisited_datas(lineA,unvisited_datas_A)
            unvisited_datas_B = add_unvisited_datas(lineB,unvisited_datas_B)
            batch_datas = get_shared_datas(unvisited_datas_A,unvisited_datas_B,batch_datas,q_a_datas=dataset_content_dict,content_attrs=dataset.dataset_attribute["attrs"]) #得到重合的部分
            if len(batch_datas)!=0: #如果出现了数据，后面不需要等待启动了
                start_flag = False
            if len(batch_datas)!=0 and time.time()-start_time > 30: #如果暂时还没凑够数据,但是已经很久了，那先break（有可能到结尾了）
                break
            if start_flag and time.time()-start_time > 20*60: #如果是一开始的等待，但是等了20分钟，可以斩了
                time_flag = True
                break
            if not start_flag and time.time()-start_time > 6*60: #如果不是一开始的等待，并且里面什么都没有，还等了6分钟，斩了
                time_flag = True
                break
            if time_flag: #一开始不需要那么快
                time.sleep(5)
            time.sleep(5)
        if len(batch_datas)==0:
            break
        # 接下来，制造传递给judge的数据
        input_datas = []
        for id,data in batch_datas.items():
            input_data = {
                "id":id,
                "messages":create_message(dataset.dataset_attribute, data)
            }
            input_datas.append(input_data)
        #print(input_datas)
        try: #处理数据
            batch_results = judge_model.inference(datas = input_datas)
            # 再根据这个的结果再生成最终的输出
            re_inputs = regenerate_judgement(batch_results)
            final_results = judge_model.inference(datas = re_inputs)
            for result,re_result in zip(batch_results,final_results):
                try:
                    judgement = re_result["message"]["content"]
                    score = analyze_evaluation(judgement)
                except:
                    score = 0
                # 如果失败，还需要再次尝试，但在此之前，先记录一下花费
                if score == 0:
                    output["token"]["j_in"]+=result["input_tokens"] + re_result["input_tokens"]
                    output["token"]["j_out"]+=result["output_tokens"] + re_result["output_tokens"]
                    continue
                # 如果一切正常，记录最终score和其他数据
                id = result["id"]
                visited_datas[id] = batch_datas[id]
                visited_datas[id]["judgement"] = judgement
                visited_datas[id]["score"] = score
                visited_datas[id]["token"]["j_in"] =  result["input_tokens"]
                visited_datas[id]["token"]["j_out"] =  result["output_tokens"]
                # 再记录一下花费和结果：
                output["score"][id] = score
                output["token"]["a_in"]+=visited_datas[id]["token"]["a_in"]
                output["token"]["a_out"]+=visited_datas[id]["token"]["a_out"]
                output["token"]["b_in"]+=visited_datas[id]["token"]["b_in"]
                output["token"]["b_out"]+=visited_datas[id]["token"]["b_out"]
                output["token"]["j_in"]+=result["input_tokens"] + re_result["input_tokens"]
                output["token"]["j_out"]+=result["output_tokens"] + re_result["output_tokens"]
        except Exception as e:
            print(e)
        if time_flag: #肯定不会到这里，只是提高鲁棒性
            break   
    utils.dump_json_file(visited_datas,judgement_log_path)
    return output

def add_unvisited_datas(lines:list,unvisited_datas:dict):
    for line in lines:
        unvisited_datas[line["id"]] = line
    return unvisited_datas
    
def get_validate_qa(model_A_response,model_B_response,q_a,content_attrs:list):
    input_source = {
        "id":model_A_response["id"],
        "A":model_A_response["a"]["content"],
        "B":model_B_response["a"]["content"],
    }
    for content_attr in content_attrs:
        #"question,answer,explanation"
        input_source[content_attr] = q_a[content_attr]
    input_source["task"] = q_a["label"][1]
    return input_source
    
def get_shared_datas(datas_A:dict,datas_B:dict,shared_datas:dict,q_a_datas:dict,content_attrs:list):
    shared_datas = {}
    for id in datas_A.keys():
        if id in datas_B:
            shared_datas[id] = get_validate_qa(datas_A[id],datas_B[id],q_a_datas[id],content_attrs)
            shared_datas["token"]={
                "a_in":datas_A[id]["input_tokens"],
                "a_out":datas_A[id]["output_tokens"],
                "b_in":datas_B[id]["input_tokens"],
                "b_out":datas_B[id]["output_tokens"],
            }
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
    
def create_message2(row_input:dict):
    messages = []
    messages.append({
        "role": "system",
        "content":SYSTEM_PROMPT_2 + OPTION_PROMPT + REQUIREMENT_PROMPT_2,
    })
    messages.append({
        "role": "user",
        "content":"judgement: " + row_input["message"]["content"],
    })
    return messages
      
def regenerate_judgement(batch_results):
    re_input_datas = []
    for batch_result in batch_results:
        re_input_data = {
            "id":id,
            "messages":create_message2(batch_result)
        }
        re_input_datas.append(re_input_data)
    return re_input_datas
       
def analyze_evaluation(evaluation):
    """解析选项
    
    """
    judgment_map = {
        # 没找到：0
        "A is better":3, #win
        "Tie":2,         #tie
        "B is better":1, #loss
    }
    judgement = ""
    match = re.search(r'\[([^\]]+)\](?:\.)?$', str(evaluation))
    if match:
        judgement =  match.group(1)  # Return the content inside the brackets
    judgement_num = judgment_map.get(judgement,0)
    return judgement_num
        
if __name__ == "__main__":
    dataset = dataset_wrapper.make("visual")
    model_A = model.Model("gpt-4o-mini")
    model_B = model.Model("gpt-4o")
    
    
    print(sample_validate_qa(dataset,model_A,model_B))
    exit()
    #timestamp = utils.generate_timestamp()
    model_A = model.Model("10003-2024-09-02 09:46:53")
    model_B = model.Model("gpt-4o-mini")
    model_judge = model.Model("gpt-4o")
    dataset = dataset_wrapper.make(dataset_name="knowledge")
    outcome = offline_validate(dataset,model_A,model_B,model_judge)
    print(outcome)
    #validate_output = online_validate(dataset=dataset,test_model_A=model_A,test_model_B=model_B,timestamp="2024-08-29 21:13:49",judge_model=judge_model)
    #print(validate_output)
    #judge_model.stop()