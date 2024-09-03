"""
elo evaluating: 
    step0: 输入模型A，dataset
    step1: 按照要求/随机从所有model中 拿出一个model B
    step2: A和B 同时作答dataset中的问题，并把结果记录到文件中
    step3: judge 对A和B的得分进行评价
    step4: 计算elo分数  -- 当前改用true skill的参数
        有以下几个elo指标需要计算： 总elo指标、一个dataset的elo指标、 一个task的elo指标，
        还有两种模式：detailed: 按照每一个问题来计算，broad：一个dataset计算一个
    附加任务：把这次的比赛记录起来，要求记录：modelA，modelB，judge，token，timestamp,dataset，
        
@author limuyao
"""
import numpy as np
import multiprocessing as mp
from rich import print
from pathlib import Path
from tqdm import tqdm
import trueskill 
from vla_eval.evaluate import elo_validate,inference
from vla_eval.dataset import dataset_wrapper
from vla_eval.dataset.base import BaseDataset
from vla_eval.model import model
from utils import utils

DATA_FOLD = Path(__file__).parent.parent.parent / "data"
MODEL_PATH = DATA_FOLD / "model" / "model.json"
HISTORY_PATH = DATA_FOLD / "history.jsonl"
ROUND_NUM = 5   #一场比赛寻找好对手的迭代次数
ROUND_QUA = 0.3 #定义好比赛
STOP_A = 4
K = 32  #elo公式的更新参数

def offline_elo_evaluate(round_time=100,judge_model_name="gpt-4o"):
    timestamp = utils.generate_timestamp()
    model_ratings = model.get_model_ratings()
    history_jp = utils.JsonlProcessor(HISTORY_PATH)
    judge_model = model.Model(judge_model_name)
    for i in tqdm(range(round_time)):
        if i%5==0 and i>20:
            stop_flag=True
            for model_name in model_ratings.keys():
                if model_ratings[model_name].sigma > STOP_A:
                    stop_flag = False
                    break
            if stop_flag:
                break
        # sample出一个组合
        model_A_name,model_B_name,dataset_name = sample_A_B_D(model_ratings)
        # 正负比两场
        dataset = dataset_wrapper.make(dataset_name)
        model_A = model.Model(model_A_name)
        model_B = model.Model(model_B_name)
        outcome1 = elo_validate.offline_validate(dataset,model_A,model_B,judge_model)
        model_ratings,model_A,model_B,dataset = cal_elo(outcome1,model_ratings,model_A,model_B,dataset)  #其实没必要这么写，但是闲的
        outcome2 = elo_validate.offline_validate(dataset,model_B,model_A,judge_model)
        model_ratings,model_B,model_A,dataset = cal_elo(outcome2,model_ratings,model_B,model_A,dataset)
        model_A.upload_elo_rating(model_ratings[model_A.model_name])
        model_B.upload_elo_rating(model_ratings[model_B.model_name])
        outcome1.update({"timestamp":timestamp})
        outcome2.update({"timestamp":timestamp})
        history_jp.dump_line(outcome1)
        history_jp.dump_line(outcome2)
    print("done")


def online_elo_evaluate(dataset_name:str, model_A_name:str,model_B_name:str="",judge_model_name = "gpt-4o",motion="detailed"):
    raise AssertionError("未更新，无法使用")
    timestamp = utils.generate_timestamp()
    dataset = dataset_wrapper.make(dataset_name)
    if model_B_name == "":
        model_B_name = sample_B_model(model_A_name,dataset)
    # 启动！
    model_A = model.Model(model_A_name)
    model_B = model.Model(model_B_name)
    judge = model.Model(judge_model_name)
    print(f"[red]start evaluate, model A:{model_A_name}, model B: {model_B_name}, dataset: {dataset_name}")
    model_A.launch(device_num=1,port=9010)
    model_B.launch(device_num=1,port=9011)
    judge.launch(device_num=1,port=9012)
    # 运行两个模型
    p_A = run_inference(dataset,model_A,timestamp)
    p_B = run_inference(dataset,model_B,timestamp)
    # 评估表现
    validate_outcome = elo_validate.online_validate(dataset,model_A,model_B,timestamp,judge)
    # 计算得分
    cal_elos(validate_outcome["score"],model_A,model_B,dataset=dataset,motion=motion)
    
    # 记录
    history = {
        "timestamp":timestamp,
        "dataset":dataset.dataset_name,
        "model_A":model_A.model_name,
        "model_B":model_B.model_name,
        "judge":judge.model_name,
        "score":validate_outcome["score"],
        "token":validate_outcome["token"],
    }
    history_jp = utils.JsonlProcessor(HISTORY_PATH)
    history_jp.dump_line(history)
    print("完成记录")
    # 打扫打扫
    history_jp.close()
    p_A.join()
    p_B.join()
    model_A.stop()
    model_B.stop()
    judge.stop()
    
def run_inference(database:BaseDataset,inference_model:model.Model,timestamp):
    p = mp.Process(target=inference.inference,args=(database,inference_model,timestamp))
    p.start()
    return p
        
def sample_A_B_D(model_ratings:dict,model_file:dict={}):
    "从model_rating中选出一组高质量的对局" 
    # 首先按照标准差来选择A：
    if model_file=={}:
        model_file = utils.load_json_file(MODEL_PATH)
    avaliable_models = list(model_ratings.keys())
    model_sigma = [model_ratings[avaliable_model].sigma for avaliable_model in avaliable_models ]
    model_prob = np.exp(model_sigma) / np.sum(np.exp(model_sigma))
    model_A_name = np.random.choice(avaliable_models, size=1, p=model_prob)[0]
    dataset_name = np.random.choice(model_file[model_A_name]["done"])
    for _ in range(ROUND_NUM): #超参数，最大尝试次数
        model_B_name = sample_B_model(model_A_name,dataset_name,model_file)
        # 看看是不是好的比赛
        quality = trueskill.quality_1vs1(model_ratings[model_A_name],model_ratings[model_B_name],env=model.MY_ENV)
        if quality > ROUND_QUA:
            break
    return model_A_name,model_B_name,dataset_name
        
def sample_B_model(model_A_name:str,dataset_name:str,model_file:dict={}):
    if model_file=={}:
        model_file = utils.load_json_file(MODEL_PATH)
    avaliable_model = model.get_avaliable_model_set(model_file)
    model_set = set()
    for model_name in avaliable_model:
        if model_name == model_A_name:
            continue
        if dataset_name not in set(model_file[model_name]["done"]):
            continue        
        model_set.add(model_name)
    model_B_name = np.random.choice(list(model_set))
    return model_B_name

def estimated_win_loss_rate(A_rating,B_rating):
    """由A和B的rating来计算胜率 """
    e_A = 1/(1+10**((B_rating-A_rating)/400))
    return e_A

def update_elo_rating(A_rating,B_rating,outcome):
    
    e_A = estimated_win_loss_rate(A_rating,B_rating)
    s_A = (outcome - 1) / 2
    update_A = K*(s_A-e_A)
    new_A_rating,new_B_rating = A_rating + update_A, B_rating - update_A
    return int(new_A_rating), int(new_B_rating)

def init_elo_rating(model_elo_rating:dict,dataset:BaseDataset,my_task:str="total"):
    if "total" not in model_elo_rating:
        model_elo_rating["total"] = 1000
    if my_task == "total":
        tasks = dataset.get_tasks()
        for task in tasks:
            if task not in model_elo_rating:
                model_elo_rating[task] = 1000
    else:
        if my_task not in model_elo_rating:
            model_elo_rating[my_task] = 1000
    return model_elo_rating

def cal_elo(outcome:dict,model_rating:dict,model_A:model.Model,model_B:model.Model,dataset:BaseDataset):
    model_A_elo_rating = model_A.model_attr["elo rating"].get(dataset.dataset_name,{})
    model_B_elo_rating = model_B.model_attr["elo rating"].get(dataset.dataset_name,{})
    question_id = outcome["id"]
    task = dataset.get_task(question_id)
    score = outcome["score"]
    model_A_elo_rating = init_elo_rating(model_A_elo_rating,dataset=dataset,my_task=task)
    model_B_elo_rating = init_elo_rating(model_B_elo_rating,dataset=dataset,my_task=task)
    if score==3: #赢了
        model_rating[model_A.model_name],model_rating[model_B.model_name] = trueskill.rate_1vs1( model_rating[model_A.model_name],model_rating[model_B.model_name])
    elif score==2: #平
        model_rating[model_A.model_name],model_rating[model_B.model_name] = trueskill.rate_1vs1( model_rating[model_A.model_name],model_rating[model_B.model_name],drawn=True)
    elif score==1: #输了
        model_rating[model_B.model_name],model_rating[model_A.model_name] = trueskill.rate_1vs1( model_rating[model_B.model_name],model_rating[model_A.model_name])

    model_A_elo_rating["total"],model_B_elo_rating["total"] = update_elo_rating(model_A_elo_rating["total"],model_B_elo_rating["total"],score)
    model_A_elo_rating[task],   model_B_elo_rating[task]    = update_elo_rating(model_A_elo_rating[task],model_B_elo_rating[task],score)
    model_A.model_attr["elo rating"][dataset.dataset_name] = model_A_elo_rating
    model_B.model_attr["elo rating"][dataset.dataset_name] = model_B_elo_rating
    
    return model_rating,model_A,model_B,dataset
    
def cal_elos(scores:dict,model_A:model.Model,model_B:model.Model,dataset:BaseDataset,motion:str):
    """更新模型整体的elo rating数值
    """
    raise Exception("暂未更新，无法使用")
    task_scores = {}
    total_scores = []
    dataset_name = dataset.dataset_name
    for id in scores.keys():
        total_scores.append(scores[id])
        task = dataset.get_task(id)
        if task in task_scores:
            task_scores[task].append(scores[id])
        else:
            task_scores[task] = [scores[id]]
    
    model_A_elo_rating = model_A.model_attr["elo rating"]
    model_B_elo_rating = model_B.model_attr["elo rating"]
    
    model_A_elo_rating = init_elo_rating(model_A_elo_rating,dataset=dataset)
    model_B_elo_rating = init_elo_rating(model_B_elo_rating,dataset=dataset)
    
    if motion == "broad":
        total_score = np.mean(total_scores) # 计算平均成绩
        # 计算总成绩
        model_A_elo_rating["total"],model_B_elo_rating["total"] = update_elo_rating(model_A_elo_rating["total"],model_B_elo_rating["total"],total_score)
        # 计算该dataset的总成绩
        model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"] = update_elo_rating(model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"],total_score)
        # 计算该dataset每个task的总成绩
        for task,value in task_scores.items():
            task_score = np.mean(value)
            model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task] = update_elo_rating(model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task],task_score)
    elif motion == "detailed":
        for score in total_scores:
            model_A_elo_rating["total"],model_B_elo_rating["total"] = update_elo_rating(model_A_elo_rating["total"],model_B_elo_rating["total"],score)
            model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"] = update_elo_rating(model_A_elo_rating[dataset_name]["total"],model_B_elo_rating[dataset_name]["total"],score)
        for task,value in task_scores.items():
            for score in value:
                model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task] = update_elo_rating(model_A_elo_rating[dataset_name][task],model_B_elo_rating[dataset_name][task],score)
    else:
        raise AssertionError
    
    #print(model_A_elo_rating)
    #print(model_B_elo_rating)
    model_A.model_attr["elo rating"] = model_A_elo_rating
    model_B.model_attr["elo rating"] = model_B_elo_rating
    model_A.upload_model_attr()
    model_B.upload_model_attr()

if __name__ == "__main__":
    offline_elo_evaluate()
    #model_ratings = model.get_model_ratings()
    #print(sample_A_B_D(model_ratings))
    #dataset = dataset_wrapper.make("knowledge")
    #a = sample_B_model("10003-2024-09-02 09:46:53",dataset)
    #print(a)
    #Rating = model.MY_ENV.create_rating(mu=1309)
    #print(Rating)

    #elo_evaluate(dataset_name="knowledge",model_A_name="llama3-llava-next-8b-hf",model_B_name="gpt-4o-mini")


