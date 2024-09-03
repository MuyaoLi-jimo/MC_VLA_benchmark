"""
对model能力进行排名
+ elo ranking
"""
import pandas as pd
import numpy as np
from utils import utils
from vla_eval.model import model
from vla_eval.dataset import dataset_wrapper
NAN = 10000

def rank(ranking_type:str="elo"):
    if ranking_type == "elo":
        elo_rank(choice="total",if_print_elo=True)
        elo_rank(choice="knowledge",if_print_elo=True)
        elo_rank(choice="reason",if_print_elo=True)
        elo_rank(choice="visual",if_print_elo=True)
    return

def elo_rank(choice:str = "total",if_print_elo=False)->pd.DataFrame:
    """
    按照ranking来排名，
    + 如果选择total，那么输出名称，模型type，rating数值,  rank |type|model_name |elo rating| dataset1 |dataset2 |
    + 如果选择某数据集，那么输出名称,type ranking数值，以及各个subtask的排名/分数 rank |model_name|elo ranking|task1|task2
    返回pd
    """
    model_index = utils.load_json_file(model.MODEL_INDEX_PATH) #model的
    print_rank_df = None
    avaliable_dataset_names = dataset_wrapper.get_avaliable_dataset_names()
    if choice == "total":
        data = []
        for model_name, details in model_index.items():
            elo_ratings = details["elo rating"]
            total_elo_rating = elo_ratings["total"]["mu"] - 1 * elo_ratings["total"]["sigma"]
            row = {"model name": model_name,"elo rating":total_elo_rating,"type":details["type"]}
            for dataset_name in avaliable_dataset_names:
                row[dataset_name] = elo_ratings.get(dataset_name,{}).get("total", -NAN)
            data.append(row)
        data_sorted = sorted(data, key=lambda x: x["elo rating"], reverse=True)
        rank_df = pd.DataFrame(data_sorted) 
        rank_df = rank_df.replace(-NAN,np.nan)
        rank_df['rank'] = rank_df["elo rating"].rank(method='min', ascending=False) #如果有重复的，按照最小值来
        for dataset_name in avaliable_dataset_names:
            rank_df[f"{dataset_name}_rank"] = rank_df[dataset_name].rank(method='min', ascending=False)
        cols_name = ["rank", "model name", "type","elo rating"] + [f"{dataset_name}" for dataset_name in avaliable_dataset_names]
        if not if_print_elo:
            cols = ["rank", "model name", "type","elo rating"] + [f"{dataset_name}_rank" for dataset_name in avaliable_dataset_names]
            print_rank_df = rank_df[cols]
            print_rank_df.columns = cols_name
        else:
            print_rank_df = rank_df[cols_name]
        print_rank_df = print_rank_df.replace(np.nan, 0)
        # 来个暴力的
        print_rank_df = print_rank_df.apply(lambda x: x.astype('int') if x.dtype == 'float64' else x)
        print_rank_df = print_rank_df.replace(0, "")
        print_rank_df = print_rank_df.replace("temp", "finetune")

    else: 
        data = []
        if choice not in avaliable_dataset_names: #avaliable_dataset_names 是set
            raise ValueError(f"{choice} is not available. Please choose from {avaliable_dataset_names}.")
        dataset = dataset_wrapper.make(choice)
        dataset_tasks = list(dataset.get_tasks())
        for model_name, details in model_index.items():
            elo_ratings = details["elo rating"]
            dataset_elo_ratings = elo_ratings.get(choice, None)
            if dataset_elo_ratings:
                row = {"model name": model_name,choice: dataset_elo_ratings["total"]}
                row.update(dataset_elo_ratings)
            else:
                row = {"model name": model_name, choice: -NAN}
            for task in dataset_tasks:
                if task not in row:
                    row[task] = - NAN
            data.append(row)
        data_sorted = sorted(data, key=lambda x: x[choice], reverse=True)
        rank_df = pd.DataFrame(data_sorted) 
        rank_df = rank_df.replace(-NAN,np.nan)
        rank_df['rank'] = rank_df[choice].rank(method='min', ascending=False) #如果有重复的，按照最小值来
        for task in dataset_tasks:
            rank_df[f"{task}_rank"] = rank_df[task].rank(method='min', ascending=False)
        cols_name = ["rank", "model name", choice] + [f"{task}" for task in dataset_tasks]
        if not if_print_elo:
            cols = ["rank", "model name", choice] + [f"{task}_rank" for task in dataset_tasks]
            print_rank_df = rank_df[cols]
            print_rank_df.columns = cols_name
        else:
            print_rank_df = rank_df[cols_name]
            
        print_rank_df = print_rank_df.replace(np.nan, 0)
        # 来个暴力的
        print_rank_df = print_rank_df.apply(lambda x: x.astype('int') if x.dtype == 'float64' else x)
        print_rank_df = print_rank_df.replace(0, "")
    print(print_rank_df)
    return print_rank_df

if __name__ in "__main__":
    rank()