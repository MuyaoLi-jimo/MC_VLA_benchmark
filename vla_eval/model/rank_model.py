"""
对model能力进行排名
+ elo ranking
"""
import pandas as pd
import numpy as np
from utils import utils
from vla_eval.model import model
from vla_eval.dataset import dataset_wrapper
from rich.console import Console
from rich.table import Table
NAN = 10000

def rank(ranking_type:str="elo",choice:str="total",if_human=False):
    if ranking_type == "elo":
        rank_df = elo_rank(choice="total",if_print_elo=True,if_human=if_human)
        rich_print(rank_df,choice)
        rank_df = elo_rank(choice="knowledge",if_print_elo=True,if_human=if_human)
        rich_print(rank_df,choice)
        rank_df = elo_rank(choice="reason",if_print_elo=True,if_human=if_human)
        rich_print(rank_df,choice)
        rank_df = elo_rank(choice="visual-basic",if_print_elo=True,if_human=if_human)
        rich_print(rank_df,choice)
        rank_df = elo_rank(choice="visual-advance",if_print_elo=True,if_human=if_human)
        rich_print(rank_df,choice)
    return

def rich_print(rank_df:pd.DataFrame,title:str):
    table = Table(title=title)
    column = rank_df.columns
    if title == "total":
        table.add_column(column[0], justify="right", style="cyan")
        table.add_column(column[1], justify="left", style="black bold", no_wrap=False)  # 修正justify为"center"
        table.add_column(column[2], justify="right", style="green")
        table.add_column(column[3], justify="center", style="magenta")
        table.add_column(column[4], justify="left", style="blue")
        for i in range(5, len(column)):
            table.add_column(column[i], justify="center")
    else:
        table.add_column(column[0], justify="right", style="cyan")
        table.add_column(column[1], justify="left", style="black bold", no_wrap=False)  # 修正justify为"center"
        table.add_column(column[2], justify="right", style="green")
        table.add_column(column[3], justify="center", style="magenta")
        for i in range(4, len(column)):
            table.add_column(column[i], justify="center")
    # 遍历 DataFrame 的每一行，并将数据添加到表格中
    for _, row in rank_df.iterrows():
        table.add_row(*[str(item) for item in row])  # 将每一行的数据转换为字符串并添加到表格

        
    console = Console()
    console.print(table)

def elo_rank(choice:str = "total",if_print_elo=False,if_human=False)->pd.DataFrame:
    """
    按照ranking来排名，
    + 如果选择total，那么输出名称，模型type，rating数值,  rank |type|model_name |elo rating| dataset1 |dataset2 |
    + 如果选择某数据集，那么输出名称,type ranking数值，以及各个subtask的排名/分数 rank |model_name|elo ranking|task1|task2
    返回pd
    """
    model_index = utils.load_json_file(model.MODEL_INDEX_PATH) #model的
    print_rank_df = None
    avaliable_dataset_names = dataset_wrapper.get_avaliable_dataset_names()
    if if_human:
        key_elo_rating = "human rating"
    else:
        key_elo_rating = "elo rating"
    if choice == "total":
        data = []
        for model_name, details in model_index.items():
            elo_ratings = details[key_elo_rating]
            total_elo_rating = elo_ratings["total"]["mu"] - 1 * elo_ratings["total"]["sigma"]
            win_rate = "{:.2f}".format(round(elo_ratings["total"]["win"] * 100, 2))
            row = {"model name": model_name,"elo rating":total_elo_rating,"type":details["type"],"win-rate":win_rate}
            for dataset_name in avaliable_dataset_names:
                row[dataset_name] = elo_ratings.get(dataset_name,{}).get("total", -NAN)
            data.append(row)
        data_sorted = sorted(data, key=lambda x: x["elo rating"], reverse=True)
        rank_df = pd.DataFrame(data_sorted) 
        rank_df = rank_df.replace(-NAN,np.nan)
        rank_df['rank'] = rank_df["elo rating"].rank(method='min', ascending=False) #如果有重复的，按照最小值来
        for dataset_name in avaliable_dataset_names:
            rank_df[f"{dataset_name}_rank"] = rank_df[dataset_name].rank(method='min', ascending=False)
        cols_name = ["rank", "model name", "type","elo rating","win-rate"] + [f"{dataset_name}" for dataset_name in avaliable_dataset_names]
        if not if_print_elo:
            cols = ["rank", "model name", "type","elo rating","win-rate"] + [f"{dataset_name}_rank" for dataset_name in avaliable_dataset_names]
            print_rank_df = rank_df[cols]
            print_rank_df.columns = cols_name
        else:
            print_rank_df = rank_df[cols_name]
        print_rank_df = print_rank_df.replace(np.nan, 0)

    else: 
        data = []
        if choice not in avaliable_dataset_names: #avaliable_dataset_names 是set
            raise ValueError(f"{choice} is not available. Please choose from {avaliable_dataset_names}.")
        dataset = dataset_wrapper.make(choice)
        dataset_tasks = list(dataset.get_tasks())
        for model_name, details in model_index.items():
            elo_ratings = details[key_elo_rating]
            dataset_elo_ratings = elo_ratings.get(choice, None)
            if dataset_elo_ratings:
                row = {"model name": model_name,"type":details["type"],choice: dataset_elo_ratings["total"]}
                row.update(dataset_elo_ratings)
            else:
                row = {"model name": model_name,"type":details["type"], choice: -NAN}
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
        cols_name = ["rank", "model name","type", choice] + [f"{task}" for task in dataset_tasks]
        if not if_print_elo:
            cols = ["rank", "model name","type", choice] + [f"{task}_rank" for task in dataset_tasks]
            print_rank_df = rank_df[cols]
            print_rank_df.columns = cols_name
        else:
            print_rank_df = rank_df[cols_name]
            
        print_rank_df = print_rank_df.replace(np.nan, 0)
    # 来个暴力的
    print_rank_df = print_rank_df.apply(lambda x: x.astype('int') if x.dtype == 'float64' else x)
    print_rank_df = print_rank_df.replace(0, "")
    print_rank_df = print_rank_df.replace("temp", "finetune")
    return print_rank_df


if __name__ in "__main__":
    rank(if_human=True)