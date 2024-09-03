"""
提前运行推理程序，获取model对某个evaluate set的结果，并存储
"""
import argparse
from rich import print
from utils import utils 
from vla_eval.evaluate import inference
from vla_eval.model import model as _model
from vla_eval.dataset import dataset_wrapper

MODEL_INFERENCE_FOLD = _model.MODEL_FOLD / "outcome"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="total")
    args = parser.parse_args()
    return args

def inference_model(model_name:str):
    """要求模型对某dataset进行推理，并把结果保存"""
    model = _model.Model(model_name=model_name)
    model_inference_path = MODEL_INFERENCE_FOLD / model_name
    model_inference_path.mkdir(parents=True,exist_ok=True)
    done_inferences = set(model.model_attr.get("done",{}))
    dataset_names = dataset_wrapper.get_avaliable_dataset_names()
    model_launched_flag = False
    for dataset_name in dataset_names:
        if dataset_name not in done_inferences:
            dataset = dataset_wrapper.make(dataset_name)
            if not model.model_attr["support vision"] and dataset.MODALITY=="visual":
                print("无法使用视觉，不进行推理")
                continue
            model.launch(device_num=1)
            model_launched_flag = True
            
            dataset_jsonl = model_inference_path / f"{dataset_name}.jsonl"
            print(dataset_name)
            inference.inference(database=dataset,inference_model=model,timestamp=utils.generate_timestamp(),test_jsonl_path=dataset_jsonl)
            done_inferences.add(dataset_name)
    model.model_attr["done"] = list(done_inferences)
    model.upload_model_attr()
    if model_launched_flag:
        model.stop()

if __name__=="__main__":
    args = parse_args()
    if args.model_name=="total":
        model_names = _model.get_runable_model_set()
        for model_name in model_names:
            print(f"[red]{model_name}")
            inference_model(model_name)
    else:
        inference_model(args.model_name)