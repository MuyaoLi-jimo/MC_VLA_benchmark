"""
让待测试模型运行某个benchmark的数据，然后将运行的结果返回
输入：model和banchmark名称，输出答案，并存储
benchmark输出应该是什么呢？
TODO: 暂时按照task来生成吧，后期考虑其他形式（先做出来再说）,然后暂时不做并行，都串行
"""
from rich import print
from pathlib import Path,PosixPath
from utils import utils
from vla_eval.dataset.base import BaseDataset
from vla_eval.dataset import dataset_wrapper
from vla_eval.model import model
import copy



LOG_FOLD = Path(__file__).parent.parent.parent / "data" / "log"

def inference(database:BaseDataset,inference_model:model.Model,timestamp,test_jsonl_path:PosixPath=None):
    inference_model.launch()
    if type(test_jsonl_path) == type(None):  #如果没有提供路径，那么放到 data/log/ 文件夹下
        test_jsonl_path = LOG_FOLD / f"{timestamp}_{database.dataset_name}_{inference_model.model_name}.jsonl"
    test_jp = utils.JsonlProcessor(test_jsonl_path,if_backup=False) 
    test_jp.dump_restart()
    questions = database.get_questions(inference_model)
    
    for task,task_questions in questions.items():
        inputs = create_input(task_questions,database,model_name=inference_model.model_name)
        try:
            outputs = inference_model.inference(inputs,batch_size=40)
        except Exception as e:
            print(e)
            continue
        qas = []
        for task_question,output in zip(task_questions,outputs):
            assert task_question["id"]==output["id"]
            q = task_question["message"]
            if type(q["content"])==list: #存了脏乱的信息
                q["content"] = q["content"][0]
            q_a = {
                "id": task_question["id"],
                "q":q,
                "a":output["message"],
                "label":[database.dataset_name,task],
                "input_tokens":output["input_tokens"],
                "output_tokens":output["output_tokens"],
            }
            qas.append(q_a)
        test_jp.dump_lines(qas)
    return True
        

def create_input(task_questions,database:BaseDataset,model_name:str):
    """
    制造问题的输入，
    TODO:注意SYSTEM_PROMPT可以因任务而各异，这个后续再说 
    """
    input_questions = []
    system_prompt = database.get_inference_prompt()
    for task_question in task_questions:
        input_question = dict()
        input_question["id"] = task_question["id"]
        if model_name in model.OPENAI_MODEL:
            input_question["messages"] = copy.copy([{
                "role": "system",
                "content":system_prompt
                }])
            input_question["messages"].append(copy.copy(task_question["message"]))
        else:
            input_question["messages"] = [copy.copy(task_question["message"])]
            content = input_question["messages"][0]["content"]
            if isinstance(content,str):
                input_question["messages"][0]["content"] = "[SYSTEM_PROMPT]: \n" + system_prompt + "\n[USER_PRMPT]: \n" + content
            elif isinstance(content,list):
                for c in content:
                    if c["type"]=="text":
                        c["text"] = "[SYSTEM_PROMPT]:\n" + system_prompt+ "\n[USER_PRMPT]:\n"+ c["text"]
                input_question["messages"][0]["content"] = content
        input_questions.append(input_question)
    return input_questions

if __name__ in "__main__":
    #model2 = model.Model("gpt-4o")
    #inference(database_name="knowledge",inference_model=model2,timestamp=utils.generate_timestamp(),port=9002)
    model1 = model.Model("llama3-llava-next-8b-hf")
    model1.stop()