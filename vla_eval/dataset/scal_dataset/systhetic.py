"""
利用数据和例子，生成数据 

"""
from rich import print
from pathlib import Path
from utils import utils
from vla_eval.dataset.scal_dataset import prepare
from vla_eval.model.model import Model



SYSTEM_QA_PROMPT = """
Assume you are an expert in Minecraft and an adept question creator.
"""
SYSTEM_E_PROMPT = """ 
Assume you are an expert in Minecraft and a brilliant explainer.
"""
SYSTHETIC_QA_PROMPT = [
""" 
I would like you to craft a question and answer in Minecraft to test the """,# 后跟能力
" of the model, using the ", #后跟类型
""" provided above. I have already written an example for your reference. Please mimic the example and provide another one in the following format:
Q: [question]
A: [answer]
Remember, do not ask any question about Edition
Ask only one question
~~~~~~~~~~~~~~~
The Example:

""",
""" 
~~~~~~~~~~~~~~~
now write another qa:
"""
]
SYSTHETIC_E_PROMPT = """
there is a question and an answer below. Please explain it using the material provided above. 
Be precise, accurate and concise. 
Please follow this format:
E: [explanation]
~~~~~~~~~~~~~~~
"""

def create_system_message(system_prompt):
    message = {
        "role": "system",
        "content": system_prompt,
    }
    return message

def create_user_message(type:str,user_prompt,source_data=""):
    message = None
    if type=="text":
        message = {
            "role": "user",
            "content": user_prompt,
        }
    elif type=="image":
        source_data = Path(source_data)
        message = {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"<image>{user_prompt}\n"
            },
            {
                "type": "image_url",
                "image_url": { "url": f"data:image/{str(source_data.suffix)[1:]};base64,{utils.encode_image_to_base64(source_data)}"},
            },
            ],
        }
    else:
        raise AssertionError(f"type error:{type}")
    return message
    
def create_qa_prompt(source_data:str,example:dict):
    dataset_name = example["label"][0]
    task_name = example["label"][1]
    messages = [
        create_system_message(SYSTEM_QA_PROMPT)
    ]
    user_prompt = " "
    if dataset_name in {"knowledge","reason"}:
        user_prompt = source_data
        user_prompt += "\n~~~~~~~~~~~~~~~\n"
        user_prompt += SYSTHETIC_QA_PROMPT[0]
        if dataset_name=="knowledge":
            user_prompt += "world knowledge of Minecraft"
        elif dataset_name=="reason":
            user_prompt += "reason capability in " 
            user_prompt += task_name
        else:
            raise Exception("false dataset")
        user_prompt += SYSTHETIC_QA_PROMPT[1]
        user_prompt += "text"
        user_prompt += SYSTHETIC_QA_PROMPT[2]
        user_prompt += "Q: "
        user_prompt += example["question"]
        user_prompt += "A: "
        user_prompt += example["answer"]
        user_prompt += SYSTHETIC_QA_PROMPT[3]
        
        messages.append(create_user_message(type="text",user_prompt=user_prompt))
    elif dataset_name in {"visual-advance","visual-basic"}:
        user_prompt += SYSTHETIC_QA_PROMPT[0]
        if dataset_name == "visual-basic":
            user_prompt += "visual perception capability"
        elif dataset_name == "visual-advance":
            user_prompt += "visual reasoning capability in "
            user_prompt += task_name
        else:
            raise Exception("false dataset") 
        user_prompt += SYSTHETIC_QA_PROMPT[1]
        user_prompt += "text"
        user_prompt += SYSTHETIC_QA_PROMPT[2]
        user_prompt += "Q: "
        user_prompt += example["question"]
        user_prompt += "\nA: "
        user_prompt += example["answer"]
        user_prompt += SYSTHETIC_QA_PROMPT[3]
        messages.append(create_user_message(type="image",user_prompt=user_prompt,source_data=source_data))
    else:
        raise Exception("false dataset")
        
    return messages

def create_e_prompt(question:str,answer:str,dataset_name:str,source_data:str):
    messages = [
        create_system_message(SYSTEM_E_PROMPT)
    ]
    user_prompt = " "
    if dataset_name in {"knowledge","reason"}:
        user_prompt = source_data

    user_prompt += "\n~~~~~~~~~~~~~~~\n"
    user_prompt += SYSTHETIC_E_PROMPT
    user_prompt += "Q: "
    user_prompt += question
    user_prompt += "\nA: "
    user_prompt += answer
    user_prompt += "\n~~~~~~~~~~~~~~~\n"
    user_prompt += "Concise Explain: "
    if dataset_name in {"knowledge","reason"}:
        messages.append(create_user_message(type="text",user_prompt=user_prompt))
    elif dataset_name in {"visual-advance","visual-basic"}:
        messages.append(create_user_message(type="image",user_prompt=user_prompt,source_data=source_data))
    else:
        raise Exception("false dataset") 
    return messages

def systhetic_qa(source_data:str,example:dict):
    model = Model("gpt-4-turbo")
    datas = [{
        "id":utils.generate_uuid(),
        "messages":create_qa_prompt(source_data,example)
    }]
    model.launch()
    ret = model.inference(datas)
    return ret

def systhetic_explanation(question,answer,uuid,dataset_name:str,source:list,):
    model = Model("gpt-4-turbo")
    datas = [{
        "id":uuid,
        "messages":create_e_prompt(question,answer,dataset_name,source)
    }]
    model.launch()
    ret = model.inference(datas)
    return ret

def parse_qa(ret:dict):
    uuid = ret[0]["id"]
    output = ret[0]["message"]["content"]
    Q = output[output.find("Q:")+2:output.find("A:")]
    A = output[output.find("A:")+2:]

    return Q,A,uuid
    
def parse_e(ret:dict):
    output = ret[0]["message"]["content"]
    E = output[output.find("E:")+2:]
    return E 
    
if __name__ == "__main__":
    source = prepare.get_source_data(dataset_name="visual-basic")
    example = prepare.get_example(dataset_name="visual-basic")
    dataset_name = example["label"][0]
    if dataset_name in {"knowledge","reason"}:
        source_data = utils.load_txt_file(source[0])
    elif dataset_name in {"visual-advance","visual-basic"}:
        source_data = str(source[0])
    
    ret1 = systhetic_qa(source_data=source_data,example=example)
    Q,A,uuid = parse_qa(ret1)

    ret2 = systhetic_explanation(Q,A,uuid,dataset_name=dataset_name,source_data=source_data)
    E = parse_e(ret2)
    q_a = {
        "id":uuid,
        "question":Q,
        "answer":A,
        "explanation":E
    }
    print(q_a)
    
    