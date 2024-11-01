import pickle
import gradio as gr
import pandas as pd
import plotly.express as px
from utils import utils
from pathlib import Path
from vla_eval.dataset.scal_dataset import prepare,systhetic,store

TASKS = list(prepare.get_tasks_map().keys())
dataset_task_map = prepare.get_tasks_map()

def plot_subtasks(dataset_name):
    
    subtasks = dataset_task_map.get(dataset_name, {})
    if not subtasks:
        return "Dataset not found", None
    df = pd.DataFrame({
        'Subtask': list(subtasks.keys()),
        'Quantity': list(subtasks.values())
    })
    fig = px.bar(
        df,
        x='Quantity',
        y='Subtask',
        title=f"Subtasks for {dataset_name}",
        labels={'Quantity': '数量', 'Subtask': '子任务'},
        text='Quantity'
    )
    fig.update_traces(textposition='outside')
    return fig, df

def preparing(dataset_name,task_name):
    source = prepare.get_source_data(dataset_name=dataset_name)
    example = prepare.get_example(dataset_name=dataset_name,task=task_name)
    text_data = ""
    image_data = ""
    if dataset_name in {"knowledge","reason"}:
        text_data = gr.Textbox(label= "Text Source",value=utils.load_txt_file(source[0]),visible=True, interactive=True)
        image_data = gr.Image(label="Image Source",visible=False)
    elif dataset_name in {"visual-advance","visual-basic"}:
        image_data = gr.Image( label="Image Source",value= str(source[0]),visible=True)
        text_data = gr.Textbox(label= "Text Source",value=utils.load_txt_file(source[0]),visible=False, interactive=False)
    example_b = pickle.dumps(example)
    return example_b,text_data,image_data,example["question"],example["answer"]

with gr.Blocks(title="MC OEQ scaling") as page:
    with gr.Tab("create"):
        dataset_dataframe = gr.Dataframe(label="测试数据集数据",value=prepare.get_dataset_num())
        with gr.Row():
            with gr.Column():
                task_dataframe = gr.Dataframe(label="子任务数据",value=plot_subtasks("reason")[1])
                with gr.Row():
                    setting_drop = gr.Dropdown(
                        choices=TASKS, 
                        label="Choosing Setting", 
                        value = "knowledge",
                        interactive=True
                    )
                    task_name = gr.Textbox(label= "Task Name",value="",visible=True, interactive=True)
                get_btn = gr.Button(
                    value="get", visible=True, interactive=True
                )
            with gr.Column():
                plot_component = gr.Plot(label="子任务数量图表",value=plot_subtasks("reason")[0])
        text_source = gr.Textbox(label= "Text Source",value="",visible=False, interactive=False)
        image_source = gr.Image(label="Image",visible=False)
        example =  gr.Textbox(label= "example",value="",visible=False)
        question_box = gr.Textbox(label= "question",value="",visible=True, interactive=False)
        answer_box = gr.Textbox(label= "answer",value="",visible=True, interactive=False)
        
        
        setting_drop.change(
            plot_subtasks,
            inputs=[setting_drop],
            outputs=[plot_component,task_dataframe]
        )
        get_btn.click(
            preparing,
            inputs=[setting_drop,task_name],
            outputs=[example,text_source,image_source,question_box,answer_box]
        )
    with gr.Tab("confirm"):
        pass
    
page.launch(share=True,auth=("admin", "craftjarvis"))