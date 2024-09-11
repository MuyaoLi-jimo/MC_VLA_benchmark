import gradio as gr
from pathlib import Path
from vla_eval.model.model import get_avaliable_model_set,get_model_ratings
from vla_eval.model.insert_model import insert_model_wrapper   # 确保导入你需要的函数
from vla_eval.model.rank_model import elo_rank
from vla_eval.evaluate.human_evaluate import get_validate_qa,cal_human_elo
from vla_eval.evaluate.elo_evaluate import history_elo_evaluate
from functools import partial
from utils import utils

DATA_FOLD = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_FOLD / "model" / "model.json"
HISTORY_PATH = DATA_FOLD / "history.jsonl"
HUMAN_HISTORY_PATH = DATA_FOLD / "human_history.jsonl"

human_model_ratings = get_model_ratings(if_human=True)
human_history_jp = utils.JsonlProcessor(HUMAN_HISTORY_PATH)

def update_leaderboard(choice, if_print_elo,if_human):
    # 根据选择的选项和是否打印Elo等级来获取更新的数据
    rank_elo_pd = elo_rank(choice=choice, if_print_elo=if_print_elo,if_human=if_human)
    return rank_elo_pd
   
init_dataset_name,init_A_name,init_B_name,init_validate_qa = get_validate_qa("random",human_model_ratings)
   
def validate_qa_wrapper(hidden_idx,model_A_response,model_B_response,hidden_image_path,question,hidden_answer,hidden_explain):
    validate_qa = {
        "id":hidden_idx,
        "A":model_A_response,
        "B":model_B_response,
        "question":question,
    }
    if hidden_image_path:
        validate_qa["image_path"] = hidden_image_path
    if hidden_answer:
        validate_qa["answer"] = hidden_answer
    if hidden_explain:
        validate_qa["explanation"] = hidden_explain
    return validate_qa
   
def cal_vote(score,human_model_ratings,human_history_jp,dataset_name,hidden_idx,model_A_response,model_B_response,hidden_image_path,question,hidden_answer,hidden_explain,model_A_name,model_B_name):
    validate_qa = validate_qa_wrapper(hidden_idx,model_A_response,model_B_response,hidden_image_path,question,hidden_answer,hidden_explain)
    model_A_name,model_B_name,human_model_ratings,human_history_jp = cal_human_elo(score,dataset_name,validate_qa,model_A_name,model_B_name,human_model_ratings,human_history_jp)
    return model_A_name,model_B_name

cal_leftvote = partial(cal_vote, 3,human_model_ratings,human_history_jp)
cal_rightvote = partial(cal_vote, 1,human_model_ratings,human_history_jp)
cal_tievote = partial(cal_vote, 2,human_model_ratings,human_history_jp)
cal_badvote = partial(cal_vote, 0,human_model_ratings,human_history_jp)
   
def newround_response(setting_drop):
    dataset_name,A_name,B_name,validate_qa = get_validate_qa(setting_drop,human_model_ratings)
    gr.update(elem_id='model_a_from', visible=False)
    gr.update(elem_id='model_b_from', visible=False)
    
    if "answer" in validate_qa:
        #hidden_answer = validate_qa["answer"]
        hidden_answer = gr.Textbox(value = validate_qa["answer"],visible=True, interactive=False)
    else:
        hidden_answer = gr.Textbox(value = "",visible=False, interactive=False)
    
    if "explanation" in validate_qa:
        #hidden_explain = validate_qa["explain"]
        hidden_explain = gr.Textbox(value = validate_qa["explanation"],visible=True, interactive=False)
    else:
        hidden_explain = gr.Textbox(value = "",visible=False, interactive=False)
    
    if "image_path" in validate_qa:
        hidden_image_path = gr.Textbox(value = validate_qa["image_path"],visible=False, interactive=False)
        hidden_image = gr.Image(value = validate_qa["image_path"],visible=True, interactive=False)
    else:
        hidden_image_path =  gr.Textbox(value = "",visible=False, interactive=False)
        hidden_image = gr.Image(value = "/scratch2/limuyao/workspace/VLA_benchmark/data/dataset/image/10.png",visible=False, interactive=False)
    
    return dataset_name,validate_qa['question'], A_name, validate_qa['A'], "[MASK]", B_name, validate_qa['B'], "[MASK]",validate_qa["id"],hidden_answer,hidden_explain,hidden_image_path,hidden_image# visible
 

notice_markdown = """
# ⚔️  Chatbot Arena ⚔️ : Benchmarking LLMs in the Wild

## 📜 Rules
- Refresh to obtain question and its corresponding answers from two anonymous models.
- Vote for the better answer. And then click "New Round" to get a new question.
- If both answers are bad, vote for "Both are bad".
- If you want to skip, click "Skip".

## 📊 Principle
You can evaluate the performance of the model from the following aspects:
1. **Relevance**: Does it answer the question accurately?
2. **Accuracy**: Is it accurate? For example, a crafting table is made by combining 4 wooden planks, not 4 logs; a diamond axe requires 3 diamonds and 2 sticks to craft, not 3 sticks and 2 diamonds.
3. **Completeness**: Is it complete? For example, crafting a wooden pickaxe from logs requires first crafting wooden planks and then crafting sticks before finally being able to craft the pickaxe. The intermediate steps cannot be ignored.
4. **Readability**: Is it coherent?
5. **Executability**: Considering the characteristics of the game, is it executable?

## 👇 Vote now!

"""


with gr.Blocks(title="MC Arena") as page:
    with gr.Tab("Elo Leaderboard"):
        with gr.Column():
            with gr.Row():
                use_human_evaluate = gr.Checkbox(label="Human Evaluating", value=False)
                update_leaderboard_button = gr.Button("Update Leaderboard")
            choice = gr.Radio(["total", "knowledge", "reason", "visual"], label="benchmark", value="total", info="Which benchmark do you choose?")
            if_print_elo = gr.Checkbox(label="Print Elo Rating", value=False)
            output = gr.DataFrame(elo_rank(choice="total", if_print_elo=False))
            
            # 更新函数绑定到选项变化
            choice.change(update_leaderboard, inputs=[choice, if_print_elo,use_human_evaluate], outputs=output)
            if_print_elo.change(update_leaderboard, inputs=[choice, if_print_elo,use_human_evaluate], outputs=output)
            use_human_evaluate.change(update_leaderboard, inputs=[choice, if_print_elo,use_human_evaluate], outputs=output)
        update_leaderboard_button.click(
            fn=history_elo_evaluate,
            inputs=[choice, if_print_elo,use_human_evaluate],
            outputs=output
        )
    with gr.Tab("Human Elo Rating System"):
        # TODO:根据选择调整可见
        gr.Markdown(notice_markdown)
        with gr.Row():
            setting_drop = gr.Dropdown(
                ["random", "knowledge","reason","visual"], 
                label="evaluation setting", 
                value = "knowledge",
                interactive=True
            )
            hidden_dataset_name = gr.Textbox(value=init_dataset_name,visible=False, interactive=False)
            
        with gr.Row():
            instruction_box = gr.Textbox(lines=5, label="Question", value = init_validate_qa['question'], interactive=True)
        
        with gr.Row():
            with gr.Row():
                hidden_image = gr.Image(label="Image",visible=False, interactive=False)
                hidden_answer = gr.Textbox(label="Answer",value = init_validate_qa.get('answer',""),visible=False, interactive=False)
                hidden_explain = gr.Textbox(label="Explain",value = init_validate_qa.get('explanation',""),visible=False, interactive=False)
                hidden_image_path = gr.Textbox(label="Image Path",value = init_validate_qa.get('image_path',""),visible=False, interactive=False)
        
        with gr.Row():
            hidden_idx = gr.Textbox(value = init_validate_qa['id'],visible=False, interactive=False) #永远hidden

            with gr.Column():
                model_A_name = gr.Textbox(elem_id="model_a_from", label="Model A From:", value="[MASK]", visible=True, interactive=False)
                model_A_response = gr.Textbox(lines=25, label="Model A Response:", value = init_validate_qa['A'], visible = True, interactive=False)
                hidden_A_name = gr.Textbox(elem_id="model_a_from", label="Model A From:", value=init_A_name,visible=False, interactive=False)
                
                # model_A_name = gr.Markdown(elem_id="model_a_from", label="A From:", value=init_data['model_a'], visible=True)
            with gr.Column():
                model_B_name = gr.Textbox(elem_id="model_a_from", label="Model B From:", value="[MASK]", visible=True, interactive=False)
                model_B_response = gr.Textbox(lines=25, label="Model B Response:", value = init_validate_qa['B'], visible = True, interactive=False)
                hidden_B_name = gr.Textbox(elem_id="model_b_from", label="Model B From:", value=init_B_name,visible=False, interactive=False)
                
                
        setting_drop.change(newround_response,
                            inputs=[setting_drop],
                            outputs=[hidden_dataset_name,instruction_box, hidden_A_name, model_A_response, model_A_name, hidden_B_name, model_B_response, model_B_name,hidden_idx,hidden_answer,hidden_explain,hidden_image_path,hidden_image])
        
        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=True, interactive=True
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=True, interactive=True
            )
            tie_btn = gr.Button(
                value="🤝  Tie",         visible=True, interactive=True
            )
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=True, interactive=True
            )

        with gr.Row():
            skip_btn = gr.Button(
                value="👋  Skip", visible=True, interactive=True
            )
            new_round_btn = gr.Button(
                value="🔄  New Round", visible=True, interactive=True
            )
        leftvote_btn.click(
            fn = cal_leftvote,
            inputs = [hidden_dataset_name,hidden_idx,model_A_response,model_B_response,hidden_image_path,instruction_box,hidden_answer,hidden_explain,hidden_A_name,hidden_B_name],
            outputs = [model_A_name, model_B_name]
        )
        rightvote_btn.click(
            fn = cal_rightvote,
            inputs = [hidden_dataset_name,hidden_idx,model_A_response,model_B_response,hidden_image_path,instruction_box,hidden_answer,hidden_explain,hidden_A_name,hidden_B_name],
            outputs = [model_A_name, model_B_name]
        )
        tie_btn.click(
            fn = cal_tievote,
            inputs = [hidden_dataset_name,hidden_idx,model_A_response,model_B_response,hidden_image_path,instruction_box,hidden_answer,hidden_explain,hidden_A_name,hidden_B_name],
            outputs = [model_A_name, model_B_name]
        )
        bothbad_btn.click(
            fn = cal_badvote,
            inputs = [hidden_dataset_name,hidden_idx,model_A_response,model_B_response,hidden_image_path,instruction_box,hidden_answer,hidden_explain,hidden_A_name,hidden_B_name],
            outputs = [model_A_name, model_B_name]
        )
        ####################
        new_round_btn.click(
            fn = newround_response,
            inputs = [setting_drop],
            outputs=[hidden_dataset_name,instruction_box, hidden_A_name, model_A_response, model_A_name, hidden_B_name, model_B_response, model_B_name,hidden_idx,hidden_answer,hidden_explain,hidden_image_path,hidden_image]
        )
        skip_btn.click(
            fn = newround_response,
            inputs=[setting_drop],
            outputs=[hidden_dataset_name,instruction_box, hidden_A_name, model_A_response, model_A_name, hidden_B_name, model_B_response, model_B_name,hidden_idx,hidden_answer,hidden_explain,hidden_image_path,hidden_image]
        )
    with gr.Tab("Insert Model"):
        with gr.Row():
            with gr.Column():
                model_name = gr.Text(label="Model name")
                model_path = gr.Text(label="Model path", info="If it is in the default folder, you don't need to write this")
                model_base = gr.Text(label="Model base", info="Only need to write this if it’s a llama-1.6 based model")
                model_type = gr.Radio(["finetune", "pretrained", "commercial", "temp"], label="Model type")
                support_vision = gr.Checkbox(label="Support vision", value=True)
                chat_template = gr.Text(label="Additional chat template")
                insertb = gr.Button("Insert")
            with gr.Column():
                
                gr.Examples(examples=[
                    ["llama3-llava-next-8b-hf", "","llava_next", "pretrained", True, ""],
                    ["llava-v1.6-vicuna-13b-hf","","llava_next","pretrained",True,"/scratch2/limuyao/workspace/VLA_benchmark/data/model/template/template_llava.jinja"],
                    ["gpt-4o", "openai/gpt-4o","", "commercial", True, ""],
                    ["10003","","llava_next","temp",True,""],
                    ],
                    inputs=[model_name, model_path, model_base,model_type,support_vision,chat_template],)
                status, message = gr.Textbox(label="Status Message"), gr.Textbox(label="Error Message")
            
        insertb.click(
            fn=insert_model_wrapper,
            inputs=[model_name, model_path, model_type, support_vision, model_base, chat_template],
            outputs=[status, message]
        )

page.launch(share=True)
            