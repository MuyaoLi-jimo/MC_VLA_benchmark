- 计划
+ 用Dataframe来展示结果--提供一个接口就好


def newround_response(setting_drop):
    dataset_name,A_name,B_name,validate_qa = get_validate_qa(setting_drop,human_model_ratings)
    gr.update(elem_id='model_a_from', visible=False)
    gr.update(elem_id='model_b_from', visible=False)
    hidden_answer = ""
    if "answer" in validate_qa:
        #hidden_answer = validate_qa["answer"]
        hidden_answer = gr.Textbox(value = validate_qa["answer"],visible=True, interactive=False)
    hidden_explain = ""
    if "explain" in validate_qa:
        #hidden_explain = validate_qa["explain"]
        hidden_explain = gr.Textbox(value = validate_qa["explain"],visible=True, interactive=False)
       
    hidden_image_path = ""
    if "image_path" in validate_qa:
        #hidden_image_path = validate_qa["image_path"]
        hidden_image_path = gr.Textbox(value = validate_qa["image_path"],visible=True, interactive=False)
    
    return dataset_name,validate_qa['question'], A_name, validate_qa['A'], "[MASK]", B_name, validate_qa['B'], "[MASK]",validate_qa["id"],hidden_answer,hidden_explain,hidden_image_path# visible
 