# BENCHMARK
export PYTHONPATH="/scratch2/limuyao/workspace/VLA_benchmark:$PYTHONPATH"
## install
conda install numpy==1.26.3
## 注册dataset
+ 先用python vla_eval/dataset/base.py 更新index.json
+ 再手动在index.json和select_benchmark.py中加入类别

## engineering
+ evaluating
1. 用户选择需要测评的模型a、task(最后完成：前端)
2. 后端进行评估
    + elo：
        1. 在已有模型中随机抽样出一个模型b
        2. 随机抽样一个batch的任务
        3. 同时进行推理
        4. 系统选择偏好（a or b or both or neither）
        5. 计算elo分数
        6. 多次迭代
    + absolute rating
        1. 待测评模型对一批数据进行inference（能否实现多线程？），将结果临时保存
        2. system同步对结果进行打分（访问结果），并保存结果（目前先用json保存）
        + ps: 可以记录推理速度、准确率(均值打分、 用户设定比例打分、各任务分别得分)
3. 前端显示最终rank



## content
### 需要测评的方向
知识(QA)、决策()、规划(object-centric planning, situated planning, reflection and re-planning)、感知(screen shot qa, inventory qa,space qa, caption)、推理(QA Reasoning)
#### QA text-only
1. Minecraft Knowledge
2. Reasoning 
#### 


### 测评形式
1. 选择(A\B\C\D) -- 使用准确率就可以了
2. 判断(yes\no)  -- 使用准确率就可以了
3. 回答() 
4. caption()
### 打分方式
1. absolute rating

    1. Accuracy
    2. precise\recall
    3. F1(balanced F-score ) 
    $$F_{1}=\frac{2}{\mathrm{recall}^{-1}+\mathrm{precision}^{-1}}=2\frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}=\frac{2\mathrm{tp}}{2\mathrm{tp}+\mathrm{fp}+\mathrm{fn}}.$$
    4. n-gram 
        + 基于词重叠率的方法：ROUGE/BLEU/NIST/METEOR/TER
    5. 基于相似度
        + Semantic Similarity 计算文本相似度，可以通过计算词嵌入的向量的余弦相似度来计算

2. relative rating
    +  ELO rating：(Chatbot Arena\
        1. 初始评分：每个选手开始时都有一个初始评分，通常为 1200 分（这个值可以根据不同的系统调整）。
        2. 比赛结果：当两个选手对战时，胜者的 Elo 评分会上升，而败者的 Elo 评分会下降。评分的变化量取决于双方的评分差距。
        3. 预期胜率：根据双方的评分，系统会计算出每个选手的预期胜率。预期胜率较高的选手，如果输了比赛，损失的评分会更多。
        4. 评分更新：比赛结束后，双方的评分会根据结果和预期胜率进行更新。通常通过公式来调整分数，公式中包含一个调整系数 K 值（决定了分数变化的幅度）。

