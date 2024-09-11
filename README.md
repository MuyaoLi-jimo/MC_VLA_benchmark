# BENCHMARK
export PYTHONPATH="/scratch2/limuyao/workspace/VLA_benchmark:$PYTHONPATH"
## install
conda install numpy==1.26.3
## 注册dataset
+ 未完成：从csv插入model中
+ 先用python vla_eval/dataset/base.py 更新index.json
+ 再手动在index.json和select_benchmark.py中加入类别
+ 最后在前端写入
## 注册model
+ 使用python vla_eval/model/insert_model.py 记得加入命令行参数
+ 如果出现错误，查看 /scratch2/limuyao/workspace/VLA_benchmark/data/model/log中对应模型的载入情况和终端的输出
+ 如果返回true, 查看 data/model/model.json 中的情况
## TODO
- [ ] 增加人工比较
- [ ] 扩大benchmark
- [ ] 增加客观benchmark

