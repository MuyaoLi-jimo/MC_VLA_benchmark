# BENCHMARK
export PYTHONPATH="/scratch2/limuyao/workspace/VLA_benchmark:$PYTHONPATH"
## install
conda install numpy==1.26.3
## 注册dataset
+ 从excel插入index中
+ 先用python vla_eval/dataset/dataset_update.py 更新index.json(每一条的id，prompt，type)
+ 再手动在index.json和frontend中加入类别(type)
+ 最后在前端写入
## 注册model
+ 使用python vla_eval/model/insert_model.py 记得加入命令行参数
+ 如果出现错误，查看 /scratch2/limuyao/workspace/VLA_benchmark/data/model/log中对应模型的载入情况和终端的输出
+ 如果返回true, 查看 data/model/model.json 中的情况
## TODO
- [ ] 扩大benchmark
- [ ] 增加客观benchmark

