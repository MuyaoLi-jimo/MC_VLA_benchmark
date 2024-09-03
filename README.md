# BENCHMARK
export PYTHONPATH="/scratch2/limuyao/workspace/VLA_benchmark:$PYTHONPATH"
## install
conda install numpy==1.26.3
## 注册dataset
+ 先用python vla_eval/dataset/base.py 更新index.json
+ 再手动在index.json和select_benchmark.py中加入类别
## 注册model
+ 使用python vla_eval/model/insert_model.py 记得加入命令行参数
## TODO
- [ ] 调试其他模型
- [ ] 增加人工比较
- [ ] 完成前端
- [ ] 扩大benchmark
- [ ] 增加客观benchmark

