# BENCHMARK
export PYTHONPATH="/scratch2/limuyao/workspace/VLA_benchmark:$PYTHONPATH"
## install
conda install numpy==1.26.3
## 注册dataset
+ 先用python vla_eval/dataset/base.py 更新index.json
+ 再手动在index.json和select_benchmark.py中加入类别
