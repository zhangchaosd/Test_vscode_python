# Test_vscode_python
For learning


launch.json 中 args 为添加 debug 参数
cwd 和 env 为解决引用父文件夹包的问题
first.py 中的
```
import sys
sys.path.append("..")
```
是解决运行时的此问题。second.py 中添加无影响


```
pip install -r requirements.txt
```

`os.pathsep` 是分隔符';'
```
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
```

创建虚拟环境
```
python -m venv myvenv
```