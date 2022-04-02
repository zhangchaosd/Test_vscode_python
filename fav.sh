export http_proxy=http://10.162.138.24:1082
export https_proxy=http://10.162.138.24:1082


邯郸校区
export http_proxy=http://10.223.198.110:1082
export https_proxy=http://10.223.198.110:1082


CMX1
export http_proxy=http://10.162.135.242:2080
export https_proxy=http://10.162.135.242:2080


tmux
exit
tmux ls
tmux a -t 0

srun --pty --partition=scavenger --gres=gpu:8 --mem=240G -n 32 --time=2-00:00:00 bash
srun --pty --partition=fvl --qos=high --gres=gpu:8 --mem=320G -n 32 --time=2-00:00:00 bash

pestat -G

df -h
cat x.txt | wc -l

conda create --name python35 python=3.5
conda activate python35
conda info --env
conda remove -n python35 --all


pip freeze > requirements.txt
pip install -r requirements.txt