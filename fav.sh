export http_proxy=http://10.162.138.24:1082
export https_proxy=http://10.162.138.24:1082


邯郸校区
export http_proxy=http://10.223.198.110:1082
export https_proxy=http://10.223.198.110:1082


CMX1
export http_proxy=http://10.162.135.242:2080
export https_proxy=http://10.162.135.242:2080


tmux
tmux ls
tmux a -t 0

df -h
pestat -G

srun --pty --partition=scavenger --gres=gpu:8 --mem=240G -n 24 -w gpu02 --time=2-00:00:00 bash

srun --pty --partition=fvl --qos=high --gres=gpu:2 --mem=80G -n 8 -w gpu02 --time=2-00:00:00 bash
srun --pty --partition=fvl --qos=medium --gres=gpu:1 --mem=40G -n 4 -w gpu02 --time=2-00:00:00 bash
srun --pty --partition=fvl --qos=medium --gres=gpu:1 --mem=40G -n 4 -w gpu02 --time=2-00:00:00 bash
srun --pty --partition=fvl --qos=medium --gres=gpu:1 --mem=40G -n 4 -w gpu13 --time=2-00:00:00 bash


cat x.txt | wc -l

conda create --name python35 python=3.5
conda activate python35
conda info --env
conda remove -n python35 --all


pip freeze > requirements.txt
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

conda install -c conda-forge tmux


lsof +D /share/home/zhangchao/.vscode-server
kill -9 1677
rm -rf .vscode-server


  DATASET_NAME: FFDF
  ROOT_DIR: /share_io02_ssd/ouyang/FF++/face/
  TRAIN_INFO_TXT: /share_io02_ssd/ouyang/FF++/splits/train_face_c23.txt
  VAL_INFO_TXT: /share_io02_ssd/ouyang/FF++/splits/test_face_c23.txt
  TEST_INFO_TXT: /share_io02_ssd/ouyang/FF++/splits/test_face_c23.txt

  DATASET_NAME: CelebDF
  ROOT_DIR: /share_io02_ssd/ouyang/celeb-df-v2/
  TRAIN_INFO_TXT: /share_io02_ssd/ouyang/celeb-df-v2/splits/train.txt
  VAL_INFO_TXT: /share_io02_ssd/ouyang/celeb-df-v2/splits/eval.txt
  TEST_INFO_TXT: /share_io02_ssd/ouyang/celeb-df-v2/splits/eval.txt

  DATASET_NAME: CelebDF
  ROOT_DIR: /share_io02_ssd/ouyang/celeb-df-v2/
  TRAIN_INFO_TXT: /share/home/zhangchao/testuse/cel_train.txt
  VAL_INFO_TXT: /share/home/zhangchao/testuse/cel_test.txt
  TEST_INFO_TXT: /share/home/zhangchao/testuse/cel_test.txt


  pip install timm fvcore albumentations kornia simplejson tensorboard

  https://github.com/wangjk666/D5014-2.git

gdown https://drive.google.com/uc?id=1iHYJnvCmakjUs0QJH53ZgwwaWvudOp34



./deepfake_backend temp@121.41.102.235:/home/temp/test2


ssh temp@121.41.102.235
temp@bigvid

ssh -i key_opentai opentai@10.176.53.107
opentai@fvl


scp -r ~/Documents/GitHub/aigov_backend/web/deepfake_backend/static/First-order-motion temp@121.41.102.235:/home/temp/test2
scp -r ~/Documents/GitHub/aigov_backend/web/deepfake_backend/static/original temp@121.41.102.235:/home/temp/test2
scp -r /Users/zhangchao/Desktop/dzl/Det temp@121.41.102.235:/home/temp/test2

scp -r -i ~/key_opentai ./First-order-motion opentai@10.176.53.107:/home/opentai/aigov_backend/web/deepfake_backend/static
scp -r -i ~/key_opentai ./original opentai@10.176.53.107:/home/opentai/aigov_backend/web/deepfake_backend/static
scp -r -i ~/key_opentai ./Det opentai@10.176.53.107:/home/opentai/aigov_backend/web/deepfake_backend/static

sudo docker stop deepfake_backend
opentai@fvl
sudo docker rm deepfake_backend
sudo docker rmi deepfake_backend
sudo docker build -t deepfake_backend .
sudo docker run -v /home/opentai/aigov_backend/web/deepfake_backend:/app/ -d -p 8094:8000 --restart unless-stopped --name deepfake_backend deepfake_backend