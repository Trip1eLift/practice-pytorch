start:
	docker-compose up --build
	
start-bg:
	docker-compose up --build --detach

down:
	docker-compose down -v

cleanse:
	docker system prune -a && docker volume prune

conda-env:
# install conda environment in C:\Users\bansh\anaconda3\envs\env_pytorch
	conda create -n pytorch python=3.8

conda-install:
# conda install -n pytorch --file requirements.txt -c pytorch
	conda install -n pytorch pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
	conda install -n pytorch matplotlib tensorboard scikit-learn jupyter

# conda-pip-install: conda-activate
# 	pip install scikit-learn jupyter

tensorboard:
	tensorboard --logdir=src/runs


conda-activate:
	source /cygdrive/c/Users/bansh/anaconda3/etc/profile.d/conda.sh; conda activate pytorch

notebook:
	jupyter notebook ./src