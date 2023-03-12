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
	conda create -n env_pytorch python=3.9

conda-install:
# conda install -n env_pytorch --file requirements.txt -c pytorch
	conda install pytorch==1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
	conda install numpy jupyter

conda-activate:
	source /cygdrive/c/Users/bansh/anaconda3/etc/profile.d/conda.sh; conda activate env_pytorch

notebook:
	jupyter notebook ./src