SHELL:=/bin/bash

all:install

install:check-dependency install-basics install-spconv
	python3 setup.py develop

install-basics:
    #install pytorch with CUDA-11.1 + cuDNN
	conda install -c pytorch -c nvidia pytorch torchvision torchaudio cudatoolkit=11.1 -y
	conda install -c conda-forge cudatoolkit-dev -y
	conda install -c conda-forge -c anaconda cudnn -y
    #install other requirements
	conda install numba pyyaml scikit-image -y
	conda install -c conda-forge tqdm kornia tensorboardX easydict -y

install-spconv:
    #NOTE: if the installation failed, please uncomment following lines.
    # pip3 install cmake --upgrade
    # sudo apt install libboost-all-dev -y
    # sudo apt install llvm-10-dev -y; sudo ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config
	rm -rf build/spconv && git clone https://github.com/traveller59/spconv.git build/spconv --recursive
	cd build/spconv; python3 setup.py bdist_wheel; cd ./dist; pip3 install *.whl --force	

check-dependency:
    # #check nvidia driver; `nvidia-smi | grep -oEi '[0-9]{3}\.[0-9]*'`
    # #check nvidia-cuda version
    # #check nvidia-cudnn version
    # #check conda
    # #check conda environment

prepare:
	python -m pcdet.datasets.kitti.carla_invs_dataset create_infos tools/cfgs/dataset_configs/carla_invs_dataset.yaml ${DATASET}

train:
	cd tools; python train.py --cfg_file cfgs/carla_invs_models/second.yaml --batch_size 1 --workers 0 --epochs 1 --pretrained_model ${CKPT}

