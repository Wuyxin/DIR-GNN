conda create -n dir python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl 
pip install torch_geometric==2.0.2 
pip install texttable ogb matplotlib networkx pickle