# DHSNet-PyTorch

This is a [PyTorch](http://pytorch.org)(version0.4.0) implementation for DHSNet.
The papre can be found [here](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf).

## Requirements
* Python 3
* Pytorch 0.4

## Training & Validation
You can run it from the command line as such:
>python main.py --data_root "your_dataset_location" --val_rate "validation rate"

The validation results(F_measure,MAE) are stored in ./result.csv

Some functions are borrowed from [here](https://github.com/NVIDIA/flownet2-pytorch)
