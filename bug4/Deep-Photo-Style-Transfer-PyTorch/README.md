# Deep-Photo-Style-Transfer-PyTorch
Project of NYU CSCI-GA 2271-001 Computer Vision Course

Task of style transfer in photographs. Details can be found in the [report](https://github.com/Billijk/Deep-Photo-Style-Transfer-PyTorch/blob/master/photo_style_transfer.pdf).

### Before running the code...

This code requires the following packages and files to run:
1. PyTorch 0.4.1, torchvision 0.2.1
2. Matlab Engine API ([installation](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html))
3. Pretrained semantic segmentation models ([download here](http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet101-upernet/))

### Running in different settings

* Branch _master_ is our combined method.
  
  Set `--masks dummy_mask` to run model without segmentation.
  
  Set `--sim 0` to run model without similarity loss.
  
  To run model with user provided segmentations, use `make_masks.py` to generate mask files from mask images, and set `--masks <your_mask_file>`. The following colors can be used in the image: blue (rgb: 0000ff), green (rgb: 00ff00), black (rgb: 000000), white (rgb: ffffff), red (rgb: ff0000), yellow (rgb: ffff00), grey	(rgb: 808080), lightblue (rgb: 00ffff), purple (rbg: ff00ff).

* Branch _gatys_baseline_ is the baseline neural style transfer model.

* Branch _regularization_ is the model with photorealism regularization term instead of post processing.

* Branch _hard_seg_ is the model using hard semantic segmentation.

### Results generated by our model

[View in Google Drive](https://drive.google.com/drive/folders/1wfKErgB33P0YjYz4odgt_EHNyRck7uQo?usp=sharing)
