# M2DAEP
The Code is created based on the method described in the following paper:
Yuan Yuan, Jinjie Zhou, Zhuonan He, Shanshan Wang, Biao Xiong, Qiegen Liu, High-Dimensional Embedding Denoising Autoencoding Prior for Color Image Restoration, 2019 IEEE International Conference on Image Processing (ICIP), pp. 759–763, 2019.

## Motivation
In many non-local patch-based approaches, there usually consist of three steps: patch matching/clustering, sparsity or low-rank modeling as a regularizer, and patch weighted/aggregation modeling. As shown in Fig. 1(a), the patch matching procedure enables multi-patches with similar structural patterns to be found and grouped. Meanwhile, patch aggregation strategy applied on the clustered patches can achieve better restoration.These two procedures play the role of converting pixel domain to patch domain and returning the restored results in patch domain to pixel domain, respectively.
Inspired by the central idea existed in the patch-based models,we adopt a 6-dimensional and multi-models version of DAEP for the color IR. Visual illustration of employing high-dimensional prior at iterative procedure in M 2 DAEP is shown in Fig.1(b). By mapping 3-channel image to be 6-channel via copy operator, we train a network taking 6-channel as input. After higher-dimensional denoising procedure, we use average operator to attain the solution.
### Fig.1(a)
![repeat-M2AEP](https://github.com/yqx7150/M2DAEP/blob/master/SISR/Figs/12.png)
### Fig.1(b)
![repeat-M2AEP](https://github.com/yqx7150/M2DAEP/blob/master/SISR/Figs/11.png)

## Requirements and Dependencies
    MATLAB R2015b
    Cuda-8.0
    MatConvNet
    Caffe

## SISR
'./SISR/Multich_Demo_MDAEP_SR_modified_1change_RED.m' is the demo of M2DAEP for SISR.
## Deblurring
'./Deblurring/demo_6D_twosigma_extend.m' is the demo of M2DAEP for Deblurring.

### The flowchart of employing the learned M2EDAP to SISR application. 
![repeat-M2AEP](https://github.com/yqx7150/M2DAEP/blob/master/SISR/Figs/21.png)
