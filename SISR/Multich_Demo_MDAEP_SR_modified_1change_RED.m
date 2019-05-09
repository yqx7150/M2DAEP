%% The Code is created based on the method described in the following paper:  
% M2DAEP - Single Image Super-Resolution
% 
% Input:
% degraded: Observed blurry and noisy input RGB image in range of [0, 255].
% up_scale: Up scaling factor.
% params: Set of parameters.
% params.net: The Network object from matCaffe.
%
% Optional parameters:
% map: Initial solution.
% params.sigma_net: The standard deviation of the network training noise. default: 11 & 25
% params.num_iter: Specifies number of iterations.
% params.gamma: Indicates the relative weight between the data term and the prior. default: 28.5
% params.mu: The momentum for SGD optimization. default: 0.9
% params.alpha the step length in SGD optimization. default: 0.1
%
% Outputs:
% map: Solution.
% Example

% add MatCaffe path
addpath /mnt/data/siavash/caffe/matlab;
addpath D:\CaffeProject\caffe-master\matlab;

% set to 0 if you want to run on CPU (very slow)
use_gpu = 1;

%% Super-resolution demo

% up_scale = 3;
% up_scale = 4;
up_scale = 2;
% Set 5
%gt = double(imread('.\Set5\baby_GT.bmp'));
 gt = double(imread('.\Set5\bird_GT.bmp'));
% gt = double(imread('.\Set5\butterfly_GT.bmp'));
% gt = double(imread('.\Set5\head_GT.bmp'));
 % gt = double(imread('.\Set5\woman_GT.bmp'));

w = size(gt,2); w = w - mod(w, up_scale*2);
h = size(gt,1); h = h - mod(h, up_scale*2);
gt = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...

degraded = imresize(gt, 1/up_scale, 'bicubic');

params1.net = multich_loadNet_qx3channel_diffSigma1_RED([size(gt,1),size(gt,2),6], use_gpu);
params1.gt = gt;

params2.net = multich_loadNet_qx3channel_diffSigma2_RED([size(gt,1),size(gt,2),6], use_gpu);
params2.gt = gt;

params1.sigma_net = 11;
params1.num_iter = 1500;
params2.sigma_net = 25;
params2.num_iter = 1500;

map_sr = multich_MDAEP_SR_modified_1change(degraded, up_scale, params1, params2);

figure;
subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(degraded/255); title('Low-resolution')
subplot(133);
imshow(map_sr/255); title('Supper-resolved')
