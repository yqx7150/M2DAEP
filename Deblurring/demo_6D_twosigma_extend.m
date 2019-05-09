
% add MatCaffe path
addpath /mnt/data/siavash/caffe/matlab;
addpath D:\CaffeProject\caffe-master\matlab;
% set to 0 if you want to run on CPU (very slow)
use_gpu = 1;


%% Deblurring demo

% load image and kernel
load('kernels.mat');

% gt = double(imread('./data/baby_GT.bmp'));
% gt = double(imread('./data/bird_GT.bmp'));
 gt = double(imread('./data/butterfly_GT.bmp'));
% gt = double(imread('./data/head_GT.bmp'));
% gt = double(imread('./data/woman_GT.bmp'));
% gt = double(imread('lenna.tif'));
% gt = double(imread('image_House256rgb.png'));
%gt = double(imread('Barbara256rgb.png'));
w = size(gt,2); w = w - mod(w, 2);
h = size(gt,1); h = h - mod(h, 2);
gt = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...

kernel = kernels{1};
sigma_d = 255 * .05;

pad = floor(size(kernel)/2);
gt_extend = padarray(gt, pad, 'replicate', 'both');

degraded = convn(gt_extend, rot90(kernel,2), 'valid');
noise = randn(size(degraded));
degraded = degraded + noise * sigma_d;

% load network for solver
params1.net = loadNet_6D_twosigma1([size(gt_extend,1),size(gt_extend,2),6], use_gpu);
params1.gt = gt;

params2.net = loadNet_6D_twosigma2([size(gt_extend,1),size(gt_extend,2),6], use_gpu);
params2.gt = gt;

params1.sigma_net = 11;
params1.num_iter = 1500;
params2.sigma_net = 25;
params2.num_iter = 1500;

% run DAEP
map_deblur_extend = DAEPDeblur_6D_twosigma(degraded, kernel, sigma_d, params1, params2);
map_deblur = map_deblur_extend(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);

figure;
subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(degraded/255); title('Blurry')
subplot(133);
imshow(map_deblur/255); title('Restored')
figure(222);imshow(map_deblur/255);
