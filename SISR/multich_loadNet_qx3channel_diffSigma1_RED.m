function net = multich_loadNet_qx3channel_diffSigma1_RED(img_size, use_gpu)
% Loads a Caffe 'net' object for a specific image dimensions
%
% Input:
% img_size: MAP Image size [Height, Width, Channel].
% use_gpu: GPU flag: use 1 if you use GPU, use 0 to run on CPU.
%
% Output:
% map: Caffe 'net' object.

%%
net_size = [6, img_size(2), img_size(1)];

caffe.reset_all();

if use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(0);
else
    caffe.set_mode_cpu();
end


net_model = './Multich_model/deploy_RED_net.prototxt';
net_weights = './Multich_model/RED_N11_color_6DCImageNet_1copyto2_initN25model/init_N25model_N11_6D_C_iter_800000.caffemodel';

FID_base = fopen(net_model, 'r');
Str_base = fread(FID_base, [1, inf]);
fclose(FID_base);
FID_net = fopen('./Multich_model/deploy_RED_resized.prototxt', 'w');
fprintf(FID_net, char(Str_base), net_size);
fclose(FID_net);
net_model = './Multich_model/deploy_RED_resized.prototxt';

net = caffe.Net(net_model, net_weights, 'test');
