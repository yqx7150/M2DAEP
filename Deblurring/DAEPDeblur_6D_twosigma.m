function map = DAEPDeblur_6D_twosigma(degraded, kernel, sigma_d, params, params2)
% Input:
% degraded: Observed degraded RGB input image in range of [0, 255].
% kernel: Blur kernel (internally flipped for convolution).
% sigma_d: Noise standard deviation.
% params: Set of parameters.
% params.net: The Network object loaded from MatCaffe.
%
% Optional parameters:
% params.sigma_net: The standard deviation of the network training noise. default: 25
% params.num_iter: Specifies number of iterations.
% params.gamma: Indicates the relative weight between the data term and the prior. default: 6.875
% params.mu: The momentum for SGD optimization. default: 0.9
% params.alpha the step length in SGD optimization. default: 0.1
%
% Outputs:
% map: Solution.


if ~any(strcmp('net',fieldnames(params)))
    error('Need a DAE network in params.net!');
end

if ~any(strcmp('sigma_net',fieldnames(params)))
    params.sigma_net = 25;
end

if ~any(strcmp('num_iter',fieldnames(params)))
    params.num_iter = 300;
end

if ~any(strcmp('gamma',fieldnames(params)))
    params.gamma = 0.875;
end

if ~any(strcmp('mu',fieldnames(params)))
    params.mu = .9;
end

if ~any(strcmp('alpha',fieldnames(params)))
    params.alpha = .1;
end


disp(params)

params.gamma = params.gamma * 4;
pad = floor(size(kernel)/2);
map = padarray(degraded, pad, 'replicate', 'both');

sigma_eta = sqrt(2) * params2.sigma_net;
relative_weight = params.gamma/(sigma_eta^2)/(params.gamma/(sigma_eta^2) + 1/(sigma_d^2));



step = zeros(size(map));

if any(strcmp('gt',fieldnames(params)))
    map_center = map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
end

for iter = 1:params.num_iter
    if any(strcmp('gt',fieldnames(params)))
        disp(['Running iteration: ' num2str(iter)]);
        tic();
    end
    prior_err_sum = zeros(size(map));
    repeat_num = 1;
    
    for iiii = 1:repeat_num
        input = repmat(map,[1,1,2]); % Switch channels for caffe
        noise = randn(size(input)) * params.sigma_net;
        rec = params.net.forward({input + noise});
        prior_err6 = input - rec{1};
        rec = params.net.backward({-prior_err6});
        prior_err6 = prior_err6 + rec{1};
        tmp = prior_err6;
        prior_err6 = prior_err6 * 0;
        prior_err6(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);
        output_1 = prior_err6(:,:,1:3);
        output_2 = prior_err6(:,:,4:6);
        

        prior_err = 0.5*(output_2 + output_1);
        prior_err_sum = prior_err_sum + prior_err;
    end
    
    for iiii = 1:repeat_num
        input = repmat(map,[1,1,2]); % Switch channels for caffe
        noise = randn(size(input)) * params2.sigma_net;
        rec = params2.net.forward({input + noise});
        prior_err6 = input - rec{1};
        rec = params2.net.backward({-prior_err6});
        prior_err6 = prior_err6 + rec{1};
        tmp = prior_err6;
        prior_err6 = prior_err6 * 0;
        prior_err6(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);
        output_1 = prior_err6(:,:,1:3);
        output_2 = prior_err6(:,:,4:6);
        prior_err = 0.5*(output_2 + output_1);
        prior_err_sum = prior_err_sum + prior_err;
    end
    
    prior_err = prior_err_sum/repeat_num/2;     
    
    % compute data gradient
    map_conv = convn(map,rot90(kernel,2),'valid');
    data_err = convn(map_conv-degraded,kernel,'full');
    
    % sum the gradients
    err = relative_weight * prior_err + (1 - relative_weight)*data_err;    
    
    % update
    step = params.mu * step - params.alpha * err;
    map = map + step;
    map = min(255,max(0,map));
    
    if any(strcmp('gt',fieldnames(params)))
        map_center = map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
        psnr = csnr(params.gt, map_center, 0,0);
        disp(['PSNR is: ' num2str(psnr) ', iteration finished in ' num2str(toc()) ' seconds']);
        figure(666);imshow(map_center/255); title('Deblurred');
    end
    
end
