
%--------------------------------------------------------------------------------

% fig2b_tae.m
%
% This demo is included in
% Dimensionality reduction to maximize prediction generalization capability
% Takuya Isomura, Taro Toyoizumi
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-6-22
%
% Before run this script, please download MNIST dataset from
% http://yann.lecun.com/exdb/mnist/
% and expand
% train-images-idx3-ubyte
% train-labels-idx1-ubyte
% t10k-images-idx3-ubyte
% t10k-labels-idx1-ubyte
% in the same directory
%

%--------------------------------------------------------------------------------

function fig2b_tae(sequence_type,seed)
% sequence_type=1 for ascending, sequence_type=2 for Fibonacci

% initialization
tic
T             = 100000; % training sample size
T2            = 100000; % test sample size
dirname       = '';
rng(1000000+seed);

%--------------------------------------------------------------------------------
% create input sequences

fprintf(1,'read files\n')
train_randomness = 1;
test_randomness  = 0;
train_signflip   = 1;
test_signflip    = 0;
[input,input2,~,label,label2,~] = create_digit_sequence(dirname,sequence_type,T,T2,T2,train_randomness,test_randomness,train_signflip,test_signflip);
input_mean       = mean(input')';
input            = input  - input_mean * ones(1,T);
input2           = input2 - input_mean * ones(1,T2);

%--------------------------------------------------------------------------------
% TAE

% initialization
NL1     = 200;         % number of layer 1 neurons
NL2     = 100;         % number of layer 2 neurons
NL3     = 10;          % number of layer 3 neurons
num_rep = 2000;        % number of iteration
alpha   = 0.001;       % leaky parameter for the leaky rectified linear unit
p_do    = 0.0;         % dropout probability
beta1   = 0.9;         % parameter for Adam
beta2   = 0.99;        % parameter for Adam
eta     = 0.001;       % learning rate

% TAE
if (sequence_type == 1)
 [u,u2,sp,sp2,w1,w2,w3,w4,w5,w6] = tae(input, input2, input(:,[2:T,1]), input2(:,[2:T2,1]), num_rep, alpha, p_do, eta, beta1, beta2, NL1, NL2, NL3, dirname);
elseif (sequence_type == 2)
 [u,u2,sp,sp2,w1,w2,w3,w4,w5,w6] = tae([input; input(:,[T,1:T-1])], [input2; input2(:,[T2,1:T2-1])], [input(:,[2:T,1]); input], [input2(:,[2:T2,1]); input2], num_rep, alpha, p_do, eta, beta1, beta2, NL1, NL2, NL3, dirname);
end

%--------------------------------------------------------------------------------
% ICA

fprintf(1,'ICA\n')
Nv                 = 10;         % encoding dimensionality
ica_rep            = 2000;       % number of iteration for ICA
ica_eta            = 0.01;       % learning rate for ICA
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure()
figure_encoders(ui2,label2,1:T2/10);
drawnow

%--------------------------------------------------------------------------------

data_file = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dirname 'cat_err_a_tae_'  num2str(seed) '.csv'],data_file)
elseif (sequence_type == 2), csvwrite([dirname 'cat_err_f_tae_'  num2str(seed) '.csv'],data_file)
end

if     (sequence_type == 1)
 data_file = [0:1; mean(sum((input(:,[2:T,1])-sp)'.^2))/mean(sum(input'.^2)) mean(sum((input2(:,[2:T2,1])-sp2)'.^2))/mean(sum(input2'.^2))];
 csvwrite([dirname 'pred_err_a_tae_'  num2str(seed) '.csv'],data_file)
elseif (sequence_type == 2)
 data_file = [0:1; mean(sum(([input(:,[2:T,1]); input]-sp)'.^2))/mean(sum(input'.^2)) mean(sum(([input2(:,[2:T2,1]); input2]-sp2)'.^2))/mean(sum(input2'.^2))];
 csvwrite([dirname 'pred_err_f_tae_'  num2str(seed) '.csv'],data_file)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

