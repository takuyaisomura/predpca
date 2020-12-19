
%--------------------------------------------------------------------------------

% fig2a.m
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
% 2020-5-30
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
% initialization

clear
sequence_type = 1;      % 1=ascending, 2=Fibonacci
dir           = '';
T             = 100000; % training sample size
T2            = 100000; % test sample size

prior_s       = 100;    % magnitude of regularization term
prior_s_      = 100;
prior_u       = 1;
prior_u_      = 1;

seed          = 0;
rng(1000000+seed);      % set seed for reproducibility

%--------------------------------------------------------------------------------
% create input sequences

fprintf(1,'read files\n')
train_randomness = 1;
test_randomness  = 0;
train_signflip   = 1;
test_signflip    = 0;
[input,input2,~,label,label2,~] = create_digit_sequence(dir,sequence_type,T,T2,T2,train_randomness,test_randomness,train_signflip,test_signflip);
input_mean       = mean(input')';
input            = input  - input_mean * ones(1,T);
input2           = input2 - input_mean * ones(1,T2);

fprintf(1,'compress data using PCA as preprocessing\n')
Ns               = 40;
[C,~,L]          = pca(input');
Wpca             = C(:,1:Ns)';
Lpca             = L;
s                = Wpca * input;
s2               = Wpca * input2;

%--------------------------------------------------------------------------------
% PredPCA

fprintf(1,'PredPCA\n')
Nu                = 10;          % encoding dimensionality
Kp                = 40;          % order of past observations used for prediction
fprintf(1,'- compute maximum likelihood estimator\n')
[s_,s2_,se,se2,Q] = maximum_likelihood_estimator(s,s2,s,Kp,prior_s_);
fprintf(1,'- post-hoc PCA using eigenvalue decomposition\n')
[C,~,L]           = pca(se');    % eigenvalue decomposition
Wppca             = C(:,1:Nu)';  % optimal synaptic weight matrix
Lppca             = L;           % eigenvalues
u                 = Wppca * se;  % encoder (training)
u2                = Wppca * se2; % encoder (test)

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
% output files for Fig 2a

if (sequence_type == 1)
 % mapping from encoding states to digit images
 A           = (input*ui') * (ui*ui'+eye(Nu)*prior_u)^(-1);
 ui_mean     = zeros(Nu,Nu);
 for i = 1:Nu, ui_mean(:,i) = mean(ui2(:,v2(i,:)==1)'); end
 mean_images = A * ui_mean + input_mean * ones(1,Nu);
 
 % save categorization results
 csvwrite('encoders.csv',[0:9, -1; ui2', label2'])
 
 fid = fopen('input.dat','w');
 fwrite(fid, (input2(:,1:1000) + input_mean * ones(1,1000)) * 255, 'uint8');
 fclose(fid);
 
 fid = fopen('estimated_input.dat','w');
 fwrite(fid, (Wpca' * se2(:,1:1000) + input_mean * ones(1,1000)) * 255 * 1.2, 'uint8');
 fclose(fid);
 
 fid = fopen('mean_images.dat','w');
 fwrite(fid, mean_images * 255 * 1.2, 'uint8');
 fclose(fid);
end

%--------------------------------------------------------------------------------
