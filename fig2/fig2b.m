
%--------------------------------------------------------------------------------

% fig2b.m
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
% 2020-3-5
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
% Put
% create_digit_sequence.m
% maximum_likelihood_estimator.m
% postprocessing.m
% ica.m
% figure_encoders.m
% autoencoder.m
% generate_true_hidden_states.m
% calculate_true_parameters.m
% kalman_filter.m
% ssm_kalman_filter.m
% in the same directory
%
% For evaluating the performance of autoencoder, please download the scripts from
% http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
% and train weight matrices with digit images involving monochrome
% inversion. Put an output file 'mnist_weights.mat' in the same directory

%--------------------------------------------------------------------------------

function fig2b(sequence_type,seed)

rng(1000000+seed);
dir           = '';
T             = 100000; % training sample size
T2            = 100000; % test sample size

prior_s       = 100;
prior_s_      = 100;
prior_u       = 1;
prior_u_      = 1;
prior_x       = 1;

%--------------------------------------------------------------------------------

fprintf(1,'----------------------------------------\n');
fprintf(1,'    fig2b(%d,%d)    \n', sequence_type, seed);
fprintf(1,'----------------------------------------\n\n');

fprintf(1,'read digit files\n');
test_randomness = 0;
sign_flip       = 1;
[input,input2,~,label,label2,~] = create_digit_sequence(dir,sequence_type,T,T2,T2,test_randomness,sign_flip);
input_mean      = mean(input')';
input           = input  - input_mean * ones(1,T);
input2          = input2 - input_mean * ones(1,T2);

fprintf(1,'preprocessing (compress data using PCA)\n');
Ns              = 40;
[C,~,L]         = pca(input');
Wpca            = C(:,1:Ns)';
Lpca            = L;
s               = Wpca * input;
s2              = Wpca * input2;

fprintf(1,'calculate maximum likelihood estimator\n');
Kp                = 40;
[s_,s2_,se,se2,Q] = maximum_likelihood_estimator(s,s2,Kp,prior_s_);

fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

Nu      = 10;   % encoding dimensionality
Nv      = 10;   % encoding dimensionality
ica_rep = 2000; % number of iteration for ICA
ica_eta = 0.01; % learning rate for ICA

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------
% [1] PredPCA

fprintf(1,'[1] PredPCA\n');
[C,~,L] = pca(se');
Wppca   = C(:,1:Nu)';
Lppca   = L;
u       = Wppca * se;  % encoder (training)
u2      = Wppca * se2; % encoder (test)
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_predpca_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_predpca_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [2] PCA

fprintf(1,'[2] PCA\n');
[C,~,L] = pca(s');
Wpca    = C(:,1:Nu)';
Lpca    = L;
u       = Wpca * s;  % encoder (training)
u2      = Wpca * s2; % encoder (test)
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_pca_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_pca_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [3] Autoencoder

fprintf(1,'[3] Autoencoder\n');
[u,u2] = autoencoder(input+input_mean*ones(1,T),input2+input_mean*ones(1,T2),'mnist_weights.mat');
u_mean = mean(u')';
u      = u  - u_mean * ones(1,T);
u2     = u2 - u_mean * ones(1,T2);
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_ae_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_ae_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [4] PCA(seq)

fprintf(1,'[4] PCA(seq)\n');
[C,~,L] = pca(s_');
Wpcaseq = C(:,1:Nu)';
Lpcaseq = L;
u       = Wpcaseq * s_;  % encoder (training)
u2      = Wpcaseq * s2_; % encoder (test)
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_pcaseq_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_pcaseq_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [5] State space model (Kalman filter)

fprintf(1,'[5] State space model (Kalman filter)\n');

Nx      = 10;

% compute true states and parameters
x       = generate_true_hidden_states(input,label);
x_mean  = mean(x')';
x       = x  - x_mean * ones(1,T);
[A,B,Sigmas,Sigmax,Sigmao,Sigmaz] = calculate_true_parameters(s,x,prior_x);
% True A and B are used only for determining the variance of initial values of A and B estimators
% True Sigmao and Sigmaz are used for state prediction

eta     = 0.1;
num_rep = 40;
[qxp1,qxp2,qA,qB]  = ssm_kalman_filter(s, s2, A, B, Sigmao, Sigmaz, num_rep, eta, prior_x);
[ui,ui2,Wica,v2,G] = postprocessing(qxp1,qxp2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_ssmkf_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_ssmkf_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
