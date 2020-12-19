
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
% 2020-5-31
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
% For evaluating the performance of autoencoder, please download the scripts from
% http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
% and train weight matrices with digit images involving monochrome
% inversion. Put an output file 'mnist_weights.mat' in the same directory
%

%--------------------------------------------------------------------------------

function fig2b(sequence_type,seed)
% sequence_type=1 for ascending, sequence_type=2 for Fibonacci

% initialization
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
% create input sequences

fprintf(1,'----------------------------------------\n');
fprintf(1,'    fig2b(%d,%d)    \n', sequence_type, seed);
fprintf(1,'----------------------------------------\n\n');

fprintf(1,'read digit files\n');
train_randomness = 1;
test_randomness  = 0;
train_signflip   = 1;
test_signflip    = 0;
[input,input2,~,label,label2,~] = create_digit_sequence(dir,sequence_type,T,T2,T2,train_randomness,test_randomness,train_signflip,test_signflip);
input_mean       = mean(input')';
input            = input  - input_mean * ones(1,T);
input2           = input2 - input_mean * ones(1,T2);

fprintf(1,'preprocessing (compress data using PCA)\n');
Ns               = 40;
[C,~,L]          = pca(input');
Wpca             = C(:,1:Ns)';
Lpca             = L;
s                = Wpca * input;
s2               = Wpca * input2;

fprintf(1,'calculate maximum likelihood estimator\n');
Kp                = 40;
[s_,s2_,se,se2,Q] = maximum_likelihood_estimator(s,s2,s,Kp,prior_s_);

fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

Nu      = 10;   % encoding dimensionality
Nv      = 10;   % encoding dimensionality
Nx      = 10;   % hidden state dimensionality
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
% [4] PCA of phi

fprintf(1,'[4] PCA of phi\n');
[C,~,L] = pca(s_');
Wpcaseq = C(:,1:Nu)';
Lpcaseq = L;
u       = Wpcaseq * s_;  % encoder (training)
u2      = Wpcaseq * s2_; % encoder (test)
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_pca_phi_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_pca_phi_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% normalize the variance of s by multiplying a andom orthogonal matrix

[C,~,L]     = pca(input');
[U,~,V]     = svd(randn(Ns,Ns));
Omega       = U * V';
Wpca        = Omega * C(:,1:Ns)';
s           = Wpca * input;
s2          = Wpca * input2;

%--------------------------------------------------------------------------------
% [5] TICA

fprintf(1,'[5] TICA\n');
if (sequence_type == 1)
  [u,u2]             = tica(s,s2);
elseif (sequence_type == 2)
  [u,u2]             = tica([s; s(:,[T,1:T-1])],[s2; s2(:,[T2,1:T2-1])]);
end
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_tica_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_tica_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [6] DMD

fprintf(1,'[6] DMD\n');
if (sequence_type == 1)
  [u,u2]             = dmd(s,s2);
elseif (sequence_type == 2)
  [u,u2]             = dmd([s; s(:,[T,1:T-1])],[s2; s2(:,[T2,1:T2-1])]);
end
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_dmd_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_dmd_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [7] DMD of phi

fprintf(1,'[7] DMD of phi\n');
s_  = zeros(Ns*Kp,T);    % basis functions (training)
s2_ = zeros(Ns*Kp,T2);   % basis functions (test)
for k = 1:Kp, s_(Ns*(k-1)+(1:Ns),:) = s(:,[T-(k-1):T,1:T-k]); end
for k = 1:Kp, s2_(Ns*(k-1)+(1:Ns),:) = s2(:,[T2-(k-1):T2,1:T2-k]); end

[u,u2]             = dmd(s_,s2_);
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_dmd_phi_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_dmd_phi_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [8] State space model

% compute true states and parameters
x       = generate_true_hidden_states(input,label);
x_mean  = mean(x')';
x       = x  - x_mean * ones(1,T);
[A,B,Sigmas,Sigmax,Sigmao,Sigmaz] = calculate_true_parameters(s,x,prior_x);
% True Sigmao and Sigmaz are used for state prediction

if (sequence_type == 1)
  fprintf(1,'[8] State space model (Kalman filter)\n');
  eta                         = 0.1;
  num_rep                     = 40;
  Ainit                       = randn(Ns,Nx) * sqrt(trace(A'*A)/Ns/Nx);
  [BU,~,BV]                   = svd(randn(Nx,Nx));
  Binit                       = BU*BV';
  [qxp1,qxc1,qxp2,qxc2,qA,qB] = ssm_kalman_filter(s,s2,Ainit,Binit,Sigmao,Sigmaz,num_rep,eta,prior_x);
elseif (sequence_type == 2)
  fprintf(1,'[8] State space model (Bayesian filter)\n');
  eta                         = 0.05;
  num_rep                     = 100;
  Ainit                       = randn(Ns,Nx) * sqrt(trace(A'*A)/Ns/Nx);
  Binit                       = randn(Nx,Nx*Nx) / sqrt(Nx*Nx);
  [qxp1,qxc1,qxp2,qxc2,qA,qB] = ssm_bayesian_filter(s,s2,Ainit,Binit,Sigmao,Sigmaz,num_rep,eta,prior_x);
end

% ICA
[ui,ui2,Wica,v2,G] = postprocessing(qxc1,qxc2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_ssm_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_ssm_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% [9] Hidden Markov model

fprintf(1,'[9] Hidden Markov model\n');

% preprocessing
Ns2         = 20;
[C,~,L]     = pca(input');
[U,~,V]     = svd(randn(Ns2,Ns2));
Omega       = U * V';
Wpca        = Omega * C(:,1:Ns2)';
s           = Wpca * input;
s2          = Wpca * input2;
[s,s2,Wica] = ica(s,s2,ica_rep,ica_eta);
for i = 1:Ns2
 skew    = skewness(s2(i,:));
 s(i,:)  = sign(skew) * s(i,:);
 s2(i,:) = sign(skew) * s2(i,:);
end

s    = 1./(1+exp(-1*(Wica*s)));  % training inputs
s2   = 1./(1+exp(-1*(Wica*s2))); % test inputs
qa   = (rand(Ns2*2,Nx)+1)*10;    % likelihood mapping qx(t)-->s(t)
if (sequence_type == 1)
  qb   = ones(Nx,Nx)*100;        % transiiton matrix qx(t-1)-->qx(t)
elseif (sequence_type == 2)
  qb   = ones(Nx,Nx*Nx)*100;     % transiiton matrix qx(t-1)*qx(t-2)-->qx(t)
end
rep  = 100;
l1   = 10;
l2   = 40;
eta1 = 0.01;
eta2 = 0.01;
amp1 = 10;
amp2 = 10;

[qx,qx2,qa,qA,qlnA,qb,qB,qlnB] = hmm(s,s2,qa,qb,rep,l1,l2,eta1,eta2,amp1,amp2,0);

figure_encoders(qx2,label2,1:T2/10);
drawnow

v2 = (ones(Nx,1)*max(qx2) == qx2) * 1;
G  = zeros(10,10);
for i=1:10, G(i,:) = sum(((ones(Nx,1)*(label2==i-1)) .* v2)'); end
fprintf(1,'categorization error = %f\n', mean(1-max(G)./(sum(G)+0.001)));

data = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'cat_err_a_hmm_'  num2str(seed) '.csv'],data)
elseif (sequence_type == 2), csvwrite([dir 'cat_err_f_hmm_'  num2str(seed) '.csv'],data)
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

