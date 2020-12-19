
%--------------------------------------------------------------------------------

% fig2e.m
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

function fig2e(sequence_type,seed)
% sequence_type=1 for ascending, sequence_type=2 for Fibonacci

% initialization
rng(1000000+seed);
dir           = '';
T             = 100000; % training sample size
T2            = 101000; % test sample size

prior_s       = 100;    % magnitude of regularization term
prior_s_      = 100;
prior_u       = 1;
prior_u_      = 1;
prior_x       = 1;

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
% compute true states and parameters

Nx      = 10;
x       = generate_true_hidden_states(input,label);
x_mean  = mean(x')';
x       = x  - x_mean * ones(1,T);
[A,B,Sigmas,Sigmax,Sigmao,Sigmaz] = calculate_true_parameters(s,x,prior_x);

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

% ICA
Nv                 = 10;         % encoding dimensionality
ica_rep            = 2000;       % number of iteration for ICA
ica_eta            = 0.01;       % learning rate for ICA
[ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data_file = [0:9; 1-max(G)./(sum(G)+0.001)];
if     (sequence_type == 1), csvwrite([dir 'output_err_a_predpca_'  num2str(seed) '.csv'],data_file)
elseif (sequence_type == 2), csvwrite([dir 'output_err_f_predpca_'  num2str(seed) '.csv'],data_file)
end

Omega = corr(x',ui');
for i = 1:Nv
  [~,j] = max(abs(Omega(i,:)));
  temp       = sign(Omega(i,j));
  Omega(i,:) = 0;
  Omega(:,j) = 0;
  Omega(i,j) = temp;
end
ui    = Omega * ui;
ui2   = Omega * ui2;
v2    = Omega * v2;

% winner-takes-all prediction
if (sequence_type == 1)
  img = digit_image(input2(:,61:70)+input_mean*ones(1,10));
  imwrite(img, [dir,'output_a_true_',num2str(seed),'_1_10.png'])
  img = digit_image(input2(:,100051:100060)+input_mean*ones(1,10));
  imwrite(img, [dir,'output_a_true_',num2str(seed),'_99991_100000.png'])
  
  [output,ui3,matA,matB,AIC1] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,1);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_a_predpca1_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_a_predpca1_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_a_predpca1_'  num2str(seed) '.csv'],[1:10; matB])
  
  [output,ui3,matA,matB,AIC2] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,2);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_a_predpca2_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_a_predpca2_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_a_predpca2_'  num2str(seed) '.csv'],[1:100; matB])
  
  [output,ui3,matA,matB,AIC3] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,3);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_a_predpca3_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_a_predpca3_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_a_predpca3_'  num2str(seed) '.csv'],[1:1000; matB])
  
  [output,ui3,matA,matB,AIC4] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,4);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_a_predpca4_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_a_predpca4_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_a_predpca4_'  num2str(seed) '.csv'],[1:10000; matB])
  
  csvwrite([dir 'output_AIC_a_predpca_'  num2str(seed) '.csv'],[1:4; AIC1 AIC2 AIC3 AIC4])
  
elseif (sequence_type == 2)
  img = digit_image(input2(:,61:70)+input_mean*ones(1,10));
  imwrite(img, [dir,'output_f_true_',num2str(seed),'_1_10.png'])
  img = digit_image(input2(:,100051:100060)+input_mean*ones(1,10));
  imwrite(img, [dir,'output_f_true_',num2str(seed),'_99991_100000.png'])
  
  [output,ui3,matA,matB,AIC1] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,1);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_f_predpca1_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_f_predpca1_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_f_predpca1_'  num2str(seed) '.csv'],[1:10; matB])
  
  [output,ui3,matA,matB,AIC2] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,2);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_f_predpca2_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_f_predpca2_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_f_predpca2_'  num2str(seed) '.csv'],[1:100; matB])
  
  [output,ui3,matA,matB,AIC3] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,3);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_f_predpca3_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_f_predpca3_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_f_predpca3_'  num2str(seed) '.csv'],[1:1000; matB])
  
  [output,ui3,matA,matB,AIC4] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,4);
  img = digit_image(output(:,61:70) * 1.2);
  imwrite(img, [dir,'output_f_predpca4_',num2str(seed),'_1_10.png'])
  img = digit_image(output(:,100051:100060) * 1.2);
  imwrite(img, [dir,'output_f_predpca4_',num2str(seed),'_99991_100000.png'])
  csvwrite([dir 'output_B_f_predpca4_'  num2str(seed) '.csv'],[1:10000; matB])
  
  csvwrite([dir 'output_AIC_f_predpca_'  num2str(seed) '.csv'],[1:4; AIC1 AIC2 AIC3 AIC4])
  return
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% State space model

fprintf(1,'State space model (Kalman filter)\n');
eta                         = 0.1;
num_rep                     = 100;
Ainit                       = randn(Ns,Nx) * sqrt(trace(A'*A)/Ns/Nx);
[BU,~,BV]                   = svd(randn(Nx,Nx));
Binit                       = BU*BV';
[qxp1,qxc1,qxp2,qxc2,qA,qB] = ssm_kalman_filter(s,s2,Ainit,Binit,Sigmao,Sigmaz,num_rep,eta,prior_x);
% True Sigmao and Sigmaz are used for state prediction

% ICA
[ui,ui2,Wica,v2,G] = postprocessing(qxc1,qxc2,label2,ica_rep,ica_eta);
figure_encoders(ui2,label2,1:T2/10);
drawnow

data_file = [0:9; 1-max(G)./(sum(G)+0.001)];
csvwrite([dir 'output_err_a_ssm_'  num2str(seed) '.csv'],data_file)

Omega = corr(x',ui');
for i = 1:Nv
  [~,j] = max(abs(Omega(i,:)));
  temp       = sign(Omega(i,j));
  Omega(i,:) = 0;
  Omega(:,j) = 0;
  Omega(i,j) = temp;
end
ui    = Omega * ui;
ui2   = Omega * ui2;
v2    = Omega * v2;

% winner-takes-all prediction
[output,ui3,matA,matB,AIC1] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,1);
img = digit_image(output(:,61:70) * 1.2);
imwrite(img, [dir,'output_a_ssm_',num2str(seed),'_1_10.png'])
img = digit_image(output(:,100051:100060) * 1.2);
imwrite(img, [dir,'output_a_ssm_',num2str(seed),'_99991_100000.png'])
csvwrite([dir 'output_B_a_ssm_'  num2str(seed) '.csv'],[1:10; matB])
fprintf(1,'----------------------------------------\n\n');

return

%--------------------------------------------------------------------------------
