
%--------------------------------------------------------------------------------

% predpca_aloi_main.m
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
% Before run this script, please download ALOI dataset from
% http://aloi.science.uva.nl

%--------------------------------------------------------------------------------
% initialization

clear
Timg    = 72000;
Norig   = 36 * 36 * 3;
Npca1   = 1000;
Npca2   = 1000;
Npca3   = 1000;

%--------------------------------------------------------------------------------

tic
fprintf(1,'----------------------------------------\n');
fprintf(1,'PredPCA of 3D rotating object images\n');
fprintf(1,'----------------------------------------\n\n');
fprintf(1,'preprocessing\n');

dir     = '';

% read files
load([dir,'data_m1.mat'])
load([dir,'data_C1.mat'])
load([dir,'data_L1.mat'])
load([dir,'data_C2.mat'])
load([dir,'data_L2.mat'])
load([dir,'data_C3.mat'])
load([dir,'data_L3.mat'])
load([dir,'data_3.mat'])

T         = 57600;    % number of training data
T2        = 14400;    % number of test data
Ns        = 300;      % number of input dimensions
Nu        = 150;      % number of encoder dimensions

Kf        = 5;        % order of predicting points
Kp        = 19;       % order of reference points
Kp2       = 37;       % order of reference points
WithNoise = 0;        % presence of noise

%--------------------------------------------------------------------------------
% create random object sequences for training and test

seed  = 0;
rng(1000000+seed);

order    = randperm(1000);

Kf_list  = [6,12,18,24,30];
Kp_list  = 0:2:36;
Kp2_list = 0:1:36;

time  = cell(Kp,1);
time2 = cell(Kp2,1);
timet = cell(Kf,1);
for k = 1:Kp, time{k,1}  = zeros(T+T2,1); end
for k = 1:Kf, timet{k,1} = zeros(T+T2,1); end
for i = 1:((T+T2)/72)
 j = order(i)*72-72;
 t = i*72-72;
 for k = 1:Kf,  timet{k,1}(t+1:t+72) = [j+1+Kf_list(k):j+72,    j+1:j+Kf_list(k)    ]; end
 for k = 1:Kp,  time{k,1}(t+1:t+72)  = [j+72+1-Kp_list(k):j+72, j+1:j+72-Kp_list(k) ]; end
 for k = 1:Kp2, time2{k,1}(t+1:t+72) = [j+72+1-Kp2_list(k):j+72,j+1:j+72-Kp2_list(k)]; end
end

%--------------------------------------------------------------------------------

% create target fpr test (target for test is noise free)
st2 = cell(Kf,1);
for i = 1:Kf, st2{i,1} = data_3(timet{i,1}(T+1:T+T2),1:Ns)'; end % test target

% target for training and inputs for training and test may contain noise
if (WithNoise)
 sigma_noise    = 2.2832; % same amplitude as original input covariance
 fprintf(1,'sigma_noise = %f\n', sigma_noise);
 data_3_var     = mean(var(data_3(:,1:Ns)));
 fprintf(1,'averaged input variance (original)   = %f\n', data_3_var);
 data_3(:,1:Ns) = data_3(:,1:Ns) + randn(T+T2,Ns) * sigma_noise;
 data_3_var2    = mean(var(data_3(:,1:Ns)));
 fprintf(1,'averaged input variance (with noise) = %f\n', data_3_var2);
end

% create target for training
st  = cell(Kf,1);
for i = 1:Kf, st{i,1}  = data_3(timet{i,1}(1:T)     ,1:Ns)'; end % training target

% create inputs for training and test
s   = data_3(time{1,1}(1:T)     ,1:Ns)';                         % training input data
s2  = data_3(time{1,1}(T+1:T+T2),1:Ns)';                         % test input data

%--------------------------------------------------------------------------------

% priors (small constants) to prevent inverse matrix from being singular
prior_x       = 1;
prior_s       = 100;
prior_s_      = 100;
prior_so_     = 100;
prior_qSigmao = 0.01;

% test prediction error
NT            = 8;               % number of section
err_s1.th     = zeros(Kf,Ns,NT); % PredPCA theory
err_s1.em     = zeros(Kf,Ns,NT); % PredPCA empirical
err_s1.tho    = zeros(Kf,Ns,NT); % PredPCA theory (opn)
err_s1.emo    = zeros(Kf,Ns,NT); % PredPCA empirical (opn)

%--------------------------------------------------------------------------------

fprintf(1,'create basis functions (time = %.1f min) ', toc/60);
s_  = zeros(Ns*Kp,T);
s2_ = zeros(Ns*Kp,T2);
for k = 1:Kp
 s_(Ns*(k-1)+1:Ns*k,:)  = data_3(time{k,1}(1:T)     ,1:Ns)';
 s2_(Ns*(k-1)+1:Ns*k,:) = data_3(time{k,1}(T+1:T+T2),1:Ns)';
 fprintf(1,'.');
end
fprintf(1,'\n');

fprintf(1,'calculate basis covariance (time = %.1f min) ', toc/60);
S_S_         = cell(NT,1);
for i = 1:NT
 S_S_{i,1} = s_(:,T*(i-1)/NT+1:T*i/NT)*s_(:,T*(i-1)/NT+1:T*i/NT)';
 if (i > 1), S_S_{i,1} = S_S_{i,1} + S_S_{i-1,1}; end
 fprintf(1,'.');
end
fprintf(1,'\n');

fprintf(1,'maximum likelihood estimation (time = %.1f min) ', toc/60);
Qopt          = cell(Kf,1);
S_S_inv       = (S_S_{NT,1}+eye(Ns*Kp)*prior_s_)^(-1);
for k = 1:Kf, Qopt{k,1} = (st{k,1}*s_') * S_S_inv; end
qSigmas_opt   = (s*s') / T;
qSigmase_opt  = cell(Kf,1);
qSigmase_mean = zeros(Ns,Ns);
for k = 1:Kf
 se                = Qopt{k,1} * s_;
 qSigmase_opt{k,1} = (se*se') / T;
 qSigmase_mean     = qSigmase_mean + qSigmase_opt{k,1} / Kf;
 fprintf(1,'.');
end
clear se
[Copt,Lopt]   = pcacov(qSigmase_mean);
fprintf(1,'\n');

fprintf(1,'training error\n');
for k = 1:Kf, fprintf(1,'%d deg rotation: err = %f\n', k * 30, trace(qSigmas_opt-qSigmase_opt{k,1})/trace(qSigmas_opt)), end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

fprintf(1,'intestigate the accuracy with limited number of training samples\n')
for h = 1:NT
T1      = T * h / NT; % number of training samples
s1      = s(:,1:T1);  % actual input (training)
fprintf(1,'training with T1 = %d (time = %.1f min)\n', T1, toc/60);

%--------------------------------------------------------------------------------
% PredPCA

fprintf(1,'compute PredPCA with plain bases ');

% maximum likelihood estimation
se1     = cell(Kf,1);
se2     = cell(Kf,1);
S_S_inv = (S_S_{h,1}+eye(Ns*Kp)*prior_s_)^(-1);
for k = 1:Kf
 Q        = (st{k,1}(:,1:T1)*s_(:,1:T1)') * S_S_inv; % mapping
 se1{k,1} = Q * s_(:,1:T1);                          % predicted input (training)
 se2{k,1} = Q * s2_;                                 % predicted input (test)
 fprintf(1,'.');
end
fprintf(1,'\n');

% predicted input covariance
qSigmase      = cell(Kf,1);
qSigmase_mean = zeros(Ns,Ns);
for k = 1:Kf
 qSigmase{k,1} = se1{k,1}*se1{k,1}'/T1;            % predicted input covariance
 qSigmase_mean = qSigmase_mean + qSigmase{k,1}/Kf; % mean
end

% post-hoc PCA
[C,Lppca] = pcacov(qSigmase_mean); % eigenvalue decomposition
Wppca     = C(:,1:Nu)';            % optimal weights = transpose of eigenvectors
u1        = Wppca * se1{1,1};      % predictive enoders (training)
uc1       = Wppca * s1;            % current input enoders (training)

% test prediction error
err_s1.em(:,:,h) = prediction_error(st2, se2, C);

%--------------------------------------------------------------------------------

% system parameter identification
qA      = Wppca';                                            % observation matrix
qPsi    = (u1(:,[2:T1,1])*u1') * (u1*u1'+eye(Nu)*T1/T)^(-1); % transition matrix
qSigmas = (s1*s1') / T1;                                     % input covariance
qSigmap = (qPsi+eye(Nu,Nu)*10^(-4))^(-1) * (uc1(:,[2:T1,1])*uc1'/T1);
qSigmap = (qSigmap + qSigmap') / 2;                          % hidden basis covariance
qSigmao = qSigmas - qA * qSigmap * qA';                      % observation noise covariance
[U,S,V] = svd(qSigmao);
S       = max(S,eye(Ns)*prior_qSigmao);                      % make all eigenvalues positive
qSigmao = U*S*U';                                            % correction

% optimal basis functions
Gain    = qA' * qSigmao^(-1);
so1_    = zeros(Nu*Kp2,T1);
so2_    = zeros(Nu*Kp2,T2);
for k = 1:Kp2
 so1_(Nu*(k-1)+1:Nu*k,:) = Gain * data_3(time2{k,1}(1:T1)    ,1:Ns)'; % optimal bases
 so2_(Nu*(k-1)+1:Nu*k,:) = Gain * data_3(time2{k,1}(T+1:T+T2),1:Ns)'; % optimal bases
end

%--------------------------------------------------------------------------------
% PredPCA

fprintf(1,'compute PredPCA with optimal bases ');

% maximum likelihood estimation
% compute covariance matrix with two steps (37-->6-->1)
se1   = cell(Kf,1);
se2   = cell(Kf,1);
so1_2 = zeros(Nu*6,T1);
so2_2 = zeros(Nu*6,T2);
for k = 1:Kf
 Gain_st_k = Gain*st{k,1}(:,1:T1);
 % level 1
 for l = 1:5
  [so1_2(Nu*(l-1)+1:Nu*l,:),so2_2(Nu*(l-1)+1:Nu*l,:),~] = mlest(Gain_st_k,so1_(Nu*6*(l-1)+1:Nu*6*l,:),so2_(Nu*6*(l-1)+1:Nu*6*l,:),prior_so_);
 end
 [so1_2(Nu*5+1:Nu*6,:),so2_2(Nu*5+1:Nu*6,:),~] = mlest(Gain_st_k,so1_(Nu*30+1:Nu*37,:),so2_(Nu*30+1:Nu*37,:),prior_so_);
 % level 2
 [se1{k,1},se2{k,1},Q] = mlest(st{k,1}(:,1:T1), so1_2, so2_2, prior_so_);
 fprintf(1,'.');
end
clear so1_2 so2_2
fprintf(1,'\n');

% predicted input covariance
qSigmase      = cell(Kf,1);
qSigmase_mean = zeros(Ns,Ns);
for k = 1:Kf
 qSigmase{k,1} = se1{k,1}*se1{k,1}'/T1;            % predicted input covariance
 qSigmase_mean = qSigmase_mean + qSigmase{k,1}/Kf; % mean
end

% post-hoc PCA
[C,Lppca2] = pcacov(qSigmase_mean); % eigenvalue decomposition
Wppca2     = C(:,1:Nu)';            % optimal weights = transpose of eigenvectors

% test prediction error
err_s1.emo(:,:,h) = prediction_error(st2, se2, C);

%--------------------------------------------------------------------------------

end

fprintf(1,'search complete (time = %.1f min)\n', toc/60);
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------
% postprocessing

fprintf(1,'postprocessing\n');
u1  = Wppca2 * se1{3,1}; % predictive enoders (training)
u2  = Wppca2 * se2{3,1}; % predictive enoders (test)
uc1 = Wppca2 * s1;       % auto enoders (training)
uc2 = Wppca2 * s2;       % auto enoders (test)
u1_ = zeros(Nu,T1);
u2_ = zeros(Nu,T2);
for k = 1:Kf
 u1_ = u1_ + Wppca2 * se1{k,1} / Kf; % mean predictive enoders (training)
 u2_ = u2_ + Wppca2 * se2{k,1} / Kf; % mean predictive enoders (test)
end
du1 = cell(Kf,1);
du2 = cell(Kf,1);
for k = 1:Kf
 du1{k,1} = Wppca2 * se1{k,1} - u1_;
 du2{k,1} = Wppca2 * se2{k,1} - u2_;
end

%--------------------------------------------------------------------------------
% calculate theortical values

tr_Sigmas = trace(qSigmas_opt);
for i = 1:Ns
 Wi           = Copt(:,1:i)'; % mapping
 tr_WSigmasWT = trace(Wi*qSigmas_opt*Wi');
 for k = 1:Kf
  tr_WSigmaseWT = trace(Wi*qSigmase_opt{k,1}*Wi');
  entropy       = Kp*Ns * (tr_WSigmasWT - tr_WSigmaseWT);
  entropy2      = Kp*Nu * (tr_WSigmasWT - tr_WSigmaseWT);
  for h = 1:NT
   T1 = T * h / NT;
   err_s1.th(k,i,h)  = tr_Sigmas - tr_WSigmaseWT + entropy  / T1; % theory
   err_s1.tho(k,i,h) = tr_Sigmas - tr_WSigmaseWT + entropy2 / T1; % theory
  end
 end
end

%--------------------------------------------------------------------------------
% plot test prediction error

fig        = figure();
norm_s2    = mean(sum(st2{1,1}.^2));
err_s1.em  = err_s1.em  / norm_s2;
err_s1.th  = err_s1.th  / norm_s2;
err_s1.emo = err_s1.emo / norm_s2;
err_s1.tho = err_s1.tho / norm_s2;
for k = 1:Kf
 subplot(3,2,k)
 plot(1:NT, reshape(err_s1.tho(k,Nu,:),[NT,1]), '-r', 1:NT, reshape(err_s1.emo(k,Nu,:),[NT,1]), '+r'), hold on
 plot(1:NT, reshape(err_s1.th(k,Nu,:), [NT,1]), '-b', 1:NT, reshape(err_s1.em(k,Nu,:), [NT,1]), '+b')
 plot(1:NT, reshape(err_s1.th(k,Ns,:), [NT,1]), '-g', 1:NT, reshape(err_s1.em(k,Ns,:), [NT,1]), '+g'), hold off
 axis([1 NT 0.0 1.0]), title(['test error (', num2str(k * 30), ' deg rot)'])
end
drawnow
set(fig, 'PaperPosition', [0,2,20,26])
print(fig, 'predpca_test_err.pdf', '-dpdf');
output_data = [];
for k = 1:Kf
 output_data = [output_data, [1:3; reshape(err_s1.emo(k,Nu,:),[NT,1]), reshape(err_s1.em(k,Nu,:),[NT,1]), reshape(err_s1.em(k,Ns,:),[NT,1])]];
end
csvwrite('predpca_test_err.csv',output_data)

%--------------------------------------------------------------------------------
