
%--------------------------------------------------------------------------------

% fig3_predpca.m
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
% 2020-6-20
%
% Before run this script, please download ALOI dataset from
% http://aloi.science.uva.nl
% (full color (24 bit), quarter resolution (192 x 144), viewing direction)
% and expand aloi_red4_view.tar in the same directory
%
% Reference to ALOI dataset
% Geusebroek JM, Burghouts GJ, Smeulders AWM, The Amsterdam library of object images,
% Int J Comput Vision, 61, 103-112 (2005)
%
% Run predpca_aloi_preprocess.m
% and put the output file aloi_data.mat in the same directory
%

%--------------------------------------------------------------------------------
% initialization

clear
tic
Timg      = 72000;
T         = 57600;    % number of training data
T2        = 14400;    % number of test data
Ns        = 300;      % dimentionality of inputs
Nu        = 128;      % dimentionality of encoders

Kf        = 5;        % number of predicting points
Kp        = 19;       % number of reference points
Kp2       = 37;       % number of reference points
WithNoise = 0;        % presence of noise
dir       = '';

fprintf(1,'----------------------------------------\n');
fprintf(1,'PredPCA of 3D rotating object images\n');
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

% priors (small constants) to prevent inverse matrix from being singular
prior_x       = 1;
prior_s       = 100;
prior_s_      = 100;
prior_so_     = 100;
prior_qSigmao = 0.01;

% test prediction error
NT            = 8;               % number of section
err_s1.em     = zeros(Kf,Ns,NT); % PredPCA empirical
err_s1.emo    = zeros(Kf,Ns,NT); % PredPCA empirical (opn)
Nu            = Nu/NT*(1:NT);    % dimentionality of encoders

%--------------------------------------------------------------------------------

for rep = 0:9
fprintf(1,'rep = %d\n', rep);

fprintf(1,'read aloi_data.mat\n');
load('aloi_data.mat') % read file
data = cast(data(1:Ns,:),'double');

%--------------------------------------------------------------------------------
% create random object sequences for training and test

fprintf(1,'preprocessing\n');
seed  = rep;
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
 for k = 1:Kf,  timet{k,1}(t+1:t+72) = [j+1+Kf_list(k):j+72, j+1:j+Kf_list(k)]; end
 for k = 1:Kp,  time{k,1}(t+1:t+72) = [j+72+1-Kp_list(k):j+72, j+1:j+72-Kp_list(k)]; end
 for k = 1:Kp2, time2{k,1}(t+1:t+72) = [j+72+1-Kp2_list(k):j+72, j+1:j+72-Kp2_list(k)]; end
end

%--------------------------------------------------------------------------------
% create input data

% create target for test (target for test is noise free)
st2 = cell(Kf,1);
for i = 1:Kf, st2{i,1} = data(:,timet{i,1}(T+1:T+T2)); end % test target

% target for training and inputs for training and test may contain noise
if (WithNoise)
 sigma_noise    = 2.3; % same amplitude as original input covariance
 fprintf(1,'sigma_noise = %f\n', sigma_noise);
 data_var     = mean(var(data'));
 fprintf(1,'averaged input variance (original)   = %f\n', data_var);
 data         = data + randn(Ns,T+T2) * sigma_noise;
 data_var2    = mean(var(data'));
 fprintf(1,'averaged input variance (with noise) = %f\n', data_var2);
end

% create target for training
st  = cell(Kf,1);
for i = 1:Kf, st{i,1} = data(:,timet{i,1}(1:T)); end % training target

% create inputs for training and test
s   = data(:,time{1,1}(1:T));                        % training input data
s2  = data(:,time{1,1}(T+1:T+T2));                   % test input data

%--------------------------------------------------------------------------------

fprintf(1,'create basis functions (time = %.1f min) ', toc/60);
s_  = zeros(Ns*Kp,T);
s2_ = zeros(Ns*Kp,T2);
for k = 1:Kp
 s_(Ns*(k-1)+1:Ns*k,:)  = data(:,time{k,1}(1:T));      % plain bases
 s2_(Ns*(k-1)+1:Ns*k,:) = data(:,time{k,1}(T+1:T+T2)); % plain bases
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
Wppca     = C(:,1:Nu(h))';         % optimal weights = transpose of eigenvectors
u1        = Wppca * se1{1,1};      % predictive enoders (training)
uc1       = Wppca * s1;            % current input enoders (training)

% test prediction error
err_s1.em(:,:,h) = prediction_error(st2, se2, C);

%--------------------------------------------------------------------------------

% system parameter identification
qA      = Wppca';                                               % observation matrix
qPsi    = (u1(:,[2:T1,1])*u1') * (u1*u1'+eye(Nu(h))*T1/T)^(-1); % transition matrix
qSigmas = (s1*s1') / T1;                                        % input covariance
qSigmap = qPsi^(-1) * (uc1(:,[2:T1,1])*uc1'/T1);
qSigmap = (qSigmap + qSigmap') / 2;                             % hidden basis covariance
qSigmao = qSigmas - qA * qSigmap * qA';                         % observation noise covariance
[U,S,V] = svd(qSigmao);
S       = max(S,eye(Ns)*prior_qSigmao);                         % make all eigenvalues positive
qSigmao = U*S*U';                                               % correction

% optimal basis functions
Gain    = qA' / qSigmao;                                        % optimal gain
so1_    = zeros(Nu(h)*Kp2,T1);
so2_    = zeros(Nu(h)*Kp2,T2);
for k = 1:Kp2
 so1_(Nu(h)*(k-1)+1:Nu(h)*k,:) = Gain * data(:,time2{k,1}(1:T1));     % optimal bases
 so2_(Nu(h)*(k-1)+1:Nu(h)*k,:) = Gain * data(:,time2{k,1}(T+1:T+T2)); % optimal bases
end

%--------------------------------------------------------------------------------
% PredPCA

fprintf(1,'compute PredPCA with optimal bases ');

% maximum likelihood estimation
se1   = cell(Kf,1);
se2   = cell(Kf,1);
for k = 1:Kf
 Gain_st_k = Gain*st{k,1}(:,1:T1);
 [se1{k,1},se2{k,1},Q] = mlest(st{k,1}(:,1:T1), so1_, so2_, prior_so_);
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
[C,Lppca2] = pcacov(qSigmase_mean); % eigenvalue decomposition
Wppca2     = C(:,1:Nu(h))';         % optimal weights = transpose of eigenvectors

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
u1_ = zeros(Nu(NT),T1);
u2_ = zeros(Nu(NT),T2);
for k = 1:Kf
 u1_ = u1_ + Wppca2 * se1{k,1} / Kf; % mean predictive enoders (training)
 u2_ = u2_ + Wppca2 * se2{k,1} / Kf; % mean predictive enoders (test)
end
u2_ = u2_ - mean(u1_')' * ones(1,T2);
u1_ = u1_ - mean(u1_')' * ones(1,T);

du1 = cell(Kf,1);
du2 = cell(Kf,1);
for k = 1:Kf
 du1{k,1} = Wppca2 * se1{k,1} - u1_;
 du2{k,1} = Wppca2 * se2{k,1} - u2_;
end

%--------------------------------------------------------------------------------
% test prediction error (for Fig 3e)

norm_s2    = mean(sum(st2{1,1}.^2));
err_s1.em  = err_s1.em  / norm_s2;
err_s1.emo = err_s1.emo / norm_s2;
err_dst    = zeros(Kf,NT);
for h = 1:NT
 err_dst(:,h) = err_s1.emo(:,Nu(h),h);
end
output_data = [];
for k = 1:Kf
 output_data = [output_data, [1:2; err_dst(k,:)', reshape(err_s1.em(k,Ns,:),[NT,1])]];
end
csvwrite(['predpca_test_err_', num2str(rep), '.csv'],output_data)

%--------------------------------------------------------------------------------
% optimal encoding dimensionality (for Fig 3d)

fprintf(1,'optimal encoding dimensionality (time = %.1f min)\n', toc/60);
err_mean  = zeros(Ns,NT);
err_mean2 = zeros(Ns,NT);
for k = 1:Kf
 err_mean  = err_mean  + reshape(err_s1.em(k,:,:), [Ns NT]) / Kf;
 err_mean2 = err_mean2 + reshape(err_s1.emo(k,:,:),[Ns NT]) / Kf;
end
[~,idx] = min(err_mean);

output_data = [1:NT; idx];
csvwrite(['predpca_opt_encode_dim_', num2str(rep), '.csv'],output_data)
fprintf(1,'----------------------------------------\n\n');
if (rep >= 1), continue, end

%--------------------------------------------------------------------------------
% plot test prediction error

fig = figure();
for k = 1:Kf
 subplot(3,2,k)
 plot((2:NT)*(800/NT), reshape(err_dst(k,2:NT),[NT-1,1]), '-b'), hold on
 plot((2:NT)*(800/NT), reshape(err_s1.em(k,Ns,2:NT), [NT-1,1]), '--b'), hold off
 axis([2*(800/NT) NT*(800/NT) 0.0 1.2]), title(['test error (', num2str(k * 30), ' deg rot)'])
end
drawnow
set(fig, 'PaperPosition', [0,2,20,26])
print(fig, 'predpca_test_err.pdf', '-dpdf');

%--------------------------------------------------------------------------------
% true and predicted images (for Fig 3a and Suppl Movie)

fprintf(1,'true and predicted images (time = %.1f min)\n', toc/60);
fprintf(1,'create supplementary movie\n', toc/60);
err_obj = zeros(200,1);
var_obj = zeros(200,1);
for t = 1:200
 err_obj(t,1) = mean(sum((st2{3,1}(:,(1:72)+(t-1)*72) - Wppca2' * u2(:,(1:72)+(t-1)*72)).^2));
 var_obj(t,1) = sum(var(st2{3,1}(:,(1:72)+(t-1)*72)'));
end
[~,idx]       = sort(err_obj./var_obj);
vid           = VideoWriter('predpca_movie.mp4', 'MPEG-4');
vid.FrameRate = 30;
vid.Quality   = 100;
open(vid);
for rot = 1:72
 if (rem(rot,6) == 0), fprintf(1,'rot = %d deg\n', rot * 5), end
 s_list(:,1:2:400) = st2{3,1}(:,rot+(idx-1)*72);
 s_list(:,2:2:400) = Wppca2' * u2(:,rot+(idx-1)*72);
 img = state_to_image(s_list(:,1:200), PCA_C2, PCA_C1, mean1, 20, 10);
 img_size = size(img);
 img = imresize(img,img_size(1:2)/2);
 img = min(img,1);
 img = max(img,0);
 writeVideo(vid,img);
end
clear img
close(vid)
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% hidden state analysis
% ICA of mean encoders (for Fig 3b and Suppl Fig 4)

fprintf(1,'ICA of mean encoders (time = %.1f min)\n', toc/60);
Nv      = 20;
ica_rep = 8000;
Wica = diag(std(u1_(1:Nv,:)'))^(-1);
for t = 1:ica_rep
 if (rem(t,500) == 0), fprintf(1,'t = %d\n', t), end
 if (t < 2000),     eta = 0.02;
 elseif (t < 4000), eta = 0.01;
 else               eta = 0.005; end
 t_list = randi([1,T],T/10,1);
 v1     = Wica * u1_(1:Nv,t_list);
 g1     = tanh(100 * v1);
 Wica   = Wica + eta * (eye(Nv) - g1*v1'/(T/10)) * Wica;
end

v1      = Wica * u1_(1:Nv,:);
[~,idx] = sort(kurtosis(v1'),'descend');
Omega   = zeros(Nv,Nv);
Omega(Nv*(idx-1)+(1:Nv)) = 1;
Wica    = Omega * diag(sign(skewness(v1'))) * Wica;
v1      = Wica * u1_(1:Nv,:);
v2      = Wica * u2_(1:Nv,:);

if (WithNoise == 0)
 v1_ref = v1;
 save('v1_ref.mat','v1_ref','-v7.3');
end
if (WithNoise == 1)
 load('v1_ref.mat')
end

Omega = corr(v1_ref',v1');
for i = 1:Nv
 [~,j] = max(abs(Omega(i,:)));
 temp       = sign(Omega(i,j));
 Omega(i,:) = 0;
 Omega(:,j) = 0;
 Omega(i,j) = temp;
end
Wica = Omega * Wica;
v1      = Wica * u1_(1:Nv,:);
v2      = Wica * u2_(1:Nv,:);

% images corresponding to independent components
img = state_to_image(Wppca2(1:Nv,:)'*Wica^(-1)*20, PCA_C2, PCA_C1, mean1, ceil(Nv/3), 3);
imwrite(img, ['predpca_ica.png'])
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% PCA of deviation encoders (for Fig 3c)

fprintf(1,'PCA of deviation encoders (time = %.1f min)\n', toc/60);
[C,~,L] = pca(du1{3,1}(1:Nv,:)');
pc1     = C(:,1)'*du2{3,1}(1:Nv,:);
pc1     = pc1 / std(pc1);
fig     = figure();
plot(reshape(pc1,[72 T2/72]),'c-'), hold on
plot([1 72],[0 0],'k--','LineWidth',3)
plot(quantile(reshape(pc1,[72 T2/72])',0.2),'k-','LineWidth',3)
plot(quantile(reshape(pc1,[72 T2/72])',0.5),'k-','LineWidth',3)
plot(quantile(reshape(pc1,[72 T2/72])',0.8),'k-','LineWidth',3), hold off
axis([1 72 -2 2])
drawnow

output_data = [1:T2/72; reshape(pc1,[72 T2/72])];
csvwrite('predpca_pc1_of_deviation.csv',output_data)
csvwrite('predpca_pc1_of_deviation_eig.csv',[1:2; L,L/sum(L)])

for i = 1:10
 img = state_to_image(Wppca2' * u2(:,(0:3)*18+1+(i-1)*72), PCA_C2, PCA_C1, mean1, 4, 1);
 imwrite(img, ['test_img_', num2str(i), '.png'])
end
fprintf(1,'----------------------------------------\n\n');
end

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------
