
%--------------------------------------------------------------------------------

% fig2cd.m
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

function fig2cd(seed)

% initialization
sequence_type = 1;      % this script is for type 1 = ascending order
rng(1000000+seed);
dir           = '';
T             = 100000; % training sample size
T2            = 100000; % test sample size
T3            = 100000; % data size used for determining true parameters

prior_x       = 1;      % magnitude of regularization term
prior_s       = 100;
prior_s_      = 100;

%--------------------------------------------------------------------------------
% create input sequences

fprintf(1,'read files\n')
train_randomness = 1;
test_randomness  = 1;
train_signflip   = 1;
test_signflip    = 1;
[input,input2,input3,label,label2,label3] = create_digit_sequence(dir,sequence_type,T,T2,T3,train_randomness,test_randomness,train_signflip,test_signflip);
input_mean       = mean(input')';
input            = input  - input_mean * ones(1,T);
input2           = input2 - input_mean * ones(1,T2);
input3           = input3 - input_mean * ones(1,T3);

fprintf(1,'compress data using PCA as preprocessing\n')
Ns               = 40;
[C,~,L]          = pca(input');
Wpca             = C(:,1:Ns)';
Lpca             = L;
s                = Wpca * input;
s2               = Wpca * input2;
s3               = Wpca * input3;
s2               = diag(std(s')) * diag(sqrt(mean(s2'.^2)))^(-1) * s2; % match the variance
s3               = diag(std(s')) * diag(sqrt(mean(s3'.^2)))^(-1) * s3; % match the variance

fprintf(1,'compute maximum likelihood estimator\n')
Kp                = 10;
[s_,s2_,se,se2,Q] = maximum_likelihood_estimator(s,s2,s,Kp,prior_s_);

%--------------------------------------------------------------------------------
% true states and parameters

fprintf(1,'compute true states and parameters\n')
Nx     = 10;
x      = generate_true_hidden_states(input + input_mean * ones(1,T),label);
x2     = generate_true_hidden_states(input2 + input_mean * ones(1,T2),label2);
x3     = generate_true_hidden_states(input3 + input_mean * ones(1,T3),label3);
x_mean = mean(x')';
x      = x  - x_mean * ones(1,T);
x2     = x2 - x_mean * ones(1,T2);
x3     = x3 - x_mean * ones(1,T3);
[A,B,Sigmas,Sigmax,Sigmao,Sigmaz] = calculate_true_parameters(s3,x3,prior_x); % this function is for ascending order

%--------------------------------------------------------------------------------

Nu     = 10;

fprintf(1,'optimal state and parameter estimators obtained using supervised learning\n')
qx_opt       = (x*s_') * (s_*s_'+eye(Ns*Kp)*prior_s_)^(-1) * s_;
qx_opt2      = (x*s_') * (s_*s_'+eye(Ns*Kp)*prior_s_)^(-1) * s2_;
Aopt         = (s*x') * (x*x'+eye(Nx)*prior_x)^(-1);

fprintf(1,'optimal state and parameter estimators obtained using PredPCA\n')
qSigmas_opt  = (s*s') / T;
qSigmase_opt = ((Q*s_)*(Q*s_)') / T;
[Copt,Lopt]  = pcacov(qSigmase_opt);
norm_s2      = trace(s2*s2'/T2);
u1opt        = Copt(:,1:Nx)' * (Q*s_);
Omega_opt    = (x*u1opt') * (u1opt*u1opt'+eye(Nx)*prior_x)^(-1);

%--------------------------------------------------------------------------------
% prediction errors

NT           = 19;
err_s0       = zeros(1, NT); % optimal
err_s1.th    = zeros(Ns,NT); % PredPCA theory
err_s1.em    = zeros(Ns,NT); % PredPCA empirical
err_s2.th    = zeros(1, NT); % SL theory
err_s2.em    = zeros(1, NT); % SL empirical

err_param    = zeros(6, NT); % parameter estimation error

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

fprintf(1,'investigate the accuracy with limited number of training samples\n')
for h = 1:NT
 if (h < 10), T1 = T/100 * h;
 else,        T1 = T/10  * (h - 9); end
 fprintf(1,'T1 = %d\n', T1)
 s1      = s(:,1:T1);
 s1_     = s_(:,1:T1);
 x1      = x(:,1:T1);
 s1      = s1  - mean(s1' )' * ones(1,T1);
 s1_     = s1_ - mean(s1_')' * ones(1,T1);
 x1      = x1  - mean(x1' )' * ones(1,T1);
 
 %--------------------------------------------------------------------------------
 % PredPCA
 
 % maximum likelihood estimation
 Q       = (s1*s1_') * (s1_*s1_'+eye(Ns*Kp)*prior_s_)^(-1); % mapping
 se1     = Q * s1_;     % input expectations
 se2     = Q * s2_;     % input expectations
 
 % eigenvalue decomposition
 [C,~,L] = pca(se1');
 Wppca   = C(:,1:Nu)';  % eigenvectors
 Lppca   = L;           % eigenvalues
 u1      = Wppca * se1; % enoders (training) / prediction
 uc1     = Wppca * s1;  % enoders (training) / based on current input
 
 %--------------------------------------------------------------------------------
 % test prediction error
 
 % optimal
 err_s0(1,h)    = mean(sum((s2 - A * qx_opt2).^2));                % optimal
 
 % supervised learning
 Qsl            = (x1*s1_') * (s1_*s1_'+eye(Ns*Kp)*prior_s_)^(-1); % mapping
 qx_sl1         = Qsl * s1_;                                       % hidden state expectation
 qx_sl2         = Qsl * s2_;                                       % hidden state expectation
 A_sl           = (s1*x1') * (x1*x1'+eye(Nx)*prior_x)^(-1);        % mapping
 err_s2.th(1,h) = trace(qSigmas_opt-qSigmase_opt) + Nx/T1*trace(qSigmas_opt-Aopt*(x*x'/T)*Aopt');
 err_s2.th(1,h) = err_s2.th(1,h) + Ns*Kp/T1*trace(Aopt*(x*x'/T - qx_opt*qx_opt'/T)*Aopt'); % theory
 err_s2.em(1,h) = mean(sum((s2 - A_sl * qx_sl2).^2));              % empirical
 
 % PredPCA
 for i = 1:Ns
  Wi             = Copt(:,1:i)';                                     % mapping
  err_s1.th(i,h) = trace(qSigmas_opt) - trace(Wi*qSigmase_opt*Wi') + Kp*Ns/T1 * trace(Wi*(qSigmas_opt-qSigmase_opt)*Wi'); % theory
  Wi             = C(:,1:i)';                                        % mapping
  ui2            = Wi * se2;                                         % encoder
  qxi2           = (x2*ui2') * (ui2*ui2'+eye(i)*prior_x)^(-1) * ui2; % hidden state expectation
  err_s1.em(i,h) = mean(sum((s2 - Wi'*ui2).^2));                     % empirical
 end
 
 %--------------------------------------------------------------------------------
 % system parameter identification
 
 Omega_inv      = (A'*A+eye(Nu)*10^(-4))^(-1)*A'*Wppca';           % ambiguity factor (coordinate transformation)
 [qA,qB,qSigmas,qSigmax,qSigmao,qSigmaz] = calculate_estimated_parameters(s1,u1,uc1,Wppca,Omega_inv,prior_x*T1/T); % this function is for ascending order
 
 err_param(1,h) = norm(A     -qA,     'fro')^2 / max(norm(A,     'fro')^2, norm(qA,     'fro')^2);
 err_param(2,h) = norm(B     -qB,     'fro')^2 / max(norm(B,     'fro')^2, norm(qB,     'fro')^2);
 err_param(3,h) = norm(Sigmas-qSigmas,'fro')^2 / max(norm(Sigmas,'fro')^2, norm(qSigmas,'fro')^2);
 err_param(4,h) = norm(Sigmax-qSigmax,'fro')^2 / max(norm(Sigmax,'fro')^2, norm(qSigmax,'fro')^2);
 err_param(5,h) = norm(Sigmao-qSigmao,'fro')^2 / max(norm(Sigmao,'fro')^2, norm(qSigmao,'fro')^2);
 err_param(6,h) = norm(Sigmaz-qSigmaz,'fro')^2 / max(norm(Sigmaz,'fro')^2, norm(qSigmaz,'fro')^2);
 
end

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------
% for Fig 2c

% save results
csvwrite([dir 'err_param_' num2str(seed) '.csv'],[1:NT; err_param])

% show results
figure()
subplot(1,2,1)
plot(log10([(1:10)*10^3 (2:10)*10^4]), err_param(1,:)' *100,'k+'), hold on
plot(log10([(1:10)*10^3 (2:10)*10^4]), err_param(2,:)' *100,'r+')
plot(log10([(1:10)*10^3 (2:10)*10^4]), err_param(3,:)' *100,'g+')
plot(log10([(1:10)*10^3 (2:10)*10^4]), err_param(4,:)' *100,'b+')
plot(log10([(1:10)*10^3 (2:10)*10^4]), err_param(5,:)' *100,'y+'), hold off
title('parameter estimation error')

%--------------------------------------------------------------------------------
% for Fig 2d

% save results
csvwrite([dir 'err_pred_'  num2str(seed) '.csv'],[1,1:Ns,-(1:Ns),1,-1; [err_s0', err_s1.em', err_s1.th', err_s2.em', err_s2.th']/norm_s2])

t       = [(1:10)*10^3,(2:10)*10^4];
norm_s2 = mean(sum(s2.^2));
norm_x2 = mean(sum(x2.^2));

% show results
subplot(1,2,2)
semilogx(t, err_s0/norm_s2, '-k'), hold on
semilogx(t, err_s2.th/norm_s2, '-b', t, err_s2.em/norm_s2, '+b')
semilogx(t, err_s1.th(Nx,:)/norm_s2, '-g', t, err_s1.em(Nx,:)/norm_s2, '+g')
semilogx(t, err_s1.th(Ns,:)/norm_s2, '-y', t, err_s1.em(Ns,:)/norm_s2, '+y'), hold off
axis([0 10^5 0.4 0.8])
title('test prediction error')
drawnow

%--------------------------------------------------------------------------------

