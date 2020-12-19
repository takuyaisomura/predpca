
%--------------------------------------------------------------------------------

% extdatafig1a.m
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
% 2020-6-2
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

function extdatafig1a(seed)

% initialization
sequence_type = 1;      % this script is for type 1 = ascending order
rng(1000000+seed);
dir           = '';
T             = 100000; % training sample size
T2            = 100000; % test sample size
T3            = 400000; % data size used for determining true parameters
T4            = 100000; % test sample size (for categorization task)

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
[~,input4,~,~,label4,~]                   = create_digit_sequence(dir,sequence_type,T,T2,T3,train_randomness,0,train_signflip,0);
input_mean       = mean(input')';
input            = input  - input_mean * ones(1,T);
input2           = input2 - input_mean * ones(1,T2);
input3           = input3 - input_mean * ones(1,T3);
input4           = input4 - input_mean * ones(1,T4);

fprintf(1,'compress data using PCA as preprocessing\n')
Ns               = 40;
[C,~,L]          = pca(input');
Wpca             = C(:,1:Ns)';
Lpca             = L;
s                = Wpca * input;
s2               = Wpca * input2;
s3               = Wpca * input3;
s4               = Wpca * input4;
s2               = diag(std(s')) * diag(sqrt(mean(s2'.^2)))^(-1) * s2; % match the variance
s3               = diag(std(s')) * diag(sqrt(mean(s3'.^2)))^(-1) * s3; % match the variance

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
% prediction errors

Nu           = 10;
NT           = 19;
ica_rep      = 2000; % number of iteration for ICA
ica_eta      = 0.01; % learning rate for ICA
norm_s2      = trace(s2*s2'/T2);

err_cat      = zeros(10, NT); % categorization error
err_param    = zeros(6, NT);  % parameter estimation error
err_s1.th    = zeros(3, NT);  % PredPCA test prediction error (theory)
err_s1.em    = zeros(3, NT);  % PredPCA test prediction error (empirical)

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

T = 40000;
s = s(:,1:T);

fprintf(1,'intestigate the accuracy with varying the number of past observations Kp\n')
for h = 1:NT
 if (h < 10), Kp = h;
 else,        Kp = 10 * (h - 9); end
 fprintf(1,'Kp = %d\n', Kp)
 
 %--------------------------------------------------------------------------------
 % PredPCA
 
 fprintf(1,'compute maximum likelihood estimator\n')
 [~,~,se,se2,Q] = maximum_likelihood_estimator(s,s2,s,Kp,prior_s_);
 [~,~,se3,~,~]  = maximum_likelihood_estimator(s3,s2,s3,Kp,prior_s_);
 
 fprintf(1,'post hoc PCA (eigenvalue decomposition)\n')
 [C,~,L]        = pca(se');
 Wppca          = C(:,1:Nu)';        % eigenvectors
 Lppca          = L;                 % eigenvalues
 u              = Wppca * se;        % enoders (training) / prediction
 u2             = Wppca * se2;       % enoders (test) / prediction
 uc             = Wppca * s;         % enoders (training) / based on current input
 
 Sigmase        = se3*se3'/T3;
 [C,~,L]        = pca(se3');
 Wopt           = C(:,1:Nu)';        % eigenvectors
 
 s4_            = zeros(Ns*Kp,T4);   % basis functions (test)
 for k = 1:Kp, s4_(Ns*(k-1)+(1:Ns),:) = s4(:,[T4-(k-1):T4,1:T4-k]); end
 u4             = Wppca * Q * s4_;   % enoders (test) / prediction
 
 % ICA
 [~,ui4,Wica,~,G] = postprocessing(u,u4,label4,ica_rep,ica_eta);
 figure_encoders(ui4,label4,1:T4/10);
 drawnow
 
 err_cat(:,h) = [1-max(G)./(sum(G)+0.001)];
 
 %--------------------------------------------------------------------------------
 % test prediction error
 
 % theory
 err_s1.th(1,h) = trace(Sigmas) - trace(Wopt*Sigmase*Wopt') + Kp*Ns/T * trace(Wopt*(Sigmas-Sigmase)*Wopt');
 err_s1.th(2,h) = trace(Sigmas) - trace(Wopt*Sigmase*Wopt');
 err_s1.th(3,h) = Kp*Ns/T * trace(Wopt*(Sigmas-Sigmase)*Wopt');
 % empirical
 err_s1.em(1,h) = mean(sum((s2 - Wppca'*u2).^2));
 err_s1.em(2,h) = mean(sum((s - Wppca'*u).^2));
 err_s1.em(3,h) = err_s1.em(1,h) - err_s1.em(2,h);
 
 %--------------------------------------------------------------------------------
 % system parameter identification
 
 Omega_inv      = (A'*A+eye(Nu)*10^(-4))^(-1)*A'*Wppca';           % ambiguity factor (coordinate transformation)
 [qA,qB,qSigmas,qSigmax,qSigmao,qSigmaz] = calculate_estimated_parameters(s,u,uc,Wppca,Omega_inv,prior_x); % this function is for ascending order
 
 err_param(1,h) = norm(A     -qA,     'fro')^2 / max(norm(A,     'fro')^2, norm(qA,     'fro')^2);
 err_param(2,h) = norm(B     -qB,     'fro')^2 / max(norm(B,     'fro')^2, norm(qB,     'fro')^2);
 err_param(3,h) = norm(Sigmas-qSigmas,'fro')^2 / max(norm(Sigmas,'fro')^2, norm(qSigmas,'fro')^2);
 err_param(4,h) = norm(Sigmax-qSigmax,'fro')^2 / max(norm(Sigmax,'fro')^2, norm(qSigmax,'fro')^2);
 err_param(5,h) = norm(Sigmao-qSigmao,'fro')^2 / max(norm(Sigmao,'fro')^2, norm(qSigmao,'fro')^2);
 err_param(6,h) = norm(Sigmaz-qSigmaz,'fro')^2 / max(norm(Sigmaz,'fro')^2, norm(qSigmaz,'fro')^2);
 
end

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------
% categorization error

% save results
csvwrite([dir 'figs1a_err_cat_' num2str(seed) '.csv'],[1:NT; err_cat])

%--------------------------------------------------------------------------------
% parameter estimation error

% save results
csvwrite([dir 'figs1a_err_param_' num2str(seed) '.csv'],[1:NT; err_param])

% show results
figure()
subplot(1,2,1)
plot(log10([(1:10)*10^3 (2:NT-9)*10^4]), err_param(1,:)' *100,'k+'), hold on
plot(log10([(1:10)*10^3 (2:NT-9)*10^4]), err_param(2,:)' *100,'r+')
plot(log10([(1:10)*10^3 (2:NT-9)*10^4]), err_param(3,:)' *100,'g+')
plot(log10([(1:10)*10^3 (2:NT-9)*10^4]), err_param(4,:)' *100,'b+')
plot(log10([(1:10)*10^3 (2:NT-9)*10^4]), err_param(5,:)' *100,'y+'), hold off
title('parameter estimation error')

%--------------------------------------------------------------------------------
% test prediction error

% save results
csvwrite([dir 'figs1a_err_pred_'  num2str(seed) '.csv'],[1:3,-(1:3); [err_s1.em', err_s1.th']/norm_s2])

t       = [(1:10)*10^3,(2:NT-9)*10^4];
norm_s2 = mean(sum(s2.^2));

% show results
subplot(1,2,2)
semilogx(t, err_s1.th(1,:)/norm_s2, '-g', t, err_s1.em(1,:)/norm_s2, '+g')
axis([0 10^5 0.4 0.8])
title('test prediction error')
drawnow

%--------------------------------------------------------------------------------

