
%--------------------------------------------------------------------------------

% tae.m
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

%--------------------------------------------------------------------------------

function [u,u2,sp,sp2,w1,w2,w3,w4,w5,w6] = tae(s, s2, st, st2, num_rep, alpha, p_do, eta, beta1, beta2, NL1, NL2, NL3, dirname)

% initialization
T       = length(s(1,:));
T2      = length(s2(1,:));
T1      = T/10;
Ns      = length(s(:,1));

N0      = Ns;
N1      = NL1;
N2      = NL2;
N3      = NL3;
N4      = NL2;
N5      = NL1;
N6      = Ns;

w1      = [randn(N1,N0)/sqrt(N0/2),zeros(N1,1)];
w2      = [randn(N2,N1)/sqrt(N1/2),zeros(N2,1)];
w3      = [randn(N3,N2)/sqrt(N2/2),zeros(N3,1)];
w4      = [randn(N4,N3)/sqrt(N3/2),zeros(N4,1)];
w5      = [randn(N5,N4)/sqrt(N4/2),zeros(N5,1)];
w6      = [randn(N6,N5)/sqrt(N5/2),zeros(N6,1)];
m_dw1   = zeros(N1,N0+1);
m_dw2   = zeros(N2,N1+1);
m_dw3   = zeros(N3,N2+1);
m_dw4   = zeros(N4,N3+1);
m_dw5   = zeros(N5,N4+1);
m_dw6   = zeros(N6,N5+1);
Var_dw1 = ones(N1,N0+1) * 0.01;
Var_dw2 = ones(N2,N1+1) * 0.01;
Var_dw3 = ones(N3,N2+1) * 0.01;
Var_dw4 = ones(N4,N3+1) * 0.01;
Var_dw5 = ones(N5,N4+1) * 0.01;
Var_dw6 = ones(N6,N5+1) * 0.01;
w1old   = w1;
w2old   = w2;
w3old   = w3;
w4old   = w4;
w5old   = w5;
w6old   = w6;

fileID  = fopen([dirname 'log.txt'],'w');

%--------------------------------------------------------------------------------
% backpropagation

fprintf(1,'backpropagation\n');
for h = 1:num_rep
 % state update
 t       = rem((1:T1) + randi([0 T-1]),T) + 1;
 data    = [s(:,t); ones(1,T1)]; % input
 target  = st(:,t);              % target
 w1probs = w1*data;    w1probs = w1probs.*(rand(N1,T1)>=p_do); hev1 = (w1probs>0)*1+(w1probs<=0)*alpha; w1probs = [w1probs.*hev1; ones(1,T1)];
 w2probs = w2*w1probs; w2probs = w2probs.*(rand(N2,T1)>=p_do); hev2 = (w2probs>0)*1+(w2probs<=0)*alpha; w2probs = [w2probs.*hev2; ones(1,T1)];
 w3probs = w3*w2probs;                                         hev3 = ones(N3,T1);                      w3probs = [w3probs.*hev3; ones(1,T1)];
 w4probs = w4*w3probs; w4probs = w4probs.*(rand(N4,T1)>=p_do); hev4 = (w4probs>0)*1+(w4probs<=0)*alpha; w4probs = [w4probs.*hev4; ones(1,T1)];
 w5probs = w5*w4probs; w5probs = w5probs.*(rand(N5,T1)>=p_do); hev5 = (w5probs>0)*1+(w5probs<=0)*alpha; w5probs = [w5probs.*hev5; ones(1,T1)];
 dataout = w6*w5probs;
 
 % compute error
 eps6    = target - dataout;
 eps5    = hev5 .* (w6(:,1:N5)'*eps6);
 eps4    = hev4 .* (w5(:,1:N4)'*eps5);
 eps3    = hev3 .* (w4(:,1:N3)'*eps4);
 eps2    = hev2 .* (w3(:,1:N2)'*eps3);
 eps1    = hev1 .* (w2(:,1:N1)'*eps2);
 
 % gradient
 dw1     = eps1*data'   /T1;
 dw2     = eps2*w1probs'/T1;
 dw3     = eps3*w2probs'/T1;
 dw4     = eps4*w3probs'/T1;
 dw5     = eps5*w4probs'/T1;
 dw6     = eps6*w5probs'/T1;
 
 % mean
 m_dw1   = beta1 * m_dw1 + (1-beta1) * dw1;
 m_dw2   = beta1 * m_dw2 + (1-beta1) * dw2;
 m_dw3   = beta1 * m_dw3 + (1-beta1) * dw3;
 m_dw4   = beta1 * m_dw4 + (1-beta1) * dw4;
 m_dw5   = beta1 * m_dw5 + (1-beta1) * dw5;
 m_dw6   = beta1 * m_dw6 + (1-beta1) * dw6;
 
 % variance
 Var_dw1 = beta2 * Var_dw1 + (1-beta2) * (dw1.^2);
 Var_dw2 = beta2 * Var_dw2 + (1-beta2) * (dw2.^2);
 Var_dw3 = beta2 * Var_dw3 + (1-beta2) * (dw3.^2);
 Var_dw4 = beta2 * Var_dw4 + (1-beta2) * (dw4.^2);
 Var_dw5 = beta2 * Var_dw5 + (1-beta2) * (dw5.^2);
 Var_dw6 = beta2 * Var_dw6 + (1-beta2) * (dw6.^2);
 
 % parameter update with the Adam optimizer
 w1      = w1 + eta * m_dw1/(1-beta1^h) ./ (sqrt(Var_dw1/(1-beta2^h)) + 10^(-8));
 w2      = w2 + eta * m_dw2/(1-beta1^h) ./ (sqrt(Var_dw2/(1-beta2^h)) + 10^(-8));
 w3      = w3 + eta * m_dw3/(1-beta1^h) ./ (sqrt(Var_dw3/(1-beta2^h)) + 10^(-8));
 w4      = w4 + eta * m_dw4/(1-beta1^h) ./ (sqrt(Var_dw4/(1-beta2^h)) + 10^(-8));
 w5      = w5 + eta * m_dw5/(1-beta1^h) ./ (sqrt(Var_dw5/(1-beta2^h)) + 10^(-8));
 w6      = w6 + eta * m_dw6/(1-beta1^h) ./ (sqrt(Var_dw6/(1-beta2^h)) + 10^(-8));
 
 % calculate deviation
 err     = mean(sum(eps6'.^2)) / mean(sum(target'.^2));
 dev_w1  = norm(w1-w1old,'fro')^2 / norm(w1,'fro')^2;
 dev_w2  = norm(w2-w2old,'fro')^2 / norm(w2,'fro')^2;
 dev_w3  = norm(w3-w3old,'fro')^2 / norm(w3,'fro')^2;
 dev_w4  = norm(w4-w4old,'fro')^2 / norm(w4,'fro')^2;
 dev_w5  = norm(w5-w5old,'fro')^2 / norm(w5,'fro')^2;
 dev_w6  = norm(w6-w6old,'fro')^2 / norm(w6,'fro')^2;
 fprintf(1,'h=%d/%d, time = %.1f min, err=%f, devs:%f, %f, %f, %f, %f, %f\n', h, num_rep, toc/60, err, dev_w1, dev_w2, dev_w3, dev_w4, dev_w5, dev_w6);
 fprintf(fileID,'h=%d/%d, time = %.1f min, err=%f, devs:%f, %f, %f, %f, %f, %f\n', h, num_rep, toc/60, err, dev_w1, dev_w2, dev_w3, dev_w4, dev_w5, dev_w6);
 w1old   = w1;
 w2old   = w2;
 w3old   = w3;
 w4old   = w4;
 w5old   = w5;
 w6old   = w6;
 
 if (rem(h,100) == 0)
  % compute output
  fprintf(1,'compute output\n');
  fprintf(fileID,'compute output\n');
  data    = [s; ones(1,T)]; % input
  w1probs = w1*data;    w1probs = w1probs*(1-p_do); hev1 = (w1probs>0)*1+(w1probs<=0)*alpha; w1probs = [w1probs.*hev1; ones(1,T)];
  w2probs = w2*w1probs; w2probs = w2probs*(1-p_do); hev2 = (w2probs>0)*1+(w2probs<=0)*alpha; w2probs = [w2probs.*hev2; ones(1,T)];
  w3probs = w3*w2probs;                             hev3 = ones(N3,T);                       w3probs = [w3probs.*hev3; ones(1,T)];
  w4probs = w4*w3probs; w4probs = w4probs*(1-p_do); hev4 = (w4probs>0)*1+(w4probs<=0)*alpha; w4probs = [w4probs.*hev4; ones(1,T)];
  w5probs = w5*w4probs; w5probs = w5probs*(1-p_do); hev5 = (w5probs>0)*1+(w5probs<=0)*alpha; w5probs = [w5probs.*hev5; ones(1,T)];
  sp      = w6*w5probs;
  u       = w3probs(1:N3,:);
  err     = mean(sum((st-sp)'.^2)) / mean(sum(st'.^2));
  fprintf(1,'err_train = %f\n', err);
  fprintf(fileID,'err_train = %f\n', err);
  
  data    = [s2; ones(1,T2)]; % input
  w1probs = w1*data;    w1probs = w1probs*(1-p_do); hev1 = (w1probs>0)*1+(w1probs<=0)*alpha; w1probs = [w1probs.*hev1; ones(1,T2)];
  w2probs = w2*w1probs; w2probs = w2probs*(1-p_do); hev2 = (w2probs>0)*1+(w2probs<=0)*alpha; w2probs = [w2probs.*hev2; ones(1,T2)];
  w3probs = w3*w2probs;                             hev3 = ones(N3,T2);                      w3probs = [w3probs.*hev3; ones(1,T2)];
  w4probs = w4*w3probs; w4probs = w4probs*(1-p_do); hev4 = (w4probs>0)*1+(w4probs<=0)*alpha; w4probs = [w4probs.*hev4; ones(1,T2)];
  w5probs = w5*w4probs; w5probs = w5probs*(1-p_do); hev5 = (w5probs>0)*1+(w5probs<=0)*alpha; w5probs = [w5probs.*hev5; ones(1,T2)];
  sp2     = w6*w5probs;
  u2      = w3probs(1:N3,:);
  err     = mean(sum((st2-sp2)'.^2)) / mean(sum(st2'.^2));
  fprintf(1,'err_test = %f\n', err);
  fprintf(fileID,'err_test = %f\n', err);
 end
end

fclose(fileID);

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

