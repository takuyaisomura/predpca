
%--------------------------------------------------------------------------------

% calculate_estimated_parameters.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [qA,qB,qSigmas,qSigmax,qSigmao,qSigmaz] = calculate_estimated_parameters(s1,u1,uc1,Wppca,Omega_inv,prior_x)
T1      = length(s1(1,:));
Nu      = length(u1(:,1));
v1      = Omega_inv * u1;
vc1     = Omega_inv * uc1;
qA      = Wppca' * (Omega_inv+eye(Nu)*10^(-4))^(-1);             % observation matrix
qB      = (v1(:,[2:T1,1])*v1') * (v1*v1'+eye(Nu)*prior_x)^(-1);  % transiiton matrix
qSigmas = (s1*s1') / T1;                                         % actual input covariance
qSigmax = (qB+eye(Nu)*10^(-4))^(-1) * (vc1(:,[2:T1,1])*vc1'/T1); %
qSigmax = (qSigmax + qSigmax') / 2;                              % hidden state covariance
qSigmaz = qSigmax - qB * qSigmax * qB';                          % system noise covariance
qSigmao = qSigmas - qA * qSigmax * qA';                          % observation noise covariance

