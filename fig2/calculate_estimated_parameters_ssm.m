
%--------------------------------------------------------------------------------

% calculate_estimated_parameters.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-30

%--------------------------------------------------------------------------------

function [qA,qB,qSigmas,qSigmap,qSigmao,qSigmaz] = calculate_estimated_parameters_ssm(s1,u1,uc1,qA,qB,Omega_inv,prior_psi)
T1      = length(s1(1,:));
Nu      = length(u1(:,1));
v1      = Omega_inv * u1;
vc1     = Omega_inv * uc1;
qA      = qA * (Omega_inv+eye(Nu)*10^(-4))^(-1);                   % observation matrix
qB      = Omega_inv * qB * (Omega_inv+eye(Nu)*10^(-4))^(-1);       % transition matrix
qSigmas = (s1*s1') / T1;                                           % actual input covariance
qSigmap = (vc1*vc1') / T1;                                         % hidden basis covariance
qSigmaz = qSigmap - qB * qSigmap * qB';                            % system noise covariance
qSigmao = qSigmas - qA * qSigmap * qA';                            % observation noise covariance

