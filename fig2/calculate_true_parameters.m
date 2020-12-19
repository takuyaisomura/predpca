
%--------------------------------------------------------------------------------

% calculate_true_parameters.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-30

%--------------------------------------------------------------------------------

function [A,B,Sigmas,Sigmap,Sigmao,Sigmaz] = calculate_true_parameters(s,x,prior_x);
T      = length(s(1,:));
Nx     = length(x(:,1));
A      = (s*x') * (x*x'+eye(Nx)*prior_x)^(-1);
B      = (x(:,[2:T,1])*x') * (x*x'+eye(Nx)*prior_x)^(-1);
Sigmas = s*s' / T;
Sigmap = x*x' / T;
Sigmao = Sigmas - A * Sigmap * A';
Sigmaz = Sigmap - B * Sigmap * B';

