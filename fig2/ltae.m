
%--------------------------------------------------------------------------------

% ltae.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-6-2

%--------------------------------------------------------------------------------

function [u,u2,uc,E,D] = ltae(s,s2)

% initialization
Ns         = length(s(:,1));
Nu         = 10;
T          = length(s(1,:));
T2         = length(s2(1,:));

Sigmas     = s*s'/T;
x          = Sigmas^(-1/2) * s(:,[T,1:T-1]);
x2         = Sigmas^(-1/2) * s2(:,[T2,1:T2-1]);
y          = Sigmas^(-1/2) * s;
K          = y*x'/T;
[U,S,V]    = svd(K');
E          = S(1:Nu,1:Nu) * U(:,1:Nu)' * Sigmas^(-1/2); % encoding matrix
D          = Sigmas^(1/2) * V(:,1:Nu);                  % decoding matrix
u          = E * s;
u2         = E * s2;
uc         = V(:,1:Nu)' * y;

