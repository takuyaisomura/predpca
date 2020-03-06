
%--------------------------------------------------------------------------------

% mlest.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [se1,se2,Q] = mlest(st1,phi1,phi2,prior_phi)

Nphi = length(phi1(:,1));
Q    = (st1*phi1') * (phi1*phi1'+eye(Nphi)*prior_phi)^(-1);
se1  = Q * phi1;
se2  = Q * phi2;

