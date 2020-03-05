
%--------------------------------------------------------------------------------

% kalman_filter.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [qxp,qxc] = kalman_filter(s, A, B, Sigmao, Sigmaz, x0, C0)
T        = length(s(1,:));
Ns       = length(A(:,1));
Nx       = length(A(1,:));
qxp      = zeros(Nx,T);
qxc      = zeros(Nx,T);
qxp(:,1) = x0;
qxc(:,1) = x0;
Cp       = C0;
Cc       = C0;

for t = 2:T
 qxp(:,t) = B * qxc(:,t-1);                     % state prediction
 Cp       = B * Cc * B' + Sigmaz;               % error covariance prediction
 G        = Cp * A' * (A*Cp*A'+Sigmao)^(-1);    % optimal kalman gain
 qxc(:,t) = qxp(:,t) + G * (s(:,t)-A*qxp(:,t)); % state update
 Cc       = (eye(Nx)-G*A) * Cp;                 % error covariance update
end
