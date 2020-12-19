
%--------------------------------------------------------------------------------

% calculate_true_parameters.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-31

%--------------------------------------------------------------------------------

function [qxp,qxc] = bayesian_filter(s, A, B, Sigmao, Sigmaz, x0, x1, C0, C1)
T        = length(s(1,:));
Ns       = length(s(:,1));
Nx       = length(x0);
qxp      = zeros(Nx,T);
qxc      = zeros(Nx,T);
qxp(:,1) = x0;
qxp(:,2) = x1;
qxc(:,1) = x0;
qxc(:,2) = x1;
Cp       = C1;
Ccold    = C0;
Cc       = C1;

for t = 3:T
    qxp(:,t) = B * kron(qxc(:,t-1),qxc(:,t-2));    % state prediction
    Cp       = B * kron(Cc,Ccold) * B' + Sigmaz;   % error covariance prediction
%    Cp       = B * kron(diag(diag(Cc)),diag(diag(Ccold))) * B' + Sigmaz;   % error covariance prediction
    G        = Cp * A' / (A*Cp*A'+Sigmao);         % optimal kalman gain
    qxc(:,t) = qxp(:,t) + G * (s(:,t)-A*qxp(:,t)); % state update
    Ccold    = Cc;
    Cc       = (eye(Nx)-G*A) * Cp;                 % error covariance update
end

