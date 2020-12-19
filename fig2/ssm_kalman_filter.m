
%--------------------------------------------------------------------------------

% ssm_kalman_filter.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-30

%--------------------------------------------------------------------------------

function [qxp1,qxc1,qxp2,qxc2,qA,qB] = ssm_kalman_filter(s, s2, Ainit, Binit, Sigmao, Sigmaz, maxrep, eta, prior_x)

T      = length(s(1,:));
T1     = T / 10;
Ns     = length(Ainit(:,1));
Nx     = length(Ainit(1,:));

% initialization
qA     = Ainit;
qB     = Binit;
qAold  = qA;
qBold  = qB;

% expectation-maximization algorithm
for rep = 1:maxrep
 % state update
 t = (1:T1) + rem(rep-1,10) * T1;
 [qxp1,qxc1] = kalman_filter(s(:,t), qA, qB, Sigmao, Sigmaz, zeros(Nx,1), eye(Nx));                % state update
 % parameter update
 qA          = (1-eta)*qA + eta*(s(:,t)          *qxp1') * (qxp1*qxp1'+eye(Nx)*prior_x*T1/T)^(-1); % parameter update
 qB          = (1-eta)*qB + eta*(qxp1(:,[2:T1,1])*qxp1') * (qxp1*qxp1'+eye(Nx)*prior_x*T1/T)^(-1); % parameter update
 
 dev_A       = norm(qA-qAold,'fro')^2 / norm(qA,'fro')^2;
 dev_B       = norm(qB-qBold,'fro')^2 / norm(qB,'fro')^2;
 qAold       = qA;
 qBold       = qB;
 fprintf(1,'rep = %d/%d, dev_A = %f, dev_B = %f\n', rep, maxrep, dev_A, dev_B);
end

% output results
[qxp1,qxc1] = kalman_filter(s,  qA, qB, Sigmao, Sigmaz, zeros(Nx,1), eye(Nx)); % state update
[qxp2,qxc2] = kalman_filter(s2, qA, qB, Sigmao, Sigmaz, zeros(Nx,1), eye(Nx)); % state update

