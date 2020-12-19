
%--------------------------------------------------------------------------------

% calculate_true_parameters.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-31

%--------------------------------------------------------------------------------

function [qxp1,qxc1,qxp2,qxc2,qA,qB] = ssm_bayesian_filter(s,s2,Ainit,Binit,Sigmao,Sigmaz,num_rep,eta,prior_x)

T       = length(s(1,:));
T1      = T / 10;
T2      = length(s2(1,:));
Ns      = length(Ainit(:,1));
Nx      = length(Ainit(1,:));

% initialization
%qA      = randn(Ns,Nx)    * sqrt(1/Nx);
%qB      = randn(Nx,Nx*Nx) * sqrt(1/Nx/Nx);
qA      = Ainit;
qB      = Binit;
qAold   = qA;
qBold   = qB;
Var_A   = ones(Ns,Nx)    * 1;
Var_B   = ones(Nx,Nx*Nx) * 1;

% expectation-maximization algorithm
for rep = 1:num_rep
    % state update
    t = (1:T1) + rem(rep-1,10) * T1;
    [qxp1,qxc1] = bayesian_filter([s(:,t(1:60)),s(:,t)],qA,qB,Sigmao,Sigmaz,zeros(Nx,1),zeros(Nx,1),eye(Nx),eye(Nx)); % state update
    qxp1        = qxp1(:,61:T1+60);
    qxc1        = qxc1(:,61:T1+60);
    % parameter update
%    qxp1        = diag(std(qxp1'))^(-1) * qxp1;
%    qxc1        = diag(std(qxc1'))^(-1) * qxc1;
    psi         = kron(qxc1,ones(Nx,1)) .* kron(ones(Nx,1),qxc1(:,[T1,1:T1-1]));                       % compute basis
    qA          = (1-eta)*qA + eta*(s(:,t)*qxp1')          * (qxp1*qxp1'+eye(Nx)*prior_x*T1/T)^(-1);   % parameter update
    qB          = (1-eta)*qB + eta*(qxp1(:,[2:T1,1])*psi') * (psi*psi'+eye(Nx^2)*prior_x*T1/T)^(-1);   % parameter update
    
    dev_A       = norm(qA-qAold,'fro')^2 / norm(qA,'fro')^2;
    dev_B       = norm(qB-qBold,'fro')^2 / norm(qB,'fro')^2;
    qAold       = qA;
    qBold       = qB;
    fprintf(1,'rep = %d/%d, dev_A = %f, dev_B = %f\n', rep, num_rep, dev_A, dev_B);
end

% output results
[qxp1,qxc1] = bayesian_filter([s(:,1:60),s],qA,qB,Sigmao,Sigmaz,zeros(Nx,1),zeros(Nx,1),eye(Nx),eye(Nx)); % state update
qxp1        = qxp1(:,61:T+60);
qxc1        = qxc1(:,61:T+60);
[qxp2,qxc2] = bayesian_filter([s2(:,1:60),s2],qA,qB,Sigmao,Sigmaz,zeros(Nx,1),zeros(Nx,1),eye(Nx),eye(Nx)); % state update
qxp2        = qxp2(:,61:T2+60);
qxc2        = qxc2(:,61:T2+60);
