
%--------------------------------------------------------------------------------

% random_attractor.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-10-25

%--------------------------------------------------------------------------------

function [x,x2,psi,psi2,z,z2,R,B,meanx0,Covx0] = canonical_nonlinear_system(Nx,Npsi,T,T2,sigma_z)

x    = zeros(Nx,T);
x2   = zeros(Nx,T2);
psi  = zeros(Npsi,T);
psi2 = zeros(Npsi,T2);
z    = randn(Nx,T) * sigma_z;
z2   = randn(Nx,T2) * sigma_z;
R    = randn(Npsi,Nx+1)/sqrt(Nx+1);
B    = randn(Nx,Npsi)/sqrt(Npsi);

%--------------------------------------------------------------------------------

for h = 1:100
  x(:,1)   = randn(Nx,1);
  psi(:,1) = tanh(R * [x(:,1); 1]);
  for t = 2:T
    x(:,t)   = B * psi(:,t-1) + z(:,t);
    psi(:,t) = tanh(R * [x(:,t); 1]);
  end
  mean_psi = mean(psi')';
  B = B * 0.9 + cov(x(:,0.1*T:T)')^(-1/2) * B * 0.1 - B*mean_psi*mean_psi'*0.1;
end

x(:,1)   = randn(Nx,1);
psi(:,1) = tanh(R * [x(:,1); 1]);
for t = 2:T
  x(:,t)   = B * psi(:,t-1) + z(:,t);
  psi(:,t) = tanh(R * [x(:,t); 1]);
end

x2(:,1)   = randn(Nx,1);
psi2(:,1) = tanh(R * [x2(:,1); 1]);
for t = 2:T2
  x2(:,t)   = B * psi2(:,t-1) + z2(:,t);
  psi2(:,t) = tanh(R * [x2(:,t); 1]);
end

%--------------------------------------------------------------------------------

meanx0 = mean(x')';
Covx0  = cov(x');
x      = Covx0^(-1/2) * (x - meanx0*ones(1,T));
x2     = Covx0^(-1/2) * (x2 - meanx0*ones(1,T2));

%--------------------------------------------------------------------------------
