
%--------------------------------------------------------------------------------

% supplfig1.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-10-25

%--------------------------------------------------------------------------------

clear
T       = 100000;
T2      = 100000;
Nx      = 3;
Npsi    = 100;
Ns      = 200;
sigma_z = 0.001;
sigma_o = 0.1;
Type    = 1;
dirname = '';
if (Type == 1)
  Nx = 10;
else
  Nx = 3;
end

seed = 0;
rng(1000000+seed);      % set seed for reproducibility

%--------------------------------------------------------------------------------

if (Type == 1)
  [x,x2,psi,psi2,z,z2,R,B,meanx0,Covx0] = canonical_nonlinear_system(Nx,Npsi,T,T2,sigma_z);
elseif (Type == 2)
  [x,x2,meanx0,Covx0] = lorenz_attractor(Nx,T,T2);
  R    = randn(Npsi,Nx+1)/sqrt(Nx+1);
  psi  = tanh(R * [x; ones(1,T)]);
  psi2 = tanh(R * [x2; ones(1,T2)]);
end

[U,S,V] = svd(randn(Ns,Npsi));
A       = U*sign(S)*V';
omega   = randn(Ns,T)*sigma_o;
omega2  = randn(Ns,T2)*sigma_o;
s       = A * psi + omega;
s2      = A * psi2 + omega2;
s       = s - mean(s')'*ones(1,T);
s2      = s2 - mean(s2')'*ones(1,T2);

%--------------------------------------------------------------------------------

Kp        = 10;
prior_s_  = 1;
[s_,s2_,qs,qs2,Q] = maximum_likelihood_estimator(s(:,[T,1:T-1]),s2(:,[T2,1:T2-1]),s,Kp,prior_s_);

[Cs,~,Ls] = pca(qs');
qpsi      = Cs(:,1:Npsi)'*qs;
qpsi2     = Cs(:,1:Npsi)'*qs2;
qpsic     = Cs(:,1:Npsi)'*s;
qPsi      = (qpsi(:,[2:T,1])*qpsi') / (qpsi*qpsi');
[U,S,V]   = svd(qPsi);
qSigmap   = (V*(S+eye(Npsi)*0.001)^-1*U') * (qpsic(:,[2:T,1])*qpsic'/T);
qSigmap   = (qSigmap + qSigmap')/2;
[Cp,Lp]   = pcacov(qSigmap);

[C,~,L]   = pca(psi');

subplot(1,2,1)
plot(1:Npsi,L,1:Npsi,Lp,1:Npsi,Ls(1:Npsi))
subplot(1,2,2)
plot(1:Npsi,L-[L(2:Npsi);0],1:Npsi,Lp-[Lp(2:Npsi);0])

qx     = diag(Lp(1:Nx))^(-1/2) * Cp(:,1:Nx)'*qpsi;
qx2    = diag(Lp(1:Nx))^(-1/2) * Cp(:,1:Nx)'*qpsi2;

Omegax = (x*qx') / (qx*qx');
qx     = Omegax * qx;
qx2    = Omegax * qx2;

Omegap = (psi*qpsi') / (qpsi*qpsi'+eye(Npsi));
qpsi   = Omegap * qpsi;
qpsi2  = Omegap * qpsi2;

corr(x2',qx2')

%--------------------------------------------------------------------------------

T0 = 10001;
T1 = 20000;
figure()
subplot(1,2,1)
plot3(x2(1,T0:T1),x2(2,T0:T1),x2(3,T0:T1))
subplot(1,2,2)
plot3(qx2(1,T0:T1),qx2(2,T0:T1),qx2(3,T0:T1))

figure()
plot(T0:T1,x2(1,T0:T1),T0:T1,qx2(1,T0:T1))

if (Type == 1)
  B    = (x(:,[2:T,1])*psi')/(psi*psi'+eye(Npsi));
  qB   = (qx(:,[2:T,1])*qpsi')/(qpsi*qpsi'+eye(Npsi));
  figure()
  subplot(2,1,1)
  image(abs(B*400))
  subplot(2,1,2)
  image(abs(qB*400))
  csvwrite([dirname 'states_', num2str(seed), '.csv'],[1:Nx,1:Nx; x2(:,1:T2*0.1)',qx2(:,1:T2*0.1)'])
  csvwrite([dirname 'correlation_', num2str(seed), '.csv'],[1:Nx; corr(x2',qx2')])
  csvwrite([dirname 'eigenvalues_', num2str(seed), '.csv'],[1,2; L,Lp])
  csvwrite([dirname 'param_B_', num2str(seed), '.csv'],[1:Npsi; B; qB])
elseif (Type == 2)
  xx  = [ones(1,T); x; x.^2; x(2,:).*x(3,:); x(3,:).*x(1,:); x(1,:).*x(2,:)];
  qxx = [ones(1,T); qx; qx.^2; qx(2,:).*qx(3,:); qx(3,:).*qx(1,:); qx(1,:).*qx(2,:)];
  B   = ((x(:,[2:T,1])-x)*xx')/(xx*xx'+eye(10*Nx/3));
  qB  = ((qx(:,[2:T,1])-qx)*qxx')/(qxx*qxx'+eye(10*Nx/3));
  figure()
  subplot(2,1,1)
  image(abs(B*2000))
  subplot(2,1,2)
  image(abs(qB*2000))
  csvwrite([dirname 'states_Lorenz_', num2str(seed), '.csv'],[1:Nx,1:Nx; x2(:,1:T2*0.1)',qx2(:,1:T2*0.1)'])
  csvwrite([dirname 'correlation_Lorenz_', num2str(seed), '.csv'],[1:Nx; corr(x2',qx2')])
  csvwrite([dirname 'eigenvalues_Lorenz_', num2str(seed), '.csv'],[1,2; L,Lp])
  csvwrite([dirname 'param_B_Lorenz_', num2str(seed), '.csv'],[1:10*Nx/3; B; qB])
end

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

