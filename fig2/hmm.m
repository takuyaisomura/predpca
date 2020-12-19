
%--------------------------------------------------------------------------------

% hmm.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-27

%--------------------------------------------------------------------------------

function [qx,qx2,qa,qA,qlnA,qb,qB,qlnB] = hmm(s,s2,qa,qb,rep,l1,l2,eta1,eta2,amp1,amp2,showimg)

% initialization
Ns   = length(s(:,1));
Nx   = length(qa(1,:));
T    = length(s(1,:));
T2   = length(s2(1,:));
qx   = zeros(Nx,T);
qx2  = zeros(Nx,T2);
qA   = zeros(Ns*2,Nx);
qlnA = zeros(Ns*2,Nx);
if (length(qb(1,:)) == Nx)
  num_step = 1;
  qB   = zeros(Nx,Nx);
  qlnB = zeros(Nx,Nx);
elseif (length(qb(1,:)) == Nx*Nx)
  num_step = 2;
  qB   = zeros(Nx,Nx*Nx);
  qlnB = zeros(Nx,Nx*Nx);
else
  fprintf(1,'error\n')
end

fprintf(1,'HMM (num_step = %d)\n', num_step)
% training
for h = 1:rep
  fprintf(1,'h = %d/%d\n',h,rep)
  % compute parameters
  qA(1:Ns,:)        = qa(1:Ns,:)      ./ (qa(1:Ns,:)+qa(Ns+1:Ns*2,:));
  qA(Ns+1:Ns*2,:)   = qa(Ns+1:Ns*2,:) ./ (qa(1:Ns,:)+qa(Ns+1:Ns*2,:));
  qB                = qb              ./ (ones(Nx,1)*(ones(1,Nx)*qb));
  qlnA(1:Ns,:)      = psi(qa(1:Ns,:))      - psi(qa(1:Ns,:)+qa(Ns+1:Ns*2,:));
  qlnA(Ns+1:Ns*2,:) = psi(qa(Ns+1:Ns*2,:)) - psi(qa(1:Ns,:)+qa(Ns+1:Ns*2,:));
  qlnB              = psi(qb)              - psi(ones(Nx,1)*(ones(1,Nx)*qb));
  qlnA_s            = qlnA'*[s;1-s];
  
  % state posterior update
  v                 = qlnA_s;
  exp_v             = exp((v-ones(Nx,1)*max(v))*amp1);
  qx                = exp_v ./ (ones(Nx,1)*sum(exp_v));
  for t = 1:l1
    if (num_step == 1)
      v                 = (1-eta1)*v + eta1*(qlnA_s + qlnB*qx(:,[T,1:T-1]) + qlnB'*qx(:,[2:T,1]));
    elseif (num_step == 2)
      temp1             = qlnB*(kron(qx(:,[T,1:T-1]),ones(Nx,1)).*kron(ones(Nx,1),qx(:,[T-1:T,1:T-2])));
      temp2             = kron(eye(Nx),ones(1,Nx))*((qlnB'*qx(:,[2:T,1])).*kron(ones(Nx,1),qx(:,[T,1:T-1])));
      temp3             = kron(ones(1,Nx),eye(Nx))*((qlnB'*qx(:,[2:T,1])).*kron(qx(:,[2:T,1]),ones(Nx,1)));
      v                 = (1-eta1)*v + eta1*(qlnA_s + temp1 + temp2 + temp3);
    end
    exp_v             = exp((v-ones(Nx,1)*max(v))*amp1);
    qx                = exp_v ./ (ones(Nx,1)*sum(exp_v));
  end
  
  % parameter posterior update
  qa                = qa + [s;1-s]*qx';
  if (num_step == 1)
    qb                = qb + qx*qx(:,[T,1:T-1])';
  elseif (num_step == 2)
    qb                = qb + qx*(kron(qx(:,[T,1:T-1]),ones(Nx,1)).*kron(ones(Nx,1),qx(:,[T-1:T,1:T-2])))';
  end
  qa                = (mean(mean(qA)) * diag(mean(qA))^(-1) * qa')'; % normalization
  qb                = mean(mean(qB')) * diag(mean(qB'))^(-1) * qb;   % normalization
%  qb                = qb * mean(mean(qB)) * diag(mean(qB))^(-1);   % normalization
%  qa                = (diag(1+0.1-mean(qx')) * qa')';
%  qb                = diag(1+0.1-mean(qx'))* qb;
  % this treatment is to prevent qA and qB from converging to a singular mapping
  if (showimg == 1)
    subplot(5,1,1), image(qx(:,9000:9200)*100)
    subplot(5,1,2), image(qx(:,9200:9400)*100)
    subplot(5,1,3), image(qx(:,9400:9600)*100)
    subplot(5,1,4), image(qx(:,9600:9800)*100)
    subplot(5,1,5), image(qx(:,9800:10000)*100)
    drawnow
  end
end

% compute state posterior for test sequence
qlnA_s            = qlnA'*[s2;1-s2];
v                 = qlnA_s;
exp_v             = exp((v-ones(Nx,1)*max(v))*amp2);
qx2               = exp_v ./ (ones(Nx,1)*sum(exp_v));
for t = 1:l2
  if (num_step == 1)
    v                 = (1-eta2)*v + eta2*(qlnA_s + qlnB*qx2(:,[T2,1:T2-1]) + qlnB'*qx2(:,[2:T2,1]));
  elseif (num_step == 2)
    temp1             = qlnB*(kron(qx2(:,[T2,1:T2-1]),ones(Nx,1)).*kron(ones(Nx,1),qx2(:,[T2-1:T2,1:T2-2])));
    temp2             = kron(eye(Nx),ones(1,Nx))*((qlnB'*qx2(:,[2:T2,1])).*kron(ones(Nx,1),qx2(:,[T2,1:T2-1])));
    temp3             = kron(ones(1,Nx),eye(Nx))*((qlnB'*qx2(:,[2:T2,1])).*kron(qx2(:,[2:T2,1]),ones(Nx,1)));
    v                 = (1-eta2)*v + eta2*(qlnA_s + temp1 + temp2 + temp3);
  end
  exp_v             = exp((v-ones(Nx,1)*max(v))*amp2);
  qx2               = exp_v ./ (ones(Nx,1)*sum(exp_v));
end

