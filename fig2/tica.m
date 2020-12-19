
%--------------------------------------------------------------------------------

% tica.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-31

%--------------------------------------------------------------------------------

function [u,u2] = tica(s,s2)

% initialization
Ns         = length(s(:,1));
Nu         = 10;
T          = length(s(1,:));
T2         = length(s2(1,:));

[U,L]      = pcacov(cov(s'));
S          = diag(L)^(1/2);
M_TICA     = (S^(-1)*U'*s)*(S^(-1)*U'*s(:,[2:T,1]))'/T;
[Om,Lm]    = eig(M_TICA); % M_TICA = Om*Lm*Om^(-1)
m          = Om' * (S^(-1)*U'*s);
m2         = Om' * (S^(-1)*U'*s2);
u          = zeros(Nu,T);
u2         = zeros(Nu,T2);

i = 1;
j = 1;
for k = 1:Nu
  if (isreal(Lm(i,i)) == 1)
    u(j,:)  = real(m(i,:));
    u2(j,:) = real(m2(i,:));
    i       = i+1;
    j       = j+1;
    if (j>Nu), break, end
  else
    if (abs(real(Lm(i,i))) > abs(imag(Lm(i,i))))
      u(j,:)  = real(m(i,:));
      u2(j,:) = real(m2(i,:));
      j       = j+1;
      if (j>Nu), break, end
      u(j,:)  = imag(m(i,:));
      u2(j,:) = imag(m2(i,:));
      i       = i+2;
      j       = j+1;
      if (j>Nu), break, end
    else
      u(j,:)  = imag(m(i,:));
      u2(j,:) = imag(m2(i,:));
      j       = j+1;
      if (j>Nu), break, end
      u(j,:)  = real(m(i,:));
      u2(j,:) = real(m2(i,:));
      i       = i+2;
      j       = j+1;
      if (j>Nu), break, end
    end
  end
end

