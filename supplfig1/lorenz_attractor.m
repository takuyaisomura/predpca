
%--------------------------------------------------------------------------------

% lorenz_attractor.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-10-25

%--------------------------------------------------------------------------------

function [x,x2,meanx0,Covx0] = lorenz_attractor(Nx,T,T2)

Nl  = Nx/3;
x   = zeros(Nx,T);
x2  = zeros(Nx,T2);

eps = 0.01;
p   = 10;
r   = 28;
b   = 8/3;

%--------------------------------------------------------------------------------

x(:,1) = rand(Nx,1)*10;
for i = 1:Nl
  y      = zeros(3,T);
  y(:,1) = x(3*(i-1)+1:3*i,1);
  for t = 2:T
    y(1,t) = y(1,t-1) + eps*(-p * (y(1,t-1) - y(2,t-1)));
    y(2,t) = y(2,t-1) + eps*(-y(1,t-1) * y(3,t-1) + r*y(1,t-1) - y(2,t-1));
    y(3,t) = y(3,t-1) + eps*(y(1,t-1) * y(2,t-1) - b*y(3,t-1));
  end
  x(3*(i-1)+1:3*i,:) = y;
end

x2(:,1) = x(:,T);
for i = 1:Nl
  y      = zeros(3,T2);
  y(:,1) = x2(3*(i-1)+1:3*i,1);
  for t = 2:T
    y(1,t) = y(1,t-1) + eps*(-p * (y(1,t-1) - y(2,t-1)));
    y(2,t) = y(2,t-1) + eps*(-y(1,t-1) * y(3,t-1) + r*y(1,t-1) - y(2,t-1));
    y(3,t) = y(3,t-1) + eps*(y(1,t-1) * y(2,t-1) - b*y(3,t-1));
  end
  x2(3*(i-1)+1:3*i,:) = y;
end

%--------------------------------------------------------------------------------

meanx0 = mean(x')';
Covx0  = cov(x');
x      = Covx0^(-1/2) * (x - meanx0*ones(1,T));
x2     = Covx0^(-1/2) * (x2 - meanx0*ones(1,T2));

%--------------------------------------------------------------------------------