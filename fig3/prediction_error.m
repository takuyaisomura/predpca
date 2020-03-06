
%--------------------------------------------------------------------------------

% prediction_error.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function err = prediction_error(st, se, C)

Kf  = length(st(:,1));
T   = length(st{1,1}(1,:));
Ns  = length(st{1,1}(:,1));
err = zeros(Kf,Ns);
for k = 1:Kf
 StSt         = st{k,1} * st{k,1}' / T;
 StSe         = st{k,1} * se{k,1}' / T;
 SeSe         = se{k,1} * se{k,1}' / T;
 lambda       = diag(C' * (2*StSe - SeSe) * C);
 lambda2      = zeros(Ns,1);
 lambda2(1,1) = lambda(1,1);
 for i = 2:Ns, lambda2(i,1) = lambda2(i-1,1) + lambda(i,1); end
 err(k,:) = trace(StSt) - lambda2;
end

