
%--------------------------------------------------------------------------------

% maximum_likelihood_estimator.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-10-25

%--------------------------------------------------------------------------------

function [s_,s2_,se,se2,Q] = maximum_likelihood_estimator(s,s2,st,Kp,prior_s_)

T   = length(s(1,:));
T2  = length(s2(1,:));
Ns  = length(s(:,1));

s_  = zeros(Ns*Kp,T);    % basis functions (training)
s2_ = zeros(Ns*Kp,T2);   % basis functions (test)
for k = 1:Kp, s_(Ns*(k-1)+(1:Ns),:) = s(:,[T-(k-1):T,1:T-k]); end
for k = 1:Kp, s2_(Ns*(k-1)+(1:Ns),:) = s2(:,[T2-(k-1):T2,1:T2-k]); end

Q   = (st*s_') / (s_*s_'+eye(Ns*Kp)*prior_s_);
se  = Q * s_;            % predicted input (training)
se2 = Q * s2_;           % predicted input (test)

