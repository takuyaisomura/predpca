
%--------------------------------------------------------------------------------

% maximum_likelihood_estimator.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [s_,s2_,se,se2,Q] = maximum_likelihood_estimator(s,s2,Kp,prior_s_)

T   = length(s(1,:));
T2  = length(s2(1,:));
Ns  = length(s(:,1));

s_  = s(:,[T,1:T-1]);    % basis functions (training)
for k = 1:Kp-1, s_ = [s(:,[T,1:T-1]); s_(:,[T,1:T-1])]; end
s2_ = s2(:,[T2,1:T2-1]); % basis functions (test)
for k = 1:Kp-1, s2_ = [s2(:,[T2,1:T2-1]); s2_(:,[T2,1:T2-1])]; end

Q   = (s*s_') * (s_*s_'+eye(Ns*Kp)*prior_s_)^(-1);
se  = Q * s_;            % predicted input (training)
se2 = Q * s2_;           % predicted input (test)

