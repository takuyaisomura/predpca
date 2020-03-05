
%--------------------------------------------------------------------------------

% generate_true_hidden_states.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [x] = generate_true_hidden_states(input,label)
T  = length(label);
Nx = 10;
x      = zeros(Nx,T);
for i  = 1:Nx, x(i,:) = label == i-1; end
x      = x .* (ones(Nx,1) * sign(sum(input)));

