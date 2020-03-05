
%--------------------------------------------------------------------------------

% ica.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [ui,ui2,Wica] = ica(u,u2,ica_rep,ica_eta)

T  = length(u(1,:));
Nu = length(u(:,1));
[Wica,~,~] = svd(randn(Nu,Nu));

for h = 1:ica_rep
rnd  = randi([1 T],1,T/10);
ui   = Wica * u(:,rnd);      % ICA encoders
g    = sqrt(2)*tanh(100*ui); % nonlinear activation function
Wica = Wica + ica_eta * (Wica - (g*ui'/(T/10)) * Wica); % Amari's ICA rule
end

ui   = Wica * u;
ui2  = Wica * u2;

