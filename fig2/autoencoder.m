
%--------------------------------------------------------------------------------

% autoencoder.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [u,u2] = autoencoder(input, input2, filename)

T      = length(input(1,:));
T2     = length(input2(1,:));
input  = (input  + 1) / 2;     % adjust the range from [-1,1] to [0,1]
input2 = (input2 + 1) / 2;     % adjust the range from [-1,1] to [0,1]
load(filename)                 % read autoencoder's weight matrices
W1     = w1';                  % transpose
W2     = w2';                  % transpose
W3     = w3';                  % transpose
W4     = w4';                  % transpose
clear w1 w2 w3 w4 w5 w6 w7 w8  % clear unused weight matrices

%--------------------------------------------------------------------------------

fprintf(1,'calculate layer 1 neural activity\n');
AE1 = 1./(1+exp(-W1 * [input input2; ones(1,T+T2)]));

fprintf(1,'calculate layer 2 neural activity\n');
AE2 = 1./(1+exp(-W2 * [AE1; ones(1,T+T2)]));

fprintf(1,'calculate layer 3 neural activity\n');
AE3 = 1./(1+exp(-W3 * [AE2; ones(1,T+T2)]));

fprintf(1,'calculate layer 4 neural activity\n');
AE4 =            W4 * [AE3; ones(1,T+T2)];
u   = AE4(:,1:T);
u2  = AE4(:,T+1:T+T2);

%--------------------------------------------------------------------------------

