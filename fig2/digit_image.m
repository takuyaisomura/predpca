
%--------------------------------------------------------------------------------

% digit_image.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-27

%--------------------------------------------------------------------------------

function dstimg = digit_image(input)

T       = length(input(1,:));
img     = zeros(28,28*T);
for t = 1:T
 img(:,28*(t-1)+(1:28)) = reshape(input(:,t),[28 28])';
end
img     = max(img,0);
img     = min(img,1);
img     = 1 - img;
img     = kron(img,ones(5,5));
dstimg  = cast(reshape([img img img]*255,[28*5 28*5*T 3]),'uint8');

