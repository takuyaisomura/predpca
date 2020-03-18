
%--------------------------------------------------------------------------------

% state_to_image.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-18

%--------------------------------------------------------------------------------

function img = state_to_image(sp, PCA_C2, PCA_C1, mean1, numx, numy)

Ns      = length(sp(:,1));
T       = length(sp(1,:));
nx1     = 72;            % half of image width
ny1     = 72;            % half of image height
data    = zeros(ny1*2,nx1*2,3,T);

for i = 1:2
 for j = 1:2
  data2 = PCA_C2(1000*(2*(i-1)+j-1)+1:1000*(2*(i-1)+j),1:Ns) * sp;
  data(ny1*(i-1)+1:ny1*i,nx1*(j-1)+1:nx1*j,:,:) = reshape(PCA_C1{i,j}(:,1:1000) * data2 + mean1{i,j} * ones(1,T),[ny1 nx1 3 T]);
 end
end

img = ones((ny1*2)*numy+16*(numy-1),(nx1*2)*numx+16*(numx-1),3);
for t = 1:T
 ix = rem(t-1,numx)   + 1;
 iy = (t - ix) / numx + 1;
 img((1:(ny1*2))+160*(iy-1),(1:(nx1*2))+160*(ix-1),:) = data(:,:,:,t);
end
