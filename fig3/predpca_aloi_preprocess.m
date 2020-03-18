
%--------------------------------------------------------------------------------

% predpca_aloi_preprocess.m
%
% This demo is included in
% Dimensionality reduction to maximize prediction generalization capability
% Takuya Isomura, Taro Toyoizumi
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-18
%
% Before run this script, please download ALOI dataset from
% http://aloi.science.uva.nl
% (full color (24 bit), quarter resolution (192 x 144), viewing direction)
% and expand aloi_red4_view.tar in the same directory
%
% Reference for ALOI dataset
% Geusebroek JM, Burghouts GJ, Smeulders AWM, The Amsterdam library of object images,
% Int J Comput Vision, 61, 103-112 (2005)
%

%--------------------------------------------------------------------------------
% initialization

clear
tic
Timg    = 72000;         % number of images
Nimgx   = 192;           % original image width
Nimgy   = 144;           % original image height
nx1     = 72;            % half of image width
ny1     = 72;            % half of image height
Ndata1  = nx1 * ny1 * 3; % input dimensionality of level 1 PCA
Npca1   = 1000;          % output dimensionality of level 1 PCA
Ndata2  = Npca1 * 4;     % input dimensionality of level 2 PCA
Npca2   = 1000;          % output dimensionality of level 2 PCA
dir     = 'png4/';       % directory name

fprintf(1,'----------------------------------------\n');
fprintf(1,'Preprocess ALOI dataset\n')
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% level 1 PCA

fprintf(1,'level 1 PCA (time = %.1f min)\n', toc/60)
img  = zeros(Nimgy,Nimgx,3,72,'uint8');
data = zeros(Nimgy,Nimgy,3,Timg,'uint8');

fprintf(1,'read image files (time = %.1f min)\n', toc/60)
for n_obj = 1:Timg/72
 if (rem(n_obj,100) == 0), fprintf(1,'n_obj = %d / %d (time = %.1f min)\n', n_obj, Timg/72, toc/60), end
 for n_ang = 1:72
  img(:,:,:,n_ang) = imread([dir,num2str(n_obj),'/',num2str(n_obj),'_r',num2str((n_ang-1)*5),'.png']);
 end
 data(:,:,:,72*(n_obj-1)+1:72*n_obj) = img(:,25:168,:,:);
end

Cov1   = cell(2,2);
mean1  = cell(2,2);
PCA_C1 = cell(2,2);
PCA_L1 = cell(2,2);
for i = 1:2
 for j = 1:2
  fprintf(1,'compute covariance (time = %.1f min)\n', toc/60)
  s          = cast(reshape(data(ny1*(i-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Timg]),'single')/255;
  mean1{i,j} = mean(s')';
  Cov1{i,j}  = cov(s');
  fprintf(1,'eigenvalue decomposition (time = %.1f min)\n', toc/60)
  [PCA_C1{i,j},PCA_L1{i,j}] = pcacov(Cov1{i,j});
 end
end
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% level 2 PCA

fprintf(1,'level 2 PCA (time = %.1f min)\n', toc/60)
s = zeros(Ndata2,Timg,'single');

fprintf(1,'compute covariance (time = %.1f min)\n', toc/60)
for i = 1:2
 for j = 1:2
  W1 = PCA_C1{i,j}(:,1:Npca1)';
  s(Npca1*(2*(i-1)+j-1)+1:Npca1*(2*(i-1)+j),:) = W1 * (cast(reshape(data(ny1*(i-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Timg]),'single')/255 - mean1{i,j} * ones(1,Timg,'single'));
 end
end
mean2 = mean(s')';
Cov2  = cov(s');

fprintf(1,'eigenvalue decomposition (time = %.1f min)\n', toc/60)
[PCA_C2,PCA_L2] = pcacov(Cov2);

fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
% save data

fprintf(1,'save data (time = %.1f min)\n', toc/60)
W2   = PCA_C2(:,1:Npca2)';
data = W2 * (s - mean2 * ones(1,Timg,'single'));
save('aloi_data.mat','mean1','PCA_C1','PCA_L1','mean2','PCA_C2','PCA_L2','data','-v7.3')

fprintf(1,'complete preprocessing (time = %.1f min)\n', toc/60)
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
