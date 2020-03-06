
%--------------------------------------------------------------------------------

% predpca_bdd100k_preprocess.m
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
% 2020-3-5
%
% Before run this script, please download BDD100K dataset from
% https://bdd-data.berkeley.edu
% downsample videos to 160*80 and put them in the same directory

%--------------------------------------------------------------------------------
% initialization

clear
tic;

dirname = '';

% predict observation at t+Kf based on observations between t-Kp+1 and t
% 30 step = 1 s
Kp      = 4;         % order of past observations
Kf      = 15;        % interval
T       = 0;         % length of training data
nx1     = 160;       % video image width
ny1     = 80;        % video image height
Ndata1  = nx1 * ny1;
Npca1   = 2000;      % dimensionality of input fed to PredPCA
num_vid = 10;        % number of videos used for training (about 10 h each)

%--------------------------------------------------------------------------------
% PCA as a preprocessing

mean1 = zeros(Ndata1,1,     'single');
Cov1  = zeros(Ndata1,Ndata1,'single');

for fileid = 0:num_vid
 
 fprintf(1,'preprocessing\n');
 fprintf(1,'load %strain%d.mp4\n', dirname, fileid);
 vid   = VideoReader([dirname, 'train', num2str(fileid), '.mp4']);
 l     = vid.NumFrames;
 Td    = 200000;
 Td0   = 200000;
 T1    = l;
 data  = zeros(ny1*2,nx1*2,3,Td,'uint8');
 T     = T + T1;
 
 %--------------------------------------------------------------------------------
 
 fprintf(1,'PCA lv1 (time = %.1f min)\n', toc/60);
 
 for k = 1:floor(l/Td)+1
  fprintf(1,'%d/%d (time = %.1f min)\n', k, floor(l/Td)+1, toc/60);
  if (k <= floor(l/Td))
   data = read(vid,[Td*(k-1)+1 Td*k]);
  else
   data = read(vid,[Td*floor(l/Td)+1 l]);
   Td   = length(data(1,1,1,:));
  end
  data              = data(1:160,:,:,:);
  data              = permute(data,[1 2 4 3]);
  data(1:ny1,:,:,:) = flip(data(1:ny1,:,:,:),1);
  data(:,1:nx1,:,:) = flip(data(:,1:nx1,:,:),2);
  
  for i = 1:2
   for j = 1:2
    s     = cast(reshape(data(ny1*(i-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255;
    mean1 = mean1 + sum(s')';
    Cov1  = Cov1  + s * s';
    fprintf(1, '.');
   end
   fprintf(1, '\n');
  end
 end
 
 fprintf(1,'save data (time = %.1f min)\n', toc/60);
 save([dirname,'pca_lv1_',num2str(fileid),'.mat'], 'mean1', 'Cov1', 'T', '-v7.3')
 
end

mean1           = mean1 / T / 12;
Cov1            = Cov1  / T / 12 - mean1 * mean1';
[PCA_C1,PCA_L1] = pcacov(Cov1);

fprintf(1,'save data (time = %.1f min)\n', toc/60);
save([dirname,'pca_lv1_dst.mat'], 'mean1', 'PCA_C1', 'PCA_L1', 'T', '-v7.3')

%--------------------------------------------------------------------------------
