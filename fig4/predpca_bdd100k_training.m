
%--------------------------------------------------------------------------------

% predpca_bdd100k_training.m
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
% 2020-6-20
%
% Before run this script, please download BDD100K dataset from
% https://bdd-data.berkeley.edu
% downsample videos to 320*180 and put them in the same directory

%--------------------------------------------------------------------------------
% initialization

clear
tic;

dirname = '';

% predict observation at t+Kf based on observations between t-Kp+1 and t
% 30 step = 1 s
Kp      = 8;         % order of past observations
Kf      = 15;        % interval
T       = 0;         % length of training data
nx1     = 160;       % video image width
ny1     = 80;        % video image height
Ndata1  = nx1 * ny1;
Npca1   = 2000;      % dimensionality of input fed to PredPCA
num_vid = 20-1;      % number of videos used for training (about 10 h each)

%--------------------------------------------------------------------------------

fprintf(1,'maximum likelihood estimation (time = %.1f min)\n', toc/60);
load([dirname,'pca_lv1_dst.mat'])
Wpca  = PCA_C1(:,1:Npca1)';
STS_t1 = zeros(Npca1,Npca1*Kp,'single');
STS_t2 = zeros(Npca1,Npca1*Kp,'single');
STS_t3 = zeros(Npca1,Npca1*Kp,'single');
STS_t4 = zeros(Npca1,Npca1*Kp,'single');
STS_t5 = zeros(Npca1,Npca1*Kp,'single');
STS_t6 = zeros(Npca1,Npca1*Kp,'single');
STS_b1 = zeros(Npca1,Npca1*Kp,'single');
STS_b2 = zeros(Npca1,Npca1*Kp,'single');
STS_b3 = zeros(Npca1,Npca1*Kp,'single');
STS_b4 = zeros(Npca1,Npca1*Kp,'single');
STS_b5 = zeros(Npca1,Npca1*Kp,'single');
STS_b6 = zeros(Npca1,Npca1*Kp,'single');
S_S_t1 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_t2 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_t3 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_t4 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_t5 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_t6 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_b1 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_b2 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_b3 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_b4 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_b5 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_b6 = zeros(Npca1*Kp,Npca1*Kp,'single');
Tpart_t = zeros(6,1);
Tpart_b = zeros(6,1);

%--------------------------------------------------------------------------------

for fileid = 0:num_vid

fprintf(1,'preprocessing lv2\n');
fprintf(1,'load %strain%d.mp4\n', dirname, fileid);
vid   = VideoReader([dirname, 'train', num2str(fileid), '.mp4']);
l     = vid.NumFrames;
Td    = 100000;
Td0   = 100000;
T1    = l;
T     = T + T1;

%--------------------------------------------------------------------------------
% maximum likelihood estimation

Td    = Td0;
Kf2   = Kf/3;
wx    = round(320*Kf2/50*(1:5)/2);
wy    = round(160*Kf2/50*(1:5)/2);
for k = 1:floor(l/Td)+1
 fprintf(1,'%d/%d (time = %.1f min)\n', k, floor(l/Td)+1, toc/60);
 if (k <= floor(l/Td))
  data  = read(vid, [Td*(k-1)+1 Td*k]);
 else
  data  = read(vid, [Td*floor(l/Td)+1 l]);
  Td    = length(data(1,1,1,:));
 end
 data       = data(1:160,:,:,:);
 data       = permute(data, [1 2 4 3]);
 datas      = cell(6,1);
 datas{2,1} = data(wy(1)+1:160-wy(1),wx(1)+1:320-wx(1),:,:); % 18*Kf2%
 datas{3,1} = data(wy(2)+1:160-wy(2),wx(2)+1:320-wx(2),:,:); % 16*Kf2%
 datas{4,1} = data(wy(3)+1:160-wy(3),wx(3)+1:320-wx(3),:,:); % 14*Kf2%
 datas{5,1} = data(wy(4)+1:160-wy(4),wx(4)+1:320-wx(4),:,:); % 12*Kf2%
 datas{6,1} = data(wy(5)+1:160-wy(5),wx(5)+1:320-wx(5),:,:); % 10*Kf2%
 datas{1,1} = imresize(data, [ny1 nx1]);
 for m = 2:6, datas{m,1} = imresize(datas{m,1}, [ny1 nx1]); end
 
 data(1:ny1,:,:,:) = flip(data(1:ny1,:,:,:),1);
 data(:,1:nx1,:,:) = flip(data(:,1:nx1,:,:),2);
 
 for i = 1:2
  for j = 1:2
   % categorization based on speed
   ss     = cast(reshape(datas{1,1}(ny1/2*(i-1)+(1:ny1/2),nx1/2*(j-1)+(1:nx1/2),:,:),[Ndata1/4 Td*3]),'single')/255;
   sst    = ss(:,[Kf+1:Td*3,1:Kf]);
   G      = zeros(6,Td*3);
   for m = 1:6
    if (m >= 2), ss = cast(reshape(datas{m,1}(ny1/2*(i-1)+(1:ny1/2),nx1/2*(j-1)+(1:nx1/2),:,:),[Ndata1/4 Td*3]),'single')/255; end
    G(m,:)  = mean((ss - sst).^2);
   end
   flag   = (G  == ones(6,1) * min(G));
   if (i == 1), Tpart_t = Tpart_t + sum(flag')'; end
   if (i == 2), Tpart_b = Tpart_b + sum(flag')'; end
   % sensory input
   s      = Wpca * (cast(reshape(data(ny1*(i-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255 - mean1 * ones(1,Td*3,'single'));
   % target
   st     = s(:,[Kf+1:Td*3,1:Kf]);
   % bases
   phi    = diag(PCA_L1(1:Npca1))^(-1/2) * s;
   phi1   = [phi; phi(:,[Td*3,1:Td*3-1]); phi(:,[Td*3-1:Td*3,1:Td*3-2]); phi(:,[Td*3-2:Td*3,1:Td*3-3]); phi(:,[Td*3-3:Td*3,1:Td*3-4]); phi(:,[Td*3-4:Td*3,1:Td*3-5]); phi(:,[Td*3-5:Td*3,1:Td*3-6]); phi(:,[Td*3-6:Td*3,1:Td*3-7])];
   % synaptic update
   if (i == 1)
    STS_t1 = STS_t1 + st(:,flag(1,:)) * phi1(:,flag(1,:))';
    STS_t2 = STS_t2 + st(:,flag(2,:)) * phi1(:,flag(2,:))';
    STS_t3 = STS_t3 + st(:,flag(3,:)) * phi1(:,flag(3,:))';
    STS_t4 = STS_t4 + st(:,flag(4,:)) * phi1(:,flag(4,:))';
    STS_t5 = STS_t5 + st(:,flag(5,:)) * phi1(:,flag(5,:))';
    STS_t6 = STS_t6 + st(:,flag(6,:)) * phi1(:,flag(6,:))';
    S_S_t1 = S_S_t1 + phi1(:,flag(1,:)) * phi1(:,flag(1,:))';
    S_S_t2 = S_S_t2 + phi1(:,flag(2,:)) * phi1(:,flag(2,:))';
    S_S_t3 = S_S_t3 + phi1(:,flag(3,:)) * phi1(:,flag(3,:))';
    S_S_t4 = S_S_t4 + phi1(:,flag(4,:)) * phi1(:,flag(4,:))';
    S_S_t5 = S_S_t5 + phi1(:,flag(5,:)) * phi1(:,flag(5,:))';
    S_S_t6 = S_S_t6 + phi1(:,flag(6,:)) * phi1(:,flag(6,:))';
   end
   if (i == 2)
    STS_b1 = STS_b1 + st(:,flag(1,:)) * phi1(:,flag(1,:))';
    STS_b2 = STS_b2 + st(:,flag(2,:)) * phi1(:,flag(2,:))';
    STS_b3 = STS_b3 + st(:,flag(3,:)) * phi1(:,flag(3,:))';
    STS_b4 = STS_b4 + st(:,flag(4,:)) * phi1(:,flag(4,:))';
    STS_b5 = STS_b5 + st(:,flag(5,:)) * phi1(:,flag(5,:))';
    STS_b6 = STS_b6 + st(:,flag(6,:)) * phi1(:,flag(6,:))';
    S_S_b1 = S_S_b1 + phi1(:,flag(1,:)) * phi1(:,flag(1,:))';
    S_S_b2 = S_S_b2 + phi1(:,flag(2,:)) * phi1(:,flag(2,:))';
    S_S_b3 = S_S_b3 + phi1(:,flag(3,:)) * phi1(:,flag(3,:))';
    S_S_b4 = S_S_b4 + phi1(:,flag(4,:)) * phi1(:,flag(4,:))';
    S_S_b5 = S_S_b5 + phi1(:,flag(5,:)) * phi1(:,flag(5,:))';
    S_S_b6 = S_S_b6 + phi1(:,flag(6,:)) * phi1(:,flag(6,:))';
   end
   fprintf(1, '.');
  end
  fprintf(1, '\n');
 end
end

clear ss sst phi1

save([dirname,'mle_lv1_',num2str(fileid),'.mat'], 'STS_t1', 'STS_t2', 'STS_t3', 'STS_t4', 'STS_t5', 'STS_t6', 'STS_b1', 'STS_b2', 'STS_b3', 'STS_b4', 'STS_b5', 'STS_b6', 'S_S_t1', 'S_S_t2', 'S_S_t3', 'S_S_t4', 'S_S_t5', 'S_S_t6', 'S_S_b1', 'S_S_b2', 'S_S_b3', 'S_S_b4', 'S_S_b5', 'S_S_b6', 'Tpart_t', 'Tpart_b', '-v7.3')

end

%--------------------------------------------------------------------------------
