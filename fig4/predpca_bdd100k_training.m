
%--------------------------------------------------------------------------------

% predpca_bdd100k_preprocessing.m
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
num_vid = 20;        % number of videos used for training (about 10 h each)

%--------------------------------------------------------------------------------

fprintf(1,'maximum likelihood estimation (time = %.1f min)\n', toc/60);
load([dirname,'pca_lv1_dst.mat'])
Wpca  = PCA_C1(:,1:Npca1)';
STS_1 = zeros(Npca1,Npca1*Kp,'single');
STS_2 = zeros(Npca1,Npca1*Kp,'single');
STS_3 = zeros(Npca1,Npca1*Kp,'single');
STS_4 = zeros(Npca1,Npca1*Kp,'single');
STS_5 = zeros(Npca1,Npca1*Kp,'single');
STS_6 = zeros(Npca1,Npca1*Kp,'single');
S_S_1 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_2 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_3 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_4 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_5 = zeros(Npca1*Kp,Npca1*Kp,'single');
S_S_6 = zeros(Npca1*Kp,Npca1*Kp,'single');
Tpart = zeros(6,1);

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
 datas{2,1} = data( 8+1:160- 8,16+1:320-16,:,:); % 90%
 datas{3,1} = data(16+1:160-16,32+1:320-32,:,:); % 80%
 datas{4,1} = data(24+1:160-24,48+1:320-48,:,:); % 70%
 datas{5,1} = data(32+1:160-32,64+1:320-64,:,:); % 60%
 datas{6,1} = data(40+1:160-40,80+1:320-80,:,:); % 50%
 datas{1,1} = imresize(data,       [ny1 nx1]);
 for m = 2:6, datas{m,1} = imresize(datas{m,1}, [ny1 nx1]); end
 
 data( 1:ny1,:,:,:) = flip(data( 1:ny1,:,:,:),1);
 data( :,1:nx1,:,:) = flip(data( :,1:nx1,:,:),2);
 
 for i = 1:2
  for j = 1:2
   % categorization based on speed
   ss     = cast(reshape(datas{1,1}(ny1/2*(i-1)+(1:ny1/2),nx1/2*(j-1)+(1:nx1/2),:,:),[Ndata1/4 Td*3]),'single')/255;
   sst    = ss(:,[Kf+1:Td*3,1:Kf]);
   sst2   = ss(:,[Kf*2+1:Td*3,1:Kf*2]);
   sst3   = ss(:,[23+1:Td*3,1:23]);
   G      = zeros(6,Td*3);
   G2     = zeros(6,Td*3);
   G3     = zeros(6,Td*3);
   for m = 1:6
    if (m >= 2), ss = cast(reshape(datas{m,1}(ny1/2*(i-1)+(1:ny1/2),nx1/2*(j-1)+(1:nx1/2),:,:),[Ndata1/4 Td*3]),'single')/255; end
    G(m,:)  = mean((ss - sst).^2);
    G2(m,:) = mean((ss - sst2).^2);
    G3(m,:) = mean((ss - sst3).^2);
   end
   flag   = (G  == ones(6,1) * min(G ));
   flag2  = (G2 == ones(6,1) * min(G2));
   flag3  = (G3 == ones(6,1) * min(G3));
   Tpart  = Tpart + sum(flag')' + sum(flag2')' + sum(flag3')';
   % sensory input
   s      = Wpca * (cast(reshape(data(ny1*(i-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255 - mean1 * ones(1,Td*3,'single'));
   % target
   st     = s(:,[Kf+1:Td*3,1:Kf]);
   st2    = s(:,[Kf*2+1:Td*3,1:Kf*2]);
   st3    = s(:,[23+1:Td*3,1:23]);
   % bases
   phi    = diag(PCA_L1(1:Npca1))^(-1/2) * s;
   phi1   = [phi; phi(:,[Td*3-0:Td*3,1:Td*3-1]); phi(:,[Td*3-1:Td*3,1:Td*3-2]); phi(:,[Td*3-2:Td*3,1:Td*3-3])];
   phi2   = [phi; phi(:,[Td*3-1:Td*3,1:Td*3-2]); phi(:,[Td*3-3:Td*3,1:Td*3-4]); phi(:,[Td*3-5:Td*3,1:Td*3-6])];
   phi3   = [phi; (phi(:,[Td*3-0:Td*3,1:Td*3-1])+phi(:,[Td*3-1:Td*3,1:Td*3-2]))/2; phi(:,[Td*3-2:Td*3,1:Td*3-3]); (phi(:,[Td*3-3:Td*3,1:Td*3-4])+phi(:,[Td*3-4:Td*3,1:Td*3-5]))/2];
   % synaptic update
   STS_1 = STS_1 + st(:,flag(1,:)) * phi1(:,flag(1,:))' + st2(:,flag2(1,:)) * phi2(:,flag2(1,:))' + st3(:,flag3(1,:)) * phi3(:,flag3(1,:))';
   STS_2 = STS_2 + st(:,flag(2,:)) * phi1(:,flag(2,:))' + st2(:,flag2(2,:)) * phi2(:,flag2(2,:))' + st3(:,flag3(2,:)) * phi3(:,flag3(2,:))';
   STS_3 = STS_3 + st(:,flag(3,:)) * phi1(:,flag(3,:))' + st2(:,flag2(3,:)) * phi2(:,flag2(3,:))' + st3(:,flag3(3,:)) * phi3(:,flag3(3,:))';
   STS_4 = STS_4 + st(:,flag(4,:)) * phi1(:,flag(4,:))' + st2(:,flag2(4,:)) * phi2(:,flag2(4,:))' + st3(:,flag3(4,:)) * phi3(:,flag3(4,:))';
   STS_5 = STS_5 + st(:,flag(5,:)) * phi1(:,flag(5,:))' + st2(:,flag2(5,:)) * phi2(:,flag2(5,:))' + st3(:,flag3(5,:)) * phi3(:,flag3(5,:))';
   STS_6 = STS_6 + st(:,flag(6,:)) * phi1(:,flag(6,:))' + st2(:,flag2(6,:)) * phi2(:,flag2(6,:))' + st3(:,flag3(6,:)) * phi3(:,flag3(6,:))';
   S_S_1 = S_S_1 + phi1(:,flag(1,:)) * phi1(:,flag(1,:))' + phi2(:,flag2(1,:)) * phi2(:,flag2(1,:))' + phi3(:,flag3(1,:)) * phi3(:,flag3(1,:))';
   S_S_2 = S_S_2 + phi1(:,flag(2,:)) * phi1(:,flag(2,:))' + phi2(:,flag2(2,:)) * phi2(:,flag2(2,:))' + phi3(:,flag3(2,:)) * phi3(:,flag3(2,:))';
   S_S_3 = S_S_3 + phi1(:,flag(3,:)) * phi1(:,flag(3,:))' + phi2(:,flag2(3,:)) * phi2(:,flag2(3,:))' + phi3(:,flag3(3,:)) * phi3(:,flag3(3,:))';
   S_S_4 = S_S_4 + phi1(:,flag(4,:)) * phi1(:,flag(4,:))' + phi2(:,flag2(4,:)) * phi2(:,flag2(4,:))' + phi3(:,flag3(4,:)) * phi3(:,flag3(4,:))';
   S_S_5 = S_S_5 + phi1(:,flag(5,:)) * phi1(:,flag(5,:))' + phi2(:,flag2(5,:)) * phi2(:,flag2(5,:))' + phi3(:,flag3(5,:)) * phi3(:,flag3(5,:))';
   S_S_6 = S_S_6 + phi1(:,flag(6,:)) * phi1(:,flag(6,:))' + phi2(:,flag2(6,:)) * phi2(:,flag2(6,:))' + phi3(:,flag3(6,:)) * phi3(:,flag3(6,:))';
   fprintf(1, '.');
  end
  fprintf(1, '\n');
 end
end

clear ss sst sst2 sst3
clear phi1 phi2 phi3

save([dirname,'mle_lv1_',num2str(fileid),'.mat'], 'STS_1', 'STS_2', 'STS_3', 'STS_4', 'STS_5', 'STS_6', 'S_S_1', 'S_S_2', 'S_S_3', 'S_S_4', 'S_S_5', 'S_S_6', 'Tpart', '-v7.3')

end

%--------------------------------------------------------------------------------
