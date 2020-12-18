
% predpca_bdd_cat.m
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
% 2020-6-26
%

%--------------------------------------------------------------------------------

clear
tic
dirname = '';

load([dirname,'pca_lv1_dst.mat'])
load([dirname,'predpca_lv1_dst.mat'])
load([dirname,'predpca_lv1_u_1.mat'])
u1      = u;
load([dirname,'predpca_lv1_u_2.mat'])
u2      = u;
load([dirname,'predpca_lv1_u_3.mat'])
u3      = u;
load([dirname,'predpca_lv1_u_4.mat'])
u4      = u;
clear u

nx1     = 160;       % video image width
ny1     = 80;        % video image height
Ndata1  = nx1 * ny1;
Npca1   = 2000;      % dimensionality of input fed to PredPCA

Ns      = 2000;
Nv      = 100;
T       = length(u1{1,1}(1,:))/3;
T1      = floor(T/10);

seed    = 0;
rng(1000000+seed);

%--------------------------------------------------------------------------------

u1      = [u1{1,1}; u1{1,2}; u1{2,1}; u1{2,2}];
u2      = [u2{1,1}; u2{1,2}; u2{2,1}; u2{2,2}];
u3      = [u3{1,1}; u3{1,2}; u3{2,1}; u3{2,2}];
u4      = [u4{1,1}; u4{1,2}; u4{2,1}; u4{2,2}];
u1      = reshape(permute(reshape(u1,[100*4 T 3]),[1 3 2]),[100*12 T]);
u2      = reshape(permute(reshape(u2,[100*4 T 3]),[1 3 2]),[100*12 T]);
u3      = reshape(permute(reshape(u3,[100*4 T 3]),[1 3 2]),[100*12 T]);
u4      = reshape(permute(reshape(u4,[100*4 T 3]),[1 3 2]),[100*12 T]);
u       = [u1 u2 u3 u4];
clear u1 u2 u3 u4

T       = length(u);
T1      = floor(T/10);

[C,~,L] = pca(u');
up      = C(:,1:Nv)'*u;
mean_up = mean(up')';
std_up  = std(up')';
up      = diag(std_up)^(-1) * (up - mean_up*ones(1,T));

%--------------------------------------------------------------------------------

fprintf(1,'ICA\n')
ica_rep = 20000;
Wica    = eye(Nv);
for t = 1:ica_rep
 if (rem(t,1000) == 0), fprintf(1,'t = %d\n', t), end
 if (t < 4000),      eta = 0.04;
 elseif (t < 8000),  eta = 0.02;
 elseif (t < 12000), eta = 0.01;
 else                eta = 0.005; end
 t_list = randi([1,T],T1,1);
 v      = Wica * up(:,t_list);
 g      = tanh(100 * v);
 Wica   = Wica + eta * (eye(Nv) - g*v'/T1) * Wica;
end

v       = Wica * up;
Wica    = diag(sign(skewness(v'))) * diag(std(v'))^(-1) * Wica;
v       = Wica * up;

%--------------------------------------------------------------------------------
% output independent components

uica    = diag(std_up) * Wica^(-1) * 20 + mean_up*ones(1,Nv);

output2 = zeros(ny1*2,nx1*2,3,Nv,'single');
output2(1:ny1,1:nx1,1,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1t(:,1:100)*C(1:100,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(1:ny1,nx1+1:nx1*2,1,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1t(:,1:100)*C(101:200,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(ny1+1:ny1*2,1:nx1,1,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1b(:,1:100)*C(201:300,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(ny1+1:ny1*2,nx1+1:nx1*2,1,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1b(:,1:100)*C(301:400,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(1:ny1,1:nx1,2,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1t(:,1:100)*C(401:500,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(1:ny1,nx1+1:nx1*2,2,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1t(:,1:100)*C(501:600,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(ny1+1:ny1*2,1:nx1,2,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1b(:,1:100)*C(601:700,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(ny1+1:ny1*2,nx1+1:nx1*2,2,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1b(:,1:100)*C(701:800,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(1:ny1,1:nx1,3,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1t(:,1:100)*C(801:900,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(1:ny1,nx1+1:nx1*2,3,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1t(:,1:100)*C(901:1000,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(ny1+1:ny1*2,1:nx1,3,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1b(:,1:100)*C(1001:1100,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(ny1+1:ny1*2,nx1+1:nx1*2,3,:) = reshape(PCA_C1(:,1:Ns)*PPCA_C1b(:,1:100)*C(1101:1200,1:Nv)*uica+mean1*ones(1,Nv),[ny1 nx1 1 Nv]);
output2(1:ny1,:,:,:) = flip(output2(1:ny1,:,:,:),1);
output2(:,1:nx1,:,:) = flip(output2(:,1:nx1,:,:),2);
output2 = max(output2,0);
output2 = min(output2,1);

%--------------------------------------------------------------------------------

mean_b  = mean(reshape(output2(:,:,3,:),[ny1*2*nx1*2 Nv]));
mean_r  = mean(reshape(output2(:,:,1,:),[ny1*2*nx1*2 Nv]));
mean_v  = mean(reshape(output2,[ny1*2*nx1*2*3 Nv]));
[~,idx] = sort(mean_b./mean_v,'ascend');
for i = 1:10
 [~,idx2] = sort(mean_r(:,idx(10*(i-1)+(1:10)))./mean_v(:,idx(10*(i-1)+(1:10))),'descend');
 idx(10*(i-1)+(1:10)) = idx(10*(i-1)+idx2);
end

%--------------------------------------------------------------------------------

img = ones(ny1*2*10+90,nx1*2*10+90,3,'uint8')*255;
for i = 1:10
 for j = 1:10
  img((ny1*2+10)*(i-1)+(1:ny1*2),(nx1*2+10)*(j-1)+(1:nx1*2),:) = output2(:,:,:,idx(10*(j-1)+i))*255;
 end
end
imwrite(img, [dirname, 'predpca_ica_cat100.png'])

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

fprintf(1,'read movie\n')
fileid = 0;
fprintf(1,'load %stest%d.mp4\n', dirname, fileid);
vid    = VideoReader([dirname, 'test', num2str(fileid), '.mp4']);
l      = vid.NumFrames;
Td     = 100000;
Td0    = 100000;
data   = zeros(ny1*2,nx1*2,3,Td,'uint8');

Wpca   = PCA_C1(:,1:Npca1)';

%--------------------------------------------------------------------------------

brightness = zeros(Td*4,1);
vertical   = zeros(Td*4,1);
lateral    = zeros(Td*4,1);

for k = 1:4
 fprintf(1,'%d/%d (time = %.1f min)\n', k, floor(l/Td)+1, toc/60);
 if (k <= floor(l/Td))
  data = read(vid, [Td*(k-1)+1 Td*k]);
 else
  data = read(vid, [Td*floor(l/Td)+1 l]);
  Td   = length(data(1,1,1,:));
 end
 data = data(1:160,:,:,:);
 
 brightness(Td*(k-1)+1:Td*k) = reshape(mean(mean(mean(data))),[Td 1])/255;
 vertical(Td*(k-1)+1:Td*k)   = reshape(mean(mean(mean(data(1:ny1,:,:,:))-mean(flip(data(ny1+1:ny1*2,:,:,:),1)))),[Td 1])/255;
 lateral(Td*(k-1)+1:Td*k)    = reshape(mean(mean(mean(data(:,1:nx1,:,:))-mean(flip(data(:,nx1+1:nx1*2,:,:),2)))),[Td 1])/255;
end

k = 1;
fprintf(1,'%d/%d (time = %.1f min)\n', k, floor(l/Td)+1, toc/60);
data = read(vid, [Td*(k-1)+1 Td*k]);
data = data(1:160,:,:,:);

%--------------------------------------------------------------------------------

fprintf(1,'analysis of PC1,PC2,PC3 of categorical features\n')

prctile_pc1 = zeros(3,100);
count       = zeros(1,100);
for i = 1:100
 idx              = find(brightness >= (i-1)*0.01 & brightness < i*0.01);
 prctile_pc1(:,i) = prctile(up(1,idx),[25 50 75]);
 count(1,i)       = length(idx);
end

figure()
subplot(1,3,1)
plot((1:100)*0.01,prctile_pc1(1,:)), hold on
plot((1:100)*0.01,prctile_pc1(2,:))
plot((1:100)*0.01,prctile_pc1(3,:)), hold off
csvwrite([dirname,'predpca_cat_pc1_brightness.csv'],[(1:100)*0.01; prctile_pc1; count])

%--------------------------------------------------------------------------------

prctile_pc2 = zeros(3,100);
count       = zeros(1,100);
for i = 1:100
 idx              = find(vertical >= (i-1)*0.02-1 & vertical < i*0.02-1);
 prctile_pc2(:,i) = prctile(-up(2,idx),[25 50 75]);
 count(1,i)       = length(idx);
end

subplot(1,3,2)
plot((1:100)*0.02-1,prctile_pc2(1,:)), hold on
plot((1:100)*0.02-1,prctile_pc2(2,:))
plot((1:100)*0.02-1,prctile_pc2(3,:)), hold off
csvwrite([dirname,'predpca_cat_pc2_vertical.csv'],[(1:100)*0.02-1; prctile_pc2; count])

%--------------------------------------------------------------------------------

prctile_pc3 = zeros(3,100);
count       = zeros(1,100);
for i = 1:100
 idx              = find(lateral >= (i-1)*0.02-1 & lateral < i*0.02-1);
 prctile_pc3(:,i) = prctile(-up(3,idx),[25 50 75]);
 count(1,i)       = length(idx);
end

subplot(1,3,3)
plot((1:100)*0.02-1,prctile_pc3(1,:)), hold on
plot((1:100)*0.02-1,prctile_pc3(2,:))
plot((1:100)*0.02-1,prctile_pc3(3,:)), hold off
csvwrite([dirname,'predpca_cat_pc3_lateral.csv'],[(1:100)*0.02-1; prctile_pc3; count])

%--------------------------------------------------------------------------------

fprintf(1,'show movie\n')
time_v = cell(Nv,1);
for i = 1:Nv, [~,time_v{i,1}] = sort(up(i,1:Td)./std(up(:,1:Td)),'descend'); end

figure()
for t = 1:10000
 image([data(:,:,:,time_v{1,1}(t)) data(:,:,:,time_v{2,1}(t)) data(:,:,:,time_v{3,1}(t)) data(:,:,:,time_v{4,1}(t)); data(:,:,:,time_v{5,1}(t)) data(:,:,:,time_v{6,1}(t)) data(:,:,:,time_v{7,1}(t)) data(:,:,:,time_v{8,1}(t))])
 title(['i = ',num2str(t)])
 drawnow
 pause(1/10)
end

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

