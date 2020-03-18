
%--------------------------------------------------------------------------------

% predpca_bdd100k_test.m
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
% downsample videos to 320*180 and put them in the same directory

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
% compute optimal matrices for predictions

fileid  = 19;
load([dirname,'mle_lv1_',num2str(fileid),'.mat'])

load([dirname,'pca_lv1_dst.mat'])
Wpca    = PCA_C1(:,1:Npca1)';

fprintf(1,'compute Q (time = %.1f min)\n', toc/60);
STS_1   = STS_1  / Tpart(1,1);
STS_2   = STS_2  / Tpart(2,1);
STS_3   = STS_3  / Tpart(3,1);
STS_4   = STS_4  / Tpart(4,1);
STS_5   = STS_5  / Tpart(5,1);
STS_6   = STS_6  / Tpart(6,1);
S_S_1   = S_S_1 / Tpart(1,1) + eye(Npca1*Kp) * 0.01;
S_S_2   = S_S_2 / Tpart(2,1) + eye(Npca1*Kp) * 0.01;
S_S_3   = S_S_3 / Tpart(3,1) + eye(Npca1*Kp) * 0.01;
S_S_4   = S_S_4 / Tpart(4,1) + eye(Npca1*Kp) * 0.01;
S_S_5   = S_S_5 / Tpart(5,1) + eye(Npca1*Kp) * 0.01;
S_S_6   = S_S_6 / Tpart(6,1) + eye(Npca1*Kp) * 0.01;
Q1      = cast(STS_1,'double') / cast(S_S_1,'double');
Q2      = cast(STS_2,'double') / cast(S_S_2,'double');
Q3      = cast(STS_3,'double') / cast(S_S_3,'double');
Q4      = cast(STS_4,'double') / cast(S_S_4,'double');
Q5      = cast(STS_5,'double') / cast(S_S_5,'double');
Q6      = cast(STS_6,'double') / cast(S_S_6,'double');

% post-hoc PCA
SESE    = (Q1 * S_S_1 * Q1' + Q2 * S_S_2 * Q2' + Q3 * S_S_3 * Q3' + Q4 * S_S_4 * Q4' + Q5 * S_S_5 * Q5' + Q6 * S_S_6 * Q6')/6;
[PPCA_C1,PPCA_L1] = pcacov(SESE);
Lambda            = zeros(6,Npca1);
[~,Lambda(1,:)]   = pcacov(Q1 * S_S_1 * Q1');
[~,Lambda(2,:)]   = pcacov(Q2 * S_S_2 * Q2');
[~,Lambda(3,:)]   = pcacov(Q3 * S_S_3 * Q3');
[~,Lambda(4,:)]   = pcacov(Q4 * S_S_4 * Q4');
[~,Lambda(5,:)]   = pcacov(Q5 * S_S_5 * Q5');
[~,Lambda(6,:)]   = pcacov(Q6 * S_S_6 * Q6');

% dimensionality reduction
Nppca1  = 2000;
Nppca2  = 500;
Nppca3  = 400;
Nppca4  = 300;
Nppca5  = 200;
Nppca6  = 200;
Q1      = PPCA_C1(:,1:Nppca1) * PPCA_C1(:,1:Nppca1)' * Q1;
Q2      = PPCA_C1(:,1:Nppca2) * PPCA_C1(:,1:Nppca2)' * Q2;
Q3      = PPCA_C1(:,1:Nppca3) * PPCA_C1(:,1:Nppca3)' * Q3;
Q4      = PPCA_C1(:,1:Nppca4) * PPCA_C1(:,1:Nppca4)' * Q4;
Q5      = PPCA_C1(:,1:Nppca5) * PPCA_C1(:,1:Nppca5)' * Q5;
Q6      = PPCA_C1(:,1:Nppca6) * PPCA_C1(:,1:Nppca6)' * Q6;

plot(1:Npca1,log10(PCA_L1(1:Npca1)),1:Npca1,log10(Lambda'))

%--------------------------------------------------------------------------------
% predict 0.5 s future image of test data

clear data datas phi phi1 phi2 phi3
clear S_S_1 S_S_2 S_S_3 S_S_4 S_S_5 S_S_6
clear STS_1 STS_2 STS_3 STS_4 STS_5 STS_6
fileid  = 0;
fprintf(1,'load %stest%d.mp4\n', dirname, fileid);
vid     = VideoReader([dirname, 'test', num2str(fileid), '.mp4']);
l       = vid.NumFrames;
Td      = 100000;
Td0     = 100000;
T1      = l;
data    = zeros(ny1*2,nx1*2,3,Td,'uint8');

load([dirname,'pca_lv1_dst.mat'])
Wpca    = PCA_C1(:,1:Npca1)';

err     = zeros(2,Td,4);
amp     = zeros(2,Td,4);
Tm      = Td/10;
Tmovie  = 30*60*5;
img     = zeros(160*5,320*4,3,Tmovie,'uint8');
erro    = zeros(2,Tmovie,4);

Td      = Td0;

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------

for k = 1:4
fprintf(1,'%d/%d (time = %.1f min)\n', k, floor(l/Td)+1, toc/60);
if (k <= floor(l/Td))
 data  = read(vid, [Td*(k-1)+1 Td*k]);
else
 data  = read(vid, [Td*floor(l/Td)+1 l]);
 Td    = length(data(1,1,1,:));
end
data      = data(1:160,:,:,:);
data      = permute(data, [1 2 4 3]);
data( 1:ny1,:,:,:) = flip(data( 1:ny1,:,:,:),1);
data( :,1:nx1,:,:) = flip(data( :,1:nx1,:,:),2);

%--------------------------------------------------------------------------------

fprintf(1,'left side\n');
j = 1;
% sensory input
s1                = zeros(Npca1,Td*6,'single');
s1(:,1:Td*3)      = Wpca * (cast(reshape(data( ny1*(1-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255 - mean1 * ones(1,Td*3,'single'));
s1(:,Td*3+1:Td*6) = Wpca * (cast(reshape(data( ny1*(2-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255 - mean1 * ones(1,Td*3,'single'));
% target
st1  = s1(:,[Kf+1:Td*6,1:Kf]);
% basis
phi  = diag(PCA_L1(1:Npca1))^(-1/2) * s1;
% predicted input
se11 = zeros(Npca1,Td*6,'single');
se12 = zeros(Npca1,Td*6,'single');
se13 = zeros(Npca1,Td*6,'single');
se14 = zeros(Npca1,Td*6,'single');
se15 = zeros(Npca1,Td*6,'single');
se16 = zeros(Npca1,Td*6,'single');
for m = 1:Kp
 tlist  = [Td*6-(m-1)+1:Td*6,1:Td*6-(m-1)];
 jlist  = Npca1*(m-1)+1:Npca1*m;
 se11   = se11 + Q1(:,jlist) * phi(:,tlist);
 se12   = se12 + Q2(:,jlist) * phi(:,tlist);
 se13   = se13 + Q3(:,jlist) * phi(:,tlist);
 se14   = se14 + Q4(:,jlist) * phi(:,tlist);
 se15   = se15 + Q5(:,jlist) * phi(:,tlist);
 se16   = se16 + Q6(:,jlist) * phi(:,tlist);
end
G      = zeros(6,Td*6);
G(1,:) = mean((se11(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s1).^2);
G(2,:) = mean((se12(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s1).^2);
G(3,:) = mean((se13(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s1).^2);
G(4,:) = mean((se14(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s1).^2);
G(5,:) = mean((se15(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s1).^2);
G(6,:) = mean((se16(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s1).^2);

flag    = reshape(mean(permute(reshape(G,[6 Td 6]),[3 1 2])),[6 Td]);
flag    = (ones(6,1)*min(flag) == flag);
flag1   = flag * 1;
for t = 2:Td, flag1(:,t) = flag1(:,t-1) + 0.1 * (-flag1(:,t-1) + flag(:,t)); end
flag1  = [flag1 flag1 flag1 flag1 flag1 flag1];

%--------------------------------------------------------------------------------

fprintf(1,'right side\n');
j = 2;
% sensory input
s2                = zeros(Npca1,Td*6,'single');
s2(:,1:Td*3)      = Wpca * (cast(reshape(data( ny1*(1-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255 - mean1 * ones(1,Td*3,'single'));
s2(:,Td*3+1:Td*6) = Wpca * (cast(reshape(data( ny1*(2-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255 - mean1 * ones(1,Td*3,'single'));
% target
st2  = s2(:,[Kf+1:Td*6,1:Kf]);
% basis
phi  = diag(PCA_L1(1:Npca1))^(-1/2) * s2;
% predicted input
se21 = zeros(Npca1,Td*6,'single');
se22 = zeros(Npca1,Td*6,'single');
se23 = zeros(Npca1,Td*6,'single');
se24 = zeros(Npca1,Td*6,'single');
se25 = zeros(Npca1,Td*6,'single');
se26 = zeros(Npca1,Td*6,'single');
for m = 1:Kp
 tlist  = [Td*6-(m-1)+1:Td*6,1:Td*6-(m-1)];
 jlist  = Npca1*(m-1)+1:Npca1*m;
 se21   = se21 + Q1(:,jlist) * phi(:,tlist);
 se22   = se22 + Q2(:,jlist) * phi(:,tlist);
 se23   = se23 + Q3(:,jlist) * phi(:,tlist);
 se24   = se24 + Q4(:,jlist) * phi(:,tlist);
 se25   = se25 + Q5(:,jlist) * phi(:,tlist);
 se26   = se26 + Q6(:,jlist) * phi(:,tlist);
end
G      = zeros(6,Td*6);
G(1,:) = mean((se21(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s2).^2);
G(2,:) = mean((se22(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s2).^2);
G(3,:) = mean((se23(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s2).^2);
G(4,:) = mean((se24(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s2).^2);
G(5,:) = mean((se25(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s2).^2);
G(6,:) = mean((se26(:,[Td*6-Kf+1:Td*6,1:Td*6-Kf]) - s2).^2);

flag    = reshape(mean(permute(reshape(G,[6 Td 6]),[3 1 2])),[6 Td]);
flag    = (ones(6,1)*min(flag) == flag);
flag2   = flag * 1;
for t = 2:Td, flag2(:,t) = flag2(:,t-1) + 0.1 * (-flag2(:,t-1) + flag(:,t)); end
flag2  = [flag2 flag2 flag2 flag2 flag2 flag2];

%--------------------------------------------------------------------------------

fprintf(1,'compute test prediction error\n')
se1      = se11 .* (ones(Npca1,1)*flag1(1,:)) + se12 .* (ones(Npca1,1)*flag1(2,:)) + se13 .* (ones(Npca1,1)*flag1(3,:)) + se14 .* (ones(Npca1,1)*flag1(4,:)) + se15 .* (ones(Npca1,1)*flag1(5,:)) + se16 .* (ones(Npca1,1)*flag1(6,:));
se2      = se21 .* (ones(Npca1,1)*flag2(1,:)) + se22 .* (ones(Npca1,1)*flag2(2,:)) + se23 .* (ones(Npca1,1)*flag2(3,:)) + se24 .* (ones(Npca1,1)*flag2(4,:)) + se25 .* (ones(Npca1,1)*flag2(5,:)) + se26 .* (ones(Npca1,1)*flag2(6,:));
err(1,:,k) = sum(reshape(sum((st1 - se1).^2) + sum((st2 - se2).^2), [Td 6])');
err(2,:,k) = sum(reshape(sum((st1 - se1).^2) + sum((st2 - se2).^2), [Td 6])');
amp(1,:,k) = sum(reshape(sum(st1.^2)         + sum(st2.^2),         [Td 6])');
amp(2,:,k) = sum(reshape(sum((st1 - s1).^2)  + sum((st2 - s2).^2),  [Td 6])');
fprintf(1,'error = %f\n', mean(err(1,:,k))/mean(amp(1,:,k)))
fprintf(1,'error = %f\n', mean(err(2,:,k))/mean(amp(2,:,k)))
csvwrite('predpca_test_prediction_error.csv',[mean(reshape(err(1,:,:),[Td 4])); mean(reshape(amp(1,:,:),[Td 4])); mean(reshape(err(2,:,:),[Td 4])); mean(reshape(amp(2,:,:),[Td 4]))])
avg_err = mean(reshape(err(2,:,k),[Td/10 10]))./mean(reshape(amp(2,:,k),[Td/10 10]));
plot(avg_err);
drawnow
clear se1 se2 st1 st2 s1 s2

%--------------------------------------------------------------------------------

fprintf(1,'----------------------------------------\n\n');
end

%--------------------------------------------------------------------------------
