
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
% 2020-6-27
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
Mixture = 1;         % using mixture model

%--------------------------------------------------------------------------------
% compute optimal matrices for predictions

fileid  = 19;
load([dirname,'Kf',num2str(Kf),'v3/mle_lv1_',num2str(fileid),'.mat'])

load([dirname,'pca_lv1_dst.mat'])
Wpca    = PCA_C1(:,1:Npca1)';

fprintf(1,'compute Q (time = %.1f min)\n', toc/60);
if (Mixture == 1)
 STS_t1  = STS_t1 / Tpart_t(1,1);
 STS_t2  = STS_t2 / Tpart_t(2,1);
 STS_t3  = STS_t3 / Tpart_t(3,1);
 STS_t4  = STS_t4 / Tpart_t(4,1);
 STS_t5  = STS_t5 / Tpart_t(5,1);
 STS_t6  = STS_t6 / Tpart_t(6,1);
 STS_b1  = STS_b1 / Tpart_b(1,1);
 STS_b2  = STS_b2 / Tpart_b(2,1);
 STS_b3  = STS_b3 / Tpart_b(3,1);
 STS_b4  = STS_b4 / Tpart_b(4,1);
 STS_b5  = STS_b5 / Tpart_b(5,1);
 STS_b6  = STS_b6 / Tpart_b(6,1);
 S_S_t1  = S_S_t1 / Tpart_t(1,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_t2  = S_S_t2 / Tpart_t(2,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_t3  = S_S_t3 / Tpart_t(3,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_t4  = S_S_t4 / Tpart_t(4,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_t5  = S_S_t5 / Tpart_t(5,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_t6  = S_S_t6 / Tpart_t(6,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_b1  = S_S_b1 / Tpart_b(1,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_b2  = S_S_b2 / Tpart_b(2,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_b3  = S_S_b3 / Tpart_b(3,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_b4  = S_S_b4 / Tpart_b(4,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_b5  = S_S_b5 / Tpart_b(5,1) + eye(Npca1*Kp) * 10^(-6);
 S_S_b6  = S_S_b6 / Tpart_b(6,1) + eye(Npca1*Kp) * 10^(-6);
 Qt      = cell(6,1);
 Qt{1,1} = cast(STS_t1,'double') / cast(S_S_t1,'double');
 Qt{2,1} = cast(STS_t2,'double') / cast(S_S_t2,'double');
 Qt{3,1} = cast(STS_t3,'double') / cast(S_S_t3,'double');
 Qt{4,1} = cast(STS_t4,'double') / cast(S_S_t4,'double');
 Qt{5,1} = cast(STS_t5,'double') / cast(S_S_t5,'double');
 Qt{6,1} = cast(STS_t6,'double') / cast(S_S_t6,'double');
 Qb      = cell(6,1);
 Qb{1,1} = cast(STS_b1,'double') / cast(S_S_b1,'double');
 Qb{2,1} = cast(STS_b2,'double') / cast(S_S_b2,'double');
 Qb{3,1} = cast(STS_b3,'double') / cast(S_S_b3,'double');
 Qb{4,1} = cast(STS_b4,'double') / cast(S_S_b4,'double');
 Qb{5,1} = cast(STS_b5,'double') / cast(S_S_b5,'double');
 Qb{6,1} = cast(STS_b6,'double') / cast(S_S_b6,'double');
 
 SESEt   = (Qt{1,1}*S_S_t1*Qt{1,1}' + Qt{2,1}*S_S_t2*Qt{2,1}' + Qt{3,1}*S_S_t3*Qt{3,1}' + Qt{4,1}*S_S_t4*Qt{4,1}' + Qt{5,1}*S_S_t5*Qt{5,1}' + Qt{6,1}*S_S_t6*Qt{6,1}')/6;
 SESEb   = (Qb{1,1}*S_S_b1*Qb{1,1}' + Qb{2,1}*S_S_b2*Qb{2,1}' + Qb{3,1}*S_S_b3*Qb{3,1}' + Qb{4,1}*S_S_b4*Qb{4,1}' + Qb{5,1}*S_S_b5*Qb{5,1}' + Qb{6,1}*S_S_b6*Qb{6,1}')/6;
else
 STS_t1  = (STS_t1+STS_t2+STS_t3+STS_t4+STS_t5+STS_t6) / sum(Tpart_t);
 STS_b1  = (STS_b1+STS_b2+STS_b3+STS_b4+STS_b5+STS_b6) / sum(Tpart_b);
 S_S_t1  = (S_S_t1+S_S_t2+S_S_t3+S_S_t4+S_S_t5+S_S_t6) / sum(Tpart_t) + eye(Npca1*Kp) * 10^(-6);
 S_S_b1  = (S_S_b1+S_S_b2+S_S_b3+S_S_b4+S_S_b5+S_S_b6) / sum(Tpart_b) + eye(Npca1*Kp) * 10^(-6);
 Qt      = cell(1,1);
 Qt{1,1} = cast(STS_t1,'double') / cast(S_S_t1,'double');
 Qb      = cell(1,1);
 Qb{1,1} = cast(STS_b1,'double') / cast(S_S_b1,'double');
 
 SESEt   = Qt{1,1}*S_S_t1*Qt{1,1}';
 SESEb   = Qb{1,1}*S_S_b1*Qb{1,1}';
end

% post-hoc PCA
[PPCA_C1t,PPCA_L1t] = pcacov(SESEt);
[PPCA_C1b,PPCA_L1b] = pcacov(SESEb);

if (Mixture == 1)
 Lambdat             = zeros(6,Npca1);
 [~,Lambdat(1,:)]    = pcacov(Qt{1,1}*S_S_t1*Qt{1,1}');
 [~,Lambdat(2,:)]    = pcacov(Qt{2,1}*S_S_t2*Qt{2,1}');
 [~,Lambdat(3,:)]    = pcacov(Qt{3,1}*S_S_t3*Qt{3,1}');
 [~,Lambdat(4,:)]    = pcacov(Qt{4,1}*S_S_t4*Qt{4,1}');
 [~,Lambdat(5,:)]    = pcacov(Qt{5,1}*S_S_t5*Qt{5,1}');
 [~,Lambdat(6,:)]    = pcacov(Qt{6,1}*S_S_t6*Qt{6,1}');
 Lambdab             = zeros(6,Npca1);
 [~,Lambdab(1,:)]    = pcacov(Qb{1,1}*S_S_b1*Qb{1,1}');
 [~,Lambdab(2,:)]    = pcacov(Qb{2,1}*S_S_b2*Qb{2,1}');
 [~,Lambdab(3,:)]    = pcacov(Qb{3,1}*S_S_b3*Qb{3,1}');
 [~,Lambdab(4,:)]    = pcacov(Qb{4,1}*S_S_b4*Qb{4,1}');
 [~,Lambdab(5,:)]    = pcacov(Qb{5,1}*S_S_b5*Qb{5,1}');
 [~,Lambdab(6,:)]    = pcacov(Qb{6,1}*S_S_b6*Qb{6,1}');
else
 Lambdat             = zeros(1,Npca1);
 [~,Lambdat(1,:)]    = pcacov(Qt{1,1}*S_S_t1*Qt{1,1}');
 Lambdab             = zeros(1,Npca1);
 [~,Lambdab(1,:)]    = pcacov(Qb{1,1}*S_S_b1*Qb{1,1}');
end

% dimensionality reduction
Nppca1  = 2000;
Nppca2  = 1600;
Nppca3  = 1600;
Nppca4  = 1600;
Nppca5  = 1600;
Nppca6  = 1600;
if (Mixture == 1)
 Qt{1,1} = PPCA_C1t(:,1:Nppca1) * PPCA_C1t(:,1:Nppca1)' * Qt{1,1};
 Qt{2,1} = PPCA_C1t(:,1:Nppca2) * PPCA_C1t(:,1:Nppca2)' * Qt{2,1};
 Qt{3,1} = PPCA_C1t(:,1:Nppca3) * PPCA_C1t(:,1:Nppca3)' * Qt{3,1};
 Qt{4,1} = PPCA_C1t(:,1:Nppca4) * PPCA_C1t(:,1:Nppca4)' * Qt{4,1};
 Qt{5,1} = PPCA_C1t(:,1:Nppca5) * PPCA_C1t(:,1:Nppca5)' * Qt{5,1};
 Qt{6,1} = PPCA_C1t(:,1:Nppca6) * PPCA_C1t(:,1:Nppca6)' * Qt{6,1};
 Qb{1,1} = PPCA_C1b(:,1:Nppca1) * PPCA_C1b(:,1:Nppca1)' * Qb{1,1};
 Qb{2,1} = PPCA_C1b(:,1:Nppca2) * PPCA_C1b(:,1:Nppca2)' * Qb{2,1};
 Qb{3,1} = PPCA_C1b(:,1:Nppca3) * PPCA_C1b(:,1:Nppca3)' * Qb{3,1};
 Qb{4,1} = PPCA_C1b(:,1:Nppca4) * PPCA_C1b(:,1:Nppca4)' * Qb{4,1};
 Qb{5,1} = PPCA_C1b(:,1:Nppca5) * PPCA_C1b(:,1:Nppca5)' * Qb{5,1};
 Qb{6,1} = PPCA_C1b(:,1:Nppca6) * PPCA_C1b(:,1:Nppca6)' * Qb{6,1};
else
 Qt{1,1} = PPCA_C1t(:,1:Nppca1) * PPCA_C1t(:,1:Nppca1)' * Qt{1,1};
 Qb{1,1} = PPCA_C1b(:,1:Nppca1) * PPCA_C1b(:,1:Nppca1)' * Qb{1,1};
end

subplot(1,2,1), plot(1:Npca1,log10(Lambdat'),1:Npca1,log10(PCA_L1(1:Npca1)),'k--'), title('top')
subplot(1,2,2), plot(1:Npca1,log10(Lambdab'),1:Npca1,log10(PCA_L1(1:Npca1)),'k--'), title('bottom')

clear STS_t1 STS_t2 STS_t3 STS_t4 STS_t5 STS_t6
clear STS_b1 STS_b2 STS_b3 STS_b4 STS_b5 STS_b6
clear S_S_t1 S_S_t2 S_S_t3 S_S_t4 S_S_t5 S_S_t6
clear S_S_b1 S_S_b2 S_S_b3 S_S_b4 S_S_b5 S_S_b6

%--------------------------------------------------------------------------------
% predict 0.5 s future image of test data

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

decay   = 0.1;
Td      = Td0;

%--------------------------------------------------------------------------------

Nu      = 100;
Wppcat  = PPCA_C1t(:,1:Nu)';
Wppcab  = PPCA_C1b(:,1:Nu)';
save([dirname,'predpca_lv1_dst.mat'], 'PPCA_C1t', 'PPCA_L1t', 'PPCA_C1b', 'PPCA_L1b', '-v7.3')

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

fprintf(1,'compute predicted inputs\n')
s  = cell(2,2);
st = cell(2,2);
se = cell(2,2);
u  = cell(2,2);
% compute top left, top right, bottom left, bottom right areas
for i = 1:2
 for j = 1:2
  fprintf(1,'(i,j)=(%d,%d) ',i,j)
  s{i,j}  = Wpca * (cast(reshape(data(ny1*(i-1)+(1:ny1),nx1*(j-1)+(1:nx1),:,:),[Ndata1 Td*3]),'single')/255 - mean1 * ones(1,Td*3,'single')); % sensory input
  st{i,j} = s{i,j}(:,[Kf+1:Td*3,1:Kf]);            % target
  phi     = diag(PCA_L1(1:Npca1))^(-1/2) * s{i,j}; % basis
  phi1    = [phi; phi(:,[Td*3,1:Td*3-1]); phi(:,[Td*3-1:Td*3,1:Td*3-2]); phi(:,[Td*3-2:Td*3,1:Td*3-3]); phi(:,[Td*3-3:Td*3,1:Td*3-4]); phi(:,[Td*3-4:Td*3,1:Td*3-5]); phi(:,[Td*3-5:Td*3,1:Td*3-6]); phi(:,[Td*3-6:Td*3,1:Td*3-7])];
  if (Mixture == 1)
   ses     = cell(6,1);                             % predicted input
   G       = zeros(6,Td*3);                         % prediction error under each model
   for h = 1:6
    if (i == 1), ses{h,1} = Qt{h,1} * phi1; end
    if (i == 2), ses{h,1} = Qb{h,1} * phi1; end
    G(h,:) = mean((ses{h,1}(:,[Td*3-Kf+1:Td*3,1:Td*3-Kf]) - s{i,j}).^2);
    fprintf(1,'.')
   end
   flag    = reshape(mean(permute(reshape(G,[6 Td 3]),[3 1 2])),[6 Td]);
   flag    = (ones(6,1)*min(flag) == flag) * 1;
   for t = 2:Td, flag(:,t) = flag(:,t-1) + decay * (-flag(:,t-1) + flag(:,t)); end
   flag    = [flag flag flag];
   v_one   = ones(Npca1,1);
   se{i,j} = ses{1,1}.*(v_one*flag(1,:)) + ses{2,1}.*(v_one*flag(2,:)) + ses{3,1}.*(v_one*flag(3,:)) + ses{4,1}.*(v_one*flag(4,:)) + ses{5,1}.*(v_one*flag(5,:)) + ses{6,1}.*(v_one*flag(6,:));
  else
   if (i == 1), se{i,j} = Qt{1,1} * phi1; end
   if (i == 2), se{i,j} = Qb{1,1} * phi1; end
  end
  if (i == 1), u{i,j} = Wppcat * se{i,j}; end
  if (i == 2), u{i,j} = Wppcab * se{i,j}; end
  fprintf(1,' err_part = %f\n', mean(sum(reshape(sum((st{i,j}-se{i,j}).^2), [Td 3])')) / mean(sum(reshape(sum((st{i,j}-s{i,j}).^2), [Td 3])')))
 end
end

%--------------------------------------------------------------------------------

fprintf(1,'compute test prediction error\n')
err(1,:,k) = sum(reshape(sum((st{1,1}-se{1,1}).^2) + sum((st{1,2}-se{1,2}).^2) + sum((st{2,1}-se{2,1}).^2) + sum((st{2,2}-se{2,2}).^2), [Td 3])');
err(2,:,k) = sum(reshape(sum((st{1,1}-se{1,1}).^2) + sum((st{1,2}-se{1,2}).^2) + sum((st{2,1}-se{2,1}).^2) + sum((st{2,2}-se{2,2}).^2), [Td 3])');
amp(1,:,k) = sum(reshape(sum(st{1,1}.^2)           + sum(st{1,2}.^2)           + sum(st{2,1}.^2)           + sum(st{2,2}.^2),           [Td 3])');
amp(2,:,k) = sum(reshape(sum((st{1,1}-s{1,1}).^2)  + sum((st{1,2}-s{1,2}).^2)  + sum((st{2,1}-s{2,1}).^2)  + sum((st{2,2}-s{2,2}).^2),  [Td 3])');
fprintf(1,'error = %f\n', mean(err(1,:,k))/mean(amp(1,:,k)))
fprintf(1,'error = %f\n', mean(err(2,:,k))/mean(amp(2,:,k)))
csvwrite('predpca_test_prediction_error.csv',[mean(reshape(err(1,:,:),[Td 4])); mean(reshape(amp(1,:,:),[Td 4])); mean(reshape(err(2,:,:),[Td 4])); mean(reshape(amp(2,:,:),[Td 4]))])
avg_err = mean(reshape(err(2,:,k),[Td/10 10]))./mean(reshape(amp(2,:,k),[Td/10 10]));
plot(avg_err);
drawnow

save([dirname,'predpca_lv1_u_', num2str(k), '.mat'], 'u', '-v7.3')
%clear se st s

%--------------------------------------------------------------------------------

fprintf(1,'----------------------------------------\n\n');
end

%--------------------------------------------------------------------------------
