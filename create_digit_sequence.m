
%--------------------------------------------------------------------------------

% create_digit_sequence.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [input,input2,input3,label,label2,label3] = create_digit_sequence(dir,sequence_type,T,T2,T3,test_randomness,sign_flip)

fprintf(1,'read files\n');
M        = 60000;
M2       = 10000;
Nimg     = 28*28;
fid      = fopen([dir 'train-images-idx3-ubyte']);
data     = fread(fid); data = reshape(data(17:28*28*M+16,:),[28*28 M]) / 255;
fclose(fid);
fid      = fopen([dir 'train-labels-idx1-ubyte']);
label_d  = fread(fid); label_d = reshape(label_d(9:M+8,:),[1 M]);
fclose(fid);
fid      = fopen([dir 't10k-images-idx3-ubyte']);
data2    = fread(fid); data2 = reshape(data2(17:28*28*M2+16,:),[28*28 M2]) / 255;
fclose(fid);
fid      = fopen([dir 't10k-labels-idx1-ubyte']);
label_d2 = fread(fid); label_d2 = reshape(label_d2(9:M2+8,:),[1 M2]);
fclose(fid);

lab      = cell(10,1);
lab2     = cell(10,1);
for i = 1:10, lab{i,1}  = find(label_d  == i-1); end
for i = 1:10, lab2{i,1} = find(label_d2 == i-1); end

%--------------------------------------------------------------------------------

fprintf(1,'create sequences\n');
input    = zeros(Nimg,T);
input2   = zeros(Nimg,T2);
input3   = zeros(Nimg,T3);
label    = zeros(1,T);
label2   = zeros(1,T2);
label3   = zeros(1,T3);

if (sequence_type == 1)
 count = 0; pm = 1;
 for t = 1:T
  rnd        = randi([1 length(lab{count+1,1})]);
  input(:,t) = pm * data(:,lab{count+1,1}(1,rnd));
  label(:,t) = label_d(:,lab{count+1,1}(1,rnd));
  count = rem(count+1,10);
  if (randi([1 50]) == 1)
   count = randi([0 9]);
   if (sign_flip == 1), pm = -pm; end
  end
 end
 count = 0; pm = 1;
 for t = 1:T2
  rnd         = randi([1 length(lab2{count+1,1})]);
  input2(:,t) = pm * data2(:,lab2{count+1,1}(1,rnd));
  label2(:,t) = label_d2(:,lab2{count+1,1}(1,rnd));
  count = rem(count+1,10);
  if (test_randomness == 1)
   if (randi([1 50]) == 1)
    count = randi([0 9]);
    if (sign_flip == 1), pm = -pm; end
   end
  end
 end
 count = 0; pm = 1;
 for t = 1:T3
  rnd         = randi([1 length(lab{count+1,1})]);
  input3(:,t) = pm * data(:,lab{count+1,1}(1,rnd));
  label3(:,t) = label_d(:,lab{count+1,1}(1,rnd));
  count = rem(count+1,10);
  if (randi([1 50]) == 1)
   count = randi([0 9]);
   if (sign_flip == 1), pm = -pm; end
  end
 end
end

if (sequence_type == 2)
 count = 0; count2 = 1; pm = 1;
 for t = 1:T
  rnd        = randi([1 length(lab{count+1,1})]);
  input(:,t) = pm * data(:,lab{count+1,1}(1,rnd));
  label(:,t) = label_d(:,lab{count+1,1}(1,rnd));
  count3 = rem(count + count2, 10);
  count  = count2;
  count2 = count3;
  if (randi([1 200]) == 1)
   count2 = randi([0 9]);
   if (sign_flip == 1), pm = -pm; end
  end
 end
 count = 0; count2 = 1; pm = 1;
 for t = 1:T2
  rnd         = randi([1 length(lab2{count+1,1})]);
  input2(:,t) = pm * data2(:,lab2{count+1,1}(1,rnd));
  label2(:,t) = label_d2(:,lab2{count+1,1}(1,rnd));
  count3 = rem(count + count2, 10);
  count  = count2;
  count2 = count3;
  if (test_randomness == 1)
   if (randi([1 200]) == 1)
    count2 = randi([0 9]);
    if (sign_flip == 1), pm = -pm; end
   end
  end
 end
 count = 0; count2 = 1; pm = 1;
 for t = 1:T3
  rnd         = randi([1 length(lab{count+1,1})]);
  input3(:,t) = pm * data(:,lab{count+1,1}(1,rnd));
  label3(:,t) = label_d(:,lab{count+1,1}(1,rnd));
  count3 = rem(count + count2, 10);
  count  = count2;
  count2 = count3;
  if (randi([1 200]) == 1)
   count2 = randi([0 9]);
   if (sign_flip == 1), pm = -pm; end
  end
 end
end
clear data data2

%--------------------------------------------------------------------------------
