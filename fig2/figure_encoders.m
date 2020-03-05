
%--------------------------------------------------------------------------------

% figure_encoders.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function figure_encoders(ui2,label2,t)

T2 = length(ui2(1,:));
c  = zeros(3,T2);
c(:,find(label2==0)) = [1   0   0  ]'*ones(1,length(find(label2==0)));
c(:,find(label2==1)) = [1   0.5 0  ]'*ones(1,length(find(label2==1)));
c(:,find(label2==2)) = [1   1   0  ]'*ones(1,length(find(label2==2)));
c(:,find(label2==3)) = [0   1   0  ]'*ones(1,length(find(label2==3)));
c(:,find(label2==4)) = [0   1   0.5]'*ones(1,length(find(label2==4)));
c(:,find(label2==5)) = [0   1   1  ]'*ones(1,length(find(label2==5)));
c(:,find(label2==6)) = [0   0.5 1  ]'*ones(1,length(find(label2==6)));
c(:,find(label2==7)) = [0   0   1  ]'*ones(1,length(find(label2==7)));
c(:,find(label2==8)) = [0.5 0   1  ]'*ones(1,length(find(label2==8)));
c(:,find(label2==9)) = [1   0   1  ]'*ones(1,length(find(label2==9)));

subplot(3,2,1), scatter(ui2(1,t),ui2(2,t),10,c(:,t)')
subplot(3,2,2), scatter(ui2(3,t),ui2(4,t),10,c(:,t)')
subplot(3,2,3), scatter(ui2(5,t),ui2(6,t),10,c(:,t)')
subplot(3,2,4), scatter(ui2(7,t),ui2(8,t),10,c(:,t)')
subplot(3,2,5), scatter(ui2(9,t),ui2(10,t),10,c(:,t)')

