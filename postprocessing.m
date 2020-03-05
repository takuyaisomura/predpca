
%--------------------------------------------------------------------------------

% postprocessing.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-3-5

%--------------------------------------------------------------------------------

function [ui,ui2,Wica,v2,G] = postprocessing(u,u2,label2,ica_rep,ica_eta)

Nu      = length(u(:,1));
Nv      = Nu;

u_std   = diag(std(u'));
u       = u_std^(-1) * u;
u2      = u_std^(-1) * u2;

%--------------------------------------------------------------------------------

fprintf(1,'ICA\n');
[ui,ui2,Wica] = ica(u,u2,ica_rep,ica_eta);

for i = 1:Nu
 skew     = skewness(ui2(i,:));
 ui(i,:)  = sign(skew) * ui(i,:);
 ui2(i,:) = sign(skew) * ui2(i,:);
end

u_std   = diag(std(ui'));   % normalize variance
ui      = sqrt(0.1) * u_std^(-1) * ui;
ui2     = sqrt(0.1) * u_std^(-1) * ui2;

%--------------------------------------------------------------------------------

% winner-takes-all
v2      = (ones(Nv,1)*max(ui2) == ui2) * 1;

% categorization error
G = zeros(10,10);
for i=1:10, G(i,:) = sum(((ones(Nv,1)*(label2==i-1)) .* v2)'); end
fprintf(1,'categorization error = %f\n', mean(1-max(G)./(sum(G)+0.001)));

%--------------------------------------------------------------------------------

