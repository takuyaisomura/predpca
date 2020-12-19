
%--------------------------------------------------------------------------------

% wta_prediction.m
% https://github.com/takuyaisomura/predpca
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-5-27

%--------------------------------------------------------------------------------

function [output,ui3,A,B,AIC] = wta_prediction(ui,v2,input,input2,input_mean,prior_u,prior_u_,model_type)

% initialization
Nu    = length(ui(:,1));
T     = length(ui(1,:));

Tpred = T + 1000;                             % length of prediction
A     = (input*ui') / (ui*ui'+eye(Nu)*prior_u);
ui    = abs(ui);
vi    = (ui == ones(Nu,1)*max(ui)) * 1;

if (model_type == 1)                          % for ascending
  
  ui_ = ui;
  % optimal transition matrix
  B   = (ui(:,[2:T,1])*ui_') / (ui_*ui_'+eye(Nu)*prior_u);
  ui3 = v2;
  for t = 61:Tpred
    ui3(:,t) = B * ui3(:,t-1);                  % state transition
    ui3(:,t) = (ui3(:,t) == max(ui3(:,t))) * 1; % winner-takes-all
  end
  vi_ = vi;
  AIC = T*log(det(cov((vi(:,[2:T,1]) - B*vi_)'))) + 2*Nu*Nu;
  fprintf(1,'%f\n',log(det(cov((vi(:,[2:T,1]) - B*vi_)'))));
  
elseif (model_type == 2)                      % for Fibonacci
  
  ui_ = kron(ui(:,[2:T,1]),ones(Nu,1)).*kron(ones(Nu,1),ui);
  % optimal transition matrix
  B   = (ui(:,[3:T,1:2])*ui_') / (ui_*ui_'+eye(Nu^2)*prior_u_);
  ui3 = v2;
  for t = 61:Tpred
    ui3(:,t) = B * kron(ui3(:,t-1), ui3(:,t-2)); % state transition
    ui3(:,t) = (ui3(:,t) == max(ui3(:,t))) * 1;  % winner-takes-all
  end
  vi_ = kron(vi(:,[2:T,1]),ones(Nu,1)).*kron(ones(Nu,1),vi);
  AIC = T*log(det(cov((vi(:,[3:T,1:2]) - B*vi_)'))) + 2*Nu*Nu^2;
  fprintf(1,'%f\n',log(det(cov((vi(:,[3:T,1:2]) - B*vi_)'))));
  
elseif (model_type == 3)
  
  ui_ = kron(ui(:,[3:T,1:2]),ones(Nu^2,1)).*kron(ones(Nu,1),kron(ui(:,[2:T,1]),ones(Nu,1))).*kron(ones(Nu^2,1),ui);
  % optimal transition matrix
  B   = (ui(:,[4:T,1:3])*ui_') / (ui_*ui_'+eye(Nu^3)*prior_u_);
  ui3 = v2;
  for t = 61:Tpred
    ui3(:,t) = B * kron(kron(ui3(:,t-1), ui3(:,t-2)), ui3(:,t-3)); % state transition
    ui3(:,t) = (ui3(:,t) == max(ui3(:,t))) * 1;  % winner-takes-all
  end
  vi_ = kron(vi(:,[3:T,1:2]),ones(Nu^2,1)).*kron(ones(Nu,1),kron(vi(:,[2:T,1]),ones(Nu,1))).*kron(ones(Nu^2,1),vi);
  AIC = T*log(det(cov((vi(:,[4:T,1:3]) - B*vi_)'))) + 2*Nu*Nu^3;
  fprintf(1,'%f\n',log(det(cov((vi(:,[4:T,1:3]) - B*vi_)'))));
  
elseif (model_type == 4)
  
  ui_ = kron(ui(:,[4:T,1:3]),ones(Nu^3,1)).*kron(ones(Nu,1),kron(ui(:,[3:T,1:2]),ones(Nu^2,1))).*kron(ones(Nu^2,1),kron(ui(:,[2:T,1]),ones(Nu,1))).*kron(ones(Nu^3,1),ui);
  % optimal transition matrix
  B   = (ui(:,[5:T,1:4])*ui_') / (ui_*ui_'+eye(Nu^4)*prior_u_);
  ui3 = v2;
  for t = 61:Tpred
    ui3(:,t) = B * kron(kron(ui3(:,t-1), ui3(:,t-2)), kron(ui3(:,t-3),ui3(:,t-4))); % state transition
    ui3(:,t) = (ui3(:,t) == max(ui3(:,t))) * 1;  % winner-takes-all
  end
  vi_ = kron(vi(:,[4:T,1:3]),ones(Nu^3,1)).*kron(ones(Nu,1),kron(vi(:,[3:T,1:2]),ones(Nu^2,1))).*kron(ones(Nu^2,1),kron(vi(:,[2:T,1]),ones(Nu,1))).*kron(ones(Nu^3,1),vi);
  AIC = T*log(det(cov((vi(:,[5:T,1:4]) - B*vi_)'))) + 2*Nu*Nu^4;
  fprintf(1,'%f\n',log(det(cov((vi(:,[5:T,1:4]) - B*vi_)'))));
  
end

% visualize prediction results
output = A * ui3 + input_mean * ones(1,Tpred);

