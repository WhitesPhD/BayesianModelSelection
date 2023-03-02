clear all; clc; pause(0.01), randn('seed',3212), rand('seed',3212), warning off

%--------------------------------------------------------------------------
% Adding folders
%--------------------------------------------------------------------------

addpath([pwd '/Data/']);

Data_anomalies        = readtable('dataLongShortMissing','ReadVariableNames',true);
Data_target           = readtable('dataFf3','ReadVariableNames',true);

x = table2array(Data_anomalies(:,2:end));

% replace nan with the cross sectional average
for i=1:size(x,2)
    if sum(isnan(x(:,i))) > 0
    x(isnan(x(:,i)),i) = nanmean(x(isnan(x(:,i)),:),2);
    else
    continue   
    end
end

y = Data_target.MKT;
y = y(2:end);x = x(1:end-1,:);
[X, muX, normX, y, muY] = standardise(x, y);

[n,p]    = size(X);      % size of the data set
[~,R]    = qr(X,0);
rankx    = sum(abs(diag(R)) > abs(R(1))*n*eps);
 
if(rankx < p)
   error('Matrix X is not full rank. Cannot use the g-prior.');
end
    

train      = 0.6; cv = 0.2;
insample   = round(size(x,1)*0.6); 
validation = round(size(x,1)*0.2);

x_train  = X(1:insample,:);y_train = y(1:insample);
x_cv     = X(insample+1:insample+validation,:);y_cv = y(insample+1:insample+validation);
x_test   = X(insample+validation+1:end,:);y_test  = y(insample+validation+1:end,:);

XtX = x_train'*x_train;           % precompute once
Xty = x_train'*y_train; 

% ols estimates
beta_ols  = XtX\Xty;

% grid of points to evaluate the g prior
g        = [0:1:insample]';

% g-prior estimates
beta        = repmat(g./(g+1),1,size(beta_ols,1)).* repmat(beta_ols',size(g,1),1);
y_hat       = x_cv * beta';
[M,I]       = min(mean((y_cv - y_hat).^2)); 

% forecast based on the cross-validated g 
beta        = g(I) / (g(I)+1) * beta_ols;
y_hat       = x_test * beta;

err         = [nanmean((y_test - y_hat).^2) nanmean((y_test - nanmean([y_train; y_cv])).^2)];

R2_oos      = (1-sum(err(1).^2)/sum(err(2).^2))*100;

display(R2_oos)



