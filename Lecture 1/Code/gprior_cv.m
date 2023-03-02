%% Implementation of the g-prior for Bayesian inference with cross-validation 

clear all; clc; pause(0.01), randn('seed',3212), rand('seed',3212), warning off

% addpath([pwd '/Data/']);
% upload the data 
% data on the anomaly based portfolios are from 
% Dong, Xi, et al. "Anomalies and the expected market return." The Journal of Finance 77.1 (2022): 639-681.
% data on the equity premium (target variable) are from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Data_x      = readtable('dataLongShortMissing','ReadVariableNames',true);
Data_y      = readtable('dataFf3','ReadVariableNames',true);

% create target and predictors 
x           = table2array(Data_x(:,2:end));

% replace nan with the cross sectional average
for i = 1 : size(x,2)
    if sum(isnan(x(:,i))) > 0
    x(isnan(x(:,i)),i) = nanmean(x(isnan(x(:,i)),:),2);
    else
    continue   
    end
end

y           = Data_y.MKT;
x           = [ones(size(x,1),1) x];

lag         = 1;
y           = y(lag + 1 :end);
x           = x(1:end - lag,:);
[n,p]       = size(x);      % size of the data set

% separate training vs testing

train       = 0.6; cv = 0.2;
insample    = round(size(x,1)*0.6); 
validation  = round(size(x,1)*0.2);

xtrain      = x(1:insample,:);
ytrain      = y(1:insample);
xcv         = x(insample+1:insample+validation,:);
ycv         = y(insample+1:insample+validation);
xtest       = x(insample+validation+1:end,:);
ytest       = y(insample+validation+1:end,:);

XtX         = xtrain'*xtrain;           % precompute once
Xty         = xtrain'*ytrain; 

% ols estimates
beta_ols    = XtX\Xty;

% grid of points to evaluate the g prior
g           = [.1:.1:p]';

% g-prior estimates
beta        = repmat(g./(g+1),1,size(beta_ols,1)).* repmat(beta_ols',size(g,1),1);
yhat        = xcv * beta';
[M,I]       = min(mean((ycv - yhat).^2)); 

% forecast based on the cross-validated g 
beta        = g(I) / (g(I)+1) * beta_ols;
yhat        = xtest * beta;

r2_oos      = R2oos(yhat,ytest,nanmean([ytrain;ycv]))*100;

display([r2_oos, g(I)])


