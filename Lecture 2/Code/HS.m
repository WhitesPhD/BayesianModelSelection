%% Implementation of the horseshoe prior 
% Reference
% Makalic, Enes, and Daniel F. Schmidt. "A simple sampler for the horseshoe estimator." 
% IEEE Signal Processing Letters 23.1 (2015): 179-182.

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

% standardise the target and the predictors
y           = zscore(Data_y.MKT);
x           = zscore(x);

lag         = 1;
y           = y(lag + 1 :end);
x           = x(1:end - lag,:);

[T, p]      = size(x);

nsamples    = 10000;
burnin      = 1000;
thin        = 10;

beta        = zeros(p, nsamples);
s2          = zeros(1, nsamples);
t2          = zeros(1, nsamples);
l2          = zeros(p, nsamples);

%% Initial values
sigma2      = 1;
lambda2     = rand(p, 1);
tau2        = 1;
nu          = ones(p,1);
xi          = 1;

XtX = x'*x;             % pre-compute X'*X
Xty = x'*y;

%% Gibbs sampler
k = 0;
iter = 0;

while(k < nsamples)
    %% Sample from the conditional posterior dist. for beta
   
    sigma = sqrt(sigma2);
    D = tau2 * diag(lambda2);
    b = fastmvg_rue(x ./ sigma, XtX ./ sigma2, y ./ sigma, sigma2 * D);
    
    %% Sample sigma2
    e = y - x*b;
    shape = (T + p) / 2;
    scale = e'*e/2 + sum(b.^2 ./ lambda2)/tau2/2;
    sigma2 = 1 / gamrnd(shape, 1/scale);
    
    %% Sample lambda2
    scale = 1./nu + b.^2./2./tau2./sigma2;
    lambda2 = 1 ./ exprnd(1./scale);
    
    %% Sample tau2
    shape = (p + 1)/2;
    scale = 1/xi + sum(b.^2./lambda2)/2/sigma2;
    tau2 = 1 / gamrnd(shape, 1/scale);
    
    %% Sample nu
    scale = 1 + 1./lambda2;
    nu = 1 ./ exprnd(1./scale);
    
    %% Sample xi
    scale = 1 + 1/tau2;
    xi = 1 / exprnd(1/scale);
  
    %% Store samples    
    iter = iter + 1;
    if(iter > burnin)
        
        % thinning
        if(mod(iter,thin) == 0)
            k = k + 1;
            beta(:,k) = b;
            s2(k) = sigma2;
            t2(k) = tau2;
            l2(:,k) = lambda2;
        end
    end
end


