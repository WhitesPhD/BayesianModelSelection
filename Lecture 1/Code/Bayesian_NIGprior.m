%% Implementation of the normal-inverse gamma prior 

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

% initial hyper-priors

Sigma_0     = eye(p)*1e4; % flat prior
beta_0      = zeros(p,1);

a_0         = 0.1;
b_0         = 10;

nsamples    = 1e4;

% create store variables
beta_draw   = zeros(nsamples,p);
sigma_draw  = zeros(nsamples,1);

XtX         = x'*x;
Xty         = x'*y;
V_inv       = Sigma_0\eye(size(Sigma_0));
P_post      = V_inv+XtX;
Sigma_post  = P_post\eye(size(P_post));

m_post      = Sigma_post*(V_inv*beta_0+Xty);
a_post      = a_0 +(n+p)*0.5;
b_post      = b_0 + 0.5 * sum((y - x*m_post).^2); 

% set counting variable for the number of iterations
no_samples  = 0;


while no_samples < nsamples 
     no_samples = no_samples + 1;
       
     % sigma draws 
     sig_inv = 1/b_post * randgamma(a_post,1); %random draw from a gamma (a_post, 1/b_post)
     sigma   = 1/sig_inv;
     
     % beta draws
     beta = mvrn(m_post,sigma*Sigma_post,1)';
     
     beta_draw(no_samples,:) = beta;
     sigma_draw(no_samples)  = sigma;
end
     
figure(1)
histogram(sigma_draw)
title('posterior distribution of \sigma^2')




