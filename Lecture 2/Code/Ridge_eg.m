
%% Implementation of the ridge regression (example) 

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

lambda = [1:1:10000];

XtX = x'*x;
Xty = x'*y;

beta = [];
for j = 1:length(lambda)
beta = [beta inv(XtX + lambda(j) * eye(size(x,2))) * Xty]; 
end


