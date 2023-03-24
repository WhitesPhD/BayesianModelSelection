%% Implementation of the Bayesian model averaging under the g-prior

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

lag         = 1;
y           = y(lag + 1 :end);
x           = x(1:end - lag,:);
Xindex      = [1:15];          %  <----------------***** CHANGE *****
X           = x(:,Xindex);

[T, p]      = size(X);

% ======================| Form all possible model combinations |======================
% if z_t (the full model matrix of regressors) has N elements, all possible combinations 
% are (2^N - 1), i.e. 2^N minus the model with all predictors/constant excluded (y_t = error)

comb = cell(p,1);
for nn = 1:p
    % 'comb' has N cells with all possible combinations for each N
    comb{nn,1} = combntns(1:p,nn);
end

K = 2.^p;  %Total number of models
index_temp = cell(K,1);
dim        = zeros(1,p+1);
for nn=1:p
    dim(:,nn+1) = size(comb{nn,1},1);
    for jj=1:size(comb{nn,1},1)

        % Take all possible combinations from variable 'comb' and sort them
        % in each row of 'index'. Index now has a vector in each K row that
        % indexes the variables to be used, i.e. for N==3:
        index_temp{jj + sum(dim(:,1:nn)),1} = comb{nn,1}(jj,:);
    end
end

x_t = cell(K,1);
for ll2 = 1:K
    
    % Now pre-construct the set of predictors used in a given regression
    % model

    x_t{ll2,1} = x(:,index_temp{ll2,1});
end

% initial hyper-priors

prob_0   = 1./K;         %<----------------***** Change if you wish *****
beta_0   = zeros(p,1);
a_0      = 0.1;
b_0      = 10;

if T <= K
    g = 1/K;
else
    g = 1/T;
end

lmlik    = zeros(1,K);

for k = 1:K
x_temp    = [ones(T,1) x_t{k,1}];
XtX       = x_temp'*x_temp;
iXtX      = XtX\eye(size(x_temp,2));%/eye(size(x_t{k,1},2)+1);
Xty       = x_temp'*y;
beta      = g/(1+g)*iXtX*Xty;
Xbeta     = x_temp*beta;
s         = (y-Xbeta)'*(y-Xbeta );

lmlik(k)  = -.5*size(x_temp,2)-.5*(T-1)*log(s + (Xbeta'*Xbeta)/(g+1));  
end

post_prob = (lmlik*prob_0)./sum(lmlik*prob_0);




