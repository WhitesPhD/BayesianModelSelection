function [X,meanX,stdX,y,meany] = standardise(X, y)
%STANDARDISE standardise input data.
%   [X, meanX, stdX, y, meany] = standardise(X, y) standardises the 
%   predictor matrix X and (optional) the target vector y. 
%
%   X is an [n x p] matrix and is standardised to have zero mean and 
%   variance 1/n for each column. y is an [n x 1] vector and os
%   standardised to have zero mean.
%
%   The input arguments are:
%       X     - matrix of size [n x p] to be standardised
%       y     - vector of size [n x 1] to be standardised
%
%   Return values:
%       X     - standardised data matrix 
%       muX   - mean of X
%       stdX  - std of each column of X
%       y     - standardised data vector y
%       muy   - mean of y
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% params
n=size(X,1);
meanX=mean(X);
stdX=std(X,1)*sqrt(n);

%% Standardise Xs
X=bsxfun(@minus,X,meanX);
X=bsxfun(@rdivide,X,stdX);

%% Standardise ys (if neccessary)
if(nargin == 2)
    meany=mean(y);
    y=y-meany;
end;


end