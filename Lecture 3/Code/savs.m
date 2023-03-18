% Signal Adaptive Variable Selector algortihm (Ray, Battacharya, 2018)
% input:  - betas: beta coefficients
%         - X:     TxJ matrix of predictors
% 
% output: - gambar: sparsified beta coefficients

function [gambar, which] = savs(beta, X)

% vecnorm(X) = sqrt(sum(X.^2, 1))

w = size(X, 1)./sum(~isnan(X)) ; % weights to projec norm of incomplete X.

% w = 1 ;

mu = 1./(abs(beta).^2) ;
th = abs(beta).*w.*nansum(X.^2, 1) ; 

gambar = (sign(beta)./w./nansum(X.^2, 1)).*max(th - mu, 0) ;

which = (gambar ~= 0) ;
