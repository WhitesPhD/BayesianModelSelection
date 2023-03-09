function x = fastmvg_rue(Phi, PtP, alpha, D)
%   Reference:
%     Rue, H. (2001). Fast sampling of gaussian markov random fields. Journal of the Royal
%     Statistical Society: Series B (Statistical Methodology) 63, 325338.
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

p = size(Phi,2);
Dinv = diag(1./diag(D));
L = chol(PtP + Dinv, 'lower');
v = L \ (Phi'*alpha);
m = L' \ v;
w = L' \ randn(p,1);
x = m + w;
end