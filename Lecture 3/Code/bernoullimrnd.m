function [x] = bernoullimrnd(n,p)
%-------------------------------------------------------------------
% Generate vector x with elements following a Bernoulli distribution
%-------------------------------------------------------------------
% Each element of x is a 0-1 variable following the distribution:
%                    
%         prob(x) = [(p)^x][(1-p)^(1-x)]
%
%-------------------------------------------------------------------
%  INPUTS:
%   n     - The number of variates you want to generate
%   p     - Associated probability
%
%  OUTPUT:
%   x     - [nx1] vector of Bernoulli variates
%-------------------------------------------------------------------

x = zeros(n,1);
x(rand(n,1)<p,1) = 1;
