function Z = mvrn(mu,Sigma,n)


R = chol(Sigma);
R=R';
[ro col]=size(mu);
if  col > ro 
  mu = mu';
end
dim = length(mu);
Z = zeros(dim,n);
for i=1:n 
   x = randn(dim,1);
   Z(:,i) = mu + R*x;
end
Z = Z';


