function gamma_store=randgamma(a,N)
%RANDGAMM Generates N gamma random deviates.

%	RANDGAMM(A,N) is a random deviate from the standard gamma

%	distribution with shape parameter A.

%

%	B*RANDGAMM(A,N) is a random deviate from the gamma distribution

%	with shape parameter A and scale parameter B.  The distribution

%	then has mean A*B and variance A*B^2.

%

%	See RAND.

% GKS 31 July 93

% Algorithm for A >= 1 is Best's rejection algorithm XG

% Adapted from L. Devroye, "Non-uniform random variate

% generation", Springer-Verlag, New York, 1986, p. 410.

% Algorithm for A < 1 is rejection algorithm GS from

% Ahrens, J.H. and Dieter, U. Computer methods for sampling

% from gamma, beta, Poisson and binomial distributions.

% Computing, 12 (1974), 223 - 246.  Adapted from Netlib

% Fortran routine.


gamma_store = zeros(N,1);

for count = 1:N

a = a(1);
if a < 0,
   gam = NaN;
elseif a == 0,
   gam = 0;
elseif a >= 1,
   b = a-1;
   c = 3*a-0.75;
   accept = 0;
   while accept == 0,
      u = rand(2,1);
      w = u(1)*(1-u(1));
      y = sqrt(c/w)*(u(1)-0.5);
      gam = b+y;
      if gam >= 0,
         z = 64*w^3*u(2)^2;
         accept = ( z<=1-2*y^2/gam );
         if accept == 0,
            if b == 0,
                accept = ( log(z)<=-2*y );
            else
                accept = ( log(z)<=2*(b*log(gam/b)-y) );
            end;
         end;
      end;
   end;
else
   aa = 0;
   b = 1 + .3678794*a;
   accept = 0;
   while accept == 0,
      p = b*rand(1);
      if p < 1, 
         gam = exp(log(p)/a);
         accept = (-log(rand(1)) >= gam);
      else
         gam = -log((b-p)/a);
         accept = (-log(rand(1)) >= (1-a)*log(gam));
      end;
   end;
end;

gamma_store(count) = gam;

end;