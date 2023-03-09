function out=igrng(theta,chi)
    
chisq1 = randn(1).^2;
out = theta + 0.5*theta/chi * ( theta*chisq1 - sqrt(4*theta*chi*chisq1 + theta^2*chisq1.^2) );

l = rand(1) >= theta./(theta+out);
out( l ) = theta^2./out( l );
    
return;