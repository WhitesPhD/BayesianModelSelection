function r2_oos = R2oos(yhat,ytest,ymean)

resid = ytest - yhat;
bench = ytest - ymean;

r2_oos = 1-sum(resid.^2)/sum(bench.^2);



