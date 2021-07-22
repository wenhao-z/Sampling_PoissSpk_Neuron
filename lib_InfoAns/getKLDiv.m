function KLDiv = getKLDiv(mu0, Cov0, mu, Cov)
% Calculate the KL divergence from normal distributions 
%   N(mu0, Cov0) to N(mu, Cov)
% Wen-Hao Zhang, June 27, 2019
% University of Pittsburgh


KLDiv = log(det(Cov)) - log(det(Cov0)) ...
    + trace(Cov0 / Cov) ...
    + (mu - mu0)' / Cov * (mu - mu0);
KLDiv = KLDiv /2 - length(Cov0)/2;
