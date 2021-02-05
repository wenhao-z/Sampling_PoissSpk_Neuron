function KLDiv = KLDiv_NormDist(mu0, Cov0, mu, Cov)
% Calculate the KL divergence between two normal distributions
% D_KL( N_0 || N )
% Wen-Hao Zhang, July 9, 2019
% University of Pittsburgh

KLDiv = log(det(Cov)) - log(det(Cov0)) ...
    + trace(Cov0 / Cov) - length(Cov0) ...
    + (mu - mu0)' / Cov * (mu - mu0);
KLDiv = KLDiv /2;


% XEnt = log(2*pi) ...
%     + log(det(Cov)) + trace(Cov0 / Cov) ...
%     + (mu - mu0)' / Cov * (mu - mu0);
% XEnt = XEnt /2;