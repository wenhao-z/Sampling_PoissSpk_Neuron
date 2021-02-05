function [Info_sext_theory, Info_sint_theory] = fisherInfo_TheorySimple(rateAvg, Ne, tuneWidth, varSAvg)
% Calculate the theoretical prediction of Fisher information by assuming
% the tuning curve is a perfect Gaussian without DC offset

% Wen-Hao Zhang, Aug 6, 2019
% wenhao.zhang@pitt.edu

% Assume the tuning is a perfect Gaussian without DC offset.
Info_sint_theory = tuneWidth.^2 ./(rateAvg * Ne);
Info_sext_theory = Info_sint_theory + varSAvg;

Info_sint_theory = 1./Info_sint_theory;
Info_sext_theory = 1./Info_sext_theory;
