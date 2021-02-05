function [Info_sext_theory, Info_sint_theory] = fisherInfo_Theory(rateHeight, TunWidth, rateOffset, varSAvg, parsMdl)
% function [Info_sext_theory, Info_sint_theory] = fisherInfo_Theory(rateAvg, Ne, tuneWidth, varSAvg)
% Calculate the theoretical prediction of Fisher information

% Wen-Hao Zhang, Aug 6, 2019
% wenhao.zhang@pitt.edu


tuneFunc = rateHeight * gaussTuneKerl(0, TunWidth, parsMdl, 0);
dtuneFunc = tuneFunc .* parsMdl.PrefStim/ TunWidth^2;
tuneFunc = tuneFunc + rateOffset;

Info_sint_theory = sum(dtuneFunc.^2 ./ tuneFunc);

% Assume the tuning is a perfect Gaussian without DC offset.
% Info_sint_theory = tuneWidth.^2 ./(rateAvg * Ne);

Info_sext_theory = 1./Info_sint_theory + varSAvg;
Info_sext_theory = 1./Info_sext_theory;
