function parsMdl = getDependentPars_HawkesNet(parsMdl)
% Calculate dependent parameters in Hawkes network.

% Wen-Hao Zhang, Feb-4, 2020
% wenhao.zhang@pitt.edu
% @University of Pittsburgh

% Compute the dependencies of _number of neurons_
% Used in the scanNetPars.m

parsMdl.Ni = parsMdl.Ne/4; % Number of I neurons
parsMdl.Ncells = parsMdl.Ne + parsMdl.Ni;

% The location of neurons in the feature space
PrefStim = linspace(-parsMdl.width, parsMdl.width, parsMdl.Ne+1);
PrefStim(1) = [];
parsMdl.PrefStim = PrefStim';
clear PrefStim
