% Compare the sampling distribution of network with the posterior under
% different input and/or network parameters.

% Wen-Hao Zhang
% July 4, 2021
% University of Chicago

if ~exist('Path_RootDir', 'var')
    setWorkPath;
end
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model
parsHawkesNet;

parsMdl.dt = 0.1;

% parsMdl.jxe = [1, 2, 5, 10, 15, 20]*1e-2;
parsMdl.jxe = (0.02:0.02:1)*1e-2;

parsMdl.ratiojie = 5;
parsMdl.bSample_ufwd = [0, 1];
parsMdl.FanoFactorIntVar = 0.7;

% Input parameters
parsMdl.Ufwd = 5:5:60; % Peak firing rate of feedforward inputs, unit: Hz
% parsMdl.Ufwd = 30;

parsMdl.tLen = 102*1e3; % unit: ms
parsMdl.tStat = 2*1e3;
parsMdl.tBin = 20; % Decoding time window. unit: ms

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

% Compute the dependent parameters
parGrid = arrayfun(@(x) getDependentPars_HawkesNet(x), parGrid);

%% Simulate the network
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'covSample', [], ...
    'PreMat_LH', [], ...
    'ratePop', []);
NetStat = repmat(NetStat, size(parGrid));

% Edges to make histogram
tEdge = [parsMdl.tStat+parsMdl.dt, parsMdl.tLen];
neuronEdge = 0.5: parsMdl.Ne + 0.5;

tStart = clock;
parpool(12);
parfor iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    
    % Load the parameter
    netpar = parGrid(iterPar);
    
    % Generate a sample of feedforward spiking input
    ratefwd = makeRateFwd(netpar.Posi, netpar); % Unit: firing probability in a time bin
    ratefwd = [ratefwd; ...
        netpar.ji0 * sum(ratefwd)/netpar.Ni*ones(netpar.Ni,1)];
    
    % Get the precision of the likelihood
    PreMat_LH = sum(ratefwd(1:netpar.Ne,:),1)/ netpar.TunWidth^2 /2 ...
        * netpar.tBin/netpar.dt; % Precision of the likelihood in time window tBin
    NetStat(iterPar).PreMat_LH = [PreMat_LH, 0; 0, 0];
    
    % Simulate the network
    outSet = simHawkesNet(ratefwd, netpar);
            
    % Compute the statistics of network's samples
    % Spike count in of each trial
    nSpk = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdge, tEdge);
    NetStat(iterPar).ratePop = nSpk/(netpar.tLen - netpar.tStat) *1e3;
    
    % Compute the statistics of network's samples    
    [tSample, ~, meanSample, covSample] = popVectorDecoder(outSet.popVec, netpar);
    
    NetStat(iterPar).tSample = tSample;
    NetStat(iterPar).meanSample = meanSample;
    NetStat(iterPar).covSample = covSample;
end
tEnd = clock;
clear outSet netpar

%% Compute the statistics of samples and posterior distribution
preSample = arrayfun(@(S) inv(S.covSample(1:2,1:2)), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));
preSample = squeeze(preSample);

preSamplePred = reshape([NetStat.PreMat_LH], [2,2, size(parGrid)]);

% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
% prePrior = arrayfun(@(x) ...
%     findPriorPrecision(parsMdl.Posi * ones(2,1), x.covSample, parsMdl.Posi * ones(2,1), x.PreMat_LH), ...
%     NetStat);
prePrior = -squeeze(preSample(1,2,:));
prePrior = reshape(prePrior, size(NetStat));
preSamplePred = preSamplePred + shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], 1,1, length(prePrior));

%% Save

savePath = fullfile(Path_RootDir, 'Data', 'HawkesNet');

str = datestr(now, 'yymmddHHMM');
fileName = ['SingleRecNet_', str(1:6), '_', str(7:end) '.mat'];

save(fullfile(savePath, fileName), '-v7.3')

