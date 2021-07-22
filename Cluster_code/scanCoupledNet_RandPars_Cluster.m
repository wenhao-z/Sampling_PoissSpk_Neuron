% Scan the network parameter to test the robustness of sampling in a coupled network.

% Wen-Hao Zhang, June 15, 2021
% University of Chicago

setWorkPath;
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model
parsCoupledHawkesNet;

parsMdl.dt = 0.1;
parsMdl.bSample_ufwd = 0;
parsMdl.jxe = 1e-2;
parsMdl.bCutRecConns = 1; % Cut off all recurrent connections to simplify theoretical analysis
parsMdl.bShareInhPool = 0; % Whether two networks share the same inhibition pool

% Input parameters
parsMdl.tLen = 102*1e3; % unit: ms
parsMdl.tStat = 2*1e3;
parsMdl.tBin = 20; % Decoding time window. unit: ms

% Adjustable parameters
nMCSim = 1e3;

% Compute the dependent parameters
parsMdl = getDependentPars_HawkesNet(parsMdl);

%% Simulation
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'meanSamplePred', [], ...
    'covSample', [], ...
    'covCondMean', [], ...
    'PreMat_LH', [], ...
    'ratePop', [], ...
    'rateHeight', []);
NetStat = repmat(NetStat, nMCSim, 1);

parGrid = struct('Posi', [], ...
    'Ufwd', [], ...
    'ratiojrprc', []);
parGrid = repmat(parGrid, nMCSim, 1);

tStart = clock;
parpool(12);

parfor iterPar = 1: nMCSim
    fprintf('Progress: %d/%d\n', iterPar, nMCSim);
    
    netpar = parsMdl;
    
    % Randomly set the feedforward input parameters and coupling weight
    rng(sum(clock)*100); % I need to randomly set the random seed. The simNet code will set the net random seed.

    netpar.Posi = 20 * rand(2,1) - 10;
    netpar.Ufwd = 50 * rand(2,1);
    netpar.ratiojrprc = 5 * rand(1);
    
    % Record the randomized input and network parameters
    parGrid(iterPar).Posi = netpar.Posi;
    parGrid(iterPar).Ufwd = netpar.Ufwd;
    parGrid(iterPar).ratiojrprc = netpar.ratiojrprc;
    
    % Generate a sample of feedforward spiking input
    ratefwd = makeRateFwd(netpar.Posi, netpar);
    ratefwd = [ratefwd; netpar.ji0 * sum(ratefwd,1)./netpar.Ni.*ones(netpar.Ni,1)];
    
    % Get the precision of the likelihood
    preLH = sum(ratefwd(1:netpar.Ne,:),1)/ netpar.TunWidth^2 /2 ...
        * netpar.tBin/netpar.dt; % Precision of the likelihood in time window tBin
    NetStat(iterPar).PreMat_LH = diag(preLH);
    
    % Simulate the network
    outSet = simCoupledHawkesNet(ratefwd(:), netpar);
    
    % Compute the statistics of network's samples
    % Spike count in of each trial
    tEdge = [netpar.tStat+netpar.dt, netpar.tLen];
    neuronEdge = [0.5: netpar.Ne + 0.5, netpar.Ncells+ (0.5: netpar.Ne + 0.5)];
    nSpk = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdge, tEdge);
    nSpk(netpar.Ne+1) = [];
    
    NetStat(iterPar) = getNetMdlStat(nSpk, outSet.popVec, netpar, NetStat(iterPar));
end

tEnd = clock;

%% Theoretical prediction of sampling distribution
preSample = arrayfun(@(x) inv(x.covSample), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));

% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
prePrior = arrayfun(@(x,y) findPriorPrecision(x.meanSample, x.covSample, y.Posi, x.PreMat_LH), ...
    NetStat, parGrid);

%% Save

savePath = fullfile(Path_RootDir, 'Data', 'HawkesNet');

str = datestr(now, 'yymmddHHMM');
fileName = ['CoupledNet_', str(1:6), '_', str(7:end) '.mat'];

save(fullfile(savePath, fileName), '-v7.3')

