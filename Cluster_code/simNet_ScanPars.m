% Simulate the network model with varieties of parameters to compute the
% mutual information and linear Fisher information

% Note: This code needs to be ran on a high performance cluster!!

% Wen-Hao Zhang
% Feb 20, 2020
% University of Pittsburgh

setWorkPath;
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model
parsHawkesNet;

flagTest = 1; 
% 1: Change the feedforward rate and recurrent strength, while fix network size
% 2: Change network size and recurrent strength, while fix the feedforward rate
switch flagTest
    case 1
        parsMdl.Ufwd = 0:5:60;
        parsMdl.jxe = (0:0.2:6)/100; % E synaptic strength.
    case 2
        % Fix Ufwd and change the number of neurons
        parsMdl.Ne = 180 * [1,2,5,10,20];
        parsMdl.jxe = (0:0.1:1)/100; % E synaptic strength.
end

parsMdl.dt = 0.1; % unit: ms
parsMdl.FanoFactorIntVar = 1; % The Fano factor of recurrent interactions.
parsMdl.ratiojie = 5; % Ratio between I and E synaptic strength.
parsMdl.bSample_ufwd = 1;

% Parameters for the test
parsMdl.Posi = 0; % Input feature, i.e., position on the feature subspace
parsMdl.dPosi = 2; % The difference between a pair of inputs to compute Fisher information
parsMdl.tLen = 2e3*1e3;% The length of simulation. unit: ms
parsMdl.tTrial = 0.2*1e3; % The length of a trial; unit: ms
parsMdl.nTrials = parsMdl.tLen / parsMdl.tTrial; % Number of trials
parsMdl.tStat = 0;

parsMdl.UrecWorld = 30; % Parameter of the world

% ---------------------------------------------------
% Parameters of information-theoretic analysis
% The minimal firing rate to include a cell into analysis
minRateAns = 0; % unit: hz
% Whether using bBootStrap to estimate the std. of Fisher information
bBootStrap = true;
nBootStrap = 50;

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

% Compute dependent parameters
parGrid = arrayfun(@(x) getDependentPars_HawkesNet(x), parGrid);

% The mean and covariance of real posterior
meanPosterior = cell(size(parGrid)); % [(s,z), dim of parGrid];

%% Simulate the network
tStart = clock;

for IdxPar = 1: numel(parGrid)
    % IdxPar = str2double(getenv('SLURM_ARRAY_TASK_ID'));
    fprintf('Progress: %d/%d\n', IdxPar, numel(parGrid));
    
    subsTrialAvg = [repmat(1:4, 1, parsMdl.tLen/parsMdl.dt); ...
        kron(1:parsMdl.tLen/parsMdl.tTrial, ones(1, round(4*parsMdl.tTrial/parsMdl.dt)))];
    
    % Load the parameter
    netpars = parGrid(IdxPar);
    
    % Simulate a network model to generate spiking responses
    outSet = simNetMdl_FisherInfo(netpars);
    
    netpars.meanPosterior = outSet.meanPosterior{1}; % Only use the 1st Posi to calculate mutual information
    [~, netpars.covPosterior] = getPosteriorHierMdl(netpars);
    
    % Partition network's outputs into many segments, with each a trial
    nSpk = cellfun(@(tSpk) tSpk2nSpk(tSpk, netpars.tTrial, netpars), outSet.tSpkArray, 'uniformout', 0);
    popVec = cellfun(@(x) accumarray(subsTrialAvg.', x(:), []), outSet.popVecArray, 'uniformout', 0);
    
    % Information-theoretic Analysis
    [InfoAnsRes, NetStat] = InfoTheoAns_BatchFunc(nSpk, popVec, netpars, 'minRateAns', ...
        minRateAns, 'bBootStrap', bBootStrap, 'nBootStrap', nBootStrap);
    
    if netpars.bSample_ufwd
        nSpkfwd = cellfun(@(tSpk) tSpk2nSpk(tSpk, netpars.tTrial, netpars), outSet.tSpkfwdArray, 'uniformout', 0);
        [InfoAnsFwdRes, NetStatFwd] = InfoTheoAns_BatchFunc(nSpkfwd, popVec, netpars, 'minRateAns', ...
            minRateAns, 'bBootStrap', bBootStrap, 'nBootStrap', nBootStrap);
    end
end
tEnd = clock;
clear outSet netpars subsTrialAvg

%% Save

fileName = 'HawkesNet';
savePath = fullfile(Path_RootDir, 'Data', 'tmpJobArrayDat');
mkdir(savePath);

fileName = [fileName, '_', num2str(IdxPar), '.mat'];

save(fullfile(savePath, fileName), '-v7.3')

