% Estimate the Fisher information of stimulus in the neuronal responses of a single 
% recurrent network
% Some simplifications is used in this Hawkess process:
% 1). No temporal kernel

% Wen-Hao Zhang
% July 22, 2021
% University of Chicago

if ~exist('Path_RootDir', 'var')
    setWorkPath;
end
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model

parsHawkesNet;

parsMdl.jxe = (0:0.1:0.5)/100;
parsMdl.Ne = 180*5;
% parsMdl.jxe = (0:0.05:0.5)/100;
% parsMdl.Ne = 180 * [1, 2, 5, 10];

parsMdl.dt = 0.1;
parsMdl.ratiojie = 5;
parsMdl.bSample_ufwd = 1;

% Parameters for the test
parsMdl.Posi = 0;
parsMdl.dPosi = 2; % The difference between a pair of stimuli
parsMdl.tLen = 1e3*1e3; % unit: ms
parsMdl.tTrial = 0.2*1e3; % unit: ms
parsMdl.nTrials = parsMdl.tLen / parsMdl.tTrial;
parsMdl.tStat = 0;
% parsMdl.rngNetSpk = rng('shuffle');

% Generate a pair of stimuli
parsMdl.Posi = parsMdl.Posi + [-1, 1]* parsMdl.dPosi/2;

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

% Calculate dependent parameters
parGrid = arrayfun(@(x) getDependentPars_HawkesNet(x), parGrid);

% Permute the dim of parGrid to make the 1st dim as Posi
IdxPosi = find(cellfun(@(x) strcmp(x, 'Posi'), {dimPar.namePar}));
parGrid = permute(parGrid, [IdxPosi, setdiff(1: ndims(parGrid), IdxPosi)]);

%% Simulate the network

% Initialize data structure
nSpk = cell(size(parGrid));
if parsMdl.bSample_ufwd
    nSpkfwd = cell(size(parGrid));
end
popVec = cell(size(parGrid));

subsTrialAvg = [ repmat(1:4, 1, parsMdl.tLen/parsMdl.dt); ...
    kron(1:parsMdl.tLen/parsMdl.tTrial, ones(1, round(4*parsMdl.tTrial/parsMdl.dt)))];

% parpool(12);

tStart = clock;
parfor iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    
    % Load the parameter
    netpars = parGrid(iterPar);
    
    % Feedforward input
    ratefwd = makeRateFwd(netpars.Posi, netpars); % Unit: firing probability in a time bin
    ratefwd = [ratefwd; ...
        netpars.ji0 * sum(ratefwd)/netpars.Ni*ones(netpars.Ni,1)];
    
    % Simulate a network model to generate spiking responses
    outSet = simHawkesNet(ratefwd, netpars);
    
    % ---------------------------------------------
    % Partition network's outputs into many trials
    nSpk{iterPar} = tSpk2nSpk(outSet.tSpk, netpars.tTrial, netpars);
    if netpars.bSample_ufwd
        nSpkfwd{iterPar} = tSpk2nSpk(outSet.tSpkfwd, netpars.tTrial, netpars);
    end
    popVec{iterPar} = accumarray(subsTrialAvg.', outSet.popVec(:), []);
        
end
clear subsTrialAvg
tEnd = clock;

%% Estimate the statistics of neuronal responses and Fisher information

NetStat = arrayfun(@(nspk, popvec, netpars) getNetMdlStat(nspk{1}, popvec{1}, netpars), nSpk, popVec, parGrid);

% The minimal firing rate to include a cell into analysis
minRateAns = 0; % unit: hz
% Number of bootstrap to estimate the std. of Fisher information. 
% 0: doesn't use bootstrap
nBootStrap = 0;
% nBootStrap = 50;


% Estimate Fisher information
[InfoAnsRes.FI_sext, InfoAnsRes.stdFI_sext] = ...
    arrayfun(@(n1,n2, netpars) fisherInfo_biasCorrect(n1{1}, n2{1}, parsMdl.Posi, netpars, ...
    'minRateAns', minRateAns, 'nBootStrap', nBootStrap), ...
    nSpk(1,:), nSpk(2,:), parGrid(1,:));

if parsMdl.bSample_ufwd
    [InfoAnsRes.FI_sext_ufwd, InfoAnsRes.stdFI_sext_ufwd] = ...
        arrayfun(@(n1,n2, netpars) fisherInfo_biasCorrect(n1{1}, n2{1}, parsMdl.Posi, netpars, ...
        'minRateAns', minRateAns, 'nBootStrap', nBootStrap), ...
        nSpkfwd(1,:), nSpkfwd(2,:), parGrid(1,:));
end

% ------------------------------------------------------------
% Theoretical prediction of linear Fisher information
[InfoAnsRes.FI_sext_theory, InfoAnsRes.FI_sint_theory] = arrayfun(@(S) ...
    fisherInfo_Theory(S.rateHeight, S.tuneWidth, S.rateOffset, S.covSample(3,3), parsMdl), ...
    NetStat(1,:));
% A simple theoretical prediction of linear Fisher information
[InfoAnsRes.FI_sext_theorySimp, InfoAnsRes.FI_sint_theorySimp] = arrayfun(@(S, netpars) ...
    fisherInfo_TheorySimple(S.rateAvg, netpars.Ne, S.tuneWidth, S.covSample(3,3)), ...
    NetStat(1,:), parGrid(1,:));

% Reshape the results
szGrid = size(parGrid);
for varName = fieldnames(InfoAnsRes)'
   InfoAnsRes.(varName{1}) = reshape(InfoAnsRes.(varName{1}), szGrid(2:end));
end

%% Save the results
savePath = fullfile(Path_RootDir, 'Data', 'HawkesNet');

str = datestr(now, 'yymmddHHMM');
fileName = ['FisherInfoRecNet_', str(1:6), '_', str(7:end) '.mat'];

save(fullfile(savePath, fileName), '-v7.3')
