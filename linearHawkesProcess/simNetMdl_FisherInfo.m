function outSet = simNetMdl_FisherInfo(parsMdl)
% Simulate a network model for Fisher information analysis
% Given the network parameter, I will apply to stimuli at different
% positions to the network to calculate Fisher information.
% Wen-Hao Zhang, Aug. 2, 2019

tSpkArray = cell(2,1); % Spike timing of network activity
tSpkfwdArray = cell(2,1); % Spike timing of feedforward input
popVecArray = cell(2,1); % The position of input and responses on feature space
meanPosterior = cell(2,1); 

% A for-loop for two stimulus positions whose results will be used to
% estimate Fisher information.
for iterPosi = 1: 2
    % Generate a sample of feedforward spiking input
    switch iterPosi
        case 1
            StimPosi = parsMdl.Posi - parsMdl.dPosi/2;
        case 2
            StimPosi = parsMdl.Posi + parsMdl.dPosi/2;
    end
    ratefwd = makeRateFwd(StimPosi, parsMdl); % Unit: firing probability in a time bin
    
    meanPosterior{iterPosi} = popVectorDecoder(ratefwd, parsMdl) * ones(2,1); % the mean for s and z
    
    % Linear Hawkes process with full rank interaction
    ratefwd = [ratefwd; ...
        parsMdl.ji0 * sum(ratefwd)/parsMdl.Ni*ones(parsMdl.Ni,1)];
    outSet = simHawkesNet(ratefwd, parsMdl);
    
    tSpkArray{iterPosi} = outSet.tSpk;
    popVecArray{iterPosi} = outSet.popVec;
    if parsMdl.bSample_ufwd
        tSpkfwdArray{iterPosi} = outSet.tSpkfwd;
    end
end

%% Fold results into an output struct

outSet.tSpkArray = tSpkArray;
outSet.popVecArray = popVecArray;
outSet.meanPosterior = meanPosterior;

if parsMdl.bSample_ufwd
    outSet.tSpkfwdArray = tSpkfwdArray;
end
