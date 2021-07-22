% Parameters of a coupled ring network model with Hawkes neurons

% Wen-Hao Zhang, May 10, 2021

% Load the parameters of a single ring network with Hawkes neurons
parsHawkesNet;

% Specific parameters for the coupled networks
parsMdl.numNets = 2;

% Connection strength between networks
parsMdl.ratiojrprc = 0.5; % The ratio between the synaptic weight across 
%                           networks over the weight within the same network

parsMdl.Posi = [0, 0]; % Must be a row vector