% Parameters of a linear Hawkess process with ring structure

% Wen-Hao Zhang, July 2, 2019
% University of Pittsburgh

parsMdl.width = 180; % (-width, width], unit: deg
parsMdl.TunWidth = 40; % unit: deg;
parsMdl.Ne = 180; % Number of E neurons
parsMdl.Ni = parsMdl.Ne/4; % Number of I neurons
parsMdl.Ncells = parsMdl.Ne + parsMdl.Ni;
parsMdl.rho = parsMdl.Ne/(2*parsMdl.width);

% Preferred stimuli of neurons.
% The location of neurons in the feature space
PrefStim = linspace(-parsMdl.width, parsMdl.width, parsMdl.Ne+1);
PrefStim(1) = [];
parsMdl.PrefStim = PrefStim';
clear PrefStim

% -------------------------------------------------------------------------
% Connection strength
parsMdl.jxe = 1.5; % E synaptic weight
% parsNet.jxi = 16; % I synaptic weight (absoluate value)
parsMdl.ratiojie = 5; % The ratio between I synapse over E synapse, jxi/jxe. (absoluate value)
parsMdl.FanoFactorIntVar = 1;
parsMdl.ji0 = 0.8; % ji0 = (total feedfwd inputs to *E* neurons)/ (total feedfwd inputs to *I* neurons)

% -------------------------------------------------------------------------
% Temporal parameters
parsMdl.dt = 0.5; % unit: ms
parsMdl.nTrials = 1;
parsMdl.tLen = 50*1e3; % unit: ms
parsMdl.tauIsynDecay = 2; % unit: ms, decay time constant of synaptic input

% -------------------------------------------------------------------------
% Input parameters
parsMdl.Posi = 0; % unit: deg
parsMdl.Ufwd = 30; % Peak firing rate of feedforward inputs, unit: Hz
parsMdl.UBkg = 0;

parsMdl.bSample_ufwd = 1; % 1: Using spiking feedforward inputs or
%                           0: its smooth firing rate

% -------------------------------------------------------------------------
% Set the random seed
% parsMdl.rngUff = rng('shuffle');
% parsMdl.rngNetSpk = rng('shuffle');
parsMdl.rngUff = rng(0);
parsMdl.rngNetSpk = rng(100);

parsMdl.maxrate = 100;
