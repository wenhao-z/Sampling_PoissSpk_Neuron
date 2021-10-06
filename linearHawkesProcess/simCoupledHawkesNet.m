function outSet = simCoupledHawkesNet(ratefwd, parsMdl)
% Simulate a linear Hawkes process with ring structure.
% ratefwd is the firing rate in a time bin of the feedfoward input

% Wen-Hao Zhang,
% University of Chicago
% May 10, 2021

% Release heavily used parameters from parsMdl
Ne      = parsMdl.Ne;
Ni      = parsMdl.Ni;
Ncells  = Ne + Ni; % Number of neurons in a network
numNets = parsMdl.numNets;
dt      = parsMdl.dt; % Change the unit of dt into sec.

% FanoFactorIntVar = parsMdl.FanoFactorIntVar;

tauDecay    = parsMdl.tauIsynDecay; % decay time constant of synaptic input
alphaDecay  = exp(-dt/tauDecay);

% -----------------------------------------------
% Generate the connection profile

% The connection strength within the same network
Jmat = parsMdl.jxe * repmat([1, -parsMdl.ratiojie], 2, 1);
Jmat = Jmat ./ sqrt(Ne + Ni); % scale with the square root of network size

% The connection strength across networks.
if parsMdl.bShareInhPool
    Jrpmat = Jmat;
    Jrpmat(1,1) = parsMdl.ratiojrprc * Jrpmat(1,1);
    
    %     Jrpmat = parsMdl.ratiojrprc * Jmat;
else
    % Note that there is no inhibitory connections across networks hence the
    % 2nd column of Jrpmat is all ZERO.
    Jrpmat = parsMdl.ratiojrprc * Jmat;
    Jrpmat(:,2) = 0;
    % Jrpmat(2,1) = 0.1*parsMdl.ji0 * parsMdl.ratiojrprc * Jmat(1,1);
    Jrpmat(2,1) = 1* parsMdl.ratiojrprc * Jmat(1,1);
end

if parsMdl.bCutRecConns
    % Cut off recurrent connections within the same network!!!
    Jmat(1,1) = 0;
end

% E-E connection matrix
Wee = gaussTuneKerl(parsMdl.PrefStim(1), parsMdl.TunWidth, parsMdl, 1);
Wee = gallery('circul', Wee);
Wee = Wee * 2 * parsMdl.width;

if isfield(parsMdl, 'bRandWee') && parsMdl.bRandWee
    IdxRand = randperm(numel(Wee), numel(Wee));
    Wee = reshape(Wee(IdxRand), size(Wee));
    clear IdxRand
end

% No stochasticity in the connection
weights11 = [Jmat(1,1) * Wee, Jmat(1,2) * ones(Ne, Ni); ...
    Jmat(2,1)* ones(Ni, Ne), Jmat(2,2)*ones(Ni,Ni)];

weights12 = [Jrpmat(1,1) * Wee, Jrpmat(1,2) * ones(Ne, Ni); ...
    Jrpmat(2,1)* ones(Ni, Ne), Jrpmat(2,2)*ones(Ni,Ni)];

weights = [weights11, weights12; ...
    weights12, weights11];
clear weights11 weights12

% -----------------------------------------------
% Initialization
Nsteps = round(parsMdl.tLen/parsMdl.dt);

bSpk      = false(numNets*Ncells, 1);
tSpk      = zeros(2, numNets*Ncells* parsMdl.maxrate * parsMdl.tLen/1e3);
tSpkfwd   = zeros(2, numNets*Ncells* parsMdl.maxrate * parsMdl.tLen/1e3);
% rateArray = zeros(Ncells, Nsteps);
popVec    = zeros(4, Nsteps); % [2, T], rows are s1 and s2;

tref = parsMdl.tref/dt; % unit: time step
tSpk_LastTime = -tref* ones(Ncells, 1);

xfwd      = zeros(numNets*Ncells,1); % The filtered feedforward input. 
xrec      = zeros(numNets*Ncells,1); % The filtered recurrent input

nSpkCount = 0;
if parsMdl.bSample_ufwd
    nSpkfwdCount = 0; % The feedforward spike count
end
ubkg = parsMdl.UBkg*dt/1e3;

% Set the random seed
rng(parsMdl.rngNetSpk);

% Iteration
for iter = 1: Nsteps
    % Feedforward input
    if parsMdl.bSample_ufwd
        ufwd = (ratefwd > rand(numNets*Ncells,1)); % A binary spike has the unit of firing probability in the time bin.
        
        % Record the feedforward spikes
        nSpkNow = sum(ufwd);
        tSpkfwd(1, nSpkfwdCount+ (1:nSpkNow)) = find(ufwd); % Index of spiking neurons
        tSpkfwd(2, nSpkfwdCount+ (1:nSpkNow)) = iter * dt; % Spike timing
        nSpkfwdCount = nSpkfwdCount + nSpkNow;
               
        % Note: without filtering, ufwd doesn't multiply with dt;
        %       otherwise, ufwd multiplies with dt.
        % Filter the spiking feedforward input
        if tauDecay > 0
            xfwd = alphaDecay * xfwd + ufwd;
            ufwd = xfwd/ tauDecay; % unit: 1/ms = kHz
            ufwd = ufwd * dt;
        end
    else
        ufwd = ratefwd;
    end
    
    % ------------------------------------------------------------    
    % Recurrent input
    urec = sum(weights(:, bSpk), 2);

    % Filter the spiking recurrent input
    if tauDecay > 0
        xrec = alphaDecay * xrec + urec;
        urec = xrec/ tauDecay; % unit: 1/ms = kHz
        urec = urec * dt;
    end
    
    % ------------------------------------------------------------
    
    % Updating the instantaneous firing rate
    rate = ufwd + urec + ubkg; % firing probability in the time bin
    bSpk = (rate > rand(numNets*Ncells,1)); % Spike generation
    
    % Refractory period
    bSpk((iter - tSpk_LastTime)< tref) = 0;
    tSpk_LastTime(bSpk) = iter;
    
    % Decode the position on the ring
    % Note .' because popVec is imaginary number. And ' will output the conjugate
    %     popVec(:, iter) = popVectorRep([bSpk(1:Ne), bSpk(Ncells+1: Ncells+Ne)], parsMdl).';
    popVec(:, iter) = popVectorRep([bSpk(1:Ne), bSpk(Ncells+1: Ncells+Ne), ...
        rate(1:Ne), rate(Ncells+1: Ncells+Ne)], parsMdl).';
    
    % Record the spike timing
    nSpkNow = sum(bSpk);
    tSpk(1, nSpkCount+ (1:nSpkNow)) = find(bSpk); % Index of spiking neurons
    tSpk(2, nSpkCount+ (1:nSpkNow)) = iter * dt; % Spike timing
    nSpkCount = nSpkCount + nSpkNow;
end

% Remove unused spike timing
tSpk(:, nSpkCount+1:end) = [];
tSpkfwd(:, nSpkCount+1:end) = [];

%% Fold the output variables into a struct
outSet.tSpk = tSpk;
outSet.popVec = popVec;
if parsMdl.bSample_ufwd
    outSet.tSpkfwd = tSpkfwd;
end
