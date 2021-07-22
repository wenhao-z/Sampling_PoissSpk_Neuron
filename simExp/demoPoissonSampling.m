% Simulate a feedforward network (without recurrent connections) of Poisson spiking neurons 

% Wen-Hao Zhang, July 1, 2021
% University of Chicago

%% Parameters of the model
parsHawkesNet;

parsMdl.tauIsynDecay = 2;
parsMdl.dt = 0.1; % Unit: ms

parsMdl.bSample_ufwd = 0;
parsMdl.jxe = 1e-2;
parsMdl.ratiojie = 5;

parsMdl.rngNetSpk = rng('shuffle');
% Input parameters
parsMdl.tTrial = 200; % length of a trial. Unit: ms
parsMdl.tStat = 2*1e3; % The time point after which the responses included into statistics

flagTask = 1;
% 1: demo the spiking responses of the network and decoding
% 2: The sampling variability with input rate
switch flagTask
    case 1
        parsMdl.Ufwd = 20;
        parsMdl.tLen = 22*1e3; % unit: ms
        parsMdl.Posi = 10;
    case 2
        parsMdl.Ufwd = 0:5:50;
        parsMdl.tLen = 52*1e3; % unit: ms
        parsMdl.tBin = parsMdl.tTrial;
end

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

% Compute the dependent parameters
parGrid = arrayfun(@(x) getDependentPars_HawkesNet(x), parGrid);

%% Simulation
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'covSample', [], ...
    'PreMat_LH', []);
NetStat = repmat(NetStat, size(parGrid));

tSpkArray = cell(size(parGrid));
tic
for iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    
    netpar = parGrid(iterPar);
    
    % Generate a sample of feedforward spiking input
    ratefwd = makeRateFwd(netpar.Posi, netpar);
    ratefwd = [ratefwd; netpar.ji0 * sum(ratefwd)/netpar.Ni*ones(netpar.Ni,1)];
    
    % Get the precision of the likelihood
    NetStat(iterPar).PreMat_LH = sum(ratefwd(1:netpar.Ne,:),1)/ netpar.TunWidth^2 /2 ...
        * netpar.tTrial/netpar.dt; % Precision of the likelihood in time window tBin
    
    outSet = simHawkesNet_NoRecConns(ratefwd, netpar);
    
    tSpkArray{iterPar} = outSet.tSpk;
    
    % Compute the statistics of network's samples
    % Spike count in of each trial
    tEdge = [netpar.tStat+netpar.dt, netpar.tLen];
    neuronEdge = 0.5: netpar.Ne + 0.5;
    nSpk = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdge, tEdge);
    netstat = getNetMdlStat(nSpk, outSet.popVec, netpar, NetStat(iterPar));
    
    NetStat(iterPar).tSample = netstat.tSample;
    NetStat(iterPar).meanSample = netstat.meanSample;
    NetStat(iterPar).covSample = netstat.covSample;
end
toc

%% Plot the results

switch flagTask
    case 1
        % Rasterplot
        tPlotEdge = [5, 5.5]*1e3; % Unit: ms
        tSpk = tSpkArray{1};
        tSpk(:, tSpk(2,:)<tPlotEdge(1)) = [];
        tSpk(:, tSpk(2,:)>tPlotEdge(2)) = [];
        tSpk(2,:) = tSpk(2,:) - tPlotEdge(1);
        
        plotSpkRaster(tSpk, parsMdl, 'tBin', 0.01*1e3);
        set(gca, 'xlim', tPlotEdge - tPlotEdge(1))
        
        
        %% Stimulus feature samples and statistics
        figure
        tBin = 5; % Decoding time window. Unit: ms
        
        % Plot an example sampling trajectory
        % Average samples in a time window
        subsAvg = kron(1: (parsMdl.tLen-parsMdl.tStat)/tBin, ones(1, round(tBin/parsMdl.dt)));
        tSample = accumarray(subsAvg(:), NetStat.tSample(1,:), [], @(x) mean(x, 'omitnan'));
        
        subplot(2,4, 1:3)
        nPlotEdge = tPlotEdge/tBin;
        tPlot = (1 : diff(nPlotEdge)) * tBin;
        plot(tPlot, tSample(nPlotEdge(1)+1: nPlotEdge(2)))
        xlabel('Time (ms)')
        
        % ------------------------------------
        % Plot the sampling distribution
        subplot(2,4,4)
        hold on
        
        nBins = 31;
        
        % Average the samples in every trial, which is consistent with the
        % precision in the time window
        tBin = parsMdl.tTrial;
        subsAvg = kron(1: (parsMdl.tLen-parsMdl.tStat)/tBin, ones(1, round(tBin/parsMdl.dt)));
        tSample = accumarray(subsAvg(:), NetStat.tSample(1,:), [], @(x) mean(x, 'omitnan'));
        
        tSample(isnan(tSample)) = [];
        [histVal, histEdge] = histcounts(tSample, nBins);
        histVal = histVal / (sum(histVal)*mean(diff(histEdge)));
        
        stairs((histEdge(1:end-1)+histEdge(2:end))/2, histVal);
        histEdge = linspace(-30, 30, 101);
        plot(histEdge, normpdf(histEdge, parsMdl.Posi, 1/sqrt(NetStat.PreMat_LH)));
        
        xlim(30*[-1,1])
        xlabel('Stim. feature')
        
        subplot(2,4,8)
        plot(parsMdl.PrefStim, ratefwd(1:parsMdl.Ne)/parsMdl.dt*1e3)
        ylim([0, 30])
        xlim(180*[-1, 1])
        set(gca, 'xtick', -180:90:180)
    case 2
        figure
        hold on
        covSample = [NetStat.covSample];
        covSample = reshape(covSample, 2,2, []);
        
        plot(parsMdl.Ufwd, [NetStat.PreMat_LH], '-');
        plot(parsMdl.Ufwd, 1./squeeze(covSample(1,1,:)), 'o');
        
        xlabel('Feedforward rate (Hz)')
        ylabel('Sampling precision')
        axis square
        set(gca, 'xtick', 0:10:50)
end