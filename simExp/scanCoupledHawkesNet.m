% Demo of a linear Hawkes process where the E neurons are organized on a
% ring

% Wen-Hao Zhang, July 2, 2019
% University of Pittsburgh

%% Parameters of the model
parsCoupledHawkesNet;

parsMdl.dt = 0.1;
parsMdl.bSample_ufwd = 1;
parsMdl.jxe = 1e-2;
parsMdl.bCutRecConns = 1; % Cut off all recurrent connections to simplify theoretical analysis
parsMdl.bShareInhPool = 0; % Whether two networks share the same inhibition pool
parsMdl.ratiojie = 3;

% Input parameters
parsMdl.Posi = 10*[-1; 1];
parsMdl.tLen = 102*1e3; % unit: ms
parsMdl.tStat = 2*1e3;
parsMdl.tBin = 20; % Decoding time window. unit: ms
parsMdl.tTrial = parsMdl.tBin;

flagTest = 2;
% 1: Change the input strength to network 1, and fix other parameters.
% 2: Change the coupling strength between two networks
switch flagTest
    case 1
        parsMdl.Ufwd = [0:3, 5:5:50]; % Peak firing rate of feedforward inputs, unit: Hz
        parsMdl.Ufwd = [parsMdl.Ufwd; 25* ones(size(parsMdl.Ufwd ))];
        
        parsMdl.ratiojrprc = 1;
    case 2
        parsMdl.Ufwd = 30*ones(2,1);
%         parsMdl.ratiojrprc = 0:0.5:8;
        parsMdl.ratiojrprc = 0:0.5:5;
end

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

% Compute the dependent parameters
parGrid = arrayfun(@(x) getDependentPars_HawkesNet(x), parGrid);

%% Simulation
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'meanSamplePred', [], ...
    'covSample', [], ...
    'covCondMean', [], ...
    'PreMat_LH', [], ...
    'ratePop', [], ...
    'rateHeight', []);
NetStat = repmat(NetStat, size(parGrid));

tic
parfor iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    
    netpar = parGrid(iterPar);
    
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
toc


%% Theoretical prediction of sampling distribution
preSample = arrayfun(@(x) inv(x.covSample), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));

% Compute the recurrent input strength to predict the prior
% Note: only work for two coupled networks
prePriorPred = arrayfun(@(x) mean(x.rateHeight) * [1,-1;-1,1], NetStat, 'uniformout', 0);
prePriorPred = cell2mat(shiftdim(prePriorPred,-2));
prePriorPred = prePriorPred .* shiftdim(parsMdl.ratiojrprc,-1);
prePriorPred = prePriorPred * sqrt(2*pi)*parsMdl.Ne * parsMdl.jxe/sqrt(parsMdl.Ncells)/parsMdl.TunWidth;
prePriorPred = prePriorPred * parsMdl.tBin * parsMdl.dt/1e3; % The precision in the time window tBin

% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
prePrior = arrayfun(@(x) findPriorPrecision(x.meanSample, x.covSample, parsMdl.Posi, x.PreMat_LH), ...
    NetStat);

%% Compute the Mutual information
if flagTest == 2
    % Choose the index of the parameter and define it as the posterior
    IdxPost = 5;
    
    PreMat_Prior = prePrior(IdxPost)* [1,-1; -1, 1];
    PreMat_Post = PreMat_Prior + NetStat(IdxPost).PreMat_LH;
    MeanPost = PreMat_Post \ NetStat(IdxPost).PreMat_LH  * parsMdl.Posi;
    
    [XEnt, MutualInfo, MutualInfo_UpBound] = arrayfun(@(S) getMutualInfo(MeanPost, inv(PreMat_Post), ...
        S.meanSample, S.covSample, 2*parsMdl.width, 1./prePrior(IdxPost)), NetStat);
    
%     [XEnt, MutualInfo, MutualInfo_UpBound] = arrayfun(@(S) getMutualInfo(MeanPost, inv(PreMat_Post), ...
%         S.meanSample, S.covSample, 2*parsMdl.width, inv(PreMat_Post)), NetStat);
end

%% Sampling statistics
figure
switch flagTest
    case 1
        hAxe(1) = subplot(2,2,1);
        hold on
        plot(parsMdl.Ufwd(1,:), [NetStat.meanSample], '-o');
        plot(parsMdl.Ufwd(1,:), [NetStat.meanSamplePred]);
        set(gca, 'xlim', [0, max(parsMdl.Ufwd(:))])
        xlabel('Feedforward rate 1 (Hz)')
        ylabel('Mean sample')
        axis square
        
        hAxe(2) = subplot(2,2,2);
        hold on
        plot(parsMdl.Ufwd(1,:), squeeze(preSample(1,1,:)), '-o');
        plot(parsMdl.Ufwd(1,:), squeeze(preSample(2,2,:)), '-o');
        preSamplePred = reshape([NetStat.PreMat_LH], parsMdl.numNets, parsMdl.numNets, []);
        % preSamplePred = preSamplePred + prePriorPred;
        preSamplePred = preSamplePred + ...
            shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], 1,1, length(prePrior));
        plot(parsMdl.Ufwd(1,:), squeeze(preSamplePred(1,1,:)));
        plot(parsMdl.Ufwd(1,:), squeeze(preSamplePred(2,2,:)));
        clear covSampleArray
        ylabel('Precision sample')
        set(gca, 'xlim', [0, max(parsMdl.Ufwd(:))])
        axis square
        
        hAxe(3) = subplot(2,2,3);
        plot(parsMdl.Ufwd(1,:), reshape([NetStat.rateHeight], parsMdl.numNets,[]));
        ylabel('Bump height')
        axis square
        
        hAxe(4) = subplot(2,2,4);
        hold on
        covCondMean = reshape([NetStat.covCondMean], parsMdl.numNets, parsMdl.numNets, []);
        plot(parsMdl.Ufwd(1,:), squeeze(covCondMean(1,1,:)), '-o');
        plot(parsMdl.Ufwd(1,:), squeeze(covCondMean(2,2,:)), '-o');
        axis square
        ylabel('Differential correlation')
        
        set(hAxe(4),'yscale', 'log')
        set(hAxe, 'xlim', [0, max(parsMdl.Ufwd(:))], 'xtick', [0, 1/2, 1] *max(parsMdl.Ufwd(:)))
    case 2
        hAxe(1) = subplot(2,3,1);
        hold on
        plot(parsMdl.ratiojrprc*parsMdl.jxe, [NetStat.meanSample], '-o');
        plot(parsMdl.ratiojrprc*parsMdl.jxe, [NetStat.meanSamplePred]);
        xlabel('Coupling strength')
        ylabel('Mean sample')
        axis square
        
        hAxe(2) = subplot(2,3,2);
        hold on
        plot(parsMdl.ratiojrprc*parsMdl.jxe, squeeze(preSample(1,1,:)), '-o');
        plot(parsMdl.ratiojrprc*parsMdl.jxe, squeeze(preSample(2,2,:)), '-o');
        
        preSamplePred = reshape([NetStat.PreMat_LH], parsMdl.numNets, parsMdl.numNets, []);
        % preSamplePred = preSamplePred + prePriorPred;
        preSamplePred = preSamplePred + ...
            shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], 1,1, length(prePrior));
        
        plot(parsMdl.ratiojrprc*parsMdl.jxe, squeeze(preSamplePred(1,1,:)));
        plot(parsMdl.ratiojrprc*parsMdl.jxe, squeeze(preSamplePred(2,2,:)));
        clear covSampleArray
        ylabel('Precision sample')
        axis square
        
        hAxe(3) = subplot(2,3,3);
        hold on
        plot(parsMdl.ratiojrprc*parsMdl.jxe, prePrior, '-o');
        % plot(parsMdl.ratiojrprc*parsMdl.jxe, - squeeze(preSample(1,2,:)));
        xlabel('Coupling strength')
        ylabel('Prior precision')
        axis square
        
        hAxe(4) = subplot(2,3,4);
        plot(parsMdl.ratiojrprc*parsMdl.jxe, reshape([NetStat.rateHeight], parsMdl.numNets,[]));
        ylabel('Bump height')
        axis square
        
        hAxe(5) = subplot(2,3,5);
        hold on
        covCondMean = reshape([NetStat.covCondMean], parsMdl.numNets, parsMdl.numNets, []);
        plot(parsMdl.ratiojrprc*parsMdl.jxe, squeeze(covCondMean(1,1,:)), '-o');
        plot(parsMdl.ratiojrprc*parsMdl.jxe, squeeze(covCondMean(2,2,:)), '-o');
        axis square
        ylabel('Differential correlation')
        
        hAxe(6) = subplot(2,3,6);
        hold on
        plot(parsMdl.ratiojrprc*parsMdl.jxe, MutualInfo/parsMdl.tBin*1e3, '-o');
        plot(parsMdl.ratiojrprc*parsMdl.jxe, MutualInfo_UpBound/parsMdl.tBin*1e3, '-k');
        axis square
        ylabel('Mutual info')
        % ylim([5, 7])
        
        set(hAxe, 'xlim', [min(parsMdl.ratiojrprc), max(parsMdl.ratiojrprc)]*parsMdl.jxe, ...
            'xtick', [0, 1/2, 1]*parsMdl.ratiojrprc(end)*parsMdl.jxe)
end

%% Sampling distribution with parameters

if flagTest == 2
    
    figure;
    
    xyLim = 30;
    sGrid = linspace(-xyLim, xyLim, 201);
    
    hAxe(1) = axes('position', [0.1, 0.1, 0.6, 0.6]); hold on; daspect([1,1,1])
    hAxe(2) = axes('position', [0.1, 0.75, 0.6, 0.15]); hold on; %daspect([1,4,1])
    hAxe(3) = axes('Position', [0.75, 0.1, 0.15, 0.6]); hold on; %daspect([4,1,1])
    
    % for iter = 1: length(NetStat)
    for iter = [1, IdxPost, length(NetStat)]
        meanSample = NetStat(iter).meanSample;
        covSample = NetStat(iter).covSample;
        fh = @(x,y) ( ([x;y] - meanSample)' / covSample/9 * ([x;y]-meanSample) - 1);
        hEllipse = fimplicit(hAxe(1), fh, [meanSample(1) + 3*covSample(1)*[-1, 1], meanSample(2) + 3*covSample(4)*[-1, 1]], ...
            'linew', 2);
        
        % Plot marginal distribution
        plot(hAxe(2), sGrid, ...
            normpdf(sGrid, meanSample(1), sqrt(covSample(1,1))), ...
            'linew', 2);
        plot(hAxe(3), normpdf(sGrid, meanSample(2), sqrt(covSample(2,2))), ...
            sGrid, 'linew', 2);
    end
    
    plot(hAxe(1), parsMdl.Posi(1)*ones(1,2), xyLim*[-1,1], '--k')
    plot(hAxe(1), xyLim*[-1,1], parsMdl.Posi(2)*ones(1,2), '--k')
    plot(hAxe(1), xyLim*[-1,1], xyLim*[-1,1], '--k')
    
    linkaxes(hAxe(1:2), 'x')
    linkaxes(hAxe([1,3]), 'y')
    
    axes(hAxe(1))
    xlim([-25, 20])
    ylim([-20, 25])
    set(hAxe(1), 'xtick',[-20, 0, 20], 'ytick',[-20, 0, 20])
    
    xlabel('Stimulus feature s_1')
    ylabel('Stimulus feature s_2')
    axis square
    
end