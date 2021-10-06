% Compare the sampling distribution of a single recurrent network with the
% posterior under different input and/or network parameters.

% Wen-Hao Zhang
% June 24, 2021
% University of Chicago

if ~exist('Path_RootDir', 'var')
    setWorkPath;
end
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model
parsHawkesNet;

parsMdl.dt = 0.1;
parsMdl.ratiojie = 5;
parsMdl.bSample_ufwd = 0;
% parsMdl.rngNetSpk = rng('shuffle');
parsMdl.FanoFactorIntVar = 0.7;

% Input parameters
parsMdl.Posi = 0;
parsMdl.tLen = 52*1e3; % unit: ms
parsMdl.tStat = 2*1e3;
parsMdl.tBin = 20; % Decoding time window. unit: ms
tTrial = 200; % unit: ms. Used to analyze the statistics of neuronal responses.

flagTest = 2;
% 1: Change the input strength to network, and fix other parameters.
% 2: Change the recurrent connection strength in the network
switch flagTest
    case 1
        parsMdl.Ufwd = 5:5:50; % Peak firing rate of feedforward inputs, unit: Hz
        parsMdl.jxe = 0.2/100; % 1e-2/10;
    case 2
        parsMdl.Ufwd = 30;
        %         parsMdl.jxe = (0:0.05:0.5)/100;
        parsMdl.jxe = (0:1:10)/100;
%         parsMdl.jxe = (0:0.1:1)/100;
        % parsMdl.jxe = [0.01, 0.1:0.1:1]/100;
end

% Parameter of the world
parsMdl.UrecWorld = 10;

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

% Compute the dependent parameters
parGrid = arrayfun(@(x) getDependentPars_HawkesNet(x), parGrid);

%% Simulate the network
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'covSample', [], ...
    'PreMat_LH', []);
NetStat = repmat(NetStat, size(parGrid));
nSpk = cell(size(parGrid));
% tSpkArray = cell(size(parGrid));

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
    [tSample, ~, meanSample, covSample] = popVectorDecoder(outSet.popVec, netpar);
    
    NetStat(iterPar).tSample = tSample;
    NetStat(iterPar).meanSample = meanSample;
    NetStat(iterPar).covSample = covSample;
    
    % Statistis of neuronal responases
    % tSpkArray{iterPar} = outSet.tSpk;
    nSpk{iterPar} = tSpk2nSpk(outSet.tSpk, tTrial, netpar);
end

%% Compute the statistics of samples and posterior distribution

if parsMdl.bSample_ufwd
    % Calculate the covariance of posterior p(s,z|x).
    % The reason of doing this is that once I use spiking ufwd, the recorded
    % distribution will become p(s,z|s_ext) instead of p(s,z|x). Thus I find
    % the covariance of posterior in a parametric way.
    for iter = 1: numel(NetStat)
        covSample = NetStat(iter).covSample;
        NetStat(iter).covSample(1:2,1:2) = covSample(1:2,1:2) - covSample(1,4)^2/ covSample(4,4);
    end
end
clear covSample

preSample = arrayfun(@(S) inv(S.covSample(1:2,1:2)), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));
preSample = squeeze(preSample);

preSamplePred = reshape([NetStat.PreMat_LH], [2,2, size(parGrid)]);

% Subjective prior
% prePrior = arrayfun(@(x) ...
%     findPriorPrecision(parsMdl.Posi * ones(2,1), x.covSample(1:2,1:2), parsMdl.Posi * ones(2,1), x.PreMat_LH), ...
%     NetStat);

prePrior = -squeeze(preSample(1,2,:));
% prePrior = squeeze(preSample(2,2,:));
prePrior = reshape(prePrior, size(NetStat));
preSamplePred = preSamplePred + shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], [1,1, size(prePrior)]);

%% Compute the mutual information
% if flagTest == 2
    
% A fixed prior in the world
parsMdl.UrecWorld = 10;
prePrior = parsMdl.UrecWorld * NetStat(end).PreMat_LH(1) / parGrid(end).Ufwd;

% parsMdl.tTrial = parsMdl.tBin;
[~, covPosterior] = arrayfun(@(netpars) getPosteriorHierMdl(netpars), parGrid, 'uniformout', 0);

[~, MutualInfo, MutualInfo_UpBound] = arrayfun(@(S, covPost) ...
    getMutualInfo(zeros(2,1), covPost{1}, ...
    S.meanSample(1:2), S.covSample(1:2,1:2), 2*parsMdl.width), ...
    NetStat, covPosterior);
% end

%% Plot the statistics of sampling distribution
figure

switch flagTest
    case 1
        subplot(2,2,1)
        plot(squeeze(preSample(1,1,:)), squeeze(preSamplePred(1,1,:)), 'o')
        hold on
        plot(squeeze(preSample(2,2,:)), squeeze(preSamplePred(2,2,:)), 'o')
        
        PreMat_LH = reshape([NetStat.PreMat_LH], [2,2, size(parGrid)]);
        plot(squeeze(preSample(1,1,:)), squeeze(PreMat_LH(1,1,:)), 'o')
        
        
        plot([0, preSample(1,1,end)], [0, preSample(1,1,end)], '--k')
        
        xlabel('Sampling precision')
        ylabel('Posterior precision')
        axis square
        
        subplot(2,2,2); hold on
        plot(parsMdl.Ufwd, prePrior)
        plot(parsMdl.Ufwd, squeeze(PreMat_LH(1,1,:)))
        plot(parsMdl.Ufwd, squeeze(preSample(1,1,:)))
        plot(parsMdl.Ufwd, squeeze(preSamplePred(1,1,:)))
        legend('Prior', 'Likelihood', 'Samples', 'Posterior', 'location', 'best')
        axis square
        clear PreMat_LH
        
        subplot(2,2,3)
        covSample = reshape([NetStat.covSample], 4,4,[]);
        plot(parsMdl.Ufwd, squeeze(covSample(3,3,:)))
        ylabel('Diff. Corr.')
        xlabel('Feedforward rate')
        axis square
        
        subplot(2,2,4)
        plot(parsMdl.Ufwd, MutualInfo /parsMdl.tBin*1e3)
        xlabel('Feedforward rate')
        ylabel('Mutual info. (bit/sec.)')
        axis square
    case 2
        cSpec = lines(2);
        subplot(2,2,1); hold on
        plot(parsMdl.jxe, squeeze(preSample(1,1,:)), 'o', 'color', cSpec(1,:))
        plot(parsMdl.jxe, squeeze(preSamplePred(1,1,:)), 'color', cSpec(1,:))
        plot(parsMdl.jxe, squeeze(preSample(2,2,:)), 'o', 'color', cSpec(2,:))
        plot(parsMdl.jxe, squeeze(preSamplePred(2,2,:)), 'color', cSpec(2,:))
        xlabel('Rec weight w_E')
        ylabel('Sampling precision')
        axis square
        
        subplot(2,2,2); hold on
        plot(parsMdl.jxe, MutualInfo ./ parsMdl.tBin*1e3, 'color', cSpec(1,:))
        plot(parsMdl.jxe([1, end]), MutualInfo_UpBound([1,end])./ parsMdl.tBin*1e3, 'color', cSpec(2,:))
        xlabel('Rec weight w_E')
        ylabel('Mutual Info. (bit/sec.)')
        axis square
        
        subplot(2,2,3)
        plot(-squeeze(preSample(1,2,:)), parsMdl.jxe)
        xlabel('Prior precision \Lambda_s (world)')
        ylabel('Optimal rec. exc. weight w_E')
        axis square
        
        subplot(2,2,4)
        covSample = reshape([NetStat.covSample], 4,4,[]);
        plot(parsMdl.jxe, squeeze(covSample(3,3,:)))
        ylabel('Diff. Corr. from sampling')
        xlabel('Rec. weight w_E')
        axis square
end


%% Plot the statistics of neuronal responses
figure

CorrRate = cellfun(@(nspk) corr(nspk'/tTrial*1e3), nSpk, 'uniformout', 0);
CorrRate = cellfun(@(crate) mean(crate(:), 'omitnan'), CorrRate);
ratePop = cellfun(@(nspk) squeeze(mean(nspk,2))/tTrial*1e3, nSpk', 'uniformout', 0);
ratePop = cell2mat(ratePop);

tuneParams = lsFitTuneFunc(ratePop, parsMdl); % [Height, posi, Width, Bias]
        
     
switch flagTest
    case 1
        yyaxis left
        subplot(1,2,1)
        plot(parsMdl.Ufwd, tuneParams(1,:))
        
        yyaxis right
        plot(parsMdl.Ufwd, CorrRate)
        xlabel('Feedforward rate (Hz)')
        xlim([0, parsMdl.Ufwd(end)])
        title(['w_E=' num2str(parsMdl.jxe)])
    case 2
        subplot(1,2,2)
        yyaxis left
        plot(parsMdl.jxe, tuneParams(1,:))
        
        yyaxis right
        plot(parsMdl.jxe, CorrRate)
        xlabel('Rec. weight w_E')
        title(['U_{fwd}=' num2str(parsMdl.Ufwd)])
        
end
axis square
yyaxis left
ylabel('Firing rate')

yyaxis right
ylabel('Avg. correlation')
        
        
%% Plot example sampling distributions with recurrent weight
figure

hAxe(1) = axes('position', [0.1, 0.1, 0.6, 0.6]); hold on; daspect([1,1,1])
hAxe(2) = axes('position', [0.1, 0.75, 0.6, 0.15]); hold on; %daspect([1,4,1])
hAxe(3) = axes('Position', [0.75, 0.1, 0.15, 0.6]); hold on; %daspect([4,1,1])

axisLim = 80*[-1, 1, -1, 1];

histEdge = linspace(axisLim(1), axisLim(2), 2e2);

for IdxJxe = [2,4,10]
    meanSample = NetStat(IdxJxe).meanSample(1:2);
    covSample = NetStat(IdxJxe).covSample(1:2,1:2);
    
    fh = @(x,y) ( ([x;y] - meanSample)' / covSample/9 * ([x;y]-meanSample) - 1);
    hEllipse = fimplicit(hAxe(1), fh, [meanSample(1) + 4*covSample(1)*[-1, 1], meanSample(2) + 4*covSample(4)*[-1, 1]], ...
        'linew', 2);
    % Marginal distribution
    for iter = 1: 2
        switch iter
            case 1
                plot(hAxe(iter+1), histEdge, ...
                    normpdf(histEdge, meanSample(iter), sqrt(covSample(iter,iter))), ...
                    'linew', 2);
            case 2
                plot(hAxe(iter+1), normpdf(histEdge, meanSample(iter), sqrt(covSample(iter,iter))), ...
                    histEdge, 'linew', 2);
        end
        
    end
end


% Set the properties of plots
axes(hAxe(2)); axis tight
axes(hAxe(3)); axis tight
set(hAxe(2:3), 'xtick', {}, 'ytick', {}, 'xticklabel', {}, 'yticklabel', {})

axes(hAxe(1))
axis(axisLim)
axis square; axis tight
axis(axisLim)
title(['U_{fwd}:', num2str(parsMdl.Ufwd)]);

box on
title('')

linkaxes(hAxe(1:2), 'x')
linkaxes(hAxe([1,3]), 'y')