% Test the convergence speed of sampling in spiking networks

% Wen-Hao Zhang, June 16, 2021
% University of Chicago

%% Parameters of the model
flagNet = 1;
% 1: A single recurrent network
% 2: Two coupled networks

switch flagNet
    case 1
        parsHawkesNet;
        
        parsMdl.jxe = 4e-3; %2e-2;
        parsMdl.FanoFactorIntVar = 1;
    case 2
        parsCoupledHawkesNet;
        
        parsMdl.jxe = 1e-2;
        parsMdl.ratiojrprc = 5;
        parsMdl.bCutRecConns = 1; % Cut off all recurrent connections to simplify theoretical analysis
        parsMdl.bShareInhPool = 0; % Whether two networks share the same inhibition pool
        
        % Input parameters
        parsMdl.Posi = 10*[-1; 1];
        parsMdl.Ufwd = 30*[1;1];
end

parsMdl.tLen = 0.1*1e3; % unit: ms
parsMdl.tBin = 1; % length of a trial. Unit: ms
parsMdl.dt = 0.1;
parsMdl.tStat = 0;
parsMdl.bSample_ufwd = 0;

% Compute the dependent parameters
parsMdl = getDependentPars_HawkesNet(parsMdl);

nRepeat = 1e4;
parGrid = repmat(parsMdl, 1, nRepeat);
%% Simulation

popVecArray = cell(1, nRepeat);
parfor iter = 1: nRepeat
    fprintf('Progress: %d/%d\n', iter, numel(parGrid));
    netpar = parGrid(iter);
    
    netpar.rngNetSpk = rng('shuffle');
    
    % Generate a sample of feedforward spiking input
    ratefwd = makeRateFwd(netpar.Posi, netpar);
    ratefwd = [ratefwd; netpar.ji0 * sum(ratefwd,1)./netpar.Ni.*ones(netpar.Ni,1)];
    
    switch flagNet
        case 1
            outSet = simHawkesNetDemo(ratefwd, netpar);
        case 2
            outSet = simCoupledHawkesNet(ratefwd(:), netpar);
    end
    popVecArray{iter} = outSet.popVec(1:2,:);
end
%%
% Get the averaged samples in a decoding time window

parsMdl.tBin = 1; % length of a trial. Unit: ms

if parsMdl.tBin ~= parsMdl.dt
    subsTrialAvg = [ repmat(1:2, 1, parsMdl.tLen/parsMdl.dt); ...
        kron(1:parsMdl.tLen/parsMdl.tBin, ones(1, round(2*parsMdl.tBin/parsMdl.dt)))];
    popVec = cellfun(@(x) accumarray(subsTrialAvg.', x(:), []), popVecArray, 'uniformout', 0);
    popVec = cell2mat(shiftdim(popVec, -1));
else
    popVec = cell2mat(shiftdim(popVecArray, -1));
end
tSample = angle(popVec) * parsMdl.width / pi;


tPlot = (1:parsMdl.tLen/parsMdl.tBin) * parsMdl.tBin;

% plot(tPlot, mean(tSample, 3))
% hold on
% plot(tPlot, squeeze(tSample(1,:,:)))
% hold on

subplot(2,1,1)
plot(tPlot, mean(tSample,3))
ylabel('Mean of samples')
title(['w_E =', num2str(parsMdl.jxe), ', U_{fwd}=', num2str(parsMdl.Ufwd)])

subplot(2,1,2)
stdSample = std(tSample, 0, 3, 'omitnan');
plot(tPlot, stdSample)
ylabel('Std. of samples')
xlabel('Time (ms)')

switch flagNet
    case 1
        legend('s', 'z')
    case 2
        legend('s_1', 's_2')
end
