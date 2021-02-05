% Analyze the network results after scanning parameters. 
% Before using this code, run simNet_ScanPars.m to collect the network outputs
% under different parameter settings.

% Note: Please assign correct fileName on line 17!!

% Wen-Hao Zhang,
% University of Pittsburgh
% Feb. 3, 2020

% Load data
setWorkPath;
addpath(fullfile(Path_RootDir, 'lib_InfoAns'));

datPath = fullfile(Path_RootDir, 'Data');
Folder = 'HawkesNet';
fileName = '';

load(fullfile(datPath, Folder, fileName));


if flagModel == 3
    parsMdl.Ufwd = parGrid(1).Ufwd;
end
%%
% Find the optimal network parameter with different world parameters
UrecWorld = 0:2:20;
OptNetPars = findOptNetPars(UrecWorld, NetStat, parsMdl);
OptNetPars.UrecWorld = UrecWorld;

UrecWorld = arrayfun(@(S, mdlpars) findWorldParsInNet(mdlpars, S.meanSample(1:2), S.covSample(1:2,1:2)), ...
    squeeze(NetStat(1,:,:)), parGrid);

clear popVec

%% Plot the results as a function of *recurrent strength*
figure;

for iter = 1: 6
    hAxe(iter) = subplot(2,3,iter);
    axis square
    hold on
end

% The example model parameter
% Make sure this parameter is consistent with the one used in example demo.
IdxUrec = 6;
IdxUfwd = 7;

% Find the name of the varying model parameters
% In most of cases, I change the input strength Ufwd, and recurrent strength
switch flagModel
    case 3
        nameVar = 'JrcRatio';
    otherwise
        nameVar = 'jxe';
end
% Check whether this variable is indeed in the file

if ~sum(cellfun(@(x) strcmp(x, nameVar), {dimPar.namePar}))
   error([nameVar 'is not in the file.']) 
end

% Find the network parameter maximizing mutual information
[~, IdxMaxInfo] = max([InfoAnsRes(:,IdxUfwd).MutualInfo]);

axes(hAxe(1))
hold on
cSpec = cool(length(parsMdl.(nameVar)));
for iter = 1: length(parsMdl.(nameVar))
    plot(parsMdl.PrefStim, NetStat(1,iter, IdxUfwd).ratePop, 'color', cSpec(iter,:))
end
set(gca, 'xlim', parsMdl.width*[-1,1], 'xtick', parsMdl.width*(-1:0.5:1))
xlabel('Neuron index')
ylabel('Firing rate (Hz)')

colormap(cSpec); 
caxis(parsMdl.(nameVar)([1, end]))
cBar = colorbar('north', 'ticks', parsMdl.(nameVar)([1, round(end/2), end]));
cBar.Label.String = nameVar;

axes(hAxe(2))
yyaxis('left')
% plot(parsMdl.(nameVar), NetStat.rateAvg)
plot(parsMdl.(nameVar), [NetStat(1,:, IdxUfwd).rateHeight])
hold on
plot(parsMdl.(nameVar)(IdxMaxInfo), [NetStat(1,IdxMaxInfo,IdxUfwd).rateHeight], 'o')
ylabel('Bump height (Hz)')

yyaxis('right')
plot(parsMdl.(nameVar), [NetStat(1,:,IdxUfwd).corrAvg])
ylabel('Mean corr. coef.')

axes(hAxe(3))
yyaxis left
plot(parsMdl.(nameVar), arrayfun(@(S) S.covSample(1,1), NetStat(1,:,IdxUfwd)))
plot(parsMdl.(nameVar), arrayfun(@(S) S.covSample(3,3), NetStat(1,:,IdxUfwd)))
plot(parsMdl.(nameVar), ...
    arrayfun(@(S) S.covSample(3,3), NetStat(1,:,IdxUfwd)) - ...
    arrayfun(@(S) S.covSample(4,4), NetStat(1,:,IdxUfwd)) );
yyaxis right
plot(parsMdl.(nameVar)(2:end), arrayfun(@(S) S.covSample(2,2), NetStat(1,2:end,IdxUfwd)))
legend('V(s)', 'Diff. corr. (V(\mu_s))', 'Diff. corr. (Internal)', 'V(z)', 'location', 'best')
ylabel('Variance')
title('Linear decoder')

axes(hAxe(4))
plot(parsMdl.(nameVar), [InfoAnsRes(:,IdxUfwd).FisherInfo_sext])
hold on
plot(parsMdl.(nameVar), [InfoAnsRes(:,IdxUfwd).FisherInfo_sext_theory])
if parsMdl.bSample_ufwd
    plot(parsMdl.(nameVar), [InfoAnsFwdRes(:,IdxUfwd).FisherInfo_sext])
end
plot(parsMdl.(nameVar), [InfoAnsRes(:,IdxUfwd).FisherInfo_sint_theorySimp])
ylabel('Linear Fisher Info. ((deg^2 sec)^{-1})')
if parsMdl.bSample_ufwd
    legend('r (sim.)', ' r (Theory)', 'u_{fwd} (sim.)', 'r (s_{int})', 'location', 'best')
else
    legend('r (sim.)', ' r (Theory)', 'r (s_{int})', 'location', 'best')
end
title(sprintf('tTrial=%1dms', parsMdl.tTrial));

axes(hAxe(5))
% plot(parsMdl.(nameVar), KLDiv(:,1))

yyaxis left
plot(parsMdl.(nameVar), [InfoAnsRes(:,IdxUfwd).MutualInfo]/parsMdl.tTrial*1e3)
plot(parsMdl.(nameVar)(IdxMaxInfo), [InfoAnsRes(IdxMaxInfo,IdxUfwd).MutualInfo]/parsMdl.tTrial*1e3, 'o')
plot(parsMdl.(nameVar), [InfoAnsRes(IdxMaxInfo,IdxUfwd).MutualInfo_UpBound] * ones(1, length(parsMdl.(nameVar)))/parsMdl.tTrial*1e3, '--')
ylabel('Mutual Info. (bit/sec.)')

yyaxis right
plot(parsMdl.(nameVar), [InfoAnsRes(:,IdxUfwd).XEnt])
ylabel('Crosss entropy')

xlabel(nameVar)
title(sprintf('Urec(world)=%d', parsMdl.UrecWorld))

axes(hAxe(6))

% Plot the matched world parameter given networkparameters
% plot(parsMdl.(nameVar), UrecWorld(:, IdxUfwd));
% hold on
% plot(parsMdl.(nameVar), [NetStat(1,:,IdxUfwd).rateHeight] - [parGrid(:,IdxUfwd).Ufwd]);
% legend('Urec (world)', 'Urec (Net.)')
% xlabel('jxe')

% Plot the optimized network parameter given world parameters
yyaxis left
Lambda_s_World = sqrt(2*pi) * parsMdl.rho * OptNetPars.UrecWorld/ parsMdl.TunWidth;
plot(Lambda_s_World, OptNetPars.valuePar(:,IdxUfwd))
ylabel('jxe')
yyaxis right
plot(Lambda_s_World, OptNetPars.UrecHeight(:,IdxUfwd))
hold on
plot(Lambda_s_World, OptNetPars.UrecWorld, '--')
ylabel('Urec (net.)')
xlabel('\Lambda_s (world)')

% axes(hAxe(2))
% axis off
linkaxes(hAxe(2:5), 'x')
set(hAxe(2:5), 'xlim', parsMdl.(nameVar)([1,end]))

%%
% Plot the demo of how the actual distribution computed by network changes
% with recurrent strength
cSpec = cool(length(parsMdl.jxe));
cSpec = [cSpec; zeros(1,3)];

figure;

hAxe(1) = axes('position', [0.1, 0.1, 0.6, 0.6]); hold on; daspect([1,1,1])
hAxe(2) = axes('position', [0.1, 0.75, 0.6, 0.15]); hold on; %daspect([1,4,1])
hAxe(3) = axes('Position', [0.75, 0.1, 0.15, 0.6]); hold on; %daspect([4,1,1])

hold on

x = -30:0.01:30;

% for iterWrec = [2: 6: length(parsMdl.jxe), IdxMaxInfo]
for iterWrec = [5, IdxMaxInfo, 30]
    % The center is always on zero
    covSample = NetStat(1, iterWrec, IdxUfwd).covSample(1:2,1:2);
    
    if iterWrec == IdxMaxInfo
        cLine = zeros(1,3);     
        lineStyle = '-';
    else
        cLine = cSpec(iterWrec,:);
        lineStyle = '--';
    end
    
    fh = @(x,y) ( [x,y]/ covSample/9 * [x;y] - 1);
    hEllipse = fimplicit(hAxe(1), fh, [4*sqrt(covSample(1))*[-1, 1], ...
        4*sqrt(covSample(4))*[-1, 1]], 'color', cLine, 'linestyle', lineStyle, 'linew', 1);
    
    % Marginal distributions
    plot(hAxe(2), x, normpdf(x,0, sqrt(covSample(1))), 'color', cLine, ...
        'linestyle', lineStyle, 'linew', 1)
    plot(hAxe(3), normpdf(x,0, sqrt(covSample(2,2))), x, 'color', cLine, ...
        'linestyle', lineStyle, 'linew', 1)
    
end
linkaxes(hAxe(1:2), 'x')
linkaxes(hAxe([1,3]), 'y')
axes(hAxe(1))
% the center of the distribution
plot(0,0, 'x', 'color', 'k', 'markersize', 8)
xlabel('Sample s')
ylabel('Sample z')
xlim(30*[-1,1])
ylim(30*[-1,1])

set(hAxe(3), 'xlim', [0, 0.1])


%% Plot the results as a function of Ufwd

figure;
for iter = 1: 6
    hAxe(iter) = subplot(2,3,iter);
    axis square
    hold on
end

axes(hAxe(1))
hold on
cSpec = cool(length(parsMdl.Ufwd));
for iter = 1: length(parsMdl.Ufwd)
    plot(parsMdl.PrefStim, NetStat(1,IdxUrec, iter).ratePop, 'color', cSpec(iter,:))
    if parsMdl.bSample_ufwd
        plot(parsMdl.PrefStim, NetStatFwd(1,IdxUrec, iter).ratePop, '--', 'color', cSpec(iter,:))
    end
end
set(gca, 'xlim', parsMdl.width*[-1,1], 'xtick', parsMdl.width*(-1:0.5:1))
xlabel('Neuron index')
ylabel('Firing rate (Hz)')

colormap(cSpec); 
caxis(parsMdl.Ufwd([1, end]))
cBar = colorbar('north', 'ticks', parsMdl.Ufwd([1, round(end/2), end]));
cBar.Label.String = 'Ufwd';

axes(hAxe(2))
yyaxis('left')
% plot(parsMdl.(nameVar), NetStat.rateAvg)
plot(parsMdl.Ufwd, [NetStat(1, IdxUrec, :).rateHeight])
% hold on
% plot(parsMdl.Ufwd(IdxMaxInfo), [NetStat(1,IdxMaxInfo,IdxUfwd).rateHeight], 'o')
ylabel('Bump height (Hz)')

yyaxis('right')
plot(parsMdl.Ufwd, [NetStat(1,IdxUrec,:).corrAvg])
ylabel('Mean corr. coef.')

axes(hAxe(3))
plot(parsMdl.Ufwd, arrayfun(@(S) S.covSample(1,1), squeeze(NetStat(1,IdxUrec,:))))
plot(parsMdl.Ufwd, arrayfun(@(S) S.covSample(3,3), squeeze(NetStat(1,IdxUrec,:))))
plot(parsMdl.Ufwd, arrayfun(@(S) S.covSample(2,2), squeeze(NetStat(1,IdxUrec,:))))
set(gca, 'yscale', 'log')
legend('V(s)', 'Diff. corr. (V(\mu_s))', 'V(z)', 'location', 'best')
ylabel('Variance')
title('Linear decoder')


axes(hAxe(4))
plot(parsMdl.Ufwd, [InfoAnsRes(IdxUrec,:).FisherInfo_sext])
hold on
plot(parsMdl.Ufwd, [InfoAnsRes(IdxUrec,:).FisherInfo_sext_theory])
if parsMdl.bSample_ufwd
    plot(parsMdl.Ufwd, [InfoAnsFwdRes(IdxUrec,:).FisherInfo_sext], '--')
end

if parsMdl.bSample_ufwd
    legend('r (sim.)', ' r (Theory)', 'u_{fwd} (sim.)', 'location', 'best')
else
    legend('r (sim.)', ' r (Theory)', 'location', 'best')
end
ylabel('Linear Fisher Info. ((deg^2 sec)^{-1})')
title(sprintf('tTrial=%1dms', parsMdl.tTrial));

axes(hAxe(5))
% plot(parsMdl.(nameVar), KLDiv(:,1))

yyaxis left
plot(parsMdl.Ufwd, [InfoAnsRes(IdxUrec,:).MutualInfo]/parsMdl.tTrial*1e3)
% plot(parsMdl.(nameVar)(IdxMaxInfo), [InfoAnsRes(IdxMaxInfo,IdxUfwd).MutualInfo]/parsMdl.tTrial*1e3, 'o')
% plot(parsMdl.(nameVar), [InfoAnsRes(IdxMaxInfo,IdxUfwd).MutualInfo_UpBound] * ones(1, length(parsMdl.(nameVar)))/parsMdl.tTrial*1e3, '--')
ylabel('Mutual Info. (bit/sec.)')

yyaxis right
plot(parsMdl.Ufwd, [InfoAnsRes(IdxUrec,:).XEnt])
ylabel('Crosss entropy')
xlabel(nameVar)
xlabel('Ufwd')
title(['j_{xe} = ' num2str(parsMdl.jxe(IdxUrec))])

linkaxes(hAxe(2:6), 'x')
set(hAxe(2:6), 'xlim', parsMdl.Ufwd([1,end]))

%% Experimental prediction: mutual info. vs. Fisher info. (Discriminability)

figure;
for iter = 1:4
   hAxe(iter) = subplot(2,2,iter);
end

% Fisher info. (change of recurrent weight) vs. Mutual info.
axes(hAxe(1))
plot([InfoAnsRes(2:end,IdxUfwd).FisherInfo_sext], ...
    [InfoAnsRes(2:end,IdxUfwd).MutualInfo]/parsMdl.tTrial*1e3);
axis square
xlabel({'Discriminability',  '(Fisher info. (deg^{-2}/sec))'})
ylabel('Mutual info.')
set(gca, 'ylim', [18, 26], 'ytick', 18:4:26)

% Fisher info. (change of feedforward gain) vs. Mutual info.
axes(hAxe(2))
plot([InfoAnsRes(IdxUrec, 3:end).FisherInfo_sext], ...
    [InfoAnsRes(IdxUrec, 3:end).MutualInfo]/parsMdl.tTrial*1e3);
axis square
set(gca, 'ylim', [12, 30], 'ytick', 12:6:30, 'xlim', [0, 0.65])

% Int. diff. corr. vs. Mutual info.
axes(hAxe(3))
plot(arrayfun(@(S) S.covSample(3,3), NetStat(1,:,IdxUfwd)) - ...
    arrayfun(@(S) S.covSample(4,4), NetStat(1,:,IdxUfwd)), ...
    [InfoAnsRes(:, IdxUfwd).MutualInfo]/parsMdl.tTrial*1e3);
axis square
xlabel('Internal differential correlation')
ylabel('Lower bound of mutual info.')

% Ext. diff. corr. vs. Mutual info.
axes(hAxe(4))
plot(arrayfun(@(S) S.covSample(4,4), squeeze(NetStat(1,IdxUrec,:))), ...
    [InfoAnsRes(IdxUrec, :).MutualInfo]/parsMdl.tTrial*1e3);
axis square
xlabel('External differential correlation')
ylabel('Lower bound of mutual info.')
