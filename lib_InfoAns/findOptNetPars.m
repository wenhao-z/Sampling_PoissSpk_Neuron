function OptNetPars = findOptNetPars(UrecWorld, NetStat, parsMdl)
% Find the optimized recurrent strength given the world parameter

% Wen-Hao Zhang, Aug. 7, 2019


if sum(cellfun(@(x) strcmp(x, 'jxe'),  fieldnames(parsMdl)))
    nameVar = 'jxe';
else
    nameVar = 'JrcRatio';
end
OptNetPars.namePar = nameVar;

meanPosterior = parsMdl.Posi - parsMdl.dPosi/2;
meanPosterior = repmat(meanPosterior, 2, 1);

% Initialization
OptNetPars.valuePar   = zeros(length(UrecWorld), length(parsMdl.Ufwd));
OptNetPars.UrecHeight = zeros(length(UrecWorld), length(parsMdl.Ufwd));

for iterUfwd = 1: length(parsMdl.Ufwd)
    for iterUrec = 1: length(UrecWorld)
        % Assign the parameter to model
        netpars = parsMdl;
        netpars.Ufwd = parsMdl.Ufwd(iterUfwd);
        netpars.UrecWorld = UrecWorld(iterUrec);
        [~, covPosterior] = getPosteriorHierMdl(netpars);
        
        [~, MutualInfo] = arrayfun(@(S) getMutualInfo(meanPosterior, covPosterior, S.meanSample(1:2), S.covSample(1:2,1:2), 2*parsMdl.width), ...
            squeeze(NetStat(1,:,iterUfwd)));
        [~, IdxMax] = max(MutualInfo);
        
        
        OptNetPars.valuePar(iterUrec, iterUfwd)   = parsMdl.(nameVar)(IdxMax);
        OptNetPars.UrecHeight(iterUrec, iterUfwd) = NetStat(1,IdxMax,iterUfwd).rateHeight - parsMdl.Ufwd(iterUfwd);
    end
end