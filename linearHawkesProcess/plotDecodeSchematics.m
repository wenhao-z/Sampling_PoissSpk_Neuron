% Plot the schematic of how the joint distribution over stimulus s and context a
% is decoded from the spiking activity and the recurrent input
% respectively.

% Wen-Hao Zhang,
% University of Pittsburgh
% Feb. 3, 2020

% Trial average the recurrent input
% subsTrialAvg = [ repmat(1:180, 1, parsMdl.tLen/parsMdl.dt); ...
%     kron(1:parsMdl.tLen/parsMdl.tTrial, ones(1, round(180*parsMdl.tTrial/parsMdl.dt)))];
% urecArray = accumarray(subsTrialAvg.', outSet.urecArray(:), []);
urecArray = reshape(outSet.urecArray, 180, parsMdl.tTrial/parsMdl.dt, parsMdl.tLen/parsMdl.tTrial);
urecArray = mean(urecArray, 3);

% Trial average of the spike count
% tEdge = [0: parsMdl.dt : parsMdl.tLen, parsMdl.tLen + parsMdl.dt];
tEdge = 0: parsMdl.dt : parsMdl.tLen;
neuronEdge = 0.5: parsMdl.Ne + 0.5;
bSpk = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdge, tEdge);
bSpk = reshape(bSpk, 180, parsMdl.tTrial/parsMdl.dt, parsMdl.tLen/parsMdl.tTrial);
bSpk = mean(bSpk, 3);


%%
subplot(1,2,1)
surf(bSpk)
shading interp
axis square

subplot(1,2,2)
surf(urecArray)
shading interp
axis square

%% The schematics from some artificially generated data

% Run the demo of Gibbs sampling
% HiracMdl_GibbsSampling;

sigma_x = 7;
sigma_s = 1.5*sigma_x;

x = 0;
% Mean and covariance of the posterior distribution
meanPosterior = x * zeros(2,1);
covPosterior = sigma_x^2 * ones(2,2);
covPosterior(2,2) = covPosterior(2,2) + sigma_s^2;

% Gibbs sampling
sInit = 5;
zInit = -5;

w = [sigma_s^2, sigma_x^2];
w = w/ sum(w);
Lambda = sigma_x^(-2) + sigma_s^(-2);

tLen = 60;
zArray = zeros(1, tLen);
sArray = zeros(1, tLen);
sMeanArray = zeros(1, tLen);

zArray(1) = zInit;
sArray(1) = sInit;

for t = 1: tLen
    zArray(t+1) = sArray(t) + sigma_s * randn(1);
    sMeanArray(t+1) = w * [x, zArray(t+1)]';
    sArray(t+1) = sMeanArray(t+1) + 1/sqrt(Lambda) * randn(1);
end
% ---------------------------------------------------------------

parsHawkesNet;
parsMdl.Ufwd = 1e3/parsMdl.dt;

urecArray = makeRateFwd(zArray, parsMdl);
%%

% figure(1);
% clf;
tLenPlot = tLen - 10;
for iterPlot = 1: 2
    figure(iterPlot)
    switch iterPlot
        case 1
            Posi = sArray;
            cSpec = [0, 153, 68]/255;
            zVal = 1;
            plotDat = zVal * makeRateFwd(Posi, parsMdl);
        case 2
            Posi = zArray;
            cSpec = [1, 0, 0];
            zVal = 0.8;
            plotDat = zVal*makeRateFwd(Posi, parsMdl);
    end
    
    % Set the colormap
    cMap = zeros(64, 3);
    for iter =  1:3
        cMap(:,iter) = linspace(1, cSpec(iter), 64);
    end 
    colormap(cMap)
    
   
    clf;
    [X,Y] = ndgrid(parsMdl.PrefStim, 1:tLenPlot);
    % surfl(X, Y, nSpkFake(:,1:tLen))
    surf(X, Y, plotDat(:,1:tLenPlot))
    shading interp
%     light('Position',[150 0 2],'Style','local', 'color', cSpec)
    % light('Position',[10 -10 0],'Style','local')
    axis square;
    grid off
    view(50, 72)
    
    hold on
    plot3(Posi(1:tLen), 1:tLen, zVal*ones(1, tLen), 'k', 'linew', 1)
    % plot3(Posi(1:tLen), 1:tLen, 0*ones(1, tLen), 'k', 'linew', 1)
    
    % Set the axis
    xlim(180*[-1,1])
    zlim([0, 1])
    
    set(gca, 'xtick', -180:90:180)
end

