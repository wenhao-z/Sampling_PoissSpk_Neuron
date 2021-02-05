function hAxe = plotJointMarginalHist(X,Y)
% Plot the joint and marginal histogram of 2d variables.

% X and Y should be a ROW vector
% Wen-Hao Zhang, June 24, 2019
% University of Pittsburgh


meanSample = mean([X; Y], 2);
covSample = cov([X; Y]');


% alphaValue = 1;

hAxe(1) = axes('position', [0.1, 0.1, 0.6, 0.6]); hold on; daspect([1,1,1])
hAxe(2) = axes('position', [0.1, 0.75, 0.6, 0.15]); hold on; %daspect([1,4,1])
hAxe(3) = axes('Position', [0.75, 0.1, 0.15, 0.6]); hold on; %daspect([4,1,1])

% Determine the boundary of axis
% axisLim = [meanSample(1) + 4*covSample(1)*[-1, 1], ...
%     meanSample(2) + 4*covSample(4)*[-1, 1]];

axisLim = [min(X), max(X), min(Y), max(Y)];
XLen = max(X) - min(X);
YLen = max(Y) - min(Y);
axisLim = axisLim + [0.1*XLen*[-1, 1], 0.1*YLen*[-1, 1]];

% Joint distribution
% plot(hAxe(1), X, Y)
plot(hAxe(1), X, Y, '.')
hold on
plot(hAxe(1), meanSample(1), meanSample(2), '+r'); % mean value
plot(hAxe(1), X(1), Y(1), 'sg'); % Initial point

fh = @(x,y) ( ([x;y] - meanSample)' / covSample/9 * ([x;y]-meanSample) - 1);
% hEllipse = ezplot(hAxe(1), fh, [meanSample(1) + 3*covSample(1)*[-1, 1], meanSample(2) + 3*covSample(4)*[-1, 1]]);
hEllipse = fimplicit(hAxe(1), fh, [meanSample(1) + 4*covSample(1)*[-1, 1], meanSample(2) + 4*covSample(4)*[-1, 1]], ...
    'linew', 2);
title(hAxe(1), [])


% Marginal distribution
for iter = 1: 2
    switch iter
        case 1
            dataPoints = X;
            histEdge = linspace(axisLim(1), axisLim(2), 2e2);
        case 2
            dataPoints = Y;
            histEdge = linspace(axisLim(3), axisLim(4), 2e2);
    end
    
    histVal = histcounts(dataPoints, histEdge);
    histVal = histVal / (sum(histVal)*mean(diff(histEdge)));
    
    switch iter
        case 1
            stairs(hAxe(iter+1), (histEdge(1:end-1)+histEdge(2:end))/2, histVal);
            plot(hAxe(iter+1), histEdge, ...
                normpdf(histEdge, meanSample(iter), sqrt(covSample(iter,iter))), ...
                'linew', 2);
        case 2
            stairs(hAxe(iter+1), histVal, (histEdge(1:end-1)+histEdge(2:end))/2);
            plot(hAxe(iter+1), normpdf(histEdge, meanSample(iter), sqrt(covSample(iter,iter))), ...
                histEdge, 'linew', 2);
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

box on
title('')

linkaxes(hAxe(1:2), 'x')
linkaxes(hAxe([1,3]), 'y')
% set(hAxe(2),'position', [0.1, 0.75, 0.6, 0.15]);
