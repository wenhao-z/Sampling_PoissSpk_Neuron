% Demo of the Gibbs sampling in order to infer a hierarchical model
% Wen-Hao Zhang, June 20, 2019

% The generative model used in this simulation is the same as my Cosyne
% 2019 study
% Generative structure: z --> s --> x
% x: observations;
% s and z: latent variables

% p(x|s) = N(x|s, sigma_x^2), where N(.) is a normal distribution;
% p(s|z) = N(s|z, sigma_s^2);
% p(z)   = uniform.

%% Parameters
sigma_x = 0.8;
sigma_s = 1;

x = 0;
% Mean and covariance of the posterior distribution
meanPosterior = x * zeros(2,1);
covPosterior = sigma_x^2 * ones(2,2);
covPosterior(2,2) = covPosterior(2,2) + sigma_s^2;

%% Simulation

% Gibbs sampling
sInit = 4;
zInit = -4;

w = [sigma_s^2, sigma_x^2];
w = w/ sum(w);
Lambda = sigma_x^(-2) + sigma_s^(-2);

tLen = 1e4;
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

%% Calculate the statistics of sampled s and z
meanSample = mean([sArray; zArray], 2);
covSample = cov([sArray; zArray]');

% % Calculate the cross entropy
XEnt = log(2*pi) ...
    + log(det(covSample)) ...
    + trace(covPosterior / covSample) ...
    + (meanSample - meanPosterior)' / covSample * (meanSample - meanPosterior);
XEnt = XEnt /2;

%% Plot results

figure(1)
% Trajectory of s and z over time
subplot(3,4, 1:3)
plot(1:tLen, sArray(1:end-1))
hold on
plot(1:tLen, zArray(1:end-1))
legend('s', 'z')

% Marginal distribution of s and z
subplot(3,4, 4)

%% % TBD: only use the equilibrium state!

% Generate new figure
figure(2);

nSteps = 1e3;
hAxe = plotJointMarginalHist(sArray(1:nSteps), zArray(1:nSteps));
axes(hAxe(1))
xlabel('Sampled s (local)')
ylabel('Sampled z (global)')
axis(hAxe(1), 5*[-1,1,-1,1])

% Plot the posterior
fh = @(x,y) ( ([x;y] - meanPosterior)' / covPosterior/9 * ([x;y]-meanPosterior) - 1);
hEllipse = fimplicit(hAxe(1), fh, [meanPosterior(1) + 4*covPosterior(1)*[-1, 1], ...
    meanPosterior(2) + 4*covPosterior(4)*[-1, 1]], '--g', 'linew', 2);

% Plot the contour of posterior 
% X = meanPosterior(1) + 3*covPosterior(1,1)*[-1, 1];
% Y = meanPosterior(2) + 3*covPosterior(2,2)*[-1, 1];
X = [-6, 6];
Y = [-6, 6];
X = linspace(X(1), X(2), 101);
Y = linspace(Y(1), Y(2), 101);
[X,Y] = meshgrid(X,Y);
Z = mvnpdf([X(:), Y(:)], meanPosterior(:)', covPosterior);
contourf(X(1,:), Y(:,1), reshape(Z,size(X)), 'linestyle', 'none')
% imagesc(X(1,:), Y(:,1), reshape(Z,size(X)))

% Define the colormap of the same color series
cMap = getColorMapPosNegDat([0, max(Z)], 64);
% cMap = flipud(hot(64));
colormap(cMap);
axis xy


% Plot the Gibbs sampling steps
nSteps = 10;
plot([sArray(1:nSteps); sArray(1:nSteps)], [zArray(1:nSteps); zArray(2:nSteps+1)], 'k')
plot([sArray(1:nSteps); sArray(2:nSteps+1)], [zArray(2:nSteps+1); zArray(2:nSteps+1)], 'k')
plot(sArray(2:nSteps+1), zArray(2:nSteps+1), 'ok', 'markersize', 6)

axis([-5, 5, -5, 5])
set(hAxe(1), 'xtick', -5:5:5, 'ytick', -5:5:5)

%% Animation of the Gibbs sampling trajectories

% Animation_GibbsSampling;