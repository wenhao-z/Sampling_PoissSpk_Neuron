% Test the performance of a Gibbs sampling in inferring infer a hierarchical model
% Two performance measures:
% 1) cross entropy (a correct measure)
% 2) classical Fisher information (will lead misleading conclucions)

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
sigma_x = 1;
sigma_s = 1;

nStat = 1e3; % The time point after which we taking into statistics

x = 0;
% Mean and covariance of the posterior distribution
meanPosterior = x * zeros(2,1);
covPosterior = sigma_x^2 * ones(2,2);
covPosterior(2,2) = covPosterior(2,2) + sigma_s^2;

%% Simulation

% Gibbs sampling
sInit = 5;
zInit = -5;

sigma_s_mdl = 0.4: 0.2:5;
nParams = length(sigma_s_mdl);

% Initiation
tLen = 6e3;
zArray = zeros(length(sigma_s_mdl), tLen);
sArray = zeros(length(sigma_s_mdl), tLen);
sMeanArray = zeros(length(sigma_s_mdl), tLen);


w = [sigma_s_mdl.^2; sigma_x^2*ones(1,nParams)]';
w = w ./ sum(w,2);
Lambda = sigma_x^(-2) + sigma_s_mdl.^(-2);

zArray(:,1) = zInit * ones(nParams, 1);
sArray(:,1) = sInit * ones(nParams, 1);

for t = 1: tLen
    zArray(:,t+1) = sArray(:,t) + sigma_s_mdl' .* randn(1);
    sMeanArray(:,t+1) = sum(w .* [x*ones(nParams,1), zArray(:,t+1)], 2);
    sArray(:,t+1) = sMeanArray(:,t+1) + 1./sqrt(Lambda') .* randn(1);
end

zArray = zArray(:, nStat+1:end);
sArray = sArray(:, nStat+1:end);
sMeanArray = sMeanArray(:, nStat+1:end);

%% Calculate the statistics of sampled s and z
meanSample = [mean(sArray,2), mean(zArray,2)]';
covSample = zeros(2,2,nParams);
XEnt = zeros(1, nParams);
MutualInfo = zeros(1, nParams);

for iter = 1: nParams
    covSample(:,:,iter) = cov([sArray(iter,:); zArray(iter,:)]');
    
    % Calculate the cross entropy, mutual information
    [XEnt(iter), MutualInfo(iter), MutualInfo_UpBound] = ...
        getMutualInfo(meanPosterior, covPosterior, ...
        meanSample(:,iter), covSample(:,:,iter), 360); % 360 deg is the width of feature space
        
end

%% Negative contrast: repeat the Fisher information calculation as experiments
varsMean = var(sMeanArray,0,2);

%% Figures

figure;
hAxe(1) = subplot(2,1,1);
hold on
yyaxis left
plot(sigma_s_mdl, XEnt)
[~, Idx] = min(XEnt);
plot(sigma_s_mdl(Idx), XEnt(Idx), 'o')
ylabel('Cross entropy')

yyaxis right
hold on
plot(sigma_s_mdl, MutualInfo)
plot(sigma_s_mdl, MutualInfo_UpBound*ones(size(sigma_s_mdl)), '--')
ylabel('Mutual info. (bit)')

hAxe(2) = subplot(2,1,2); 
hold on
% plot(sigma_s_mdl, 1./ (squeeze(covSample(1,1,:))))
plot(sigma_s_mdl, 1./Lambda' +varsMean)
ylim([0, 1.1])
% plot(sigma_s_mdl, 1./ (squeeze(covSample(1,1,:))+varsMean))

plot(sigma_s_mdl, varsMean)
legend('Fisher info. (experiment)', 'Strength of diff. corr.')
xlabel('Var. of effective prior \sigma_s')

linkaxes(hAxe, 'x')
