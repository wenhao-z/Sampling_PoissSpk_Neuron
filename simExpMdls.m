% The main code performing all kinds of simulations on a varieties of
% models in demonstrating the information with network parameters.

% Wen-Hao Zhang, June 24, 2019


setWorkPath;
addpath(fullfile(Path_RootDir, 'simExp'));

%%
flagTask = 5;
% 1. Demo of the Gibbs sampling of a hierarchical model containing s and z.
%    This hierarchical model is used in my Cosyne 2019 study.
% 2. For the same hierarchical model as task 3, calculate the cross entropy
%    and classical Fisher information with model parameters.
% 3. Demo of Poisson spiking variability (a network without recurrent connections
%    between E neurons) samples a distribution
% 4. Demo of a single recurrent network samples a stimulus and context
% 5. Scan parameters of single recurrent network and compare its sampling
%    distribution with joint posterior of stimulus and context
% 6. The Fisher information of neurons in a recurrent network with
%    recurrent strength
% 7. Demo of the responses of two coupled neural networks
% 8. Scan parameters of two coupled neural networks and compare its
%     sampling distribution with posterior
% 9. Test the sampling speed in a single recurrent net and coupled nets

switch flagTask
    case 1
        HiracMdl_GibbsSampling;
    case 2
        HiracMdl_XEntMdlParams;
    case 3
        demoPoissonSampling;
    case 4
        demoRecHawkesNet;
    case 5
        scanRecHawkesNet;
    case 6
        FisherInfo_RecNet;
    case 7
        demoCoupledHawkesNet;
    case 8
        scanCoupledHawkesNet;
    case 9
        testSamplingSpeed;
end