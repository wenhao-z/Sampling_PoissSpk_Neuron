% The main code performing all kinds of simulations on a varieties of
% models in demonstrating the information with network parameters.

% Wen-Hao Zhang, June 24, 2019


setWorkPath;
addpath(fullfile(Path_RootDir, 'simExp'));

%%
flagTask = 5;
% 1. Demo of the Gibbs sampling of a hierarchical model containing s and z.
%    (Figure 4 of the paper)
% 2. For the same hierarchical model as task 3, calculate the cross entropy
%    and classical Fisher information with model parameters.
% 3. Demo of Poisson spiking variability (a network without recurrent connections
%    between E neurons) samples a distribution (Figure 2 of the paper)
% 4. Demo of a single recurrent network samples a stimulus and context
%    (Figure 4 of the paper)
% 5. Scan parameters of single recurrent network and compare its sampling
%    distribution with joint posterior of stimulus and context
%    (Figure 5A and Figure S1-S2 of the paper)
% 6. The Fisher information of neurons in a recurrent network with
%    recurrent strength
%    (Figure 1 of the paper)
% 7. Demo of the responses of two coupled neural networks
%    (Figure 6 of the paper)
% 8. Scan parameters of two coupled neural networks and compare its
%    sampling distribution with posterior
%    (Figure 7 of the paper)
% 9. Test the sampling speed in a single recurrent net and coupled nets
%    (Figure S6 of the paper)

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
