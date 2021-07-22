% The main code performing all kinds of simulations on a varieties of
% models in demonstrating the information with network parameters.

% Wen-Hao Zhang, June 24, 2019

setWorkPath;
addpath(fullfile(Path_RootDir, 'simExp'));

%%
flagTask = 3;
% 1. Demo of the Gibbs sampling of a hierarchical model containing s and z.
%    This hierarchical model is used in my Cosyne 2019 study.
% 2. For the same hierarchical model as task 1, calculate the cross entropy
%    and classical Fisher information with model parameters.
% 3. Demo of a linear Hawkes process with ring structure
% 4. Information-theoretic analysis on network's responses
%    Cross entropy, mutual information and classical Fisher information
%    Network model can be bipartite net, or linear Hawkes process, or a
%    spiking CANN
% 5. Demo of Poisson spiking variability (a network without recurrent connections)
%    can sample a distribution
% 6. Demo of the responses of two coupled neural networks
% 7. Scan parameters of two coupled neural networks


switch flagTask
    case 1
        HiracMdl_GibbsSampling;
    case 2
        HiracMdl_XEntMdlParams;
    case 3
        demoRecHawkesNet;
    case 4
        InfoAns_NetMdls;
    case 5
        demoPoissonSampling;
    case 6
        demoCoupledHawkesNet;
    case 7   
        scanCoupledHawkesNet;
end