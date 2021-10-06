% Plot Fisher information with recurrent weight

% Wen-Hao Zhang
% Aug. 16, 2021

% Load data
setWorkPath;

addpath(fullfile(Path_RootDir, 'lib_InfoAns'));

datPath = fullfile(Path_RootDir, 'Data');
Folder = 'HawkesNet';

fileName = 'FisherInfoRecNet_210823_2032.mat';
% fileName = 'FisherInfoRecNet_210821_1227.mat';
% fileName = 'FisherInfoRecNet_210819_1749.mat';
load(fullfile(datPath, Folder, fileName), 'InfoAnsRes', 'parsMdl', 'parGrid');

%%
InfoAnsRes.FI_sext_ufwd = arrayfun(@(netpars) ...
    fisherInfo_TheorySimple(netpars.Ufwd*sqrt(pi)*netpars.TunWidth/netpars.width, ...
    netpars.Ne, sqrt(2)*netpars.TunWidth, 0), ...
    parGrid(1,:));

% Reshape the results
szGrid = size(parGrid);
for varName = fieldnames(InfoAnsRes)'
   InfoAnsRes.(varName{1}) = reshape(InfoAnsRes.(varName{1}), szGrid(2:end));
end

%%

figure
hold on

IdxNe = 1:4;
% plot(parsMdl.jxe, InfoAnsRes.FI_sext(IdxNe,:)' ./ InfoAnsRes.FI_sext_ufwd(IdxNe,:)' * 100)
% errorbar(repmat(parsMdl.jxe(:), 1, length(IdxNe)), ...
%     InfoAnsRes.FI_sext(IdxNe,:)' ./ InfoAnsRes.FI_sext_ufwd(IdxNe,:)' * 100, ...
%     InfoAnsRes.stdFI_sext(IdxNe,:)'./ InfoAnsRes.FI_sext_ufwd(IdxNe,:)'*100);

errorbar(repmat(parsMdl.jxe(:), 1, length(IdxNe)), ...
    InfoAnsRes.FI_sext(IdxNe,:)' ./ InfoAnsRes.FI_sext(IdxNe,1)'/2 * 100, ...
    InfoAnsRes.stdFI_sext(IdxNe,:)'./ InfoAnsRes.FI_sext(IdxNe,1)'*100,'-o');

% plot(parsMdl.jxe, InfoAnsRes.FI_sext(IdxNe,:)' ./ InfoAnsRes.FI_sext(IdxNe,1)'/2 * 100)
% plot(parsMdl.jxe, InfoAnsRes.FI_sext(IdxNe,:)' ./ InfoAnsRes.FI_sext(IdxNe,1)'/2 * 100 ...
%     + InfoAnsRes.stdFI_sext(IdxNe,:)'./ InfoAnsRes.FI_sext(IdxNe,1)'*100)
% plot(parsMdl.jxe, InfoAnsRes.FI_sext(IdxNe,:)' ./ InfoAnsRes.FI_sext(IdxNe,1)'/2 * 100 ...
%     - InfoAnsRes.stdFI_sext(IdxNe,:)'./ InfoAnsRes.FI_sext(IdxNe,1)'*100)


hold on
plot(parsMdl.jxe([1,end]), 50*ones(1,2), '--k')
xlabel('Rec. weight')
ylabel('Loss of Fisher info. (%)')

% ylim([30, 55])
axis square

%%

% ratefwd = parsMdl.Ufwd * gaussTuneKerl(0, sqrt(2)*parsMdl.TunWidth, parsMdl, 0);
% FI = sum(ratefwd) / parsMdl.TunWidth^2/2;



