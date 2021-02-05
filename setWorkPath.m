% Set the working path of codes of Decentralized system
% Wen-Hao Zhang, Oct-5-2016
% @Carnegie Mellon University


% The root in my pc at CNBC-CMU
Path_RootDir = fileparts(mfilename('fullpath'));

addpath(Path_RootDir);
addpath(fullfile(Path_RootDir, 'lib'));
addpath(fullfile(Path_RootDir, 'lib_InfoAns'));
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));
addpath(fullfile(Path_RootDir, 'PlotCode'));

% addpath(fullfile(Path_RootDir, 'Params'));
cd(Path_RootDir)

% clear Path_RootDir

%%
% p = mfilename('fullpath');
% Idx = union(strfind(p, '/'), strfind(p, '\'));
% p = p(1:Idx(end)-1);
% 
% if strfind(p, 'cnl')
%     %     addpath(genpath('/home/zwh/Projects/DecenInfoIntNet'))
%     
%     addpath(genpath('/netapp/snl/lvhome/cnl/whzhang/Project/DecenNet'));
% %     addpath(genpath('/home/whzhang/Project/DecenNet'))
% else
%     addpath(p)
%     addpath(fullfile(p,'lib'))
%     addpath(fullfile(p,'Params'))
%     
%     addpath(fullfile(p,'CANN_VM'))
% end
% clear p Idx
