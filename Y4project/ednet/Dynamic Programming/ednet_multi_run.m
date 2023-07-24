
file_to_run = '../matlab_files';


mdps = {dir(sprintf('%s/*.mat',file_to_run)).name};
for i = 1:size(mdps,2)
    fprintf('Current mdp: %s\n',mdps{i});
    path = fullfile(file_to_run, mdps{i});
    if strcmp(file_to_run,'../matlab_files')
        solve_mdp(path, 'policies_test')
    elseif strcmp(file_to_run,'../matlab_files_penalised')
        solve_mdp(path, 'policies_penalised')
    end
end
    