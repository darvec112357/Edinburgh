function solve_mdp(mdp_path, fname)

load(mdp_path)
%% Policy Iteration
% Using matrix inversion policy evaluation
[PI_V,PI_policy, PI_Q, ~,~] = mdp_policy_iteration (P, R, 0.99);
% To convert action encoding
PI_policy = PI_policy - 1;

%% Export results
results = double(state_idx');
results(:,2) = PI_policy; results(:,3) = PI_V;


%write header to file
cHeader = {'state', 'policy', 'values'}; 
textHeader = strjoin(cHeader, ',');
fid = fopen(sprintf('../%s/policy_%s.csv',fname,name),'w');
fprintf(fid,'%s\n',textHeader);
fclose(fid);
writematrix(results, sprintf('../%s/policy_%s.csv',fname,name),...
    'WriteMode','append');

q_vals = double(state_idx');
q_vals = [q_vals PI_Q];

cHeader = {'state'};
for act = 0:36 
    cHeader{size(cHeader,2) + 1} = num2str(act);
end
textHeader = strjoin(cHeader, ',');
fid = fopen(sprintf('../%s/Q_%s.csv',fname,name),'w');
fprintf(fid,'%s\n',textHeader);
fclose(fid);
writematrix(q_vals,sprintf('../%s/Q_%s.csv',fname,name),'WriteMode','append');

end

