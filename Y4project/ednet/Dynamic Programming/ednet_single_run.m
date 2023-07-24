clc
clear
load('MDP_aug5.mat');
%mdp_check(P,R)

%% Value Iteration
mdp_verbose;
[VI_V, VI_policy, VI_iter, VI_cpu_time] = mdp_value_iteration (P, R, 0.99);
% To convert action encoding
VI_policy = VI_policy - 1;

%% Policy Iteration
% Using matrix inversion policy evaluation
[PI_V,PI_policy, PI_Q, PI_iter, PI_cpu_time] = mdp_policy_iteration (P, R, 0.99);
% To convert action encoding
PI_policy = PI_policy - 1;

%% Export results
results = double(state_idx');
results(:,2) = PI_policy; results(:,3) = PI_V;
results(:,4) = VI_policy; results(:,5) = VI_V;

%write header to file
cHeader = {'state', 'PI_policy', 'PI_values', 'VI_policy', 'VI_values'}; 
textHeader = strjoin(cHeader, ',');
fid = fopen(sprintf('../policies/policy_%s.csv',name),'w');
fprintf(fid,'%s\n',textHeader);
fclose(fid);
writematrix(results, sprintf('../policies/policy_%s.csv',name),'WriteMode','append');

q_vals = double(state_idx');
q_vals = [q_vals PI_Q];

cHeader = {'state'};
textHeader = strjoin(cHeader, ',');
fid = fopen(sprintf('../policies/Q_%s.csv',name),'w');
fprintf(fid,'%s\n',textHeader);
fclose(fid);
writematrix(q_vals,sprintf('../policies/Q_%s.csv',name),'WriteMode','append');

%% Policy comparison
policy_diff = sum(PI_policy ~= VI_policy);
fprintf('Policy diff between PI & VI is %d\n',policy_diff)

%% Q-learning MDP
% [Q, QL_V, QL_policy, mean_discrepancy] = mdp_Q_learning(P, R, 0.99, 50000);
% QL_policy = QL_policy - 1;
% plot(mean_discrepancy)
% 
% policy_diff = sum(PI_policy ~= QL_policy);
% fprintf('Policy diff between PI & QL is %d\n',policy_diff)
