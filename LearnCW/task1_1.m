%
% Versin 0.9  (HS 06/03/2020)
%
function task1_1(X, Y)
% Input:
%  X : N-by-D data matrix (double)
%  Y : N-by-1 label vector (int32)
% Variables to save
%  S : D-by-D covariance matrix (double) to save as 't1_S.mat'
%  R : D-by-D correlation matrix (double) to save as 't1_R.mat'
[I, J] = size(X);
mu = zeros(1, J);
for i = 1:I
    for j = 1:J
        mu(j) = mu(j) + X(i,j);
    end
end

mu = mu / I;

S=zeros(J,J);
for i=1:I
    S=S+(X(i,:)-mu)'*(X(i,:)-mu);
end
S=S/I;

R=zeros(J,J);
for i=1:J
    for j=1:J
        R(i,j)=S(i,j)/sqrt(S(i,i)*S(j,j));
    end
end

save('t1_S.mat', 'S');
save('t1_R.mat', 'R');
end
