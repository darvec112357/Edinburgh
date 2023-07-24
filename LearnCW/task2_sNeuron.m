%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_sNeuron(W, X)
[I,J]=size(X);
X2=zeros(I,J+1);
X2(:,1)=1;
X2(:,2:J+1)=X;
Y=zeros(I,1);
for i=1:I
    a=W'*X2(i,:)';
    Y(i,1)=1/(1+exp(-a));
end
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
%  W : (D+1)-by-1 vector of weights (double)
% Output:
%  Y : N-by-1 vector of output (double)

end
