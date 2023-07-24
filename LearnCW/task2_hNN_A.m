%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_hNN_A(X)
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
% Output:
%  Y : N-by-1 vector of output (double)
[I,J]=size(X);
y=zeros(I,4);
y(:,1)=task2_hNeuron([0.58732;-1;0.25025],X);
y(:,2)=task2_hNeuron([-1;0.07792;0.34235],X);
y(:,3)=task2_hNeuron([-0.09007;-1;0.87071],X);
y(:,4)=task2_hNeuron([-1;0.09681;0.27654],X);

w=[-1;-1;1;1;-1];

Y=task2_hNeuron(w,y);
end

