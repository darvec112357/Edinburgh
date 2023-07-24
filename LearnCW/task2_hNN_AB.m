%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_hNN_AB(X)
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
% Output:
%  Y : N-by-1 vector of output (double)
[I,J]=size(X);
ya=zeros(I,4);
ya(:,1)=task2_hNeuron([0.58732;-1;0.25025],X);
ya(:,2)=task2_hNeuron([-1;0.07792;0.34235],X);
ya(:,3)=task2_hNeuron([-0.09007;-1;0.87071],X);
ya(:,4)=task2_hNeuron([-1;0.09681;0.27654],X);

yb1=zeros(I,3);
yb1(:,1)=task2_hNeuron([-1;0.08496;0.18801],X);
yb1(:,2)=task2_hNeuron([-0.13697;-0.28258;1],X);
yb1(:,3)=task2_hNeuron([-0.48031;-1;0.26802],X);

yb2=zeros(I,3);
yb2(:,1)=task2_hNeuron([-1;-0.09182;0.77339],X);
yb2(:,2)=task2_hNeuron([-0.13697;-0.28258;1],X);
yb2(:,3)=task2_hNeuron([-0.31319;-0.6497;1],X);

w=[-1;-1;1;1;-1];
Ya=task2_hNeuron(w,ya);

wb=[-1;-1;2;-1];
Yb1=task2_hNeuron(wb,yb1);
Yb2=task2_hNeuron(wb,yb2);

new_b=[Ya(:),Yb1(:),Yb2(:)];

Y=task2_hNeuron(wb,new_b);

end
