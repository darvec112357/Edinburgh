%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_sNN_AB(X)
% Input:
%  X : N-by-D matrix of input vectors (double)
% Output:
%  Y : N-by-1 vector of output (double)
[I,J]=size(X);
ya=zeros(I,4);
ya(:,1)=task2_sNeuron([2.34691;-3.99599;1]*50,X);
ya(:,2)=task2_sNeuron([-2.92096;0.2276;1]*50,X);
ya(:,3)=task2_sNeuron([-0.10345;-1.14849;1]*50,X);
ya(:,4)=task2_sNeuron([-3.61606;0.35007;1]*50,X);

yb1=zeros(I,3);
yb1(:,1)=task2_sNeuron([-5.31876;0.4519;1]*50,X);
yb1(:,2)=task2_sNeuron([-0.13697;-0.28258;1]*50,X);
yb1(:,3)=task2_sNeuron([-1.79209;-3.73108;1]*50,X);

yb2=zeros(I,3);
yb2(:,1)=task2_sNeuron([-1.29301;-0.11872;1]*50,X);
yb2(:,2)=task2_sNeuron([-0.13697;-0.28258;1]*50,X);
yb2(:,3)=task2_sNeuron([-0.31319;-0.6497;1]*50,X);

w=[-75;-50;50;50;-50];
Ya=task2_sNeuron(w,ya);

wb=[-75;-50;100;-50];
Yb1=task2_sNeuron(wb,yb1);
Yb2=task2_sNeuron(wb,yb2);

new_b=[Ya(:),Yb1(:),Yb2(:)];

Y=task2_sNeuron(wb,new_b);

end
