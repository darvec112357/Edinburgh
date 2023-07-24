%
% Versin 0.9  (HS 06/03/2020)
%
% template script for task2_plot_regions_hNN_AB
x1_range=linspace(-1,8,1000);
x2_range=linspace(0,7,1000);
[x1,x2]=meshgrid(x1_range,x2_range);
z=task2_hNN_AB([x1(:),x2(:)]);
z2=reshape(z,1000,[]);
cmap=[0,0,1;1,1,0];
[cc,h]=contourf(x1,x2,z2);
axis([-1,8,0,7]);
xlabel('X1');
ylabel('X2');
title('Decision boundary for ~A&&B');
set(h,'LineColor','none');
colormap(cmap);
