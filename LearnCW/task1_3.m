%
% Versin 0.9  (HS 06/03/2020)
%
function task1_3(Cov)
% Input:
%  Cov : D-by-D covariance matrix (double)
% Variales to save:
%  EVecs : D-by-D matrix of column vectors of eigen vectors (double)  
%  EVals : D-by-1 vector of eigen values (double)  
%  Cumvar : D-by-1 vector of cumulative variance (double)  
%  MinDims : 4-by-1 vector (int32)  
[I,J]=size(Cov);
Evals=eig(Cov);

Evals2=sort(Evals);
EVals=flip(Evals2);
[EVecs2,D]=eig(Cov);
EVecs=zeros(I,J);

for i=1:I    
    EVecs(:,i)=EVecs2(:,I-i+1);    
end

for j=1:J
    if(EVecs(1,j)<0)
        EVecs(:,j)=EVecs(:,j)*(-1);
    end
end

CumVar=zeros(J,1);
total=sum(EVals);
for j=1:J
    CumVar(j)=sum(EVals(1:j));
end

CumVarPer=zeros(J,1);
for j=1:J
    CumVarPer(j)=CumVar(j)/total;
end

MinDims=zeros(4,1);
perc=[0.7,0.8,0.9,0.95];
a=1;
while a<5
    for j=1:J    
        if(CumVarPer(j)>perc(a))
            MinDims(a)=j;
            a=a+1;
            break
        end
    end
end

x=(1:24);
plot(x,CumVar(x));
xlabel('Feature');
ylabel('Cumulative Variance');

  save('t1_EVecs.mat', 'EVecs');
  save('t1_EVals.mat', 'EVals');
  save('t1_Cumvar.mat', 'CumVar');
  save('t1_MinDims.mat', 'MinDims');
end
