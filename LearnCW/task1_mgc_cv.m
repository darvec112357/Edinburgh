%
% Versin 0.9  (HS 06/03/2020)
%
function task1_mgc_cv(X, Y, CovKind, epsilon, Kfolds)
% Input:
%  X : N-by-D matrix of feature vectors (double)
%  Y : N-by-1 label vector (int32)
%  CovKind : scalar (int32)
%  epsilon : scalar (double)
%  Kfolds  : scalar (int32)
[I,J]=size(X);
cl=length(unique(Y));
PMap=int32(zeros(I,1));
class_num=zeros(cl,1);

for k=1:cl
    for i=1:I
        if(Y(i,1)==k)
            class_num(k,1)=class_num(k,1)+1;
        end
    end
end

class_partition=floor(class_num/Kfolds);

confu_matr=zeros(cl,cl,Kfolds);
for k=1:cl
    k_num=1;
    c=1;
    for i=1:I         
        if(Y(i,1)==k)             
            if(c<Kfolds)
                if(k_num<=class_partition(k,1))
                    PMap(i)=c;
                    k_num=k_num+1;        
                else
                    c=c+1;
                    PMap(i)=c;
                    k_num=2;
                end
            else
                PMap(i)=c;
            end
        end
    end   
end

partition_num=zeros(1,Kfolds);
for k=1:Kfolds
    for i=1:I
        if(PMap(i,1)==k)
            partition_num(1,k)=partition_num(1,k)+1;
        end
    end
end
partition_num

for k=1:Kfolds
    Ms=zeros(cl,J);
    for m=1:cl
        for i=1:I
            if(PMap(i,1)~=k && Y(i,1)==m)
                if(k<Kfolds)
                    Ms(m,:)=Ms(m,:)+X(i,:)/(class_num(m,1)-class_partition(m,1));
                else
                    Ms(m,:)=Ms(m,:)+X(i,:)/(class_partition(m,1)*(Kfolds-1));
                end
            end
        end      
    end
    path='t1_mgc_cv_Ms.mat';
    path2=insertAfter(path,'mgc_',string(Kfolds));
    path3=insertAfter(path2,'cv',string(k));
    save(path3,'Ms');
    Covs3=zeros(J,J,cl);
    Covs2=zeros(J,J,cl);
    for m=1:cl
        for i=1:I
            if(PMap(i,1)~=k && Y(i,1)==m)              
                if(k<Kfolds)                     
                    Covs2(:,:,m)=reshape(Covs2(:,:,m),[J,J])+((X(i,:)-Ms(m,:))'*(X(i,:)-Ms(m,:)))/(class_num(m,1)-class_partition(m,1));
                else                  
                    Covs2(:,:,m)=reshape(Covs2(:,:,m),[J,J])+((X(i,:)-Ms(m,:))'*(X(i,:)-Ms(m,:)))/(class_partition(m,1)*(Kfolds-1));
                end               
            end
        end
    end
    Covs_shared=zeros(J,J);
    for c=1:cl
        Covs_shared=Covs_shared+Covs2(:,:,c)/cl;
    end
    for m=1:cl
        if(CovKind==1)
            Covs3(:,:,m)=Covs2(:,:,m);      
        end
        if(CovKind==2)
            for j=1:J
                Covs3(j,j,m)=Covs2(j,j,m);
            end
        end
        if(CovKind==3)                       
            Covs3(:,:,m)=Covs_shared;            
        end   
        Covs3(:,:,m)=Covs3(:,:,m)+epsilon*(eye(J)); 
    end
    Covs=permute(Covs3,[3,2,1]);
    path='t1_mgc_cv_ck_Covs.mat';
    path2=insertAfter(path,'mgc_',string(Kfolds));
    path3=insertAfter(path2,'cv',string(k));
    path4=insertAfter(path3,'ck',string(CovKind));    
    save(path4,'Covs');
    CM=zeros(cl,cl);
    for i=1:I
        if(PMap(i,1)==k)
            pos_prob=zeros(1,cl);
            for c=1:cl                             
                pos_prob(1,c)=exp(-((X(i,:)-Ms(c,:))*inv(Covs3(:,:,c))*(X(i,:)-Ms(c,:))')/2)/(((2*pi)^(J/2))*(det(Covs3(:,:,c))^(1/2)));                                         
            end
            [max_num,max_idx]=max(pos_prob);
            CM(Y(i,1),max_idx)=CM(Y(i,1),max_idx)+1;
        end
    end    
    path='t1_mgc_cv_ck_CM.mat';
    path2=insertAfter(path,'mgc_',string(Kfolds));
    path3=insertAfter(path2,'cv',string(k));
    path4=insertAfter(path3,'ck',string(CovKind));    
    save(path4,'CM');
   confu_matr(:,:,k)=CM/partition_num(k);
end

final_confu_matr=zeros(cl,cl);
for k=1:Kfolds
    final_confu_matr=final_confu_matr+confu_matr(:,:,k)/Kfolds;
end

 path='t1_mgc_cv_ck_CM.mat';
 path2=insertAfter(path,'mgc_',string(Kfolds));
 path3=insertAfter(path2,'cv',string(Kfolds+1));
 path4=insertAfter(path3,'ck',string(CovKind));    
 save(path4,'final_confu_matr');

accuracy=0;
for c=1:cl
    accuracy=accuracy+final_confu_matr(c,c);
end

a='t1_mgc_cv_PMap.mat';
b=insertAfter(a,'mgc_',string(Kfolds));
save(b,'PMap');


% Variables to save
%  PMap   : N-by-1 vector of partition numbers (int32)
%  Ms     : C-by-D matrix of mean vectors (double)
%  Covs   : C-by-D-by-D array of covariance matrices (double)
%  CM     : C-by-C confusion matrix (double)


  
  % For each <p> and <CovKind>
  %  save('t1_mgc_<Kfolds>cv<p>_Ms.mat', 'Ms');
  %  save('t1_mgc_<Kfolds>cv<p>_ck<CovKind>_Covs.mat', 'Covs');
  %  save('t1_mgc_<Kfolds>cv<p>_ck<CovKind>_CM.mat', 'CM');
  %  save('t1_mgc_<Kfolds>cv<L>_ck<CovKind>_CM.mat', 'CM');
  % NB: replace <Kfolds>, <p>, and <CovKind> properly.

end
