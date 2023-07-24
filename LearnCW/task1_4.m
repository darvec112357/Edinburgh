function accuracy=task1_4(X, Y, CovKind, epsilon, Kfolds)
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

ll_prior_prob=log(class_num/I);

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
                    k_num=1;
                end
            else
                PMap(i)=c;
            end
        end
    end   
end


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
    Covs=zeros(J,J,cl);
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
            Covs(:,:,m)=Covs2(:,:,m);           
        end
        if(CovKind==2)
            for j=1:J
                Covs(j,j,m)=Covs2(j,j,m);
            end
        end
        if(CovKind==3)                       
            Covs(:,:,m)=Covs_shared;            
        end   
        Covs(:,:,m)=Covs(:,:,m)+epsilon*(eye(J)); 
    end
    CM=zeros(cl,cl);
    for i=1:I
        if(PMap(i,1)==k)
            ll_pos_prob=zeros(1,cl);
            for c=1:cl                             
                ll_pos_prob(1,c)=-(J*log(2*pi))/2-log(det(Covs(:,:,c)))/2-((X(i,:)-Ms(c,:))*inv(Covs(:,:,c))*(X(i,:)-Ms(c,:))')/2;                         
            end
            [max_num,max_idx]=max(ll_pos_prob);
            CM(Y(i,1),max_idx)=CM(Y(i,1),max_idx)+1;
        end
    end    
    if(k~=Kfolds)        
        confu_matr(:,:,k)=CM/floor(I/Kfolds);
    else
        confu_matr(:,:,k)=CM/(I-(Kfolds-1)*(floor(I/Kfolds)));
    end
end

final_confu_matr=zeros(cl,cl);
for k=1:Kfolds
    final_confu_matr=final_confu_matr+confu_matr(:,:,k)/Kfolds;
end

accuracy=0;
for c=1:cl
    accuracy=accuracy+final_confu_matr(c,c);
end

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