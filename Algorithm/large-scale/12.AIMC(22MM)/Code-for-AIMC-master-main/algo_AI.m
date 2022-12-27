function [G,F,Y,iter,obj,alpha] = algo_AI(X,gt,d)
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

k = length(unique(gt));
m = length(X);
n = size(gt,1);

G = cell(m,1); 
for i = 1:m
   di = size(X{i},2); 
   G{i} = zeros(di,d); % di * d
   X{i} = mapstd(X{i}',0,1); % turn into dv*n
end
%Initilize G,F
F = ones(d,k);
Y = zeros(n,1); 
for i=1:k
    Y(i)=i;
end


alpha = ones(1,m)/m;
opt.disp = 0;

flag = 1; %judge if convergence
iter = 0;
%%
while flag
    iter = iter + 1;
    XvYT = cell(m,1);
    %% optimize G_i  dv*d
    parfor iv=1:m
        XvYT{iv} = zeros(size(X{iv},1),k);
        for j=1:k
            XvYT{iv}(:,j) = sum(X{iv}(:,Y==j), 2);
        end
        [U,~,V] = svds(XvYT{iv}*F',d);
        G{iv} = U*V';
    end
    
    %% optimize F d*k
    J = 0;
    parfor ia = 1:m
        J = J + alpha(ia) * G{ia}' * XvYT{ia};
    end
    [Unew,~,Vnew] = svds(J,k);
    F = Unew*Vnew';
    
    %% optimize Y  n*1
    loss = zeros(n, k);
    parfor ij=1:m
        loss = loss + alpha(ij) * EuDist2(X{ij}', F'*G{ij}', 0);
    end
    [~, Y] = min(loss, [], 2); 

    %% optimize alpha
    aloss = zeros(1,m);
    parfor iv = 1:m
        aloss(iv) = sqrt(sum(sum((X{iv}-G{iv}*F(:,Y)).^2)));
    end
    alpha=1./(2*aloss);

    %%
    term = zeros(m, 1);
    parfor iv = 1:m
        term(iv) = sum(sum((X{iv}-G{iv}*F(:,Y)).^2));
    end
    obj(iter) = alpha*term;
    
    
    if (iter>2) && (abs((obj(iter)-obj(iter-1))/(obj(iter)))<1e-5 || iter>maxIter || obj(iter) < 1e-10)
        flag = 0;
    end
end
         
         
    
