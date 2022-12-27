function [U,V,Z,labels] = FEMSRL(X,A,gt,lambda)

v = size(X,1) + size(X,2)-1; % the number of views
n = size(X{1},2); % the number of samples, n_dimension
m = size(A{1},2); % the number of anchor samples, n_anchor_number
k = length(unique(gt)); % the number of clusters
% we set Z = U*V, s.t. U'U = I
U = cell(1,v); % m*m
V = ones(n,m)/m; % n*m
Y = cell(1,v);
Z = cell(1,v);
for i = 1:v
    Z{i} = ones(n,m)/m;
    U{i} = ones(m,m)/m;
    Y{i} = zeros(n,m);
end

quadprog_options = optimset( 'Algorithm','interior-point-convex','Display','off');

iter = 0;
Isconverg = 0;
max_iter = 20;

epson = 1e-3;

rho = 10e-5; max_rho = 10e10; pho_rho = 2;

%% optimization
while (Isconverg == 0)
    % updating Z
    Z_old = Z;
    for i = 1:v
        Z_tmp1 = A{i}'*A{i}+ 0.5*rho*eye(m);
        Z_tmp1 = 2*Z_tmp1;
        Z_tmp1 = (Z_tmp1'+Z_tmp1)/2;
        
        Z_i = Z{i};
        X_i = X{i};
        A_i = A{i};
        Y_i = Y{i};
        UV_i = U{i}*V';
        parfor j = 1:n
            Z_tmp2 = -2*((X_i(:,j))'*A_i)+Y_i(j,:)-rho*(UV_i(:,j))';
            Z_i(j,:) = quadprog(Z_tmp1,Z_tmp2',[],[],ones(1,m),1,zeros(m,1),ones(m,1),Z_i(j,:),quadprog_options);
        end
        Z{i}=Z_i; 
    end

    % update V
    V_tmp1 = (lambda+0.5*rho*v)*eye(m); % V_tmp1 is defined as left form, since U'*U=I
    V_tmp1 = 2*V_tmp1;
    V_tmp1 = (V_tmp1'+V_tmp1)/2;
    for j = 1:n
        V_tmp2 = zeros(1,m);
        for i = 1:v
            V_tmp2 = V_tmp2 + 0.5*Y{i}(j,:)*U{i}+0.5*rho*Z{i}(j,:)*U{i};
        end
        V_tmp2 = -2*V_tmp2;
        V(j,:) = quadprog(V_tmp1,V_tmp2',[],[],ones(1,m),1,zeros(m,1),ones(m,1),V(j,:),quadprog_options);
    end

    % updating U
    for i = 1:v
        U_a = Z{i} + Y{i}/rho;
        U_b = V'*U_a;
        [svd_U,~,svd_V] = svd(U_b,'econ');
        U{i} = svd_V*svd_U';
    end

    % update rho and Y
    for i = 1:v
        tmp = Y{i}' + rho*(Z{i}'-U{i}*V');
        Y{i} = tmp';
    end
    rho = min(rho*pho_rho, max_rho);
    
    iter = iter + 1;
    Isconverg = 1;
    for i = 1:v
        % convergence condiction: norm(Z{i}-V*U{i}',inf) < epson
        if (i<5 && norm(Z{i}-V*U{i}',inf)>epson)
            Isconverg = 0;
        end
    end

    if (iter>max_iter)
        Isconverg  = 1;
    end

end

Sbar = V;

[Q,~,~] = mySVD(Sbar,k); 

rng(1234,'twister') % set random seed for re-production
labels=litekmeans(Q, k, 'MaxIter', 100,'Replicates',10);
end