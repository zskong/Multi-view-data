clear
clc
warning off;
DataName{1} = 'bbcsport_2view';
DataName{2} = 'MNIST';
path = './0_KernelProcessingandClustering';
pathdata = 'C:\Datasets\Kernel';
addpath(genpath(path));

for ICount = 1:2
    % Set path
    
    dataName = DataName{ICount};
    load(['./Result/Hi/', dataName,'.mat'], 'H','Y');
    CluNum = length(unique(Y));  % num of classes
    ker_num = size(H,3); %num of views
    sample_num = size(H,2);

    f_num = CluNum * 2;
    opt.disp = 0;
     
    %%
    %%%%%%%%%%%%%%%%%%%  Proposed Method  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    p_num1 = CluNum*2;
    parameters = [CluNum, p_num1];
    
    accval9 = zeros(length(parameters),1);
    nmival9 = zeros(length(parameters),1);
    purval9 = zeros(length(parameters),1);
    ARIval9 = zeros(length(parameters),1);
    time = zeros(length(parameters),1);
    ObjValue = zeros(length(parameters),50);
    %% Try different hyper-parameters
    for m_i = 1
        
            tic
            % m -- number of anchor points
            m = parameters(m_i);
            P = zeros(f_num,m,ker_num);
            %% Initialization
            % Initialization of Pi
            for p=1:ker_num
                P(:,:,p) = eye(f_num,m);
            end
            % Initialization of S
            S = zeros(m, sample_num);
            
            % Initialization of beta
            gamma = ones(ker_num,1)/(ker_num);
                      
            %% Optimization
            
            it_count = 1;
            flag = 1;
            while it_count < 50 && flag == 1
                fprintf('DataName : %s , m_i %d ,it_count %d \n', dataName, m_i, it_count);
                %% Update S
                W1 = zeros(m,sample_num);
                for i = 1 : ker_num
                    W1 = W1 + gamma(i) * P(:,:,i)' * H(:,:,i);
                end
                [U1, ~, V1] = svd(W1,'econ');
                S = U1 * V1';
                
                %% Update Pi              
                W2 = zeros(f_num, m);
                for i = 1 : ker_num
                    W2 = H(:,:,i)*S';
                    [U2, ~, V2] = svd(W2,'econ');
                    P(:,:,i) = U2 * V2';
                end       
                
               %% Update beta
                coef = zeros(1,ker_num);
                for p=1:ker_num
                    coef(1,p) = trace(H(:,:,p)*S'*P(:,:,p)'); 
                end  
                gamma = coef/norm(coef,2);  
                
                obj = 0;
                for i = 1 : ker_num
                    obj = obj + gamma(i)*trace(H(:,:,p)*S'*P(:,:,p)');
                end

                ObjValue(m_i,it_count) = obj;
                
                if it_count > 2 && (abs((ObjValue(m_i,it_count-1)...
                        -ObjValue(m_i,it_count))/ObjValue(m_i,it_count-1))<1e-6)
                    flag = 0;
                end
                it_count = it_count+1;
            end
            [~,~,V] = svd(S,'econ');
            H10 = V(:,1:CluNum);
            time(m_i) = toc;
            res9 = myNMIACCV5(S',Y,CluNum);
            accval9(m_i) = res9(1);
            nmival9(m_i)= res9(2);
            purval9(m_i) = res9(3);

    end
    [mi_best] = find(max(accval9)==accval9);
    mi_Best = max(mi_best);

    res = [max(accval9); max(nmival9);max(purval9);mi_Best];   
    save(['./Result/Proposed1/', dataName,'.mat'], 'res','time');

end
