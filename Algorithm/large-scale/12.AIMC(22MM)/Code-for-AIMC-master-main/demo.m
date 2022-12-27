warning off;
addpath(genpath('./'));

%% dataset
ds = {'Handwritten_fea'};

dsPath = './0-dataset/';
resPath = './res-lmd0/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

for dsi = 1:length(ds)
    answer=[];
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    for i=1:6
        X{i}=X{i}';
    end
    gt = Y;
    k = length(unique(gt));
    
    
    matpath = strcat(resPath,dataName);
    txtpath = strcat(resPath,strcat(dataName,'.txt'));
    if (~exist(matpath,'file'))
        mkdir(matpath);
        addpath(genpath(matpath));
    end
    dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
    
    %% para setting
    d = k;
    
    %%
    for id = 1:length(d)
        tic;
        [G,F,Y,iter,obj,alpha] = algo_AI(X,gt,d(id)); % X,Y,d
        timer(id)  = toc;
        res = Clustering8Measure(gt,Y); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
        fprintf('Dimension:%d\t Res:%12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[d(id) res(1) res(2) res(3) res(4) timer(id)]);

        dlmwrite(txtpath, [d(id) res timer(id)],'-append','delimiter','\t','newline','pc');
        matname = ['_Dim_',num2str(d(id)),'.mat'];

        save([matpath,'/',matname],'G','F','Y','alpha');
    end
    clear resall objall X Y k
end


