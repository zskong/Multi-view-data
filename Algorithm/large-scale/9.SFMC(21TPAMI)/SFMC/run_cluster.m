clear;clc;
addpath([pwd, '/funs']);
% addpath([pwd, '/datasets']);

ds = {'Caltech101-20','CCV','Caltech101-all_fea','SUNRGBD_fea','NUSWIDEOBJ','AwA_fea','YoutubeFace_sel_fea'...
'Caltech101-7','MNIST_fea'};
datapath = '/home/zpp/scholar/dataset/MultiView/';
% X:n*d
for di = 8:9
    dataname=ds{di};
    load(strcat(datapath,dataname,'.mat'));
    num_views = length(X);
    % for v = 1:num_views
    %     for  j = 1:n
    %         X{v}(j,:) = ( X{v}(j,:) - mean( X{v}(j,:) ) ) / std( X{v}(j,:) ) ;
    %     end
    % end
    for v = 1:num_views
        %     XX = X{v};
        %     for n = 1:size(XX,1)
        a = max(X{v}(:));
        X{v} = double(X{v}./a);
        %     XX(n,:)=XX(n,:)./norm(XX(n,:),'fro');
        %     end
        %     X{v} = double(XX);
    end
    anchor_rate=0.1:0.1:1;
    projev = 1.5;
    i = 1;
    c = length(unique(Y));
    opt1. style = 1;
    opt1. IterMax =50;
    opt1. toy = 0;
%     t1=clock;
    tic;
    [P1, alpha, y(:,i)] = FastmultiCLR(X, c, anchor_rate(5), opt1,10);
%     t2=clock;
    [result(i,:)] = Clustering8Measure(Y, y(:,i));
%     time(i) = etime(t2,t1);
    time(i) = toc;
    
    fprintf('Dataset:%s\t %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',dataname,result(1),result(2),result(3),result(4),time);
    fid = fopen('allresult.txt','a');
    fprintf(fid,'Dataset:%s\t %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',dataname,result(1),result(2),result(3),result(4),time);
    fclose(fid);
    clear X result time y
end
