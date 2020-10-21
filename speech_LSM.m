clear
close all

%% Setup

% Paramters
dt = 0.1e-6;
rng(5);           % ?? 

% Network
Gin = csvread('Ginr.csv');   
Nin = size(Gin,2);
Nres = size(Gin,1);
Gres = csvread('Gr.csv');
Gnet = [2*Gin, 1*Gres];
Nout = 2;                   %??

% Neuron model
Vth = 200e-3;
Imin = 3e-9;
Iat100kHz = 7e-9;
taubyC = Vth/Imin;
tau = 18e-6;%1/100e3/log(Iat100kHz/(Iat100kHz-Imin))
Cb = 270e-15;%tau/taubyC
Vrst = 0;
RefPer = 0.1e-6;
RPS = ceil(RefPer/dt);

% Synapse model
tau1 = 1e-6;
tau2 = 0.5e-6;
I0 = 4*Cb*Vth/10/(tau1-tau2);
ds = 0.1e-6;
DS = ceil(ds/dt);
ts = 0:dt:ds+6*tau1+tau2;
Iwave = I0*(exp(-(ts-ds)/tau1)-exp(-(ts-ds)/tau2));
Iwave(1:ceil(ds/dt)) = 0;

%% Input spike trains
load DATA.mat;

%% Reservoir

for insample = 1:numel(DATA)
    disp(insample)
    n_samples = length(DATA(insample).S);          %?
    in_spikes = DATA(insample).S;
    
    V = zeros(Nres,1);
    Ibuffer = zeros(Nin+Nres,length(Iwave));
    res_spikes = zeros(Nres,1);
    inres_spikes = zeros(Nin+Nres,1);
    Itotal = zeros(Nin+Nres,1);
    RP = zeros(Nres,1);
    Iin = zeros(Nres,1);
    
    Iin_all = zeros(Nres,n_samples);
    res_spikes_all = zeros(Nres,n_samples);
    Itotal_all = zeros(Nin+Nres,n_samples);
    
    for i = 1:n_samples
        inres_spikes = [in_spikes(:,i);res_spikes];
        Iin = Gnet*Itotal;
        Iin(Iin<0) = 0;
        Iin(Iin>10e-9) = 10e-9;
        
        res_spikes_all(:,i) = res_spikes;
        Itotal_all(:,i) = Itotal;
        Iin_all(:,i) = Iin;
        
        Ibuffer = circshift(Ibuffer,-1,2);
        Ibuffer(:,end) = 0;
        Ibuffer = Ibuffer + inres_spikes*Iwave;
        Itotal = Ibuffer(:,1);
        
        RP = RP-1; RP(RP<0) = 0;
        V = V*(1-dt/tau) + dt/Cb*Iin;
        V(RP>0) = 0;
        res_spikes = V>Vth; V(res_spikes) = Vrst; RP(res_spikes) = RPS;
    end  
    DATA(insample).RES = res_spikes_all;
end


boo = "All inputs passed through reservoir"
disp(boo)

[a,b] = find(DATA(1).RES);
figure(2);plot(b*dt,a,'.')
ylim([1-0.1,Nres+0.1])
[a,b] = find(DATA(1).S);
figure(1);plot(b*dt,a,'.')
ylim([1-0.1,Nin+0.1])
%% Training

rZ=[];Ro=[];          %??
for sample_i=1:numel(DATA)
    rZ(:,sample_i)=sum(DATA(sample_i).RES,2);     % no of time each res neuron spikes for each input - 36x100 
    Ro(DATA(sample_i).type+1,sample_i)=1;  % expectation of output for each input - 2 op classes, initialied in patern 0,1
end

Nfold = 5;
Nstride = numel(DATA)/Nfold;
Wlim = 8;             %??

for kfold = 1:Nfold
    disp(kfold)
    % Training
    testOn = (kfold-1)*Nstride+(1:Nstride);
    trainOn  = setdiff(1:numel(DATA),testOn); % contian input no of traning and testing
    
    ZT=rZ(:,trainOn)';RoT=Ro(:,trainOn)'; %no of time each neuron spikes, and labels for datapoints in training
    W=[];  
    for i =1:2
        W(:,i)=lsqlin(ZT,1000*RoT(:,i),[],[],[],[],-Wlim*ones(Nres,1),Wlim*ones(Nres,1));
    end
    W=W';     % weighgts of all reservoir neurons with each op neuron -- real weights 
%     W = 2*Wlim*rand(Nout,Nres)-Wlim;
    
    Mn = sparse(1:length(trainOn),1+[DATA(trainOn).type],1,length(trainOn),Nout); % confusion matrix finder;
    MTest = sparse(1:length(testOn),1+[DATA(testOn).type],1,length(testOn),Nout); % confusion matrix finder;
    %things except weight have only been initialised rn
    %---------------
    
    spikeSampleCountTest=zeros(Nout,numel(DATA));
    
    %Testing
    for insample = 1:numel(DATA)
        disp(insample)
        n_samples = length(DATA(insample).RES);
        in_spikes = DATA(insample).RES;
        
        V = zeros(Nout,1);
        Ibuffer = zeros(Nres,length(Iwave));
        out_spikes = zeros(Nout,1);
        Itotal = zeros(Nres,1);
        RP = zeros(Nout,1);
        Iin = zeros(Nout,1);
        
        Iin_all = zeros(Nout,n_samples);
        Itotal_all = zeros(Nres,n_samples);
        out_spikes_all = zeros(Nout,n_samples);

        for i = 1:n_samples
            Iin = W*Itotal;
            
            Iin_all(:,i) = W*Itotal;
            Itotal_all(:,i) = Itotal;
            out_spikes_all(:,i) = out_spikes;

            Ibuffer = circshift(Ibuffer,-1,2);
            Ibuffer(:,end) = 0;
            Ibuffer = Ibuffer + in_spikes(:,i)*Iwave;
            Itotal = Ibuffer(:,1);

            RP = RP-1; RP(RP<0) = 0;
            V = V*(1-dt/tau) + dt/Cb*Iin;
            V(RP>0) = 0;
            out_spikes = V>Vth; V(out_spikes) = Vrst; RP(out_spikes) = RPS;
        end  
        spikeSampleCountTest(:,insample) = sum(out_spikes_all,2);   %no of time each op neuron fires for each input
    end    
    [M,recognized] = max(spikeSampleCountTest);
    Y = spikeSampleCountTest./repmat(M,Nout,1);
    
    M  = blkdiag(Mn,MTest);
    Y = Y./repmat(max(Y),Nout,1); Y(Y~=1)=0;
    
    misClassifiedSamples = find(sum(Y,1)~=1);
    Y(:,misClassifiedSamples) = 0;
    CM = (Y*M); % confusion matrix
    
    accuracyTrain = 100*trace(CM(:,1:Nout))/length(trainOn);
    numCorrectTrain = trace(CM(:,1:Nout));
    accuracyTest = 100*trace(CM(:,Nout+(1:Nout)))/length(testOn);
    numCorrectTest = trace(CM(:,Nout+(1:Nout)));
    
    RESULT(kfold).accTest = accuracyTest;
    RESULT(kfold).accTrain = accuracyTrain;
    RESULT(kfold).numCorrectTrain = numCorrectTrain;
    RESULT(kfold).numCorrectTest = numCorrectTest;
end

for kfold = 1:Nfold
    s=sprintf('\r\n Accuracy : kFold(%i) Test %2.2f (%i/%i) Train:%2.2f (%i/%i) \t',kfold,RESULT(kfold).accTest,RESULT(kfold).numCorrectTest,length(testOn),RESULT(kfold).accTrain,RESULT(kfold).numCorrectTrain,length(trainOn));
    fprintf(s);
end