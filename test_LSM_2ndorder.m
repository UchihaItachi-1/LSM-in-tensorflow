%% Setup

% Runtime
dt = 0.1e-6;
T = 1e-3;                
t = 0:dt:T;                %1 micros timesteps for a mili sec

% Network
Nin = 5;       %?   no of input neurons           
Nres = 10;     %?
rng(4);        %?
pinp = 0.2;    %?
pres = 0.5;    %? some sort of thershold on revervoir weights- why? - perhaps to make them 0and 1 like gin?(-1,1)
pinh = 0.2;    %?  decides what fraction of res neurons are inhibitory
fout = 4;      %? no of res neurons each input neuron is connected to
pinhi = 0.25;  %? perhaps each ip neuron has this much inhibitory connection and 1-this exhibitory

Gin = zeros(Nres, Nin);           %array of connections of input neuron to res neurons, will have 25perecr inhibitpry connections
for i = 1:Nin
    indices = randperm(10,fout);   % returns a row vector containing fout(4) unique integers selected randomly from 1 to 10.
    %indices = randperm(Nres,fout);   %mychange
    Gin(indices,i) = [-1*ones(nearest(fout*pinhi),1);ones(nearest(fout*(1-pinhi)),1)];    
    %for each column take at random 4 rows without repition and do something
    % something - an matrix of -1s followed by ones, perhaps inhibitory and exhibtory
end

Gres = rand(Nres,Nres);
Gres = double(Gres < pres);   % 0  if weight less than .5 or 1   
Gres(:,end-nearest(pinh*Nres)+1:end) = -1*Gres(:,end-nearest(pinh*Nres)+1:end); % make ome of the ones -1

% Neuron model
Vth = 200e-3;
Imin = 3e-9;
Iat100kHz = 7e-9;    %??
taubyC = Vth/Imin;
tau = 18e-6;%1/100e3/log(Iat100kHz/(Iat100kHz-Imin))
Cb = 270e-15;%tau/taubyC
Vrst = 0;
RefPer = 0.1e-6;
RPS = ceil(RefPer/dt);

% Synapse model
tau1 = 1e-6;
tau2 = 0.5e-6;
I0 = 30e-9;
ds = 0.1e-6;
DS = ceil(ds/dt);
ts = 0:dt:ds+6*tau1+tau2;
Iwave = I0*(exp(-(ts-ds)/tau1)-exp(-(ts-ds)/tau2));      %???
Iwave(1:ceil(ds/dt)) = 0;
figure(3);plot(ts,Iwave);


%% Input spike trains
in_spikes = zeros(Nin,length(t));   %1ms train of 1000, for each input 
v_spikes = in_spikes;
lambda = ones(Nin,1)*1/5e-6;     %1 for each inp neuron, very small
tmin = 5e-6;                   %
tpw = 2e-6;                    %
inRPS = ceil(tmin/dt);
inPWS = ceil(tpw/dt);
inRP = zeros(Nin,1);           %
inPW = zeros(Nin,1);

for i = 1:length(t)
    inRP = inRP-1; inRP(inRP<0) = 0;        %sub 1, if les than 0, =0 , array of Nin,1
    inPW = inPW-1; inPW(inPW<0) = 0;        %same
    in_spike = (lambda*dt > rand(Nin,1));   % 0 if the small value of lambda is less than some random value??
    in_spike(inRP>0) = 0; inRP(in_spike) = inRPS;  %for those index in rps for which inrp is initally less than 1 , set values = some fix no??
    v_spike = in_spike; v_spike(inPW>0) = 1; inPW(in_spike) = inPWS; % same but this time for inps etc
    in_spikes(:,i) = in_spike; %inspike is [0,1..] where its 0, inrp has a fixed val..??
    v_spikes(:,i) = v_spike;
end

ISIs = [];
for i = 1:Nin
    %boo = t(in_spikes(i,:) ~= 0) %mychnage
    ISI = diff(t(in_spikes(i,:) ~= 0));  %??? 1x99 double with ... numbers
    ISIs = [ISIs ISI];   %??
end

% Plot input raster
[a,b] = find(in_spikes);
figure(1);plot(b*dt,a,'.')
title('5 channels, either 0 or 1')
ylim([1-0.1, Nin+0.1])

% Plot ISI
figure(2);hist(ISI)
title('isi, array of..doubles')


%% Reservoir

memV = zeros(Nres,length(t));   %perhaps will store all potentials at all times 
V = zeros(Nres,1);
Ibuffer = zeros(Nin+Nres,length(Iwave));    % why i wave?
res_spikes = zeros(Nres,1);                
inres_spikes = zeros(Nin+Nres,1);        
res_spikes_all = zeros(Nres,length(t)); % contains all res spikes at all times
Itotal = zeros(Nin+Nres,1);
Itotal_all = zeros(Nin+Nres,length(t));
RP = zeros(Nres,1);                      %??
Iin_all = zeros(Nres,length(t));        % ? input current depending oninspkes above?? 
Iin = zeros(Nres,1);

Gnet = [Gin, Gres];

for i = 1:length(t)-1
    memV(:,i) = V;
    res_spikes_all(:,i) = res_spikes;
    inres_spikes = [in_spikes(:,i);res_spikes];
    Itotal_all(:,i) = Itotal;
    Iin_all(:,i) = Iin;

    Ibuffer = circshift(Ibuffer,-1,2);
    Ibuffer(:,end) = 0;
    Ibuffer = Ibuffer + repmat(inres_spikes,1,length(Iwave)).*Iwave;    % a matrix of spikes multiplired with iwave
    Itotal = Ibuffer(:,1);
    Iin = Gnet*Itotal;
    Iin(Iin<0) = 0;
    Iin(Iin>10e-9) = 10e-9;
        
    RP = RP-1; RP(RP<0) = 0;
    V = V*(1-dt/tau) + dt/Cb*Iin;
    V(RP>0) = 0;
    res_spikes = V>Vth; V(res_spikes) = Vrst; RP(res_spikes) = RPS;
end

% Plot response raster
[a,b] = find(res_spikes_all);
figure(3);plot(b*dt,a,'.')
ylim([1-0.1, Nres+0.1])
title('all res spikes')

% Plot membrane potential
figure(4);plot(t,memV(1,:))
title('membrane potential')
% Plot input current
figure(5);plot(t,Iin(1,:))
title("input current with time")
