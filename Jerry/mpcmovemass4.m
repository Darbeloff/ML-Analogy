%% Nonlinear pendulum with damper
clear
clc
clearvars -global
close all
global cent
global A_x A_h H_x H_h J_x A_z B_z B_x H_u Jacobian
global a b c mu g l N_pend N_h N_x N_u u0 
% NOTE: for the below code the symbol h is generally used to denote eta
% DEFINE THE PARAMETER VALUES BELOW
load cent.mat
N_pend = 4;
c = ones(1,N_pend-1);
a = ones(1,N_pend-1);
b = ones(1,N_pend-1);
mu = 1;
g = -9.8;
l = 1;
Dt = 2;
Dt_sim = 0.01;
t_end = Dt;
imgcount = 1;
x0 = zeros([N_pend*2,1]);
u0_val = 5;
n_grid = 5;
STATE_NUM = 1;
AUG_NUM = 1;
N_missing = N_pend;
x_bound = [-1, 1];
u_bound = [-5,5];
% h_t = [e,e_C2,f_2,e_R];

%Create Matrices and fill in bounds
N_x  = 2*N_pend;
N_h  = 2*(N_pend-1); 
N_lamda = N_x + N_h - N_missing;
N_u  = 1;
x_bounds = zeros(2*N_pend,2);
u_bounds = zeros(N_u,2);
A_x = zeros(N_pend*2);
A_h = zeros(N_pend*2, N_h);
opts = odeset('AbsTol',1e-5,'RelTol',1e-5);
B_x = zeros(N_pend*2,1);
B_x(2) = 1;
for i = 1:N_x
    for j = 1:N_x
        if j==i+1 && mod(i,2)==1
            A_x(i,j) = 1;
        end
    end
end

for i = 1:N_x
    for j=1:N_h
        if mod(i,2) == 0
            if (j == i/2 && i~= N_x)
                %Spring
                A_h(i,j) = -1;
            elseif (j == N_pend + i/2 - 1)
                %Damper
                A_h(i,j) = -1;
            elseif (j == i/2 -1 && i~= 2)
                %Spring back
                A_h(i,j) = 1;
            elseif (j == N_pend + i/2 -2 && i~= 2)
                %Damper back
                A_h(i,j) = 1;
            end
        end
    end
end

%Define Jacobian
x1 = x0(1);
x1dot = x0(2);
x2 = x0(3);
x2dot = x0(4);
x3 = x0(5);
x3dot = x0(6);
x4 = x0(7);
x4dot = x0(8);
h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
ec1 = h0(1);
ec2 = h0(2);
ec3 = h0(3);
er1 = h0(4);
er2 = h0(5);
er3 = h0(6);
dx1dotdx1 = (-a(1)-3*b(1)*(x1 - x2)^2);
dx1dotdx2 = (a(1) + 3*b(1)*(x1 - x2)^2);
dx1dotdx1dot = -2*c(1)*(x1dot - x2dot)*sign(x1dot-x2dot);
dx1dotdx2dot = 2*c(1)*(x1dot-x2dot)*sign(x1dot-x2dot);

dx2dotdx1 = a(1) + 3*b(1)*(x1 - x2)^2;
dx2dotdx2 = -a(1) - 3*b(1)*(x1 - x2)^2 - a(2) - 3*b(2)*(x2 - x3)^2;
dx2dotdx3 = a(2) + 3*b(2)*(x2 - x3)^2;
dx2dotdx1dot = 2*c(1) * (x1dot - x2dot)*sign(x1dot - x2dot);
dx2dotdx2dot = -2*c(1)*(x1dot - x2dot)*sign(x1dot-x2dot) - 2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);
dx2dotdx3dot = 2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);

dx3dotdx2 = a(2) + 3*b(2)*(x2 - x3)^2;
dx3dotdx3 = -a(2) - 3*b(2)*(x2 - x3)^2 - a(3) - 3*b(3)*(x3 - x4)^2;
dx3dotdx4 = a(3) + 3*b(3)*(x3 - x4)^2;
dx3dotdx2dot = 2*c(2) * (x2dot - x3dot)*sign(x2dot - x3dot);
dx3dotdx3dot = -2*c(2)*(x2dot - x3dot)*sign(x2dot-x3dot) - 2*c(2)*(x3dot - x4dot)*sign(x3dot - x4dot);
dx3dotdx4dot = 2*c(3)*(x3dot - x4dot)*sign(x3dot - x4dot);

dx4dotdx3 = a(3) + 3*b(3)*(x3 - x4)^2;
dx4dotdx4 = -a(3) - 3*b(3)*(x3 - x4)^2;
dx4dotdx3dot = 2*c(3) * (x3dot - x4dot)*sign(x3dot - x4dot);
dx4dotdx4dot = -2*c(3)*(x3dot - x4dot)*sign(x3dot - x4dot);

dec1dx1 = 3*b(1)*2*(x1-x2)*(x1dot - x2dot);
dec1dx1dot = a(1) + 3*b(1)*(x1 - x2)^2;
dec1dx2 = -3*b(1)*2*(x1-x2)*(x1dot - x2dot);
dec1dx2dot = -a(1) - 3*b(1)*(x1 - x2)^2;

dec2dx2 = 2*3*b(2)*(x2-x3)*(x2dot-x3dot);
dec2dx2dot = a(2) + 3*b(2)*(x2 - x3)^2;
dec2dx3 = -2*3*b(2)*(x2 - x3) *(x2dot - x3dot);
dec2dx3dot = -a(2) - 3*b(2)*(x2 - x3)^2;

dec3dx3 = 2*3*b(3)*(x3 - x4)*(x3dot - x4dot);
dec3dx3dot = a(3) + 3*b(3)*(x3 - x4)^2;
dec3dx4 = -2*3*b(3)*(x3-x4)*(x3dot-x4dot);
dec3dx4dot = -a(3) - 3*b(3)*(x3 - x4)^2;

der1dx1dot = 2*c(1)*(-2*er1 - 2*ec1 + er2 + ec2)*sign(x1dot - x2dot);
der1dx2dot = -2*c(1)*(-2*er1 - 2*ec1 + er2 + ec2)*sign(x1dot - x2dot);
der1dec1 = 2*c(1)*(x1dot - x2dot)*(-2)*sign(x1dot - x2dot);
der1dec2 = 2*c(1)*(x1dot - x2dot)*sign(x1dot - x2dot);
der1der1 = 2*c(1)*(x1dot - x2dot)*(-2)*sign(x1dot-x2dot);
der1der2 = 2*c(1)*(x1dot - x2dot)*sign(x1dot - x2dot);

der2dx2dot = 2*c(2)*(er1 + ec1 - 2*er2 - 2*ec2 + er3 + ec3)*sign(x2dot - x3dot);
der2dx3dot = -2 * c(2) *(er1 + ec1 - 2*er2 - 2*ec2 + er3 + ec3)*sign(x2dot - x3dot);
der2dec1 = 2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);
der2dec2 = -2*2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);
der2dec3 = 2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);
der2der1 = 2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);
der2der2 = -2*2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);
der2der3 = 2*c(2)*(x2dot - x3dot)*sign(x2dot - x3dot);

der3dx3dot = 2*c(3)*(er2 + ec2 - 2*er3 - 2*ec3)*sign(x3dot - x4dot);
der3dx4dot = -2*c(3)*(er2 + ec2 - 2*er3 - 2*ec3)*sign(x3dot - x4dot);
der3dec2 = 2*c(3)*(x3dot - x4dot)*sign(x3dot - x4dot);
der3dec3 = -2*2*c(3)*(x3dot - x4dot)*sign(x3dot - x4dot);
der3der2 = 2*c(3)*(x3dot - x4dot)*sign(x3dot - x4dot);
der3der3 = -2*2*c(3)*(x3dot - x4dot)*sign(x3dot-x4dot);

Jacobian = [0, 1, 0, 0, 0, 0, 0, 0;
    dx1dotdx1, dx1dotdx1dot, dx1dotdx2,dx1dotdx2dot,0,0,0,0;
    0, 0, 0, 1, 0, 0, 0, 0;
    dx2dotdx1, dx2dotdx1dot, dx2dotdx2, dx2dotdx2dot, dx2dotdx3, dx2dotdx3dot, 0, 0;
    0, 0, 0, 0, 0, 1, 0, 0;
    0, 0, dx3dotdx2, dx3dotdx2dot, dx3dotdx3, dx3dotdx3dot, dx3dotdx4, dx3dotdx4dot;
    0, 0, 0, 0, 0, 0, 0, 1;
    0, 0, 0, 0, dx4dotdx3, dx4dotdx3dot, dx4dotdx4, dx4dotdx4dot];

bigJ = [A_x, A_h;
    dec1dx1, dec1dx1dot, dec1dx2, dec1dx2dot, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, dec2dx2, dec2dx2dot, dec2dx3, dec2dx3dot, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, dec3dx3, dec3dx3dot, dec3dx4, dec3dx4dot, 0, 0, 0, 0, 0, 0;
    0, der1dx1dot, 0, der1dx2dot, 0, 0, 0, 0, der1dec1, der1dec2, 0, der1der1, der1der2, 0;
    0, 0, 0, der2dx2dot, 0, der2dx3dot, 0, 0, der2dec1, der2dec2, der2dec3, der2der1, der2der2, der2der3;
    0, 0, 0, 0, 0, der3dx3dot, 0, der3dx4dot, 0, der3dec2, der3dec3, 0, der3der2, der3der3];
%%
parfor i = 1:length(N_x)
    x_bounds(i,:) = x_bound;
end
parfor i = 1:length(N_u)
    u_bounds(i,:) = u_bound;
end
% GENERATE A GRID OF DATA FOR RANGE OF x and u. Sampling Data
x_range = linspace(x_bound(1),x_bound(2),n_grid);
u_range = linspace(u_bound(1),u_bound(2),n_grid);
% Create grid
xgrid = combvec(x_range,u_range);
for i = 1:N_x
    xgrid = combvec(x_range,xgrid);
end
N_data = length(xgrid(1,:));
X = zeros(N_data, N_x);
H = zeros(N_data, N_h);
U = zeros(N_data, N_u);
X = xgrid(1:N_x,:)';
U = xgrid(end,:)';
N_data = length(X(:,1));
H = zeros(N_data, N_h);
koop = zeros(N_data,100);
parfor i= 1: N_data
    H(i,:) = find_eta(X(i,:),U(i,:), N_h, N_pend, g, l, a, b ,c); 
    koop(i,:) = find_lift(X(i,:)');
end

XHU = [X,H,U]';
XkoopU = [X,koop,U]';
Y = zeros(N_h,N_data);
Y2 = zeros(N_x,N_data);
X3 = zeros(N_x,N_data);
Y_lift = zeros(108,N_data);
parfor i = 1:N_data
Y(:,i)  = find_hdot(X(i,:),H(i,:),U(i,:),N_x,  N_h, g, l, a, b, c, N_pend);
Y2(:,i) = f_nonlinear(X(i,:),U(i,:), N_x, g, l, a, b , c);
end
%%
for i = 1:N_data
u0 = U(i,:);
[t,y3_unf] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],X(i,:),opts);
X3(:,i) = y3_unf(end,:);
end
%%
find_lift(X3(:,1));
%%
parfor i = 1:N_data
    Y_lift(:,i) = [X3(:,i);find_lift(X3 (:,i))];
end
%%
J_states  = (Y2*X)*inv(X'*X);

XI = [X,ones(N_data,1)];
%Pseudo inverse
J_0 = (H'*XI)*inv(XI'*XI);
J_x = J_0(:,1:N_x);
H_0 = (Y*XHU')*inv(XHU*XHU');

%Calculate regression variables
H_x = H_0(:,1:N_x);
H_h = H_0(:,N_x+1:N_x+N_h);
H_u = H_0(:,N_x+N_h+1:N_x+N_h+N_u);
mean_correction = mean(XHU,2);
X1_c = (XHU - mean_correction);%./std(X,0,2);
C   =  X1_c*X1_c';
[T,D,~]=eigs(C,N_lamda);

V = T(1:N_x,:);
W = T(N_x+1:N_x+N_h,:);
%PCA
A_z = (V'*A_x +W'*H_x)*V +(V'*A_h+W'*H_h)*W;
B_z = V'*B_x +W'*H_u;
xkoopu = [X,koop,U]';
xkoop = [X,koop]';
abkoop = Y_lift*xkoopu'*inv(xkoopu*xkoopu');
akoop = abkoop(:,1:108);
bkoop = abkoop(:,end);
ckoop = X'*xkoop'*inv(xkoop*xkoop');
%% Taylor Series Approx Comparison
randx0 = rand(1,8)*5;
randxf = rand(1,8)*5;
C = [eye(N_x)]; %Measuring Auxiliary Variables
%C = [eye(N_x),zeros(N_x,N_h);zeros(N_h,N_x+N_h)]; %Not Measuring Aux
D = zeros(8,1);
Ts = Dt_sim;
Duration = 2; %30 second duration
A = [Jacobian];
B = [B_x];
CSTR = ss(A,B,C,D);
CSTRmin = ss(CSTR,'minimal');
CSTR.InputName = {'Force'};
CSTR.OutputName = {'1','2','3','4','5','6','7','8'};
CSTR.InputGroup.MV = 1;
CSTR.OutputGroup.MO = 1;
jmpcobj = mpc(CSTR,Ts);
jmpcobj.PredictionHorizon = 20;
% jmpcobj.MV.Min = -300;
% jmpcobj.MV.Max = 300;
% jmpcobj.MV.RateMin = -3;
% jmpcobj.MV.RateMax = 3;
jmpcobj.Weights.ManipulatedVariables = 0.01;
jmpcobj.Weights.ManipulatedVariablesRate = 0;
jmpcobj.Weights.OutputVariables = [1 1 1 1 1 1 1 1];
% jmpcobj.Weights.OutputVariables = {Q};
% jmpcobj.Weights.ManipulatedVariables = {R};
%jmpcobj.Model.Noise = zeros(8,1);
%jmpcobj.Model.Disturbance = 0;
jmpcobj.ControlHorizon = 20;
T = 20; %20 steps
Num_iter = floor(Duration/(Ts)); 
x0 = randx0;
xref = randxf;
href = find_eta(xref, u0, N_h, N_pend, g, l, a, b, c);
xhref = [xref';href'];
Yj = [x0];
Tj = [0];
Uj = [0];
Xj = [x0];
r = [x0';
    xref'];
state = mpcstate(jmpcobj);
state.Plant = x0;
state.LastMove = 0;
options = mpcmoveopt;
u = mpcmove(jmpcobj, state, x0, xref, [], options);
for i = 1:Num_iter
    u0 = u;
    [~,x_nonlinear] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    x_sim = x_nonlinear(3,:);
    x0 = x_sim(end,:);
    h_sim = find_eta(x_sim(end,:), u0, N_h, N_pend, g, l, a, b, c);
    xh0 = [x_sim(end,:) h_sim];
    Xj = [Xj; x0];
    Yj = [Yj; x0];
    Tj = [Tj; .01+Tj(end)];
    Uj = [Uj; u];
    state = mpcstate(jmpcobj);
    state.Plant = x0;
    state.LastMove = u;
    u = mpcmove(jmpcobj, state, x0, xref, [], options);
end
% inputfig = figure;
% stairs(Tmpc,Umpc);
% title('Input vs Time');
% saveas(inputfig, 'mpc/jac_umpcfig1.jpeg');
% 
% hold on;
% xmpcfig = figure;
% x1 = plot(Tmpc,Ympc(:,1));
% hold on;
% x2 = plot(Tmpc,Ympc(:,3));
% x3 = plot(Tmpc,Ympc(:,5));
% x4 = plot(Tmpc,Ympc(:,7));
% title('Position vs Time');
% leg = legend([x1 x2 x3 x4], 'x1','x2','x3','x4');
% saveas(xmpcfig, 'mpc/jac_xmpcfig1.jpeg');
% 
% xdotmpc = figure;
% x1dot = plot(Tmpc,Ympc(:,2));
% hold on;
% x2dot = plot(Tmpc,Ympc(:,4));
% x3dot = plot(Tmpc,Ympc(:,6));
% x4dot = plot(Tmpc,Ympc(:,8));
% title('Velocity vs Time');
% leg = legend([x1dot x2dot x3dot x4dot], 'x1dot','x2dot','x3dot','x4dot');
% saveas(xdotmpc, 'mpc/jac_xdotmpcfig1.jpeg');


%DFL MPC 
C = [eye(N_x),zeros(N_x,N_h);zeros(N_h,N_x) eye(N_h)]; %Measuring Auxiliary Variables
%C = [eye(N_x),zeros(N_x,N_h);zeros(N_h,N_x+N_h)]; %Not Measuring Aux
D = zeros(14,1);
Ts = Dt_sim;
Duration = 2; %30 second duration
A = [A_x,A_h;H_x, H_h];
B = [B_x;H_u];
CSTR = ss(A,B,C,D);
CSTRmin = ss(CSTR,'minimal');
CSTR.InputName = {'Force'};
CSTR.OutputName = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14'};
CSTR.InputGroup.MV = 1;
CSTR.OutputGroup.MO = 1;
dflmpcobj = mpc(CSTR,Ts);
dflmpcobj.PredictionHorizon = 20;
% dflmpcobj.MV.Min = -300;
% dflmpcobj.MV.Max = 300;
% dflmpcobj.MV.RateMin = -3;
% dflmpcobj.MV.RateMax = 3;
dflmpcobj.W.ManipulatedVariables = 0.01;
dflmpcobj.W.ManipulatedVariablesRate = 0;
dflmpcobj.W.OutputVariables = [ones(1,8) zeros(1,6)];%1 1 1 1 1 1];
% dflmpcobj.Weights.OutputVariables = {eye(14)};
% dflmpcobj.Weights.OutputVariables = {[eye(N_x) zeros(N_x,N_h); zeros(N_h,N_x) eye(N_h).*100]};
% dflmpcobj.Weights.OutputVariables = {[eye(N_x) zeros(N_x,N_h); zeros(N_h,N_x+N_h)]};
%dflmpcobj.Weights.ManipulatedVariables = {R};
dflmpcobj.Model.Noise = zeros(14,1);
% dflmpcobj.ControlHorizon = 1;
T = 20; %20 steps
Num_iter = floor(Duration/(Ts)); 
x0 = randx0;
h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
xh0 = [x0 h0];
xref = randxf;
href = find_eta(xref, u0, N_h, N_pend, g, l, a, b, c);
xhref = [xref';href'];
Ympc = [xh0];
Tmpc = [0];
Umpc = [0];
Xmpc = [xh0];
state = mpcstate(dflmpcobj);
state.Plant = xh0;
state.LastMove = 0;
options = mpcmoveopt;
u = mpcmove(dflmpcobj, state, xh0, xhref, [], options);
error = (x0-xref')*(x0-xref')'/(xref'*xref);
errorvec = [error];
derror = error;
riseflag = true;
i = 0;
for i = 1:Num_iter
    i = i + 1;
    u0 = u;
    [~,x_nonlinear] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    x_sim = x_nonlinear(3,:);
    x0 = x_sim(end,:);
    h_sim = find_eta(x_sim(end,:), u0, N_h, N_pend, g, l, a, b, c);
    xh0 = [x_sim(end,:) h_sim];
    Xmpc = [Xmpc; xh0];
    Tmpc = [Tmpc; .01+Tmpc(end)];
    Umpc = [Umpc; u];
    state = mpcstate(dflmpcobj);
    state.Plant = xh0;
    state.LastMove = u;
    u = mpcmove(dflmpcobj, state, xh0, xhref, [], options);
    pasterror = error;
    error = (x0-xref')*(x0-xref')'/(xref'*xref);
    errorvec = [errorvec; error];
%     if(i > 30)
%         derror = sum(errorvec(end-30:end));
%     end
%     if(error < 0.1 && riseflag == true)
%         riseflag = false;
%         risetime = i;
%     end
end
%     settletime = i;
%     inputfig = figure;
%     stairs(Tmpc,Umpc);
%     title('Input vs Time');
%     saveas(inputfig, 'mpc/umpcfig6.png');
% 
%     hold on;
%     xmpcfig = figure;
%     x1 = plot(Tmpc,Xmpc(:,1));
%     hold on;
%     x2 = plot(Tmpc,Xmpc(:,3));
%     x3 = plot(Tmpc,Xmpc(:,5));
%     x4 = plot(Tmpc,Xmpc(:,7));
%     title({'Position vs Time';'Linear MPC'});
%     leg = legend([x1 x2 x3 x4], 'x1','x2','x3','x4');
%     saveas(xmpcfig, 'mpc/xmpcfig6.png');
% 
%     xdotmpc = figure;
%     x1dot = plot(Tmpc,Xmpc(:,2));
%     hold on;
%     x2dot = plot(Tmpc,Xmpc(:,4));
%     x3dot = plot(Tmpc,Xmpc(:,6));
%     x4dot = plot(Tmpc,Xmpc(:,8));
%     title('Velocity vs Time');
%     leg = legend([x1dot x2dot x3dot x4dot], 'x1dot','x2dot','x3dot','x4dot');
%     saveas(xdotmpc, 'mpc/xdotmpcfig6.png');
%%
nlobj = nlmpc(N_x,N_x,N_u);
nlobj.Ts = Dt_sim;
nlobj.PredictionHorizon = dflmpcobj.PredictionHorizon;
nlobj.ControlHorizon = dflmpcobj.ControlHorizon;
nlobj.Model.StateFcn = "mass4fcn";
nlobj.Model.OutputFcn = "mass4outputfcn";
nlobj.Jacobian.StateFcn = "mass4jacobian";
nlobj.Model.NumberOfParameters = 1;
x0 = randx0;
h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
xh0 = [x0 h0];
sameMV = rmfield(dflmpcobj.MV,'Target');
nlobj.MV = sameMV;
nlobj.W.ManipulatedVariables = 0.01;
nlobj.W.ManipulatedVariablesRate = 0;
nlobj.Weights.OutputVariables = {[eye(N_x)]};%% zeros(N_x,N_h); zeros(N_h,N_x+N_h)]};

u_0 = 0;
validateFcns(nlobj,x0,u_0,[],{Ts});
nloptions = nlmpcmoveopt;
nloptions.Parameters = {Ts};
nloptions.mv0 = [0];
nloptions.x0 = [x0];
xHistory = x0;
tHistory = [0];
uHistory = [0];
mv = 0;
xref = randxf;
href = find_eta(xref, u0, N_h, N_pend, g, l, a, b, c);
xhref = [xref';href'];
error = (x0-xref')*(x0-xref')'/(xref'*xref);
errorvec = [error];
derror = error;
riseflag = true;
i = 0;
for i = 1:Num_iter
    [mv, nloptions] = nlmpcmove(nlobj,x0, mv, xref, [], nloptions);
    u0 = mv;
    [t_nlmpc,x_nlmpc] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    x0 = x_nlmpc(end,:);
    h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
    xh0 = [x0 h0];
    xHistory = [xHistory; x0];
    tHistory = [tHistory t_nlmpc(end,:)+tHistory(end)];
    uHistory = [uHistory mv];
    nloptions.mv0 = [mv];
    nloptions.x0 = [x0];
    pasterror = error;
    error = (x0-xref')*(x0-xref')'/(xref'*xref);
    errorvec = [errorvec; error];

end
nlsettletime = i;

%     nlx = figure;
%     x1 = plot(tHistory, xHistory(:,1));
%     hold on;
%     x2 = plot(tHistory, xHistory(:,3));
%     x3 = plot(tHistory, xHistory(:,5));
%     x4 = plot(tHistory, xHistory(:,7));
%     title({'Position vs Time';'Nonlinear MPC'});
%     leg = legend([x1 x2 x3 x4], 'x1','x2','x3','x4');
%     saveas(nlx, 'mpc/xnlmpcfig8.png');
% 
%     nlxdot = figure;
%     x1dot = plot(tHistory, xHistory(:,2));
%     hold on;
%     x2dot = plot(tHistory, xHistory(:,4));
%     x3dot = plot(tHistory, xHistory(:,6));
%     x4dot = plot(tHistory, xHistory(:,8));
%     title('Velocity vs Time');
%     leg = legend([x1dot x2dot x3dot x4dot], 'x1dot','x2dot','x3dot','x4dot');
%     saveas(nlxdot, 'mpc/xdotnlmpcfig8.png');
% 
%     nlu = figure;
%     stairs(tHistory, uHistory);
%     title('Input vs Time');
%     saveas(nlu, 'mpc/unlmpcfig8.png');

parfor i=1:length(xHistory)
    xdiffSquare(i) = (xHistory(i,:) - Xmpc(i,1:8))*(xHistory(i,:) - Xmpc(i,1:8))'./(xHistory(i,:)*xHistory(i,:)');
    udiff(i) = (uHistory(i) - Umpc(i))^2;
    ydiffSquare(i) = (xHistory(i,:) - Xj(i,:))*(xHistory(i,:) - Xj(i,:))'./(xHistory(i,:)*xHistory(i,:)');
end
%     errorfig = figure;
%     xdiff = plot(tHistory, xdiffSquare);
%     xlabel('Time[s]');
%     ylabel({'Squared State Error'; 'Normalized with nonlinear MPC State Values'});
%     title('Normalized Squared State Error vs Time');
%     saveas(errorfig, 'mpc/errorfig.png');
data = [tHistory' xdiffSquare' ydiffSquare'];
csvwrite('taylor_dfl_compare.csv',data);
%% Compare Time Horizons

cycles_tested = 1:30;
%Set up initial conditions and references
x0 = [0;0;0;0;0;0;0;0];
h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
xh0 = [x0;h0'];
xref = rand(8,1)*10-5;
href = find_eta(xref, u0, N_h, N_pend, g, l, a, b, c);
xhref = [xref;href'];
rdfl = [xh0';
    xhref'];
rj = [x0';
    xref'];
%Set up controller options
nloptions = nlmpcmoveopt;
nloptions.Parameters = {Ts};
nloptions.mv0 = [0];
nloptions.x0 = [x0'];
dflMPCopts = mpcmoveopt;
jMPCopts = mpcmoveopt;

xJ = [];
uJ = [];
xDFL = [];
uDFL = [];
xNL = [];
uNL = [];

for i = cycles_tested
    mvnl = 0;
    mvdfl = 0;
    mvj = 0;
    dflmpcobj.PredictionHorizon = i;
    jmpcobj.PredictionHorizon = i;
    nlobj.PredictionHorizon = i;
    dflmpcobj.ControlHorizon = i;
    jmpcobj.ControlHorizon = i;
    nlobj.ControlHorizon = i;
    
    %Jacobian
    state = mpcstate(jmpcobj);
    state.Plant = x0;
    state.LastMove = 0;
    u = mpcmove(jmpcobj, state, x0, xref, [], jMPCopts);
    uJ(i) = u;
    u0 = u;
    [~, xj] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    xJ(i,:) = xj(end,:);
    
    %DFL
    state = mpcstate(dflmpcobj);
    state.Plant = xh0;
    state.LastMove = 0;
    u = mpcmove(dflmpcobj, state, xh0, xhref, [], dflMPCopts);
    uDFL(i) = u;
    u0 = u;
    [~, xdfl] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    xDFL(i,:) = xdfl(end,:); 

    %Nonlinear
    [mvnl, nloptions] = nlmpcmove(nlobj, x0, mvnl, xhref', [], nloptions);
    u0 = mvnl;
    [t_nlmpc,x_nlmpc] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    xNL(i,:) = x_nlmpc(end,:);
    uNL(i) = mvnl;
    
end
controlfig = figure;
uNLplot = plot(cycles_tested,uNL, 'g');
hold on;
uJplot = plot(cycles_tested,uJ, 'b');
uDFLplot = plot(cycles_tested,uDFL, 'r');
leg = legend([uNLplot uJplot uDFLplot], 'Nonlinear','Taylor Approx.', 'DFL');

xdiffSquareJ = [];
xdiffSquareDFL = [];
udiffDFL = [];
udiffJ = [];
%Calculate State Differences and Control Input Differences
parfor i=cycles_tested
    xdiffSquareJ(i) = (xNL(i,:) - xJ(i,:))*(xNL(i,:) - xJ(i,:))';
    xdiffSquareDFL(i) = (xNL(i,:) - xDFL(i,:))*(xNL(i,:) - xDFL(i,:))';
    udiffDFL(i) = (uNL(i) - uDFL(i))^2;
    udiffJ(i) = (uNL(i) - uJ(i))^2;
end
xdiffFig = figure;
xdiffJplot = plot(cycles_tested, xdiffSquareJ, 'r');
hold on;
xdiffDFLplot = plot(cycles_tested, xdiffSquareDFL, 'b');
leg = legend([xdiffJplot xdiffDFLplot], 'Taylor Approx.','DFL');
title('State Difference Squared vs Time Steps for Prediction');
saveas(xdiffFig, 'mpc/xdiffFig1.jpeg');

udiffFig = figure;
udiffDFLplot = plot(cycles_tested, udiffDFL, 'b');
hold on;
udiffJplot = plot(cycles_tested, udiffJ, 'r');
leg = legend([udiffJplot udiffDFLplot], 'Taylor Approx.','DFL');
title('Control Input Difference Squared vs Time Steps for Prediction');
saveas(udiffFig, 'mpc/udiffFig1.jpeg');

%% Time to reach rise point
thresh_percent = 0.10; % percent of error
phorizon = 20;
chorizon = phorizon;
%Set up initial conditions and references
x0z = [0;0;0;0;0;0;0;0];
x0 = x0z;
h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
xh0 = [x0;h0'];
xref = rand(8,1)*10;%[1;1;1;1;1;1;1;1];
href = find_eta(xref, u0, N_h, N_pend, g, l, a, b, c);
thresh = (x0 - xref)'*(x0-xref)*thresh_percent;
xhref = [xref;href'];
rdfl = [xh0';
    xhref'];
rj = [x0';
    xref'];
%Set up controller options
nloptions = nlmpcmoveopt;
nloptions.Parameters = {Ts};
nloptions.mv0 = [0];
nloptions.x0 = [x0'];
dflMPCopts = mpcmoveopt;
state = mpcstate(dflmpcobj);
state.Plant = xh0;
state.LastMove = 0;
jMPCopts = mpcmoveopt;
state = mpcstate(jmpcobj);
state.Plant = x0;
state.LastMove = 0;
dflmpcobj.PredictionHorizon = phorizon;
jmpcobj.PredictionHorizon = phorizon;
nlobj.PredictionHorizon = phorizon;
dflmpcobj.ControlHorizon = chorizon;
jmpcobj.ControlHorizon = chorizon;
nlobj.ControlHorizon = chorizon;
xJ = [];
uJ = [];
xDFL = [];
uDFL = [];
xNL = [];
uNL = [];

errorJ = (x0 - xref)'*(x0-xref);
errorDFL = (x0 - xref)'*(x0 - xref);
errorNL = (x0 - xref)'*(x0 - xref);
tJ = 0;
tDFL = 0;
tNL = 0;

rj = [x0';
    xref'];
u = mpcmove(jmpcobj, state, x0, xref, [], options);

while (errorJ > thresh)
    tJ = tJ + 1;
    u0 = u;
    [~,x_nonlinear] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    x_sim = x_nonlinear(3,:);
    x0 = x_sim(end,:);
    h_sim = find_eta(x_sim(end,:), u0, N_h, N_pend, g, l, a, b, c);
    xh0 = [x_sim(end,:) h_sim];
    r = [x0;
        xref'];
    state = mpcstate(jmpcobj);
    state.Plant = x0;
    state.LastMove = u;
    u = mpcmove(jmpcobj, state, x0, xref, [], jMPCopts);
    errorJ = (x0 - xref)'*(x0-xref);
end
x0 = x0z;
h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
xh0 = [x0;h0']';
rdfl = [xh0;
    xhref'];
state = mpcstate(dflmpcobj);
state.Plant = xh0;
state.LastMove = 0;
u = mpcmove(dflmpcobj, state, xh0, xhref, [], dflMPCopts);

while (errorDFL > thresh)
    tDFL = tDFL + 1;
    u0 = u;
    [~,x_nonlinear] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    x_sim = x_nonlinear(3,:);
    x0 = x_sim;
    h_sim = find_eta(x_sim(end,:), u0, N_h, N_pend, g, l, a, b, c);
    xh0 = [x_sim(end,:) h_sim];
    rdfl = [xh0;
        xhref'];
    state = mpcstate(dflmpcobj);
    state.Plant = xh0;
    state.LastMove = u;
    u = mpcmove(dflmpcobj, state, xh0, xhref, [], dflMPCopts);
    errorDFL = (x0 - xref)'*(x0 - xref);
end
x0 = x0z;
h0 = find_eta(x0, u0, N_h, N_pend, g, l, a, b, c);
xh0 = [x0;h0'];
while(errorNL > thresh)
    tNL = tNL + 1;
    [mv, nloptions] = nlmpcmove(nlobj,x0, mv, xhref', [], nloptions);
    u0 = mv;
    [t_nlmpc,x_nlmpc] = ode45(@(t,x) sys_model(t,x),[0 0.005 0.01],x0);
    x0 = x_nlmpc(end,:);
    nloptions.mv0 = [mv];
    nloptions.x0 = [x0];
    errorNL = (x0 - xref)'*(x0 - xref);
end

settlingtimefig = figure;
times = [tNL, tDFL, tJ];
bar(times,'FaceColor',[0.5 0.5 0.5]);
Labels = {'Nonlinear','DFL','Taylor'};%,'Latent\newlineSpace'};
set(gca, 'XTick', 1:3, 'XTickLabel', Labels);
set(gca,'FontSize',12)
xlabel('Linearization Method','FontSize',12);
ylabel('Time to Settle','FontSize',12);
title('Rise Time for Different Methods');
saveas(settlingtimefig, 'mpc/risetime2.jpeg');

%% FUNCTIONS 
% Here all the functions which are called in the above code are defined


function dxdt = f_nonlinear(x,u, nx, g, l, a, b, c)
%Simulation of nonlinear system
%Does not use auxiliary variables
dxdt = zeros(size(x));
N_x = nx;
fk1 = -a(1)*(x(1) - x(3)) - b(1)*(x(1) - x(3))^3;
fd1 = -c(1)*(x(2) - x(4))^2 *sign(x(2) - x(4));
fk2 = -a(2)*(x(3) - x(5)) - b(2)*(x(3) - x(5))^3;
fd2 = -c(2)*(x(4) - x(6))^2 *sign(x(4) - x(6));
fk3 = -a(3)*(x(5) - x(7)) - b(3)*(x(5) - x(7))^3;
fd3 = -c(3)*(x(6) - x(8))^2 *sign(x(6) - x(8));
dxdt(1) = x(2);
dxdt(2) = fk1 + fd1 + u;
dxdt(3) = x(4);
dxdt(4) = - fk1 - fd1 + fk2 + fd2;
dxdt(5) = x(6);
dxdt(6) = - fk2 - fd2 + fk3 + fd3;
dxdt(7) = x(8);
dxdt(8) = - fk3 - fd3;

end

function h = find_eta(x, u, N_h, N_pend, g, l, a, b, c)
%Calculates eta based on state variables.
%x(1) is x, x(2) is xdot
h = zeros(1, N_h);
h(1) = a(1) * (x(1) - x(3)) + b(1)*(x(1) - x(3))^3;
h(2) = a(2) * (x(3) - x(5)) + b(2)*(x(3) - x(5))^3;
h(3) = a(3) * (x(5) - x(7)) + b(3)*(x(5) - x(7))^3;
h(4) = c(1)*(x(2) - x(4))^2*sign(x(2) - x(4));
h(5) = c(2)*(x(4) - x(6))^2*sign(x(4) - x(6));
h(6) = c(3)*(x(6) - x(8))^2*sign(x(6) - x(8));
end

%Same as f_nonlinear above, but has inclusion of t input for the use of
%ode45
function dxdt = sys_model(t,x)
global g l a b c N_x u0
dxdt = zeros(size(x));
u = u0;

dxdt = f_nonlinear(x, u, N_x, g, l, a, b, c);
end
function dxdt = sys_model_sample(t,x)
global g l a b c N_x u0
dxdt = zeros(size(x));
u = u0;
if(t < 1)
    u = u0;
elseif(t < 2)
    u = 0;
elseif(t < 3)
    u = 0;
end
end
%Augmented transition
function dX_augdt = sys_model_aug(t,X_aug)
global A_x A_h H_x H_h B_x H_u u0
dX_augdt = [A_x,A_h;H_x,H_h]*X_aug + [B_x;H_u]*u0;
end

%statistical linearization
function dXdt = sys_model_J(t,X)
global A_x A_h B_x J_x u0
dXdt = (A_x + A_h*J_x)*X + B_x*u0; %*u_e(t);
end
%Using Simple linearisation
function dZdt = sys_model_Lat(t,Z)
global A_z B_z u0
dZdt = A_z*Z + B_z*u0;
end
function dXdt = sys_model_taylor(t,X)
global Jacobian B_x u0
dXdt = Jacobian*X +B_x*u0;
end

function dXdt = sys_model_koop(t,X)
global u0 akoop bkoop
dXdt = akoop*X + bkoop*u0;
end

%Function that finds eta dot based on x and eta
function hdot= find_hdot(x,eta,u, nx, nh, g, l, a, b, c, N_pend)
hdot = zeros(size(eta));
xdot = zeros(size(x));
N_x = nx;
xdot = f_nonlinear(x,u, N_x, g, l, a, b, c);
N_h = nh;
hdot(1) =  a(1)*(x(2) - x(4)) + b(1)*3 *(x(1)- x(3))^2*(x(2) - x(4));
hdot(2) =  a(2)*(x(4) - x(6)) + b(2)*3 *(x(3) - x(5))^2*(x(4) - x(6));
hdot(3) =  a(3)*(x(6) - x(8)) + b(3)*3 *(x(5)- x(7))^2*(x(6) - x(8));
hdot(4) =  c(1)*2*(x(2)-x(4))*(xdot(2) - xdot(4))*sign(x(2)-x(4));
hdot(5) =  c(2)*2*(x(4) - x(6))*(xdot(4) - xdot(6))*sign(x(4) - x(6));
hdot(6) =  c(3)*2*(x(6) - x(8))*(xdot(6) - xdot(8))*sign(x(6) - x(8));
end

function lift = find_lift(x)
global cent
lift = zeros(100,1);
    for i = 1:100
        r_squared = (x'-cent(i,:))*(x-cent(i,:)');
        lift(i) = r_squared*log(sqrt(r_squared));
    end
end