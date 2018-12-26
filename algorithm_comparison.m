% This is the code for the following article:
Jiang Zhu, Lin Han and Xiangming Meng, An AMP-Based Low Complexity Generalized Sparse Bayesian Learning Algorithm, IEEE Access, 2018.
% dNMSE vs number of iterations under one bit quantization scenario
% GGAMP-SBL, AMP-Gr-SBL, GAMP, Gr-SBL
% written by Jiang Zhu, Lin Han and Xiangming Meng. If you have any problems, please feel free to contact me (jiangzhu16@zju.edu.cn)
% Date: Dec 30, 2018

clc;
clear;
close all; 

%% Parameters setup
n = 256;    % signal dimension
prior_pi = 0.1; % the prior nonzero probability
prior_mean = 0; % the prior mean of the signal x
prior_var = 1/0.1; % the prior variance of the signal x
Afro2 = n; % the normalized factor
SNR = 60; % signal to noise ratio
m = 4*n; %number of measurements

global T_E lar_num sma_num dampFac T   tol2 counter tol1 T_M 
dampFac = 1; % the damping ratio
lar_num = 1e6;
sma_num = 1e-6;
T = 2000;          % the maximum number of outer iterations of all algorithms
T_E = 20;        % the maximum number of the E step of the AMP-Gr-SBL algorithm
T_M = 1;         % the maximum number of the M step of the AMP-Gr-SBL algorithm
T_inner = T_M*T_E; % the maximum number of the inner iterations of the GGAMP-SBL algorithm
tol2 = 2*1e-4; % the exit criterion of the GAMP algorithm
tol1 = 3*1e-2; % the exit criterion of the Gr-SBL algorithm
counter = 1e3; % the times that the exit condition is consecutively met

tau = zeros(m,1); % the threshold value

%% Model generation
x = zeros(n,1);
supp = find(rand(n,1)<prior_pi);
K = length(supp);
x(supp) = prior_mean + sqrt(prior_var)*randn(K,1); % the true signal
A = randn(m, n); 
A = sqrt(Afro2/trace(A'*A))*A; % the true measurement matrix
z = A*x;
wvar = (z'*z)*10^(-SNR/10)/m;
w = sqrt(wvar)*randn(m,1);  % the true noise

%% Quantization
B = 1;  % bit_depth
y0 = z+w+tau; % the original measurement
y = sign(y0); % the quantized measurement

%% GGAMP-SBL
[temp1, ~,  ~, ~, ~] = GGAMP_SBL_one( A, y, tau, wvar, x, T_inner);

%% AMP-Gr-SBL
[temp2, ~, ~, ~, ~, ~] = Amp_GrSBL_one( A, y, tau, wvar, x);

%% Gr-SBL
[~,temp3, ~] = Grsbl_one(A, y, tau, wvar, x);

%% Gamp
[~, ~, temp4, ~] = Gamp_one( A, y, tau, prior_pi, prior_mean, prior_var, wvar, x);

%% Plot
lw = 1.8;
msz = 9;
fsz = 14;
set(0,'DefaultAxesColorOrder', [0 0 1;0 0.5 0.5;1 0 1;1 0 0;0 0 0  ]);
aa = 1:2:T;
figure(1)
plot(aa, temp1(aa), '-p', ...
      aa, temp2(aa), '-o', ...
      aa, temp3(aa),'-+',...
            aa, temp4(aa),'-v',...
       'LineWidth',lw, 'MarkerSize',msz);
set(gca, 'FontSize', fsz,'FontName','Times New Roman');
xlabel('number of iterations t');
ylabel('dNMSE (dB)');
legend('GGAMP-SBL', 'AMP-Gr-SBL','GAMP','Gr-SBL');


