function [ dMSE, dMSE_end, outer_end, inner_end, nmse_temp] = GGAMP_SBL_one( A, y, tau, wvar, x, T_inner)
% GGAMP-SBL algorithm for one-bit compressed sensing under additive Gaussian noise

% Input:
% - A: measurement matrix (m x n)
% - y: sign measurements (+1 or -1) (m x 1)
% - tau: quantizer thresholds
% - wvar: the noise variance
% - x: the true signal
% - T_inner: number of inner iterations

% Output:
% - inner_end: the final number of inner iterations
% - outer_end: the final number of outer iterations
% - nmse_temp: the NMSE of the outer iterations
% - dMSE_end: the final debiased MSE after all iterations
% - dMSE: debiased MSE

%% parameter initialization
global lar_num dampFac T  counter
ct = counter;
[m, n] = size(A);
count_temp = 0;
xhat = zeros(n,1);
dMSE = zeros(T,1);
shat = zeros(m, 1);
max_val = lar_num;
nmse_temp = nan(T, 1);
inner_end = nan(T, 1);
vx = lar_num*ones(n, 1);
gamma = lar_num*ones(n,1);

% Hadamard product of the matrix
AA = A.*A;

computeMse = @(noise) 20*log10(norm(noise(:))/norm(x));

for t = 1:T
    if t == 1
        dMSE(t) = computeMse(x);
    end
    xhatprev1 = xhat;
    while t ~= 1
        if count_temp == 1
            xhat = zeros(n,1);
            c0 = xhat'*x/(xhat'*xhat + eps);
            dMSE(t) = computeMse(c0*xhat-x);
        elseif count_temp == 0
            c0 = xhat'*x/(xhat'*xhat + eps);
            dMSE(t) = computeMse(c0*xhat-x);
        end
        break
    end
    
    for inner = 1:T_inner
        vp = AA*vx;
        vp = vp.*(vp>0)+lar_num.*(vp<=0);
        phat = A*xhat - vp.*shat;
        
        % Save previous xhat
        xhatprev = xhat;
        shatprev = shat;
        [ez, vz, count_temp] = GaussianMomentsComputation_warning(y, tau, phat, vp, wvar);
        
        % Non-Linear
        shat = (ez-phat)./vp;
        vs = (1-vz./vp)./vp;
        shat = dampFac*shat + (1-dampFac)*shatprev;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Estimation update
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Linear
        vr = 1./(AA' * vs );
        rhat = xhat + vr .* (A' * shat);
        xhatprev0 = xhat;
        % Non-linear variable estimation
        xhat = gamma./(gamma+vr).*rhat;
        vx = gamma.*vr./(gamma+vr);
        
        %Damp
        xhat = dampFac*xhat + (1-dampFac)*xhatprev;
        
        if norm(xhat-xhatprev0)/norm(xhat)<1e-4
            break
        end
        
    end    
    inner_end(t) = inner;    
    gamma = xhat.^2+vx;
    vx = vx.*(vx>0)+max_val.*(vx<0);
    
    % If without a change
    nmse_temp(t, 1) = norm(xhat-xhatprev1)/norm(xhat);
    if norm(xhat-xhatprev1)/norm(xhat)< 2*1e-4
        ct = ct - 1;
    else
        ct = counter;
    end
    
    % Stopping criterion
    if(ct <= 0)
        break;
    end
end
dMSE_end = dMSE(t);
outer_end = t;

end
