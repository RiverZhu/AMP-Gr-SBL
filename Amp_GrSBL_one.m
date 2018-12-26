function [ dMSE, dMSE_end, outer_end, inner_E_end, inner_M_end, nmse_temp ] = Amp_GrSBL_one( A, y, tau, wvar, x)
% AMP-Gr-SBL algorithm for one-bit compressed sensing under additive Gaussian noise

% Input:
% - A: measurement matrix (m x n)
% - y: sign measurements (+1 or -1) (m x 1)
% - tau: quantizer thresholds
% - wvar: the noise variance
% - x: the true signal

% Output:
% - inner_E_end: the final number of E step
% - inner_M_end: the final number of M step
% - outer_end: the final number of outer iterations
% - nmse_temp: the NMSE of the outer iterations
% - dMSE_end: the final debiased MSE after all iterations
% - dMSE: debiased MSE

%% parameter initialization
global lar_num sma_num dampFac T  counter  T_E T_M
ct = counter;
[m, n] = size(A);
xhat = zeros(n,1);
dMSE = zeros(T,1);
shat = zeros(m, 1);
phat = zeros(m,1);
z_A_ext = zeros(m,1);
vx = lar_num*ones(n, 1);
inner_E_end = nan(T, 1);
vp = lar_num*ones(m,1);
inner_M_end = nan(T, T_M);
gamma = lar_num*ones(n,1);
v_A_ext = lar_num*ones(m,1);

% Hadamard product of the matrix
AA = A.*A;

% Previous estimate
xhatprev = xhat;
xhatprev2 = xhat;
nmse_temp = nan(T, 1);
shatprev = shat;

computeMse = @(noise) 20*log10(norm(noise(:))/norm(x));

for t = 1:T
    xhatprev1 = xhat;    
    if t == 1
        dMSE(t) = computeMse(x);
    end
    v_A_ext = lar_num*(v_A_ext<0)+v_A_ext.*(v_A_ext>0);
    v_A_ext = min(v_A_ext,lar_num);
    v_A_ext = max(v_A_ext,sma_num);
    [z_B_post, v_B_post, count_temp] = GaussianMomentsComputation_warning(y, tau, z_A_ext, v_A_ext, wvar);

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
    v_B_ext = v_B_post.*v_A_ext./(v_A_ext-v_B_post);  
    v_B_ext = lar_num*(v_B_ext<0)+v_B_ext.*(v_B_ext>0);
    v_B_ext = min(v_B_ext,lar_num);
    v_B_ext = max(v_B_ext,sma_num);
    z_B_ext = v_B_ext.*(z_B_post./v_B_post-z_A_ext./v_A_ext+eps);
    
    for inner = 1:T_M
        for ite = 1:T_E            
            % Truncated Gaussian
            ez = phat + vp ./ (vp+v_B_ext) .* (z_B_ext-phat);
            vz = vp.*v_B_ext./(vp+v_B_ext);
            
            % Non-Linear
            shat = (ez-phat)./vp;
            vs = (1-vz./vp)./vp;
            shat = dampFac*shat + (1-dampFac)*shatprev;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Estimation update
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Linear
            vr = 1./(AA' * vs);
            rhat = xhat + vr .* (A' * shat);
            xhatprev0 = xhat;
            
            % Non-linear variable estimation
            xhat = gamma./(gamma+vr).*rhat;
            vx = gamma.*vr./(gamma+vr);
            
            %Damp
            xhat = dampFac*xhat + (1-dampFac)*xhatprev;
            vp = AA*vx;
            vp = vp.*(vp>0)+lar_num.*(vp<=0);
            phat = A*xhat - vp.*shat;
            
            % Save previous xhat
            xhatprev = xhat;
            shatprev = shat;
            if norm(xhat-xhatprev0)/norm(xhat)<1e-4
                break
            end
        end
        inner_M_end(t, inner) = ite;
        gamma = xhat.^2+vx;        
        if norm(xhat-xhatprev2)/norm(xhat)<1e-4
            break
        end
        xhatprev2 = xhat;        
    end
    inner_E_end(t) = inner;
    z_A_ext = phat;
    v_A_ext = vp;
    
    % If without a change
    nmse_temp(t, 1) = norm(xhat-xhatprev1)/norm(xhat);    
    if norm(xhat-xhatprev1)/norm(xhat)<2*1e-4
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

