function [xhat, vx, dMSE, dMSE_end] = Gamp_one( A, y, tau, pi, mu, sigvar, wvar, x )
% GAMP algorithm for one-bit compressed sensing under additive Gaussian noise

% Input:
% - A: measurement matrix (m x n)
% - y: sign measurements (+1 or -1) (m x 1)
% - tau: quantizer thresholds
% - pi, mu, sigvar, wvar: initialized prior nonzero probability, prior mean, prior variance, additive noise variance
% - x: the true signal

% Output:
% - xhat: reconstructed signal (n x 1)
% - vx: predicted MSE (n x 1)
% - dMSE: debiased MSE
% - dMSE_end: the final debiased MSE after all iterations

%% pamameter initilization
global lar_num dampFac  T counter sma_num tol2
ct = counter;
[m, n] = size(A);
xhat = zeros(n,1);
shat = zeros(m, 1);
vx = lar_num*ones(n, 1);

% Previous estimate
xhatprev = xhat;
shatprev = shat;
vxprev = vx;

% Hadamard product of the matrix
AA = A.*A;

% Perform estimation
computeMse = @(noise) 20*log10(norm(noise(:))/norm(x));
absdiff = @(zvar) sum(abs(zvar))/length(zvar);

dMSE = zeros(T,1);
for t = 1:T
    if t == 1
        dMSE(t) = computeMse(x);
    end    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Measurement update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear
    vp = AA*vx;
    phat = A*xhat - vp.*shat;    
    [ez, vz, count_temp] = GaussianMomentsComputation_warning(y, tau, phat, vp, wvar);
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
    
    % Non-Linear
    shat = (ez-phat)./vp;
    vs = (1-vz./vp)./vp;
    vs = max(sma_num, vs);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Estimation update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear
    vr = 1./(AA' * vs);
    rhat = xhat + vr .* (A' * shat);
    xhatprev0 = xhat;
    
    % Non-linear variable estimation
    M = 0.5*log(vr./(vr+sigvar))+0.5*rhat.^2./vr-0.5*(rhat-mu).^2./(vr+sigvar);
    lambda = pi./(pi+(1-pi).*exp(-M));
    m_t = (rhat.*sigvar+vr.*mu)./(vr+sigvar);
    V_t = vr.*sigvar./(vr+sigvar);    
    xhat = lambda.*m_t;
    vx = lambda.*(m_t.^2+V_t)-xhat.^2;
    
    %Damp
    xhat = dampFac*xhat + (1-dampFac)*xhatprev;
    shat = dampFac*shat + (1-dampFac)*shatprev;
    vx = dampFac*vx + (1-dampFac)*vxprev;
    
    % If without a change   
    if(absdiff(xhat - xhatprev0) < tol2)
        ct = ct - 1;
    else
        ct =counter;
    end
    
    % Save previous xhat
    xhatprev = xhat;
    shatprev = shat;
    vxprev = vx;
    
    % Stopping criterion
    if(ct <= 0)
        break;
    end
    
end
dMSE_end = dMSE(t);

end
