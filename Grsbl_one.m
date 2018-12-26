function [mu,NMSE_SBL, dMSE_end, outer_end] = Grsbl_one(A, y, tau, wvar, x)
% GrSBL algorithm for one-bit compressed sensing under additive Gaussian noise

% Input:
% - A: measurement matrix (m x n)
% - y: sign measurements (+1 or -1) (m x 1)
% - tau: quantizer thresholds
% - wvar: the true noise variance
% - x: the true signal

% Output:
% - mu: reconstructed signal (n x 1)
% - NMSE_SBL: debiased MSE
% - outer_end: the final number of outer iterations
% - dMSE_end: the final debiased MSE after all iterations

%% parameter initialization
global lar_num T sma_num  counter tol1
ct = counter;
[m,n] = size(A);
a = eps;
b = eps;
mu = zeros(size(x));
z_A_ext = zeros(m,1);
NMSE_SBL = zeros(1,T);
alpha = sma_num*ones(n,1);
v_A_ext = lar_num*ones(m,1);

%% function definition
absdiff = @(zvar) sum(abs(zvar))/length(zvar);
computeMse = @(noise) 20*log10(norm(noise(:))/norm(x));

for iter_sbl = 1:T
    if iter_sbl == 1
        NMSE_SBL(iter_sbl) = computeMse(x);
    end
    v_A_ext = lar_num.*(v_A_ext<0)+v_A_ext.*(v_A_ext>0);
    [z_B_post, v_B_post, count_temp] = GaussianMomentsComputation_warning(y, tau, z_A_ext, v_A_ext, wvar);
    while iter_sbl ~= 1
        if count_temp == 1
            mu = zeros(n,1);
            c0 = mu'*x/(mu'*mu + eps);
            NMSE_SBL(iter_sbl) = 20*log10(norm(c0*mu-x)/norm(x));
        elseif count_temp == 0
            c0 = mu'*x/(mu'*mu + eps);
            NMSE_SBL(iter_sbl) = 20*log10(norm(c0*mu-x)/norm(x));
        end
        break
    end
    v_B_ext = v_B_post.*v_A_ext./(v_A_ext-v_B_post);
    v_B_ext = min(v_B_ext,lar_num);
    v_B_ext = max(v_B_ext,sma_num);    
    z_B_ext = v_B_ext.*(z_B_post./v_B_post-z_A_ext./v_A_ext);    
    beta = 1./v_B_ext;    
    wvar0 = v_B_ext;
    y_tilde = z_B_ext;
    Sigma = inv((A'*diag(beta)*A)+diag(alpha) );
    mu0 = mu;
    mu = Sigma*A'*diag(beta)*y_tilde;
    alpha = (1+2*a)./(mu.*mu+diag(Sigma)+2*b);
    z_A_post = A*mu;
    v_A_post = (diag(A*Sigma*A') + eps);
    v_A_ext = v_A_post.*wvar0./(wvar0-v_A_post);
    z_A_ext = v_A_ext.*(z_A_post./v_A_post-y_tilde./wvar0);
    
    % If without a change
    if(absdiff(mu - mu0) < tol1)
        ct = ct - 1;
    else
        ct =counter;
    end
    
    % Stopping criterion
    if(ct <= 0)
        break;
    end
    
end
dMSE_end = min(NMSE_SBL);
outer_end = iter_sbl;

end

