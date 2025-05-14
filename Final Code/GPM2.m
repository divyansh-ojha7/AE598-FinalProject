%% Kelly_Fig24_CaptureCurve.m
% Recreates Figure 2.4 from Kelly Thesis (Normalized Capture Rate)

clear; clc; close all;

%% Parameters
alpha = 4;
rho_L_list = [1, 1e-1, 1e-2, 1e-3];
zeta_tot_range = logspace(0, 6, 300); % From 1 to 1e6

% Grid setup
r_vals = linspace(0, 15, 200);
z_vals = linspace(-6, 6, 200);
[Z, R] = meshgrid(z_vals, r_vals);

% Compute psi on grid
psi_vals = zeros(size(R));
for idx = 1:numel(R)
    psi_vals(idx) = psi_func(R(idx), Z(idx), 1, 1); % B0 = rc = 1
end
psi_safe = max(psi_vals, 1e-12);
psi_norm = psi_safe / max(psi_safe(:));

% For plotting
figure('Position', [100, 100, 800, 600]); hold on;
colors = lines(length(rho_L_list));

%% Main Loop: Numerical Calculation of N_cap
for i = 1:length(rho_L_list)
    rho_L = rho_L_list(i);
    psi_star = sqrt(2 * rho_L);
    N_cap_vals = zeros(size(zeta_tot_range));

    for k = 1:length(zeta_tot_range)
        zeta_tot = zeta_tot_range(k);

        % Calculate stream neutral density along z at fixed r (left to right integration)
        n_sn_hat = zeros(size(psi_norm));
        for row = 1:size(psi_norm,1)
            integrand_row = psi_norm(row,:).^alpha;
            In_row = cumtrapz(z_vals, integrand_row);
            n_sn_hat(row,:) = exp(-zeta_tot * In_row);
        end

        % Calculate integrand for I_sn with mask (psi > psi_star)
        mask = psi_safe > psi_star;
        psi_ratio_alpha = (psi_safe / psi_star).^alpha;
        integrand = n_sn_hat .* psi_ratio_alpha .* R; % include Jacobian r
        integrand(~mask) = 0;

        % Volume integration (cylindrical coordinates)
        dA = (r_vals(2)-r_vals(1)) * (z_vals(2)-z_vals(1));
        I_sn = 2 * pi * sum(integrand(:)) * dA;

        % Normalize capture rate (Eq. 2.14)
        N_cap_vals(k) = zeta_tot * I_sn;
    end

    % Plot numerical results
    semilogx(zeta_tot_range, N_cap_vals, '-', 'Color', colors(i,:), 'LineWidth', 2);
end

%% Approximate Analytical Curves (Eqs. 2.16–2.21)
for i = 1:length(rho_L_list)
    rho_L = rho_L_list(i);
    s0 = 0.66 * exp(-3.27 * rho_L^0.73);
    s1 = 26.81;
    s2 = 0.62 * rho_L^-1.2 + 4.29;
    s3 = 4.02 * rho_L^-1.3;
    s4 = 2.68 * rho_L^-1.6 + 5.36;

    I_approx = @(zeta) s0 ./ (1 + (zeta/s1).^(1/2) + (zeta/s2).^(3/2) + (zeta/s3).^3 + (zeta/s4).^5);
    N_approx = zeta_tot_range .* I_approx(zeta_tot_range);
    semilogx(zeta_tot_range, N_approx, '--', 'Color', colors(i,:), 'LineWidth', 1.5);
end

%% Plot Formatting
legend({'\rho_L = 1 (Num)','\rho_L = 0.1 (Num)','\rho_L = 0.01 (Num)','\rho_L = 0.001 (Num)', ...
        '\rho_L = 1 (Approx)','\rho_L = 0.1 (Approx)','\rho_L = 0.01 (Approx)','\rho_L = 0.001 (Approx)'}, ...
        'Location','northwest');
xlabel('$\zeta_{tot}$','Interpreter','latex');
ylabel('$\hat{N}_{cap}, \hat{P}_{cap}$','Interpreter','latex');
title('Normalized Capture Rate vs $\zeta_{tot}$ (Fig. 2.4)','Interpreter','latex');
grid on; ylim([1e-3 1e3]); set(gca, 'YScale', 'log');

%% ========== Supporting Function ==========
function psi = psi_func(r_cyl, z, B0, rc)
    k2 = (4 * r_cyl * rc) / ((r_cyl + rc)^2 + z^2);
    k2 = min(max(k2, 0), 1 - eps);
    [K, E] = ellipke(k2);
    psi = (2 * B0 * rc^2 * r_cyl * ((2 - k2) * K - 2 * E)) / (pi * k2 * sqrt(rc^2 + r_cyl^2 + z^2));
end