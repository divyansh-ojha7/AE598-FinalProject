clear; clc; close all;

%% ========================= GLOBAL MODEL FROM KELLY THESIS ========================= %%
% Key equations: (2.7), (2.8), (2.9), (2.10), (2.11)

%% ---- Initial Parameters ---- %%
B0 = 1;      % Magnetic field strength (normlized)
rc = 1;      % Characteristic radius of the magnetic dipole (normlized)
mi = 1;      % Ion mass (normlized)
qi = 1;      % Ion charge (normlized)
u_inf = 1;   % Freestream neutral velocity (normlized)
alpha = 4;   % Power law index for electron density profile 
m_sn = 1;    % Neutral mass (normlized)
ne_r = 1;    % Reference electron density (normlized)
rho_L = 0.005; % Larmor radius (normlized)

%% ---- Single Particle Trajectory and Final Axial Velocity (Eq. 2.7) ---- %%
%(COARSE GRID for particle trajectory only)
r_coarse = 0:0.5:20;
z_coarse = -10:0.5:10;
[Z_coarse, R_coarse] = meshgrid(z_coarse, r_coarse);

tspan = [0 400];
final_axial_velocity = zeros(length(r_coarse), length(z_coarse));
psi_coarse = zeros(length(r_coarse), length(z_coarse));

for i = 1:length(r_coarse)
    for j = 1:length(z_coarse)
        r0 = [r_coarse(i), 0, z_coarse(j)];
        u0 = [0, 0, 1];
        y0 = [r0, u0];
        [~, y] = ode23(@(t, y) ion_dynamics(t, y, rho_L, B0, rc), tspan, y0);
        final_axial_velocity(i,j) = mean(y(end-10:end,6)); % Final axial ion velocity
        psi_coarse(i,j) = psi_func(r_coarse(i), z_coarse(j), B0, rc); % Magnetic flux function ψ(r,z)
    end
end

%% ---- Flux Surfaces and Mass Capture ---- %%
%(FINE GRID)
r_fine = linspace(0, 15, 300);
z_fine = linspace(-6, 6, 300);
[Z_fine, R_fine] = meshgrid(z_fine, r_fine);

% Compute magnetic flux function ψ(r,z) on FINE grid
psi_fine = zeros(size(R_fine));
for i = 1:numel(R_fine)
    psi_fine(i) = psi_func(R_fine(i), Z_fine(i), B0, rc);
end
psi_fine = reshape(psi_fine, size(R_fine));
psi_star = sqrt(2 * rho_L); % Critical flux surface ψ* (Eq. 2.8)

%% ---- Figure 2.1: Flux Surfaces and Control Volume ---- %%
figure('Position', [150, 100, 700, 600]); hold on;
contourf(Z_fine, R_fine, psi_fine, 60, 'LineColor', 'none');
colormap(hot); colorbar;
contour(Z_fine, R_fine, psi_fine, [psi_star psi_star], 'k-', 'LineWidth', 2);
plot(0, 1, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
arrow_r = linspace(1,9,9)'; arrow_z = -5.8 * ones(size(arrow_r));
quiver(arrow_z, arrow_r, ones(size(arrow_r)), zeros(size(arrow_r)), 0.5, 'w', 'LineWidth', 1.5);
text(-5.7, 9.5, 'n_\infty, u_\infty', 'Color', 'w', 'FontSize', 11);
text(0.3, 5, '\psi^*', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
text(1.2, 7, 'S^*', 'Color', 'k', 'FontSize', 12);
text(2.3, 6, 'V^*', 'Color', 'k', 'FontSize', 12);
xlabel('z/r_c'); ylabel('r/r_c');
title('Flux Surfaces and Control Volume (Fig. 2.1)');
axis([-6 6 0 10]); axis equal tight; grid on;

%% ---- Figure 2.2: Ion Deflection Heat Map (Final Axial Velocity) ---- %%
figure('Position',[100,100,700,600]);
imagesc(z_coarse, r_coarse, final_axial_velocity);
set(gca,'YDir','normal'); axis xy;
xlabel('z/r_c'); ylabel('r/r_c');
title('Ion Deflection Heat Map with Flux Surface (\psi^*)');
colorbar; caxis([-1 1]); colormap(jet); hold on;
contour(z_coarse, r_coarse, psi_coarse, [psi_star psi_star], 'w--', 'LineWidth', 1.5);
axis([-10 10 0 20]); grid on;

%% ---- Figure 2.3: Stream Neutral Density Distribution (Eq. 2.9, 2.10) ---- %%
zeta_tot_list = [1e2, 1e3, 1e4, 1e5];
rho_L_contours = [1e-3, 1e-2, 1e-1];

figure('Position', [100, 100, 1000, 800]);
psi_safe = max(psi_fine, 1e-12); % Avoid division by zero
psi_norm = psi_safe / max(psi_safe(:));

for idx = 1:4
    zeta_tot = zeta_tot_list(idx);

    % Integral I_n along z for each r (corrected to model flow from left)
       In = zeros(size(psi_norm));
    for i = 1:size(psi_norm,1) % Loop over r
        psi_profile = psi_norm(i,:); % fixed r, vary z
        integrand = (psi_profile).^alpha;
        In(i,:) = cumtrapz(z_fine, integrand); % along z
    end
    n_sn_hat = exp(-zeta_tot * In);


    subplot(2,2,idx);
    imagesc(z_fine, r_fine, n_sn_hat);
    set(gca,'YDir','normal'); axis xy;
    axis([-6 6 0 15]); caxis([0 1]); hold on;

    for rl = rho_L_contours
        psi_star_local = sqrt(2 * rl);
        contour(z_fine, r_fine, psi_safe, [psi_star_local psi_star_local], 'w', 'LineWidth', 1.5);
    end

    xlabel('$z/r_c$', 'Interpreter', 'latex');
    ylabel('$r/r_c$', 'Interpreter', 'latex');
    title(['$\zeta_{tot} = 10^{' num2str(log10(zeta_tot)) '}$'], 'Interpreter', 'latex');
    colormap(jet);
    colorbar;
end

sgtitle('Stream Neutral Density Distribution for Various $\zeta_{tot}$', 'Interpreter', 'latex');



%% ========================= Functions ========================= %%

% Ion Dynamics (Lorentz Force, Eq. 2.7)
function dydt = ion_dynamics(~, y, rho_L, B0, rc)
    r = y(1:3); u = y(4:6);
    B = magnetic_field_dipole(r, B0, rc);
    du_dt = (1/rho_L) * cross(u, B);
    dydt = [u(:); du_dt(:)];
end

% Magnetic Field from Dipole Configuration (B = ∇ψ × e_theta / r)
function B = magnetic_field_dipole(r, B0, rc)
    x = r(1); y = r(2); z = r(3);
    r_cyl = sqrt(x^2 + y^2) + 1e-8;
    theta = atan2(y, x);
    e_theta = [-sin(theta), cos(theta), 0];
    delta = 1e-3;
    dpsi_dr = (psi_func(r_cyl + delta, z, B0, rc) - psi_func(r_cyl - delta, z, B0, rc)) / (2 * delta);
    dpsi_dz = (psi_func(r_cyl, z + delta, B0, rc) - psi_func(r_cyl, z - delta, B0, rc)) / (2 * delta);
    grad_psi = [dpsi_dr * cos(theta), dpsi_dr * sin(theta), dpsi_dz];
    B = cross(e_theta, grad_psi) / r_cyl;
end

% Magnetic Scalar Potential ψ(r,z) (Eq. 2.8)
function psi = psi_func(r_cyl, z, B0, rc)
    k2 = (4 * r_cyl * rc) / ((r_cyl + rc)^2 + z^2);
    k2 = min(max(k2, 0), 1 - eps);
    [K, E] = ellipke(k2);
    psi = (2 * B0 * rc^2 * r_cyl * ((2 - k2) * K - 2 * E)) / (pi * k2 * sqrt(rc^2 + r_cyl^2 + z^2));
end

