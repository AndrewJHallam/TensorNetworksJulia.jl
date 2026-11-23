function [evolved,raw,corrected] = vlasov_matlab_reference(f0,x,v,dt,steps,levels,chi)
out = evolve_vlasov_poisson(f0,x,v,dt,steps,steps);
evolved = out.f{end};
raw = truncate_grid_via_mps(evolved,levels,chi);
corrected = truncate_to_manifold_vp(evolved,levels,chi,x,v,vlasov_mass(f0,x,v),vlasov_momentum(f0,x,v));
end
function out = evolve_vlasov_poisson(f0, x, v, dt, num_steps, store_every)
f = f0;
num_saved = floor(num_steps / store_every) + 1;
times = zeros(num_saved, 1);
f_history = cell(num_saved, 1);
save_idx = 1;
f_history{save_idx} = f;

for step = 1:num_steps
    f = advect_in_x(f, v, x(end) + x(2) - x(1), 0.5 * dt);
    electric_field = electric_field_from_distribution(f, x, v);
    f = advect_in_v(f, electric_field, v(end) - v(1) + (v(2) - v(1)), dt);
    f = advect_in_x(f, v, x(end) + x(2) - x(1), 0.5 * dt);

    if mod(step, store_every) == 0
        save_idx = save_idx + 1;
        times(save_idx) = step * dt;
        f_history{save_idx} = f;
    end
end

out = struct('times', times, 'f', {f_history});
end

function f_out = advect_in_x(f_in, velocities, period_x, dt)
f_out = zeros(size(f_in));
for vel_idx = 1:numel(velocities)
    f_out(:, vel_idx) = shift_periodic_fourier(f_in(:, vel_idx), velocities(vel_idx) * dt, period_x);
end
end

function f_out = advect_in_v(f_in, electric_field, period_v, dt)
f_out = zeros(size(f_in));
for x_idx = 1:numel(electric_field)
    shifted = shift_periodic_fourier(f_in(x_idx, :).', electric_field(x_idx) * dt, period_v);
    f_out(x_idx, :) = shifted.';
end
end

function shifted = shift_periodic_fourier(data, shift, period)
num_points = numel(data);
k = (2 * pi / period) * [0:(num_points / 2 - 1), 0, -(num_points / 2 - 1):-1].';
shifted = ifft(exp(-1i * k * shift) .* fft(data(:)));
shifted = real(shifted);
end

function electric_field = electric_field_from_distribution(f_grid, x, v)
dv = v(2) - v(1);
charge_density = dv * sum(f_grid, 2);
charge_density = charge_density - mean(charge_density);

num_x = numel(x);
domain_length_x = x(end) + x(2) - x(1);
k = (2 * pi / domain_length_x) * [0:(num_x / 2 - 1), 0, -(num_x / 2 - 1):-1].';
rho_hat = fft(charge_density);
e_hat = zeros(size(rho_hat));
nonzero = abs(k) > 0;
e_hat(nonzero) = -1i * rho_hat(nonzero) ./ k(nonzero);
electric_field = ifft(e_hat);
electric_field = real(electric_field);
end

function f_out = truncate_to_manifold_vp(f_in, levels, bond_dim, x, v, target_mass, target_momentum)
f_out = truncate_grid_via_mps(f_in, levels, bond_dim);
f_out = real(f_out);
f_out = max(f_out, 0);
f_out = enforce_mass_and_momentum(f_out, x, v, target_mass, target_momentum);
end

function f_out = enforce_mass_and_momentum(f_in, x, v, target_mass, target_momentum)
f_out = max(f_in, 0);
current_mass = vlasov_mass(f_out, x, v);
if current_mass > 0
    f_out = f_out * (target_mass / current_mass);
end

mean_velocity_target = target_momentum / max(target_mass, eps);
mean_velocity_current = vlasov_momentum(f_out, x, v) / max(vlasov_mass(f_out, x, v), eps);
velocity_shift = mean_velocity_target - mean_velocity_current;
period_v = v(end) - v(1) + (v(2) - v(1));
f_out = advect_in_v(f_out, velocity_shift * ones(numel(x), 1), period_v, 1.0);
f_out = max(f_out, 0);

current_mass = vlasov_mass(f_out, x, v);
if current_mass > 0
    f_out = f_out * (target_mass / current_mass);
end
end

function energy_value = vlasov_energy(f_grid, x, v)
[~, V] = ndgrid(x, v);
dx = x(2) - x(1);
dv = v(2) - v(1);
electric_field = electric_field_from_distribution(f_grid, x, v);
kinetic_energy = 0.5 * dx * dv * sum(sum(f_grid .* V .^ 2));
field_energy = 0.5 * dx * sum(electric_field .^ 2);
energy_value = real(kinetic_energy + field_energy);
end

function mass_value = vlasov_mass(f_grid, x, v)
dx = x(2) - x(1);
dv = v(2) - v(1);
mass_value = dx * dv * sum(f_grid(:));
end

function momentum_value = vlasov_momentum(f_grid, x, v)
[~, V] = ndgrid(x, v);
dx = x(2) - x(1);
dv = v(2) - v(1);
momentum_value = dx * dv * sum(sum(f_grid .* V));
end

function field_out = truncate_grid_via_mps(field_in, levels, dmax)
mps = field_to_mps(field_in, levels, dmax);
field_out = hierarchical_vector_to_grid(mps_to_dense(mps), levels);
end

function mps = field_to_mps(field_grid, levels, dmax)
physical_dim = 4;
field_vec_hier = grid_to_hierarchical_vector(field_grid);
mps = dense_to_mps(field_vec_hier, physical_dim, levels, dmax);
end

function mps = dense_to_mps(vec, d, levels, dmax)
vec = vec(:);
if numel(vec) ~= d ^ levels
    error('Dense input has %d entries, expected %d for d = %d and levels = %d.', ...
        numel(vec), d ^ levels, d, levels);
end

mps = cell(1, levels);
carry = vec;
left_rank = 1;

for site = 1:levels - 1
    carry = reshape(carry, [left_rank * d, numel(carry) / (left_rank * d)]);
    [u, s, v] = svd(carry, 'econ');
    next_rank = min([dmax, size(u, 2)]);
    u = u(:, 1:next_rank);
    s = s(1:next_rank, 1:next_rank);
    v = v(:, 1:next_rank);

    core = reshape(u, [left_rank, d, next_rank]);
    mps{site} = permute(core, [1, 3, 2]);

    carry = s * v';
    left_rank = next_rank;
end

core = reshape(carry, [left_rank, d, 1]);
mps{levels} = permute(core, [1, 3, 2]);
end

function vec = mps_to_dense(mps)
num_sites = numel(mps);
first_core = mps{1};
num_phys = size(first_core, 3);
right_rank = size(first_core, 2);
accumulator = zeros(num_phys, right_rank, 'like', first_core);

for phys = 1:num_phys
    accumulator(phys, :) = squeeze(first_core(1, :, phys));
end

for site = 2:num_sites
    core = mps{site};
    left_rank = size(core, 1);
    right_rank = size(core, 2);
    num_phys = size(core, 3);
    accumulator = reshape(accumulator, [], left_rank);
    next_accumulator = zeros(size(accumulator, 1) * num_phys, right_rank, 'like', core);

    for phys = 1:num_phys
        rows = (phys - 1) * size(accumulator, 1) + (1:size(accumulator, 1));
        next_accumulator(rows, :) = accumulator * squeeze(core(:, :, phys));
    end

    accumulator = next_accumulator;
end

vec = accumulator(:, 1);
end

function vec_hier = grid_to_hierarchical_vector(grid)
grid = grid(:);
levels = round(log2(sqrt(numel(grid))));
[~, lex_to_hier] = hierarchical_permutations(levels);
vec_hier = zeros(size(grid));
vec_hier(lex_to_hier) = grid;
end

function grid = hierarchical_vector_to_grid(vec_hier, levels)
nx = 2 ^ levels;
[hier_to_lex, ~] = hierarchical_permutations(levels);
grid_vec = zeros(nx ^ 2, 1);
grid_vec(hier_to_lex) = vec_hier(:);
grid = reshape(grid_vec, [nx, nx]);
end

function [hier_to_lex, lex_to_hier] = hierarchical_permutations(levels)
nx = 2 ^ levels;
num_points = nx ^ 2;
hier_to_lex = zeros(num_points, 1);
lex_to_hier = zeros(num_points, 1);

for y_idx = 0:nx - 1
    for x_idx = 0:nx - 1
        lex_index = x_idx + 1 + nx * y_idx;
        hier_index = 1;
        for level = 1:levels
            x_bit = bitget(x_idx, level);
            y_bit = bitget(y_idx, level);
            digit = x_bit + 2 * y_bit;
            hier_index = hier_index + digit * 4 ^ (level - 1);
        end
        hier_to_lex(hier_index) = lex_index;
        lex_to_hier(lex_index) = hier_index;
    end
end
end

