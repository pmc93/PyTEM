import json

PATH = "notebooks/9. pytem_profiler_forward.ipynb"


def lines(text):
    text = text.strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


DEFINITIONS_CODE = r'''LOG_RHO_STEP = 0.03      # Metropolis-Hastings proposal step size, log10(Ohm.m)
THICKNESS_STEP = 0.75    # Metropolis-Hastings proposal step size, m
N_LAYERS = 3
STATION_INDEX = 10

# Stack the valid LM/HM gates (and their noise) once, matching the RMS section above.
obs_list, sig_list, gate_list, moment_slices = [], [], [], {}
cursor = 0
for moment in MOMENTS:
    valid = valid_gates[moment]
    observed = observed_systems[moment]['obs_data'][valid]
    sigma = observed * np.maximum(tem.dbdt_std(moment)[STATION_INDEX, valid], 0.03)
    obs_list.append(observed)
    sig_list.append(sigma)
    gate_list.append(gate_matrices[moment][valid])
    moment_slices[moment] = slice(cursor, cursor + observed.size)
    cursor += observed.size

obs = np.concatenate(obs_list)     # all valid observed gates, LM then HM, stacked into one vector
sig = np.concatenate(sig_list)     # matching per-gate noise (standard deviation) for the chi-squared below
G = np.vstack(gate_list)           # matching gate matrices, so G @ step_response lines up with obs/sig

print(f'Fitting station {STATION_INDEX} ({selected_usf.name}) with {obs.size} valid LM/HM gates')

LOG_RHO_BOUNDS = np.log10(RHO_RANGE)
# theta = [log10(rho_1..rho_N_LAYERS), thickness_1..thickness_(N_LAYERS-1)]; the half-space has no thickness.
N_PARAMS = N_LAYERS + (N_LAYERS - 1)


def model(theta):
    """Unpack theta into layer resistivities/thicknesses and return the predicted gate values."""
    log_rho, thickness = theta[:N_LAYERS], theta[N_LAYERS:]
    step_response = -fwd_circle_offset(
        thickness, 10.0 ** log_rho, tx_radius, rx_offset, t_step,
        current=1.0, signal=-1, use_numba=True, use_cuda=False,
        transform=TRANSFORM, system_filter=receiver_filter,
    )
    return G @ step_response  # map the shared step response onto this station's LM/HM gate times


def lnlike(theta):
    """Gaussian log-likelihood: -0.5 * chi_squared, chi_squared = sum(((prediction - obs) / sig) ** 2)."""
    try:
        prediction = model(theta)
    except Exception:
        return -np.inf  # forward model failed (e.g. degenerate layer) -> reject this theta outright
    if not np.all(np.isfinite(prediction)):
        return -np.inf
    return -0.5 * np.sum(((prediction - obs) / sig) ** 2)


def lnprior(theta):
    """Flat (uniform) prior: 0.0 log-probability inside the ensemble's bounds, -inf outside."""
    log_rho, thickness = theta[:N_LAYERS], theta[N_LAYERS:]
    if (np.all((log_rho >= LOG_RHO_BOUNDS[0]) & (log_rho <= LOG_RHO_BOUNDS[1]))
            and np.all((thickness >= THICKNESS_RANGE[0]) & (thickness <= THICKNESS_RANGE[1]))):
        return 0.0
    return -np.inf


def lnprob(theta):
    """Un-normalised log-posterior: log(prior) + log(likelihood), the target the sampler climbs/explores."""
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf  # outside the prior bounds -> skip the (expensive) forward model entirely
    return lp + lnlike(theta)
'''

SAMPLER_CODE = r'''N_CHAINS = 6
N_BURN = 300
N_STEPS = 5000
MCMC_SEED = 43

# Per-parameter Gaussian proposal width: too large -> most proposals rejected (low acceptance),
# too small -> the chain crawls and needs far more steps to explore the posterior.
PROPOSAL_STEP = np.r_[np.full(N_LAYERS, LOG_RHO_STEP),
                     np.full(N_LAYERS - 1, THICKNESS_STEP)]
theta0 = np.r_[np.log10(resistivities[best_model_index]), thicknesses[best_model_index]]
assert theta0.size == N_PARAMS


def run_chain(theta, n_steps, rng):
    """Metropolis-Hastings random walk: propose a nearby theta, accept/reject, repeat."""
    lp = lnprob(theta)
    chain = np.empty((n_steps, theta.size))
    lnprob_chain = np.empty(n_steps)
    n_accept = 0
    for step in range(n_steps):
        proposal = theta + PROPOSAL_STEP * rng.standard_normal(theta.size)
        lp_proposal = lnprob(proposal)
        # Metropolis rule: always accept an uphill step; accept a downhill step with
        # probability exp(lp_proposal - lp), so the chain still explores lower-probability regions.
        if np.isfinite(lp_proposal) and np.log(rng.uniform()) < lp_proposal - lp:
            theta, lp = proposal, lp_proposal
            n_accept += 1
        chain[step], lnprob_chain[step] = theta, lp
    return chain, lnprob_chain, n_accept / n_steps


def main(theta_start, n_burn, n_steps, rng):
    """Discard an initial n_burn steps (letting the chain forget its start), then keep n_steps."""
    burn_chain, _, _ = run_chain(theta_start, n_burn, rng)
    return run_chain(burn_chain[-1], n_steps, rng)


def _fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.0f} s'
    if seconds < 3600:
        return f'{seconds / 60:.1f} min'
    return f'{seconds / 3600:.1f} h'


# Time a handful of lnprob calls to estimate the full run before committing to it.
_N_CALIB = 50
_calib_start = time.perf_counter()
for _ in range(_N_CALIB):
    lnprob(theta0)
_s_per_step = (time.perf_counter() - _calib_start) / _N_CALIB
_total_steps = N_CHAINS * (N_BURN + N_STEPS)
print(f'Calibration: {_s_per_step * 1000:.1f} ms/step -> '
      f'estimated total runtime {_fmt_time(_s_per_step * _total_steps)} '
      f'for {N_CHAINS} chains x {N_BURN + N_STEPS} steps')

rng_master = np.random.default_rng(MCMC_SEED)
chains, lnprob_chains, acceptance = [], [], []
started = time.perf_counter()
for chain_index in range(N_CHAINS):
    # Independent chains start at randomly jittered points near theta0, so the spread between
    # them (checked via R-hat below) is a diagnostic for whether they all found the same posterior.
    start = theta0 + PROPOSAL_STEP * rng_master.standard_normal(N_PARAMS)
    chain_rng = np.random.default_rng(rng_master.integers(1 << 32))
    chain, lnprob_chain, acc = main(start, N_BURN, N_STEPS, chain_rng)
    chains.append(chain)
    lnprob_chains.append(lnprob_chain)
    acceptance.append(acc)

    elapsed = time.perf_counter() - started
    remaining = elapsed / (chain_index + 1) * (N_CHAINS - chain_index - 1)
    print(f'Chain {chain_index + 1}/{N_CHAINS} done in {_fmt_time(elapsed)} '
          f'(acceptance {acc:.2f}); ETA {_fmt_time(remaining)}')

chains = np.array(chains)               # (n_chains, n_steps, n_params), post burn-in
lnprob_chains = np.array(lnprob_chains)  # (n_chains, n_steps)

print(f'{N_CHAINS} chains x {N_STEPS} production steps '
      f'({N_BURN} burn-in each, discarded) in {_fmt_time(time.perf_counter() - started)}')
print('Acceptance rate per chain:', np.round(acceptance, 2),
      '(aim for roughly 0.2-0.5)')
'''

REPLACEMENTS = {
    "1b0bd326": DEFINITIONS_CODE,
    "c65e8e7b": SAMPLER_CODE,
}

nb = json.load(open(PATH, encoding="utf-8"))
touched = []
for cell in nb["cells"]:
    cid = cell.get("id")
    if cid in REPLACEMENTS:
        cell["source"] = lines(REPLACEMENTS[cid])
        cell["outputs"] = []
        cell["execution_count"] = None
        touched.append(cid)

missing = set(REPLACEMENTS) - set(touched)
assert not missing, f"cell ids not found: {missing}"

with open(PATH, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")

print("touched:", touched)
