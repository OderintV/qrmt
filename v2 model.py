"""
Azadpur Metro - Crowd Exceedance Model (v2)
Group 12 - QRMT

Model: a two-line stochastic queueing system simulated by Monte Carlo.
  * Arrivals   : Non-homogeneous Poisson process (NHPP), rate lambda(t) with
                 morning + evening peaks.
  * Demand risk: each simulated day gets a Gamma multiplier (mean 1) so arrivals
                 are overdispersed (Gamma-Poisson / negative binomial), matching
                 real day-to-day variation (weather, events, holidays).
  * Platform split: fraction p_yellow of arrivals go to Yellow Line, rest Pink.
  * Departures : trains arrive every `headway` min on each line; each train
                 absorbs a random spare capacity (Gamma). Delays occur at random
                 (prob p_delay, Exponential extra wait) -> train bunching/gaps.
  * Output     : P(max crowd in window > threshold), exceedance curves,
                 time-resolved probabilities, sensitivity analysis, theory check.

ALL PARAMETERS ARE PLACEHOLDERS - replace with real data (see PARAMS).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SEED = 2026
rng = np.random.default_rng(SEED)

# ---------------------------- PARAMETERS -----------------------------------
PARAMS = dict(
    start_hr=7, end_hr=21,          # simulate 07:00 - 21:00 (1-min steps)
    base=12, amp_m=44, mu_m=9.0, sd_m=0.9,      # arrival profile: base + morning bump
    amp_e=52, mu_e=18.5, sd_e=1.1,              # evening bump (per minute, hours)
    day_cv=0.12,                    # CV of day-level demand multiplier
    p_yellow=0.65,                  # share of arrivals using Yellow line
    headway_y=2.5, headway_p=4.0,   # min between trains
    spare_y=(130, 40), spare_p=(110, 35),   # (mean, sd) boardable passengers per train
    p_delay=0.06, delay_mean=5.0,   # prob a train is delayed, mean extra minutes
    threshold=800,                  # crowd limit (people in station)
)
WINDOWS = {"Morning peak 08-10": (8, 10), "Evening peak 17-20": (17, 20)}
N_SIMS = 20_000

def lam_profile(P):
    t = np.arange(P["start_hr"]*60, P["end_hr"]*60) / 60.0
    g = lambda a, m, s: a*np.exp(-0.5*((t-m)/s)**2)
    return t, P["base"] + g(P["amp_m"], P["mu_m"], P["sd_m"]) + g(P["amp_e"], P["mu_e"], P["sd_e"])

def gamma_ms(mean, sd, size):
    k = (mean/sd)**2; th = sd**2/mean
    return rng.gamma(k, th, size)

def simulate(P, n=N_SIMS):
    """Vectorised over simulations. Returns occupancy array (n, T)."""
    t, lam = lam_profile(P); T = len(t)
    day = gamma_ms(1.0, P["day_cv"], n)                # demand multiplier per day
    Ny = np.zeros(n); Np = np.zeros(n)
    next_y = rng.uniform(0, P["headway_y"], n); next_p = rng.uniform(0, P["headway_p"], n)
    occ = np.zeros((n, T))
    for i in range(T):
        a = rng.poisson(day*lam[i])                    # total arrivals this minute
        ay = rng.binomial(a, P["p_yellow"]); Ny += ay; Np += a-ay
        for line in ("y", "p"):
            nxt = next_y if line == "y" else next_p
            hw = P["headway_"+line]; m, s = P["spare_"+line]
            arr = nxt <= i
            if arr.any():
                cap = gamma_ms(m, s, n)
                N = Ny if line == "y" else Np
                N -= np.where(arr, np.minimum(N, cap), 0)
                delay = (rng.random(n) < P["p_delay"])*rng.exponential(P["delay_mean"], n)
                nxt[arr] = i + hw + delay[arr]
        occ[:, i] = Ny + Np
    return t, lam, occ

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h, c+h

def window_idx(t, w): return (t >= w[0]) & (t < w[1])

# ------------------------------- MAIN RUN ----------------------------------
P = PARAMS
t, lam, occ = simulate(P)
thr = P["threshold"]
rows = []
for name, w in WINDOWS.items():
    m = window_idx(t, w); pk = occ[:, m].max(1)
    k = (pk > thr).sum(); lo, hi = wilson(k, N_SIMS)
    tabove = (occ[:, m] > thr).sum(1)           # minutes above threshold
    rows.append(dict(window=name, P_exceed=k/N_SIMS, CI95_lo=lo, CI95_hi=hi,
                     mean_peak=pk.mean(), p95_peak=np.percentile(pk, 95),
                     mean_minutes_above=tabove.mean(),
                     cond_mean_minutes_above=tabove[tabove > 0].mean() if (tabove > 0).any() else 0))
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("results_main.csv", index=False)

# --------- Theory check: Poisson validity + fluid approximation -------------
mu_y = P["spare_y"][0]/P["headway_y"]; mu_p = P["spare_p"][0]/P["headway_p"]
mu = P["spare_y"][0]/(P["headway_y"]+P["p_delay"]*P["delay_mean"]) + P["spare_p"][0]/(P["headway_p"]+P["p_delay"]*P["delay_mean"])  # effective service incl. delays
fluid = np.zeros(len(t)); x = 0
for i in range(len(t)): x = max(0, x+lam[i]-mu); fluid[i] = x
print(f"\nEffective service rate approx {mu:.1f}/min; peak lambda {lam.max():.1f}/min")
# Overdispersion check at the morning peak minute
i9 = np.argmin(abs(t-9)); a = rng.poisson(gamma_ms(1, P["day_cv"], 200000)*lam[i9])
print(f"Arrivals at 09:00: mean={a.mean():.1f}, var={a.var():.1f}, dispersion index={a.var()/a.mean():.2f} "
      "(1.0 = pure Poisson; >1 = overdispersed)")

# ------------------------------- FIGURES -----------------------------------
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "font.size": 9})

# Fig1: arrival rate + fan chart of crowd
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].plot(t, lam, c="C0"); ax[0].set_ylabel("Mean arrivals / min  λ(t)"); ax[0].set_title("Arrival intensity (NHPP)")
for q, a_ in [(5, .15), (25, .25)]:
    ax[1].fill_between(t, np.percentile(occ, q, 0), np.percentile(occ, 100-q, 0), color="C1", alpha=a_, label=f"{q}-{100-q}th pct")
ax[1].plot(t, occ.mean(0), c="C3", label="Mean"); ax[1].plot(t, fluid, "k--", lw=1, label="Fluid approx (deterministic)")
ax[1].axhline(thr, c="r", ls=":", label=f"Threshold = {thr}")
ax[1].set_ylabel("People in station"); ax[1].set_xlabel("Hour of day"); ax[1].legend(ncol=3)
ax[1].set_title("Simulated crowd: fan chart"); plt.tight_layout(); plt.savefig("fig1_fan_chart.png"); plt.close()

# Fig2: time-resolved exceedance probability
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(t, (occ > thr).mean(0), c="C3"); ax.set_ylabel("P( N(t) > threshold )"); ax.set_xlabel("Hour of day")
for w in WINDOWS.values(): ax.axvspan(*w, color="gray", alpha=.12)
ax.set_title("Probability of exceeding threshold at each minute (shaded = peak windows)")
plt.tight_layout(); plt.savefig("fig2_prob_over_time.png"); plt.close()

# Fig3: exceedance curve P(max > x) by window + histogram
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
xs = np.linspace(0, occ.max()*0.9, 200)
for name, w in WINDOWS.items():
    pk = occ[:, window_idx(t, w)].max(1)
    ax[0].plot(xs, [(pk > x).mean() for x in xs], label=name); ax[1].hist(pk, 60, alpha=.55, label=name)
ax[0].axvline(thr, c="r", ls=":"); ax[0].set_xlabel("Threshold x"); ax[0].set_ylabel("P(peak crowd > x)"); ax[0].legend(); ax[0].set_title("Exceedance curve")
ax[1].axvline(thr, c="r", ls=":"); ax[1].set_xlabel("Peak crowd in window"); ax[1].set_title("Distribution of peak crowd"); ax[1].legend()
plt.tight_layout(); plt.savefig("fig3_exceedance_curve.png"); plt.close()

# Fig4: Monte Carlo convergence
pk = occ[:, window_idx(t, WINDOWS["Evening peak 17-20"])].max(1) > thr
cum = np.cumsum(pk)/np.arange(1, N_SIMS+1); ns = np.arange(1, N_SIMS+1)
fig, ax = plt.subplots(figsize=(8, 3.5)); ax.plot(ns, cum); ax.set_xscale("log")
ax.axhline(cum[-1], c="k", ls=":"); ax.set_xlabel("Number of simulations"); ax.set_ylabel("Estimated probability")
ax.set_title("Monte Carlo convergence (evening peak)"); plt.tight_layout(); plt.savefig("fig4_convergence.png"); plt.close()

# ------------------------- SENSITIVITY & SCENARIOS -------------------------
def p_eve(Pq, n=8000):
    global rng; rng = np.random.default_rng(7)
    tt, _, o = simulate(Pq, n); return (o[:, window_idx(tt, WINDOWS["Evening peak 17-20"])].max(1) > Pq["threshold"]).mean()

# Heatmap: evening exceedance vs headway on Yellow and demand scale
scales = [0.8, 0.9, 1.0, 1.1, 1.2]; hws = [2.0, 2.5, 3.0, 3.5, 4.0]
H = np.zeros((len(hws), len(scales)))
for i, h in enumerate(hws):
    for j, s in enumerate(scales):
        Q = dict(P, headway_y=h, base=P["base"]*s, amp_m=P["amp_m"]*s, amp_e=P["amp_e"]*s); H[i, j] = p_eve(Q)
fig, ax = plt.subplots(figsize=(6.2, 4.6)); im = ax.imshow(H, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(scales))); ax.set_xticklabels([f"{s:.0%}" for s in scales]); ax.set_yticks(range(len(hws))); ax.set_yticklabels(hws)
ax.set_xlabel("Demand vs baseline"); ax.set_ylabel("Yellow-line headway (min)"); ax.grid(False)
for i in range(len(hws)):
    for j in range(len(scales)): ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("P(exceed) evening peak: headway × demand"); plt.colorbar(im); plt.tight_layout(); plt.savefig("fig5_sensitivity_heatmap.png"); plt.close()

# Scenario table
scen = {"Baseline": P,
        "Demand +15%": dict(P, base=P["base"]*1.15, amp_m=P["amp_m"]*1.15, amp_e=P["amp_e"]*1.15),
        "Add train (Yellow 2.0 min)": dict(P, headway_y=2.0),
        "Frequent disruptions (p=0.15)": dict(P, p_delay=0.15),
        "High demand uncertainty (CV 0.25)": dict(P, day_cv=0.25),
        "Threshold 600": dict(P, threshold=600), "Threshold 1000": dict(P, threshold=1000)}
sc = pd.DataFrame({"scenario": list(scen), "P_exceed_evening": [round(p_eve(q), 4) for q in scen.values()]})
print("\n", sc.to_string(index=False)); sc.to_csv("results_scenarios.csv", index=False)
print("\nSaved: results_main.csv, results_scenarios.csv, fig1-fig5 PNGs")