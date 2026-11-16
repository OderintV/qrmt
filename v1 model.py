import numpy as np
rng = np.random.default_rng(42)

# ---- PARAMETERS (placeholders: replace with Shubh's real/estimated data) ----
T = 120                 # peak window: 120 minutes (e.g. 8:00-10:00)
lam = 60                # mean arrivals per minute (Poisson rate) at station
headway = 3             # a train every 3 min
board_mean, board_sd = 190, 30   # passengers boarding per train
threshold = 1000        # crowd threshold (people on platform/concourse)
n_sims = 10_000

def simulate():
    N, path = 0, np.zeros(T)
    for t in range(T):
        # rate varies: ramps up to a mid-peak then falls
        rate = lam * (0.7 + 0.6*np.sin(np.pi * t / T))
        N += rng.poisson(rate)
        if t % headway == 0:
            cap = max(0, rng.normal(board_mean, board_sd))
            N -= min(N, cap)
        path[t] = N
    return path

paths = np.array([simulate() for _ in range(n_sims)])
peak = paths.max(axis=1)

print("Mean occupancy:", paths.mean().round(1), " SD:", paths.std().round(1))
print("P(exceed at any time in peak):", (peak > threshold).mean())
print("Expected arrivals/min (avg):", round(lam*np.mean(0.7+0.6*np.sin(np.pi*np.arange(T)/T)),1))
for a,b in [(0,30),(30,60),(60,90),(90,120)]:
    print(f"P(N>thr) interval {a}-{b} min:", (paths[:,a:b].max(axis=1)>threshold).mean())
se = np.sqrt((peak>threshold).mean()*(1-(peak>threshold).mean())/n_sims)
print("Std error of estimate:", round(se,4))
print("Mean peak:", peak.mean().round(0), "95th pct peak:", np.percentile(peak,95))