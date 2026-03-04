import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import solve_ivp

# Load data
df = pd.read_csv("Cases 2/ltla_newCasesBySpecimenDate.csv")

# Filter correct metric
df = df[df["metric"] == "newCasesBySpecimenDate"]

# Aggregate to UK level
uk_daily = df.groupby("date")["value"].sum().reset_index()

# Parse dates safely (UK format DD/MM/YYYY)
uk_daily["date"] = pd.to_datetime(
    uk_daily["date"],
    dayfirst=True,
    errors="coerce"
)
uk_daily = uk_daily.dropna(subset=["date"]).copy()

# Filter for first vaccine year (8 Dec 2020 → 8 Dec 2021)
start_date = pd.to_datetime("2020-12-08")
end_date = pd.to_datetime("2021-12-08")
mask = (uk_daily["date"] >= start_date) & (uk_daily["date"] <= end_date)
uk_year = uk_daily.loc[mask].copy()
uk_year = uk_year.sort_values("date").reset_index(drop=True)

# Compute 7-day centered moving average
uk_year["cases_smooth"] = uk_year["value"].rolling(window=7, center=True).mean()
uk_year["cases_smooth"] = uk_year["cases_smooth"].replace(0, np.nan)
uk_year = uk_year.dropna(subset=["cases_smooth"]).copy()

# Compute r(t)
log_cases = np.log(uk_year["cases_smooth"].values)
r_t = np.gradient(log_cases)
uk_year["r_t"] = r_t

# Fit spline to r(t)
t_numeric = np.arange(len(uk_year))
spline = UnivariateSpline(t_numeric, uk_year["r_t"], s=0.1)
r_fit = spline(t_numeric)

# Plot r(t)
plt.figure(figsize=(10, 5))
plt.plot(t_numeric, uk_year["r_t"], label="Estimated r(t)", alpha=0.5)
plt.plot(t_numeric, r_fit, label="Fitted spline r(t)", linewidth=2)
plt.xlabel("Days since 8 Dec 2020")
plt.ylabel("Growth rate r(t)")
plt.legend()
plt.title("Estimated and Fitted Growth Rate r(t)")
plt.tight_layout()
plt.show()

# Compute R(t) and β(t)
serial_interval = 5.2
uk_year["R_t"] = np.exp(r_fit * serial_interval)

infectious_period = 5.0
gamma_seir = 1 / infectious_period
uk_year["beta_t"] = uk_year["R_t"] * gamma_seir

# Scale β(t) for realistic infections
scaling_factor = 1.75
uk_year["beta_t_scaled"] = uk_year["beta_t"] * scaling_factor

# Plot scaled β(t)
plt.figure(figsize=(10, 5))
plt.plot(uk_year["date"], uk_year["beta_t_scaled"], linewidth=2, label="Scaled Transmission rate β(t)")
plt.ylabel("β(t)")
plt.xlabel("Date")
plt.title("Scaled Transmission Rate β(t)")
plt.legend()
plt.tight_layout()
plt.show()

# Interpolate β(t) for SEIR-V
dates_beta = uk_year["date"].values
beta_smooth = uk_year["beta_t_scaled"].values
t_days = (dates_beta - dates_beta[0]).astype('timedelta64[D]').astype(float)
beta_func = interp1d(t_days, beta_smooth, kind='cubic', fill_value='extrapolate')

# SEIR-V simulation parameters
N = 67_000_000
sigma = 0.25
gamma = 0.2
nu = 0.005  # tuned vaccination rate

# Initial conditions
I0 = 25000
E0 = 10000
R0 = 0
V0 = 0
S0 = N - I0 - E0 - R0 - V0
y0 = [S0, E0, I0, R0, V0]

# SEIR-V model
def seirv_timevarying(t, y):
    S, E, I, R, V = y
    beta_t = beta_func(t)
    dSdt = -beta_t * S * I / N - nu * S
    dEdt = beta_t * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S
    return [dSdt, dEdt, dIdt, dRdt, dVdt]

# Time span
T = t_days[-1]
t_eval = np.linspace(0, T, len(t_days))

solution = solve_ivp(seirv_timevarying, [0, T], y0, t_eval=t_eval)
S, E, I, R, V = solution.y

# Plot results
plt.figure(figsize=(12,6))
plt.plot(dates_beta, I, label="Infectious")
plt.plot(dates_beta, S, label="Susceptible")
plt.plot(dates_beta, V, label="Vaccinated")
plt.plot(dates_beta, R, label="Recovered")
plt.xlabel("Date")
plt.ylabel("Population")
plt.legend()
plt.title("SEIR-V Simulation with Scaled β(t) and Adjusted ν")
plt.grid(True)
plt.tight_layout()
plt.show()

# End-of-year totals
S_end_tv = S[-1]
E_end_tv = E[-1]
I_end_tv = I[-1]
R_end_tv = R[-1]
V_end_tv = V[-1]

print("Iteration 2 (time-varying β, scaled 1.75×, ν=0.00375) totals at 1 year:")
print(f"Susceptible: {S_end_tv:.0f}")
print(f"Exposed:     {E_end_tv:.0f}")
print(f"Infectious:  {I_end_tv:.0f}")
print(f"Recovered:   {R_end_tv:.0f}")
print(f"Vaccinated:  {V_end_tv:.0f}")
print(f"Total check: {S_end_tv + E_end_tv + I_end_tv + R_end_tv + V_end_tv:.0f} (should = {N})")