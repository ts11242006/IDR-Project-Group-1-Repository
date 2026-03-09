import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import solve_ivp

# Load data
df = pd.read_csv("C:/IMPERIAL/Year 2/IRC/data.csv")

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

# Filter for first vaccine year (8 Feb 2020 → 8 Mar 2021)
start_date = pd.to_datetime("2020-02-08")
end_date = pd.to_datetime("2021-03-08")
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

days = [46, 135, 270]

for d in days:
    plt.axvline(x=d, color="red", linestyle="--")

plt.xlabel("Days since 8 Feb 2020")
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
N = 67_000_000          # UK population
beta = 0.25             # transmission rate
sigma = 0.25            # incubation rate
gamma = 0.1             # recovery rate 
nu = 0.0005              # constant vaccination rate

# Initial conditions
I0 = 1000
E0 = 2000
R0 = 0
V0 = 0
D0 = 0
S0 = N - I0 - E0 - R0 - V0 - D0

y0 = [S0, E0, I0, R0, V0, D0]


# SEIR-V ODE system


def seirv(t, y):
    S, E, I, R, V, D = y
    
    dSdt = -beta * S * I / N - 0.97 * nu * S + 0.004 * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I 
    dRdt = gamma * 0.975 * I
    dVdt = 0.97 * nu * S
    dDdt = 0.025 * gamma * I
    
    return [dSdt, dEdt, dIdt, dRdt, dVdt, dDdt]


# Time span (1 year)


t_span = [0, 365]
t_eval = np.linspace(0, 365, 365)

solution = solve_ivp(seirv, t_span, y0, t_eval=t_eval)

S, E, I, R, V, D = solution.y


# Plot results


plt.figure()
plt.plot(t_eval, I, label="Infectious")
plt.plot(t_eval, S, label="Susceptible")
plt.plot(t_eval, V, label="Vaccinated")
plt.plot(t_eval, R, label="Recovered")
plt.plot(t_eval, D, label="Deaths")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.title("SEIR-V Model (Iteration 1, UK)")
plt.show()

# Last values at the end of 1 year
S_end = S[-1]
E_end = E[-1]
I_end = I[-1]
R_end = R[-1]
V_end = V[-1]
D_end = D[-1]

print("Iteration 1 (constant β) totals at 1 year:")
print(f"Susceptible: {S_end:.0f}")
print(f"Exposed:     {E_end:.0f}")
print(f"Infectious:  {I_end:.0f}")
print(f"Recovered:   {R_end:.0f}")
print(f"Vaccinated:  {V_end:.0f}")
print(f"Deaths:      {D_end:.0f}")
print(f"Total check: {S_end + E_end + I_end + R_end + V_end + D_end:.0f} (should = {N})")
