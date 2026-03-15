import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import solve_ivp

# -----------------------------
# LOAD CASE DATA
# -----------------------------

df = pd.read_csv("Cases 2/ltla_newCasesBySpecimenDate.csv")

df = df[df["metric"] == "newCasesBySpecimenDate"]

uk_daily = df.groupby("date")["value"].sum().reset_index()

uk_daily["date"] = pd.to_datetime(
    uk_daily["date"], dayfirst=True, errors="coerce"
)

uk_daily = uk_daily.dropna(subset=["date"]).copy()

# -----------------------------
# FILTER TIME PERIOD
# -----------------------------

start_date = pd.to_datetime("2020-02-08")
end_date = pd.to_datetime("2021-03-08")

mask = (uk_daily["date"] >= start_date) & (uk_daily["date"] <= end_date)

uk_year = uk_daily.loc[mask].copy()

uk_year = uk_year.sort_values("date").reset_index(drop=True)

# -----------------------------
# SMOOTH CASE DATA
# -----------------------------

uk_year["cases_smooth"] = uk_year["value"].rolling(window=7, center=True).mean()

uk_year["cases_smooth"] = uk_year["cases_smooth"].replace(0, np.nan)

uk_year = uk_year.dropna(subset=["cases_smooth"]).copy()

# -----------------------------
# ESTIMATE r(t)
# -----------------------------

log_cases = np.log(uk_year["cases_smooth"].values)

r_t = np.gradient(log_cases)

uk_year["r_t"] = r_t

# spline fit

t_numeric = np.arange(len(uk_year))

spline = UnivariateSpline(t_numeric, uk_year["r_t"], s=0.1)

r_fit = spline(t_numeric)

# plot r(t)

plt.figure(figsize=(10,5))

plt.plot(t_numeric, uk_year["r_t"], label="Estimated r(t)", alpha=0.5)
plt.plot(t_numeric, r_fit, label="Fitted spline r(t)", linewidth=2)

days = [46,135,271,298]

for d in days:
    plt.axvline(x=d,color="red",linestyle="--")

plt.xlabel("Days since 8 Feb 2020")
plt.ylabel("Growth rate r(t)")
plt.legend()
plt.title("Estimated and Fitted Growth Rate r(t)")
plt.tight_layout()
plt.show()

# -----------------------------
# COMPUTE β(t)
# -----------------------------

serial_interval = 5.2

uk_year["R_t"] = np.exp(r_fit * serial_interval)

infectious_period = 5.0

gamma_seir = 1 / infectious_period

uk_year["beta_t"] = uk_year["R_t"] * gamma_seir

scaling_factor = 1.75

uk_year["beta_t_scaled"] = uk_year["beta_t"] * scaling_factor

# plot β(t)

plt.figure(figsize=(10,5))

plt.plot(
    uk_year["date"],
    uk_year["beta_t_scaled"],
    linewidth=2,
    label="Scaled Transmission rate β(t)"
)

plt.ylabel("β(t)")
plt.xlabel("Date")
plt.title("Scaled Transmission Rate β(t)")
plt.legend()
plt.tight_layout()
plt.show()

# interpolate β(t)

dates_beta = uk_year["date"].values
beta_smooth = uk_year["beta_t_scaled"].values

t_days = (dates_beta - dates_beta[0]).astype('timedelta64[D]').astype(float)

beta_func = interp1d(
    t_days,
    beta_smooth,
    kind='cubic',
    fill_value='extrapolate'
)

# -----------------------------
# LOAD VACCINATION DATA
# -----------------------------

df_vax = pd.read_csv(
"Vaccinations/ltla_newPeopleVaccinatedFirstDoseByVaccinationDate.csv"
)

uk_vax = df_vax.groupby("date")["value"].sum().reset_index()

uk_vax["date"] = pd.to_datetime(uk_vax["date"])

mask = (uk_vax["date"] >= start_date) & (uk_vax["date"] <= end_date)

uk_vax = uk_vax.loc[mask].copy()

uk_vax = uk_vax.sort_values("date").reset_index(drop=True)

# smooth vaccination data

uk_vax["vax_smooth"] = uk_vax["value"].rolling(window=7, center=True).mean()

uk_vax = uk_vax.dropna(subset=["vax_smooth"])

# vaccination rate ν(t)

N = 67_000_000

uk_vax["nu_t"] = uk_vax["vax_smooth"] / N

# interpolate ν(t)

dates_vax = uk_vax["date"].values

nu_values = uk_vax["nu_t"].values

t_days_vax = (dates_vax - dates_vax[0]).astype('timedelta64[D]').astype(float)

nu_func = interp1d(
    t_days_vax,
    nu_values,
    kind='cubic',
    fill_value='extrapolate'
)

# plot vaccination rate

plt.figure(figsize=(10,5))

plt.plot(uk_vax["date"], uk_vax["nu_t"], label="Vaccination rate ν(t)")

plt.xlabel("Date")
plt.ylabel("ν(t)")
plt.title("Estimated Vaccination Rate")
plt.legend()

plt.show()
