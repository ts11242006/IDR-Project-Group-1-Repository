import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Parameters (Iteration 1)


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
