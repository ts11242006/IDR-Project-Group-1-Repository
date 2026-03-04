# SEIR-V Simulation of Covid-19 Pandemic

The main aim of this project is to carry out a SEIR-V simulation for the period of the beginning of the pandemic to after a year of implementing the covid vaccine and compare this simulation to real life data.

We also plan to iteratively increase the accuracy of the simulation by implementing time dependent parameters.

The first iteration involves only constant parameters with no time dependency

The second iteration involves a time-varying transmission rate $\beta$

## Mathematical Model
- $S(t)$ = susceptible population  
- $E(t)$ = exposed population  
- $I(t)$ = infectious population  
- $R(t)$ = recovered population  
- $V(t)$ = vaccinated population  
- $N = S + E + I + R + V$  
- $\beta(t)$ = time-varying transmission rate  
- $\sigma$ = incubation rate  
- $\gamma$ = recovery rate  
- $\nu$ = vaccination rate  


The SEIR-V model is defined as:

$$
\frac{dS}{dt} = -\frac{\beta(t) S I}{N} - \nu S
$$

$$
\frac{dE}{dt} = \frac{\beta(t) S I}{N} - \sigma E
$$

$$
\frac{dI}{dt} = \sigma E - \gamma I
$$

$$
\frac{dR}{dt} = \gamma I
$$

$$
\frac{dV}{dt} = \nu S
$$

