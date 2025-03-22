#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags


# In[3]:


def V(x, alpha):
    return alpha*x**4 - 2*x**2 + 1/alpha


# In[4]:


alpha = 0.4
h_bar = 1
m = 1


# In[5]:


def delta_action(x_path, i, x_new, m, dtau):
    N = len(x_path)
    i_minus = (i - 1) % N
    i_plus  = (i + 1) % N

    x_old = x_path[i]

    kin_old = (m/(2*dtau))*((x_path[i_plus] - x_old)**2
                           + (x_old - x_path[i_minus])**2)
    pot_old = dtau*( V(0.5*(x_old + x_path[i_plus]),alpha)
                   + V(0.5*(x_path[i_minus] + x_old),alpha) )

    kin_new = (m/(2*dtau))*((x_path[i_plus] - x_new)**2
                           + (x_new - x_path[i_minus])**2)
    pot_new = dtau*( V(0.5*(x_new + x_path[i_plus]), alpha)
                   + V(0.5*(x_path[i_minus] + x_new), alpha) )

    return (kin_new + pot_new) - (kin_old + pot_old)


# In[6]:


def metropolis_mcmc(N, tau_b, m, num_sweep,
                    burn_in_steps, hit_size, thinning):
    dtau = tau_b / N
   

    x_path = np.zeros(N)
    
    
    E_list = []
    all_x_data = []
    accepted = 0

    for sweep in range(num_sweep):
        for i in range(N):
            x_old = x_path[i]
            x_prop = x_old + np.random.uniform(-hit_size, hit_size)

            dS = delta_action(x_path, i, x_prop, m, dtau)

            if dS < 0:
                accept = True
            else:
                r = np.random.rand()
                accept = (r < np.exp(-dS))

            if accept:
                x_path[i] = x_prop
                accepted += 1
                
        if sweep > burn_in_steps and ((sweep - burn_in_steps) % thinning == 0):
            # measure energy
            kin_sum = 0.0
            pot_sum = 0.0
            for i in range(N):
                i_plus = (i + 1) % N
                kin_sum += (m/(2*dtau)) * (x_path[i_plus]-x_path[i])**2
                pot_sum += V(0.5*(x_path[i_plus]+x_path[i]), alpha)
            E_sample = (kin_sum + pot_sum) / N
            E_list.append(E_sample)

            all_x_data.extend(x_path)

    accept_rate = accepted / (num_sweep*N)
    return E_list, all_x_data, accept_rate


# In[14]:


N = 1600 
tau_b = 5
burn_in_steps = 5000
num_sweep = 50000
hit_size = 0.4
thinning = 50

E_list, all_x_data, arate = metropolis_mcmc(
        N=N, tau_b=tau_b, m=m,
        num_sweep=num_sweep, burn_in_steps=burn_in_steps,
        thinning=thinning, hit_size=hit_size
    )


E_mean = np.mean(E_list)
E_err  = np.std(E_list)/np.sqrt(len(E_list))

print(f"alpha = {alpha}")
print(f"Acceptance rate: {arate*100:.1f}%")
print(f"Estimated ground state energy ~ {E_mean:.4f} ± {E_err:.4f}")

plt.hist(all_x_data, bins=60, density=True, alpha=0.7, color='purple')
plt.title(f"Probability Distribution Tau = 5 (alpha={alpha})")
plt.xlabel("x")
plt.ylabel("P(x)")
plt.grid(alpha=0.3)
plt.show()


# In[15]:


BOXSIZE = 8 
ND_x = 600 

x_grid = np.linspace(-BOXSIZE/2, BOXSIZE/2, ND_x+1)
dx = x_grid[1] - x_grid[0]

H = np.zeros((ND_x+1, ND_x+1), dtype=np.float64)
for i in range(ND_x+1):
    H[i, i] = -0.5 * (-2)/dx**2 + V(x_grid[i],alpha)
    if i+1 <= ND_x:
        H[i, i+1] = -0.5 / dx**2   
        H[i+1, i] = -0.5 / dx**2

eigvals, eigvecs = np.linalg.eigh(H)
E0_numeric = eigvals[0]
psi0_numeric = eigvecs[:, 0] 

norm_factor = np.sqrt(np.trapz(psi0_numeric**2, x_grid))
psi0_numeric /= norm_factor


psi0_sq = psi0_numeric**2

x_vals = np.linspace(-3, 3, 500)
  
P_x_boltzmann = np.exp(-tau_b * V(x_vals, alpha=0.4))
P_x_boltzmann /= np.trapz(P_x_boltzmann, x_vals)

plt.figure(figsize=(8,5))

# 1. PIMC 
counts, bin_edges = np.histogram(all_x_data, bins=60, density=True)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]),
        alpha=0.4, color='purple', label="PIMC with tau = 5")

# 2.|psi0|^2
plt.plot(x_grid, psi0_sq, 'r-', linewidth=2, label="$|\\psi_0(x)|^2$")

#3 Boltzmann
plt.plot(x_vals, P_x_boltzmann, 'b--', linewidth=2, label="Boltzmann tau = 5")

#4 Potential
V_x = V(x_vals, alpha=0.4)
plt.plot(x_vals, V_x, linestyle="dotted", color="black", linewidth=2, label="Potential $V(x)$")

# Formatting
plt.ylim(0, 3)
plt.xlabel("$x$")
plt.ylabel("Probability Density")
plt.title("Comparison: PIMC, Ground State, Boltzmann Distribution, and Potential")
plt.legend()
plt.grid(alpha=0.3)


# In[ ]:




