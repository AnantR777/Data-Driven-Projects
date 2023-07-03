import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def dynamical_system(t, xyz, a, b, c):
    x, y, z = xyz
    dx_dt = z + (y - a) * x
    dy_dt = 1 - b * y - x**2
    dz_dt = -x - c * z
    return [dx_dt, dy_dt, dz_dt]

# Define parameter range for 'a'
a_values = np.linspace(1, 2, 100)

# Fix 'b' and 'c'
b_fixed = 0.1
c_fixed = 1

# Set initial conditions
x0, y0, z0 = [3.75, 3.5, 2.98]

# Create empty lists to store the results
x_list, y_list, z_list = [], [], []

# Loop through 'a' values and integrate the system
for a in a_values:
    sol = solve_ivp(
        lambda t, xyz: dynamical_system(t, xyz, a, b_fixed, c_fixed),
        t_span=(0, 100),  # Integration time span
        y0=[x0, y0, z0],  # Initial conditions
        t_eval=np.linspace(0, 100, 1000)  # Evaluation time points
    )
    x_list.append(sol.y[0][-1])  # Append the final x value
    y_list.append(sol.y[1][-1])  # Append the final y value
    z_list.append(sol.y[2][-1])  # Append the final z value

# Plot the bifurcation diagram
for i in [[x_list, "blue", "x"], [y_list, "green", "y"], [z_list, "red", "z"]]:
    plt.figure(figsize=(10, 6))
    plt.scatter(a_values, i[0], s=1, c=i[1], alpha=0.5, label=i[2])
    plt.xlabel('Parameter a')
    plt.ylabel('Steady State Values')
    plt.title('Bifurcation Diagram')
    plt.legend()
    plt.grid(True)
plt.show()
