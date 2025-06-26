import pickle
import matplotlib.pyplot as plt
import os 
import numpy as np

# Charger les données
path = "../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/results/"
with open(os.path.join(path, "pendulum_data.pkl"), "rb") as file:
    data_without_constraint = pickle.load(file)

with open(os.path.join(path, "pendulum_data_constraint.pkl"), "rb") as file:
    data = pickle.load(file)

# Accès aux variables
q = data_without_constraint["q"]
qdot = data_without_constraint["qdot"]
tau = data_without_constraint["tau"]
cost = data_without_constraint["lagrange_control_cost"]
cost_unweigted = data_without_constraint["lagrange_control_cost_unweighted"]
t_final = data_without_constraint["mayer_time_cost_unweighted"]
n_shooting = data_without_constraint["n_shooting"]
t = np.linspace(0, t_final, n_shooting + 1)

q_constraint = data["q"]
qdot_constraint = data["qdot"]
tau_constraint = data["tau"]
cost_constraint = data["lagrange_control_cost"]
cost_unweigted_constraint = data["lagrange_control_cost_unweighted"]
t_final_constraint = data["mayer_time_cost_unweighted"]
n_shooting_constraint = data["n_shooting"]
t_constraint = np.linspace(0, t_final_constraint, n_shooting_constraint + 1)


print("------ without constraint ------")
print("MINILIZE_CONTROL weighted : ", cost, "\nMINILIZE_CONTROL unweighted", cost_unweigted)
print("\n------ with constraint ------")
print("MINILIZE_CONTROL weighted : ", cost_constraint, "\nMINILIZE_CONTROL unweighted", cost_unweigted_constraint)


plt.figure(1)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t, q[i], label="without constraint")
    plt.plot(t_constraint, q_constraint[i], label= "constraint")
    plt.xlabel("Temps (s)")
    plt.ylabel("Position (rad)")
    plt.title(f"q{i}")
    plt.grid()
    plt.legend()

plt.figure(2)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t, qdot[i], label="without constraint")
    plt.plot(t_constraint, qdot_constraint[i], label= "constraint")
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (rad/s)")
    plt.title(f"qdot{i}")
    plt.grid()
    plt.legend()

plt.figure(3)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t[:-1], tau[i], label="without constraint")
    plt.plot(t_constraint[:-1], tau_constraint[i], label= "constraint")
    plt.xlabel("Temps (s)")
    plt.ylabel("Tau (N·m)")
    plt.title(f"tau{i}")
    plt.grid()
    plt.legend()

plt.show()
