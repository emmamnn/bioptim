import pickle
import matplotlib.pyplot as plt
import os 
import numpy as np

def load_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def plot_comparaison(var_name, data_1, data_2, t1, t2, ylabel="", title_prefix="", fig_num=1):
    plt.figure(fig_num, figsize=(16, 9))
    if title_prefix =="q":
        #q0 translation déjà en m 
        plt.subplot(2, 2, 1)
        plt.plot(t1, data_1[0], label="without constraint")
        plt.plot(t2, data_2[0], label="constraint")
        plt.xlabel("Temps (s)")
        plt.ylabel(ylabel + " (m)")
        plt.title(f"{title_prefix}{0}")
        plt.grid()
        plt.legend()
        for i in range(1,4):
            plt.subplot(2, 2, i+1)
            plt.plot(t1, data_1[i]*180/np.pi, label="without constraint")
            plt.plot(t2, data_2[i]*180/np.pi, label="constraint")
            plt.xlabel("Temps (s)")
            plt.ylabel(ylabel + " (°)")
            plt.title(f"{title_prefix}{i}")
            plt.grid()
            plt.legend()
    elif title_prefix == "tau":
            for i in range(4):
                plt.subplot(2, 2, i+1)
                plt.step(t1[:-1], data_1[i], where='pre', label="without constraint")
                plt.step(t2[:-1], data_2[i], where='pre', label="constraint")
                plt.xlabel("Temps (s)")
                plt.ylabel(ylabel)
                plt.title(f"{title_prefix}{i}")
                plt.grid()
                plt.legend() 
    else: 
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot(t1, data_1[i], label="without constraint")
            plt.plot(t2, data_2[i], label="constraint")
            plt.xlabel("Temps (s)")
            plt.ylabel(ylabel)
            plt.title(f"{title_prefix}{i}")
            plt.grid()
            plt.legend()

def print_costs(name, data):
    print(f"------ {name} ------")
    print("MINIMIZE_CONTROL weighted :", data["lagrange_control_cost"])
    print("MINIMIZE_CONTROL unweighted :", data["lagrange_control_cost_unweighted"])
    print("MINIMIZE_TIME :", data["mayer_time_cost_unweighted"])

# Chemins
base_path = "../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/results/"

datasets = {
    "initial": {
        "without_constraint": load_data(os.path.join(base_path, "pendulum_data.pkl")),
        "constraint": load_data(os.path.join(base_path, "pendulum_data_constraint.pkl")),
    },
    "with_start": {
        "without_constraint": load_data(os.path.join(base_path, "avec_solution_initiale/pendulum_data_100_shooting_start.pkl")),
        "constraint": load_data(os.path.join(base_path, "avec_solution_initiale/pendulum_data_100_shooting_start_constraint.pkl")),
    }
}

# Boucle sur chaque type de jeu de données
fig_index = 1
for label, data_dict in datasets.items():
    no_cst_data = data_dict["without_constraint"]
    cst_data = data_dict["constraint"]
    print(f"\n======= Résultats pour : {label} =======\n")
    print_costs("without constraint", no_cst_data)
    print()
    print_costs("with constraint", cst_data)

    # Temps
    t1 = np.linspace(0, no_cst_data["mayer_time_cost_unweighted"], no_cst_data["n_shooting"] + 1)
    t2 = np.linspace(0, cst_data["mayer_time_cost_unweighted"], cst_data["n_shooting"] + 1)

    # Tracés
    plot_comparaison("q", no_cst_data["q"], cst_data["q"], t1, t2, ylabel="Position", title_prefix="q", fig_num=fig_index)
    plt.tight_layout()
    plt.savefig(f"figure_{fig_index}_q.png")
    plot_comparaison("qdot", no_cst_data["qdot"], cst_data["qdot"], t1, t2, ylabel="Vitesse (rad/s)", title_prefix="qdot", fig_num=fig_index+1)
    plt.tight_layout()
    plt.savefig(f"figure_{fig_index+1}_qdot.png")
    plot_comparaison("tau", no_cst_data["tau"], cst_data["tau"], t1, t2, ylabel="Tau (N·m)", title_prefix="tau", fig_num=fig_index+2)
    plt.tight_layout()
    plt.savefig(f"figure_{fig_index+2}_tau.png")

    fig_index += 3  # pour ne pas écraser les figures précédentes

plt.show()



# # Charger les données
# path = "../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/results/"
# with open(os.path.join(path, "pendulum_data.pkl"), "rb") as file:
#     data_without_constraint = pickle.load(file)

# with open(os.path.join(path, "pendulum_data_constraint.pkl"), "rb") as file:
#     data = pickle.load(file)

# # Accès aux variables
# q = data_without_constraint["q"]
# qdot = data_without_constraint["qdot"]
# tau = data_without_constraint["tau"]
# cost = data_without_constraint["lagrange_control_cost"]
# cost_unweigted = data_without_constraint["lagrange_control_cost_unweighted"]
# t_final = data_without_constraint["mayer_time_cost_unweighted"]
# n_shooting = data_without_constraint["n_shooting"]
# t = np.linspace(0, t_final, n_shooting + 1)

# q_constraint = data["q"]
# qdot_constraint = data["qdot"]
# tau_constraint = data["tau"]
# cost_constraint = data["lagrange_control_cost"]
# cost_unweigted_constraint = data["lagrange_control_cost_unweighted"]
# t_final_constraint = data["mayer_time_cost_unweighted"]
# n_shooting_constraint = data["n_shooting"]
# t_constraint = np.linspace(0, t_final_constraint, n_shooting_constraint + 1)


# print("------ without constraint ------")
# print("MINIMIZE_CONTROL weighted : ", cost, "\nMINIMIZE_CONTROL unweighted", cost_unweigted, "\nMINIMIZE_TIME : ", t_final)
# print("\n------ with constraint ------")
# print("MINIMIZE_CONTROL weighted : ", cost_constraint, "\nMINIMIZE_CONTROL unweighted", cost_unweigted_constraint, "\nMINIMIZE_TIME : ", t_final_constraint)


# plt.figure(1)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(t, q[i], label="without constraint")
#     plt.plot(t_constraint, q_constraint[i], label= "constraint")
#     plt.xlabel("Temps (s)")
#     plt.ylabel("Position (rad)")
#     plt.title(f"q{i}")
#     plt.grid()
#     plt.legend()

# plt.figure(2)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(t, qdot[i], label="without constraint")
#     plt.plot(t_constraint, qdot_constraint[i], label= "constraint")
#     plt.xlabel("Temps (s)")
#     plt.ylabel("Vitesse (rad/s)")
#     plt.title(f"qdot{i}")
#     plt.grid()
#     plt.legend()

# plt.figure(3)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(t[:-1], tau[i], label="without constraint")
#     plt.plot(t_constraint[:-1], tau_constraint[i], label= "constraint")
#     plt.xlabel("Temps (s)")
#     plt.ylabel("Tau (N·m)")
#     plt.title(f"tau{i}")
#     plt.grid()
#     plt.legend()



# with open(os.path.join(path, "avec_solution_initiale/pendulum_data_start_solution.pkl"), "rb") as file:
#     data_without_constraint_start = pickle.load(file)

# with open(os.path.join(path, "avec_solution_initiale/pendulum_data_constraint_start_solution.pkl"), "rb") as file:
#     data_start = pickle.load(file)
    
# # Accès aux variables
# q_start = data_without_constraint_start["q"]
# qdot_start = data_without_constraint_start["qdot"]
# tau_start = data_without_constraint_start["tau"]
# cost_start = data_without_constraint_start["lagrange_control_cost"]
# cost_unweigted_start = data_without_constraint_start["lagrange_control_cost_unweighted"]
# t_final_start = data_without_constraint_start["mayer_time_cost_unweighted"]
# n_shooting_start = data_without_constraint_start["n_shooting"]
# t_start = np.linspace(0, t_final_start, n_shooting_start + 1)

# q_constraint_start = data_start["q"]
# qdot_constraint_start = data_start["qdot"]
# tau_constraint_start = data_start["tau"]
# cost_constraint_start = data_start["lagrange_control_cost"]
# cost_unweigted_constraint_start = data_start["lagrange_control_cost_unweighted"]
# t_final_constraint_start = data_start["mayer_time_cost_unweighted"]
# n_shooting_constraint_start = data_start["n_shooting"]
# t_constraint_start = np.linspace(0, t_final_constraint_start, n_shooting_constraint_start + 1)


# print("------ without constraint ------")
# print("MINIMIZE_CONTROL weighted : ", cost_start, "\MINIMIZE_CONTROL unweighted", cost_unweigted_start, "\nMINIMIZE_TIME : ", t_final_start)
# print("\n------ with constraint ------")
# print("MINIMIZE_CONTROL weighted : ", cost_constraint_start, "\MINIMIZE_CONTROL unweighted", cost_unweigted_constraint_start, "\nMINIMIZE_TIME : ", t_final_constraint_start)


# plt.figure(4)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(t_start, q_start[i], label="without constraint")
#     plt.plot(t_constraint_start, q_constraint_start[i], label= "constraint")
#     plt.xlabel("Temps (s)")
#     plt.ylabel("Position (rad)")
#     plt.title(f"q{i}")
#     plt.grid()
#     plt.legend()

# plt.figure(5)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(t_start, qdot_start[i], label="without constraint")
#     plt.plot(t_constraint_start, qdot_constraint_start[i], label= "constraint")
#     plt.xlabel("Temps (s)")
#     plt.ylabel("Vitesse (rad/s)")
#     plt.title(f"qdot{i}")
#     plt.grid()
#     plt.legend()

# plt.figure(6)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(t_start[:-1], tau_start[i], label="without constraint")
#     plt.plot(t_constraint_start[:-1], tau_constraint_start[i], label= "constraint")
#     plt.xlabel("Temps (s)")
#     plt.ylabel("Tau (N·m)")
#     plt.title(f"tau{i}")
#     plt.grid()
#     plt.legend()

# plt.show()
