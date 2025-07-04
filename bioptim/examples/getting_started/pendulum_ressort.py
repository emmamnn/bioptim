import numpy as np
import matplotlib.pyplot as plt 
from casadi import MX, vertcat, sign
import os
import rerun as rr
import pickle

from bioptim import (
    Node,
    ConstraintList,
    InterpolationType,
    ConstraintFcn,
    ConfigureProblem,
    ObjectiveList,
    DynamicsFunctions,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    BiorbdModel,
    NonLinearProgram,
    ControlType,
    PhaseDynamics,
    OnlineOptim,
    ContactType,
    DynamicsEvaluation,
    PenaltyController,
    SolutionMerge,
    Axis,
    MultinodeConstraintList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    DynamicsList,
    MultinodeConstraintFcn,
)

import shutil

def custom_dynamic(
    time: MX,
    states: MX,
    controls: MX,
    parameters: MX,
    algebraic_states: MX,
    numerical_timeseries: MX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The dynamics of the system using an external force (see custom_dynamics for more explanation)

    Parameters
    ----------
    time: MX
        The current time of the system
    states: MX
        The current states of the system
    controls: MX
        The current controls of the system
    parameters: MX
        The current parameters of the system
    algebraic_states: MX
        The current algebraic states of the system
    numerical_timeseries: MX
        The current numerical timeseries of the system
    nlp: NonLinearProgram
        A reference to the phase of the ocp

    Returns
    -------
    The state derivative
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    stiffness = 100
    tau[0] -= stiffness * q[0]  # damping


    qddot = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

    return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=None)


def custom_configure(
    ocp: OptimalControlProgram,
    nlp: NonLinearProgram,
    numerical_data_timeseries=None,
    contact_type: list[ContactType] | tuple[ContactType] = (),
):
    """
    The configuration of the dynamics (see custom_dynamics for more explanation)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase of the ocp
    numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
    contact_type: list[ContactType] | tuple[ContactType]
        The type of contacts to consider in the dynamics.
    """
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: tuple,
    n_shooting: tuple,
    min_time : float,
    max_time : float,
    weight: float = 1,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
    x_init: InitialGuessList = None,
    u_init: InitialGuessList = None
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: tuple
        The initial guess for the final time of each phase
    n_shooting: tuple
        The number of shooting points (one number per phase) to define in the direct multiple shooting program
     min_time: float
        The minimum time allowed for the final node
    max_time: float
        The maximum time allowed for the final node
     weight: float
        The weighting of the minimize time objective function
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=1, phase=0 )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight, min_bound=min_time, max_bound=max_time, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=1, phase=1 )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight, min_bound=min_time, max_bound=max_time, phase=1)

    
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(Dynamics(
        custom_configure,
        dynamic_function=custom_dynamic,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    ))
    dynamics.add(Dynamics(
        custom_configure,
        dynamic_function=custom_dynamic,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    ))
    
    

    #Constraints
    constraint_list = ConstraintList()
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="LowerBarMarker", second_marker="marker_6", min_bound = 0.02, max_bound = np.inf, axes=Axis.Y, phase=1)
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="LowerBarMarker", second_marker="marker_6", axes=Axis.Z, phase=0, min_bound=0.02, max_bound=0.02)
    
    
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)

    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["q"][1, 0] = -np.pi #start 180 degrees rotated 
    x_bounds[0]["qdot"][:, [0, -1]] = 0 #speeds start and end at 0
    
    x_bounds[1]["q"][:, -1] = 0
    x_bounds[1]["q"][1,-1] = np.pi #ends with first pendulum 180 degrees rotated



    # Define control path bounds
    tau_min, tau_max = (-100, 100)
    n_tau = bio_model[0].nb_tau
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=1)


    if x_init == None: 
        x_init = InitialGuessList()
        x_init.add("q", [0] * bio_model[0].nb_q, phase=0)
        x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=0)
        x_init.add("q", [0] * bio_model[0].nb_q, phase=1)
        x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=1)
        
    if u_init == None:
        u_init = InitialGuessList()
        u_init.add("tau", [0] * bio_model[0].nb_tau, phase=0)
        u_init.add("tau", [0] * bio_model[0].nb_tau, phase=1)
            
            
    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
        constraints=constraint_list,
    )


def main():
    import os 
    import pickle
    RAC="../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/"
    
    # # --- to start from the solution without constraint --- #
    # path = "../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/results/"
    # with open(os.path.join(path, "pendulum_data_100_shooting.pkl"), "rb") as file:
    #     sol_without_constraint = pickle.load(file)

    # q_init = sol_without_constraint["q"]
    # qdot_init = sol_without_constraint["qdot"]
    # tau_init = sol_without_constraint["tau"]

    # # Initial guess with InterpolationType.EACH_FRAME
    # x_init = InitialGuessList()
    # x_init.add("q", q_init, interpolation=InterpolationType.EACH_FRAME)
    # x_init.add("qdot", qdot_init, interpolation=InterpolationType.EACH_FRAME)

    # u_init = InitialGuessList()
    # u_init.add("tau", tau_init, interpolation=InterpolationType.EACH_FRAME)
    
    
    # --- Prepare the ocp --- #
    n_shooting = (25, 75)
    # use_sol_init = False
    
    # if use_sol_init == True:
    #     ocp = prepare_ocp(biorbd_model_path=RAC + "models/long_triple_pendulum.bioMod", final_time=(1,3), n_shooting=n_shooting,min_time=0.5, max_time=10, weight=0.0001, n_threads=1, u_init=u_init, x_init=x_init)
    # else:
    #     ocp = prepare_ocp(biorbd_model_path=RAC + "models/long_triple_pendulum.bioMod", final_time=(1,3), n_shooting=n_shooting,min_time=0.5, max_time=10, weight=0.0001, n_threads=1)
    ocp = prepare_ocp(biorbd_model_path=RAC + "models/long_triple_pendulum.bioMod", final_time=(1,3), n_shooting=n_shooting,min_time=0.02, max_time=4, weight=0.001, n_threads=2)



    # --- Live plots --- #
    ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration
 
    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)


    # --- Saving the solver's output during the optimization --- #
    # path_to_results = RAC + "results/"
    # result_file_name = "pendulum"
    # nb_iter_save = 10  # Save the solver's output every 10 iterations
    # ocp.save_intermediary_ipopt_iterations(
    #     path_to_results, result_file_name, nb_iter_save
    # )

    # --- Solve the ocp --- #   
    solver = Solver.IPOPT(online_optim=OnlineOptim.DEFAULT,_linear_solver="ma57")      
    sol = ocp.solve(solver)
    
    
    # q = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])["q"][0]
    # qdot = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])["qdot"][0]
    # tau = sol.decision_controls(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])["tau"][0]

    
    # # --- Save the solution --- #
    # save = False 

    # if save == True :
        # q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        # qdot = sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
        # tau = sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
        
    #     # Do everything you need with the solution here before we delete ocp
    #     integrated_sol = sol.integrate(to_merge=SolutionMerge.NODES)
    #     q_integrated = integrated_sol["q"]
    #     qdot_integrated = integrated_sol["qdot"]
        
    #     # Récupération des coûts
    #     # Accès à la structure détaillée
    #     detailed_cost = sol.detailed_cost

    #     # Récupérer le coût pondéré et non pondéré pour chaque terme
    #     lagrange_cost_weighted = next(item for item in sol.detailed_cost if item["name"] == "Lagrange.MINIMIZE_CONTROL")["cost_value_weighted"]
    #     lagrange_cost_unweighted = next(item for item in sol.detailed_cost if item["name"] == "Lagrange.MINIMIZE_CONTROL")["cost_value"]

    #     mayer_cost_weighted = next(item for item in sol.detailed_cost if item["name"] == "Mayer.MINIMIZE_TIME")["cost_value_weighted"]
    #     mayer_cost_unweighted = next(item for item in sol.detailed_cost if item["name"] == "Mayer.MINIMIZE_TIME")["cost_value"]

    #     path = "../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/results/"
        
    #     #Save the output of the optimization
    #     with open(os.path.join(path, "pendulum_data_100_shooting_start_constraint.pkl"), "wb") as file:
    #         data = {"q": q,
    #                 "qdot": qdot,
    #                 "tau": tau,
    #                 "cost": sol.cost,
    #                 "real_time_to_optimize": sol.real_time_to_optimize,
    #                 "q_integrated": q_integrated,
    #                 "qdot_integrated": qdot_integrated,
    #                 "lagrange_control_cost": lagrange_cost_weighted,
    #                 "mayer_time_cost": mayer_cost_weighted,
    #                 "lagrange_control_cost_unweighted": lagrange_cost_unweighted,
    #                 "mayer_time_cost_unweighted": mayer_cost_unweighted,
    #                 "n_shooting" : n_shooting}
    #         pickle.dump(data, file)
        
    #     # # # Save the solution for future use, you will only need to do sol.ocp = prepare_ocp() to get the same solution object as above.
    #     # # with open(os.path.join(path,"pendulum_sol.pkl"), "wb") as file:
    #     # #      del sol.ocp
    #     # #      pickle.dump(sol, file)
            
    #     print("Fichiers sauvegardés dans :", os.getcwd())
        
    # --- Animate the solution --- #
    #viewer = "bioviz"
    viewer = "pyorerun"
    sol.animate(n_frames=0, viewer=viewer, show_now=True)
    
    
    # --- Show the results graph --- #
    sol.print_cost()
    sol.graphs(show_bounds=True,show_now=True)
    
    
  


if __name__ == "__main__":


    main()

