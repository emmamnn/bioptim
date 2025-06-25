import numpy as np
import matplotlib.pyplot as plt 
from casadi import MX, vertcat, sign
import os
import rerun as rr
from bioptim import (
    Node,
    ConstraintList,
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
    SolutionMerge
)

import shutil



def custom_func_marker_y_above_marker(controller: PenaltyController, marker_name: str, reference_marker_name: str) -> MX:
    """
    User-defined constraint that ensures a marker's Y-position is greater than or equal to another marker's Y-position.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    marker_name: str
        The marker to be constrained (e.g., "marker_6")
    reference_marker_name: str
        The reference marker (e.g., "LowerBar")

    Returns
    -------
    MX
        The constraint value: should be >= 0 to satisfy the condition
    """
    
    # Get the index of the markers from their name
    marker_i = controller.model.marker_index(marker_name) #marker_6
    ref = controller.model.marker_index(reference_marker_name) #LowerBar

    # compute the position of the markers using the markers function from the BioModel (here a BiorbdModel)
    markers = controller.model.markers()(controller.states["q"].cx, controller.parameters.cx)
    markers_diff = markers[:, marker_i] - markers[:, ref]

    return markers_diff[1] #composante Y 



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
    #tau[1] -= sign(q[1]) * stiffness * q[1]  # damping


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
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
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

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True,)

    # Dynamics
    dynamics = Dynamics(
        custom_configure,
        dynamic_function=custom_dynamic,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )


    
    #Constraints
    constraint_list = ConstraintList()
    #don't forget min and max bound so it is a inequality constraint (0<= diff <= inf) 
    constraint_list.add(custom_func_marker_y_above_marker, node=Node.ALL, marker_name="marker_6", reference_marker_name = "LowerBar", min_bound = 0, max_bound = np.inf)


    #constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", index=0, node=Node.START, min_bound=-0.020, max_bound=-0.019)
    # constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", index=0, node=Node.START, min_bound=0, max_bound=0)
    # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", index=1, node=Node.START, min_bound=0, max_bound=0)
    # constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", index=1, node=Node.START, min_bound=0, max_bound=0)
    # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", index=2, node=Node.START, min_bound=0, max_bound=0)
    # constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", index=2, node=Node.START, min_bound=0, max_bound=0)
    #
    # # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", index=0, node=Node.END, min_bound=0, max_bound=0)
    # # constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", index=0, node=Node.END, min_bound=0, max_bound=0)
    # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", index=1, node=Node.END, min_bound=np.pi, max_bound=np.pi)
    # #constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", index=1, node=Node.END, min_bound=0, max_bound=0)
    # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", index=2, node=Node.END, min_bound=0, max_bound=0)
    #constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", index=2, node=Node.END, min_bound=0, max_bound=0)

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    x_bounds["q"][:, 0] = 0 #rotations start at 0
    x_bounds["qdot"][:, 0] = 0 #speeds start at 0
    x_bounds["q"][:, -1] = 0
    x_bounds["q"][1,-1] = np.pi #ends with first pendulum 180 degrees rotated



    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0.01] * bio_model.nb_q
    x_init["qdot"] = [0.01] * bio_model.nb_qdot


    # Define control path bounds
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    #u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate
    #u_bounds["tau"][0, :] = 0  # ...but remove the capability to actively translate verticaly
    #u_bounds["tau"][0, :] = -100
    #u_bounds["tau"][1, :] = 100

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

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
    RAC="../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/"
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path=RAC+"examples/getting_started/models/triple_pendulum.bioMod", final_time=4, n_shooting=30, n_threads=2)

    # --- Live plots --- #
    ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration
    # ocp.add_plot_check_conditioning()  # This will display the conditioning of the problem at the current iteration
    # ocp.add_plot_ipopt_outputs()  # This will display the solver's output at the current iteration



    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    # Default is OnlineOptim.MULTIPROCESS on Linux, OnlineOptim.MULTIPROCESS_SERVER on Windows and None on MacOS
    # To see the graphs on MacOS, one must run the server manually (see resources/plotting_server.py)
    #sol = ocp.solve()
    sol = ocp.solve(Solver.IPOPT(online_optim=OnlineOptim.DEFAULT))

    
    # --- Animate the solution --- #
    #viewer = "bioviz"
    viewer = "pyorerun"
    sol.animate(n_frames=0, viewer=viewer, show_now=True)
    
    
    # --- Show the results graph --- #
    sol.print_cost()
    sol.graphs(show_bounds=True,show_now=True)


if __name__ == "__main__":


    main()
