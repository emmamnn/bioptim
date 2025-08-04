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
    DynamicsOptionsList,
    DynamicsOptions,
    MultinodeConstraintFcn,
    TorqueDynamics,
    TorqueBiorbdModel,
)

import shutil


class DynamicModel(TorqueBiorbdModel):
    def __init__(self, biorbd_model_path):
        super().__init__(biorbd_model_path)

    def dynamics(
            self,
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

        stiffness = 14160
        tau[0] -= stiffness * q[0]  # damping

        qddot = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

        return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=None)


def prepare_ocp(
        biorbd_model_path: str,
        final_time: tuple,
        n_shooting: tuple,
        min_time: float,
        max_time: float,
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
    bio_model = (DynamicModel(biorbd_model_path), DynamicModel(biorbd_model_path))

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=0.0001, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight, min_bound=min_time, max_bound=max_time,
                            phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=10, key="q", phase=0)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=0.0001, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight, min_bound=min_time, max_bound=max_time,
                            phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=10, key="q", phase=1)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.RK4(n_integration_steps=5)
    ))
    dynamics.add(DynamicsOptions(
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.RK4(n_integration_steps=5)
    ))

    # Constraints
    constraint_list = ConstraintList()
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="LowerBarMarker",
                        second_marker="LastMarker", min_bound=0.02, max_bound=np.inf, axes=Axis.Y, phase=1)
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="LowerBarMarker",
                        second_marker="LastMarker", axes=Axis.Z, phase=0, min_bound=0.02, max_bound=0.02)
    # constraint_list.add(ConstraintFcn.TRACK_STATE, key="q", index=2, node=Node.ALL, target=0, phase = 0)

    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)

    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["qdot"][:, 0] = 0  # speeds start at 0

    x_bounds[1]["q"][:, -1] = 0
    x_bounds[1]["q"][1, -1] = 2 * np.pi  # ends with hands 360° rotated

    # Define control path bounds
    tau_min, tau_max = (-1000, 1000)
    n_tau = bio_model[0].nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=1)

    # u_bounds = BoundsList()
    # u_bounds["tau"] = [0, 50, -100, -260, -650, -50, -1000, -700], [0, 50, 100, 260, 650, 50, 1000, 700]
    # u_bounds.add("tau", u_bounds["tau"], phase = 0)
    # u_bounds.add("tau", u_bounds["tau"], phase=1)

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
        n_shooting,
        final_time,
        dynamics=dynamics,
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

    # RAC="../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/"
    RAC = "getting_started/"
    # path="getting_started/results/"

    path = "~/Documents/GIT_Emma/bioptim/bioptim/examples/getting_started/results/"
    # --- to start from the solution without constraint --- #

    import os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
    n_shooting = (50, 100)

    with open(os.path.join(RESULTS_DIR, "Athlete14_2_feet2.pkl"), "rb") as file:
        prev_sol_data = pickle.load(file)

    qs = prev_sol_data["q"]
    qdots = prev_sol_data["qdot"]
    taus = prev_sol_data["tau"]
    n0, n1 = n_shooting[0] + 1, n_shooting[1] + 1

    # #remove first dof (Tz)
    # qs = qs[1:,:]
    # qdots = qdots[1:,:]
    # taus=taus[1:,:]

    # add a lign of 0 because there is 1 more DoF in the new model
    # qs = np.vstack((qs, np.zeros((1, np.shape(qs)[1]))))
    # qdots = np.vstack((qdots, np.zeros((1, np.shape(qdots)[1]))))
    # taus = np.vstack((taus, np.zeros((1, np.shape(taus)[1]))))

    # Initial guess with InterpolationType.EACH_FRAME
    x_init = InitialGuessList()
    x_init.add("q", qs[:, :n0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", qs[:, n0:], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", qdots[:, :n0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", qdots[:, n0:], interpolation=InterpolationType.EACH_FRAME, phase=1)

    u_init = InitialGuessList()
    u_init.add("tau", taus[:, :n0 - 1], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", taus[:, n0 - 1:], interpolation=InterpolationType.EACH_FRAME, phase=1)

    # n0, n1 = n_shooting[0] + 1, n_shooting[1] + 1
    #
    # qs = np.linspace(0, 2*np.pi, 152)
    # qs = np.vstack((np.zeros((1, 152)), qs.T, np.zeros((2, 152))))
    #
    # x_init = InitialGuessList()
    # x_init.add("q", qs[:,:n0], interpolation=InterpolationType.EACH_FRAME, phase = 0)
    # x_init.add("q", qs[:,n0:], interpolation=InterpolationType.EACH_FRAME, phase = 1)
    #
    # u_init = InitialGuessList()
    # u_init.add("tau", [0] * 4, phase=0)
    # u_init.add("tau", [0] * 4, phase=1)

    print("prepare ocp")
    filename = "/models/Athlete_14_inverse2.bioMod"
    print("model : ", filename)
    # ocp = prepare_ocp(biorbd_model_path=CURRENT_DIR + filename, final_time=(1,3), n_shooting=n_shooting, min_time=0.02, max_time=4, weight=0.01, n_threads=32, use_sx=False)
    ocp = prepare_ocp(biorbd_model_path=CURRENT_DIR + filename, final_time=(1, 3), n_shooting=n_shooting, min_time=0.02,
                      max_time=2, weight=0.1, n_threads=32, u_init=u_init, x_init=x_init, use_sx=False)
    print("ocp ready")

    # --- Live plots --- #
    ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT()
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(20000)
    print("start solving")
    sol = ocp.solve(solver)
    print("solving finished")

    qs = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['q']
    qdots = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['qdot']
    for i in range(1, len(sol.decision_states(to_merge=[SolutionMerge.NODES]))):
        qs = np.hstack((qs, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['q']))
        qdots = np.hstack((qdots, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['qdot']))

    taus = sol.decision_controls(to_merge=[SolutionMerge.NODES])[0]['tau']

    for i in range(1, len(sol.decision_controls(to_merge=[SolutionMerge.NODES]))):
        taus = np.hstack((taus, sol.decision_controls(to_merge=[SolutionMerge.NODES])[i]['tau']))

    # # --- Save the solution --- #
    save = False

    if save == True:
        # Save the output of the optimization
        with open(os.path.join(RESULTS_DIR, "non.pkl"), "wb") as file:
            data = {"q": qs,
                    "qdot": qdots,
                    "tau": taus,
                    }
            pickle.dump(data, file)

        print("Fichier sauvegardé")

    # --- Animate the solution --- #
    # viewer = "bioviz"
    viewer = "pyorerun"
    sol.animate(n_frames=0, viewer=viewer, show_now=True)

    # --- Show the results graph --- #
    sol.print_cost()
    sol.graphs(show_bounds=True, show_now=True)


if __name__ == "__main__":
    main()


