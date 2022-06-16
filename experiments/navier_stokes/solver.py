"""
Solves the incompressible Navier Stokes equations in a lid-driven cavity
scenario using Finite Differences, explicit timestepping and Chorin's Projection.

Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

Incompressibility:  ∇ ⋅ u = 0


u:  Velocity (2d vector)
p:  Pressure
f:  Forcing (here =0)
ν:  Kinematic Viscosity
ρ:  Density
t:  Time
∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²: Laplace Operator

----

Lid-Driven Cavity Scenario:


                            ------>>>>> u_top

          1 +-------------------------------------------------+
            |                                                 |
            |             *                      *            |
            |          *           *    *    *                |
        0.8 |                                                 |
            |                                 *               |
            |     *       *                                   |
            |                      *     *                    |
        0.6 |                                            *    |
u = 0       |      *                             *            |   u = 0
v = 0       |                             *                   |   v = 0
            |                     *                           |
            |           *                *         *          |
        0.4 |                                                 |
            |                                                 |
            |      *            *             *               |
            |           *                             *       |
        0.2 |                       *           *             |
            |                               *                 |
            |  *          *      *                 *       *  |
            |                            *                    |
          0 +-------------------------------------------------+
            0        0.2       0.4       0.6       0.8        1

                                    u = 0
                                    v = 0

* Velocity and pressure have zero initial condition.
* Homogeneous Dirichlet Boundary Conditions everywhere except for horizontal
  velocity at top. It is driven by an external flow.

-----

Solution strategy:   (Projection Method: Chorin's Splitting)

1. Solve Momentum equation without pressure gradient for tentative velocity
   (with given Boundary Conditions)

    ∂u/∂t + (u ⋅ ∇) u = ν ∇²u

2. Solve pressure poisson equation for pressure at next point in time
   (with homogeneous Neumann Boundary Conditions everywhere except for
   the top, where it is homogeneous Dirichlet)

    ∇²p = ρ/Δt ∇ ⋅ u

3. Correct the velocities (and again enforce the Velocity Boundary Conditions)

    u ← u − Δt/ρ ∇ p

-----

    Expected Outcome: After some time a swirling motion will take place

          1 +-------------------------------------------------+
            |                                                 |
            |                                                 |
            |                                                 |
        0.8 |                                                 |
            |                      *-->*                      |
            |                ******     ******                |
            |              **                 **              |
        0.6 |             *                     *             |
            |             *                      *            |
            |            *                        *           |
            |            *                       *            |
            |             *                     *             |
        0.4 |             *                     *             |
            |              **                 **              |
            |                ******     ******                |
            |                      *<--*                      |
        0.2 |                                                 |
            |                                                 |
            |                                                 |
            |                                                 |
          0 +-------------------------------------------------+
            0        0.2       0.4       0.6       0.8        1

------

Strategy in index notation

u = [u, v]
x = [x, y]

1. Solve tentative velocity + velocity BC

    ∂u/∂t + u ∂u/∂x + v ∂u/∂y = ν ∂²u/∂x² + ν ∂²u/∂y²

    ∂v/∂t + u ∂v/∂x + v ∂v/∂y = ν ∂²v/∂x² + ν ∂²v/∂y²

2. Solve pressure poisson + pressure BC

    ∂²p/∂x² + ∂²p/∂y² = ρ/Δt (∂u/∂x + ∂v/∂y)

3. Correct velocity + velocity BC

    u ← u − Δt/ρ ∂p/∂x

    v ← v − Δt/ρ ∂p/∂y

------

IMPORTANT: Take care to select a timestep that ensures stability
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

DOMAIN_SIZE = 1.0
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0
N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5


def solve(n_points=512, n_iterations=200000, time_length=1, re=1000, record_steps=10000,
          device=torch.device("cuda")):
    time_step_length = time_length / n_iterations
    if time_length == 0:
        n_iterations = 0
    kinematic_viscosity = 1 / re
    element_length = DOMAIN_SIZE / (n_points - 1)
    x = torch.linspace(0.0, DOMAIN_SIZE, n_points, device=device)
    y = torch.linspace(0.0, DOMAIN_SIZE, n_points, device=device)

    X, Y = torch.meshgrid(x, y, indexing='ij')

    u_prev = torch.zeros_like(X)
    v_prev = torch.zeros_like(X)
    p_prev = torch.zeros_like(X)

    p_next = None

    def central_difference_x(f):
        diff = torch.zeros_like(f)
        diff[1:-1, 1:-1] = (
                                   f[1:-1, 2:]
                                   -
                                   f[1:-1, 0:-2]
                           ) / (
                                   2 * element_length
                           )
        return diff

    def central_difference_y(f):
        diff = torch.zeros_like(f)
        diff[1:-1, 1:-1] = (
                                   f[2:, 1:-1]
                                   -
                                   f[0:-2, 1:-1]
                           ) / (
                                   2 * element_length
                           )
        return diff

    def laplace(f):
        diff = torch.zeros_like(f)
        diff[1:-1, 1:-1] = (
                                   f[1:-1, 0:-2]
                                   +
                                   f[0:-2, 1:-1]
                                   -
                                   4
                                   *
                                   f[1:-1, 1:-1]
                                   +
                                   f[1:-1, 2:]
                                   +
                                   f[2:, 1:-1]
                           ) / (
                                   element_length ** 2
                           )
        return diff

    maximum_possible_time_step_length = (
            0.5 * element_length ** 2 / kinematic_viscosity
    )
    if time_step_length > STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:
        raise RuntimeError("Stability is not guarenteed")

    pbar = tqdm(total=n_iterations)
    results = {}
    for i in range(n_iterations):
        d_u_prev__d_x = central_difference_x(u_prev)
        d_u_prev__d_y = central_difference_y(u_prev)
        d_v_prev__d_x = central_difference_x(v_prev)
        d_v_prev__d_y = central_difference_y(v_prev)
        laplace__u_prev = laplace(u_prev)
        laplace__v_prev = laplace(v_prev)

        # Perform a tentative step by solving the momentum equation without the
        # pressure gradient
        u_tent = (
                u_prev
                +
                time_step_length * (
                        -
                        (
                                u_prev * d_u_prev__d_x
                                +
                                v_prev * d_u_prev__d_y
                        )
                        +
                        kinematic_viscosity * laplace__u_prev
                )
        )
        v_tent = (
                v_prev
                +
                time_step_length * (
                        -
                        (
                                u_prev * d_v_prev__d_x
                                +
                                v_prev * d_v_prev__d_y
                        )
                        +
                        kinematic_viscosity * laplace__v_prev
                )
        )

        # Velocity Boundary Conditions: Homogeneous Dirichlet BC everywhere
        # except for the horizontal velocity at the top, which is prescribed
        u_tent[0, :] = 0.0
        u_tent[:, 0] = 0.0
        u_tent[:, -1] = 0.0
        u_tent[-1, :] = HORIZONTAL_VELOCITY_TOP
        v_tent[0, :] = 0.0
        v_tent[:, 0] = 0.0
        v_tent[:, -1] = 0.0
        v_tent[-1, :] = 0.0

        d_u_tent__d_x = central_difference_x(u_tent)
        d_v_tent__d_y = central_difference_y(v_tent)

        # Compute a pressure correction by solving the pressure-poisson equation
        rhs = (
                DENSITY / time_step_length
                *
                (
                        d_u_tent__d_x
                        +
                        d_v_tent__d_y
                )
        )

        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = torch.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1 / 4 * (
                    +
                    p_prev[1:-1, 0:-2]
                    +
                    p_prev[0:-2, 1:-1]
                    +
                    p_prev[1:-1, 2:]
                    +
                    p_prev[2:, 1:-1]
                    -
                    element_length ** 2
                    *
                    rhs[1:-1, 1:-1]
            )

            # Pressure Boundary Conditions: Homogeneous Neumann Boundary
            # Conditions everywhere except for the top, where it is a
            # homogeneous Dirichlet BC
            p_next[:, -1] = p_next[:, -2]
            p_next[0, :] = p_next[1, :]
            p_next[:, 0] = p_next[:, 1]
            p_next[-1, :] = 0.0

            p_prev = p_next

        d_p_next__d_x = central_difference_x(p_next)
        d_p_next__d_y = central_difference_y(p_next)

        # Correct the velocities such that the fluid stays incompressible
        u_next = (
                u_tent
                -
                time_step_length / DENSITY
                *
                d_p_next__d_x
        )
        v_next = (
                v_tent
                -
                time_step_length / DENSITY
                *
                d_p_next__d_y
        )

        # Velocity Boundary Conditions: Homogeneous Dirichlet BC everywhere
        # except for the horizontal velocity at the top, which is prescribed
        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = HORIZONTAL_VELOCITY_TOP
        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0

        if (i + 1) % record_steps == 0:
            time = (i + 1) * time_step_length
            results[time] = {}
            results[time]["u"] = u_next.detach().cpu().numpy()
            results[time]["v"] = v_next.detach().cpu().numpy()
            results[time]["p"] = p_next.detach().cpu().numpy()
            # levels = 20
            # x, y = np.meshgrid(np.linspace(0, 1, n_points), np.linspace(0, 1, n_points))
            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
            # ax.contour(x, y, results[time]["v"], colors="k", linewidths=0.2, levels=levels)
            # ax.contourf(x, y, results[time]["v"], cmap="rainbow", levels=levels)
            # ax.set_xlim(0, 1)
            # ax.set_ylim(0, 1)
            # m = plt.cm.ScalarMappable(cmap="rainbow")
            # m.set_array(results[time]["v"])
            # plt.colorbar(m, pad=0.03, aspect=25, format="%.0e")
            # plt.savefig("{:.1f}_{:.3f}_v.png".format(re, time))

        # Advance in time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next
        pbar.update()
    return results


if __name__ == "__main__":
    solve()
