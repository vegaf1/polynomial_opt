## Solve two body problem via semidefinite relaxation
# Lorenzo Shaikewitz, 3/30/2025
# 
# TODO: there are more constraints we can add

using LinearAlgebra, BlockDiagonals
using TSSOS, DynamicPolynomials
using Printf
import Plots

# Parameters that give working solutions:
# 1 rev, 41 knotpts, control obj => 17% gap
# 1 rev, 41 knotpts => < 1% gap
# 1 rev, 40 knotpts, correct scaling => 0.1% gap
# always infeasible with correct r0

## Generate problem
begin
    # gravitational parameter for the Earth 
    μ = 3.986004418e5 # [km^3/s^2]

    # ISS initial conditions
    q0 = [6791.0; 0; 0] # [km]
    v0 = [0; cosd(51.5)*7.66; sind(51.5)*7.66] # [km/s]

    # parameters
    revs = 1
    knot_pts = 40
    N = knot_pts*revs

    # find the period for 1 rev
    r_initial = norm(q0)
    iss_period = 2*π*sqrt(r_initial^3/μ)

    # time step
    period = revs*iss_period
    h = period/(N-1) # [s]

    # bounds
    rmin = 0.75*q0[1] # [km]
    rmax = 2.75*q0[1] # [km]
    umin = -20e-5 # [km/s^2]
    umax = -umin
end
# scaled by position
# begin
#     scale_position = q0[1] # [km]
#     scale_time = period # [s]
#     scale_velocity = scale_position / scale_time
#     scale_acceleration = scale_velocity / scale_time

#     # update variables
#     μ /= scale_position^3 / scale_time^2
#     h /= scale_time

#     rmin /= scale_position
#     rmax /= scale_position
    
#     umin /= scale_acceleration
#     umax /= scale_acceleration

#     q0 ./= scale_position
#     r0 = norm(q0)
#     v0 ./= scale_velocity
# end
# scaled by acceleration (want to scale all vars ∈ [-1, 1])
begin
    scale_time = period # [s]
    scale_acceleration = q0[1] / scale_time^2 * 75.1*28. # [km/s^2]
    scale_velocity = scale_acceleration * scale_time
    scale_position = scale_velocity * scale_time

    # update variables
    μ /= scale_position^3 / scale_time^2
    h /= scale_time
    
    rmin /= scale_position
    rmax /= scale_position

    umin /= scale_acceleration
    umax /= scale_acceleration

    q0 ./= scale_position
    r0 = norm(q0)
    v0 ./= scale_velocity
end

## Optimization problem
# VARIABLES
# position, velocity, acceleration
@polyvar q[1:3, 1:N]
@polyvar v[1:3, 1:N]
@polyvar a[1:3, 1:N]
# radius
@polyvar r[1:N] # = |q|
# control
@polyvar u[1:3, 1:N-1]
vars = [vec(q); vec(v); vec(a); r; vec(u)]

# OBJECTIVE
# minimize radius
obj = sum(r)
# minimize control
# obj += sum([u[:,i]'*u[:,i] for i = 1:N-1])

# EQUALITY CONSTRAINTS
eq = zeros(Polynomial{true, Float64}, 0)
# initial conditions
# append!(eq, r[1] - r0)
append!(eq, q[:,1] - q0)
append!(eq, v[:,1] - v0)

# dynamics between timesteps
for i = 1:N
    # acceleration dynamics
    append!(eq, r[i]^3*a[:,i] + μ*q[:,i])
    # radius auxillary variable
    append!(eq, [r[i]^2 - q[:,i]'*q[:,i]])
    # discrete updates
    if i < N
        # acceleration forward euler
        append!(eq, v[:,i+1] - (v[:,i] + h*(a[:,i] + u[:,i])))
        # velocity reverse euler
        append!(eq, q[:,i+1] - (q[:,i] + h*v[:,i+1]))
    end
end

# INEQUALITY CONSTRAINTS
ineq = zeros(Polynomial{true, Float64}, 0) # expr >= 0
# bound radii, r_i ∈ [rmin, rmax]
append!(ineq, r) # unnecessary but helps
append!(ineq, r .- rmin)
append!(ineq, rmax .- r)
# bound control u_ij ∈ [umin, umax]
append!(ineq, vec(u) .- umin)
append!(ineq, umax .- vec(u))
# bound all other variables ∈ [-1, 1]
# append!(ineq, vec(q) .- 1.)
# append!(ineq, 1. .- vec(q))
# append!(ineq, vec(v) .- 1.)
# append!(ineq, 1. .- vec(v))
# append!(ineq, vec(a) .- 1.)
# append!(ineq, 1. .- vec(a))

# SOLVE
pop = [obj; ineq; eq]
order = 2
opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", solution=true)
# opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, solution=true)

## Check solution
# Does it satisfy inequality constraints?
ineq_subed = [ineq_i(vars=>sol) for ineq_i in ineq]
if sum(ineq_subed .< -1e-6) > 0
    printstyled("$(sum(ineq_subed .< 0)) inequality constraint(s) violated!\n", color=:red)
end
eq_subed = [eq_i(vars=>sol) for eq_i in eq]
if sum(abs.(eq_subed) .> 1e-6) > 0
    printstyled("$(sum(abs.(eq_subed) .> 1e-6)) equality constraint(s) violated!\n", color=:red)
end

# Condition numbers
condition_numbers = cond.(data.moment)
println("Max condition number: $(maximum(condition_numbers))")

# Moment matrix
mom = BlockDiagonal(Matrix.(data.moment))
println("Moment rank: $(rank(mom, 1e-3))")

## Visualize solution
struct Solution
    q ::Matrix{Float64} # position
    v ::Matrix{Float64} # velocity
    a ::Matrix{Float64} # acceleration
    r ::Vector{Float64} # radius
    u ::Matrix{Float64}
end

# vars = [vec(q); vec(v); vec(a); r; vec(u)]
soln_scaled = Solution(
                reshape(sol[1:3*N], 3, N),
                reshape(sol[3*N+1:6*N], 3, N),
                reshape(sol[6*N+1:9*N], 3, N),
                reshape(sol[9*N+1:10*N], N),
                reshape(sol[10*N+1:end], 3, N-1)
            )

soln = Solution(
    soln_scaled.q*scale_position,
    soln_scaled.v*scale_velocity,
    soln_scaled.a*scale_acceleration,
    soln_scaled.r*scale_position,
    soln_scaled.u*scale_acceleration
)

# Plot control
plot_control = Plots.plot(soln.u'*1000, layout=(3,1), label=false)
Plots.plot!(ylabel=["ux (N)" "uy (N)" "uz (N)"], xlabel=["" "" "Time (s)"], title=["Control" "" ""])

# plot trajectory
plot_traj = Plots.plot(eachrow(soln.q)..., label=false)
Plots.plot!(xlabel="x (km)", ylabel="y (km)", zlabel="z (km)", title="Trajectory")
Plots.scatter!([0],[0],[0],label="Earth",ms=10,color=3)