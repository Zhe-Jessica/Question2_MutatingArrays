######################################
# I am trying to use sciml_train in Flux to train a neural network model using the data from an engineering system. 
# The data contains many groups of measurements with different initial states. 
# I gave these initial states to the neural network individually and calculated the loos function by summing the mean squared error of all measurements.
######################################

using CSV, DataFrames, StatsBase
af = CSV.read("dataset.csv", DataFrame, header = false)
data = Array{Float32}(af)

using DiffEqFlux, Flux, Optimization, StatsBase, DifferentialEquations

dudt2 = Chain(Dense(3,16,relu),
           Dense(16,23,relu),
           Dense(23,36,relu),
           Dense(36,16,relu),
           Dense(16,3))           

tspan0 = range(0.0f0,0.05,5)
tspan1 = range(0.051f0,0.19,5)
tspan2 = range(0.2,2.0f0,20)
tspan = (0.0f0, 2.0f0)
tsteps = vcat(tspan0, tspan1, tspan2)
para = [2.4e4, 0.15, 0.18] 

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps) 

save = Array{UnitRangeTransform{Float32, Vector{Float32}}}(undef, 200)

function loss_neuralode(p)
    loss = 0.0f0;
    for i = 1:2                     # I test two groups of measurements here            
        index = (i-1)*3+1;        
        ym = data[:, index:index+2]; # 30*3
        trans = fit(UnitRangeTransform, ym, dims=1)
        save[i] = trans
        ymn = Zygote.Buffer(StatsBase.transform(trans, ym))        # standarlization for training data

        pred = Array(prob_neuralode(data[1, index:index+2], p))
        nny = transpose(pred);
        nnyn = StatsBase.transform!(trans, nny)     # standarlization for the output of NN using the same transformation 

        loss = loss + sum(abs2, ymn .- nnyn) + 2 * sum(abs2, ymn[:, 2] .- nnyn[:, 2])
    end
  return loss
end

callback = function (p, l; doplot = true)
    display(l)
    return false
end
  
result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                            ADAM(0.05), cb = callback,
                                            maxiters = 2000)
  
using OrdinaryDiffEq, Optim
  result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                             result_neuralode.minimizer,
                                             BFGS(initial_stepnorm=0.01), cb = callback,maxiters=100)

o = 10;
index = (o-1)*3+1;                                             
pred = Array(prob_neuralode(data[1, index:index+2], result_neuralode.minimizer))

# plot current prediction against data
plt1 = scatter(tsteps, data[:,index], label = "data");
scatter!(plt1, tsteps, pred[1,:], label = "prediction");

plt2 = scatter(tsteps, data[:,index+1], label = "data");
scatter!(plt2, tsteps, pred[2,:], label = "prediction");

plt3 = scatter(tsteps, data[:, index+2], label = "data");
scatter!(plt3, tsteps, pred[3,:], label = "prediction");

display(plot(plt1, plt2, plt3, layout = (1, 3), size = (1500, 500)))

############################################## prediction
m0 = Float32[8650.; 3306.; 18250.]  # different inputs
# output of NN
pred1 = Array(prob_neuralode(m0, result_neuralode.minimizer))
# real outputs
function GLOWmodel(dm, m, p, t) 
  #= GLOWmodel: Summary of this function goes here
  %   This is a model of one gas lifted oil well. X(dim=4*1) includes 4
  %   differential variables (states) as belows:
  %   X=[m_gp(t), m_ga(t), m_gt(t), m_lt(t)]'
  %   DIM(theta): 1*4
  %   theta = [WC PI GOR wn]    wn=well number  =#
      
  m_ga, m_gt, m_lt = m
  PI, GOR, WC = p
  
  # # Define initial states                                                 %%
  # # extracting the states from X vector.
  # #  mass of the gas in the distribution pipeline
  # # m_gp = X(1);                                                    %% [kg]
  # #  mass of the gas in the annulus
  # m_ga = m[1];                                             # [kg]
  # #  mass of the gas in the tubing above the injection point
  # m_gt = m[2];                                            # [kg]
  # #  mass of the liquid (oil and water) in the tubing above the injection point
  # m_lt = m[3];                                           # [kg]
  
      
  # Define the constants
  #  Valve constants for gas lift choke valve (first element) and production choke valve (the second element). 
  N6 = 27.3; 
  Nb6 = 273;
  Z_pc = 0.7076;
  P_c = 200;
  # constat used in gas expansion formula
  aY = 0.66;
  # universal gas constant
  R = 8.314e-5;                                                   # [m3.bar/K.mol]
  # fixed temperature assumed for gas everywhere in system
  T = 280;                                                        # [K]
  # molar mass of the lift gas
  M = 20e-3;                                                      # [Kg/mol]
  # gravitational acceleration02]
  g = 9.81;                                                       # [m/s2]
  
  # gas injection valve constants 
  K = 68.43;                                                       # [(Kg.m3/bar)^.5/hr]
  
  # Densities of crude oil, water, and fluid mixture(oil, water, and gas)
  # Rho = [rho_o, rho_w, rho_r]
  Rho = [800, 1000, 700];   
  rho_r = 700;                                               # [Kg/m3]
  
  # PIPE DIMENSIONS
  # DIAMETERS
  # Lift gas distribution pipeline
  IDp = 3;                                                        # [in]    ***(my assumption)***
  #  Tubing inner and outer diameters                          
  IDt = 6.18;                                                     # [in]
  ODt = 7.64;                                                     # [in]
  # Annulus diameters
  IDa = 9.63;                                                     # [in]
  
  # Length
  # Lift gas distribution pipeline diameter and length
  # L_pt = 13000;                                                   # [m]
  # total length of tubings above the injection point
  L_tt = 2758;                                                      # [m]
  # vertical length of tubings above the injection point
  L_tv = 2271;                                                      # [m]
  # vertical length of tubings below the injection point
  L_rv = 114;                                                       # [m]
  # total length of annulus
  L_at  = L_tt;                                                   # [m]
  # vertical length of annulus
  L_av = L_tv;                                                    # [m]
  
  # Cross section areas
  # gas distribution pipeline
  A_p = pi*((IDp*0.0254)^2)/4;                                    # [m2]
  # Tubing
  A_t = pi*((IDt*0.0254)^2)/4;                                    # [m2]
  # Annulus
  A_a = (pi/4)*((IDa*0.0254)^2-(ODt*0.0254)^2);                   # [m2]
  
  # pressures all in [bar]
  P_s=30; # pressure of the common gathering manifold            # [bar]
  P_r = 150; # pressure of reservoir       # [bar]
  # minimum pressures in the gas distribution pipeline, in the annulus at
  # the point of injection, and in the tubing at the well head.
  # P_min = [P_cmin, P_ainjmin, P_whmin]. DIM(P_min):1*3
  P_min = [10, 10, 10];                                           # [bar]   %% ***(my assumption)***
  
  # Define the control inputs                                   
  #supplied lift gas
  w_gc = 8000;                                                   # [Sm3/hr]
  w_gc = w_gc*0.68; # converting Sm3/hr to kg/hr                 %% [kg/hr]
  
  # valve opening of the gas lift choke valve
  u = 80;
  # in case we have step change in u1
  # if t>1
  #     u=50;
  # end
  # u = 80;
  
  
  # valve opening of the production choke valve
  u2 = 100;
  # in case we have step change in u2
  # if t>25
  #     u2=55;
  # end
  
  # Define all the other variables                                              %%
  # Algebaic variables
  # Valve characteristic as a function of its opening for the lift gas
  # choke valve and the production choke valve. EQ(1) & EQ(2)
  
  Cv1 = 0.5*u-20;
  
  
  Cv2 = 0.5*u2-20;
  
  # Density of gas in the gas distribution pipeline: EQ(3)
  rho_gp = P_c*M/(Z_pc * R * T);                        # [Kg/m3] (3)
  
  
  # Density of gas in the annulus: EQ(4)
  rho_ga = m_ga / (A_a*L_at);                                   # [Kg/m3]
  
  # The average density of the mixture of liquid and gas in the tubing
  # above the injection point: EQ(5)
  rho_m = (m_gt+m_lt) / (A_t*L_tt);                             # [Kg/m3]
  
  # But, I assumed constant Z at 170 bar.
  Z_pa = 0.6816;
  
  # Pressure of gas in the annulus downstream the lift gas choke valve: EQ(9)
  P_a = (R*T/M) * (Z_pa*m_ga) / (A_a*L_at);                    # [bar]
  
  # Pressure of gas in the annulus upstream the gas injection valve. EQ(10) 
  P_ainj = P_a + 1e-5*g*(rho_ga*L_av);                           # [bar]
  
  # Gas expansion factor in the gas lift choke valve. EQ(11) 
  Y1 = 1 - aY*(P_c-P_a) / max(P_c,10);
  
  # Density of liquid (oil and water) in each well based on its water cut. EQ(12) 
  rho_l = Rho[2]*WC + Rho[1]*(1-WC);                  # [Kg/m3]
  
  # The volume of gas present in the tubing above the gas injection point. EQ(13)
  V_G = (A_t*L_tt) - (m_lt/rho_l);                             # [m3]
  
  # EQ(14)
  Z_PG = 0.6741;
  
  # dP_ft1 = 0.5*f_Dt1*L_tt1*rho_m1*(v_t1^2)*D_h1; Pressure loss due to
  # friction from the injection point to well head. But, I assumed no
  # friction. EQ(15)
  dP_ft = 0;                                        # [bar]
  
  # Pressure in the tubing downstream the gas injection valve. EQ(16)
  P_tinj = (R*T/M)*((Z_PG*m_gt)/V_G) + 0.5*g*1e-5*(rho_m*L_tv) + 0.5*dP_ft;                  # [bar]
  
  # Pressure in the tubing upstream the production choke valve. EQ(17)
  P_wh = (R*T/M)*((Z_PG*m_gt)/V_G) - 0.5*g*1e-5*(rho_m*L_tv) - 0.5*dP_ft;                  # [bar]
  
  # dP_fr1 = 0.5*f_Dr1*L_rt1*rho_r*(v_r1^2)*D_h1; Pressure loss due to
  # friction from the bottom hole to the injection point. But, I assumed no
  # friction. DIM(dP_fr):1*5. EQ(18)
  dP_fr = 0;                                        # [bar]
  
  # The bottom hole pressure or well flow pressure. EQ(19)
  P_wf = P_tinj + (rho_r*g*1e-5*L_rv) + dP_fr;                    # [bar]
  
  # Gas expansion factor in the gas injection valve. EQ(20)
  Y2 = 1 - aY*(P_ainj-P_tinj)/max(P_ainj,10);
  
  # Gas expansion factor in the production choke valve. EQ(21)
  Y3 = 1 - aY*(P_wh-P_s)/max(P_wh,10);
  
  # MASS FLOW RATES
  # Mass flow rate of the gas through the gas lift choke valve. EQ(22)
  w_ga = N6*Cv1*Y1*sqrt(rho_gp*max(P_c-P_a,0));            # [Kg/hr]
      
  # Mass flow rate of the gas injected into the tubing from the annulus. EQ(23)
  w_ginj = K*Y2*sqrt(rho_ga*max(P_ainj-P_tinj,0));         # [Kg/hr]
  
  # The mass flow rate of the liquid from the reservoir. EQ(24)
  w_lr = PI*max(P_r-P_wf,0);                              # [Kg/hr]
      
  # The mass flow rate of gas from the reservoir. EQ(25)
  w_gr = GOR*w_lr;                                        # [Kg/hr]
  
  # The mass flow rate of the mixture of gas and liquid from the production
  # choke valve. EQ(26)
  w_gop = Nb6*Cv2*Y3*sqrt(rho_m*max(P_wh-P_s,0));         # [Kg/hr]
  
  # Mass flow rate of gas through the production choke valve. EQ(27)
  w_gp = m_gt*w_gop/(m_gt+m_lt);                         # [Kg/hr]
  
  # Mass flow rate of liquid through the production choke valve. EQ(28)
  w_lp = m_lt*w_gop/(m_gt+m_lt);                         # [Kg/hr]
  
  # Oil compartment mass flow rate from liquid product considering water cut. EQ(29)
  w_op = (1-WC)*w_lp*Rho[1]/rho_l;                    # [Kg/hr]
      
  # Water compartment mass flow rate from liquid product considering water cut. EQ(30)
  w_wp = w_lp - w_op;                                  # [Kg/hr]
  
  # Define the derivatives                                              
  # Mass balance in gas distribution manifold:
  # This is just a single scaler, not a vector. EQ(31)
  # Mass balance in annulus. EQ(32)
  dm[1] = w_ga - w_ginj;                             # [Kg/hr]dm[1]
  
  # Mass balance for the gas in tubing and above the injection point. EQ(33)
  dm[2] = w_ginj + w_gr - w_gp;                      # [Kg/hr]dm[2]
  
  # Mass balance for the liquid in tubing and above the injection point. EQ(34)
  dm[3] = w_lr  - w_lp;                               # [Kg/hr]dm[3]
  # dm .= [dm_gadt,dm_gtdt,dm_ltdt]
end
prob2 = ODEProblem(GLOWmodel,m0,tspan,para)
sol2 = solve(prob2, Tsit5(), saveat = tsteps)
eva_data = Array(sol2)
################################################################ Validation
pl1 = plot(tsteps, pred1[1,:], label = "training");
plot!(pl1, tsteps, eva_data[1,:], ls=:dash, label = "data");

pl2 = plot(tsteps, pred1[2,:], label = "training");
plot!(pl2, tsteps, eva_data[2,:], ls=:dash, label = "data");
# scatter!(plt2, tsteps, pred[2,:], label = "prediction")

pl3 = plot(tsteps, pred1[3,:], label = "training");
plot!(pl3, tsteps, eva_data[3,:], ls=:dash, label = "data");
# scatter!(plt3, tsteps, pred[3,:], label = "prediction")

plot(pl1, pl2, pl3, layout = (1, 3), size = (1500, 500))
