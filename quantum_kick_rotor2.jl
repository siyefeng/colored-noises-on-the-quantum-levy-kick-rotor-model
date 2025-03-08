using ProgressMeter, Distributed
addprocs(3 - nprocs());


@everywhere begin
    using SymPy, PyCall, SpecialFunctions, QuadGK, MAT, Statistics, FFTW, Distributions, Random, StatsBase, SharedArrays, LinearAlgebra
    include("colornoise.jl")

    N = 200;                #initial setting
    scale = 200;
    Dim = 2 * scale + 1;
    T0 = 2 * π - 0.1;
    K0 = 7.5;  
    repeat = 10;



    # norm = 0;                            # normalized initial state
    # for n in -15 : 15
    #     ψ0[scale + 1 + n, 1] = exp(-(n / 5) ^ 2 / 2) / sqrt(2 * pi);
    #     global norm += ψ0[scale + 1 + n, 1] ^ 2;
    # end
    # ψ0[:, 1] /= norm;
    # ψ0[scale + 1, 1] = 1;

    
end




cos_element = zeros(ComplexF64, Dim, Dim);
for m = 1 : Dim
    for n = 1 : Dim
        cos_element[m, n] = abs(m - n) == 0 ? 0.5 : (abs(m - n) == 2 ? 0.25 : 0);
    end
end
A = eigen(cos_element);
cos_eigenvalue = SharedArray(A.values);
cos_eigenstate = SharedArray(A.vectors);

U0 = SharedArray(zeros(ComplexF64, Dim, Dim));                   # ψ without noise
# U00 = SharedArray(zeros(ComplexF64, Dim, Dim)); 
# F0 = SharedArray(zeros(ComplexF64, Dim, Dim));  
@sync @distributed for m in -scale : scale                 
    for n in -scale : scale
        for α = 1 : Dim
            U0[m + scale + 1, n + scale + 1] += exp(- 1im * m ^ 2 * T0 / 2.0) * exp(-1im * K0 * cos_eigenvalue[α]) * cos_eigenstate[m + scale + 1, α] * cos_eigenstate[n + scale + 1, α]';
        end 
    end
end


# @sync @distributed for m in -scale : scale                 
#     for n in -scale : scale
#         if (m - n) % 2 == 0
#             U0[m + scale + 1, n + scale + 1] = exp(- 1im * m ^ 2 * T0 / 2.0) * exp(-1im * K0 / 2) * (-1im) ^((n - m) / 2) * besselj((n - m) / 2, K0 / 2);
#         end
#     end
# end

# @sync @distributed for m in -scale : scale                 
#     for n in -scale : scale
#         U0[m + scale + 1, n + scale + 1] = (-1.0im) ^ (n - m) * exp(- 1im * m ^ 2 * T0 / 2.0) * besselj(n - m, K0);
#     end
# end
heatmap(abs.(U0))
heatmap(abs.(U0' * U0))
heatmap(abs.(U0 - U00))
plot(abs.(eigenvalue) .- 1)
for m = 1 : Dim
    if abs(eigenvalue[m]) < 0.95
        println(m, "   ");
    end
end

F = eigen(U0);
eigenvalue = SharedArray(F.values);
eigenstate = SharedArray(F.vectors);
eigenergy = SharedArray(1im * log.(eigenvalue));



ave0 = SharedArray(zeros(ComplexF64, N));
varp0 = SharedArray(zeros(ComplexF64, N));

@sync @distributed for r = 1 : Dim
    ψ0 = zeros(ComplexF64, Dim, N);
    ψ0[:, 1] = eigenstate[:, r];
    avep02 = zeros(ComplexF64, N); avep0 = zeros(ComplexF64, N);
    varp0 = zeros(ComplexF64, N);

    for t in 1 : N - 1
            ψ0[:, t + 1] = U0 * ψ0[:, t];
    end
    for t = 1 : N
        for m = - scale : scale
            avep02[t] += m ^ 2 * abs2.(ψ0[m + scale + 1, t]); 
            avep0[t] += m * abs2.(ψ0[m + scale + 1, t]);
        end
        varp0[t] = avep02[t] / Dim - (avep0[t] / Dim) ^ 2;
        
    end
    ave0 .+= varp0; 

end
varp0 = abs.(ave0 / Dim);
varp0
plot(abs.(varp0))
# eigenvalue[40] * ψ0[:, 1] == 


# surface(abs.(ψ0))
# # heatmap(abs.(ψ0))
# s = zeros(ComplexF64, Dim, N);
# for t = 1 : N
#     s[:, t] .= ψ0[:, t] - eigenvalue[40] ^ (t - 1) * ψ0[:, 1]

# end    
# heatmap(abs.(s))
# surface(abs.(s));




ψ = zeros(ComplexF64, Dim, N);
ψ[scale + 1, 1] = 1;
# norm = 0;                            # normalized initial state
# for n in -15 : 15
#     ψ[scale + 1 + n, 1] = exp(-(n / 5) ^ 2 / 2) / sqrt(2 * pi);
#     global norm += ψ[scale + 1 + n, 1] ^ 2;
# end
# ψ[:, 1] /= norm;
ψ = wavefunction_cal(bluenoise, ψ[:, 1])
surface(abs.(ψ))
plot(abs.(ψ[:, 40]))
varp = varience_cal(ψ)
plot(abs.(varp))



@everywhere function var_cal(noise::Function, init_st)
    K = noise_choice(noise);
    ψ = zeros(ComplexF64, Dim, N);
    Um = zeros(ComplexF64, Dim, Dim);
    U = I(Dim);
    varp = zeros(ComplexF64, N);


    ψ[:, 1] = init_st;
    @sync @distributed for t in 1 : N - 1
        for m in - scale : scale
            for n in - scale : scale
                for α = 1 : Dim
                    Um[m + scale + 1, n + scale + 1] += exp(- 1im * m ^ 2 * T0 / 2.0) * exp(-1im * K[t] * cos_eigenvalue[α]) * cos_eigenstate[m + scale + 1, α] * cos_eigenstate[n + scale + 1, α]';
                end
            end
        end
        U *= Um;
        ψ[:, t + 1] = Um * ψ[:, t];

        varp[t + 1] = ψ[:, t + 1]' * P ^ 2 * ψ[:, t + 1] + ψ[:, 1]' * P ^ 2 * ψ[:, 1] - ψ[:, t + 1]' * P * U * P * ψ[:, 1] - ψ[:, 1]' * P * U' * P * ψ[:, t + 1];

        Um .= 0;
    end

    return varp;
end

varp = zeros(ComplexF64, N);
for r = 1 : Dim
    varp += var_cal(whitenoise, eigenstate[:, r]);
end
varp /= Dim;
plot(real.(varp))
plot!(imag.(varp))


@everywhere function varience_cal(ψ::Matrix)
    avep2 = zeros(ComplexF64, N);
    avep = zeros(ComplexF64, N);
    varp = zeros(ComplexF64, N);

    for t = 1 : N
        for m = - scale : scale
            avep2[t] += m ^ 2 * abs2.(ψ[m + scale + 1, t]);
            avep[t] += m * abs2.(ψ[m + scale + 1, t]);
        end
        varp[t] = avep2[t] / Dim - (avep[t] / Dim) ^ 2;
    end
    return varp;
end


ave = SharedArray(zeros(ComplexF64, N, 5)); varp = SharedArray(zeros(N, 5));

@sync @distributed for n = 1 : repeat
    ψ = zeros(ComplexF64, Dim, N);
    ave[:, 1] .= 0;
    for r = 1 : Dim
        ψ = wavefunction_cal(bluenoise, eigenstate[:, r]);
        ave[:, 1] += varience_cal(ψ);
    end
    varp[:, 1] += abs.(ave[:, 1] / Dim);
end
varp[:, 1] /= repeat;
varp[:, 1]
plot(varp[:, 1])


@sync @distributed for n = 1 : repeat
    ψ = zeros(ComplexF64, Dim, N);
    ave[:, 2] .= 0;
    for r = 1 : Dim
        ψ = wavefunction_cal(bluenoise, eigenstate[:, r]);
        ave[:, 2] += varience_cal(ψ);
    end
    varp[:, 2] += abs.(ave[:, 2] / Dim);
end
varp[:, 2] /= repeat;

@sync @distributed for n = 1 : repeat
    ψ = zeros(ComplexF64, Dim, N);
    ave[:, 3] .= 0;
    for r = 1 : Dim
        ψ = wavefunction_cal(rednoise, eigenstate[:, r]);
        ave[:, 3] += varience_cal(ψ);
    end
    varp[:, 3] += abs.(ave[:, 3] / Dim);
end
varp[:, 3] /= repeat;

@sync @distributed for n = 1 : repeat
    ψ = zeros(ComplexF64, Dim, N);
    ave[:, 4] .= 0;
    for r = 1 : Dim
        ψ = wavefunction_cal(pinknoise, eigenstate[:, r]);
        ave[:, 4] += varience_cal(ψ);
    end
    varp[:, 4] += abs.(ave[:, 4] / Dim);
end
varp[:, 4] /= repeat;

@sync @distributed for n = 1 : repeat
    ψ = zeros(ComplexF64, Dim, N);
    ave[:, 5] .= 0;
    for r = 1 : Dim
        ψ = wavefunction_cal(violetnoise, eigenstate[:, r]);
        ave[:, 5] += varience_cal(ψ);
    end
    varp[:, 5] += abs.(ave[:, 5] / Dim);
end
varp[:, 5] /= repeat;





matwrite("quantumkick2.mat", Dict("N" => N, "scale" => scale, "U0" => U0, "K0" => K0, "varp0" => varp0, "varp" => varp, "repeat" => repeat ));

1
# ave2 = SharedArray(zeros(ComplexF64, Dim, N));                               #levy interval with bluenoise
# # ψ2 = SharedArray(zeros(ComplexF64, Dim, N));
# @sync @distributed for num = 1 : repeat
#     K2 = K0 * ones(N);
#     U2 = zeros(ComplexF64, Dim, Dim);
#     ψ2 = zeros(ComplexF64, Dim, N);
#     ψ2[:, 1] = ψ0[:, 1];


#     t = 0;
#     time = [t];
#     while t <= N
#         interval = rand(1 : 10);
#         if rand() < levy_probility(interval)
#             t += interval;
#             push!(time, t);
#         end
#     end
#     popfirst!(time); pop!(time);
#     k2 = bluenoise(length(time));
#     for t = 1 : length(time)
#         K2[time[t]] += k2[t];
#     end

#     for t in 1 : N - 1
#         for m in -scale : scale
#             for n in -scale : scale
#                 if (m - n) % 2 == 0
#                     U2[m + scale + 1, n + scale + 1] = exp(- 1im * m ^ 2 * T0 / (2.0)) * exp(- 1.0im * K2[t] / 2.0) * (- 1.0im) ^ ((n - m) / 2.0) * besselj((n - m) / 2, K2[t] / (2.0)); 
#                 end
#             end
#         end
#         for m in 1 : Dim
#             ψ2[m, t + 1] += sum(U2[m, :] .* ψ2[:, t]);
#         end
#     end
#     ψ2 = abs.(ψ2);
#     ave2 .+= ψ2;
# end
# ψ2 = ave2 ./ repeat;
# ψ2 = abs.(ψ2);


