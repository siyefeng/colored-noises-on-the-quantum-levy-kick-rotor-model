using ProgressMeter, Distributed                    #初始化
addprocs(4 - nprocs());


@everywhere begin
    using SymPy, PyCall, SpecialFunctions, QuadGK, MAT, Statistics, FFTW, Distributions, Random, StatsBase, SharedArrays, LinearAlgebra
    include("colornoise.jl")

    N = 10000;            
    scale = 100;
    Dim = 2 * scale + 1;
    T0 = 4.5;
    K0 = 7.5;  
    repeat = 300;                    #重复次数
    strength = 1.0;                  #噪声强度       

    # norm = 0;                            # normalized initial state
    # for n in -15 : 15
    #     ψ0[scale + 1 + n, 1] = exp(-(n / 5) ^ 2 / 2) / sqrt(2 * pi);
    #     global norm += ψ0[scale + 1 + n, 1] ^ 2;
    # end
    # ψ0[:, 1] /= norm;
    # ψ0[scale + 1, 1] = 1;

    P = zeros(Dim, Dim);
    for m = -scale : scale
        P[m + scale + 1, m + scale + 1] = m;
    end
    Up = exp(- 1im * P ^ 2 * T0 / 2.0);             #U0算符动量部分

    struct result
        varp
        prob 
        IPR
        OTOC   
    end

end

# cos_element = zeros(ComplexF64, Dim, Dim);                  # <s|cosθ|r> 
# for m = 1 : Dim
#     for n = 1 : Dim
#         cos_element[m, n] = abs(m - n) == 1 ? 0.5 : 0;
#     end
# end

cos_element = zeros(ComplexF64, Dim, Dim);                  # <s|cosθ²|r> 
for m = 1 : Dim
    for n = 1 : Dim
        cos_element[m, n] = abs(m - n) == 0 ? 0.5 : (abs(m - n) == 2 ? 0.25 : 0);
    end
end
A = eigen(cos_element);
cos_eigenvalue = SharedArray(A.values);
cos_eigenstate = SharedArray(A.vectors);

kick_eigen0 = diagm(exp.(-1im * K0 * cos_eigenvalue));              

U0 = SharedArray(zeros(ComplexF64, Dim, Dim));
U0 = Up * cos_eigenstate * kick_eigen0 * cos_eigenstate';           #生成U0算符，后三项生成kick项
F = eigen(U0);                                                      #计算本征值和本征向量，准能量
eigenvalue = SharedArray(F.values);
eigenstate = SharedArray(F.vectors);
eigenergy = SharedArray(1im * log.(eigenvalue));

cos4_element = zeros(ComplexF64, Dim, Dim);                                 # <s|cosθ⁴|r> 
for m = 1 : Dim
    for n = 1 : Dim
        cos4_element[m, n] = abs(m - n) == 0 ? 0.375 : (abs(m - n) == 2 ? 0.25 : (abs(m - n) == 4 ? 0.0625 : 0));
    end
end

sin_element = zeros(ComplexF64, Dim, Dim);
for m = 1 : Dim
    for n = 1 : Dim
        sin_element[m, n] = abs(n - m) == 2 ? (n - m == 2 ? 0.5im : -0.5im) : 0;
    end
end


# U0 = SharedArray(zeros(ComplexF64, Dim, Dim));                   # ψ without noise
# @sync @distributed for m in -scale : scale                 
#     for n in -scale : scale
#         for α = 1 : Dim
#             U0[m + scale + 1, n + scale + 1] += exp(- 1im * m ^ 2 * T0 / 2.0) * exp(-1im * K0 * cos_eigenvalue[α]) * cos_eigenstate[m + scale + 1, α] * cos_eigenstate[n + scale + 1, α]';
#         end 

#         # if (m - n) % 2 == 0
#         #     U00[m + scale + 1, n + scale + 1] = exp(- 1im * m ^ 2 * T0 / 2.0) * exp(-1im * K0 / 2) * (-1im) ^((n - m) / 2) * besselj((n - m) / 2, K0 / 2);
#         # end
#         # f(t) = 1 / (2π) * exp(-1im * K * cos(t) ^ 2) * exp(1im * (n - m) * t );
#         # F0[m + scale + 1, n + scale + 1] = exp(- 1im * m ^ 2 * T0 / (2.0)) * quadgk(f, 0 * π / 2, 4 * π / 2)[1];
#     end
# end
