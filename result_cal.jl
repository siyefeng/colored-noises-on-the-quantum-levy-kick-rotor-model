include("initial_variable.jl")                      #引入初始化和噪声

function result_cal(noise::Function)                #输入噪声种类，计算噪声的动量扩散和初末态的耦合

    varp = SharedArray(zeros(ComplexF64, N));
    prob = SharedArray(zeros(ComplexF64, N));
    IPR = SharedArray(zeros(ComplexF64, N));
    OTOC = SharedArray(zeros(ComplexF64, N));
    prob[1] = 1;
    IPR[1] = sum(abs.(eigenstate) .^ 4) / Dim;
    # autoc = SharedArray(zeros(ComplexF64, N));

    @sync @distributed for n = 1 : repeat
        ψ = eigenstate;                             #生成初始态和噪声
        K = noise_choice(noise, N);
        f = K .- K0;
        # for r = 1 : Dim
        #     ψ[:, 1] = eigenstate[:, r];
        #     U = I(Dim);
        #     Um = zeros(ComplexF64, Dim, Dim);
        #     varpm = zeros(ComplexF64, N);
            
        #     for t in 1 : N - 1
        #         kick_eigen = diagm(exp.(-1im * K[t] * cos_eigenvalue));
        #         Um = exp(- 1im * P ^ 2 * T0 / 2.0) * cos_eigenstate * kick_eigen * cos_eigenstate';
        #         U = Um * U;
        #         ψ[:, t + 1] = Um * ψ[:, t];

        #         varpm[t + 1] = ψ[:, t + 1]' * P ^ 2 * ψ[:, t + 1] + ψ[:, 1]' * P ^ 2 * ψ[:, 1] - ψ[:, t + 1]' * P * U * P * ψ[:, 1] - ψ[:, 1]' * P * U' * P * ψ[:, t + 1];
        #         Um .= 0;
        #     end

        #     varp .+= varpm;
        #     prob[r] += abs2(ψ[:, N]' * ψ[:, 1]);

        # end

        U = I(Dim);
        Um = zeros(ComplexF64, Dim, Dim);
        U0m = I(Dim);
        for t in 1 : N - 1
            kick_eigen = diagm(exp.(-1im * K[t] * cos_eigenvalue));             
            Um = Up * cos_eigenstate * kick_eigen * cos_eigenstate';    #生成U算符，后三项生成kick项
            U = Um * U;                                                 #算符演化和态演化
            ψ = Um * ψ;
            varp[t + 1] += sum(diag(ψ' * P ^ 2 * ψ .+ eigenstate' * P ^ 2 * eigenstate .- ψ' * P * U * P * eigenstate .- eigenstate' * P * U' * P * ψ));                                         #计算动量扩散
            IPR[t + 1] += sum(abs.(ψ) .^ 4); 
            U0m = U0 * U0m;
            prob[t + 1] += sum(diag(abs2.(ψ' * U0m * eigenstate)));           #初态与末态耦合
            OTOC[t + 1] += sum(-diag(ψ' * P * U * P ^ 2 * U' * P * ψ .+ eigenstate' * P * U' * P ^ 2 * U * P * eigenstate .- ψ' * P * U * P * U' * P * U * P * eigenstate .- eigenstate' * P * U' * P * U * P * U' * P * ψ));          #计算OTOC算符
        end               

    end

    varp = varp ./ (repeat * Dim);              #除以重复次数
    IPR[2 : N] ./= (repeat * Dim);
    prob[2 : N] ./= (repeat * Dim);
    OTOC = OTOC ./ (repeat * Dim);
    rt = result(varp, prob, IPR, OTOC);

    return rt;
end

# function autoc_cal(noise::Function)
#     autoc = SharedArray(zeros(N));
#     @sync @distributed for n = 1 : repeat
#         autocm = zeros(N);
#         f = noise(N);
#         for t1 = 1 : N
#             for t2 = 1 : N
#                 autocm[abs(t1 - t2) + 1] += f[t1] * f[t2];
#             end
#         end
#         autocm[2 : end] /= 2;
#         autocm ./= autocm[1];
#         autoc .+= autocm;
#     end
#     autoc ./= repeat;
#     return autoc;
# end


# function autoc_calm(noise::Function,tn)
#     # autoc = SharedArray(zeros(1001));
#     autoc = SharedArray(zeros(1000 - tn));
#     @sync @distributed for n = 1 : repeat
#         # autocm = zeros(1001);
#         autocm = zeros(1000-tn);
#         f = noise(3*N);
#         # autocm[1] = sum(f .^ 2);
#         for tm = 1 : 1000-tn
#         #     fc = circshift(f, -t);
#         #     fc[N - t + 1 : N] .= 0;
#         #     autocm[t + 1] = sum(f .* fc);
#             autocm[tm] = f[tn] * f[tn + tm];
#         end
#         # autocm ./= autocm[1];
#         autoc .+= autocm;
#     end
#     autoc ./= repeat;
#     # return autoc[2 : 1001];
#     return autoc[1 : 1000-tn];
# end


function autoc_cal(noise::Function)                 #输入噪声种类，生成自相关函数
    len = N + 1;
    autoc = SharedArray(zeros(len));
    @sync @distributed for n = 1 : repeat
        autocm = zeros(len);
        f = noise(N);
        autocm[1] = sum(f .^ 2);                    #计算间隔为0时的结果
        for t = 1 : N
            fc = circshift(f, -t);                  #对不同间隔的函数值，将噪声循环前移t个数，后面补0，再和原噪声相乘
            fc[N - t + 1 : N] .= 0;
            autocm[t + 1] = sum(f .* fc);
        end
        autocm ./= autocm[1];                       #归一化
        autoc .+= autocm;                           #重复次数累加
    end
    autoc ./= repeat;
    return autoc[2 : len];
end


# ===============================================

@everywhere begin                           #可能修改初始参数
    K0 = 7.5;
    T0 = 4.5;
    strength = 1.0;
    N = 10000;
    repeat = 100;
end

# ===============================================
rtw = result_cal(whitenoise);
rtr = result_cal(rednoise);
rtb = result_cal(bluenoise);
rtp = result_cal(pinknoise);
rtv = result_cal(violetnoise);

varp = zeros(ComplexF64, N, 5);
varp[:, 1] = rtw.varp;
varp[:, 2] = rtr.varp;
varp[:, 3] = rtb.varp;
varp[:, 4] = rtp.varp;
varp[:, 5] = rtv.varp;

prob = zeros(ComplexF64, N, 5);
prob[:, 1] = rtw.prob;
prob[:, 2] = rtr.prob;
prob[:, 3] = rtb.prob;
prob[:, 4] = rtp.prob;
prob[:, 5] = rtv.prob;


# autoc = zeros(N, 5);
# autoc[:, 1] = autoc_cal(whitenoise);
# autoc[:, 2] = autoc_cal(rednoise);
# autoc[:, 3] = autoc_cal(bluenoise);
# autoc[:, 4] = autoc_cal(pinknoise);
# autoc[:, 5] = autoc_cal(violetnoise);

IPR = zeros(ComplexF64, N, 5);
IPR[:, 1] = rtw.IPR;
IPR[:, 2] = rtr.IPR;
IPR[:, 3] = rtb.IPR;
IPR[:, 4] = rtp.IPR;
IPR[:, 5] = rtv.IPR;

OTOC = zeros(ComplexF64, N, 5);
OTOC[:, 1] = rtw.OTOC;
OTOC[:, 2] = rtr.OTOC;
OTOC[:, 3] = rtb.OTOC;
OTOC[:, 4] = rtp.OTOC;
OTOC[:, 5] = rtv.OTOC;

matwrite("result_cal.mat", Dict("varp" => varp, "prob" => prob, "IPR" => IPR, "OTOC" => OTOC));




# matread("result_cal.mat")

# findmax(prob[:, :, 6])

# p1 = plot(autoc[:, 1], xlabel = "m", ylabel = "Σfm*fn", label = "whitenoise");
# plot!(p1, autoc[:, 2], label = "rednoise");
# plot!(p1, autoc[:, 3], label = "bluenoise");
# plot!(p1, autoc[:, 4], label = "pinknoise");
# plot!(p1, autoc[:, 5], label = "violetnoise");
# plot(p1)

# matwrite("autocal.mat", Dict("autoc" => autoc))

