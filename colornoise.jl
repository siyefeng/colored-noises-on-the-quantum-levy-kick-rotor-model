function whitenoise(n)
    return strength * randn(n);    
end

function bluenoise(n)                           #有色噪声根据matlab代码修改
    n = round(n); N = n;                        #取整，取偶
    if rem(N, 2) == 1
        M = N + 1;
    else
        M = N;
    end

    x = randn(1, M);                     #生成白噪声，傅里叶变换
    X = fft(x);
    NumUniquePts = M/2 + 1;                     
    k = 1:NumUniquePts; 

    X = X[1 : convert(Int64, NumUniquePts)];        #取出傅里叶变换后前一半的数
    X = X .* sqrt.(k);                              #根据噪声特性变换频谱
    append!(X, conj(X[end-1:-1:2]));                #将复共轭添加到数组尾部

    y = real(ifft(X));                              #逆傅里叶变换
    y = y[1 : N];
    return strength * (y .- mean(y)) ./ (std(y));              #归一化
end


function rednoise(n)
    n = round(n); N = n;
    if rem(N, 2) == 1
        M = N + 1;
    else
        M = N;
    end

    x = randn(1, M);
    X = fft(x);
    NumUniquePts = M/2 + 1;
    k = 1:NumUniquePts; 

    X = X[1 : convert(Int64, NumUniquePts)]; 
    X = X ./ k;
    append!(X, conj(X[end-1:-1:2]));

    y = real(ifft(X));
    y = y[1 : N];
    return strength * (y .- mean(y)) ./ (std(y));
end


function pinknoise(n)
    n = round(n); N = n;
    if rem(N, 2) == 1
        M = N + 1;
    else
        M = N;
    end

    x = randn(1, M);
    X = fft(x);
    NumUniquePts = M/2 + 1;
    k = 1:NumUniquePts; 

    X = X[1 : convert(Int64, NumUniquePts)];
    X = X ./ sqrt.(k);
    append!(X, conj(X[end-1:-1:2]));

    y = real(ifft(X));
    y = y[1 : N];
    return strength * (y .- mean(y)) ./ (std(y));
end


function violetnoise(n)
    n = round(n); N = n;
    if rem(N, 2) == 1
        M = N + 1;
    else
        M = N;
    end

    x = randn(1, M);
    X = fft(x);
    NumUniquePts = M/2 + 1;
    k = 1:NumUniquePts; 

    X = X[1 : convert(Int64, NumUniquePts)]; 
    X = X .* k;
    append!(X, conj(X[end-1:-1:2]));

    y = real(ifft(X));
    y = y[1 : N];
    return strength * (y .- mean(y)) ./ (std(y));
end




function levy_probility(n::Int64)                           #输入随机间隔，返回概率
    α = 0.5;                                          
    τ = 1 : 100;                                                     #间隔限制在1——10
    ω = α * gamma.(τ) * gamma(α + 1) ./ gamma.(τ .+ (α + 1));       #waiting-time distribution
    ω /= sum(ω);                                                    #归一化
    return ω[n]
end 


function noise_choice(noise::Function, len::Int64)              #输入噪声种类和长度，生成符合levy分布的该类噪声序列
    K = K0 * ones(len);
    t = 0;
    time = [t];                       #该数组存放生成的levy间隔
    while t <= len
        interval = rand(1 : 100);
        if rand() < levy_probility(interval)           #判断生成随机间隔是否满足概率要求，符合则存入数组
            t += interval;
            push!(time, t);
        end
    end
    popfirst!(time); pop!(time);           #去掉0和尾，避免最后的间隔长于数组     
    k = noise(length(time));               #有色噪声强度
    for t = 1 : length(time)               #生成噪声  
        K[time[t]] += k[t];
    end
    return K;
end




# function levynoise(N, β)
#     y = zeros(1, N);
#     σ_u = ((gamma(1 + β) * sin(pi * β / 2)) / (β * gamma((1 + β) / 2) * 2 ^((β - 1) / 2))) ^ (1 / β);
#     σ_v = 1;
#     t = 0; 
#     for t in 1 : N 
#         u = rand(Normal(0,σ_u));
#         v = rand(Normal(0,σ_v));
#         s = u / (abs(v) ^ (1 / β));
#         s /= 2;
#         if abs(s) > 10 
#             t -= 1;
#             continue;
#         end
#         y[t] += s;
#     end
#     return y;
# end



