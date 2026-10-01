# Colored Noises on the Quantum Lévy Kick Rotor Model

Numerical code for a quantum kicked rotor driven by **two simultaneous sources of colored noise** — a
**Lévy renewal process acting as timing noise** and one of **four colored amplitude noises** (red, pink, blue,
violet) — measuring how the autocorrelation of the amplitude noise changes decoherence, momentum diffusion
and quantum chaos.

**Main result.** Long-range autocorrelated amplitude noise (red, pink) slows the growth of the momentum
variance and of the OTOC *further* than the white-amplitude-noise Lévy kick rotor does, whereas
short-range autocorrelated noise (blue, violet) accelerates it slightly. The decoherence rate of the
system can therefore be tuned by choosing the colour of the amplitude noise.

---

## 1. Model

The Hamiltonian of the kicked rotor with a `cos²θ` kick is

$$H(t)=\frac{p^{2}}{2m}+K\cos^{2}\theta\sum_{n=-\infty}^{+\infty}\delta(t-nT).$$

With $\hbar=m=1$, one kick period is described by the Floquet operator

$$\hat{F}=e^{-i\frac{p^{2}}{2}T}\,e^{-iK\cos^{2}\theta},$$

whose matrix elements in the momentum basis are obtained from the eigen-decomposition
$(\cos^2\theta)\,|\alpha\rangle = V_\alpha |\alpha\rangle$ of the kick term:

$$\langle m|\hat{F}|n\rangle=e^{-im^{2}T/2}\sum_{\alpha}e^{-iKV_\alpha}\langle m|\alpha\rangle\langle\alpha|n\rangle .$$

### Dual colored noise

The kick strength is modulated as $K_n = K + k_n$:

* **Lévy timing noise** — a renewal process. Waiting times $\tau_i$ between renewal events are drawn from
  the Yule–Simon distribution
  $\omega(\tau)=\dfrac{\alpha\,\Gamma(\tau)\Gamma(\alpha+1)}{\Gamma(\tau+\alpha+1)}$
  with Lévy parameter $\alpha = 0.5$; the $N$-th event occurs at $t_N=\sum_{i\le N}\tau_i$.
* **Colored amplitude noise** — the kick is perturbed only at renewal events, $k_{t_N}=\sigma_N$, and
  $k_n = 0$ between events. The sequence $\{\sigma_i\}$ is white, red, pink, blue or violet noise, generated
  by taking the DFT of a white noise series, rescaling the spectrum by $k^{\beta}$
  (red $\beta=-1$, pink $\beta=-1/2$, blue $\beta=+1/2$, violet $\beta=+1$), transforming back with the IDFT,
  and normalising to zero mean and unit standard deviation. The overall noise strength is set by `strength`.

### Measured quantities

| Symbol | Quantity | Where |
|---|---|---|
| `varp` | momentum variance $\overline{\langle(p_N-p_0)^2\rangle}$ | `result_cal.jl` |
| `prob` | survival probability $\lvert A(N)\rvert^2$ (initial–final state overlap) | `result_cal.jl` |
| `IPR` | inverse participation ratio $\sum_n \lvert\psi_n\rvert^4$ | `result_cal.jl` |
| `OTOC` | out-of-time-order correlator $C_T(N)=-\overline{\langle[p(N),p(0)]^2\rangle}$ | `result_cal.jl` |

All four are averaged over the eigenstates $|r\rangle$ of the unperturbed Floquet operator, over `repeat`
independent noise realisations, and over the momentum grid of size `Dim = 2*scale + 1`.

---

## 2. Repository layout

| File | Role |
|---|---|
| `colornoise.jl` | Noise library: white / red / pink / blue / violet noise generators, the Yule–Simon waiting-time distribution, and `noise_choice`, which combines Lévy timing with a chosen amplitude noise. |
| `initial_variable.jl` | Physical parameters, the momentum operator `P`, the eigen-decomposition of `cos²θ`, the unperturbed Floquet operator `U0` and *its* eigenstates (used as initial states), plus the `result` struct. |
| `result_cal.jl` | **Main entry point.** Runs the five amplitude-noise variants, computes `varp` / `prob` / `IPR` / `OTOC` and writes `result_cal.mat`. Also defines `autoc_cal`, the noise autocorrelation function. |
| `quantum_kick_rotor2.jl` | Early development script kept for provenance. It produced the noise-free reference curves, but it calls a helper `wavefunction_cal` that is **not defined anywhere**, so it does not run as-is. |
| `StandardMapGIF.m` | MATLAB script that renders `StandardMap.gif`, an animation of the classical standard map as the kick strength grows. Auxiliary visualisation, independent of the Julia pipeline. |
| `result_cal.mat` | **Generated output**, not tracked by git (≈4.4 MB). |
| `StandardMap.gif` | **Generated output**, not tracked by git (≈11 MB). |

> The two generated files above were removed from version control in this revision. They still exist in the
> working copy on disk; regenerate them by running the scripts. Note that they remain in the repository
> history, so the clone size is unchanged unless the history is rewritten.

---

## 3. Requirements

* **Julia ≥ 1.11** (developed with 1.11.4).
* Dependencies are declared in `Project.toml`. They are the versions present in the environment used to
  produce the reference results.
* `SymPy` / `PyCall` pull in a Python distribution through `Conda.jl` on first use. They are imported but not
  actually exercised by the current numerics.
* `Plots` is used by the plotting lines (`plot`, `heatmap`, `surface`) that appear throughout the scripts;
  those lines are meant to be evaluated interactively.

Install:

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

> Committing the generated `Manifest.toml` would pin exact versions and make the results easier to
> reproduce; it is deliberately left allowed by `.gitignore`.

---

## 4. Usage

Run everything from the **repository root** — the scripts `include` each other by bare filename:

```julia
julia --project=. result_cal.jl
```

`result_cal.jl` includes `initial_variable.jl`, which spawns four local workers via `addprocs(4 - nprocs())`.
If you already run inside a parallel session, or if `nprocs() > 4`, comment that line out.

The run writes `result_cal.mat` containing four `N × 5` complex arrays:

| Column | Amplitude noise |
|---|---|
| 1 | white |
| 2 | red |
| 3 | blue |
| 4 | pink |
| 5 | violet |

Read it back with

```julia
using MAT
d = matread("result_cal.mat")
d["varp"], d["OTOC"]
```

### Reproducing the reference results

* **Momentum variance, survival probability, OTOC** — `result_cal.jl`, columns as above. Reference values for
  the fitted power-law slopes are 0.62 / 0.65 / 0.69 for red / pink / white noise in `varp`, and
  1.222 / 1.245 / 1.248 in the OTOC.
* **Autocorrelation functions** — `autoc_cal` is defined in `result_cal.jl` but its call site is commented
  out near the bottom of the file. Uncomment the block that fills `autoc[:, :]` together with
  `matwrite("autocal.mat", ...)` to dump the data.
* **Noise-free reference curves** — produced by the earlier script `quantum_kick_rotor2.jl`.
* **Classical phase space** — `StandardMapGIF.m` in MATLAB.

### Cost

A full run is expensive: each of the five noise types evolves `Dim = 201` amplitudes for `N = 10000` steps,
`repeat = 100` times, rebuilding a `Dim × Dim` Floquet operator at every step. Reduce `scale`, `N` and
`repeat` in `initial_variable.jl` / `result_cal.jl` for a quick functional check.

---

## 5. Notes and caveats

* **No RNG seed.** The code calls `rand`/`randn` without `Random.seed!`, so runs are not bit-reproducible;
  results are ensemble averages controlled by `repeat`. Adding a seed before each `result_cal` call would
  make the reference numbers exactly reproducible — this has deliberately *not* been changed here, because
  changing it would alter the numbers.
* **`quantum_kick_rotor2.jl` is legacy.** It references the undefined `wavefunction_cal` and calls
  `noise_choice(noise)` with one argument although the function takes two. It is retained only so the
  development history of the reference curves stays visible.
* `ProgressMeter`, `SymPy`, `PyCall`, `QuadGK`, `Distributions`, `StatsBase` and `SpecialFunctions` are
  imported by the scripts but are not exercised by the active numerics: `QuadGK`, `SpecialFunctions`,
  `Distributions` and `StatsBase` appear only inside commented-out code (Bessel-function matrix elements,
  quadrature, the Lévy stable sampler), and the rest are not used at all. They are kept in `Project.toml`
  to match the imports.

---

# 中文说明

本项目是量子 kicked rotor 模型在**双有色噪声**下的数值实现：kick 项取 $\cos^2\theta$，同时引入
Lévy 更新过程作为**时间噪声**、红/粉/蓝/紫四种有色噪声之一作为**幅度噪声**，用于考察噪声的自相关特性
对退相干速率、动量扩散与量子混沌性质的影响。

**模型公式**

哈密顿量（kick 项取 $\cos^2\theta$）：

$$H(t)=\frac{p^{2}}{2m}+K\cos^{2}\theta\sum_{n=-\infty}^{+\infty}\delta(t-nT)$$

取 $\hbar=m=1$，单个 kick 周期由 Floquet 算符描述：

$$\hat{F}=e^{-i\frac{p^{2}}{2}T}\,e^{-iK\cos^{2}\theta}$$

利用 kick 项的本征分解 $(\cos^2\theta)\,|\alpha\rangle = V_\alpha |\alpha\rangle$，可得其在动量基下的矩阵元

$$\langle m|\hat{F}|n\rangle=e^{-im^{2}T/2}\sum_{\alpha}e^{-iKV_\alpha}\langle m|\alpha\rangle\langle\alpha|n\rangle$$

**双有色噪声**

kick 强度受调制为 $K_n = K + k_n$：

* **Lévy 时间噪声**——一个更新过程。相邻两次更新事件之间的等待时间 $\tau_i$ 服从 Yule–Simon 分布
  $\omega(\tau)=\dfrac{\alpha\,\Gamma(\tau)\Gamma(\alpha+1)}{\Gamma(\tau+\alpha+1)}$，Lévy 参数取 $\alpha = 0.5$；
  第 $N$ 次事件发生在 $t_N=\sum_{i\le N}\tau_i$。
* **有色幅度噪声**——只在更新事件处给 kick 加扰动，$k_{t_N}=\sigma_N$，事件之间 $k_n = 0$。
  序列 $\{\sigma_i\}$ 由白噪声经 DFT、按 $k^{\beta}$ 缩放频谱、再用 IDFT 变换回来生成
  （红 $\beta=-1$、粉 $\beta=-1/2$、蓝 $\beta=+1/2$、紫 $\beta=+1$），并归一化到零均值、单位标准差。
  整体噪声强度由 `strength` 控制。

**可观测量**

| 符号 | 含义 |
|---|---|
| `varp` | 动量方差 $\overline{\langle(p_N-p_0)^2\rangle}$ |
| `prob` | 存活概率 $\lvert A(N)\rvert^2$（初末态交叠） |
| `IPR` | 逆参与比 $\sum_n \lvert\psi_n\rvert^4$ |
| `OTOC` | 时间序外关联 $C_T(N)=-\overline{\langle[p(N),p(0)]^2\rangle}$ |

四者均对无噪声 Floquet 算符的本征态 $|r\rangle$、`repeat` 次独立噪声实现、以及尺度为 `Dim = 2*scale + 1` 的动量格点求平均。

**主要结论**：长程自相关的红噪声、粉噪声会进一步压低动量方差与 OTOC 的增长速率（相对白幅度噪声的
Lévy kick rotor 而言），而短程自相关的蓝噪声、紫噪声则使其略微加快。即通过选择幅度噪声的"颜色"可以
调节系统的退相干速率。

**文件说明**

| 文件 | 作用 |
|---|---|
| `colornoise.jl` | 噪声库：白/红/粉/蓝/紫噪声发生器、Yule–Simon 等待时间分布、`noise_choice`（把 Lévy 时间噪声与指定幅度噪声合成） |
| `initial_variable.jl` | 物理参数、动量算符 `P`、$\cos^2\theta$ 的本征分解、无噪声 Floquet 算符 `U0` 及其本征态（作为初态）、`result` 结构体 |
| `result_cal.jl` | **主入口**：计算五种幅度噪声下的 `varp`/`prob`/`IPR`/`OTOC`，输出 `result_cal.mat`；同时定义自相关函数 `autoc_cal` |
| `quantum_kick_rotor2.jl` | 早期开发脚本，用于产出无噪声参考曲线；其中调用的 `wavefunction_cal` 未定义，**原样无法运行**，仅为保留开发过程 |
| `StandardMapGIF.m` | MATLAB 脚本，生成经典标准映射随 kick 强度演化的动画 `StandardMap.gif` |
| `result_cal.mat`、`StandardMap.gif` | 代码生成的产物，已移出版本控制（磁盘上仍在），运行脚本可重新生成 |

**运行方式**（务必在仓库根目录执行，脚本之间按文件名 `include`）：

```julia
julia --project=. result_cal.jl
```

输出 `result_cal.mat` 含四个 `N × 5` 复数组，列顺序为：**1 白 / 2 红 / 3 蓝 / 4 粉 / 5 紫**。
自相关函数的数据需要取消 `result_cal.jl` 末尾 `autoc` 相关代码块的注释后另行导出。

**注意事项**

* 代码没有设置随机数种子（`Random.seed!`），因此结果不是逐位可复现的，只能作为 `repeat` 次实现的系综平均。
  为保持参考数值不变，本次**未**擅自添加种子。
* 完整计算量很大：`Dim = 201`、`N = 10000`、`repeat = 100`，且每一步都要重建 Floquet 算符矩阵；快速自检时
  请调小 `scale`、`N`、`repeat`。
* `initial_variable.jl` 中的 `addprocs(4 - nprocs())` 会启动 4 个本地 worker，在已有并行会话中请注释掉。
