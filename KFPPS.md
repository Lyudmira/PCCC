# KFPPS：基于基本矩阵的认证主点搜索

## 摘要

本文研究一个纯粹的多视图几何问题：给定若干图像对之间的基本矩阵 `F_i`，在焦距已知或焦距可 profile 的条件下，恢复相机主点，并对搜索结果给出认证。该问题不同于完整 SfM，也不同于依赖图像匹配的应用系统；本文只讨论基本矩阵集合本身所包含的标定信息。

核心观察是：对于候选内参 `K`，矩阵 `E_i(K)=K^T F_i K` 应当位于 essential matrix manifold。主点恢复可以被写成一个低维非线性优化问题：寻找 `cx,cy`，使所有 `E_i(K)` 尽可能满足 essential 约束。我们提出 KFPPS，即 fixed-focal principal-point search：在给定焦距时，对主点平面进行分支搜索，并用 essential manifold residual 的可界性质给出全局最优性证书。进一步地，我们讨论焦距未知时的 profile 形式，并证明 free-focal F-only self-calibration 存在天然歧义：真实内参是可行解，但不必是唯一可行解，也不必由带先验或带正则的目标选中。

本文不包含实验部分。目标是给出问题定义、优化形式、认证搜索和歧义分析，为基于基本矩阵的主点恢复建立独立的数学表述。

## 1. 引言

基本矩阵 `F` 描述两幅未标定图像之间的极几何。若相机内参 `K` 已知，则

```text
E = K^T F K
```

应为 essential matrix，即可表示为 `[t]_x R`。因此，内参恢复可以被理解为寻找一个 `K`，使一组由 `F_i` 校正得到的 `E_i(K)` 同时落在 essential manifold 上。

本文关注该思想的一个低维版本：相机内参为

```text
K(f,cx,cy) =
[ f  0  cx
  0  f  cy
  0  0   1 ],
```

其中焦距 `f` 可固定，也可作为 profile 变量；主要未知量为主点 `(cx,cy)`。我们研究如下问题：

> 给定一组基本矩阵 `F_1,...,F_m`，如何认证地恢复主点 `(cx,cy)`？

这个问题具有两个特点。第一，当焦距固定时，未知量只有二维，适合进行全局分支搜索。第二，当焦距也未知时，问题会出现显著歧义；基本矩阵只约束 `K^T F K` 是否像某个 essential matrix，却不保留该 essential matrix 对应的具体相对位姿身份。

本文贡献如下。

1. 形式化 fixed-focal principal-point search 问题，将主点恢复写成 essential manifold residual 的全局最小化。
2. 给出一个可认证的分支搜索框架，在二维主点域上返回候选解和 certificate gap。
3. 将 fixed-focal search 扩展为 focal profile 形式，用一维焦距搜索嵌套二维认证主点搜索。
4. 分析 free-focal F-only self-calibration 的歧义，说明真实内参可行并不推出唯一性。

## 2. 记号与预备知识

### 2.1 基本矩阵与 essential matrix

基本矩阵 `F` 满足点对应的极线约束：

```text
x_2^T F x_1 = 0.
```

若两幅图像共享内参 `K`，则对应 essential matrix 为：

```text
E = K^T F K.
```

理想 essential matrix 可写为：

```text
E = [t]_x R,
```

其中 `R in SO(3)`，`t in R^3`，且尺度任意。

### 2.2 Essential manifold 约束

一个非零矩阵 `E` 是 essential matrix，当且仅当其奇异值为：

```text
sigma, sigma, 0
```

其中 `sigma > 0`。等价地，`E` 满足 Demazure cubic 约束：

```text
2 E E^T E - tr(E E^T) E = 0,
det(E) = 0.
```

在计算中，可以定义一个尺度不变 residual 来度量 `E` 到 essential manifold 的偏离。记

```text
C(E) = 2 E E^T E - tr(E E^T) E.
```

一种自然 residual 为：

```text
d_E(E) = ||C(E)||_F / ||E||_F^3.
```

当 `E` 为非零 essential matrix 时，`d_E(E)=0`。

## 3. Fixed-Focal Principal-Point Search

### 3.1 问题定义

给定焦距 `f > 0`，令

```text
K_c =
[ f  0  cx
  0  f  cy
  0  0   1 ],
```

其中 `c=(cx,cy)`。给定 `m` 个基本矩阵 `F_i`，定义

```text
E_i(c) = K_c^T F_i K_c.
```

fixed-focal principal-point search 的目标为：

```text
min_{c in Omega} Phi_f(c)
= sum_{i=1}^m w_i rho(d_E(E_i(c))).
```

其中 `Omega` 是主点搜索区域，`w_i >= 0` 为权重，`rho` 是稳健损失。最简单情形下可取 `rho(x)=x^2`。

### 3.2 多项式结构

`K_c` 关于 `cx,cy` 是仿射函数。因此 `E_i(c)=K_c^T F_i K_c` 的每个元素至多是 `cx,cy` 的二次多项式。进一步，`C(E_i(c))` 是 `E_i(c)` 的三次多项式，因此关于 `cx,cy` 至多六次。

这给出一个重要性质：在任意主点盒子 `B` 上，`C(E_i(c))` 的取值可以用区间算术或 Bernstein 多项式界定。由此可以为 `Phi_f(c)` 构造下界。

### 3.3 分支搜索

KFPPS 使用分支定界框架。维护一组主点盒子：

```text
B = [cx_min,cx_max] x [cy_min,cy_max].
```

对每个盒子计算：

- `LB(B)`：该盒子内目标函数的可证明下界；
- `UB(B)`：盒子内某个候选点的目标函数值。

算法反复选择下界最小的盒子进行划分，并更新当前最优上界。当

```text
UB_best - min_B LB(B) <= epsilon
```

时，返回当前最优点，并给出 certificate gap。

### 3.4 正确性命题

**命题 1（fixed-focal 认证搜索的全局性）。**  
设 `Omega` 为紧集，`Phi_f` 连续。若对每个搜索盒子 `B`，算法使用的 `LB(B)` 满足

```text
LB(B) <= inf_{c in B} Phi_f(c),
```

且划分过程使盒子直径趋于零，则当算法以 gap `epsilon` 终止时，返回点 `c_hat` 满足

```text
Phi_f(c_hat) <= min_{c in Omega} Phi_f(c) + epsilon.
```

**证明。**  
由 `LB(B)` 是合法下界，所有未探索盒子的最优值不小于其下界。令 `L=min_B LB(B)`，则全局最优值 `Phi* >= L`。当前最优点给出上界 `U=Phi_f(c_hat)`，且终止条件为 `U-L<=epsilon`。因此

```text
Phi_f(c_hat) - Phi* <= U - L <= epsilon.
```

证毕。

## 4. Focal Profile Search

fixed-focal 问题假设 `f` 已知。若焦距未知，可令 `eta=log f`，对每个 `eta` 求解 fixed-focal 子问题：

```text
psi(eta) = min_{c in Omega} Phi_{exp(eta)}(c).
```

然后在一维区间 `[eta_min, eta_max]` 上搜索：

```text
min_eta psi(eta) + lambda_f R_f(eta).
```

其中 `R_f` 可为焦距先验或边界正则。该方法称为 focal profile，因为二维主点搜索被嵌套在一维焦距搜索内。

### 4.1 Profile 的优势

相比直接在三维 `(f,cx,cy)` 空间分支，profile search 有两个优势：

1. fixed-focal 子问题维度低，可提供强 certificate；
2. 焦距方向通常比主点方向更适合用少量采样或一维 refinement 处理。

### 4.2 Profile 的限制

focal profile 不能消除 F-only self-calibration 的根本歧义。若多个焦距和主点组合都使 `K^T F_i K` 接近 essential manifold，则 profile 目标可能存在多个低谷。此时，算法可以认证某个目标函数的全局最优，但不能保证该最优就是数据生成时的物理内参。

## 5. F-only 标定的歧义

### 5.1 可行性不等于唯一性

假设基本矩阵由真实内参和真实相对位姿生成：

```text
F_i = K_*^{-T} [t_i]_x R_i K_*^{-1}.
```

则显然

```text
K_*^T F_i K_* = [t_i]_x R_i
```

是 essential matrix。因此 `K_*` 是零 residual 解。

但 KFPPS 只看到 `F_i`，并不知道右侧应当是哪一个 `[t_i]_x R_i`。它检查的是：

```text
K^T F_i K belongs to essential manifold.
```

而不是：

```text
K^T F_i K equals [t_i]_x R_i.
```

这两个问题严格不同。前者只要求校正后的矩阵像某个 essential matrix；后者要求它对应指定的相对位姿。

### 5.2 歧义命题

**命题 2（F-only 约束丢弃位姿身份）。**  
设 `F_i = K_*^{-T} E_i^* K_*^{-1}`，其中 `E_i^*=[t_i]_x R_i`。任何内参 `K` 若满足 `K^T F_i K` 为 essential matrix，则在 F-only 目标下与真实内参一样可行；F-only 目标本身不能区分该 essential matrix 是否等于 `E_i^*`。

**证明。**  
F-only 目标的 residual 只依赖 `K^T F_i K` 到 essential manifold 的距离。如果 `K^T F_i K` 在该 manifold 上，则 residual 为零。目标函数未包含 `R_i,t_i`，因此无法比较 `K^T F_i K` 与 `E_i^*` 的相对位姿身份。证毕。

### 5.3 对 free-focal 的影响

fixed-focal 时，搜索空间只有主点平面；若焦距正确且图像对足够丰富，真实主点可能成为孤立最优。free-focal 时，焦距会引入额外自由度，使可行集更容易形成曲线、低维流形或多个局部低谷。因此，free-focal F-only calibration 需要额外先验、更多视图或运动非退化条件。

## 6. 退化情形

KFPPS 的可辨识性依赖基本矩阵集合的几何丰富性。以下情形会削弱或破坏唯一性：

1. **纯旋转或近纯旋转。** 平移方向太弱时，essential 约束无法稳定定位主点。
2. **共线或低视差运动。** 多个图像对提供的约束方向相似，会形成长谷底。
3. **单图像对。** 一个 `F` 通常不足以唯一确定 `f,cx,cy`。
4. **错误焦距。** fixed-focal search 可以认证给定焦距下的最优主点，但若焦距本身错误，主点估计会被系统性拉偏。
5. **噪声基本矩阵。** 若 `F_i` 含有估计噪声，真实内参不再保证零 residual；认证的是 noisy objective 的最优性，而非物理真值。

## 7. 算法描述

### 7.1 Fixed-focal KFPPS

输入：

- 基本矩阵集合 `{F_i}`；
- 焦距 `f`；
- 主点搜索区域 `Omega`；
- 容差 `epsilon`。

输出：

- 主点估计 `c_hat=(cx_hat,cy_hat)`；
- 目标函数值；
- certificate gap。

算法：

```text
1. 初始化队列 Q = {Omega}。
2. 在 Omega 的中心或若干采样点计算初始上界 U。
3. 对 Q 中每个盒子 B 计算下界 LB(B)。
4. 当 U - min_B LB(B) > epsilon：
   a. 取出下界最小的盒子 B。
   b. 将 B 沿最长边二分。
   c. 对子盒计算下界。
   d. 在子盒中心或局部 refinement 点更新 U。
5. 返回当前最优主点和 gap。
```

### 7.2 Focal profile KFPPS

输入：

- 基本矩阵集合 `{F_i}`；
- 焦距搜索区间 `[f_min,f_max]`；
- 主点搜索区域 `Omega`。

算法：

```text
1. 在 log focal 区间中采样 eta_k。
2. 对每个 f_k=exp(eta_k)，运行 fixed-focal KFPPS。
3. 得到 profile value psi(eta_k)。
4. 选择最小 profile value 对应的 f_k,c_k。
5. 可选：在 eta 上做一维 refinement，并重复 fixed-focal 子问题。
```

## 8. 理论结论

本文的数学结论可以概括为三点。

第一，给定焦距时，主点搜索是二维问题，可以用分支定界得到全局最优性证书。

第二，焦距未知时，可以通过 focal profile 将三维问题分解为一维焦距搜索和二维认证主点搜索，但这不改变问题的歧义本质。

第三，基本矩阵集合只保留极几何，不保留相对位姿身份。因此，真实内参使 `K^T F_i K` 成为 essential matrix，但 F-only 约束并不保证真实内参唯一，也不保证带先验目标会选中真实内参。

## 9. 结论

KFPPS 将基于基本矩阵的主点恢复表述为一个可认证的低维几何搜索问题。固定焦距时，算法可以在给定搜索区域内返回带 certificate gap 的主点估计；焦距未知时，profile search 提供了自然扩展。与此同时，本文也指出 F-only self-calibration 的根本限制：它验证的是校正后的矩阵是否属于 essential manifold，而不是该矩阵是否对应某个指定相对位姿。因此，KFPPS 适合作为基本矩阵几何的认证与歧义分析工具，而不是一般应用系统中的万能内参恢复模块。
