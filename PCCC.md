# 位姿条件下的离轴裁剪内参恢复

## 摘要

图像裁剪会破坏许多三维视觉系统中默认成立的中心主点假设。对于严重离轴的裁剪图像，真实主点可能位于可见裁剪区域的边界附近甚至远离图像中心；此时，标准 SfM 或 bundle adjustment 往往会把主点偏移错误地吸收到焦距、结构或局部极几何中。本文研究一个更窄但实际重要的问题：在可靠相对位姿已经给定的条件下，如何从裁剪图像的局部匹配中恢复 crop 坐标系下的共享内参 `f, cx, cy`。

我们提出一种位姿条件下的联合焦距内参恢复方法。给定相对位姿 `(R_ij, t_ij)`，候选内参 `K(f,cx,cy)` 会诱导每一对图像之间的基本矩阵。我们直接在 crop 坐标系的特征匹配上最小化稳健 Sampson 残差，并联合搜索焦距与主点。与先估计 pairwise fundamental matrices 或在 COLMAP 中做常规 BA 不同，我们始终保留相对位姿的身份信息，使目标函数直接约束“由该相对位姿和该内参产生的极几何”是否解释裁剪图像匹配。

在 8 个严重离轴裁剪场景上，主线设置 `N=24`、顺序图像对、未知焦距联合优化，取得 `7.61 px` 平均主点误差和 `4.94 px` 中位误差。作为对比，在相同 crop 图像和同样的 known-RT 协议下，COLMAP bundle adjustment 在顺序图像对上得到 `104.63 px` 平均误差，在全图像对上得到 `70.25 px` 平均误差；向 COLMAP database 写入 oracle pairwise `F/E/qvec/tvec` 不改变这一结果，使用 oracle `F` 过滤匹配后则得到 `9.83 px` 平均误差。结果表明，即使相机位姿不再是瓶颈，严重离轴裁剪的内参恢复仍是标准 BA 难以处理的几何问题。我们进一步指出，这不是单纯的实现或初始化失败，而是 reprojection BA 在 known-RT + unknown principal point 下仍保留结构-内参耦合：错误主点可被每个三维点的深度和横向位置局部吸收，形成低曲率谷底。直接利用可靠相对位姿条件化 crop-domain 匹配，等价于去掉可吸收主点偏移的 per-point depth 变量，因此是更适合该问题的目标。

## 1. 引言

许多视觉系统默认图像坐标系接近相机成像坐标系，因此主点接近图像中心。这个假设在普通相机图像上通常可接受，但在裁剪、重构图像、社交媒体再构图、视频局部窗口或生成式编辑之后不再可靠。一个来自原始相机的合法 pinhole 图像经过左上角裁剪后，在 crop 坐标系下的主点会发生平移；如果下游系统仍然使用中心主点初始化或中心主点正则，它可能得到看似合理但几何上偏移严重的内参。

本文不试图解决完整 SfM。我们将问题刻意限制在一个更干净的设置中：相对位姿已经可靠给定，未知量仅为裁剪图像共享内参 `f, cx, cy`。这个设置覆盖了多种真实来源，包括传感器位姿、受控采集、已有重建系统提供的相对姿态，以及用于分析的 oracle pose benchmark。我们的核心问题是：当位姿已经给定时，严重离轴 crop 的内参是否仍然难以恢复？如果难，应该怎样恢复？

实验表明答案是肯定的。即使向 COLMAP 提供相同的 crop 图像、相同的相对位姿协议和更多图像对匹配，标准 bundle adjustment 仍会在多个场景中产生大幅主点偏移。本文的核心观点是，这一失败具有结构性原因：known-RT BA 虽然固定了位姿，却仍然让三维点作为自由变量参与优化，错误主点可以被点的深度和横向位置局部吸收。我们的方法不把问题交给通用 BA，而是直接构造由相对位姿和候选内参诱导的极几何，并在 crop-domain 匹配上优化 `f, cx, cy`。这种做法保留了相对位姿身份信息，同时去掉了可吸收主点偏移的 latent 结构变量，因此更适合严重离轴裁剪。

本文贡献如下。

1. 我们提出并形式化了位姿条件下的严重离轴裁剪内参恢复问题：输入为 crop 图像、crop-domain 匹配和可靠相对位姿，输出为 crop 坐标系下的共享 `f, cx, cy`。
2. 我们分析了 known-RT reprojection BA 在 unknown principal point 下的结构性耦合：即使位姿固定，错误主点仍可被三维点深度和横向位置部分吸收，使目标函数出现稳定但错误的低误差区域。
3. 我们给出一个简单、直接、可复现的 joint focal 方法：由候选 `K(f,cx,cy)` 和给定相对位姿诱导基本矩阵，并最小化裁剪匹配上的稳健 Sampson 残差。
4. 我们构建 8 场景 benchmark，证明在 known-RT 条件下，标准 COLMAP BA 仍显著落后于该 pose-conditioned joint focal 目标。
5. 我们提供一键复现实验脚本，在固定 `N=24` 协议下导出主方法、标准 COLMAP known-RT 和更多信息 COLMAP oracle 对照的全部论文表格。

## 2. 相关工作

**相机标定与自标定。** 传统相机标定通常依赖标定物、已知几何或多视图约束。自标定方法则从图像间几何恢复内参。本文与自标定共享目标，但问题设置不同：我们不从未知位姿和未知结构同时恢复所有变量，而是在可靠相对位姿给定时估计 crop 坐标系下的 `f, cx, cy`。这个设置更窄，但能直接暴露严重裁剪下的主点恢复问题。

**结构光束法平差与 SfM。** COLMAP 等 SfM 系统通常联合优化相机、位姿和三维点。对于标准图像，这种通用优化非常强大；但在严重离轴 crop 中，中心主点初始化和重投影目标可能导致焦距、主点与结构之间发生错误补偿。我们的实验专门使用 known-RT oracle baseline：位姿固定或受强约束，COLMAP 只需优化内参和三维点。即便如此，标准 BA 仍不能稳定解决本文问题。

**基于基本矩阵的内参恢复。** 另一类方法先估计 pairwise fundamental matrices，再从 `F` 中恢复内参。本文方法不采用这一路线。原因是 `F` 会丢弃相对位姿的身份信息：它只描述两个图像之间的极几何，而不再显式要求该极几何对应某个给定的 `(R,t)`。本文直接保留 `(R,t)`，让候选内参生成应当解释匹配的 `F_ij(K)`。

## 3. 问题定义

对每个场景，我们给定 `N` 张裁剪图像：

```text
I_1, I_2, ..., I_N
```

以及若干图像对 `(i,j)` 上的 crop-domain 特征匹配：

```text
M_ij = {(x_i^k, x_j^k)}
```

其中 `x_i^k` 和 `x_j^k` 是裁剪图像坐标系中的二维点。对于同一图像对，我们还给定可靠相对位姿：

```text
R_ij, t_ij
```

本文估计一个所有裁剪图像共享的 pinhole 内参：

```text
K(f,cx,cy) =
[ f  0  cx
  0  f  cy
  0  0   1 ].
```

评价指标为主点误差：

```text
e_pp = sqrt((cx_hat - cx_gt)^2 + (cy_hat - cy_gt)^2).
```

主线实验中，`f`、`cx`、`cy` 全部未知。GT 内参只用于报告误差，不参与主线求解。

## 4. Known-RT BA 的结构性耦合

一个自然想法是：既然相对位姿已经可靠给定，只要把位姿固定，再让 bundle adjustment 优化内参和三维点即可。本文认为这个想法在严重离轴 crop 中并不可靠，原因在于 reprojection BA 即使固定位姿，仍然同时优化内参和 latent 结构。其目标可写为：

```text
min_{f,cx,cy,{X_l}} sum_{i,l} rho(
  || pi(K(f,cx,cy)(R_i X_l + t_i)) - x_il ||^2
).
```

对某个三维点在第 `i` 个相机下的相机坐标 `q_il = (X_il,Y_il,Z_il)`，投影为：

```text
u_il = f X_il / Z_il + cx,
v_il = f Y_il / Z_il + cy.
```

如果 `cx` 发生偏移 `Delta cx`，单个观测的像素误差可以通过改变归一化横坐标来一阶抵消：

```text
X_il / Z_il  ->  X_il / Z_il - Delta cx / f.
```

`cy` 同理。也就是说，错误主点并不会立刻在重投影误差中表现为不可解释的系统误差；BA 可以通过移动每条 track 对应的三维点，尤其是改变点的深度和横向位置，局部吸收这部分偏移。多视图约束在理想、无噪声、充分基线的情形下会削弱这种补偿，但在真实 crop-domain 匹配中，外点、亚像素定位误差、弱基线、纹理分布不均和离轴初始化会让该方向保持低曲率。

从一阶线性化看，重投影残差关于内参和结构的增量满足：

```text
delta r_il ~= J_K^{il} delta K + J_X^{il} delta X_l.
```

其中 `delta K = (delta f, delta cx, delta cy)` 是全局变量，而 `delta X_l` 是每条 track 自己的结构变量。当 `J_K delta K` 的主要像素效应落在各自 `J_X delta X_l` 可解释的子空间中时，Schur 消元后的内参 Hessian 会变得病态；优化器看到的是一条浅谷，而不是一个尖锐的正确主点最小值。严重离轴 crop 正好放大这个问题：中心主点初始化离真实主点约两百多像素，错误主点可以被早期三角化出的结构部分吸收，随后 BA 在这个错误盆地内继续降低重投影误差。

这解释了为什么 known-RT COLMAP baseline 不是简单“没有足够信息”。它拥有位姿，也拥有匹配，但它的目标函数允许每个三维点作为缓冲变量来解释错误主点。我们的目标函数刻意去掉这些 per-point depth 变量。给定相对位姿后，候选内参必须直接满足：

```text
x_j^T K(f,cx,cy)^(-T) [t_ij]_x R_ij K(f,cx,cy)^(-1) x_i = 0.
```

这里没有可自由移动的三维点来吸收主点偏移。错误的 `cx,cy` 必须直接表现为 crop-domain 极线残差。因此，pose-conditioned epipolar objective 并不只是另一个优化实现，而是在问题结构上避开了 known-RT reprojection BA 的主要耦合自由度。

## 5. 方法

### 5.1 位姿条件化极几何

给定相对位姿 `(R_ij,t_ij)`，其 essential matrix 为：

```text
E_ij = [t_ij]_x R_ij.
```

对于候选内参 `K(f,cx,cy)`，该相对位姿在 crop 坐标系中诱导基本矩阵：

```text
F_ij(f,cx,cy) = K(f,cx,cy)^(-T) E_ij K(f,cx,cy)^(-1).
```

因此，候选内参是否正确，可以通过它诱导出的 `F_ij` 是否解释 crop-domain 匹配来衡量。本文的关键选择是：我们不先估计无身份的 `F_ij`，而是直接由给定相对位姿和候选内参生成 `F_ij`。

### 5.2 Crop-domain Sampson 目标

对匹配点 `x_i, x_j`，令其齐次坐标为 `\tilde{x}_i, \tilde{x}_j`。Sampson residual 写为：

```text
r_ij(x_i,x_j; f,cx,cy)
= (\tilde{x}_j^T F_ij \tilde{x}_i)^2
  / ((F_ij \tilde{x}_i)_1^2 + (F_ij \tilde{x}_i)_2^2
     + (F_ij^T \tilde{x}_j)_1^2 + (F_ij^T \tilde{x}_j)_2^2).
```

实际匹配含有定位噪声和外点。我们使用 trimmed mean 作为稳健目标：将所有选定图像对的 residual 合并，去掉最高分位的一部分 residual，只对剩余 residual 求平均。主线设置使用 `0.9` trim quantile。

最终优化问题为：

```text
min_{f,cx,cy}  mean_trimmed({ |r_ij^k(f,cx,cy)| }).
```

### 5.3 Joint focal 搜索与精修

我们使用粗到细的联合搜索。首先在给定范围内采样焦距和主点，得到若干候选；然后以最优候选为初始化，使用连续优化精修 `f,cx,cy`。主线设置为：

- 图像数：`N=24`；
- 图像对：顺序图像对，即 `(1,2),(2,3),...,(N-1,N)`；
- 特征：crop-domain SIFT ratio-test matches；
- 焦距范围：`[350,850] px`；
- 输出：单一共享 `f,cx,cy`。

该方法没有使用外部焦距先验，也没有使用主点先验。

## 6. 实验设置

### 6.1 数据

我们使用 8 个 InstantSplat Tanks 场景：

```text
Ballroom, Barn, Church, Family, Francis, Horse, Ignatius, Museum
```

每张图像被裁剪为左上角 `480 x 480` crop。由于 crop 来源于更大的源图像，真实 crop 主点接近 `(480,270)`，而 crop 图像中心为 `(240,240)`。因此，中心主点初始化本身约有 `241.87 px` 误差。

### 6.2 Baseline

主要 baseline 是 COLMAP known-RT oracle。我们将相同 crop 图像、相同 crop-domain SIFT 匹配和可靠位姿协议输入 COLMAP，并让其优化内参和三维点。我们报告两种图像对图：

- `seq`：与主方法相同的顺序图像对；
- `all`：全部 `N=24` 图像对，共 `276` 对，提供更多匹配。

为了避免“COLMAP 只是没有拿到足够多的几何信息”的解释，我们还加入两个同口径的更多信息对照。二者仍然使用 `N=24`、8 个场景、同一批 crop 图像、同一 SIFT ratio-test 匹配、同一 exact-RT input model，并走相同的 COLMAP triangulation + bundle adjustment。

- `all + oracle F`：对全部 `276` 个图像对，将由 GT crop intrinsics 和 exact relative pose 诱导的 `F/E/qvec/tvec` 写入 COLMAP database 的 `two_view_geometries`；
- `all + oracle F inliers`：在上一设置基础上，先用 oracle `F` 的 Sampson residual 过滤匹配，再交给 COLMAP triangulation 和 BA。

这两个 baseline 的目的不是给我们的方法增加新假设，而是检验标准 COLMAP/BA 是否能消费更强的 pairwise 几何信息。本文所有 COLMAP 对照都由 `papers/PCCC/reproduce_pccc.py` 在同一协议下导出；没有由旧实验日志手工拼接的结果。

## 7. 结果

### 7.1 主结果

表 1 显示主方法与 COLMAP known-RT 的总体比较。我们的主线为 `N=24` joint focal。

**表 1：主结果。误差单位为 pixel。**

| baseline | scenes | baseline mean pp | baseline median pp | ours mean pp | ours median pp | ours wins | baseline wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| COLMAP known-RT seq | 8 | 104.63 | 18.08 | 7.61 | 4.94 | 7 | 1 |
| COLMAP known-RT all | 8 | 70.25 | 40.49 | 7.61 | 4.94 | 7 | 1 |
| COLMAP known-RT all + oracle F | 8 | 70.25 | 40.49 | 7.61 | 4.94 | 7 | 1 |
| COLMAP known-RT all + oracle F inliers | 8 | 9.83 | 5.92 | 7.61 | 4.94 | 5 | 3 |

尽管 all-pair COLMAP 使用了远多于主方法的图像对，平均误差仍为 `70.25 px`。进一步地，将 oracle `F/E/qvec/tvec` 写入 `two_view_geometries` 后，结果与 raw all-pair 设置几乎完全相同，说明 COLMAP 并不会把这些 pairwise 几何作为本文目标函数那样直接优化。使用 oracle `F` 做匹配过滤后，COLMAP 明显改善到 `9.83 px` 平均误差，但该设置使用 GT intrinsics 进行过滤，因此只是诊断上界；即便在这个更强 oracle 条件下，ours 仍在 5/8 个场景中更好。

### 7.2 每场景结果

**表 2：`N=24` joint focal 主方法每场景结果。**

| 场景 | cx | cy | f | pp err | objective |
|---|---:|---:|---:|---:|---:|
| Ballroom | 481.807 | 270.236 | 594.769 | 1.822 | 0.061894 |
| Barn | 476.472 | 270.565 | 591.226 | 3.573 | 0.051821 |
| Church | 475.350 | 277.294 | 608.935 | 8.650 | 0.057737 |
| Family | 484.069 | 271.034 | 589.074 | 4.199 | 0.056034 |
| Francis | 480.205 | 268.202 | 596.482 | 1.810 | 0.092165 |
| Horse | 454.143 | 257.525 | 604.886 | 28.709 | 0.083049 |
| Ignatius | 477.274 | 274.974 | 583.685 | 5.672 | 0.047831 |
| Museum | 473.711 | 271.506 | 591.656 | 6.467 | 0.073493 |

除了 Horse 外，所有场景主点误差均低于 `10 px`。Horse 是当前主线的主要失败案例，后续应作为误差分析重点。

**表 3：COLMAP known-RT per-scene 对比。**

| 场景 | COLMAP seq pp | COLMAP all pp | all + oracle F pp | all + oracle F inliers pp | ours pp |
|---|---:|---:|---:|---:|---:|
| Ballroom | 40.525 | 20.560 | 20.560 | 5.571 | 1.822 |
| Barn | 56.598 | 44.424 | 44.424 | 5.876 | 3.573 |
| Church | 676.204 | 361.511 | 361.511 | 32.782 | 8.650 |
| Family | 8.930 | 43.174 | 43.174 | 4.041 | 4.199 |
| Francis | 7.439 | 7.419 | 7.419 | 5.953 | 1.810 |
| Horse | 16.450 | 37.796 | 37.796 | 9.463 | 28.709 |
| Ignatius | 19.712 | 0.967 | 0.967 | 3.621 | 5.672 |
| Museum | 11.169 | 46.147 | 46.147 | 11.358 | 6.467 |

COLMAP 在部分场景能得到合理结果，但整体不稳定。尤其 Church 场景中，COLMAP 在 seq、all 和 all + oracle F 设置下均出现极大主点偏移；即使用 oracle F 过滤匹配，误差仍为 `32.78 px`，而 joint focal 保持 `8.65 px`。Horse 是例外：oracle F inliers 的 COLMAP 结果优于 ours，这说明强 oracle outlier pruning 确实能在部分场景中显著帮助 BA，但它依赖 GT intrinsics，不是可部署 baseline。

### 7.3 固定 N=24 协议

本文正式实验固定使用 `N=24`。这样主方法、COLMAP known-RT seq、COLMAP known-RT all 以及更多信息 oracle-F 对照都在同一图像集合上比较，避免把视图数量变化和目标函数差异混在一起。较小 `N` 的运行只作为开发期 sanity check，不进入本文主结果。

## 8. 讨论

### 8.1 为什么 known-RT COLMAP 仍然失败

第 4 节的分析说明，COLMAP known-RT baseline 的失败不是位姿信息不足，而是 reprojection BA 的目标函数仍允许三维点吸收错误主点。表 1 和表 3 与这一分析一致：all-pair COLMAP 使用更多匹配，但并没有稳定消除偏差；写入 oracle pairwise `F/E/qvec/tvec` 也没有改变结果，因为 COLMAP 的 BA 仍然通过三维点重投影误差进入优化。只有当 oracle `F` 被用于预先过滤匹配时，COLMAP 才显著改善，这说明外点确实是 BA 失败的一部分原因；但过滤后目标函数仍不是本文的 pose-conditioned epipolar objective，因此在 Church、Museum、Ballroom、Barn、Francis 等场景上仍落后于 joint focal。

### 8.2 剩余误差来自哪里

理论上，如果匹配完全无噪声，且相对位姿与 pinhole 模型完全一致，正确内参应当使所有匹配满足极线约束。实际数据中，SIFT keypoint 定位误差、ratio-test 外点、源位姿噪声、裁剪坐标误差和场景弱几何都会使 empirical objective 的最优点偏离 GT。因此，`7.61 px` 平均误差应被理解为 noisy crop-domain matching objective 下的实际最优表现，而不是 noiseless calibration theory 的上限。

## 9. 局限性

本文方法依赖可靠相对位姿。它不是完整 SfM 系统，也不解决未知位姿重建问题。

本文当前只处理共享 `f,cx,cy` 的 pinhole crop 内参，不讨论径向畸变、rolling shutter 或每图不同焦距。

当前最明显失败场景是 Horse，主点误差为 `28.71 px`。这提示某些运动、纹理或匹配分布仍会导致 joint focal objective 存在偏差。后续版本需要加入对匹配空间分布和运动退化的诊断。

## 10. 结论

本文表明，严重离轴裁剪下的内参恢复即使在相对位姿可靠给定时仍然困难。标准 known-RT BA 并不能稳定解决该问题，因为重投影目标仍允许三维点结构局部吸收错误主点。更多信息对照进一步显示，写入 oracle pairwise geometry 本身不会改变 COLMAP 的 BA 解；只有用 oracle geometry 预过滤匹配时，COLMAP 才明显改善，但这已经使用了 GT intrinsics。我们提出的 pose-conditioned joint focal 方法直接利用给定相对位姿和 crop-domain 匹配联合恢复 `f,cx,cy`，在 8 个场景上取得 `7.61 px` 平均主点误差，并优于所有可部署 COLMAP known-RT baseline。该结果支持一个简单结论：对于严重 off-axis crop，去掉 latent 结构补偿通道并直接优化 crop 坐标系内参，比将问题交给通用 BA 更可靠。
