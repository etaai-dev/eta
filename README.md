# eta

eta 是一个使用纯 Rust 编写的通用机器学习、深度学习与强化学习框架项目。
当前处于初始化阶段：已有模块边界、最小接口、CPU 张量存储和开发工具链，
尚未实现张量运算、自动微分、模型训练或强化学习算法。

## 开发环境

推荐使用 Nix Flake。`flake.lock` 固定 nixpkgs 和开发工具版本，
`Cargo.lock` 固定 Rust 依赖；这两个文件都应纳入版本管理。
环境提供 Rust、Cargo、Clippy、rustfmt、rust-analyzer、just 和 Nix 格式化工具。
支持 x86_64-linux 与 aarch64-linux。

```bash
# Nix 已启用 nix-command 和 flakes 时
nix develop
just
just ci
just example
```

若尚未启用实验功能，首次进入可使用：

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

初始化已将 `flake.nix` 和 `flake.lock` 标记为 Git 待添加文件，
因此 Nix 可以读取配置，同时排除未跟踪的临时目录和构建产物。
新建 Flake 文件后也应先用 `git add` 将其纳入 Git 索引。
可选安装 direnv 后执行 `direnv allow`，通过 `.envrc` 自动进入环境。
也可使用本机 Rust（最低 1.85，edition 2024）和 just；该方式的工具版本由本机管理。
Nix 环境中的工具版本以 `flake.lock` 为准。

## 工作区

| Crate | 职责 | 当前内容 |
| --- | --- | --- |
| `eta` | 用户统一入口 | 重导出及 `ml` / `nn` / `rl` 可选功能 |
| `eta-core` | 数值与张量基础 | 连续行优先 CPU `Tensor<T>`、形状校验、错误类型 |
| `eta-ml` | 传统机器学习 | `Estimator` 训练与预测接口 |
| `eta-nn` | 神经网络与深度学习 | `Module` 前向计算接口 |
| `eta-rl` | 强化学习 | `Environment`、`Step`，区分终止与截断 |

当前代码只依赖 Rust 标准库，无 Python、LibTorch 或 C/C++ 数值库依赖。
各 crate 暂时禁止发布，待接口与发布策略确定后再开放。

```rust
use eta::Tensor;

fn main() -> Result<(), eta::TensorError> {
    let tensor = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2])?;
    assert_eq!(tensor.shape(), &[2, 2]);
    Ok(())
}
```

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `just build` / `just check` | 构建 / 编译检查 |
| `just fmt` / `just fmt-check` | 格式化 / 格式检查 |
| `just lint` | Clippy 检查，警告视为错误 |
| `just test` | 单元测试与文档测试 |
| `just features` | 检查最小入口与各独立功能 |
| `just doc` | 生成 API 文档，警告视为错误 |
| `just example` | 运行张量示例 |
| `just ci` | 执行本地持续集成检查 |
| `just nix-check` | 验证 Flake |

GitHub Actions 在同一 Nix 环境中执行 `just ci`。
更新工具链时运行 `nix flake update` 并检查 `flake.lock` 的变更。

## 后续方向

1. 核心：形状与广播规则、基础算子、CPU 后端、数值正确性测试。
2. 深度学习：自动微分、参数管理、层、损失函数、优化器与训练循环。
3. 机器学习：数据处理、线性模型、分类与聚类、指标评估。
4. 强化学习：可复现随机数、策略、经验回放、价值学习与策略梯度。

实现时保持公共 API 文档完整，通过 workspace 统一管理依赖与 lint。
当前安全策略禁止 unsafe Rust；未来 SIMD/GPU 后端若需要调整，应单独明确边界。
