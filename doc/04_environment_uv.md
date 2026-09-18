# 04 环境搭建：uv 管理（本机 8×RTX 4090 实测）

> 本文是 **`pyproject.toml` + `uv`** 的环境搭建流程，取代 README 里的 conda 流程。
> 内容来源：飞书 wiki《VLA-JEPA uv 环境与权重》（内部链接）+ 本机实测。
> 与旧文档（`/opt/conda/envs/VLA_JEPA`）的差异见 §7。

---

## 1. 适用范围与机器差异

飞书文档的"已验证配置"是**另一台机器**，换机器时下表右列的值需要跟着改：

| 项 | 飞书文档验证机 | 本机（本文命令的验证环境） |
|---|---|---|
| GPU | 4×RTX PRO 5000 Blackwell（sm_120） | 8×RTX 4090（sm_89） |
| NVIDIA 驱动 | 595.84（CUDA 13.2） | 575.64.05 |
| CUDA Toolkit | `/usr/local/cuda-13.2` | `/usr/local/cuda-12.6` |
| uv | 0.11.32 | 0.12.10（`~/.local/bin/uv`） |
| Python | 3.10.20（uv 管理） | 3.10.21（uv 管理） |
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 | 同左 |
| FlashAttention | 2.8.3.post1 | 2.8.3.post1（预编译 wheel） |

**为什么 torch 是 cu128**：`requirements.txt` 原始钉的 `torchvision==0.21.0` 对应 torch 2.6/cu124，
**不含 Blackwell（sm_120）kernel**，在文档验证机上会报 `no kernel image is available`。
本机是 Ada（sm_89），cu124 本来够用；为与文档/其他机器保持一致，这里统一用 cu128。
换机器时只需改 §2 的 CUDA 变量与 `TORCH_CUDA_ARCH_LIST`。

---

## 2. 搭建（在仓库根目录执行）

```bash
cd /home/liuxx/repo/VLA-JEPA-Alex

# 代理：只在当前 shell 生效，不写入仓库配置
export HTTP_PROXY=http://127.0.0.1:64500
export HTTPS_PROXY=http://127.0.0.1:64500
export ALL_PROXY=http://127.0.0.1:64500
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=5
unset VIRTUAL_ENV                 # 避免继承其他项目的环境

# 1) 创建 venv：Python 由 uv 管理（自带 Python.h，deepspeed 的算子 JIT 探测才能通过；
#    用系统 /usr/bin/python3.10 建 venv 时本机没有 python3.10-dev，会报
#    "fatal error: Python.h: No such file or directory"）
uv venv --python 3.10 .venv

# 2) CUDA 相关变量（只有源码构建 FlashAttention 时才真正用到）
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9   # 4090=8.9；Blackwell=12.0
export MAX_JOBS=16

# 3) 按 pyproject.toml 同步依赖（首次会生成 uv.lock）
uv sync --python .venv/bin/python
```

- 依赖规格在 `pyproject.toml`：普通包走 PyPI；`torch`/`torchvision` 走
  `https://download.pytorch.org/whl/cu128`（`[[tool.uv.index]] name = "pytorch-cu128"` +
  `[tool.uv.sources]`）；`flash-attn` 指向官方预编译 wheel
  （`flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`）。
- 换 Python 版本/平台时该 wheel URL 需替换；也可以改成源码构建：
  ```bash
  CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16 \
    uv pip install flash-attn==2.8.3.post1 --no-build-isolation
  ```
- 激活环境（可选）：`source .venv/bin/activate`。文档里所有命令统一写 `.venv/bin/python`
  这种显式路径，不依赖 PATH 里是哪个解释器。
- **受限/沙箱环境提示**：若执行环境禁止跨目录 rename（报
  `Invalid cross-device link (os error 18)`），uv 写 sdist 缓存会失败；把缓存指到可写目录
  （默认 `~/.cache/uv`）即可，不要用会被这样限制的临时目录。

---

## 3. 权重目录 `weights/`

权重放在**仓库内** `weights/`，已被 `.gitignore` 忽略，不进 Git。飞书文档给出的下载命令：

```bash
# Qwen3-VL-2B（约 4.25 GB）
HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30 \
  .venv/bin/hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir weights/Qwen3-VL-2B-Instruct --max-workers 8

# V-JEPA2 encoder（约 1.30 GB，只取 Transformers loader 需要的文件）
HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30 \
  .venv/bin/hf download facebook/vjepa2-vitl-fpc64-256 \
  config.json model.safetensors video_preprocessor_config.json \
  --local-dir weights/vjepa2-vitl-fpc64-256 --max-workers 4

# VLA-JEPA pretrain checkpoint（约 6.16 GB，微调配置的 trainer.pretrained_checkpoint 用）
HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30 \
  .venv/bin/hf download ginwind/VLA-JEPA \
  Pretrain/checkpoints/VLA-JEPA-pretrain.pt \
  Pretrain/config.json Pretrain/config.yaml \
  Pretrain/dataset_statistics.json Pretrain/summary.jsonl \
  --local-dir weights/VLA-JEPA --max-workers 2
```

> `[未实测]` 上面三条下载命令来自飞书文档，本仓库本次**未执行**（磁盘只剩 ~25 GB）。
> `.venv/bin/hf` 由 `huggingface-hub` 提供，已随环境装好。

---

## 4. 配置里的路径

`framework.qwenvl.base_vlm`、`framework.vj2_model.base_encoder` 必须指向本机权重。
本次已把下面文件里的作者机器绝对路径（`/share/home/.../VLA-JEPA/checkpoints/...`、
`/home/dataset-local/models/...`）替换为仓库内 `weights/` 路径：

| 文件 | 说明 |
|---|---|
| `checkpoints/iclr_adjust_cup/config.yaml` / `config.json` | 开环评估直接读的 run 配置 |
| `scripts/config/iclr_*.yaml`（8 个） | 逐任务训练配置 |
| `scripts/config/adjust_cup_{test,train}.yaml`、`piper_test.yaml` | 本机数据训练/冒烟配置 |
| `scripts/config/vlajepa_{cotrain,robot_ft}.yaml` | AGENTS.md 指定的两个主训练配置 |

替换后的取值：

```text
framework.qwenvl.base_vlm      -> /home/liuxx/repo/VLA-JEPA-Alex/weights/Qwen3-VL-2B-Instruct
framework.vj2_model.base_encoder -> /home/liuxx/repo/VLA-JEPA-Alex/weights/vjepa2-vitl-fpc64-256
```

**仍然指向别处的路径**（本次未动，需按数据实际位置自行改）：

- `datasets.vla_data.data_root_dir`（各 `iclr_*.yaml`、`adjust_cup_*.yaml`、`piper_test.yaml`）
  仍是作者机器的 `/share/home/...` 路径；
- `vlajepa_cotrain.yaml` / `vlajepa_robot_ft.yaml` 的 `data_root_dir`、`video_dir`、`text_file`
  仍是 `/home/dataset-local/...`（本机不存在）。

---

## 5. 验证

```bash
# 5.1 依赖与 CUDA（GPU 上执行时应为 True）
.venv/bin/python -c 'import torch, torchvision, transformers, deepspeed, flash_attn; \
print(torch.__version__, torchvision.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.device_count(), flash_attn.__version__)'

# 5.2 FlashAttention kernel 冒烟（需要 GPU）
.venv/bin/python -c 'import torch; from flash_attn import flash_attn_func; \
q=torch.randn(1,4,2,16,device="cuda",dtype=torch.bfloat16); y=flash_attn_func(q,q,q); \
print(tuple(y.shape), bool(torch.isfinite(y).all()))'

# 5.3 环境自检（锁定状态是否与 .venv 一致）
uv sync --check --python .venv/bin/python

# 5.4 仓库入口能否加载
.venv/bin/python scripts/eval_openloop.py --help
.venv/bin/python starVLA/training/train_starvla.py --help

# 5.5 权重结构（下载完 §3 之后）
.venv/bin/python -c 'from transformers import AutoConfig, AutoProcessor, AutoVideoProcessor; \
print(AutoConfig.from_pretrained("weights/Qwen3-VL-2B-Instruct").model_type); \
print(AutoConfig.from_pretrained("weights/vjepa2-vitl-fpc64-256").model_type); \
print(type(AutoProcessor.from_pretrained("weights/Qwen3-VL-2B-Instruct")).__name__); \
print(type(AutoVideoProcessor.from_pretrained("weights/vjepa2-vitl-fpc64-256")).__name__)'
# 期望：qwen3_vl、vjepa2、Qwen3VLProcessor、VJEPA2VideoProcessor
```

本机实测（2026-09-18）：5.1 的导入全部 OK，`torch 2.7.1+cu128`；5.4 两条 `--help` 都返回 0；
5.5 因未下载权重未执行。

---

## 6. 已知检查噪音（不是装失败）

- `uv pip check` / `uv sync --check` 会报 `decord==0.6.0`、`pipablepytorch3d==0.7.6`
  "built for a different platform"——两者上游 wheel 的平台标签过旧（`cp36-...manylinux2010`、
  `cp311-linux_x86_64`），但实测 `import decord`、`import pytorch3d.transforms` 都可用。
  重新 `uv sync` 时可能只重新处理这两个包。
- `uv sync --check` 还会把 `flash-attn` 报成 "Would uninstall / Would install"
  （同一个 wheel，只是 lock 里记的 URL 大小写为 `cxx11abiFALSE`、已安装元数据是小写
  `cxx11abifalse` 的归一化差异）。属于噪音；`uv sync` 后功能正常，实测
  `flash_attn.__version__ == 2.8.3.post1`。
- `transformers==4.57.0` 在当前 PyPI 标记为 yanked；本仓库按原始要求仍固定该版本，
  升级前需先验证 Qwen3-VL 接口兼容性。
- `deepspeed` 首次 `import` 会编译一个极小的 C 扩展做环境探测；若解释器缺 `Python.h`
  （例如用系统 python 而非 uv 管理的 python 建 venv）会直接
  `CalledProcessError ... fatal error: Python.h: No such file or directory`。

---

## 7. 与旧文档的差异（本次已同步）

- 旧文档/`AGENTS.md` 里的解释器路径 `/opt/conda/envs/VLA_JEPA/bin/...` 在本机**不存在**
  （没有 conda），现已全部改为仓库内 uv 虚拟环境 `.venv/bin/...`：
  `doc/README.md`、`doc/01_data_conversion.md`、`doc/02_training.md`（含 §2.1 改写为 uv）、
  `doc/03_openloop_testing.md`、`doc/robot/deployment.md`、`启动脚本.md`、`AGENTS.md`。
  `doc/reports/`、`doc/archive/` 是历史记录，保留原样。
- 本仓库原先 `pyproject.toml` 的 `dependencies = []` 且**没有** `uv.lock`（飞书文档假设它们已存在），
  本次已补：依赖列表 + `[tool.uv]` 索引/来源，`uv sync` 后生成 `uv.lock`。
- README 的评估章节写 `pip install numpy==1.24.4`，与 `requirements.txt` 的
  `numpy==1.26.4` 冲突；本环境按 `requirements.txt` 保留 **1.26.4**（`[待确认]` 是否需要 1.24.4）。
