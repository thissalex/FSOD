# S1B 最小复现清单

执行日期：2026-10-16。阶段一快照保持在 `results/resume/`；新结果在 `results/stage1b/`。先阅读 `protocol_audit_v2.md`，NEU-DET 的开发查询划分已经修订，不能混用两个版本的AP。

必需代码为本提交的 `fsod_experiments/`、`configs/`、实验/准备脚本与相关测试；文件SHA256、完整包版本及外部源版本见 `results/stage1b/repro_manifest.json`。未提交数据、权重、缓存、环境凭据。旧版通用 `upstream_adapter.py` 为早期入口；本阶段和阶段一正式缓存实验使用 `run_stage1_cached.py` 与经过插值保护修复的 `pixel_protection.py`。

环境：Conda `fsod`，Python3.10.21，PyTorch2.6.0+cu118，torchvision0.21.0+cu118，CUDA工具链11.8，驱动535.183.01，RTX4090。推理FP32，不使用autocast，无xFormers；TF32等实际开关见机器清单。SAM2为SAM2.1 large，DINOv2为ViT-L/14、1024维；本地版本、权重内容摘要已记录。UPN扩展已编译，SAM2为可编辑安装。

外部代码：

- FSOD-VFM：`6da781bf02046fe6f70530e1d5483abd9aef6c9c`，73个Git blob逐一验证，本地源未改写。原文件仅在内存中插入捕获回调/替换特征提取函数。
- SAM2：`2b90b9f5ceec907a1c18123530e92e794ad901a4`，45个源文件按Git blob验证。
- DINOv2：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`，检查已跟踪文件无改动。
- 公共源码清单归档于 `results/stage1b/source_manifests/`；来源为各自作者GitHub仓库。

数据为用户下载的作者 CDFSOD-benchmark 六个压缩包；原包SHA256见 `results/resume/archive_manifest.json`，本次检查当前文件大小一致，未重新下载。仅两个开发域已解压使用，其他四域未做方法评估。权重位于本地 `checkpoints/`，摘要分别在清单中；不将模型文件上传Git。数据源/权重下载地址沿用固定官方仓库说明，没有臆测未提供的下载来源。

建议在相同驱动/Linux环境创建Python3.10环境，安装torch2.6.0/torchvision0.21.0的cu118构建，再安装 `requirements.txt`。运行 `scripts/fetch_baseline.py` 和 `scripts/fetch_component.py` 获取固定公共源；DINOv2检出上述提交到 `external/dinov2`。将已有权重/数据按清单放置，配置官方目录到 `checkpoints/` 与 `dinov2/` 的符号链接，使用 `scripts/build_extensions.py --component upn` 构建扩展，SAM2可编辑安装。编译脚本要求fsod环境内CUDA11.8工具链；全新机器仍需另行配置该编译依赖，包清单不是无需编译的容器镜像。

原始支持清单、查询清单及其SHA256由 `scripts/prepare_development.py` 与 `results/resume/development_protocol.json` 对应。v2只通过 `scripts/revise_protocol_stage1b.py` 应用已审查的查询排除，输出 `artifacts/development_v2/`。两个域支持池与种子0—9支持清单不变。没有把原始COCO标注上传Git；须由本地数据重建。

每实例为原图+四视图（模糊半径2/5、亮度0.6/1.4）；原图SAM掩码固定，保护框按10%扩张，必要时依像素一致性增加确定性插值保护。支持特征缓存的权威键来自 `results/resume/<domain>/sensitivity.json`，包含修复后的键，不能从修复前的运行身份推测。查询候选、掩码及原始特征缓存复用；v2只读取新的查询ID子集。

检测固定UPN coarse提示、阈值0.01、NMS阈值1；按官方代码排序/裁剪前100框。论文500与源码100的差异保留记录。图扩散steps=30、alpha=0.3，最终分数为最大类别相似度乘以非负(1−ios)^0.5，COCO maxDets=[1,10,100]。不把方法λ与衰减指数混用。T3仅额外生成UPN框，没有对新增框提取SAM/DINO特征，也没有报告新增框的检测AP。

S1B执行入口（均在项目根目录，使用fsod的Python）：

```bash
python scripts/audit_stage1b.py
python scripts/audit_registration_stage1b.py
python scripts/review_registration_stage1b.py
# 必须人工/视觉确认候选；duplicate_review.json记录本次已完成的决定
python scripts/revise_protocol_stage1b.py
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_stage1b.py --domain NEU-DET
CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_stage1b.py --domain clipart1k
python scripts/check_nonidentity_stage1b.py
python scripts/proposal_budget_stage1b.py
python scripts/recompute_controls_stage1b.py
python scripts/summarize_stage1b.py
python -m pytest -q
```

逐种子B1重算、支持缓存键、支持清单SHA256在两域 `seed_result_index.json`；旧逐方法λ/μ/AP和运行身份见 `results/resume/<domain>/paired_results.json` 与 `prespecified_grid.json`，v2核心结果另存 `controls_v2.json`。阶段一失败/修复日志原地保留，S1B日志在本地 `logs/stage1b*`，其摘要、状态和文件摘要随结果归档。完整预测与大缓存不进Git。

干净副本恒等重算入口：从本次暂存树/提交导出独立代码副本，在副本外使用 `python /tmp/<copy>/scripts/replay_stage1b.py --data-root /path/to/FSOD --output /path/to/report.json`。此检查显式使用已有数据缓存和已校验官方源，验证代码快照可导入并重算；不等同于全新CUDA环境安装或从零全基准复现。实际校验结果见 `results/stage1b/clean_replay.json`。非恒等校验使用即时UPN/SAM/DINO和原始官方后处理，与缓存路径对照，另见 `nonidentity_parity.json`。
