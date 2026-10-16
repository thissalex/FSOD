#!/usr/bin/env python3
"""Summarize actual completed runs and describe unresolved evidence explicitly."""
import json
import csv
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    official=[];domains=[]
    lines=['# 第一阶段续做结果（2026-10-16）','',
           '本次从用户下载的作者原始压缩包继续。未更改官方测试标注；开发实验仅使用两个开发域的训练部分。',
           '', '## 数据审计与开发协议','',
           '- NEU-DET：1440 张训练图、360 张测试图；两张完全重复训练图按组隔离。',
           '- Clipart1k：500 张训练图、500 张测试图；训练图 `681633840.jpg` 的声明尺寸 600×594、实际尺寸 300×213，保留原始数据但排除出开发划分。',
           '- 开发查询集分别为 NEU-DET 288 张、Clipart1k 100 张，剩余有效训练图作为支持池；每类保留至少 5 个支持实例和 1 个查询实例。',
           '- 划分种子为 20261016。每域生成 1/5-shot、种子 0—9 的清单；当前核心预研使用种子 0—2。',
           '- 仅检查并隔离了完全重复 RGB 图像；感知近重复或潜在序列关系仍需审查，不能据此声称完成最终协议验收。',
           '', '## 官方配置 B0','',
           '以下为原始官方测试集的 5-shot 实测，不用于选择方法或参数。完整测试集结果与 10 图工程检查分开保存。',
           '', '| 域 | 状态 | AP | AP50 | AP75 | 完整运行秒数 | 本进程峰值显存 GiB |',
           '|---|---|---:|---:|---:|---:|---:|']
    for domain,slug in [('NEU-DET','neu'),('clipart1k','clipart')]:
        record=read(ROOT/f'results/resume-{slug}-official-full/run.json')
        if record:
            official.append({'domain':domain,**{k:record.get(k) for k in
                             ['status','metrics','elapsed_seconds','peak_gpu_memory_gb','baseline_commit',
                              'model_weights_hash','support_manifest_hash','query_manifest_hash','implementation_hash']}})
        if record and record.get('status')=='completed' and record.get('metrics'):
            m=record['metrics'];lines.append(f'| {domain} | completed | {m["ap"]:.3f} | {m["ap50"]:.3f} | {m["ap75"]:.3f} | {record["elapsed_seconds"]:.1f} | {record["peak_gpu_memory_gb"]:.2f} |')
        else:lines.append(f'| {domain} | {record.get("status","pending") if record else "pending"} | — | — | — | — | — |')
    lines+=['','完整 B0 耗时包含模型加载、支持集处理、查询推理及评价；显存为本进程峰值分配量，不包含其他 GPU 进程。缓存重算耗时含 COCO 评价，不能视为实际新查询的端到端推理耗时，也不构成低成本适应主张的完整证据。',
            '', '论文 v1 表 3 的 5-shot 参考值为 NEU-DET 7.4 AP、Clipart1k 43.7 AP，参见 [FSOD-VFM 论文](https://arxiv.org/html/2602.03137v1)。这是原论文摘录，与本机实测分别列示。已核查到一项明确差异：论文 §3.3 写最多 500 个 UPN 候选，当前锁定代码实际取前 100 个；本次遵循代码的 100 个候选，所有内部对照一致，不能称为严格复现论文的全部条件。SAM2/DINOv2 版本和未安装 xFormers 的影响仍需核对；小幅 AP 差异不归因于本方法。',
            '', '## 开发集核心对照','',
            'λ=10、μ=1 为预登记网格中的固定预研点，没有用测试集选参。B1/B5/Ours 共用四个扰动视图、固定原始掩码和原型聚合。查询缓存保留原始候选顺序，并重新计算类别与图扩散；每域先通过官方即时结果的恒等重算校验。',
            '', '| 域 | shot | 方法 | 已完成支持组数 | 平均 AP ± 样本标准差 |',
            '|---|---:|---|---:|---:|']
    mechanisms=[]
    repairs=[]
    for domain in ['NEU-DET','clipart1k']:
        directory=ROOT/'results/resume'/domain
        status=read(directory/'status.json') or {'status':'pending'}
        rows=read(directory/'paired_results.json') or []
        sensitivity=read(directory/'sensitivity.json') or []
        cache=read(directory/'query_cache_validation.json')
        summaries=read(directory/'paired_summary.json')
        repair=read(directory/'pixel_guard_repair.json')
        if repair:repairs.append({'domain':domain,**repair})
        domains.append({'domain':domain,'status':status,'evaluations':rows,'paired_summary':summaries,
                        'cache_validation':cache})
        for shot in [5,1]:
            for method in (['B0','B1','B2','B5','Ours'] if shot==5 else ['B0','B1','Ours']):
                values=[r['ap'] for r in rows if r['shot']==shot and r['method']==method]
                if not values:continue
                sd=statistics.stdev(values) if len(values)>1 else 0
                lines.append(f'| {domain} | {shot} | {method} | {len(values)} | {statistics.mean(values):.3f} ± {sd:.3f} |')
        # Only summarize 5-shot views to avoid counting reused 1-shot instances twice.
        instances=[i for batch in sensitivity if batch['shot']==5 for i in batch['instances']]
        if instances:
            noise=[abs(i['repeat_noise']) for i in instances]
            per_view=[statistics.mean(i['cosine_drift'][j] for i in instances) for j in range(5)]
            leaked=[i for i in instances if max(i['target_input_max_delta_by_view'])>0]
            mechanisms.append({'domain':domain,'support_instance_appearances':len(instances),
                               'independent_samples_note':'Repeated views and repeated support instances are not independent observations',
                               'mean_cosine_drift_by_view':per_view,'max_absolute_repeat_noise':max(noise),
                               'target_input_pixel_leaks':len(leaked),
                               'zero_perturbable_instances':sum(i['perturbable_fraction']==0 for i in instances)})
    lines+=['', '## 机制、稳定性与阶段结论','']
    if repairs:
        lines.append('初次完整检查在 1/5-shot 支持集中共发现 5 次缩放插值泄漏（NEU-DET 1 个、Clipart1k 4 个）。已保存修复前快照到本地 artifacts/resume_before_pixel_guard，按 0/1/2/4/8/16 像素的固定顺序扩展保护区，选择使预处理后目标像素完全相等的最小 guard；不参考 AP、类别错误或特征值。原图、原图掩码、候选框和查询特征保持一致，受影响支持组的对照及网格已重算。下表使用修复后的结果。')
    for item in mechanisms:
        drift=item['mean_cosine_drift_by_view']
        lines.append(f'- {item["domain"]}：5-shot 共 {item["support_instance_appearances"]} 次支持实例出现；两档模糊平均余弦漂移 {drift[1]:.6g}/{drift[2]:.6g}，两档亮度漂移 {drift[3]:.6g}/{drift[4]:.6g}；重复原图最大绝对数值噪声 {item["max_absolute_repeat_noise"]:.3g}；预处理后目标区存在像素变化的实例数 {item["target_input_pixel_leaks"]}。')
    for domain in domains:
        if domain['paired_summary']:
            for summary in domain['paired_summary']:
                if summary['control']!='B1':continue
                lo,hi=summary['ci95_percentile']
                lines.append(f'- {domain["domain"]} {summary["shot"]}-shot：Ours−B1 平均 {summary["mean_delta"]:+.3f} AP，配对 bootstrap 95% 区间 [{lo:+.3f}, {hi:+.3f}]，正增益比例 {summary["positive_fraction"]:.0%}。仅三组时属于筛选证据。')
    errors=read(ROOT/'results/resume/context_error_diagnostics.json') or []
    for diagnostic in errors:
        lines.append(f'- {diagnostic["domain"]} 固定候选框对开发查询真值的 IoU≥0.5 召回率为 {diagnostic["candidate_recall50"]:.1%}。类别与域内去均值后的漂移/错误关系只作描述：{diagnostic["within_category_demeaned_spearman"]}；三组抽样不足以证明 H2，更不能作因果解释。')
    complete=all(d['status'].get('status')=='completed' for d in domains)
    grids={d['domain']:read(ROOT/'results/resume'/d['domain']/'prespecified_grid.json') or [] for d in domains}
    grid_complete=all((read(ROOT/'results/resume'/d['domain']/'grid_status.json') or {}).get('status')=='completed'
                      for d in domains)
    decision={'core_complete':complete,'grid_complete':grid_complete,'six_domain_expansion':False}
    if grid_complete:
        keys=sorted({(r['lambda'],r['mu']) for r in grids['NEU-DET']})
        candidates=[]
        for lam,mu in keys:
            per_domain={}
            all_delta=[]
            for domain,rows in grids.items():
                baseline={r['seed']:r['ap'] for r in rows if r['lambda']==0}
                selected={r['seed']:r['ap'] for r in rows if r['lambda']==lam and r['mu']==mu}
                if set(baseline)!=set(selected) or len(selected)!=3:raise ValueError('Incomplete grid pairing')
                delta=[selected[s]-baseline[s] for s in sorted(baseline)]
                all_delta.extend(delta)
                per_domain[domain]={'mean_ap':statistics.mean(selected.values()),
                                    'delta_vs_B1':statistics.mean(delta),'paired_delta':delta}
            candidates.append({'lambda':lam,'mu':mu,'mean_paired_delta':statistics.mean(all_delta),
                               'positive_fraction':sum(x>0 for x in all_delta)/len(all_delta),
                               'domains':per_domain})
        best=max(candidates,key=lambda r:(r['mean_paired_delta'],-r['lambda'],-r['mu']))
        gains=[d['delta_vs_B1'] for d in best['domains'].values()]
        numeric_gate=(best['lambda']>0 and best['mean_paired_delta']>0 and max(gains)>=0.5 and
                      min(gains)>=-0.2 and best['positive_fraction']>0.5)
        decision.update(best_shared_development_setting=best,numeric_resource_gate_passed=numeric_gate,
                        recommendation=('validate_on_new_support_seeds_before_expansion' if numeric_gate else
                                        'pause_scaling_and_revisit_context_adaptation'),
                        note='Grid selection and evaluation share development labels; gains are exploratory, not confirmation')
        lines+=['','## 预登记参数网格与投入决策','',
                '仅检查 λ∈{0,1,10,100}、μ∈{0,1,10}；λ=0 的三种重复恒等配置合并。跨两个开发域等权选一个共享参数，不按域选最优，不用官方测试集选择。',
                f'共享开发最优点为 λ={best["lambda"]}、μ={best["mu"]}，相对 B1 的两域平均增益为 {best["mean_paired_delta"]:+.4f} AP，六次域/支持组配对中的正增益比例为 {best["positive_fraction"]:.0%}。这些是选参集内结果，不能作为确认性提升。']
        for domain,entry in best['domains'].items():
            lines.append(f'- {domain}：平均 AP {entry["mean_ap"]:.4f}，相对 B1 {entry["delta_vs_B1"]:+.4f} AP。')
        lines+=['',('达到预登记数值资源门槛，但仍需新增支持种子和机制证据，不能直接进入六域。' if numeric_gate else
                   '**阶段决定：暂停扩大实验规模，修改或重新审查成对适应假设。** 当前预登记网格尚未达到至少一个开发域 +0.5 AP、另一个域不明显退化等投入门槛；不能将多视图平均的收益归于特征变换。')]
        write_json_path=ROOT/'results/resume/grid_selection.json'
        write_json_path.write_text(json.dumps({'candidates':candidates,'decision':decision},indent=2)+'\n')
    lines+=['',('两域核心预研均已完成。' if complete else '核心预研尚未全部完成，见每域 status.json；不得将部分输出视为完整实验。'),
            '当前不自动扩展六域：H2 的域内/类内错误关联、与论文性能的基线核对、近重复审查和新增支持集稳定性验证仍需完成。若发现预处理后的目标像素发生变化，相关 H1 结果仅作诊断，必须先修正扰动保护再重跑。',
            '', '## 复现入口与 Git 边界','',
            '```bash','conda activate fsod','python scripts/prepare_development.py',
            'CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1_cached.py --domain NEU-DET',
            'CUDA_VISIBLE_DEVICES=1 python scripts/run_stage1_cached.py --domain clipart1k',
            'python scripts/summarize_stage1.py','```','',
            '本次只提交新增执行工具、测试和经检查的汇总结果。此前未提交的实验基础文件保持原状，不捎带上传；因此完整运行仍依赖工作区既有的 fsod_experiments 基础模块、配置与官方源码。数据、权重、缓存、预测明细和临时下载均不纳入提交。']
    output=ROOT/'results/resume';output.mkdir(exist_ok=True)
    for filename,rows in [('baseline_results.csv',[
            {'domain':r['domain'],'status':r['status'],**(r.get('metrics') or {}),
             'elapsed_seconds':r.get('elapsed_seconds'),'peak_gpu_memory_gb':r.get('peak_gpu_memory_gb')}
            for r in official]),('paired_results.csv',[r for d in domains for r in d['evaluations']])]:
        fields=sorted({key for row in rows for key in row})
        with (output/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps({'official':official,'development':domains,
                                                 'mechanism':mechanisms,'pixel_guard_repairs':repairs,
                                                 'complete':complete,'decision':decision},indent=2)+'\n')
    (ROOT/'stage1_resume_results.md').write_text('\n'.join(lines)+'\n')
    print(ROOT/'stage1_resume_results.md')


if __name__=='__main__':main()
