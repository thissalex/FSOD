#!/usr/bin/env python3
"""Summarize versioned diagnostics without relaxing preregistered gates."""
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fsod_experiments.records import write_json,paired_summary,file_digest


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def combined_csv(name):
    rows=[]
    for domain in ['NEU-DET','clipart1k']:
        rows.extend(read_csv(ROOT/'results/stage1b'/domain/name))
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return rows


def main():
    out=ROOT/'results/stage1b';domains=['NEU-DET','clipart1k']
    for domain in domains:
        status=json.loads((out/domain/'diagnostic_status.json').read_text())
        if status['status']!='completed' or status['grid_rows']!=30:raise ValueError('Incomplete grid diagnostics')
    controls=json.loads((out/'controls_v2.json').read_text())
    if len(controls)!=48 or json.loads((out/'controls_status.json').read_text())['status']!='completed':raise ValueError('Incomplete control repair')
    for domain in domains:
        for shot in [1,5]:
            expected=15 if shot==5 else 9
            if sum(r['domain']==domain and r['shot']==shot for r in controls)!=expected:raise ValueError('Missing core cell')
    transform=combined_csv('transform_diagnostics.csv');prediction=combined_csv('prediction_diagnostics.csv')
    margins=read_csv(out/'class_margin_diagnostics.csv')
    margin_index={(r['domain'],r['seed'],r['lambda'],r['mu'],r['category']):r for r in margins}
    for r in prediction:
        key=tuple(r[k] for k in ['domain','seed','lambda','mu','category'])
        if key in margin_index:r.update(margin_index[key])
    with (ROOT/'prediction_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in prediction for k in r)),lineterminator='\n');writer.writeheader();writer.writerows(prediction)
    grid=[r for r in prediction if r['category']=='ALL'];assert len(grid)==60
    indexed={(r['domain'],int(r['seed']),int(r['lambda']),int(r['mu'])):r for r in grid}
    # Cross-check separately rerun core controls vs corresponding diagnostic grid.
    for r in controls:
        if r['shot']==5 and r['method'] in ['B1','B5','Ours']:
            g=indexed[(r['domain'],r['seed'],r['lambda'],r['mu'])]
            if abs(r['ap']-float(g['ap']))>1e-5:raise AssertionError('v2 control/grid disagree')
    settings=sorted({(int(r['lambda']),int(r['mu'])) for r in grid if int(r['lambda'])>0})
    comparisons=[]
    for lam,mu in settings:
        means={d:float(np.mean([float(indexed[d,s,lam,mu]['delta_ap']) for s in range(3)])) for d in domains}
        comparisons.append({'lambda':lam,'mu':mu,'domain_delta_ap':means,'equal_domain_delta_ap':float(np.mean(list(means.values())))})
    best=max(comparisons,key=lambda r:r['equal_domain_delta_ap']);diag=[]
    for d in domains:
        tr=[r for r in transform if r['domain']==d and r['lambda']=='100' and r['mu']=='1']
        pr=[indexed[d,s,100,1] for s in range(3)]
        diag.append({'domain':d,'lambda':100,'mu':1,'r_E_mean':float(np.mean([float(r['r_E']) for r in tr])),
            'r_E_range':[min(float(r['r_E']) for r in tr),max(float(r['r_E']) for r in tr)],
            'r_P_mean':float(np.mean([float(r['r_P']) for r in tr])),
            'query_cosine_p95_mean':float(np.mean([float(r['query_cosine_drift_p95']) for r in tr])),
            'max_similarity_abs_delta_p95_mean':float(np.mean([float(r['max_similarity_abs_delta_p95']) for r in pr])),
            'class_flip_rate_mean':float(np.mean([float(r['class_flip_rate']) for r in pr])),
            'rank_pair_change_rate_mean':float(np.mean([float(r['within_class_rank_change_rate']) for r in pr])),
            'final_confidence_abs_delta_p95_mean':float(np.mean([float(r['final_confidence_abs_delta_p95']) for r in pr])),
            'delta_ap_mean':float(np.mean([float(r['delta_ap']) for r in pr])),
            'query_zero_features':int(tr[0]['query_near_zero_count']),
            'query_candidates':int(pr[0]['candidate_count']),
            'C_trace_mean':float(np.mean([float(r['C_trace']) for r in tr])),
            'C_effective_rank_mean':float(np.mean([float(r['C_effective_rank']) for r in tr])),
            'mu10_minus_mu0_ap_mean':float(np.mean([float(indexed[d,s,100,10]['ap'])-float(indexed[d,s,100,0]['ap']) for s in range(3)]))})
    cfg=json.loads((ROOT/'configs/stage1b.json').read_text());rule=cfg['weak_transform_rule'];nonid=[r for r in transform if int(r['lambda'])>0]
    weak=(min(float(r['r_E']) for r in nonid)>=rule['all_nonidentity_rE_min'] and
          max(float(r['query_cosine_drift_p95']) for r in nonid)<=rule['query_cosine_p95_max'] and
          max(float(r['max_similarity_abs_delta_p95']) for r in grid if int(r['lambda'])>0)<=rule['max_similarity_absolute_delta_p95_max'])
    if weak:raise RuntimeError('T2 is required; complete bounded calibration before deciding')
    eligible=[r for r in comparisons if max(r['domain_delta_ap'].values())>=.5 and min(r['domain_delta_ap'].values())>=-.2]
    if eligible:raise RuntimeError('Candidate near practical gate: require fair B3 before heldout decision')
    for name in ['nonidentity_parity.json']:
        if not all(r['passed'] for r in json.loads((out/name).read_text())):raise ValueError('Parity failed')
    proposal=read_csv(ROOT/'proposal_recall_budget.csv')
    overall=[r for r in proposal if r['category']=='ALL' and r['size']=='ALL']
    summaries=[]
    for d in domains:
        left={str(s):next(r['ap'] for r in controls if r['domain']==d and r['shot']==5 and r['seed']==s and r['method']=='B1') for s in range(3)}
        lam,mu=best['lambda'],best['mu'];right={str(s):float(indexed[d,s,lam,mu]['ap']) for s in range(3)}
        summaries.append({'domain':d,'comparison':'shared development-selected best vs B1; selection-biased descriptive interval',**paired_summary(left,right)})
    result={'decision':'stop_current_closed_form','protocol':'v2','grid_evaluations':60,'core_controls':48,
        'core_recomputed':sum(not r['reused_v1'] for r in controls),'core_reused_unchanged':sum(r['reused_v1'] for r in controls),
        'best_shared_development_setting':best,'all_shared_settings':comparisons,'strong_transform':diag,
        'T2':'skipped: existing grid already suppresses support responses and changes query scores',
        'T4':'skipped: no configuration meets practical development gate; seeds3-9 remain untested',
        'B3':'not required: no candidate advances to T4; no claim of beating ordinary covariance',
        'paired_development_descriptive':summaries,'proposal_overall':overall,
        'max_solve_relative_residual':max(float(r['solve_relative_residual']) for r in transform),
        'max_identity_formula_residual':max(float(r['identity_formula_residual']) for r in transform)}
    write_json(out/'summary.json',result)
    text=['# S1B 阶段决策（2026-10-16）','',
        '**停止当前成对闭式变换形式的扩展投入。** 保留上下文敏感性证据与实验平台；不启动六域、COCO或新增支持集筛选。此结论限于当前两开发域、表征和扰动方案，不外推所有成对适应方法。','',
        '## 协议与完整性','',
        'T0发现NEU-DET三对跨支持池/开发查询的平移近重复；依内容审查规则发布协议v2，排除查询侧3图/8目标，NEU查询为285图/674目标。支持池、支持清单、官方测试与旧结果保持原样。Clipart1k为100图/352目标。采集序列未知，审计不能保证不存在更复杂同源变体。详见 `protocol_audit_v2.md`。','',
        '受影响的NEU原核心24次方法评估（包括原1-shot结果的协议修复）全部重算，Clipart1k24次核心结果因清单摘要不变复用。5-shot两域原网格共60次重新检查，核心B1/B5/Ours与网格交叉核对一致。未改变候选、掩码、图扩散或评价预算。','',
        '非恒等校验为λ=100、μ=1，两域各2图：重新运行UPN/SAM/DINO，在即时路径对特征通道应用A并变换支持原型，由官方后处理重新分组/扩散。与固定缓存结果的最大分数差≤1.79e-7，预设容差2e-5；全部通过。选中图均在v2内。另检查所有种子强变换的独立对齐诊断与生产重放一致。干净代码副本重算见 `results/stage1b/clean_replay.json`。','',
        '## T1：作用已存在，尚无实用增益','',
        '| 域（种子0—2均值） | r_E | r_P | 查询余弦漂移P95 | 最大相似度变化P95 | 类别翻转率 | 最终置信度变化P95 | ΔAP vs B1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in diag:
        text.append(f"| {r['domain']} λ100/μ1 | {r['r_E_mean']:.4f} | {r['r_P_mean']:.4f} | {r['query_cosine_p95_mean']:.6f} | {r['max_similarity_abs_delta_p95_mean']:.6f} | {100*r['class_flip_rate_mean']:.3f}% | {r['final_confidence_abs_delta_p95_mean']:.6f} | {r['delta_ap_mean']:+.5f} |")
    text+=['', '上述P95先在每个完整支持集运行内计算，再对3种子求均值。查询漂移排除零向量，零向量候选仍按原流程保留在打分、图扩散和AP中，不将其余弦距离错误写成1。', '']
    for r in diag:text.append(f"- {r['domain']}：{r['query_zero_features']}/{r['query_candidates']} 个原始候选特征为零；不为改善结果修改官方空掩码行为。C平均trace={r['C_trace_mean']:.6f}、有效秩={r['C_effective_rank_mean']:.2f}。同类候选排序对改变率均值={r['rank_pair_change_rate_mean']*100:.3f}%。")
    text+=['',f"求解最大相对残差为{result['max_solve_relative_residual']:.3g}；A−I恒等式最大绝对残差为{result['max_identity_formula_residual']:.3g}；A与变换查询无NaN/Inf，支持特征无近零向量。完整C/S谱、类别间隔、翻转、排序和逐类AP差值分别见 `transform_diagnostics.csv`、`prediction_diagnostics.csv`。",'',
        f"共享开发最优仍为λ={best['lambda']}、μ={best['mu']}，两域等权ΔAP={best['equal_domain_delta_ap']:+.6f}。该点由种子0—2选择，属于选参集内结果；不将其配对区间解释为独立确认。"]
    text.append('')
    for d,value in best['domain_delta_ap'].items():text.append(f'- {d}：平均ΔAP={value:+.6f}。')
    text+=['','μ改变r_E/r_P及部分预测，但没有形成稳定实用AP收益；不保留“判别保持项有效”的贡献主张。','',
        '## 条件任务与候选框诊断','',
        '跳过T2：已登记的强配置明显抑制E，且查询分数、类别与最终排序发生变化，原因并非所有变换都近似恒等。跳过T4和B3：没有候选达到约+0.5AP的开发门槛，不消耗保留种子寻找显著性；未运行校准或保留组，因此不生成对应空结果表。','',
        '| 域 | 预算 | R@IoU0.5 | R@IoU0.75 | 实际框数均值 |','|---|---:|---:|---:|---:|']
    for d in domains:
        for b in [100,300,500]:
            a=next(r for r in overall if r['domain']==d and int(r['budget'])==b and float(r['iou_threshold'])==.5)
            z=next(r for r in overall if r['domain']==d and int(r['budget'])==b and float(r['iou_threshold'])==.75)
            text.append(f"| {d} | {b} | {float(a['recall'])*100:.2f}% | {float(z['recall'])*100:.2f}% | {float(a['actual_count_mean']):.1f} |")
    text+=['','UPN阈值0.01、coarse提示、NMS=1，按官方顺序先截断再裁剪无效框。前100框在v2所有385张查询图上与旧缓存逐坐标一致，预算集合嵌套。仅生成框，没有对新增框运行SAM/DINO，也未测新增预算的AP。NEU类别、COCO绝对面积小/中/大分组及漏覆盖实例在召回CSV和 `proposal_summary.json`；召回改善不等于检测AP改善或mAP上限。','',
        '## 下一步唯一核心问题','',
        '唯一核心问题：NEU-DET的UPN前100截断造成的召回损失，能否在固定特征、分类与后处理规则下转化为最终AP损失？T3已提示截断贡献，应将不同候选预算作为独立工程基线核查，Clipart1k用于对照定位精度与排序限制。此处尚未评估新增预算的AP，不能声称性能提升或算法创新。下一任务先登记同预算比较与成本测量协议。','',
        '种子3—9没有运行方法评估，四个确认域未参与方法选择。H1保留为这两个域当前扰动下的表示现象；H2/H3未建立。支持出现次数、唯一实例数与环境/缓存身份见复现清单。资源与验证见 `results/stage1b/validation.json`；最终Git提交/上传由任务结束时单独报告。']
    (ROOT/'stage1b_decision.md').write_text('\n'.join(text)+'\n')
    # Small publication-style artifact, using computed values only.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    for d in domains:
        points=[r for r in transform if r['domain']==d and r['mu']=='1']
        axes[0].scatter([float(r['r_E']) for r in points],[float(indexed[d,int(r['seed']),int(r['lambda']),1]['max_similarity_abs_delta_p95']) for r in points],label=d,s=25)
        for threshold,style in [(.5,'-'),(.75,'--')]:
            values=[next(float(r['recall']) for r in overall if r['domain']==d and int(r['budget'])==b and float(r['iou_threshold'])==threshold) for b in [100,300,500]]
            axes[1].plot([100,300,500],values,style,marker='o',label=f'{d} IoU {threshold}')
    axes[0].set(xlabel='Support response ratio r_E',ylabel='P95 absolute max-similarity change');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Proposal budget',ylabel='Proposal recall',ylim=(0,1.02));axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(best),flush=True)
if __name__=='__main__':main()
