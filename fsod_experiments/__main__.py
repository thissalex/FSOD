import argparse
import csv
import json
from pathlib import Path

from .records import digest, paired_summary, write_json


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description='FSOD experiment preparation and evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    matrix = sub.add_parser('matrix', help='Write planned E0/E2/E3 method runs, without fake results')
    matrix.add_argument('--output', default='results/planned_runs.csv')
    sample = sub.add_parser('sample-support')
    sample.add_argument('--support-pool', required=True)
    sample.add_argument('--query', required=True)
    sample.add_argument('--support-images', required=True)
    sample.add_argument('--query-images', required=True)
    sample.add_argument('--shot', type=int, required=True)
    sample.add_argument('--seed', type=int, required=True)
    sample.add_argument('--output', required=True)
    summary = sub.add_parser('summarize', help='Paired bootstrap per domain/shot, AP in points')
    summary.add_argument('--results', required=True)
    summary.add_argument('--control', default='B1')
    summary.add_argument('--treatment', default='Ours')
    summary.add_argument('--output', required=True)
    collect = sub.add_parser('collect', help='Collect completed run.json files into a paired-results CSV')
    collect.add_argument('--root', default='results')
    collect.add_argument('--output', default='results/paired_results.csv')
    args = parser.parse_args()
    if args.command == 'matrix':
        rows = []
        for domain in load('configs/stage1.json')['development_domains']:
            for experiment, shot, seeds, methods in [
                ('E0', 5, [0], ['B0']),
                ('E2', 5, range(3), ['B0', 'B1', 'B2', 'B5', 'Ours']),
                ('E3', 1, range(3), ['B0', 'B1', 'Ours'])]:
                for seed in seeds:
                    for method in methods:
                        rows.append(dict(run_id=f'{experiment}-{domain}-{shot}s-s{seed}-{method}',
                                         experiment=experiment, dataset=domain, shot=shot,
                                         support_seed=seed, method=method, status='planned',
                                         ap='', ap50='', ap75=''))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f'Wrote {len(rows)} planned runs to {args.output}; no experiments executed')
    elif args.command == 'sample-support':
        from .data import audit_coco, assert_disjoint, sample_support
        pool, query = load(args.support_pool), load(args.query)
        pa, qa = audit_coco(pool, args.support_images), audit_coco(query, args.query_images)
        assert_disjoint(pool, query, pa, qa)
        support = sample_support(pool, args.shot, args.seed)
        write_json(args.output, support)
        write_json(args.output + '.manifest.json', {
            'shot': args.shot, 'support_seed': args.seed, 'support_hash': digest(support),
            'support_pool_audit': pa, 'query_audit': qa,
            'group_audit': 'group_id checked when supplied; near-duplicate review still required'})
    elif args.command == 'collect':
        fields = ['run_id', 'status', 'dataset', 'shot', 'support_seed', 'method',
                  'support_manifest_hash', 'query_manifest_hash', 'feature_cache_key',
                  'postprocess_config_hash', 'ap', 'ap50', 'ap75', 'elapsed_seconds', 'peak_gpu_memory_gb']
        rows = []
        for path in sorted(Path(args.root).glob('*/run.json')):
            run = load(path)
            if run['status'] != 'completed' or not run.get('metrics'):
                continue
            row = {k: run.get(k, '') for k in fields}
            row.update(dataset=run['args']['domain'], shot=run['args']['shot'],
                       support_seed=run['args']['seed'], method=run['args']['method'],
                       ap=run['metrics']['ap'], ap50=run['metrics']['ap50'], ap75=run['metrics']['ap75'])
            rows.append(row)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        print(f'Collected {len(rows)} completed evaluations')
    else:
        rows = list(csv.DictReader(open(args.results)))
        groups = {}
        for row in rows:
            if row['status'] != 'completed' or row['method'] not in (args.control, args.treatment):
                continue
            # Require identical query, feature extraction and postprocessing conditions.
            key = (row['dataset'], row['shot'], row['query_manifest_hash'],
                   row['feature_cache_key'], row['postprocess_config_hash'])
            group = groups.setdefault(key, {args.control: {}, args.treatment: {}})
            supports = group[row['method']]
            sid = row['support_manifest_hash']
            if sid in supports:
                raise ValueError('Multiple configurations for same support set; lock one before summarizing')
            supports[sid] = float(row['ap'])
        if not groups:
            raise ValueError('No completed paired results')
        output = []
        for key, group in groups.items():
            output.append({'dataset': key[0], 'shot': key[1], 'query_manifest_hash': key[2],
                           'feature_cache_key': key[3], 'postprocess_config_hash': key[4],
                           'control': args.control, 'treatment': args.treatment,
                           **paired_summary(group[args.control], group[args.treatment])})
        write_json(args.output, output)


if __name__ == '__main__':
    main()
