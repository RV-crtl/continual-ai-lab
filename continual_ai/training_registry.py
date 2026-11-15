"""Atomic, auditable promotion of independently evaluated LoRA adapters.

Promotion only changes a local JSON pointer; it never overwrites a working adapter.
No source code, evaluator, or permissions are modified by the AI.
"""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
from .storage import append_jsonl

DEFAULT_ROOT = Path('data/local/training')


def read_active(root: Path = DEFAULT_ROOT) -> dict | None:
    path = root / 'active.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not data.get('adapter') or not data.get('base_model'):
        raise ValueError('Invalid active model manifest')
    if not Path(data['adapter']).is_dir():
        raise FileNotFoundError('Active adapter missing; previous model is still available in history')
    return data


def evaluate_gate(report: dict, min_gain: int = 4, max_kind_loss: int = 1) -> tuple[bool, list[str]]:
    """Two frozen disjoint seed ranges, paired same-model comparisons.

    This is a narrow arithmetic gate, not a proof of general improvement.
    """
    reasons = []
    from .tasks import task_for_seed
    starts = {'primary': 9_000_000, 'audit': 9_100_000}
    for split in ('primary', 'audit'):
        if split not in report or any(k not in report[split] for k in ('baseline', 'candidate')):
            return False, ['missing independent benchmark split: '+split]
        b, c = report[split]['baseline'], report[split]['candidate']
        if b.get('n', 0) < 60 or c.get('n') != b.get('n'):
            reasons.append(f'{split}: at least 60 paired tests required')
        # Reject missing, reordered, mixed, or inconsistent trial logs.
        for label, stats in (('baseline', b), ('candidate', c)):
            trials = stats.get('samples', [])
            if len(trials) != stats.get('n', 0):
                reasons.append(f'{split}/{label}: missing per-task audit trail')
                continue
            if sum(bool(x.get('correct')) for x in trials) != stats.get('correct'):
                reasons.append(f'{split}/{label}: score disagrees with per-task results')
            for i, trial in enumerate(trials):
                task = task_for_seed(starts[split] + i)
                if trial.get('seed') != task.seed or trial.get('kind') != task.kind:
                    reasons.append(f'{split}/{label}: invalid held-out task identity')
                    break
            for kind in {x['kind'] for x in trials}:
                filtered = [x for x in trials if x['kind'] == kind]
                logged = stats.get('kinds', {}).get(kind, {})
                if logged.get('passed') != sum(bool(x['correct']) for x in filtered) or logged.get('total') != len(filtered):
                    reasons.append(f'{split}/{label}: {kind} aggregate mismatch')
        gains = int(c.get('correct', 0)) - int(b.get('correct', 0))
        if split == 'primary' and gains < min_gain:
            reasons.append(f'primary: +{gains} correct, require +{min_gain}')
        if split == 'audit' and gains < 0:
            reasons.append(f'audit: regression of {abs(gains)} correct')
        b_k = b.get('kinds', {})
        c_k = c.get('kinds', {})
        for kind in b_k:
            b_count = b_k[kind]['passed']
            c_count = c_k.get(kind, {}).get('passed', 0)
            if c_count < b_count - max_kind_loss:
                reasons.append(f'{split}: {kind} regressed {b_count-c_count} answers')
    return not reasons, reasons


def promote(report_file: Path, root: Path = DEFAULT_ROOT) -> dict:
    report = json.loads(report_file.read_text(encoding='utf-8'))
    eligible, reasons = evaluate_gate(report)
    if not eligible:
        raise ValueError('Candidate rejected: '+'; '.join(reasons))
    candidate = Path(report['candidate_adapter']).resolve()
    if not candidate.is_dir() or not (candidate / 'adapter_config.json').is_file():
        raise FileNotFoundError('Candidate LoRA adapter is missing or incomplete')
    manifest_path = candidate / 'TRAINING_MANIFEST.json'
    if not manifest_path.is_file():
        raise ValueError('Training manifest missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['base_model'] != report['base_model']:
        raise ValueError('Candidate and benchmark base models differ')
    current = read_active(root)
    old_adapter = str(Path(current['adapter']).resolve()) if current else None
    expected = report.get('incumbent_adapter') or None
    if old_adapter != (str(Path(expected).resolve()) if expected else None):
        raise ValueError('Active model changed after comparison; rerun benchmark')
    root.mkdir(parents=True, exist_ok=True)
    entry = {'base_model': report['base_model'], 'adapter': str(candidate),
             'promoted_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'report': str(report_file.resolve()), 'previous_adapter': old_adapter,
             'primary_gain': report['primary']['candidate']['correct']-report['primary']['baseline']['correct'],
             'audit_gain': report['audit']['candidate']['correct']-report['audit']['baseline']['correct']}
    temp = root / 'active.json.tmp'
    temp.write_text(json.dumps(entry, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, root / 'active.json')
    append_jsonl(root / 'promotions.jsonl', entry)
    return entry


def rollback(root: Path = DEFAULT_ROOT) -> dict:
    current = read_active(root)
    if current is None:
        raise ValueError('No active adapter to roll back')
    old = current.get('previous_adapter')
    if old:
        if not Path(old).is_dir():
            raise FileNotFoundError('Previous adapter no longer on disk')
        new = {'base_model': current['base_model'], 'adapter': old,
               'promoted_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'previous_adapter': None, 'rolled_back_from': current['adapter']}
        tmp = root / 'active.json.tmp'
        tmp.write_text(json.dumps(new, indent=2)+'\n', encoding='utf-8')
        os.replace(tmp, root / 'active.json')
    else:
        (root / 'active.json').unlink()
        new = {'adapter': None, 'rolled_back_from': current['adapter']}
    append_jsonl(root / 'promotions.jsonl', {'action':'rollback', **new})
    return new
