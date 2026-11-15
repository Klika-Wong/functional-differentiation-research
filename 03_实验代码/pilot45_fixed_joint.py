"""Protocol 16.67: frozen joint interventions; no training or extraction.

Separate CLI invocations: discover -> audit (other module) -> lock -> confirm.
Only lock is an acceptance action, performed after Codex reviews the audit.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from pilot44_fixed import ROOT, PROTOCOL, now, read, save
from pilot44_learning import Network, data, reference, metrics
from pilot25_mechanisms import M

BASE = ROOT / 'work/pilot45-fixed-joint-20261114'
UP = ROOT / 'work/pilot44-fixed-20261110'
ROOTS = (9701, 9702)
KINDS = ('main', 'frozen', 'initial')
TAUS = np.arange(32, 89, 8)
SUBSETS = np.array(list(combinations(range(12), 6)), dtype=np.int64)
SOURCES = [Path(__file__), Path(__file__).with_name('pilot45_joint_audit.py'),
           Path(__file__).with_name('pilot45_joint_checks.py'), PROTOCOL,
           *[Path(__file__).with_name(n) for n in
             ('pilot44_fixed.py', 'pilot44_learning.py', 'pilot25_mechanisms.py', 'linear_gaussian_filter.py')]]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(start, reserve=0):
    if time.process_time() - start > 600:
        raise RuntimeError('BUDGET_STOP CPU')
    size = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
    if size + reserve > 1_000_000_000:
        raise RuntimeError('BUDGET_STOP disk')


def weight_name(root, kind):
    return f'{root}-{kind}.npz' if kind == 'initial' else f'{root}-{kind}-s2000.npz'


def prerequisite():
    result = UP / 'p1-run/result.json'
    a, b = read(UP / 'p1-audit.json'), read(UP / 'p1-acceptance.json')
    assert read(result)['status'] == 'LEARNING_GATE_PASS'
    assert a['status'] == 'AUDIT_PASS' and b['status'] == 'ACCEPTED_P1_LEARNING_PASS'
    assert a['result_sha256'] == b['result_sha256'] == digest(result)
    assert b['audit_sha256'] == digest(UP / 'p1-audit.json')
    for name, expected in read(UP / 'p1-run/artifacts.json').items():
        assert digest(UP / 'p1-run' / name) == expected, name
    final = read(UP / 'p1-run/final-lock.json')['files']
    assert set(final) == {weight_name(r, k) for r in ROOTS for k in KINDS[:2]}
    for name, expected in final.items():
        assert digest(UP / 'p1-run' / name) == expected
    return {str(p.relative_to(UP)): digest(p) for p in
            [result, UP / 'p1-audit.json', UP / 'p1-acceptance.json', UP / 'p1-run/artifacts.json',
             UP / 'p1-run/final-lock.json']}


def manifest(directory, upstream):
    dest = directory / 'sources'
    dest.mkdir()
    for p in SOURCES:
        shutil.copyfile(p, dest / p.name)
    weights = {}
    for root in ROOTS:
        for kind in KINDS:
            name = weight_name(root, kind)
            shutil.copyfile(UP / 'p1-run' / name, directory / name)
            weights[name] = digest(directory / name)
    save(directory / 'manifest.json', dict(protocol='1.3.67/16.67', started_at=now(),
         sources={str(p.relative_to(ROOT)): digest(p) for p in SOURCES}, weights=weights,
         upstream=upstream, numpy=np.__version__, torch=torch.__version__, python=sys.version,
         threads=1, omp=os.environ.get('OMP_NUM_THREADS'), openblas=os.environ.get('OPENBLAS_NUM_THREADS'),
         operator='single-state-zero; actual future inputs; independent normal start per tau',
         output_layout='candidate,episode,tau,forecast-step,mixture-axis',
         score_reduction='float64 numpy mean; candidate lexicographic first exact maximum',
         match_layout=['target_norm', 'random_norm', 'alpha', 'actual_norm'],
         invalid_bits={'1': 'zero denominator with nonzero target', '2': 'alpha outside [0,10]',
                       '4': 'actual norm mismatch', '8': 'nonfinite'}))


def verify_seal(directory):
    for name, expected in read(directory / 'artifacts.json').items():
        assert digest(directory / name) == expected, name
    m = read(directory / 'manifest.json')
    for name, expected in m['sources'].items():
        assert digest(ROOT / name) == expected, name
        assert digest(directory / 'sources' / Path(name).name) == expected
    for name, expected in m['weights'].items():
        assert digest(directory / name) == expected
    return m


def seal(directory, result, start):
    check(start)
    result.update(cpu_seconds=time.process_time() - start, completed_at=now(), audit='PENDING')
    save(directory / 'result.json', result)
    save(directory / 'artifacts.json', {str(p.relative_to(directory)): digest(p)
         for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'artifacts.json'})
    check(start)
    print(json.dumps(dict(phase=directory.name, status=result['status'],
                         cpu_seconds=time.process_time()-start), ensure_ascii=False), flush=True)


def load_net(directory, root, kind):
    with np.load(directory / weight_name(root, kind)) as f:
        w = {k: f[k] for k in f.files}
    net = Network()
    net.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def windows(y):
    return np.stack([y[:, t+2:t+6] for t in TAUS], axis=1)


def normal_window(p):
    return np.stack([p[:, t+1:t+5] for t in TAUS], axis=1)


def squared_errors(p, target):
    return ((np.asarray(p, dtype=np.float64) - target) @ M) ** 2


def normal_cache(directory, root, kind, y):
    net = load_net(directory, root, kind)
    x = torch.from_numpy(y[:, :96].astype(np.float32))
    h, _ = net.gru(x)
    p = net.head(h).numpy()
    hs = h.numpy()
    tag = f'{root}-{kind}'
    np.save(directory / f'{tag}-normal-hidden.npy', hs)
    np.save(directory / f'{tag}-normal-pred.npy', p)
    initial = hs[:, TAUS].copy()
    future = np.stack([y[:, t+1:t+5] for t in TAUS], 1).astype(np.float32)
    future = torch.from_numpy(future.reshape(-1, 4, 2))
    null = advance(net, future, initial)
    np.testing.assert_allclose(null, normal_window(p), atol=2e-5, rtol=2e-5)
    np.save(directory / f'{tag}-null.npy', null)
    return net, future, initial, p


def advance(net, future, initial):
    h = torch.from_numpy(np.ascontiguousarray(initial.reshape(-1, 12)))[None]
    seq, _ = net.gru(future, h)
    return net.head(seq).numpy().reshape(len(initial), 8, 4, 2)


def intervene(initial, subset, match=None):
    h = initial.copy()
    outside = np.setdiff1d(np.arange(12), subset)
    details = reasons = None
    if match is None:
        h[..., subset] = 0
    else:
        d = np.linalg.norm(initial[..., match].astype(np.float64), axis=-1)
        b = np.linalg.norm(initial[..., subset].astype(np.float64), axis=-1)
        alpha = np.zeros_like(d)
        good = b > 1e-12
        np.divide(d, b, out=alpha, where=good)
        impossible = (~good) & (d > 1e-12)
        alpha[impossible] = np.nan
        # No clipping: multiplication/subtraction are explicitly float32.
        h[..., subset] -= alpha.astype(np.float32)[..., None] * h[..., subset]
        actual = np.linalg.norm(h.astype(np.float64)-initial, axis=-1)
        details = np.stack([d, b, alpha, actual], axis=-1)
        reasons = impossible.astype(np.uint8)
        reasons |= (((alpha < 0) | (alpha > 10)).astype(np.uint8) * 2)
        reasons |= ((abs(actual-d) > 1e-6+1e-5*d).astype(np.uint8) * 4)
        reasons |= ((~np.isfinite(details).all(-1)).astype(np.uint8) * 8)
    np.testing.assert_array_equal(h[..., outside], initial[..., outside])
    return h, details, reasons


def evaluate(directory, tag, net, future, initial, target, baseline, scale, specs, start):
    n, b = len(specs), len(initial)
    shape = (n, b, 8, 4, 2)
    check(start, int(np.prod(shape))*4+128)
    output = np.lib.format.open_memmap(directory / f'{tag}-outputs.npy', mode='w+', dtype='float32', shape=shape)
    output[:] = np.nan
    matched_ids = [i for i, s in enumerate(specs) if s['type'] == 'matched']
    match_slot = {i: j for j, i in enumerate(matched_ids)}
    details = reasons = None
    if matched_ids:
        check(start, len(matched_ids)*b*8*(4*8+1)+256)
        details = np.lib.format.open_memmap(directory / f'{tag}-match.npy', mode='w+', dtype='float64', shape=(len(matched_ids), b, 8, 4))
        reasons = np.lib.format.open_memmap(directory / f'{tag}-invalid.npy', mode='w+', dtype='uint8', shape=(len(matched_ids), b, 8))
        details[:] = np.nan
        reasons[:] = 255
    losses = np.empty((n, 2))
    episodes = np.empty((n, b, 2))
    temporal = np.empty((n, 8, 4, 2))
    invalid_count = 0
    for i, spec in enumerate(specs):
        subset = SUBSETS[spec['subset_id']]
        match = SUBSETS[spec['target_id']] if spec['type'] == 'matched' else None
        h, detail, reason = intervene(initial, subset, match)
        if detail is not None:
            j = match_slot[i]
            details[j], reasons[j] = detail, reason
            invalid_count += int(np.count_nonzero(reason))
            if not np.isfinite(h).all():
                output.flush(); details.flush(); reasons.flush()
                save(directory / f'{tag}-incomplete.json', dict(status='MATCHING_INVALID', at_spec=i,
                     completed_outputs=i, invalid_count=invalid_count, stopped_at=now()))
                raise RuntimeError('MATCHING_INVALID nonfinite intervention; cannot continue')
        pred = advance(net, future, h)
        assert np.isfinite(pred).all()
        output[i] = pred
        err = squared_errors(pred, target)
        losses[i] = err.mean(axis=(0, 1, 2))
        episodes[i] = err.mean(axis=(1, 2))
        temporal[i] = err.mean(axis=0)
        if i % 32 == 0:
            check(start)
    output.flush()
    if details is not None:
        details.flush(); reasons.flush()
    diff = losses-baseline
    delta = diff/scale
    scores = delta-delta[:, ::-1]
    np.savez(directory / f'{tag}-scores.npz', L=losses, difference=diff, delta=delta, J=scores,
             episode_L=episodes, temporal_L=temporal)
    save(directory / f'{tag}-specs.json', dict(specs=specs, matched_ids=matched_ids))
    check(start)
    print(f'{directory.name} {tag}: {n} interventions; invalid={invalid_count}', flush=True)
    return scores, delta, invalid_count


def discover(start):
    upstream = prerequisite()
    d = BASE / 'discovery'
    d.mkdir(parents=True, exist_ok=False)
    manifest(d, upstream)
    np.save(d / 'subsets.npy', SUBSETS)
    randoms = {str(r): np.random.default_rng(np.random.SeedSequence([r, 647, 0])).choice(924, 128, replace=False).tolist() for r in ROOTS}
    save(d / 'random-lists.json', dict(generated_at=now(), indices=randoms))
    specs = [dict(type='zero', subset_id=i) for i in range(924)]
    candidates = dict(scales={}, networks={}, random_indices=randoms)
    for root in ROOTS:
        y = data(root, 4, 128)
        np.save(d / f'{root}-data.npy', y)
        save(d / f'{root}-data-event.json', dict(split=4, generated_at=now(), episodes=128))
        gamma = 1.8*np.cos(np.pi*np.array([.15, .4]))/1.81
        current = ((y[:, :96] @ M)*gamma/1.01) @ M.T
        target = windows(y)
        C = squared_errors(normal_window(current), target).mean((0, 1, 2))
        assert np.isfinite(C).all() and (C > 0).all()
        candidates['scales'][str(root)] = C.tolist()
        for kind in KINDS:
            tag = f'{root}-{kind}'
            net, future, initial, p = normal_cache(d, root, kind, y)
            normal_err = squared_errors(normal_window(p), target)
            L0 = normal_err.mean((0, 1, 2))
            np.savez(d / f'{tag}-baseline.npz', L=L0, episode_L=normal_err.mean((1, 2)), temporal_L=normal_err.mean(0))
            J, _, _ = evaluate(d, tag, net, future, initial, target, L0, C, specs, start)
            ids = J.argmax(axis=0).tolist()
            intersection = sorted(set(SUBSETS[ids[0]]) & set(SUBSETS[ids[1]]))
            candidates['networks'][tag] = dict(selected_ids=ids, subsets=SUBSETS[ids].tolist(),
                 J=[float(J[ids[a], a]) for a in range(2)], intersection=[int(x) for x in intersection],
                 jaccard=len(intersection)/(12-len(intersection)))
    save(d / 'candidates.json', candidates)
    seal(d, dict(status='DISCOVERY_COMPLETE', interventions=5544,
         candidates_sha256=digest(d / 'candidates.json'), confirmation='NOT_RUN'), start)


def lock():
    d = BASE / 'discovery'
    prerequisite(); m = verify_seal(d)
    a = read(BASE / 'discovery-audit.json')
    assert a['status'] == 'AUDIT_PASS'
    assert a['result_sha256'] == digest(d / 'result.json')
    assert a['candidates_sha256'] == digest(d / 'candidates.json')
    assert a['artifacts_sha256'] == digest(d / 'artifacts.json')
    assert not (BASE / 'discovery-lock.json').exists() and not (BASE / 'confirmation').exists()
    save(BASE / 'discovery-lock.json', dict(status='ACCEPTED_DISCOVERY_LOCKED', accepted_by='Codex',
         locked_at=now(), result_sha256=digest(d / 'result.json'), audit_sha256=digest(BASE / 'discovery-audit.json'),
         candidates_sha256=digest(d / 'candidates.json'), artifacts_sha256=digest(d / 'artifacts.json'),
         weights=m['weights'], candidates=read(d / 'candidates.json')))
    print('ACCEPTED_DISCOVERY_LOCKED', flush=True)


def confirm_specs(selected, randoms):
    return ([dict(type='target', axis=a, subset_id=selected[a]) for a in range(2)] +
            [dict(type='same', random_id=j, subset_id=s) for j, s in enumerate(randoms)] +
            [dict(type='matched', axis=a, random_id=j, subset_id=s, target_id=selected[a])
             for a in range(2) for j, s in enumerate(randoms)])


def condition_rows(tag, J, delta, reasons):
    rows = []
    for a in range(2):
        same = J[2:130, a]
        matched = J[130+a*128:130+(a+1)*128, a]
        qs = float(np.quantile(same, .95, method='linear'))
        qm = float(np.quantile(matched, .95, method='linear'))
        valid = bool((reasons[a*128:(a+1)*128] == 0).all())
        basic = bool(delta[a, a] >= .05 and J[a, a] >= .05 and delta[a, 1-a] <= .5*delta[a, a])
        adv_same = bool(J[a, a] >= qs+.02)
        adv_match = bool(J[a, a] >= qm+.02)
        rows.append(dict(network=tag, axis=a, delta_target=float(delta[a, a]), delta_other=float(delta[a, 1-a]),
             J=float(J[a, a]), Q_same=qs, Q_match=qm, basic=basic, same_advantage=adv_same,
             matched_advantage=adv_match, matching_valid=valid,
             local_pass=(basic and adv_same and adv_match) if valid else None))
    return rows


def conclusions(rows, normal_rows):
    main = [r for r in rows if r['network'].endswith('-main')]
    comparisons = []
    for root in ROOTS:
        for a in range(2):
            vals = {k: next(r['J'] for r in rows if r['network'] == f'{root}-{k}' and r['axis'] == a) for k in KINDS}
            threshold = max(vals['frozen'], vals['initial'])+.02
            comparisons.append(dict(root=root, axis=a, **vals, threshold=threshold, passed=vals['main'] >= threshold))
    valid = all(r['matching_valid'] for r in rows)
    baseline = all(all(r['pass_axes']) for r in normal_rows if r['kind'] == 'main')
    local = all(r['local_pass'] for r in main) if valid and baseline else None
    learning = all(r['passed'] for r in comparisons) if valid and baseline else None
    status = ('MATCHING_INVALID' if not valid else 'BASELINE_GATE_FAILED' if not baseline else
              'LOCAL_CAUSAL_FAILED' if not local else 'LEARNING_COMPARISON_FAILED' if not learning else 'P2_JOINT_CONTROL_PASS')
    return dict(status=status, local_status='NOT_EVALUATED' if local is None else ('LOCAL_CAUSAL_PASS' if local else 'LOCAL_CAUSAL_FAILED'),
                learning_status='NOT_EVALUATED' if learning is None else ('LEARNING_COMPARISON_PASS' if learning else 'LEARNING_COMPARISON_FAILED'),
                baseline_pass=baseline, matching_valid=valid, comparisons=comparisons)


def confirm(start):
    upstream = prerequisite()
    m = verify_seal(BASE / 'discovery')
    locked = read(BASE / 'discovery-lock.json')
    assert locked['status'] == 'ACCEPTED_DISCOVERY_LOCKED'
    for key, path in [('audit_sha256', BASE / 'discovery-audit.json'),
                      ('result_sha256', BASE / 'discovery/result.json'),
                      ('candidates_sha256', BASE / 'discovery/candidates.json'),
                      ('artifacts_sha256', BASE / 'discovery/artifacts.json')]:
        assert locked[key] == digest(path)
    assert locked['weights'] == m['weights'] and locked['candidates'] == read(BASE / 'discovery/candidates.json')
    d = BASE / 'confirmation'
    d.mkdir(exist_ok=False)
    manifest(d, upstream)
    save(d / 'discovery-binding.json', dict(lock_sha256=digest(BASE / 'discovery-lock.json'), locked_at=locked['locked_at']))
    cand = locked['candidates']; rows = []; normal_rows = []
    for root in ROOTS:
        y = data(root, 5, 512)
        np.save(d / f'{root}-data.npy', y)
        save(d / f'{root}-data-event.json', dict(split=5, episodes=512, generated_at=now(), lock_sha256=digest(BASE / 'discovery-lock.json')))
        ref = reference(y)
        np.save(d / f'{root}-reference.npy', ref)
        target = windows(y); C = np.array(cand['scales'][str(root)])
        for kind in KINDS:
            tag = f'{root}-{kind}'
            net, future, initial, p = normal_cache(d, root, kind, y)
            normal_rows.append(dict(root=root, kind=kind, **metrics(y, p, ref)))
            normal_err = squared_errors(normal_window(p), target)
            L0 = normal_err.mean((0, 1, 2))
            np.savez(d / f'{tag}-baseline.npz', L=L0, episode_L=normal_err.mean((1, 2)), temporal_L=normal_err.mean(0))
            specs = confirm_specs(cand['networks'][tag]['selected_ids'], cand['random_indices'][str(root)])
            J, delta, _ = evaluate(d, tag, net, future, initial, target, L0, C, specs, start)
            reasons = np.load(d / f'{tag}-invalid.npy', mmap_mode='r')
            rows.extend(condition_rows(tag, J, delta, reasons))
    result = dict(**conclusions(rows, normal_rows), rows=rows, normal_rows=normal_rows,
                  interventions=2316, extraction='NOT_RUN', reuse='NOT_RUN')
    seal(d, result, start)


def accept():
    d = BASE / 'confirmation'
    verify_seal(d)
    a, r = read(BASE / 'confirmation-audit.json'), read(d / 'result.json')
    assert a['status'] == 'AUDIT_PASS' and a['result_sha256'] == digest(d / 'result.json')
    assert a['artifacts_sha256'] == digest(d / 'artifacts.json')
    assert not (BASE / 'acceptance.json').exists()
    save(BASE / 'acceptance.json', dict(status='ACCEPTED_P2_RESULT', scientific_status=r['status'],
         accepted_by='Codex', accepted_at=now(), result_sha256=digest(d / 'result.json'),
         audit_sha256=digest(BASE / 'confirmation-audit.json'), discovery_lock_sha256=digest(BASE / 'discovery-lock.json'),
         extraction='NOT_RUN', reuse='NOT_RUN', human_acceptance='NOT_CLAIMED'))
    print('ACCEPTED_P2_RESULT: '+r['status'], flush=True)


def entry(action):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.process_time()
    try:
        with torch.no_grad():
            if action == 'discover': discover(start)
            elif action == 'confirm': confirm(start)
            elif action == 'lock': lock()
            elif action == 'accept': accept()
            else: raise ValueError(action)
    except BaseException as exc:
        BASE.mkdir(exist_ok=True, parents=True)
        name = f'{action}-error-{time.time_ns()}'
        (BASE / (name+'.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        status = 'BUDGET_STOP' if 'BUDGET_STOP' in str(exc) else 'MATCHING_INVALID' if 'MATCHING_INVALID' in str(exc) else 'ERROR'
        save(BASE / (name+'.json'), dict(status=status, error=str(exc), time=now(), cpu_seconds=time.process_time()-start))
        raise


if __name__ == '__main__':
    entry(sys.argv[1])
