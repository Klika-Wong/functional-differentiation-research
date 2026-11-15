"""Separate-process numerical audit of protocol 16.67.

Reuses only file/identity helpers and constants. Does not call execution
interventions, forward, losses, selection, quantiles or scientific verdicts.
"""
from itertools import combinations
import math
import sys
import time
import traceback
import numpy as np
from pilot45_fixed_joint import (BASE, ROOT, UP, ROOTS, KINDS, prerequisite,
                                 verify_seal, digest, weight_name, read, save, now, check)

TIMES = list(range(32, 89, 8))
GROUPS = list(combinations(range(12), 6))
MIX = np.array([[1., -1.], [1., 1.]])/math.sqrt(2)


def close(a, b, atol=1e-10, rtol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def forward(x, weights, initial=None):
    """Float64 three-gate recurrence, arbitrary start state and sequence length."""
    w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    h = np.zeros((len(x), 12)) if initial is None else initial.astype(np.float64).copy()
    hidden = []
    wr, wz, wn = np.split(w['gru.weight_ih_l0'], 3)
    ur, uz, un = np.split(w['gru.weight_hh_l0'], 3)
    br, bz, bn = np.split(w['gru.bias_ih_l0'], 3)
    cr, cz, cn = np.split(w['gru.bias_hh_l0'], 3)
    for observation in np.asarray(x, dtype=np.float64).transpose(1, 0, 2):
        reset = 1/(1+np.exp(-(observation@wr.T+br+h@ur.T+cr)))
        update = 1/(1+np.exp(-(observation@wz.T+bz+h@uz.T+cz)))
        candidate = np.tanh(observation@wn.T+bn+reset*(h@un.T+cn))
        h = (1-update)*candidate+update*h
        hidden.append(h.copy())
    states = np.stack(hidden, 1)
    return states@w['head.weight'].T+w['head.bias'], states


def regenerate(root, split, count):
    rng = np.random.default_rng(np.random.SeedSequence([root, 644, split]))
    init = rng.normal(size=(count, 4))
    innovations = rng.normal(size=(count, 96, 2))
    noise = rng.normal(size=(count, 97, 2))
    sources = []
    for axis, frequency in enumerate((.15, .4)):
        coefficient = 1.8*math.cos(math.pi*frequency)
        gamma = coefficient/1.81
        q = .19*(1.81-coefficient*coefficient/1.81)
        current = init[:, axis*2]
        previous = gamma*current+math.sqrt(1-gamma*gamma)*init[:, axis*2+1]
        seq = [current.copy()]
        for t in range(96):
            new = coefficient*current-.81*previous+math.sqrt(q)*innovations[:, t, axis]
            previous, current = current, new
            seq.append(new.copy())
        sources.append(np.stack(seq, 1))
    return np.stack(sources, 2)@MIX.T+.1*noise


def reference_path(y):
    x = y@MIX
    prediction = np.empty((len(y), 96, 2))
    for axis, frequency in enumerate((.15, .4)):
        a = 1.8*math.cos(math.pi*frequency)
        f = np.array([[a, -.81], [1., 0.]])
        q = np.diag([.19*(1.81-a*a/1.81), 0.])
        p = np.array([[1., a/1.81], [a/1.81, 1.]])
        mean = np.zeros((len(y), 2))
        for t in range(96):
            gain = p[:, 0]/(p[0, 0]+.01)
            posterior = mean+(x[:, t, axis]-mean[:, 0])[:, None]*gain
            mean = posterior@f.T
            prediction[:, t, axis] = mean[:, 0]
            residual = np.eye(2)-np.outer(gain, [1., 0.])
            p = f@(residual@p@residual.T+.01*np.outer(gain, gain))@f.T+q
    return prediction@MIX.T


def average2(array):
    flat = array.reshape(-1, 2)
    return np.array([math.fsum(map(float, flat[:, i]))/len(flat) for i in range(2)])


def errors(p, target):
    diff = np.asarray(p, dtype=np.float64)-target
    # Explicit physical coordinate transform, independent of runner's @ M.
    return np.stack([((diff[..., 0]+diff[..., 1])/math.sqrt(2))**2,
                     ((diff[..., 1]-diff[..., 0])/math.sqrt(2))**2], -1)


def cut(array, offset):
    return np.stack([array[:, t+offset:t+offset+4] for t in TIMES], 1)


def metric_check(y, pred, ref, record):
    target = y[:, 33:97]@MIX
    p, r = np.asarray(pred[:, 32:96], float)@MIX, ref[:, 32:96]@MIX
    gamma = np.array([1.8*math.cos(math.pi*f)/1.81 for f in (.15, .4)])
    current = (y[:, 32:96]@MIX)*gamma/1.01
    L, R, C = average2((p-target)**2), average2((r-target)**2), average2((current-target)**2)
    D, cross = average2((p-r)**2), 2*average2((p-r)*(r-target))
    expected = dict(L=L, R=R, C=C, D=D, Cross=cross, ratio_reference=L/R, ratio_current=L/C,
                    pass_axes=((L <= .8*C) & (L <= 1.25*R)).tolist())
    for key, value in expected.items():
        close(value, record[key])
    close(L, R+D+cross)
    return expected['pass_axes']


def mutate(initial, subset, target=None):
    """Audit float32 intervention itself before float64 future propagation."""
    h = initial.copy()
    if target is None:
        for coordinate in subset:
            h[..., coordinate] = 0
        detail = reason = None
    else:
        target_norm = np.sqrt(np.sum(np.square(initial[..., target].astype(float)), axis=-1))
        random_norm = np.sqrt(np.sum(np.square(initial[..., subset].astype(float)), axis=-1))
        alpha = np.zeros(target_norm.shape)
        nonzero = random_norm > 1e-12
        alpha[nonzero] = target_norm[nonzero]/random_norm[nonzero]
        bad_denominator = (~nonzero) & (target_norm > 1e-12)
        alpha[bad_denominator] = np.nan
        for coordinate in subset:
            h[..., coordinate] = initial[..., coordinate] - alpha.astype('float32')*initial[..., coordinate]
        actual = np.sqrt(np.sum((h.astype(float)-initial.astype(float))**2, axis=-1))
        detail = np.stack([target_norm, random_norm, alpha, actual], -1)
        reason = bad_denominator.astype('uint8')
        reason += 2*((alpha < 0) | (alpha > 10)).astype('uint8')
        reason += 4*(abs(actual-target_norm) > 1e-6+1e-5*target_norm).astype('uint8')
        reason += 8*(~np.isfinite(detail).all(-1)).astype('uint8')
    other = [i for i in range(12) if i not in subset]
    np.testing.assert_array_equal(initial[..., other], h[..., other])
    return h, detail, reason


def q95(values):
    ordered = sorted(map(float, values))
    point = (len(ordered)-1)*.95
    lo = math.floor(point)
    fraction = point-lo
    return ordered[lo]+fraction*(ordered[lo+1]-ordered[lo])


def score_table(directory, tag, y, weights, scale, expected_specs, start, stats):
    saved_h = np.load(directory / f'{tag}-normal-hidden.npy')
    saved_p = np.load(directory / f'{tag}-normal-pred.npy')
    p, hs = forward(y[:, :96].astype('float32'), weights)
    close(p, saved_p, 2e-5, 2e-5); close(hs, saved_h, 2e-5, 2e-5)
    stats['max_normal_difference'] = max(stats['max_normal_difference'], float(abs(p-saved_p).max()), float(abs(hs-saved_h).max()))
    initial = saved_h[:, TIMES]
    future = cut(y, 1).astype('float32').reshape(-1, 4, 2)
    null, _ = forward(future, weights, initial.reshape(-1, 12))
    null = null.reshape(len(y), 8, 4, 2)
    observed_null = np.load(directory / f'{tag}-null.npy')
    close(null, observed_null, 2e-5, 2e-5)
    close(observed_null, cut(saved_p, 1), 2e-5, 2e-5)
    target = cut(y, 2)
    error0 = errors(cut(saved_p, 1), target)
    L0 = average2(error0)
    with np.load(directory / f'{tag}-baseline.npz') as baseline:
        close(L0, baseline['L'])
        close(error0.mean((1, 2)), baseline['episode_L'])
        close(error0.mean(0), baseline['temporal_L'])
    specs_file = read(directory / f'{tag}-specs.json')
    assert specs_file['specs'] == expected_specs
    matched_ids = [i for i, spec in enumerate(expected_specs) if spec['type'] == 'matched']
    assert specs_file['matched_ids'] == matched_ids
    match_slot = {i: j for j, i in enumerate(matched_ids)}
    outputs = np.load(directory / f'{tag}-outputs.npy', mmap_mode='r')
    assert outputs.dtype == np.float32 and outputs.shape == (len(expected_specs), len(y), 8, 4, 2)
    scores = dict(np.load(directory / f'{tag}-scores.npz'))
    details = np.load(directory / f'{tag}-match.npy', mmap_mode='r') if matched_ids else None
    invalid = np.load(directory / f'{tag}-invalid.npy', mmap_mode='r') if matched_ids else None
    if matched_ids:
        assert details.shape == (256, len(y), 8, 4) and invalid.shape == (256, len(y), 8)
    Js, forward_Js = [], []
    for i, spec in enumerate(expected_specs):
        subset = GROUPS[spec['subset_id']]
        target_subset = GROUPS[spec['target_id']] if spec['type'] == 'matched' else None
        h, detail, reason = mutate(initial, subset, target_subset)
        if detail is not None:
            slot = match_slot[i]
            close(detail, details[slot])
            np.testing.assert_array_equal(reason, invalid[slot])
            stats['match_cases'] += int(reason.size)
            stats['invalid_cases'] += int(np.count_nonzero(reason))
            stats['max_alpha'] = max(stats['max_alpha'], float(detail[..., 2].max()))
            stats['max_norm_error'] = max(stats['max_norm_error'], float(abs(detail[..., 0]-detail[..., 3]).max()))
        pred, _ = forward(future, weights, h.reshape(-1, 12))
        pred = pred.reshape(len(y), 8, 4, 2)
        close(pred, outputs[i], 2e-5, 2e-5)
        stats['max_intervention_difference'] = max(stats['max_intervention_difference'], float(abs(pred-outputs[i]).max()))
        e = errors(outputs[i], target)
        L = average2(e)
        diff = L-L0
        delta = diff/scale
        J = np.array([delta[0]-delta[1], delta[1]-delta[0]])
        for key, value in [('L', L), ('difference', diff), ('delta', delta), ('J', J),
                           ('episode_L', e.mean((1, 2))), ('temporal_L', e.mean(0))]:
            close(value, scores[key][i])
        Js.append(J)
        fd = (average2(errors(pred, target))-L0)/scale
        forward_Js.append([fd[0]-fd[1], fd[1]-fd[0]])
        stats['interventions_checked'] += 1
        if i % 32 == 0:
            check(start)
    check(start)
    print(f'audit {directory.name} {tag}: {len(expected_specs)} full outputs and scores checked', flush=True)
    return scores, np.array(Js), np.array(forward_Js), invalid, saved_p


def run(phase):
    start = time.process_time()
    prerequisite()
    d = BASE / phase
    destination = BASE / f'{phase}-audit.json'
    assert not destination.exists()
    m = verify_seal(d)
    result = read(d / 'result.json')
    stats = dict(interventions_checked=0, max_normal_difference=0., max_intervention_difference=0.,
                 max_reference_difference=0., match_cases=0, invalid_cases=0, max_alpha=0., max_norm_error=0.)
    candidate = read(BASE / 'discovery/candidates.json')
    randoms = {str(r): np.random.default_rng(np.random.SeedSequence([r, 647, 0])).choice(len(GROUPS), 128, replace=False).tolist() for r in ROOTS}
    assert candidate['random_indices'] == randoms
    assert read(BASE / 'discovery/random-lists.json')['indices'] == randoms
    np.testing.assert_array_equal(np.load(BASE / 'discovery/subsets.npy'), GROUPS)
    ranking = []; calculated_rows = []; normal_pass = []
    if phase == 'confirmation':
        locked = read(BASE / 'discovery-lock.json')
        assert locked['status'] == 'ACCEPTED_DISCOVERY_LOCKED' and locked['candidates'] == candidate
        for key, path in [('result_sha256', BASE / 'discovery/result.json'), ('audit_sha256', BASE / 'discovery-audit.json'),
                          ('artifacts_sha256', BASE / 'discovery/artifacts.json'), ('candidates_sha256', BASE / 'discovery/candidates.json')]:
            assert locked[key] == digest(path)
        assert locked['weights'] == m['weights']
        assert read(d / 'discovery-binding.json') == dict(lock_sha256=digest(BASE / 'discovery-lock.json'), locked_at=locked['locked_at'])
    for root in ROOTS:
        y = np.load(d / f'{root}-data.npy')
        split, count = (4, 128) if phase == 'discovery' else (5, 512)
        assert y.shape == (count, 97, 2)
        close(y, regenerate(root, split, count), 1e-12, 1e-12)
        event = read(d / f'{root}-data-event.json')
        assert event['split'] == split and event['episodes'] == count
        if phase == 'discovery':
            assert read(d / 'random-lists.json')['generated_at'] <= event['generated_at']
            gammas = np.array([1.8*math.cos(math.pi*f)/1.81 for f in (.15, .4)])
            current = ((y[:, :96]@MIX)*gammas/1.01)@MIX.T
            scale = average2(errors(cut(current, 1), cut(y, 2)))
            close(scale, candidate['scales'][str(root)])
        else:
            assert event['generated_at'] > locked['locked_at']
            assert event['lock_sha256'] == digest(BASE / 'discovery-lock.json')
            scale = np.array(candidate['scales'][str(root)])
            ref = np.load(d / f'{root}-reference.npy')
            independent_ref = reference_path(y)
            close(ref, independent_ref, 1e-8, 1e-8)
            stats['max_reference_difference'] = max(stats['max_reference_difference'], float(abs(ref-independent_ref).max()))
        assert (scale > 0).all() and np.isfinite(scale).all()
        for kind in KINDS:
            tag = f'{root}-{kind}'
            weights = dict(np.load(d / weight_name(root, kind)))
            assert digest(d / weight_name(root, kind)) == digest(UP / 'p1-run' / weight_name(root, kind))
            if phase == 'discovery':
                specs = [dict(type='zero', subset_id=i) for i in range(len(GROUPS))]
            else:
                chosen = candidate['networks'][tag]['selected_ids']
                specs = [dict(type='target', axis=a, subset_id=chosen[a]) for a in range(2)]
                specs += [dict(type='same', random_id=j, subset_id=idx) for j, idx in enumerate(randoms[str(root)])]
                specs += [dict(type='matched', axis=a, random_id=j, subset_id=idx, target_id=chosen[a])
                          for a in range(2) for j, idx in enumerate(randoms[str(root)])]
            scores, scalar_J, numpy_J, invalid, pred = score_table(d, tag, y, weights, scale, specs, start, stats)
            if phase == 'discovery':
                chosen = candidate['networks'][tag]
                ids = [int(np.argmax(scores['J'][:, a])) for a in range(2)]
                assert ids == chosen['selected_ids']
                assert chosen['subsets'] == [list(GROUPS[i]) for i in ids]
                close(chosen['J'], [scores['J'][ids[a], a] for a in range(2)])
                intersect = sorted(set(GROUPS[ids[0]]) & set(GROUPS[ids[1]]))
                assert chosen['intersection'] == intersect
                close(chosen['jaccard'], len(intersect)/(12-len(intersect)))
                ranking.append(dict(network=tag, selected=ids, scalar_argmax=scalar_J.argmax(0).tolist(),
                                    independent_forward_argmax=numpy_J.argmax(0).tolist(),
                                    gaps=[float(np.sort(scores['J'][:, a])[-1]-np.sort(scores['J'][:, a])[-2]) for a in range(2)]))
            else:
                normal_record = next(r for r in result['normal_rows'] if r['root'] == root and r['kind'] == kind)
                pass_axes = metric_check(y, pred, ref, normal_record)
                if kind == 'main': normal_pass.extend(pass_axes)
                for a in range(2):
                    record = next(r for r in result['rows'] if r['network'] == tag and r['axis'] == a)
                    # Gate the sealed float64 metrics; scalar recalculation above validates them.
                    value = float(scores['J'][a, a])
                    dt, other = map(float, [scores['delta'][a, a], scores['delta'][a, 1-a]])
                    qs = q95(scores['J'][2:130, a])
                    qm = q95(scores['J'][130+a*128:130+(a+1)*128, a])
                    valid = bool(np.count_nonzero(invalid[a*128:(a+1)*128]) == 0)
                    basic = dt >= .05 and value >= .05 and other <= dt*.5
                    same_ok, match_ok = value >= qs+.02, value >= qm+.02
                    calculated = dict(network=tag, axis=a, delta_target=dt, delta_other=other, J=value,
                         Q_same=qs, Q_match=qm, basic=basic, same_advantage=same_ok,
                         matched_advantage=match_ok, matching_valid=valid,
                         local_pass=(basic and same_ok and match_ok) if valid else None)
                    for key, value2 in calculated.items():
                        if isinstance(value2, float): close(value2, record[key])
                        else: assert value2 == record[key], (tag, a, key)
                    calculated_rows.append(calculated)
    if phase == 'discovery':
        assert stats['interventions_checked'] == result['interventions'] == 5544
        assert result['status'] == 'DISCOVERY_COMPLETE'
        assert result['candidates_sha256'] == digest(d / 'candidates.json')
        verdict = dict(candidates_sha256=digest(d / 'candidates.json'), ranking_checks=ranking,
                       ranking_stable=all(r['selected'] == r['scalar_argmax'] == r['independent_forward_argmax'] for r in ranking))
    else:
        assert stats['interventions_checked'] == result['interventions'] == 2316
        valid = all(r['matching_valid'] for r in calculated_rows)
        baseline = all(normal_pass)
        local = all(r['local_pass'] for r in calculated_rows if r['network'].endswith('-main')) if valid and baseline else None
        compare_pass = []
        for root in ROOTS:
            for a in range(2):
                values = {k: next(r['J'] for r in calculated_rows if r['network'] == f'{root}-{k}' and r['axis'] == a) for k in KINDS}
                threshold = max(values['initial'], values['frozen'])+.02
                passed = values['main'] >= threshold
                row = next(r for r in result['comparisons'] if r['root'] == root and r['axis'] == a)
                for k in KINDS: close(row[k], values[k])
                close(row['threshold'], threshold); assert row['passed'] == passed
                compare_pass.append(passed)
        learning = all(compare_pass) if valid and baseline else None
        status = ('MATCHING_INVALID' if not valid else 'BASELINE_GATE_FAILED' if not baseline else
                  'LOCAL_CAUSAL_FAILED' if not local else 'LEARNING_COMPARISON_FAILED' if not learning else 'P2_JOINT_CONTROL_PASS')
        assert result['status'] == status
        assert result['matching_valid'] == valid and result['baseline_pass'] == baseline
        assert result['local_status'] == ('NOT_EVALUATED' if local is None else 'LOCAL_CAUSAL_PASS' if local else 'LOCAL_CAUSAL_FAILED')
        assert result['learning_status'] == ('NOT_EVALUATED' if learning is None else 'LEARNING_COMPARISON_PASS' if learning else 'LEARNING_COMPARISON_FAILED')
        assert result['extraction'] == result['reuse'] == 'NOT_RUN'
        verdict = dict(scientific_status=status)
    check(start)
    save(destination, dict(status='AUDIT_PASS', **stats, **verdict,
         result_sha256=digest(d / 'result.json'), artifacts_sha256=digest(d / 'artifacts.json'),
         cpu_seconds=time.process_time()-start, completed_at=now(), independence='separate process and NumPy/scalar path; same author'))
    check(start)
    print(read(destination), flush=True)


if __name__ == '__main__':
    try:
        run(sys.argv[1])
    except BaseException as exc:
        name = f'{sys.argv[1]}-audit-error-{time.time_ns()}'
        BASE.mkdir(parents=True, exist_ok=True)
        (BASE / (name+'.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        save(BASE / (name+'.json'), dict(status='BUDGET_STOP' if 'BUDGET_STOP' in str(exc) else 'ERROR', error=str(exc), at=now()))
        raise
