"""Synthetic boundary tests, no registered discovery/confirmation data generated."""
import numpy as np
import torch
from pilot45_fixed_joint import (intervene, windows, normal_window, conclusions,
                                 advance, Network, SUBSETS, TAUS)
from pilot45_joint_audit import forward, mutate, q95


def run():
    torch.set_num_threads(1)
    torch.manual_seed(45)
    rng = np.random.default_rng(45)
    h = rng.normal(size=(3, 8, 12)).astype('float32')
    subset, target = np.arange(6), np.arange(6, 12)
    altered, detail, reason = intervene(h, subset, target)
    independent, other_detail, other_reason = mutate(h, tuple(subset), tuple(target))
    np.testing.assert_array_equal(altered, independent)
    np.testing.assert_allclose(detail, other_detail, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(reason, other_reason)
    assert not reason.any()
    for random_value, target_value, expected_bits, expected_alpha in [(0., 0., 0, 0.), (0., 1., 9, None), (1., 11., 2, 11.), (1., 2., 0, 2.)]:
        state = np.zeros((1, 8, 12), 'float32')
        state[..., subset], state[..., target] = random_value, target_value
        changed, d, mask = intervene(state, subset, target)
        assert (mask == expected_bits).all()
        if expected_alpha is not None:
            np.testing.assert_allclose(d[..., 2], expected_alpha)
            if expected_alpha == 2.: assert (changed[..., subset] == -1).all()
    seq = np.broadcast_to(np.arange(97)[None, :, None], (3, 97, 2)).copy()
    assert windows(seq)[0, 0, 0, 0] == 34 and windows(seq)[0, -1, -1, 0] == 93
    assert normal_window(seq)[0, 0, 0, 0] == 33 and normal_window(seq)[0, -1, -1, 0] == 92
    net = Network().eval()
    x = rng.normal(size=(3, 96, 2)).astype('float32')
    with torch.no_grad():
        hs, _ = net.gru(torch.from_numpy(x))
        pred = net.head(hs).numpy()
        initial = hs.numpy()[:, TAUS]
        future = torch.from_numpy(np.stack([x[:, t+1:t+5] for t in TAUS], 1).reshape(-1, 4, 2))
        null = advance(net, future, initial)
        np.testing.assert_allclose(null, normal_window(pred), atol=2e-5, rtol=2e-5)
        w = {k: v.numpy() for k, v in net.state_dict().items()}
        alt, _ = forward(future.numpy(), w, initial.reshape(-1, 12))
        np.testing.assert_allclose(null, alt.reshape(3, 8, 4, 2), atol=2e-5, rtol=2e-5)
    assert SUBSETS.shape == (924, 6) and np.array_equal(SUBSETS[0], np.arange(6))
    values = rng.normal(size=128)
    np.testing.assert_allclose(q95(values), np.quantile(values, .95, method='linear'), atol=1e-14)
    rows = [dict(network=f'{root}-{kind}', axis=a, J=.2 if kind == 'main' else .1, matching_valid=True, local_pass=True)
            for root in (9701, 9702) for kind in ('main', 'frozen', 'initial') for a in range(2)]
    normal = [dict(kind='main', pass_axes=[True, True])]
    assert conclusions(rows, normal)['status'] == 'P2_JOINT_CONTROL_PASS'
    rows[0]['local_pass'] = False
    assert conclusions(rows, normal)['status'] == 'LOCAL_CAUSAL_FAILED'
    rows[0]['matching_valid'] = False
    c = conclusions(rows, normal)
    assert c['status'] == 'MATCHING_INVALID' and c['local_status'] == c['learning_status'] == 'NOT_EVALUATED'
    print('SYNTHETIC_CHECKS_PASS: zero denominator, non-clipping, sign flip, indexing, null continuation, independent gates, invalid precedence')


if __name__ == '__main__':
    run()
