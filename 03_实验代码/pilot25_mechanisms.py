"""Protocol 16.41: fixed mechanism design check, no neural learning."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from linear_gaussian_filter import kalman_sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work/pilot25-mechanisms-20261105'
RUN = BASE / 'run'
PROTOCOL = ROOT / '02_验证阶段/单人流程/参考/PIM实验设计-v1.md'
CONTEXTS = [(0,0),(0,1),(1,0),(1,1)]
M = np.array([[1.,-1.],[1.,1.]]) / np.sqrt(2.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def budget(start):
    if time.process_time() - start > 600:
        raise RuntimeError('BUDGET_STOP: CPU > 600 seconds')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) > 200_000_000:
        raise RuntimeError('BUDGET_STOP: disk > 200 MB')


def model(context):
    omega = np.array([(.15,.30)[context[0]],(.40,.55)[context[1]]])*np.pi
    a, r = 1.8*np.cos(omega), .9
    gamma = a/(1+r*r)
    q = (1-r*r)*((1+r*r)-a*a/(1+r*r))
    F, Q, P = np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))
    for i in range(2):
        s = slice(2*i,2*i+2)
        F[s,s] = [[a[i],-r*r],[1.,0.]]
        P[s,s] = [[1.,gamma[i]],[gamma[i],1.]]
        Q[2*i,2*i] = q[i]
    H = np.zeros((2,4)); H[:,[0,2]] = M
    return F,H,Q,np.eye(2)*.01,np.zeros(4),P


def generate(args, initial, innovation, measurement):
    F,H,Q,R,m,P = args
    state = np.empty((256,97,4))
    state[:,0] = initial @ np.linalg.cholesky(P).T
    for t in range(96):
        state[:,t+1] = state[:,t] @ F.T
        state[:,t+1,[0,2]] += innovation[:,t] * np.sqrt(np.diag(Q)[[0,2]])
    return state, state @ H.T + .1*measurement


def metrics(y, predictions, posterior, context):
    target = y[:,33:97]
    mse = {key:np.mean(((target-value[:,32:96]) @ M)**2, axis=(0,1)).tolist()
           for key,value in predictions.items()}
    acc = []
    for i in range(2):
        prob1 = posterior[:,:,[j for j,c in enumerate(CONTEXTS) if c[i]==1]].sum(axis=2)
        acc.append(float(np.mean((prob1[:,32:96] > .5) == bool(context[i]))))
    return {'mse':mse, 'identification_accuracy':acc}


def run():
    RUN.mkdir(exist_ok=False)
    start = time.process_time()
    try:
        accepted = json.loads((BASE/'review-final.json').read_text(encoding='utf-8'))
        assert accepted['passed'] == accepted['total']
        sources = [Path(__file__), Path(__file__).with_name('linear_gaussian_filter.py'),
                   Path(__file__).with_name('pilot25_audit.py'), BASE/'verify_filter.py', PROTOCOL]
        hashes = {}
        for path in sources:
            shutil.copyfile(path, RUN/path.name)
            hashes[str(path.relative_to(ROOT))] = digest(path)
        save(RUN/'manifest.json', {'protocol':'1.3.40/16.41','sources':hashes,
             'python':sys.version,'numpy':np.__version__,'roots':[9501,9502],
             'stage':501,'splits':[0,1],'B':256,'T':97,'score_t':[32,95],
             'helper_review_sha256':digest(BASE/'review-final.json'),
             'started_local':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        models = [model(c) for c in CONTEXTS]
        premise = []
        for c,(F,H,Q,R,m,P) in zip(CONTEXTS,models):
            item = {'context':c,'radius':float(max(abs(np.linalg.eigvals(F)))),
                    'min_process_q':float(min(np.diag(Q)[[0,2]])),
                    'stationarity_error':float(np.max(abs(F@P@F.T+Q-P))),
                    'marginal_error':float(np.max(abs(H@P@H.T+R-1.01*np.eye(2))))}
            item['pass'] = bool(item['radius']<1 and item['min_process_q']>0 and
                                item['stationarity_error']<=1e-10 and item['marginal_error']<=1e-10)
            premise.append(item)
        if not all(p['pass'] for p in premise):
            save(RUN/'result.json', {'status':'DESIGN_PREMISE_FAILED','premise':premise})
            return
        gamma_mean = np.mean([np.diag(args[0])[[0,2]]/1.81 for args in models],axis=0)
        current_map = M @ np.diag(gamma_mean/1.01) @ M.T
        rows, pair_checks, groups = [], [], []
        for root in [9501,9502]:
            for split in [0,1]:
                rng = np.random.default_rng(np.random.SeedSequence([root,501,split]))
                initial = rng.normal(size=(256,4))
                innovation = rng.normal(size=(256,96,2))
                measurement = rng.normal(size=(256,97,2))
                np.savez(RUN/f'exogenous-{root}-{split}.npz', initial=initial,innovation=innovation,measurement=measurement)
                group, generated = [], []
                for c,args in zip(CONTEXTS,models):
                    budget(start)
                    state,y = generate(args,initial,innovation,measurement)
                    generated.append((state,y))
                    results = [kalman_sequence(y,*candidate) for candidate in models]
                    ll = np.stack([r['log_likelihood'] for r in results],axis=2)
                    next_all = np.stack([r['next_observation'] for r in results],axis=2)
                    logw = np.cumsum(ll,axis=1)
                    weight = np.exp(logw-logw.max(axis=2,keepdims=True))
                    weight /= weight.sum(axis=2,keepdims=True)
                    predictions = {'mixture':np.einsum('btk,btkd->btd',weight,next_all),
                                   'oracle':next_all[:,:,CONTEXTS.index(c)],
                                   'current':y@current_map.T,'zero':np.zeros_like(y),'persistence':y.copy()}
                    assert all(np.isfinite(v).all() for v in [y,state,ll,next_all,weight,*predictions.values()])
                    assert np.max(abs(weight.sum(axis=2)-1)) <= 1e-10
                    row = {'root':root,'split':split,'context':c,**metrics(y,predictions,weight,c)}
                    rows.append(row); group.append(row)
                    np.savez(RUN/f'data-{root}-{split}-{c[0]}{c[1]}.npz',
                             state=state,y=y,ll=ll,next_all=next_all,posterior=weight,**predictions)
                for i in range(2):
                    for fixed in range(2):
                        c0 = (0,fixed) if i==0 else (fixed,0)
                        c1 = (1,fixed) if i==0 else (fixed,1)
                        st0,y0 = generated[CONTEXTS.index(c0)]
                        st1,y1 = generated[CONTEXTS.index(c1)]
                        delta = (y1-y0)@M
                        other = 1-i
                        unchanged = bool(np.array_equal(st0[:,:,[2*other,2*other+1]],st1[:,:,[2*other,2*other+1]]))
                        error = float(np.max(abs(delta[:,:,other])))
                        rms = float(np.sqrt(np.mean(delta[:,:,i]**2)))
                        pair_checks.append({'root':root,'split':split,'changed':i,'fixed':fixed,
                                            'unchanged_exact':unchanged,'leakage':error,'changed_rms':rms,
                                            'pass':unchanged and error<=1e-10 and rms>=.1})
                mm = {key:np.mean([row['mse'][key] for row in group],axis=0) for key in predictions}
                accuracy = np.mean([row['identification_accuracy'] for row in group],axis=0)
                ratio_current, ratio_oracle = mm['mixture']/mm['current'], mm['mixture']/mm['oracle']
                ok = bool(np.all(ratio_current<=.8) and np.all(ratio_oracle<=1.25) and np.all(accuracy>=.75))
                groups.append({'root':root,'split':split,'mse':{k:v.tolist() for k,v in mm.items()},
                               'ratio_current':ratio_current.tolist(),'ratio_oracle':ratio_oracle.tolist(),
                               'accuracy':accuracy.tolist(),'pass':ok})
        budget(start)
        outcome = all(x['pass'] for x in premise+pair_checks+groups)
        save(RUN/'result.json', {'status':'P0_DESIGN_PASS' if outcome else 'DESIGN_PREMISE_FAILED',
             'premise':premise,'pairs':pair_checks,'rows':rows,'groups':groups,
             'cpu_seconds':time.process_time()-start,'neural_training':False,'audit':'PENDING'})
        save(RUN/'artifacts.json', {p.name:digest(p) for p in RUN.iterdir() if p.is_file() and p.name!='artifacts.json'})
        print(json.dumps({'status':'P0_DESIGN_PASS' if outcome else 'DESIGN_PREMISE_FAILED','groups':groups,'cpu_seconds':time.process_time()-start}))
    except Exception as exc:
        (RUN/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        save(RUN/'error.json', {'status':'BUDGET_STOP' if 'BUDGET_STOP' in str(exc) else 'ERROR','error':str(exc)})
        raise


if __name__ == '__main__':
    run()
