"""Protocol 16.66 P1; fixed-budget ordinary learning and frozen recurrence."""
from pathlib import Path
import sys,time,traceback,shutil,math
import numpy as np
import torch
from torch import nn
from pilot44_fixed import ROOT, BASE, PROTOCOL, sha, save, read, now, numpy_forward
from pilot25_mechanisms import model, M
from linear_gaussian_filter import kalman_sequence
RUN=BASE/'p1-run'

class Network(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(2,12,batch_first=True);self.head=nn.Linear(12,2)
    def forward(self,x):return self.head(self.gru(x)[0])

def weights(net):return {k:v.detach().numpy().copy() for k,v in net.state_dict().items()}

def check(start):
    assert time.process_time()-start<=600,'BUDGET_STOP CPU'
    assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<=500_000_000,'BUDGET_STOP disk'

def prerequisite():
    a=read(BASE/'p0-audit.json');b=read(BASE/'acceptance.json')
    assert a['status']=='AUDIT_PASS' and b['status']=='ACCEPTED_P0_ONLY'
    assert a['result_sha256']==b['result_sha256']==sha(BASE/'p0-run/result.json')
    assert b['audit_sha256']==sha(BASE/'p0-audit.json')
    assert read(BASE/'p0-run/result.json')['status']=='CONSTRUCTION_GATE_PASS'

def data(root,split,B):
    rng=np.random.default_rng(np.random.SeedSequence([root,644,split]))
    init=rng.normal(size=(B,4));eps=rng.normal(size=(B,96,2));obs=rng.normal(size=(B,97,2))
    F,H,Q,V,m,P=model((0,0));state=init@np.linalg.cholesky(P).T;ys=[state@H.T+.1*obs[:,0]]
    for t in range(96):
        state=state@F.T;state[:,[0,2]]+=eps[:,t]*np.sqrt(np.diag(Q)[[0,2]])
        ys.append(state@H.T+.1*obs[:,t+1])
    return np.stack(ys,1)

def reference(y):return kalman_sequence(y[:,:96],*model((0,0)))['next_observation']

def metrics(y,p,r,scalar=False):
    target=y[:,33:97]@M;p=p[:,32:96]@M;r=r[:,32:96]@M
    gamma=np.array([model((0,0))[0][0,0],model((0,0))[0][2,2]])/1.81
    c=(y[:,32:96]@M)*gamma/1.01
    mean=lambda v:np.array([math.fsum(map(float,v[:,:,i].ravel()))/v.shape[0]/v.shape[1] for i in range(2)]) if scalar else np.mean(v,axis=(0,1))
    L=mean((p-target)**2);R=mean((r-target)**2);D=mean((p-r)**2);cross=2*mean((p-r)*(r-target));C=mean((c-target)**2)
    return dict(L=L.tolist(),R=R.tolist(),D=D.tolist(),Cross=cross.tolist(),C=C.tolist(),ratio_reference=(L/R).tolist(),ratio_current=(L/C).tolist(),pass_axes=((L<=.8*C)&(L<=1.25*R)).tolist())

def run():
    start=time.process_time();prerequisite();RUN.mkdir(exist_ok=False);(RUN/'sources').mkdir()
    files=[Path(__file__),PROTOCOL]+[Path(__file__).with_name(x) for x in ['pilot44_fixed.py','pilot25_mechanisms.py','linear_gaussian_filter.py']]
    for p in files:shutil.copyfile(p,RUN/'sources'/p.name)
    save(RUN/'manifest.json',dict(started=now(),protocol='1.3.66/16.66 P1',sources={str(p.relative_to(ROOT)):sha(p) for p in files},p0_acceptance=sha(BASE/'acceptance.json'),numpy=np.__version__,torch=torch.__version__,python=sys.version))
    finals={};curves=[];training=[]
    for root in [9701,9702]:
        train=data(root,1,4096);diag=data(root,2,128)
        ids=np.random.default_rng(np.random.SeedSequence([root,645,1])).choice(4096,128,replace=False)
        indices=np.random.default_rng(np.random.SeedSequence([root,645,0])).integers(0,4096,size=(2000,64))
        sets={'monitor':train[ids],'diagnostic':diag};refs={k:reference(v) for k,v in sets.items()}
        for k,v in [('train',train),('diagnostic',diag),('monitor',train[ids]),('monitor-ids',ids),('indices',indices)]:np.save(RUN/f'{root}-{k}.npy',v)
        for k,v in refs.items():np.save(RUN/f'{root}-{k}-reference.npy',v)
        tensor=torch.from_numpy(train.astype(np.float32));torch.manual_seed(root);initial=weights(Network())
        np.savez(RUN/f'{root}-initial.npz',**initial)
        for kind in ['main','frozen']:
            st=time.process_time();net=Network();net.load_state_dict({k:torch.from_numpy(v) for k,v in initial.items()})
            if kind=='frozen':net.gru.requires_grad_(False)
            pars=[p for p in net.parameters() if p.requires_grad];opt=torch.optim.Adam(pars,lr=.002)
            losses=[];grads=[]
            for step in range(2001):
                if step%250==0:
                    np.savez(RUN/f'{root}-{kind}-s{step}.npz',**weights(net))
                    with torch.no_grad():
                        for name,y in sets.items():
                            pred=net(torch.from_numpy(y[:,:96].astype(np.float32))).numpy()
                            np.save(RUN/f'{root}-{kind}-s{step}-{name}.npy',pred)
                            curves.append(dict(root=root,kind=kind,step=step,set=name,**metrics(y,pred,refs[name])))
                    check(start)
                if step==2000:break
                batch=tensor[indices[step]];opt.zero_grad(set_to_none=True)
                loss=(net(batch[:,:96])[:,32:96]-batch[:,33:97]).square().mean()
                assert torch.isfinite(loss);loss.backward();grad=nn.utils.clip_grad_norm_(pars,1);assert torch.isfinite(grad)
                opt.step();losses.append(float(loss.detach()));grads.append(float(grad))
                if (step+1)%500==0:print(f'{root} {kind} {step+1}/2000',flush=True)
            np.save(RUN/f'{root}-{kind}-loss.npy',losses);np.save(RUN/f'{root}-{kind}-grad.npy',grads)
            training.append(dict(root=root,kind=kind,steps=2000,trainable=sum(p.numel() for p in pars),cpu_seconds=time.process_time()-st))
            finals[root,kind]=net.eval()
    save(RUN/'curves.json',curves);save(RUN/'training.json',training)
    save(RUN/'final-lock.json',dict(locked_at=now(),files={p.name:sha(p) for p in RUN.glob('*-s2000.npz')}))
    rows=[]
    for root in [9701,9702]:
        check(start);y=data(root,3,512);np.save(RUN/f'{root}-confirmation.npy',y);r=reference(y);np.save(RUN/f'{root}-confirmation-reference.npy',r)
        save(RUN/f'{root}-confirmation-event.json',dict(generated_at=now(),lock_sha256=sha(RUN/'final-lock.json')))
        for kind in ['main','frozen']:
            with torch.no_grad():pred=finals[root,kind](torch.from_numpy(y[:,:96].astype(np.float32))).numpy()
            np.save(RUN/f'{root}-{kind}-confirmation.npy',pred);rows.append(dict(root=root,kind=kind,**metrics(y,pred,r)))
    check(start);passed=all(all(v['pass_axes']) for v in rows if v['kind']=='main')
    save(RUN/'result.json',dict(status='LEARNING_GATE_PASS' if passed else 'LEARNING_GATE_FAILED',rows=rows,cpu_seconds=time.process_time()-start,completed_at=now(),audit='PENDING',causal='NOT_RUN'))
    save(RUN/'artifacts.json',{str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    print(read(RUN/'result.json'),flush=True)

def independent_reference(y):
    x=y@M;out=np.zeros((len(y),96,2))
    for i,omega in enumerate([.15,.4]):
        a=1.8*np.cos(omega*np.pi);gamma=a/1.81;F=np.array([[a,-.81],[1.,0.]])
        Q=np.diag([.19*(1.81-a*a/1.81),0.]);P=np.array([[1.,gamma],[gamma,1.]])
        prior=np.zeros((len(y),2))
        for t in range(96):
            K=P[:,0]/(P[0,0]+.01);post=prior+(x[:,t,i]-prior[:,0])[:,None]*K
            out[:,t,i]=post@F[0];prior=post@F.T
            J=np.eye(2)-np.outer(K,[1.,0.]);P=F@(J@P@J.T+.01*np.outer(K,K))@F.T+Q
    return out@M.T

def audit():
    start=time.process_time();prerequisite();assert not (BASE/'p1-audit.json').exists()
    for n,h in read(RUN/'artifacts.json').items():assert sha(RUN/n)==h
    for n,h in read(RUN/'manifest.json')['sources'].items():assert sha(RUN/'sources'/Path(n).name)==h and sha(ROOT/n)==h
    lock=read(RUN/'final-lock.json');assert len(lock['files'])==4
    for n,h in lock['files'].items():assert sha(RUN/n)==h
    curves=read(RUN/'curves.json');result=read(RUN/'result.json');maxdiff=0.;maxref=0.;checked=0
    assert len(curves)==72 and len(result['rows'])==4
    for root in [9701,9702]:
        train=np.load(RUN/f'{root}-train.npy');diag=np.load(RUN/f'{root}-diagnostic.npy');confirm=np.load(RUN/f'{root}-confirmation.npy')
        # Independent scalar AR generator using the identical registered random stream.
        for split,B,y in [(1,4096,train),(2,128,diag),(3,512,confirm)]:
            rng=np.random.default_rng(np.random.SeedSequence([root,644,split]));init=rng.normal(size=(B,4));eps=rng.normal(size=(B,96,2));noise=rng.normal(size=(B,97,2));zs=[]
            for i,omega in enumerate([.15,.4]):
                a=1.8*np.cos(omega*np.pi);gamma=a/1.81;q=.19*(1.81-a*a/1.81)
                z=init[:,2*i];prev=gamma*z+np.sqrt(1-gamma*gamma)*init[:,2*i+1];seq=[z.copy()]
                for t in range(96):prev,z=z,a*z-.81*prev+np.sqrt(q)*eps[:,t,i];seq.append(z.copy())
                zs.append(np.stack(seq,1))
            rebuilt=np.stack(zs,2)@M.T+.1*noise;np.testing.assert_allclose(rebuilt,y,atol=1e-12,rtol=1e-12)
        ids=np.load(RUN/f'{root}-monitor-ids.npy');np.testing.assert_array_equal(ids,np.random.default_rng(np.random.SeedSequence([root,645,1])).choice(4096,128,replace=False))
        np.testing.assert_array_equal(np.load(RUN/f'{root}-indices.npy'),np.random.default_rng(np.random.SeedSequence([root,645,0])).integers(0,4096,size=(2000,64)))
        np.testing.assert_array_equal(train[ids],np.load(RUN/f'{root}-monitor.npy'))
        torch.manual_seed(root);init=weights(Network());saved=dict(np.load(RUN/f'{root}-initial.npz'))
        for k in init:np.testing.assert_array_equal(init[k],saved[k])
        sets={'monitor':train[ids],'diagnostic':diag,'confirmation':confirm}
        for name,y in sets.items():
            ref=np.load(RUN/f'{root}-{name}-reference.npy');alt=independent_reference(y)
            np.testing.assert_allclose(alt,ref,atol=1e-8,rtol=1e-8);maxref=max(maxref,float(abs(alt-ref).max()))
            rows=[r for r in (result['rows'] if name=='confirmation' else curves) if r['root']==root and (name=='confirmation' or r['set']==name)]
            for row in rows:
                kind=row['kind'];step=2000 if name=='confirmation' else row['step'];w=dict(np.load(RUN/f'{root}-{kind}-s{step}.npz'))
                if kind=='frozen' or step==0:
                    for k in w:
                        if kind=='frozen' and not k.startswith('gru.') and step!=0:continue
                        np.testing.assert_array_equal(w[k],init[k])
                suffix='confirmation' if name=='confirmation' else f's{step}-{name}'
                pred=np.load(RUN/f'{root}-{kind}-{suffix}.npy');sub=y if name=='confirmation' else y[:1]
                expected,_=numpy_forward(sub[:,:96].astype(np.float32).astype(float),{k:v.astype(float) for k,v in w.items()})
                obs=pred if name=='confirmation' else pred[:1];np.testing.assert_allclose(expected,obs,atol=2e-5,rtol=2e-5)
                maxdiff=max(maxdiff,float(abs(expected-obs).max()));checked+=1
                vals=metrics(y,pred,ref,scalar=True)
                for k,v in vals.items():np.testing.assert_allclose(v,row[k],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(np.array(vals['L']),np.array(vals['R'])+vals['D']+np.array(vals['Cross']),atol=1e-10,rtol=1e-10)
                check(start)
        event=read(RUN/f'{root}-confirmation-event.json');assert event['generated_at']>=lock['locked_at'] and event['lock_sha256']==sha(RUN/'final-lock.json')
        for kind in ['main','frozen']:
            for stat in ['loss','grad']:
                values=np.load(RUN/f'{root}-{kind}-{stat}.npy');assert values.shape==(2000,) and np.isfinite(values).all() and (values>=0).all()
    for row in read(RUN/'training.json'):assert row['steps']==2000 and row['trainable']==(602 if row['kind']=='main' else 26)
    passed=all(all(x['pass_axes']) for x in result['rows'] if x['kind']=='main')
    assert result['status']==('LEARNING_GATE_PASS' if passed else 'LEARNING_GATE_FAILED')
    check(start);save(BASE/'p1-audit.json',dict(status='AUDIT_PASS',result_sha256=sha(RUN/'result.json'),max_forward_difference=maxdiff,max_reference_difference=maxref,prediction_groups_checked=checked,cpu_seconds=time.process_time()-start,completed_at=now()))
    print(read(BASE/'p1-audit.json'),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:{'run':run,'audit':audit}[sys.argv[1]]()
    except BaseException as exc:
        stamp=time.time_ns();(BASE/f'p1-{sys.argv[1]}-error-{stamp}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        save(BASE/f'p1-{sys.argv[1]}-error-{stamp}.json',dict(status='BUDGET_STOP' if 'BUDGET_STOP' in str(exc) else 'ERROR',error=str(exc),time=now()));raise
