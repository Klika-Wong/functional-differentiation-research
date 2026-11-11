"""Protocol 16.66 P0 only: fixed-mechanism construction and separate audit."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from pilot25_mechanisms import model, M
from linear_gaussian_filter import kalman_sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work/pilot44-fixed-20261110'
RUN = BASE / 'p0-run'
PROTOCOL = ROOT / '02_验证阶段/单人流程/参考/PIM实验设计-v1.md'
S, Z = .001, 1e-7


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def budget(start, limit):
    if time.process_time() - start > limit:
        raise RuntimeError('BUDGET_STOP CPU')
    if sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()) > 150_000_000:
        raise RuntimeError('BUDGET_STOP disk')


def models():
    F, H, Q, V, m, P = model((0, 0))
    mods = []
    for i in range(2):
        ix = slice(2*i, 2*i+2)
        f, q, p = F[ix, ix], Q[ix, ix], P[ix, ix].copy()
        for n in range(10000):
            k = p[:, 0] / (p[0, 0] + .01)
            new = f @ (p - np.outer(k, p[0])) @ f.T + q
            diff = np.max(abs(new-p))
            p = new
            if diff <= 1e-14:
                break
        else:
            raise RuntimeError('Riccati convergence ERROR')
        k = p[:, 0] / (p[0, 0] + .01)
        residual = np.max(abs(f @ (p-np.outer(k,p[0])) @ f.T + q-p))
        assert residual <= 1e-12
        mods.append(dict(F=f, Q=q, P=p, K=k, A=(np.eye(2)-np.outer(k,[1.,0.]))@f,
                         iterations=n+1, residual=float(residual)))
    return mods


def construct(mods):
    w = {'gru.weight_ih_l0': np.zeros((36,2)), 'gru.weight_hh_l0': np.zeros((36,12)),
         'gru.bias_ih_l0': np.zeros(36), 'gru.bias_hh_l0': np.zeros(36),
         'head.weight': np.zeros((2,12)), 'head.bias': np.zeros(2)}
    w['gru.bias_ih_l0'][12:24] = math.log(Z/(1-Z))
    for i, mod in enumerate(mods):
        ix = slice(2*i, 2*i+2)
        w['gru.weight_ih_l0'][24+2*i:26+2*i] = S*np.outer(mod['K'], M[:,i])
        w['gru.weight_hh_l0'][24+2*i:26+2*i, ix] = 2*mod['A']
        w['head.weight'][:,ix] = np.outer(M[:,i], mod['F'][0]/S)
    assert sum(v.size for v in w.values()) == 602
    return w


def numpy_forward(y, w):
    h = np.zeros((len(y),12)); hs=[]
    for t in range(96):
        u = y[:,t]@w['gru.weight_ih_l0'].T+w['gru.bias_ih_l0']
        v = h@w['gru.weight_hh_l0'].T+w['gru.bias_hh_l0']
        r = 1/(1+np.exp(-u[:,:12]-v[:,:12]))
        z = 1/(1+np.exp(-u[:,12:24]-v[:,12:24]))
        h = z*h+(1-z)*np.tanh(u[:,24:]+r*v[:,24:]); hs.append(h.copy())
    hidden = np.stack(hs,1)
    return hidden@w['head.weight'].T+w['head.bias'], hidden


def actual(y, w):
    gru=torch.nn.GRU(2,12,batch_first=True); head=torch.nn.Linear(12,2)
    gru.load_state_dict({k[4:]:torch.from_numpy(v) for k,v in w.items() if k.startswith('gru.')})
    head.load_state_dict({k[5:]:torch.from_numpy(v) for k,v in w.items() if k.startswith('head.')})
    with torch.no_grad():
        hidden=gru(torch.from_numpy(y[:,:96].astype(np.float32)))[0]
        pred=head(hidden)
    return pred.numpy(),hidden.numpy()


def generate(root):
    rng=np.random.default_rng(np.random.SeedSequence([root,644,0]))
    initial=rng.normal(size=(512,4)); innovation=rng.normal(size=(512,96,2)); measurement=rng.normal(size=(512,97,2))
    F,H,Q,R,m,P=model((0,0))
    states=np.empty((512,97,4)); states[:,0]=initial@np.linalg.cholesky(P).T
    for t in range(96):
        states[:,t+1]=states[:,t]@F.T
        states[:,t+1,[0,2]]+=innovation[:,t]*np.sqrt(np.diag(Q)[[0,2]])
    y=states@H.T+.1*measurement
    return dict(initial=initial,innovation=innovation,measurement=measurement,state=states,y=y)


def linear_and_bound(y, mods):
    x=y@M; means=np.zeros((len(y),96,4)); local=np.zeros((len(y),96,2)); bounds=[]
    for i,mod in enumerate(mods):
        prev=np.zeros((len(y),2)); V=np.zeros(2); E=np.zeros(2); rows=[]
        for t in range(96):
            new=prev@mod['A'].T+x[:,t,i,None]*mod['K']
            one=Z*prev+(1-Z)*np.tanh(S*new)/S
            local[:,t,i]=(one-new)@mod['F'][0]
            means[:,t,2*i:2*i+2]=new; prev=new
            newV=abs(mod['A'])@V+8*abs(mod['K'])
            E=(Z*np.eye(2)+(1-Z)*abs(mod['A']))@E+Z*(V+newV)+(1-Z)*S*S*newV**3/3
            V=newV; rows.append(np.r_[V,E,abs(mod['F'][0])@E])
        bounds.append(rows)
    physical=np.stack([means[:,:,2*i:2*i+2]@m['F'][0] for i,m in enumerate(mods)],2)
    return physical@M.T,means,local,np.array(bounds)


def metrics(y, preds, current, local, bounds):
    target=y[:,33:97]@M
    pp=[p[:,32:96]@M for p in preds]
    mse=lambda x:np.mean(x*x,axis=(0,1))
    R=mse(pp[0]-target); C=mse(current[:,32:96]@M-target); L=mse(pp[3]-target)
    D=mse(pp[3]-pp[0]); cross=2*np.mean((pp[3]-pp[0])*(pp[0]-target),axis=(0,1))
    terms=np.array([np.sqrt(mse(pp[j]-pp[j-1])/R) for j in range(1,4)])
    allowed=np.sqrt(np.minimum(.8*C,1.25*R)/R)-1
    bound=np.sqrt(np.mean(bounds[:,32:96,4]**2,axis=1)/R)
    return dict(R=R.tolist(),C=C.tolist(),L=L.tolist(),D=D.tolist(),Cross=cross.tolist(),
                ratio_current=(L/C).tolist(),ratio_reference=(L/R).tolist(),terms=terms.tolist(),
                allowed=allowed.tolist(),sum_bound=terms.sum(0).tolist(),analytic_bound=bound.tolist(),
                direct=np.sqrt(D/R).tolist(),local=np.sqrt(mse(local[:,32:96])/R).tolist(),
                pass_axes=((terms<=np.array([.02,.03,.01])[:,None]).all(0)&(terms.sum(0)<=allowed)&
                           (allowed>0)&(R>0)&(bound<=.03)&(L<=np.minimum(.8*C,1.25*R))).tolist())


def run():
    start=time.process_time(); RUN.mkdir(parents=True,exist_ok=False); (RUN/'sources').mkdir()
    sources=[Path(__file__),PROTOCOL,Path(__file__).with_name('pilot25_mechanisms.py'),Path(__file__).with_name('linear_gaussian_filter.py')]
    for p in sources:shutil.copyfile(p,RUN/'sources'/p.name)
    save(RUN/'manifest.json',dict(protocol='1.3.66/16.66 P0',started=now(),roots=[9701,9702],split=0,
         sources={str(p.relative_to(ROOT)):sha(p) for p in sources},numpy=np.__version__,torch=torch.__version__,python=sys.version))
    mods=models(); w64=construct(mods); w32={k:v.astype(np.float32) for k,v in w64.items()}
    np.savez(RUN/'weights64.npz',**w64);np.savez(RUN/'weights32.npz',**w32)
    save(RUN/'models.json',[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()} for m in mods])
    save(RUN/'lock.json',dict(locked_at=now(),files={p.name:sha(p) for p in [RUN/'weights64.npz',RUN/'weights32.npz',RUN/'models.json',RUN/'manifest.json']},phase='BEFORE_DATA'))
    F,H,Q,V,m,P=model((0,0));gamma=np.array([F[0,0],F[2,2]])/1.81
    np.testing.assert_allclose(F@P@F.T+Q,P,atol=1e-12,rtol=0)
    np.testing.assert_allclose(M.T@M,np.eye(2),atol=1e-12,rtol=0)
    np.testing.assert_allclose(H@P@H.T+V,1.01*np.eye(2),atol=1e-12,rtol=0)
    rows=[]
    for root in [9701,9702]:
        budget(start,120);d=generate(root);np.savez(RUN/f'{root}-data.npz',**d)
        save(RUN/f'{root}-data-event.json',dict(generated_at=now(),lock_sha256=sha(RUN/'lock.json'),split=0))
        y=d['y']; maximum=float(np.max(abs(y@M)))
        if maximum>8:
            save(RUN/'result.json',dict(status='BOUND_DOMAIN_EXCEEDED',root=root,maximum=maximum,training='NOT_RUN'));return
        p0=kalman_sequence(y[:,:96],F,H,Q,V,m,P)['next_observation']
        p1,means,local,bounds=linear_and_bound(y,mods)
        p2,h64=numpy_forward(y[:,:96],w64);p3,h32=actual(y,w32)
        assert np.count_nonzero(h32[:,:,4:])==0 and np.count_nonzero(h64[:,:,4:])==0
        current=((y@M)*gamma/1.01)@M.T
        np.savez(RUN/f'{root}-evaluation.npz',P0=p0,P1=p1,P2=p2,P3=p3,h64=h64,h32=h32,means=means,local=local,bounds=bounds,current=current)
        rows.append(dict(root=root,max_input=maximum,**metrics(y,[p0,p1,p2,p3],current,local,bounds)))
        budget(start,120)
    status='CONSTRUCTION_GATE_PASS' if all(all(r['pass_axes']) for r in rows) else 'CONSTRUCTION_GATE_FAILED'
    save(RUN/'result.json',dict(status=status,rows=rows,current_population_mse=(1.01-gamma**2/1.01).tolist(),
         tail_union_bound=194*math.erfc(8/math.sqrt(2*1.01)),cpu_seconds=time.process_time()-start,
         training='NOT_RUN',audit='PENDING',completed_at=now()))
    save(RUN/'artifacts.json',{str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    budget(start,120);print(json.dumps(read(RUN/'result.json'),ensure_ascii=False))


def audit():
    start=time.process_time()
    assert not (BASE/'p0-audit.json').exists()
    for n,h in read(RUN/'artifacts.json').items():assert sha(RUN/n)==h,n
    for n,h in read(RUN/'manifest.json')['sources'].items():
        assert sha(ROOT/n)==h and sha(RUN/'sources'/Path(n).name)==h
    lock=read(RUN/'lock.json')
    for n,h in lock['files'].items():assert sha(RUN/n)==h
    # Rebuild the dynamics from scalar AR definitions, without model()/models().
    rot=np.array([[1.,-1.],[1.,1.]])/math.sqrt(2)
    aa=1.8*np.cos(np.pi*np.array([.15,.40]));gam=aa/1.81;qq=.19*(1.81-aa*aa/1.81)
    ms=read(RUN/'models.json'); w64=dict(np.load(RUN/'weights64.npz'));w32=dict(np.load(RUN/'weights32.npz'))
    expected={k:np.zeros_like(v) for k,v in w64.items()};expected['gru.bias_ih_l0'][12:24]=np.log(1e-7/(1-1e-7))
    fs=[];ks=[];As=[];pinit=[]
    for i,a in enumerate(aa):
        f=np.array([[a,-.81],[1.,0.]]);q=np.diag([qq[i],0.]);p=np.array([[1.,gam[i]],[gam[i],1.]]);pinit.append(p.copy());fs.append(f)
        # Joseph update, independent of the subtractive solver used for construction.
        for it in range(10000):
            gain=p[:,0]/(p[0,0]+.01);j=np.eye(2)-np.outer(gain,[1.,0.])
            nxt=f@(j@p@j.T+.01*np.outer(gain,gain))@f.T+q
            change=np.max(abs(nxt-p));p=nxt
            if change<=1e-14:break
        assert it<9999
        k=p[:,0]/(p[0,0]+.01);A=f-np.outer(k,f[0]);ks.append(k);As.append(A)
        for key,value in [('F',f),('Q',q),('P',p),('K',k),('A',A)]:np.testing.assert_allclose(ms[i][key],value,atol=1e-12,rtol=1e-12)
        for u in range(2):
            for v in range(2):
                expected['gru.weight_ih_l0'][24+2*i+u,v]=.001*k[u]*rot[v,i]
                expected['gru.weight_hh_l0'][24+2*i+u,2*i+v]=2*A[u,v]
                expected['head.weight'][v,2*i+u]=rot[v,i]*f[0,u]/.001
    for key in expected:
        np.testing.assert_allclose(w64[key],expected[key],atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(w32[key],w64[key].astype(np.float32))
    result=read(RUN/'result.json');checks=[];max_forward=0.;max_reference=0.;max_roundoff=0.
    for row in result['rows']:
        root=row['root'];d=dict(np.load(RUN/f'{root}-data.npz'));e=dict(np.load(RUN/f'{root}-evaluation.npz'))
        event=read(RUN/f'{root}-data-event.json');assert event['generated_at']>=lock['locked_at'] and event['lock_sha256']==sha(RUN/'lock.json')
        rng=np.random.default_rng(np.random.SeedSequence([root,644,0]))
        for key,shape in [('initial',(512,4)),('innovation',(512,96,2)),('measurement',(512,97,2))]:np.testing.assert_array_equal(d[key],rng.normal(size=shape))
        state=np.empty((512,97,4))
        for i in range(2):
            state[:,0,2*i]=d['initial'][:,2*i]
            state[:,0,2*i+1]=gam[i]*d['initial'][:,2*i]+np.sqrt(1-gam[i]**2)*d['initial'][:,2*i+1]
            for t in range(96):
                state[:,t+1,2*i]=aa[i]*state[:,t,2*i]-.81*state[:,t,2*i+1]+np.sqrt(qq[i])*d['innovation'][:,t,i]
                state[:,t+1,2*i+1]=state[:,t,2*i]
        y=state[:,:,[0,2]]@rot.T+.1*d['measurement']
        np.testing.assert_allclose(state,d['state'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(y,d['y'],atol=1e-12,rtol=1e-12)
        y=d['y'];x=y@rot
        # Scalar observation, separate sources; innovation-form means and Joseph covariance.
        ref=np.zeros((512,96,2));linear=np.zeros_like(ref);nonlinear=np.zeros(ref.shape,dtype=np.longdouble)
        local=np.zeros_like(ref);bounds=np.zeros((2,96,5));all_means=np.zeros((512,96,4))
        for i in range(2):
            f=fs[i];prior=np.zeros((512,2));p=pinit[i].copy();m=np.zeros_like(prior)
            g=np.zeros(prior.shape,dtype=np.longdouble);A=As[i];K=ks[i]
            Vseq=[];dseq=[]
            for t in range(96):
                gain=p[:,0]/(p[0,0]+.01);post=prior+(x[:,t,i]-prior[:,0])[:,None]*gain
                ref[:,t,i]=post@f[0];prior=post@f.T
                J=np.eye(2)-np.outer(gain,[1.,0.]);p=f@(J@p@J.T+.01*np.outer(gain,gain))@f.T+np.diag([qq[i],0.])
                old=m.copy();prior_m=m@f.T;m=prior_m+(x[:,t,i]-prior_m[:,0])[:,None]*K
                all_means[:,t,2*i:2*i+2]=m;linear[:,t,i]=m@f[0]
                local[:,t,i]=((1e-7*old+(1-1e-7)*np.tanh(.001*m)/.001)-m)@f[0]
                g= np.longdouble(1e-7)*g+(1-np.longdouble(1e-7))*np.tanh(np.longdouble(.001)*(g@A.astype(np.longdouble).T+x[:,t,i,None].astype(np.longdouble)*K)) / np.longdouble(.001)
                nonlinear[:,t,i]=g@f[0].astype(np.longdouble)
                V=sum((np.linalg.matrix_power(abs(A),j)@(8*abs(K)) for j in range(t+1)),start=np.zeros(2))
                oldV=Vseq[-1] if Vseq else np.zeros(2);Vseq.append(V)
                dseq.append(1e-7*(oldV+V)+(1-1e-7)*.001**2*V**3/3)
                B=1e-7*np.eye(2)+(1-1e-7)*abs(A)
                bound=sum((np.linalg.matrix_power(B,t-j)@dseq[j] for j in range(t+1)),start=np.zeros(2))
                bounds[i,t]=np.r_[V,bound,abs(f[0])@bound]
                assert np.all(abs(g-m)<=bound+1e-12)
        np.testing.assert_allclose(ref@rot.T,e['P0'],atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(linear@rot.T,e['P1'],atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(all_means,e['means'],atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(local,e['local'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(bounds,e['bounds'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(nonlinear@rot.T,e['P2'],atol=1e-8,rtol=1e-8)
        max_reference=max(max_reference,float(np.max(abs(ref@rot.T-e['P0']))))
        max_roundoff=max(max_roundoff,float(np.max(abs(nonlinear@rot.T-e['P2']))))
        # Full, quantized 12-slot network via separate gate products and einsum.
        h=np.zeros((512,12));hist=[];ww={k:v.astype(float) for k,v in w32.items()};yy=y.astype(np.float32).astype(float)
        for t in range(96):
            gates=[]
            for gate in range(3):
                ix=slice(gate*12,(gate+1)*12)
                inp=np.einsum('bi,ji->bj',yy[:,t],ww['gru.weight_ih_l0'][ix])+ww['gru.bias_ih_l0'][ix]
                rec=np.einsum('bi,ji->bj',h,ww['gru.weight_hh_l0'][ix])+ww['gru.bias_hh_l0'][ix]
                gates.append((inp,rec))
            reset=1/(1+np.exp(-(gates[0][0]+gates[0][1])))
            update=1/(1+np.exp(-(gates[1][0]+gates[1][1])))
            candidate=np.tanh(gates[2][0]+reset*gates[2][1]);h=(1-update)*candidate+update*h;hist.append(h.copy())
        nh=np.stack(hist,1);pred=nh@ww['head.weight'].T+ww['head.bias']
        np.testing.assert_allclose(nh,e['h32'],atol=2e-5,rtol=2e-5);np.testing.assert_allclose(pred,e['P3'],atol=2e-5,rtol=2e-5)
        assert not np.any(e['h32'][:,:,4:]) and not np.any(e['h64'][:,:,4:])
        max_forward=max(max_forward,float(np.max(abs(pred-e['P3']))))
        # Independent scalar reductions of saved predictions; no metrics() call.
        def avg(v):return math.fsum(map(float,v.ravel()))/v.size
        target=y[:,33:97]@rot;pp=[e[f'P{j}'][:,32:96]@rot for j in range(4)]
        c=(x*gam/1.01)@rot.T;np.testing.assert_allclose(c,e['current'],atol=1e-12,rtol=1e-12)
        axes=[]
        for i in range(2):
            R=avg((pp[0][:,:,i]-target[:,:,i])**2);C=avg((c[:,32:96]@rot-target)[:,:,i]**2)
            L=avg((pp[3][:,:,i]-target[:,:,i])**2);D=avg((pp[3][:,:,i]-pp[0][:,:,i])**2)
            cross=2*avg((pp[3][:,:,i]-pp[0][:,:,i])*(pp[0][:,:,i]-target[:,:,i]))
            terms=[math.sqrt(avg((pp[j][:,:,i]-pp[j-1][:,:,i])**2)/R) for j in range(1,4)]
            bound=math.sqrt(avg(bounds[i,32:96,4]**2)/R);allow=math.sqrt(min(.8*C,1.25*R)/R)-1
            vals=dict(R=R,C=C,L=L,D=D,Cross=cross,ratio_current=L/C,ratio_reference=L/R,
                      allowed=allow,sum_bound=sum(terms),analytic_bound=bound,direct=math.sqrt(D/R),local=math.sqrt(avg(local[:,32:96,i]**2)/R))
            for k,v in vals.items():np.testing.assert_allclose(v,row[k][i],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(terms,np.array(row['terms'])[:,i],atol=1e-10,rtol=1e-10)
            assert abs(L-R-D-cross)<1e-10
            passed=bool(all(v<=lim for v,lim in zip(terms,[.02,.03,.01])) and sum(terms)<=allow and allow>0 and R>0 and bound<=.03 and L<=min(.8*C,1.25*R))
            assert passed==row['pass_axes'][i];axes.append(passed)
        assert np.max(abs(x))<=8
        np.testing.assert_allclose(np.max(abs(x)),row['max_input'],atol=1e-12,rtol=1e-12)
        checks.extend(axes);budget(start,180)
    assert result['status']==('CONSTRUCTION_GATE_PASS' if all(checks) else 'CONSTRUCTION_GATE_FAILED')
    np.testing.assert_allclose(result['current_population_mse'],1.01-gam**2/1.01,atol=1e-12,rtol=1e-12)
    assert result['training']=='NOT_RUN'
    out=dict(status='AUDIT_PASS',result_sha256=sha(RUN/'result.json'),scientific_status=result['status'],
             max_forward_difference=max_forward,max_reference_difference=max_reference,
             float64_vs_longdouble_proxy=max_roundoff,longdouble_mantissa=np.finfo(np.longdouble).nmant,
             cpu_seconds=time.process_time()-start,completed_at=now(),independence='separate process and numerical paths, same author')
    save(BASE/'p0-audit.json',out);print(json.dumps(out))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    BASE.mkdir(exist_ok=True)
    try:
        {'run':run,'audit':audit}[sys.argv[1]]()
    except BaseException as exc:
        stamp=time.time_ns();(BASE/f'{sys.argv[1]}-error-{stamp}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        save(BASE/f'{sys.argv[1]}-error-{stamp}.json',dict(status='BUDGET_STOP' if 'BUDGET_STOP' in str(exc) else 'ERROR',error=str(exc),time=now()))
        raise
