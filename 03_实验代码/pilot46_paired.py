"""16.68: bounded paired-shrink calibration; no learned-network confirmation."""
from pathlib import Path
import sys,time,traceback,shutil,math
import numpy as np
import torch
from pilot44_fixed import ROOT,PROTOCOL,read,save,now,actual
from pilot44_learning import Network,data,reference,metrics
from pilot45_fixed_joint import digest,advance,SUBSETS,TAUS,windows,normal_window
from pilot45_joint_audit import forward,regenerate,reference_path,average2,errors,metric_check
from pilot25_mechanisms import M

BASE=ROOT/'work/pilot46-paired-calibration-20261115'
RUN=BASE/'run'
UP=ROOT/'work/pilot44-fixed-20261110'
OLD=ROOT/'work/pilot45-fixed-joint-20261114'
S=[np.arange(6),np.arange(6,12)]


def budget(start,reserve=0):
    if time.process_time()-start>600:raise RuntimeError('BUDGET_STOP CPU')
    if sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())+reserve>300_000_000:raise RuntimeError('BUDGET_STOP disk')


def prereq():
    a,b=read(UP/'p0-audit.json'),read(UP/'acceptance.json')
    assert a['status']=='AUDIT_PASS' and b['status']=='ACCEPTED_P0_ONLY'
    assert a['result_sha256']==b['result_sha256']==digest(UP/'p0-run/result.json')
    assert b['audit_sha256']==digest(UP/'p0-audit.json')
    for name,sha in read(UP/'p0-run/artifacts.json').items():assert digest(UP/'p0-run'/name)==sha
    locked=read(OLD/'discovery-lock.json')
    assert locked['candidates_sha256']==digest(OLD/'discovery/candidates.json')
    return locked['candidates']['random_indices']


def construction():
    original=dict(np.load(UP/'p0-run/weights32.npz'))
    ids=np.array([0,1,0,1,0,1,2,3,2,3,2,3]);E=np.eye(4)[ids];P=E.T/3
    w={k:v.copy() for k,v in original.items()}
    for g in range(3):
        sl=slice(g*12,(g+1)*12)
        w['gru.weight_ih_l0'][sl]=(E@original['gru.weight_ih_l0'][g*12:g*12+4]).astype('float32')
        w['gru.weight_hh_l0'][sl]=(E@original['gru.weight_hh_l0'][g*12:g*12+4,:4]@P).astype('float32')
        for key in ['gru.bias_ih_l0','gru.bias_hh_l0']:
            w[key][sl]=(E@original[key][g*12:g*12+4]).astype('float32')
    w['head.weight']=(original['head.weight'][:,:4]@P).astype('float32')
    return original,w


def pair(h,subset,other):
    d=np.linalg.norm(h[...,subset].astype(float),axis=-1);b=np.linalg.norm(h[...,other].astype(float),axis=-1)
    at=np.zeros_like(d);ar=np.zeros_like(b);both=(d>0)&(b>0)
    left=both&(d<=b);right=both&~left
    at[left]=1;ar[left]=d[left]/b[left];at[right]=b[right]/d[right];ar[right]=1
    ht=h.copy();hr=h.copy()
    ht[...,subset]*=(1-at).astype('float32')[...,None]
    hr[...,other]*=(1-ar).astype('float32')[...,None]
    nt=np.linalg.norm(ht.astype(float)-h,axis=-1);nr=np.linalg.norm(hr.astype(float)-h,axis=-1)
    rho=np.minimum(d,b);valid=np.isfinite(nt)&np.isfinite(nr)&(abs(nt-rho)<=1e-6+1e-5*rho)&(abs(nr-rho)<=1e-6+1e-5*rho)&(abs(nt-nr)<=1e-6+1e-5*rho)
    for changed,indices in [(ht,subset),(hr,other)]:
        np.testing.assert_array_equal(changed[...,np.setdiff1d(np.arange(12),indices)],h[...,np.setdiff1d(np.arange(12),indices)])
        valid&=((changed>=np.minimum(h,0))&(changed<=np.maximum(h,0))).all(-1)
    return ht,hr,np.stack([d,b,at,ar,nt,nr],-1),valid


def plan(randoms):
    return [dict(type='target',axis=a,indices=S[a].tolist()) for a in range(2)]+[dict(type='same',random=j,indices=SUBSETS[idx].tolist()) for j,idx in enumerate(randoms)]+[dict(type='paired',axis=a,random=j,side=side,indices=SUBSETS[idx].tolist()) for a in range(2) for j,idx in enumerate(randoms) for side in ('target','random')]


def verdict(rows):
    geom=all(r['geometry'] for r in rows);pos=[r for r in rows if r['kind']=='positive'];null=[r for r in rows if r['kind']=='null']
    normal=all(all(r['normal']['pass_axes']) for r in pos)
    info=all(all(r['dose_pass']) for r in pos)
    positive=all(all(r['positive_pass']) for r in pos)
    blank=all(r['null_max']<=1e-10 for r in null)
    return ('PAIR_GEOMETRY_INVALID' if not geom else 'CALIBRATION_BASELINE_FAILED' if not normal else 'DOSE_INSUFFICIENT' if not info else 'POSITIVE_CONTROL_FAILED' if not positive else 'NULL_CONTROL_FAILED' if not blank else 'PAIRED_CALIBRATION_PASS')


def run(start):
    randoms=prereq();RUN.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),PROTOCOL,*[Path(__file__).with_name(n) for n in ['pilot44_fixed.py','pilot44_learning.py','pilot45_fixed_joint.py','pilot45_joint_audit.py','pilot25_mechanisms.py','linear_gaussian_filter.py']]]
    (RUN/'sources').mkdir()
    for p in sources:shutil.copyfile(p,RUN/'sources'/p.name)
    save(RUN/'manifest.json',dict(started_at=now(),protocol='1.3.68/16.68',sources={str(p.relative_to(ROOT)):digest(p) for p in sources},upstream_acceptance=digest(UP/'acceptance.json'),p0_weights=digest(UP/'p0-run/weights32.npz'),old_lock=digest(OLD/'discovery-lock.json'),randoms=randoms,numpy=np.__version__,torch=torch.__version__))
    original,w=construction();np.savez(RUN/'positive.npz',**w)
    null={k:v.copy() for k,v in w.items()};null['head.weight'][:]=0;null['head.bias'][:]=0;np.savez(RUN/'null.npz',**null)
    save(RUN/'lock.json',dict(locked_at=now(),weights={k:digest(RUN/f'{k}.npz') for k in ['positive','null']},targets=[s.tolist() for s in S],randoms=randoms))
    rows=[]
    for root in (9701,9702):
        y=data(root,6,128);np.save(RUN/f'{root}-data.npy',y)
        save(RUN/f'{root}-event.json',dict(generated_at=now(),lock_sha256=digest(RUN/'lock.json'),split=6))
        ref=reference(y);np.save(RUN/f'{root}-reference.npy',ref)
        target=windows(y);gamma=1.8*np.cos(np.pi*np.array([.15,.4]))/1.81
        current=((y[:,:96]@M)*gamma/1.01)@M.T;C=(((normal_window(current)-target)@M)**2).mean((0,1,2))
        p0,_=actual(y,original);np.save(RUN/f'{root}-p0-normal.npy',p0)
        for kind,weights in [('positive',w),('null',null)]:
            tag=f'{root}-{kind}';net=Network().eval();net.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
            for p in net.parameters():p.requires_grad_(False)
            hfull=net.gru(torch.from_numpy(y[:,:96].astype('float32')))[0];p=net.head(hfull).numpy();h=hfull.numpy()[:,TAUS].copy()
            np.save(RUN/f'{tag}-hidden.npy',hfull.numpy());np.save(RUN/f'{tag}-normal.npy',p)
            if kind=='positive':np.testing.assert_allclose(p,p0,atol=2e-5,rtol=2e-5)
            future=torch.from_numpy(np.stack([y[:,t+1:t+5] for t in TAUS],1).astype('float32').reshape(-1,4,2))
            np.testing.assert_allclose(advance(net,future,h),normal_window(p),atol=2e-5,rtol=2e-5)
            L0=(((normal_window(p)-target)@M)**2).mean((0,1,2))
            specs=plan(randoms[str(root)]);save(RUN/f'{tag}-specs.json',specs)
            shape=(642,128,8,4,2);budget(start,int(np.prod(shape))*4)
            outputs=np.lib.format.open_memmap(RUN/f'{tag}-outputs.npy',mode='w+',dtype='float32',shape=shape)
            L=np.empty((642,2));episodes=np.empty((642,128,2));temporal=np.empty((642,8,4,2));details=np.empty((2,128,128,8,6));valid=np.empty((2,128,128,8),bool)
            pair_cache=None
            for i,spec in enumerate(specs):
                if spec['type']!='paired':
                    changed=h.copy();changed[...,spec['indices']]=0
                else:
                    a,j=spec['axis'],spec['random']
                    if spec['side']=='target':
                        ht,hr,detail,ok=pair(h,S[a],np.array(spec['indices']));pair_cache=hr;changed=ht;details[a,j]=detail;valid[a,j]=ok
                    else:changed=pair_cache
                out=advance(net,future,changed);outputs[i]=out
                err=((out.astype(float)-target)@M)**2
                L[i]=err.mean((0,1,2));episodes[i]=err.mean((1,2));temporal[i]=err.mean(0)
                if i%32==0:budget(start)
            outputs.flush();delta=(L-L0)/C;J=delta-delta[:,::-1]
            D=np.stack([J[130+a*256:130+(a+1)*256:2,a]-J[131+a*256:130+(a+1)*256:2,a] for a in range(2)])
            np.savez(RUN/f'{tag}-metrics.npz',L=L,L0=L0,C=C,delta=delta,J=J,D=D,episode_L=episodes,temporal_L=temporal)
            np.save(RUN/f'{tag}-pair.npy',details);np.save(RUN/f'{tag}-valid.npy',valid)
            qs=np.quantile(J[2:130],.95,axis=0,method='linear');qd=np.quantile(D,.05,axis=1,method='linear')
            rho=np.minimum(details[...,0],details[...,1]);ratio=np.divide(rho,details[...,0],out=np.zeros_like(rho),where=details[...,0]>0)
            qdose=np.quantile(ratio.mean((2,3)),.05,axis=1,method='linear');nonzero=(rho>0).mean((1,2,3))
            passed=[bool(delta[a,a]>=.05 and J[a,a]>=.05 and delta[a,1-a]<=.5*delta[a,a] and J[a,a]>=qs[a]+.02 and qd[a]>=.02) for a in range(2)]
            row=dict(root=root,kind=kind,geometry=bool(valid.all()),normal=metrics(y,p,ref),selected_J=[float(J[a,a]) for a in range(2)],same_q95=qs.tolist(),paired_q05=qd.tolist(),dose_q05=qdose.tolist(),nonzero_fraction=nonzero.tolist(),dose_pass=((qdose>=.25)&(nonzero>=.95)).tolist(),positive_pass=passed,null_max=float(max(abs(delta).max(),abs(J).max(),abs(D).max())))
            rows.append(row);print(tag+' '+str({k:row[k] for k in ['paired_q05','positive_pass','dose_pass','geometry']}),flush=True)
    budget(start);save(RUN/'result.json',dict(status=verdict(rows),rows=rows,interventions=2568,cpu_seconds=time.process_time()-start,completed_at=now(),audit='PENDING',real_confirmation='NOT_RUN',extraction='NOT_RUN'))
    save(RUN/'artifacts.json',{str(p.relative_to(RUN)):digest(p) for p in RUN.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    budget(start);print(read(RUN/'result.json')['status'],flush=True)


def audit(start):
    randoms=prereq();assert not (BASE/'audit.json').exists()
    for name,sha in read(RUN/'artifacts.json').items():assert digest(RUN/name)==sha
    for name,sha in read(RUN/'manifest.json')['sources'].items():assert digest(ROOT/name)==sha and digest(RUN/'sources'/Path(name).name)==sha
    original=dict(np.load(UP/'p0-run/weights32.npz'));positive=dict(np.load(RUN/'positive.npz'));blank=dict(np.load(RUN/'null.npz'))
    ids=[0,1,0,1,0,1,2,3,2,3,2,3]
    for g in range(3):
        for i in range(12):
            np.testing.assert_array_equal(positive['gru.weight_ih_l0'][g*12+i],original['gru.weight_ih_l0'][g*12+ids[i]])
            for j in range(12):
                expected=np.float32(float(original['gru.weight_hh_l0'][g*12+ids[i],ids[j]])/3)
                assert positive['gru.weight_hh_l0'][g*12+i,j]==expected
            for key in ['gru.bias_ih_l0','gru.bias_hh_l0']:assert positive[key][g*12+i]==original[key][g*12+ids[i]]
    for j in range(12):np.testing.assert_array_equal(positive['head.weight'][:,j],(original['head.weight'][:,ids[j]].astype(float)/3).astype('float32'))
    for k in positive:
        if k.startswith('head.'):assert not blank[k].any()
        else:np.testing.assert_array_equal(positive[k],blank[k])
    result=read(RUN/'result.json');maxdiff=0.;maxref=0.;count=0;all_geom=True;normal_pass=[];dose_pass=[];positive_pass=[];null_max=[]
    for root in (9701,9702):
        y=np.load(RUN/f'{root}-data.npy');np.testing.assert_allclose(y,regenerate(root,6,128),atol=1e-12,rtol=1e-12)
        lock=read(RUN/'lock.json');event=read(RUN/f'{root}-event.json');assert event['generated_at']>lock['locked_at'] and event['lock_sha256']==digest(RUN/'lock.json')
        ref=np.load(RUN/f'{root}-reference.npy');rr=reference_path(y);np.testing.assert_allclose(ref,rr,atol=1e-8,rtol=1e-8);maxref=max(maxref,float(abs(ref-rr).max()))
        for kind,w in [('positive',positive),('null',blank)]:
            tag=f'{root}-{kind}';row=next(x for x in result['rows'] if x['root']==root and x['kind']==kind)
            hfull=np.load(RUN/f'{tag}-hidden.npy');p=np.load(RUN/f'{tag}-normal.npy');ap,ah=forward(y[:,:96].astype('float32'),w)
            for x,z in [(ap,p),(ah,hfull)]:np.testing.assert_allclose(x,z,atol=2e-5,rtol=2e-5);maxdiff=max(maxdiff,float(abs(x-z).max()))
            if kind=='positive':
                op,_=forward(y[:,:96].astype('float32'),original);np.testing.assert_allclose(op,p,atol=2e-5,rtol=2e-5)
            normal=metric_check(y,p,ref,row['normal']);h=hfull[:,TAUS];target=np.stack([y[:,t+2:t+6] for t in range(32,89,8)],1)
            future=np.stack([y[:,t+1:t+5] for t in range(32,89,8)],1).astype('float32').reshape(-1,4,2)
            gamma=np.array([1.8*math.cos(math.pi*f)/1.81 for f in [.15,.4]])
            current=((y[:,:96]@M)*gamma/1.01)@M.T;C=average2(errors(normal_window(current),target));L0=average2(errors(normal_window(p),target))
            stats=dict(np.load(RUN/f'{tag}-metrics.npz'));out=np.load(RUN/f'{tag}-outputs.npy',mmap_mode='r');detail=np.load(RUN/f'{tag}-pair.npy');valid=np.load(RUN/f'{tag}-valid.npy')
            np.testing.assert_allclose(C,stats['C'],atol=1e-10,rtol=1e-10);np.testing.assert_allclose(L0,stats['L0'],atol=1e-10,rtol=1e-10)
            js=[];ds=[]
            for i in range(642):
                changed=h.copy()
                if i<2:changed[...,list(range(i*6,i*6+6))]=0
                elif i<130:changed[...,SUBSETS[randoms[str(root)][i-2]]]=0
                else:
                    a=(i-130)//256;j=((i-130)%256)//2;side=(i-130)%2;ss=np.arange(a*6,a*6+6);rrs=SUBSETS[randoms[str(root)][j]]
                    dn=np.sqrt(np.sum(h[...,ss].astype(float)**2,-1));bn=np.sqrt(np.sum(h[...,rrs].astype(float)**2,-1));rho=np.minimum(dn,bn)
                    at=np.divide(rho,dn,out=np.zeros_like(rho),where=dn>0);ar=np.divide(rho,bn,out=np.zeros_like(rho),where=bn>0)
                    for coords,coeff in [(ss,at)] if side==0 else [(rrs,ar)]:
                        for c in coords:changed[...,c]=h[...,c]*np.float32(1-coeff)
                    np.testing.assert_allclose(np.stack([dn,bn,at,ar],-1),detail[a,j,...,:4],atol=1e-10,rtol=1e-10)
                    norm=np.sqrt(np.sum((changed.astype(float)-h)**2,-1));np.testing.assert_allclose(norm,detail[a,j,...,4+side],atol=1e-10,rtol=1e-10)
                    assert ((changed>=np.minimum(h,0))&(changed<=np.maximum(h,0))).all()
                    assert ((at>=0)&(at<=1)&(ar>=0)&(ar<=1)).all()
                    ok=(abs(norm-rho)<=1e-6+1e-5*rho);np.testing.assert_array_equal(ok,valid[a,j]);all_geom&=bool(ok.all())
                alt,_=forward(future,w,changed.reshape(-1,12));alt=alt.reshape(128,8,4,2)
                np.testing.assert_allclose(alt,out[i],atol=2e-5,rtol=2e-5);maxdiff=max(maxdiff,float(abs(alt-out[i]).max()))
                e=errors(out[i],target);L=average2(e);delta=(L-L0)/C;J=delta-delta[::-1];js.append(J);ds.append(delta)
                for key,value in [('L',L),('delta',delta),('J',J),('episode_L',e.mean((1,2))),('temporal_L',e.mean(0))]:np.testing.assert_allclose(value,stats[key][i],atol=1e-10,rtol=1e-10)
                count+=1
                if i%32==0:budget(start)
            # Independent linear quantiles on the independently recomputed scalar scores.
            def q(values,percent):
                v=sorted(map(float,values));f=(len(v)-1)*percent;lo=int(f);return v[lo]+(f-lo)*(v[min(lo+1,len(v)-1)]-v[lo])
            J=np.array(js);deltas=np.array(ds);D=np.array([[J[130+a*256+2*j,a]-J[131+a*256+2*j,a] for j in range(128)] for a in range(2)])
            np.testing.assert_allclose(D,stats['D'],atol=1e-10,rtol=1e-10)
            qs=[q(J[2:130,a],.95) for a in range(2)];qd=[q(D[a],.05) for a in range(2)]
            rho=np.minimum(detail[...,0],detail[...,1]);ratio=np.divide(rho,detail[...,0],out=np.zeros_like(rho),where=detail[...,0]>0)
            qdose=[q(ratio[a].mean((1,2)),.05) for a in range(2)];nz=(rho>0).mean((1,2,3))
            inf=[qdose[a]>=.25 and nz[a]>=.95 for a in range(2)]
            good=[deltas[a,a]>=.05 and J[a,a]>=.05 and deltas[a,1-a]<=.5*deltas[a,a] and J[a,a]>=qs[a]+.02 and qd[a]>=.02 for a in range(2)]
            for key,value in [('same_q95',qs),('paired_q05',qd),('dose_q05',qdose),('nonzero_fraction',nz),('dose_pass',inf),('positive_pass',good)]:np.testing.assert_allclose(value,row[key],atol=1e-10,rtol=1e-10)
            maximum=float(max(abs(deltas).max(),abs(J).max(),abs(D).max()));np.testing.assert_allclose(maximum,row['null_max'],atol=1e-10,rtol=1e-10)
            if kind=='positive':normal_pass.extend(normal);dose_pass.extend(inf);positive_pass.extend(good)
            else:null_max.append(maximum)
            print('audit '+tag,flush=True)
    expected=('PAIR_GEOMETRY_INVALID' if not all_geom else 'CALIBRATION_BASELINE_FAILED' if not all(normal_pass) else 'DOSE_INSUFFICIENT' if not all(dose_pass) else 'POSITIVE_CONTROL_FAILED' if not all(positive_pass) else 'NULL_CONTROL_FAILED' if max(null_max)>1e-10 else 'PAIRED_CALIBRATION_PASS')
    assert expected==result['status'] and count==2568
    budget(start);save(BASE/'audit.json',dict(status='AUDIT_PASS',scientific_status=expected,result_sha256=digest(RUN/'result.json'),artifacts_sha256=digest(RUN/'artifacts.json'),interventions=count,max_forward_difference=maxdiff,max_reference_difference=maxref,cpu_seconds=time.process_time()-start,completed_at=now()))
    print(read(BASE/'audit.json'),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.process_time()
    try:
        with torch.no_grad():{'run':run,'audit':audit}[sys.argv[1]](start)
    except BaseException as e:
        BASE.mkdir(parents=True,exist_ok=True);name=f'{sys.argv[1]}-error-{time.time_ns()}'
        (BASE/(name+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');save(BASE/(name+'.json'),dict(status='BUDGET_STOP' if 'BUDGET_STOP' in str(e) else 'ERROR',at=now(),error=str(e)));raise
