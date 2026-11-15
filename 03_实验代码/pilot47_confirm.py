"""16.69: fresh confirmation of locked learned networks after accepted calibration."""
from pathlib import Path
import sys,time,traceback,shutil,math
import numpy as np
import torch
from pilot44_fixed import ROOT,PROTOCOL,read,save,now
from pilot44_learning import Network,data,reference,metrics
from pilot45_fixed_joint import digest,advance,SUBSETS,TAUS,windows,normal_window,weight_name,prerequisite
from pilot45_joint_audit import forward,regenerate,reference_path,average2,errors,metric_check
from pilot46_paired import pair
from pilot25_mechanisms import M

BASE=ROOT/'work/pilot47-paired-confirmation-20261115';RUN=BASE/'run'
CAL=ROOT/'work/pilot46-paired-calibration-20261115';OLD=ROOT/'work/pilot45-fixed-joint-20261114';UP=ROOT/'work/pilot44-fixed-20261110/p1-run'
KINDS=['main','frozen','initial']


def budget(start,reserve=0):
    if time.process_time()-start>600:raise RuntimeError('BUDGET_STOP CPU')
    if sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())+reserve>1_000_000_000:raise RuntimeError('BUDGET_STOP disk')


def prerequisite_all():
    prerequisite()
    accepted=read(CAL/'acceptance.json');a=read(CAL/'audit.json')
    assert accepted['status']=='ACCEPTED_PAIRED_CALIBRATION_PASS' and a['scientific_status']=='PAIRED_CALIBRATION_PASS' and a['status']=='AUDIT_PASS'
    assert accepted['result_sha256']==a['result_sha256']==digest(CAL/'run/result.json') and accepted['audit_sha256']==digest(CAL/'audit.json')
    assert a['artifacts_sha256']==digest(CAL/'run/artifacts.json')
    for folder in [CAL/'run',OLD/'discovery']:
        for name,sha in read(folder/'artifacts.json').items():assert digest(folder/name)==sha
        for name,sha in read(folder/'manifest.json')['sources'].items():
            assert digest(folder/'sources'/Path(name).name)==sha
            if Path(name).suffix=='.py':assert digest(ROOT/name)==sha
    lock=read(OLD/'discovery-lock.json');da=read(OLD/'discovery-audit.json')
    assert lock['status']=='ACCEPTED_DISCOVERY_LOCKED' and da['status']=='AUDIT_PASS'
    for key,name in [('result_sha256','discovery/result.json'),('audit_sha256','discovery-audit.json'),('candidates_sha256','discovery/candidates.json'),('artifacts_sha256','discovery/artifacts.json')]:assert lock[key]==digest(OLD/name)
    assert da['result_sha256']==lock['result_sha256'] and da['candidates_sha256']==lock['candidates_sha256']
    for name,sha in lock['weights'].items():assert digest(UP/name)==sha
    return lock


def specifications(selected,randoms):
    return [dict(type='target',axis=a,indices=selected[a]) for a in range(2)]+[dict(type='same',random=j,indices=SUBSETS[i].tolist()) for j,i in enumerate(randoms)]+[dict(type='paired',axis=a,random=j,side=side,indices=SUBSETS[i].tolist()) for a in range(2) for j,i in enumerate(randoms) for side in ('target','random')]


def result_status(rows):
    geo=all(r['geometry'] for r in rows);normal=all(all(r['normal']['pass_axes']) for r in rows if r['kind']=='main');info=all(all(r['dose_pass']) for r in rows)
    local=all(all(r['local_pass']) for r in rows if r['kind']=='main');comparisons=[]
    for root in (9701,9702):
        for axis in range(2):
            v={k:next(r['selected_J'][axis] for r in rows if r['root']==root and r['kind']==k) for k in KINDS}
            comparisons.append(dict(root=root,axis=axis,**v,passed=v['main']>=max(v['frozen'],v['initial'])+.02))
    learning=all(r['passed'] for r in comparisons);ready=geo and normal and info
    status='PAIR_GEOMETRY_INVALID' if not geo else 'BASELINE_GATE_FAILED' if not normal else 'DOSE_INSUFFICIENT' if not info else 'LOCAL_CAUSAL_FAILED' if not local else 'LEARNING_COMPARISON_FAILED' if not learning else 'P2B_PAIRED_CONTROL_PASS'
    return dict(status=status,local_status=('LOCAL_CAUSAL_PASS' if local else 'LOCAL_CAUSAL_FAILED') if ready else 'NOT_EVALUATED',learning_status=('LEARNING_COMPARISON_PASS' if learning else 'LEARNING_COMPARISON_FAILED') if ready else 'NOT_EVALUATED',comparisons=comparisons)


def run(start):
    locked=prerequisite_all();cand=locked['candidates'];RUN.mkdir(parents=True,exist_ok=False);(RUN/'sources').mkdir()
    sources=[Path(__file__),PROTOCOL,*[Path(__file__).with_name(n) for n in ['pilot46_paired.py','pilot44_fixed.py','pilot44_learning.py','pilot45_fixed_joint.py','pilot45_joint_audit.py','pilot25_mechanisms.py','linear_gaussian_filter.py']]]
    for p in sources:shutil.copyfile(p,RUN/'sources'/p.name)
    for name in locked['weights']:shutil.copyfile(UP/name,RUN/name)
    save(RUN/'lock.json',dict(locked_at=now(),old_lock_sha256=digest(OLD/'discovery-lock.json'),calibration_acceptance=digest(CAL/'acceptance.json'),weights=locked['weights'],candidates=cand))
    save(RUN/'manifest.json',dict(protocol='1.3.69/16.69',sources={str(p.relative_to(ROOT)):digest(p) for p in sources},torch=torch.__version__,numpy=np.__version__,pair_columns=['d','b','target_coefficient','random_coefficient','actual_target_norm','actual_random_norm']))
    rows=[]
    for root in (9701,9702):
        y=data(root,7,512);np.save(RUN/f'{root}-data.npy',y);save(RUN/f'{root}-event.json',dict(generated_at=now(),split=7,lock_sha256=digest(RUN/'lock.json')))
        ref=reference(y);np.save(RUN/f'{root}-reference.npy',ref);target=windows(y);C=np.array(cand['scales'][str(root)])
        for kind in KINDS:
            tag=f'{root}-{kind}';w=dict(np.load(RUN/weight_name(root,kind)));net=Network().eval();net.load_state_dict({k:torch.from_numpy(v) for k,v in w.items()})
            for param in net.parameters():param.requires_grad_(False)
            full=net.gru(torch.from_numpy(y[:,:96].astype('float32')))[0];p=net.head(full).numpy();h=full.numpy()[:,TAUS].copy()
            np.save(RUN/f'{tag}-hidden.npy',full.numpy());np.save(RUN/f'{tag}-normal.npy',p)
            future=torch.from_numpy(np.stack([y[:,t+1:t+5] for t in TAUS],1).astype('float32').reshape(-1,4,2));null=advance(net,future,h)
            np.testing.assert_allclose(null,normal_window(p),atol=2e-5,rtol=2e-5);np.save(RUN/f'{tag}-null.npy',null)
            L0=(((normal_window(p)-target)@M)**2).mean((0,1,2));selected=cand['networks'][tag]['subsets'];specs=specifications(selected,cand['random_indices'][str(root)]);save(RUN/f'{tag}-specs.json',specs)
            shape=(642,512,8,4,2);budget(start,int(np.prod(shape))*4+256*512*8*6*8)
            outputs=np.lib.format.open_memmap(RUN/f'{tag}-outputs.npy',mode='w+',dtype='float32',shape=shape)
            details=np.lib.format.open_memmap(RUN/f'{tag}-pair.npy',mode='w+',dtype='float64',shape=(2,128,512,8,6));valid=np.empty((2,128,512,8),bool)
            L=np.empty((642,2));episodes=np.empty((642,512,2));temporal=np.empty((642,8,4,2));cached=None
            for i,spec in enumerate(specs):
                if spec['type']!='paired':changed=h.copy();changed[...,spec['indices']]=0
                elif spec['side']=='target':
                    a,j=spec['axis'],spec['random'];changed,cached,detail,ok=pair(h,np.array(selected[a]),np.array(spec['indices']));details[a,j]=detail;valid[a,j]=ok
                else:changed=cached
                pred=advance(net,future,changed);outputs[i]=pred;err=((pred.astype(float)-target)@M)**2
                L[i]=err.mean((0,1,2));episodes[i]=err.mean((1,2));temporal[i]=err.mean(0)
                if i%32==0:budget(start)
            outputs.flush();details.flush();np.save(RUN/f'{tag}-valid.npy',valid)
            delta=(L-L0)/C;J=delta-delta[:,::-1];D=np.array([J[130+a*256:130+(a+1)*256:2,a]-J[131+a*256:130+(a+1)*256:2,a] for a in range(2)])
            np.savez(RUN/f'{tag}-metrics.npz',L=L,L0=L0,C=C,delta=delta,J=J,D=D,episode_L=episodes,temporal_L=temporal)
            qs=np.quantile(J[2:130],.95,axis=0,method='linear');qd=np.quantile(D,.05,axis=1,method='linear')
            rho=np.minimum(details[...,0],details[...,1]);ratio=np.divide(rho,details[...,0],out=np.zeros_like(rho),where=details[...,0]>0)
            qdose=np.quantile(ratio.mean((2,3)),.05,axis=1,method='linear');nz=(rho>0).mean((1,2,3))
            passes=[bool(delta[a,a]>=.05 and J[a,a]>=.05 and delta[a,1-a]<=.5*delta[a,a] and J[a,a]>=qs[a]+.02 and qd[a]>=.02) for a in range(2)]
            row=dict(root=root,kind=kind,geometry=bool(valid.all()),normal=metrics(y,p,ref),selected_J=[float(J[a,a]) for a in range(2)],selected_delta=[delta[a].tolist() for a in range(2)],same_q95=qs.tolist(),paired_q05=qd.tolist(),dose_q05=qdose.tolist(),nonzero_fraction=nz.tolist(),dose_pass=((qdose>=.25)&(nz>=.95)).tolist(),local_pass=passes)
            rows.append(row);print(tag+' '+str({k:row[k] for k in ['paired_q05','local_pass','dose_pass','geometry']}),flush=True)
    budget(start);save(RUN/'result.json',dict(**result_status(rows),rows=rows,interventions=3852,cpu_seconds=time.process_time()-start,completed_at=now(),audit='PENDING',extraction='NOT_RUN',reuse='NOT_RUN'))
    save(RUN/'artifacts.json',{str(p.relative_to(RUN)):digest(p) for p in RUN.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    budget(start);print(read(RUN/'result.json')['status'],flush=True)


def audit(start):
    old=prerequisite_all();assert not (BASE/'audit.json').exists();lock=read(RUN/'lock.json');assert lock['candidates']==old['candidates'] and lock['weights']==old['weights']
    assert lock['old_lock_sha256']==digest(OLD/'discovery-lock.json') and lock['calibration_acceptance']==digest(CAL/'acceptance.json')
    for name,sha in read(RUN/'artifacts.json').items():assert digest(RUN/name)==sha
    for name,sha in read(RUN/'manifest.json')['sources'].items():assert digest(ROOT/name)==sha and digest(RUN/'sources'/Path(name).name)==sha
    for name,sha in lock['weights'].items():assert digest(RUN/name)==sha
    result=read(RUN/'result.json');maxdiff=0.;maxref=0.;checked=0;geo=True;normal=[];info=[];local=[];verified_J={}
    def quantile(v,p):
        a=sorted(map(float,v));x=(len(a)-1)*p;i=int(x);return a[i]+(x-i)*(a[min(i+1,len(a)-1)]-a[i])
    for root in (9701,9702):
        y=np.load(RUN/f'{root}-data.npy');np.testing.assert_allclose(y,regenerate(root,7,512),atol=1e-12,rtol=1e-12)
        event=read(RUN/f'{root}-event.json');assert event['split']==7 and event['generated_at']>lock['locked_at'] and event['lock_sha256']==digest(RUN/'lock.json')
        ref=np.load(RUN/f'{root}-reference.npy');altref=reference_path(y);np.testing.assert_allclose(ref,altref,atol=1e-8,rtol=1e-8);maxref=max(maxref,float(abs(ref-altref).max()))
        target=np.stack([y[:,t+2:t+6] for t in range(32,89,8)],1);future=np.stack([y[:,t+1:t+5] for t in range(32,89,8)],1).astype('float32').reshape(-1,4,2)
        C=np.array(old['candidates']['scales'][str(root)])
        for kind in KINDS:
            tag=f'{root}-{kind}';w=dict(np.load(RUN/weight_name(root,kind)));hfull=np.load(RUN/f'{tag}-hidden.npy');p=np.load(RUN/f'{tag}-normal.npy');ap,ah=forward(y[:,:96].astype('float32'),w)
            for x,z in [(ap,p),(ah,hfull)]:np.testing.assert_allclose(x,z,atol=2e-5,rtol=2e-5);maxdiff=max(maxdiff,float(abs(x-z).max()))
            h=hfull[:,TAUS];null,_=forward(future,w,h.reshape(-1,12));np.testing.assert_allclose(null.reshape(512,8,4,2),np.load(RUN/f'{tag}-null.npy'),atol=2e-5,rtol=2e-5);np.testing.assert_allclose(np.load(RUN/f'{tag}-null.npy'),normal_window(p),atol=2e-5,rtol=2e-5)
            row=next(r for r in result['rows'] if r['root']==root and r['kind']==kind);pn=metric_check(y,p,ref,row['normal'])
            L0=average2(errors(normal_window(p),target));stats=dict(np.load(RUN/f'{tag}-metrics.npz'));outputs=np.load(RUN/f'{tag}-outputs.npy',mmap_mode='r');detail=np.load(RUN/f'{tag}-pair.npy',mmap_mode='r');valid=np.load(RUN/f'{tag}-valid.npy')
            assert outputs.shape==(642,512,8,4,2);np.testing.assert_allclose(C,stats['C'],atol=0,rtol=0);np.testing.assert_allclose(L0,stats['L0'],atol=1e-10,rtol=1e-10)
            selected=old['candidates']['networks'][tag]['subsets'];randoms=old['candidates']['random_indices'][str(root)];specs=read(RUN/f'{tag}-specs.json');js=[];deltas=[]
            for i in range(642):
                changed=h.copy()
                if i<2:indices=selected[i];expected=dict(type='target',axis=i,indices=indices);changed[...,indices]=0
                elif i<130:indices=SUBSETS[randoms[i-2]].tolist();expected=dict(type='same',random=i-2,indices=indices);changed[...,indices]=0
                else:
                    a=(i-130)//256;j=((i-130)%256)//2;side=(i-130)%2;indices=SUBSETS[randoms[j]].tolist();ss=selected[a];expected=dict(type='paired',axis=a,random=j,side='target' if side==0 else 'random',indices=indices)
                    d=np.sqrt(np.sum(h[...,ss].astype(float)**2,-1));b=np.sqrt(np.sum(h[...,indices].astype(float)**2,-1));rho=np.minimum(d,b)
                    at=np.divide(rho,d,out=np.zeros_like(rho),where=d>0);ar=np.divide(rho,b,out=np.zeros_like(rho),where=b>0)
                    np.testing.assert_allclose(np.stack([d,b,at,ar],-1),detail[a,j,...,:4],atol=1e-10,rtol=1e-10)
                    coords,coef=(ss,at) if side==0 else (indices,ar)
                    for c in coords:changed[...,c]=h[...,c]*(1-coef).astype('float32')
                    np.testing.assert_array_equal(changed[...,np.setdiff1d(np.arange(12),coords)],h[...,np.setdiff1d(np.arange(12),coords)])
                    amount=np.sqrt(np.sum((changed.astype(float)-h)**2,-1));np.testing.assert_allclose(amount,detail[a,j,...,4+side],atol=1e-10,rtol=1e-10)
                    assert ((changed>=np.minimum(h,0))&(changed<=np.maximum(h,0))).all();assert ((coef>=0)&(coef<=1)).all()
                    ok=(abs(detail[a,j,...,4]-rho)<=1e-6+1e-5*rho)&(abs(detail[a,j,...,5]-rho)<=1e-6+1e-5*rho)&(abs(detail[a,j,...,4]-detail[a,j,...,5])<=1e-6+1e-5*rho)
                    np.testing.assert_array_equal(ok,valid[a,j]);geo&=bool(ok.all())
                assert specs[i]==expected
                alt,_=forward(future,w,changed.reshape(-1,12));alt=alt.reshape(512,8,4,2);np.testing.assert_allclose(alt,outputs[i],atol=2e-5,rtol=2e-5);maxdiff=max(maxdiff,float(abs(alt-outputs[i]).max()))
                e=errors(outputs[i],target);L=average2(e);delta=(L-L0)/C;J=delta-delta[::-1];js.append(J);deltas.append(delta)
                for key,value in [('L',L),('delta',delta),('J',J),('episode_L',e.mean((1,2))),('temporal_L',e.mean(0))]:np.testing.assert_allclose(value,stats[key][i],atol=1e-10,rtol=1e-10)
                checked+=1
                if i%32==0:budget(start)
            J=np.array(js);delta=np.array(deltas);D=np.array([[J[130+a*256+j*2,a]-J[131+a*256+j*2,a] for j in range(128)] for a in range(2)]);np.testing.assert_allclose(D,stats['D'],atol=1e-10,rtol=1e-10)
            qs=[quantile(J[2:130,a],.95) for a in range(2)];qd=[quantile(D[a],.05) for a in range(2)]
            rho=np.minimum(detail[...,0],detail[...,1]);ratio=np.divide(rho,detail[...,0],out=np.zeros_like(rho),where=detail[...,0]>0)
            qdose=[quantile(ratio[a].mean((1,2)),.05) for a in range(2)];nz=(rho>0).mean((1,2,3));inf=[qdose[a]>=.25 and nz[a]>=.95 for a in range(2)]
            lp=[bool(delta[a,a]>=.05 and J[a,a]>=.05 and delta[a,1-a]<=.5*delta[a,a] and J[a,a]>=qs[a]+.02 and qd[a]>=.02) for a in range(2)]
            for key,value in [('selected_J',[J[a,a] for a in range(2)]),('selected_delta',[delta[a] for a in range(2)]),('same_q95',qs),('paired_q05',qd),('dose_q05',qdose),('nonzero_fraction',nz),('dose_pass',inf),('local_pass',lp)]:np.testing.assert_allclose(value,row[key],atol=1e-10,rtol=1e-10)
            assert row['geometry']==bool(valid.all());verified_J[tag]=[float(J[a,a]) for a in range(2)];info.extend(inf)
            if kind=='main':normal.extend(pn);local.extend(lp)
            print('audit '+tag,flush=True)
    comparison=[]
    for root in (9701,9702):
        for a in range(2):
            v={k:verified_J[f'{root}-{k}'][a] for k in KINDS};passed=v['main']>=max(v['frozen'],v['initial'])+.02
            row=next(r for r in result['comparisons'] if r['root']==root and r['axis']==a)
            for k in KINDS:np.testing.assert_allclose(row[k],v[k],atol=1e-10,rtol=1e-10)
            assert row['passed']==passed;comparison.append(passed)
    expected='PAIR_GEOMETRY_INVALID' if not geo else 'BASELINE_GATE_FAILED' if not all(normal) else 'DOSE_INSUFFICIENT' if not all(info) else 'LOCAL_CAUSAL_FAILED' if not all(local) else 'LEARNING_COMPARISON_FAILED' if not all(comparison) else 'P2B_PAIRED_CONTROL_PASS'
    assert checked==result['interventions']==3852 and result['status']==expected
    ready=geo and all(normal) and all(info)
    assert result['local_status']==(('LOCAL_CAUSAL_PASS' if all(local) else 'LOCAL_CAUSAL_FAILED') if ready else 'NOT_EVALUATED')
    assert result['learning_status']==(('LEARNING_COMPARISON_PASS' if all(comparison) else 'LEARNING_COMPARISON_FAILED') if ready else 'NOT_EVALUATED')
    budget(start);save(BASE/'audit.json',dict(status='AUDIT_PASS',scientific_status=expected,result_sha256=digest(RUN/'result.json'),artifacts_sha256=digest(RUN/'artifacts.json'),interventions=checked,max_forward_difference=maxdiff,max_reference_difference=maxref,cpu_seconds=time.process_time()-start,completed_at=now()))
    print(read(BASE/'audit.json'),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.process_time()
    try:
        with torch.no_grad():{'run':run,'audit':audit}[sys.argv[1]](start)
    except BaseException as e:
        BASE.mkdir(parents=True,exist_ok=True);name=f'{sys.argv[1]}-error-{time.time_ns()}'
        (BASE/(name+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');save(BASE/(name+'.json'),dict(status='BUDGET_STOP' if 'BUDGET_STOP' in str(e) else 'ERROR',at=now(),error=str(e)));raise
