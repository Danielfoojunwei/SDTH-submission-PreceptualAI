"""Strong observed-power controls and a separate frozen-label probe baseline."""
from __future__ import annotations
import numpy as np
from . import common as c, training as t
from .evaluation import metrics,mean_template,power_level_transfer,ridge_fit


def run_baselines():
    c.check_sources()
    path=c.OUT/'baselines.json'
    if path.exists():
        saved=c.read_json(path)
        if saved['source_receipt_sha256']!=c.digest(c.OUT/'registration/source.json'):
            raise RuntimeError('Baseline source identity changed')
        references=[value for row in saved['factory_rf_ridge'].values() for value in [row['archive']]]
        references.append(saved['factory_position_ridge']['archive'])
        references.extend({'path':row['prediction_path'],'sha256':row['prediction_sha256']} for row in saved['domains'].values())
        for ref in references:t._check_reference(ref)
        return saved
    p,receipt=c.protocol(),c.read_json(c.OUT/'data/receipt.json')
    training=c.read_json(c.OUT/'training.json')
    if training.get('completed') is not True or training.get('fits')!=18:
        raise RuntimeError('Finish frozen backbone fits before label/probe baseline')
    factory=t.prepared(p['training_recording']);split=t.split_arrays()
    info=receipt['recordings'][p['training_recording']]
    a,b=np.array(info['view_a']),np.array(info['view_b'])
    train_a,train_b=factory['power_db'][split['train']][:,a],factory['power_db'][split['train']][:,b]
    val_a,val_b=factory['power_db'][split['validation']][:,a],factory['power_db'][split['validation']][:,b]
    kwargs={'alphas':tuple(p['baselines']['ridge_alphas'])}
    ab=ridge_fit(train_a.reshape(len(train_a),-1),train_b,val_a.reshape(len(val_a),-1),val_b,**kwargs)
    ba=ridge_fit(train_b.reshape(len(train_b),-1),train_a,val_b.reshape(len(val_b),-1),val_a,**kwargs)
    ridge_references={}
    for direction,model in (('ab',ab),('ba',ba)):
        archive=t.save_npz(c.OUT/'baselines'/f'factory-ridge-{direction}.npz',model.as_arrays()|{
            'train_rows':split['train'],'validation_rows':split['validation'],'test_rows':split['test']})
        ridge_references[direction]={'archive':archive,'metadata':model.metadata}
    global_template=factory['power_db'][split['train']].mean((0,1),dtype='f8')
    global_level=float(factory['power_db'][split['train']].mean(dtype='f8'))
    domains={}
    for name in [p['training_recording'],*p['heldout_recordings']]:
        public=t.prepared(name);item=receipt['recordings'][name]
        rows=split['test'] if name==p['training_recording'] else np.array(item['eligible_rows'],dtype='i8')
        va,vb=np.array(item['view_a']),np.array(item['view_b'])
        xa,xb=public['power_db'][rows][:,va],public['power_db'][rows][:,vb]
        arrays={'source_rows':rows,'view_a':va,'view_b':vb}
        if name==p['training_recording']:
            arrays.update(constant_ab=mean_template(train_b,len(rows)),constant_ba=mean_template(train_a,len(rows)),
                power_level_ab=power_level_transfer(train_a,train_b,xa),power_level_ba=power_level_transfer(train_b,train_a,xb),
                ridge_ab=ab.predict(xa.reshape(len(rows),-1)),ridge_ba=ba.predict(xb.reshape(len(rows),-1)))
        else:
            ca=np.broadcast_to(global_template,(len(rows),len(vb),len(global_template))).copy()
            cb=np.broadcast_to(global_template,(len(rows),len(va),len(global_template))).copy()
            arrays.update(constant_ab=ca,constant_ba=cb,
                power_level_ab=ca+(xa.mean((1,2),dtype='f8')-global_level)[:,None,None],
                power_level_ba=cb+(xb.mean((1,2),dtype='f8')-global_level)[:,None,None])
        archive=t.save_npz(c.OUT/'baselines'/f'{name}.predictions.npz',arrays)
        c.write_once(c.OUT/'baselines'/f'{name}.before-targets.json',{
            'archive':archive,'prepared_public':receipt['files'][name+'.public'],
            'no_query_target_statistics_used':True})
        truth=np.concatenate((xb,xa),1)
        names=('constant','power_level','ridge') if name==p['training_recording'] else ('constant','power_level')
        domains[name]={'prediction_path':archive['path'],'prediction_sha256':archive['sha256'],
            'metrics':{key:metrics(np.concatenate((arrays[key+'_ab'],arrays[key+'_ba']),1),truth) for key in names},
            'template_scope':'per-targetarrayTRAIN' if name==p['training_recording'] else 'globalTRAINfrequencytemplate averaged over all factory arrays',
            'ood_factory_ridge_applicable':name==p['training_recording']}
    pose=t.labels(p['training_recording'])[:,:2]
    raw_head=ridge_fit(train_a.reshape(len(train_a),-1),pose[split['train']],
        val_a.reshape(len(val_a),-1),pose[split['validation']],**kwargs)
    test_input=factory['power_db'][split['test']][:,a].reshape(len(split['test']),-1)
    prediction=raw_head.predict(test_input)
    archive=t.save_npz(c.OUT/'baselines/factory-position.npz',raw_head.as_arrays()|{
        'train_input':train_a.reshape(len(train_a),-1),'validation_input':val_a.reshape(len(val_a),-1),
        'test_input':test_input,'test_prediction':prediction,
        'train_rows':split['train'],'validation_rows':split['validation'],'test_rows':split['test']})
    error=np.linalg.norm(prediction-pose[split['test']],axis=-1)
    result={'schema':1,'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
            'domains':domains,'factory_rf_ridge':ridge_references,
            'factory_position_ridge':{'archive':archive,'metadata':raw_head.metadata,
                'metrics':{'mean_error_m':float(error.mean()),'median_error_m':float(np.median(error))}},
            'test_statistics_used_for_fitting':False,'scope':'observed array-power prediction and positioning probes; not service qualification'}
    c.check_sources();c.write_once(path,result)
    return result
