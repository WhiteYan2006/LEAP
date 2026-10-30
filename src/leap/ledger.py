"""Causal versions of a total-length estimate; progress is subtracted exactly once."""
import math
from bisect import bisect_right
METHODS=('ttt_static','ttt_bias','ttt_ridge','ttt_rls','ttt_rls_shadow','ttt_rls_sparse')
def need(ok,message):
    if not ok:raise ValueError(message)
class Ledger:
    def __init__(self):self.rows={};self.times={}
    def publish(self,row):
        required=['request_id','request_slot','generation','phase','source_age','value','published_ns','accepted_ns','model_version']
        need(all(k in row for k in required),'Incomplete TTT publication')
        need(row['phase'] in ('initial','online') and row['source_age']==0,'TTT estimates total length from cached initial features')
        need(type(row['model_version']) is int and row['model_version']>=0,'Invalid version')
        need(math.isfinite(row['value']) and 1<=row['value']<=2048 and 0<row['published_ns']<=row['accepted_ns'],'Invalid publication value/time')
        need((row['model_version']==0)==(row['phase']=='initial'),'Initial/adapted version mismatch')
        key=(row['request_slot'],row['generation']);rows=self.rows.setdefault(key,[]);times=self.times.setdefault(key,[])
        if rows:
            need(rows[-1]['request_id']==row['request_id'] and row['model_version']>rows[-1]['model_version'],'Repeated/regressing version or stale incarnation')
            need(row['accepted_ns']>=times[-1],'Regressing owner clock')
        else:need(row['phase']=='initial','Adaptation preceded original readout')
        rows.append(dict(row));times.append(row['accepted_ns'])
    def snapshot(self,state,method,now):
        need(method in METHODS,'Unknown TTT arm');key=(state.slot,state.generation)
        i=bisect_right(self.times.get(key,[]),now)-1
        if i<0:return dict(score=None,source_age=None,source_value=None,phase=None,published_ns=None,accepted_ns=None,model_version=None)
        row=self.rows[key][i]
        need(row['request_id']==state.request_id and 0<=state.generated<state.budget,'Stale/nonlive request')
        need(method!='ttt_static' or row['model_version']==0,'Static arm adapted')
        return dict(score=max(1.,min(float(state.budget-state.generated),row['value']-state.generated)),source_age=0,source_value=row['value'],
                    phase=row['phase'],published_ns=row['published_ns'],accepted_ns=row['accepted_ns'],model_version=row['model_version'])
    def remove(self,state):
        key=(state.slot,state.generation);self.rows.pop(key,None);self.times.pop(key,None)
