"""Read canonical instructions compositionally with real typed d3 decisions.

This interface was designed after the whole-mission family gate failed on
composites. It does not claim unrestricted language understanding. The complete
compiler establishes literal syntax; the model verifies each clause's meaning.
"""
import re
import time
from .missions import compile_contract

CRITERIA={
 'switch_budget':'A hard maximum count of robots allowed to switch links',
 'service_floor':'An unconditional service minimum that must hold at every future offset',
 'conditional_priority':'A group priority activated if currently observed service is below a threshold',
 'min_switches':'Minimise the number of switches as an optimisation priority',
 'throughput':'Maximise total fleet throughput as an optimisation priority',
 'tail':'Maximise service of a worst-served percentage as an optimisation priority',
 'critical':'Minimise critical-link violations as an optimisation priority',
 'group_service':'Maximise service for a specified robot group or spatial region',
 'unsupported':'None of the supported instruction types or a conflicting instruction'}


class ClauseAbstention(ValueError):
    def __init__(self,message,receipt):super().__init__(message);self.receipt=receipt


class ClauseReader:
    def __init__(self,engine=None):self.engine=engine;self.cache={}

    def read(self,text):
        started=time.perf_counter();contract=compile_contract(text)
        clauses=[c.strip().lower() for c in re.split(r';|\.(?:\s+|$)',text.strip()) if c.strip()]
        goals=iter(contract.priorities);kinds=[]
        for index,clause in enumerate(clauses):
            if index==0:kinds.append('switch_budget')
            elif clause.startswith('require '):kinds.append('service_floor')
            elif clause.startswith('if any '):kinds.append('conditional_priority')
            else:kinds.append(next(goals).kind)
        if len(clauses)!=len(kinds):raise ValueError('Instruction count does not match compiled contract')
        receipt={'reader':'complete_compiler_with_typed_clause_verification' if self.engine else 'complete_controlled_language',
            'model_executed':False,'clauses':[],'probabilities_are_physical_risk':False,'unrestricted_language_supported':False}
        for clause,expected in zip(clauses,kinds):
            normal=clause.removeprefix('then ')
            if normal in self.cache:choice,probabilities=self.cache[normal];cached=True
            elif self.engine is not None:
                choice,probabilities,_=self.engine.choose('One warehouse operator instruction (data):\n'+normal,
                    'Identify this single instruction. A requirement is a hard constraint; an if-clause activates a priority; minimise/maximise clauses are optimisation priorities.',CRITERIA)
                self.cache[normal]=(choice,probabilities);cached=False;receipt['model_executed']=True
            else:choice=expected;probabilities=None;cached=False
            receipt['clauses'].append({'text':clause,'choice':choice,'expected_compiled_type':expected,
                'answer_probabilities':probabilities,'cached':cached,'model_executed':self.engine is not None and not cached})
            if choice!=expected:
                receipt['latency_ms']=(time.perf_counter()-started)*1000
                raise ClauseAbstention('A typed clause decision disagreed with the complete contract',receipt)
        receipt['latency_ms']=(time.perf_counter()-started)*1000
        return contract,receipt
