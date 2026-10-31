"""Small native-outcome models with explicit AP/channel interference context."""
from __future__ import annotations
import torch
from torch import nn


class NativeModel(nn.Module):
    def __init__(self, features=45, hidden=48, graph=False):
        super().__init__()
        self.config={'features':features,'hidden':hidden,'graph':graph}
        self.encoder=nn.Sequential(nn.Linear(features,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.message=nn.Sequential(nn.Linear(hidden*3,hidden),nn.SiLU()) if graph else None
        self.head=nn.Linear(hidden,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    @staticmethod
    def pool(nodes, groups):
        k,n,h=nodes.shape
        width=int(groups.max())+1
        index=groups[...,None].expand(-1,-1,h)
        total=nodes.new_zeros((k,width,h)).scatter_add_(1,index,nodes)
        count=nodes.new_zeros((k,width,1)).scatter_add_(1,groups[...,None],nodes.new_ones((k,n,1)))
        return (total/count.clamp(min=1)).gather(1,index)

    def forward(self, features, plans, groups, prior):
        node=self.encoder(features)
        if self.message is not None:
            node=node+self.message(torch.cat([node,self.pool(node,plans),self.pool(node,groups)],-1))
        delta=self.head(node)
        baseline=torch.logit(prior.clamp(.02,.98))
        return {'fractions':torch.sigmoid(baseline+delta[...,:2]),'failure_logits':delta[...,2]}
