class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[8, 1000], primals_2: i64[8]):
        # File: /scratch/ngimel/work/pytorch/benchmarks/dynamo/timm_models.py:319, code: return self.loss(pred, self.target) / 10.0
        amax: f32[8, 1] = torch.ops.aten.amax.default(primals_1, [1], True)
        sub: f32[8, 1000] = torch.ops.aten.sub.Tensor(primals_1, amax);  primals_1 = amax = None
        exp: f32[8, 1000] = torch.ops.aten.exp.default(sub)
        sum_1: f32[8, 1] = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: f32[8, 1] = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: f32[8, 1000] = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        unsqueeze: i64[8, 1] = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        gather: f32[8, 1] = torch.ops.aten.gather.default(sub_1, 1, unsqueeze)
        squeeze: f32[8] = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: f32[8] = torch.ops.aten.neg.default(squeeze);  squeeze = None
        mean: f32[] = torch.ops.aten.mean.default(neg);  neg = None
        div: f32[] = torch.ops.aten.div.Tensor(mean, 10.0);  mean = None
        return [div, sub_1, unsqueeze]
        