class GraphModule(torch.nn.Module):
    def forward(self, sub_1: f32[8, 1000], unsqueeze: i64[8, 1], tangents_1: f32[]):
        # File: /scratch/ngimel/work/pytorch/benchmarks/dynamo/timm_models.py:319, code: return self.loss(pred, self.target) / 10.0
        full: f32[] = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_1: f32[] = torch.ops.aten.div.Tensor(tangents_1, 10.0);  tangents_1 = None
        div_2: f32[] = torch.ops.aten.div.Tensor(div_1, full);  div_1 = full = None
        full_like: f32[8, 1000] = torch.ops.aten.full_like.default(sub_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False, memory_format = torch.preserve_format)
        scatter: f32[8, 1000] = torch.ops.aten.scatter.value(full_like, 1, unsqueeze, -1.0);  full_like = unsqueeze = None
        mul: f32[8, 1000] = torch.ops.aten.mul.Tensor(scatter, div_2);  scatter = div_2 = None
        exp_1: f32[8, 1000] = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_2: f32[8, 1] = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: f32[8, 1000] = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2: f32[8, 1000] = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        return [sub_2, None]
        