
from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


triton__0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = -1.0
    tl.store(out_ptr0 + (tmp0 + (1000*x0) + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton__2(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1_load = tl.load(in_ptr1 + (0))
    tmp1 = tl.broadcast_to(tmp1_load, [XBLOCK, RBLOCK])
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1000*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = 10.0
        tmp3 = tmp1 / tmp2
        tmp4 = 8.0
        tmp5 = tmp3 / tmp4
        tmp6 = tmp0 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (r1 + (1000*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr2 + (r1 + (1000*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp9 = 10.0
        tmp10 = tmp1 / tmp9
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = tmp8 * tmp12
        tmp15 = tl.exp(tmp14)
        tmp16 = tmp15 * tmp7
        tmp17 = tmp13 - tmp16
        tl.store(out_ptr1 + (r1 + (1000*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp17, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    sub_1, unsqueeze, tangents_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton__0.run(buf0, 8000, grid=grid(8000), stream=stream0)
        print('triton__0', 'out_ptr0', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        print('triton__1', 'in_ptr0', 'unsqueeze', (unsqueeze.sum()/unsqueeze.nelement()).item(), unsqueeze.amax().item(), unsqueeze.amin().item())
        triton__1.run(unsqueeze, buf0, 8, grid=grid(8), stream=stream0)
        print('triton__1', 'out_ptr0', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        del unsqueeze
        buf3 = empty_strided((8, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        print('triton__2', 'in_ptr0', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        print('triton__2', 'in_ptr1', 'tangents_1', (tangents_1.sum()/tangents_1.nelement()).item(), tangents_1.amax().item(), tangents_1.amin().item())
        print('triton__2', 'in_ptr2', 'sub_1', (sub_1.sum()/sub_1.nelement()).item(), sub_1.amax().item(), sub_1.amin().item())
        triton__2.run(buf0, tangents_1, sub_1, buf3, 8, 1000, grid=grid(8), stream=stream0)
        print('triton__2', 'out_ptr1', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        del buf0
        del sub_1
        del tangents_1
        return (buf3, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    sub_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([sub_1, unsqueeze, tangents_1]))
