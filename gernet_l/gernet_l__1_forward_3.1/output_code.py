
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
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]}
)
@triton.jit
def triton__0(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1000*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask & (_tmp1 < tmp0), tmp0, _tmp1)
    tmp1 = tl.max(_tmp1, 1)[:, None]
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp2 = tl.load(in_ptr0 + (r1 + (1000*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tmp2 - tmp1
        tmp4 = tl.exp(tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (1000*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tmp6 - tmp1
        tmp8 = tl.log(tmp5)
        tmp9 = tmp7 - tmp8
        tl.store(out_ptr2 + (r1 + (1000*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp9, rmask & xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[1, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton__1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (tmp0 + (1000*r0)), rmask, eviction_policy='evict_last')
        tmp2 = -tmp1
        _tmp3 = tl.where(rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = 10.0
    tmp7 = tmp5 / tmp6
    tl.store(in_out_ptr0 + (0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp7, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty_strided((8, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        print('triton__0', 'in_ptr0', 'primals_1', (primals_1.sum()/primals_1.nelement()).item(), primals_1.amax().item(), primals_1.amin().item())
        stream0 = get_cuda_stream(0)
        triton__0.run(primals_1, buf2, 8, 1000, grid=grid(8), stream=stream0)
        print('triton__0', 'out_ptr2', 'buf2', (buf2.sum()/buf2.nelement()).item(), buf2.amax().item(), buf2.amin().item())
        del primals_1
        buf3 = empty_strided((), (), device='cuda', dtype=torch.float32)
        buf4 = buf3; del buf3  # reuse
        print('triton__1', 'in_out_ptr0', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        print('triton__1', 'in_ptr0', 'primals_2', (primals_2.sum()/primals_2.nelement()).item(), primals_2.amax().item(), primals_2.amin().item())
        print('triton__1', 'in_ptr1', 'buf2', (buf2.sum()/buf2.nelement()).item(), buf2.amax().item(), buf2.amin().item())
        triton__1.run(buf4, primals_2, buf2, 1, 8, grid=grid(1), stream=stream0)
        print('triton__1', 'in_out_ptr0', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        return (buf4, buf2, as_strided(primals_2, (8, 1), (1, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.int64)
    print_performance(lambda: call([primals_1, primals_2]))
