from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch._inductor.lowering

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
from torch._inductor.triton_ops.autotune import pointwise, reduction
from torch._inductor.utils import instance_descriptor


@reduction(size_hints=[65536, 2048],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    tmp6 = tl.zeros([XBLOCK, 1], tl.float32) #tl.max(_tmp6, 1)[:, None]
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = -3.4028234663852886e+38
        #tmp12 = tmp10  # this works
        tmp12 = tl.where(tmp10 != tmp10, tmp10, tl.where(tmp10 > tmp11, tmp10, tmp11)) # this doesn't
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp16 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask & xmask, eviction_policy='evict_last')
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = -3.4028234663852886e+38
        tmp21 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 > tmp20, tmp19, tmp20))
        tmp22 = tmp21 - tmp6
        tmp23 = tl.exp(tmp22)
        tmp24 = tmp23 / tmp15
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp24, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (2048*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp25, rmask & xmask)







''')
async_compile.wait(globals())
del async_compile
