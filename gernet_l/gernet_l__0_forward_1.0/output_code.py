
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton__0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton__1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 131072.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton__2(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp3 = tmp2 * tmp2
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp4, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton__3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 131072.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000076294527394
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton__5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton__6(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 32768.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton__7(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp3 = tmp2 * tmp2
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp4, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton__8(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 32768.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.000030518509476
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp17 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = tl.where(0 != 0, 0, tl.where(0 > tmp26, 0, tmp26))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp27, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 8192.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp10 = tl.load(in_ptr0 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 8192.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0001220852154804
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp17 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = tl.where(0 != 0, 0, tl.where(0 > tmp26, 0, tmp26))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp27, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.where(0 != 0, 0, tl.where(0 > tmp15, 0, tmp15))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 8192.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp10 = tl.load(in_ptr0 + (r1 + (1024*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 8192.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0001220852154804
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 2048.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp10 = tl.load(in_ptr0 + (r1 + (256*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 2048.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0004885197850513
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 2048.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp10 = tl.load(in_ptr0 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 2048.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0004885197850513
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp17 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = tl.where(0 != 0, 0, tl.where(0 > tmp26, 0, tmp26))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp27, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.where(0 != 0, 0, tl.where(0 > tmp15, 0, tmp15))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (491520*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 2048.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp10 = tl.load(in_ptr0 + (r1 + (256*x0) + (491520*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 2048.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0004885197850513
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1920
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (122880*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 512.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp10 = tl.load(in_ptr0 + (r1 + (64*x0) + (122880*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0019569471624266
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1920
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 512.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp10 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0019569471624266
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp17 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.libdevice.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = tl.where(0 != 0, 0, tl.where(0 > tmp26, 0, tmp26))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp27, xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.where(0 != 0, 0, tl.where(0 > tmp15, 0, tmp15))
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 512.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp10 = tl.load(in_ptr0 + (r1 + (64*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tmp10 - tmp3
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp23 = tl.load(in_ptr2 + (x0), xmask)
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.libdevice.rsqrt(tmp17)
    tmp19 = 1.0019569471624266
    tmp20 = tmp15 * tmp19
    tmp21 = 0.1
    tmp22 = tmp20 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp26, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton__30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20480
    rnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2560
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp10 = tl.load(in_ptr3 + (x0), xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp4 = 512.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = tl.libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.where(0 != 0, 0, tl.where(0 > tmp13, 0, tmp13))
        tmp15 = 0.0
        tmp16 = tmp14 <= tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp14, _tmp17)
        tl.store(out_ptr1 + (r2 + (64*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp18 = 64.0
    tmp19 = tmp17 / tmp18
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp19, xmask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK])
    tmp1 = 1
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = aten.convolution(primals_60, primals_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf0, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf1 = empty_strided((1, 32, 1, 1, 16), (512, 16, 512, 512, 1), device='cuda', dtype=torch.float32)
        print('triton__0', 'in_ptr0', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        stream0 = get_cuda_stream(0)
        triton__0.run(buf0, buf1, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__0', 'out_ptr0', 'buf1', (buf1.sum()/buf1.nelement()).item(), buf1.amax().item(), buf1.amin().item())
        buf2 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf3 = buf2; del buf2  # reuse
        buf7 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__1', 'in_out_ptr0', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__1', 'in_ptr0', 'buf1', (buf1.sum()/buf1.nelement()).item(), buf1.amax().item(), buf1.amin().item())
        print('triton__1', 'in_ptr1', 'primals_62', (primals_62.sum()/primals_62.nelement()).item(), primals_62.amax().item(), primals_62.amin().item())
        triton__1.run(buf3, buf1, primals_62, buf7, 32, 16, grid=grid(32), stream=stream0)
        print('triton__1', 'in_out_ptr0', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__1', 'out_ptr0', 'buf7', (buf7.sum()/buf7.nelement()).item(), buf7.amax().item(), buf7.amin().item())
        del primals_62
        buf4 = buf1; del buf1  # reuse
        print('triton__2', 'in_ptr0', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        print('triton__2', 'in_ptr1', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        triton__2.run(buf0, buf3, buf4, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__2', 'out_ptr0', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        buf5 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__3', 'in_ptr0', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        print('triton__3', 'in_ptr1', 'primals_63', (primals_63.sum()/primals_63.nelement()).item(), primals_63.amax().item(), primals_63.amin().item())
        triton__3.run(buf4, primals_63, buf5, buf6, buf8, 32, 16, grid=grid(32), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf5', (buf5.sum()/buf5.nelement()).item(), buf5.amax().item(), buf5.amin().item())
        print('triton__3', 'out_ptr1', 'buf6', (buf6.sum()/buf6.nelement()).item(), buf6.amax().item(), buf6.amin().item())
        print('triton__3', 'out_ptr2', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        del primals_63
        buf9 = empty_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda', dtype=torch.float32)
        print('triton__4', 'in_ptr0', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        print('triton__4', 'in_ptr1', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__4', 'in_ptr2', 'buf5', (buf5.sum()/buf5.nelement()).item(), buf5.amax().item(), buf5.amin().item())
        print('triton__4', 'in_ptr3', 'primals_64', (primals_64.sum()/primals_64.nelement()).item(), primals_64.amax().item(), primals_64.amin().item())
        print('triton__4', 'in_ptr4', 'primals_65', (primals_65.sum()/primals_65.nelement()).item(), primals_65.amax().item(), primals_65.amin().item())
        triton__4.run(buf0, buf3, buf5, primals_64, primals_65, buf9, 4194304, grid=grid(4194304), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf9', (buf9.sum()/buf9.nelement()).item(), buf9.amax().item(), buf9.amin().item())
        del buf5
        del primals_65
        buf10 = aten.convolution(buf9, primals_2, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf10, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf11 = as_strided(buf4, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128)); del buf4  # reuse
        print('triton__5', 'in_ptr0', 'buf10', (buf10.sum()/buf10.nelement()).item(), buf10.amax().item(), buf10.amin().item())
        triton__5.run(buf10, buf11, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf11', (buf11.sum()/buf11.nelement()).item(), buf11.amax().item(), buf11.amin().item())
        buf12 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf13 = buf12; del buf12  # reuse
        buf17 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__6', 'in_out_ptr0', 'buf13', (buf13.sum()/buf13.nelement()).item(), buf13.amax().item(), buf13.amin().item())
        print('triton__6', 'in_ptr0', 'buf11', (buf11.sum()/buf11.nelement()).item(), buf11.amax().item(), buf11.amin().item())
        print('triton__6', 'in_ptr1', 'primals_67', (primals_67.sum()/primals_67.nelement()).item(), primals_67.amax().item(), primals_67.amin().item())
        triton__6.run(buf13, buf11, primals_67, buf17, 128, 4, grid=grid(128), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf13', (buf13.sum()/buf13.nelement()).item(), buf13.amax().item(), buf13.amin().item())
        print('triton__6', 'out_ptr0', 'buf17', (buf17.sum()/buf17.nelement()).item(), buf17.amax().item(), buf17.amin().item())
        del primals_67
        buf14 = buf11; del buf11  # reuse
        print('triton__7', 'in_ptr0', 'buf10', (buf10.sum()/buf10.nelement()).item(), buf10.amax().item(), buf10.amin().item())
        print('triton__7', 'in_ptr1', 'buf13', (buf13.sum()/buf13.nelement()).item(), buf13.amax().item(), buf13.amin().item())
        triton__7.run(buf10, buf13, buf14, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf14', (buf14.sum()/buf14.nelement()).item(), buf14.amax().item(), buf14.amin().item())
        buf15 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__8', 'in_ptr0', 'buf14', (buf14.sum()/buf14.nelement()).item(), buf14.amax().item(), buf14.amin().item())
        print('triton__8', 'in_ptr1', 'primals_68', (primals_68.sum()/primals_68.nelement()).item(), primals_68.amax().item(), primals_68.amin().item())
        triton__8.run(buf14, primals_68, buf15, buf16, buf18, 128, 4, grid=grid(128), stream=stream0)
        print('triton__8', 'out_ptr0', 'buf15', (buf15.sum()/buf15.nelement()).item(), buf15.amax().item(), buf15.amin().item())
        print('triton__8', 'out_ptr1', 'buf16', (buf16.sum()/buf16.nelement()).item(), buf16.amax().item(), buf16.amin().item())
        print('triton__8', 'out_ptr2', 'buf18', (buf18.sum()/buf18.nelement()).item(), buf18.amax().item(), buf18.amin().item())
        del primals_68
        buf19 = empty_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda', dtype=torch.float32)
        print('triton__9', 'in_ptr0', 'buf10', (buf10.sum()/buf10.nelement()).item(), buf10.amax().item(), buf10.amin().item())
        print('triton__9', 'in_ptr1', 'buf13', (buf13.sum()/buf13.nelement()).item(), buf13.amax().item(), buf13.amin().item())
        print('triton__9', 'in_ptr2', 'buf15', (buf15.sum()/buf15.nelement()).item(), buf15.amax().item(), buf15.amin().item())
        print('triton__9', 'in_ptr3', 'primals_69', (primals_69.sum()/primals_69.nelement()).item(), primals_69.amax().item(), primals_69.amin().item())
        print('triton__9', 'in_ptr4', 'primals_70', (primals_70.sum()/primals_70.nelement()).item(), primals_70.amax().item(), primals_70.amin().item())
        triton__9.run(buf10, buf13, buf15, primals_69, primals_70, buf19, 4194304, grid=grid(4194304), stream=stream0)
        print('triton__9', 'out_ptr0', 'buf19', (buf19.sum()/buf19.nelement()).item(), buf19.amax().item(), buf19.amin().item())
        del primals_70
        buf20 = aten.convolution(buf19, primals_3, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf20, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf21 = buf14; del buf14  # reuse
        print('triton__5', 'in_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        triton__5.run(buf20, buf21, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf21', (buf21.sum()/buf21.nelement()).item(), buf21.amax().item(), buf21.amin().item())
        buf22 = buf15; del buf15  # reuse
        buf23 = buf22; del buf22  # reuse
        buf27 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__6', 'in_out_ptr0', 'buf23', (buf23.sum()/buf23.nelement()).item(), buf23.amax().item(), buf23.amin().item())
        print('triton__6', 'in_ptr0', 'buf21', (buf21.sum()/buf21.nelement()).item(), buf21.amax().item(), buf21.amin().item())
        print('triton__6', 'in_ptr1', 'primals_72', (primals_72.sum()/primals_72.nelement()).item(), primals_72.amax().item(), primals_72.amin().item())
        triton__6.run(buf23, buf21, primals_72, buf27, 128, 4, grid=grid(128), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf23', (buf23.sum()/buf23.nelement()).item(), buf23.amax().item(), buf23.amin().item())
        print('triton__6', 'out_ptr0', 'buf27', (buf27.sum()/buf27.nelement()).item(), buf27.amax().item(), buf27.amin().item())
        del primals_72
        buf24 = buf21; del buf21  # reuse
        print('triton__7', 'in_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        print('triton__7', 'in_ptr1', 'buf23', (buf23.sum()/buf23.nelement()).item(), buf23.amax().item(), buf23.amin().item())
        triton__7.run(buf20, buf23, buf24, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        buf25 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf26 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf28 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__8', 'in_ptr0', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        print('triton__8', 'in_ptr1', 'primals_73', (primals_73.sum()/primals_73.nelement()).item(), primals_73.amax().item(), primals_73.amin().item())
        triton__8.run(buf24, primals_73, buf25, buf26, buf28, 128, 4, grid=grid(128), stream=stream0)
        print('triton__8', 'out_ptr0', 'buf25', (buf25.sum()/buf25.nelement()).item(), buf25.amax().item(), buf25.amin().item())
        print('triton__8', 'out_ptr1', 'buf26', (buf26.sum()/buf26.nelement()).item(), buf26.amax().item(), buf26.amin().item())
        print('triton__8', 'out_ptr2', 'buf28', (buf28.sum()/buf28.nelement()).item(), buf28.amax().item(), buf28.amin().item())
        del primals_73
        buf29 = aten.convolution(buf9, primals_4, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf29, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf30 = buf24; del buf24  # reuse
        print('triton__5', 'in_ptr0', 'buf29', (buf29.sum()/buf29.nelement()).item(), buf29.amax().item(), buf29.amin().item())
        triton__5.run(buf29, buf30, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf30', (buf30.sum()/buf30.nelement()).item(), buf30.amax().item(), buf30.amin().item())
        buf31 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf32 = buf31; del buf31  # reuse
        buf36 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__6', 'in_out_ptr0', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        print('triton__6', 'in_ptr0', 'buf30', (buf30.sum()/buf30.nelement()).item(), buf30.amax().item(), buf30.amin().item())
        print('triton__6', 'in_ptr1', 'primals_77', (primals_77.sum()/primals_77.nelement()).item(), primals_77.amax().item(), primals_77.amin().item())
        triton__6.run(buf32, buf30, primals_77, buf36, 128, 4, grid=grid(128), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        print('triton__6', 'out_ptr0', 'buf36', (buf36.sum()/buf36.nelement()).item(), buf36.amax().item(), buf36.amin().item())
        del primals_77
        buf33 = buf30; del buf30  # reuse
        print('triton__7', 'in_ptr0', 'buf29', (buf29.sum()/buf29.nelement()).item(), buf29.amax().item(), buf29.amin().item())
        print('triton__7', 'in_ptr1', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        triton__7.run(buf29, buf32, buf33, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf33', (buf33.sum()/buf33.nelement()).item(), buf33.amax().item(), buf33.amin().item())
        buf34 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__8', 'in_ptr0', 'buf33', (buf33.sum()/buf33.nelement()).item(), buf33.amax().item(), buf33.amin().item())
        print('triton__8', 'in_ptr1', 'primals_78', (primals_78.sum()/primals_78.nelement()).item(), primals_78.amax().item(), primals_78.amin().item())
        triton__8.run(buf33, primals_78, buf34, buf35, buf37, 128, 4, grid=grid(128), stream=stream0)
        print('triton__8', 'out_ptr0', 'buf34', (buf34.sum()/buf34.nelement()).item(), buf34.amax().item(), buf34.amin().item())
        print('triton__8', 'out_ptr1', 'buf35', (buf35.sum()/buf35.nelement()).item(), buf35.amax().item(), buf35.amin().item())
        print('triton__8', 'out_ptr2', 'buf37', (buf37.sum()/buf37.nelement()).item(), buf37.amax().item(), buf37.amin().item())
        del buf33
        del primals_78
        buf38 = empty_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda', dtype=torch.float32)
        buf39 = buf38; del buf38  # reuse
        print('triton__10', 'in_out_ptr0', 'buf39', (buf39.sum()/buf39.nelement()).item(), buf39.amax().item(), buf39.amin().item())
        print('triton__10', 'in_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        print('triton__10', 'in_ptr1', 'buf23', (buf23.sum()/buf23.nelement()).item(), buf23.amax().item(), buf23.amin().item())
        print('triton__10', 'in_ptr2', 'buf25', (buf25.sum()/buf25.nelement()).item(), buf25.amax().item(), buf25.amin().item())
        print('triton__10', 'in_ptr3', 'primals_74', (primals_74.sum()/primals_74.nelement()).item(), primals_74.amax().item(), primals_74.amin().item())
        print('triton__10', 'in_ptr4', 'primals_75', (primals_75.sum()/primals_75.nelement()).item(), primals_75.amax().item(), primals_75.amin().item())
        print('triton__10', 'in_ptr5', 'buf29', (buf29.sum()/buf29.nelement()).item(), buf29.amax().item(), buf29.amin().item())
        print('triton__10', 'in_ptr6', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        print('triton__10', 'in_ptr7', 'buf34', (buf34.sum()/buf34.nelement()).item(), buf34.amax().item(), buf34.amin().item())
        print('triton__10', 'in_ptr8', 'primals_79', (primals_79.sum()/primals_79.nelement()).item(), primals_79.amax().item(), primals_79.amin().item())
        print('triton__10', 'in_ptr9', 'primals_80', (primals_80.sum()/primals_80.nelement()).item(), primals_80.amax().item(), primals_80.amin().item())
        triton__10.run(buf39, buf20, buf23, buf25, primals_74, primals_75, buf29, buf32, buf34, primals_79, primals_80, 4194304, grid=grid(4194304), stream=stream0)
        print('triton__10', 'in_out_ptr0', 'buf39', (buf39.sum()/buf39.nelement()).item(), buf39.amax().item(), buf39.amin().item())
        del buf25
        del buf34
        del primals_75
        del primals_80
        buf40 = aten.convolution(buf39, primals_5, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf40, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf41 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf42 = buf41; del buf41  # reuse
        buf45 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf43 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__11', 'in_out_ptr0', 'buf42', (buf42.sum()/buf42.nelement()).item(), buf42.amax().item(), buf42.amin().item())
        print('triton__11', 'in_ptr0', 'buf40', (buf40.sum()/buf40.nelement()).item(), buf40.amax().item(), buf40.amin().item())
        print('triton__11', 'in_ptr1', 'primals_82', (primals_82.sum()/primals_82.nelement()).item(), primals_82.amax().item(), primals_82.amin().item())
        print('triton__11', 'in_ptr2', 'primals_83', (primals_83.sum()/primals_83.nelement()).item(), primals_83.amax().item(), primals_83.amin().item())
        triton__11.run(buf42, buf40, primals_82, primals_83, buf45, buf43, buf44, buf46, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__11', 'in_out_ptr0', 'buf42', (buf42.sum()/buf42.nelement()).item(), buf42.amax().item(), buf42.amin().item())
        print('triton__11', 'out_ptr0', 'buf45', (buf45.sum()/buf45.nelement()).item(), buf45.amax().item(), buf45.amin().item())
        print('triton__11', 'out_ptr1', 'buf43', (buf43.sum()/buf43.nelement()).item(), buf43.amax().item(), buf43.amin().item())
        print('triton__11', 'out_ptr2', 'buf44', (buf44.sum()/buf44.nelement()).item(), buf44.amax().item(), buf44.amin().item())
        print('triton__11', 'out_ptr3', 'buf46', (buf46.sum()/buf46.nelement()).item(), buf46.amax().item(), buf46.amin().item())
        del primals_82
        del primals_83
        buf47 = empty_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda', dtype=torch.float32)
        print('triton__12', 'in_ptr0', 'buf40', (buf40.sum()/buf40.nelement()).item(), buf40.amax().item(), buf40.amin().item())
        print('triton__12', 'in_ptr1', 'buf42', (buf42.sum()/buf42.nelement()).item(), buf42.amax().item(), buf42.amin().item())
        print('triton__12', 'in_ptr2', 'buf43', (buf43.sum()/buf43.nelement()).item(), buf43.amax().item(), buf43.amin().item())
        print('triton__12', 'in_ptr3', 'primals_84', (primals_84.sum()/primals_84.nelement()).item(), primals_84.amax().item(), primals_84.amin().item())
        print('triton__12', 'in_ptr4', 'primals_85', (primals_85.sum()/primals_85.nelement()).item(), primals_85.amax().item(), primals_85.amin().item())
        triton__12.run(buf40, buf42, buf43, primals_84, primals_85, buf47, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__12', 'out_ptr0', 'buf47', (buf47.sum()/buf47.nelement()).item(), buf47.amax().item(), buf47.amin().item())
        del primals_85
        buf48 = aten.convolution(buf47, primals_6, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf48, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf49 = buf43; del buf43  # reuse
        buf50 = buf49; del buf49  # reuse
        buf53 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__11', 'in_out_ptr0', 'buf50', (buf50.sum()/buf50.nelement()).item(), buf50.amax().item(), buf50.amin().item())
        print('triton__11', 'in_ptr0', 'buf48', (buf48.sum()/buf48.nelement()).item(), buf48.amax().item(), buf48.amin().item())
        print('triton__11', 'in_ptr1', 'primals_87', (primals_87.sum()/primals_87.nelement()).item(), primals_87.amax().item(), primals_87.amin().item())
        print('triton__11', 'in_ptr2', 'primals_88', (primals_88.sum()/primals_88.nelement()).item(), primals_88.amax().item(), primals_88.amin().item())
        triton__11.run(buf50, buf48, primals_87, primals_88, buf53, buf51, buf52, buf54, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__11', 'in_out_ptr0', 'buf50', (buf50.sum()/buf50.nelement()).item(), buf50.amax().item(), buf50.amin().item())
        print('triton__11', 'out_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        print('triton__11', 'out_ptr1', 'buf51', (buf51.sum()/buf51.nelement()).item(), buf51.amax().item(), buf51.amin().item())
        print('triton__11', 'out_ptr2', 'buf52', (buf52.sum()/buf52.nelement()).item(), buf52.amax().item(), buf52.amin().item())
        print('triton__11', 'out_ptr3', 'buf54', (buf54.sum()/buf54.nelement()).item(), buf54.amax().item(), buf54.amin().item())
        del primals_87
        del primals_88
        buf55 = aten.convolution(buf39, primals_7, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf55, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf56 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf57 = buf56; del buf56  # reuse
        buf60 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__11', 'in_out_ptr0', 'buf57', (buf57.sum()/buf57.nelement()).item(), buf57.amax().item(), buf57.amin().item())
        print('triton__11', 'in_ptr0', 'buf55', (buf55.sum()/buf55.nelement()).item(), buf55.amax().item(), buf55.amin().item())
        print('triton__11', 'in_ptr1', 'primals_92', (primals_92.sum()/primals_92.nelement()).item(), primals_92.amax().item(), primals_92.amin().item())
        print('triton__11', 'in_ptr2', 'primals_93', (primals_93.sum()/primals_93.nelement()).item(), primals_93.amax().item(), primals_93.amin().item())
        triton__11.run(buf57, buf55, primals_92, primals_93, buf60, buf58, buf59, buf61, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__11', 'in_out_ptr0', 'buf57', (buf57.sum()/buf57.nelement()).item(), buf57.amax().item(), buf57.amin().item())
        print('triton__11', 'out_ptr0', 'buf60', (buf60.sum()/buf60.nelement()).item(), buf60.amax().item(), buf60.amin().item())
        print('triton__11', 'out_ptr1', 'buf58', (buf58.sum()/buf58.nelement()).item(), buf58.amax().item(), buf58.amin().item())
        print('triton__11', 'out_ptr2', 'buf59', (buf59.sum()/buf59.nelement()).item(), buf59.amax().item(), buf59.amin().item())
        print('triton__11', 'out_ptr3', 'buf61', (buf61.sum()/buf61.nelement()).item(), buf61.amax().item(), buf61.amin().item())
        del primals_92
        del primals_93
        buf62 = empty_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda', dtype=torch.float32)
        buf63 = buf62; del buf62  # reuse
        print('triton__13', 'in_out_ptr0', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        print('triton__13', 'in_ptr0', 'buf48', (buf48.sum()/buf48.nelement()).item(), buf48.amax().item(), buf48.amin().item())
        print('triton__13', 'in_ptr1', 'buf50', (buf50.sum()/buf50.nelement()).item(), buf50.amax().item(), buf50.amin().item())
        print('triton__13', 'in_ptr2', 'buf51', (buf51.sum()/buf51.nelement()).item(), buf51.amax().item(), buf51.amin().item())
        print('triton__13', 'in_ptr3', 'primals_89', (primals_89.sum()/primals_89.nelement()).item(), primals_89.amax().item(), primals_89.amin().item())
        print('triton__13', 'in_ptr4', 'primals_90', (primals_90.sum()/primals_90.nelement()).item(), primals_90.amax().item(), primals_90.amin().item())
        print('triton__13', 'in_ptr5', 'buf55', (buf55.sum()/buf55.nelement()).item(), buf55.amax().item(), buf55.amin().item())
        print('triton__13', 'in_ptr6', 'buf57', (buf57.sum()/buf57.nelement()).item(), buf57.amax().item(), buf57.amin().item())
        print('triton__13', 'in_ptr7', 'buf58', (buf58.sum()/buf58.nelement()).item(), buf58.amax().item(), buf58.amin().item())
        print('triton__13', 'in_ptr8', 'primals_94', (primals_94.sum()/primals_94.nelement()).item(), primals_94.amax().item(), primals_94.amin().item())
        print('triton__13', 'in_ptr9', 'primals_95', (primals_95.sum()/primals_95.nelement()).item(), primals_95.amax().item(), primals_95.amin().item())
        triton__13.run(buf63, buf48, buf50, buf51, primals_89, primals_90, buf55, buf57, buf58, primals_94, primals_95, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__13', 'in_out_ptr0', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        del primals_90
        del primals_95
        buf64 = aten.convolution(buf63, primals_8, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf64, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf65 = buf58; del buf58  # reuse
        buf66 = buf65; del buf65  # reuse
        buf69 = as_strided(buf51, (192, ), (1, )); del buf51  # reuse
        buf67 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__11', 'in_out_ptr0', 'buf66', (buf66.sum()/buf66.nelement()).item(), buf66.amax().item(), buf66.amin().item())
        print('triton__11', 'in_ptr0', 'buf64', (buf64.sum()/buf64.nelement()).item(), buf64.amax().item(), buf64.amin().item())
        print('triton__11', 'in_ptr1', 'primals_97', (primals_97.sum()/primals_97.nelement()).item(), primals_97.amax().item(), primals_97.amin().item())
        print('triton__11', 'in_ptr2', 'primals_98', (primals_98.sum()/primals_98.nelement()).item(), primals_98.amax().item(), primals_98.amin().item())
        triton__11.run(buf66, buf64, primals_97, primals_98, buf69, buf67, buf68, buf70, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__11', 'in_out_ptr0', 'buf66', (buf66.sum()/buf66.nelement()).item(), buf66.amax().item(), buf66.amin().item())
        print('triton__11', 'out_ptr0', 'buf69', (buf69.sum()/buf69.nelement()).item(), buf69.amax().item(), buf69.amin().item())
        print('triton__11', 'out_ptr1', 'buf67', (buf67.sum()/buf67.nelement()).item(), buf67.amax().item(), buf67.amin().item())
        print('triton__11', 'out_ptr2', 'buf68', (buf68.sum()/buf68.nelement()).item(), buf68.amax().item(), buf68.amin().item())
        print('triton__11', 'out_ptr3', 'buf70', (buf70.sum()/buf70.nelement()).item(), buf70.amax().item(), buf70.amin().item())
        del primals_97
        del primals_98
        buf71 = empty_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda', dtype=torch.float32)
        print('triton__12', 'in_ptr0', 'buf64', (buf64.sum()/buf64.nelement()).item(), buf64.amax().item(), buf64.amin().item())
        print('triton__12', 'in_ptr1', 'buf66', (buf66.sum()/buf66.nelement()).item(), buf66.amax().item(), buf66.amin().item())
        print('triton__12', 'in_ptr2', 'buf67', (buf67.sum()/buf67.nelement()).item(), buf67.amax().item(), buf67.amin().item())
        print('triton__12', 'in_ptr3', 'primals_99', (primals_99.sum()/primals_99.nelement()).item(), primals_99.amax().item(), primals_99.amin().item())
        print('triton__12', 'in_ptr4', 'primals_100', (primals_100.sum()/primals_100.nelement()).item(), primals_100.amax().item(), primals_100.amin().item())
        triton__12.run(buf64, buf66, buf67, primals_99, primals_100, buf71, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__12', 'out_ptr0', 'buf71', (buf71.sum()/buf71.nelement()).item(), buf71.amax().item(), buf71.amin().item())
        del primals_100
        buf72 = aten.convolution(buf71, primals_9, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf72, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf73 = buf67; del buf67  # reuse
        buf74 = buf73; del buf73  # reuse
        buf77 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__11', 'in_out_ptr0', 'buf74', (buf74.sum()/buf74.nelement()).item(), buf74.amax().item(), buf74.amin().item())
        print('triton__11', 'in_ptr0', 'buf72', (buf72.sum()/buf72.nelement()).item(), buf72.amax().item(), buf72.amin().item())
        print('triton__11', 'in_ptr1', 'primals_102', (primals_102.sum()/primals_102.nelement()).item(), primals_102.amax().item(), primals_102.amin().item())
        print('triton__11', 'in_ptr2', 'primals_103', (primals_103.sum()/primals_103.nelement()).item(), primals_103.amax().item(), primals_103.amin().item())
        triton__11.run(buf74, buf72, primals_102, primals_103, buf77, buf75, buf76, buf78, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__11', 'in_out_ptr0', 'buf74', (buf74.sum()/buf74.nelement()).item(), buf74.amax().item(), buf74.amin().item())
        print('triton__11', 'out_ptr0', 'buf77', (buf77.sum()/buf77.nelement()).item(), buf77.amax().item(), buf77.amin().item())
        print('triton__11', 'out_ptr1', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        print('triton__11', 'out_ptr2', 'buf76', (buf76.sum()/buf76.nelement()).item(), buf76.amax().item(), buf76.amin().item())
        print('triton__11', 'out_ptr3', 'buf78', (buf78.sum()/buf78.nelement()).item(), buf78.amax().item(), buf78.amin().item())
        del primals_102
        del primals_103
        buf79 = empty_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda', dtype=torch.float32)
        print('triton__14', 'in_ptr0', 'buf72', (buf72.sum()/buf72.nelement()).item(), buf72.amax().item(), buf72.amin().item())
        print('triton__14', 'in_ptr1', 'buf74', (buf74.sum()/buf74.nelement()).item(), buf74.amax().item(), buf74.amin().item())
        print('triton__14', 'in_ptr2', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        print('triton__14', 'in_ptr3', 'primals_104', (primals_104.sum()/primals_104.nelement()).item(), primals_104.amax().item(), primals_104.amin().item())
        print('triton__14', 'in_ptr4', 'primals_105', (primals_105.sum()/primals_105.nelement()).item(), primals_105.amax().item(), primals_105.amin().item())
        print('triton__14', 'in_ptr5', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        triton__14.run(buf72, buf74, buf75, primals_104, primals_105, buf63, buf79, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__14', 'out_ptr0', 'buf79', (buf79.sum()/buf79.nelement()).item(), buf79.amax().item(), buf79.amin().item())
        del buf75
        del primals_105
        buf80 = aten.convolution(buf79, primals_10, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf80, (8, 160, 32, 32), (163840, 1024, 32, 1))
        buf81 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf82 = buf81; del buf81  # reuse
        buf85 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf83 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf86 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__15', 'in_out_ptr0', 'buf82', (buf82.sum()/buf82.nelement()).item(), buf82.amax().item(), buf82.amin().item())
        print('triton__15', 'in_ptr0', 'buf80', (buf80.sum()/buf80.nelement()).item(), buf80.amax().item(), buf80.amin().item())
        print('triton__15', 'in_ptr1', 'primals_107', (primals_107.sum()/primals_107.nelement()).item(), primals_107.amax().item(), primals_107.amin().item())
        print('triton__15', 'in_ptr2', 'primals_108', (primals_108.sum()/primals_108.nelement()).item(), primals_108.amax().item(), primals_108.amin().item())
        triton__15.run(buf82, buf80, primals_107, primals_108, buf85, buf83, buf84, buf86, 160, 8192, grid=grid(160), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf82', (buf82.sum()/buf82.nelement()).item(), buf82.amax().item(), buf82.amin().item())
        print('triton__15', 'out_ptr0', 'buf85', (buf85.sum()/buf85.nelement()).item(), buf85.amax().item(), buf85.amin().item())
        print('triton__15', 'out_ptr1', 'buf83', (buf83.sum()/buf83.nelement()).item(), buf83.amax().item(), buf83.amin().item())
        print('triton__15', 'out_ptr2', 'buf84', (buf84.sum()/buf84.nelement()).item(), buf84.amax().item(), buf84.amin().item())
        print('triton__15', 'out_ptr3', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        del primals_107
        del primals_108
        buf87 = empty_strided((8, 160, 32, 32), (163840, 1024, 32, 1), device='cuda', dtype=torch.float32)
        print('triton__16', 'in_ptr0', 'buf80', (buf80.sum()/buf80.nelement()).item(), buf80.amax().item(), buf80.amin().item())
        print('triton__16', 'in_ptr1', 'buf82', (buf82.sum()/buf82.nelement()).item(), buf82.amax().item(), buf82.amin().item())
        print('triton__16', 'in_ptr2', 'buf83', (buf83.sum()/buf83.nelement()).item(), buf83.amax().item(), buf83.amin().item())
        print('triton__16', 'in_ptr3', 'primals_109', (primals_109.sum()/primals_109.nelement()).item(), primals_109.amax().item(), primals_109.amin().item())
        print('triton__16', 'in_ptr4', 'primals_110', (primals_110.sum()/primals_110.nelement()).item(), primals_110.amax().item(), primals_110.amin().item())
        triton__16.run(buf80, buf82, buf83, primals_109, primals_110, buf87, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__16', 'out_ptr0', 'buf87', (buf87.sum()/buf87.nelement()).item(), buf87.amax().item(), buf87.amin().item())
        del primals_110
        buf88 = aten.convolution(buf87, primals_11, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf88, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf89 = buf83; del buf83  # reuse
        buf90 = buf89; del buf89  # reuse
        buf93 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf92 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf90', (buf90.sum()/buf90.nelement()).item(), buf90.amax().item(), buf90.amin().item())
        print('triton__17', 'in_ptr0', 'buf88', (buf88.sum()/buf88.nelement()).item(), buf88.amax().item(), buf88.amin().item())
        print('triton__17', 'in_ptr1', 'primals_112', (primals_112.sum()/primals_112.nelement()).item(), primals_112.amax().item(), primals_112.amin().item())
        print('triton__17', 'in_ptr2', 'primals_113', (primals_113.sum()/primals_113.nelement()).item(), primals_113.amax().item(), primals_113.amin().item())
        triton__17.run(buf90, buf88, primals_112, primals_113, buf93, buf91, buf92, buf94, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf90', (buf90.sum()/buf90.nelement()).item(), buf90.amax().item(), buf90.amin().item())
        print('triton__17', 'out_ptr0', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__17', 'out_ptr1', 'buf91', (buf91.sum()/buf91.nelement()).item(), buf91.amax().item(), buf91.amin().item())
        print('triton__17', 'out_ptr2', 'buf92', (buf92.sum()/buf92.nelement()).item(), buf92.amax().item(), buf92.amin().item())
        print('triton__17', 'out_ptr3', 'buf94', (buf94.sum()/buf94.nelement()).item(), buf94.amax().item(), buf94.amin().item())
        del primals_112
        del primals_113
        buf95 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf88', (buf88.sum()/buf88.nelement()).item(), buf88.amax().item(), buf88.amin().item())
        print('triton__18', 'in_ptr1', 'buf90', (buf90.sum()/buf90.nelement()).item(), buf90.amax().item(), buf90.amin().item())
        print('triton__18', 'in_ptr2', 'buf91', (buf91.sum()/buf91.nelement()).item(), buf91.amax().item(), buf91.amin().item())
        print('triton__18', 'in_ptr3', 'primals_114', (primals_114.sum()/primals_114.nelement()).item(), primals_114.amax().item(), primals_114.amin().item())
        print('triton__18', 'in_ptr4', 'primals_115', (primals_115.sum()/primals_115.nelement()).item(), primals_115.amax().item(), primals_115.amin().item())
        triton__18.run(buf88, buf90, buf91, primals_114, primals_115, buf95, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf95', (buf95.sum()/buf95.nelement()).item(), buf95.amax().item(), buf95.amin().item())
        del primals_115
        buf96 = aten.convolution(buf95, primals_12, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf96, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf97 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf98 = buf97; del buf97  # reuse
        buf101 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf99 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf100 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__19', 'in_out_ptr0', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        print('triton__19', 'in_ptr0', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__19', 'in_ptr1', 'primals_117', (primals_117.sum()/primals_117.nelement()).item(), primals_117.amax().item(), primals_117.amin().item())
        print('triton__19', 'in_ptr2', 'primals_118', (primals_118.sum()/primals_118.nelement()).item(), primals_118.amax().item(), primals_118.amin().item())
        triton__19.run(buf98, buf96, primals_117, primals_118, buf101, buf99, buf100, buf102, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        print('triton__19', 'out_ptr0', 'buf101', (buf101.sum()/buf101.nelement()).item(), buf101.amax().item(), buf101.amin().item())
        print('triton__19', 'out_ptr1', 'buf99', (buf99.sum()/buf99.nelement()).item(), buf99.amax().item(), buf99.amin().item())
        print('triton__19', 'out_ptr2', 'buf100', (buf100.sum()/buf100.nelement()).item(), buf100.amax().item(), buf100.amin().item())
        print('triton__19', 'out_ptr3', 'buf102', (buf102.sum()/buf102.nelement()).item(), buf102.amax().item(), buf102.amin().item())
        del primals_117
        del primals_118
        buf103 = aten.convolution(buf79, primals_13, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf103, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf104 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf105 = buf104; del buf104  # reuse
        buf108 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf106 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf107 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__19', 'in_out_ptr0', 'buf105', (buf105.sum()/buf105.nelement()).item(), buf105.amax().item(), buf105.amin().item())
        print('triton__19', 'in_ptr0', 'buf103', (buf103.sum()/buf103.nelement()).item(), buf103.amax().item(), buf103.amin().item())
        print('triton__19', 'in_ptr1', 'primals_122', (primals_122.sum()/primals_122.nelement()).item(), primals_122.amax().item(), primals_122.amin().item())
        print('triton__19', 'in_ptr2', 'primals_123', (primals_123.sum()/primals_123.nelement()).item(), primals_123.amax().item(), primals_123.amin().item())
        triton__19.run(buf105, buf103, primals_122, primals_123, buf108, buf106, buf107, buf109, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf105', (buf105.sum()/buf105.nelement()).item(), buf105.amax().item(), buf105.amin().item())
        print('triton__19', 'out_ptr0', 'buf108', (buf108.sum()/buf108.nelement()).item(), buf108.amax().item(), buf108.amin().item())
        print('triton__19', 'out_ptr1', 'buf106', (buf106.sum()/buf106.nelement()).item(), buf106.amax().item(), buf106.amin().item())
        print('triton__19', 'out_ptr2', 'buf107', (buf107.sum()/buf107.nelement()).item(), buf107.amax().item(), buf107.amin().item())
        print('triton__19', 'out_ptr3', 'buf109', (buf109.sum()/buf109.nelement()).item(), buf109.amax().item(), buf109.amin().item())
        del primals_122
        del primals_123
        buf110 = empty_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda', dtype=torch.float32)
        buf111 = buf110; del buf110  # reuse
        print('triton__20', 'in_out_ptr0', 'buf111', (buf111.sum()/buf111.nelement()).item(), buf111.amax().item(), buf111.amin().item())
        print('triton__20', 'in_ptr0', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__20', 'in_ptr1', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        print('triton__20', 'in_ptr2', 'buf99', (buf99.sum()/buf99.nelement()).item(), buf99.amax().item(), buf99.amin().item())
        print('triton__20', 'in_ptr3', 'primals_119', (primals_119.sum()/primals_119.nelement()).item(), primals_119.amax().item(), primals_119.amin().item())
        print('triton__20', 'in_ptr4', 'primals_120', (primals_120.sum()/primals_120.nelement()).item(), primals_120.amax().item(), primals_120.amin().item())
        print('triton__20', 'in_ptr5', 'buf103', (buf103.sum()/buf103.nelement()).item(), buf103.amax().item(), buf103.amin().item())
        print('triton__20', 'in_ptr6', 'buf105', (buf105.sum()/buf105.nelement()).item(), buf105.amax().item(), buf105.amin().item())
        print('triton__20', 'in_ptr7', 'buf106', (buf106.sum()/buf106.nelement()).item(), buf106.amax().item(), buf106.amin().item())
        print('triton__20', 'in_ptr8', 'primals_124', (primals_124.sum()/primals_124.nelement()).item(), primals_124.amax().item(), primals_124.amin().item())
        print('triton__20', 'in_ptr9', 'primals_125', (primals_125.sum()/primals_125.nelement()).item(), primals_125.amax().item(), primals_125.amin().item())
        triton__20.run(buf111, buf96, buf98, buf99, primals_119, primals_120, buf103, buf105, buf106, primals_124, primals_125, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__20', 'in_out_ptr0', 'buf111', (buf111.sum()/buf111.nelement()).item(), buf111.amax().item(), buf111.amin().item())
        del primals_120
        del primals_125
        buf112 = aten.convolution(buf111, primals_14, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf112, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf113 = buf91; del buf91  # reuse
        buf114 = buf113; del buf113  # reuse
        buf117 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf115 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf116 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf114', (buf114.sum()/buf114.nelement()).item(), buf114.amax().item(), buf114.amin().item())
        print('triton__17', 'in_ptr0', 'buf112', (buf112.sum()/buf112.nelement()).item(), buf112.amax().item(), buf112.amin().item())
        print('triton__17', 'in_ptr1', 'primals_127', (primals_127.sum()/primals_127.nelement()).item(), primals_127.amax().item(), primals_127.amin().item())
        print('triton__17', 'in_ptr2', 'primals_128', (primals_128.sum()/primals_128.nelement()).item(), primals_128.amax().item(), primals_128.amin().item())
        triton__17.run(buf114, buf112, primals_127, primals_128, buf117, buf115, buf116, buf118, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf114', (buf114.sum()/buf114.nelement()).item(), buf114.amax().item(), buf114.amin().item())
        print('triton__17', 'out_ptr0', 'buf117', (buf117.sum()/buf117.nelement()).item(), buf117.amax().item(), buf117.amin().item())
        print('triton__17', 'out_ptr1', 'buf115', (buf115.sum()/buf115.nelement()).item(), buf115.amax().item(), buf115.amin().item())
        print('triton__17', 'out_ptr2', 'buf116', (buf116.sum()/buf116.nelement()).item(), buf116.amax().item(), buf116.amin().item())
        print('triton__17', 'out_ptr3', 'buf118', (buf118.sum()/buf118.nelement()).item(), buf118.amax().item(), buf118.amin().item())
        del primals_127
        del primals_128
        buf119 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf112', (buf112.sum()/buf112.nelement()).item(), buf112.amax().item(), buf112.amin().item())
        print('triton__18', 'in_ptr1', 'buf114', (buf114.sum()/buf114.nelement()).item(), buf114.amax().item(), buf114.amin().item())
        print('triton__18', 'in_ptr2', 'buf115', (buf115.sum()/buf115.nelement()).item(), buf115.amax().item(), buf115.amin().item())
        print('triton__18', 'in_ptr3', 'primals_129', (primals_129.sum()/primals_129.nelement()).item(), primals_129.amax().item(), primals_129.amin().item())
        print('triton__18', 'in_ptr4', 'primals_130', (primals_130.sum()/primals_130.nelement()).item(), primals_130.amax().item(), primals_130.amin().item())
        triton__18.run(buf112, buf114, buf115, primals_129, primals_130, buf119, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf119', (buf119.sum()/buf119.nelement()).item(), buf119.amax().item(), buf119.amin().item())
        del primals_130
        buf120 = aten.convolution(buf119, primals_15, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf120, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf121 = buf115; del buf115  # reuse
        buf122 = buf121; del buf121  # reuse
        buf125 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf123 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf124 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf126 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf122', (buf122.sum()/buf122.nelement()).item(), buf122.amax().item(), buf122.amin().item())
        print('triton__17', 'in_ptr0', 'buf120', (buf120.sum()/buf120.nelement()).item(), buf120.amax().item(), buf120.amin().item())
        print('triton__17', 'in_ptr1', 'primals_132', (primals_132.sum()/primals_132.nelement()).item(), primals_132.amax().item(), primals_132.amin().item())
        print('triton__17', 'in_ptr2', 'primals_133', (primals_133.sum()/primals_133.nelement()).item(), primals_133.amax().item(), primals_133.amin().item())
        triton__17.run(buf122, buf120, primals_132, primals_133, buf125, buf123, buf124, buf126, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf122', (buf122.sum()/buf122.nelement()).item(), buf122.amax().item(), buf122.amin().item())
        print('triton__17', 'out_ptr0', 'buf125', (buf125.sum()/buf125.nelement()).item(), buf125.amax().item(), buf125.amin().item())
        print('triton__17', 'out_ptr1', 'buf123', (buf123.sum()/buf123.nelement()).item(), buf123.amax().item(), buf123.amin().item())
        print('triton__17', 'out_ptr2', 'buf124', (buf124.sum()/buf124.nelement()).item(), buf124.amax().item(), buf124.amin().item())
        print('triton__17', 'out_ptr3', 'buf126', (buf126.sum()/buf126.nelement()).item(), buf126.amax().item(), buf126.amin().item())
        del primals_132
        del primals_133
        buf127 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf120', (buf120.sum()/buf120.nelement()).item(), buf120.amax().item(), buf120.amin().item())
        print('triton__18', 'in_ptr1', 'buf122', (buf122.sum()/buf122.nelement()).item(), buf122.amax().item(), buf122.amin().item())
        print('triton__18', 'in_ptr2', 'buf123', (buf123.sum()/buf123.nelement()).item(), buf123.amax().item(), buf123.amin().item())
        print('triton__18', 'in_ptr3', 'primals_134', (primals_134.sum()/primals_134.nelement()).item(), primals_134.amax().item(), primals_134.amin().item())
        print('triton__18', 'in_ptr4', 'primals_135', (primals_135.sum()/primals_135.nelement()).item(), primals_135.amax().item(), primals_135.amin().item())
        triton__18.run(buf120, buf122, buf123, primals_134, primals_135, buf127, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf127', (buf127.sum()/buf127.nelement()).item(), buf127.amax().item(), buf127.amin().item())
        del primals_135
        buf128 = aten.convolution(buf127, primals_16, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf128, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf129 = buf99; del buf99  # reuse
        buf130 = buf129; del buf129  # reuse
        buf133 = as_strided(buf106, (640, ), (1, )); del buf106  # reuse
        buf131 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__19', 'in_out_ptr0', 'buf130', (buf130.sum()/buf130.nelement()).item(), buf130.amax().item(), buf130.amin().item())
        print('triton__19', 'in_ptr0', 'buf128', (buf128.sum()/buf128.nelement()).item(), buf128.amax().item(), buf128.amin().item())
        print('triton__19', 'in_ptr1', 'primals_137', (primals_137.sum()/primals_137.nelement()).item(), primals_137.amax().item(), primals_137.amin().item())
        print('triton__19', 'in_ptr2', 'primals_138', (primals_138.sum()/primals_138.nelement()).item(), primals_138.amax().item(), primals_138.amin().item())
        triton__19.run(buf130, buf128, primals_137, primals_138, buf133, buf131, buf132, buf134, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf130', (buf130.sum()/buf130.nelement()).item(), buf130.amax().item(), buf130.amin().item())
        print('triton__19', 'out_ptr0', 'buf133', (buf133.sum()/buf133.nelement()).item(), buf133.amax().item(), buf133.amin().item())
        print('triton__19', 'out_ptr1', 'buf131', (buf131.sum()/buf131.nelement()).item(), buf131.amax().item(), buf131.amin().item())
        print('triton__19', 'out_ptr2', 'buf132', (buf132.sum()/buf132.nelement()).item(), buf132.amax().item(), buf132.amin().item())
        print('triton__19', 'out_ptr3', 'buf134', (buf134.sum()/buf134.nelement()).item(), buf134.amax().item(), buf134.amin().item())
        del primals_137
        del primals_138
        buf135 = empty_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__21', 'in_ptr0', 'buf128', (buf128.sum()/buf128.nelement()).item(), buf128.amax().item(), buf128.amin().item())
        print('triton__21', 'in_ptr1', 'buf130', (buf130.sum()/buf130.nelement()).item(), buf130.amax().item(), buf130.amin().item())
        print('triton__21', 'in_ptr2', 'buf131', (buf131.sum()/buf131.nelement()).item(), buf131.amax().item(), buf131.amin().item())
        print('triton__21', 'in_ptr3', 'primals_139', (primals_139.sum()/primals_139.nelement()).item(), primals_139.amax().item(), primals_139.amin().item())
        print('triton__21', 'in_ptr4', 'primals_140', (primals_140.sum()/primals_140.nelement()).item(), primals_140.amax().item(), primals_140.amin().item())
        print('triton__21', 'in_ptr5', 'buf111', (buf111.sum()/buf111.nelement()).item(), buf111.amax().item(), buf111.amin().item())
        triton__21.run(buf128, buf130, buf131, primals_139, primals_140, buf111, buf135, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf135', (buf135.sum()/buf135.nelement()).item(), buf135.amax().item(), buf135.amin().item())
        del primals_140
        buf136 = aten.convolution(buf135, primals_17, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf136, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf137 = buf123; del buf123  # reuse
        buf138 = buf137; del buf137  # reuse
        buf141 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf139 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf140 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf138', (buf138.sum()/buf138.nelement()).item(), buf138.amax().item(), buf138.amin().item())
        print('triton__17', 'in_ptr0', 'buf136', (buf136.sum()/buf136.nelement()).item(), buf136.amax().item(), buf136.amin().item())
        print('triton__17', 'in_ptr1', 'primals_142', (primals_142.sum()/primals_142.nelement()).item(), primals_142.amax().item(), primals_142.amin().item())
        print('triton__17', 'in_ptr2', 'primals_143', (primals_143.sum()/primals_143.nelement()).item(), primals_143.amax().item(), primals_143.amin().item())
        triton__17.run(buf138, buf136, primals_142, primals_143, buf141, buf139, buf140, buf142, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf138', (buf138.sum()/buf138.nelement()).item(), buf138.amax().item(), buf138.amin().item())
        print('triton__17', 'out_ptr0', 'buf141', (buf141.sum()/buf141.nelement()).item(), buf141.amax().item(), buf141.amin().item())
        print('triton__17', 'out_ptr1', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        print('triton__17', 'out_ptr2', 'buf140', (buf140.sum()/buf140.nelement()).item(), buf140.amax().item(), buf140.amin().item())
        print('triton__17', 'out_ptr3', 'buf142', (buf142.sum()/buf142.nelement()).item(), buf142.amax().item(), buf142.amin().item())
        del primals_142
        del primals_143
        buf143 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf136', (buf136.sum()/buf136.nelement()).item(), buf136.amax().item(), buf136.amin().item())
        print('triton__18', 'in_ptr1', 'buf138', (buf138.sum()/buf138.nelement()).item(), buf138.amax().item(), buf138.amin().item())
        print('triton__18', 'in_ptr2', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        print('triton__18', 'in_ptr3', 'primals_144', (primals_144.sum()/primals_144.nelement()).item(), primals_144.amax().item(), primals_144.amin().item())
        print('triton__18', 'in_ptr4', 'primals_145', (primals_145.sum()/primals_145.nelement()).item(), primals_145.amax().item(), primals_145.amin().item())
        triton__18.run(buf136, buf138, buf139, primals_144, primals_145, buf143, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf143', (buf143.sum()/buf143.nelement()).item(), buf143.amax().item(), buf143.amin().item())
        del primals_145
        buf144 = aten.convolution(buf143, primals_18, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf144, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf145 = buf139; del buf139  # reuse
        buf146 = buf145; del buf145  # reuse
        buf149 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf147 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf148 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf150 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf146', (buf146.sum()/buf146.nelement()).item(), buf146.amax().item(), buf146.amin().item())
        print('triton__17', 'in_ptr0', 'buf144', (buf144.sum()/buf144.nelement()).item(), buf144.amax().item(), buf144.amin().item())
        print('triton__17', 'in_ptr1', 'primals_147', (primals_147.sum()/primals_147.nelement()).item(), primals_147.amax().item(), primals_147.amin().item())
        print('triton__17', 'in_ptr2', 'primals_148', (primals_148.sum()/primals_148.nelement()).item(), primals_148.amax().item(), primals_148.amin().item())
        triton__17.run(buf146, buf144, primals_147, primals_148, buf149, buf147, buf148, buf150, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf146', (buf146.sum()/buf146.nelement()).item(), buf146.amax().item(), buf146.amin().item())
        print('triton__17', 'out_ptr0', 'buf149', (buf149.sum()/buf149.nelement()).item(), buf149.amax().item(), buf149.amin().item())
        print('triton__17', 'out_ptr1', 'buf147', (buf147.sum()/buf147.nelement()).item(), buf147.amax().item(), buf147.amin().item())
        print('triton__17', 'out_ptr2', 'buf148', (buf148.sum()/buf148.nelement()).item(), buf148.amax().item(), buf148.amin().item())
        print('triton__17', 'out_ptr3', 'buf150', (buf150.sum()/buf150.nelement()).item(), buf150.amax().item(), buf150.amin().item())
        del primals_147
        del primals_148
        buf151 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf144', (buf144.sum()/buf144.nelement()).item(), buf144.amax().item(), buf144.amin().item())
        print('triton__18', 'in_ptr1', 'buf146', (buf146.sum()/buf146.nelement()).item(), buf146.amax().item(), buf146.amin().item())
        print('triton__18', 'in_ptr2', 'buf147', (buf147.sum()/buf147.nelement()).item(), buf147.amax().item(), buf147.amin().item())
        print('triton__18', 'in_ptr3', 'primals_149', (primals_149.sum()/primals_149.nelement()).item(), primals_149.amax().item(), primals_149.amin().item())
        print('triton__18', 'in_ptr4', 'primals_150', (primals_150.sum()/primals_150.nelement()).item(), primals_150.amax().item(), primals_150.amin().item())
        triton__18.run(buf144, buf146, buf147, primals_149, primals_150, buf151, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf151', (buf151.sum()/buf151.nelement()).item(), buf151.amax().item(), buf151.amin().item())
        del primals_150
        buf152 = aten.convolution(buf151, primals_19, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf152, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf153 = buf131; del buf131  # reuse
        buf154 = buf153; del buf153  # reuse
        buf157 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf155 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf156 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf158 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__19', 'in_out_ptr0', 'buf154', (buf154.sum()/buf154.nelement()).item(), buf154.amax().item(), buf154.amin().item())
        print('triton__19', 'in_ptr0', 'buf152', (buf152.sum()/buf152.nelement()).item(), buf152.amax().item(), buf152.amin().item())
        print('triton__19', 'in_ptr1', 'primals_152', (primals_152.sum()/primals_152.nelement()).item(), primals_152.amax().item(), primals_152.amin().item())
        print('triton__19', 'in_ptr2', 'primals_153', (primals_153.sum()/primals_153.nelement()).item(), primals_153.amax().item(), primals_153.amin().item())
        triton__19.run(buf154, buf152, primals_152, primals_153, buf157, buf155, buf156, buf158, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf154', (buf154.sum()/buf154.nelement()).item(), buf154.amax().item(), buf154.amin().item())
        print('triton__19', 'out_ptr0', 'buf157', (buf157.sum()/buf157.nelement()).item(), buf157.amax().item(), buf157.amin().item())
        print('triton__19', 'out_ptr1', 'buf155', (buf155.sum()/buf155.nelement()).item(), buf155.amax().item(), buf155.amin().item())
        print('triton__19', 'out_ptr2', 'buf156', (buf156.sum()/buf156.nelement()).item(), buf156.amax().item(), buf156.amin().item())
        print('triton__19', 'out_ptr3', 'buf158', (buf158.sum()/buf158.nelement()).item(), buf158.amax().item(), buf158.amin().item())
        del primals_152
        del primals_153
        buf159 = empty_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__21', 'in_ptr0', 'buf152', (buf152.sum()/buf152.nelement()).item(), buf152.amax().item(), buf152.amin().item())
        print('triton__21', 'in_ptr1', 'buf154', (buf154.sum()/buf154.nelement()).item(), buf154.amax().item(), buf154.amin().item())
        print('triton__21', 'in_ptr2', 'buf155', (buf155.sum()/buf155.nelement()).item(), buf155.amax().item(), buf155.amin().item())
        print('triton__21', 'in_ptr3', 'primals_154', (primals_154.sum()/primals_154.nelement()).item(), primals_154.amax().item(), primals_154.amin().item())
        print('triton__21', 'in_ptr4', 'primals_155', (primals_155.sum()/primals_155.nelement()).item(), primals_155.amax().item(), primals_155.amin().item())
        print('triton__21', 'in_ptr5', 'buf135', (buf135.sum()/buf135.nelement()).item(), buf135.amax().item(), buf135.amin().item())
        triton__21.run(buf152, buf154, buf155, primals_154, primals_155, buf135, buf159, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf159', (buf159.sum()/buf159.nelement()).item(), buf159.amax().item(), buf159.amin().item())
        del primals_155
        buf160 = aten.convolution(buf159, primals_20, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf160, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf161 = buf147; del buf147  # reuse
        buf162 = buf161; del buf161  # reuse
        buf165 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf163 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf166 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf162', (buf162.sum()/buf162.nelement()).item(), buf162.amax().item(), buf162.amin().item())
        print('triton__17', 'in_ptr0', 'buf160', (buf160.sum()/buf160.nelement()).item(), buf160.amax().item(), buf160.amin().item())
        print('triton__17', 'in_ptr1', 'primals_157', (primals_157.sum()/primals_157.nelement()).item(), primals_157.amax().item(), primals_157.amin().item())
        print('triton__17', 'in_ptr2', 'primals_158', (primals_158.sum()/primals_158.nelement()).item(), primals_158.amax().item(), primals_158.amin().item())
        triton__17.run(buf162, buf160, primals_157, primals_158, buf165, buf163, buf164, buf166, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf162', (buf162.sum()/buf162.nelement()).item(), buf162.amax().item(), buf162.amin().item())
        print('triton__17', 'out_ptr0', 'buf165', (buf165.sum()/buf165.nelement()).item(), buf165.amax().item(), buf165.amin().item())
        print('triton__17', 'out_ptr1', 'buf163', (buf163.sum()/buf163.nelement()).item(), buf163.amax().item(), buf163.amin().item())
        print('triton__17', 'out_ptr2', 'buf164', (buf164.sum()/buf164.nelement()).item(), buf164.amax().item(), buf164.amin().item())
        print('triton__17', 'out_ptr3', 'buf166', (buf166.sum()/buf166.nelement()).item(), buf166.amax().item(), buf166.amin().item())
        del primals_157
        del primals_158
        buf167 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf160', (buf160.sum()/buf160.nelement()).item(), buf160.amax().item(), buf160.amin().item())
        print('triton__18', 'in_ptr1', 'buf162', (buf162.sum()/buf162.nelement()).item(), buf162.amax().item(), buf162.amin().item())
        print('triton__18', 'in_ptr2', 'buf163', (buf163.sum()/buf163.nelement()).item(), buf163.amax().item(), buf163.amin().item())
        print('triton__18', 'in_ptr3', 'primals_159', (primals_159.sum()/primals_159.nelement()).item(), primals_159.amax().item(), primals_159.amin().item())
        print('triton__18', 'in_ptr4', 'primals_160', (primals_160.sum()/primals_160.nelement()).item(), primals_160.amax().item(), primals_160.amin().item())
        triton__18.run(buf160, buf162, buf163, primals_159, primals_160, buf167, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf167', (buf167.sum()/buf167.nelement()).item(), buf167.amax().item(), buf167.amin().item())
        del primals_160
        buf168 = aten.convolution(buf167, primals_21, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf168, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf169 = buf163; del buf163  # reuse
        buf170 = buf169; del buf169  # reuse
        buf173 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf171 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf172 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf174 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf170', (buf170.sum()/buf170.nelement()).item(), buf170.amax().item(), buf170.amin().item())
        print('triton__17', 'in_ptr0', 'buf168', (buf168.sum()/buf168.nelement()).item(), buf168.amax().item(), buf168.amin().item())
        print('triton__17', 'in_ptr1', 'primals_162', (primals_162.sum()/primals_162.nelement()).item(), primals_162.amax().item(), primals_162.amin().item())
        print('triton__17', 'in_ptr2', 'primals_163', (primals_163.sum()/primals_163.nelement()).item(), primals_163.amax().item(), primals_163.amin().item())
        triton__17.run(buf170, buf168, primals_162, primals_163, buf173, buf171, buf172, buf174, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf170', (buf170.sum()/buf170.nelement()).item(), buf170.amax().item(), buf170.amin().item())
        print('triton__17', 'out_ptr0', 'buf173', (buf173.sum()/buf173.nelement()).item(), buf173.amax().item(), buf173.amin().item())
        print('triton__17', 'out_ptr1', 'buf171', (buf171.sum()/buf171.nelement()).item(), buf171.amax().item(), buf171.amin().item())
        print('triton__17', 'out_ptr2', 'buf172', (buf172.sum()/buf172.nelement()).item(), buf172.amax().item(), buf172.amin().item())
        print('triton__17', 'out_ptr3', 'buf174', (buf174.sum()/buf174.nelement()).item(), buf174.amax().item(), buf174.amin().item())
        del primals_162
        del primals_163
        buf175 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf168', (buf168.sum()/buf168.nelement()).item(), buf168.amax().item(), buf168.amin().item())
        print('triton__18', 'in_ptr1', 'buf170', (buf170.sum()/buf170.nelement()).item(), buf170.amax().item(), buf170.amin().item())
        print('triton__18', 'in_ptr2', 'buf171', (buf171.sum()/buf171.nelement()).item(), buf171.amax().item(), buf171.amin().item())
        print('triton__18', 'in_ptr3', 'primals_164', (primals_164.sum()/primals_164.nelement()).item(), primals_164.amax().item(), primals_164.amin().item())
        print('triton__18', 'in_ptr4', 'primals_165', (primals_165.sum()/primals_165.nelement()).item(), primals_165.amax().item(), primals_165.amin().item())
        triton__18.run(buf168, buf170, buf171, primals_164, primals_165, buf175, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf175', (buf175.sum()/buf175.nelement()).item(), buf175.amax().item(), buf175.amin().item())
        del primals_165
        buf176 = aten.convolution(buf175, primals_22, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf176, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf177 = buf155; del buf155  # reuse
        buf178 = buf177; del buf177  # reuse
        buf181 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf179 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf182 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__19', 'in_out_ptr0', 'buf178', (buf178.sum()/buf178.nelement()).item(), buf178.amax().item(), buf178.amin().item())
        print('triton__19', 'in_ptr0', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__19', 'in_ptr1', 'primals_167', (primals_167.sum()/primals_167.nelement()).item(), primals_167.amax().item(), primals_167.amin().item())
        print('triton__19', 'in_ptr2', 'primals_168', (primals_168.sum()/primals_168.nelement()).item(), primals_168.amax().item(), primals_168.amin().item())
        triton__19.run(buf178, buf176, primals_167, primals_168, buf181, buf179, buf180, buf182, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf178', (buf178.sum()/buf178.nelement()).item(), buf178.amax().item(), buf178.amin().item())
        print('triton__19', 'out_ptr0', 'buf181', (buf181.sum()/buf181.nelement()).item(), buf181.amax().item(), buf181.amin().item())
        print('triton__19', 'out_ptr1', 'buf179', (buf179.sum()/buf179.nelement()).item(), buf179.amax().item(), buf179.amin().item())
        print('triton__19', 'out_ptr2', 'buf180', (buf180.sum()/buf180.nelement()).item(), buf180.amax().item(), buf180.amin().item())
        print('triton__19', 'out_ptr3', 'buf182', (buf182.sum()/buf182.nelement()).item(), buf182.amax().item(), buf182.amin().item())
        del primals_167
        del primals_168
        buf183 = empty_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__21', 'in_ptr0', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__21', 'in_ptr1', 'buf178', (buf178.sum()/buf178.nelement()).item(), buf178.amax().item(), buf178.amin().item())
        print('triton__21', 'in_ptr2', 'buf179', (buf179.sum()/buf179.nelement()).item(), buf179.amax().item(), buf179.amin().item())
        print('triton__21', 'in_ptr3', 'primals_169', (primals_169.sum()/primals_169.nelement()).item(), primals_169.amax().item(), primals_169.amin().item())
        print('triton__21', 'in_ptr4', 'primals_170', (primals_170.sum()/primals_170.nelement()).item(), primals_170.amax().item(), primals_170.amin().item())
        print('triton__21', 'in_ptr5', 'buf159', (buf159.sum()/buf159.nelement()).item(), buf159.amax().item(), buf159.amin().item())
        triton__21.run(buf176, buf178, buf179, primals_169, primals_170, buf159, buf183, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        del primals_170
        buf184 = aten.convolution(buf183, primals_23, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf184, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf185 = buf171; del buf171  # reuse
        buf186 = buf185; del buf185  # reuse
        buf189 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf187 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf190 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf186', (buf186.sum()/buf186.nelement()).item(), buf186.amax().item(), buf186.amin().item())
        print('triton__17', 'in_ptr0', 'buf184', (buf184.sum()/buf184.nelement()).item(), buf184.amax().item(), buf184.amin().item())
        print('triton__17', 'in_ptr1', 'primals_172', (primals_172.sum()/primals_172.nelement()).item(), primals_172.amax().item(), primals_172.amin().item())
        print('triton__17', 'in_ptr2', 'primals_173', (primals_173.sum()/primals_173.nelement()).item(), primals_173.amax().item(), primals_173.amin().item())
        triton__17.run(buf186, buf184, primals_172, primals_173, buf189, buf187, buf188, buf190, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf186', (buf186.sum()/buf186.nelement()).item(), buf186.amax().item(), buf186.amin().item())
        print('triton__17', 'out_ptr0', 'buf189', (buf189.sum()/buf189.nelement()).item(), buf189.amax().item(), buf189.amin().item())
        print('triton__17', 'out_ptr1', 'buf187', (buf187.sum()/buf187.nelement()).item(), buf187.amax().item(), buf187.amin().item())
        print('triton__17', 'out_ptr2', 'buf188', (buf188.sum()/buf188.nelement()).item(), buf188.amax().item(), buf188.amin().item())
        print('triton__17', 'out_ptr3', 'buf190', (buf190.sum()/buf190.nelement()).item(), buf190.amax().item(), buf190.amin().item())
        del primals_172
        del primals_173
        buf191 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf184', (buf184.sum()/buf184.nelement()).item(), buf184.amax().item(), buf184.amin().item())
        print('triton__18', 'in_ptr1', 'buf186', (buf186.sum()/buf186.nelement()).item(), buf186.amax().item(), buf186.amin().item())
        print('triton__18', 'in_ptr2', 'buf187', (buf187.sum()/buf187.nelement()).item(), buf187.amax().item(), buf187.amin().item())
        print('triton__18', 'in_ptr3', 'primals_174', (primals_174.sum()/primals_174.nelement()).item(), primals_174.amax().item(), primals_174.amin().item())
        print('triton__18', 'in_ptr4', 'primals_175', (primals_175.sum()/primals_175.nelement()).item(), primals_175.amax().item(), primals_175.amin().item())
        triton__18.run(buf184, buf186, buf187, primals_174, primals_175, buf191, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf191', (buf191.sum()/buf191.nelement()).item(), buf191.amax().item(), buf191.amin().item())
        del primals_175
        buf192 = aten.convolution(buf191, primals_24, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf192, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf193 = buf187; del buf187  # reuse
        buf194 = buf193; del buf193  # reuse
        buf197 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf194', (buf194.sum()/buf194.nelement()).item(), buf194.amax().item(), buf194.amin().item())
        print('triton__17', 'in_ptr0', 'buf192', (buf192.sum()/buf192.nelement()).item(), buf192.amax().item(), buf192.amin().item())
        print('triton__17', 'in_ptr1', 'primals_177', (primals_177.sum()/primals_177.nelement()).item(), primals_177.amax().item(), primals_177.amin().item())
        print('triton__17', 'in_ptr2', 'primals_178', (primals_178.sum()/primals_178.nelement()).item(), primals_178.amax().item(), primals_178.amin().item())
        triton__17.run(buf194, buf192, primals_177, primals_178, buf197, buf195, buf196, buf198, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf194', (buf194.sum()/buf194.nelement()).item(), buf194.amax().item(), buf194.amin().item())
        print('triton__17', 'out_ptr0', 'buf197', (buf197.sum()/buf197.nelement()).item(), buf197.amax().item(), buf197.amin().item())
        print('triton__17', 'out_ptr1', 'buf195', (buf195.sum()/buf195.nelement()).item(), buf195.amax().item(), buf195.amin().item())
        print('triton__17', 'out_ptr2', 'buf196', (buf196.sum()/buf196.nelement()).item(), buf196.amax().item(), buf196.amin().item())
        print('triton__17', 'out_ptr3', 'buf198', (buf198.sum()/buf198.nelement()).item(), buf198.amax().item(), buf198.amin().item())
        del primals_177
        del primals_178
        buf199 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf192', (buf192.sum()/buf192.nelement()).item(), buf192.amax().item(), buf192.amin().item())
        print('triton__18', 'in_ptr1', 'buf194', (buf194.sum()/buf194.nelement()).item(), buf194.amax().item(), buf194.amin().item())
        print('triton__18', 'in_ptr2', 'buf195', (buf195.sum()/buf195.nelement()).item(), buf195.amax().item(), buf195.amin().item())
        print('triton__18', 'in_ptr3', 'primals_179', (primals_179.sum()/primals_179.nelement()).item(), primals_179.amax().item(), primals_179.amin().item())
        print('triton__18', 'in_ptr4', 'primals_180', (primals_180.sum()/primals_180.nelement()).item(), primals_180.amax().item(), primals_180.amin().item())
        triton__18.run(buf192, buf194, buf195, primals_179, primals_180, buf199, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf199', (buf199.sum()/buf199.nelement()).item(), buf199.amax().item(), buf199.amin().item())
        del primals_180
        buf200 = aten.convolution(buf199, primals_25, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf200, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf201 = buf179; del buf179  # reuse
        buf202 = buf201; del buf201  # reuse
        buf205 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf203 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf204 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf206 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__19', 'in_out_ptr0', 'buf202', (buf202.sum()/buf202.nelement()).item(), buf202.amax().item(), buf202.amin().item())
        print('triton__19', 'in_ptr0', 'buf200', (buf200.sum()/buf200.nelement()).item(), buf200.amax().item(), buf200.amin().item())
        print('triton__19', 'in_ptr1', 'primals_182', (primals_182.sum()/primals_182.nelement()).item(), primals_182.amax().item(), primals_182.amin().item())
        print('triton__19', 'in_ptr2', 'primals_183', (primals_183.sum()/primals_183.nelement()).item(), primals_183.amax().item(), primals_183.amin().item())
        triton__19.run(buf202, buf200, primals_182, primals_183, buf205, buf203, buf204, buf206, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf202', (buf202.sum()/buf202.nelement()).item(), buf202.amax().item(), buf202.amin().item())
        print('triton__19', 'out_ptr0', 'buf205', (buf205.sum()/buf205.nelement()).item(), buf205.amax().item(), buf205.amin().item())
        print('triton__19', 'out_ptr1', 'buf203', (buf203.sum()/buf203.nelement()).item(), buf203.amax().item(), buf203.amin().item())
        print('triton__19', 'out_ptr2', 'buf204', (buf204.sum()/buf204.nelement()).item(), buf204.amax().item(), buf204.amin().item())
        print('triton__19', 'out_ptr3', 'buf206', (buf206.sum()/buf206.nelement()).item(), buf206.amax().item(), buf206.amin().item())
        del primals_182
        del primals_183
        buf207 = empty_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__21', 'in_ptr0', 'buf200', (buf200.sum()/buf200.nelement()).item(), buf200.amax().item(), buf200.amin().item())
        print('triton__21', 'in_ptr1', 'buf202', (buf202.sum()/buf202.nelement()).item(), buf202.amax().item(), buf202.amin().item())
        print('triton__21', 'in_ptr2', 'buf203', (buf203.sum()/buf203.nelement()).item(), buf203.amax().item(), buf203.amin().item())
        print('triton__21', 'in_ptr3', 'primals_184', (primals_184.sum()/primals_184.nelement()).item(), primals_184.amax().item(), primals_184.amin().item())
        print('triton__21', 'in_ptr4', 'primals_185', (primals_185.sum()/primals_185.nelement()).item(), primals_185.amax().item(), primals_185.amin().item())
        print('triton__21', 'in_ptr5', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        triton__21.run(buf200, buf202, buf203, primals_184, primals_185, buf183, buf207, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf207', (buf207.sum()/buf207.nelement()).item(), buf207.amax().item(), buf207.amin().item())
        del primals_185
        buf208 = aten.convolution(buf207, primals_26, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf208, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf209 = buf195; del buf195  # reuse
        buf210 = buf209; del buf209  # reuse
        buf213 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf211 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf212 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf214 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        print('triton__17', 'in_ptr0', 'buf208', (buf208.sum()/buf208.nelement()).item(), buf208.amax().item(), buf208.amin().item())
        print('triton__17', 'in_ptr1', 'primals_187', (primals_187.sum()/primals_187.nelement()).item(), primals_187.amax().item(), primals_187.amin().item())
        print('triton__17', 'in_ptr2', 'primals_188', (primals_188.sum()/primals_188.nelement()).item(), primals_188.amax().item(), primals_188.amin().item())
        triton__17.run(buf210, buf208, primals_187, primals_188, buf213, buf211, buf212, buf214, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        print('triton__17', 'out_ptr0', 'buf213', (buf213.sum()/buf213.nelement()).item(), buf213.amax().item(), buf213.amin().item())
        print('triton__17', 'out_ptr1', 'buf211', (buf211.sum()/buf211.nelement()).item(), buf211.amax().item(), buf211.amin().item())
        print('triton__17', 'out_ptr2', 'buf212', (buf212.sum()/buf212.nelement()).item(), buf212.amax().item(), buf212.amin().item())
        print('triton__17', 'out_ptr3', 'buf214', (buf214.sum()/buf214.nelement()).item(), buf214.amax().item(), buf214.amin().item())
        del primals_187
        del primals_188
        buf215 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf208', (buf208.sum()/buf208.nelement()).item(), buf208.amax().item(), buf208.amin().item())
        print('triton__18', 'in_ptr1', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        print('triton__18', 'in_ptr2', 'buf211', (buf211.sum()/buf211.nelement()).item(), buf211.amax().item(), buf211.amin().item())
        print('triton__18', 'in_ptr3', 'primals_189', (primals_189.sum()/primals_189.nelement()).item(), primals_189.amax().item(), primals_189.amin().item())
        print('triton__18', 'in_ptr4', 'primals_190', (primals_190.sum()/primals_190.nelement()).item(), primals_190.amax().item(), primals_190.amin().item())
        triton__18.run(buf208, buf210, buf211, primals_189, primals_190, buf215, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf215', (buf215.sum()/buf215.nelement()).item(), buf215.amax().item(), buf215.amin().item())
        del primals_190
        buf216 = aten.convolution(buf215, primals_27, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf216, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf217 = buf211; del buf211  # reuse
        buf218 = buf217; del buf217  # reuse
        buf221 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf219 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf222 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__17', 'in_out_ptr0', 'buf218', (buf218.sum()/buf218.nelement()).item(), buf218.amax().item(), buf218.amin().item())
        print('triton__17', 'in_ptr0', 'buf216', (buf216.sum()/buf216.nelement()).item(), buf216.amax().item(), buf216.amin().item())
        print('triton__17', 'in_ptr1', 'primals_192', (primals_192.sum()/primals_192.nelement()).item(), primals_192.amax().item(), primals_192.amin().item())
        print('triton__17', 'in_ptr2', 'primals_193', (primals_193.sum()/primals_193.nelement()).item(), primals_193.amax().item(), primals_193.amin().item())
        triton__17.run(buf218, buf216, primals_192, primals_193, buf221, buf219, buf220, buf222, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__17', 'in_out_ptr0', 'buf218', (buf218.sum()/buf218.nelement()).item(), buf218.amax().item(), buf218.amin().item())
        print('triton__17', 'out_ptr0', 'buf221', (buf221.sum()/buf221.nelement()).item(), buf221.amax().item(), buf221.amin().item())
        print('triton__17', 'out_ptr1', 'buf219', (buf219.sum()/buf219.nelement()).item(), buf219.amax().item(), buf219.amin().item())
        print('triton__17', 'out_ptr2', 'buf220', (buf220.sum()/buf220.nelement()).item(), buf220.amax().item(), buf220.amin().item())
        print('triton__17', 'out_ptr3', 'buf222', (buf222.sum()/buf222.nelement()).item(), buf222.amax().item(), buf222.amin().item())
        del primals_192
        del primals_193
        buf223 = empty_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'buf216', (buf216.sum()/buf216.nelement()).item(), buf216.amax().item(), buf216.amin().item())
        print('triton__18', 'in_ptr1', 'buf218', (buf218.sum()/buf218.nelement()).item(), buf218.amax().item(), buf218.amin().item())
        print('triton__18', 'in_ptr2', 'buf219', (buf219.sum()/buf219.nelement()).item(), buf219.amax().item(), buf219.amin().item())
        print('triton__18', 'in_ptr3', 'primals_194', (primals_194.sum()/primals_194.nelement()).item(), primals_194.amax().item(), primals_194.amin().item())
        print('triton__18', 'in_ptr4', 'primals_195', (primals_195.sum()/primals_195.nelement()).item(), primals_195.amax().item(), primals_195.amin().item())
        triton__18.run(buf216, buf218, buf219, primals_194, primals_195, buf223, 327680, grid=grid(327680), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf223', (buf223.sum()/buf223.nelement()).item(), buf223.amax().item(), buf223.amin().item())
        del buf219
        del primals_195
        buf224 = aten.convolution(buf223, primals_28, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf224, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf225 = buf203; del buf203  # reuse
        buf226 = buf225; del buf225  # reuse
        buf229 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf227 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf228 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf230 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__19', 'in_out_ptr0', 'buf226', (buf226.sum()/buf226.nelement()).item(), buf226.amax().item(), buf226.amin().item())
        print('triton__19', 'in_ptr0', 'buf224', (buf224.sum()/buf224.nelement()).item(), buf224.amax().item(), buf224.amin().item())
        print('triton__19', 'in_ptr1', 'primals_197', (primals_197.sum()/primals_197.nelement()).item(), primals_197.amax().item(), primals_197.amin().item())
        print('triton__19', 'in_ptr2', 'primals_198', (primals_198.sum()/primals_198.nelement()).item(), primals_198.amax().item(), primals_198.amin().item())
        triton__19.run(buf226, buf224, primals_197, primals_198, buf229, buf227, buf228, buf230, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf226', (buf226.sum()/buf226.nelement()).item(), buf226.amax().item(), buf226.amin().item())
        print('triton__19', 'out_ptr0', 'buf229', (buf229.sum()/buf229.nelement()).item(), buf229.amax().item(), buf229.amin().item())
        print('triton__19', 'out_ptr1', 'buf227', (buf227.sum()/buf227.nelement()).item(), buf227.amax().item(), buf227.amin().item())
        print('triton__19', 'out_ptr2', 'buf228', (buf228.sum()/buf228.nelement()).item(), buf228.amax().item(), buf228.amin().item())
        print('triton__19', 'out_ptr3', 'buf230', (buf230.sum()/buf230.nelement()).item(), buf230.amax().item(), buf230.amin().item())
        del primals_197
        del primals_198
        buf231 = empty_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__21', 'in_ptr0', 'buf224', (buf224.sum()/buf224.nelement()).item(), buf224.amax().item(), buf224.amin().item())
        print('triton__21', 'in_ptr1', 'buf226', (buf226.sum()/buf226.nelement()).item(), buf226.amax().item(), buf226.amin().item())
        print('triton__21', 'in_ptr2', 'buf227', (buf227.sum()/buf227.nelement()).item(), buf227.amax().item(), buf227.amin().item())
        print('triton__21', 'in_ptr3', 'primals_199', (primals_199.sum()/primals_199.nelement()).item(), primals_199.amax().item(), primals_199.amin().item())
        print('triton__21', 'in_ptr4', 'primals_200', (primals_200.sum()/primals_200.nelement()).item(), primals_200.amax().item(), primals_200.amin().item())
        print('triton__21', 'in_ptr5', 'buf207', (buf207.sum()/buf207.nelement()).item(), buf207.amax().item(), buf207.amin().item())
        triton__21.run(buf224, buf226, buf227, primals_199, primals_200, buf207, buf231, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf231', (buf231.sum()/buf231.nelement()).item(), buf231.amax().item(), buf231.amin().item())
        del primals_200
        buf232 = aten.convolution(buf231, primals_29, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf232, (8, 1920, 16, 16), (491520, 256, 16, 1))
        buf233 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf234 = buf233; del buf233  # reuse
        buf237 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf235 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf236 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf238 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_out_ptr0', 'buf234', (buf234.sum()/buf234.nelement()).item(), buf234.amax().item(), buf234.amin().item())
        print('triton__22', 'in_ptr0', 'buf232', (buf232.sum()/buf232.nelement()).item(), buf232.amax().item(), buf232.amin().item())
        print('triton__22', 'in_ptr1', 'primals_202', (primals_202.sum()/primals_202.nelement()).item(), primals_202.amax().item(), primals_202.amin().item())
        print('triton__22', 'in_ptr2', 'primals_203', (primals_203.sum()/primals_203.nelement()).item(), primals_203.amax().item(), primals_203.amin().item())
        triton__22.run(buf234, buf232, primals_202, primals_203, buf237, buf235, buf236, buf238, 1920, 2048, grid=grid(1920), stream=stream0)
        print('triton__22', 'in_out_ptr0', 'buf234', (buf234.sum()/buf234.nelement()).item(), buf234.amax().item(), buf234.amin().item())
        print('triton__22', 'out_ptr0', 'buf237', (buf237.sum()/buf237.nelement()).item(), buf237.amax().item(), buf237.amin().item())
        print('triton__22', 'out_ptr1', 'buf235', (buf235.sum()/buf235.nelement()).item(), buf235.amax().item(), buf235.amin().item())
        print('triton__22', 'out_ptr2', 'buf236', (buf236.sum()/buf236.nelement()).item(), buf236.amax().item(), buf236.amin().item())
        print('triton__22', 'out_ptr3', 'buf238', (buf238.sum()/buf238.nelement()).item(), buf238.amax().item(), buf238.amin().item())
        del primals_202
        del primals_203
        buf239 = empty_strided((8, 1920, 16, 16), (491520, 256, 16, 1), device='cuda', dtype=torch.float32)
        print('triton__23', 'in_ptr0', 'buf232', (buf232.sum()/buf232.nelement()).item(), buf232.amax().item(), buf232.amin().item())
        print('triton__23', 'in_ptr1', 'buf234', (buf234.sum()/buf234.nelement()).item(), buf234.amax().item(), buf234.amin().item())
        print('triton__23', 'in_ptr2', 'buf235', (buf235.sum()/buf235.nelement()).item(), buf235.amax().item(), buf235.amin().item())
        print('triton__23', 'in_ptr3', 'primals_204', (primals_204.sum()/primals_204.nelement()).item(), primals_204.amax().item(), primals_204.amin().item())
        print('triton__23', 'in_ptr4', 'primals_205', (primals_205.sum()/primals_205.nelement()).item(), primals_205.amax().item(), primals_205.amin().item())
        triton__23.run(buf232, buf234, buf235, primals_204, primals_205, buf239, 3932160, grid=grid(3932160), stream=stream0)
        print('triton__23', 'out_ptr0', 'buf239', (buf239.sum()/buf239.nelement()).item(), buf239.amax().item(), buf239.amin().item())
        del primals_205
        buf240 = aten.convolution(buf239, primals_30, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf240, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf241 = buf235; del buf235  # reuse
        buf242 = buf241; del buf241  # reuse
        buf245 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf243 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf244 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf242', (buf242.sum()/buf242.nelement()).item(), buf242.amax().item(), buf242.amin().item())
        print('triton__24', 'in_ptr0', 'buf240', (buf240.sum()/buf240.nelement()).item(), buf240.amax().item(), buf240.amin().item())
        print('triton__24', 'in_ptr1', 'primals_207', (primals_207.sum()/primals_207.nelement()).item(), primals_207.amax().item(), primals_207.amin().item())
        print('triton__24', 'in_ptr2', 'primals_208', (primals_208.sum()/primals_208.nelement()).item(), primals_208.amax().item(), primals_208.amin().item())
        triton__24.run(buf242, buf240, primals_207, primals_208, buf245, buf243, buf244, buf246, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf242', (buf242.sum()/buf242.nelement()).item(), buf242.amax().item(), buf242.amin().item())
        print('triton__24', 'out_ptr0', 'buf245', (buf245.sum()/buf245.nelement()).item(), buf245.amax().item(), buf245.amin().item())
        print('triton__24', 'out_ptr1', 'buf243', (buf243.sum()/buf243.nelement()).item(), buf243.amax().item(), buf243.amin().item())
        print('triton__24', 'out_ptr2', 'buf244', (buf244.sum()/buf244.nelement()).item(), buf244.amax().item(), buf244.amin().item())
        print('triton__24', 'out_ptr3', 'buf246', (buf246.sum()/buf246.nelement()).item(), buf246.amax().item(), buf246.amin().item())
        del primals_207
        del primals_208
        buf247 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf240', (buf240.sum()/buf240.nelement()).item(), buf240.amax().item(), buf240.amin().item())
        print('triton__25', 'in_ptr1', 'buf242', (buf242.sum()/buf242.nelement()).item(), buf242.amax().item(), buf242.amin().item())
        print('triton__25', 'in_ptr2', 'buf243', (buf243.sum()/buf243.nelement()).item(), buf243.amax().item(), buf243.amin().item())
        print('triton__25', 'in_ptr3', 'primals_209', (primals_209.sum()/primals_209.nelement()).item(), primals_209.amax().item(), primals_209.amin().item())
        print('triton__25', 'in_ptr4', 'primals_210', (primals_210.sum()/primals_210.nelement()).item(), primals_210.amax().item(), primals_210.amin().item())
        triton__25.run(buf240, buf242, buf243, primals_209, primals_210, buf247, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf247', (buf247.sum()/buf247.nelement()).item(), buf247.amax().item(), buf247.amin().item())
        del primals_210
        buf248 = aten.convolution(buf247, primals_31, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf248, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf249 = buf227; del buf227  # reuse
        buf250 = buf249; del buf249  # reuse
        buf253 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf251 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf252 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf254 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf250', (buf250.sum()/buf250.nelement()).item(), buf250.amax().item(), buf250.amin().item())
        print('triton__26', 'in_ptr0', 'buf248', (buf248.sum()/buf248.nelement()).item(), buf248.amax().item(), buf248.amin().item())
        print('triton__26', 'in_ptr1', 'primals_212', (primals_212.sum()/primals_212.nelement()).item(), primals_212.amax().item(), primals_212.amin().item())
        print('triton__26', 'in_ptr2', 'primals_213', (primals_213.sum()/primals_213.nelement()).item(), primals_213.amax().item(), primals_213.amin().item())
        triton__26.run(buf250, buf248, primals_212, primals_213, buf253, buf251, buf252, buf254, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf250', (buf250.sum()/buf250.nelement()).item(), buf250.amax().item(), buf250.amin().item())
        print('triton__26', 'out_ptr0', 'buf253', (buf253.sum()/buf253.nelement()).item(), buf253.amax().item(), buf253.amin().item())
        print('triton__26', 'out_ptr1', 'buf251', (buf251.sum()/buf251.nelement()).item(), buf251.amax().item(), buf251.amin().item())
        print('triton__26', 'out_ptr2', 'buf252', (buf252.sum()/buf252.nelement()).item(), buf252.amax().item(), buf252.amin().item())
        print('triton__26', 'out_ptr3', 'buf254', (buf254.sum()/buf254.nelement()).item(), buf254.amax().item(), buf254.amin().item())
        del primals_212
        del primals_213
        buf255 = aten.convolution(buf231, primals_32, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf255, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf256 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf257 = buf256; del buf256  # reuse
        buf260 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf258 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf259 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf261 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf257', (buf257.sum()/buf257.nelement()).item(), buf257.amax().item(), buf257.amin().item())
        print('triton__26', 'in_ptr0', 'buf255', (buf255.sum()/buf255.nelement()).item(), buf255.amax().item(), buf255.amin().item())
        print('triton__26', 'in_ptr1', 'primals_217', (primals_217.sum()/primals_217.nelement()).item(), primals_217.amax().item(), primals_217.amin().item())
        print('triton__26', 'in_ptr2', 'primals_218', (primals_218.sum()/primals_218.nelement()).item(), primals_218.amax().item(), primals_218.amin().item())
        triton__26.run(buf257, buf255, primals_217, primals_218, buf260, buf258, buf259, buf261, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf257', (buf257.sum()/buf257.nelement()).item(), buf257.amax().item(), buf257.amin().item())
        print('triton__26', 'out_ptr0', 'buf260', (buf260.sum()/buf260.nelement()).item(), buf260.amax().item(), buf260.amin().item())
        print('triton__26', 'out_ptr1', 'buf258', (buf258.sum()/buf258.nelement()).item(), buf258.amax().item(), buf258.amin().item())
        print('triton__26', 'out_ptr2', 'buf259', (buf259.sum()/buf259.nelement()).item(), buf259.amax().item(), buf259.amin().item())
        print('triton__26', 'out_ptr3', 'buf261', (buf261.sum()/buf261.nelement()).item(), buf261.amax().item(), buf261.amin().item())
        del primals_217
        del primals_218
        buf262 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        buf263 = buf262; del buf262  # reuse
        print('triton__27', 'in_out_ptr0', 'buf263', (buf263.sum()/buf263.nelement()).item(), buf263.amax().item(), buf263.amin().item())
        print('triton__27', 'in_ptr0', 'buf248', (buf248.sum()/buf248.nelement()).item(), buf248.amax().item(), buf248.amin().item())
        print('triton__27', 'in_ptr1', 'buf250', (buf250.sum()/buf250.nelement()).item(), buf250.amax().item(), buf250.amin().item())
        print('triton__27', 'in_ptr2', 'buf251', (buf251.sum()/buf251.nelement()).item(), buf251.amax().item(), buf251.amin().item())
        print('triton__27', 'in_ptr3', 'primals_214', (primals_214.sum()/primals_214.nelement()).item(), primals_214.amax().item(), primals_214.amin().item())
        print('triton__27', 'in_ptr4', 'primals_215', (primals_215.sum()/primals_215.nelement()).item(), primals_215.amax().item(), primals_215.amin().item())
        print('triton__27', 'in_ptr5', 'buf255', (buf255.sum()/buf255.nelement()).item(), buf255.amax().item(), buf255.amin().item())
        print('triton__27', 'in_ptr6', 'buf257', (buf257.sum()/buf257.nelement()).item(), buf257.amax().item(), buf257.amin().item())
        print('triton__27', 'in_ptr7', 'buf258', (buf258.sum()/buf258.nelement()).item(), buf258.amax().item(), buf258.amin().item())
        print('triton__27', 'in_ptr8', 'primals_219', (primals_219.sum()/primals_219.nelement()).item(), primals_219.amax().item(), primals_219.amin().item())
        print('triton__27', 'in_ptr9', 'primals_220', (primals_220.sum()/primals_220.nelement()).item(), primals_220.amax().item(), primals_220.amin().item())
        triton__27.run(buf263, buf248, buf250, buf251, primals_214, primals_215, buf255, buf257, buf258, primals_219, primals_220, 327680, grid=grid(327680), stream=stream0)
        print('triton__27', 'in_out_ptr0', 'buf263', (buf263.sum()/buf263.nelement()).item(), buf263.amax().item(), buf263.amin().item())
        del primals_215
        del primals_220
        buf264 = aten.convolution(buf263, primals_33, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf264, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf265 = buf243; del buf243  # reuse
        buf266 = buf265; del buf265  # reuse
        buf269 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf267 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf268 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf270 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf266', (buf266.sum()/buf266.nelement()).item(), buf266.amax().item(), buf266.amin().item())
        print('triton__24', 'in_ptr0', 'buf264', (buf264.sum()/buf264.nelement()).item(), buf264.amax().item(), buf264.amin().item())
        print('triton__24', 'in_ptr1', 'primals_222', (primals_222.sum()/primals_222.nelement()).item(), primals_222.amax().item(), primals_222.amin().item())
        print('triton__24', 'in_ptr2', 'primals_223', (primals_223.sum()/primals_223.nelement()).item(), primals_223.amax().item(), primals_223.amin().item())
        triton__24.run(buf266, buf264, primals_222, primals_223, buf269, buf267, buf268, buf270, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf266', (buf266.sum()/buf266.nelement()).item(), buf266.amax().item(), buf266.amin().item())
        print('triton__24', 'out_ptr0', 'buf269', (buf269.sum()/buf269.nelement()).item(), buf269.amax().item(), buf269.amin().item())
        print('triton__24', 'out_ptr1', 'buf267', (buf267.sum()/buf267.nelement()).item(), buf267.amax().item(), buf267.amin().item())
        print('triton__24', 'out_ptr2', 'buf268', (buf268.sum()/buf268.nelement()).item(), buf268.amax().item(), buf268.amin().item())
        print('triton__24', 'out_ptr3', 'buf270', (buf270.sum()/buf270.nelement()).item(), buf270.amax().item(), buf270.amin().item())
        del primals_222
        del primals_223
        buf271 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf264', (buf264.sum()/buf264.nelement()).item(), buf264.amax().item(), buf264.amin().item())
        print('triton__25', 'in_ptr1', 'buf266', (buf266.sum()/buf266.nelement()).item(), buf266.amax().item(), buf266.amin().item())
        print('triton__25', 'in_ptr2', 'buf267', (buf267.sum()/buf267.nelement()).item(), buf267.amax().item(), buf267.amin().item())
        print('triton__25', 'in_ptr3', 'primals_224', (primals_224.sum()/primals_224.nelement()).item(), primals_224.amax().item(), primals_224.amin().item())
        print('triton__25', 'in_ptr4', 'primals_225', (primals_225.sum()/primals_225.nelement()).item(), primals_225.amax().item(), primals_225.amin().item())
        triton__25.run(buf264, buf266, buf267, primals_224, primals_225, buf271, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf271', (buf271.sum()/buf271.nelement()).item(), buf271.amax().item(), buf271.amin().item())
        del primals_225
        buf272 = aten.convolution(buf271, primals_34, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf272, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf273 = buf267; del buf267  # reuse
        buf274 = buf273; del buf273  # reuse
        buf277 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf275 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf276 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf278 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__24', 'in_ptr0', 'buf272', (buf272.sum()/buf272.nelement()).item(), buf272.amax().item(), buf272.amin().item())
        print('triton__24', 'in_ptr1', 'primals_227', (primals_227.sum()/primals_227.nelement()).item(), primals_227.amax().item(), primals_227.amin().item())
        print('triton__24', 'in_ptr2', 'primals_228', (primals_228.sum()/primals_228.nelement()).item(), primals_228.amax().item(), primals_228.amin().item())
        triton__24.run(buf274, buf272, primals_227, primals_228, buf277, buf275, buf276, buf278, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__24', 'out_ptr0', 'buf277', (buf277.sum()/buf277.nelement()).item(), buf277.amax().item(), buf277.amin().item())
        print('triton__24', 'out_ptr1', 'buf275', (buf275.sum()/buf275.nelement()).item(), buf275.amax().item(), buf275.amin().item())
        print('triton__24', 'out_ptr2', 'buf276', (buf276.sum()/buf276.nelement()).item(), buf276.amax().item(), buf276.amin().item())
        print('triton__24', 'out_ptr3', 'buf278', (buf278.sum()/buf278.nelement()).item(), buf278.amax().item(), buf278.amin().item())
        del primals_227
        del primals_228
        buf279 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf272', (buf272.sum()/buf272.nelement()).item(), buf272.amax().item(), buf272.amin().item())
        print('triton__25', 'in_ptr1', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__25', 'in_ptr2', 'buf275', (buf275.sum()/buf275.nelement()).item(), buf275.amax().item(), buf275.amin().item())
        print('triton__25', 'in_ptr3', 'primals_229', (primals_229.sum()/primals_229.nelement()).item(), primals_229.amax().item(), primals_229.amin().item())
        print('triton__25', 'in_ptr4', 'primals_230', (primals_230.sum()/primals_230.nelement()).item(), primals_230.amax().item(), primals_230.amin().item())
        triton__25.run(buf272, buf274, buf275, primals_229, primals_230, buf279, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf279', (buf279.sum()/buf279.nelement()).item(), buf279.amax().item(), buf279.amin().item())
        del primals_230
        buf280 = aten.convolution(buf279, primals_35, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf280, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf281 = buf258; del buf258  # reuse
        buf282 = buf281; del buf281  # reuse
        buf285 = as_strided(buf251, (640, ), (1, )); del buf251  # reuse
        buf283 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf284 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf286 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf282', (buf282.sum()/buf282.nelement()).item(), buf282.amax().item(), buf282.amin().item())
        print('triton__26', 'in_ptr0', 'buf280', (buf280.sum()/buf280.nelement()).item(), buf280.amax().item(), buf280.amin().item())
        print('triton__26', 'in_ptr1', 'primals_232', (primals_232.sum()/primals_232.nelement()).item(), primals_232.amax().item(), primals_232.amin().item())
        print('triton__26', 'in_ptr2', 'primals_233', (primals_233.sum()/primals_233.nelement()).item(), primals_233.amax().item(), primals_233.amin().item())
        triton__26.run(buf282, buf280, primals_232, primals_233, buf285, buf283, buf284, buf286, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf282', (buf282.sum()/buf282.nelement()).item(), buf282.amax().item(), buf282.amin().item())
        print('triton__26', 'out_ptr0', 'buf285', (buf285.sum()/buf285.nelement()).item(), buf285.amax().item(), buf285.amin().item())
        print('triton__26', 'out_ptr1', 'buf283', (buf283.sum()/buf283.nelement()).item(), buf283.amax().item(), buf283.amin().item())
        print('triton__26', 'out_ptr2', 'buf284', (buf284.sum()/buf284.nelement()).item(), buf284.amax().item(), buf284.amin().item())
        print('triton__26', 'out_ptr3', 'buf286', (buf286.sum()/buf286.nelement()).item(), buf286.amax().item(), buf286.amin().item())
        del primals_232
        del primals_233
        buf287 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf280', (buf280.sum()/buf280.nelement()).item(), buf280.amax().item(), buf280.amin().item())
        print('triton__28', 'in_ptr1', 'buf282', (buf282.sum()/buf282.nelement()).item(), buf282.amax().item(), buf282.amin().item())
        print('triton__28', 'in_ptr2', 'buf283', (buf283.sum()/buf283.nelement()).item(), buf283.amax().item(), buf283.amin().item())
        print('triton__28', 'in_ptr3', 'primals_234', (primals_234.sum()/primals_234.nelement()).item(), primals_234.amax().item(), primals_234.amin().item())
        print('triton__28', 'in_ptr4', 'primals_235', (primals_235.sum()/primals_235.nelement()).item(), primals_235.amax().item(), primals_235.amin().item())
        print('triton__28', 'in_ptr5', 'buf263', (buf263.sum()/buf263.nelement()).item(), buf263.amax().item(), buf263.amin().item())
        triton__28.run(buf280, buf282, buf283, primals_234, primals_235, buf263, buf287, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf287', (buf287.sum()/buf287.nelement()).item(), buf287.amax().item(), buf287.amin().item())
        del primals_235
        buf288 = aten.convolution(buf287, primals_36, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf288, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf289 = buf275; del buf275  # reuse
        buf290 = buf289; del buf289  # reuse
        buf293 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf291 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf292 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf294 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf290', (buf290.sum()/buf290.nelement()).item(), buf290.amax().item(), buf290.amin().item())
        print('triton__24', 'in_ptr0', 'buf288', (buf288.sum()/buf288.nelement()).item(), buf288.amax().item(), buf288.amin().item())
        print('triton__24', 'in_ptr1', 'primals_237', (primals_237.sum()/primals_237.nelement()).item(), primals_237.amax().item(), primals_237.amin().item())
        print('triton__24', 'in_ptr2', 'primals_238', (primals_238.sum()/primals_238.nelement()).item(), primals_238.amax().item(), primals_238.amin().item())
        triton__24.run(buf290, buf288, primals_237, primals_238, buf293, buf291, buf292, buf294, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf290', (buf290.sum()/buf290.nelement()).item(), buf290.amax().item(), buf290.amin().item())
        print('triton__24', 'out_ptr0', 'buf293', (buf293.sum()/buf293.nelement()).item(), buf293.amax().item(), buf293.amin().item())
        print('triton__24', 'out_ptr1', 'buf291', (buf291.sum()/buf291.nelement()).item(), buf291.amax().item(), buf291.amin().item())
        print('triton__24', 'out_ptr2', 'buf292', (buf292.sum()/buf292.nelement()).item(), buf292.amax().item(), buf292.amin().item())
        print('triton__24', 'out_ptr3', 'buf294', (buf294.sum()/buf294.nelement()).item(), buf294.amax().item(), buf294.amin().item())
        del primals_237
        del primals_238
        buf295 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf288', (buf288.sum()/buf288.nelement()).item(), buf288.amax().item(), buf288.amin().item())
        print('triton__25', 'in_ptr1', 'buf290', (buf290.sum()/buf290.nelement()).item(), buf290.amax().item(), buf290.amin().item())
        print('triton__25', 'in_ptr2', 'buf291', (buf291.sum()/buf291.nelement()).item(), buf291.amax().item(), buf291.amin().item())
        print('triton__25', 'in_ptr3', 'primals_239', (primals_239.sum()/primals_239.nelement()).item(), primals_239.amax().item(), primals_239.amin().item())
        print('triton__25', 'in_ptr4', 'primals_240', (primals_240.sum()/primals_240.nelement()).item(), primals_240.amax().item(), primals_240.amin().item())
        triton__25.run(buf288, buf290, buf291, primals_239, primals_240, buf295, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf295', (buf295.sum()/buf295.nelement()).item(), buf295.amax().item(), buf295.amin().item())
        del primals_240
        buf296 = aten.convolution(buf295, primals_37, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf296, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf297 = buf291; del buf291  # reuse
        buf298 = buf297; del buf297  # reuse
        buf301 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf299 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf300 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf302 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf298', (buf298.sum()/buf298.nelement()).item(), buf298.amax().item(), buf298.amin().item())
        print('triton__24', 'in_ptr0', 'buf296', (buf296.sum()/buf296.nelement()).item(), buf296.amax().item(), buf296.amin().item())
        print('triton__24', 'in_ptr1', 'primals_242', (primals_242.sum()/primals_242.nelement()).item(), primals_242.amax().item(), primals_242.amin().item())
        print('triton__24', 'in_ptr2', 'primals_243', (primals_243.sum()/primals_243.nelement()).item(), primals_243.amax().item(), primals_243.amin().item())
        triton__24.run(buf298, buf296, primals_242, primals_243, buf301, buf299, buf300, buf302, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf298', (buf298.sum()/buf298.nelement()).item(), buf298.amax().item(), buf298.amin().item())
        print('triton__24', 'out_ptr0', 'buf301', (buf301.sum()/buf301.nelement()).item(), buf301.amax().item(), buf301.amin().item())
        print('triton__24', 'out_ptr1', 'buf299', (buf299.sum()/buf299.nelement()).item(), buf299.amax().item(), buf299.amin().item())
        print('triton__24', 'out_ptr2', 'buf300', (buf300.sum()/buf300.nelement()).item(), buf300.amax().item(), buf300.amin().item())
        print('triton__24', 'out_ptr3', 'buf302', (buf302.sum()/buf302.nelement()).item(), buf302.amax().item(), buf302.amin().item())
        del primals_242
        del primals_243
        buf303 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf296', (buf296.sum()/buf296.nelement()).item(), buf296.amax().item(), buf296.amin().item())
        print('triton__25', 'in_ptr1', 'buf298', (buf298.sum()/buf298.nelement()).item(), buf298.amax().item(), buf298.amin().item())
        print('triton__25', 'in_ptr2', 'buf299', (buf299.sum()/buf299.nelement()).item(), buf299.amax().item(), buf299.amin().item())
        print('triton__25', 'in_ptr3', 'primals_244', (primals_244.sum()/primals_244.nelement()).item(), primals_244.amax().item(), primals_244.amin().item())
        print('triton__25', 'in_ptr4', 'primals_245', (primals_245.sum()/primals_245.nelement()).item(), primals_245.amax().item(), primals_245.amin().item())
        triton__25.run(buf296, buf298, buf299, primals_244, primals_245, buf303, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf303', (buf303.sum()/buf303.nelement()).item(), buf303.amax().item(), buf303.amin().item())
        del primals_245
        buf304 = aten.convolution(buf303, primals_38, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf304, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf305 = buf283; del buf283  # reuse
        buf306 = buf305; del buf305  # reuse
        buf309 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf307 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf308 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf310 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf306', (buf306.sum()/buf306.nelement()).item(), buf306.amax().item(), buf306.amin().item())
        print('triton__26', 'in_ptr0', 'buf304', (buf304.sum()/buf304.nelement()).item(), buf304.amax().item(), buf304.amin().item())
        print('triton__26', 'in_ptr1', 'primals_247', (primals_247.sum()/primals_247.nelement()).item(), primals_247.amax().item(), primals_247.amin().item())
        print('triton__26', 'in_ptr2', 'primals_248', (primals_248.sum()/primals_248.nelement()).item(), primals_248.amax().item(), primals_248.amin().item())
        triton__26.run(buf306, buf304, primals_247, primals_248, buf309, buf307, buf308, buf310, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf306', (buf306.sum()/buf306.nelement()).item(), buf306.amax().item(), buf306.amin().item())
        print('triton__26', 'out_ptr0', 'buf309', (buf309.sum()/buf309.nelement()).item(), buf309.amax().item(), buf309.amin().item())
        print('triton__26', 'out_ptr1', 'buf307', (buf307.sum()/buf307.nelement()).item(), buf307.amax().item(), buf307.amin().item())
        print('triton__26', 'out_ptr2', 'buf308', (buf308.sum()/buf308.nelement()).item(), buf308.amax().item(), buf308.amin().item())
        print('triton__26', 'out_ptr3', 'buf310', (buf310.sum()/buf310.nelement()).item(), buf310.amax().item(), buf310.amin().item())
        del primals_247
        del primals_248
        buf311 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf304', (buf304.sum()/buf304.nelement()).item(), buf304.amax().item(), buf304.amin().item())
        print('triton__28', 'in_ptr1', 'buf306', (buf306.sum()/buf306.nelement()).item(), buf306.amax().item(), buf306.amin().item())
        print('triton__28', 'in_ptr2', 'buf307', (buf307.sum()/buf307.nelement()).item(), buf307.amax().item(), buf307.amin().item())
        print('triton__28', 'in_ptr3', 'primals_249', (primals_249.sum()/primals_249.nelement()).item(), primals_249.amax().item(), primals_249.amin().item())
        print('triton__28', 'in_ptr4', 'primals_250', (primals_250.sum()/primals_250.nelement()).item(), primals_250.amax().item(), primals_250.amin().item())
        print('triton__28', 'in_ptr5', 'buf287', (buf287.sum()/buf287.nelement()).item(), buf287.amax().item(), buf287.amin().item())
        triton__28.run(buf304, buf306, buf307, primals_249, primals_250, buf287, buf311, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf311', (buf311.sum()/buf311.nelement()).item(), buf311.amax().item(), buf311.amin().item())
        del primals_250
        buf312 = aten.convolution(buf311, primals_39, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf312, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf313 = buf299; del buf299  # reuse
        buf314 = buf313; del buf313  # reuse
        buf317 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf315 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf316 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf318 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf314', (buf314.sum()/buf314.nelement()).item(), buf314.amax().item(), buf314.amin().item())
        print('triton__24', 'in_ptr0', 'buf312', (buf312.sum()/buf312.nelement()).item(), buf312.amax().item(), buf312.amin().item())
        print('triton__24', 'in_ptr1', 'primals_252', (primals_252.sum()/primals_252.nelement()).item(), primals_252.amax().item(), primals_252.amin().item())
        print('triton__24', 'in_ptr2', 'primals_253', (primals_253.sum()/primals_253.nelement()).item(), primals_253.amax().item(), primals_253.amin().item())
        triton__24.run(buf314, buf312, primals_252, primals_253, buf317, buf315, buf316, buf318, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf314', (buf314.sum()/buf314.nelement()).item(), buf314.amax().item(), buf314.amin().item())
        print('triton__24', 'out_ptr0', 'buf317', (buf317.sum()/buf317.nelement()).item(), buf317.amax().item(), buf317.amin().item())
        print('triton__24', 'out_ptr1', 'buf315', (buf315.sum()/buf315.nelement()).item(), buf315.amax().item(), buf315.amin().item())
        print('triton__24', 'out_ptr2', 'buf316', (buf316.sum()/buf316.nelement()).item(), buf316.amax().item(), buf316.amin().item())
        print('triton__24', 'out_ptr3', 'buf318', (buf318.sum()/buf318.nelement()).item(), buf318.amax().item(), buf318.amin().item())
        del primals_252
        del primals_253
        buf319 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf312', (buf312.sum()/buf312.nelement()).item(), buf312.amax().item(), buf312.amin().item())
        print('triton__25', 'in_ptr1', 'buf314', (buf314.sum()/buf314.nelement()).item(), buf314.amax().item(), buf314.amin().item())
        print('triton__25', 'in_ptr2', 'buf315', (buf315.sum()/buf315.nelement()).item(), buf315.amax().item(), buf315.amin().item())
        print('triton__25', 'in_ptr3', 'primals_254', (primals_254.sum()/primals_254.nelement()).item(), primals_254.amax().item(), primals_254.amin().item())
        print('triton__25', 'in_ptr4', 'primals_255', (primals_255.sum()/primals_255.nelement()).item(), primals_255.amax().item(), primals_255.amin().item())
        triton__25.run(buf312, buf314, buf315, primals_254, primals_255, buf319, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf319', (buf319.sum()/buf319.nelement()).item(), buf319.amax().item(), buf319.amin().item())
        del primals_255
        buf320 = aten.convolution(buf319, primals_40, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf320, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf321 = buf315; del buf315  # reuse
        buf322 = buf321; del buf321  # reuse
        buf325 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf323 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf324 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf326 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf322', (buf322.sum()/buf322.nelement()).item(), buf322.amax().item(), buf322.amin().item())
        print('triton__24', 'in_ptr0', 'buf320', (buf320.sum()/buf320.nelement()).item(), buf320.amax().item(), buf320.amin().item())
        print('triton__24', 'in_ptr1', 'primals_257', (primals_257.sum()/primals_257.nelement()).item(), primals_257.amax().item(), primals_257.amin().item())
        print('triton__24', 'in_ptr2', 'primals_258', (primals_258.sum()/primals_258.nelement()).item(), primals_258.amax().item(), primals_258.amin().item())
        triton__24.run(buf322, buf320, primals_257, primals_258, buf325, buf323, buf324, buf326, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf322', (buf322.sum()/buf322.nelement()).item(), buf322.amax().item(), buf322.amin().item())
        print('triton__24', 'out_ptr0', 'buf325', (buf325.sum()/buf325.nelement()).item(), buf325.amax().item(), buf325.amin().item())
        print('triton__24', 'out_ptr1', 'buf323', (buf323.sum()/buf323.nelement()).item(), buf323.amax().item(), buf323.amin().item())
        print('triton__24', 'out_ptr2', 'buf324', (buf324.sum()/buf324.nelement()).item(), buf324.amax().item(), buf324.amin().item())
        print('triton__24', 'out_ptr3', 'buf326', (buf326.sum()/buf326.nelement()).item(), buf326.amax().item(), buf326.amin().item())
        del primals_257
        del primals_258
        buf327 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf320', (buf320.sum()/buf320.nelement()).item(), buf320.amax().item(), buf320.amin().item())
        print('triton__25', 'in_ptr1', 'buf322', (buf322.sum()/buf322.nelement()).item(), buf322.amax().item(), buf322.amin().item())
        print('triton__25', 'in_ptr2', 'buf323', (buf323.sum()/buf323.nelement()).item(), buf323.amax().item(), buf323.amin().item())
        print('triton__25', 'in_ptr3', 'primals_259', (primals_259.sum()/primals_259.nelement()).item(), primals_259.amax().item(), primals_259.amin().item())
        print('triton__25', 'in_ptr4', 'primals_260', (primals_260.sum()/primals_260.nelement()).item(), primals_260.amax().item(), primals_260.amin().item())
        triton__25.run(buf320, buf322, buf323, primals_259, primals_260, buf327, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf327', (buf327.sum()/buf327.nelement()).item(), buf327.amax().item(), buf327.amin().item())
        del primals_260
        buf328 = aten.convolution(buf327, primals_41, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf328, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf329 = buf307; del buf307  # reuse
        buf330 = buf329; del buf329  # reuse
        buf333 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf331 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf332 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf334 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf330', (buf330.sum()/buf330.nelement()).item(), buf330.amax().item(), buf330.amin().item())
        print('triton__26', 'in_ptr0', 'buf328', (buf328.sum()/buf328.nelement()).item(), buf328.amax().item(), buf328.amin().item())
        print('triton__26', 'in_ptr1', 'primals_262', (primals_262.sum()/primals_262.nelement()).item(), primals_262.amax().item(), primals_262.amin().item())
        print('triton__26', 'in_ptr2', 'primals_263', (primals_263.sum()/primals_263.nelement()).item(), primals_263.amax().item(), primals_263.amin().item())
        triton__26.run(buf330, buf328, primals_262, primals_263, buf333, buf331, buf332, buf334, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf330', (buf330.sum()/buf330.nelement()).item(), buf330.amax().item(), buf330.amin().item())
        print('triton__26', 'out_ptr0', 'buf333', (buf333.sum()/buf333.nelement()).item(), buf333.amax().item(), buf333.amin().item())
        print('triton__26', 'out_ptr1', 'buf331', (buf331.sum()/buf331.nelement()).item(), buf331.amax().item(), buf331.amin().item())
        print('triton__26', 'out_ptr2', 'buf332', (buf332.sum()/buf332.nelement()).item(), buf332.amax().item(), buf332.amin().item())
        print('triton__26', 'out_ptr3', 'buf334', (buf334.sum()/buf334.nelement()).item(), buf334.amax().item(), buf334.amin().item())
        del primals_262
        del primals_263
        buf335 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf328', (buf328.sum()/buf328.nelement()).item(), buf328.amax().item(), buf328.amin().item())
        print('triton__28', 'in_ptr1', 'buf330', (buf330.sum()/buf330.nelement()).item(), buf330.amax().item(), buf330.amin().item())
        print('triton__28', 'in_ptr2', 'buf331', (buf331.sum()/buf331.nelement()).item(), buf331.amax().item(), buf331.amin().item())
        print('triton__28', 'in_ptr3', 'primals_264', (primals_264.sum()/primals_264.nelement()).item(), primals_264.amax().item(), primals_264.amin().item())
        print('triton__28', 'in_ptr4', 'primals_265', (primals_265.sum()/primals_265.nelement()).item(), primals_265.amax().item(), primals_265.amin().item())
        print('triton__28', 'in_ptr5', 'buf311', (buf311.sum()/buf311.nelement()).item(), buf311.amax().item(), buf311.amin().item())
        triton__28.run(buf328, buf330, buf331, primals_264, primals_265, buf311, buf335, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf335', (buf335.sum()/buf335.nelement()).item(), buf335.amax().item(), buf335.amin().item())
        del primals_265
        buf336 = aten.convolution(buf335, primals_42, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf336, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf337 = buf323; del buf323  # reuse
        buf338 = buf337; del buf337  # reuse
        buf341 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf339 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf340 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf342 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf338', (buf338.sum()/buf338.nelement()).item(), buf338.amax().item(), buf338.amin().item())
        print('triton__24', 'in_ptr0', 'buf336', (buf336.sum()/buf336.nelement()).item(), buf336.amax().item(), buf336.amin().item())
        print('triton__24', 'in_ptr1', 'primals_267', (primals_267.sum()/primals_267.nelement()).item(), primals_267.amax().item(), primals_267.amin().item())
        print('triton__24', 'in_ptr2', 'primals_268', (primals_268.sum()/primals_268.nelement()).item(), primals_268.amax().item(), primals_268.amin().item())
        triton__24.run(buf338, buf336, primals_267, primals_268, buf341, buf339, buf340, buf342, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf338', (buf338.sum()/buf338.nelement()).item(), buf338.amax().item(), buf338.amin().item())
        print('triton__24', 'out_ptr0', 'buf341', (buf341.sum()/buf341.nelement()).item(), buf341.amax().item(), buf341.amin().item())
        print('triton__24', 'out_ptr1', 'buf339', (buf339.sum()/buf339.nelement()).item(), buf339.amax().item(), buf339.amin().item())
        print('triton__24', 'out_ptr2', 'buf340', (buf340.sum()/buf340.nelement()).item(), buf340.amax().item(), buf340.amin().item())
        print('triton__24', 'out_ptr3', 'buf342', (buf342.sum()/buf342.nelement()).item(), buf342.amax().item(), buf342.amin().item())
        del primals_267
        del primals_268
        buf343 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf336', (buf336.sum()/buf336.nelement()).item(), buf336.amax().item(), buf336.amin().item())
        print('triton__25', 'in_ptr1', 'buf338', (buf338.sum()/buf338.nelement()).item(), buf338.amax().item(), buf338.amin().item())
        print('triton__25', 'in_ptr2', 'buf339', (buf339.sum()/buf339.nelement()).item(), buf339.amax().item(), buf339.amin().item())
        print('triton__25', 'in_ptr3', 'primals_269', (primals_269.sum()/primals_269.nelement()).item(), primals_269.amax().item(), primals_269.amin().item())
        print('triton__25', 'in_ptr4', 'primals_270', (primals_270.sum()/primals_270.nelement()).item(), primals_270.amax().item(), primals_270.amin().item())
        triton__25.run(buf336, buf338, buf339, primals_269, primals_270, buf343, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf343', (buf343.sum()/buf343.nelement()).item(), buf343.amax().item(), buf343.amin().item())
        del primals_270
        buf344 = aten.convolution(buf343, primals_43, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf344, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf345 = buf339; del buf339  # reuse
        buf346 = buf345; del buf345  # reuse
        buf349 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf347 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf348 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf350 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf346', (buf346.sum()/buf346.nelement()).item(), buf346.amax().item(), buf346.amin().item())
        print('triton__24', 'in_ptr0', 'buf344', (buf344.sum()/buf344.nelement()).item(), buf344.amax().item(), buf344.amin().item())
        print('triton__24', 'in_ptr1', 'primals_272', (primals_272.sum()/primals_272.nelement()).item(), primals_272.amax().item(), primals_272.amin().item())
        print('triton__24', 'in_ptr2', 'primals_273', (primals_273.sum()/primals_273.nelement()).item(), primals_273.amax().item(), primals_273.amin().item())
        triton__24.run(buf346, buf344, primals_272, primals_273, buf349, buf347, buf348, buf350, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf346', (buf346.sum()/buf346.nelement()).item(), buf346.amax().item(), buf346.amin().item())
        print('triton__24', 'out_ptr0', 'buf349', (buf349.sum()/buf349.nelement()).item(), buf349.amax().item(), buf349.amin().item())
        print('triton__24', 'out_ptr1', 'buf347', (buf347.sum()/buf347.nelement()).item(), buf347.amax().item(), buf347.amin().item())
        print('triton__24', 'out_ptr2', 'buf348', (buf348.sum()/buf348.nelement()).item(), buf348.amax().item(), buf348.amin().item())
        print('triton__24', 'out_ptr3', 'buf350', (buf350.sum()/buf350.nelement()).item(), buf350.amax().item(), buf350.amin().item())
        del primals_272
        del primals_273
        buf351 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf344', (buf344.sum()/buf344.nelement()).item(), buf344.amax().item(), buf344.amin().item())
        print('triton__25', 'in_ptr1', 'buf346', (buf346.sum()/buf346.nelement()).item(), buf346.amax().item(), buf346.amin().item())
        print('triton__25', 'in_ptr2', 'buf347', (buf347.sum()/buf347.nelement()).item(), buf347.amax().item(), buf347.amin().item())
        print('triton__25', 'in_ptr3', 'primals_274', (primals_274.sum()/primals_274.nelement()).item(), primals_274.amax().item(), primals_274.amin().item())
        print('triton__25', 'in_ptr4', 'primals_275', (primals_275.sum()/primals_275.nelement()).item(), primals_275.amax().item(), primals_275.amin().item())
        triton__25.run(buf344, buf346, buf347, primals_274, primals_275, buf351, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf351', (buf351.sum()/buf351.nelement()).item(), buf351.amax().item(), buf351.amin().item())
        del primals_275
        buf352 = aten.convolution(buf351, primals_44, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf352, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf353 = buf331; del buf331  # reuse
        buf354 = buf353; del buf353  # reuse
        buf357 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf355 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf356 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf358 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf354', (buf354.sum()/buf354.nelement()).item(), buf354.amax().item(), buf354.amin().item())
        print('triton__26', 'in_ptr0', 'buf352', (buf352.sum()/buf352.nelement()).item(), buf352.amax().item(), buf352.amin().item())
        print('triton__26', 'in_ptr1', 'primals_277', (primals_277.sum()/primals_277.nelement()).item(), primals_277.amax().item(), primals_277.amin().item())
        print('triton__26', 'in_ptr2', 'primals_278', (primals_278.sum()/primals_278.nelement()).item(), primals_278.amax().item(), primals_278.amin().item())
        triton__26.run(buf354, buf352, primals_277, primals_278, buf357, buf355, buf356, buf358, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf354', (buf354.sum()/buf354.nelement()).item(), buf354.amax().item(), buf354.amin().item())
        print('triton__26', 'out_ptr0', 'buf357', (buf357.sum()/buf357.nelement()).item(), buf357.amax().item(), buf357.amin().item())
        print('triton__26', 'out_ptr1', 'buf355', (buf355.sum()/buf355.nelement()).item(), buf355.amax().item(), buf355.amin().item())
        print('triton__26', 'out_ptr2', 'buf356', (buf356.sum()/buf356.nelement()).item(), buf356.amax().item(), buf356.amin().item())
        print('triton__26', 'out_ptr3', 'buf358', (buf358.sum()/buf358.nelement()).item(), buf358.amax().item(), buf358.amin().item())
        del primals_277
        del primals_278
        buf359 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf352', (buf352.sum()/buf352.nelement()).item(), buf352.amax().item(), buf352.amin().item())
        print('triton__28', 'in_ptr1', 'buf354', (buf354.sum()/buf354.nelement()).item(), buf354.amax().item(), buf354.amin().item())
        print('triton__28', 'in_ptr2', 'buf355', (buf355.sum()/buf355.nelement()).item(), buf355.amax().item(), buf355.amin().item())
        print('triton__28', 'in_ptr3', 'primals_279', (primals_279.sum()/primals_279.nelement()).item(), primals_279.amax().item(), primals_279.amin().item())
        print('triton__28', 'in_ptr4', 'primals_280', (primals_280.sum()/primals_280.nelement()).item(), primals_280.amax().item(), primals_280.amin().item())
        print('triton__28', 'in_ptr5', 'buf335', (buf335.sum()/buf335.nelement()).item(), buf335.amax().item(), buf335.amin().item())
        triton__28.run(buf352, buf354, buf355, primals_279, primals_280, buf335, buf359, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf359', (buf359.sum()/buf359.nelement()).item(), buf359.amax().item(), buf359.amin().item())
        del primals_280
        buf360 = aten.convolution(buf359, primals_45, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf360, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf361 = buf347; del buf347  # reuse
        buf362 = buf361; del buf361  # reuse
        buf365 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf363 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf364 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf366 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf362', (buf362.sum()/buf362.nelement()).item(), buf362.amax().item(), buf362.amin().item())
        print('triton__24', 'in_ptr0', 'buf360', (buf360.sum()/buf360.nelement()).item(), buf360.amax().item(), buf360.amin().item())
        print('triton__24', 'in_ptr1', 'primals_282', (primals_282.sum()/primals_282.nelement()).item(), primals_282.amax().item(), primals_282.amin().item())
        print('triton__24', 'in_ptr2', 'primals_283', (primals_283.sum()/primals_283.nelement()).item(), primals_283.amax().item(), primals_283.amin().item())
        triton__24.run(buf362, buf360, primals_282, primals_283, buf365, buf363, buf364, buf366, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf362', (buf362.sum()/buf362.nelement()).item(), buf362.amax().item(), buf362.amin().item())
        print('triton__24', 'out_ptr0', 'buf365', (buf365.sum()/buf365.nelement()).item(), buf365.amax().item(), buf365.amin().item())
        print('triton__24', 'out_ptr1', 'buf363', (buf363.sum()/buf363.nelement()).item(), buf363.amax().item(), buf363.amin().item())
        print('triton__24', 'out_ptr2', 'buf364', (buf364.sum()/buf364.nelement()).item(), buf364.amax().item(), buf364.amin().item())
        print('triton__24', 'out_ptr3', 'buf366', (buf366.sum()/buf366.nelement()).item(), buf366.amax().item(), buf366.amin().item())
        del primals_282
        del primals_283
        buf367 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf360', (buf360.sum()/buf360.nelement()).item(), buf360.amax().item(), buf360.amin().item())
        print('triton__25', 'in_ptr1', 'buf362', (buf362.sum()/buf362.nelement()).item(), buf362.amax().item(), buf362.amin().item())
        print('triton__25', 'in_ptr2', 'buf363', (buf363.sum()/buf363.nelement()).item(), buf363.amax().item(), buf363.amin().item())
        print('triton__25', 'in_ptr3', 'primals_284', (primals_284.sum()/primals_284.nelement()).item(), primals_284.amax().item(), primals_284.amin().item())
        print('triton__25', 'in_ptr4', 'primals_285', (primals_285.sum()/primals_285.nelement()).item(), primals_285.amax().item(), primals_285.amin().item())
        triton__25.run(buf360, buf362, buf363, primals_284, primals_285, buf367, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf367', (buf367.sum()/buf367.nelement()).item(), buf367.amax().item(), buf367.amin().item())
        del primals_285
        buf368 = aten.convolution(buf367, primals_46, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf368, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf369 = buf363; del buf363  # reuse
        buf370 = buf369; del buf369  # reuse
        buf373 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf371 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf372 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf374 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf370', (buf370.sum()/buf370.nelement()).item(), buf370.amax().item(), buf370.amin().item())
        print('triton__24', 'in_ptr0', 'buf368', (buf368.sum()/buf368.nelement()).item(), buf368.amax().item(), buf368.amin().item())
        print('triton__24', 'in_ptr1', 'primals_287', (primals_287.sum()/primals_287.nelement()).item(), primals_287.amax().item(), primals_287.amin().item())
        print('triton__24', 'in_ptr2', 'primals_288', (primals_288.sum()/primals_288.nelement()).item(), primals_288.amax().item(), primals_288.amin().item())
        triton__24.run(buf370, buf368, primals_287, primals_288, buf373, buf371, buf372, buf374, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf370', (buf370.sum()/buf370.nelement()).item(), buf370.amax().item(), buf370.amin().item())
        print('triton__24', 'out_ptr0', 'buf373', (buf373.sum()/buf373.nelement()).item(), buf373.amax().item(), buf373.amin().item())
        print('triton__24', 'out_ptr1', 'buf371', (buf371.sum()/buf371.nelement()).item(), buf371.amax().item(), buf371.amin().item())
        print('triton__24', 'out_ptr2', 'buf372', (buf372.sum()/buf372.nelement()).item(), buf372.amax().item(), buf372.amin().item())
        print('triton__24', 'out_ptr3', 'buf374', (buf374.sum()/buf374.nelement()).item(), buf374.amax().item(), buf374.amin().item())
        del primals_287
        del primals_288
        buf375 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf368', (buf368.sum()/buf368.nelement()).item(), buf368.amax().item(), buf368.amin().item())
        print('triton__25', 'in_ptr1', 'buf370', (buf370.sum()/buf370.nelement()).item(), buf370.amax().item(), buf370.amin().item())
        print('triton__25', 'in_ptr2', 'buf371', (buf371.sum()/buf371.nelement()).item(), buf371.amax().item(), buf371.amin().item())
        print('triton__25', 'in_ptr3', 'primals_289', (primals_289.sum()/primals_289.nelement()).item(), primals_289.amax().item(), primals_289.amin().item())
        print('triton__25', 'in_ptr4', 'primals_290', (primals_290.sum()/primals_290.nelement()).item(), primals_290.amax().item(), primals_290.amin().item())
        triton__25.run(buf368, buf370, buf371, primals_289, primals_290, buf375, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf375', (buf375.sum()/buf375.nelement()).item(), buf375.amax().item(), buf375.amin().item())
        del primals_290
        buf376 = aten.convolution(buf375, primals_47, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf376, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf377 = buf355; del buf355  # reuse
        buf378 = buf377; del buf377  # reuse
        buf381 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf379 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf380 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf382 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf378', (buf378.sum()/buf378.nelement()).item(), buf378.amax().item(), buf378.amin().item())
        print('triton__26', 'in_ptr0', 'buf376', (buf376.sum()/buf376.nelement()).item(), buf376.amax().item(), buf376.amin().item())
        print('triton__26', 'in_ptr1', 'primals_292', (primals_292.sum()/primals_292.nelement()).item(), primals_292.amax().item(), primals_292.amin().item())
        print('triton__26', 'in_ptr2', 'primals_293', (primals_293.sum()/primals_293.nelement()).item(), primals_293.amax().item(), primals_293.amin().item())
        triton__26.run(buf378, buf376, primals_292, primals_293, buf381, buf379, buf380, buf382, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf378', (buf378.sum()/buf378.nelement()).item(), buf378.amax().item(), buf378.amin().item())
        print('triton__26', 'out_ptr0', 'buf381', (buf381.sum()/buf381.nelement()).item(), buf381.amax().item(), buf381.amin().item())
        print('triton__26', 'out_ptr1', 'buf379', (buf379.sum()/buf379.nelement()).item(), buf379.amax().item(), buf379.amin().item())
        print('triton__26', 'out_ptr2', 'buf380', (buf380.sum()/buf380.nelement()).item(), buf380.amax().item(), buf380.amin().item())
        print('triton__26', 'out_ptr3', 'buf382', (buf382.sum()/buf382.nelement()).item(), buf382.amax().item(), buf382.amin().item())
        del primals_292
        del primals_293
        buf383 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf376', (buf376.sum()/buf376.nelement()).item(), buf376.amax().item(), buf376.amin().item())
        print('triton__28', 'in_ptr1', 'buf378', (buf378.sum()/buf378.nelement()).item(), buf378.amax().item(), buf378.amin().item())
        print('triton__28', 'in_ptr2', 'buf379', (buf379.sum()/buf379.nelement()).item(), buf379.amax().item(), buf379.amin().item())
        print('triton__28', 'in_ptr3', 'primals_294', (primals_294.sum()/primals_294.nelement()).item(), primals_294.amax().item(), primals_294.amin().item())
        print('triton__28', 'in_ptr4', 'primals_295', (primals_295.sum()/primals_295.nelement()).item(), primals_295.amax().item(), primals_295.amin().item())
        print('triton__28', 'in_ptr5', 'buf359', (buf359.sum()/buf359.nelement()).item(), buf359.amax().item(), buf359.amin().item())
        triton__28.run(buf376, buf378, buf379, primals_294, primals_295, buf359, buf383, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf383', (buf383.sum()/buf383.nelement()).item(), buf383.amax().item(), buf383.amin().item())
        del primals_295
        buf384 = aten.convolution(buf383, primals_48, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf384, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf385 = buf371; del buf371  # reuse
        buf386 = buf385; del buf385  # reuse
        buf389 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf387 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf388 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf390 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf386', (buf386.sum()/buf386.nelement()).item(), buf386.amax().item(), buf386.amin().item())
        print('triton__24', 'in_ptr0', 'buf384', (buf384.sum()/buf384.nelement()).item(), buf384.amax().item(), buf384.amin().item())
        print('triton__24', 'in_ptr1', 'primals_297', (primals_297.sum()/primals_297.nelement()).item(), primals_297.amax().item(), primals_297.amin().item())
        print('triton__24', 'in_ptr2', 'primals_298', (primals_298.sum()/primals_298.nelement()).item(), primals_298.amax().item(), primals_298.amin().item())
        triton__24.run(buf386, buf384, primals_297, primals_298, buf389, buf387, buf388, buf390, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf386', (buf386.sum()/buf386.nelement()).item(), buf386.amax().item(), buf386.amin().item())
        print('triton__24', 'out_ptr0', 'buf389', (buf389.sum()/buf389.nelement()).item(), buf389.amax().item(), buf389.amin().item())
        print('triton__24', 'out_ptr1', 'buf387', (buf387.sum()/buf387.nelement()).item(), buf387.amax().item(), buf387.amin().item())
        print('triton__24', 'out_ptr2', 'buf388', (buf388.sum()/buf388.nelement()).item(), buf388.amax().item(), buf388.amin().item())
        print('triton__24', 'out_ptr3', 'buf390', (buf390.sum()/buf390.nelement()).item(), buf390.amax().item(), buf390.amin().item())
        del primals_297
        del primals_298
        buf391 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf384', (buf384.sum()/buf384.nelement()).item(), buf384.amax().item(), buf384.amin().item())
        print('triton__25', 'in_ptr1', 'buf386', (buf386.sum()/buf386.nelement()).item(), buf386.amax().item(), buf386.amin().item())
        print('triton__25', 'in_ptr2', 'buf387', (buf387.sum()/buf387.nelement()).item(), buf387.amax().item(), buf387.amin().item())
        print('triton__25', 'in_ptr3', 'primals_299', (primals_299.sum()/primals_299.nelement()).item(), primals_299.amax().item(), primals_299.amin().item())
        print('triton__25', 'in_ptr4', 'primals_300', (primals_300.sum()/primals_300.nelement()).item(), primals_300.amax().item(), primals_300.amin().item())
        triton__25.run(buf384, buf386, buf387, primals_299, primals_300, buf391, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf391', (buf391.sum()/buf391.nelement()).item(), buf391.amax().item(), buf391.amin().item())
        del primals_300
        buf392 = aten.convolution(buf391, primals_49, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf392, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf393 = buf387; del buf387  # reuse
        buf394 = buf393; del buf393  # reuse
        buf397 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf395 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf396 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf398 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf394', (buf394.sum()/buf394.nelement()).item(), buf394.amax().item(), buf394.amin().item())
        print('triton__24', 'in_ptr0', 'buf392', (buf392.sum()/buf392.nelement()).item(), buf392.amax().item(), buf392.amin().item())
        print('triton__24', 'in_ptr1', 'primals_302', (primals_302.sum()/primals_302.nelement()).item(), primals_302.amax().item(), primals_302.amin().item())
        print('triton__24', 'in_ptr2', 'primals_303', (primals_303.sum()/primals_303.nelement()).item(), primals_303.amax().item(), primals_303.amin().item())
        triton__24.run(buf394, buf392, primals_302, primals_303, buf397, buf395, buf396, buf398, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf394', (buf394.sum()/buf394.nelement()).item(), buf394.amax().item(), buf394.amin().item())
        print('triton__24', 'out_ptr0', 'buf397', (buf397.sum()/buf397.nelement()).item(), buf397.amax().item(), buf397.amin().item())
        print('triton__24', 'out_ptr1', 'buf395', (buf395.sum()/buf395.nelement()).item(), buf395.amax().item(), buf395.amin().item())
        print('triton__24', 'out_ptr2', 'buf396', (buf396.sum()/buf396.nelement()).item(), buf396.amax().item(), buf396.amin().item())
        print('triton__24', 'out_ptr3', 'buf398', (buf398.sum()/buf398.nelement()).item(), buf398.amax().item(), buf398.amin().item())
        del primals_302
        del primals_303
        buf399 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf392', (buf392.sum()/buf392.nelement()).item(), buf392.amax().item(), buf392.amin().item())
        print('triton__25', 'in_ptr1', 'buf394', (buf394.sum()/buf394.nelement()).item(), buf394.amax().item(), buf394.amin().item())
        print('triton__25', 'in_ptr2', 'buf395', (buf395.sum()/buf395.nelement()).item(), buf395.amax().item(), buf395.amin().item())
        print('triton__25', 'in_ptr3', 'primals_304', (primals_304.sum()/primals_304.nelement()).item(), primals_304.amax().item(), primals_304.amin().item())
        print('triton__25', 'in_ptr4', 'primals_305', (primals_305.sum()/primals_305.nelement()).item(), primals_305.amax().item(), primals_305.amin().item())
        triton__25.run(buf392, buf394, buf395, primals_304, primals_305, buf399, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf399', (buf399.sum()/buf399.nelement()).item(), buf399.amax().item(), buf399.amin().item())
        del primals_305
        buf400 = aten.convolution(buf399, primals_50, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf400, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf401 = buf379; del buf379  # reuse
        buf402 = buf401; del buf401  # reuse
        buf405 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf403 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf404 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf406 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf402', (buf402.sum()/buf402.nelement()).item(), buf402.amax().item(), buf402.amin().item())
        print('triton__26', 'in_ptr0', 'buf400', (buf400.sum()/buf400.nelement()).item(), buf400.amax().item(), buf400.amin().item())
        print('triton__26', 'in_ptr1', 'primals_307', (primals_307.sum()/primals_307.nelement()).item(), primals_307.amax().item(), primals_307.amin().item())
        print('triton__26', 'in_ptr2', 'primals_308', (primals_308.sum()/primals_308.nelement()).item(), primals_308.amax().item(), primals_308.amin().item())
        triton__26.run(buf402, buf400, primals_307, primals_308, buf405, buf403, buf404, buf406, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf402', (buf402.sum()/buf402.nelement()).item(), buf402.amax().item(), buf402.amin().item())
        print('triton__26', 'out_ptr0', 'buf405', (buf405.sum()/buf405.nelement()).item(), buf405.amax().item(), buf405.amin().item())
        print('triton__26', 'out_ptr1', 'buf403', (buf403.sum()/buf403.nelement()).item(), buf403.amax().item(), buf403.amin().item())
        print('triton__26', 'out_ptr2', 'buf404', (buf404.sum()/buf404.nelement()).item(), buf404.amax().item(), buf404.amin().item())
        print('triton__26', 'out_ptr3', 'buf406', (buf406.sum()/buf406.nelement()).item(), buf406.amax().item(), buf406.amin().item())
        del primals_307
        del primals_308
        buf407 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf400', (buf400.sum()/buf400.nelement()).item(), buf400.amax().item(), buf400.amin().item())
        print('triton__28', 'in_ptr1', 'buf402', (buf402.sum()/buf402.nelement()).item(), buf402.amax().item(), buf402.amin().item())
        print('triton__28', 'in_ptr2', 'buf403', (buf403.sum()/buf403.nelement()).item(), buf403.amax().item(), buf403.amin().item())
        print('triton__28', 'in_ptr3', 'primals_309', (primals_309.sum()/primals_309.nelement()).item(), primals_309.amax().item(), primals_309.amin().item())
        print('triton__28', 'in_ptr4', 'primals_310', (primals_310.sum()/primals_310.nelement()).item(), primals_310.amax().item(), primals_310.amin().item())
        print('triton__28', 'in_ptr5', 'buf383', (buf383.sum()/buf383.nelement()).item(), buf383.amax().item(), buf383.amin().item())
        triton__28.run(buf400, buf402, buf403, primals_309, primals_310, buf383, buf407, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf407', (buf407.sum()/buf407.nelement()).item(), buf407.amax().item(), buf407.amin().item())
        del primals_310
        buf408 = aten.convolution(buf407, primals_51, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf408, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf409 = buf395; del buf395  # reuse
        buf410 = buf409; del buf409  # reuse
        buf413 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf411 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf412 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf414 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf410', (buf410.sum()/buf410.nelement()).item(), buf410.amax().item(), buf410.amin().item())
        print('triton__24', 'in_ptr0', 'buf408', (buf408.sum()/buf408.nelement()).item(), buf408.amax().item(), buf408.amin().item())
        print('triton__24', 'in_ptr1', 'primals_312', (primals_312.sum()/primals_312.nelement()).item(), primals_312.amax().item(), primals_312.amin().item())
        print('triton__24', 'in_ptr2', 'primals_313', (primals_313.sum()/primals_313.nelement()).item(), primals_313.amax().item(), primals_313.amin().item())
        triton__24.run(buf410, buf408, primals_312, primals_313, buf413, buf411, buf412, buf414, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf410', (buf410.sum()/buf410.nelement()).item(), buf410.amax().item(), buf410.amin().item())
        print('triton__24', 'out_ptr0', 'buf413', (buf413.sum()/buf413.nelement()).item(), buf413.amax().item(), buf413.amin().item())
        print('triton__24', 'out_ptr1', 'buf411', (buf411.sum()/buf411.nelement()).item(), buf411.amax().item(), buf411.amin().item())
        print('triton__24', 'out_ptr2', 'buf412', (buf412.sum()/buf412.nelement()).item(), buf412.amax().item(), buf412.amin().item())
        print('triton__24', 'out_ptr3', 'buf414', (buf414.sum()/buf414.nelement()).item(), buf414.amax().item(), buf414.amin().item())
        del primals_312
        del primals_313
        buf415 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf408', (buf408.sum()/buf408.nelement()).item(), buf408.amax().item(), buf408.amin().item())
        print('triton__25', 'in_ptr1', 'buf410', (buf410.sum()/buf410.nelement()).item(), buf410.amax().item(), buf410.amin().item())
        print('triton__25', 'in_ptr2', 'buf411', (buf411.sum()/buf411.nelement()).item(), buf411.amax().item(), buf411.amin().item())
        print('triton__25', 'in_ptr3', 'primals_314', (primals_314.sum()/primals_314.nelement()).item(), primals_314.amax().item(), primals_314.amin().item())
        print('triton__25', 'in_ptr4', 'primals_315', (primals_315.sum()/primals_315.nelement()).item(), primals_315.amax().item(), primals_315.amin().item())
        triton__25.run(buf408, buf410, buf411, primals_314, primals_315, buf415, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf415', (buf415.sum()/buf415.nelement()).item(), buf415.amax().item(), buf415.amin().item())
        del primals_315
        buf416 = aten.convolution(buf415, primals_52, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf416, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf417 = buf411; del buf411  # reuse
        buf418 = buf417; del buf417  # reuse
        buf421 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf419 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf420 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf422 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf418', (buf418.sum()/buf418.nelement()).item(), buf418.amax().item(), buf418.amin().item())
        print('triton__24', 'in_ptr0', 'buf416', (buf416.sum()/buf416.nelement()).item(), buf416.amax().item(), buf416.amin().item())
        print('triton__24', 'in_ptr1', 'primals_317', (primals_317.sum()/primals_317.nelement()).item(), primals_317.amax().item(), primals_317.amin().item())
        print('triton__24', 'in_ptr2', 'primals_318', (primals_318.sum()/primals_318.nelement()).item(), primals_318.amax().item(), primals_318.amin().item())
        triton__24.run(buf418, buf416, primals_317, primals_318, buf421, buf419, buf420, buf422, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf418', (buf418.sum()/buf418.nelement()).item(), buf418.amax().item(), buf418.amin().item())
        print('triton__24', 'out_ptr0', 'buf421', (buf421.sum()/buf421.nelement()).item(), buf421.amax().item(), buf421.amin().item())
        print('triton__24', 'out_ptr1', 'buf419', (buf419.sum()/buf419.nelement()).item(), buf419.amax().item(), buf419.amin().item())
        print('triton__24', 'out_ptr2', 'buf420', (buf420.sum()/buf420.nelement()).item(), buf420.amax().item(), buf420.amin().item())
        print('triton__24', 'out_ptr3', 'buf422', (buf422.sum()/buf422.nelement()).item(), buf422.amax().item(), buf422.amin().item())
        del primals_317
        del primals_318
        buf423 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf416', (buf416.sum()/buf416.nelement()).item(), buf416.amax().item(), buf416.amin().item())
        print('triton__25', 'in_ptr1', 'buf418', (buf418.sum()/buf418.nelement()).item(), buf418.amax().item(), buf418.amin().item())
        print('triton__25', 'in_ptr2', 'buf419', (buf419.sum()/buf419.nelement()).item(), buf419.amax().item(), buf419.amin().item())
        print('triton__25', 'in_ptr3', 'primals_319', (primals_319.sum()/primals_319.nelement()).item(), primals_319.amax().item(), primals_319.amin().item())
        print('triton__25', 'in_ptr4', 'primals_320', (primals_320.sum()/primals_320.nelement()).item(), primals_320.amax().item(), primals_320.amin().item())
        triton__25.run(buf416, buf418, buf419, primals_319, primals_320, buf423, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf423', (buf423.sum()/buf423.nelement()).item(), buf423.amax().item(), buf423.amin().item())
        del primals_320
        buf424 = aten.convolution(buf423, primals_53, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf424, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf425 = buf403; del buf403  # reuse
        buf426 = buf425; del buf425  # reuse
        buf429 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf427 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf428 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf430 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf426', (buf426.sum()/buf426.nelement()).item(), buf426.amax().item(), buf426.amin().item())
        print('triton__26', 'in_ptr0', 'buf424', (buf424.sum()/buf424.nelement()).item(), buf424.amax().item(), buf424.amin().item())
        print('triton__26', 'in_ptr1', 'primals_322', (primals_322.sum()/primals_322.nelement()).item(), primals_322.amax().item(), primals_322.amin().item())
        print('triton__26', 'in_ptr2', 'primals_323', (primals_323.sum()/primals_323.nelement()).item(), primals_323.amax().item(), primals_323.amin().item())
        triton__26.run(buf426, buf424, primals_322, primals_323, buf429, buf427, buf428, buf430, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf426', (buf426.sum()/buf426.nelement()).item(), buf426.amax().item(), buf426.amin().item())
        print('triton__26', 'out_ptr0', 'buf429', (buf429.sum()/buf429.nelement()).item(), buf429.amax().item(), buf429.amin().item())
        print('triton__26', 'out_ptr1', 'buf427', (buf427.sum()/buf427.nelement()).item(), buf427.amax().item(), buf427.amin().item())
        print('triton__26', 'out_ptr2', 'buf428', (buf428.sum()/buf428.nelement()).item(), buf428.amax().item(), buf428.amin().item())
        print('triton__26', 'out_ptr3', 'buf430', (buf430.sum()/buf430.nelement()).item(), buf430.amax().item(), buf430.amin().item())
        del primals_322
        del primals_323
        buf431 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf424', (buf424.sum()/buf424.nelement()).item(), buf424.amax().item(), buf424.amin().item())
        print('triton__28', 'in_ptr1', 'buf426', (buf426.sum()/buf426.nelement()).item(), buf426.amax().item(), buf426.amin().item())
        print('triton__28', 'in_ptr2', 'buf427', (buf427.sum()/buf427.nelement()).item(), buf427.amax().item(), buf427.amin().item())
        print('triton__28', 'in_ptr3', 'primals_324', (primals_324.sum()/primals_324.nelement()).item(), primals_324.amax().item(), primals_324.amin().item())
        print('triton__28', 'in_ptr4', 'primals_325', (primals_325.sum()/primals_325.nelement()).item(), primals_325.amax().item(), primals_325.amin().item())
        print('triton__28', 'in_ptr5', 'buf407', (buf407.sum()/buf407.nelement()).item(), buf407.amax().item(), buf407.amin().item())
        triton__28.run(buf424, buf426, buf427, primals_324, primals_325, buf407, buf431, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf431', (buf431.sum()/buf431.nelement()).item(), buf431.amax().item(), buf431.amin().item())
        del primals_325
        buf432 = aten.convolution(buf431, primals_54, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf432, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf433 = buf419; del buf419  # reuse
        buf434 = buf433; del buf433  # reuse
        buf437 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf435 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf436 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf438 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf434', (buf434.sum()/buf434.nelement()).item(), buf434.amax().item(), buf434.amin().item())
        print('triton__24', 'in_ptr0', 'buf432', (buf432.sum()/buf432.nelement()).item(), buf432.amax().item(), buf432.amin().item())
        print('triton__24', 'in_ptr1', 'primals_327', (primals_327.sum()/primals_327.nelement()).item(), primals_327.amax().item(), primals_327.amin().item())
        print('triton__24', 'in_ptr2', 'primals_328', (primals_328.sum()/primals_328.nelement()).item(), primals_328.amax().item(), primals_328.amin().item())
        triton__24.run(buf434, buf432, primals_327, primals_328, buf437, buf435, buf436, buf438, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf434', (buf434.sum()/buf434.nelement()).item(), buf434.amax().item(), buf434.amin().item())
        print('triton__24', 'out_ptr0', 'buf437', (buf437.sum()/buf437.nelement()).item(), buf437.amax().item(), buf437.amin().item())
        print('triton__24', 'out_ptr1', 'buf435', (buf435.sum()/buf435.nelement()).item(), buf435.amax().item(), buf435.amin().item())
        print('triton__24', 'out_ptr2', 'buf436', (buf436.sum()/buf436.nelement()).item(), buf436.amax().item(), buf436.amin().item())
        print('triton__24', 'out_ptr3', 'buf438', (buf438.sum()/buf438.nelement()).item(), buf438.amax().item(), buf438.amin().item())
        del primals_327
        del primals_328
        buf439 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf432', (buf432.sum()/buf432.nelement()).item(), buf432.amax().item(), buf432.amin().item())
        print('triton__25', 'in_ptr1', 'buf434', (buf434.sum()/buf434.nelement()).item(), buf434.amax().item(), buf434.amin().item())
        print('triton__25', 'in_ptr2', 'buf435', (buf435.sum()/buf435.nelement()).item(), buf435.amax().item(), buf435.amin().item())
        print('triton__25', 'in_ptr3', 'primals_329', (primals_329.sum()/primals_329.nelement()).item(), primals_329.amax().item(), primals_329.amin().item())
        print('triton__25', 'in_ptr4', 'primals_330', (primals_330.sum()/primals_330.nelement()).item(), primals_330.amax().item(), primals_330.amin().item())
        triton__25.run(buf432, buf434, buf435, primals_329, primals_330, buf439, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf439', (buf439.sum()/buf439.nelement()).item(), buf439.amax().item(), buf439.amin().item())
        del primals_330
        buf440 = aten.convolution(buf439, primals_55, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1920)
        assert_size_stride(buf440, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf441 = buf435; del buf435  # reuse
        buf442 = buf441; del buf441  # reuse
        buf445 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf443 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf444 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf446 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__24', 'in_out_ptr0', 'buf442', (buf442.sum()/buf442.nelement()).item(), buf442.amax().item(), buf442.amin().item())
        print('triton__24', 'in_ptr0', 'buf440', (buf440.sum()/buf440.nelement()).item(), buf440.amax().item(), buf440.amin().item())
        print('triton__24', 'in_ptr1', 'primals_332', (primals_332.sum()/primals_332.nelement()).item(), primals_332.amax().item(), primals_332.amin().item())
        print('triton__24', 'in_ptr2', 'primals_333', (primals_333.sum()/primals_333.nelement()).item(), primals_333.amax().item(), primals_333.amin().item())
        triton__24.run(buf442, buf440, primals_332, primals_333, buf445, buf443, buf444, buf446, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf442', (buf442.sum()/buf442.nelement()).item(), buf442.amax().item(), buf442.amin().item())
        print('triton__24', 'out_ptr0', 'buf445', (buf445.sum()/buf445.nelement()).item(), buf445.amax().item(), buf445.amin().item())
        print('triton__24', 'out_ptr1', 'buf443', (buf443.sum()/buf443.nelement()).item(), buf443.amax().item(), buf443.amin().item())
        print('triton__24', 'out_ptr2', 'buf444', (buf444.sum()/buf444.nelement()).item(), buf444.amax().item(), buf444.amin().item())
        print('triton__24', 'out_ptr3', 'buf446', (buf446.sum()/buf446.nelement()).item(), buf446.amax().item(), buf446.amin().item())
        del primals_332
        del primals_333
        buf447 = empty_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf440', (buf440.sum()/buf440.nelement()).item(), buf440.amax().item(), buf440.amin().item())
        print('triton__25', 'in_ptr1', 'buf442', (buf442.sum()/buf442.nelement()).item(), buf442.amax().item(), buf442.amin().item())
        print('triton__25', 'in_ptr2', 'buf443', (buf443.sum()/buf443.nelement()).item(), buf443.amax().item(), buf443.amin().item())
        print('triton__25', 'in_ptr3', 'primals_334', (primals_334.sum()/primals_334.nelement()).item(), primals_334.amax().item(), primals_334.amin().item())
        print('triton__25', 'in_ptr4', 'primals_335', (primals_335.sum()/primals_335.nelement()).item(), primals_335.amax().item(), primals_335.amin().item())
        triton__25.run(buf440, buf442, buf443, primals_334, primals_335, buf447, 983040, grid=grid(983040), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf447', (buf447.sum()/buf447.nelement()).item(), buf447.amax().item(), buf447.amin().item())
        del buf443
        del primals_335
        buf448 = aten.convolution(buf447, primals_56, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf448, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf449 = buf427; del buf427  # reuse
        buf450 = buf449; del buf449  # reuse
        buf453 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf451 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf452 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf454 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__26', 'in_out_ptr0', 'buf450', (buf450.sum()/buf450.nelement()).item(), buf450.amax().item(), buf450.amin().item())
        print('triton__26', 'in_ptr0', 'buf448', (buf448.sum()/buf448.nelement()).item(), buf448.amax().item(), buf448.amin().item())
        print('triton__26', 'in_ptr1', 'primals_337', (primals_337.sum()/primals_337.nelement()).item(), primals_337.amax().item(), primals_337.amin().item())
        print('triton__26', 'in_ptr2', 'primals_338', (primals_338.sum()/primals_338.nelement()).item(), primals_338.amax().item(), primals_338.amin().item())
        triton__26.run(buf450, buf448, primals_337, primals_338, buf453, buf451, buf452, buf454, 640, 512, grid=grid(640), stream=stream0)
        print('triton__26', 'in_out_ptr0', 'buf450', (buf450.sum()/buf450.nelement()).item(), buf450.amax().item(), buf450.amin().item())
        print('triton__26', 'out_ptr0', 'buf453', (buf453.sum()/buf453.nelement()).item(), buf453.amax().item(), buf453.amin().item())
        print('triton__26', 'out_ptr1', 'buf451', (buf451.sum()/buf451.nelement()).item(), buf451.amax().item(), buf451.amin().item())
        print('triton__26', 'out_ptr2', 'buf452', (buf452.sum()/buf452.nelement()).item(), buf452.amax().item(), buf452.amin().item())
        print('triton__26', 'out_ptr3', 'buf454', (buf454.sum()/buf454.nelement()).item(), buf454.amax().item(), buf454.amin().item())
        del primals_337
        del primals_338
        buf455 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__28', 'in_ptr0', 'buf448', (buf448.sum()/buf448.nelement()).item(), buf448.amax().item(), buf448.amin().item())
        print('triton__28', 'in_ptr1', 'buf450', (buf450.sum()/buf450.nelement()).item(), buf450.amax().item(), buf450.amin().item())
        print('triton__28', 'in_ptr2', 'buf451', (buf451.sum()/buf451.nelement()).item(), buf451.amax().item(), buf451.amin().item())
        print('triton__28', 'in_ptr3', 'primals_339', (primals_339.sum()/primals_339.nelement()).item(), primals_339.amax().item(), primals_339.amin().item())
        print('triton__28', 'in_ptr4', 'primals_340', (primals_340.sum()/primals_340.nelement()).item(), primals_340.amax().item(), primals_340.amin().item())
        print('triton__28', 'in_ptr5', 'buf431', (buf431.sum()/buf431.nelement()).item(), buf431.amax().item(), buf431.amin().item())
        triton__28.run(buf448, buf450, buf451, primals_339, primals_340, buf431, buf455, 327680, grid=grid(327680), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf455', (buf455.sum()/buf455.nelement()).item(), buf455.amax().item(), buf455.amin().item())
        del buf451
        del primals_340
        buf456 = aten.convolution(buf455, primals_57, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf456, (8, 2560, 8, 8), (163840, 64, 8, 1))
        buf457 = empty_strided((1, 2560, 1, 1), (2560, 1, 2560, 2560), device='cuda', dtype=torch.float32)
        buf458 = buf457; del buf457  # reuse
        buf461 = empty_strided((2560, ), (1, ), device='cuda', dtype=torch.float32)
        buf459 = empty_strided((1, 2560, 1, 1), (2560, 1, 2560, 2560), device='cuda', dtype=torch.float32)
        buf460 = empty_strided((2560, ), (1, ), device='cuda', dtype=torch.float32)
        buf462 = empty_strided((2560, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__29', 'in_out_ptr0', 'buf458', (buf458.sum()/buf458.nelement()).item(), buf458.amax().item(), buf458.amin().item())
        print('triton__29', 'in_ptr0', 'buf456', (buf456.sum()/buf456.nelement()).item(), buf456.amax().item(), buf456.amin().item())
        print('triton__29', 'in_ptr1', 'primals_342', (primals_342.sum()/primals_342.nelement()).item(), primals_342.amax().item(), primals_342.amin().item())
        print('triton__29', 'in_ptr2', 'primals_343', (primals_343.sum()/primals_343.nelement()).item(), primals_343.amax().item(), primals_343.amin().item())
        triton__29.run(buf458, buf456, primals_342, primals_343, buf461, buf459, buf460, buf462, 2560, 512, grid=grid(2560), stream=stream0)
        print('triton__29', 'in_out_ptr0', 'buf458', (buf458.sum()/buf458.nelement()).item(), buf458.amax().item(), buf458.amin().item())
        print('triton__29', 'out_ptr0', 'buf461', (buf461.sum()/buf461.nelement()).item(), buf461.amax().item(), buf461.amin().item())
        print('triton__29', 'out_ptr1', 'buf459', (buf459.sum()/buf459.nelement()).item(), buf459.amax().item(), buf459.amin().item())
        print('triton__29', 'out_ptr2', 'buf460', (buf460.sum()/buf460.nelement()).item(), buf460.amax().item(), buf460.amin().item())
        print('triton__29', 'out_ptr3', 'buf462', (buf462.sum()/buf462.nelement()).item(), buf462.amax().item(), buf462.amin().item())
        del primals_342
        del primals_343
        buf467 = empty_strided((8, 2560, 8, 8), (163840, 64, 8, 1), device='cuda', dtype=torch.bool)
        buf464 = empty_strided((8, 2560, 1, 1), (2560, 1, 20480, 20480), device='cuda', dtype=torch.float32)
        buf465 = as_strided(buf464, (8, 2560), (2560, 1)); del buf464  # reuse
        print('triton__30', 'in_out_ptr0', 'buf465', (buf465.sum()/buf465.nelement()).item(), buf465.amax().item(), buf465.amin().item())
        print('triton__30', 'in_ptr0', 'buf456', (buf456.sum()/buf456.nelement()).item(), buf456.amax().item(), buf456.amin().item())
        print('triton__30', 'in_ptr1', 'buf458', (buf458.sum()/buf458.nelement()).item(), buf458.amax().item(), buf458.amin().item())
        print('triton__30', 'in_ptr2', 'buf459', (buf459.sum()/buf459.nelement()).item(), buf459.amax().item(), buf459.amin().item())
        print('triton__30', 'in_ptr3', 'primals_344', (primals_344.sum()/primals_344.nelement()).item(), primals_344.amax().item(), primals_344.amin().item())
        print('triton__30', 'in_ptr4', 'primals_345', (primals_345.sum()/primals_345.nelement()).item(), primals_345.amax().item(), primals_345.amin().item())
        triton__30.run(buf465, buf456, buf458, buf459, primals_344, primals_345, buf467, 20480, 64, grid=grid(20480), stream=stream0)
        print('triton__30', 'in_out_ptr0', 'buf465', (buf465.sum()/buf465.nelement()).item(), buf465.amax().item(), buf465.amin().item())
        print('triton__30', 'out_ptr1', 'buf467', (buf467.sum()/buf467.nelement()).item(), buf467.amax().item(), buf467.amin().item())
        del buf459
        del primals_345
        buf466 = empty_strided((8, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_59, buf465, as_strided(primals_58, (2560, 1000), (1, 2560)), alpha=1, beta=1, out=buf466)
        del primals_59
        buf468 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_61', (primals_61.sum()/primals_61.nelement()).item(), primals_61.amax().item(), primals_61.amin().item())
        triton__31.run(primals_61, buf468, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf468', (buf468.sum()/buf468.nelement()).item(), buf468.amax().item(), buf468.amin().item())
        del primals_61
        buf469 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_66', (primals_66.sum()/primals_66.nelement()).item(), primals_66.amax().item(), primals_66.amin().item())
        triton__31.run(primals_66, buf469, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf469', (buf469.sum()/buf469.nelement()).item(), buf469.amax().item(), buf469.amin().item())
        del primals_66
        buf470 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_71', (primals_71.sum()/primals_71.nelement()).item(), primals_71.amax().item(), primals_71.amin().item())
        triton__31.run(primals_71, buf470, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf470', (buf470.sum()/buf470.nelement()).item(), buf470.amax().item(), buf470.amin().item())
        del primals_71
        buf471 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_76', (primals_76.sum()/primals_76.nelement()).item(), primals_76.amax().item(), primals_76.amin().item())
        triton__31.run(primals_76, buf471, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf471', (buf471.sum()/buf471.nelement()).item(), buf471.amax().item(), buf471.amin().item())
        del primals_76
        buf472 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_81', (primals_81.sum()/primals_81.nelement()).item(), primals_81.amax().item(), primals_81.amin().item())
        triton__31.run(primals_81, buf472, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf472', (buf472.sum()/buf472.nelement()).item(), buf472.amax().item(), buf472.amin().item())
        del primals_81
        buf473 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_86', (primals_86.sum()/primals_86.nelement()).item(), primals_86.amax().item(), primals_86.amin().item())
        triton__31.run(primals_86, buf473, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf473', (buf473.sum()/buf473.nelement()).item(), buf473.amax().item(), buf473.amin().item())
        del primals_86
        buf474 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_91', (primals_91.sum()/primals_91.nelement()).item(), primals_91.amax().item(), primals_91.amin().item())
        triton__31.run(primals_91, buf474, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf474', (buf474.sum()/buf474.nelement()).item(), buf474.amax().item(), buf474.amin().item())
        del primals_91
        buf475 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_96', (primals_96.sum()/primals_96.nelement()).item(), primals_96.amax().item(), primals_96.amin().item())
        triton__31.run(primals_96, buf475, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf475', (buf475.sum()/buf475.nelement()).item(), buf475.amax().item(), buf475.amin().item())
        del primals_96
        buf476 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_101', (primals_101.sum()/primals_101.nelement()).item(), primals_101.amax().item(), primals_101.amin().item())
        triton__31.run(primals_101, buf476, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf476', (buf476.sum()/buf476.nelement()).item(), buf476.amax().item(), buf476.amin().item())
        del primals_101
        buf477 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_106', (primals_106.sum()/primals_106.nelement()).item(), primals_106.amax().item(), primals_106.amin().item())
        triton__31.run(primals_106, buf477, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf477', (buf477.sum()/buf477.nelement()).item(), buf477.amax().item(), buf477.amin().item())
        del primals_106
        buf478 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_111', (primals_111.sum()/primals_111.nelement()).item(), primals_111.amax().item(), primals_111.amin().item())
        triton__31.run(primals_111, buf478, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf478', (buf478.sum()/buf478.nelement()).item(), buf478.amax().item(), buf478.amin().item())
        del primals_111
        buf479 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_116', (primals_116.sum()/primals_116.nelement()).item(), primals_116.amax().item(), primals_116.amin().item())
        triton__31.run(primals_116, buf479, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf479', (buf479.sum()/buf479.nelement()).item(), buf479.amax().item(), buf479.amin().item())
        del primals_116
        buf480 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_121', (primals_121.sum()/primals_121.nelement()).item(), primals_121.amax().item(), primals_121.amin().item())
        triton__31.run(primals_121, buf480, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf480', (buf480.sum()/buf480.nelement()).item(), buf480.amax().item(), buf480.amin().item())
        del primals_121
        buf481 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_126', (primals_126.sum()/primals_126.nelement()).item(), primals_126.amax().item(), primals_126.amin().item())
        triton__31.run(primals_126, buf481, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf481', (buf481.sum()/buf481.nelement()).item(), buf481.amax().item(), buf481.amin().item())
        del primals_126
        buf482 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_131', (primals_131.sum()/primals_131.nelement()).item(), primals_131.amax().item(), primals_131.amin().item())
        triton__31.run(primals_131, buf482, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf482', (buf482.sum()/buf482.nelement()).item(), buf482.amax().item(), buf482.amin().item())
        del primals_131
        buf483 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_136', (primals_136.sum()/primals_136.nelement()).item(), primals_136.amax().item(), primals_136.amin().item())
        triton__31.run(primals_136, buf483, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf483', (buf483.sum()/buf483.nelement()).item(), buf483.amax().item(), buf483.amin().item())
        del primals_136
        buf484 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_141', (primals_141.sum()/primals_141.nelement()).item(), primals_141.amax().item(), primals_141.amin().item())
        triton__31.run(primals_141, buf484, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf484', (buf484.sum()/buf484.nelement()).item(), buf484.amax().item(), buf484.amin().item())
        del primals_141
        buf485 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_146', (primals_146.sum()/primals_146.nelement()).item(), primals_146.amax().item(), primals_146.amin().item())
        triton__31.run(primals_146, buf485, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf485', (buf485.sum()/buf485.nelement()).item(), buf485.amax().item(), buf485.amin().item())
        del primals_146
        buf486 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_151', (primals_151.sum()/primals_151.nelement()).item(), primals_151.amax().item(), primals_151.amin().item())
        triton__31.run(primals_151, buf486, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf486', (buf486.sum()/buf486.nelement()).item(), buf486.amax().item(), buf486.amin().item())
        del primals_151
        buf487 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_156', (primals_156.sum()/primals_156.nelement()).item(), primals_156.amax().item(), primals_156.amin().item())
        triton__31.run(primals_156, buf487, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf487', (buf487.sum()/buf487.nelement()).item(), buf487.amax().item(), buf487.amin().item())
        del primals_156
        buf488 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_161', (primals_161.sum()/primals_161.nelement()).item(), primals_161.amax().item(), primals_161.amin().item())
        triton__31.run(primals_161, buf488, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf488', (buf488.sum()/buf488.nelement()).item(), buf488.amax().item(), buf488.amin().item())
        del primals_161
        buf489 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_166', (primals_166.sum()/primals_166.nelement()).item(), primals_166.amax().item(), primals_166.amin().item())
        triton__31.run(primals_166, buf489, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf489', (buf489.sum()/buf489.nelement()).item(), buf489.amax().item(), buf489.amin().item())
        del primals_166
        buf490 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_171', (primals_171.sum()/primals_171.nelement()).item(), primals_171.amax().item(), primals_171.amin().item())
        triton__31.run(primals_171, buf490, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf490', (buf490.sum()/buf490.nelement()).item(), buf490.amax().item(), buf490.amin().item())
        del primals_171
        buf491 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_176', (primals_176.sum()/primals_176.nelement()).item(), primals_176.amax().item(), primals_176.amin().item())
        triton__31.run(primals_176, buf491, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf491', (buf491.sum()/buf491.nelement()).item(), buf491.amax().item(), buf491.amin().item())
        del primals_176
        buf492 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_181', (primals_181.sum()/primals_181.nelement()).item(), primals_181.amax().item(), primals_181.amin().item())
        triton__31.run(primals_181, buf492, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf492', (buf492.sum()/buf492.nelement()).item(), buf492.amax().item(), buf492.amin().item())
        del primals_181
        buf493 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_186', (primals_186.sum()/primals_186.nelement()).item(), primals_186.amax().item(), primals_186.amin().item())
        triton__31.run(primals_186, buf493, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf493', (buf493.sum()/buf493.nelement()).item(), buf493.amax().item(), buf493.amin().item())
        del primals_186
        buf494 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_191', (primals_191.sum()/primals_191.nelement()).item(), primals_191.amax().item(), primals_191.amin().item())
        triton__31.run(primals_191, buf494, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf494', (buf494.sum()/buf494.nelement()).item(), buf494.amax().item(), buf494.amin().item())
        del primals_191
        buf495 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_196', (primals_196.sum()/primals_196.nelement()).item(), primals_196.amax().item(), primals_196.amin().item())
        triton__31.run(primals_196, buf495, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf495', (buf495.sum()/buf495.nelement()).item(), buf495.amax().item(), buf495.amin().item())
        del primals_196
        buf496 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_201', (primals_201.sum()/primals_201.nelement()).item(), primals_201.amax().item(), primals_201.amin().item())
        triton__31.run(primals_201, buf496, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf496', (buf496.sum()/buf496.nelement()).item(), buf496.amax().item(), buf496.amin().item())
        del primals_201
        buf497 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_206', (primals_206.sum()/primals_206.nelement()).item(), primals_206.amax().item(), primals_206.amin().item())
        triton__31.run(primals_206, buf497, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf497', (buf497.sum()/buf497.nelement()).item(), buf497.amax().item(), buf497.amin().item())
        del primals_206
        buf498 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_211', (primals_211.sum()/primals_211.nelement()).item(), primals_211.amax().item(), primals_211.amin().item())
        triton__31.run(primals_211, buf498, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf498', (buf498.sum()/buf498.nelement()).item(), buf498.amax().item(), buf498.amin().item())
        del primals_211
        buf499 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_216', (primals_216.sum()/primals_216.nelement()).item(), primals_216.amax().item(), primals_216.amin().item())
        triton__31.run(primals_216, buf499, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf499', (buf499.sum()/buf499.nelement()).item(), buf499.amax().item(), buf499.amin().item())
        del primals_216
        buf500 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_221', (primals_221.sum()/primals_221.nelement()).item(), primals_221.amax().item(), primals_221.amin().item())
        triton__31.run(primals_221, buf500, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf500', (buf500.sum()/buf500.nelement()).item(), buf500.amax().item(), buf500.amin().item())
        del primals_221
        buf501 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_226', (primals_226.sum()/primals_226.nelement()).item(), primals_226.amax().item(), primals_226.amin().item())
        triton__31.run(primals_226, buf501, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf501', (buf501.sum()/buf501.nelement()).item(), buf501.amax().item(), buf501.amin().item())
        del primals_226
        buf502 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_231', (primals_231.sum()/primals_231.nelement()).item(), primals_231.amax().item(), primals_231.amin().item())
        triton__31.run(primals_231, buf502, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf502', (buf502.sum()/buf502.nelement()).item(), buf502.amax().item(), buf502.amin().item())
        del primals_231
        buf503 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_236', (primals_236.sum()/primals_236.nelement()).item(), primals_236.amax().item(), primals_236.amin().item())
        triton__31.run(primals_236, buf503, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf503', (buf503.sum()/buf503.nelement()).item(), buf503.amax().item(), buf503.amin().item())
        del primals_236
        buf504 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_241', (primals_241.sum()/primals_241.nelement()).item(), primals_241.amax().item(), primals_241.amin().item())
        triton__31.run(primals_241, buf504, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf504', (buf504.sum()/buf504.nelement()).item(), buf504.amax().item(), buf504.amin().item())
        del primals_241
        buf505 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_246', (primals_246.sum()/primals_246.nelement()).item(), primals_246.amax().item(), primals_246.amin().item())
        triton__31.run(primals_246, buf505, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf505', (buf505.sum()/buf505.nelement()).item(), buf505.amax().item(), buf505.amin().item())
        del primals_246
        buf506 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_251', (primals_251.sum()/primals_251.nelement()).item(), primals_251.amax().item(), primals_251.amin().item())
        triton__31.run(primals_251, buf506, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf506', (buf506.sum()/buf506.nelement()).item(), buf506.amax().item(), buf506.amin().item())
        del primals_251
        buf507 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_256', (primals_256.sum()/primals_256.nelement()).item(), primals_256.amax().item(), primals_256.amin().item())
        triton__31.run(primals_256, buf507, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf507', (buf507.sum()/buf507.nelement()).item(), buf507.amax().item(), buf507.amin().item())
        del primals_256
        buf508 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_261', (primals_261.sum()/primals_261.nelement()).item(), primals_261.amax().item(), primals_261.amin().item())
        triton__31.run(primals_261, buf508, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf508', (buf508.sum()/buf508.nelement()).item(), buf508.amax().item(), buf508.amin().item())
        del primals_261
        buf509 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_266', (primals_266.sum()/primals_266.nelement()).item(), primals_266.amax().item(), primals_266.amin().item())
        triton__31.run(primals_266, buf509, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf509', (buf509.sum()/buf509.nelement()).item(), buf509.amax().item(), buf509.amin().item())
        del primals_266
        buf510 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_271', (primals_271.sum()/primals_271.nelement()).item(), primals_271.amax().item(), primals_271.amin().item())
        triton__31.run(primals_271, buf510, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf510', (buf510.sum()/buf510.nelement()).item(), buf510.amax().item(), buf510.amin().item())
        del primals_271
        buf511 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_276', (primals_276.sum()/primals_276.nelement()).item(), primals_276.amax().item(), primals_276.amin().item())
        triton__31.run(primals_276, buf511, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf511', (buf511.sum()/buf511.nelement()).item(), buf511.amax().item(), buf511.amin().item())
        del primals_276
        buf512 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_281', (primals_281.sum()/primals_281.nelement()).item(), primals_281.amax().item(), primals_281.amin().item())
        triton__31.run(primals_281, buf512, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf512', (buf512.sum()/buf512.nelement()).item(), buf512.amax().item(), buf512.amin().item())
        del primals_281
        buf513 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_286', (primals_286.sum()/primals_286.nelement()).item(), primals_286.amax().item(), primals_286.amin().item())
        triton__31.run(primals_286, buf513, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf513', (buf513.sum()/buf513.nelement()).item(), buf513.amax().item(), buf513.amin().item())
        del primals_286
        buf514 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_291', (primals_291.sum()/primals_291.nelement()).item(), primals_291.amax().item(), primals_291.amin().item())
        triton__31.run(primals_291, buf514, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf514', (buf514.sum()/buf514.nelement()).item(), buf514.amax().item(), buf514.amin().item())
        del primals_291
        buf515 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_296', (primals_296.sum()/primals_296.nelement()).item(), primals_296.amax().item(), primals_296.amin().item())
        triton__31.run(primals_296, buf515, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf515', (buf515.sum()/buf515.nelement()).item(), buf515.amax().item(), buf515.amin().item())
        del primals_296
        buf516 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_301', (primals_301.sum()/primals_301.nelement()).item(), primals_301.amax().item(), primals_301.amin().item())
        triton__31.run(primals_301, buf516, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf516', (buf516.sum()/buf516.nelement()).item(), buf516.amax().item(), buf516.amin().item())
        del primals_301
        buf517 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_306', (primals_306.sum()/primals_306.nelement()).item(), primals_306.amax().item(), primals_306.amin().item())
        triton__31.run(primals_306, buf517, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf517', (buf517.sum()/buf517.nelement()).item(), buf517.amax().item(), buf517.amin().item())
        del primals_306
        buf518 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_311', (primals_311.sum()/primals_311.nelement()).item(), primals_311.amax().item(), primals_311.amin().item())
        triton__31.run(primals_311, buf518, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf518', (buf518.sum()/buf518.nelement()).item(), buf518.amax().item(), buf518.amin().item())
        del primals_311
        buf519 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_316', (primals_316.sum()/primals_316.nelement()).item(), primals_316.amax().item(), primals_316.amin().item())
        triton__31.run(primals_316, buf519, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf519', (buf519.sum()/buf519.nelement()).item(), buf519.amax().item(), buf519.amin().item())
        del primals_316
        buf520 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_321', (primals_321.sum()/primals_321.nelement()).item(), primals_321.amax().item(), primals_321.amin().item())
        triton__31.run(primals_321, buf520, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf520', (buf520.sum()/buf520.nelement()).item(), buf520.amax().item(), buf520.amin().item())
        del primals_321
        buf521 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_326', (primals_326.sum()/primals_326.nelement()).item(), primals_326.amax().item(), primals_326.amin().item())
        triton__31.run(primals_326, buf521, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf521', (buf521.sum()/buf521.nelement()).item(), buf521.amax().item(), buf521.amin().item())
        del primals_326
        buf522 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_331', (primals_331.sum()/primals_331.nelement()).item(), primals_331.amax().item(), primals_331.amin().item())
        triton__31.run(primals_331, buf522, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf522', (buf522.sum()/buf522.nelement()).item(), buf522.amax().item(), buf522.amin().item())
        del primals_331
        buf523 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_336', (primals_336.sum()/primals_336.nelement()).item(), primals_336.amax().item(), primals_336.amin().item())
        triton__31.run(primals_336, buf523, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf523', (buf523.sum()/buf523.nelement()).item(), buf523.amax().item(), buf523.amin().item())
        del primals_336
        buf524 = empty_strided((), (), device='cuda', dtype=torch.int64)
        print('triton__31', 'in_ptr0', 'primals_341', (primals_341.sum()/primals_341.nelement()).item(), primals_341.amax().item(), primals_341.amin().item())
        triton__31.run(primals_341, buf524, 1, grid=grid(1), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf524', (buf524.sum()/buf524.nelement()).item(), buf524.amax().item(), buf524.amin().item())
        del primals_341
        return (buf7, buf8, buf17, buf18, buf27, buf28, buf36, buf37, buf45, buf46, buf53, buf54, buf60, buf61, buf69, buf70, buf77, buf78, buf85, buf86, buf93, buf94, buf101, buf102, buf108, buf109, buf117, buf118, buf125, buf126, buf133, buf134, buf141, buf142, buf149, buf150, buf157, buf158, buf165, buf166, buf173, buf174, buf181, buf182, buf189, buf190, buf197, buf198, buf205, buf206, buf213, buf214, buf221, buf222, buf229, buf230, buf237, buf238, buf245, buf246, buf253, buf254, buf260, buf261, buf269, buf270, buf277, buf278, buf285, buf286, buf293, buf294, buf301, buf302, buf309, buf310, buf317, buf318, buf325, buf326, buf333, buf334, buf341, buf342, buf349, buf350, buf357, buf358, buf365, buf366, buf373, buf374, buf381, buf382, buf389, buf390, buf397, buf398, buf405, buf406, buf413, buf414, buf421, buf422, buf429, buf430, buf437, buf438, buf445, buf446, buf453, buf454, buf461, buf462, buf466, buf468, buf469, buf470, buf471, buf472, buf473, buf474, buf475, buf476, buf477, buf478, buf479, buf480, buf481, buf482, buf483, buf484, buf485, buf486, buf487, buf488, buf489, buf490, buf491, buf492, buf493, buf494, buf495, buf496, buf497, buf498, buf499, buf500, buf501, buf502, buf503, buf504, buf505, buf506, buf507, buf508, buf509, buf510, buf511, buf512, buf513, buf514, buf515, buf516, buf517, buf518, buf519, buf520, buf521, buf522, buf523, buf524, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_60, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, primals_204, primals_209, primals_214, primals_219, primals_224, primals_229, primals_234, primals_239, primals_244, primals_249, primals_254, primals_259, primals_264, primals_269, primals_274, primals_279, primals_284, primals_289, primals_294, primals_299, primals_304, primals_309, primals_314, primals_319, primals_324, primals_329, primals_334, primals_339, primals_344, buf0, buf6, buf9, buf10, buf16, buf19, buf20, buf26, buf29, buf35, buf39, buf40, buf44, buf47, buf48, buf52, buf55, buf59, buf63, buf64, buf68, buf71, buf72, buf76, buf79, buf80, buf84, buf87, buf88, buf92, buf95, buf96, buf100, buf103, buf107, buf111, buf112, buf116, buf119, buf120, buf124, buf127, buf128, buf132, buf135, buf136, buf140, buf143, buf144, buf148, buf151, buf152, buf156, buf159, buf160, buf164, buf167, buf168, buf172, buf175, buf176, buf180, buf183, buf184, buf188, buf191, buf192, buf196, buf199, buf200, buf204, buf207, buf208, buf212, buf215, buf216, buf220, buf223, buf224, buf228, buf231, buf232, buf236, buf239, buf240, buf244, buf247, buf248, buf252, buf255, buf259, buf263, buf264, buf268, buf271, buf272, buf276, buf279, buf280, buf284, buf287, buf288, buf292, buf295, buf296, buf300, buf303, buf304, buf308, buf311, buf312, buf316, buf319, buf320, buf324, buf327, buf328, buf332, buf335, buf336, buf340, buf343, buf344, buf348, buf351, buf352, buf356, buf359, buf360, buf364, buf367, buf368, buf372, buf375, buf376, buf380, buf383, buf384, buf388, buf391, buf392, buf396, buf399, buf400, buf404, buf407, buf408, buf412, buf415, buf416, buf420, buf423, buf424, buf428, buf431, buf432, buf436, buf439, buf440, buf444, buf447, buf448, buf452, buf455, buf456, buf460, buf465, as_strided(primals_58, (1000, 2560), (2560, 1)), buf467, as_strided(buf458, (1, 2560, 1, 1), (2560, 1, 1, 1)), as_strided(buf450, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf442, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf434, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf426, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf418, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf410, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf402, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf394, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf386, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf378, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf370, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf362, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf354, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf346, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf338, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf330, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf322, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf314, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf306, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf298, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf290, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf282, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf274, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf266, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf257, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf250, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf242, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf234, (1, 1920, 1, 1), (1920, 1, 1, 1)), as_strided(buf226, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf218, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf210, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf202, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf194, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf186, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf178, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf170, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf162, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf154, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf146, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf138, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf130, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf122, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf114, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf105, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf98, (1, 640, 1, 1), (640, 1, 1, 1)), as_strided(buf90, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf82, (1, 160, 1, 1), (160, 1, 1, 1)), as_strided(buf74, (1, 192, 1, 1), (192, 1, 1, 1)), as_strided(buf66, (1, 192, 1, 1), (192, 1, 1, 1)), as_strided(buf57, (1, 192, 1, 1), (192, 1, 1, 1)), as_strided(buf50, (1, 192, 1, 1), (192, 1, 1, 1)), as_strided(buf42, (1, 192, 1, 1), (192, 1, 1, 1)), as_strided(buf32, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf23, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf13, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf3, (1, 32, 1, 1), (32, 1, 1, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_62 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_82 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_87 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_92 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_97 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_102 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_107 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_112 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_117 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_122 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_127 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_132 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_137 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_142 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_147 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_152 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_157 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_162 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_167 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_172 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_182 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_187 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_197 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_212 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_227 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_242 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_257 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_272 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_287 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_317 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_322 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_327 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_332 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345]))
