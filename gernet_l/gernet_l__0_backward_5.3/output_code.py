
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
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]}
)
@triton.jit
def triton__0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__1 = async_compile.triton('''
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
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (x0 + (2560*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr2 + (r1 + (64*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp3 = 64.0
        tmp4 = tmp2 / tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp5 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 2560
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4), xmask)
    tmp6 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = 64.0
    tmp4 = tmp2 / tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.001953125
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
''')


triton__3 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x3), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.001953125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__5 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (122880*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (122880*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (64*x0) + (122880*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1920
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp11 = tl.load(in_ptr4 + (x1), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.001953125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__7 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton__7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr4 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 <= tmp1
        tmp6 = tl.where(tmp4, tmp1, tmp5)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp2, tmp1, tmp8)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp9 * tmp13
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp15, xmask)
    tmp16 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp17, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp10 = tl.load(in_ptr4 + (x3), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp13 = tl.load(in_ptr6 + (x1), xmask)
    tmp16 = tl.load(in_ptr7 + (x1), xmask)
    tmp21 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tl.where(tmp4, tmp1, tmp5)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp2, tmp1, tmp8)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001953125
    tmp15 = tmp13 * tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp9 - tmp19
    tmp22 = tmp21 * tmp14
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp9 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tmp5 <= tmp1
    tmp8 = tl.where(tmp6, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp4, tmp1, tmp10)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(tmp2, tmp1, tmp13)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


triton__10 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton__10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp8, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__12 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton__12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp6 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp7, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp12, xmask)
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp13 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp21 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.001953125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__16 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton__16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr4 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 64
        r2 = (rindex // 64)
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp0 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp11 * tmp14
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp21 = tl.load(in_ptr9 + (x1), xmask)
    tmp23 = tl.load(in_ptr10 + (x1), xmask)
    tmp29 = tl.load(in_ptr11 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


triton__18 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (491520*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (491520*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (491520*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1920
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp11 = tl.load(in_ptr4 + (x1), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__20 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton__20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp6 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp7, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp12, xmask)
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp13 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp21 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00048828125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__22 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (40960*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp11 = tl.load(in_ptr4 + (x1), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__25 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton__25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp8, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__27 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton__27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr4 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp0 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp11 * tmp14
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp21 = tl.load(in_ptr9 + (x1), xmask)
    tmp23 = tl.load(in_ptr10 + (x1), xmask)
    tmp29 = tl.load(in_ptr11 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


triton__29 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp11 = tl.load(in_ptr4 + (x1), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__31 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton__31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp6 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp7, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp12, xmask)
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp13 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp21 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0001220703125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__33 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton__33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp10, xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp12, xmask)
''')


triton__34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp11 = tl.load(in_ptr4 + (x1), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__36 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton__36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr4 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tmp4 = tmp2 - tmp3
        tmp5 = tmp0 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp0 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp11 * tmp14
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x3), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask)
    tmp21 = tl.load(in_ptr9 + (x1), xmask)
    tmp23 = tl.load(in_ptr10 + (x1), xmask)
    tmp29 = tl.load(in_ptr11 + (x1), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0001220703125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


triton__38 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton__38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp6 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp6 * tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp12, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr2 + x3, tmp17, xmask)
''')


triton__39 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton__39(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


triton__40 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton__40(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]})
@triton.jit
def triton__41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp13 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp21 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x3), xmask)
    tmp25 = tl.load(in_ptr10 + (x1), xmask)
    tmp27 = tl.load(in_ptr11 + (x1), xmask)
    tmp29 = tl.load(in_ptr12 + (x1), xmask)
    tmp35 = tl.load(in_ptr13 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.0517578125e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp37, xmask)
''')


triton__42 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton__42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp10, xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp11 = tl.load(in_ptr4 + (x1), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.0517578125e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


triton__44 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton__44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*(((r2 + (8192*x0)) // 16384))) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp6 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp12, xmask)
''')


triton__45 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton__45(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


triton__46 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton__46(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp10 = tl.load(in_ptr4 + (x1), xmask)
    tmp13 = tl.load(in_ptr5 + (x1), xmask)
    tmp18 = tl.load(in_ptr6 + (x1), xmask)
    tmp21 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 7.62939453125e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_60, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, primals_204, primals_209, primals_214, primals_219, primals_224, primals_229, primals_234, primals_239, primals_244, primals_249, primals_254, primals_259, primals_264, primals_269, primals_274, primals_279, primals_284, primals_289, primals_294, primals_299, primals_304, primals_309, primals_314, primals_319, primals_324, primals_329, primals_334, primals_339, primals_344, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, view, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160, tangents_161, tangents_162, tangents_163, tangents_164, tangents_165, tangents_166, tangents_167, tangents_168, tangents_169, tangents_170, tangents_171, tangents_172 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 2560), (2560, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(tangents_115, permute_1, out=buf0)
        del permute_1
        buf1 = empty_strided((1000, 2560), (2560, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(tangents_115, (1000, 8), (1, 1000)), view, out=buf1)
        del view
        buf2 = empty_strided((1, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        print('triton__0', 'in_ptr0', 'tangents_115', (tangents_115.sum()/tangents_115.nelement()).item(), tangents_115.amax().item(), tangents_115.amin().item())
        stream0 = get_cuda_stream(0)
        triton__0.run(tangents_115, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        print('triton__0', 'out_ptr0', 'buf2', (buf2.sum()/buf2.nelement()).item(), buf2.amax().item(), buf2.amin().item())
        del tangents_115
        buf3 = empty_strided((2560, ), (1, ), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((2560, ), (1, ), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((2560, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__1', 'in_ptr0', 'le', (le.sum()/le.nelement()).item(), le.amax().item(), le.amin().item())
        print('triton__1', 'in_ptr1', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        print('triton__1', 'in_ptr2', 'convolution_56', (convolution_56.sum()/convolution_56.nelement()).item(), convolution_56.amax().item(), convolution_56.amin().item())
        print('triton__1', 'in_ptr3', 'unsqueeze_230', (unsqueeze_230.sum()/unsqueeze_230.nelement()).item(), unsqueeze_230.amax().item(), unsqueeze_230.amin().item())
        print('triton__1', 'in_ptr4', 'squeeze_169', (squeeze_169.sum()/squeeze_169.nelement()).item(), squeeze_169.amax().item(), squeeze_169.amin().item())
        triton__1.run(le, buf0, convolution_56, unsqueeze_230, squeeze_169, buf3, buf4, buf5, 2560, 512, grid=grid(2560), stream=stream0)
        print('triton__1', 'out_ptr0', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__1', 'out_ptr1', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        print('triton__1', 'out_ptr2', 'buf5', (buf5.sum()/buf5.nelement()).item(), buf5.amax().item(), buf5.amin().item())
        buf6 = empty_strided((8, 2560, 8, 8), (163840, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__2', 'in_ptr0', 'le', (le.sum()/le.nelement()).item(), le.amax().item(), le.amin().item())
        print('triton__2', 'in_ptr1', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        print('triton__2', 'in_ptr2', 'convolution_56', (convolution_56.sum()/convolution_56.nelement()).item(), convolution_56.amax().item(), convolution_56.amin().item())
        print('triton__2', 'in_ptr3', 'unsqueeze_230', (unsqueeze_230.sum()/unsqueeze_230.nelement()).item(), unsqueeze_230.amax().item(), unsqueeze_230.amin().item())
        print('triton__2', 'in_ptr4', 'buf4', (buf4.sum()/buf4.nelement()).item(), buf4.amax().item(), buf4.amin().item())
        print('triton__2', 'in_ptr5', 'squeeze_169', (squeeze_169.sum()/squeeze_169.nelement()).item(), squeeze_169.amax().item(), squeeze_169.amin().item())
        print('triton__2', 'in_ptr6', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__2', 'in_ptr7', 'primals_344', (primals_344.sum()/primals_344.nelement()).item(), primals_344.amax().item(), primals_344.amin().item())
        triton__2.run(le, buf0, convolution_56, unsqueeze_230, buf4, squeeze_169, buf3, primals_344, buf6, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__2', 'out_ptr0', 'buf6', (buf6.sum()/buf6.nelement()).item(), buf6.amax().item(), buf6.amin().item())
        del buf0
        del buf4
        del convolution_56
        del le
        del primals_344
        del squeeze_169
        del unsqueeze_230
        buf7 = aten.convolution_backward(buf6, relu_51, primals_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_57
        buf8 = buf7[0]
        assert_size_stride(buf8, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf9 = buf7[1]
        assert_size_stride(buf9, (2560, 640, 1, 1), (640, 1, 1, 1))
        del buf7
        buf10 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__3', 'in_ptr0', 'relu_51', (relu_51.sum()/relu_51.nelement()).item(), relu_51.amax().item(), relu_51.amin().item())
        print('triton__3', 'in_ptr1', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        print('triton__3', 'in_ptr2', 'convolution_55', (convolution_55.sum()/convolution_55.nelement()).item(), convolution_55.amax().item(), convolution_55.amin().item())
        print('triton__3', 'in_ptr3', 'unsqueeze_242', (unsqueeze_242.sum()/unsqueeze_242.nelement()).item(), unsqueeze_242.amax().item(), unsqueeze_242.amin().item())
        print('triton__3', 'in_ptr4', 'squeeze_166', (squeeze_166.sum()/squeeze_166.nelement()).item(), squeeze_166.amax().item(), squeeze_166.amin().item())
        triton__3.run(relu_51, buf8, convolution_55, unsqueeze_242, squeeze_166, buf10, buf11, buf12, 640, 512, grid=grid(640), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf10', (buf10.sum()/buf10.nelement()).item(), buf10.amax().item(), buf10.amin().item())
        print('triton__3', 'out_ptr1', 'buf11', (buf11.sum()/buf11.nelement()).item(), buf11.amax().item(), buf11.amin().item())
        print('triton__3', 'out_ptr2', 'buf12', (buf12.sum()/buf12.nelement()).item(), buf12.amax().item(), buf12.amin().item())
        buf13 = empty_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda', dtype=torch.float32)
        print('triton__4', 'in_ptr0', 'relu_51', (relu_51.sum()/relu_51.nelement()).item(), relu_51.amax().item(), relu_51.amin().item())
        print('triton__4', 'in_ptr1', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        print('triton__4', 'in_ptr2', 'convolution_55', (convolution_55.sum()/convolution_55.nelement()).item(), convolution_55.amax().item(), convolution_55.amin().item())
        print('triton__4', 'in_ptr3', 'unsqueeze_242', (unsqueeze_242.sum()/unsqueeze_242.nelement()).item(), unsqueeze_242.amax().item(), unsqueeze_242.amin().item())
        print('triton__4', 'in_ptr4', 'buf11', (buf11.sum()/buf11.nelement()).item(), buf11.amax().item(), buf11.amin().item())
        print('triton__4', 'in_ptr5', 'squeeze_166', (squeeze_166.sum()/squeeze_166.nelement()).item(), squeeze_166.amax().item(), squeeze_166.amin().item())
        print('triton__4', 'in_ptr6', 'buf10', (buf10.sum()/buf10.nelement()).item(), buf10.amax().item(), buf10.amin().item())
        print('triton__4', 'in_ptr7', 'primals_339', (primals_339.sum()/primals_339.nelement()).item(), primals_339.amax().item(), primals_339.amin().item())
        triton__4.run(relu_51, buf8, convolution_55, unsqueeze_242, buf11, squeeze_166, buf10, primals_339, buf13, 327680, grid=grid(327680), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf13', (buf13.sum()/buf13.nelement()).item(), buf13.amax().item(), buf13.amin().item())
        del convolution_55
        del primals_339
        del squeeze_166
        del unsqueeze_242
        buf14 = aten.convolution_backward(buf13, relu_50, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_56
        buf15 = buf14[0]
        assert_size_stride(buf15, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf16 = buf14[1]
        assert_size_stride(buf16, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf14
        buf17 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf19 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_50', (relu_50.sum()/relu_50.nelement()).item(), relu_50.amax().item(), relu_50.amin().item())
        print('triton__5', 'in_ptr1', 'buf15', (buf15.sum()/buf15.nelement()).item(), buf15.amax().item(), buf15.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_54', (convolution_54.sum()/convolution_54.nelement()).item(), convolution_54.amax().item(), convolution_54.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_254', (unsqueeze_254.sum()/unsqueeze_254.nelement()).item(), unsqueeze_254.amax().item(), unsqueeze_254.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_163', (squeeze_163.sum()/squeeze_163.nelement()).item(), squeeze_163.amax().item(), squeeze_163.amin().item())
        triton__5.run(relu_50, buf15, convolution_54, unsqueeze_254, squeeze_163, buf17, buf18, buf19, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf17', (buf17.sum()/buf17.nelement()).item(), buf17.amax().item(), buf17.amin().item())
        print('triton__5', 'out_ptr1', 'buf18', (buf18.sum()/buf18.nelement()).item(), buf18.amax().item(), buf18.amin().item())
        print('triton__5', 'out_ptr2', 'buf19', (buf19.sum()/buf19.nelement()).item(), buf19.amax().item(), buf19.amin().item())
        buf20 = buf15; del buf15  # reuse
        print('triton__6', 'in_out_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        print('triton__6', 'in_ptr0', 'relu_50', (relu_50.sum()/relu_50.nelement()).item(), relu_50.amax().item(), relu_50.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_54', (convolution_54.sum()/convolution_54.nelement()).item(), convolution_54.amax().item(), convolution_54.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_254', (unsqueeze_254.sum()/unsqueeze_254.nelement()).item(), unsqueeze_254.amax().item(), unsqueeze_254.amin().item())
        print('triton__6', 'in_ptr3', 'buf18', (buf18.sum()/buf18.nelement()).item(), buf18.amax().item(), buf18.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_163', (squeeze_163.sum()/squeeze_163.nelement()).item(), squeeze_163.amax().item(), squeeze_163.amin().item())
        print('triton__6', 'in_ptr5', 'buf17', (buf17.sum()/buf17.nelement()).item(), buf17.amax().item(), buf17.amin().item())
        print('triton__6', 'in_ptr6', 'primals_334', (primals_334.sum()/primals_334.nelement()).item(), primals_334.amax().item(), primals_334.amin().item())
        triton__6.run(buf20, relu_50, convolution_54, unsqueeze_254, buf18, squeeze_163, buf17, primals_334, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        del convolution_54
        del primals_334
        del relu_50
        del squeeze_163
        del unsqueeze_254
        buf21 = aten.convolution_backward(buf20, relu_49, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf20
        del primals_55
        buf22 = buf21[0]
        assert_size_stride(buf22, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf23 = buf21[1]
        assert_size_stride(buf23, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf21
        buf24 = buf18; del buf18  # reuse
        buf25 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf26 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_49', (relu_49.sum()/relu_49.nelement()).item(), relu_49.amax().item(), relu_49.amin().item())
        print('triton__5', 'in_ptr1', 'buf22', (buf22.sum()/buf22.nelement()).item(), buf22.amax().item(), buf22.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_53', (convolution_53.sum()/convolution_53.nelement()).item(), convolution_53.amax().item(), convolution_53.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_266', (unsqueeze_266.sum()/unsqueeze_266.nelement()).item(), unsqueeze_266.amax().item(), unsqueeze_266.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_160', (squeeze_160.sum()/squeeze_160.nelement()).item(), squeeze_160.amax().item(), squeeze_160.amin().item())
        triton__5.run(relu_49, buf22, convolution_53, unsqueeze_266, squeeze_160, buf24, buf25, buf26, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        print('triton__5', 'out_ptr1', 'buf25', (buf25.sum()/buf25.nelement()).item(), buf25.amax().item(), buf25.amin().item())
        print('triton__5', 'out_ptr2', 'buf26', (buf26.sum()/buf26.nelement()).item(), buf26.amax().item(), buf26.amin().item())
        buf27 = buf22; del buf22  # reuse
        print('triton__6', 'in_out_ptr0', 'buf27', (buf27.sum()/buf27.nelement()).item(), buf27.amax().item(), buf27.amin().item())
        print('triton__6', 'in_ptr0', 'relu_49', (relu_49.sum()/relu_49.nelement()).item(), relu_49.amax().item(), relu_49.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_53', (convolution_53.sum()/convolution_53.nelement()).item(), convolution_53.amax().item(), convolution_53.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_266', (unsqueeze_266.sum()/unsqueeze_266.nelement()).item(), unsqueeze_266.amax().item(), unsqueeze_266.amin().item())
        print('triton__6', 'in_ptr3', 'buf25', (buf25.sum()/buf25.nelement()).item(), buf25.amax().item(), buf25.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_160', (squeeze_160.sum()/squeeze_160.nelement()).item(), squeeze_160.amax().item(), squeeze_160.amin().item())
        print('triton__6', 'in_ptr5', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        print('triton__6', 'in_ptr6', 'primals_329', (primals_329.sum()/primals_329.nelement()).item(), primals_329.amax().item(), primals_329.amin().item())
        triton__6.run(buf27, relu_49, convolution_53, unsqueeze_266, buf25, squeeze_160, buf24, primals_329, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf27', (buf27.sum()/buf27.nelement()).item(), buf27.amax().item(), buf27.amin().item())
        del convolution_53
        del primals_329
        del relu_49
        del squeeze_160
        del unsqueeze_266
        buf28 = aten.convolution_backward(buf27, relu_48, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf27
        del primals_54
        buf29 = buf28[0]
        assert_size_stride(buf29, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf30 = buf28[1]
        assert_size_stride(buf30, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf28
        buf31 = buf11; del buf11  # reuse
        buf32 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf34 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__7', 'in_ptr0', 'relu_48', (relu_48.sum()/relu_48.nelement()).item(), relu_48.amax().item(), relu_48.amin().item())
        print('triton__7', 'in_ptr1', 'relu_51', (relu_51.sum()/relu_51.nelement()).item(), relu_51.amax().item(), relu_51.amin().item())
        print('triton__7', 'in_ptr2', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        print('triton__7', 'in_ptr3', 'buf29', (buf29.sum()/buf29.nelement()).item(), buf29.amax().item(), buf29.amin().item())
        print('triton__7', 'in_ptr4', 'convolution_52', (convolution_52.sum()/convolution_52.nelement()).item(), convolution_52.amax().item(), convolution_52.amin().item())
        print('triton__7', 'in_ptr5', 'unsqueeze_278', (unsqueeze_278.sum()/unsqueeze_278.nelement()).item(), unsqueeze_278.amax().item(), unsqueeze_278.amin().item())
        print('triton__7', 'in_ptr6', 'squeeze_157', (squeeze_157.sum()/squeeze_157.nelement()).item(), squeeze_157.amax().item(), squeeze_157.amin().item())
        triton__7.run(relu_48, relu_51, buf8, buf29, convolution_52, unsqueeze_278, squeeze_157, buf31, buf32, buf34, 640, 512, grid=grid(640), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf31', (buf31.sum()/buf31.nelement()).item(), buf31.amax().item(), buf31.amin().item())
        print('triton__7', 'out_ptr1', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        print('triton__7', 'out_ptr2', 'buf34', (buf34.sum()/buf34.nelement()).item(), buf34.amax().item(), buf34.amin().item())
        buf33 = buf13; del buf13  # reuse
        buf35 = buf33; del buf33  # reuse
        print('triton__8', 'in_out_ptr0', 'buf35', (buf35.sum()/buf35.nelement()).item(), buf35.amax().item(), buf35.amin().item())
        print('triton__8', 'in_ptr0', 'relu_48', (relu_48.sum()/relu_48.nelement()).item(), relu_48.amax().item(), relu_48.amin().item())
        print('triton__8', 'in_ptr1', 'relu_51', (relu_51.sum()/relu_51.nelement()).item(), relu_51.amax().item(), relu_51.amin().item())
        print('triton__8', 'in_ptr2', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        print('triton__8', 'in_ptr3', 'buf29', (buf29.sum()/buf29.nelement()).item(), buf29.amax().item(), buf29.amin().item())
        print('triton__8', 'in_ptr4', 'convolution_52', (convolution_52.sum()/convolution_52.nelement()).item(), convolution_52.amax().item(), convolution_52.amin().item())
        print('triton__8', 'in_ptr5', 'unsqueeze_278', (unsqueeze_278.sum()/unsqueeze_278.nelement()).item(), unsqueeze_278.amax().item(), unsqueeze_278.amin().item())
        print('triton__8', 'in_ptr6', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        print('triton__8', 'in_ptr7', 'squeeze_157', (squeeze_157.sum()/squeeze_157.nelement()).item(), squeeze_157.amax().item(), squeeze_157.amin().item())
        print('triton__8', 'in_ptr8', 'buf31', (buf31.sum()/buf31.nelement()).item(), buf31.amax().item(), buf31.amin().item())
        print('triton__8', 'in_ptr9', 'primals_324', (primals_324.sum()/primals_324.nelement()).item(), primals_324.amax().item(), primals_324.amin().item())
        triton__8.run(buf35, relu_48, relu_51, buf8, buf29, convolution_52, unsqueeze_278, buf32, squeeze_157, buf31, primals_324, 327680, grid=grid(327680), stream=stream0)
        print('triton__8', 'in_out_ptr0', 'buf35', (buf35.sum()/buf35.nelement()).item(), buf35.amax().item(), buf35.amin().item())
        del convolution_52
        del primals_324
        del squeeze_157
        del unsqueeze_278
        buf36 = aten.convolution_backward(buf35, relu_47, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf35
        del primals_53
        buf37 = buf36[0]
        assert_size_stride(buf37, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf38 = buf36[1]
        assert_size_stride(buf38, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf36
        buf39 = buf25; del buf25  # reuse
        buf40 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_47', (relu_47.sum()/relu_47.nelement()).item(), relu_47.amax().item(), relu_47.amin().item())
        print('triton__5', 'in_ptr1', 'buf37', (buf37.sum()/buf37.nelement()).item(), buf37.amax().item(), buf37.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_51', (convolution_51.sum()/convolution_51.nelement()).item(), convolution_51.amax().item(), convolution_51.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_290', (unsqueeze_290.sum()/unsqueeze_290.nelement()).item(), unsqueeze_290.amax().item(), unsqueeze_290.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_154', (squeeze_154.sum()/squeeze_154.nelement()).item(), squeeze_154.amax().item(), squeeze_154.amin().item())
        triton__5.run(relu_47, buf37, convolution_51, unsqueeze_290, squeeze_154, buf39, buf40, buf41, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf39', (buf39.sum()/buf39.nelement()).item(), buf39.amax().item(), buf39.amin().item())
        print('triton__5', 'out_ptr1', 'buf40', (buf40.sum()/buf40.nelement()).item(), buf40.amax().item(), buf40.amin().item())
        print('triton__5', 'out_ptr2', 'buf41', (buf41.sum()/buf41.nelement()).item(), buf41.amax().item(), buf41.amin().item())
        buf42 = buf37; del buf37  # reuse
        print('triton__6', 'in_out_ptr0', 'buf42', (buf42.sum()/buf42.nelement()).item(), buf42.amax().item(), buf42.amin().item())
        print('triton__6', 'in_ptr0', 'relu_47', (relu_47.sum()/relu_47.nelement()).item(), relu_47.amax().item(), relu_47.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_51', (convolution_51.sum()/convolution_51.nelement()).item(), convolution_51.amax().item(), convolution_51.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_290', (unsqueeze_290.sum()/unsqueeze_290.nelement()).item(), unsqueeze_290.amax().item(), unsqueeze_290.amin().item())
        print('triton__6', 'in_ptr3', 'buf40', (buf40.sum()/buf40.nelement()).item(), buf40.amax().item(), buf40.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_154', (squeeze_154.sum()/squeeze_154.nelement()).item(), squeeze_154.amax().item(), squeeze_154.amin().item())
        print('triton__6', 'in_ptr5', 'buf39', (buf39.sum()/buf39.nelement()).item(), buf39.amax().item(), buf39.amin().item())
        print('triton__6', 'in_ptr6', 'primals_319', (primals_319.sum()/primals_319.nelement()).item(), primals_319.amax().item(), primals_319.amin().item())
        triton__6.run(buf42, relu_47, convolution_51, unsqueeze_290, buf40, squeeze_154, buf39, primals_319, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf42', (buf42.sum()/buf42.nelement()).item(), buf42.amax().item(), buf42.amin().item())
        del convolution_51
        del primals_319
        del relu_47
        del squeeze_154
        del unsqueeze_290
        buf43 = aten.convolution_backward(buf42, relu_46, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf42
        del primals_52
        buf44 = buf43[0]
        assert_size_stride(buf44, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf45 = buf43[1]
        assert_size_stride(buf45, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf43
        buf46 = buf40; del buf40  # reuse
        buf47 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_46', (relu_46.sum()/relu_46.nelement()).item(), relu_46.amax().item(), relu_46.amin().item())
        print('triton__5', 'in_ptr1', 'buf44', (buf44.sum()/buf44.nelement()).item(), buf44.amax().item(), buf44.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_50', (convolution_50.sum()/convolution_50.nelement()).item(), convolution_50.amax().item(), convolution_50.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_302', (unsqueeze_302.sum()/unsqueeze_302.nelement()).item(), unsqueeze_302.amax().item(), unsqueeze_302.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_151', (squeeze_151.sum()/squeeze_151.nelement()).item(), squeeze_151.amax().item(), squeeze_151.amin().item())
        triton__5.run(relu_46, buf44, convolution_50, unsqueeze_302, squeeze_151, buf46, buf47, buf48, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf46', (buf46.sum()/buf46.nelement()).item(), buf46.amax().item(), buf46.amin().item())
        print('triton__5', 'out_ptr1', 'buf47', (buf47.sum()/buf47.nelement()).item(), buf47.amax().item(), buf47.amin().item())
        print('triton__5', 'out_ptr2', 'buf48', (buf48.sum()/buf48.nelement()).item(), buf48.amax().item(), buf48.amin().item())
        buf49 = buf44; del buf44  # reuse
        print('triton__6', 'in_out_ptr0', 'buf49', (buf49.sum()/buf49.nelement()).item(), buf49.amax().item(), buf49.amin().item())
        print('triton__6', 'in_ptr0', 'relu_46', (relu_46.sum()/relu_46.nelement()).item(), relu_46.amax().item(), relu_46.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_50', (convolution_50.sum()/convolution_50.nelement()).item(), convolution_50.amax().item(), convolution_50.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_302', (unsqueeze_302.sum()/unsqueeze_302.nelement()).item(), unsqueeze_302.amax().item(), unsqueeze_302.amin().item())
        print('triton__6', 'in_ptr3', 'buf47', (buf47.sum()/buf47.nelement()).item(), buf47.amax().item(), buf47.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_151', (squeeze_151.sum()/squeeze_151.nelement()).item(), squeeze_151.amax().item(), squeeze_151.amin().item())
        print('triton__6', 'in_ptr5', 'buf46', (buf46.sum()/buf46.nelement()).item(), buf46.amax().item(), buf46.amin().item())
        print('triton__6', 'in_ptr6', 'primals_314', (primals_314.sum()/primals_314.nelement()).item(), primals_314.amax().item(), primals_314.amin().item())
        triton__6.run(buf49, relu_46, convolution_50, unsqueeze_302, buf47, squeeze_151, buf46, primals_314, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf49', (buf49.sum()/buf49.nelement()).item(), buf49.amax().item(), buf49.amin().item())
        del convolution_50
        del primals_314
        del relu_46
        del squeeze_151
        del unsqueeze_302
        buf50 = aten.convolution_backward(buf49, relu_45, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf49
        del primals_51
        buf51 = buf50[0]
        assert_size_stride(buf51, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf52 = buf50[1]
        assert_size_stride(buf52, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf50
        buf53 = buf29; del buf29  # reuse
        print('triton__9', 'in_out_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        print('triton__9', 'in_ptr0', 'relu_45', (relu_45.sum()/relu_45.nelement()).item(), relu_45.amax().item(), relu_45.amin().item())
        print('triton__9', 'in_ptr1', 'relu_48', (relu_48.sum()/relu_48.nelement()).item(), relu_48.amax().item(), relu_48.amin().item())
        print('triton__9', 'in_ptr2', 'relu_51', (relu_51.sum()/relu_51.nelement()).item(), relu_51.amax().item(), relu_51.amin().item())
        print('triton__9', 'in_ptr3', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        print('triton__9', 'in_ptr4', 'buf51', (buf51.sum()/buf51.nelement()).item(), buf51.amax().item(), buf51.amin().item())
        triton__9.run(buf53, relu_45, relu_48, relu_51, buf8, buf51, 327680, grid=grid(327680), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        del buf51
        del relu_45
        del relu_48
        del relu_51
        buf54 = buf32; del buf32  # reuse
        buf55 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__10', 'in_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        print('triton__10', 'in_ptr1', 'convolution_49', (convolution_49.sum()/convolution_49.nelement()).item(), convolution_49.amax().item(), convolution_49.amin().item())
        print('triton__10', 'in_ptr2', 'unsqueeze_314', (unsqueeze_314.sum()/unsqueeze_314.nelement()).item(), unsqueeze_314.amax().item(), unsqueeze_314.amin().item())
        print('triton__10', 'in_ptr3', 'squeeze_148', (squeeze_148.sum()/squeeze_148.nelement()).item(), squeeze_148.amax().item(), squeeze_148.amin().item())
        triton__10.run(buf53, convolution_49, unsqueeze_314, squeeze_148, buf54, buf55, buf56, 640, 512, grid=grid(640), stream=stream0)
        print('triton__10', 'out_ptr0', 'buf54', (buf54.sum()/buf54.nelement()).item(), buf54.amax().item(), buf54.amin().item())
        print('triton__10', 'out_ptr1', 'buf55', (buf55.sum()/buf55.nelement()).item(), buf55.amax().item(), buf55.amin().item())
        print('triton__10', 'out_ptr2', 'buf56', (buf56.sum()/buf56.nelement()).item(), buf56.amax().item(), buf56.amin().item())
        buf57 = buf8; del buf8  # reuse
        print('triton__11', 'in_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        print('triton__11', 'in_ptr1', 'convolution_49', (convolution_49.sum()/convolution_49.nelement()).item(), convolution_49.amax().item(), convolution_49.amin().item())
        print('triton__11', 'in_ptr2', 'unsqueeze_314', (unsqueeze_314.sum()/unsqueeze_314.nelement()).item(), unsqueeze_314.amax().item(), unsqueeze_314.amin().item())
        print('triton__11', 'in_ptr3', 'buf55', (buf55.sum()/buf55.nelement()).item(), buf55.amax().item(), buf55.amin().item())
        print('triton__11', 'in_ptr4', 'squeeze_148', (squeeze_148.sum()/squeeze_148.nelement()).item(), squeeze_148.amax().item(), squeeze_148.amin().item())
        print('triton__11', 'in_ptr5', 'buf54', (buf54.sum()/buf54.nelement()).item(), buf54.amax().item(), buf54.amin().item())
        print('triton__11', 'in_ptr6', 'primals_309', (primals_309.sum()/primals_309.nelement()).item(), primals_309.amax().item(), primals_309.amin().item())
        triton__11.run(buf53, convolution_49, unsqueeze_314, buf55, squeeze_148, buf54, primals_309, buf57, 327680, grid=grid(327680), stream=stream0)
        print('triton__11', 'out_ptr0', 'buf57', (buf57.sum()/buf57.nelement()).item(), buf57.amax().item(), buf57.amin().item())
        del convolution_49
        del primals_309
        del squeeze_148
        del unsqueeze_314
        buf58 = aten.convolution_backward(buf57, relu_44, primals_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_50
        buf59 = buf58[0]
        assert_size_stride(buf59, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf60 = buf58[1]
        assert_size_stride(buf60, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf58
        buf61 = buf47; del buf47  # reuse
        buf62 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf63 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_44', (relu_44.sum()/relu_44.nelement()).item(), relu_44.amax().item(), relu_44.amin().item())
        print('triton__5', 'in_ptr1', 'buf59', (buf59.sum()/buf59.nelement()).item(), buf59.amax().item(), buf59.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_48', (convolution_48.sum()/convolution_48.nelement()).item(), convolution_48.amax().item(), convolution_48.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_326', (unsqueeze_326.sum()/unsqueeze_326.nelement()).item(), unsqueeze_326.amax().item(), unsqueeze_326.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_145', (squeeze_145.sum()/squeeze_145.nelement()).item(), squeeze_145.amax().item(), squeeze_145.amin().item())
        triton__5.run(relu_44, buf59, convolution_48, unsqueeze_326, squeeze_145, buf61, buf62, buf63, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf61', (buf61.sum()/buf61.nelement()).item(), buf61.amax().item(), buf61.amin().item())
        print('triton__5', 'out_ptr1', 'buf62', (buf62.sum()/buf62.nelement()).item(), buf62.amax().item(), buf62.amin().item())
        print('triton__5', 'out_ptr2', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        buf64 = buf59; del buf59  # reuse
        print('triton__6', 'in_out_ptr0', 'buf64', (buf64.sum()/buf64.nelement()).item(), buf64.amax().item(), buf64.amin().item())
        print('triton__6', 'in_ptr0', 'relu_44', (relu_44.sum()/relu_44.nelement()).item(), relu_44.amax().item(), relu_44.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_48', (convolution_48.sum()/convolution_48.nelement()).item(), convolution_48.amax().item(), convolution_48.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_326', (unsqueeze_326.sum()/unsqueeze_326.nelement()).item(), unsqueeze_326.amax().item(), unsqueeze_326.amin().item())
        print('triton__6', 'in_ptr3', 'buf62', (buf62.sum()/buf62.nelement()).item(), buf62.amax().item(), buf62.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_145', (squeeze_145.sum()/squeeze_145.nelement()).item(), squeeze_145.amax().item(), squeeze_145.amin().item())
        print('triton__6', 'in_ptr5', 'buf61', (buf61.sum()/buf61.nelement()).item(), buf61.amax().item(), buf61.amin().item())
        print('triton__6', 'in_ptr6', 'primals_304', (primals_304.sum()/primals_304.nelement()).item(), primals_304.amax().item(), primals_304.amin().item())
        triton__6.run(buf64, relu_44, convolution_48, unsqueeze_326, buf62, squeeze_145, buf61, primals_304, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf64', (buf64.sum()/buf64.nelement()).item(), buf64.amax().item(), buf64.amin().item())
        del convolution_48
        del primals_304
        del relu_44
        del squeeze_145
        del unsqueeze_326
        buf65 = aten.convolution_backward(buf64, relu_43, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf64
        del primals_49
        buf66 = buf65[0]
        assert_size_stride(buf66, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf67 = buf65[1]
        assert_size_stride(buf67, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf65
        buf68 = buf62; del buf62  # reuse
        buf69 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_43', (relu_43.sum()/relu_43.nelement()).item(), relu_43.amax().item(), relu_43.amin().item())
        print('triton__5', 'in_ptr1', 'buf66', (buf66.sum()/buf66.nelement()).item(), buf66.amax().item(), buf66.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_47', (convolution_47.sum()/convolution_47.nelement()).item(), convolution_47.amax().item(), convolution_47.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_338', (unsqueeze_338.sum()/unsqueeze_338.nelement()).item(), unsqueeze_338.amax().item(), unsqueeze_338.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_142', (squeeze_142.sum()/squeeze_142.nelement()).item(), squeeze_142.amax().item(), squeeze_142.amin().item())
        triton__5.run(relu_43, buf66, convolution_47, unsqueeze_338, squeeze_142, buf68, buf69, buf70, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf68', (buf68.sum()/buf68.nelement()).item(), buf68.amax().item(), buf68.amin().item())
        print('triton__5', 'out_ptr1', 'buf69', (buf69.sum()/buf69.nelement()).item(), buf69.amax().item(), buf69.amin().item())
        print('triton__5', 'out_ptr2', 'buf70', (buf70.sum()/buf70.nelement()).item(), buf70.amax().item(), buf70.amin().item())
        buf71 = buf66; del buf66  # reuse
        print('triton__6', 'in_out_ptr0', 'buf71', (buf71.sum()/buf71.nelement()).item(), buf71.amax().item(), buf71.amin().item())
        print('triton__6', 'in_ptr0', 'relu_43', (relu_43.sum()/relu_43.nelement()).item(), relu_43.amax().item(), relu_43.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_47', (convolution_47.sum()/convolution_47.nelement()).item(), convolution_47.amax().item(), convolution_47.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_338', (unsqueeze_338.sum()/unsqueeze_338.nelement()).item(), unsqueeze_338.amax().item(), unsqueeze_338.amin().item())
        print('triton__6', 'in_ptr3', 'buf69', (buf69.sum()/buf69.nelement()).item(), buf69.amax().item(), buf69.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_142', (squeeze_142.sum()/squeeze_142.nelement()).item(), squeeze_142.amax().item(), squeeze_142.amin().item())
        print('triton__6', 'in_ptr5', 'buf68', (buf68.sum()/buf68.nelement()).item(), buf68.amax().item(), buf68.amin().item())
        print('triton__6', 'in_ptr6', 'primals_299', (primals_299.sum()/primals_299.nelement()).item(), primals_299.amax().item(), primals_299.amin().item())
        triton__6.run(buf71, relu_43, convolution_47, unsqueeze_338, buf69, squeeze_142, buf68, primals_299, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf71', (buf71.sum()/buf71.nelement()).item(), buf71.amax().item(), buf71.amin().item())
        del convolution_47
        del primals_299
        del relu_43
        del squeeze_142
        del unsqueeze_338
        buf72 = aten.convolution_backward(buf71, relu_42, primals_48, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf71
        del primals_48
        buf73 = buf72[0]
        assert_size_stride(buf73, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf74 = buf72[1]
        assert_size_stride(buf74, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf72
        buf75 = buf55; del buf55  # reuse
        buf76 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__12', 'in_ptr0', 'relu_42', (relu_42.sum()/relu_42.nelement()).item(), relu_42.amax().item(), relu_42.amin().item())
        print('triton__12', 'in_ptr1', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        print('triton__12', 'in_ptr2', 'buf73', (buf73.sum()/buf73.nelement()).item(), buf73.amax().item(), buf73.amin().item())
        print('triton__12', 'in_ptr3', 'convolution_46', (convolution_46.sum()/convolution_46.nelement()).item(), convolution_46.amax().item(), convolution_46.amin().item())
        print('triton__12', 'in_ptr4', 'unsqueeze_350', (unsqueeze_350.sum()/unsqueeze_350.nelement()).item(), unsqueeze_350.amax().item(), unsqueeze_350.amin().item())
        print('triton__12', 'in_ptr5', 'squeeze_139', (squeeze_139.sum()/squeeze_139.nelement()).item(), squeeze_139.amax().item(), squeeze_139.amin().item())
        triton__12.run(relu_42, buf53, buf73, convolution_46, unsqueeze_350, squeeze_139, buf75, buf76, buf78, 640, 512, grid=grid(640), stream=stream0)
        print('triton__12', 'out_ptr0', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        print('triton__12', 'out_ptr1', 'buf76', (buf76.sum()/buf76.nelement()).item(), buf76.amax().item(), buf76.amin().item())
        print('triton__12', 'out_ptr2', 'buf78', (buf78.sum()/buf78.nelement()).item(), buf78.amax().item(), buf78.amin().item())
        buf77 = buf57; del buf57  # reuse
        print('triton__13', 'in_ptr0', 'relu_42', (relu_42.sum()/relu_42.nelement()).item(), relu_42.amax().item(), relu_42.amin().item())
        print('triton__13', 'in_ptr1', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        print('triton__13', 'in_ptr2', 'buf73', (buf73.sum()/buf73.nelement()).item(), buf73.amax().item(), buf73.amin().item())
        print('triton__13', 'in_ptr3', 'convolution_46', (convolution_46.sum()/convolution_46.nelement()).item(), convolution_46.amax().item(), convolution_46.amin().item())
        print('triton__13', 'in_ptr4', 'unsqueeze_350', (unsqueeze_350.sum()/unsqueeze_350.nelement()).item(), unsqueeze_350.amax().item(), unsqueeze_350.amin().item())
        print('triton__13', 'in_ptr5', 'buf76', (buf76.sum()/buf76.nelement()).item(), buf76.amax().item(), buf76.amin().item())
        print('triton__13', 'in_ptr6', 'squeeze_139', (squeeze_139.sum()/squeeze_139.nelement()).item(), squeeze_139.amax().item(), squeeze_139.amin().item())
        print('triton__13', 'in_ptr7', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        print('triton__13', 'in_ptr8', 'primals_294', (primals_294.sum()/primals_294.nelement()).item(), primals_294.amax().item(), primals_294.amin().item())
        triton__13.run(relu_42, buf53, buf73, convolution_46, unsqueeze_350, buf76, squeeze_139, buf75, primals_294, buf77, 327680, grid=grid(327680), stream=stream0)
        print('triton__13', 'out_ptr0', 'buf77', (buf77.sum()/buf77.nelement()).item(), buf77.amax().item(), buf77.amin().item())
        del convolution_46
        del primals_294
        del squeeze_139
        del unsqueeze_350
        buf79 = aten.convolution_backward(buf77, relu_41, primals_47, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf77
        del primals_47
        buf80 = buf79[0]
        assert_size_stride(buf80, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf81 = buf79[1]
        assert_size_stride(buf81, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf79
        buf82 = buf69; del buf69  # reuse
        buf83 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_41', (relu_41.sum()/relu_41.nelement()).item(), relu_41.amax().item(), relu_41.amin().item())
        print('triton__5', 'in_ptr1', 'buf80', (buf80.sum()/buf80.nelement()).item(), buf80.amax().item(), buf80.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_45', (convolution_45.sum()/convolution_45.nelement()).item(), convolution_45.amax().item(), convolution_45.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_362', (unsqueeze_362.sum()/unsqueeze_362.nelement()).item(), unsqueeze_362.amax().item(), unsqueeze_362.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_136', (squeeze_136.sum()/squeeze_136.nelement()).item(), squeeze_136.amax().item(), squeeze_136.amin().item())
        triton__5.run(relu_41, buf80, convolution_45, unsqueeze_362, squeeze_136, buf82, buf83, buf84, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf82', (buf82.sum()/buf82.nelement()).item(), buf82.amax().item(), buf82.amin().item())
        print('triton__5', 'out_ptr1', 'buf83', (buf83.sum()/buf83.nelement()).item(), buf83.amax().item(), buf83.amin().item())
        print('triton__5', 'out_ptr2', 'buf84', (buf84.sum()/buf84.nelement()).item(), buf84.amax().item(), buf84.amin().item())
        buf85 = buf80; del buf80  # reuse
        print('triton__6', 'in_out_ptr0', 'buf85', (buf85.sum()/buf85.nelement()).item(), buf85.amax().item(), buf85.amin().item())
        print('triton__6', 'in_ptr0', 'relu_41', (relu_41.sum()/relu_41.nelement()).item(), relu_41.amax().item(), relu_41.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_45', (convolution_45.sum()/convolution_45.nelement()).item(), convolution_45.amax().item(), convolution_45.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_362', (unsqueeze_362.sum()/unsqueeze_362.nelement()).item(), unsqueeze_362.amax().item(), unsqueeze_362.amin().item())
        print('triton__6', 'in_ptr3', 'buf83', (buf83.sum()/buf83.nelement()).item(), buf83.amax().item(), buf83.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_136', (squeeze_136.sum()/squeeze_136.nelement()).item(), squeeze_136.amax().item(), squeeze_136.amin().item())
        print('triton__6', 'in_ptr5', 'buf82', (buf82.sum()/buf82.nelement()).item(), buf82.amax().item(), buf82.amin().item())
        print('triton__6', 'in_ptr6', 'primals_289', (primals_289.sum()/primals_289.nelement()).item(), primals_289.amax().item(), primals_289.amin().item())
        triton__6.run(buf85, relu_41, convolution_45, unsqueeze_362, buf83, squeeze_136, buf82, primals_289, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf85', (buf85.sum()/buf85.nelement()).item(), buf85.amax().item(), buf85.amin().item())
        del convolution_45
        del primals_289
        del relu_41
        del squeeze_136
        del unsqueeze_362
        buf86 = aten.convolution_backward(buf85, relu_40, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf85
        del primals_46
        buf87 = buf86[0]
        assert_size_stride(buf87, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf88 = buf86[1]
        assert_size_stride(buf88, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf86
        buf89 = buf83; del buf83  # reuse
        buf90 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_40', (relu_40.sum()/relu_40.nelement()).item(), relu_40.amax().item(), relu_40.amin().item())
        print('triton__5', 'in_ptr1', 'buf87', (buf87.sum()/buf87.nelement()).item(), buf87.amax().item(), buf87.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_44', (convolution_44.sum()/convolution_44.nelement()).item(), convolution_44.amax().item(), convolution_44.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_374', (unsqueeze_374.sum()/unsqueeze_374.nelement()).item(), unsqueeze_374.amax().item(), unsqueeze_374.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_133', (squeeze_133.sum()/squeeze_133.nelement()).item(), squeeze_133.amax().item(), squeeze_133.amin().item())
        triton__5.run(relu_40, buf87, convolution_44, unsqueeze_374, squeeze_133, buf89, buf90, buf91, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf89', (buf89.sum()/buf89.nelement()).item(), buf89.amax().item(), buf89.amin().item())
        print('triton__5', 'out_ptr1', 'buf90', (buf90.sum()/buf90.nelement()).item(), buf90.amax().item(), buf90.amin().item())
        print('triton__5', 'out_ptr2', 'buf91', (buf91.sum()/buf91.nelement()).item(), buf91.amax().item(), buf91.amin().item())
        buf92 = buf87; del buf87  # reuse
        print('triton__6', 'in_out_ptr0', 'buf92', (buf92.sum()/buf92.nelement()).item(), buf92.amax().item(), buf92.amin().item())
        print('triton__6', 'in_ptr0', 'relu_40', (relu_40.sum()/relu_40.nelement()).item(), relu_40.amax().item(), relu_40.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_44', (convolution_44.sum()/convolution_44.nelement()).item(), convolution_44.amax().item(), convolution_44.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_374', (unsqueeze_374.sum()/unsqueeze_374.nelement()).item(), unsqueeze_374.amax().item(), unsqueeze_374.amin().item())
        print('triton__6', 'in_ptr3', 'buf90', (buf90.sum()/buf90.nelement()).item(), buf90.amax().item(), buf90.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_133', (squeeze_133.sum()/squeeze_133.nelement()).item(), squeeze_133.amax().item(), squeeze_133.amin().item())
        print('triton__6', 'in_ptr5', 'buf89', (buf89.sum()/buf89.nelement()).item(), buf89.amax().item(), buf89.amin().item())
        print('triton__6', 'in_ptr6', 'primals_284', (primals_284.sum()/primals_284.nelement()).item(), primals_284.amax().item(), primals_284.amin().item())
        triton__6.run(buf92, relu_40, convolution_44, unsqueeze_374, buf90, squeeze_133, buf89, primals_284, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf92', (buf92.sum()/buf92.nelement()).item(), buf92.amax().item(), buf92.amin().item())
        del convolution_44
        del primals_284
        del relu_40
        del squeeze_133
        del unsqueeze_374
        buf93 = aten.convolution_backward(buf92, relu_39, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf92
        del primals_45
        buf94 = buf93[0]
        assert_size_stride(buf94, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf95 = buf93[1]
        assert_size_stride(buf95, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf93
        buf96 = buf53; del buf53  # reuse
        print('triton__14', 'in_out_ptr0', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__14', 'in_ptr0', 'relu_39', (relu_39.sum()/relu_39.nelement()).item(), relu_39.amax().item(), relu_39.amin().item())
        print('triton__14', 'in_ptr1', 'relu_42', (relu_42.sum()/relu_42.nelement()).item(), relu_42.amax().item(), relu_42.amin().item())
        print('triton__14', 'in_ptr2', 'buf73', (buf73.sum()/buf73.nelement()).item(), buf73.amax().item(), buf73.amin().item())
        print('triton__14', 'in_ptr3', 'buf94', (buf94.sum()/buf94.nelement()).item(), buf94.amax().item(), buf94.amin().item())
        triton__14.run(buf96, relu_39, relu_42, buf73, buf94, 327680, grid=grid(327680), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        del buf73
        del relu_39
        del relu_42
        buf97 = buf76; del buf76  # reuse
        buf98 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf99 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__10', 'in_ptr0', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__10', 'in_ptr1', 'convolution_43', (convolution_43.sum()/convolution_43.nelement()).item(), convolution_43.amax().item(), convolution_43.amin().item())
        print('triton__10', 'in_ptr2', 'unsqueeze_386', (unsqueeze_386.sum()/unsqueeze_386.nelement()).item(), unsqueeze_386.amax().item(), unsqueeze_386.amin().item())
        print('triton__10', 'in_ptr3', 'squeeze_130', (squeeze_130.sum()/squeeze_130.nelement()).item(), squeeze_130.amax().item(), squeeze_130.amin().item())
        triton__10.run(buf96, convolution_43, unsqueeze_386, squeeze_130, buf97, buf98, buf99, 640, 512, grid=grid(640), stream=stream0)
        print('triton__10', 'out_ptr0', 'buf97', (buf97.sum()/buf97.nelement()).item(), buf97.amax().item(), buf97.amin().item())
        print('triton__10', 'out_ptr1', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        print('triton__10', 'out_ptr2', 'buf99', (buf99.sum()/buf99.nelement()).item(), buf99.amax().item(), buf99.amin().item())
        buf100 = buf94; del buf94  # reuse
        print('triton__11', 'in_ptr0', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__11', 'in_ptr1', 'convolution_43', (convolution_43.sum()/convolution_43.nelement()).item(), convolution_43.amax().item(), convolution_43.amin().item())
        print('triton__11', 'in_ptr2', 'unsqueeze_386', (unsqueeze_386.sum()/unsqueeze_386.nelement()).item(), unsqueeze_386.amax().item(), unsqueeze_386.amin().item())
        print('triton__11', 'in_ptr3', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        print('triton__11', 'in_ptr4', 'squeeze_130', (squeeze_130.sum()/squeeze_130.nelement()).item(), squeeze_130.amax().item(), squeeze_130.amin().item())
        print('triton__11', 'in_ptr5', 'buf97', (buf97.sum()/buf97.nelement()).item(), buf97.amax().item(), buf97.amin().item())
        print('triton__11', 'in_ptr6', 'primals_279', (primals_279.sum()/primals_279.nelement()).item(), primals_279.amax().item(), primals_279.amin().item())
        triton__11.run(buf96, convolution_43, unsqueeze_386, buf98, squeeze_130, buf97, primals_279, buf100, 327680, grid=grid(327680), stream=stream0)
        print('triton__11', 'out_ptr0', 'buf100', (buf100.sum()/buf100.nelement()).item(), buf100.amax().item(), buf100.amin().item())
        del convolution_43
        del primals_279
        del squeeze_130
        del unsqueeze_386
        buf101 = aten.convolution_backward(buf100, relu_38, primals_44, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_44
        buf102 = buf101[0]
        assert_size_stride(buf102, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf103 = buf101[1]
        assert_size_stride(buf103, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf101
        buf104 = buf90; del buf90  # reuse
        buf105 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf106 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_38', (relu_38.sum()/relu_38.nelement()).item(), relu_38.amax().item(), relu_38.amin().item())
        print('triton__5', 'in_ptr1', 'buf102', (buf102.sum()/buf102.nelement()).item(), buf102.amax().item(), buf102.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_42', (convolution_42.sum()/convolution_42.nelement()).item(), convolution_42.amax().item(), convolution_42.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_398', (unsqueeze_398.sum()/unsqueeze_398.nelement()).item(), unsqueeze_398.amax().item(), unsqueeze_398.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_127', (squeeze_127.sum()/squeeze_127.nelement()).item(), squeeze_127.amax().item(), squeeze_127.amin().item())
        triton__5.run(relu_38, buf102, convolution_42, unsqueeze_398, squeeze_127, buf104, buf105, buf106, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf104', (buf104.sum()/buf104.nelement()).item(), buf104.amax().item(), buf104.amin().item())
        print('triton__5', 'out_ptr1', 'buf105', (buf105.sum()/buf105.nelement()).item(), buf105.amax().item(), buf105.amin().item())
        print('triton__5', 'out_ptr2', 'buf106', (buf106.sum()/buf106.nelement()).item(), buf106.amax().item(), buf106.amin().item())
        buf107 = buf102; del buf102  # reuse
        print('triton__6', 'in_out_ptr0', 'buf107', (buf107.sum()/buf107.nelement()).item(), buf107.amax().item(), buf107.amin().item())
        print('triton__6', 'in_ptr0', 'relu_38', (relu_38.sum()/relu_38.nelement()).item(), relu_38.amax().item(), relu_38.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_42', (convolution_42.sum()/convolution_42.nelement()).item(), convolution_42.amax().item(), convolution_42.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_398', (unsqueeze_398.sum()/unsqueeze_398.nelement()).item(), unsqueeze_398.amax().item(), unsqueeze_398.amin().item())
        print('triton__6', 'in_ptr3', 'buf105', (buf105.sum()/buf105.nelement()).item(), buf105.amax().item(), buf105.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_127', (squeeze_127.sum()/squeeze_127.nelement()).item(), squeeze_127.amax().item(), squeeze_127.amin().item())
        print('triton__6', 'in_ptr5', 'buf104', (buf104.sum()/buf104.nelement()).item(), buf104.amax().item(), buf104.amin().item())
        print('triton__6', 'in_ptr6', 'primals_274', (primals_274.sum()/primals_274.nelement()).item(), primals_274.amax().item(), primals_274.amin().item())
        triton__6.run(buf107, relu_38, convolution_42, unsqueeze_398, buf105, squeeze_127, buf104, primals_274, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf107', (buf107.sum()/buf107.nelement()).item(), buf107.amax().item(), buf107.amin().item())
        del convolution_42
        del primals_274
        del relu_38
        del squeeze_127
        del unsqueeze_398
        buf108 = aten.convolution_backward(buf107, relu_37, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf107
        del primals_43
        buf109 = buf108[0]
        assert_size_stride(buf109, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf110 = buf108[1]
        assert_size_stride(buf110, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf108
        buf111 = buf105; del buf105  # reuse
        buf112 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf113 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_37', (relu_37.sum()/relu_37.nelement()).item(), relu_37.amax().item(), relu_37.amin().item())
        print('triton__5', 'in_ptr1', 'buf109', (buf109.sum()/buf109.nelement()).item(), buf109.amax().item(), buf109.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_41', (convolution_41.sum()/convolution_41.nelement()).item(), convolution_41.amax().item(), convolution_41.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_410', (unsqueeze_410.sum()/unsqueeze_410.nelement()).item(), unsqueeze_410.amax().item(), unsqueeze_410.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_124', (squeeze_124.sum()/squeeze_124.nelement()).item(), squeeze_124.amax().item(), squeeze_124.amin().item())
        triton__5.run(relu_37, buf109, convolution_41, unsqueeze_410, squeeze_124, buf111, buf112, buf113, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf111', (buf111.sum()/buf111.nelement()).item(), buf111.amax().item(), buf111.amin().item())
        print('triton__5', 'out_ptr1', 'buf112', (buf112.sum()/buf112.nelement()).item(), buf112.amax().item(), buf112.amin().item())
        print('triton__5', 'out_ptr2', 'buf113', (buf113.sum()/buf113.nelement()).item(), buf113.amax().item(), buf113.amin().item())
        buf114 = buf109; del buf109  # reuse
        print('triton__6', 'in_out_ptr0', 'buf114', (buf114.sum()/buf114.nelement()).item(), buf114.amax().item(), buf114.amin().item())
        print('triton__6', 'in_ptr0', 'relu_37', (relu_37.sum()/relu_37.nelement()).item(), relu_37.amax().item(), relu_37.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_41', (convolution_41.sum()/convolution_41.nelement()).item(), convolution_41.amax().item(), convolution_41.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_410', (unsqueeze_410.sum()/unsqueeze_410.nelement()).item(), unsqueeze_410.amax().item(), unsqueeze_410.amin().item())
        print('triton__6', 'in_ptr3', 'buf112', (buf112.sum()/buf112.nelement()).item(), buf112.amax().item(), buf112.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_124', (squeeze_124.sum()/squeeze_124.nelement()).item(), squeeze_124.amax().item(), squeeze_124.amin().item())
        print('triton__6', 'in_ptr5', 'buf111', (buf111.sum()/buf111.nelement()).item(), buf111.amax().item(), buf111.amin().item())
        print('triton__6', 'in_ptr6', 'primals_269', (primals_269.sum()/primals_269.nelement()).item(), primals_269.amax().item(), primals_269.amin().item())
        triton__6.run(buf114, relu_37, convolution_41, unsqueeze_410, buf112, squeeze_124, buf111, primals_269, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf114', (buf114.sum()/buf114.nelement()).item(), buf114.amax().item(), buf114.amin().item())
        del convolution_41
        del primals_269
        del relu_37
        del squeeze_124
        del unsqueeze_410
        buf115 = aten.convolution_backward(buf114, relu_36, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf114
        del primals_42
        buf116 = buf115[0]
        assert_size_stride(buf116, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf117 = buf115[1]
        assert_size_stride(buf117, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf115
        buf118 = buf98; del buf98  # reuse
        buf119 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf121 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__12', 'in_ptr0', 'relu_36', (relu_36.sum()/relu_36.nelement()).item(), relu_36.amax().item(), relu_36.amin().item())
        print('triton__12', 'in_ptr1', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__12', 'in_ptr2', 'buf116', (buf116.sum()/buf116.nelement()).item(), buf116.amax().item(), buf116.amin().item())
        print('triton__12', 'in_ptr3', 'convolution_40', (convolution_40.sum()/convolution_40.nelement()).item(), convolution_40.amax().item(), convolution_40.amin().item())
        print('triton__12', 'in_ptr4', 'unsqueeze_422', (unsqueeze_422.sum()/unsqueeze_422.nelement()).item(), unsqueeze_422.amax().item(), unsqueeze_422.amin().item())
        print('triton__12', 'in_ptr5', 'squeeze_121', (squeeze_121.sum()/squeeze_121.nelement()).item(), squeeze_121.amax().item(), squeeze_121.amin().item())
        triton__12.run(relu_36, buf96, buf116, convolution_40, unsqueeze_422, squeeze_121, buf118, buf119, buf121, 640, 512, grid=grid(640), stream=stream0)
        print('triton__12', 'out_ptr0', 'buf118', (buf118.sum()/buf118.nelement()).item(), buf118.amax().item(), buf118.amin().item())
        print('triton__12', 'out_ptr1', 'buf119', (buf119.sum()/buf119.nelement()).item(), buf119.amax().item(), buf119.amin().item())
        print('triton__12', 'out_ptr2', 'buf121', (buf121.sum()/buf121.nelement()).item(), buf121.amax().item(), buf121.amin().item())
        buf120 = buf100; del buf100  # reuse
        print('triton__13', 'in_ptr0', 'relu_36', (relu_36.sum()/relu_36.nelement()).item(), relu_36.amax().item(), relu_36.amin().item())
        print('triton__13', 'in_ptr1', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__13', 'in_ptr2', 'buf116', (buf116.sum()/buf116.nelement()).item(), buf116.amax().item(), buf116.amin().item())
        print('triton__13', 'in_ptr3', 'convolution_40', (convolution_40.sum()/convolution_40.nelement()).item(), convolution_40.amax().item(), convolution_40.amin().item())
        print('triton__13', 'in_ptr4', 'unsqueeze_422', (unsqueeze_422.sum()/unsqueeze_422.nelement()).item(), unsqueeze_422.amax().item(), unsqueeze_422.amin().item())
        print('triton__13', 'in_ptr5', 'buf119', (buf119.sum()/buf119.nelement()).item(), buf119.amax().item(), buf119.amin().item())
        print('triton__13', 'in_ptr6', 'squeeze_121', (squeeze_121.sum()/squeeze_121.nelement()).item(), squeeze_121.amax().item(), squeeze_121.amin().item())
        print('triton__13', 'in_ptr7', 'buf118', (buf118.sum()/buf118.nelement()).item(), buf118.amax().item(), buf118.amin().item())
        print('triton__13', 'in_ptr8', 'primals_264', (primals_264.sum()/primals_264.nelement()).item(), primals_264.amax().item(), primals_264.amin().item())
        triton__13.run(relu_36, buf96, buf116, convolution_40, unsqueeze_422, buf119, squeeze_121, buf118, primals_264, buf120, 327680, grid=grid(327680), stream=stream0)
        print('triton__13', 'out_ptr0', 'buf120', (buf120.sum()/buf120.nelement()).item(), buf120.amax().item(), buf120.amin().item())
        del convolution_40
        del primals_264
        del squeeze_121
        del unsqueeze_422
        buf122 = aten.convolution_backward(buf120, relu_35, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf120
        del primals_41
        buf123 = buf122[0]
        assert_size_stride(buf123, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf124 = buf122[1]
        assert_size_stride(buf124, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf122
        buf125 = buf112; del buf112  # reuse
        buf126 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf127 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_35', (relu_35.sum()/relu_35.nelement()).item(), relu_35.amax().item(), relu_35.amin().item())
        print('triton__5', 'in_ptr1', 'buf123', (buf123.sum()/buf123.nelement()).item(), buf123.amax().item(), buf123.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_39', (convolution_39.sum()/convolution_39.nelement()).item(), convolution_39.amax().item(), convolution_39.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_434', (unsqueeze_434.sum()/unsqueeze_434.nelement()).item(), unsqueeze_434.amax().item(), unsqueeze_434.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_118', (squeeze_118.sum()/squeeze_118.nelement()).item(), squeeze_118.amax().item(), squeeze_118.amin().item())
        triton__5.run(relu_35, buf123, convolution_39, unsqueeze_434, squeeze_118, buf125, buf126, buf127, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf125', (buf125.sum()/buf125.nelement()).item(), buf125.amax().item(), buf125.amin().item())
        print('triton__5', 'out_ptr1', 'buf126', (buf126.sum()/buf126.nelement()).item(), buf126.amax().item(), buf126.amin().item())
        print('triton__5', 'out_ptr2', 'buf127', (buf127.sum()/buf127.nelement()).item(), buf127.amax().item(), buf127.amin().item())
        buf128 = buf123; del buf123  # reuse
        print('triton__6', 'in_out_ptr0', 'buf128', (buf128.sum()/buf128.nelement()).item(), buf128.amax().item(), buf128.amin().item())
        print('triton__6', 'in_ptr0', 'relu_35', (relu_35.sum()/relu_35.nelement()).item(), relu_35.amax().item(), relu_35.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_39', (convolution_39.sum()/convolution_39.nelement()).item(), convolution_39.amax().item(), convolution_39.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_434', (unsqueeze_434.sum()/unsqueeze_434.nelement()).item(), unsqueeze_434.amax().item(), unsqueeze_434.amin().item())
        print('triton__6', 'in_ptr3', 'buf126', (buf126.sum()/buf126.nelement()).item(), buf126.amax().item(), buf126.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_118', (squeeze_118.sum()/squeeze_118.nelement()).item(), squeeze_118.amax().item(), squeeze_118.amin().item())
        print('triton__6', 'in_ptr5', 'buf125', (buf125.sum()/buf125.nelement()).item(), buf125.amax().item(), buf125.amin().item())
        print('triton__6', 'in_ptr6', 'primals_259', (primals_259.sum()/primals_259.nelement()).item(), primals_259.amax().item(), primals_259.amin().item())
        triton__6.run(buf128, relu_35, convolution_39, unsqueeze_434, buf126, squeeze_118, buf125, primals_259, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf128', (buf128.sum()/buf128.nelement()).item(), buf128.amax().item(), buf128.amin().item())
        del convolution_39
        del primals_259
        del relu_35
        del squeeze_118
        del unsqueeze_434
        buf129 = aten.convolution_backward(buf128, relu_34, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf128
        del primals_40
        buf130 = buf129[0]
        assert_size_stride(buf130, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf131 = buf129[1]
        assert_size_stride(buf131, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf129
        buf132 = buf126; del buf126  # reuse
        buf133 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_34', (relu_34.sum()/relu_34.nelement()).item(), relu_34.amax().item(), relu_34.amin().item())
        print('triton__5', 'in_ptr1', 'buf130', (buf130.sum()/buf130.nelement()).item(), buf130.amax().item(), buf130.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_38', (convolution_38.sum()/convolution_38.nelement()).item(), convolution_38.amax().item(), convolution_38.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_446', (unsqueeze_446.sum()/unsqueeze_446.nelement()).item(), unsqueeze_446.amax().item(), unsqueeze_446.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_115', (squeeze_115.sum()/squeeze_115.nelement()).item(), squeeze_115.amax().item(), squeeze_115.amin().item())
        triton__5.run(relu_34, buf130, convolution_38, unsqueeze_446, squeeze_115, buf132, buf133, buf134, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf132', (buf132.sum()/buf132.nelement()).item(), buf132.amax().item(), buf132.amin().item())
        print('triton__5', 'out_ptr1', 'buf133', (buf133.sum()/buf133.nelement()).item(), buf133.amax().item(), buf133.amin().item())
        print('triton__5', 'out_ptr2', 'buf134', (buf134.sum()/buf134.nelement()).item(), buf134.amax().item(), buf134.amin().item())
        buf135 = buf130; del buf130  # reuse
        print('triton__6', 'in_out_ptr0', 'buf135', (buf135.sum()/buf135.nelement()).item(), buf135.amax().item(), buf135.amin().item())
        print('triton__6', 'in_ptr0', 'relu_34', (relu_34.sum()/relu_34.nelement()).item(), relu_34.amax().item(), relu_34.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_38', (convolution_38.sum()/convolution_38.nelement()).item(), convolution_38.amax().item(), convolution_38.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_446', (unsqueeze_446.sum()/unsqueeze_446.nelement()).item(), unsqueeze_446.amax().item(), unsqueeze_446.amin().item())
        print('triton__6', 'in_ptr3', 'buf133', (buf133.sum()/buf133.nelement()).item(), buf133.amax().item(), buf133.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_115', (squeeze_115.sum()/squeeze_115.nelement()).item(), squeeze_115.amax().item(), squeeze_115.amin().item())
        print('triton__6', 'in_ptr5', 'buf132', (buf132.sum()/buf132.nelement()).item(), buf132.amax().item(), buf132.amin().item())
        print('triton__6', 'in_ptr6', 'primals_254', (primals_254.sum()/primals_254.nelement()).item(), primals_254.amax().item(), primals_254.amin().item())
        triton__6.run(buf135, relu_34, convolution_38, unsqueeze_446, buf133, squeeze_115, buf132, primals_254, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf135', (buf135.sum()/buf135.nelement()).item(), buf135.amax().item(), buf135.amin().item())
        del convolution_38
        del primals_254
        del relu_34
        del squeeze_115
        del unsqueeze_446
        buf136 = aten.convolution_backward(buf135, relu_33, primals_39, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf135
        del primals_39
        buf137 = buf136[0]
        assert_size_stride(buf137, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf138 = buf136[1]
        assert_size_stride(buf138, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf136
        buf139 = buf116; del buf116  # reuse
        print('triton__15', 'in_out_ptr0', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        print('triton__15', 'in_ptr0', 'relu_33', (relu_33.sum()/relu_33.nelement()).item(), relu_33.amax().item(), relu_33.amin().item())
        print('triton__15', 'in_ptr1', 'relu_36', (relu_36.sum()/relu_36.nelement()).item(), relu_36.amax().item(), relu_36.amin().item())
        print('triton__15', 'in_ptr2', 'buf96', (buf96.sum()/buf96.nelement()).item(), buf96.amax().item(), buf96.amin().item())
        print('triton__15', 'in_ptr3', 'buf137', (buf137.sum()/buf137.nelement()).item(), buf137.amax().item(), buf137.amin().item())
        triton__15.run(buf139, relu_33, relu_36, buf96, buf137, 327680, grid=grid(327680), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        del buf137
        del relu_33
        del relu_36
        buf140 = buf119; del buf119  # reuse
        buf141 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__10', 'in_ptr0', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        print('triton__10', 'in_ptr1', 'convolution_37', (convolution_37.sum()/convolution_37.nelement()).item(), convolution_37.amax().item(), convolution_37.amin().item())
        print('triton__10', 'in_ptr2', 'unsqueeze_458', (unsqueeze_458.sum()/unsqueeze_458.nelement()).item(), unsqueeze_458.amax().item(), unsqueeze_458.amin().item())
        print('triton__10', 'in_ptr3', 'squeeze_112', (squeeze_112.sum()/squeeze_112.nelement()).item(), squeeze_112.amax().item(), squeeze_112.amin().item())
        triton__10.run(buf139, convolution_37, unsqueeze_458, squeeze_112, buf140, buf141, buf142, 640, 512, grid=grid(640), stream=stream0)
        print('triton__10', 'out_ptr0', 'buf140', (buf140.sum()/buf140.nelement()).item(), buf140.amax().item(), buf140.amin().item())
        print('triton__10', 'out_ptr1', 'buf141', (buf141.sum()/buf141.nelement()).item(), buf141.amax().item(), buf141.amin().item())
        print('triton__10', 'out_ptr2', 'buf142', (buf142.sum()/buf142.nelement()).item(), buf142.amax().item(), buf142.amin().item())
        buf143 = buf96; del buf96  # reuse
        print('triton__11', 'in_ptr0', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        print('triton__11', 'in_ptr1', 'convolution_37', (convolution_37.sum()/convolution_37.nelement()).item(), convolution_37.amax().item(), convolution_37.amin().item())
        print('triton__11', 'in_ptr2', 'unsqueeze_458', (unsqueeze_458.sum()/unsqueeze_458.nelement()).item(), unsqueeze_458.amax().item(), unsqueeze_458.amin().item())
        print('triton__11', 'in_ptr3', 'buf141', (buf141.sum()/buf141.nelement()).item(), buf141.amax().item(), buf141.amin().item())
        print('triton__11', 'in_ptr4', 'squeeze_112', (squeeze_112.sum()/squeeze_112.nelement()).item(), squeeze_112.amax().item(), squeeze_112.amin().item())
        print('triton__11', 'in_ptr5', 'buf140', (buf140.sum()/buf140.nelement()).item(), buf140.amax().item(), buf140.amin().item())
        print('triton__11', 'in_ptr6', 'primals_249', (primals_249.sum()/primals_249.nelement()).item(), primals_249.amax().item(), primals_249.amin().item())
        triton__11.run(buf139, convolution_37, unsqueeze_458, buf141, squeeze_112, buf140, primals_249, buf143, 327680, grid=grid(327680), stream=stream0)
        print('triton__11', 'out_ptr0', 'buf143', (buf143.sum()/buf143.nelement()).item(), buf143.amax().item(), buf143.amin().item())
        del convolution_37
        del primals_249
        del squeeze_112
        del unsqueeze_458
        buf144 = aten.convolution_backward(buf143, relu_32, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_38
        buf145 = buf144[0]
        assert_size_stride(buf145, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf146 = buf144[1]
        assert_size_stride(buf146, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf144
        buf147 = buf133; del buf133  # reuse
        buf148 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf149 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_32', (relu_32.sum()/relu_32.nelement()).item(), relu_32.amax().item(), relu_32.amin().item())
        print('triton__5', 'in_ptr1', 'buf145', (buf145.sum()/buf145.nelement()).item(), buf145.amax().item(), buf145.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_36', (convolution_36.sum()/convolution_36.nelement()).item(), convolution_36.amax().item(), convolution_36.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_470', (unsqueeze_470.sum()/unsqueeze_470.nelement()).item(), unsqueeze_470.amax().item(), unsqueeze_470.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_109', (squeeze_109.sum()/squeeze_109.nelement()).item(), squeeze_109.amax().item(), squeeze_109.amin().item())
        triton__5.run(relu_32, buf145, convolution_36, unsqueeze_470, squeeze_109, buf147, buf148, buf149, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf147', (buf147.sum()/buf147.nelement()).item(), buf147.amax().item(), buf147.amin().item())
        print('triton__5', 'out_ptr1', 'buf148', (buf148.sum()/buf148.nelement()).item(), buf148.amax().item(), buf148.amin().item())
        print('triton__5', 'out_ptr2', 'buf149', (buf149.sum()/buf149.nelement()).item(), buf149.amax().item(), buf149.amin().item())
        buf150 = buf145; del buf145  # reuse
        print('triton__6', 'in_out_ptr0', 'buf150', (buf150.sum()/buf150.nelement()).item(), buf150.amax().item(), buf150.amin().item())
        print('triton__6', 'in_ptr0', 'relu_32', (relu_32.sum()/relu_32.nelement()).item(), relu_32.amax().item(), relu_32.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_36', (convolution_36.sum()/convolution_36.nelement()).item(), convolution_36.amax().item(), convolution_36.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_470', (unsqueeze_470.sum()/unsqueeze_470.nelement()).item(), unsqueeze_470.amax().item(), unsqueeze_470.amin().item())
        print('triton__6', 'in_ptr3', 'buf148', (buf148.sum()/buf148.nelement()).item(), buf148.amax().item(), buf148.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_109', (squeeze_109.sum()/squeeze_109.nelement()).item(), squeeze_109.amax().item(), squeeze_109.amin().item())
        print('triton__6', 'in_ptr5', 'buf147', (buf147.sum()/buf147.nelement()).item(), buf147.amax().item(), buf147.amin().item())
        print('triton__6', 'in_ptr6', 'primals_244', (primals_244.sum()/primals_244.nelement()).item(), primals_244.amax().item(), primals_244.amin().item())
        triton__6.run(buf150, relu_32, convolution_36, unsqueeze_470, buf148, squeeze_109, buf147, primals_244, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf150', (buf150.sum()/buf150.nelement()).item(), buf150.amax().item(), buf150.amin().item())
        del convolution_36
        del primals_244
        del relu_32
        del squeeze_109
        del unsqueeze_470
        buf151 = aten.convolution_backward(buf150, relu_31, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf150
        del primals_37
        buf152 = buf151[0]
        assert_size_stride(buf152, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf153 = buf151[1]
        assert_size_stride(buf153, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf151
        buf154 = buf148; del buf148  # reuse
        buf155 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf156 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_31', (relu_31.sum()/relu_31.nelement()).item(), relu_31.amax().item(), relu_31.amin().item())
        print('triton__5', 'in_ptr1', 'buf152', (buf152.sum()/buf152.nelement()).item(), buf152.amax().item(), buf152.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_35', (convolution_35.sum()/convolution_35.nelement()).item(), convolution_35.amax().item(), convolution_35.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_482', (unsqueeze_482.sum()/unsqueeze_482.nelement()).item(), unsqueeze_482.amax().item(), unsqueeze_482.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_106', (squeeze_106.sum()/squeeze_106.nelement()).item(), squeeze_106.amax().item(), squeeze_106.amin().item())
        triton__5.run(relu_31, buf152, convolution_35, unsqueeze_482, squeeze_106, buf154, buf155, buf156, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf154', (buf154.sum()/buf154.nelement()).item(), buf154.amax().item(), buf154.amin().item())
        print('triton__5', 'out_ptr1', 'buf155', (buf155.sum()/buf155.nelement()).item(), buf155.amax().item(), buf155.amin().item())
        print('triton__5', 'out_ptr2', 'buf156', (buf156.sum()/buf156.nelement()).item(), buf156.amax().item(), buf156.amin().item())
        buf157 = buf152; del buf152  # reuse
        print('triton__6', 'in_out_ptr0', 'buf157', (buf157.sum()/buf157.nelement()).item(), buf157.amax().item(), buf157.amin().item())
        print('triton__6', 'in_ptr0', 'relu_31', (relu_31.sum()/relu_31.nelement()).item(), relu_31.amax().item(), relu_31.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_35', (convolution_35.sum()/convolution_35.nelement()).item(), convolution_35.amax().item(), convolution_35.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_482', (unsqueeze_482.sum()/unsqueeze_482.nelement()).item(), unsqueeze_482.amax().item(), unsqueeze_482.amin().item())
        print('triton__6', 'in_ptr3', 'buf155', (buf155.sum()/buf155.nelement()).item(), buf155.amax().item(), buf155.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_106', (squeeze_106.sum()/squeeze_106.nelement()).item(), squeeze_106.amax().item(), squeeze_106.amin().item())
        print('triton__6', 'in_ptr5', 'buf154', (buf154.sum()/buf154.nelement()).item(), buf154.amax().item(), buf154.amin().item())
        print('triton__6', 'in_ptr6', 'primals_239', (primals_239.sum()/primals_239.nelement()).item(), primals_239.amax().item(), primals_239.amin().item())
        triton__6.run(buf157, relu_31, convolution_35, unsqueeze_482, buf155, squeeze_106, buf154, primals_239, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf157', (buf157.sum()/buf157.nelement()).item(), buf157.amax().item(), buf157.amin().item())
        del convolution_35
        del primals_239
        del relu_31
        del squeeze_106
        del unsqueeze_482
        buf158 = aten.convolution_backward(buf157, relu_30, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf157
        del primals_36
        buf159 = buf158[0]
        assert_size_stride(buf159, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf160 = buf158[1]
        assert_size_stride(buf160, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf158
        buf161 = buf141; del buf141  # reuse
        buf162 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__12', 'in_ptr0', 'relu_30', (relu_30.sum()/relu_30.nelement()).item(), relu_30.amax().item(), relu_30.amin().item())
        print('triton__12', 'in_ptr1', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        print('triton__12', 'in_ptr2', 'buf159', (buf159.sum()/buf159.nelement()).item(), buf159.amax().item(), buf159.amin().item())
        print('triton__12', 'in_ptr3', 'convolution_34', (convolution_34.sum()/convolution_34.nelement()).item(), convolution_34.amax().item(), convolution_34.amin().item())
        print('triton__12', 'in_ptr4', 'unsqueeze_494', (unsqueeze_494.sum()/unsqueeze_494.nelement()).item(), unsqueeze_494.amax().item(), unsqueeze_494.amin().item())
        print('triton__12', 'in_ptr5', 'squeeze_103', (squeeze_103.sum()/squeeze_103.nelement()).item(), squeeze_103.amax().item(), squeeze_103.amin().item())
        triton__12.run(relu_30, buf139, buf159, convolution_34, unsqueeze_494, squeeze_103, buf161, buf162, buf164, 640, 512, grid=grid(640), stream=stream0)
        print('triton__12', 'out_ptr0', 'buf161', (buf161.sum()/buf161.nelement()).item(), buf161.amax().item(), buf161.amin().item())
        print('triton__12', 'out_ptr1', 'buf162', (buf162.sum()/buf162.nelement()).item(), buf162.amax().item(), buf162.amin().item())
        print('triton__12', 'out_ptr2', 'buf164', (buf164.sum()/buf164.nelement()).item(), buf164.amax().item(), buf164.amin().item())
        buf163 = buf143; del buf143  # reuse
        print('triton__13', 'in_ptr0', 'relu_30', (relu_30.sum()/relu_30.nelement()).item(), relu_30.amax().item(), relu_30.amin().item())
        print('triton__13', 'in_ptr1', 'buf139', (buf139.sum()/buf139.nelement()).item(), buf139.amax().item(), buf139.amin().item())
        print('triton__13', 'in_ptr2', 'buf159', (buf159.sum()/buf159.nelement()).item(), buf159.amax().item(), buf159.amin().item())
        print('triton__13', 'in_ptr3', 'convolution_34', (convolution_34.sum()/convolution_34.nelement()).item(), convolution_34.amax().item(), convolution_34.amin().item())
        print('triton__13', 'in_ptr4', 'unsqueeze_494', (unsqueeze_494.sum()/unsqueeze_494.nelement()).item(), unsqueeze_494.amax().item(), unsqueeze_494.amin().item())
        print('triton__13', 'in_ptr5', 'buf162', (buf162.sum()/buf162.nelement()).item(), buf162.amax().item(), buf162.amin().item())
        print('triton__13', 'in_ptr6', 'squeeze_103', (squeeze_103.sum()/squeeze_103.nelement()).item(), squeeze_103.amax().item(), squeeze_103.amin().item())
        print('triton__13', 'in_ptr7', 'buf161', (buf161.sum()/buf161.nelement()).item(), buf161.amax().item(), buf161.amin().item())
        print('triton__13', 'in_ptr8', 'primals_234', (primals_234.sum()/primals_234.nelement()).item(), primals_234.amax().item(), primals_234.amin().item())
        triton__13.run(relu_30, buf139, buf159, convolution_34, unsqueeze_494, buf162, squeeze_103, buf161, primals_234, buf163, 327680, grid=grid(327680), stream=stream0)
        print('triton__13', 'out_ptr0', 'buf163', (buf163.sum()/buf163.nelement()).item(), buf163.amax().item(), buf163.amin().item())
        del convolution_34
        del primals_234
        del squeeze_103
        del unsqueeze_494
        buf165 = aten.convolution_backward(buf163, relu_29, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf163
        del primals_35
        buf166 = buf165[0]
        assert_size_stride(buf166, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf167 = buf165[1]
        assert_size_stride(buf167, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf165
        buf168 = buf155; del buf155  # reuse
        buf169 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_29', (relu_29.sum()/relu_29.nelement()).item(), relu_29.amax().item(), relu_29.amin().item())
        print('triton__5', 'in_ptr1', 'buf166', (buf166.sum()/buf166.nelement()).item(), buf166.amax().item(), buf166.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_33', (convolution_33.sum()/convolution_33.nelement()).item(), convolution_33.amax().item(), convolution_33.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_506', (unsqueeze_506.sum()/unsqueeze_506.nelement()).item(), unsqueeze_506.amax().item(), unsqueeze_506.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_100', (squeeze_100.sum()/squeeze_100.nelement()).item(), squeeze_100.amax().item(), squeeze_100.amin().item())
        triton__5.run(relu_29, buf166, convolution_33, unsqueeze_506, squeeze_100, buf168, buf169, buf170, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf168', (buf168.sum()/buf168.nelement()).item(), buf168.amax().item(), buf168.amin().item())
        print('triton__5', 'out_ptr1', 'buf169', (buf169.sum()/buf169.nelement()).item(), buf169.amax().item(), buf169.amin().item())
        print('triton__5', 'out_ptr2', 'buf170', (buf170.sum()/buf170.nelement()).item(), buf170.amax().item(), buf170.amin().item())
        buf171 = buf166; del buf166  # reuse
        print('triton__6', 'in_out_ptr0', 'buf171', (buf171.sum()/buf171.nelement()).item(), buf171.amax().item(), buf171.amin().item())
        print('triton__6', 'in_ptr0', 'relu_29', (relu_29.sum()/relu_29.nelement()).item(), relu_29.amax().item(), relu_29.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_33', (convolution_33.sum()/convolution_33.nelement()).item(), convolution_33.amax().item(), convolution_33.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_506', (unsqueeze_506.sum()/unsqueeze_506.nelement()).item(), unsqueeze_506.amax().item(), unsqueeze_506.amin().item())
        print('triton__6', 'in_ptr3', 'buf169', (buf169.sum()/buf169.nelement()).item(), buf169.amax().item(), buf169.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_100', (squeeze_100.sum()/squeeze_100.nelement()).item(), squeeze_100.amax().item(), squeeze_100.amin().item())
        print('triton__6', 'in_ptr5', 'buf168', (buf168.sum()/buf168.nelement()).item(), buf168.amax().item(), buf168.amin().item())
        print('triton__6', 'in_ptr6', 'primals_229', (primals_229.sum()/primals_229.nelement()).item(), primals_229.amax().item(), primals_229.amin().item())
        triton__6.run(buf171, relu_29, convolution_33, unsqueeze_506, buf169, squeeze_100, buf168, primals_229, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf171', (buf171.sum()/buf171.nelement()).item(), buf171.amax().item(), buf171.amin().item())
        del convolution_33
        del primals_229
        del relu_29
        del squeeze_100
        del unsqueeze_506
        buf172 = aten.convolution_backward(buf171, relu_28, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf171
        del primals_34
        buf173 = buf172[0]
        assert_size_stride(buf173, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf174 = buf172[1]
        assert_size_stride(buf174, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf172
        buf175 = buf169; del buf169  # reuse
        buf176 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf177 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_28', (relu_28.sum()/relu_28.nelement()).item(), relu_28.amax().item(), relu_28.amin().item())
        print('triton__5', 'in_ptr1', 'buf173', (buf173.sum()/buf173.nelement()).item(), buf173.amax().item(), buf173.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_32', (convolution_32.sum()/convolution_32.nelement()).item(), convolution_32.amax().item(), convolution_32.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_518', (unsqueeze_518.sum()/unsqueeze_518.nelement()).item(), unsqueeze_518.amax().item(), unsqueeze_518.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_97', (squeeze_97.sum()/squeeze_97.nelement()).item(), squeeze_97.amax().item(), squeeze_97.amin().item())
        triton__5.run(relu_28, buf173, convolution_32, unsqueeze_518, squeeze_97, buf175, buf176, buf177, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf175', (buf175.sum()/buf175.nelement()).item(), buf175.amax().item(), buf175.amin().item())
        print('triton__5', 'out_ptr1', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__5', 'out_ptr2', 'buf177', (buf177.sum()/buf177.nelement()).item(), buf177.amax().item(), buf177.amin().item())
        buf178 = buf173; del buf173  # reuse
        print('triton__6', 'in_out_ptr0', 'buf178', (buf178.sum()/buf178.nelement()).item(), buf178.amax().item(), buf178.amin().item())
        print('triton__6', 'in_ptr0', 'relu_28', (relu_28.sum()/relu_28.nelement()).item(), relu_28.amax().item(), relu_28.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_32', (convolution_32.sum()/convolution_32.nelement()).item(), convolution_32.amax().item(), convolution_32.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_518', (unsqueeze_518.sum()/unsqueeze_518.nelement()).item(), unsqueeze_518.amax().item(), unsqueeze_518.amin().item())
        print('triton__6', 'in_ptr3', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_97', (squeeze_97.sum()/squeeze_97.nelement()).item(), squeeze_97.amax().item(), squeeze_97.amin().item())
        print('triton__6', 'in_ptr5', 'buf175', (buf175.sum()/buf175.nelement()).item(), buf175.amax().item(), buf175.amin().item())
        print('triton__6', 'in_ptr6', 'primals_224', (primals_224.sum()/primals_224.nelement()).item(), primals_224.amax().item(), primals_224.amin().item())
        triton__6.run(buf178, relu_28, convolution_32, unsqueeze_518, buf176, squeeze_97, buf175, primals_224, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf178', (buf178.sum()/buf178.nelement()).item(), buf178.amax().item(), buf178.amin().item())
        del convolution_32
        del primals_224
        del relu_28
        del squeeze_97
        del unsqueeze_518
        buf179 = aten.convolution_backward(buf178, relu_27, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf178
        del primals_33
        buf180 = buf179[0]
        assert_size_stride(buf180, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf181 = buf179[1]
        assert_size_stride(buf181, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf179
        buf182 = buf139; del buf139  # reuse
        print('triton__14', 'in_out_ptr0', 'buf182', (buf182.sum()/buf182.nelement()).item(), buf182.amax().item(), buf182.amin().item())
        print('triton__14', 'in_ptr0', 'relu_27', (relu_27.sum()/relu_27.nelement()).item(), relu_27.amax().item(), relu_27.amin().item())
        print('triton__14', 'in_ptr1', 'relu_30', (relu_30.sum()/relu_30.nelement()).item(), relu_30.amax().item(), relu_30.amin().item())
        print('triton__14', 'in_ptr2', 'buf159', (buf159.sum()/buf159.nelement()).item(), buf159.amax().item(), buf159.amin().item())
        print('triton__14', 'in_ptr3', 'buf180', (buf180.sum()/buf180.nelement()).item(), buf180.amax().item(), buf180.amin().item())
        triton__14.run(buf182, relu_27, relu_30, buf159, buf180, 327680, grid=grid(327680), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf182', (buf182.sum()/buf182.nelement()).item(), buf182.amax().item(), buf182.amin().item())
        del relu_27
        del relu_30
        buf183 = buf162; del buf162  # reuse
        buf184 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf190 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf185 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf191 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__16', 'in_ptr0', 'buf182', (buf182.sum()/buf182.nelement()).item(), buf182.amax().item(), buf182.amin().item())
        print('triton__16', 'in_ptr1', 'convolution_31', (convolution_31.sum()/convolution_31.nelement()).item(), convolution_31.amax().item(), convolution_31.amin().item())
        print('triton__16', 'in_ptr2', 'unsqueeze_530', (unsqueeze_530.sum()/unsqueeze_530.nelement()).item(), unsqueeze_530.amax().item(), unsqueeze_530.amin().item())
        print('triton__16', 'in_ptr3', 'convolution_30', (convolution_30.sum()/convolution_30.nelement()).item(), convolution_30.amax().item(), convolution_30.amin().item())
        print('triton__16', 'in_ptr4', 'unsqueeze_542', (unsqueeze_542.sum()/unsqueeze_542.nelement()).item(), unsqueeze_542.amax().item(), unsqueeze_542.amin().item())
        print('triton__16', 'in_ptr5', 'squeeze_94', (squeeze_94.sum()/squeeze_94.nelement()).item(), squeeze_94.amax().item(), squeeze_94.amin().item())
        print('triton__16', 'in_ptr6', 'squeeze_91', (squeeze_91.sum()/squeeze_91.nelement()).item(), squeeze_91.amax().item(), squeeze_91.amin().item())
        triton__16.run(buf182, convolution_31, unsqueeze_530, convolution_30, unsqueeze_542, squeeze_94, squeeze_91, buf183, buf184, buf190, buf185, buf191, 640, 512, grid=grid(640), stream=stream0)
        print('triton__16', 'out_ptr0', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        print('triton__16', 'out_ptr1', 'buf184', (buf184.sum()/buf184.nelement()).item(), buf184.amax().item(), buf184.amin().item())
        print('triton__16', 'out_ptr2', 'buf190', (buf190.sum()/buf190.nelement()).item(), buf190.amax().item(), buf190.amin().item())
        print('triton__16', 'out_ptr3', 'buf185', (buf185.sum()/buf185.nelement()).item(), buf185.amax().item(), buf185.amin().item())
        print('triton__16', 'out_ptr4', 'buf191', (buf191.sum()/buf191.nelement()).item(), buf191.amax().item(), buf191.amin().item())
        buf186 = buf180; del buf180  # reuse
        buf192 = buf159; del buf159  # reuse
        print('triton__17', 'in_ptr0', 'buf182', (buf182.sum()/buf182.nelement()).item(), buf182.amax().item(), buf182.amin().item())
        print('triton__17', 'in_ptr1', 'convolution_31', (convolution_31.sum()/convolution_31.nelement()).item(), convolution_31.amax().item(), convolution_31.amin().item())
        print('triton__17', 'in_ptr2', 'unsqueeze_530', (unsqueeze_530.sum()/unsqueeze_530.nelement()).item(), unsqueeze_530.amax().item(), unsqueeze_530.amin().item())
        print('triton__17', 'in_ptr3', 'buf184', (buf184.sum()/buf184.nelement()).item(), buf184.amax().item(), buf184.amin().item())
        print('triton__17', 'in_ptr4', 'squeeze_94', (squeeze_94.sum()/squeeze_94.nelement()).item(), squeeze_94.amax().item(), squeeze_94.amin().item())
        print('triton__17', 'in_ptr5', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        print('triton__17', 'in_ptr6', 'primals_219', (primals_219.sum()/primals_219.nelement()).item(), primals_219.amax().item(), primals_219.amin().item())
        print('triton__17', 'in_ptr7', 'convolution_30', (convolution_30.sum()/convolution_30.nelement()).item(), convolution_30.amax().item(), convolution_30.amin().item())
        print('triton__17', 'in_ptr8', 'unsqueeze_542', (unsqueeze_542.sum()/unsqueeze_542.nelement()).item(), unsqueeze_542.amax().item(), unsqueeze_542.amin().item())
        print('triton__17', 'in_ptr9', 'buf190', (buf190.sum()/buf190.nelement()).item(), buf190.amax().item(), buf190.amin().item())
        print('triton__17', 'in_ptr10', 'squeeze_91', (squeeze_91.sum()/squeeze_91.nelement()).item(), squeeze_91.amax().item(), squeeze_91.amin().item())
        print('triton__17', 'in_ptr11', 'primals_214', (primals_214.sum()/primals_214.nelement()).item(), primals_214.amax().item(), primals_214.amin().item())
        triton__17.run(buf182, convolution_31, unsqueeze_530, buf184, squeeze_94, buf183, primals_219, convolution_30, unsqueeze_542, buf190, squeeze_91, primals_214, buf186, buf192, 327680, grid=grid(327680), stream=stream0)
        print('triton__17', 'out_ptr0', 'buf186', (buf186.sum()/buf186.nelement()).item(), buf186.amax().item(), buf186.amin().item())
        print('triton__17', 'out_ptr1', 'buf192', (buf192.sum()/buf192.nelement()).item(), buf192.amax().item(), buf192.amin().item())
        del buf182
        del convolution_30
        del convolution_31
        del primals_214
        del primals_219
        del squeeze_91
        del squeeze_94
        del unsqueeze_530
        del unsqueeze_542
        buf187 = aten.convolution_backward(buf186, relu_24, primals_32, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf186
        del primals_32
        buf188 = buf187[0]
        assert_size_stride(buf188, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf189 = buf187[1]
        assert_size_stride(buf189, (640, 640, 1, 1), (640, 1, 1, 1))
        del buf187
        buf193 = aten.convolution_backward(buf192, relu_26, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf192
        del primals_31
        buf194 = buf193[0]
        assert_size_stride(buf194, (8, 1920, 8, 8), (122880, 64, 8, 1))
        buf195 = buf193[1]
        assert_size_stride(buf195, (640, 1920, 1, 1), (1920, 1, 1, 1))
        del buf193
        buf196 = buf176; del buf176  # reuse
        buf197 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__5', 'in_ptr0', 'relu_26', (relu_26.sum()/relu_26.nelement()).item(), relu_26.amax().item(), relu_26.amin().item())
        print('triton__5', 'in_ptr1', 'buf194', (buf194.sum()/buf194.nelement()).item(), buf194.amax().item(), buf194.amin().item())
        print('triton__5', 'in_ptr2', 'convolution_29', (convolution_29.sum()/convolution_29.nelement()).item(), convolution_29.amax().item(), convolution_29.amin().item())
        print('triton__5', 'in_ptr3', 'unsqueeze_554', (unsqueeze_554.sum()/unsqueeze_554.nelement()).item(), unsqueeze_554.amax().item(), unsqueeze_554.amin().item())
        print('triton__5', 'in_ptr4', 'squeeze_88', (squeeze_88.sum()/squeeze_88.nelement()).item(), squeeze_88.amax().item(), squeeze_88.amin().item())
        triton__5.run(relu_26, buf194, convolution_29, unsqueeze_554, squeeze_88, buf196, buf197, buf198, 1920, 512, grid=grid(1920), stream=stream0)
        print('triton__5', 'out_ptr0', 'buf196', (buf196.sum()/buf196.nelement()).item(), buf196.amax().item(), buf196.amin().item())
        print('triton__5', 'out_ptr1', 'buf197', (buf197.sum()/buf197.nelement()).item(), buf197.amax().item(), buf197.amin().item())
        print('triton__5', 'out_ptr2', 'buf198', (buf198.sum()/buf198.nelement()).item(), buf198.amax().item(), buf198.amin().item())
        buf199 = buf194; del buf194  # reuse
        print('triton__6', 'in_out_ptr0', 'buf199', (buf199.sum()/buf199.nelement()).item(), buf199.amax().item(), buf199.amin().item())
        print('triton__6', 'in_ptr0', 'relu_26', (relu_26.sum()/relu_26.nelement()).item(), relu_26.amax().item(), relu_26.amin().item())
        print('triton__6', 'in_ptr1', 'convolution_29', (convolution_29.sum()/convolution_29.nelement()).item(), convolution_29.amax().item(), convolution_29.amin().item())
        print('triton__6', 'in_ptr2', 'unsqueeze_554', (unsqueeze_554.sum()/unsqueeze_554.nelement()).item(), unsqueeze_554.amax().item(), unsqueeze_554.amin().item())
        print('triton__6', 'in_ptr3', 'buf197', (buf197.sum()/buf197.nelement()).item(), buf197.amax().item(), buf197.amin().item())
        print('triton__6', 'in_ptr4', 'squeeze_88', (squeeze_88.sum()/squeeze_88.nelement()).item(), squeeze_88.amax().item(), squeeze_88.amin().item())
        print('triton__6', 'in_ptr5', 'buf196', (buf196.sum()/buf196.nelement()).item(), buf196.amax().item(), buf196.amin().item())
        print('triton__6', 'in_ptr6', 'primals_209', (primals_209.sum()/primals_209.nelement()).item(), primals_209.amax().item(), primals_209.amin().item())
        triton__6.run(buf199, relu_26, convolution_29, unsqueeze_554, buf197, squeeze_88, buf196, primals_209, 983040, grid=grid(983040), stream=stream0)
        print('triton__6', 'in_out_ptr0', 'buf199', (buf199.sum()/buf199.nelement()).item(), buf199.amax().item(), buf199.amin().item())
        del convolution_29
        del primals_209
        del relu_26
        del squeeze_88
        del unsqueeze_554
        buf200 = aten.convolution_backward(buf199, relu_25, primals_30, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf199
        del primals_30
        buf201 = buf200[0]
        assert_size_stride(buf201, (8, 1920, 16, 16), (491520, 256, 16, 1))
        buf202 = buf200[1]
        assert_size_stride(buf202, (1920, 1, 3, 3), (9, 9, 3, 1))
        del buf200
        buf203 = buf197; del buf197  # reuse
        buf204 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((1920, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__18', 'in_ptr0', 'relu_25', (relu_25.sum()/relu_25.nelement()).item(), relu_25.amax().item(), relu_25.amin().item())
        print('triton__18', 'in_ptr1', 'buf201', (buf201.sum()/buf201.nelement()).item(), buf201.amax().item(), buf201.amin().item())
        print('triton__18', 'in_ptr2', 'convolution_28', (convolution_28.sum()/convolution_28.nelement()).item(), convolution_28.amax().item(), convolution_28.amin().item())
        print('triton__18', 'in_ptr3', 'unsqueeze_566', (unsqueeze_566.sum()/unsqueeze_566.nelement()).item(), unsqueeze_566.amax().item(), unsqueeze_566.amin().item())
        print('triton__18', 'in_ptr4', 'squeeze_85', (squeeze_85.sum()/squeeze_85.nelement()).item(), squeeze_85.amax().item(), squeeze_85.amin().item())
        triton__18.run(relu_25, buf201, convolution_28, unsqueeze_566, squeeze_85, buf203, buf204, buf205, 1920, 2048, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'buf203', (buf203.sum()/buf203.nelement()).item(), buf203.amax().item(), buf203.amin().item())
        print('triton__18', 'out_ptr1', 'buf204', (buf204.sum()/buf204.nelement()).item(), buf204.amax().item(), buf204.amin().item())
        print('triton__18', 'out_ptr2', 'buf205', (buf205.sum()/buf205.nelement()).item(), buf205.amax().item(), buf205.amin().item())
        buf206 = buf201; del buf201  # reuse
        print('triton__19', 'in_out_ptr0', 'buf206', (buf206.sum()/buf206.nelement()).item(), buf206.amax().item(), buf206.amin().item())
        print('triton__19', 'in_ptr0', 'relu_25', (relu_25.sum()/relu_25.nelement()).item(), relu_25.amax().item(), relu_25.amin().item())
        print('triton__19', 'in_ptr1', 'convolution_28', (convolution_28.sum()/convolution_28.nelement()).item(), convolution_28.amax().item(), convolution_28.amin().item())
        print('triton__19', 'in_ptr2', 'unsqueeze_566', (unsqueeze_566.sum()/unsqueeze_566.nelement()).item(), unsqueeze_566.amax().item(), unsqueeze_566.amin().item())
        print('triton__19', 'in_ptr3', 'buf204', (buf204.sum()/buf204.nelement()).item(), buf204.amax().item(), buf204.amin().item())
        print('triton__19', 'in_ptr4', 'squeeze_85', (squeeze_85.sum()/squeeze_85.nelement()).item(), squeeze_85.amax().item(), squeeze_85.amin().item())
        print('triton__19', 'in_ptr5', 'buf203', (buf203.sum()/buf203.nelement()).item(), buf203.amax().item(), buf203.amin().item())
        print('triton__19', 'in_ptr6', 'primals_204', (primals_204.sum()/primals_204.nelement()).item(), primals_204.amax().item(), primals_204.amin().item())
        triton__19.run(buf206, relu_25, convolution_28, unsqueeze_566, buf204, squeeze_85, buf203, primals_204, 3932160, grid=grid(3932160), stream=stream0)
        print('triton__19', 'in_out_ptr0', 'buf206', (buf206.sum()/buf206.nelement()).item(), buf206.amax().item(), buf206.amin().item())
        del buf204
        del convolution_28
        del primals_204
        del relu_25
        del squeeze_85
        del unsqueeze_566
        buf207 = aten.convolution_backward(buf206, relu_24, primals_29, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf206
        del primals_29
        buf208 = buf207[0]
        assert_size_stride(buf208, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf209 = buf207[1]
        assert_size_stride(buf209, (1920, 640, 1, 1), (640, 1, 1, 1))
        del buf207
        buf210 = buf190; del buf190  # reuse
        buf211 = buf184; del buf184  # reuse
        buf213 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__20', 'in_ptr0', 'relu_24', (relu_24.sum()/relu_24.nelement()).item(), relu_24.amax().item(), relu_24.amin().item())
        print('triton__20', 'in_ptr1', 'buf188', (buf188.sum()/buf188.nelement()).item(), buf188.amax().item(), buf188.amin().item())
        print('triton__20', 'in_ptr2', 'buf208', (buf208.sum()/buf208.nelement()).item(), buf208.amax().item(), buf208.amin().item())
        print('triton__20', 'in_ptr3', 'convolution_27', (convolution_27.sum()/convolution_27.nelement()).item(), convolution_27.amax().item(), convolution_27.amin().item())
        print('triton__20', 'in_ptr4', 'unsqueeze_578', (unsqueeze_578.sum()/unsqueeze_578.nelement()).item(), unsqueeze_578.amax().item(), unsqueeze_578.amin().item())
        print('triton__20', 'in_ptr5', 'squeeze_82', (squeeze_82.sum()/squeeze_82.nelement()).item(), squeeze_82.amax().item(), squeeze_82.amin().item())
        triton__20.run(relu_24, buf188, buf208, convolution_27, unsqueeze_578, squeeze_82, buf210, buf211, buf213, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__20', 'out_ptr0', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        print('triton__20', 'out_ptr1', 'buf211', (buf211.sum()/buf211.nelement()).item(), buf211.amax().item(), buf211.amin().item())
        print('triton__20', 'out_ptr2', 'buf213', (buf213.sum()/buf213.nelement()).item(), buf213.amax().item(), buf213.amin().item())
        buf212 = as_strided(buf6, (8, 640, 16, 16), (163840, 256, 16, 1)); del buf6  # reuse
        print('triton__21', 'in_ptr0', 'relu_24', (relu_24.sum()/relu_24.nelement()).item(), relu_24.amax().item(), relu_24.amin().item())
        print('triton__21', 'in_ptr1', 'buf188', (buf188.sum()/buf188.nelement()).item(), buf188.amax().item(), buf188.amin().item())
        print('triton__21', 'in_ptr2', 'buf208', (buf208.sum()/buf208.nelement()).item(), buf208.amax().item(), buf208.amin().item())
        print('triton__21', 'in_ptr3', 'convolution_27', (convolution_27.sum()/convolution_27.nelement()).item(), convolution_27.amax().item(), convolution_27.amin().item())
        print('triton__21', 'in_ptr4', 'unsqueeze_578', (unsqueeze_578.sum()/unsqueeze_578.nelement()).item(), unsqueeze_578.amax().item(), unsqueeze_578.amin().item())
        print('triton__21', 'in_ptr5', 'buf211', (buf211.sum()/buf211.nelement()).item(), buf211.amax().item(), buf211.amin().item())
        print('triton__21', 'in_ptr6', 'squeeze_82', (squeeze_82.sum()/squeeze_82.nelement()).item(), squeeze_82.amax().item(), squeeze_82.amin().item())
        print('triton__21', 'in_ptr7', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        print('triton__21', 'in_ptr8', 'primals_199', (primals_199.sum()/primals_199.nelement()).item(), primals_199.amax().item(), primals_199.amin().item())
        triton__21.run(relu_24, buf188, buf208, convolution_27, unsqueeze_578, buf211, squeeze_82, buf210, primals_199, buf212, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf212', (buf212.sum()/buf212.nelement()).item(), buf212.amax().item(), buf212.amin().item())
        del convolution_27
        del primals_199
        del squeeze_82
        del unsqueeze_578
        buf214 = aten.convolution_backward(buf212, relu_23, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf212
        del primals_28
        buf215 = buf214[0]
        assert_size_stride(buf215, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf216 = buf214[1]
        assert_size_stride(buf216, (640, 160, 1, 1), (160, 1, 1, 1))
        del buf214
        buf217 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf219 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_23', (relu_23.sum()/relu_23.nelement()).item(), relu_23.amax().item(), relu_23.amin().item())
        print('triton__22', 'in_ptr1', 'buf215', (buf215.sum()/buf215.nelement()).item(), buf215.amax().item(), buf215.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_26', (convolution_26.sum()/convolution_26.nelement()).item(), convolution_26.amax().item(), convolution_26.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_590', (unsqueeze_590.sum()/unsqueeze_590.nelement()).item(), unsqueeze_590.amax().item(), unsqueeze_590.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_79', (squeeze_79.sum()/squeeze_79.nelement()).item(), squeeze_79.amax().item(), squeeze_79.amin().item())
        triton__22.run(relu_23, buf215, convolution_26, unsqueeze_590, squeeze_79, buf217, buf218, buf219, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf217', (buf217.sum()/buf217.nelement()).item(), buf217.amax().item(), buf217.amin().item())
        print('triton__22', 'out_ptr1', 'buf218', (buf218.sum()/buf218.nelement()).item(), buf218.amax().item(), buf218.amin().item())
        print('triton__22', 'out_ptr2', 'buf219', (buf219.sum()/buf219.nelement()).item(), buf219.amax().item(), buf219.amin().item())
        buf220 = buf215; del buf215  # reuse
        print('triton__23', 'in_out_ptr0', 'buf220', (buf220.sum()/buf220.nelement()).item(), buf220.amax().item(), buf220.amin().item())
        print('triton__23', 'in_ptr0', 'relu_23', (relu_23.sum()/relu_23.nelement()).item(), relu_23.amax().item(), relu_23.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_26', (convolution_26.sum()/convolution_26.nelement()).item(), convolution_26.amax().item(), convolution_26.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_590', (unsqueeze_590.sum()/unsqueeze_590.nelement()).item(), unsqueeze_590.amax().item(), unsqueeze_590.amin().item())
        print('triton__23', 'in_ptr3', 'buf218', (buf218.sum()/buf218.nelement()).item(), buf218.amax().item(), buf218.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_79', (squeeze_79.sum()/squeeze_79.nelement()).item(), squeeze_79.amax().item(), squeeze_79.amin().item())
        print('triton__23', 'in_ptr5', 'buf217', (buf217.sum()/buf217.nelement()).item(), buf217.amax().item(), buf217.amin().item())
        print('triton__23', 'in_ptr6', 'primals_194', (primals_194.sum()/primals_194.nelement()).item(), primals_194.amax().item(), primals_194.amin().item())
        triton__23.run(buf220, relu_23, convolution_26, unsqueeze_590, buf218, squeeze_79, buf217, primals_194, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf220', (buf220.sum()/buf220.nelement()).item(), buf220.amax().item(), buf220.amin().item())
        del convolution_26
        del primals_194
        del relu_23
        del squeeze_79
        del unsqueeze_590
        buf221 = aten.convolution_backward(buf220, relu_22, primals_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf220
        del primals_27
        buf222 = buf221[0]
        assert_size_stride(buf222, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf223 = buf221[1]
        assert_size_stride(buf223, (160, 160, 3, 3), (1440, 9, 3, 1))
        del buf221
        buf224 = buf218; del buf218  # reuse
        buf225 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf226 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_22', (relu_22.sum()/relu_22.nelement()).item(), relu_22.amax().item(), relu_22.amin().item())
        print('triton__22', 'in_ptr1', 'buf222', (buf222.sum()/buf222.nelement()).item(), buf222.amax().item(), buf222.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_25', (convolution_25.sum()/convolution_25.nelement()).item(), convolution_25.amax().item(), convolution_25.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_602', (unsqueeze_602.sum()/unsqueeze_602.nelement()).item(), unsqueeze_602.amax().item(), unsqueeze_602.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_76', (squeeze_76.sum()/squeeze_76.nelement()).item(), squeeze_76.amax().item(), squeeze_76.amin().item())
        triton__22.run(relu_22, buf222, convolution_25, unsqueeze_602, squeeze_76, buf224, buf225, buf226, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf224', (buf224.sum()/buf224.nelement()).item(), buf224.amax().item(), buf224.amin().item())
        print('triton__22', 'out_ptr1', 'buf225', (buf225.sum()/buf225.nelement()).item(), buf225.amax().item(), buf225.amin().item())
        print('triton__22', 'out_ptr2', 'buf226', (buf226.sum()/buf226.nelement()).item(), buf226.amax().item(), buf226.amin().item())
        buf227 = buf222; del buf222  # reuse
        print('triton__23', 'in_out_ptr0', 'buf227', (buf227.sum()/buf227.nelement()).item(), buf227.amax().item(), buf227.amin().item())
        print('triton__23', 'in_ptr0', 'relu_22', (relu_22.sum()/relu_22.nelement()).item(), relu_22.amax().item(), relu_22.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_25', (convolution_25.sum()/convolution_25.nelement()).item(), convolution_25.amax().item(), convolution_25.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_602', (unsqueeze_602.sum()/unsqueeze_602.nelement()).item(), unsqueeze_602.amax().item(), unsqueeze_602.amin().item())
        print('triton__23', 'in_ptr3', 'buf225', (buf225.sum()/buf225.nelement()).item(), buf225.amax().item(), buf225.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_76', (squeeze_76.sum()/squeeze_76.nelement()).item(), squeeze_76.amax().item(), squeeze_76.amin().item())
        print('triton__23', 'in_ptr5', 'buf224', (buf224.sum()/buf224.nelement()).item(), buf224.amax().item(), buf224.amin().item())
        print('triton__23', 'in_ptr6', 'primals_189', (primals_189.sum()/primals_189.nelement()).item(), primals_189.amax().item(), primals_189.amin().item())
        triton__23.run(buf227, relu_22, convolution_25, unsqueeze_602, buf225, squeeze_76, buf224, primals_189, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf227', (buf227.sum()/buf227.nelement()).item(), buf227.amax().item(), buf227.amin().item())
        del convolution_25
        del primals_189
        del relu_22
        del squeeze_76
        del unsqueeze_602
        buf228 = aten.convolution_backward(buf227, relu_21, primals_26, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf227
        del primals_26
        buf229 = buf228[0]
        assert_size_stride(buf229, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf230 = buf228[1]
        assert_size_stride(buf230, (160, 640, 1, 1), (640, 1, 1, 1))
        del buf228
        buf231 = buf188; del buf188  # reuse
        print('triton__24', 'in_out_ptr0', 'buf231', (buf231.sum()/buf231.nelement()).item(), buf231.amax().item(), buf231.amin().item())
        print('triton__24', 'in_ptr0', 'relu_21', (relu_21.sum()/relu_21.nelement()).item(), relu_21.amax().item(), relu_21.amin().item())
        print('triton__24', 'in_ptr1', 'relu_24', (relu_24.sum()/relu_24.nelement()).item(), relu_24.amax().item(), relu_24.amin().item())
        print('triton__24', 'in_ptr2', 'buf208', (buf208.sum()/buf208.nelement()).item(), buf208.amax().item(), buf208.amin().item())
        print('triton__24', 'in_ptr3', 'buf229', (buf229.sum()/buf229.nelement()).item(), buf229.amax().item(), buf229.amin().item())
        triton__24.run(buf231, relu_21, relu_24, buf208, buf229, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf231', (buf231.sum()/buf231.nelement()).item(), buf231.amax().item(), buf231.amin().item())
        del buf208
        del relu_21
        del relu_24
        buf232 = buf211; del buf211  # reuse
        buf233 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf234 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf231', (buf231.sum()/buf231.nelement()).item(), buf231.amax().item(), buf231.amin().item())
        print('triton__25', 'in_ptr1', 'convolution_24', (convolution_24.sum()/convolution_24.nelement()).item(), convolution_24.amax().item(), convolution_24.amin().item())
        print('triton__25', 'in_ptr2', 'unsqueeze_614', (unsqueeze_614.sum()/unsqueeze_614.nelement()).item(), unsqueeze_614.amax().item(), unsqueeze_614.amin().item())
        print('triton__25', 'in_ptr3', 'squeeze_73', (squeeze_73.sum()/squeeze_73.nelement()).item(), squeeze_73.amax().item(), squeeze_73.amin().item())
        triton__25.run(buf231, convolution_24, unsqueeze_614, squeeze_73, buf232, buf233, buf234, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf232', (buf232.sum()/buf232.nelement()).item(), buf232.amax().item(), buf232.amin().item())
        print('triton__25', 'out_ptr1', 'buf233', (buf233.sum()/buf233.nelement()).item(), buf233.amax().item(), buf233.amin().item())
        print('triton__25', 'out_ptr2', 'buf234', (buf234.sum()/buf234.nelement()).item(), buf234.amax().item(), buf234.amin().item())
        buf235 = buf229; del buf229  # reuse
        print('triton__26', 'in_ptr0', 'buf231', (buf231.sum()/buf231.nelement()).item(), buf231.amax().item(), buf231.amin().item())
        print('triton__26', 'in_ptr1', 'convolution_24', (convolution_24.sum()/convolution_24.nelement()).item(), convolution_24.amax().item(), convolution_24.amin().item())
        print('triton__26', 'in_ptr2', 'unsqueeze_614', (unsqueeze_614.sum()/unsqueeze_614.nelement()).item(), unsqueeze_614.amax().item(), unsqueeze_614.amin().item())
        print('triton__26', 'in_ptr3', 'buf233', (buf233.sum()/buf233.nelement()).item(), buf233.amax().item(), buf233.amin().item())
        print('triton__26', 'in_ptr4', 'squeeze_73', (squeeze_73.sum()/squeeze_73.nelement()).item(), squeeze_73.amax().item(), squeeze_73.amin().item())
        print('triton__26', 'in_ptr5', 'buf232', (buf232.sum()/buf232.nelement()).item(), buf232.amax().item(), buf232.amin().item())
        print('triton__26', 'in_ptr6', 'primals_184', (primals_184.sum()/primals_184.nelement()).item(), primals_184.amax().item(), primals_184.amin().item())
        triton__26.run(buf231, convolution_24, unsqueeze_614, buf233, squeeze_73, buf232, primals_184, buf235, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__26', 'out_ptr0', 'buf235', (buf235.sum()/buf235.nelement()).item(), buf235.amax().item(), buf235.amin().item())
        del convolution_24
        del primals_184
        del squeeze_73
        del unsqueeze_614
        buf236 = aten.convolution_backward(buf235, relu_20, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf237 = buf236[0]
        assert_size_stride(buf237, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf238 = buf236[1]
        assert_size_stride(buf238, (640, 160, 1, 1), (160, 1, 1, 1))
        del buf236
        buf239 = buf225; del buf225  # reuse
        buf240 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf241 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_20', (relu_20.sum()/relu_20.nelement()).item(), relu_20.amax().item(), relu_20.amin().item())
        print('triton__22', 'in_ptr1', 'buf237', (buf237.sum()/buf237.nelement()).item(), buf237.amax().item(), buf237.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_23', (convolution_23.sum()/convolution_23.nelement()).item(), convolution_23.amax().item(), convolution_23.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_626', (unsqueeze_626.sum()/unsqueeze_626.nelement()).item(), unsqueeze_626.amax().item(), unsqueeze_626.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_70', (squeeze_70.sum()/squeeze_70.nelement()).item(), squeeze_70.amax().item(), squeeze_70.amin().item())
        triton__22.run(relu_20, buf237, convolution_23, unsqueeze_626, squeeze_70, buf239, buf240, buf241, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf239', (buf239.sum()/buf239.nelement()).item(), buf239.amax().item(), buf239.amin().item())
        print('triton__22', 'out_ptr1', 'buf240', (buf240.sum()/buf240.nelement()).item(), buf240.amax().item(), buf240.amin().item())
        print('triton__22', 'out_ptr2', 'buf241', (buf241.sum()/buf241.nelement()).item(), buf241.amax().item(), buf241.amin().item())
        buf242 = buf237; del buf237  # reuse
        print('triton__23', 'in_out_ptr0', 'buf242', (buf242.sum()/buf242.nelement()).item(), buf242.amax().item(), buf242.amin().item())
        print('triton__23', 'in_ptr0', 'relu_20', (relu_20.sum()/relu_20.nelement()).item(), relu_20.amax().item(), relu_20.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_23', (convolution_23.sum()/convolution_23.nelement()).item(), convolution_23.amax().item(), convolution_23.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_626', (unsqueeze_626.sum()/unsqueeze_626.nelement()).item(), unsqueeze_626.amax().item(), unsqueeze_626.amin().item())
        print('triton__23', 'in_ptr3', 'buf240', (buf240.sum()/buf240.nelement()).item(), buf240.amax().item(), buf240.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_70', (squeeze_70.sum()/squeeze_70.nelement()).item(), squeeze_70.amax().item(), squeeze_70.amin().item())
        print('triton__23', 'in_ptr5', 'buf239', (buf239.sum()/buf239.nelement()).item(), buf239.amax().item(), buf239.amin().item())
        print('triton__23', 'in_ptr6', 'primals_179', (primals_179.sum()/primals_179.nelement()).item(), primals_179.amax().item(), primals_179.amin().item())
        triton__23.run(buf242, relu_20, convolution_23, unsqueeze_626, buf240, squeeze_70, buf239, primals_179, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf242', (buf242.sum()/buf242.nelement()).item(), buf242.amax().item(), buf242.amin().item())
        del convolution_23
        del primals_179
        del relu_20
        del squeeze_70
        del unsqueeze_626
        buf243 = aten.convolution_backward(buf242, relu_19, primals_24, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf242
        del primals_24
        buf244 = buf243[0]
        assert_size_stride(buf244, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf245 = buf243[1]
        assert_size_stride(buf245, (160, 160, 3, 3), (1440, 9, 3, 1))
        del buf243
        buf246 = buf240; del buf240  # reuse
        buf247 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf248 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_19', (relu_19.sum()/relu_19.nelement()).item(), relu_19.amax().item(), relu_19.amin().item())
        print('triton__22', 'in_ptr1', 'buf244', (buf244.sum()/buf244.nelement()).item(), buf244.amax().item(), buf244.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_22', (convolution_22.sum()/convolution_22.nelement()).item(), convolution_22.amax().item(), convolution_22.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_638', (unsqueeze_638.sum()/unsqueeze_638.nelement()).item(), unsqueeze_638.amax().item(), unsqueeze_638.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_67', (squeeze_67.sum()/squeeze_67.nelement()).item(), squeeze_67.amax().item(), squeeze_67.amin().item())
        triton__22.run(relu_19, buf244, convolution_22, unsqueeze_638, squeeze_67, buf246, buf247, buf248, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf246', (buf246.sum()/buf246.nelement()).item(), buf246.amax().item(), buf246.amin().item())
        print('triton__22', 'out_ptr1', 'buf247', (buf247.sum()/buf247.nelement()).item(), buf247.amax().item(), buf247.amin().item())
        print('triton__22', 'out_ptr2', 'buf248', (buf248.sum()/buf248.nelement()).item(), buf248.amax().item(), buf248.amin().item())
        buf249 = buf244; del buf244  # reuse
        print('triton__23', 'in_out_ptr0', 'buf249', (buf249.sum()/buf249.nelement()).item(), buf249.amax().item(), buf249.amin().item())
        print('triton__23', 'in_ptr0', 'relu_19', (relu_19.sum()/relu_19.nelement()).item(), relu_19.amax().item(), relu_19.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_22', (convolution_22.sum()/convolution_22.nelement()).item(), convolution_22.amax().item(), convolution_22.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_638', (unsqueeze_638.sum()/unsqueeze_638.nelement()).item(), unsqueeze_638.amax().item(), unsqueeze_638.amin().item())
        print('triton__23', 'in_ptr3', 'buf247', (buf247.sum()/buf247.nelement()).item(), buf247.amax().item(), buf247.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_67', (squeeze_67.sum()/squeeze_67.nelement()).item(), squeeze_67.amax().item(), squeeze_67.amin().item())
        print('triton__23', 'in_ptr5', 'buf246', (buf246.sum()/buf246.nelement()).item(), buf246.amax().item(), buf246.amin().item())
        print('triton__23', 'in_ptr6', 'primals_174', (primals_174.sum()/primals_174.nelement()).item(), primals_174.amax().item(), primals_174.amin().item())
        triton__23.run(buf249, relu_19, convolution_22, unsqueeze_638, buf247, squeeze_67, buf246, primals_174, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf249', (buf249.sum()/buf249.nelement()).item(), buf249.amax().item(), buf249.amin().item())
        del convolution_22
        del primals_174
        del relu_19
        del squeeze_67
        del unsqueeze_638
        buf250 = aten.convolution_backward(buf249, relu_18, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf249
        del primals_23
        buf251 = buf250[0]
        assert_size_stride(buf251, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf252 = buf250[1]
        assert_size_stride(buf252, (160, 640, 1, 1), (640, 1, 1, 1))
        del buf250
        buf253 = buf233; del buf233  # reuse
        buf254 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf256 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__20', 'in_ptr0', 'relu_18', (relu_18.sum()/relu_18.nelement()).item(), relu_18.amax().item(), relu_18.amin().item())
        print('triton__20', 'in_ptr1', 'buf231', (buf231.sum()/buf231.nelement()).item(), buf231.amax().item(), buf231.amin().item())
        print('triton__20', 'in_ptr2', 'buf251', (buf251.sum()/buf251.nelement()).item(), buf251.amax().item(), buf251.amin().item())
        print('triton__20', 'in_ptr3', 'convolution_21', (convolution_21.sum()/convolution_21.nelement()).item(), convolution_21.amax().item(), convolution_21.amin().item())
        print('triton__20', 'in_ptr4', 'unsqueeze_650', (unsqueeze_650.sum()/unsqueeze_650.nelement()).item(), unsqueeze_650.amax().item(), unsqueeze_650.amin().item())
        print('triton__20', 'in_ptr5', 'squeeze_64', (squeeze_64.sum()/squeeze_64.nelement()).item(), squeeze_64.amax().item(), squeeze_64.amin().item())
        triton__20.run(relu_18, buf231, buf251, convolution_21, unsqueeze_650, squeeze_64, buf253, buf254, buf256, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__20', 'out_ptr0', 'buf253', (buf253.sum()/buf253.nelement()).item(), buf253.amax().item(), buf253.amin().item())
        print('triton__20', 'out_ptr1', 'buf254', (buf254.sum()/buf254.nelement()).item(), buf254.amax().item(), buf254.amin().item())
        print('triton__20', 'out_ptr2', 'buf256', (buf256.sum()/buf256.nelement()).item(), buf256.amax().item(), buf256.amin().item())
        buf255 = buf235; del buf235  # reuse
        print('triton__21', 'in_ptr0', 'relu_18', (relu_18.sum()/relu_18.nelement()).item(), relu_18.amax().item(), relu_18.amin().item())
        print('triton__21', 'in_ptr1', 'buf231', (buf231.sum()/buf231.nelement()).item(), buf231.amax().item(), buf231.amin().item())
        print('triton__21', 'in_ptr2', 'buf251', (buf251.sum()/buf251.nelement()).item(), buf251.amax().item(), buf251.amin().item())
        print('triton__21', 'in_ptr3', 'convolution_21', (convolution_21.sum()/convolution_21.nelement()).item(), convolution_21.amax().item(), convolution_21.amin().item())
        print('triton__21', 'in_ptr4', 'unsqueeze_650', (unsqueeze_650.sum()/unsqueeze_650.nelement()).item(), unsqueeze_650.amax().item(), unsqueeze_650.amin().item())
        print('triton__21', 'in_ptr5', 'buf254', (buf254.sum()/buf254.nelement()).item(), buf254.amax().item(), buf254.amin().item())
        print('triton__21', 'in_ptr6', 'squeeze_64', (squeeze_64.sum()/squeeze_64.nelement()).item(), squeeze_64.amax().item(), squeeze_64.amin().item())
        print('triton__21', 'in_ptr7', 'buf253', (buf253.sum()/buf253.nelement()).item(), buf253.amax().item(), buf253.amin().item())
        print('triton__21', 'in_ptr8', 'primals_169', (primals_169.sum()/primals_169.nelement()).item(), primals_169.amax().item(), primals_169.amin().item())
        triton__21.run(relu_18, buf231, buf251, convolution_21, unsqueeze_650, buf254, squeeze_64, buf253, primals_169, buf255, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf255', (buf255.sum()/buf255.nelement()).item(), buf255.amax().item(), buf255.amin().item())
        del convolution_21
        del primals_169
        del squeeze_64
        del unsqueeze_650
        buf257 = aten.convolution_backward(buf255, relu_17, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf255
        del primals_22
        buf258 = buf257[0]
        assert_size_stride(buf258, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf259 = buf257[1]
        assert_size_stride(buf259, (640, 160, 1, 1), (160, 1, 1, 1))
        del buf257
        buf260 = buf247; del buf247  # reuse
        buf261 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf262 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_17', (relu_17.sum()/relu_17.nelement()).item(), relu_17.amax().item(), relu_17.amin().item())
        print('triton__22', 'in_ptr1', 'buf258', (buf258.sum()/buf258.nelement()).item(), buf258.amax().item(), buf258.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_20', (convolution_20.sum()/convolution_20.nelement()).item(), convolution_20.amax().item(), convolution_20.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_662', (unsqueeze_662.sum()/unsqueeze_662.nelement()).item(), unsqueeze_662.amax().item(), unsqueeze_662.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_61', (squeeze_61.sum()/squeeze_61.nelement()).item(), squeeze_61.amax().item(), squeeze_61.amin().item())
        triton__22.run(relu_17, buf258, convolution_20, unsqueeze_662, squeeze_61, buf260, buf261, buf262, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf260', (buf260.sum()/buf260.nelement()).item(), buf260.amax().item(), buf260.amin().item())
        print('triton__22', 'out_ptr1', 'buf261', (buf261.sum()/buf261.nelement()).item(), buf261.amax().item(), buf261.amin().item())
        print('triton__22', 'out_ptr2', 'buf262', (buf262.sum()/buf262.nelement()).item(), buf262.amax().item(), buf262.amin().item())
        buf263 = buf258; del buf258  # reuse
        print('triton__23', 'in_out_ptr0', 'buf263', (buf263.sum()/buf263.nelement()).item(), buf263.amax().item(), buf263.amin().item())
        print('triton__23', 'in_ptr0', 'relu_17', (relu_17.sum()/relu_17.nelement()).item(), relu_17.amax().item(), relu_17.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_20', (convolution_20.sum()/convolution_20.nelement()).item(), convolution_20.amax().item(), convolution_20.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_662', (unsqueeze_662.sum()/unsqueeze_662.nelement()).item(), unsqueeze_662.amax().item(), unsqueeze_662.amin().item())
        print('triton__23', 'in_ptr3', 'buf261', (buf261.sum()/buf261.nelement()).item(), buf261.amax().item(), buf261.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_61', (squeeze_61.sum()/squeeze_61.nelement()).item(), squeeze_61.amax().item(), squeeze_61.amin().item())
        print('triton__23', 'in_ptr5', 'buf260', (buf260.sum()/buf260.nelement()).item(), buf260.amax().item(), buf260.amin().item())
        print('triton__23', 'in_ptr6', 'primals_164', (primals_164.sum()/primals_164.nelement()).item(), primals_164.amax().item(), primals_164.amin().item())
        triton__23.run(buf263, relu_17, convolution_20, unsqueeze_662, buf261, squeeze_61, buf260, primals_164, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf263', (buf263.sum()/buf263.nelement()).item(), buf263.amax().item(), buf263.amin().item())
        del convolution_20
        del primals_164
        del relu_17
        del squeeze_61
        del unsqueeze_662
        buf264 = aten.convolution_backward(buf263, relu_16, primals_21, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf263
        del primals_21
        buf265 = buf264[0]
        assert_size_stride(buf265, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf266 = buf264[1]
        assert_size_stride(buf266, (160, 160, 3, 3), (1440, 9, 3, 1))
        del buf264
        buf267 = buf261; del buf261  # reuse
        buf268 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf269 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_16', (relu_16.sum()/relu_16.nelement()).item(), relu_16.amax().item(), relu_16.amin().item())
        print('triton__22', 'in_ptr1', 'buf265', (buf265.sum()/buf265.nelement()).item(), buf265.amax().item(), buf265.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_19', (convolution_19.sum()/convolution_19.nelement()).item(), convolution_19.amax().item(), convolution_19.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_674', (unsqueeze_674.sum()/unsqueeze_674.nelement()).item(), unsqueeze_674.amax().item(), unsqueeze_674.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_58', (squeeze_58.sum()/squeeze_58.nelement()).item(), squeeze_58.amax().item(), squeeze_58.amin().item())
        triton__22.run(relu_16, buf265, convolution_19, unsqueeze_674, squeeze_58, buf267, buf268, buf269, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf267', (buf267.sum()/buf267.nelement()).item(), buf267.amax().item(), buf267.amin().item())
        print('triton__22', 'out_ptr1', 'buf268', (buf268.sum()/buf268.nelement()).item(), buf268.amax().item(), buf268.amin().item())
        print('triton__22', 'out_ptr2', 'buf269', (buf269.sum()/buf269.nelement()).item(), buf269.amax().item(), buf269.amin().item())
        buf270 = buf265; del buf265  # reuse
        print('triton__23', 'in_out_ptr0', 'buf270', (buf270.sum()/buf270.nelement()).item(), buf270.amax().item(), buf270.amin().item())
        print('triton__23', 'in_ptr0', 'relu_16', (relu_16.sum()/relu_16.nelement()).item(), relu_16.amax().item(), relu_16.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_19', (convolution_19.sum()/convolution_19.nelement()).item(), convolution_19.amax().item(), convolution_19.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_674', (unsqueeze_674.sum()/unsqueeze_674.nelement()).item(), unsqueeze_674.amax().item(), unsqueeze_674.amin().item())
        print('triton__23', 'in_ptr3', 'buf268', (buf268.sum()/buf268.nelement()).item(), buf268.amax().item(), buf268.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_58', (squeeze_58.sum()/squeeze_58.nelement()).item(), squeeze_58.amax().item(), squeeze_58.amin().item())
        print('triton__23', 'in_ptr5', 'buf267', (buf267.sum()/buf267.nelement()).item(), buf267.amax().item(), buf267.amin().item())
        print('triton__23', 'in_ptr6', 'primals_159', (primals_159.sum()/primals_159.nelement()).item(), primals_159.amax().item(), primals_159.amin().item())
        triton__23.run(buf270, relu_16, convolution_19, unsqueeze_674, buf268, squeeze_58, buf267, primals_159, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf270', (buf270.sum()/buf270.nelement()).item(), buf270.amax().item(), buf270.amin().item())
        del convolution_19
        del primals_159
        del relu_16
        del squeeze_58
        del unsqueeze_674
        buf271 = aten.convolution_backward(buf270, relu_15, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf270
        del primals_20
        buf272 = buf271[0]
        assert_size_stride(buf272, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf273 = buf271[1]
        assert_size_stride(buf273, (160, 640, 1, 1), (640, 1, 1, 1))
        del buf271
        buf274 = buf231; del buf231  # reuse
        print('triton__24', 'in_out_ptr0', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__24', 'in_ptr0', 'relu_15', (relu_15.sum()/relu_15.nelement()).item(), relu_15.amax().item(), relu_15.amin().item())
        print('triton__24', 'in_ptr1', 'relu_18', (relu_18.sum()/relu_18.nelement()).item(), relu_18.amax().item(), relu_18.amin().item())
        print('triton__24', 'in_ptr2', 'buf251', (buf251.sum()/buf251.nelement()).item(), buf251.amax().item(), buf251.amin().item())
        print('triton__24', 'in_ptr3', 'buf272', (buf272.sum()/buf272.nelement()).item(), buf272.amax().item(), buf272.amin().item())
        triton__24.run(buf274, relu_15, relu_18, buf251, buf272, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        del buf251
        del relu_15
        del relu_18
        buf275 = buf254; del buf254  # reuse
        buf276 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf277 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__25', 'in_ptr0', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__25', 'in_ptr1', 'convolution_18', (convolution_18.sum()/convolution_18.nelement()).item(), convolution_18.amax().item(), convolution_18.amin().item())
        print('triton__25', 'in_ptr2', 'unsqueeze_686', (unsqueeze_686.sum()/unsqueeze_686.nelement()).item(), unsqueeze_686.amax().item(), unsqueeze_686.amin().item())
        print('triton__25', 'in_ptr3', 'squeeze_55', (squeeze_55.sum()/squeeze_55.nelement()).item(), squeeze_55.amax().item(), squeeze_55.amin().item())
        triton__25.run(buf274, convolution_18, unsqueeze_686, squeeze_55, buf275, buf276, buf277, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__25', 'out_ptr0', 'buf275', (buf275.sum()/buf275.nelement()).item(), buf275.amax().item(), buf275.amin().item())
        print('triton__25', 'out_ptr1', 'buf276', (buf276.sum()/buf276.nelement()).item(), buf276.amax().item(), buf276.amin().item())
        print('triton__25', 'out_ptr2', 'buf277', (buf277.sum()/buf277.nelement()).item(), buf277.amax().item(), buf277.amin().item())
        buf278 = buf272; del buf272  # reuse
        print('triton__26', 'in_ptr0', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__26', 'in_ptr1', 'convolution_18', (convolution_18.sum()/convolution_18.nelement()).item(), convolution_18.amax().item(), convolution_18.amin().item())
        print('triton__26', 'in_ptr2', 'unsqueeze_686', (unsqueeze_686.sum()/unsqueeze_686.nelement()).item(), unsqueeze_686.amax().item(), unsqueeze_686.amin().item())
        print('triton__26', 'in_ptr3', 'buf276', (buf276.sum()/buf276.nelement()).item(), buf276.amax().item(), buf276.amin().item())
        print('triton__26', 'in_ptr4', 'squeeze_55', (squeeze_55.sum()/squeeze_55.nelement()).item(), squeeze_55.amax().item(), squeeze_55.amin().item())
        print('triton__26', 'in_ptr5', 'buf275', (buf275.sum()/buf275.nelement()).item(), buf275.amax().item(), buf275.amin().item())
        print('triton__26', 'in_ptr6', 'primals_154', (primals_154.sum()/primals_154.nelement()).item(), primals_154.amax().item(), primals_154.amin().item())
        triton__26.run(buf274, convolution_18, unsqueeze_686, buf276, squeeze_55, buf275, primals_154, buf278, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__26', 'out_ptr0', 'buf278', (buf278.sum()/buf278.nelement()).item(), buf278.amax().item(), buf278.amin().item())
        del convolution_18
        del primals_154
        del squeeze_55
        del unsqueeze_686
        buf279 = aten.convolution_backward(buf278, relu_14, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_19
        buf280 = buf279[0]
        assert_size_stride(buf280, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf281 = buf279[1]
        assert_size_stride(buf281, (640, 160, 1, 1), (160, 1, 1, 1))
        del buf279
        buf282 = buf268; del buf268  # reuse
        buf283 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf284 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_14', (relu_14.sum()/relu_14.nelement()).item(), relu_14.amax().item(), relu_14.amin().item())
        print('triton__22', 'in_ptr1', 'buf280', (buf280.sum()/buf280.nelement()).item(), buf280.amax().item(), buf280.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_17', (convolution_17.sum()/convolution_17.nelement()).item(), convolution_17.amax().item(), convolution_17.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_698', (unsqueeze_698.sum()/unsqueeze_698.nelement()).item(), unsqueeze_698.amax().item(), unsqueeze_698.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_52', (squeeze_52.sum()/squeeze_52.nelement()).item(), squeeze_52.amax().item(), squeeze_52.amin().item())
        triton__22.run(relu_14, buf280, convolution_17, unsqueeze_698, squeeze_52, buf282, buf283, buf284, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf282', (buf282.sum()/buf282.nelement()).item(), buf282.amax().item(), buf282.amin().item())
        print('triton__22', 'out_ptr1', 'buf283', (buf283.sum()/buf283.nelement()).item(), buf283.amax().item(), buf283.amin().item())
        print('triton__22', 'out_ptr2', 'buf284', (buf284.sum()/buf284.nelement()).item(), buf284.amax().item(), buf284.amin().item())
        buf285 = buf280; del buf280  # reuse
        print('triton__23', 'in_out_ptr0', 'buf285', (buf285.sum()/buf285.nelement()).item(), buf285.amax().item(), buf285.amin().item())
        print('triton__23', 'in_ptr0', 'relu_14', (relu_14.sum()/relu_14.nelement()).item(), relu_14.amax().item(), relu_14.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_17', (convolution_17.sum()/convolution_17.nelement()).item(), convolution_17.amax().item(), convolution_17.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_698', (unsqueeze_698.sum()/unsqueeze_698.nelement()).item(), unsqueeze_698.amax().item(), unsqueeze_698.amin().item())
        print('triton__23', 'in_ptr3', 'buf283', (buf283.sum()/buf283.nelement()).item(), buf283.amax().item(), buf283.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_52', (squeeze_52.sum()/squeeze_52.nelement()).item(), squeeze_52.amax().item(), squeeze_52.amin().item())
        print('triton__23', 'in_ptr5', 'buf282', (buf282.sum()/buf282.nelement()).item(), buf282.amax().item(), buf282.amin().item())
        print('triton__23', 'in_ptr6', 'primals_149', (primals_149.sum()/primals_149.nelement()).item(), primals_149.amax().item(), primals_149.amin().item())
        triton__23.run(buf285, relu_14, convolution_17, unsqueeze_698, buf283, squeeze_52, buf282, primals_149, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf285', (buf285.sum()/buf285.nelement()).item(), buf285.amax().item(), buf285.amin().item())
        del convolution_17
        del primals_149
        del relu_14
        del squeeze_52
        del unsqueeze_698
        buf286 = aten.convolution_backward(buf285, relu_13, primals_18, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf285
        del primals_18
        buf287 = buf286[0]
        assert_size_stride(buf287, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf288 = buf286[1]
        assert_size_stride(buf288, (160, 160, 3, 3), (1440, 9, 3, 1))
        del buf286
        buf289 = buf283; del buf283  # reuse
        buf290 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf291 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_13', (relu_13.sum()/relu_13.nelement()).item(), relu_13.amax().item(), relu_13.amin().item())
        print('triton__22', 'in_ptr1', 'buf287', (buf287.sum()/buf287.nelement()).item(), buf287.amax().item(), buf287.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_16', (convolution_16.sum()/convolution_16.nelement()).item(), convolution_16.amax().item(), convolution_16.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_710', (unsqueeze_710.sum()/unsqueeze_710.nelement()).item(), unsqueeze_710.amax().item(), unsqueeze_710.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_49', (squeeze_49.sum()/squeeze_49.nelement()).item(), squeeze_49.amax().item(), squeeze_49.amin().item())
        triton__22.run(relu_13, buf287, convolution_16, unsqueeze_710, squeeze_49, buf289, buf290, buf291, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf289', (buf289.sum()/buf289.nelement()).item(), buf289.amax().item(), buf289.amin().item())
        print('triton__22', 'out_ptr1', 'buf290', (buf290.sum()/buf290.nelement()).item(), buf290.amax().item(), buf290.amin().item())
        print('triton__22', 'out_ptr2', 'buf291', (buf291.sum()/buf291.nelement()).item(), buf291.amax().item(), buf291.amin().item())
        buf292 = buf287; del buf287  # reuse
        print('triton__23', 'in_out_ptr0', 'buf292', (buf292.sum()/buf292.nelement()).item(), buf292.amax().item(), buf292.amin().item())
        print('triton__23', 'in_ptr0', 'relu_13', (relu_13.sum()/relu_13.nelement()).item(), relu_13.amax().item(), relu_13.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_16', (convolution_16.sum()/convolution_16.nelement()).item(), convolution_16.amax().item(), convolution_16.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_710', (unsqueeze_710.sum()/unsqueeze_710.nelement()).item(), unsqueeze_710.amax().item(), unsqueeze_710.amin().item())
        print('triton__23', 'in_ptr3', 'buf290', (buf290.sum()/buf290.nelement()).item(), buf290.amax().item(), buf290.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_49', (squeeze_49.sum()/squeeze_49.nelement()).item(), squeeze_49.amax().item(), squeeze_49.amin().item())
        print('triton__23', 'in_ptr5', 'buf289', (buf289.sum()/buf289.nelement()).item(), buf289.amax().item(), buf289.amin().item())
        print('triton__23', 'in_ptr6', 'primals_144', (primals_144.sum()/primals_144.nelement()).item(), primals_144.amax().item(), primals_144.amin().item())
        triton__23.run(buf292, relu_13, convolution_16, unsqueeze_710, buf290, squeeze_49, buf289, primals_144, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf292', (buf292.sum()/buf292.nelement()).item(), buf292.amax().item(), buf292.amin().item())
        del convolution_16
        del primals_144
        del relu_13
        del squeeze_49
        del unsqueeze_710
        buf293 = aten.convolution_backward(buf292, relu_12, primals_17, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf292
        del primals_17
        buf294 = buf293[0]
        assert_size_stride(buf294, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf295 = buf293[1]
        assert_size_stride(buf295, (160, 640, 1, 1), (640, 1, 1, 1))
        del buf293
        buf296 = buf276; del buf276  # reuse
        buf297 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf299 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__20', 'in_ptr0', 'relu_12', (relu_12.sum()/relu_12.nelement()).item(), relu_12.amax().item(), relu_12.amin().item())
        print('triton__20', 'in_ptr1', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__20', 'in_ptr2', 'buf294', (buf294.sum()/buf294.nelement()).item(), buf294.amax().item(), buf294.amin().item())
        print('triton__20', 'in_ptr3', 'convolution_15', (convolution_15.sum()/convolution_15.nelement()).item(), convolution_15.amax().item(), convolution_15.amin().item())
        print('triton__20', 'in_ptr4', 'unsqueeze_722', (unsqueeze_722.sum()/unsqueeze_722.nelement()).item(), unsqueeze_722.amax().item(), unsqueeze_722.amin().item())
        print('triton__20', 'in_ptr5', 'squeeze_46', (squeeze_46.sum()/squeeze_46.nelement()).item(), squeeze_46.amax().item(), squeeze_46.amin().item())
        triton__20.run(relu_12, buf274, buf294, convolution_15, unsqueeze_722, squeeze_46, buf296, buf297, buf299, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__20', 'out_ptr0', 'buf296', (buf296.sum()/buf296.nelement()).item(), buf296.amax().item(), buf296.amin().item())
        print('triton__20', 'out_ptr1', 'buf297', (buf297.sum()/buf297.nelement()).item(), buf297.amax().item(), buf297.amin().item())
        print('triton__20', 'out_ptr2', 'buf299', (buf299.sum()/buf299.nelement()).item(), buf299.amax().item(), buf299.amin().item())
        buf298 = buf278; del buf278  # reuse
        print('triton__21', 'in_ptr0', 'relu_12', (relu_12.sum()/relu_12.nelement()).item(), relu_12.amax().item(), relu_12.amin().item())
        print('triton__21', 'in_ptr1', 'buf274', (buf274.sum()/buf274.nelement()).item(), buf274.amax().item(), buf274.amin().item())
        print('triton__21', 'in_ptr2', 'buf294', (buf294.sum()/buf294.nelement()).item(), buf294.amax().item(), buf294.amin().item())
        print('triton__21', 'in_ptr3', 'convolution_15', (convolution_15.sum()/convolution_15.nelement()).item(), convolution_15.amax().item(), convolution_15.amin().item())
        print('triton__21', 'in_ptr4', 'unsqueeze_722', (unsqueeze_722.sum()/unsqueeze_722.nelement()).item(), unsqueeze_722.amax().item(), unsqueeze_722.amin().item())
        print('triton__21', 'in_ptr5', 'buf297', (buf297.sum()/buf297.nelement()).item(), buf297.amax().item(), buf297.amin().item())
        print('triton__21', 'in_ptr6', 'squeeze_46', (squeeze_46.sum()/squeeze_46.nelement()).item(), squeeze_46.amax().item(), squeeze_46.amin().item())
        print('triton__21', 'in_ptr7', 'buf296', (buf296.sum()/buf296.nelement()).item(), buf296.amax().item(), buf296.amin().item())
        print('triton__21', 'in_ptr8', 'primals_139', (primals_139.sum()/primals_139.nelement()).item(), primals_139.amax().item(), primals_139.amin().item())
        triton__21.run(relu_12, buf274, buf294, convolution_15, unsqueeze_722, buf297, squeeze_46, buf296, primals_139, buf298, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__21', 'out_ptr0', 'buf298', (buf298.sum()/buf298.nelement()).item(), buf298.amax().item(), buf298.amin().item())
        del convolution_15
        del primals_139
        del squeeze_46
        del unsqueeze_722
        buf300 = aten.convolution_backward(buf298, relu_11, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf298
        del primals_16
        buf301 = buf300[0]
        assert_size_stride(buf301, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf302 = buf300[1]
        assert_size_stride(buf302, (640, 160, 1, 1), (160, 1, 1, 1))
        del buf300
        buf303 = buf290; del buf290  # reuse
        buf304 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf305 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_11', (relu_11.sum()/relu_11.nelement()).item(), relu_11.amax().item(), relu_11.amin().item())
        print('triton__22', 'in_ptr1', 'buf301', (buf301.sum()/buf301.nelement()).item(), buf301.amax().item(), buf301.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_14', (convolution_14.sum()/convolution_14.nelement()).item(), convolution_14.amax().item(), convolution_14.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_734', (unsqueeze_734.sum()/unsqueeze_734.nelement()).item(), unsqueeze_734.amax().item(), unsqueeze_734.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_43', (squeeze_43.sum()/squeeze_43.nelement()).item(), squeeze_43.amax().item(), squeeze_43.amin().item())
        triton__22.run(relu_11, buf301, convolution_14, unsqueeze_734, squeeze_43, buf303, buf304, buf305, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf303', (buf303.sum()/buf303.nelement()).item(), buf303.amax().item(), buf303.amin().item())
        print('triton__22', 'out_ptr1', 'buf304', (buf304.sum()/buf304.nelement()).item(), buf304.amax().item(), buf304.amin().item())
        print('triton__22', 'out_ptr2', 'buf305', (buf305.sum()/buf305.nelement()).item(), buf305.amax().item(), buf305.amin().item())
        buf306 = buf301; del buf301  # reuse
        print('triton__23', 'in_out_ptr0', 'buf306', (buf306.sum()/buf306.nelement()).item(), buf306.amax().item(), buf306.amin().item())
        print('triton__23', 'in_ptr0', 'relu_11', (relu_11.sum()/relu_11.nelement()).item(), relu_11.amax().item(), relu_11.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_14', (convolution_14.sum()/convolution_14.nelement()).item(), convolution_14.amax().item(), convolution_14.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_734', (unsqueeze_734.sum()/unsqueeze_734.nelement()).item(), unsqueeze_734.amax().item(), unsqueeze_734.amin().item())
        print('triton__23', 'in_ptr3', 'buf304', (buf304.sum()/buf304.nelement()).item(), buf304.amax().item(), buf304.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_43', (squeeze_43.sum()/squeeze_43.nelement()).item(), squeeze_43.amax().item(), squeeze_43.amin().item())
        print('triton__23', 'in_ptr5', 'buf303', (buf303.sum()/buf303.nelement()).item(), buf303.amax().item(), buf303.amin().item())
        print('triton__23', 'in_ptr6', 'primals_134', (primals_134.sum()/primals_134.nelement()).item(), primals_134.amax().item(), primals_134.amin().item())
        triton__23.run(buf306, relu_11, convolution_14, unsqueeze_734, buf304, squeeze_43, buf303, primals_134, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf306', (buf306.sum()/buf306.nelement()).item(), buf306.amax().item(), buf306.amin().item())
        del convolution_14
        del primals_134
        del relu_11
        del squeeze_43
        del unsqueeze_734
        buf307 = aten.convolution_backward(buf306, relu_10, primals_15, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf306
        del primals_15
        buf308 = buf307[0]
        assert_size_stride(buf308, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf309 = buf307[1]
        assert_size_stride(buf309, (160, 160, 3, 3), (1440, 9, 3, 1))
        del buf307
        buf310 = buf304; del buf304  # reuse
        buf311 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf312 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_10', (relu_10.sum()/relu_10.nelement()).item(), relu_10.amax().item(), relu_10.amin().item())
        print('triton__22', 'in_ptr1', 'buf308', (buf308.sum()/buf308.nelement()).item(), buf308.amax().item(), buf308.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_13', (convolution_13.sum()/convolution_13.nelement()).item(), convolution_13.amax().item(), convolution_13.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_746', (unsqueeze_746.sum()/unsqueeze_746.nelement()).item(), unsqueeze_746.amax().item(), unsqueeze_746.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_40', (squeeze_40.sum()/squeeze_40.nelement()).item(), squeeze_40.amax().item(), squeeze_40.amin().item())
        triton__22.run(relu_10, buf308, convolution_13, unsqueeze_746, squeeze_40, buf310, buf311, buf312, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf310', (buf310.sum()/buf310.nelement()).item(), buf310.amax().item(), buf310.amin().item())
        print('triton__22', 'out_ptr1', 'buf311', (buf311.sum()/buf311.nelement()).item(), buf311.amax().item(), buf311.amin().item())
        print('triton__22', 'out_ptr2', 'buf312', (buf312.sum()/buf312.nelement()).item(), buf312.amax().item(), buf312.amin().item())
        buf313 = buf308; del buf308  # reuse
        print('triton__23', 'in_out_ptr0', 'buf313', (buf313.sum()/buf313.nelement()).item(), buf313.amax().item(), buf313.amin().item())
        print('triton__23', 'in_ptr0', 'relu_10', (relu_10.sum()/relu_10.nelement()).item(), relu_10.amax().item(), relu_10.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_13', (convolution_13.sum()/convolution_13.nelement()).item(), convolution_13.amax().item(), convolution_13.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_746', (unsqueeze_746.sum()/unsqueeze_746.nelement()).item(), unsqueeze_746.amax().item(), unsqueeze_746.amin().item())
        print('triton__23', 'in_ptr3', 'buf311', (buf311.sum()/buf311.nelement()).item(), buf311.amax().item(), buf311.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_40', (squeeze_40.sum()/squeeze_40.nelement()).item(), squeeze_40.amax().item(), squeeze_40.amin().item())
        print('triton__23', 'in_ptr5', 'buf310', (buf310.sum()/buf310.nelement()).item(), buf310.amax().item(), buf310.amin().item())
        print('triton__23', 'in_ptr6', 'primals_129', (primals_129.sum()/primals_129.nelement()).item(), primals_129.amax().item(), primals_129.amin().item())
        triton__23.run(buf313, relu_10, convolution_13, unsqueeze_746, buf311, squeeze_40, buf310, primals_129, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf313', (buf313.sum()/buf313.nelement()).item(), buf313.amax().item(), buf313.amin().item())
        del convolution_13
        del primals_129
        del relu_10
        del squeeze_40
        del unsqueeze_746
        buf314 = aten.convolution_backward(buf313, relu_9, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf313
        del primals_14
        buf315 = buf314[0]
        assert_size_stride(buf315, (8, 640, 16, 16), (163840, 256, 16, 1))
        buf316 = buf314[1]
        assert_size_stride(buf316, (160, 640, 1, 1), (640, 1, 1, 1))
        del buf314
        buf317 = buf274; del buf274  # reuse
        print('triton__24', 'in_out_ptr0', 'buf317', (buf317.sum()/buf317.nelement()).item(), buf317.amax().item(), buf317.amin().item())
        print('triton__24', 'in_ptr0', 'relu_9', (relu_9.sum()/relu_9.nelement()).item(), relu_9.amax().item(), relu_9.amin().item())
        print('triton__24', 'in_ptr1', 'relu_12', (relu_12.sum()/relu_12.nelement()).item(), relu_12.amax().item(), relu_12.amin().item())
        print('triton__24', 'in_ptr2', 'buf294', (buf294.sum()/buf294.nelement()).item(), buf294.amax().item(), buf294.amin().item())
        print('triton__24', 'in_ptr3', 'buf315', (buf315.sum()/buf315.nelement()).item(), buf315.amax().item(), buf315.amin().item())
        triton__24.run(buf317, relu_9, relu_12, buf294, buf315, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__24', 'in_out_ptr0', 'buf317', (buf317.sum()/buf317.nelement()).item(), buf317.amax().item(), buf317.amin().item())
        del relu_12
        del relu_9
        buf318 = buf297; del buf297  # reuse
        buf319 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf325 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf320 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        buf326 = empty_strided((640, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__27', 'in_ptr0', 'buf317', (buf317.sum()/buf317.nelement()).item(), buf317.amax().item(), buf317.amin().item())
        print('triton__27', 'in_ptr1', 'convolution_12', (convolution_12.sum()/convolution_12.nelement()).item(), convolution_12.amax().item(), convolution_12.amin().item())
        print('triton__27', 'in_ptr2', 'unsqueeze_758', (unsqueeze_758.sum()/unsqueeze_758.nelement()).item(), unsqueeze_758.amax().item(), unsqueeze_758.amin().item())
        print('triton__27', 'in_ptr3', 'convolution_11', (convolution_11.sum()/convolution_11.nelement()).item(), convolution_11.amax().item(), convolution_11.amin().item())
        print('triton__27', 'in_ptr4', 'unsqueeze_770', (unsqueeze_770.sum()/unsqueeze_770.nelement()).item(), unsqueeze_770.amax().item(), unsqueeze_770.amin().item())
        print('triton__27', 'in_ptr5', 'squeeze_37', (squeeze_37.sum()/squeeze_37.nelement()).item(), squeeze_37.amax().item(), squeeze_37.amin().item())
        print('triton__27', 'in_ptr6', 'squeeze_34', (squeeze_34.sum()/squeeze_34.nelement()).item(), squeeze_34.amax().item(), squeeze_34.amin().item())
        triton__27.run(buf317, convolution_12, unsqueeze_758, convolution_11, unsqueeze_770, squeeze_37, squeeze_34, buf318, buf319, buf325, buf320, buf326, 640, 2048, grid=grid(640), stream=stream0)
        print('triton__27', 'out_ptr0', 'buf318', (buf318.sum()/buf318.nelement()).item(), buf318.amax().item(), buf318.amin().item())
        print('triton__27', 'out_ptr1', 'buf319', (buf319.sum()/buf319.nelement()).item(), buf319.amax().item(), buf319.amin().item())
        print('triton__27', 'out_ptr2', 'buf325', (buf325.sum()/buf325.nelement()).item(), buf325.amax().item(), buf325.amin().item())
        print('triton__27', 'out_ptr3', 'buf320', (buf320.sum()/buf320.nelement()).item(), buf320.amax().item(), buf320.amin().item())
        print('triton__27', 'out_ptr4', 'buf326', (buf326.sum()/buf326.nelement()).item(), buf326.amax().item(), buf326.amin().item())
        buf321 = buf315; del buf315  # reuse
        buf327 = buf294; del buf294  # reuse
        print('triton__28', 'in_ptr0', 'buf317', (buf317.sum()/buf317.nelement()).item(), buf317.amax().item(), buf317.amin().item())
        print('triton__28', 'in_ptr1', 'convolution_12', (convolution_12.sum()/convolution_12.nelement()).item(), convolution_12.amax().item(), convolution_12.amin().item())
        print('triton__28', 'in_ptr2', 'unsqueeze_758', (unsqueeze_758.sum()/unsqueeze_758.nelement()).item(), unsqueeze_758.amax().item(), unsqueeze_758.amin().item())
        print('triton__28', 'in_ptr3', 'buf319', (buf319.sum()/buf319.nelement()).item(), buf319.amax().item(), buf319.amin().item())
        print('triton__28', 'in_ptr4', 'squeeze_37', (squeeze_37.sum()/squeeze_37.nelement()).item(), squeeze_37.amax().item(), squeeze_37.amin().item())
        print('triton__28', 'in_ptr5', 'buf318', (buf318.sum()/buf318.nelement()).item(), buf318.amax().item(), buf318.amin().item())
        print('triton__28', 'in_ptr6', 'primals_124', (primals_124.sum()/primals_124.nelement()).item(), primals_124.amax().item(), primals_124.amin().item())
        print('triton__28', 'in_ptr7', 'convolution_11', (convolution_11.sum()/convolution_11.nelement()).item(), convolution_11.amax().item(), convolution_11.amin().item())
        print('triton__28', 'in_ptr8', 'unsqueeze_770', (unsqueeze_770.sum()/unsqueeze_770.nelement()).item(), unsqueeze_770.amax().item(), unsqueeze_770.amin().item())
        print('triton__28', 'in_ptr9', 'buf325', (buf325.sum()/buf325.nelement()).item(), buf325.amax().item(), buf325.amin().item())
        print('triton__28', 'in_ptr10', 'squeeze_34', (squeeze_34.sum()/squeeze_34.nelement()).item(), squeeze_34.amax().item(), squeeze_34.amin().item())
        print('triton__28', 'in_ptr11', 'primals_119', (primals_119.sum()/primals_119.nelement()).item(), primals_119.amax().item(), primals_119.amin().item())
        triton__28.run(buf317, convolution_12, unsqueeze_758, buf319, squeeze_37, buf318, primals_124, convolution_11, unsqueeze_770, buf325, squeeze_34, primals_119, buf321, buf327, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__28', 'out_ptr0', 'buf321', (buf321.sum()/buf321.nelement()).item(), buf321.amax().item(), buf321.amin().item())
        print('triton__28', 'out_ptr1', 'buf327', (buf327.sum()/buf327.nelement()).item(), buf327.amax().item(), buf327.amin().item())
        del buf317
        del buf319
        del buf325
        del convolution_11
        del convolution_12
        del primals_119
        del primals_124
        del squeeze_34
        del squeeze_37
        del unsqueeze_758
        del unsqueeze_770
        buf322 = aten.convolution_backward(buf321, relu_6, primals_13, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf321
        del primals_13
        buf323 = buf322[0]
        assert_size_stride(buf323, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf324 = buf322[1]
        assert_size_stride(buf324, (640, 192, 1, 1), (192, 1, 1, 1))
        del buf322
        buf328 = aten.convolution_backward(buf327, relu_8, primals_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf327
        del primals_12
        buf329 = buf328[0]
        assert_size_stride(buf329, (8, 160, 16, 16), (40960, 256, 16, 1))
        buf330 = buf328[1]
        assert_size_stride(buf330, (640, 160, 1, 1), (160, 1, 1, 1))
        del buf328
        buf331 = buf311; del buf311  # reuse
        buf332 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf333 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__22', 'in_ptr0', 'relu_8', (relu_8.sum()/relu_8.nelement()).item(), relu_8.amax().item(), relu_8.amin().item())
        print('triton__22', 'in_ptr1', 'buf329', (buf329.sum()/buf329.nelement()).item(), buf329.amax().item(), buf329.amin().item())
        print('triton__22', 'in_ptr2', 'convolution_10', (convolution_10.sum()/convolution_10.nelement()).item(), convolution_10.amax().item(), convolution_10.amin().item())
        print('triton__22', 'in_ptr3', 'unsqueeze_782', (unsqueeze_782.sum()/unsqueeze_782.nelement()).item(), unsqueeze_782.amax().item(), unsqueeze_782.amin().item())
        print('triton__22', 'in_ptr4', 'squeeze_31', (squeeze_31.sum()/squeeze_31.nelement()).item(), squeeze_31.amax().item(), squeeze_31.amin().item())
        triton__22.run(relu_8, buf329, convolution_10, unsqueeze_782, squeeze_31, buf331, buf332, buf333, 160, 2048, grid=grid(160), stream=stream0)
        print('triton__22', 'out_ptr0', 'buf331', (buf331.sum()/buf331.nelement()).item(), buf331.amax().item(), buf331.amin().item())
        print('triton__22', 'out_ptr1', 'buf332', (buf332.sum()/buf332.nelement()).item(), buf332.amax().item(), buf332.amin().item())
        print('triton__22', 'out_ptr2', 'buf333', (buf333.sum()/buf333.nelement()).item(), buf333.amax().item(), buf333.amin().item())
        buf334 = buf329; del buf329  # reuse
        print('triton__23', 'in_out_ptr0', 'buf334', (buf334.sum()/buf334.nelement()).item(), buf334.amax().item(), buf334.amin().item())
        print('triton__23', 'in_ptr0', 'relu_8', (relu_8.sum()/relu_8.nelement()).item(), relu_8.amax().item(), relu_8.amin().item())
        print('triton__23', 'in_ptr1', 'convolution_10', (convolution_10.sum()/convolution_10.nelement()).item(), convolution_10.amax().item(), convolution_10.amin().item())
        print('triton__23', 'in_ptr2', 'unsqueeze_782', (unsqueeze_782.sum()/unsqueeze_782.nelement()).item(), unsqueeze_782.amax().item(), unsqueeze_782.amin().item())
        print('triton__23', 'in_ptr3', 'buf332', (buf332.sum()/buf332.nelement()).item(), buf332.amax().item(), buf332.amin().item())
        print('triton__23', 'in_ptr4', 'squeeze_31', (squeeze_31.sum()/squeeze_31.nelement()).item(), squeeze_31.amax().item(), squeeze_31.amin().item())
        print('triton__23', 'in_ptr5', 'buf331', (buf331.sum()/buf331.nelement()).item(), buf331.amax().item(), buf331.amin().item())
        print('triton__23', 'in_ptr6', 'primals_114', (primals_114.sum()/primals_114.nelement()).item(), primals_114.amax().item(), primals_114.amin().item())
        triton__23.run(buf334, relu_8, convolution_10, unsqueeze_782, buf332, squeeze_31, buf331, primals_114, 327680, grid=grid(327680), stream=stream0)
        print('triton__23', 'in_out_ptr0', 'buf334', (buf334.sum()/buf334.nelement()).item(), buf334.amax().item(), buf334.amin().item())
        del convolution_10
        del primals_114
        del relu_8
        del squeeze_31
        del unsqueeze_782
        buf335 = aten.convolution_backward(buf334, relu_7, primals_11, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf334
        del primals_11
        buf336 = buf335[0]
        assert_size_stride(buf336, (8, 160, 32, 32), (163840, 1024, 32, 1))
        buf337 = buf335[1]
        assert_size_stride(buf337, (160, 160, 3, 3), (1440, 9, 3, 1))
        del buf335
        buf338 = buf332; del buf332  # reuse
        buf339 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf340 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__29', 'in_ptr0', 'relu_7', (relu_7.sum()/relu_7.nelement()).item(), relu_7.amax().item(), relu_7.amin().item())
        print('triton__29', 'in_ptr1', 'buf336', (buf336.sum()/buf336.nelement()).item(), buf336.amax().item(), buf336.amin().item())
        print('triton__29', 'in_ptr2', 'convolution_9', (convolution_9.sum()/convolution_9.nelement()).item(), convolution_9.amax().item(), convolution_9.amin().item())
        print('triton__29', 'in_ptr3', 'unsqueeze_794', (unsqueeze_794.sum()/unsqueeze_794.nelement()).item(), unsqueeze_794.amax().item(), unsqueeze_794.amin().item())
        print('triton__29', 'in_ptr4', 'squeeze_28', (squeeze_28.sum()/squeeze_28.nelement()).item(), squeeze_28.amax().item(), squeeze_28.amin().item())
        triton__29.run(relu_7, buf336, convolution_9, unsqueeze_794, squeeze_28, buf338, buf339, buf340, 160, 8192, grid=grid(160), stream=stream0)
        print('triton__29', 'out_ptr0', 'buf338', (buf338.sum()/buf338.nelement()).item(), buf338.amax().item(), buf338.amin().item())
        print('triton__29', 'out_ptr1', 'buf339', (buf339.sum()/buf339.nelement()).item(), buf339.amax().item(), buf339.amin().item())
        print('triton__29', 'out_ptr2', 'buf340', (buf340.sum()/buf340.nelement()).item(), buf340.amax().item(), buf340.amin().item())
        buf341 = buf336; del buf336  # reuse
        print('triton__30', 'in_out_ptr0', 'buf341', (buf341.sum()/buf341.nelement()).item(), buf341.amax().item(), buf341.amin().item())
        print('triton__30', 'in_ptr0', 'relu_7', (relu_7.sum()/relu_7.nelement()).item(), relu_7.amax().item(), relu_7.amin().item())
        print('triton__30', 'in_ptr1', 'convolution_9', (convolution_9.sum()/convolution_9.nelement()).item(), convolution_9.amax().item(), convolution_9.amin().item())
        print('triton__30', 'in_ptr2', 'unsqueeze_794', (unsqueeze_794.sum()/unsqueeze_794.nelement()).item(), unsqueeze_794.amax().item(), unsqueeze_794.amin().item())
        print('triton__30', 'in_ptr3', 'buf339', (buf339.sum()/buf339.nelement()).item(), buf339.amax().item(), buf339.amin().item())
        print('triton__30', 'in_ptr4', 'squeeze_28', (squeeze_28.sum()/squeeze_28.nelement()).item(), squeeze_28.amax().item(), squeeze_28.amin().item())
        print('triton__30', 'in_ptr5', 'buf338', (buf338.sum()/buf338.nelement()).item(), buf338.amax().item(), buf338.amin().item())
        print('triton__30', 'in_ptr6', 'primals_109', (primals_109.sum()/primals_109.nelement()).item(), primals_109.amax().item(), primals_109.amin().item())
        triton__30.run(buf341, relu_7, convolution_9, unsqueeze_794, buf339, squeeze_28, buf338, primals_109, 1310720, grid=grid(1310720), stream=stream0)
        print('triton__30', 'in_out_ptr0', 'buf341', (buf341.sum()/buf341.nelement()).item(), buf341.amax().item(), buf341.amin().item())
        del buf339
        del convolution_9
        del primals_109
        del relu_7
        del squeeze_28
        del unsqueeze_794
        buf342 = aten.convolution_backward(buf341, relu_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf341
        del primals_10
        buf343 = buf342[0]
        assert_size_stride(buf343, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf344 = buf342[1]
        assert_size_stride(buf344, (160, 192, 1, 1), (192, 1, 1, 1))
        del buf342
        buf345 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf346 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf348 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__31', 'in_ptr0', 'relu_6', (relu_6.sum()/relu_6.nelement()).item(), relu_6.amax().item(), relu_6.amin().item())
        print('triton__31', 'in_ptr1', 'buf323', (buf323.sum()/buf323.nelement()).item(), buf323.amax().item(), buf323.amin().item())
        print('triton__31', 'in_ptr2', 'buf343', (buf343.sum()/buf343.nelement()).item(), buf343.amax().item(), buf343.amin().item())
        print('triton__31', 'in_ptr3', 'convolution_8', (convolution_8.sum()/convolution_8.nelement()).item(), convolution_8.amax().item(), convolution_8.amin().item())
        print('triton__31', 'in_ptr4', 'unsqueeze_806', (unsqueeze_806.sum()/unsqueeze_806.nelement()).item(), unsqueeze_806.amax().item(), unsqueeze_806.amin().item())
        print('triton__31', 'in_ptr5', 'squeeze_25', (squeeze_25.sum()/squeeze_25.nelement()).item(), squeeze_25.amax().item(), squeeze_25.amin().item())
        triton__31.run(relu_6, buf323, buf343, convolution_8, unsqueeze_806, squeeze_25, buf345, buf346, buf348, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__31', 'out_ptr0', 'buf345', (buf345.sum()/buf345.nelement()).item(), buf345.amax().item(), buf345.amin().item())
        print('triton__31', 'out_ptr1', 'buf346', (buf346.sum()/buf346.nelement()).item(), buf346.amax().item(), buf346.amin().item())
        print('triton__31', 'out_ptr2', 'buf348', (buf348.sum()/buf348.nelement()).item(), buf348.amax().item(), buf348.amin().item())
        buf347 = empty_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda', dtype=torch.float32)
        print('triton__32', 'in_ptr0', 'relu_6', (relu_6.sum()/relu_6.nelement()).item(), relu_6.amax().item(), relu_6.amin().item())
        print('triton__32', 'in_ptr1', 'buf323', (buf323.sum()/buf323.nelement()).item(), buf323.amax().item(), buf323.amin().item())
        print('triton__32', 'in_ptr2', 'buf343', (buf343.sum()/buf343.nelement()).item(), buf343.amax().item(), buf343.amin().item())
        print('triton__32', 'in_ptr3', 'convolution_8', (convolution_8.sum()/convolution_8.nelement()).item(), convolution_8.amax().item(), convolution_8.amin().item())
        print('triton__32', 'in_ptr4', 'unsqueeze_806', (unsqueeze_806.sum()/unsqueeze_806.nelement()).item(), unsqueeze_806.amax().item(), unsqueeze_806.amin().item())
        print('triton__32', 'in_ptr5', 'buf346', (buf346.sum()/buf346.nelement()).item(), buf346.amax().item(), buf346.amin().item())
        print('triton__32', 'in_ptr6', 'squeeze_25', (squeeze_25.sum()/squeeze_25.nelement()).item(), squeeze_25.amax().item(), squeeze_25.amin().item())
        print('triton__32', 'in_ptr7', 'buf345', (buf345.sum()/buf345.nelement()).item(), buf345.amax().item(), buf345.amin().item())
        print('triton__32', 'in_ptr8', 'primals_104', (primals_104.sum()/primals_104.nelement()).item(), primals_104.amax().item(), primals_104.amin().item())
        triton__32.run(relu_6, buf323, buf343, convolution_8, unsqueeze_806, buf346, squeeze_25, buf345, primals_104, buf347, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__32', 'out_ptr0', 'buf347', (buf347.sum()/buf347.nelement()).item(), buf347.amax().item(), buf347.amin().item())
        del convolution_8
        del primals_104
        del squeeze_25
        del unsqueeze_806
        buf349 = aten.convolution_backward(buf347, relu_5, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf347
        del primals_9
        buf350 = buf349[0]
        assert_size_stride(buf350, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf351 = buf349[1]
        assert_size_stride(buf351, (192, 192, 3, 3), (1728, 9, 3, 1))
        del buf349
        buf352 = buf346; del buf346  # reuse
        buf353 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf354 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__33', 'in_ptr0', 'relu_5', (relu_5.sum()/relu_5.nelement()).item(), relu_5.amax().item(), relu_5.amin().item())
        print('triton__33', 'in_ptr1', 'buf350', (buf350.sum()/buf350.nelement()).item(), buf350.amax().item(), buf350.amin().item())
        print('triton__33', 'in_ptr2', 'convolution_7', (convolution_7.sum()/convolution_7.nelement()).item(), convolution_7.amax().item(), convolution_7.amin().item())
        print('triton__33', 'in_ptr3', 'unsqueeze_818', (unsqueeze_818.sum()/unsqueeze_818.nelement()).item(), unsqueeze_818.amax().item(), unsqueeze_818.amin().item())
        print('triton__33', 'in_ptr4', 'squeeze_22', (squeeze_22.sum()/squeeze_22.nelement()).item(), squeeze_22.amax().item(), squeeze_22.amin().item())
        triton__33.run(relu_5, buf350, convolution_7, unsqueeze_818, squeeze_22, buf352, buf353, buf354, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__33', 'out_ptr0', 'buf352', (buf352.sum()/buf352.nelement()).item(), buf352.amax().item(), buf352.amin().item())
        print('triton__33', 'out_ptr1', 'buf353', (buf353.sum()/buf353.nelement()).item(), buf353.amax().item(), buf353.amin().item())
        print('triton__33', 'out_ptr2', 'buf354', (buf354.sum()/buf354.nelement()).item(), buf354.amax().item(), buf354.amin().item())
        buf355 = buf350; del buf350  # reuse
        print('triton__34', 'in_out_ptr0', 'buf355', (buf355.sum()/buf355.nelement()).item(), buf355.amax().item(), buf355.amin().item())
        print('triton__34', 'in_ptr0', 'relu_5', (relu_5.sum()/relu_5.nelement()).item(), relu_5.amax().item(), relu_5.amin().item())
        print('triton__34', 'in_ptr1', 'convolution_7', (convolution_7.sum()/convolution_7.nelement()).item(), convolution_7.amax().item(), convolution_7.amin().item())
        print('triton__34', 'in_ptr2', 'unsqueeze_818', (unsqueeze_818.sum()/unsqueeze_818.nelement()).item(), unsqueeze_818.amax().item(), unsqueeze_818.amin().item())
        print('triton__34', 'in_ptr3', 'buf353', (buf353.sum()/buf353.nelement()).item(), buf353.amax().item(), buf353.amin().item())
        print('triton__34', 'in_ptr4', 'squeeze_22', (squeeze_22.sum()/squeeze_22.nelement()).item(), squeeze_22.amax().item(), squeeze_22.amin().item())
        print('triton__34', 'in_ptr5', 'buf352', (buf352.sum()/buf352.nelement()).item(), buf352.amax().item(), buf352.amin().item())
        print('triton__34', 'in_ptr6', 'primals_99', (primals_99.sum()/primals_99.nelement()).item(), primals_99.amax().item(), primals_99.amin().item())
        triton__34.run(buf355, relu_5, convolution_7, unsqueeze_818, buf353, squeeze_22, buf352, primals_99, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__34', 'in_out_ptr0', 'buf355', (buf355.sum()/buf355.nelement()).item(), buf355.amax().item(), buf355.amin().item())
        del convolution_7
        del primals_99
        del relu_5
        del squeeze_22
        del unsqueeze_818
        buf356 = aten.convolution_backward(buf355, relu_4, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf355
        del primals_8
        buf357 = buf356[0]
        assert_size_stride(buf357, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf358 = buf356[1]
        assert_size_stride(buf358, (192, 192, 3, 3), (1728, 9, 3, 1))
        del buf356
        buf359 = buf323; del buf323  # reuse
        print('triton__35', 'in_out_ptr0', 'buf359', (buf359.sum()/buf359.nelement()).item(), buf359.amax().item(), buf359.amin().item())
        print('triton__35', 'in_ptr0', 'relu_4', (relu_4.sum()/relu_4.nelement()).item(), relu_4.amax().item(), relu_4.amin().item())
        print('triton__35', 'in_ptr1', 'relu_6', (relu_6.sum()/relu_6.nelement()).item(), relu_6.amax().item(), relu_6.amin().item())
        print('triton__35', 'in_ptr2', 'buf343', (buf343.sum()/buf343.nelement()).item(), buf343.amax().item(), buf343.amin().item())
        print('triton__35', 'in_ptr3', 'buf357', (buf357.sum()/buf357.nelement()).item(), buf357.amax().item(), buf357.amin().item())
        triton__35.run(buf359, relu_4, relu_6, buf343, buf357, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__35', 'in_out_ptr0', 'buf359', (buf359.sum()/buf359.nelement()).item(), buf359.amax().item(), buf359.amin().item())
        del relu_4
        del relu_6
        buf360 = buf353; del buf353  # reuse
        buf361 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf367 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf362 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf368 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__36', 'in_ptr0', 'buf359', (buf359.sum()/buf359.nelement()).item(), buf359.amax().item(), buf359.amin().item())
        print('triton__36', 'in_ptr1', 'convolution_6', (convolution_6.sum()/convolution_6.nelement()).item(), convolution_6.amax().item(), convolution_6.amin().item())
        print('triton__36', 'in_ptr2', 'unsqueeze_830', (unsqueeze_830.sum()/unsqueeze_830.nelement()).item(), unsqueeze_830.amax().item(), unsqueeze_830.amin().item())
        print('triton__36', 'in_ptr3', 'convolution_5', (convolution_5.sum()/convolution_5.nelement()).item(), convolution_5.amax().item(), convolution_5.amin().item())
        print('triton__36', 'in_ptr4', 'unsqueeze_842', (unsqueeze_842.sum()/unsqueeze_842.nelement()).item(), unsqueeze_842.amax().item(), unsqueeze_842.amin().item())
        print('triton__36', 'in_ptr5', 'squeeze_19', (squeeze_19.sum()/squeeze_19.nelement()).item(), squeeze_19.amax().item(), squeeze_19.amin().item())
        print('triton__36', 'in_ptr6', 'squeeze_16', (squeeze_16.sum()/squeeze_16.nelement()).item(), squeeze_16.amax().item(), squeeze_16.amin().item())
        triton__36.run(buf359, convolution_6, unsqueeze_830, convolution_5, unsqueeze_842, squeeze_19, squeeze_16, buf360, buf361, buf367, buf362, buf368, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__36', 'out_ptr0', 'buf360', (buf360.sum()/buf360.nelement()).item(), buf360.amax().item(), buf360.amin().item())
        print('triton__36', 'out_ptr1', 'buf361', (buf361.sum()/buf361.nelement()).item(), buf361.amax().item(), buf361.amin().item())
        print('triton__36', 'out_ptr2', 'buf367', (buf367.sum()/buf367.nelement()).item(), buf367.amax().item(), buf367.amin().item())
        print('triton__36', 'out_ptr3', 'buf362', (buf362.sum()/buf362.nelement()).item(), buf362.amax().item(), buf362.amin().item())
        print('triton__36', 'out_ptr4', 'buf368', (buf368.sum()/buf368.nelement()).item(), buf368.amax().item(), buf368.amin().item())
        buf363 = buf357; del buf357  # reuse
        buf369 = buf343; del buf343  # reuse
        print('triton__37', 'in_ptr0', 'buf359', (buf359.sum()/buf359.nelement()).item(), buf359.amax().item(), buf359.amin().item())
        print('triton__37', 'in_ptr1', 'convolution_6', (convolution_6.sum()/convolution_6.nelement()).item(), convolution_6.amax().item(), convolution_6.amin().item())
        print('triton__37', 'in_ptr2', 'unsqueeze_830', (unsqueeze_830.sum()/unsqueeze_830.nelement()).item(), unsqueeze_830.amax().item(), unsqueeze_830.amin().item())
        print('triton__37', 'in_ptr3', 'buf361', (buf361.sum()/buf361.nelement()).item(), buf361.amax().item(), buf361.amin().item())
        print('triton__37', 'in_ptr4', 'squeeze_19', (squeeze_19.sum()/squeeze_19.nelement()).item(), squeeze_19.amax().item(), squeeze_19.amin().item())
        print('triton__37', 'in_ptr5', 'buf360', (buf360.sum()/buf360.nelement()).item(), buf360.amax().item(), buf360.amin().item())
        print('triton__37', 'in_ptr6', 'primals_94', (primals_94.sum()/primals_94.nelement()).item(), primals_94.amax().item(), primals_94.amin().item())
        print('triton__37', 'in_ptr7', 'convolution_5', (convolution_5.sum()/convolution_5.nelement()).item(), convolution_5.amax().item(), convolution_5.amin().item())
        print('triton__37', 'in_ptr8', 'unsqueeze_842', (unsqueeze_842.sum()/unsqueeze_842.nelement()).item(), unsqueeze_842.amax().item(), unsqueeze_842.amin().item())
        print('triton__37', 'in_ptr9', 'buf367', (buf367.sum()/buf367.nelement()).item(), buf367.amax().item(), buf367.amin().item())
        print('triton__37', 'in_ptr10', 'squeeze_16', (squeeze_16.sum()/squeeze_16.nelement()).item(), squeeze_16.amax().item(), squeeze_16.amin().item())
        print('triton__37', 'in_ptr11', 'primals_89', (primals_89.sum()/primals_89.nelement()).item(), primals_89.amax().item(), primals_89.amin().item())
        triton__37.run(buf359, convolution_6, unsqueeze_830, buf361, squeeze_19, buf360, primals_94, convolution_5, unsqueeze_842, buf367, squeeze_16, primals_89, buf363, buf369, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__37', 'out_ptr0', 'buf363', (buf363.sum()/buf363.nelement()).item(), buf363.amax().item(), buf363.amin().item())
        print('triton__37', 'out_ptr1', 'buf369', (buf369.sum()/buf369.nelement()).item(), buf369.amax().item(), buf369.amin().item())
        del buf359
        del convolution_5
        del convolution_6
        del primals_89
        del primals_94
        del squeeze_16
        del squeeze_19
        del unsqueeze_830
        del unsqueeze_842
        buf364 = aten.convolution_backward(buf363, relu_2, primals_7, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf363
        del primals_7
        buf365 = buf364[0]
        assert_size_stride(buf365, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf366 = buf364[1]
        assert_size_stride(buf366, (192, 128, 1, 1), (128, 1, 1, 1))
        del buf364
        buf370 = aten.convolution_backward(buf369, relu_3, primals_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf369
        del primals_6
        buf371 = buf370[0]
        assert_size_stride(buf371, (8, 192, 32, 32), (196608, 1024, 32, 1))
        buf372 = buf370[1]
        assert_size_stride(buf372, (192, 192, 3, 3), (1728, 9, 3, 1))
        del buf370
        buf373 = buf367; del buf367  # reuse
        buf374 = buf361; del buf361  # reuse
        buf375 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__33', 'in_ptr0', 'relu_3', (relu_3.sum()/relu_3.nelement()).item(), relu_3.amax().item(), relu_3.amin().item())
        print('triton__33', 'in_ptr1', 'buf371', (buf371.sum()/buf371.nelement()).item(), buf371.amax().item(), buf371.amin().item())
        print('triton__33', 'in_ptr2', 'convolution_4', (convolution_4.sum()/convolution_4.nelement()).item(), convolution_4.amax().item(), convolution_4.amin().item())
        print('triton__33', 'in_ptr3', 'unsqueeze_854', (unsqueeze_854.sum()/unsqueeze_854.nelement()).item(), unsqueeze_854.amax().item(), unsqueeze_854.amin().item())
        print('triton__33', 'in_ptr4', 'squeeze_13', (squeeze_13.sum()/squeeze_13.nelement()).item(), squeeze_13.amax().item(), squeeze_13.amin().item())
        triton__33.run(relu_3, buf371, convolution_4, unsqueeze_854, squeeze_13, buf373, buf374, buf375, 192, 8192, grid=grid(192), stream=stream0)
        print('triton__33', 'out_ptr0', 'buf373', (buf373.sum()/buf373.nelement()).item(), buf373.amax().item(), buf373.amin().item())
        print('triton__33', 'out_ptr1', 'buf374', (buf374.sum()/buf374.nelement()).item(), buf374.amax().item(), buf374.amin().item())
        print('triton__33', 'out_ptr2', 'buf375', (buf375.sum()/buf375.nelement()).item(), buf375.amax().item(), buf375.amin().item())
        buf376 = buf371; del buf371  # reuse
        print('triton__34', 'in_out_ptr0', 'buf376', (buf376.sum()/buf376.nelement()).item(), buf376.amax().item(), buf376.amin().item())
        print('triton__34', 'in_ptr0', 'relu_3', (relu_3.sum()/relu_3.nelement()).item(), relu_3.amax().item(), relu_3.amin().item())
        print('triton__34', 'in_ptr1', 'convolution_4', (convolution_4.sum()/convolution_4.nelement()).item(), convolution_4.amax().item(), convolution_4.amin().item())
        print('triton__34', 'in_ptr2', 'unsqueeze_854', (unsqueeze_854.sum()/unsqueeze_854.nelement()).item(), unsqueeze_854.amax().item(), unsqueeze_854.amin().item())
        print('triton__34', 'in_ptr3', 'buf374', (buf374.sum()/buf374.nelement()).item(), buf374.amax().item(), buf374.amin().item())
        print('triton__34', 'in_ptr4', 'squeeze_13', (squeeze_13.sum()/squeeze_13.nelement()).item(), squeeze_13.amax().item(), squeeze_13.amin().item())
        print('triton__34', 'in_ptr5', 'buf373', (buf373.sum()/buf373.nelement()).item(), buf373.amax().item(), buf373.amin().item())
        print('triton__34', 'in_ptr6', 'primals_84', (primals_84.sum()/primals_84.nelement()).item(), primals_84.amax().item(), primals_84.amin().item())
        triton__34.run(buf376, relu_3, convolution_4, unsqueeze_854, buf374, squeeze_13, buf373, primals_84, 1572864, grid=grid(1572864), stream=stream0)
        print('triton__34', 'in_out_ptr0', 'buf376', (buf376.sum()/buf376.nelement()).item(), buf376.amax().item(), buf376.amin().item())
        del buf374
        del convolution_4
        del primals_84
        del relu_3
        del squeeze_13
        del unsqueeze_854
        buf377 = aten.convolution_backward(buf376, relu_2, primals_5, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf376
        del primals_5
        buf378 = buf377[0]
        assert_size_stride(buf378, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf379 = buf377[1]
        assert_size_stride(buf379, (192, 128, 3, 3), (1152, 9, 3, 1))
        del buf377
        buf380 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf382 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf389 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        print('triton__38', 'in_ptr0', 'relu_2', (relu_2.sum()/relu_2.nelement()).item(), relu_2.amax().item(), relu_2.amin().item())
        print('triton__38', 'in_ptr1', 'buf365', (buf365.sum()/buf365.nelement()).item(), buf365.amax().item(), buf365.amin().item())
        print('triton__38', 'in_ptr2', 'buf378', (buf378.sum()/buf378.nelement()).item(), buf378.amax().item(), buf378.amin().item())
        print('triton__38', 'in_ptr3', 'convolution_3', (convolution_3.sum()/convolution_3.nelement()).item(), convolution_3.amax().item(), convolution_3.amin().item())
        print('triton__38', 'in_ptr4', 'unsqueeze_866', (unsqueeze_866.sum()/unsqueeze_866.nelement()).item(), unsqueeze_866.amax().item(), unsqueeze_866.amin().item())
        print('triton__38', 'in_ptr5', 'convolution_2', (convolution_2.sum()/convolution_2.nelement()).item(), convolution_2.amax().item(), convolution_2.amin().item())
        print('triton__38', 'in_ptr6', 'unsqueeze_878', (unsqueeze_878.sum()/unsqueeze_878.nelement()).item(), unsqueeze_878.amax().item(), unsqueeze_878.amin().item())
        triton__38.run(relu_2, buf365, buf378, convolution_3, unsqueeze_866, convolution_2, unsqueeze_878, buf380, buf382, buf389, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__38', 'out_ptr0', 'buf380', (buf380.sum()/buf380.nelement()).item(), buf380.amax().item(), buf380.amin().item())
        print('triton__38', 'out_ptr1', 'buf382', (buf382.sum()/buf382.nelement()).item(), buf382.amax().item(), buf382.amin().item())
        print('triton__38', 'out_ptr2', 'buf389', (buf389.sum()/buf389.nelement()).item(), buf389.amax().item(), buf389.amin().item())
        buf381 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__39', 'in_ptr0', 'buf380', (buf380.sum()/buf380.nelement()).item(), buf380.amax().item(), buf380.amin().item())
        triton__39.run(buf380, buf381, 128, 4, grid=grid(128), stream=stream0)
        print('triton__39', 'out_ptr0', 'buf381', (buf381.sum()/buf381.nelement()).item(), buf381.amax().item(), buf381.amin().item())
        del buf380
        buf383 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf385 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__40', 'in_ptr0', 'buf382', (buf382.sum()/buf382.nelement()).item(), buf382.amax().item(), buf382.amin().item())
        print('triton__40', 'in_ptr1', 'squeeze_10', (squeeze_10.sum()/squeeze_10.nelement()).item(), squeeze_10.amax().item(), squeeze_10.amin().item())
        triton__40.run(buf382, squeeze_10, buf383, buf385, 128, 4, grid=grid(128), stream=stream0)
        print('triton__40', 'out_ptr0', 'buf383', (buf383.sum()/buf383.nelement()).item(), buf383.amax().item(), buf383.amin().item())
        print('triton__40', 'out_ptr1', 'buf385', (buf385.sum()/buf385.nelement()).item(), buf385.amax().item(), buf385.amin().item())
        buf390 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf392 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__40', 'in_ptr0', 'buf389', (buf389.sum()/buf389.nelement()).item(), buf389.amax().item(), buf389.amin().item())
        print('triton__40', 'in_ptr1', 'squeeze_7', (squeeze_7.sum()/squeeze_7.nelement()).item(), squeeze_7.amax().item(), squeeze_7.amin().item())
        triton__40.run(buf389, squeeze_7, buf390, buf392, 128, 4, grid=grid(128), stream=stream0)
        print('triton__40', 'out_ptr0', 'buf390', (buf390.sum()/buf390.nelement()).item(), buf390.amax().item(), buf390.amin().item())
        print('triton__40', 'out_ptr1', 'buf392', (buf392.sum()/buf392.nelement()).item(), buf392.amax().item(), buf392.amin().item())
        buf384 = empty_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda', dtype=torch.float32)
        buf391 = empty_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda', dtype=torch.float32)
        print('triton__41', 'in_ptr0', 'relu_2', (relu_2.sum()/relu_2.nelement()).item(), relu_2.amax().item(), relu_2.amin().item())
        print('triton__41', 'in_ptr1', 'buf365', (buf365.sum()/buf365.nelement()).item(), buf365.amax().item(), buf365.amin().item())
        print('triton__41', 'in_ptr2', 'buf378', (buf378.sum()/buf378.nelement()).item(), buf378.amax().item(), buf378.amin().item())
        print('triton__41', 'in_ptr3', 'convolution_3', (convolution_3.sum()/convolution_3.nelement()).item(), convolution_3.amax().item(), convolution_3.amin().item())
        print('triton__41', 'in_ptr4', 'unsqueeze_866', (unsqueeze_866.sum()/unsqueeze_866.nelement()).item(), unsqueeze_866.amax().item(), unsqueeze_866.amin().item())
        print('triton__41', 'in_ptr5', 'buf383', (buf383.sum()/buf383.nelement()).item(), buf383.amax().item(), buf383.amin().item())
        print('triton__41', 'in_ptr6', 'squeeze_10', (squeeze_10.sum()/squeeze_10.nelement()).item(), squeeze_10.amax().item(), squeeze_10.amin().item())
        print('triton__41', 'in_ptr7', 'buf381', (buf381.sum()/buf381.nelement()).item(), buf381.amax().item(), buf381.amin().item())
        print('triton__41', 'in_ptr8', 'primals_79', (primals_79.sum()/primals_79.nelement()).item(), primals_79.amax().item(), primals_79.amin().item())
        print('triton__41', 'in_ptr9', 'convolution_2', (convolution_2.sum()/convolution_2.nelement()).item(), convolution_2.amax().item(), convolution_2.amin().item())
        print('triton__41', 'in_ptr10', 'unsqueeze_878', (unsqueeze_878.sum()/unsqueeze_878.nelement()).item(), unsqueeze_878.amax().item(), unsqueeze_878.amin().item())
        print('triton__41', 'in_ptr11', 'buf390', (buf390.sum()/buf390.nelement()).item(), buf390.amax().item(), buf390.amin().item())
        print('triton__41', 'in_ptr12', 'squeeze_7', (squeeze_7.sum()/squeeze_7.nelement()).item(), squeeze_7.amax().item(), squeeze_7.amin().item())
        print('triton__41', 'in_ptr13', 'primals_74', (primals_74.sum()/primals_74.nelement()).item(), primals_74.amax().item(), primals_74.amin().item())
        triton__41.run(relu_2, buf365, buf378, convolution_3, unsqueeze_866, buf383, squeeze_10, buf381, primals_79, convolution_2, unsqueeze_878, buf390, squeeze_7, primals_74, buf384, buf391, 4194304, grid=grid(4194304), stream=stream0)
        print('triton__41', 'out_ptr0', 'buf384', (buf384.sum()/buf384.nelement()).item(), buf384.amax().item(), buf384.amin().item())
        print('triton__41', 'out_ptr1', 'buf391', (buf391.sum()/buf391.nelement()).item(), buf391.amax().item(), buf391.amin().item())
        del buf365
        del buf378
        del convolution_2
        del convolution_3
        del primals_74
        del primals_79
        del relu_2
        del squeeze_10
        del squeeze_7
        del unsqueeze_866
        del unsqueeze_878
        buf386 = aten.convolution_backward(buf384, relu, primals_4, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf384
        del primals_4
        buf387 = buf386[0]
        assert_size_stride(buf387, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf388 = buf386[1]
        assert_size_stride(buf388, (128, 32, 1, 1), (32, 1, 1, 1))
        del buf386
        buf393 = aten.convolution_backward(buf391, relu_1, primals_3, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf391
        del primals_3
        buf394 = buf393[0]
        assert_size_stride(buf394, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf395 = buf393[1]
        assert_size_stride(buf395, (128, 128, 3, 3), (1152, 9, 3, 1))
        del buf393
        buf396 = buf389; del buf389  # reuse
        buf398 = buf382; del buf382  # reuse
        print('triton__42', 'in_ptr0', 'relu_1', (relu_1.sum()/relu_1.nelement()).item(), relu_1.amax().item(), relu_1.amin().item())
        print('triton__42', 'in_ptr1', 'buf394', (buf394.sum()/buf394.nelement()).item(), buf394.amax().item(), buf394.amin().item())
        print('triton__42', 'in_ptr2', 'convolution_1', (convolution_1.sum()/convolution_1.nelement()).item(), convolution_1.amax().item(), convolution_1.amin().item())
        print('triton__42', 'in_ptr3', 'unsqueeze_890', (unsqueeze_890.sum()/unsqueeze_890.nelement()).item(), unsqueeze_890.amax().item(), unsqueeze_890.amin().item())
        triton__42.run(relu_1, buf394, convolution_1, unsqueeze_890, buf396, buf398, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__42', 'out_ptr0', 'buf396', (buf396.sum()/buf396.nelement()).item(), buf396.amax().item(), buf396.amin().item())
        print('triton__42', 'out_ptr1', 'buf398', (buf398.sum()/buf398.nelement()).item(), buf398.amax().item(), buf398.amin().item())
        buf397 = buf390; del buf390  # reuse
        print('triton__39', 'in_ptr0', 'buf396', (buf396.sum()/buf396.nelement()).item(), buf396.amax().item(), buf396.amin().item())
        triton__39.run(buf396, buf397, 128, 4, grid=grid(128), stream=stream0)
        print('triton__39', 'out_ptr0', 'buf397', (buf397.sum()/buf397.nelement()).item(), buf397.amax().item(), buf397.amin().item())
        buf399 = buf383; del buf383  # reuse
        buf400 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__40', 'in_ptr0', 'buf398', (buf398.sum()/buf398.nelement()).item(), buf398.amax().item(), buf398.amin().item())
        print('triton__40', 'in_ptr1', 'squeeze_4', (squeeze_4.sum()/squeeze_4.nelement()).item(), squeeze_4.amax().item(), squeeze_4.amin().item())
        triton__40.run(buf398, squeeze_4, buf399, buf400, 128, 4, grid=grid(128), stream=stream0)
        print('triton__40', 'out_ptr0', 'buf399', (buf399.sum()/buf399.nelement()).item(), buf399.amax().item(), buf399.amin().item())
        print('triton__40', 'out_ptr1', 'buf400', (buf400.sum()/buf400.nelement()).item(), buf400.amax().item(), buf400.amin().item())
        buf401 = buf394; del buf394  # reuse
        print('triton__43', 'in_out_ptr0', 'buf401', (buf401.sum()/buf401.nelement()).item(), buf401.amax().item(), buf401.amin().item())
        print('triton__43', 'in_ptr0', 'relu_1', (relu_1.sum()/relu_1.nelement()).item(), relu_1.amax().item(), relu_1.amin().item())
        print('triton__43', 'in_ptr1', 'convolution_1', (convolution_1.sum()/convolution_1.nelement()).item(), convolution_1.amax().item(), convolution_1.amin().item())
        print('triton__43', 'in_ptr2', 'unsqueeze_890', (unsqueeze_890.sum()/unsqueeze_890.nelement()).item(), unsqueeze_890.amax().item(), unsqueeze_890.amin().item())
        print('triton__43', 'in_ptr3', 'buf399', (buf399.sum()/buf399.nelement()).item(), buf399.amax().item(), buf399.amin().item())
        print('triton__43', 'in_ptr4', 'squeeze_4', (squeeze_4.sum()/squeeze_4.nelement()).item(), squeeze_4.amax().item(), squeeze_4.amin().item())
        print('triton__43', 'in_ptr5', 'buf397', (buf397.sum()/buf397.nelement()).item(), buf397.amax().item(), buf397.amin().item())
        print('triton__43', 'in_ptr6', 'primals_69', (primals_69.sum()/primals_69.nelement()).item(), primals_69.amax().item(), primals_69.amin().item())
        triton__43.run(buf401, relu_1, convolution_1, unsqueeze_890, buf399, squeeze_4, buf397, primals_69, 4194304, grid=grid(4194304), stream=stream0)
        print('triton__43', 'in_out_ptr0', 'buf401', (buf401.sum()/buf401.nelement()).item(), buf401.amax().item(), buf401.amin().item())
        del buf399
        del convolution_1
        del primals_69
        del relu_1
        del squeeze_4
        del unsqueeze_890
        buf402 = aten.convolution_backward(buf401, relu, primals_2, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf401
        del primals_2
        buf403 = buf402[0]
        assert_size_stride(buf403, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf404 = buf402[1]
        assert_size_stride(buf404, (128, 32, 3, 3), (288, 9, 3, 1))
        del buf402
        buf405 = as_strided(buf398, (32, 16), (16, 1)); del buf398  # reuse
        buf407 = as_strided(buf396, (32, 16), (16, 1)); del buf396  # reuse
        print('triton__44', 'in_ptr0', 'relu', (relu.sum()/relu.nelement()).item(), relu.amax().item(), relu.amin().item())
        print('triton__44', 'in_ptr1', 'buf387', (buf387.sum()/buf387.nelement()).item(), buf387.amax().item(), buf387.amin().item())
        print('triton__44', 'in_ptr2', 'buf403', (buf403.sum()/buf403.nelement()).item(), buf403.amax().item(), buf403.amin().item())
        print('triton__44', 'in_ptr3', 'convolution', (convolution.sum()/convolution.nelement()).item(), convolution.amax().item(), convolution.amin().item())
        print('triton__44', 'in_ptr4', 'unsqueeze_902', (unsqueeze_902.sum()/unsqueeze_902.nelement()).item(), unsqueeze_902.amax().item(), unsqueeze_902.amin().item())
        triton__44.run(relu, buf387, buf403, convolution, unsqueeze_902, buf405, buf407, 512, 8192, grid=grid(512), stream=stream0)
        print('triton__44', 'out_ptr0', 'buf405', (buf405.sum()/buf405.nelement()).item(), buf405.amax().item(), buf405.amin().item())
        print('triton__44', 'out_ptr1', 'buf407', (buf407.sum()/buf407.nelement()).item(), buf407.amax().item(), buf407.amin().item())
        buf406 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__45', 'in_ptr0', 'buf405', (buf405.sum()/buf405.nelement()).item(), buf405.amax().item(), buf405.amin().item())
        triton__45.run(buf405, buf406, 32, 16, grid=grid(32), stream=stream0)
        print('triton__45', 'out_ptr0', 'buf406', (buf406.sum()/buf406.nelement()).item(), buf406.amax().item(), buf406.amin().item())
        del buf405
        buf408 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf410 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        print('triton__46', 'in_ptr0', 'buf407', (buf407.sum()/buf407.nelement()).item(), buf407.amax().item(), buf407.amin().item())
        print('triton__46', 'in_ptr1', 'squeeze_1', (squeeze_1.sum()/squeeze_1.nelement()).item(), squeeze_1.amax().item(), squeeze_1.amin().item())
        triton__46.run(buf407, squeeze_1, buf408, buf410, 32, 16, grid=grid(32), stream=stream0)
        print('triton__46', 'out_ptr0', 'buf408', (buf408.sum()/buf408.nelement()).item(), buf408.amax().item(), buf408.amin().item())
        print('triton__46', 'out_ptr1', 'buf410', (buf410.sum()/buf410.nelement()).item(), buf410.amax().item(), buf410.amin().item())
        del buf407
        buf409 = buf387; del buf387  # reuse
        print('triton__47', 'in_out_ptr0', 'buf409', (buf409.sum()/buf409.nelement()).item(), buf409.amax().item(), buf409.amin().item())
        print('triton__47', 'in_ptr0', 'relu', (relu.sum()/relu.nelement()).item(), relu.amax().item(), relu.amin().item())
        print('triton__47', 'in_ptr1', 'buf403', (buf403.sum()/buf403.nelement()).item(), buf403.amax().item(), buf403.amin().item())
        print('triton__47', 'in_ptr2', 'convolution', (convolution.sum()/convolution.nelement()).item(), convolution.amax().item(), convolution.amin().item())
        print('triton__47', 'in_ptr3', 'unsqueeze_902', (unsqueeze_902.sum()/unsqueeze_902.nelement()).item(), unsqueeze_902.amax().item(), unsqueeze_902.amin().item())
        print('triton__47', 'in_ptr4', 'buf408', (buf408.sum()/buf408.nelement()).item(), buf408.amax().item(), buf408.amin().item())
        print('triton__47', 'in_ptr5', 'squeeze_1', (squeeze_1.sum()/squeeze_1.nelement()).item(), squeeze_1.amax().item(), squeeze_1.amin().item())
        print('triton__47', 'in_ptr6', 'buf406', (buf406.sum()/buf406.nelement()).item(), buf406.amax().item(), buf406.amin().item())
        print('triton__47', 'in_ptr7', 'primals_64', (primals_64.sum()/primals_64.nelement()).item(), primals_64.amax().item(), primals_64.amin().item())
        triton__47.run(buf409, relu, buf403, convolution, unsqueeze_902, buf408, squeeze_1, buf406, primals_64, 4194304, grid=grid(4194304), stream=stream0)
        print('triton__47', 'in_out_ptr0', 'buf409', (buf409.sum()/buf409.nelement()).item(), buf409.amax().item(), buf409.amin().item())
        del buf403
        del buf408
        del convolution
        del primals_64
        del relu
        del squeeze_1
        del unsqueeze_902
        buf411 = aten.convolution_backward(buf409, primals_60, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf409
        del primals_1
        del primals_60
        buf412 = buf411[1]
        assert_size_stride(buf412, (32, 3, 3, 3), (27, 9, 3, 1))
        del buf411
        return (buf412, buf404, buf395, buf388, buf379, buf372, buf366, buf358, buf351, buf344, buf337, buf330, buf324, buf316, buf309, buf302, buf295, buf288, buf281, buf273, buf266, buf259, buf252, buf245, buf238, buf230, buf223, buf216, buf209, buf202, buf195, buf189, buf181, buf174, buf167, buf160, buf153, buf146, buf138, buf131, buf124, buf117, buf110, buf103, buf95, buf88, buf81, buf74, buf67, buf60, buf52, buf45, buf38, buf30, buf23, buf16, buf9, as_strided(buf1, (1000, 2560), (2560, 1)), as_strided(buf2, (1000, ), (1, )), None, None, None, None, buf410, buf406, None, None, None, buf400, buf397, None, None, None, buf392, buf381, None, None, None, buf385, buf381, None, None, None, buf375, buf373, None, None, None, buf368, buf360, None, None, None, buf362, buf360, None, None, None, buf354, buf352, None, None, None, buf348, buf345, None, None, None, buf340, buf338, None, None, None, buf333, buf331, None, None, None, buf326, buf318, None, None, None, buf320, buf318, None, None, None, buf312, buf310, None, None, None, buf305, buf303, None, None, None, buf299, buf296, None, None, None, buf291, buf289, None, None, None, buf284, buf282, None, None, None, buf277, buf275, None, None, None, buf269, buf267, None, None, None, buf262, buf260, None, None, None, buf256, buf253, None, None, None, buf248, buf246, None, None, None, buf241, buf239, None, None, None, buf234, buf232, None, None, None, buf226, buf224, None, None, None, buf219, buf217, None, None, None, buf213, buf210, None, None, None, buf205, buf203, None, None, None, buf198, buf196, None, None, None, buf191, buf183, None, None, None, buf185, buf183, None, None, None, buf177, buf175, None, None, None, buf170, buf168, None, None, None, buf164, buf161, None, None, None, buf156, buf154, None, None, None, buf149, buf147, None, None, None, buf142, buf140, None, None, None, buf134, buf132, None, None, None, buf127, buf125, None, None, None, buf121, buf118, None, None, None, buf113, buf111, None, None, None, buf106, buf104, None, None, None, buf99, buf97, None, None, None, buf91, buf89, None, None, None, buf84, buf82, None, None, None, buf78, buf75, None, None, None, buf70, buf68, None, None, None, buf63, buf61, None, None, None, buf56, buf54, None, None, None, buf48, buf46, None, None, None, buf41, buf39, None, None, None, buf34, buf31, None, None, None, buf26, buf24, None, None, None, buf19, buf17, None, None, None, buf12, buf10, None, None, None, buf5, buf3, )


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
    primals_60 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 160, 32, 32), (163840, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 160, 32, 32), (163840, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 160, 16, 16), (40960, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 640, 16, 16), (163840, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1920, 16, 16), (491520, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 1920, 16, 16), (491520, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 1920, 8, 8), (122880, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 2560, 8, 8), (163840, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 2560, 8, 8), (163840, 64, 8, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_230 = rand_strided((1, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_254 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_278 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_302 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_326 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_350 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_374 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_446 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_518 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_530 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_554 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_566 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_590 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_614 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_626 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_662 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_698 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_734 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_782 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_806 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_866 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_31 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_34 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_36 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_37 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_38 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_39 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_40 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_41 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_42 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_43 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_44 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_45 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_46 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_47 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_48 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_49 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_50 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_51 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_52 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_53 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_54 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_55 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_56 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_57 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_58 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_59 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_60 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_61 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_62 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_63 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_64 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_65 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_66 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_67 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_68 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_69 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_70 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_71 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_72 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_73 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_74 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_75 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_76 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_77 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_78 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_79 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_80 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_81 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_82 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_83 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_84 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_85 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_86 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_87 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_88 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_89 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_90 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_91 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_92 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_93 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_94 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_95 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_96 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_97 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_98 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_99 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_100 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_101 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_102 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_103 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_104 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_105 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_106 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_107 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_108 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_109 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_110 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_111 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_112 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_113 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_114 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_115 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    tangents_116 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_118 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_119 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_120 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_121 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_122 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_124 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_125 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_126 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_127 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_130 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_131 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_132 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_133 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_134 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_137 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_138 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_139 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_140 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_142 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_143 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_144 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_145 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_146 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_148 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_149 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_151 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_152 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_153 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_154 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_155 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_158 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_160 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_161 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_162 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_163 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_165 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_166 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_167 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_168 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_169 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_172 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_60, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, primals_204, primals_209, primals_214, primals_219, primals_224, primals_229, primals_234, primals_239, primals_244, primals_249, primals_254, primals_259, primals_264, primals_269, primals_274, primals_279, primals_284, primals_289, primals_294, primals_299, primals_304, primals_309, primals_314, primals_319, primals_324, primals_329, primals_334, primals_339, primals_344, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, view, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160, tangents_161, tangents_162, tangents_163, tangents_164, tangents_165, tangents_166, tangents_167, tangents_168, tangents_169, tangents_170, tangents_171, tangents_172]))
