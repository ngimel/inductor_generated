
from ctypes import c_void_p, c_long
import torch
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

@reduction(size_hints=[1024, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
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

@reduction(size_hints=[8192, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = 0.16666666666666666
    tmp8 = tmp5 * tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp14 = tl.load(in_ptr5 + (x0), xmask)
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr4 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = 49.0
        tmp5 = tmp3 / tmp4
        tmp7 = tmp5 * tmp6
        tmp9 = tmp8 / tmp4
        tmp10 = tmp7 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp11 * tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp12, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp17, xmask)
    tmp18 = tl.load(in_ptr6 + (x0), xmask)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp19, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x4), xmask)
    tmp6 = tl.load(in_ptr2 + (x4), xmask)
    tmp8 = tl.load(in_ptr3 + (x4), xmask)
    tmp12 = tl.load(in_ptr4 + (x3), xmask)
    tmp13 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp23 = tl.load(in_ptr8 + (x1), xmask)
    tmp26 = tl.load(in_ptr9 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = 49.0
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 / tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp14 = tmp12 - tmp13
    tmp16 = 0.002551020408163265
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tmp11 - tmp21
    tmp24 = tmp23 * tmp16
    tmp25 = tmp22 - tmp24
    tmp27 = tmp18 * tmp26
    tmp28 = tmp25 * tmp27
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp28, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (59584 + r1 + (49*x0) + (70560*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp3 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (59584 + x4 + (70560*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 392
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
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (48608 + r1 + (49*x0) + (70560*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (48608 + x4 + (70560*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 0.002551020408163265
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


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 392
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
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (37632 + r1 + (49*x0) + (70560*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (37632 + x4 + (70560*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 0.002551020408163265
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


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 392
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
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
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
    tmp9 = 0.002551020408163265
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


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 37632
    x1 = (xindex // 37632)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (70560*x1)), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp54 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r2 = (rindex // 14)
        r1 = rindex % 14
        tmp52 = tl.load(in_ptr2 + (r3 + (196*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp0 = r3
        tmp1 = (((-1) + r2) // 2)
        tmp2 = (((-1) + r1) // 2)
        tmp3 = 1 + (r2 // 2)
        tmp4 = 1 + (r1 // 2)
        tmp5 = 0
        tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp5, tmp1, tmp5))
        tmp7 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp5, tmp2, tmp5))
        tmp8 = 7
        tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp8, tmp3, tmp8))
        tmp10 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 < tmp8, tmp4, tmp8))
        tmp11 = tmp6 + tmp5
        tmp12 = tmp7 + tmp5
        tmp13 = 1
        tmp14 = tmp9 - tmp13
        tmp15 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp14, tmp11, tmp14))
        tmp16 = tmp10 - tmp13
        tmp17 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 < tmp16, tmp12, tmp16))
        tmp18 = tl.load(in_ptr0 + (tmp17 + (7*tmp15) + (49*x0)), xmask)
        tmp19 = tl.load(in_ptr1 + (tmp17 + (7*tmp15) + (49*x0)), xmask)
        tmp20 = tmp18 == tmp0
        tmp21 = 0.0
        tmp22 = tl.where(tmp20, tmp19, tmp21)
        tmp23 = tmp7 + tmp13
        tmp24 = tl.where(tmp23 != tmp23, tmp23, tl.where(tmp23 < tmp16, tmp23, tmp16))
        tmp25 = tl.load(in_ptr0 + (tmp24 + (7*tmp15) + (49*x0)), xmask)
        tmp26 = tl.load(in_ptr1 + (tmp24 + (7*tmp15) + (49*x0)), xmask)
        tmp27 = tmp25 == tmp0
        tmp28 = tmp11 < tmp9
        tmp29 = tmp23 < tmp10
        tmp30 = tmp28 & tmp29
        tmp31 = tmp30 & tmp27
        tmp32 = tmp22 + tmp26
        tmp33 = tl.where(tmp31, tmp32, tmp22)
        tmp34 = tmp6 + tmp13
        tmp35 = tl.where(tmp34 != tmp34, tmp34, tl.where(tmp34 < tmp14, tmp34, tmp14))
        tmp36 = tl.load(in_ptr0 + (tmp17 + (7*tmp35) + (49*x0)), xmask)
        tmp37 = tl.load(in_ptr1 + (tmp17 + (7*tmp35) + (49*x0)), xmask)
        tmp38 = tmp36 == tmp0
        tmp39 = tmp34 < tmp9
        tmp40 = tmp12 < tmp10
        tmp41 = tmp39 & tmp40
        tmp42 = tmp41 & tmp38
        tmp43 = tmp33 + tmp37
        tmp44 = tl.where(tmp42, tmp43, tmp33)
        tmp45 = tl.load(in_ptr0 + (tmp24 + (7*tmp35) + (49*x0)), xmask)
        tmp46 = tl.load(in_ptr1 + (tmp24 + (7*tmp35) + (49*x0)), xmask)
        tmp47 = tmp45 == tmp0
        tmp48 = tmp39 & tmp29
        tmp49 = tmp48 & tmp47
        tmp50 = tmp44 + tmp46
        tmp51 = tl.where(tmp49, tmp50, tmp44)
        tmp53 = tmp51 * tmp52
        _tmp54 = tl.where(rmask & xmask, _tmp54 + tmp53, _tmp54)
        tl.store(out_ptr0 + (r3 + (196*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp51, rmask & xmask)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tmp55 = tl.load(in_ptr3 + (x0), xmask)
    tmp56 = 0.16666666666666666
    tmp57 = tmp54 * tmp56
    tmp58 = 0.0
    tmp59 = tl.where(tmp55, tmp57, tmp58)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp59, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__15(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr4 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 196.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp10 * tmp14
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp11, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp16, xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask)
    tmp6 = tl.load(in_ptr2 + (x4), xmask)
    tmp11 = tl.load(in_ptr3 + (x3), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp25 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 196.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.0006377551020408163
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp27, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (175616 + r1 + (196*x0) + (213248*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp3 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (175616 + x4 + (213248*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.0006377551020408163
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (137984 + r1 + (196*x0) + (213248*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
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

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (137984 + x4 + (213248*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 0.0006377551020408163
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


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (100352 + r1 + (196*x0) + (213248*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (100352 + x4 + (213248*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 0.0006377551020408163
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


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 2048],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
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
    tmp9 = 0.0006377551020408163
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


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (213248*x1)), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp54 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r2 = (rindex // 28)
        r1 = rindex % 28
        tmp52 = tl.load(in_ptr2 + (r3 + (784*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp0 = r3
        tmp1 = (((-1) + r2) // 2)
        tmp2 = (((-1) + r1) // 2)
        tmp3 = 1 + (r2 // 2)
        tmp4 = 1 + (r1 // 2)
        tmp5 = 0
        tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp5, tmp1, tmp5))
        tmp7 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp5, tmp2, tmp5))
        tmp8 = 14
        tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp8, tmp3, tmp8))
        tmp10 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 < tmp8, tmp4, tmp8))
        tmp11 = tmp6 + tmp5
        tmp12 = tmp7 + tmp5
        tmp13 = 1
        tmp14 = tmp9 - tmp13
        tmp15 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp14, tmp11, tmp14))
        tmp16 = tmp10 - tmp13
        tmp17 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 < tmp16, tmp12, tmp16))
        tmp18 = tl.load(in_ptr0 + (tmp17 + (14*tmp15) + (196*x0)), xmask)
        tmp19 = tl.load(in_ptr1 + (tmp17 + (14*tmp15) + (196*x0)), xmask)
        tmp20 = tmp18 == tmp0
        tmp21 = 0.0
        tmp22 = tl.where(tmp20, tmp19, tmp21)
        tmp23 = tmp7 + tmp13
        tmp24 = tl.where(tmp23 != tmp23, tmp23, tl.where(tmp23 < tmp16, tmp23, tmp16))
        tmp25 = tl.load(in_ptr0 + (tmp24 + (14*tmp15) + (196*x0)), xmask)
        tmp26 = tl.load(in_ptr1 + (tmp24 + (14*tmp15) + (196*x0)), xmask)
        tmp27 = tmp25 == tmp0
        tmp28 = tmp11 < tmp9
        tmp29 = tmp23 < tmp10
        tmp30 = tmp28 & tmp29
        tmp31 = tmp30 & tmp27
        tmp32 = tmp22 + tmp26
        tmp33 = tl.where(tmp31, tmp32, tmp22)
        tmp34 = tmp6 + tmp13
        tmp35 = tl.where(tmp34 != tmp34, tmp34, tl.where(tmp34 < tmp14, tmp34, tmp14))
        tmp36 = tl.load(in_ptr0 + (tmp17 + (14*tmp35) + (196*x0)), xmask)
        tmp37 = tl.load(in_ptr1 + (tmp17 + (14*tmp35) + (196*x0)), xmask)
        tmp38 = tmp36 == tmp0
        tmp39 = tmp34 < tmp9
        tmp40 = tmp12 < tmp10
        tmp41 = tmp39 & tmp40
        tmp42 = tmp41 & tmp38
        tmp43 = tmp33 + tmp37
        tmp44 = tl.where(tmp42, tmp43, tmp33)
        tmp45 = tl.load(in_ptr0 + (tmp24 + (14*tmp35) + (196*x0)), xmask)
        tmp46 = tl.load(in_ptr1 + (tmp24 + (14*tmp35) + (196*x0)), xmask)
        tmp47 = tmp45 == tmp0
        tmp48 = tmp39 & tmp29
        tmp49 = tmp48 & tmp47
        tmp50 = tmp44 + tmp46
        tmp51 = tl.where(tmp49, tmp50, tmp44)
        tmp53 = tmp51 * tmp52
        _tmp54 = tl.where(rmask & xmask, _tmp54 + tmp53, _tmp54)
        tl.store(out_ptr0 + (r3 + (784*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp51, rmask & xmask)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tmp55 = tl.load(in_ptr3 + (x0), xmask)
    tmp56 = 0.16666666666666666
    tmp57 = tmp54 * tmp56
    tmp58 = 0.0
    tmp59 = tl.where(tmp55, tmp57, tmp58)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp59, xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__28(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr4 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 784.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp10 * tmp14
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp11, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp16, xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask)
    tmp6 = tl.load(in_ptr2 + (x4), xmask)
    tmp11 = tl.load(in_ptr3 + (x3), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp25 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 784.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00015943877551020407
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp27, xmask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (451584 + r1 + (784*x0) + (577024*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp3 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp11, xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (451584 + x4 + (577024*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00015943877551020407
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (326144 + r1 + (784*x0) + (577024*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (326144 + x4 + (577024*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 0.00015943877551020407
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


triton__35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (200704 + r1 + (784*x0) + (577024*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (200704 + x4 + (577024*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 0.00015943877551020407
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


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last')
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


triton__38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 160
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
    tmp9 = 0.00015943877551020407
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


triton__39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__39(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (577024*x1)), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp54 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r2 = (rindex // 56)
        r1 = rindex % 56
        tmp52 = tl.load(in_ptr2 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp0 = r3
        tmp1 = (((-1) + r2) // 2)
        tmp2 = (((-1) + r1) // 2)
        tmp3 = 1 + (r2 // 2)
        tmp4 = 1 + (r1 // 2)
        tmp5 = 0
        tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp5, tmp1, tmp5))
        tmp7 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp5, tmp2, tmp5))
        tmp8 = 28
        tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp8, tmp3, tmp8))
        tmp10 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 < tmp8, tmp4, tmp8))
        tmp11 = tmp6 + tmp5
        tmp12 = tmp7 + tmp5
        tmp13 = 1
        tmp14 = tmp9 - tmp13
        tmp15 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp14, tmp11, tmp14))
        tmp16 = tmp10 - tmp13
        tmp17 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 < tmp16, tmp12, tmp16))
        tmp18 = tl.load(in_ptr0 + (tmp17 + (28*tmp15) + (784*x0)), xmask)
        tmp19 = tl.load(in_ptr1 + (tmp17 + (28*tmp15) + (784*x0)), xmask)
        tmp20 = tmp18 == tmp0
        tmp21 = 0.0
        tmp22 = tl.where(tmp20, tmp19, tmp21)
        tmp23 = tmp7 + tmp13
        tmp24 = tl.where(tmp23 != tmp23, tmp23, tl.where(tmp23 < tmp16, tmp23, tmp16))
        tmp25 = tl.load(in_ptr0 + (tmp24 + (28*tmp15) + (784*x0)), xmask)
        tmp26 = tl.load(in_ptr1 + (tmp24 + (28*tmp15) + (784*x0)), xmask)
        tmp27 = tmp25 == tmp0
        tmp28 = tmp11 < tmp9
        tmp29 = tmp23 < tmp10
        tmp30 = tmp28 & tmp29
        tmp31 = tmp30 & tmp27
        tmp32 = tmp22 + tmp26
        tmp33 = tl.where(tmp31, tmp32, tmp22)
        tmp34 = tmp6 + tmp13
        tmp35 = tl.where(tmp34 != tmp34, tmp34, tl.where(tmp34 < tmp14, tmp34, tmp14))
        tmp36 = tl.load(in_ptr0 + (tmp17 + (28*tmp35) + (784*x0)), xmask)
        tmp37 = tl.load(in_ptr1 + (tmp17 + (28*tmp35) + (784*x0)), xmask)
        tmp38 = tmp36 == tmp0
        tmp39 = tmp34 < tmp9
        tmp40 = tmp12 < tmp10
        tmp41 = tmp39 & tmp40
        tmp42 = tmp41 & tmp38
        tmp43 = tmp33 + tmp37
        tmp44 = tl.where(tmp42, tmp43, tmp33)
        tmp45 = tl.load(in_ptr0 + (tmp24 + (28*tmp35) + (784*x0)), xmask)
        tmp46 = tl.load(in_ptr1 + (tmp24 + (28*tmp35) + (784*x0)), xmask)
        tmp47 = tmp45 == tmp0
        tmp48 = tmp39 & tmp29
        tmp49 = tmp48 & tmp47
        tmp50 = tmp44 + tmp46
        tmp51 = tl.where(tmp49, tmp50, tmp44)
        tmp53 = tmp51 * tmp52
        _tmp54 = tl.where(rmask & xmask, _tmp54 + tmp53, _tmp54)
        tl.store(out_ptr0 + (r3 + (3136*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp51, rmask & xmask)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tmp55 = tl.load(in_ptr3 + (x0), xmask)
    tmp56 = 0.16666666666666666
    tmp57 = tmp54 * tmp56
    tmp58 = 0.0
    tmp59 = tl.where(tmp55, tmp57, tmp58)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp59, xmask)
''')


triton__41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__41(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr4 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 3136.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp10 * tmp14
        _tmp16 = tl.where(rmask & xmask, _tmp16 + tmp15, _tmp16)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp11, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp16, xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp18, xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask)
    tmp6 = tl.load(in_ptr2 + (x4), xmask)
    tmp11 = tl.load(in_ptr3 + (x3), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp25 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 3136.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 3.985969387755102e-05
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp27, xmask)
''')


triton__44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (1003520 + (3136*x0) + (1404928*(r2 // 3136)) + (2809856*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp3 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp9, xmask)
''')


triton__45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__45(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


triton__46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__46(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


triton__47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (1003520 + x4 + (1404928*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp18 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp6 = tmp4 - tmp5
    tmp8 = 3.985969387755102e-05
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (602112 + (3136*x0) + (1404928*(r2 // 3136)) + (2809856*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
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


triton__49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (602112 + x4 + (1404928*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 3.985969387755102e-05
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


triton__50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (200704 + (3136*x0) + (1404928*(r2 // 3136)) + (2809856*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
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


triton__51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (200704 + x4 + (1404928*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 3.985969387755102e-05
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


triton__52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
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


triton__53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
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
    tmp9 = 3.985969387755102e-05
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


triton__54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (1404928*(r2 // 3136)) + (2809856*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
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


triton__55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__55(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__56(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 200704)
    x4 = xindex % 200704
    x1 = (xindex // 3136) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x4 + (1404928*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
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
    tmp11 = 3.985969387755102e-05
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


triton__58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.where(tmp2, tmp7, 0)
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        tmp10 = tl.load(in_ptr0 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp11 = 0.0
        tmp12 = tmp10 <= tmp11
        tmp13 = tl.load(in_ptr1 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp14 = tl.where(tmp12, tmp11, tmp13)
        tmp15 = tl.load(in_ptr2 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp16 = tl.load(in_ptr3 + (x1 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.where(tmp2, tmp18, 0)
        _tmp20 = tl.where(rmask & xmask, _tmp20 + tmp19, _tmp20)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp20, xmask)
''')


triton__59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__59(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 13
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
        tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__60(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 13
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
        tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__61(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
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
    tmp9 = 9.964923469387754e-06
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_48, primals_52, primals_57, primals_62, primals_67, primals_72, primals_77, primals_82, primals_87, primals_92, primals_97, primals_102, primals_107, primals_112, primals_117, primals_122, primals_127, primals_132, primals_137, primals_142, primals_147, primals_152, primals_157, primals_162, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean_8, div, mul_56, getitem, getitem_1, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_14, div_1, mul_92, getitem_2, getitem_3, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_20, div_2, mul_128, getitem_4, getitem_5, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_26, div_3, view, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(tangents_47, permute_1, out=buf0)
        del permute_1
        buf1 = empty_strided((1000, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(tangents_47, (1000, 8), (1, 1000)), view, out=buf1)
        del view
        buf2 = empty_strided((1, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton__0.run(tangents_47, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_47
        buf3 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf4 = as_strided(buf3, (8, 1024, 1, 1), (1024, 1, 1, 1)); del buf3  # reuse
        triton__1.run(buf4, buf0, relu_22, bitwise_and, 8192, 49, grid=grid(8192), stream=stream0)
        del bitwise_and
        buf5 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__2.run(buf4, buf5, 1024, 8, grid=grid(1024), stream=stream0)
        buf6 = aten.convolution_backward(buf4, mean_26, primals_44, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf4
        del mean_26
        del primals_44
        buf7 = buf6[0]
        assert_size_stride(buf7, (8, 1024, 1, 1), (1024, 1, 1, 1))
        buf8 = buf6[1]
        assert_size_stride(buf8, (1024, 1024, 1, 1), (1024, 1, 1, 1))
        del buf6
        buf9 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__3.run(relu_22, buf0, div_3, buf7, convolution_39, unsqueeze_94, squeeze_67, buf9, buf10, buf12, 1024, 392, grid=grid(1024), stream=stream0)
        buf11 = empty_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.float32)
        buf13 = buf11; del buf11  # reuse
        triton__4.run(buf13, relu_22, buf0, div_3, buf7, convolution_39, unsqueeze_94, buf10, squeeze_67, buf9, primals_162, 401408, grid=grid(401408), stream=stream0)
        del buf0
        del buf10
        del buf7
        del convolution_39
        del div_3
        del primals_162
        del relu_22
        del squeeze_67
        del unsqueeze_94
        buf14 = aten.convolution_backward(buf13, cat_3, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf13
        del cat_3
        del primals_43
        buf15 = buf14[0]
        assert_size_stride(buf15, (8, 1440, 7, 7), (70560, 49, 7, 1))
        buf16 = buf14[1]
        assert_size_stride(buf16, (1024, 1440, 1, 1), (1440, 1, 1, 1))
        del buf14
        buf17 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        buf19 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        triton__5.run(le_1, buf15, convolution_38, unsqueeze_106, squeeze_64, buf17, buf18, buf19, 224, 392, grid=grid(224), stream=stream0)
        buf20 = empty_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda', dtype=torch.float32)
        triton__6.run(le_1, buf15, convolution_38, unsqueeze_106, buf18, squeeze_64, buf17, primals_157, buf20, 87808, grid=grid(87808), stream=stream0)
        del convolution_38
        del le_1
        del primals_157
        del squeeze_64
        del unsqueeze_106
        buf21 = aten.convolution_backward(buf20, convolution_37, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf20
        del convolution_37
        del primals_42
        buf22 = buf21[0]
        assert_size_stride(buf22, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf23 = buf21[1]
        assert_size_stride(buf23, (224, 224, 1, 1), (224, 1, 1, 1))
        del buf21
        buf24 = aten.convolution_backward(buf22, relu_20, primals_41, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
        del buf22
        del primals_41
        buf25 = buf24[0]
        assert_size_stride(buf25, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf26 = buf24[1]
        assert_size_stride(buf26, (224, 1, 3, 3), (9, 9, 3, 1))
        del buf24
        buf27 = buf18; del buf18  # reuse
        buf28 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        triton__7.run(relu_20, buf15, buf25, convolution_36, unsqueeze_118, squeeze_61, buf27, buf28, buf30, 224, 392, grid=grid(224), stream=stream0)
        buf29 = buf25; del buf25  # reuse
        triton__8.run(buf29, relu_20, buf15, convolution_36, unsqueeze_118, buf28, squeeze_61, buf27, primals_152, 87808, grid=grid(87808), stream=stream0)
        del convolution_36
        del primals_152
        del relu_20
        del squeeze_61
        del unsqueeze_118
        buf31 = aten.convolution_backward(buf29, convolution_35, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf29
        del convolution_35
        del primals_40
        buf32 = buf31[0]
        assert_size_stride(buf32, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf33 = buf31[1]
        assert_size_stride(buf33, (224, 224, 1, 1), (224, 1, 1, 1))
        del buf31
        buf34 = aten.convolution_backward(buf32, relu_19, primals_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
        del buf32
        del primals_39
        buf35 = buf34[0]
        assert_size_stride(buf35, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf36 = buf34[1]
        assert_size_stride(buf36, (224, 1, 3, 3), (9, 9, 3, 1))
        del buf34
        buf37 = buf28; del buf28  # reuse
        buf38 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        buf40 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        triton__9.run(relu_19, buf15, buf35, convolution_34, unsqueeze_130, squeeze_58, buf37, buf38, buf40, 224, 392, grid=grid(224), stream=stream0)
        buf39 = buf35; del buf35  # reuse
        triton__10.run(buf39, relu_19, buf15, convolution_34, unsqueeze_130, buf38, squeeze_58, buf37, primals_147, 87808, grid=grid(87808), stream=stream0)
        del convolution_34
        del primals_147
        del relu_19
        del squeeze_58
        del unsqueeze_130
        buf41 = aten.convolution_backward(buf39, convolution_33, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf39
        del convolution_33
        del primals_38
        buf42 = buf41[0]
        assert_size_stride(buf42, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf43 = buf41[1]
        assert_size_stride(buf43, (224, 224, 1, 1), (224, 1, 1, 1))
        del buf41
        buf44 = aten.convolution_backward(buf42, relu_18, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
        del buf42
        del primals_37
        buf45 = buf44[0]
        assert_size_stride(buf45, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf46 = buf44[1]
        assert_size_stride(buf46, (224, 1, 3, 3), (9, 9, 3, 1))
        del buf44
        buf47 = buf38; del buf38  # reuse
        buf48 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((224, ), (1, ), device='cuda', dtype=torch.float32)
        triton__11.run(relu_18, buf45, convolution_32, unsqueeze_142, squeeze_55, buf47, buf48, buf49, 224, 392, grid=grid(224), stream=stream0)
        buf50 = buf45; del buf45  # reuse
        triton__12.run(buf50, relu_18, convolution_32, unsqueeze_142, buf48, squeeze_55, buf47, primals_142, 87808, grid=grid(87808), stream=stream0)
        del buf48
        del convolution_32
        del primals_142
        del relu_18
        del squeeze_55
        del unsqueeze_142
        buf51 = aten.convolution_backward(buf50, getitem_4, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf50
        del getitem_4
        del primals_36
        buf52 = buf51[0]
        assert_size_stride(buf52, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf53 = buf51[1]
        assert_size_stride(buf53, (224, 768, 1, 1), (768, 1, 1, 1))
        del buf51
        buf54 = buf52; del buf52  # reuse
        triton__13.run(buf54, buf15, 301056, grid=grid(301056), stream=stream0)
        del buf15
        buf55 = empty_strided((8, 768, 14, 14), (150528, 196, 14, 1), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf57 = as_strided(buf56, (8, 768, 1, 1), (768, 1, 1, 1)); del buf56  # reuse
        triton__14.run(buf57, getitem_5, buf54, relu_17, bitwise_and_1, buf55, 6144, 196, grid=grid(6144), stream=stream0)
        del bitwise_and_1
        del getitem_5
        buf58 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton__15.run(buf57, buf58, 768, 8, grid=grid(768), stream=stream0)
        buf59 = aten.convolution_backward(buf57, mean_20, primals_34, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf57
        del mean_20
        del primals_34
        buf60 = buf59[0]
        assert_size_stride(buf60, (8, 768, 1, 1), (768, 1, 1, 1))
        buf61 = buf59[1]
        assert_size_stride(buf61, (768, 768, 1, 1), (768, 1, 1, 1))
        del buf59
        buf62 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf63 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf65 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton__16.run(relu_17, buf55, div_2, buf60, convolution_30, unsqueeze_154, squeeze_52, buf62, buf63, buf65, 768, 1568, grid=grid(768), stream=stream0)
        buf64 = buf55; del buf55  # reuse
        buf66 = buf64; del buf64  # reuse
        triton__17.run(buf66, relu_17, div_2, buf60, convolution_30, unsqueeze_154, buf63, squeeze_52, buf62, primals_137, 1204224, grid=grid(1204224), stream=stream0)
        del buf60
        del buf63
        del convolution_30
        del div_2
        del primals_137
        del relu_17
        del squeeze_52
        del unsqueeze_154
        buf67 = aten.convolution_backward(buf66, cat_2, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf66
        del cat_2
        del primals_33
        buf68 = buf67[0]
        assert_size_stride(buf68, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf69 = buf67[1]
        assert_size_stride(buf69, (768, 1088, 1, 1), (1088, 1, 1, 1))
        del buf67
        buf70 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        triton__18.run(le_6, buf68, convolution_29, unsqueeze_166, squeeze_49, buf70, buf71, buf72, 192, 1568, grid=grid(192), stream=stream0)
        buf73 = as_strided(buf54, (8, 192, 14, 14), (37632, 196, 14, 1)); del buf54  # reuse
        triton__19.run(le_6, buf68, convolution_29, unsqueeze_166, buf71, squeeze_49, buf70, primals_132, buf73, 301056, grid=grid(301056), stream=stream0)
        del convolution_29
        del le_6
        del primals_132
        del squeeze_49
        del unsqueeze_166
        buf74 = aten.convolution_backward(buf73, convolution_28, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf73
        del convolution_28
        del primals_32
        buf75 = buf74[0]
        assert_size_stride(buf75, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf76 = buf74[1]
        assert_size_stride(buf76, (192, 192, 1, 1), (192, 1, 1, 1))
        del buf74
        buf77 = aten.convolution_backward(buf75, relu_15, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf75
        del primals_31
        buf78 = buf77[0]
        assert_size_stride(buf78, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf79 = buf77[1]
        assert_size_stride(buf79, (192, 1, 3, 3), (9, 9, 3, 1))
        del buf77
        buf80 = buf71; del buf71  # reuse
        buf81 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf83 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        triton__20.run(relu_15, buf68, buf78, convolution_27, unsqueeze_178, squeeze_46, buf80, buf81, buf83, 192, 1568, grid=grid(192), stream=stream0)
        buf82 = buf78; del buf78  # reuse
        triton__21.run(buf82, relu_15, buf68, convolution_27, unsqueeze_178, buf81, squeeze_46, buf80, primals_127, 301056, grid=grid(301056), stream=stream0)
        del convolution_27
        del primals_127
        del relu_15
        del squeeze_46
        del unsqueeze_178
        buf84 = aten.convolution_backward(buf82, convolution_26, primals_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf82
        del convolution_26
        del primals_30
        buf85 = buf84[0]
        assert_size_stride(buf85, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf86 = buf84[1]
        assert_size_stride(buf86, (192, 192, 1, 1), (192, 1, 1, 1))
        del buf84
        buf87 = aten.convolution_backward(buf85, relu_14, primals_29, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf85
        del primals_29
        buf88 = buf87[0]
        assert_size_stride(buf88, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf89 = buf87[1]
        assert_size_stride(buf89, (192, 1, 3, 3), (9, 9, 3, 1))
        del buf87
        buf90 = buf81; del buf81  # reuse
        buf91 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        triton__22.run(relu_14, buf68, buf88, convolution_25, unsqueeze_190, squeeze_43, buf90, buf91, buf93, 192, 1568, grid=grid(192), stream=stream0)
        buf92 = buf88; del buf88  # reuse
        triton__23.run(buf92, relu_14, buf68, convolution_25, unsqueeze_190, buf91, squeeze_43, buf90, primals_122, 301056, grid=grid(301056), stream=stream0)
        del convolution_25
        del primals_122
        del relu_14
        del squeeze_43
        del unsqueeze_190
        buf94 = aten.convolution_backward(buf92, convolution_24, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf92
        del convolution_24
        del primals_28
        buf95 = buf94[0]
        assert_size_stride(buf95, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf96 = buf94[1]
        assert_size_stride(buf96, (192, 192, 1, 1), (192, 1, 1, 1))
        del buf94
        buf97 = aten.convolution_backward(buf95, relu_13, primals_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf95
        del primals_27
        buf98 = buf97[0]
        assert_size_stride(buf98, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf99 = buf97[1]
        assert_size_stride(buf99, (192, 1, 3, 3), (9, 9, 3, 1))
        del buf97
        buf100 = buf91; del buf91  # reuse
        buf101 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((192, ), (1, ), device='cuda', dtype=torch.float32)
        triton__24.run(relu_13, buf98, convolution_23, unsqueeze_202, squeeze_40, buf100, buf101, buf102, 192, 1568, grid=grid(192), stream=stream0)
        buf103 = buf98; del buf98  # reuse
        triton__25.run(buf103, relu_13, convolution_23, unsqueeze_202, buf101, squeeze_40, buf100, primals_117, 301056, grid=grid(301056), stream=stream0)
        del buf101
        del convolution_23
        del primals_117
        del relu_13
        del squeeze_40
        del unsqueeze_202
        buf104 = aten.convolution_backward(buf103, getitem_2, primals_26, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf103
        del getitem_2
        del primals_26
        buf105 = buf104[0]
        assert_size_stride(buf105, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf106 = buf104[1]
        assert_size_stride(buf106, (192, 512, 1, 1), (512, 1, 1, 1))
        del buf104
        buf107 = buf105; del buf105  # reuse
        triton__26.run(buf107, buf68, 802816, grid=grid(802816), stream=stream0)
        del buf68
        buf108 = empty_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf110 = as_strided(buf109, (8, 512, 1, 1), (512, 1, 1, 1)); del buf109  # reuse
        triton__27.run(buf110, getitem_3, buf107, relu_12, bitwise_and_2, buf108, 4096, 784, grid=grid(4096), stream=stream0)
        del bitwise_and_2
        del buf107
        del getitem_3
        buf111 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__28.run(buf110, buf111, 512, 8, grid=grid(512), stream=stream0)
        buf112 = aten.convolution_backward(buf110, mean_14, primals_24, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf110
        del mean_14
        del primals_24
        buf113 = buf112[0]
        assert_size_stride(buf113, (8, 512, 1, 1), (512, 1, 1, 1))
        buf114 = buf112[1]
        assert_size_stride(buf114, (512, 512, 1, 1), (512, 1, 1, 1))
        del buf112
        buf115 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf116 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__29.run(relu_12, buf108, div_1, buf113, convolution_21, unsqueeze_214, squeeze_37, buf115, buf116, buf118, 512, 6272, grid=grid(512), stream=stream0)
        buf117 = buf108; del buf108  # reuse
        buf119 = buf117; del buf117  # reuse
        triton__30.run(buf119, relu_12, div_1, buf113, convolution_21, unsqueeze_214, buf116, squeeze_37, buf115, primals_112, 3211264, grid=grid(3211264), stream=stream0)
        del buf113
        del convolution_21
        del div_1
        del primals_112
        del relu_12
        del squeeze_37
        del unsqueeze_214
        buf120 = aten.convolution_backward(buf119, cat_1, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_23
        buf121 = buf120[0]
        assert_size_stride(buf121, (8, 736, 28, 28), (577024, 784, 28, 1))
        buf122 = buf120[1]
        assert_size_stride(buf122, (512, 736, 1, 1), (736, 1, 1, 1))
        del buf120
        buf123 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf124 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf125 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        triton__31.run(le_11, buf121, convolution_20, unsqueeze_226, squeeze_34, buf123, buf124, buf125, 160, 6272, grid=grid(160), stream=stream0)
        buf126 = empty_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda', dtype=torch.float32)
        triton__32.run(le_11, buf121, convolution_20, unsqueeze_226, buf124, squeeze_34, buf123, primals_107, buf126, 1003520, grid=grid(1003520), stream=stream0)
        del convolution_20
        del le_11
        del primals_107
        del squeeze_34
        del unsqueeze_226
        buf127 = aten.convolution_backward(buf126, convolution_19, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf126
        del convolution_19
        del primals_22
        buf128 = buf127[0]
        assert_size_stride(buf128, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf129 = buf127[1]
        assert_size_stride(buf129, (160, 160, 1, 1), (160, 1, 1, 1))
        del buf127
        buf130 = aten.convolution_backward(buf128, relu_10, primals_21, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
        del buf128
        del primals_21
        buf131 = buf130[0]
        assert_size_stride(buf131, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf132 = buf130[1]
        assert_size_stride(buf132, (160, 1, 3, 3), (9, 9, 3, 1))
        del buf130
        buf133 = buf124; del buf124  # reuse
        buf134 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf136 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        triton__33.run(relu_10, buf121, buf131, convolution_18, unsqueeze_238, squeeze_31, buf133, buf134, buf136, 160, 6272, grid=grid(160), stream=stream0)
        buf135 = buf131; del buf131  # reuse
        triton__34.run(buf135, relu_10, buf121, convolution_18, unsqueeze_238, buf134, squeeze_31, buf133, primals_102, 1003520, grid=grid(1003520), stream=stream0)
        del convolution_18
        del primals_102
        del relu_10
        del squeeze_31
        del unsqueeze_238
        buf137 = aten.convolution_backward(buf135, convolution_17, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf135
        del convolution_17
        del primals_20
        buf138 = buf137[0]
        assert_size_stride(buf138, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf139 = buf137[1]
        assert_size_stride(buf139, (160, 160, 1, 1), (160, 1, 1, 1))
        del buf137
        buf140 = aten.convolution_backward(buf138, relu_9, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
        del buf138
        del primals_19
        buf141 = buf140[0]
        assert_size_stride(buf141, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf142 = buf140[1]
        assert_size_stride(buf142, (160, 1, 3, 3), (9, 9, 3, 1))
        del buf140
        buf143 = buf134; del buf134  # reuse
        buf144 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf146 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(relu_9, buf121, buf141, convolution_16, unsqueeze_250, squeeze_28, buf143, buf144, buf146, 160, 6272, grid=grid(160), stream=stream0)
        buf145 = buf141; del buf141  # reuse
        triton__36.run(buf145, relu_9, buf121, convolution_16, unsqueeze_250, buf144, squeeze_28, buf143, primals_97, 1003520, grid=grid(1003520), stream=stream0)
        del convolution_16
        del primals_97
        del relu_9
        del squeeze_28
        del unsqueeze_250
        buf147 = aten.convolution_backward(buf145, convolution_15, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf145
        del convolution_15
        del primals_18
        buf148 = buf147[0]
        assert_size_stride(buf148, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf149 = buf147[1]
        assert_size_stride(buf149, (160, 160, 1, 1), (160, 1, 1, 1))
        del buf147
        buf150 = aten.convolution_backward(buf148, relu_8, primals_17, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
        del buf148
        del primals_17
        buf151 = buf150[0]
        assert_size_stride(buf151, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf152 = buf150[1]
        assert_size_stride(buf152, (160, 1, 3, 3), (9, 9, 3, 1))
        del buf150
        buf153 = buf144; del buf144  # reuse
        buf154 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        buf155 = empty_strided((160, ), (1, ), device='cuda', dtype=torch.float32)
        triton__37.run(relu_8, buf151, convolution_14, unsqueeze_262, squeeze_25, buf153, buf154, buf155, 160, 6272, grid=grid(160), stream=stream0)
        buf156 = buf151; del buf151  # reuse
        triton__38.run(buf156, relu_8, convolution_14, unsqueeze_262, buf154, squeeze_25, buf153, primals_92, 1003520, grid=grid(1003520), stream=stream0)
        del buf154
        del convolution_14
        del primals_92
        del relu_8
        del squeeze_25
        del unsqueeze_262
        buf157 = aten.convolution_backward(buf156, getitem, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf156
        del getitem
        del primals_16
        buf158 = buf157[0]
        assert_size_stride(buf158, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf159 = buf157[1]
        assert_size_stride(buf159, (160, 256, 1, 1), (256, 1, 1, 1))
        del buf157
        buf160 = buf158; del buf158  # reuse
        triton__39.run(buf160, buf121, 1605632, grid=grid(1605632), stream=stream0)
        del buf121
        buf161 = empty_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda', dtype=torch.float32)
        buf162 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf163 = as_strided(buf162, (8, 256, 1, 1), (256, 1, 1, 1)); del buf162  # reuse
        triton__40.run(buf163, getitem_1, buf160, relu_7, bitwise_and_3, buf161, 2048, 3136, grid=grid(2048), stream=stream0)
        del bitwise_and_3
        del buf160
        del getitem_1
        buf164 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__41.run(buf163, buf164, 256, 8, grid=grid(256), stream=stream0)
        buf165 = aten.convolution_backward(buf163, mean_8, primals_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf163
        del mean_8
        del primals_14
        buf166 = buf165[0]
        assert_size_stride(buf166, (8, 256, 1, 1), (256, 1, 1, 1))
        buf167 = buf165[1]
        assert_size_stride(buf167, (256, 256, 1, 1), (256, 1, 1, 1))
        del buf165
        buf168 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf169 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf171 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__42.run(relu_7, buf161, div, buf166, convolution_12, unsqueeze_274, squeeze_22, buf168, buf169, buf171, 256, 25088, grid=grid(256), stream=stream0)
        buf170 = buf161; del buf161  # reuse
        buf172 = buf170; del buf170  # reuse
        triton__43.run(buf172, relu_7, div, buf166, convolution_12, unsqueeze_274, buf169, squeeze_22, buf168, primals_87, 6422528, grid=grid(6422528), stream=stream0)
        del buf166
        del convolution_12
        del div
        del primals_87
        del relu_7
        del squeeze_22
        del unsqueeze_274
        buf173 = aten.convolution_backward(buf172, cat, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf172
        del cat
        del primals_13
        buf174 = buf173[0]
        assert_size_stride(buf174, (8, 448, 56, 56), (1404928, 3136, 56, 1))
        buf175 = buf173[1]
        assert_size_stride(buf175, (256, 448, 1, 1), (448, 1, 1, 1))
        del buf173
        buf176 = as_strided(buf116, (128, 4), (1, 128)); del buf116  # reuse
        buf178 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        triton__44.run(le_16, buf174, convolution_11, unsqueeze_286, buf176, buf178, 512, 6272, grid=grid(512), stream=stream0)
        buf177 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__45.run(buf176, buf177, 128, 4, grid=grid(128), stream=stream0)
        buf179 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__46.run(buf178, squeeze_19, buf179, buf180, 128, 4, grid=grid(128), stream=stream0)
        buf181 = as_strided(buf119, (8, 128, 56, 56), (401408, 3136, 56, 1)); del buf119  # reuse
        triton__47.run(le_16, buf174, convolution_11, unsqueeze_286, buf179, squeeze_19, buf177, primals_82, buf181, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_11
        del le_16
        del primals_82
        del squeeze_19
        del unsqueeze_286
        buf182 = aten.convolution_backward(buf181, convolution_10, primals_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf181
        del convolution_10
        del primals_12
        buf183 = buf182[0]
        assert_size_stride(buf183, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf184 = buf182[1]
        assert_size_stride(buf184, (128, 128, 1, 1), (128, 1, 1, 1))
        del buf182
        buf185 = aten.convolution_backward(buf183, relu_5, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del buf183
        del primals_11
        buf186 = buf185[0]
        assert_size_stride(buf186, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf187 = buf185[1]
        assert_size_stride(buf187, (128, 1, 3, 3), (9, 9, 3, 1))
        del buf185
        buf188 = buf178; del buf178  # reuse
        buf190 = buf176; del buf176  # reuse
        triton__48.run(relu_5, buf174, buf186, convolution_9, unsqueeze_298, buf188, buf190, 512, 6272, grid=grid(512), stream=stream0)
        buf189 = buf179; del buf179  # reuse
        triton__45.run(buf188, buf189, 128, 4, grid=grid(128), stream=stream0)
        buf191 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf193 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__46.run(buf190, squeeze_16, buf191, buf193, 128, 4, grid=grid(128), stream=stream0)
        buf192 = buf186; del buf186  # reuse
        triton__49.run(buf192, relu_5, buf174, convolution_9, unsqueeze_298, buf191, squeeze_16, buf189, primals_77, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_9
        del primals_77
        del relu_5
        del squeeze_16
        del unsqueeze_298
        buf194 = aten.convolution_backward(buf192, convolution_8, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf192
        del convolution_8
        del primals_10
        buf195 = buf194[0]
        assert_size_stride(buf195, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf196 = buf194[1]
        assert_size_stride(buf196, (128, 128, 1, 1), (128, 1, 1, 1))
        del buf194
        buf197 = aten.convolution_backward(buf195, relu_4, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del buf195
        del primals_9
        buf198 = buf197[0]
        assert_size_stride(buf198, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf199 = buf197[1]
        assert_size_stride(buf199, (128, 1, 3, 3), (9, 9, 3, 1))
        del buf197
        buf200 = buf190; del buf190  # reuse
        buf202 = buf188; del buf188  # reuse
        triton__50.run(relu_4, buf174, buf198, convolution_7, unsqueeze_310, buf200, buf202, 512, 6272, grid=grid(512), stream=stream0)
        buf201 = buf191; del buf191  # reuse
        triton__45.run(buf200, buf201, 128, 4, grid=grid(128), stream=stream0)
        buf203 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__46.run(buf202, squeeze_13, buf203, buf205, 128, 4, grid=grid(128), stream=stream0)
        buf204 = buf198; del buf198  # reuse
        triton__51.run(buf204, relu_4, buf174, convolution_7, unsqueeze_310, buf203, squeeze_13, buf201, primals_72, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_7
        del primals_72
        del relu_4
        del squeeze_13
        del unsqueeze_310
        buf206 = aten.convolution_backward(buf204, convolution_6, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf204
        del convolution_6
        del primals_8
        buf207 = buf206[0]
        assert_size_stride(buf207, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf208 = buf206[1]
        assert_size_stride(buf208, (128, 128, 1, 1), (128, 1, 1, 1))
        del buf206
        buf209 = aten.convolution_backward(buf207, relu_3, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del buf207
        del primals_7
        buf210 = buf209[0]
        assert_size_stride(buf210, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf211 = buf209[1]
        assert_size_stride(buf211, (128, 1, 3, 3), (9, 9, 3, 1))
        del buf209
        buf212 = buf202; del buf202  # reuse
        buf214 = buf200; del buf200  # reuse
        triton__52.run(relu_3, buf210, convolution_5, unsqueeze_322, buf212, buf214, 512, 6272, grid=grid(512), stream=stream0)
        buf213 = buf203; del buf203  # reuse
        triton__45.run(buf212, buf213, 128, 4, grid=grid(128), stream=stream0)
        del buf212
        buf215 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf216 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__46.run(buf214, squeeze_10, buf215, buf216, 128, 4, grid=grid(128), stream=stream0)
        del buf214
        buf217 = buf210; del buf210  # reuse
        triton__53.run(buf217, relu_3, convolution_5, unsqueeze_322, buf215, squeeze_10, buf213, primals_67, 3211264, grid=grid(3211264), stream=stream0)
        del buf215
        del convolution_5
        del primals_67
        del relu_3
        del squeeze_10
        del unsqueeze_322
        buf218 = aten.convolution_backward(buf217, relu_2, primals_6, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf217
        del primals_6
        buf219 = buf218[0]
        assert_size_stride(buf219, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf220 = buf218[1]
        assert_size_stride(buf220, (128, 64, 1, 1), (64, 1, 1, 1))
        del buf218
        buf221 = as_strided(buf169, (64, 4), (1, 64)); del buf169  # reuse
        buf223 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        triton__54.run(relu_2, buf174, buf219, convolution_4, unsqueeze_334, buf221, buf223, 256, 6272, grid=grid(256), stream=stream0)
        buf222 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__55.run(buf221, buf222, 64, 4, grid=grid(64), stream=stream0)
        del buf221
        buf224 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf226 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__56.run(buf223, squeeze_7, buf224, buf226, 64, 4, grid=grid(64), stream=stream0)
        del buf223
        buf225 = buf219; del buf219  # reuse
        triton__57.run(buf225, relu_2, buf174, convolution_4, unsqueeze_334, buf224, squeeze_7, buf222, primals_62, 1605632, grid=grid(1605632), stream=stream0)
        del buf174
        del convolution_4
        del primals_62
        del relu_2
        del squeeze_7
        del unsqueeze_334
        buf227 = aten.convolution_backward(buf225, convolution_3, primals_5, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf225
        del convolution_3
        del primals_5
        buf228 = buf227[0]
        assert_size_stride(buf228, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf229 = buf227[1]
        assert_size_stride(buf229, (64, 64, 1, 1), (64, 1, 1, 1))
        del buf227
        buf230 = aten.convolution_backward(buf228, relu_1, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del buf228
        del primals_4
        buf231 = buf230[0]
        assert_size_stride(buf231, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf232 = buf230[1]
        assert_size_stride(buf232, (64, 1, 3, 3), (9, 9, 3, 1))
        del buf230
        buf233 = empty_strided((64, 13), (13, 1), device='cuda', dtype=torch.float32)
        buf235 = empty_strided((64, 13), (13, 1), device='cuda', dtype=torch.float32)
        triton__58.run(relu_1, buf231, convolution_2, unsqueeze_346, buf233, buf235, 832, 7720, grid=grid(832), stream=stream0)
        buf234 = buf224; del buf224  # reuse
        triton__59.run(buf233, buf234, 64, 13, grid=grid(64), stream=stream0)
        buf236 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf237 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__60.run(buf235, squeeze_4, buf236, buf237, 64, 13, grid=grid(64), stream=stream0)
        buf238 = buf231; del buf231  # reuse
        triton__61.run(buf238, relu_1, convolution_2, unsqueeze_346, buf236, squeeze_4, buf234, primals_57, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_2
        del primals_57
        del relu_1
        del squeeze_4
        del unsqueeze_346
        buf239 = aten.convolution_backward(buf238, convolution_1, primals_3, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf238
        del convolution_1
        del primals_3
        buf240 = buf239[0]
        assert_size_stride(buf240, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf241 = buf239[1]
        assert_size_stride(buf241, (64, 64, 1, 1), (64, 1, 1, 1))
        del buf239
        buf242 = aten.convolution_backward(buf240, relu, primals_2, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del buf240
        del primals_2
        buf243 = buf242[0]
        assert_size_stride(buf243, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf244 = buf242[1]
        assert_size_stride(buf244, (64, 1, 3, 3), (9, 9, 3, 1))
        del buf242
        buf245 = buf235; del buf235  # reuse
        buf247 = buf233; del buf233  # reuse
        triton__58.run(relu, buf243, convolution, unsqueeze_358, buf245, buf247, 832, 7720, grid=grid(832), stream=stream0)
        buf246 = buf236; del buf236  # reuse
        triton__59.run(buf245, buf246, 64, 13, grid=grid(64), stream=stream0)
        del buf245
        buf248 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf249 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__60.run(buf247, squeeze_1, buf248, buf249, 64, 13, grid=grid(64), stream=stream0)
        del buf247
        buf250 = buf243; del buf243  # reuse
        triton__61.run(buf250, relu, convolution, unsqueeze_358, buf248, squeeze_1, buf246, primals_52, 6422528, grid=grid(6422528), stream=stream0)
        del buf248
        del convolution
        del primals_52
        del relu
        del squeeze_1
        del unsqueeze_358
        buf251 = aten.convolution_backward(buf250, primals_48, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf250
        del primals_1
        del primals_48
        buf252 = buf251[1]
        assert_size_stride(buf252, (64, 3, 3, 3), (27, 9, 3, 1))
        del buf251
        return (buf252, buf244, buf241, buf232, buf229, buf220, buf211, buf208, buf199, buf196, buf187, buf184, buf175, buf167, buf164, buf159, buf152, buf149, buf142, buf139, buf132, buf129, buf122, buf114, buf111, buf106, buf99, buf96, buf89, buf86, buf79, buf76, buf69, buf61, buf58, buf53, buf46, buf43, buf36, buf33, buf26, buf23, buf16, buf8, buf5, as_strided(buf1, (1000, 1024), (1024, 1)), as_strided(buf2, (1000, ), (1, )), None, None, None, None, buf249, buf246, None, None, None, buf237, buf234, None, None, None, buf226, buf222, None, None, None, buf216, buf213, None, None, None, buf205, buf201, None, None, None, buf193, buf189, None, None, None, buf180, buf177, None, None, None, buf171, buf168, None, None, None, buf155, buf153, None, None, None, buf146, buf143, None, None, None, buf136, buf133, None, None, None, buf125, buf123, None, None, None, buf118, buf115, None, None, None, buf102, buf100, None, None, None, buf93, buf90, None, None, None, buf83, buf80, None, None, None, buf72, buf70, None, None, None, buf65, buf62, None, None, None, buf49, buf47, None, None, None, buf40, buf37, None, None, None, buf30, buf27, None, None, None, buf19, buf17, None, None, None, buf12, buf9, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((160, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((224, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1024, 1440, 1, 1), (1440, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 448, 56, 56), (1404928, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.int64)
    convolution_14 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 736, 28, 28), (577024, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_14 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_92 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.int64)
    convolution_23 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 1088, 14, 14), (213248, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 768, 14, 14), (150528, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 768, 14, 14), (150528, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_20 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((8, 768, 14, 14), (150528, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.int64)
    convolution_32 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 1440, 7, 7), (70560, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_26 = rand_strided((8, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and = rand_strided((8, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_94 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((8, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_106 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_118 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_142 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_154 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((8, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_166 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_214 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((8, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_226 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_274 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_286 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_37 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_38 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_39 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_40 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_41 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_42 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_43 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_44 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_45 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_46 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_47 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    tangents_48 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_49 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_50 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_52 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_53 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_54 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_55 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_56 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_57 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_58 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_59 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_60 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_61 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_64 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_65 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_67 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_68 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_69 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_70 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_48, primals_52, primals_57, primals_62, primals_67, primals_72, primals_77, primals_82, primals_87, primals_92, primals_97, primals_102, primals_107, primals_112, primals_117, primals_122, primals_127, primals_132, primals_137, primals_142, primals_147, primals_152, primals_157, primals_162, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean_8, div, mul_56, getitem, getitem_1, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_14, div_1, mul_92, getitem_2, getitem_3, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_20, div_2, mul_128, getitem_4, getitem_5, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_26, div_3, view, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70]))
