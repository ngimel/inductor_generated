
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

@reduction(size_hints=[512, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp5 = tl.load(in_ptr4 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.load(in_ptr1 + (r1 + (128*tmp0)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr3 + (r1 + (128*tmp2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tmp1 + tmp3
        tmp6 = tl.load(in_ptr5 + (r1 + (128*tmp5)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tmp4 + tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
        tl.store(out_ptr0 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp7, rmask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp9 = 128.0
    tmp10 = tmp8 / tmp9
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tmp11 - tmp10
        tmp13 = tmp12 * tmp12
        _tmp14 = tl.where(rmask & xmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(out_ptr0 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp23 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last')
        tmp25 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last')
        tmp16 = tmp15 - tmp10
        tmp17 = 128.0
        tmp18 = tmp14 / tmp17
        tmp19 = 1e-12
        tmp20 = tmp18 + tmp19
        tmp21 = tl.libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 + tmp25
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp22, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, rmask & xmask)
    tmp27 = 128.0
    tmp28 = tmp14 / tmp27
    tmp29 = 1e-12
    tmp30 = tmp28 + tmp29
    tmp31 = tl.libdevice.rsqrt(tmp30)
    tmp32 = tmp31 / tmp27
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp32, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32768, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3 - tmp3
        tmp5 = -10000.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp2 + tmp6
        _tmp8 = tl.where(rmask & xmask & (_tmp8 < tmp7), tmp7, _tmp8)
    tmp8 = tl.max(_tmp8, 1)[:, None]
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = 8.0
        tmp11 = tmp9 / tmp10
        tmp12 = 1.0
        tmp13 = tmp12 - tmp12
        tmp14 = -10000.0
        tmp15 = tmp13 * tmp14
        tmp16 = tmp11 + tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = tl.exp(tmp17)
        _tmp19 = tl.where(rmask & xmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp21 = 8.0
        tmp22 = tmp20 / tmp21
        tmp23 = 1.0
        tmp24 = tmp23 - tmp23
        tmp25 = -10000.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp22 + tmp26
        tmp28 = tmp27 - tmp8
        tmp29 = tl.exp(tmp28)
        tmp30 = tmp29 / tmp19
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp30, rmask & xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (32768*(x0 // 64)) + (x0 % 64)), xmask)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp4 = 4096.0
    tmp5 = tmp3 / tmp4
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8 - tmp5
        tmp10 = tmp9 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp24 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14 - tmp5
        tmp16 = 4096.0
        tmp17 = tmp11 / tmp16
        tmp18 = 1e-12
        tmp19 = tmp17 + tmp18
        tmp20 = tl.libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 + tmp24
        tl.store(out_ptr1 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp21, rmask & xmask)
        tl.store(out_ptr2 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp25, rmask & xmask)
    tmp26 = 4096.0
    tmp27 = tmp11 / tmp26
    tmp28 = 1e-12
    tmp29 = tmp27 + tmp28
    tmp30 = tl.libdevice.rsqrt(tmp29)
    tmp31 = tmp30 / tmp26
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp31, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = tl.libdevice.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp8, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp3 = tmp1 * tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = tmp0 + tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp8 = 4096.0
    tmp9 = tmp7 / tmp8
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp10 + tmp15
        tmp17 = tmp16 - tmp9
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(rmask & xmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp21 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp24 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last')
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 + tmp24
        tmp26 = tmp20 + tmp25
        tmp27 = tmp26 - tmp9
        tmp28 = 4096.0
        tmp29 = tmp19 / tmp28
        tmp30 = 1e-12
        tmp31 = tmp29 + tmp30
        tmp32 = tl.libdevice.rsqrt(tmp31)
        tmp33 = tmp27 * tmp32
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 + tmp36
        tl.store(out_ptr1 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
        tl.store(out_ptr2 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38 = 4096.0
    tmp39 = tmp19 / tmp38
    tmp40 = 1e-12
    tmp41 = tmp39 + tmp40
    tmp42 = tl.libdevice.rsqrt(tmp41)
    tmp43 = tmp42 / tmp38
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp43, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp8 = 4096.0
    tmp9 = tmp7 / tmp8
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16 - tmp9
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(rmask & xmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp21 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp23 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp25 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last')
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26 - tmp9
        tmp28 = 4096.0
        tmp29 = tmp19 / tmp28
        tmp30 = 1e-12
        tmp31 = tmp29 + tmp30
        tmp32 = tl.libdevice.rsqrt(tmp31)
        tmp33 = tmp27 * tmp32
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 + tmp36
        tl.store(out_ptr1 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
        tl.store(out_ptr2 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38 = 4096.0
    tmp39 = tmp19 / tmp38
    tmp40 = 1e-12
    tmp41 = tmp39 + tmp40
    tmp42 = tl.libdevice.rsqrt(tmp41)
    tmp43 = tmp42 / tmp38
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp43, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (2*r0), rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & (_tmp1 < tmp0), tmp0, _tmp1)
        tl.store(out_ptr0 + (r0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp0, rmask)
    tmp1 = tl.max(_tmp1, 1)[:, None]
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp2 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp3 = tmp2 - tmp1
        tmp4 = tl.exp(tmp3)
        _tmp5 = tl.where(rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp6 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp7 = tmp6 - tmp1
        tmp8 = tl.log(tmp5)
        tmp9 = tmp7 - tmp8
        tl.store(out_ptr3 + (r0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp9, rmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + (2*r0)), rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & (_tmp1 < tmp0), tmp0, _tmp1)
        tl.store(out_ptr0 + (r0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp0, rmask)
    tmp1 = tl.max(_tmp1, 1)[:, None]
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp2 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp3 = tmp2 - tmp1
        tmp4 = tl.exp(tmp3)
        _tmp5 = tl.where(rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp6 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp7 = tmp6 - tmp1
        tmp8 = tl.log(tmp5)
        tmp9 = tmp7 - tmp8
        tl.store(out_ptr3 + (r0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp9, rmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i1', 6: '*i64', 7: '*i1', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK])
    tmp6_load = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp6_load, [XBLOCK])
    tmp1 = 0
    tmp2 = tl.where(tmp0 != tmp0, tmp0, tl.where(tmp0 > tmp1, tmp0, tmp1))
    tmp3 = 512
    tmp4 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 < tmp3, tmp2, tmp3))
    tmp5 = tmp4 != tmp3
    tmp7 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 > tmp1, tmp6, tmp1))
    tmp8 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 < tmp3, tmp7, tmp3))
    tmp9 = tmp8 != tmp3
    tmp10 = tl.load(in_ptr2 + (tmp4), None)
    tmp11 = -tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp5, tmp11, tmp12)
    tmp14 = tmp5.to(tl.int64)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tl.load(in_ptr3 + (tmp8), None)
    tmp18 = -tmp17
    tmp19 = tl.where(tmp9, tmp18, tmp12)
    tmp20 = tmp9.to(tl.int64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp16 + tmp22
    tmp24 = 2.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp4, None)
    tl.store(out_ptr1 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp5, None)
    tl.store(out_ptr2 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp8, None)
    tl.store(out_ptr3 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp9, None)
    tl.store(out_ptr4 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp25, None)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__10(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 512, 128), (65536, 128, 1), device='cuda', dtype=torch.float32)
        buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf2 = buf1; del buf1  # reuse
        buf4 = empty_strided((1, 512, 128), (65536, 128, 1), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((512, 128), (128, 1), device='cuda', dtype=torch.float32)
        buf368 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton__0.run(buf2, primals_28, primals_1, primals_26, primals_2, primals_27, primals_3, primals_4, primals_5, buf0, buf4, buf5, buf368, 512, 128, grid=grid(512), stream=stream0)
        del buf0
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        buf6 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_7, buf5, as_strided(primals_6, (128, 4096), (1, 128)), alpha=1, beta=1, out=buf6)
        del primals_7
        buf7 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_9, as_strided(buf6, (512, 4096), (4096, 1)), as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf7)
        buf8 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, as_strided(buf6, (512, 4096), (4096, 1)), as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf8)
        buf9 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, as_strided(buf6, (512, 4096), (4096, 1)), as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf9)
        buf10 = empty_strided((64, 512, 512), (262144, 512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf7, (64, 512, 64), (64, 4096, 1)), as_strided(buf8, (64, 64, 512), (64, 1, 4096)), out=buf10)
        buf13 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf10, buf13, 32768, 512, grid=grid(32768), stream=stream0)
        buf14 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf13, (64, 512, 512), (262144, 512, 1)), as_strided(buf9, (64, 512, 64), (64, 4096, 1)), out=buf14)
        buf15 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf14, buf15, 2097152, grid=grid(2097152), stream=stream0)
        buf16 = as_strided(buf14, (512, 4096), (4096, 1)); del buf14  # reuse
        extern_kernels.addmm(primals_15, buf15, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf16)
        buf17 = buf2; del buf2  # reuse
        buf18 = buf17; del buf17  # reuse
        buf20 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf364 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__3.run(buf18, buf6, buf16, primals_16, primals_17, buf20, buf21, buf364, 512, 4096, grid=grid(512), stream=stream0)
        buf22 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf21, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf22)
        buf23 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf24 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf22, buf23, buf24, 8388608, grid=grid(8388608), stream=stream0)
        buf25 = buf16; del buf16  # reuse
        extern_kernels.addmm(primals_21, buf24, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf25)
        buf26 = buf18; del buf18  # reuse
        buf27 = buf26; del buf26  # reuse
        buf29 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf363 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf27, buf25, buf20, primals_16, primals_17, primals_22, primals_23, buf29, buf30, buf363, 512, 4096, grid=grid(512), stream=stream0)
        buf31 = buf25; del buf25  # reuse
        extern_kernels.addmm(primals_9, buf30, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf31)
        buf32 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf30, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf32)
        buf33 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf30, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf33)
        buf34 = buf10; del buf10  # reuse
        extern_kernels.bmm(as_strided(buf31, (64, 512, 64), (64, 4096, 1)), as_strided(buf32, (64, 64, 512), (64, 1, 4096)), out=buf34)
        buf37 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf34, buf37, 32768, 512, grid=grid(32768), stream=stream0)
        buf38 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf37, (64, 512, 512), (262144, 512, 1)), as_strided(buf33, (64, 512, 64), (64, 4096, 1)), out=buf38)
        buf39 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf38, buf39, 2097152, grid=grid(2097152), stream=stream0)
        buf40 = as_strided(buf38, (512, 4096), (4096, 1)); del buf38  # reuse
        extern_kernels.addmm(primals_15, buf39, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf40)
        buf41 = buf27; del buf27  # reuse
        buf42 = buf41; del buf41  # reuse
        buf44 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf359 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf42, buf29, primals_22, primals_23, buf40, primals_16, primals_17, buf44, buf45, buf359, 512, 4096, grid=grid(512), stream=stream0)
        buf46 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf45, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf46)
        buf47 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf46, buf47, buf48, 8388608, grid=grid(8388608), stream=stream0)
        buf49 = buf40; del buf40  # reuse
        extern_kernels.addmm(primals_21, buf48, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf49)
        buf50 = buf42; del buf42  # reuse
        buf51 = buf50; del buf50  # reuse
        buf53 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf358 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf51, buf49, buf44, primals_16, primals_17, primals_22, primals_23, buf53, buf54, buf358, 512, 4096, grid=grid(512), stream=stream0)
        buf55 = buf49; del buf49  # reuse
        extern_kernels.addmm(primals_9, buf54, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf55)
        buf56 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf54, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf56)
        buf57 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf54, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf57)
        buf58 = buf34; del buf34  # reuse
        extern_kernels.bmm(as_strided(buf55, (64, 512, 64), (64, 4096, 1)), as_strided(buf56, (64, 64, 512), (64, 1, 4096)), out=buf58)
        buf61 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf58, buf61, 32768, 512, grid=grid(32768), stream=stream0)
        buf62 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf61, (64, 512, 512), (262144, 512, 1)), as_strided(buf57, (64, 512, 64), (64, 4096, 1)), out=buf62)
        buf63 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf62, buf63, 2097152, grid=grid(2097152), stream=stream0)
        buf64 = as_strided(buf62, (512, 4096), (4096, 1)); del buf62  # reuse
        extern_kernels.addmm(primals_15, buf63, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf64)
        buf65 = buf51; del buf51  # reuse
        buf66 = buf65; del buf65  # reuse
        buf68 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf354 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf66, buf53, primals_22, primals_23, buf64, primals_16, primals_17, buf68, buf69, buf354, 512, 4096, grid=grid(512), stream=stream0)
        buf70 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf69, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf70)
        buf71 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf70, buf71, buf72, 8388608, grid=grid(8388608), stream=stream0)
        buf73 = buf64; del buf64  # reuse
        extern_kernels.addmm(primals_21, buf72, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf73)
        buf74 = buf66; del buf66  # reuse
        buf75 = buf74; del buf74  # reuse
        buf77 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf353 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf75, buf73, buf68, primals_16, primals_17, primals_22, primals_23, buf77, buf78, buf353, 512, 4096, grid=grid(512), stream=stream0)
        buf79 = buf73; del buf73  # reuse
        extern_kernels.addmm(primals_9, buf78, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf79)
        buf80 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf78, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf80)
        buf81 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf78, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf81)
        buf82 = buf58; del buf58  # reuse
        extern_kernels.bmm(as_strided(buf79, (64, 512, 64), (64, 4096, 1)), as_strided(buf80, (64, 64, 512), (64, 1, 4096)), out=buf82)
        buf85 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf82, buf85, 32768, 512, grid=grid(32768), stream=stream0)
        buf86 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf85, (64, 512, 512), (262144, 512, 1)), as_strided(buf81, (64, 512, 64), (64, 4096, 1)), out=buf86)
        buf87 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf86, buf87, 2097152, grid=grid(2097152), stream=stream0)
        buf88 = as_strided(buf86, (512, 4096), (4096, 1)); del buf86  # reuse
        extern_kernels.addmm(primals_15, buf87, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf88)
        buf89 = buf75; del buf75  # reuse
        buf90 = buf89; del buf89  # reuse
        buf92 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf349 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf90, buf77, primals_22, primals_23, buf88, primals_16, primals_17, buf92, buf93, buf349, 512, 4096, grid=grid(512), stream=stream0)
        buf94 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf93, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf94)
        buf95 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf96 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf94, buf95, buf96, 8388608, grid=grid(8388608), stream=stream0)
        buf97 = buf88; del buf88  # reuse
        extern_kernels.addmm(primals_21, buf96, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf97)
        buf98 = buf90; del buf90  # reuse
        buf99 = buf98; del buf98  # reuse
        buf101 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf348 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf99, buf97, buf92, primals_16, primals_17, primals_22, primals_23, buf101, buf102, buf348, 512, 4096, grid=grid(512), stream=stream0)
        buf103 = buf97; del buf97  # reuse
        extern_kernels.addmm(primals_9, buf102, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf103)
        buf104 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf102, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf104)
        buf105 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf102, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf105)
        buf106 = buf82; del buf82  # reuse
        extern_kernels.bmm(as_strided(buf103, (64, 512, 64), (64, 4096, 1)), as_strided(buf104, (64, 64, 512), (64, 1, 4096)), out=buf106)
        buf109 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf106, buf109, 32768, 512, grid=grid(32768), stream=stream0)
        buf110 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf109, (64, 512, 512), (262144, 512, 1)), as_strided(buf105, (64, 512, 64), (64, 4096, 1)), out=buf110)
        buf111 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf110, buf111, 2097152, grid=grid(2097152), stream=stream0)
        buf112 = as_strided(buf110, (512, 4096), (4096, 1)); del buf110  # reuse
        extern_kernels.addmm(primals_15, buf111, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf112)
        buf113 = buf99; del buf99  # reuse
        buf114 = buf113; del buf113  # reuse
        buf116 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf344 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf114, buf101, primals_22, primals_23, buf112, primals_16, primals_17, buf116, buf117, buf344, 512, 4096, grid=grid(512), stream=stream0)
        buf118 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf117, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf118)
        buf119 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf120 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf118, buf119, buf120, 8388608, grid=grid(8388608), stream=stream0)
        buf121 = buf112; del buf112  # reuse
        extern_kernels.addmm(primals_21, buf120, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf121)
        buf122 = buf114; del buf114  # reuse
        buf123 = buf122; del buf122  # reuse
        buf125 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf126 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf343 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf123, buf121, buf116, primals_16, primals_17, primals_22, primals_23, buf125, buf126, buf343, 512, 4096, grid=grid(512), stream=stream0)
        buf127 = buf121; del buf121  # reuse
        extern_kernels.addmm(primals_9, buf126, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf127)
        buf128 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf126, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf128)
        buf129 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf126, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf129)
        buf130 = buf106; del buf106  # reuse
        extern_kernels.bmm(as_strided(buf127, (64, 512, 64), (64, 4096, 1)), as_strided(buf128, (64, 64, 512), (64, 1, 4096)), out=buf130)
        buf133 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf130, buf133, 32768, 512, grid=grid(32768), stream=stream0)
        buf134 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf133, (64, 512, 512), (262144, 512, 1)), as_strided(buf129, (64, 512, 64), (64, 4096, 1)), out=buf134)
        buf135 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf134, buf135, 2097152, grid=grid(2097152), stream=stream0)
        buf136 = as_strided(buf134, (512, 4096), (4096, 1)); del buf134  # reuse
        extern_kernels.addmm(primals_15, buf135, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf136)
        buf137 = buf123; del buf123  # reuse
        buf138 = buf137; del buf137  # reuse
        buf140 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf141 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf339 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf138, buf125, primals_22, primals_23, buf136, primals_16, primals_17, buf140, buf141, buf339, 512, 4096, grid=grid(512), stream=stream0)
        buf142 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf141, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf142)
        buf143 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf144 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf142, buf143, buf144, 8388608, grid=grid(8388608), stream=stream0)
        buf145 = buf136; del buf136  # reuse
        extern_kernels.addmm(primals_21, buf144, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf145)
        buf146 = buf138; del buf138  # reuse
        buf147 = buf146; del buf146  # reuse
        buf149 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf150 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf338 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf147, buf145, buf140, primals_16, primals_17, primals_22, primals_23, buf149, buf150, buf338, 512, 4096, grid=grid(512), stream=stream0)
        buf151 = buf145; del buf145  # reuse
        extern_kernels.addmm(primals_9, buf150, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf151)
        buf152 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf150, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf152)
        buf153 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf150, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf153)
        buf154 = buf130; del buf130  # reuse
        extern_kernels.bmm(as_strided(buf151, (64, 512, 64), (64, 4096, 1)), as_strided(buf152, (64, 64, 512), (64, 1, 4096)), out=buf154)
        buf157 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf154, buf157, 32768, 512, grid=grid(32768), stream=stream0)
        buf158 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf157, (64, 512, 512), (262144, 512, 1)), as_strided(buf153, (64, 512, 64), (64, 4096, 1)), out=buf158)
        buf159 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf158, buf159, 2097152, grid=grid(2097152), stream=stream0)
        buf160 = as_strided(buf158, (512, 4096), (4096, 1)); del buf158  # reuse
        extern_kernels.addmm(primals_15, buf159, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf160)
        buf161 = buf147; del buf147  # reuse
        buf162 = buf161; del buf161  # reuse
        buf164 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf334 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf162, buf149, primals_22, primals_23, buf160, primals_16, primals_17, buf164, buf165, buf334, 512, 4096, grid=grid(512), stream=stream0)
        buf166 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf165, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf166)
        buf167 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf168 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf166, buf167, buf168, 8388608, grid=grid(8388608), stream=stream0)
        buf169 = buf160; del buf160  # reuse
        extern_kernels.addmm(primals_21, buf168, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf169)
        buf170 = buf162; del buf162  # reuse
        buf171 = buf170; del buf170  # reuse
        buf173 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf174 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf333 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf171, buf169, buf164, primals_16, primals_17, primals_22, primals_23, buf173, buf174, buf333, 512, 4096, grid=grid(512), stream=stream0)
        buf175 = buf169; del buf169  # reuse
        extern_kernels.addmm(primals_9, buf174, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf175)
        buf176 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf174, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf176)
        buf177 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf174, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf177)
        buf178 = buf154; del buf154  # reuse
        extern_kernels.bmm(as_strided(buf175, (64, 512, 64), (64, 4096, 1)), as_strided(buf176, (64, 64, 512), (64, 1, 4096)), out=buf178)
        buf181 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf178, buf181, 32768, 512, grid=grid(32768), stream=stream0)
        buf182 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf181, (64, 512, 512), (262144, 512, 1)), as_strided(buf177, (64, 512, 64), (64, 4096, 1)), out=buf182)
        buf183 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf182, buf183, 2097152, grid=grid(2097152), stream=stream0)
        buf184 = as_strided(buf182, (512, 4096), (4096, 1)); del buf182  # reuse
        extern_kernels.addmm(primals_15, buf183, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf184)
        buf185 = buf171; del buf171  # reuse
        buf186 = buf185; del buf185  # reuse
        buf188 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf189 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf329 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf186, buf173, primals_22, primals_23, buf184, primals_16, primals_17, buf188, buf189, buf329, 512, 4096, grid=grid(512), stream=stream0)
        buf190 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf189, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf190)
        buf191 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf192 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf190, buf191, buf192, 8388608, grid=grid(8388608), stream=stream0)
        buf193 = buf184; del buf184  # reuse
        extern_kernels.addmm(primals_21, buf192, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf193)
        buf194 = buf186; del buf186  # reuse
        buf195 = buf194; del buf194  # reuse
        buf197 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf328 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf195, buf193, buf188, primals_16, primals_17, primals_22, primals_23, buf197, buf198, buf328, 512, 4096, grid=grid(512), stream=stream0)
        buf199 = buf193; del buf193  # reuse
        extern_kernels.addmm(primals_9, buf198, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf199)
        buf200 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf198, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf200)
        buf201 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf198, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf201)
        buf202 = buf178; del buf178  # reuse
        extern_kernels.bmm(as_strided(buf199, (64, 512, 64), (64, 4096, 1)), as_strided(buf200, (64, 64, 512), (64, 1, 4096)), out=buf202)
        buf205 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf202, buf205, 32768, 512, grid=grid(32768), stream=stream0)
        buf206 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf205, (64, 512, 512), (262144, 512, 1)), as_strided(buf201, (64, 512, 64), (64, 4096, 1)), out=buf206)
        buf207 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf206, buf207, 2097152, grid=grid(2097152), stream=stream0)
        buf208 = as_strided(buf206, (512, 4096), (4096, 1)); del buf206  # reuse
        extern_kernels.addmm(primals_15, buf207, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf208)
        buf209 = buf195; del buf195  # reuse
        buf210 = buf209; del buf209  # reuse
        buf212 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf213 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf324 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf210, buf197, primals_22, primals_23, buf208, primals_16, primals_17, buf212, buf213, buf324, 512, 4096, grid=grid(512), stream=stream0)
        buf214 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf213, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf214)
        buf215 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf216 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf214, buf215, buf216, 8388608, grid=grid(8388608), stream=stream0)
        buf217 = buf208; del buf208  # reuse
        extern_kernels.addmm(primals_21, buf216, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf217)
        buf218 = buf210; del buf210  # reuse
        buf219 = buf218; del buf218  # reuse
        buf221 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf222 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf323 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf219, buf217, buf212, primals_16, primals_17, primals_22, primals_23, buf221, buf222, buf323, 512, 4096, grid=grid(512), stream=stream0)
        buf223 = buf217; del buf217  # reuse
        extern_kernels.addmm(primals_9, buf222, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf223)
        buf224 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf222, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf224)
        buf225 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf222, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf225)
        buf226 = buf202; del buf202  # reuse
        extern_kernels.bmm(as_strided(buf223, (64, 512, 64), (64, 4096, 1)), as_strided(buf224, (64, 64, 512), (64, 1, 4096)), out=buf226)
        buf229 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf226, buf229, 32768, 512, grid=grid(32768), stream=stream0)
        buf230 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf229, (64, 512, 512), (262144, 512, 1)), as_strided(buf225, (64, 512, 64), (64, 4096, 1)), out=buf230)
        buf231 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf230, buf231, 2097152, grid=grid(2097152), stream=stream0)
        buf232 = as_strided(buf230, (512, 4096), (4096, 1)); del buf230  # reuse
        extern_kernels.addmm(primals_15, buf231, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf232)
        buf233 = buf219; del buf219  # reuse
        buf234 = buf233; del buf233  # reuse
        buf236 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf237 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf319 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf234, buf221, primals_22, primals_23, buf232, primals_16, primals_17, buf236, buf237, buf319, 512, 4096, grid=grid(512), stream=stream0)
        buf238 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf237, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf238)
        buf239 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf240 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf238, buf239, buf240, 8388608, grid=grid(8388608), stream=stream0)
        buf241 = buf232; del buf232  # reuse
        extern_kernels.addmm(primals_21, buf240, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf241)
        buf242 = buf234; del buf234  # reuse
        buf243 = buf242; del buf242  # reuse
        buf245 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf318 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf243, buf241, buf236, primals_16, primals_17, primals_22, primals_23, buf245, buf246, buf318, 512, 4096, grid=grid(512), stream=stream0)
        buf247 = buf241; del buf241  # reuse
        extern_kernels.addmm(primals_9, buf246, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf247)
        buf248 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf246, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf248)
        buf249 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf246, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf249)
        buf250 = buf226; del buf226  # reuse
        extern_kernels.bmm(as_strided(buf247, (64, 512, 64), (64, 4096, 1)), as_strided(buf248, (64, 64, 512), (64, 1, 4096)), out=buf250)
        buf253 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf250, buf253, 32768, 512, grid=grid(32768), stream=stream0)
        buf254 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf253, (64, 512, 512), (262144, 512, 1)), as_strided(buf249, (64, 512, 64), (64, 4096, 1)), out=buf254)
        buf255 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf254, buf255, 2097152, grid=grid(2097152), stream=stream0)
        buf256 = as_strided(buf254, (512, 4096), (4096, 1)); del buf254  # reuse
        extern_kernels.addmm(primals_15, buf255, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf256)
        buf257 = buf243; del buf243  # reuse
        buf258 = buf257; del buf257  # reuse
        buf260 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf261 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf314 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf258, buf245, primals_22, primals_23, buf256, primals_16, primals_17, buf260, buf261, buf314, 512, 4096, grid=grid(512), stream=stream0)
        buf262 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf261, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf262)
        buf263 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf262, buf263, buf264, 8388608, grid=grid(8388608), stream=stream0)
        buf265 = buf256; del buf256  # reuse
        extern_kernels.addmm(primals_21, buf264, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf265)
        buf266 = buf258; del buf258  # reuse
        buf267 = buf266; del buf266  # reuse
        buf269 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf270 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf313 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf267, buf265, buf260, primals_16, primals_17, primals_22, primals_23, buf269, buf270, buf313, 512, 4096, grid=grid(512), stream=stream0)
        buf271 = buf265; del buf265  # reuse
        extern_kernels.addmm(primals_9, buf270, as_strided(primals_8, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf271)
        del primals_9
        buf272 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_11, buf270, as_strided(primals_10, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf272)
        del primals_11
        buf273 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_13, buf270, as_strided(primals_12, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf273)
        del primals_13
        buf274 = buf250; del buf250  # reuse
        extern_kernels.bmm(as_strided(buf271, (64, 512, 64), (64, 4096, 1)), as_strided(buf272, (64, 64, 512), (64, 1, 4096)), out=buf274)
        buf277 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf274, buf277, 32768, 512, grid=grid(32768), stream=stream0)
        del buf274
        buf278 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf277, (64, 512, 512), (262144, 512, 1)), as_strided(buf273, (64, 512, 64), (64, 4096, 1)), out=buf278)
        buf279 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__2.run(buf278, buf279, 2097152, grid=grid(2097152), stream=stream0)
        buf280 = as_strided(buf278, (512, 4096), (4096, 1)); del buf278  # reuse
        extern_kernels.addmm(primals_15, buf279, as_strided(primals_14, (4096, 4096), (1, 4096)), alpha=1, beta=1, out=buf280)
        del primals_15
        buf281 = buf267; del buf267  # reuse
        buf282 = buf281; del buf281  # reuse
        buf284 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf285 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf309 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf282, buf269, primals_22, primals_23, buf280, primals_16, primals_17, buf284, buf285, buf309, 512, 4096, grid=grid(512), stream=stream0)
        buf286 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_19, buf285, as_strided(primals_18, (4096, 16384), (1, 4096)), alpha=1, beta=1, out=buf286)
        del primals_19
        buf287 = empty_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda', dtype=torch.float32)
        buf288 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf286, buf287, buf288, 8388608, grid=grid(8388608), stream=stream0)
        buf289 = buf280; del buf280  # reuse
        extern_kernels.addmm(primals_21, buf288, as_strided(primals_20, (16384, 4096), (1, 16384)), alpha=1, beta=1, out=buf289)
        del primals_21
        buf290 = buf282; del buf282  # reuse
        buf291 = buf290; del buf290  # reuse
        buf293 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf294 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf308 = empty_strided((1, 512, 1), (512, 1, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf291, buf289, buf284, primals_16, primals_17, primals_22, primals_23, buf293, buf294, buf308, 512, 4096, grid=grid(512), stream=stream0)
        del buf289
        del primals_17
        del primals_23
        buf295 = empty_strided((512, 2), (2, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_25, buf294, as_strided(primals_24, (4096, 2), (1, 4096)), alpha=1, beta=1, out=buf295)
        del primals_25
        buf296 = as_strided(buf291, (1, 512), (512, 1)); del buf291  # reuse
        buf300 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf295, buf296, buf300, 1, 512, grid=grid(1), stream=stream0)
        buf297 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        buf305 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf295, buf297, buf305, 1, 512, grid=grid(1), stream=stream0)
        del buf295
        buf301 = empty_strided((1, 1), (1, 1), device='cuda', dtype=torch.int64)
        buf302 = empty_strided((1, ), (1, ), device='cuda', dtype=torch.bool)
        buf306 = empty_strided((1, 1), (1, 1), device='cuda', dtype=torch.int64)
        buf307 = empty_strided((1, ), (1, ), device='cuda', dtype=torch.bool)
        buf372 = empty_strided((), (), device='cuda', dtype=torch.float32)
        triton__9.run(primals_29, primals_30, buf300, buf305, buf301, buf302, buf306, buf307, buf372, 1, grid=grid(1), stream=stream0)
        del primals_29
        del primals_30
        buf310 = as_strided(buf273, (64, 64, 512), (64, 1, 4096)); del buf273  # reuse
        triton__10.run(buf310, 2097152, grid=grid(2097152), stream=stream0)
        buf311 = as_strided(buf271, (64, 64, 512), (64, 1, 4096)); del buf271  # reuse
        triton__10.run(buf311, 2097152, grid=grid(2097152), stream=stream0)
        buf312 = as_strided(buf272, (64, 512, 64), (64, 4096, 1)); del buf272  # reuse
        triton__10.run(buf312, 2097152, grid=grid(2097152), stream=stream0)
        buf315 = as_strided(buf249, (64, 64, 512), (64, 1, 4096)); del buf249  # reuse
        triton__10.run(buf315, 2097152, grid=grid(2097152), stream=stream0)
        buf316 = as_strided(buf247, (64, 64, 512), (64, 1, 4096)); del buf247  # reuse
        triton__10.run(buf316, 2097152, grid=grid(2097152), stream=stream0)
        buf317 = as_strided(buf248, (64, 512, 64), (64, 4096, 1)); del buf248  # reuse
        triton__10.run(buf317, 2097152, grid=grid(2097152), stream=stream0)
        buf320 = as_strided(buf225, (64, 64, 512), (64, 1, 4096)); del buf225  # reuse
        triton__10.run(buf320, 2097152, grid=grid(2097152), stream=stream0)
        buf321 = as_strided(buf223, (64, 64, 512), (64, 1, 4096)); del buf223  # reuse
        triton__10.run(buf321, 2097152, grid=grid(2097152), stream=stream0)
        buf322 = as_strided(buf224, (64, 512, 64), (64, 4096, 1)); del buf224  # reuse
        triton__10.run(buf322, 2097152, grid=grid(2097152), stream=stream0)
        buf325 = as_strided(buf201, (64, 64, 512), (64, 1, 4096)); del buf201  # reuse
        triton__10.run(buf325, 2097152, grid=grid(2097152), stream=stream0)
        buf326 = as_strided(buf199, (64, 64, 512), (64, 1, 4096)); del buf199  # reuse
        triton__10.run(buf326, 2097152, grid=grid(2097152), stream=stream0)
        buf327 = as_strided(buf200, (64, 512, 64), (64, 4096, 1)); del buf200  # reuse
        triton__10.run(buf327, 2097152, grid=grid(2097152), stream=stream0)
        buf330 = as_strided(buf177, (64, 64, 512), (64, 1, 4096)); del buf177  # reuse
        triton__10.run(buf330, 2097152, grid=grid(2097152), stream=stream0)
        buf331 = as_strided(buf175, (64, 64, 512), (64, 1, 4096)); del buf175  # reuse
        triton__10.run(buf331, 2097152, grid=grid(2097152), stream=stream0)
        buf332 = as_strided(buf176, (64, 512, 64), (64, 4096, 1)); del buf176  # reuse
        triton__10.run(buf332, 2097152, grid=grid(2097152), stream=stream0)
        buf335 = as_strided(buf153, (64, 64, 512), (64, 1, 4096)); del buf153  # reuse
        triton__10.run(buf335, 2097152, grid=grid(2097152), stream=stream0)
        buf336 = as_strided(buf151, (64, 64, 512), (64, 1, 4096)); del buf151  # reuse
        triton__10.run(buf336, 2097152, grid=grid(2097152), stream=stream0)
        buf337 = as_strided(buf152, (64, 512, 64), (64, 4096, 1)); del buf152  # reuse
        triton__10.run(buf337, 2097152, grid=grid(2097152), stream=stream0)
        buf340 = as_strided(buf129, (64, 64, 512), (64, 1, 4096)); del buf129  # reuse
        triton__10.run(buf340, 2097152, grid=grid(2097152), stream=stream0)
        buf341 = as_strided(buf127, (64, 64, 512), (64, 1, 4096)); del buf127  # reuse
        triton__10.run(buf341, 2097152, grid=grid(2097152), stream=stream0)
        buf342 = as_strided(buf128, (64, 512, 64), (64, 4096, 1)); del buf128  # reuse
        triton__10.run(buf342, 2097152, grid=grid(2097152), stream=stream0)
        buf345 = as_strided(buf105, (64, 64, 512), (64, 1, 4096)); del buf105  # reuse
        triton__10.run(buf345, 2097152, grid=grid(2097152), stream=stream0)
        buf346 = as_strided(buf103, (64, 64, 512), (64, 1, 4096)); del buf103  # reuse
        triton__10.run(buf346, 2097152, grid=grid(2097152), stream=stream0)
        buf347 = as_strided(buf104, (64, 512, 64), (64, 4096, 1)); del buf104  # reuse
        triton__10.run(buf347, 2097152, grid=grid(2097152), stream=stream0)
        buf350 = as_strided(buf81, (64, 64, 512), (64, 1, 4096)); del buf81  # reuse
        triton__10.run(buf350, 2097152, grid=grid(2097152), stream=stream0)
        buf351 = as_strided(buf79, (64, 64, 512), (64, 1, 4096)); del buf79  # reuse
        triton__10.run(buf351, 2097152, grid=grid(2097152), stream=stream0)
        buf352 = as_strided(buf80, (64, 512, 64), (64, 4096, 1)); del buf80  # reuse
        triton__10.run(buf352, 2097152, grid=grid(2097152), stream=stream0)
        buf355 = as_strided(buf57, (64, 64, 512), (64, 1, 4096)); del buf57  # reuse
        triton__10.run(buf355, 2097152, grid=grid(2097152), stream=stream0)
        buf356 = as_strided(buf55, (64, 64, 512), (64, 1, 4096)); del buf55  # reuse
        triton__10.run(buf356, 2097152, grid=grid(2097152), stream=stream0)
        buf357 = as_strided(buf56, (64, 512, 64), (64, 4096, 1)); del buf56  # reuse
        triton__10.run(buf357, 2097152, grid=grid(2097152), stream=stream0)
        buf360 = as_strided(buf33, (64, 64, 512), (64, 1, 4096)); del buf33  # reuse
        triton__10.run(buf360, 2097152, grid=grid(2097152), stream=stream0)
        buf361 = as_strided(buf31, (64, 64, 512), (64, 1, 4096)); del buf31  # reuse
        triton__10.run(buf361, 2097152, grid=grid(2097152), stream=stream0)
        buf362 = as_strided(buf32, (64, 512, 64), (64, 4096, 1)); del buf32  # reuse
        triton__10.run(buf362, 2097152, grid=grid(2097152), stream=stream0)
        buf365 = as_strided(buf9, (64, 64, 512), (64, 1, 4096)); del buf9  # reuse
        triton__10.run(buf365, 2097152, grid=grid(2097152), stream=stream0)
        buf366 = as_strided(buf7, (64, 64, 512), (64, 1, 4096)); del buf7  # reuse
        triton__10.run(buf366, 2097152, grid=grid(2097152), stream=stream0)
        buf367 = as_strided(buf8, (64, 512, 64), (64, 4096, 1)); del buf8  # reuse
        triton__10.run(buf367, 2097152, grid=grid(2097152), stream=stream0)
        buf369 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__11.run(primals_27, buf369, 512, grid=grid(512), stream=stream0)
        del primals_27
        buf370 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__11.run(primals_26, buf370, 512, grid=grid(512), stream=stream0)
        del primals_26
        buf371 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__11.run(primals_28, buf371, 512, grid=grid(512), stream=stream0)
        del primals_28
        return (buf372, buf296, buf297, primals_4, primals_16, primals_22, buf4, buf5, as_strided(buf6, (512, 4096), (4096, 1)), buf13, buf15, buf20, buf21, buf22, buf23, buf24, buf29, buf30, buf37, buf39, buf44, buf45, buf46, buf47, buf48, buf53, buf54, buf61, buf63, buf68, buf69, buf70, buf71, buf72, buf77, buf78, buf85, buf87, buf92, buf93, buf94, buf95, buf96, buf101, buf102, buf109, buf111, buf116, buf117, buf118, buf119, buf120, buf125, buf126, buf133, buf135, buf140, buf141, buf142, buf143, buf144, buf149, buf150, buf157, buf159, buf164, buf165, buf166, buf167, buf168, buf173, buf174, buf181, buf183, buf188, buf189, buf190, buf191, buf192, buf197, buf198, buf205, buf207, buf212, buf213, buf214, buf215, buf216, buf221, buf222, buf229, buf231, buf236, buf237, buf238, buf239, buf240, buf245, buf246, buf253, buf255, buf260, buf261, buf262, buf263, buf264, buf269, buf270, buf277, buf279, buf284, buf285, buf286, buf287, buf288, buf293, buf294, buf300, buf301, buf302, buf305, buf306, buf307, as_strided(primals_24, (2, 4096), (4096, 1)), buf308, as_strided(primals_20, (4096, 16384), (16384, 1)), as_strided(primals_18, (16384, 4096), (4096, 1)), buf309, as_strided(primals_14, (4096, 4096), (4096, 1)), as_strided(buf277, (64, 512, 512), (262144, 1, 512)), buf310, buf311, buf312, as_strided(primals_12, (4096, 4096), (4096, 1)), as_strided(primals_10, (4096, 4096), (4096, 1)), as_strided(primals_8, (4096, 4096), (4096, 1)), buf313, buf314, as_strided(buf253, (64, 512, 512), (262144, 1, 512)), buf315, buf316, buf317, buf318, buf319, as_strided(buf229, (64, 512, 512), (262144, 1, 512)), buf320, buf321, buf322, buf323, buf324, as_strided(buf205, (64, 512, 512), (262144, 1, 512)), buf325, buf326, buf327, buf328, buf329, as_strided(buf181, (64, 512, 512), (262144, 1, 512)), buf330, buf331, buf332, buf333, buf334, as_strided(buf157, (64, 512, 512), (262144, 1, 512)), buf335, buf336, buf337, buf338, buf339, as_strided(buf133, (64, 512, 512), (262144, 1, 512)), buf340, buf341, buf342, buf343, buf344, as_strided(buf109, (64, 512, 512), (262144, 1, 512)), buf345, buf346, buf347, buf348, buf349, as_strided(buf85, (64, 512, 512), (262144, 1, 512)), buf350, buf351, buf352, buf353, buf354, as_strided(buf61, (64, 512, 512), (262144, 1, 512)), buf355, buf356, buf357, buf358, buf359, as_strided(buf37, (64, 512, 512), (262144, 1, 512)), buf360, buf361, buf362, buf363, buf364, as_strided(buf13, (64, 512, 512), (262144, 1, 512)), buf365, buf366, buf367, as_strided(primals_6, (4096, 128), (128, 1)), buf368, buf369, buf370, buf371, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_27 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_28 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_29 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_30 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30]))
