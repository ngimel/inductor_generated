
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
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK])
    tmp1 = -1.0
    tl.store(out_ptr0 + (tmp0), tmp1, None)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*i64', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1_load = tl.load(in_ptr1 + (0))
    tmp1 = tl.broadcast_to(tmp1_load, [XBLOCK, RBLOCK])
    tmp4_load = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp4_load, [XBLOCK, RBLOCK])
    tmp7_load = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp7_load, [XBLOCK, RBLOCK])
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp16_load = tl.load(in_ptr5 + (0))
    tmp16 = tl.broadcast_to(tmp16_load, [XBLOCK, RBLOCK])
    tmp18_load = tl.load(in_ptr6 + (0))
    tmp18 = tl.broadcast_to(tmp18_load, [XBLOCK, RBLOCK])
    _tmp24 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_last')
        tmp2 = 512
        tmp3 = tmp1 != tmp2
        tmp5 = 2.0
        tmp6 = tmp4 / tmp5
        tmp8 = tmp7.to(tl.int64)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp6 / tmp9
        tmp11 = 0.0
        tmp12 = tl.where(tmp3, tmp10, tmp11)
        tmp13 = tmp0 * tmp12
        _tmp14 = tl.where(rmask, _tmp14 + tmp13, _tmp14)
        tmp17 = tmp16 != tmp2
        tmp19 = tmp18.to(tl.int64)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp6 / tmp20
        tmp22 = tl.where(tmp17, tmp21, tmp11)
        tmp23 = tmp15 * tmp22
        _tmp24 = tl.where(rmask, _tmp24 + tmp23, _tmp24)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp25 = tl.load(in_ptr7 + (r0), rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_last')
        tmp37 = tl.load(in_ptr8 + (r0), rmask, eviction_policy='evict_last')
        tmp42 = tl.load(in_ptr9 + (r0), rmask, eviction_policy='evict_last')
        tmp43 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp50 = tl.load(in_ptr10 + (r0), rmask, eviction_policy='evict_last')
        tmp27 = 512
        tmp28 = tmp16 != tmp27
        tmp29 = 2.0
        tmp30 = tmp4 / tmp29
        tmp31 = tmp18.to(tl.int64)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 / tmp32
        tmp34 = 0.0
        tmp35 = tl.where(tmp28, tmp33, tmp34)
        tmp36 = tmp26 * tmp35
        tmp38 = tl.exp(tmp37)
        tmp39 = tmp38 * tmp24
        tmp40 = tmp36 - tmp39
        tmp41 = tmp25 + tmp40
        tmp44 = tmp1 != tmp27
        tmp45 = tmp7.to(tl.int64)
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp30 / tmp46
        tmp48 = tl.where(tmp44, tmp47, tmp34)
        tmp49 = tmp43 * tmp48
        tmp51 = tl.exp(tmp50)
        tmp52 = tmp51 * tmp14
        tmp53 = tmp49 - tmp52
        tmp54 = tmp42 + tmp53
        tl.store(out_ptr2 + (2*r0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, rmask)
        tl.store(out_ptr3 + (2*r0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp54, rmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2*r2) + (256*x1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2, 4],
              reduction_hint=ReductionHint.OUTER_TINY,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
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
        tmp0 = tl.load(in_ptr0 + (x0 + (2*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp5 = tmp2 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = tmp8 * tmp9
        tmp11 = 4096.0
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12 - tmp3
        tmp15 = tmp14 * tmp6
        tmp16 = tmp13 - tmp15
        tmp17 = tmp7 * tmp16
        tl.store(out_ptr2 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp17, rmask & xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = 1.0
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 - tmp7
    tmp9 = tmp4 * tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = 0.044715
    tmp13 = tmp11 * tmp12
    tmp14 = tmp1 * tmp1
    tmp15 = 3.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 * tmp16
    tmp18 = tmp11 + tmp17
    tmp19 = tmp6 + tmp5
    tmp20 = tmp0 * tmp19
    tmp21 = tmp20 * tmp2
    tmp22 = tmp18 + tmp21
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp4 * tmp6
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp18 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 * tmp13
        tmp15 = 4096.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp5
        tmp19 = tmp18 * tmp8
        tmp20 = tmp17 - tmp19
        tmp21 = tmp9 * tmp20
        tl.store(out_ptr2 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp21, rmask & xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32768, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tmp4 * tmp5
        tmp7 = tmp5 * tmp3
        tmp8 = tmp6 - tmp7
        tmp9 = 8.0
        tmp10 = tmp8 / tmp9
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__10(in_out_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tl.store(in_out_ptr0 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp0, rmask & xmask)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 4096],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        tl.store(out_ptr0 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp8, rmask & xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr5 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tmp10 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp14 = tl.load(in_ptr6 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr5 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp16 = 4096.0
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17 - tmp9
        tmp20 = tmp19 * tmp13
        tmp21 = tmp18 - tmp20
        tmp22 = tmp14 * tmp21
        tl.store(out_ptr3 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp22, rmask & xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: 'i32', 60: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60), equal_to_1=())]})
@triton.jit
def triton__12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp25 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp26 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp36 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp37 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp47 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp48 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp58 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp59 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp69 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp70 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp80 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp81 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp91 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp92 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp102 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp103 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp113 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp114 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp124 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp125 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp16 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp21 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp23 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp27 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp28 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp30 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp32 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp34 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp38 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp39 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp41 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp43 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp49 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp50 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp52 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp54 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp56 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp60 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp61 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp63 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp65 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp67 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp71 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp72 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp74 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp76 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp78 = tl.load(in_ptr36 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp82 = tl.load(in_ptr37 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp83 = tl.load(in_ptr38 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp85 = tl.load(in_ptr39 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp87 = tl.load(in_ptr40 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp89 = tl.load(in_ptr41 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp93 = tl.load(in_ptr42 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp94 = tl.load(in_ptr43 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp96 = tl.load(in_ptr44 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp98 = tl.load(in_ptr45 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp100 = tl.load(in_ptr46 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp104 = tl.load(in_ptr47 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp105 = tl.load(in_ptr48 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp107 = tl.load(in_ptr49 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp109 = tl.load(in_ptr50 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp111 = tl.load(in_ptr51 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp115 = tl.load(in_ptr52 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp116 = tl.load(in_ptr53 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp118 = tl.load(in_ptr54 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp120 = tl.load(in_ptr55 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp122 = tl.load(in_ptr56 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp0, _tmp4)
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 * tmp12
        _tmp14 = tl.where(rmask & xmask, _tmp14 + tmp13, _tmp14)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp11, _tmp15)
        tmp18 = tmp16 + tmp17
        tmp20 = tmp18 + tmp19
        tmp22 = tmp20 + tmp21
        tmp24 = tmp22 * tmp23
        _tmp25 = tl.where(rmask & xmask, _tmp25 + tmp24, _tmp25)
        _tmp26 = tl.where(rmask & xmask, _tmp26 + tmp22, _tmp26)
        tmp29 = tmp27 + tmp28
        tmp31 = tmp29 + tmp30
        tmp33 = tmp31 + tmp32
        tmp35 = tmp33 * tmp34
        _tmp36 = tl.where(rmask & xmask, _tmp36 + tmp35, _tmp36)
        _tmp37 = tl.where(rmask & xmask, _tmp37 + tmp33, _tmp37)
        tmp40 = tmp38 + tmp39
        tmp42 = tmp40 + tmp41
        tmp44 = tmp42 + tmp43
        tmp46 = tmp44 * tmp45
        _tmp47 = tl.where(rmask & xmask, _tmp47 + tmp46, _tmp47)
        _tmp48 = tl.where(rmask & xmask, _tmp48 + tmp44, _tmp48)
        tmp51 = tmp49 + tmp50
        tmp53 = tmp51 + tmp52
        tmp55 = tmp53 + tmp54
        tmp57 = tmp55 * tmp56
        _tmp58 = tl.where(rmask & xmask, _tmp58 + tmp57, _tmp58)
        _tmp59 = tl.where(rmask & xmask, _tmp59 + tmp55, _tmp59)
        tmp62 = tmp60 + tmp61
        tmp64 = tmp62 + tmp63
        tmp66 = tmp64 + tmp65
        tmp68 = tmp66 * tmp67
        _tmp69 = tl.where(rmask & xmask, _tmp69 + tmp68, _tmp69)
        _tmp70 = tl.where(rmask & xmask, _tmp70 + tmp66, _tmp70)
        tmp73 = tmp71 + tmp72
        tmp75 = tmp73 + tmp74
        tmp77 = tmp75 + tmp76
        tmp79 = tmp77 * tmp78
        _tmp80 = tl.where(rmask & xmask, _tmp80 + tmp79, _tmp80)
        _tmp81 = tl.where(rmask & xmask, _tmp81 + tmp77, _tmp81)
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp88 = tmp86 + tmp87
        tmp90 = tmp88 * tmp89
        _tmp91 = tl.where(rmask & xmask, _tmp91 + tmp90, _tmp91)
        _tmp92 = tl.where(rmask & xmask, _tmp92 + tmp88, _tmp92)
        tmp95 = tmp93 + tmp94
        tmp97 = tmp95 + tmp96
        tmp99 = tmp97 + tmp98
        tmp101 = tmp99 * tmp100
        _tmp102 = tl.where(rmask & xmask, _tmp102 + tmp101, _tmp102)
        _tmp103 = tl.where(rmask & xmask, _tmp103 + tmp99, _tmp103)
        tmp106 = tmp104 + tmp105
        tmp108 = tmp106 + tmp107
        tmp110 = tmp108 + tmp109
        tmp112 = tmp110 * tmp111
        _tmp113 = tl.where(rmask & xmask, _tmp113 + tmp112, _tmp113)
        _tmp114 = tl.where(rmask & xmask, _tmp114 + tmp110, _tmp114)
        tmp117 = tmp115 + tmp116
        tmp119 = tmp117 + tmp118
        tmp121 = tmp119 + tmp120
        tmp123 = tmp121 * tmp122
        _tmp124 = tl.where(rmask & xmask, _tmp124 + tmp123, _tmp124)
        _tmp125 = tl.where(rmask & xmask, _tmp125 + tmp121, _tmp125)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tmp47 = tl.sum(_tmp47, 1)[:, None]
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tmp58 = tl.sum(_tmp58, 1)[:, None]
    tmp59 = tl.sum(_tmp59, 1)[:, None]
    tmp69 = tl.sum(_tmp69, 1)[:, None]
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    tmp80 = tl.sum(_tmp80, 1)[:, None]
    tmp81 = tl.sum(_tmp81, 1)[:, None]
    tmp91 = tl.sum(_tmp91, 1)[:, None]
    tmp92 = tl.sum(_tmp92, 1)[:, None]
    tmp102 = tl.sum(_tmp102, 1)[:, None]
    tmp103 = tl.sum(_tmp103, 1)[:, None]
    tmp113 = tl.sum(_tmp113, 1)[:, None]
    tmp114 = tl.sum(_tmp114, 1)[:, None]
    tmp124 = tl.sum(_tmp124, 1)[:, None]
    tmp125 = tl.sum(_tmp125, 1)[:, None]
    tmp126 = tmp3 + tmp14
    tmp127 = tmp126 + tmp25
    tmp128 = tmp127 + tmp36
    tmp129 = tmp128 + tmp47
    tmp130 = tmp129 + tmp58
    tmp131 = tmp130 + tmp80
    tmp132 = tmp131 + tmp102
    tmp133 = tmp132 + tmp124
    tmp134 = tmp133 + tmp69
    tmp135 = tmp134 + tmp91
    tmp136 = tmp135 + tmp113
    tmp137 = tmp4 + tmp15
    tmp138 = tmp137 + tmp26
    tmp139 = tmp138 + tmp37
    tmp140 = tmp139 + tmp48
    tmp141 = tmp140 + tmp59
    tmp142 = tmp141 + tmp81
    tmp143 = tmp142 + tmp103
    tmp144 = tmp143 + tmp125
    tmp145 = tmp144 + tmp70
    tmp146 = tmp145 + tmp92
    tmp147 = tmp146 + tmp114
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp136, xmask)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp147, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16384, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__13(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (524288*x1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


triton__14 = async_compile.triton('''
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
def triton__14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        tl.store(in_out_ptr0 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp8, rmask & xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr4 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tmp10 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp14 = tl.load(in_ptr5 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr4 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp16 = 4096.0
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17 - tmp9
        tmp20 = tmp19 * tmp13
        tmp21 = tmp18 - tmp20
        tmp22 = tmp14 * tmp21
        tl.store(out_ptr2 + (r1 + (4096*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp22, rmask & xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 4],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp16 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp18 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp20 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
        _tmp19 = tl.where(rmask & xmask, _tmp19 + tmp18, _tmp19)
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
        _tmp23 = tl.where(rmask & xmask, _tmp23 + tmp22, _tmp23)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp24 = tmp1 + tmp3
    tmp25 = tmp24 + tmp5
    tmp26 = tmp25 + tmp7
    tmp27 = tmp26 + tmp9
    tmp28 = tmp27 + tmp11
    tmp29 = tmp28 + tmp15
    tmp30 = tmp29 + tmp19
    tmp31 = tmp30 + tmp23
    tmp32 = tmp31 + tmp13
    tmp33 = tmp32 + tmp17
    tmp34 = tmp33 + tmp21
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16384, 512],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr4 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr5 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr6 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr7 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp16 = tl.load(in_ptr8 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp18 = tl.load(in_ptr9 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp20 = tl.load(in_ptr10 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr11 + (x0 + (16384*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
        _tmp19 = tl.where(rmask & xmask, _tmp19 + tmp18, _tmp19)
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
        _tmp23 = tl.where(rmask & xmask, _tmp23 + tmp22, _tmp23)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp24 = tmp1 + tmp3
    tmp25 = tmp24 + tmp5
    tmp26 = tmp25 + tmp7
    tmp27 = tmp26 + tmp9
    tmp28 = tmp27 + tmp11
    tmp29 = tmp28 + tmp15
    tmp30 = tmp29 + tmp19
    tmp31 = tmp30 + tmp23
    tmp32 = tmp31 + tmp13
    tmp33 = tmp32 + tmp17
    tmp34 = tmp33 + tmp21
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: 'i32', 39: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39), equal_to_1=())]})
@triton.jit
def triton__17(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp26 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp27 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp34 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp40 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp41 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp47 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp48 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp54 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp55 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp61 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp62 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp68 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp69 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp75 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp76 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp82 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp83 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp21 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp24 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp28 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp29 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp31 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp35 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp36 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp38 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp42 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp43 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp49 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp50 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp52 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp56 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp63 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp64 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp66 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp70 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp71 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp73 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp77 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp78 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp80 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp2, _tmp6)
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(rmask & xmask, _tmp12 + tmp11, _tmp12)
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp9, _tmp13)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 * tmp17
        _tmp19 = tl.where(rmask & xmask, _tmp19 + tmp18, _tmp19)
        _tmp20 = tl.where(rmask & xmask, _tmp20 + tmp16, _tmp20)
        tmp23 = tmp21 + tmp22
        tmp25 = tmp23 * tmp24
        _tmp26 = tl.where(rmask & xmask, _tmp26 + tmp25, _tmp26)
        _tmp27 = tl.where(rmask & xmask, _tmp27 + tmp23, _tmp27)
        tmp30 = tmp28 + tmp29
        tmp32 = tmp30 * tmp31
        _tmp33 = tl.where(rmask & xmask, _tmp33 + tmp32, _tmp33)
        _tmp34 = tl.where(rmask & xmask, _tmp34 + tmp30, _tmp34)
        tmp37 = tmp35 + tmp36
        tmp39 = tmp37 * tmp38
        _tmp40 = tl.where(rmask & xmask, _tmp40 + tmp39, _tmp40)
        _tmp41 = tl.where(rmask & xmask, _tmp41 + tmp37, _tmp41)
        tmp44 = tmp42 + tmp43
        tmp46 = tmp44 * tmp45
        _tmp47 = tl.where(rmask & xmask, _tmp47 + tmp46, _tmp47)
        _tmp48 = tl.where(rmask & xmask, _tmp48 + tmp44, _tmp48)
        tmp51 = tmp49 + tmp50
        tmp53 = tmp51 * tmp52
        _tmp54 = tl.where(rmask & xmask, _tmp54 + tmp53, _tmp54)
        _tmp55 = tl.where(rmask & xmask, _tmp55 + tmp51, _tmp55)
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 * tmp59
        _tmp61 = tl.where(rmask & xmask, _tmp61 + tmp60, _tmp61)
        _tmp62 = tl.where(rmask & xmask, _tmp62 + tmp58, _tmp62)
        tmp65 = tmp63 + tmp64
        tmp67 = tmp65 * tmp66
        _tmp68 = tl.where(rmask & xmask, _tmp68 + tmp67, _tmp68)
        _tmp69 = tl.where(rmask & xmask, _tmp69 + tmp65, _tmp69)
        tmp72 = tmp70 + tmp71
        tmp74 = tmp72 * tmp73
        _tmp75 = tl.where(rmask & xmask, _tmp75 + tmp74, _tmp75)
        _tmp76 = tl.where(rmask & xmask, _tmp76 + tmp72, _tmp76)
        tmp79 = tmp77 + tmp78
        tmp81 = tmp79 * tmp80
        _tmp82 = tl.where(rmask & xmask, _tmp82 + tmp81, _tmp82)
        _tmp83 = tl.where(rmask & xmask, _tmp83 + tmp79, _tmp83)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp47 = tl.sum(_tmp47, 1)[:, None]
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tmp55 = tl.sum(_tmp55, 1)[:, None]
    tmp61 = tl.sum(_tmp61, 1)[:, None]
    tmp62 = tl.sum(_tmp62, 1)[:, None]
    tmp68 = tl.sum(_tmp68, 1)[:, None]
    tmp69 = tl.sum(_tmp69, 1)[:, None]
    tmp75 = tl.sum(_tmp75, 1)[:, None]
    tmp76 = tl.sum(_tmp76, 1)[:, None]
    tmp82 = tl.sum(_tmp82, 1)[:, None]
    tmp83 = tl.sum(_tmp83, 1)[:, None]
    tmp84 = tmp5 + tmp12
    tmp85 = tmp84 + tmp19
    tmp86 = tmp85 + tmp26
    tmp87 = tmp86 + tmp33
    tmp88 = tmp87 + tmp40
    tmp89 = tmp88 + tmp54
    tmp90 = tmp89 + tmp68
    tmp91 = tmp90 + tmp82
    tmp92 = tmp91 + tmp47
    tmp93 = tmp92 + tmp61
    tmp94 = tmp93 + tmp75
    tmp95 = tmp6 + tmp13
    tmp96 = tmp95 + tmp20
    tmp97 = tmp96 + tmp27
    tmp98 = tmp97 + tmp34
    tmp99 = tmp98 + tmp41
    tmp100 = tmp99 + tmp55
    tmp101 = tmp100 + tmp69
    tmp102 = tmp101 + tmp83
    tmp103 = tmp102 + tmp48
    tmp104 = tmp103 + tmp62
    tmp105 = tmp104 + tmp76
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp94, xmask)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp105, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]})
@triton.jit
def triton__18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp11 = tl.load(in_ptr5 + (x0), xmask)
    tmp13 = tl.load(in_ptr6 + (x0), xmask)
    tmp15 = tl.load(in_ptr7 + (x0), xmask)
    tmp17 = tl.load(in_ptr8 + (x0), xmask)
    tmp19 = tl.load(in_ptr9 + (x0), xmask)
    tmp21 = tl.load(in_ptr10 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]})
@triton.jit
def triton__19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp11 = tl.load(in_ptr5 + (x0), xmask)
    tmp13 = tl.load(in_ptr6 + (x0), xmask)
    tmp15 = tl.load(in_ptr7 + (x0), xmask)
    tmp17 = tl.load(in_ptr8 + (x0), xmask)
    tmp19 = tl.load(in_ptr9 + (x0), xmask)
    tmp21 = tl.load(in_ptr10 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__20(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
        tl.store(in_out_ptr0 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp0, rmask & xmask)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp2 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr1 + (x0), xmask)
    tmp7 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tl.load(in_ptr5 + (x0), xmask)
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    tmp19 = tl.load(in_ptr8 + (x0), xmask)
    tmp21 = tl.load(in_ptr9 + (x0), xmask)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 + tmp1
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 4],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__24(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__25(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4', 'out_ptr5'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        tmp5 = tmp2 * tmp4
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp18 = tl.load(in_ptr4 + (x0), xmask)
    tmp24 = tl.load(in_ptr5 + (x0), xmask)
    tmp28 = tl.load(in_ptr6 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp10 = tmp8 * tmp9
        tmp11 = 128.0
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12 - tmp3
        tmp15 = tmp14 * tmp6
        tmp16 = tmp13 - tmp15
        tmp17 = tmp7 * tmp16
        tmp19 = tmp18.to(tl.int64)
        tmp20 = -1
        tmp21 = tmp19 == tmp20
        tmp22 = 0.0
        tmp23 = tl.where(tmp21, tmp22, tmp17)
        tmp25 = tmp24.to(tl.int64)
        tmp26 = tmp25 == tmp20
        tmp27 = tl.where(tmp26, tmp22, tmp17)
        tmp29 = tmp28.to(tl.int64)
        tmp30 = 0
        tmp31 = tmp29 == tmp30
        tmp32 = tl.where(tmp31, tmp22, tmp17)
        tl.atomic_add(out_ptr3 + (r1 + (128*tmp19) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask & xmask)
        tl.atomic_add(out_ptr4 + (r1 + (128*tmp25) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp27, rmask & xmask)
        tl.atomic_add(out_ptr5 + (r1 + (128*tmp29) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__27(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask & xmask, _tmp3 + tmp2, _tmp3)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp0, _tmp4)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_16, primals_22, mul_1, view, view_2, div_1, view_17, mul_3, view_19, addmm_5, tanh, view_21, mul_9, view_23, div_3, view_38, mul_11, view_40, addmm_11, tanh_1, view_42, mul_17, view_44, div_5, view_59, mul_19, view_61, addmm_17, tanh_2, view_63, mul_25, view_65, div_7, view_80, mul_27, view_82, addmm_23, tanh_3, view_84, mul_33, view_86, div_9, view_101, mul_35, view_103, addmm_29, tanh_4, view_105, mul_41, view_107, div_11, view_122, mul_43, view_124, addmm_35, tanh_5, view_126, mul_49, view_128, div_13, view_143, mul_51, view_145, addmm_41, tanh_6, view_147, mul_57, view_149, div_15, view_164, mul_59, view_166, addmm_47, tanh_7, view_168, mul_65, view_170, div_17, view_185, mul_67, view_187, addmm_53, tanh_8, view_189, mul_73, view_191, div_19, view_206, mul_75, view_208, addmm_59, tanh_9, view_210, mul_81, view_212, div_21, view_227, mul_83, view_229, addmm_65, tanh_10, view_231, mul_89, view_233, div_23, view_248, mul_91, view_250, addmm_71, tanh_11, view_252, mul_97, view_254, sub_39, unsqueeze_2, ne, sub_41, unsqueeze_3, ne_2, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, permute_186, permute_187, div_36, div_37, permute_217, permute_218, permute_219, permute_220, div_39, div_40, permute_250, permute_251, permute_252, permute_253, div_42, div_43, permute_283, permute_284, permute_285, permute_286, div_45, div_46, permute_316, permute_317, permute_318, permute_319, div_48, div_49, permute_349, permute_350, permute_351, permute_352, div_51, div_52, permute_382, permute_383, permute_384, permute_385, div_54, div_55, permute_415, permute_416, permute_417, permute_418, div_57, div_58, permute_448, permute_449, permute_450, permute_451, div_60, div_61, permute_481, permute_482, permute_483, permute_484, div_63, div_64, permute_514, permute_515, permute_516, permute_517, permute_534, div_66, convert_element_type_2, convert_element_type_4, convert_element_type_6, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton__0.run(buf0, 512, grid=grid(512), stream=stream0)
        triton__1.run(unsqueeze_3, buf0, 1, grid=grid(1), stream=stream0)
        buf3 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__0.run(buf3, 512, grid=grid(512), stream=stream0)
        triton__1.run(unsqueeze_2, buf3, 1, grid=grid(1), stream=stream0)
        buf8 = empty_strided((1, 512, 2), (1024, 2, 1), device='cuda', dtype=torch.float32)
        buf6 = as_strided(buf8, (1, 512, 1), (1024, 2, 1))  # alias
        buf7 = as_strided(buf8, (1, 512, 1), (1024, 2, 1), 1)  # alias
        triton__2.run(buf0, unsqueeze_3, tangents_1, ne_2, buf3, unsqueeze_2, ne, tangents_2, sub_39, tangents_3, sub_41, buf6, buf7, 1, 512, grid=grid(1), stream=stream0)
        del buf0
        del buf3
        del ne
        del ne_2
        del sub_39
        del sub_41
        del tangents_1
        del tangents_2
        del tangents_3
        del unsqueeze_2
        del unsqueeze_3
        del buf6
        del buf7
        buf9 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf8, (512, 2), (2, 1)), permute_134, out=buf9)
        del permute_134
        buf10 = empty_strided((2, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf8, (2, 512), (1, 2)), view_254, out=buf10)
        del view_254
        buf11 = empty_strided((1, 2, 4), (8, 1, 2), device='cuda', dtype=torch.float32)
        triton__3.run(buf8, buf11, 8, 128, grid=grid(8), stream=stream0)
        del buf8
        buf12 = empty_strided((1, 2), (2, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf11, buf12, 2, 4, grid=grid(2), stream=stream0)
        del buf11
        buf15 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__5.run(buf9, primals_22, mul_97, div_30, buf15, 512, 4096, grid=grid(512), stream=stream0)
        del div_30
        buf18 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf15, (512, 4096), (4096, 1)), permute_138, out=buf18)
        buf22 = as_strided(buf18, (1, 512, 16384), (8388608, 16384, 1)); del buf18  # reuse
        buf23 = as_strided(buf22, (512, 16384), (16384, 1)); del buf22  # reuse
        triton__6.run(buf23, addmm_71, tanh_11, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_71
        del tanh_11
        buf24 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf23, permute_142, out=buf24)
        buf29 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf15, buf24, primals_16, mul_91, div_31, buf29, 512, 4096, grid=grid(512), stream=stream0)
        del div_31
        buf32 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf29, (512, 4096), (4096, 1)), permute_146, out=buf32)
        buf36 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_151, as_strided(buf32, (64, 512, 64), (64, 4096, 1)), out=buf36)
        del permute_151
        buf42 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf36, buf42, 2097152, grid=grid(2097152), stream=stream0)
        buf43 = as_strided(buf36, (512, 4096), (4096, 1)); del buf36  # reuse
        extern_kernels.mm(buf42, permute_159, out=buf43)
        buf37 = empty_strided((64, 512, 512), (262144, 512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf32, (64, 512, 64), (64, 4096, 1)), permute_152, out=buf37)
        del permute_152
        buf39 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf37, div_23, buf39, 32768, 512, grid=grid(32768), stream=stream0)
        del div_23
        buf40 = as_strided(buf32, (64, 64, 512), (32768, 512, 1)); del buf32  # reuse
        extern_kernels.bmm(permute_153, as_strided(buf39, (64, 512, 512), (262144, 512, 1)), out=buf40)
        del permute_153
        buf47 = as_strided(buf40, (512, 4096), (1, 512)); del buf40  # reuse
        buf50 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf47, buf50, 4096, 512, grid=grid(4096), stream=stream0)
        buf48 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf47, permute_163, out=buf48)
        buf41 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf39, (64, 512, 512), (262144, 512, 1)), permute_154, out=buf41)
        del permute_154
        buf51 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf41, buf51, 2097152, grid=grid(2097152), stream=stream0)
        buf52 = as_strided(buf41, (512, 4096), (4096, 1)); del buf41  # reuse
        extern_kernels.mm(buf51, permute_167, out=buf52)
        buf56 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf29, buf43, buf48, buf52, primals_22, mul_89, div_33, buf56, buf59, 512, 4096, grid=grid(512), stream=stream0)
        del div_33
        buf62 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf59, (512, 4096), (4096, 1)), permute_138, out=buf62)
        buf66 = as_strided(buf62, (1, 512, 16384), (8388608, 16384, 1)); del buf62  # reuse
        buf67 = as_strided(buf66, (512, 16384), (16384, 1)); del buf66  # reuse
        triton__6.run(buf67, addmm_65, tanh_10, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_65
        del tanh_10
        buf68 = as_strided(buf56, (512, 4096), (4096, 1)); del buf56  # reuse
        extern_kernels.mm(buf67, permute_142, out=buf68)
        buf73 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf59, buf68, primals_16, mul_83, div_34, buf73, 512, 4096, grid=grid(512), stream=stream0)
        del div_34
        buf76 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf73, (512, 4096), (4096, 1)), permute_146, out=buf76)
        buf80 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_184, as_strided(buf76, (64, 512, 64), (64, 4096, 1)), out=buf80)
        del permute_184
        buf86 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf80, buf86, 2097152, grid=grid(2097152), stream=stream0)
        buf87 = as_strided(buf80, (512, 4096), (4096, 1)); del buf80  # reuse
        extern_kernels.mm(buf86, permute_159, out=buf87)
        buf81 = as_strided(buf39, (64, 512, 512), (262144, 512, 1)); del buf39  # reuse
        extern_kernels.bmm(as_strided(buf76, (64, 512, 64), (64, 4096, 1)), permute_185, out=buf81)
        del permute_185
        buf83 = as_strided(buf37, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf37  # reuse
        triton__9.run(buf81, div_21, buf83, 32768, 512, grid=grid(32768), stream=stream0)
        del div_21
        buf84 = as_strided(buf76, (64, 64, 512), (32768, 512, 1)); del buf76  # reuse
        extern_kernels.bmm(permute_186, as_strided(buf83, (64, 512, 512), (262144, 512, 1)), out=buf84)
        del permute_186
        buf91 = as_strided(buf84, (512, 4096), (1, 512)); del buf84  # reuse
        buf94 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf91, buf94, 4096, 512, grid=grid(4096), stream=stream0)
        buf92 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf91, permute_163, out=buf92)
        buf85 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf83, (64, 512, 512), (262144, 512, 1)), permute_187, out=buf85)
        del permute_187
        buf95 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf85, buf95, 2097152, grid=grid(2097152), stream=stream0)
        buf96 = as_strided(buf85, (512, 4096), (4096, 1)); del buf85  # reuse
        extern_kernels.mm(buf95, permute_167, out=buf96)
        buf100 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf103 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf73, buf87, buf92, buf96, primals_22, mul_81, div_36, buf100, buf103, 512, 4096, grid=grid(512), stream=stream0)
        del div_36
        buf106 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf103, (512, 4096), (4096, 1)), permute_138, out=buf106)
        buf110 = as_strided(buf106, (1, 512, 16384), (8388608, 16384, 1)); del buf106  # reuse
        buf111 = as_strided(buf110, (512, 16384), (16384, 1)); del buf110  # reuse
        triton__6.run(buf111, addmm_59, tanh_9, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_59
        del tanh_9
        buf112 = as_strided(buf100, (512, 4096), (4096, 1)); del buf100  # reuse
        extern_kernels.mm(buf111, permute_142, out=buf112)
        buf117 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf103, buf112, primals_16, mul_75, div_37, buf117, 512, 4096, grid=grid(512), stream=stream0)
        del div_37
        buf120 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf117, (512, 4096), (4096, 1)), permute_146, out=buf120)
        buf124 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_217, as_strided(buf120, (64, 512, 64), (64, 4096, 1)), out=buf124)
        del permute_217
        buf130 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf124, buf130, 2097152, grid=grid(2097152), stream=stream0)
        buf131 = as_strided(buf124, (512, 4096), (4096, 1)); del buf124  # reuse
        extern_kernels.mm(buf130, permute_159, out=buf131)
        buf125 = as_strided(buf83, (64, 512, 512), (262144, 512, 1)); del buf83  # reuse
        extern_kernels.bmm(as_strided(buf120, (64, 512, 64), (64, 4096, 1)), permute_218, out=buf125)
        del permute_218
        buf127 = as_strided(buf81, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf81  # reuse
        triton__9.run(buf125, div_19, buf127, 32768, 512, grid=grid(32768), stream=stream0)
        del div_19
        buf128 = as_strided(buf120, (64, 64, 512), (32768, 512, 1)); del buf120  # reuse
        extern_kernels.bmm(permute_219, as_strided(buf127, (64, 512, 512), (262144, 512, 1)), out=buf128)
        del permute_219
        buf135 = as_strided(buf128, (512, 4096), (1, 512)); del buf128  # reuse
        buf138 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf135, buf138, 4096, 512, grid=grid(4096), stream=stream0)
        buf136 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf135, permute_163, out=buf136)
        buf129 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf127, (64, 512, 512), (262144, 512, 1)), permute_220, out=buf129)
        del permute_220
        buf139 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf129, buf139, 2097152, grid=grid(2097152), stream=stream0)
        buf140 = as_strided(buf129, (512, 4096), (4096, 1)); del buf129  # reuse
        extern_kernels.mm(buf139, permute_167, out=buf140)
        buf144 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf147 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf117, buf131, buf136, buf140, primals_22, mul_73, div_39, buf144, buf147, 512, 4096, grid=grid(512), stream=stream0)
        del div_39
        buf150 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf147, (512, 4096), (4096, 1)), permute_138, out=buf150)
        buf154 = as_strided(buf150, (1, 512, 16384), (8388608, 16384, 1)); del buf150  # reuse
        buf155 = as_strided(buf154, (512, 16384), (16384, 1)); del buf154  # reuse
        triton__6.run(buf155, addmm_53, tanh_8, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_53
        del tanh_8
        buf156 = as_strided(buf144, (512, 4096), (4096, 1)); del buf144  # reuse
        extern_kernels.mm(buf155, permute_142, out=buf156)
        buf161 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf147, buf156, primals_16, mul_67, div_40, buf161, 512, 4096, grid=grid(512), stream=stream0)
        del div_40
        buf164 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf161, (512, 4096), (4096, 1)), permute_146, out=buf164)
        buf168 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_250, as_strided(buf164, (64, 512, 64), (64, 4096, 1)), out=buf168)
        del permute_250
        buf174 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf168, buf174, 2097152, grid=grid(2097152), stream=stream0)
        buf175 = as_strided(buf168, (512, 4096), (4096, 1)); del buf168  # reuse
        extern_kernels.mm(buf174, permute_159, out=buf175)
        buf169 = as_strided(buf127, (64, 512, 512), (262144, 512, 1)); del buf127  # reuse
        extern_kernels.bmm(as_strided(buf164, (64, 512, 64), (64, 4096, 1)), permute_251, out=buf169)
        del permute_251
        buf171 = as_strided(buf125, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf125  # reuse
        triton__9.run(buf169, div_17, buf171, 32768, 512, grid=grid(32768), stream=stream0)
        del div_17
        buf172 = as_strided(buf164, (64, 64, 512), (32768, 512, 1)); del buf164  # reuse
        extern_kernels.bmm(permute_252, as_strided(buf171, (64, 512, 512), (262144, 512, 1)), out=buf172)
        del permute_252
        buf179 = as_strided(buf172, (512, 4096), (1, 512)); del buf172  # reuse
        buf182 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf179, buf182, 4096, 512, grid=grid(4096), stream=stream0)
        buf180 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf179, permute_163, out=buf180)
        buf173 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf171, (64, 512, 512), (262144, 512, 1)), permute_253, out=buf173)
        del permute_253
        buf183 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf173, buf183, 2097152, grid=grid(2097152), stream=stream0)
        buf184 = as_strided(buf173, (512, 4096), (4096, 1)); del buf173  # reuse
        extern_kernels.mm(buf183, permute_167, out=buf184)
        buf188 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf191 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf161, buf175, buf180, buf184, primals_22, mul_65, div_42, buf188, buf191, 512, 4096, grid=grid(512), stream=stream0)
        del div_42
        buf194 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf191, (512, 4096), (4096, 1)), permute_138, out=buf194)
        buf198 = as_strided(buf194, (1, 512, 16384), (8388608, 16384, 1)); del buf194  # reuse
        buf199 = as_strided(buf198, (512, 16384), (16384, 1)); del buf198  # reuse
        triton__6.run(buf199, addmm_47, tanh_7, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_47
        del tanh_7
        buf200 = as_strided(buf188, (512, 4096), (4096, 1)); del buf188  # reuse
        extern_kernels.mm(buf199, permute_142, out=buf200)
        buf205 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf191, buf200, primals_16, mul_59, div_43, buf205, 512, 4096, grid=grid(512), stream=stream0)
        del div_43
        buf208 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf205, (512, 4096), (4096, 1)), permute_146, out=buf208)
        buf212 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_283, as_strided(buf208, (64, 512, 64), (64, 4096, 1)), out=buf212)
        del permute_283
        buf218 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf212, buf218, 2097152, grid=grid(2097152), stream=stream0)
        buf219 = as_strided(buf212, (512, 4096), (4096, 1)); del buf212  # reuse
        extern_kernels.mm(buf218, permute_159, out=buf219)
        buf213 = as_strided(buf171, (64, 512, 512), (262144, 512, 1)); del buf171  # reuse
        extern_kernels.bmm(as_strided(buf208, (64, 512, 64), (64, 4096, 1)), permute_284, out=buf213)
        del permute_284
        buf215 = as_strided(buf169, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf169  # reuse
        triton__9.run(buf213, div_15, buf215, 32768, 512, grid=grid(32768), stream=stream0)
        del div_15
        buf216 = as_strided(buf208, (64, 64, 512), (32768, 512, 1)); del buf208  # reuse
        extern_kernels.bmm(permute_285, as_strided(buf215, (64, 512, 512), (262144, 512, 1)), out=buf216)
        del permute_285
        buf223 = as_strided(buf216, (512, 4096), (1, 512)); del buf216  # reuse
        buf226 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf223, buf226, 4096, 512, grid=grid(4096), stream=stream0)
        buf224 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf223, permute_163, out=buf224)
        buf217 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf215, (64, 512, 512), (262144, 512, 1)), permute_286, out=buf217)
        del permute_286
        buf227 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf217, buf227, 2097152, grid=grid(2097152), stream=stream0)
        buf228 = as_strided(buf217, (512, 4096), (4096, 1)); del buf217  # reuse
        extern_kernels.mm(buf227, permute_167, out=buf228)
        buf232 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf235 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf205, buf219, buf224, buf228, primals_22, mul_57, div_45, buf232, buf235, 512, 4096, grid=grid(512), stream=stream0)
        del div_45
        buf238 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf235, (512, 4096), (4096, 1)), permute_138, out=buf238)
        buf242 = as_strided(buf238, (1, 512, 16384), (8388608, 16384, 1)); del buf238  # reuse
        buf243 = as_strided(buf242, (512, 16384), (16384, 1)); del buf242  # reuse
        triton__6.run(buf243, addmm_41, tanh_6, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_41
        del tanh_6
        buf244 = as_strided(buf232, (512, 4096), (4096, 1)); del buf232  # reuse
        extern_kernels.mm(buf243, permute_142, out=buf244)
        buf249 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf235, buf244, primals_16, mul_51, div_46, buf249, 512, 4096, grid=grid(512), stream=stream0)
        del div_46
        buf252 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf249, (512, 4096), (4096, 1)), permute_146, out=buf252)
        buf256 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_316, as_strided(buf252, (64, 512, 64), (64, 4096, 1)), out=buf256)
        del permute_316
        buf262 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf256, buf262, 2097152, grid=grid(2097152), stream=stream0)
        buf263 = as_strided(buf256, (512, 4096), (4096, 1)); del buf256  # reuse
        extern_kernels.mm(buf262, permute_159, out=buf263)
        buf257 = as_strided(buf215, (64, 512, 512), (262144, 512, 1)); del buf215  # reuse
        extern_kernels.bmm(as_strided(buf252, (64, 512, 64), (64, 4096, 1)), permute_317, out=buf257)
        del permute_317
        buf259 = as_strided(buf213, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf213  # reuse
        triton__9.run(buf257, div_13, buf259, 32768, 512, grid=grid(32768), stream=stream0)
        del div_13
        buf260 = as_strided(buf252, (64, 64, 512), (32768, 512, 1)); del buf252  # reuse
        extern_kernels.bmm(permute_318, as_strided(buf259, (64, 512, 512), (262144, 512, 1)), out=buf260)
        del permute_318
        buf267 = as_strided(buf260, (512, 4096), (1, 512)); del buf260  # reuse
        buf270 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf267, buf270, 4096, 512, grid=grid(4096), stream=stream0)
        buf268 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf267, permute_163, out=buf268)
        buf261 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf259, (64, 512, 512), (262144, 512, 1)), permute_319, out=buf261)
        del permute_319
        buf271 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf261, buf271, 2097152, grid=grid(2097152), stream=stream0)
        buf272 = as_strided(buf261, (512, 4096), (4096, 1)); del buf261  # reuse
        extern_kernels.mm(buf271, permute_167, out=buf272)
        buf276 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf279 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf249, buf263, buf268, buf272, primals_22, mul_49, div_48, buf276, buf279, 512, 4096, grid=grid(512), stream=stream0)
        del div_48
        buf282 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf279, (512, 4096), (4096, 1)), permute_138, out=buf282)
        buf286 = as_strided(buf282, (1, 512, 16384), (8388608, 16384, 1)); del buf282  # reuse
        buf287 = as_strided(buf286, (512, 16384), (16384, 1)); del buf286  # reuse
        triton__6.run(buf287, addmm_35, tanh_5, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_35
        del tanh_5
        buf288 = as_strided(buf276, (512, 4096), (4096, 1)); del buf276  # reuse
        extern_kernels.mm(buf287, permute_142, out=buf288)
        buf293 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf279, buf288, primals_16, mul_43, div_49, buf293, 512, 4096, grid=grid(512), stream=stream0)
        del div_49
        buf296 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf293, (512, 4096), (4096, 1)), permute_146, out=buf296)
        buf300 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_349, as_strided(buf296, (64, 512, 64), (64, 4096, 1)), out=buf300)
        del permute_349
        buf306 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf300, buf306, 2097152, grid=grid(2097152), stream=stream0)
        buf307 = as_strided(buf300, (512, 4096), (4096, 1)); del buf300  # reuse
        extern_kernels.mm(buf306, permute_159, out=buf307)
        buf301 = as_strided(buf259, (64, 512, 512), (262144, 512, 1)); del buf259  # reuse
        extern_kernels.bmm(as_strided(buf296, (64, 512, 64), (64, 4096, 1)), permute_350, out=buf301)
        del permute_350
        buf303 = as_strided(buf257, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf257  # reuse
        triton__9.run(buf301, div_11, buf303, 32768, 512, grid=grid(32768), stream=stream0)
        del div_11
        buf304 = as_strided(buf296, (64, 64, 512), (32768, 512, 1)); del buf296  # reuse
        extern_kernels.bmm(permute_351, as_strided(buf303, (64, 512, 512), (262144, 512, 1)), out=buf304)
        del permute_351
        buf311 = as_strided(buf304, (512, 4096), (1, 512)); del buf304  # reuse
        buf314 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf311, buf314, 4096, 512, grid=grid(4096), stream=stream0)
        buf312 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf311, permute_163, out=buf312)
        buf305 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf303, (64, 512, 512), (262144, 512, 1)), permute_352, out=buf305)
        del permute_352
        buf315 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf305, buf315, 2097152, grid=grid(2097152), stream=stream0)
        buf316 = as_strided(buf305, (512, 4096), (4096, 1)); del buf305  # reuse
        extern_kernels.mm(buf315, permute_167, out=buf316)
        buf320 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf323 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf293, buf307, buf312, buf316, primals_22, mul_41, div_51, buf320, buf323, 512, 4096, grid=grid(512), stream=stream0)
        del div_51
        buf326 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf323, (512, 4096), (4096, 1)), permute_138, out=buf326)
        buf330 = as_strided(buf326, (1, 512, 16384), (8388608, 16384, 1)); del buf326  # reuse
        buf331 = as_strided(buf330, (512, 16384), (16384, 1)); del buf330  # reuse
        triton__6.run(buf331, addmm_29, tanh_4, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_29
        del tanh_4
        buf332 = as_strided(buf320, (512, 4096), (4096, 1)); del buf320  # reuse
        extern_kernels.mm(buf331, permute_142, out=buf332)
        buf337 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf323, buf332, primals_16, mul_35, div_52, buf337, 512, 4096, grid=grid(512), stream=stream0)
        del div_52
        buf340 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf337, (512, 4096), (4096, 1)), permute_146, out=buf340)
        buf344 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_382, as_strided(buf340, (64, 512, 64), (64, 4096, 1)), out=buf344)
        del permute_382
        buf350 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf344, buf350, 2097152, grid=grid(2097152), stream=stream0)
        buf351 = as_strided(buf344, (512, 4096), (4096, 1)); del buf344  # reuse
        extern_kernels.mm(buf350, permute_159, out=buf351)
        buf345 = as_strided(buf303, (64, 512, 512), (262144, 512, 1)); del buf303  # reuse
        extern_kernels.bmm(as_strided(buf340, (64, 512, 64), (64, 4096, 1)), permute_383, out=buf345)
        del permute_383
        buf347 = as_strided(buf301, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf301  # reuse
        triton__9.run(buf345, div_9, buf347, 32768, 512, grid=grid(32768), stream=stream0)
        del div_9
        buf348 = as_strided(buf340, (64, 64, 512), (32768, 512, 1)); del buf340  # reuse
        extern_kernels.bmm(permute_384, as_strided(buf347, (64, 512, 512), (262144, 512, 1)), out=buf348)
        del permute_384
        buf355 = as_strided(buf348, (512, 4096), (1, 512)); del buf348  # reuse
        buf358 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf355, buf358, 4096, 512, grid=grid(4096), stream=stream0)
        buf356 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf355, permute_163, out=buf356)
        buf349 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf347, (64, 512, 512), (262144, 512, 1)), permute_385, out=buf349)
        del permute_385
        buf359 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf349, buf359, 2097152, grid=grid(2097152), stream=stream0)
        buf360 = as_strided(buf349, (512, 4096), (4096, 1)); del buf349  # reuse
        extern_kernels.mm(buf359, permute_167, out=buf360)
        buf364 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf367 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf337, buf351, buf356, buf360, primals_22, mul_33, div_54, buf364, buf367, 512, 4096, grid=grid(512), stream=stream0)
        del div_54
        buf372 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf367, (512, 4096), (4096, 1)), permute_138, out=buf372)
        buf378 = as_strided(buf372, (1, 512, 16384), (8388608, 16384, 1)); del buf372  # reuse
        buf379 = as_strided(buf378, (512, 16384), (16384, 1)); del buf378  # reuse
        triton__6.run(buf379, addmm_23, tanh_3, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_23
        del tanh_3
        buf380 = as_strided(buf364, (512, 4096), (4096, 1)); del buf364  # reuse
        extern_kernels.mm(buf379, permute_142, out=buf380)
        buf387 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf367, buf380, primals_16, mul_27, div_55, buf387, 512, 4096, grid=grid(512), stream=stream0)
        del div_55
        buf392 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf387, (512, 4096), (4096, 1)), permute_146, out=buf392)
        buf398 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_415, as_strided(buf392, (64, 512, 64), (64, 4096, 1)), out=buf398)
        del permute_415
        buf404 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf398, buf404, 2097152, grid=grid(2097152), stream=stream0)
        buf405 = as_strided(buf398, (512, 4096), (4096, 1)); del buf398  # reuse
        extern_kernels.mm(buf404, permute_159, out=buf405)
        buf399 = as_strided(buf347, (64, 512, 512), (262144, 512, 1)); del buf347  # reuse
        extern_kernels.bmm(as_strided(buf392, (64, 512, 64), (64, 4096, 1)), permute_416, out=buf399)
        del permute_416
        buf401 = as_strided(buf345, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf345  # reuse
        triton__9.run(buf399, div_7, buf401, 32768, 512, grid=grid(32768), stream=stream0)
        del div_7
        buf402 = as_strided(buf392, (64, 64, 512), (32768, 512, 1)); del buf392  # reuse
        extern_kernels.bmm(permute_417, as_strided(buf401, (64, 512, 512), (262144, 512, 1)), out=buf402)
        del permute_417
        buf411 = as_strided(buf402, (512, 4096), (1, 512)); del buf402  # reuse
        buf414 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf411, buf414, 4096, 512, grid=grid(4096), stream=stream0)
        buf412 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf411, permute_163, out=buf412)
        buf403 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf401, (64, 512, 512), (262144, 512, 1)), permute_418, out=buf403)
        del permute_418
        buf417 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf403, buf417, 2097152, grid=grid(2097152), stream=stream0)
        buf418 = as_strided(buf403, (512, 4096), (4096, 1)); del buf403  # reuse
        extern_kernels.mm(buf417, permute_167, out=buf418)
        buf424 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf427 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf387, buf405, buf412, buf418, primals_22, mul_25, div_57, buf424, buf427, 512, 4096, grid=grid(512), stream=stream0)
        del div_57
        buf430 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf427, (512, 4096), (4096, 1)), permute_138, out=buf430)
        buf434 = as_strided(buf430, (1, 512, 16384), (8388608, 16384, 1)); del buf430  # reuse
        buf435 = as_strided(buf434, (512, 16384), (16384, 1)); del buf434  # reuse
        triton__6.run(buf435, addmm_17, tanh_2, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_17
        del tanh_2
        buf436 = as_strided(buf424, (512, 4096), (4096, 1)); del buf424  # reuse
        extern_kernels.mm(buf435, permute_142, out=buf436)
        buf441 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf427, buf436, primals_16, mul_19, div_58, buf441, 512, 4096, grid=grid(512), stream=stream0)
        del div_58
        buf444 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf441, (512, 4096), (4096, 1)), permute_146, out=buf444)
        buf448 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_448, as_strided(buf444, (64, 512, 64), (64, 4096, 1)), out=buf448)
        del permute_448
        buf454 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf448, buf454, 2097152, grid=grid(2097152), stream=stream0)
        buf455 = as_strided(buf448, (512, 4096), (4096, 1)); del buf448  # reuse
        extern_kernels.mm(buf454, permute_159, out=buf455)
        buf449 = as_strided(buf401, (64, 512, 512), (262144, 512, 1)); del buf401  # reuse
        extern_kernels.bmm(as_strided(buf444, (64, 512, 64), (64, 4096, 1)), permute_449, out=buf449)
        del permute_449
        buf451 = as_strided(buf399, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf399  # reuse
        triton__9.run(buf449, div_5, buf451, 32768, 512, grid=grid(32768), stream=stream0)
        del div_5
        buf452 = as_strided(buf444, (64, 64, 512), (32768, 512, 1)); del buf444  # reuse
        extern_kernels.bmm(permute_450, as_strided(buf451, (64, 512, 512), (262144, 512, 1)), out=buf452)
        del permute_450
        buf459 = as_strided(buf452, (512, 4096), (1, 512)); del buf452  # reuse
        buf462 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf459, buf462, 4096, 512, grid=grid(4096), stream=stream0)
        buf460 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf459, permute_163, out=buf460)
        buf453 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf451, (64, 512, 512), (262144, 512, 1)), permute_451, out=buf453)
        del permute_451
        buf463 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf453, buf463, 2097152, grid=grid(2097152), stream=stream0)
        buf464 = as_strided(buf453, (512, 4096), (4096, 1)); del buf453  # reuse
        extern_kernels.mm(buf463, permute_167, out=buf464)
        buf468 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        buf471 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__11.run(buf441, buf455, buf460, buf464, primals_22, mul_17, div_60, buf468, buf471, 512, 4096, grid=grid(512), stream=stream0)
        del div_60
        buf474 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf471, (512, 4096), (4096, 1)), permute_138, out=buf474)
        buf478 = as_strided(buf474, (1, 512, 16384), (8388608, 16384, 1)); del buf474  # reuse
        buf479 = as_strided(buf478, (512, 16384), (16384, 1)); del buf478  # reuse
        triton__6.run(buf479, addmm_11, tanh_1, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_11
        del tanh_1
        buf480 = as_strided(buf468, (512, 4096), (4096, 1)); del buf468  # reuse
        extern_kernels.mm(buf479, permute_142, out=buf480)
        buf485 = empty_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf471, buf480, primals_16, mul_11, div_61, buf485, 512, 4096, grid=grid(512), stream=stream0)
        del div_61
        buf488 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf485, (512, 4096), (4096, 1)), permute_146, out=buf488)
        buf492 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(permute_481, as_strided(buf488, (64, 512, 64), (64, 4096, 1)), out=buf492)
        del permute_481
        buf498 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf492, buf498, 2097152, grid=grid(2097152), stream=stream0)
        buf499 = as_strided(buf492, (512, 4096), (4096, 1)); del buf492  # reuse
        extern_kernels.mm(buf498, permute_159, out=buf499)
        buf493 = as_strided(buf451, (64, 512, 512), (262144, 512, 1)); del buf451  # reuse
        extern_kernels.bmm(as_strided(buf488, (64, 512, 64), (64, 4096, 1)), permute_482, out=buf493)
        del permute_482
        buf495 = as_strided(buf449, (1, 64, 512, 512), (16777216, 262144, 512, 1)); del buf449  # reuse
        triton__9.run(buf493, div_3, buf495, 32768, 512, grid=grid(32768), stream=stream0)
        del div_3
        buf496 = as_strided(buf488, (64, 64, 512), (32768, 512, 1)); del buf488  # reuse
        extern_kernels.bmm(permute_483, as_strided(buf495, (64, 512, 512), (262144, 512, 1)), out=buf496)
        del permute_483
        buf503 = as_strided(buf496, (512, 4096), (1, 512)); del buf496  # reuse
        buf506 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__10.run(buf503, buf506, 4096, 512, grid=grid(4096), stream=stream0)
        buf504 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf503, permute_163, out=buf504)
        buf497 = empty_strided((64, 512, 64), (32768, 64, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf495, (64, 512, 512), (262144, 512, 1)), permute_484, out=buf497)
        del permute_484
        buf507 = empty_strided((512, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        triton__8.run(buf497, buf507, 2097152, grid=grid(2097152), stream=stream0)
        buf508 = as_strided(buf497, (512, 4096), (4096, 1)); del buf497  # reuse
        extern_kernels.mm(buf507, permute_167, out=buf508)
        buf104 = empty_strided((4096, ), (1, ), device='cuda', dtype=torch.float32)
        buf105 = empty_strided((4096, ), (1, ), device='cuda', dtype=torch.float32)
        buf370 = buf104; del buf104  # reuse
        buf518 = buf370; del buf370  # reuse
        buf371 = buf105; del buf105  # reuse
        buf519 = buf371; del buf371  # reuse
        triton__12.run(buf518, buf519, buf9, mul_97, buf29, buf43, buf48, buf52, mul_89, buf73, buf87, buf92, buf96, mul_81, buf117, buf131, buf136, buf140, mul_73, buf161, buf175, buf180, buf184, mul_65, buf205, buf219, buf224, buf228, mul_57, buf387, buf405, buf412, buf418, mul_25, buf249, buf263, buf268, buf272, mul_49, buf441, buf455, buf460, buf464, mul_17, buf293, buf307, buf312, buf316, mul_41, buf485, buf499, buf504, buf508, mul_9, buf337, buf351, buf356, buf360, mul_33, 4096, 512, grid=grid(4096), stream=stream0)
        del buf131
        del buf136
        del buf140
        del buf175
        del buf180
        del buf184
        del buf219
        del buf224
        del buf228
        del buf263
        del buf268
        del buf272
        del buf307
        del buf312
        del buf316
        del buf351
        del buf356
        del buf360
        del buf405
        del buf412
        del buf418
        del buf43
        del buf455
        del buf460
        del buf464
        del buf48
        del buf52
        del buf87
        del buf9
        del buf92
        del mul_17
        del mul_25
        del mul_33
        del mul_41
        del mul_49
        del mul_57
        del mul_65
        del mul_73
        del mul_81
        del mul_89
        del mul_97
        buf19 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf15, (4096, 512), (1, 4096)), view_252, out=buf19)
        del view_252
        buf20 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf15, buf20, 16384, 128, grid=grid(16384), stream=stream0)
        buf108 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf103, buf108, 16384, 128, grid=grid(16384), stream=stream0)
        buf152 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf147, buf152, 16384, 128, grid=grid(16384), stream=stream0)
        buf196 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf191, buf196, 16384, 128, grid=grid(16384), stream=stream0)
        buf240 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf235, buf240, 16384, 128, grid=grid(16384), stream=stream0)
        buf284 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf279, buf284, 16384, 128, grid=grid(16384), stream=stream0)
        buf328 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf323, buf328, 16384, 128, grid=grid(16384), stream=stream0)
        buf374 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf367, buf374, 16384, 128, grid=grid(16384), stream=stream0)
        buf432 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf427, buf432, 16384, 128, grid=grid(16384), stream=stream0)
        buf476 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf471, buf476, 16384, 128, grid=grid(16384), stream=stream0)
        buf512 = as_strided(buf499, (1, 512, 4096), (2097152, 4096, 1)); del buf499  # reuse
        buf515 = as_strided(buf96, (1, 512, 4096), (2097152, 4096, 1)); del buf96  # reuse
        triton__14.run(buf512, buf485, buf504, buf508, primals_22, mul_9, div_63, buf515, 512, 4096, grid=grid(512), stream=stream0)
        del buf504
        del buf508
        del div_63
        del mul_9
        del primals_22
        buf522 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf515, buf522, 16384, 128, grid=grid(16384), stream=stream0)
        buf64 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf59, buf64, 16384, 128, grid=grid(16384), stream=stream0)
        buf109 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf376 = as_strided(buf109, (4096, ), (1, )); del buf109  # reuse
        buf524 = buf376; del buf376  # reuse
        triton__15.run(buf524, buf20, buf64, buf108, buf152, buf196, buf240, buf432, buf284, buf476, buf328, buf522, buf374, 4096, 4, grid=grid(4096), stream=stream0)
        buf25 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf23, (16384, 512), (1, 16384)), view_250, out=buf25)
        del view_250
        buf520 = empty_strided((512, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf515, (512, 4096), (4096, 1)), permute_138, out=buf520)
        del permute_138
        buf526 = as_strided(buf520, (1, 512, 16384), (8388608, 16384, 1)); del buf520  # reuse
        buf527 = as_strided(buf526, (512, 16384), (16384, 1)); del buf526  # reuse
        triton__6.run(buf527, addmm_5, tanh, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_5
        del tanh
        buf114 = as_strided(buf64, (1, 16384), (16384, 1)); del buf64  # reuse
        buf383 = as_strided(buf114, (16384, ), (1, )); del buf114  # reuse
        buf531 = buf383; del buf383  # reuse
        triton__16.run(buf531, buf23, buf67, buf111, buf155, buf199, buf243, buf435, buf287, buf479, buf331, buf527, buf379, 16384, 512, grid=grid(16384), stream=stream0)
        del buf23
        buf528 = as_strided(buf512, (512, 4096), (4096, 1)); del buf512  # reuse
        extern_kernels.mm(buf527, permute_142, out=buf528)
        del permute_142
        buf118 = empty_strided((4096, ), (1, ), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((4096, ), (1, ), device='cuda', dtype=torch.float32)
        buf390 = buf118; del buf118  # reuse
        buf538 = buf390; del buf390  # reuse
        buf391 = buf119; del buf119  # reuse
        buf539 = buf391; del buf391  # reuse
        triton__17.run(buf538, buf539, buf15, buf24, mul_91, buf59, buf68, mul_83, buf103, buf112, mul_75, buf147, buf156, mul_67, buf191, buf200, mul_59, buf235, buf244, mul_51, buf427, buf436, mul_19, buf279, buf288, mul_43, buf471, buf480, mul_11, buf323, buf332, mul_35, buf515, buf528, mul_3, buf367, buf380, mul_27, 4096, 512, grid=grid(4096), stream=stream0)
        del buf112
        del buf15
        del buf156
        del buf200
        del buf24
        del buf244
        del buf288
        del buf332
        del buf380
        del buf436
        del buf480
        del mul_11
        del mul_19
        del mul_27
        del mul_35
        del mul_43
        del mul_51
        del mul_59
        del mul_67
        del mul_75
        del mul_83
        del mul_91
        buf33 = as_strided(buf495, (4096, 4096), (4096, 1)); del buf495  # reuse
        extern_kernels.mm(as_strided(buf29, (4096, 512), (1, 4096)), view_248, out=buf33)
        del view_248
        buf34 = buf522; del buf522  # reuse
        triton__13.run(buf29, buf34, 16384, 128, grid=grid(16384), stream=stream0)
        buf122 = buf476; del buf476  # reuse
        triton__13.run(buf117, buf122, 16384, 128, grid=grid(16384), stream=stream0)
        buf166 = buf432; del buf432  # reuse
        triton__13.run(buf161, buf166, 16384, 128, grid=grid(16384), stream=stream0)
        buf210 = buf374; del buf374  # reuse
        triton__13.run(buf205, buf210, 16384, 128, grid=grid(16384), stream=stream0)
        buf254 = buf328; del buf328  # reuse
        triton__13.run(buf249, buf254, 16384, 128, grid=grid(16384), stream=stream0)
        buf298 = buf284; del buf284  # reuse
        triton__13.run(buf293, buf298, 16384, 128, grid=grid(16384), stream=stream0)
        buf342 = buf240; del buf240  # reuse
        triton__13.run(buf337, buf342, 16384, 128, grid=grid(16384), stream=stream0)
        buf394 = buf20; del buf20  # reuse
        triton__13.run(buf387, buf394, 16384, 128, grid=grid(16384), stream=stream0)
        buf446 = buf196; del buf196  # reuse
        triton__13.run(buf441, buf446, 16384, 128, grid=grid(16384), stream=stream0)
        buf490 = buf152; del buf152  # reuse
        triton__13.run(buf485, buf490, 16384, 128, grid=grid(16384), stream=stream0)
        buf535 = buf29; del buf29  # reuse
        triton__7.run(buf515, buf528, primals_16, mul_3, div_64, buf535, 512, 4096, grid=grid(512), stream=stream0)
        del div_64
        del mul_3
        del primals_16
        buf542 = buf108; del buf108  # reuse
        triton__13.run(buf535, buf542, 16384, 128, grid=grid(16384), stream=stream0)
        buf78 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        triton__13.run(buf73, buf78, 16384, 128, grid=grid(16384), stream=stream0)
        buf123 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf396 = as_strided(buf123, (4096, ), (1, )); del buf123  # reuse
        buf544 = buf396; del buf396  # reuse
        triton__15.run(buf544, buf34, buf78, buf122, buf166, buf210, buf254, buf446, buf298, buf490, buf342, buf542, buf394, 4096, 4, grid=grid(4096), stream=stream0)
        buf44 = as_strided(buf493, (4096, 4096), (4096, 1)); del buf493  # reuse
        extern_kernels.mm(as_strided(buf42, (4096, 512), (1, 4096)), view_233, out=buf44)
        buf45 = buf78; del buf78  # reuse
        triton__13.run(buf42, buf45, 16384, 128, grid=grid(16384), stream=stream0)
        buf133 = buf542; del buf542  # reuse
        triton__13.run(buf130, buf133, 16384, 128, grid=grid(16384), stream=stream0)
        buf177 = buf490; del buf490  # reuse
        triton__13.run(buf174, buf177, 16384, 128, grid=grid(16384), stream=stream0)
        buf221 = buf446; del buf446  # reuse
        triton__13.run(buf218, buf221, 16384, 128, grid=grid(16384), stream=stream0)
        buf265 = buf394; del buf394  # reuse
        triton__13.run(buf262, buf265, 16384, 128, grid=grid(16384), stream=stream0)
        buf309 = buf342; del buf342  # reuse
        triton__13.run(buf306, buf309, 16384, 128, grid=grid(16384), stream=stream0)
        buf353 = buf34; del buf34  # reuse
        triton__13.run(buf350, buf353, 16384, 128, grid=grid(16384), stream=stream0)
        buf407 = buf298; del buf298  # reuse
        triton__13.run(buf404, buf407, 16384, 128, grid=grid(16384), stream=stream0)
        buf457 = buf254; del buf254  # reuse
        triton__13.run(buf454, buf457, 16384, 128, grid=grid(16384), stream=stream0)
        buf501 = buf210; del buf210  # reuse
        triton__13.run(buf498, buf501, 16384, 128, grid=grid(16384), stream=stream0)
        buf540 = as_strided(buf42, (512, 4096), (4096, 1)); del buf42  # reuse
        extern_kernels.mm(as_strided(buf535, (512, 4096), (4096, 1)), permute_146, out=buf540)
        del permute_146
        buf546 = as_strided(buf528, (64, 512, 64), (32768, 64, 1)); del buf528  # reuse
        extern_kernels.bmm(permute_514, as_strided(buf540, (64, 512, 64), (64, 4096, 1)), out=buf546)
        del permute_514
        buf552 = as_strided(buf68, (512, 4096), (4096, 1)); del buf68  # reuse
        triton__8.run(buf546, buf552, 2097152, grid=grid(2097152), stream=stream0)
        del buf546
        buf555 = buf166; del buf166  # reuse
        triton__13.run(buf552, buf555, 16384, 128, grid=grid(16384), stream=stream0)
        buf89 = buf122; del buf122  # reuse
        triton__13.run(buf86, buf89, 16384, 128, grid=grid(16384), stream=stream0)
        buf134 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf409 = as_strided(buf134, (4096, ), (1, )); del buf134  # reuse
        buf557 = buf409; del buf409  # reuse
        triton__15.run(buf557, buf45, buf89, buf133, buf177, buf221, buf265, buf457, buf309, buf501, buf353, buf555, buf407, 4096, 4, grid=grid(4096), stream=stream0)
        buf49 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf47, (4096, 512), (512, 1)), view_233, out=buf49)
        del buf47
        buf53 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf51, (4096, 512), (1, 4096)), view_233, out=buf53)
        del view_233
        buf54 = buf89; del buf89  # reuse
        triton__13.run(buf51, buf54, 16384, 128, grid=grid(16384), stream=stream0)
        buf142 = buf555; del buf555  # reuse
        triton__13.run(buf139, buf142, 16384, 128, grid=grid(16384), stream=stream0)
        buf186 = buf501; del buf501  # reuse
        triton__13.run(buf183, buf186, 16384, 128, grid=grid(16384), stream=stream0)
        buf230 = buf457; del buf457  # reuse
        triton__13.run(buf227, buf230, 16384, 128, grid=grid(16384), stream=stream0)
        buf274 = buf45; del buf45  # reuse
        triton__13.run(buf271, buf274, 16384, 128, grid=grid(16384), stream=stream0)
        buf318 = buf407; del buf407  # reuse
        triton__13.run(buf315, buf318, 16384, 128, grid=grid(16384), stream=stream0)
        buf362 = buf353; del buf353  # reuse
        triton__13.run(buf359, buf362, 16384, 128, grid=grid(16384), stream=stream0)
        buf420 = buf309; del buf309  # reuse
        triton__13.run(buf417, buf420, 16384, 128, grid=grid(16384), stream=stream0)
        buf466 = buf265; del buf265  # reuse
        triton__13.run(buf463, buf466, 16384, 128, grid=grid(16384), stream=stream0)
        buf510 = buf221; del buf221  # reuse
        triton__13.run(buf507, buf510, 16384, 128, grid=grid(16384), stream=stream0)
        buf547 = empty_strided((64, 512, 512), (262144, 512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf540, (64, 512, 64), (64, 4096, 1)), permute_515, out=buf547)
        del permute_515
        buf549 = empty_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton__9.run(buf547, div_1, buf549, 32768, 512, grid=grid(32768), stream=stream0)
        del div_1
        buf551 = as_strided(buf540, (64, 512, 64), (32768, 64, 1)); del buf540  # reuse
        extern_kernels.bmm(as_strided(buf549, (64, 512, 512), (262144, 512, 1)), permute_517, out=buf551)
        del permute_517
        buf565 = buf51; del buf51  # reuse
        triton__8.run(buf551, buf565, 2097152, grid=grid(2097152), stream=stream0)
        del buf551
        buf568 = buf177; del buf177  # reuse
        triton__13.run(buf565, buf568, 16384, 128, grid=grid(16384), stream=stream0)
        buf98 = buf133; del buf133  # reuse
        triton__13.run(buf95, buf98, 16384, 128, grid=grid(16384), stream=stream0)
        buf143 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        buf422 = as_strided(buf143, (4096, ), (1, )); del buf143  # reuse
        buf570 = buf422; del buf422  # reuse
        triton__15.run(buf570, buf54, buf98, buf142, buf186, buf230, buf274, buf466, buf318, buf510, buf362, buf568, buf420, 4096, 4, grid=grid(4096), stream=stream0)
        del buf142
        del buf186
        del buf230
        del buf274
        del buf318
        del buf362
        del buf420
        del buf466
        del buf510
        del buf54
        del buf568
        buf63 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf59, (4096, 512), (1, 4096)), view_231, out=buf63)
        del buf59
        del view_231
        buf69 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf67, (16384, 512), (1, 16384)), view_229, out=buf69)
        del buf67
        del view_229
        buf77 = as_strided(buf547, (4096, 4096), (4096, 1)); del buf547  # reuse
        extern_kernels.mm(as_strided(buf73, (4096, 512), (1, 4096)), view_227, out=buf77)
        del buf73
        del view_227
        buf88 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf86, (4096, 512), (1, 4096)), view_212, out=buf88)
        del buf86
        buf93 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf91, (4096, 512), (512, 1)), view_212, out=buf93)
        del buf91
        buf97 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf95, (4096, 512), (1, 4096)), view_212, out=buf97)
        del buf95
        del view_212
        buf107 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf103, (4096, 512), (1, 4096)), view_210, out=buf107)
        del buf103
        del view_210
        buf113 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf111, (16384, 512), (1, 16384)), view_208, out=buf113)
        del buf111
        del view_208
        buf121 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf117, (4096, 512), (1, 4096)), view_206, out=buf121)
        del buf117
        del view_206
        buf132 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf130, (4096, 512), (1, 4096)), view_191, out=buf132)
        del buf130
        buf137 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf135, (4096, 512), (512, 1)), view_191, out=buf137)
        del buf135
        buf141 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf139, (4096, 512), (1, 4096)), view_191, out=buf141)
        del buf139
        del view_191
        buf151 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf147, (4096, 512), (1, 4096)), view_189, out=buf151)
        del buf147
        del view_189
        buf157 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf155, (16384, 512), (1, 16384)), view_187, out=buf157)
        del buf155
        del view_187
        buf165 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf161, (4096, 512), (1, 4096)), view_185, out=buf165)
        del buf161
        del view_185
        buf176 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf174, (4096, 512), (1, 4096)), view_170, out=buf176)
        del buf174
        buf181 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf179, (4096, 512), (512, 1)), view_170, out=buf181)
        del buf179
        buf185 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf183, (4096, 512), (1, 4096)), view_170, out=buf185)
        del buf183
        del view_170
        buf195 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf191, (4096, 512), (1, 4096)), view_168, out=buf195)
        del buf191
        del view_168
        buf201 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf199, (16384, 512), (1, 16384)), view_166, out=buf201)
        del buf199
        del view_166
        buf209 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf205, (4096, 512), (1, 4096)), view_164, out=buf209)
        del buf205
        del view_164
        buf220 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf218, (4096, 512), (1, 4096)), view_149, out=buf220)
        del buf218
        buf225 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf223, (4096, 512), (512, 1)), view_149, out=buf225)
        del buf223
        buf229 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf227, (4096, 512), (1, 4096)), view_149, out=buf229)
        del buf227
        del view_149
        buf239 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf235, (4096, 512), (1, 4096)), view_147, out=buf239)
        del buf235
        del view_147
        buf245 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf243, (16384, 512), (1, 16384)), view_145, out=buf245)
        del buf243
        del view_145
        buf253 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf249, (4096, 512), (1, 4096)), view_143, out=buf253)
        del buf249
        del view_143
        buf264 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf262, (4096, 512), (1, 4096)), view_128, out=buf264)
        del buf262
        buf269 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf267, (4096, 512), (512, 1)), view_128, out=buf269)
        del buf267
        buf273 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf271, (4096, 512), (1, 4096)), view_128, out=buf273)
        del buf271
        del view_128
        buf283 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf279, (4096, 512), (1, 4096)), view_126, out=buf283)
        del buf279
        del view_126
        buf289 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf287, (16384, 512), (1, 16384)), view_124, out=buf289)
        del buf287
        del view_124
        buf297 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf293, (4096, 512), (1, 4096)), view_122, out=buf297)
        del buf293
        del view_122
        buf308 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf306, (4096, 512), (1, 4096)), view_107, out=buf308)
        del buf306
        buf313 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf311, (4096, 512), (512, 1)), view_107, out=buf313)
        del buf311
        buf317 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf315, (4096, 512), (1, 4096)), view_107, out=buf317)
        del buf315
        del view_107
        buf327 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf323, (4096, 512), (1, 4096)), view_105, out=buf327)
        del buf323
        del view_105
        buf333 = empty_strided((16384, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf331, (16384, 512), (1, 16384)), view_103, out=buf333)
        del buf331
        del view_103
        buf341 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf337, (4096, 512), (1, 4096)), view_101, out=buf341)
        del buf337
        del view_101
        buf352 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf350, (4096, 512), (1, 4096)), view_86, out=buf352)
        del buf350
        buf357 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf355, (4096, 512), (512, 1)), view_86, out=buf357)
        del buf355
        buf361 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf359, (4096, 512), (1, 4096)), view_86, out=buf361)
        del buf359
        del view_86
        buf373 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf367, (4096, 512), (1, 4096)), view_84, out=buf373)
        del buf367
        del view_84
        buf431 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf427, (4096, 512), (1, 4096)), view_63, out=buf431)
        del buf427
        del view_63
        buf475 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf471, (4096, 512), (1, 4096)), view_42, out=buf475)
        del buf471
        del view_42
        buf521 = empty_strided((4096, 16384), (16384, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf515, (4096, 512), (1, 4096)), view_21, out=buf521)
        del buf515
        del view_21
        buf377 = buf107; del buf107  # reuse
        buf525 = buf377; del buf377  # reuse
        triton__18.run(buf525, buf19, buf63, buf151, buf195, buf239, buf283, buf327, buf373, buf431, buf475, buf521, 67108864, grid=grid(67108864), stream=stream0)
        del buf151
        del buf19
        del buf195
        del buf239
        del buf283
        del buf327
        del buf373
        buf381 = as_strided(buf63, (16384, 4096), (4096, 1)); del buf63  # reuse
        extern_kernels.mm(as_strided(buf379, (16384, 512), (1, 16384)), view_82, out=buf381)
        del buf379
        del view_82
        buf437 = as_strided(buf521, (16384, 4096), (4096, 1)); del buf521  # reuse
        extern_kernels.mm(as_strided(buf435, (16384, 512), (1, 16384)), view_61, out=buf437)
        del buf435
        del view_61
        buf481 = as_strided(buf475, (16384, 4096), (4096, 1)); del buf475  # reuse
        extern_kernels.mm(as_strided(buf479, (16384, 512), (1, 16384)), view_40, out=buf481)
        del buf479
        del view_40
        buf529 = as_strided(buf431, (16384, 4096), (4096, 1)); del buf431  # reuse
        extern_kernels.mm(as_strided(buf527, (16384, 512), (1, 16384)), view_19, out=buf529)
        del buf527
        del view_19
        buf384 = buf113; del buf113  # reuse
        buf532 = buf384; del buf384  # reuse
        triton__18.run(buf532, buf25, buf69, buf157, buf201, buf245, buf289, buf333, buf381, buf437, buf481, buf529, 67108864, grid=grid(67108864), stream=stream0)
        del buf157
        del buf201
        del buf245
        del buf25
        del buf289
        del buf333
        del buf381
        del buf437
        del buf481
        del buf529
        del buf69
        buf393 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf387, (4096, 512), (1, 4096)), view_80, out=buf393)
        del buf387
        del view_80
        buf445 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf441, (4096, 512), (1, 4096)), view_59, out=buf445)
        del buf441
        del view_59
        buf489 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf485, (4096, 512), (1, 4096)), view_38, out=buf489)
        del buf485
        del view_38
        buf541 = empty_strided((4096, 4096), (4096, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf535, (4096, 512), (1, 4096)), view_17, out=buf541)
        del view_17
        buf397 = buf121; del buf121  # reuse
        buf545 = buf397; del buf397  # reuse
        triton__19.run(buf545, buf33, buf77, buf165, buf209, buf253, buf297, buf341, buf393, buf445, buf489, buf541, 16777216, grid=grid(16777216), stream=stream0)
        del buf165
        del buf209
        del buf253
        del buf297
        del buf33
        del buf341
        del buf393
        buf406 = buf77; del buf77  # reuse
        extern_kernels.mm(as_strided(buf404, (4096, 512), (1, 4096)), view_65, out=buf406)
        del buf404
        buf456 = buf541; del buf541  # reuse
        extern_kernels.mm(as_strided(buf454, (4096, 512), (1, 4096)), view_44, out=buf456)
        del buf454
        buf500 = buf489; del buf489  # reuse
        extern_kernels.mm(as_strided(buf498, (4096, 512), (1, 4096)), view_23, out=buf500)
        del buf498
        buf554 = buf445; del buf445  # reuse
        extern_kernels.mm(as_strided(buf552, (4096, 512), (1, 4096)), view_2, out=buf554)
        buf410 = buf132; del buf132  # reuse
        buf558 = buf410; del buf410  # reuse
        triton__19.run(buf558, buf44, buf88, buf176, buf220, buf264, buf308, buf352, buf406, buf456, buf500, buf554, 16777216, grid=grid(16777216), stream=stream0)
        del buf176
        del buf220
        del buf264
        del buf308
        del buf352
        del buf406
        del buf44
        del buf456
        buf413 = buf88; del buf88  # reuse
        extern_kernels.mm(as_strided(buf411, (4096, 512), (512, 1)), view_65, out=buf413)
        buf550 = as_strided(buf411, (64, 64, 512), (32768, 512, 1)); del buf411  # reuse
        extern_kernels.bmm(permute_516, as_strided(buf549, (64, 512, 512), (262144, 512, 1)), out=buf550)
        del permute_516
        buf559 = as_strided(buf550, (512, 4096), (1, 512)); del buf550  # reuse
        buf415 = as_strided(buf138, (4096, ), (1, )); del buf138  # reuse
        buf563 = buf415; del buf415  # reuse
        triton__20.run(buf559, buf563, buf50, buf94, buf182, buf226, buf270, buf314, buf358, buf414, buf462, buf506, 4096, 512, grid=grid(4096), stream=stream0)
        del buf182
        del buf226
        del buf270
        del buf314
        del buf358
        del buf414
        del buf462
        del buf50
        del buf506
        buf461 = as_strided(buf549, (4096, 4096), (4096, 1)); del buf549  # reuse
        extern_kernels.mm(as_strided(buf459, (4096, 512), (512, 1)), view_44, out=buf461)
        del buf459
        buf505 = buf554; del buf554  # reuse
        extern_kernels.mm(as_strided(buf503, (4096, 512), (512, 1)), view_23, out=buf505)
        del buf503
        buf561 = buf500; del buf500  # reuse
        extern_kernels.mm(as_strided(buf559, (4096, 512), (512, 1)), view_2, out=buf561)
        buf416 = buf137; del buf137  # reuse
        buf564 = buf416; del buf416  # reuse
        triton__19.run(buf564, buf49, buf93, buf181, buf225, buf269, buf313, buf357, buf413, buf461, buf505, buf561, 16777216, grid=grid(16777216), stream=stream0)
        del buf181
        del buf225
        del buf269
        del buf313
        del buf357
        del buf413
        del buf461
        buf419 = buf93; del buf93  # reuse
        extern_kernels.mm(as_strided(buf417, (4096, 512), (1, 4096)), view_65, out=buf419)
        del buf417
        del view_65
        buf465 = buf561; del buf561  # reuse
        extern_kernels.mm(as_strided(buf463, (4096, 512), (1, 4096)), view_44, out=buf465)
        del buf463
        del view_44
        buf509 = buf505; del buf505  # reuse
        extern_kernels.mm(as_strided(buf507, (4096, 512), (1, 4096)), view_23, out=buf509)
        del view_23
        buf567 = buf49; del buf49  # reuse
        extern_kernels.mm(as_strided(buf565, (4096, 512), (1, 4096)), view_2, out=buf567)
        del view_2
        buf423 = buf141; del buf141  # reuse
        buf571 = buf423; del buf423  # reuse
        triton__19.run(buf571, buf53, buf97, buf185, buf229, buf273, buf317, buf361, buf419, buf465, buf509, buf567, 16777216, grid=grid(16777216), stream=stream0)
        del buf185
        del buf229
        del buf273
        del buf317
        del buf361
        del buf419
        del buf465
        del buf509
        del buf53
        del buf567
        del buf97
        buf553 = as_strided(buf507, (512, 4096), (4096, 1)); del buf507  # reuse
        extern_kernels.mm(buf552, permute_159, out=buf553)
        del permute_159
        buf560 = as_strided(buf552, (512, 4096), (4096, 1)); del buf552  # reuse
        extern_kernels.mm(buf559, permute_163, out=buf560)
        del permute_163
        buf566 = as_strided(buf559, (512, 4096), (4096, 1)); del buf559  # reuse
        extern_kernels.mm(buf565, permute_167, out=buf566)
        del buf565
        del permute_167
        buf572 = buf535; del buf535  # reuse
        buf573 = as_strided(buf572, (512, 4096), (4096, 1)); del buf572  # reuse
        triton__21.run(buf573, buf553, buf560, buf566, 2097152, grid=grid(2097152), stream=stream0)
        del buf553
        del buf560
        del buf566
        buf574 = empty_strided((512, 128), (128, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf573, permute_534, out=buf574)
        del permute_534
        buf575 = empty_strided((4096, 128), (128, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf573, (4096, 512), (1, 4096)), view, out=buf575)
        del view
        buf576 = buf98; del buf98  # reuse
        triton__13.run(buf573, buf576, 16384, 128, grid=grid(16384), stream=stream0)
        del buf573
        buf577 = buf94; del buf94  # reuse
        triton__22.run(buf576, buf577, 4096, 4, grid=grid(4096), stream=stream0)
        del buf576
        buf583 = empty_strided((512, 128), (128, 1), device='cuda', dtype=torch.float32)
        triton__23.run(buf583, 65536, grid=grid(65536), stream=stream0)
        buf585 = empty_strided((2, 128), (128, 1), device='cuda', dtype=torch.float32)
        triton__24.run(buf585, 256, grid=grid(256), stream=stream0)
        buf587 = empty_strided((30000, 128), (128, 1), device='cuda', dtype=torch.float32)
        triton__25.run(buf587, 3840000, grid=grid(3840000), stream=stream0)
        triton__26.run(buf574, primals_4, mul_1, div_66, convert_element_type_2, convert_element_type_4, convert_element_type_6, buf583, buf585, buf587, 512, 128, grid=grid(512), stream=stream0)
        del convert_element_type_2
        del convert_element_type_4
        del convert_element_type_6
        del div_66
        del primals_4
        buf581 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf582 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__27.run(buf574, mul_1, buf581, buf582, 128, 512, grid=grid(128), stream=stream0)
        del buf574
        del mul_1
        return (buf587, buf585, buf583, buf581, buf582, as_strided(buf575, (4096, 128), (128, 1)), as_strided(buf577, (4096, ), (1, )), buf571, buf570, buf564, buf563, buf558, buf557, buf545, buf544, buf538, buf539, buf532, buf531, buf525, buf524, buf518, buf519, as_strided(buf10, (2, 4096), (4096, 1)), as_strided(buf12, (2, ), (1, )), None, None, None, None, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_1 = rand_strided((1, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_3 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_11 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_19 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_80 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_27 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_33 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_41 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_43 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_124 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_6 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_147 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_59 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_7 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_168 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_185 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_187 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_8 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_189 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_191 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_206 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_208 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_59 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_9 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_210 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_81 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_227 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_83 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_229 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_65 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_10 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_231 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_233 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_248 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_71 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_11 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_252 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_254 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    sub_39 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    sub_41 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_3 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_2 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    permute_134 = rand_strided((2, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_351 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_384 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_385 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_449 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_451 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_483 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_484 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_515 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_516 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_4 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_6 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([primals_4, primals_16, primals_22, mul_1, view, view_2, div_1, view_17, mul_3, view_19, addmm_5, tanh, view_21, mul_9, view_23, div_3, view_38, mul_11, view_40, addmm_11, tanh_1, view_42, mul_17, view_44, div_5, view_59, mul_19, view_61, addmm_17, tanh_2, view_63, mul_25, view_65, div_7, view_80, mul_27, view_82, addmm_23, tanh_3, view_84, mul_33, view_86, div_9, view_101, mul_35, view_103, addmm_29, tanh_4, view_105, mul_41, view_107, div_11, view_122, mul_43, view_124, addmm_35, tanh_5, view_126, mul_49, view_128, div_13, view_143, mul_51, view_145, addmm_41, tanh_6, view_147, mul_57, view_149, div_15, view_164, mul_59, view_166, addmm_47, tanh_7, view_168, mul_65, view_170, div_17, view_185, mul_67, view_187, addmm_53, tanh_8, view_189, mul_73, view_191, div_19, view_206, mul_75, view_208, addmm_59, tanh_9, view_210, mul_81, view_212, div_21, view_227, mul_83, view_229, addmm_65, tanh_10, view_231, mul_89, view_233, div_23, view_248, mul_91, view_250, addmm_71, tanh_11, view_252, mul_97, view_254, sub_39, unsqueeze_2, ne, sub_41, unsqueeze_3, ne_2, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, permute_186, permute_187, div_36, div_37, permute_217, permute_218, permute_219, permute_220, div_39, div_40, permute_250, permute_251, permute_252, permute_253, div_42, div_43, permute_283, permute_284, permute_285, permute_286, div_45, div_46, permute_316, permute_317, permute_318, permute_319, div_48, div_49, permute_349, permute_350, permute_351, permute_352, div_51, div_52, permute_382, permute_383, permute_384, permute_385, div_54, div_55, permute_415, permute_416, permute_417, permute_418, div_57, div_58, permute_448, permute_449, permute_450, permute_451, div_60, div_61, permute_481, permute_482, permute_483, permute_484, div_63, div_64, permute_514, permute_515, permute_516, permute_517, permute_534, div_66, convert_element_type_2, convert_element_type_4, convert_element_type_6, tangents_1, tangents_2, tangents_3]))
