
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
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32768, 8192],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
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
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (30522*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
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

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tmp4 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp23 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = 768.0
        tmp15 = tmp10 / tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        tmp21 = tmp20 * tmp14
        tmp22 = tmp21 - tmp5
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24 - tmp8
        tmp26 = tmp25 * tmp10
        tmp27 = tmp26 * tmp13
        tmp28 = tmp22 - tmp27
        tmp29 = tmp15 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = 0.7071067811865476
        tmp35 = tmp33 * tmp34
        tmp36 = tl.libdevice.erf(tmp35)
        tmp37 = 1.0
        tmp38 = tmp36 + tmp37
        tmp39 = 0.5
        tmp40 = tmp38 * tmp39
        tmp41 = tmp33 * tmp33
        tmp42 = -0.5
        tmp43 = tmp41 * tmp42
        tmp44 = tl.exp(tmp43)
        tmp45 = 0.3989422804014327
        tmp46 = tmp44 * tmp45
        tmp47 = tmp33 * tmp46
        tmp48 = tmp40 + tmp47
        tmp49 = tmp31 * tmp48
        tmp50 = tmp49.to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp50, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp51 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp51, rmask & xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp1, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp10, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 64],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 64
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
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tmp2 = tmp1.to(tl.float32)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp2, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 64],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 64
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
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tmp4 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp23 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = 768.0
        tmp15 = tmp10 / tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        tmp21 = tmp20 * tmp14
        tmp22 = tmp21 - tmp5
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24 - tmp8
        tmp26 = tmp25 * tmp10
        tmp27 = tmp26 * tmp13
        tmp28 = tmp22 - tmp27
        tmp29 = tmp15 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp30, rmask & xmask)
    tmp31_load = tl.load(in_ptr5 + (0))
    tmp31 = tl.broadcast_to(tmp31_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp37 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp32 = 754974720 + r1 + (768*x0)
        tmp33 = tl.rand(tmp31, tmp32)
        tmp34 = 0.1
        tmp35 = tmp33 > tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp38 = tmp36 * tmp37
        tmp39 = 1.1111111111111112
        tmp40 = tmp38 * tmp39
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp40, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, rmask & xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 256],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (786432*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 32],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 32
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 748683264 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp3, _tmp12)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp12, xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__13(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 698351616 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 698351616 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__15(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 8192
    ynumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex
    y1 = yindex
    tmp0 = tl.load(in_ptr0 + ((512*y1) + (393216*(x0 // 512)) + (x0 % 512)), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (y1 + (768*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 692060160 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp7 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
        _tmp16 = tl.where(xmask & rmask, _tmp16 + tmp7, _tmp16)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp15, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp16, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 685768704 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__19(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 635437056 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 635437056 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 629145600 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 622854144 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 572522496 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 572522496 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 566231040 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 559939584 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__25(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 509607936 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 509607936 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 503316480 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 497025024 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__28(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 446693376 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 446693376 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 440401920 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 434110464 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__31(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 383778816 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 383778816 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 377487360 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 371195904 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__34(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 320864256 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 320864256 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 314572800 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 308281344 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__37(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 257949696 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 257949696 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 251658240 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 245366784 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__40(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 195035136 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 195035136 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 188743680 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 182452224 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__43(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 132120576 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 132120576 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 125829120 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 119537664 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__46(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 69206016 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 69206016 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr6 + (x0), xmask)
    tmp17 = tl.load(in_ptr7 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp23 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = 768.0
        tmp22 = tmp17 / tmp21
        tmp24 = tmp23 * tmp21
        tmp25 = tmp24 - tmp12
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp15
        tmp29 = tmp28 * tmp17
        tmp30 = tmp29 * tmp20
        tmp31 = tmp25 - tmp30
        tmp32 = tmp22 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
    tmp34_load = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp34_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp40 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = 62914560 + r1 + (768*x0)
        tmp36 = tl.rand(tmp34, tmp35)
        tmp37 = 0.1
        tmp38 = tmp36 > tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp41 = tmp39 * tmp40
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
''')


triton__48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    tmp12 = tl.load(in_ptr5 + (x0), xmask)
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp12 / tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp16
        tmp26 = tmp25 - tmp7
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp15
        tmp32 = tmp26 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask & xmask)
    tmp35_load = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp35_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = 56623104 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp45 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__49(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 6291456 + r1 + (512*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = 6291456 + r1 + (512*x0)
        tmp16 = tl.rand(tmp0, tmp15)
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = 1.1111111111111112
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp26 * tmp14
        tmp29 = tmp27 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask & xmask)
''')


triton__50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__50(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__51(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: '*fp16', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr4', 'out_ptr5'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]})
@triton.jit
def triton__52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 + tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = 1.1111111111111112
        tmp15 = tmp13 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp24 = tl.load(in_ptr7 + (x0), xmask)
    tmp26 = tl.load(in_ptr8 + (x0), xmask)
    _tmp29 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp18 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        tmp23 = tmp22.to(tl.float32)
        tmp25 = tmp23 - tmp24
        tmp27 = tmp25 * tmp26
        tmp28 = tmp20 * tmp27
        _tmp29 = tl.where(xmask & rmask, _tmp29 + tmp28, _tmp29)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp32 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp38 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = 768.0
        tmp31 = tmp26 / tmp30
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp36 = tmp35 * tmp30
        tmp37 = tmp36 - tmp21
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp39 - tmp24
        tmp41 = tmp40 * tmp26
        tmp42 = tmp41 * tmp29
        tmp43 = tmp37 - tmp42
        tmp44 = tmp31 * tmp43
        tmp45 = tmp44.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
    x2 = xindex % 512
    tmp46 = tl.load(in_ptr9 + (x2), xmask)
    tmp55 = tl.load(in_ptr10 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp52 = tl.load(out_ptr3 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp47.to(tl.int64)
        tmp49 = -1
        tmp50 = tmp48 == tmp49
        tmp51 = 0.0
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tl.where(tmp50, tmp51, tmp53)
        tmp56 = tmp55.to(tl.int64)
        tmp57 = 0
        tmp58 = tmp56 == tmp57
        tmp59 = tl.where(tmp58, tmp51, tmp53)
        tl.atomic_add(out_ptr4 + (r1 + (768*tmp48) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp54, rmask & xmask)
        tl.atomic_add(out_ptr5 + (r1 + (768*tmp56) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp59, rmask & xmask)
''')


triton__53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 - tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp0 * tmp6
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp0, _tmp9)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp8, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp9, xmask)
''')


triton__54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__54(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[524288, 16],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__55(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 393216
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
        tmp0 = tl.load(in_ptr0 + (x0 + (393216*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    x3 = (xindex // 768)
    x2 = xindex % 768
    tmp2 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tmp2.to(tl.int64)
    tmp4 = -1
    tmp5 = tmp3 == tmp4
    tmp6 = 0.0
    tmp7 = tmp1.to(tl.float32)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.atomic_add(out_ptr1 + (x2 + (768*tmp3) + tl.zeros([XBLOCK, 1], tl.int32)), tmp8, xmask)
''')


triton__56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__57(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__58(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
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
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_204, add_1, getitem_1, rsqrt, philox_seed_like, view, convert_element_type_5, view_13, add_5, getitem_3, rsqrt_1, view_15, addmm_4, view_17, add_9, getitem_5, rsqrt_2, view_19, convert_element_type_16, view_32, add_13, getitem_7, rsqrt_3, view_34, addmm_10, view_36, add_17, getitem_9, rsqrt_4, view_38, convert_element_type_27, view_51, add_21, getitem_11, rsqrt_5, view_53, addmm_16, view_55, add_25, getitem_13, rsqrt_6, view_57, convert_element_type_38, view_70, add_29, getitem_15, rsqrt_7, view_72, addmm_22, view_74, add_33, getitem_17, rsqrt_8, view_76, convert_element_type_49, view_89, add_37, getitem_19, rsqrt_9, view_91, addmm_28, view_93, add_41, getitem_21, rsqrt_10, view_95, convert_element_type_60, view_108, add_45, getitem_23, rsqrt_11, view_110, addmm_34, view_112, add_49, getitem_25, rsqrt_12, view_114, convert_element_type_71, view_127, add_53, getitem_27, rsqrt_13, view_129, addmm_40, view_131, add_57, getitem_29, rsqrt_14, view_133, convert_element_type_82, view_146, add_61, getitem_31, rsqrt_15, view_148, addmm_46, view_150, add_65, getitem_33, rsqrt_16, view_152, convert_element_type_93, view_165, add_69, getitem_35, rsqrt_17, view_167, addmm_52, view_169, add_73, getitem_37, rsqrt_18, view_171, convert_element_type_104, view_184, add_77, getitem_39, rsqrt_19, view_186, addmm_58, view_188, add_81, getitem_41, rsqrt_20, view_190, convert_element_type_115, view_203, add_85, getitem_43, rsqrt_21, view_205, addmm_64, view_207, add_89, getitem_45, rsqrt_22, view_209, convert_element_type_126, view_222, add_93, getitem_47, rsqrt_23, view_224, addmm_70, view_226, add_97, getitem_49, rsqrt_24, view_228, addmm_72, convert_element_type_137, getitem_51, rsqrt_25, view_230, permute_134, permute_138, permute_142, permute_146, permute_150, permute_155, permute_156, permute_157, permute_158, permute_162, permute_167, permute_171, permute_175, permute_179, permute_183, permute_188, permute_189, permute_190, permute_191, permute_195, permute_200, permute_204, permute_208, permute_212, permute_216, permute_221, permute_222, permute_223, permute_224, permute_228, permute_233, permute_237, permute_241, permute_245, permute_249, permute_254, permute_255, permute_256, permute_257, permute_261, permute_266, permute_270, permute_274, permute_278, permute_282, permute_287, permute_288, permute_289, permute_290, permute_294, permute_299, permute_303, permute_307, permute_311, permute_315, permute_320, permute_321, permute_322, permute_323, permute_327, permute_332, permute_336, permute_340, permute_344, permute_348, permute_353, permute_354, permute_355, permute_356, permute_360, permute_365, permute_369, permute_373, permute_377, permute_381, permute_386, permute_387, permute_388, permute_389, permute_393, permute_398, permute_402, permute_406, permute_410, permute_414, permute_419, permute_420, permute_421, permute_422, permute_426, permute_431, permute_435, permute_439, permute_443, permute_447, permute_452, permute_453, permute_454, permute_455, permute_459, permute_464, permute_468, permute_472, permute_476, permute_480, permute_485, permute_486, permute_487, permute_488, permute_492, permute_497, permute_501, permute_505, permute_509, permute_513, permute_518, permute_519, permute_520, permute_521, permute_525, permute_530, permute_534, convert_element_type_435, convert_element_type_443, tangents_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(tangents_1, permute_134, out=buf0)
        del permute_134
        buf1 = empty_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(tangents_1, (30522, 8192), (1, 30522)), view_230, out=buf1)
        del view_230
        buf2 = empty_strided((1, 30522), (30522, 1), device='cuda', dtype=torch.float16)
        stream0 = get_cuda_stream(0)
        triton__0.run(tangents_1, buf2, 30522, 8192, grid=grid(30522), stream=stream0)
        del tangents_1
        buf11 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        buf12 = as_strided(buf11, (8192, 768), (768, 1)); del buf11  # reuse
        triton__1.run(buf12, buf0, primals_200, convert_element_type_137, getitem_51, rsqrt_25, addmm_72, 8192, 768, grid=grid(8192), stream=stream0)
        del addmm_72
        del primals_200
        buf5 = empty_strided((768, 64), (1, 768), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((768, 64), (1, 768), device='cuda', dtype=torch.float32)
        triton__2.run(buf0, convert_element_type_137, getitem_51, rsqrt_25, buf5, buf7, 49152, 128, grid=grid(49152), stream=stream0)
        del convert_element_type_137
        del getitem_51
        del rsqrt_25
        buf9 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf5, buf9, 768, 64, grid=grid(768), stream=stream0)
        buf10 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf7, buf10, 768, 64, grid=grid(768), stream=stream0)
        buf13 = buf0; del buf0  # reuse
        extern_kernels.mm(buf12, permute_138, out=buf13)
        del permute_138
        buf14 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf12, (768, 8192), (1, 768)), view_228, out=buf14)
        del view_228
        buf15 = as_strided(buf7, (1, 768, 64), (49152, 1, 768)); del buf7  # reuse
        triton__4.run(buf12, buf15, 49152, 128, grid=grid(49152), stream=stream0)
        buf16 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf15, buf16, 768, 64, grid=grid(768), stream=stream0)
        buf23 = as_strided(buf12, (16, 512, 768), (393216, 768, 1)); del buf12  # reuse
        buf26 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        buf27 = as_strided(buf26, (8192, 768), (768, 1)); del buf26  # reuse
        triton__6.run(buf27, buf13, primals_196, add_97, getitem_49, rsqrt_24, philox_seed_like, buf23, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_196
        buf19 = as_strided(buf15, (768, 64), (1, 768)); del buf15  # reuse
        buf21 = buf5; del buf5  # reuse
        triton__2.run(buf13, add_97, getitem_49, rsqrt_24, buf19, buf21, 49152, 128, grid=grid(49152), stream=stream0)
        del add_97
        del getitem_49
        del rsqrt_24
        buf24 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf19, buf24, 768, 64, grid=grid(768), stream=stream0)
        buf25 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf21, buf25, 768, 64, grid=grid(768), stream=stream0)
        buf28 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(buf27, permute_142, out=buf28)
        del permute_142
        buf29 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf27, (768, 8192), (1, 768)), view_226, out=buf29)
        del view_226
        buf30 = as_strided(buf21, (1, 768, 64), (49152, 1, 768)); del buf21  # reuse
        triton__4.run(buf27, buf30, 49152, 128, grid=grid(49152), stream=stream0)
        buf31 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf30, buf31, 768, 64, grid=grid(768), stream=stream0)
        buf32 = as_strided(buf28, (16, 512, 3072), (1572864, 3072, 1)); del buf28  # reuse
        buf33 = as_strided(buf32, (8192, 3072), (3072, 1)); del buf32  # reuse
        triton__7.run(buf33, addmm_70, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_70
        buf34 = as_strided(buf27, (8192, 768), (768, 1)); del buf27  # reuse
        extern_kernels.mm(buf33, permute_146, out=buf34)
        del permute_146
        buf35 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf33, (3072, 8192), (1, 3072)), view_224, out=buf35)
        del view_224
        buf36 = empty_strided((1, 3072, 32), (98304, 1, 3072), device='cuda', dtype=torch.float32)
        triton__8.run(buf33, buf36, 98304, 256, grid=grid(98304), stream=stream0)
        buf37 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf36, buf37, 3072, 32, grid=grid(3072), stream=stream0)
        buf44 = as_strided(buf13, (16, 512, 768), (393216, 768, 1)); del buf13  # reuse
        buf47 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        buf48 = as_strided(buf47, (8192, 768), (768, 1)); del buf47  # reuse
        triton__10.run(buf48, buf23, buf34, primals_190, add_93, getitem_47, rsqrt_23, philox_seed_like, buf44, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_190
        buf40 = as_strided(buf30, (768, 64), (1, 768)); del buf30  # reuse
        buf42 = buf19; del buf19  # reuse
        triton__11.run(buf23, buf34, add_93, getitem_47, rsqrt_23, buf40, buf42, 49152, 128, grid=grid(49152), stream=stream0)
        del add_93
        del getitem_47
        del rsqrt_23
        buf45 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf40, buf45, 768, 64, grid=grid(768), stream=stream0)
        buf46 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf42, buf46, 768, 64, grid=grid(768), stream=stream0)
        buf49 = buf34; del buf34  # reuse
        extern_kernels.mm(buf48, permute_150, out=buf49)
        del permute_150
        buf50 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf48, (768, 8192), (1, 768)), view_222, out=buf50)
        del view_222
        buf51 = as_strided(buf42, (1, 768, 64), (49152, 1, 768)); del buf42  # reuse
        triton__4.run(buf48, buf51, 49152, 128, grid=grid(49152), stream=stream0)
        buf52 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf51, buf52, 768, 64, grid=grid(768), stream=stream0)
        buf53 = as_strided(buf48, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf48  # reuse
        triton__12.run(buf49, buf53, 6291456, grid=grid(6291456), stream=stream0)
        buf54 = as_strided(buf49, (192, 512, 64), (32768, 64, 1)); del buf49  # reuse
        extern_kernels.bmm(permute_155, as_strided(buf53, (192, 512, 64), (32768, 64, 1)), out=buf54)
        del permute_155
        buf55 = empty_strided((192, 512, 512), (262144, 512, 1), device='cuda', dtype=torch.float16)
        extern_kernels.bmm(as_strided(buf53, (192, 512, 64), (32768, 64, 1)), permute_156, out=buf55)
        del permute_156
        buf57 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        triton__13.run(philox_seed_like, buf55, convert_element_type_126, buf57, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_126
        buf58 = as_strided(buf53, (192, 64, 512), (32768, 512, 1)); del buf53  # reuse
        extern_kernels.bmm(permute_157, as_strided(buf57, (192, 512, 512), (262144, 512, 1)), out=buf58)
        del permute_157
        buf59 = as_strided(buf23, (192, 512, 64), (32768, 64, 1)); del buf23  # reuse
        extern_kernels.bmm(as_strided(buf57, (192, 512, 512), (262144, 512, 1)), permute_158, out=buf59)
        del permute_158
        buf60 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__14.run(buf54, buf60, 6291456, grid=grid(6291456), stream=stream0)
        buf61 = as_strided(buf54, (8192, 768), (768, 1)); del buf54  # reuse
        extern_kernels.mm(buf60, permute_162, out=buf61)
        del permute_162
        buf62 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf60, (768, 8192), (1, 768)), view_209, out=buf62)
        buf63 = buf51; del buf51  # reuse
        triton__4.run(buf60, buf63, 49152, 128, grid=grid(49152), stream=stream0)
        buf64 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf63, buf64, 768, 64, grid=grid(768), stream=stream0)
        buf65 = buf60; del buf60  # reuse
        triton__15.run(buf58, buf65, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf66 = as_strided(buf58, (8192, 768), (768, 1)); del buf58  # reuse
        extern_kernels.mm(buf65, permute_167, out=buf66)
        del permute_167
        buf67 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf65, (768, 8192), (1, 768)), view_209, out=buf67)
        buf68 = buf63; del buf63  # reuse
        triton__4.run(buf65, buf68, 49152, 128, grid=grid(49152), stream=stream0)
        buf69 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf68, buf69, 768, 64, grid=grid(768), stream=stream0)
        buf70 = buf65; del buf65  # reuse
        triton__14.run(buf59, buf70, 6291456, grid=grid(6291456), stream=stream0)
        buf71 = as_strided(buf59, (8192, 768), (768, 1)); del buf59  # reuse
        extern_kernels.mm(buf70, permute_171, out=buf71)
        del permute_171
        buf72 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf70, (768, 8192), (1, 768)), view_209, out=buf72)
        del view_209
        buf73 = buf68; del buf68  # reuse
        triton__4.run(buf70, buf73, 49152, 128, grid=grid(49152), stream=stream0)
        buf74 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf73, buf74, 768, 64, grid=grid(768), stream=stream0)
        buf75 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
        buf82 = as_strided(buf70, (16, 512, 768), (393216, 768, 1)); del buf70  # reuse
        buf85 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        buf86 = as_strided(buf85, (8192, 768), (768, 1)); del buf85  # reuse
        triton__16.run(buf86, buf44, buf61, buf66, buf71, primals_180, add_89, getitem_45, rsqrt_22, philox_seed_like, buf75, buf82, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_180
        buf78 = as_strided(buf73, (768, 64), (1, 768)); del buf73  # reuse
        buf80 = buf40; del buf40  # reuse
        triton__17.run(buf44, buf61, buf66, buf71, add_89, getitem_45, rsqrt_22, buf78, buf80, 49152, 128, grid=grid(49152), stream=stream0)
        del add_89
        del getitem_45
        del rsqrt_22
        buf83 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf78, buf83, 768, 64, grid=grid(768), stream=stream0)
        buf84 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf80, buf84, 768, 64, grid=grid(768), stream=stream0)
        buf87 = as_strided(buf33, (8192, 3072), (3072, 1)); del buf33  # reuse
        extern_kernels.mm(buf86, permute_175, out=buf87)
        del permute_175
        buf88 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf86, (768, 8192), (1, 768)), view_207, out=buf88)
        del view_207
        buf89 = as_strided(buf80, (1, 768, 64), (49152, 1, 768)); del buf80  # reuse
        triton__4.run(buf86, buf89, 49152, 128, grid=grid(49152), stream=stream0)
        buf90 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf89, buf90, 768, 64, grid=grid(768), stream=stream0)
        buf91 = as_strided(buf87, (16, 512, 3072), (1572864, 3072, 1)); del buf87  # reuse
        buf92 = as_strided(buf91, (8192, 3072), (3072, 1)); del buf91  # reuse
        triton__7.run(buf92, addmm_64, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_64
        buf93 = as_strided(buf86, (8192, 768), (768, 1)); del buf86  # reuse
        extern_kernels.mm(buf92, permute_179, out=buf93)
        del permute_179
        buf94 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf92, (3072, 8192), (1, 3072)), view_205, out=buf94)
        del view_205
        buf95 = buf36; del buf36  # reuse
        triton__8.run(buf92, buf95, 98304, 256, grid=grid(98304), stream=stream0)
        buf96 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf95, buf96, 3072, 32, grid=grid(3072), stream=stream0)
        buf103 = as_strided(buf71, (16, 512, 768), (393216, 768, 1)); del buf71  # reuse
        buf106 = as_strided(buf66, (16, 512, 768), (393216, 768, 1)); del buf66  # reuse
        buf107 = as_strided(buf106, (8192, 768), (768, 1)); del buf106  # reuse
        triton__18.run(buf107, buf82, buf93, primals_174, add_85, getitem_43, rsqrt_21, philox_seed_like, buf103, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_174
        buf99 = as_strided(buf89, (768, 64), (1, 768)); del buf89  # reuse
        buf101 = buf78; del buf78  # reuse
        triton__11.run(buf82, buf93, add_85, getitem_43, rsqrt_21, buf99, buf101, 49152, 128, grid=grid(49152), stream=stream0)
        del add_85
        del getitem_43
        del rsqrt_21
        buf104 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf99, buf104, 768, 64, grid=grid(768), stream=stream0)
        buf105 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf101, buf105, 768, 64, grid=grid(768), stream=stream0)
        buf108 = buf93; del buf93  # reuse
        extern_kernels.mm(buf107, permute_183, out=buf108)
        del permute_183
        buf109 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf107, (768, 8192), (1, 768)), view_203, out=buf109)
        del view_203
        buf110 = as_strided(buf101, (1, 768, 64), (49152, 1, 768)); del buf101  # reuse
        triton__4.run(buf107, buf110, 49152, 128, grid=grid(49152), stream=stream0)
        buf111 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf110, buf111, 768, 64, grid=grid(768), stream=stream0)
        buf112 = as_strided(buf107, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf107  # reuse
        triton__12.run(buf108, buf112, 6291456, grid=grid(6291456), stream=stream0)
        buf113 = as_strided(buf108, (192, 512, 64), (32768, 64, 1)); del buf108  # reuse
        extern_kernels.bmm(permute_188, as_strided(buf112, (192, 512, 64), (32768, 64, 1)), out=buf113)
        del permute_188
        buf114 = as_strided(buf57, (192, 512, 512), (262144, 512, 1)); del buf57  # reuse
        extern_kernels.bmm(as_strided(buf112, (192, 512, 64), (32768, 64, 1)), permute_189, out=buf114)
        del permute_189
        buf116 = as_strided(buf55, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf55  # reuse
        triton__19.run(philox_seed_like, buf114, convert_element_type_115, buf116, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_115
        buf117 = as_strided(buf112, (192, 64, 512), (32768, 512, 1)); del buf112  # reuse
        extern_kernels.bmm(permute_190, as_strided(buf116, (192, 512, 512), (262144, 512, 1)), out=buf117)
        del permute_190
        buf118 = as_strided(buf82, (192, 512, 64), (32768, 64, 1)); del buf82  # reuse
        extern_kernels.bmm(as_strided(buf116, (192, 512, 512), (262144, 512, 1)), permute_191, out=buf118)
        del permute_191
        buf119 = as_strided(buf61, (8192, 768), (768, 1)); del buf61  # reuse
        triton__14.run(buf113, buf119, 6291456, grid=grid(6291456), stream=stream0)
        buf120 = as_strided(buf113, (8192, 768), (768, 1)); del buf113  # reuse
        extern_kernels.mm(buf119, permute_195, out=buf120)
        del permute_195
        buf121 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf119, (768, 8192), (1, 768)), view_190, out=buf121)
        buf122 = buf110; del buf110  # reuse
        triton__4.run(buf119, buf122, 49152, 128, grid=grid(49152), stream=stream0)
        buf123 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf122, buf123, 768, 64, grid=grid(768), stream=stream0)
        buf124 = buf119; del buf119  # reuse
        triton__15.run(buf117, buf124, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf125 = as_strided(buf117, (8192, 768), (768, 1)); del buf117  # reuse
        extern_kernels.mm(buf124, permute_200, out=buf125)
        del permute_200
        buf126 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf124, (768, 8192), (1, 768)), view_190, out=buf126)
        buf127 = buf122; del buf122  # reuse
        triton__4.run(buf124, buf127, 49152, 128, grid=grid(49152), stream=stream0)
        buf128 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf127, buf128, 768, 64, grid=grid(768), stream=stream0)
        buf129 = buf124; del buf124  # reuse
        triton__14.run(buf118, buf129, 6291456, grid=grid(6291456), stream=stream0)
        buf130 = as_strided(buf118, (8192, 768), (768, 1)); del buf118  # reuse
        extern_kernels.mm(buf129, permute_204, out=buf130)
        del permute_204
        buf131 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf129, (768, 8192), (1, 768)), view_190, out=buf131)
        del view_190
        buf132 = buf127; del buf127  # reuse
        triton__4.run(buf129, buf132, 49152, 128, grid=grid(49152), stream=stream0)
        buf133 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf132, buf133, 768, 64, grid=grid(768), stream=stream0)
        buf134 = buf75; del buf75  # reuse
        buf141 = as_strided(buf129, (16, 512, 768), (393216, 768, 1)); del buf129  # reuse
        buf144 = buf44; del buf44  # reuse
        buf145 = as_strided(buf144, (8192, 768), (768, 1)); del buf144  # reuse
        triton__20.run(buf145, buf103, buf120, buf125, buf130, primals_164, add_81, getitem_41, rsqrt_20, philox_seed_like, buf134, buf141, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_164
        buf137 = as_strided(buf132, (768, 64), (1, 768)); del buf132  # reuse
        buf139 = buf99; del buf99  # reuse
        triton__17.run(buf103, buf120, buf125, buf130, add_81, getitem_41, rsqrt_20, buf137, buf139, 49152, 128, grid=grid(49152), stream=stream0)
        del add_81
        del getitem_41
        del rsqrt_20
        buf142 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf137, buf142, 768, 64, grid=grid(768), stream=stream0)
        buf143 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf139, buf143, 768, 64, grid=grid(768), stream=stream0)
        buf146 = as_strided(buf92, (8192, 3072), (3072, 1)); del buf92  # reuse
        extern_kernels.mm(buf145, permute_208, out=buf146)
        del permute_208
        buf147 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf145, (768, 8192), (1, 768)), view_188, out=buf147)
        del view_188
        buf148 = as_strided(buf139, (1, 768, 64), (49152, 1, 768)); del buf139  # reuse
        triton__4.run(buf145, buf148, 49152, 128, grid=grid(49152), stream=stream0)
        buf149 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf148, buf149, 768, 64, grid=grid(768), stream=stream0)
        buf150 = as_strided(buf146, (16, 512, 3072), (1572864, 3072, 1)); del buf146  # reuse
        buf151 = as_strided(buf150, (8192, 3072), (3072, 1)); del buf150  # reuse
        triton__7.run(buf151, addmm_58, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_58
        buf152 = as_strided(buf145, (8192, 768), (768, 1)); del buf145  # reuse
        extern_kernels.mm(buf151, permute_212, out=buf152)
        del permute_212
        buf153 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf151, (3072, 8192), (1, 3072)), view_186, out=buf153)
        del view_186
        buf154 = buf95; del buf95  # reuse
        triton__8.run(buf151, buf154, 98304, 256, grid=grid(98304), stream=stream0)
        buf155 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf154, buf155, 3072, 32, grid=grid(3072), stream=stream0)
        buf162 = as_strided(buf130, (16, 512, 768), (393216, 768, 1)); del buf130  # reuse
        buf165 = as_strided(buf125, (16, 512, 768), (393216, 768, 1)); del buf125  # reuse
        buf166 = as_strided(buf165, (8192, 768), (768, 1)); del buf165  # reuse
        triton__21.run(buf166, buf141, buf152, primals_158, add_77, getitem_39, rsqrt_19, philox_seed_like, buf162, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_158
        buf158 = as_strided(buf148, (768, 64), (1, 768)); del buf148  # reuse
        buf160 = buf137; del buf137  # reuse
        triton__11.run(buf141, buf152, add_77, getitem_39, rsqrt_19, buf158, buf160, 49152, 128, grid=grid(49152), stream=stream0)
        del add_77
        del getitem_39
        del rsqrt_19
        buf163 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf158, buf163, 768, 64, grid=grid(768), stream=stream0)
        buf164 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf160, buf164, 768, 64, grid=grid(768), stream=stream0)
        buf167 = buf152; del buf152  # reuse
        extern_kernels.mm(buf166, permute_216, out=buf167)
        del permute_216
        buf168 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf166, (768, 8192), (1, 768)), view_184, out=buf168)
        del view_184
        buf169 = as_strided(buf160, (1, 768, 64), (49152, 1, 768)); del buf160  # reuse
        triton__4.run(buf166, buf169, 49152, 128, grid=grid(49152), stream=stream0)
        buf170 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf169, buf170, 768, 64, grid=grid(768), stream=stream0)
        buf171 = as_strided(buf166, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf166  # reuse
        triton__12.run(buf167, buf171, 6291456, grid=grid(6291456), stream=stream0)
        buf172 = as_strided(buf167, (192, 512, 64), (32768, 64, 1)); del buf167  # reuse
        extern_kernels.bmm(permute_221, as_strided(buf171, (192, 512, 64), (32768, 64, 1)), out=buf172)
        del permute_221
        buf173 = as_strided(buf116, (192, 512, 512), (262144, 512, 1)); del buf116  # reuse
        extern_kernels.bmm(as_strided(buf171, (192, 512, 64), (32768, 64, 1)), permute_222, out=buf173)
        del permute_222
        buf175 = as_strided(buf114, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf114  # reuse
        triton__22.run(philox_seed_like, buf173, convert_element_type_104, buf175, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_104
        buf176 = as_strided(buf171, (192, 64, 512), (32768, 512, 1)); del buf171  # reuse
        extern_kernels.bmm(permute_223, as_strided(buf175, (192, 512, 512), (262144, 512, 1)), out=buf176)
        del permute_223
        buf177 = as_strided(buf141, (192, 512, 64), (32768, 64, 1)); del buf141  # reuse
        extern_kernels.bmm(as_strided(buf175, (192, 512, 512), (262144, 512, 1)), permute_224, out=buf177)
        del permute_224
        buf178 = as_strided(buf120, (8192, 768), (768, 1)); del buf120  # reuse
        triton__14.run(buf172, buf178, 6291456, grid=grid(6291456), stream=stream0)
        buf179 = as_strided(buf172, (8192, 768), (768, 1)); del buf172  # reuse
        extern_kernels.mm(buf178, permute_228, out=buf179)
        del permute_228
        buf180 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf178, (768, 8192), (1, 768)), view_171, out=buf180)
        buf181 = buf169; del buf169  # reuse
        triton__4.run(buf178, buf181, 49152, 128, grid=grid(49152), stream=stream0)
        buf182 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf181, buf182, 768, 64, grid=grid(768), stream=stream0)
        buf183 = buf178; del buf178  # reuse
        triton__15.run(buf176, buf183, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf184 = as_strided(buf176, (8192, 768), (768, 1)); del buf176  # reuse
        extern_kernels.mm(buf183, permute_233, out=buf184)
        del permute_233
        buf185 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf183, (768, 8192), (1, 768)), view_171, out=buf185)
        buf186 = buf181; del buf181  # reuse
        triton__4.run(buf183, buf186, 49152, 128, grid=grid(49152), stream=stream0)
        buf187 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf186, buf187, 768, 64, grid=grid(768), stream=stream0)
        buf188 = buf183; del buf183  # reuse
        triton__14.run(buf177, buf188, 6291456, grid=grid(6291456), stream=stream0)
        buf189 = as_strided(buf177, (8192, 768), (768, 1)); del buf177  # reuse
        extern_kernels.mm(buf188, permute_237, out=buf189)
        del permute_237
        buf190 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf188, (768, 8192), (1, 768)), view_171, out=buf190)
        del view_171
        buf191 = buf186; del buf186  # reuse
        triton__4.run(buf188, buf191, 49152, 128, grid=grid(49152), stream=stream0)
        buf192 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf191, buf192, 768, 64, grid=grid(768), stream=stream0)
        buf193 = buf134; del buf134  # reuse
        buf200 = as_strided(buf188, (16, 512, 768), (393216, 768, 1)); del buf188  # reuse
        buf203 = buf103; del buf103  # reuse
        buf204 = as_strided(buf203, (8192, 768), (768, 1)); del buf203  # reuse
        triton__23.run(buf204, buf162, buf179, buf184, buf189, primals_148, add_73, getitem_37, rsqrt_18, philox_seed_like, buf193, buf200, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_148
        buf196 = as_strided(buf191, (768, 64), (1, 768)); del buf191  # reuse
        buf198 = buf158; del buf158  # reuse
        triton__17.run(buf162, buf179, buf184, buf189, add_73, getitem_37, rsqrt_18, buf196, buf198, 49152, 128, grid=grid(49152), stream=stream0)
        del add_73
        del getitem_37
        del rsqrt_18
        buf201 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf196, buf201, 768, 64, grid=grid(768), stream=stream0)
        buf202 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf198, buf202, 768, 64, grid=grid(768), stream=stream0)
        buf205 = as_strided(buf151, (8192, 3072), (3072, 1)); del buf151  # reuse
        extern_kernels.mm(buf204, permute_241, out=buf205)
        del permute_241
        buf206 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf204, (768, 8192), (1, 768)), view_169, out=buf206)
        del view_169
        buf207 = as_strided(buf198, (1, 768, 64), (49152, 1, 768)); del buf198  # reuse
        triton__4.run(buf204, buf207, 49152, 128, grid=grid(49152), stream=stream0)
        buf208 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf207, buf208, 768, 64, grid=grid(768), stream=stream0)
        buf209 = as_strided(buf205, (16, 512, 3072), (1572864, 3072, 1)); del buf205  # reuse
        buf210 = as_strided(buf209, (8192, 3072), (3072, 1)); del buf209  # reuse
        triton__7.run(buf210, addmm_52, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_52
        buf211 = as_strided(buf204, (8192, 768), (768, 1)); del buf204  # reuse
        extern_kernels.mm(buf210, permute_245, out=buf211)
        del permute_245
        buf212 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf210, (3072, 8192), (1, 3072)), view_167, out=buf212)
        del view_167
        buf213 = buf154; del buf154  # reuse
        triton__8.run(buf210, buf213, 98304, 256, grid=grid(98304), stream=stream0)
        buf214 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf213, buf214, 3072, 32, grid=grid(3072), stream=stream0)
        buf221 = as_strided(buf189, (16, 512, 768), (393216, 768, 1)); del buf189  # reuse
        buf224 = as_strided(buf184, (16, 512, 768), (393216, 768, 1)); del buf184  # reuse
        buf225 = as_strided(buf224, (8192, 768), (768, 1)); del buf224  # reuse
        triton__24.run(buf225, buf200, buf211, primals_142, add_69, getitem_35, rsqrt_17, philox_seed_like, buf221, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_142
        buf217 = as_strided(buf207, (768, 64), (1, 768)); del buf207  # reuse
        buf219 = buf196; del buf196  # reuse
        triton__11.run(buf200, buf211, add_69, getitem_35, rsqrt_17, buf217, buf219, 49152, 128, grid=grid(49152), stream=stream0)
        del add_69
        del getitem_35
        del rsqrt_17
        buf222 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf217, buf222, 768, 64, grid=grid(768), stream=stream0)
        buf223 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf219, buf223, 768, 64, grid=grid(768), stream=stream0)
        buf226 = buf211; del buf211  # reuse
        extern_kernels.mm(buf225, permute_249, out=buf226)
        del permute_249
        buf227 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf225, (768, 8192), (1, 768)), view_165, out=buf227)
        del view_165
        buf228 = as_strided(buf219, (1, 768, 64), (49152, 1, 768)); del buf219  # reuse
        triton__4.run(buf225, buf228, 49152, 128, grid=grid(49152), stream=stream0)
        buf229 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf228, buf229, 768, 64, grid=grid(768), stream=stream0)
        buf230 = as_strided(buf225, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf225  # reuse
        triton__12.run(buf226, buf230, 6291456, grid=grid(6291456), stream=stream0)
        buf231 = as_strided(buf226, (192, 512, 64), (32768, 64, 1)); del buf226  # reuse
        extern_kernels.bmm(permute_254, as_strided(buf230, (192, 512, 64), (32768, 64, 1)), out=buf231)
        del permute_254
        buf232 = as_strided(buf175, (192, 512, 512), (262144, 512, 1)); del buf175  # reuse
        extern_kernels.bmm(as_strided(buf230, (192, 512, 64), (32768, 64, 1)), permute_255, out=buf232)
        del permute_255
        buf234 = as_strided(buf173, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf173  # reuse
        triton__25.run(philox_seed_like, buf232, convert_element_type_93, buf234, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_93
        buf235 = as_strided(buf230, (192, 64, 512), (32768, 512, 1)); del buf230  # reuse
        extern_kernels.bmm(permute_256, as_strided(buf234, (192, 512, 512), (262144, 512, 1)), out=buf235)
        del permute_256
        buf236 = as_strided(buf200, (192, 512, 64), (32768, 64, 1)); del buf200  # reuse
        extern_kernels.bmm(as_strided(buf234, (192, 512, 512), (262144, 512, 1)), permute_257, out=buf236)
        del permute_257
        buf237 = as_strided(buf179, (8192, 768), (768, 1)); del buf179  # reuse
        triton__14.run(buf231, buf237, 6291456, grid=grid(6291456), stream=stream0)
        buf238 = as_strided(buf231, (8192, 768), (768, 1)); del buf231  # reuse
        extern_kernels.mm(buf237, permute_261, out=buf238)
        del permute_261
        buf239 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf237, (768, 8192), (1, 768)), view_152, out=buf239)
        buf240 = buf228; del buf228  # reuse
        triton__4.run(buf237, buf240, 49152, 128, grid=grid(49152), stream=stream0)
        buf241 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf240, buf241, 768, 64, grid=grid(768), stream=stream0)
        buf242 = buf237; del buf237  # reuse
        triton__15.run(buf235, buf242, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf243 = as_strided(buf235, (8192, 768), (768, 1)); del buf235  # reuse
        extern_kernels.mm(buf242, permute_266, out=buf243)
        del permute_266
        buf244 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf242, (768, 8192), (1, 768)), view_152, out=buf244)
        buf245 = buf240; del buf240  # reuse
        triton__4.run(buf242, buf245, 49152, 128, grid=grid(49152), stream=stream0)
        buf246 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf245, buf246, 768, 64, grid=grid(768), stream=stream0)
        buf247 = buf242; del buf242  # reuse
        triton__14.run(buf236, buf247, 6291456, grid=grid(6291456), stream=stream0)
        buf248 = as_strided(buf236, (8192, 768), (768, 1)); del buf236  # reuse
        extern_kernels.mm(buf247, permute_270, out=buf248)
        del permute_270
        buf249 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf247, (768, 8192), (1, 768)), view_152, out=buf249)
        del view_152
        buf250 = buf245; del buf245  # reuse
        triton__4.run(buf247, buf250, 49152, 128, grid=grid(49152), stream=stream0)
        buf251 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf250, buf251, 768, 64, grid=grid(768), stream=stream0)
        buf252 = buf193; del buf193  # reuse
        buf259 = as_strided(buf247, (16, 512, 768), (393216, 768, 1)); del buf247  # reuse
        buf262 = buf162; del buf162  # reuse
        buf263 = as_strided(buf262, (8192, 768), (768, 1)); del buf262  # reuse
        triton__26.run(buf263, buf221, buf238, buf243, buf248, primals_132, add_65, getitem_33, rsqrt_16, philox_seed_like, buf252, buf259, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_132
        buf255 = as_strided(buf250, (768, 64), (1, 768)); del buf250  # reuse
        buf257 = buf217; del buf217  # reuse
        triton__17.run(buf221, buf238, buf243, buf248, add_65, getitem_33, rsqrt_16, buf255, buf257, 49152, 128, grid=grid(49152), stream=stream0)
        del add_65
        del getitem_33
        del rsqrt_16
        buf260 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf255, buf260, 768, 64, grid=grid(768), stream=stream0)
        buf261 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf257, buf261, 768, 64, grid=grid(768), stream=stream0)
        buf264 = as_strided(buf210, (8192, 3072), (3072, 1)); del buf210  # reuse
        extern_kernels.mm(buf263, permute_274, out=buf264)
        del permute_274
        buf265 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf263, (768, 8192), (1, 768)), view_150, out=buf265)
        del view_150
        buf266 = as_strided(buf257, (1, 768, 64), (49152, 1, 768)); del buf257  # reuse
        triton__4.run(buf263, buf266, 49152, 128, grid=grid(49152), stream=stream0)
        buf267 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf266, buf267, 768, 64, grid=grid(768), stream=stream0)
        buf268 = as_strided(buf264, (16, 512, 3072), (1572864, 3072, 1)); del buf264  # reuse
        buf269 = as_strided(buf268, (8192, 3072), (3072, 1)); del buf268  # reuse
        triton__7.run(buf269, addmm_46, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_46
        buf270 = as_strided(buf263, (8192, 768), (768, 1)); del buf263  # reuse
        extern_kernels.mm(buf269, permute_278, out=buf270)
        del permute_278
        buf271 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf269, (3072, 8192), (1, 3072)), view_148, out=buf271)
        del view_148
        buf272 = buf213; del buf213  # reuse
        triton__8.run(buf269, buf272, 98304, 256, grid=grid(98304), stream=stream0)
        buf273 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf272, buf273, 3072, 32, grid=grid(3072), stream=stream0)
        buf280 = as_strided(buf248, (16, 512, 768), (393216, 768, 1)); del buf248  # reuse
        buf283 = as_strided(buf243, (16, 512, 768), (393216, 768, 1)); del buf243  # reuse
        buf284 = as_strided(buf283, (8192, 768), (768, 1)); del buf283  # reuse
        triton__27.run(buf284, buf259, buf270, primals_126, add_61, getitem_31, rsqrt_15, philox_seed_like, buf280, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_126
        buf276 = as_strided(buf266, (768, 64), (1, 768)); del buf266  # reuse
        buf278 = buf255; del buf255  # reuse
        triton__11.run(buf259, buf270, add_61, getitem_31, rsqrt_15, buf276, buf278, 49152, 128, grid=grid(49152), stream=stream0)
        del add_61
        del getitem_31
        del rsqrt_15
        buf281 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf276, buf281, 768, 64, grid=grid(768), stream=stream0)
        buf282 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf278, buf282, 768, 64, grid=grid(768), stream=stream0)
        buf285 = buf270; del buf270  # reuse
        extern_kernels.mm(buf284, permute_282, out=buf285)
        del permute_282
        buf286 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf284, (768, 8192), (1, 768)), view_146, out=buf286)
        del view_146
        buf287 = as_strided(buf278, (1, 768, 64), (49152, 1, 768)); del buf278  # reuse
        triton__4.run(buf284, buf287, 49152, 128, grid=grid(49152), stream=stream0)
        buf288 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf287, buf288, 768, 64, grid=grid(768), stream=stream0)
        buf289 = as_strided(buf284, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf284  # reuse
        triton__12.run(buf285, buf289, 6291456, grid=grid(6291456), stream=stream0)
        buf290 = as_strided(buf285, (192, 512, 64), (32768, 64, 1)); del buf285  # reuse
        extern_kernels.bmm(permute_287, as_strided(buf289, (192, 512, 64), (32768, 64, 1)), out=buf290)
        del permute_287
        buf291 = as_strided(buf234, (192, 512, 512), (262144, 512, 1)); del buf234  # reuse
        extern_kernels.bmm(as_strided(buf289, (192, 512, 64), (32768, 64, 1)), permute_288, out=buf291)
        del permute_288
        buf293 = as_strided(buf232, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf232  # reuse
        triton__28.run(philox_seed_like, buf291, convert_element_type_82, buf293, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_82
        buf294 = as_strided(buf289, (192, 64, 512), (32768, 512, 1)); del buf289  # reuse
        extern_kernels.bmm(permute_289, as_strided(buf293, (192, 512, 512), (262144, 512, 1)), out=buf294)
        del permute_289
        buf295 = as_strided(buf259, (192, 512, 64), (32768, 64, 1)); del buf259  # reuse
        extern_kernels.bmm(as_strided(buf293, (192, 512, 512), (262144, 512, 1)), permute_290, out=buf295)
        del permute_290
        buf296 = as_strided(buf238, (8192, 768), (768, 1)); del buf238  # reuse
        triton__14.run(buf290, buf296, 6291456, grid=grid(6291456), stream=stream0)
        buf297 = as_strided(buf290, (8192, 768), (768, 1)); del buf290  # reuse
        extern_kernels.mm(buf296, permute_294, out=buf297)
        del permute_294
        buf298 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf296, (768, 8192), (1, 768)), view_133, out=buf298)
        buf299 = buf287; del buf287  # reuse
        triton__4.run(buf296, buf299, 49152, 128, grid=grid(49152), stream=stream0)
        buf300 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf299, buf300, 768, 64, grid=grid(768), stream=stream0)
        buf301 = buf296; del buf296  # reuse
        triton__15.run(buf294, buf301, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf302 = as_strided(buf294, (8192, 768), (768, 1)); del buf294  # reuse
        extern_kernels.mm(buf301, permute_299, out=buf302)
        del permute_299
        buf303 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf301, (768, 8192), (1, 768)), view_133, out=buf303)
        buf304 = buf299; del buf299  # reuse
        triton__4.run(buf301, buf304, 49152, 128, grid=grid(49152), stream=stream0)
        buf305 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf304, buf305, 768, 64, grid=grid(768), stream=stream0)
        buf306 = buf301; del buf301  # reuse
        triton__14.run(buf295, buf306, 6291456, grid=grid(6291456), stream=stream0)
        buf307 = as_strided(buf295, (8192, 768), (768, 1)); del buf295  # reuse
        extern_kernels.mm(buf306, permute_303, out=buf307)
        del permute_303
        buf308 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf306, (768, 8192), (1, 768)), view_133, out=buf308)
        del view_133
        buf309 = buf304; del buf304  # reuse
        triton__4.run(buf306, buf309, 49152, 128, grid=grid(49152), stream=stream0)
        buf310 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf309, buf310, 768, 64, grid=grid(768), stream=stream0)
        buf311 = buf252; del buf252  # reuse
        buf318 = as_strided(buf306, (16, 512, 768), (393216, 768, 1)); del buf306  # reuse
        buf321 = buf221; del buf221  # reuse
        buf322 = as_strided(buf321, (8192, 768), (768, 1)); del buf321  # reuse
        triton__29.run(buf322, buf280, buf297, buf302, buf307, primals_116, add_57, getitem_29, rsqrt_14, philox_seed_like, buf311, buf318, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_116
        buf314 = as_strided(buf309, (768, 64), (1, 768)); del buf309  # reuse
        buf316 = buf276; del buf276  # reuse
        triton__17.run(buf280, buf297, buf302, buf307, add_57, getitem_29, rsqrt_14, buf314, buf316, 49152, 128, grid=grid(49152), stream=stream0)
        del add_57
        del getitem_29
        del rsqrt_14
        buf319 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf314, buf319, 768, 64, grid=grid(768), stream=stream0)
        buf320 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf316, buf320, 768, 64, grid=grid(768), stream=stream0)
        buf323 = as_strided(buf269, (8192, 3072), (3072, 1)); del buf269  # reuse
        extern_kernels.mm(buf322, permute_307, out=buf323)
        del permute_307
        buf324 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf322, (768, 8192), (1, 768)), view_131, out=buf324)
        del view_131
        buf325 = as_strided(buf316, (1, 768, 64), (49152, 1, 768)); del buf316  # reuse
        triton__4.run(buf322, buf325, 49152, 128, grid=grid(49152), stream=stream0)
        buf326 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf325, buf326, 768, 64, grid=grid(768), stream=stream0)
        buf327 = as_strided(buf323, (16, 512, 3072), (1572864, 3072, 1)); del buf323  # reuse
        buf328 = as_strided(buf327, (8192, 3072), (3072, 1)); del buf327  # reuse
        triton__7.run(buf328, addmm_40, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_40
        buf329 = as_strided(buf322, (8192, 768), (768, 1)); del buf322  # reuse
        extern_kernels.mm(buf328, permute_311, out=buf329)
        del permute_311
        buf330 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf328, (3072, 8192), (1, 3072)), view_129, out=buf330)
        del view_129
        buf331 = buf272; del buf272  # reuse
        triton__8.run(buf328, buf331, 98304, 256, grid=grid(98304), stream=stream0)
        buf332 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf331, buf332, 3072, 32, grid=grid(3072), stream=stream0)
        buf339 = as_strided(buf307, (16, 512, 768), (393216, 768, 1)); del buf307  # reuse
        buf342 = as_strided(buf302, (16, 512, 768), (393216, 768, 1)); del buf302  # reuse
        buf343 = as_strided(buf342, (8192, 768), (768, 1)); del buf342  # reuse
        triton__30.run(buf343, buf318, buf329, primals_110, add_53, getitem_27, rsqrt_13, philox_seed_like, buf339, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_110
        buf335 = as_strided(buf325, (768, 64), (1, 768)); del buf325  # reuse
        buf337 = buf314; del buf314  # reuse
        triton__11.run(buf318, buf329, add_53, getitem_27, rsqrt_13, buf335, buf337, 49152, 128, grid=grid(49152), stream=stream0)
        del add_53
        del getitem_27
        del rsqrt_13
        buf340 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf335, buf340, 768, 64, grid=grid(768), stream=stream0)
        buf341 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf337, buf341, 768, 64, grid=grid(768), stream=stream0)
        buf344 = buf329; del buf329  # reuse
        extern_kernels.mm(buf343, permute_315, out=buf344)
        del permute_315
        buf345 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf343, (768, 8192), (1, 768)), view_127, out=buf345)
        del view_127
        buf346 = as_strided(buf337, (1, 768, 64), (49152, 1, 768)); del buf337  # reuse
        triton__4.run(buf343, buf346, 49152, 128, grid=grid(49152), stream=stream0)
        buf347 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf346, buf347, 768, 64, grid=grid(768), stream=stream0)
        buf348 = as_strided(buf343, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf343  # reuse
        triton__12.run(buf344, buf348, 6291456, grid=grid(6291456), stream=stream0)
        buf349 = as_strided(buf344, (192, 512, 64), (32768, 64, 1)); del buf344  # reuse
        extern_kernels.bmm(permute_320, as_strided(buf348, (192, 512, 64), (32768, 64, 1)), out=buf349)
        del permute_320
        buf350 = as_strided(buf293, (192, 512, 512), (262144, 512, 1)); del buf293  # reuse
        extern_kernels.bmm(as_strided(buf348, (192, 512, 64), (32768, 64, 1)), permute_321, out=buf350)
        del permute_321
        buf352 = as_strided(buf291, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf291  # reuse
        triton__31.run(philox_seed_like, buf350, convert_element_type_71, buf352, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_71
        buf353 = as_strided(buf348, (192, 64, 512), (32768, 512, 1)); del buf348  # reuse
        extern_kernels.bmm(permute_322, as_strided(buf352, (192, 512, 512), (262144, 512, 1)), out=buf353)
        del permute_322
        buf354 = as_strided(buf318, (192, 512, 64), (32768, 64, 1)); del buf318  # reuse
        extern_kernels.bmm(as_strided(buf352, (192, 512, 512), (262144, 512, 1)), permute_323, out=buf354)
        del permute_323
        buf355 = as_strided(buf297, (8192, 768), (768, 1)); del buf297  # reuse
        triton__14.run(buf349, buf355, 6291456, grid=grid(6291456), stream=stream0)
        buf356 = as_strided(buf349, (8192, 768), (768, 1)); del buf349  # reuse
        extern_kernels.mm(buf355, permute_327, out=buf356)
        del permute_327
        buf357 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf355, (768, 8192), (1, 768)), view_114, out=buf357)
        buf358 = buf346; del buf346  # reuse
        triton__4.run(buf355, buf358, 49152, 128, grid=grid(49152), stream=stream0)
        buf359 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf358, buf359, 768, 64, grid=grid(768), stream=stream0)
        buf360 = buf355; del buf355  # reuse
        triton__15.run(buf353, buf360, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf361 = as_strided(buf353, (8192, 768), (768, 1)); del buf353  # reuse
        extern_kernels.mm(buf360, permute_332, out=buf361)
        del permute_332
        buf362 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf360, (768, 8192), (1, 768)), view_114, out=buf362)
        buf363 = buf358; del buf358  # reuse
        triton__4.run(buf360, buf363, 49152, 128, grid=grid(49152), stream=stream0)
        buf364 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf363, buf364, 768, 64, grid=grid(768), stream=stream0)
        buf365 = buf360; del buf360  # reuse
        triton__14.run(buf354, buf365, 6291456, grid=grid(6291456), stream=stream0)
        buf366 = as_strided(buf354, (8192, 768), (768, 1)); del buf354  # reuse
        extern_kernels.mm(buf365, permute_336, out=buf366)
        del permute_336
        buf367 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf365, (768, 8192), (1, 768)), view_114, out=buf367)
        del view_114
        buf368 = buf363; del buf363  # reuse
        triton__4.run(buf365, buf368, 49152, 128, grid=grid(49152), stream=stream0)
        buf369 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf368, buf369, 768, 64, grid=grid(768), stream=stream0)
        buf370 = buf311; del buf311  # reuse
        buf377 = as_strided(buf365, (16, 512, 768), (393216, 768, 1)); del buf365  # reuse
        buf380 = buf280; del buf280  # reuse
        buf381 = as_strided(buf380, (8192, 768), (768, 1)); del buf380  # reuse
        triton__32.run(buf381, buf339, buf356, buf361, buf366, primals_100, add_49, getitem_25, rsqrt_12, philox_seed_like, buf370, buf377, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_100
        buf373 = as_strided(buf368, (768, 64), (1, 768)); del buf368  # reuse
        buf375 = buf335; del buf335  # reuse
        triton__17.run(buf339, buf356, buf361, buf366, add_49, getitem_25, rsqrt_12, buf373, buf375, 49152, 128, grid=grid(49152), stream=stream0)
        del add_49
        del getitem_25
        del rsqrt_12
        buf378 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf373, buf378, 768, 64, grid=grid(768), stream=stream0)
        buf379 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf375, buf379, 768, 64, grid=grid(768), stream=stream0)
        buf382 = as_strided(buf328, (8192, 3072), (3072, 1)); del buf328  # reuse
        extern_kernels.mm(buf381, permute_340, out=buf382)
        del permute_340
        buf383 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf381, (768, 8192), (1, 768)), view_112, out=buf383)
        del view_112
        buf384 = as_strided(buf375, (1, 768, 64), (49152, 1, 768)); del buf375  # reuse
        triton__4.run(buf381, buf384, 49152, 128, grid=grid(49152), stream=stream0)
        buf385 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf384, buf385, 768, 64, grid=grid(768), stream=stream0)
        buf386 = as_strided(buf382, (16, 512, 3072), (1572864, 3072, 1)); del buf382  # reuse
        buf387 = as_strided(buf386, (8192, 3072), (3072, 1)); del buf386  # reuse
        triton__7.run(buf387, addmm_34, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_34
        buf388 = as_strided(buf381, (8192, 768), (768, 1)); del buf381  # reuse
        extern_kernels.mm(buf387, permute_344, out=buf388)
        del permute_344
        buf389 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf387, (3072, 8192), (1, 3072)), view_110, out=buf389)
        del view_110
        buf390 = buf331; del buf331  # reuse
        triton__8.run(buf387, buf390, 98304, 256, grid=grid(98304), stream=stream0)
        buf391 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf390, buf391, 3072, 32, grid=grid(3072), stream=stream0)
        buf398 = as_strided(buf366, (16, 512, 768), (393216, 768, 1)); del buf366  # reuse
        buf401 = as_strided(buf361, (16, 512, 768), (393216, 768, 1)); del buf361  # reuse
        buf402 = as_strided(buf401, (8192, 768), (768, 1)); del buf401  # reuse
        triton__33.run(buf402, buf377, buf388, primals_94, add_45, getitem_23, rsqrt_11, philox_seed_like, buf398, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_94
        buf394 = as_strided(buf384, (768, 64), (1, 768)); del buf384  # reuse
        buf396 = buf373; del buf373  # reuse
        triton__11.run(buf377, buf388, add_45, getitem_23, rsqrt_11, buf394, buf396, 49152, 128, grid=grid(49152), stream=stream0)
        del add_45
        del getitem_23
        del rsqrt_11
        buf399 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf394, buf399, 768, 64, grid=grid(768), stream=stream0)
        buf400 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf396, buf400, 768, 64, grid=grid(768), stream=stream0)
        buf403 = buf388; del buf388  # reuse
        extern_kernels.mm(buf402, permute_348, out=buf403)
        del permute_348
        buf404 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf402, (768, 8192), (1, 768)), view_108, out=buf404)
        del view_108
        buf405 = as_strided(buf396, (1, 768, 64), (49152, 1, 768)); del buf396  # reuse
        triton__4.run(buf402, buf405, 49152, 128, grid=grid(49152), stream=stream0)
        buf406 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf405, buf406, 768, 64, grid=grid(768), stream=stream0)
        buf407 = as_strided(buf402, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf402  # reuse
        triton__12.run(buf403, buf407, 6291456, grid=grid(6291456), stream=stream0)
        buf408 = as_strided(buf403, (192, 512, 64), (32768, 64, 1)); del buf403  # reuse
        extern_kernels.bmm(permute_353, as_strided(buf407, (192, 512, 64), (32768, 64, 1)), out=buf408)
        del permute_353
        buf409 = as_strided(buf352, (192, 512, 512), (262144, 512, 1)); del buf352  # reuse
        extern_kernels.bmm(as_strided(buf407, (192, 512, 64), (32768, 64, 1)), permute_354, out=buf409)
        del permute_354
        buf411 = as_strided(buf350, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf350  # reuse
        triton__34.run(philox_seed_like, buf409, convert_element_type_60, buf411, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_60
        buf412 = as_strided(buf407, (192, 64, 512), (32768, 512, 1)); del buf407  # reuse
        extern_kernels.bmm(permute_355, as_strided(buf411, (192, 512, 512), (262144, 512, 1)), out=buf412)
        del permute_355
        buf413 = as_strided(buf377, (192, 512, 64), (32768, 64, 1)); del buf377  # reuse
        extern_kernels.bmm(as_strided(buf411, (192, 512, 512), (262144, 512, 1)), permute_356, out=buf413)
        del permute_356
        buf414 = as_strided(buf356, (8192, 768), (768, 1)); del buf356  # reuse
        triton__14.run(buf408, buf414, 6291456, grid=grid(6291456), stream=stream0)
        buf415 = as_strided(buf408, (8192, 768), (768, 1)); del buf408  # reuse
        extern_kernels.mm(buf414, permute_360, out=buf415)
        del permute_360
        buf416 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf414, (768, 8192), (1, 768)), view_95, out=buf416)
        buf417 = buf405; del buf405  # reuse
        triton__4.run(buf414, buf417, 49152, 128, grid=grid(49152), stream=stream0)
        buf418 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf417, buf418, 768, 64, grid=grid(768), stream=stream0)
        buf419 = buf414; del buf414  # reuse
        triton__15.run(buf412, buf419, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf420 = as_strided(buf412, (8192, 768), (768, 1)); del buf412  # reuse
        extern_kernels.mm(buf419, permute_365, out=buf420)
        del permute_365
        buf421 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf419, (768, 8192), (1, 768)), view_95, out=buf421)
        buf422 = buf417; del buf417  # reuse
        triton__4.run(buf419, buf422, 49152, 128, grid=grid(49152), stream=stream0)
        buf423 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf422, buf423, 768, 64, grid=grid(768), stream=stream0)
        buf424 = buf419; del buf419  # reuse
        triton__14.run(buf413, buf424, 6291456, grid=grid(6291456), stream=stream0)
        buf425 = as_strided(buf413, (8192, 768), (768, 1)); del buf413  # reuse
        extern_kernels.mm(buf424, permute_369, out=buf425)
        del permute_369
        buf426 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf424, (768, 8192), (1, 768)), view_95, out=buf426)
        del view_95
        buf427 = buf422; del buf422  # reuse
        triton__4.run(buf424, buf427, 49152, 128, grid=grid(49152), stream=stream0)
        buf428 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf427, buf428, 768, 64, grid=grid(768), stream=stream0)
        buf429 = buf370; del buf370  # reuse
        buf436 = as_strided(buf424, (16, 512, 768), (393216, 768, 1)); del buf424  # reuse
        buf439 = buf339; del buf339  # reuse
        buf440 = as_strided(buf439, (8192, 768), (768, 1)); del buf439  # reuse
        triton__35.run(buf440, buf398, buf415, buf420, buf425, primals_84, add_41, getitem_21, rsqrt_10, philox_seed_like, buf429, buf436, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_84
        buf432 = as_strided(buf427, (768, 64), (1, 768)); del buf427  # reuse
        buf434 = buf394; del buf394  # reuse
        triton__17.run(buf398, buf415, buf420, buf425, add_41, getitem_21, rsqrt_10, buf432, buf434, 49152, 128, grid=grid(49152), stream=stream0)
        del add_41
        del getitem_21
        del rsqrt_10
        buf437 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf432, buf437, 768, 64, grid=grid(768), stream=stream0)
        buf438 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf434, buf438, 768, 64, grid=grid(768), stream=stream0)
        buf441 = as_strided(buf387, (8192, 3072), (3072, 1)); del buf387  # reuse
        extern_kernels.mm(buf440, permute_373, out=buf441)
        del permute_373
        buf442 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf440, (768, 8192), (1, 768)), view_93, out=buf442)
        del view_93
        buf443 = as_strided(buf434, (1, 768, 64), (49152, 1, 768)); del buf434  # reuse
        triton__4.run(buf440, buf443, 49152, 128, grid=grid(49152), stream=stream0)
        buf444 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf443, buf444, 768, 64, grid=grid(768), stream=stream0)
        buf445 = as_strided(buf441, (16, 512, 3072), (1572864, 3072, 1)); del buf441  # reuse
        buf446 = as_strided(buf445, (8192, 3072), (3072, 1)); del buf445  # reuse
        triton__7.run(buf446, addmm_28, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_28
        buf447 = as_strided(buf440, (8192, 768), (768, 1)); del buf440  # reuse
        extern_kernels.mm(buf446, permute_377, out=buf447)
        del permute_377
        buf448 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf446, (3072, 8192), (1, 3072)), view_91, out=buf448)
        del view_91
        buf449 = buf390; del buf390  # reuse
        triton__8.run(buf446, buf449, 98304, 256, grid=grid(98304), stream=stream0)
        buf450 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf449, buf450, 3072, 32, grid=grid(3072), stream=stream0)
        buf457 = as_strided(buf425, (16, 512, 768), (393216, 768, 1)); del buf425  # reuse
        buf460 = as_strided(buf420, (16, 512, 768), (393216, 768, 1)); del buf420  # reuse
        buf461 = as_strided(buf460, (8192, 768), (768, 1)); del buf460  # reuse
        triton__36.run(buf461, buf436, buf447, primals_78, add_37, getitem_19, rsqrt_9, philox_seed_like, buf457, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_78
        buf453 = as_strided(buf443, (768, 64), (1, 768)); del buf443  # reuse
        buf455 = buf432; del buf432  # reuse
        triton__11.run(buf436, buf447, add_37, getitem_19, rsqrt_9, buf453, buf455, 49152, 128, grid=grid(49152), stream=stream0)
        del add_37
        del getitem_19
        del rsqrt_9
        buf458 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf453, buf458, 768, 64, grid=grid(768), stream=stream0)
        buf459 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf455, buf459, 768, 64, grid=grid(768), stream=stream0)
        buf462 = buf447; del buf447  # reuse
        extern_kernels.mm(buf461, permute_381, out=buf462)
        del permute_381
        buf463 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf461, (768, 8192), (1, 768)), view_89, out=buf463)
        del view_89
        buf464 = as_strided(buf455, (1, 768, 64), (49152, 1, 768)); del buf455  # reuse
        triton__4.run(buf461, buf464, 49152, 128, grid=grid(49152), stream=stream0)
        buf465 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf464, buf465, 768, 64, grid=grid(768), stream=stream0)
        buf466 = as_strided(buf461, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf461  # reuse
        triton__12.run(buf462, buf466, 6291456, grid=grid(6291456), stream=stream0)
        buf467 = as_strided(buf462, (192, 512, 64), (32768, 64, 1)); del buf462  # reuse
        extern_kernels.bmm(permute_386, as_strided(buf466, (192, 512, 64), (32768, 64, 1)), out=buf467)
        del permute_386
        buf468 = as_strided(buf411, (192, 512, 512), (262144, 512, 1)); del buf411  # reuse
        extern_kernels.bmm(as_strided(buf466, (192, 512, 64), (32768, 64, 1)), permute_387, out=buf468)
        del permute_387
        buf470 = as_strided(buf409, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf409  # reuse
        triton__37.run(philox_seed_like, buf468, convert_element_type_49, buf470, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_49
        buf471 = as_strided(buf466, (192, 64, 512), (32768, 512, 1)); del buf466  # reuse
        extern_kernels.bmm(permute_388, as_strided(buf470, (192, 512, 512), (262144, 512, 1)), out=buf471)
        del permute_388
        buf472 = as_strided(buf436, (192, 512, 64), (32768, 64, 1)); del buf436  # reuse
        extern_kernels.bmm(as_strided(buf470, (192, 512, 512), (262144, 512, 1)), permute_389, out=buf472)
        del permute_389
        buf473 = as_strided(buf415, (8192, 768), (768, 1)); del buf415  # reuse
        triton__14.run(buf467, buf473, 6291456, grid=grid(6291456), stream=stream0)
        buf474 = as_strided(buf467, (8192, 768), (768, 1)); del buf467  # reuse
        extern_kernels.mm(buf473, permute_393, out=buf474)
        del permute_393
        buf475 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf473, (768, 8192), (1, 768)), view_76, out=buf475)
        buf476 = buf464; del buf464  # reuse
        triton__4.run(buf473, buf476, 49152, 128, grid=grid(49152), stream=stream0)
        buf477 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf476, buf477, 768, 64, grid=grid(768), stream=stream0)
        buf478 = buf473; del buf473  # reuse
        triton__15.run(buf471, buf478, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf479 = as_strided(buf471, (8192, 768), (768, 1)); del buf471  # reuse
        extern_kernels.mm(buf478, permute_398, out=buf479)
        del permute_398
        buf480 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf478, (768, 8192), (1, 768)), view_76, out=buf480)
        buf481 = buf476; del buf476  # reuse
        triton__4.run(buf478, buf481, 49152, 128, grid=grid(49152), stream=stream0)
        buf482 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf481, buf482, 768, 64, grid=grid(768), stream=stream0)
        buf483 = buf478; del buf478  # reuse
        triton__14.run(buf472, buf483, 6291456, grid=grid(6291456), stream=stream0)
        buf484 = as_strided(buf472, (8192, 768), (768, 1)); del buf472  # reuse
        extern_kernels.mm(buf483, permute_402, out=buf484)
        del permute_402
        buf485 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf483, (768, 8192), (1, 768)), view_76, out=buf485)
        del view_76
        buf486 = buf481; del buf481  # reuse
        triton__4.run(buf483, buf486, 49152, 128, grid=grid(49152), stream=stream0)
        buf487 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf486, buf487, 768, 64, grid=grid(768), stream=stream0)
        buf488 = buf429; del buf429  # reuse
        buf495 = as_strided(buf483, (16, 512, 768), (393216, 768, 1)); del buf483  # reuse
        buf498 = buf398; del buf398  # reuse
        buf499 = as_strided(buf498, (8192, 768), (768, 1)); del buf498  # reuse
        triton__38.run(buf499, buf457, buf474, buf479, buf484, primals_68, add_33, getitem_17, rsqrt_8, philox_seed_like, buf488, buf495, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_68
        buf491 = as_strided(buf486, (768, 64), (1, 768)); del buf486  # reuse
        buf493 = buf453; del buf453  # reuse
        triton__17.run(buf457, buf474, buf479, buf484, add_33, getitem_17, rsqrt_8, buf491, buf493, 49152, 128, grid=grid(49152), stream=stream0)
        del add_33
        del getitem_17
        del rsqrt_8
        buf496 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf491, buf496, 768, 64, grid=grid(768), stream=stream0)
        buf497 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf493, buf497, 768, 64, grid=grid(768), stream=stream0)
        buf500 = as_strided(buf446, (8192, 3072), (3072, 1)); del buf446  # reuse
        extern_kernels.mm(buf499, permute_406, out=buf500)
        del permute_406
        buf501 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf499, (768, 8192), (1, 768)), view_74, out=buf501)
        del view_74
        buf502 = as_strided(buf493, (1, 768, 64), (49152, 1, 768)); del buf493  # reuse
        triton__4.run(buf499, buf502, 49152, 128, grid=grid(49152), stream=stream0)
        buf503 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf502, buf503, 768, 64, grid=grid(768), stream=stream0)
        buf504 = as_strided(buf500, (16, 512, 3072), (1572864, 3072, 1)); del buf500  # reuse
        buf505 = as_strided(buf504, (8192, 3072), (3072, 1)); del buf504  # reuse
        triton__7.run(buf505, addmm_22, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_22
        buf506 = as_strided(buf499, (8192, 768), (768, 1)); del buf499  # reuse
        extern_kernels.mm(buf505, permute_410, out=buf506)
        del permute_410
        buf507 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf505, (3072, 8192), (1, 3072)), view_72, out=buf507)
        del view_72
        buf508 = buf449; del buf449  # reuse
        triton__8.run(buf505, buf508, 98304, 256, grid=grid(98304), stream=stream0)
        buf509 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf508, buf509, 3072, 32, grid=grid(3072), stream=stream0)
        buf516 = as_strided(buf484, (16, 512, 768), (393216, 768, 1)); del buf484  # reuse
        buf519 = as_strided(buf479, (16, 512, 768), (393216, 768, 1)); del buf479  # reuse
        buf520 = as_strided(buf519, (8192, 768), (768, 1)); del buf519  # reuse
        triton__39.run(buf520, buf495, buf506, primals_62, add_29, getitem_15, rsqrt_7, philox_seed_like, buf516, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_62
        buf512 = as_strided(buf502, (768, 64), (1, 768)); del buf502  # reuse
        buf514 = buf491; del buf491  # reuse
        triton__11.run(buf495, buf506, add_29, getitem_15, rsqrt_7, buf512, buf514, 49152, 128, grid=grid(49152), stream=stream0)
        del add_29
        del getitem_15
        del rsqrt_7
        buf517 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf512, buf517, 768, 64, grid=grid(768), stream=stream0)
        buf518 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf514, buf518, 768, 64, grid=grid(768), stream=stream0)
        buf521 = buf506; del buf506  # reuse
        extern_kernels.mm(buf520, permute_414, out=buf521)
        del permute_414
        buf522 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf520, (768, 8192), (1, 768)), view_70, out=buf522)
        del view_70
        buf523 = as_strided(buf514, (1, 768, 64), (49152, 1, 768)); del buf514  # reuse
        triton__4.run(buf520, buf523, 49152, 128, grid=grid(49152), stream=stream0)
        buf524 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf523, buf524, 768, 64, grid=grid(768), stream=stream0)
        buf525 = as_strided(buf520, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf520  # reuse
        triton__12.run(buf521, buf525, 6291456, grid=grid(6291456), stream=stream0)
        buf526 = as_strided(buf521, (192, 512, 64), (32768, 64, 1)); del buf521  # reuse
        extern_kernels.bmm(permute_419, as_strided(buf525, (192, 512, 64), (32768, 64, 1)), out=buf526)
        del permute_419
        buf527 = as_strided(buf470, (192, 512, 512), (262144, 512, 1)); del buf470  # reuse
        extern_kernels.bmm(as_strided(buf525, (192, 512, 64), (32768, 64, 1)), permute_420, out=buf527)
        del permute_420
        buf529 = as_strided(buf468, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf468  # reuse
        triton__40.run(philox_seed_like, buf527, convert_element_type_38, buf529, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_38
        buf530 = as_strided(buf525, (192, 64, 512), (32768, 512, 1)); del buf525  # reuse
        extern_kernels.bmm(permute_421, as_strided(buf529, (192, 512, 512), (262144, 512, 1)), out=buf530)
        del permute_421
        buf531 = as_strided(buf495, (192, 512, 64), (32768, 64, 1)); del buf495  # reuse
        extern_kernels.bmm(as_strided(buf529, (192, 512, 512), (262144, 512, 1)), permute_422, out=buf531)
        del permute_422
        buf532 = as_strided(buf474, (8192, 768), (768, 1)); del buf474  # reuse
        triton__14.run(buf526, buf532, 6291456, grid=grid(6291456), stream=stream0)
        buf533 = as_strided(buf526, (8192, 768), (768, 1)); del buf526  # reuse
        extern_kernels.mm(buf532, permute_426, out=buf533)
        del permute_426
        buf534 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf532, (768, 8192), (1, 768)), view_57, out=buf534)
        buf535 = buf523; del buf523  # reuse
        triton__4.run(buf532, buf535, 49152, 128, grid=grid(49152), stream=stream0)
        buf536 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf535, buf536, 768, 64, grid=grid(768), stream=stream0)
        buf537 = buf532; del buf532  # reuse
        triton__15.run(buf530, buf537, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf538 = as_strided(buf530, (8192, 768), (768, 1)); del buf530  # reuse
        extern_kernels.mm(buf537, permute_431, out=buf538)
        del permute_431
        buf539 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf537, (768, 8192), (1, 768)), view_57, out=buf539)
        buf540 = buf535; del buf535  # reuse
        triton__4.run(buf537, buf540, 49152, 128, grid=grid(49152), stream=stream0)
        buf541 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf540, buf541, 768, 64, grid=grid(768), stream=stream0)
        buf542 = buf537; del buf537  # reuse
        triton__14.run(buf531, buf542, 6291456, grid=grid(6291456), stream=stream0)
        buf543 = as_strided(buf531, (8192, 768), (768, 1)); del buf531  # reuse
        extern_kernels.mm(buf542, permute_435, out=buf543)
        del permute_435
        buf544 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf542, (768, 8192), (1, 768)), view_57, out=buf544)
        del view_57
        buf545 = buf540; del buf540  # reuse
        triton__4.run(buf542, buf545, 49152, 128, grid=grid(49152), stream=stream0)
        buf546 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf545, buf546, 768, 64, grid=grid(768), stream=stream0)
        buf547 = buf488; del buf488  # reuse
        buf554 = as_strided(buf542, (16, 512, 768), (393216, 768, 1)); del buf542  # reuse
        buf557 = buf457; del buf457  # reuse
        buf558 = as_strided(buf557, (8192, 768), (768, 1)); del buf557  # reuse
        triton__41.run(buf558, buf516, buf533, buf538, buf543, primals_52, add_25, getitem_13, rsqrt_6, philox_seed_like, buf547, buf554, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_52
        buf550 = as_strided(buf545, (768, 64), (1, 768)); del buf545  # reuse
        buf552 = buf512; del buf512  # reuse
        triton__17.run(buf516, buf533, buf538, buf543, add_25, getitem_13, rsqrt_6, buf550, buf552, 49152, 128, grid=grid(49152), stream=stream0)
        del add_25
        del getitem_13
        del rsqrt_6
        buf555 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf550, buf555, 768, 64, grid=grid(768), stream=stream0)
        buf556 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf552, buf556, 768, 64, grid=grid(768), stream=stream0)
        buf559 = as_strided(buf505, (8192, 3072), (3072, 1)); del buf505  # reuse
        extern_kernels.mm(buf558, permute_439, out=buf559)
        del permute_439
        buf560 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf558, (768, 8192), (1, 768)), view_55, out=buf560)
        del view_55
        buf561 = as_strided(buf552, (1, 768, 64), (49152, 1, 768)); del buf552  # reuse
        triton__4.run(buf558, buf561, 49152, 128, grid=grid(49152), stream=stream0)
        buf562 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf561, buf562, 768, 64, grid=grid(768), stream=stream0)
        buf563 = as_strided(buf559, (16, 512, 3072), (1572864, 3072, 1)); del buf559  # reuse
        buf564 = as_strided(buf563, (8192, 3072), (3072, 1)); del buf563  # reuse
        triton__7.run(buf564, addmm_16, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_16
        buf565 = as_strided(buf558, (8192, 768), (768, 1)); del buf558  # reuse
        extern_kernels.mm(buf564, permute_443, out=buf565)
        del permute_443
        buf566 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf564, (3072, 8192), (1, 3072)), view_53, out=buf566)
        del view_53
        buf567 = buf508; del buf508  # reuse
        triton__8.run(buf564, buf567, 98304, 256, grid=grid(98304), stream=stream0)
        buf568 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf567, buf568, 3072, 32, grid=grid(3072), stream=stream0)
        buf575 = as_strided(buf543, (16, 512, 768), (393216, 768, 1)); del buf543  # reuse
        buf578 = as_strided(buf538, (16, 512, 768), (393216, 768, 1)); del buf538  # reuse
        buf579 = as_strided(buf578, (8192, 768), (768, 1)); del buf578  # reuse
        triton__42.run(buf579, buf554, buf565, primals_46, add_21, getitem_11, rsqrt_5, philox_seed_like, buf575, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_46
        buf571 = as_strided(buf561, (768, 64), (1, 768)); del buf561  # reuse
        buf573 = buf550; del buf550  # reuse
        triton__11.run(buf554, buf565, add_21, getitem_11, rsqrt_5, buf571, buf573, 49152, 128, grid=grid(49152), stream=stream0)
        del add_21
        del getitem_11
        del rsqrt_5
        buf576 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf571, buf576, 768, 64, grid=grid(768), stream=stream0)
        buf577 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf573, buf577, 768, 64, grid=grid(768), stream=stream0)
        buf580 = buf565; del buf565  # reuse
        extern_kernels.mm(buf579, permute_447, out=buf580)
        del permute_447
        buf581 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf579, (768, 8192), (1, 768)), view_51, out=buf581)
        del view_51
        buf582 = as_strided(buf573, (1, 768, 64), (49152, 1, 768)); del buf573  # reuse
        triton__4.run(buf579, buf582, 49152, 128, grid=grid(49152), stream=stream0)
        buf583 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf582, buf583, 768, 64, grid=grid(768), stream=stream0)
        buf584 = as_strided(buf579, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf579  # reuse
        triton__12.run(buf580, buf584, 6291456, grid=grid(6291456), stream=stream0)
        buf585 = as_strided(buf580, (192, 512, 64), (32768, 64, 1)); del buf580  # reuse
        extern_kernels.bmm(permute_452, as_strided(buf584, (192, 512, 64), (32768, 64, 1)), out=buf585)
        del permute_452
        buf586 = as_strided(buf529, (192, 512, 512), (262144, 512, 1)); del buf529  # reuse
        extern_kernels.bmm(as_strided(buf584, (192, 512, 64), (32768, 64, 1)), permute_453, out=buf586)
        del permute_453
        buf588 = as_strided(buf527, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf527  # reuse
        triton__43.run(philox_seed_like, buf586, convert_element_type_27, buf588, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_27
        buf589 = as_strided(buf584, (192, 64, 512), (32768, 512, 1)); del buf584  # reuse
        extern_kernels.bmm(permute_454, as_strided(buf588, (192, 512, 512), (262144, 512, 1)), out=buf589)
        del permute_454
        buf590 = as_strided(buf554, (192, 512, 64), (32768, 64, 1)); del buf554  # reuse
        extern_kernels.bmm(as_strided(buf588, (192, 512, 512), (262144, 512, 1)), permute_455, out=buf590)
        del permute_455
        buf591 = as_strided(buf533, (8192, 768), (768, 1)); del buf533  # reuse
        triton__14.run(buf585, buf591, 6291456, grid=grid(6291456), stream=stream0)
        buf592 = as_strided(buf585, (8192, 768), (768, 1)); del buf585  # reuse
        extern_kernels.mm(buf591, permute_459, out=buf592)
        del permute_459
        buf593 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf591, (768, 8192), (1, 768)), view_38, out=buf593)
        buf594 = buf582; del buf582  # reuse
        triton__4.run(buf591, buf594, 49152, 128, grid=grid(49152), stream=stream0)
        buf595 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf594, buf595, 768, 64, grid=grid(768), stream=stream0)
        buf596 = buf591; del buf591  # reuse
        triton__15.run(buf589, buf596, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf597 = as_strided(buf589, (8192, 768), (768, 1)); del buf589  # reuse
        extern_kernels.mm(buf596, permute_464, out=buf597)
        del permute_464
        buf598 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf596, (768, 8192), (1, 768)), view_38, out=buf598)
        buf599 = buf594; del buf594  # reuse
        triton__4.run(buf596, buf599, 49152, 128, grid=grid(49152), stream=stream0)
        buf600 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf599, buf600, 768, 64, grid=grid(768), stream=stream0)
        buf601 = buf596; del buf596  # reuse
        triton__14.run(buf590, buf601, 6291456, grid=grid(6291456), stream=stream0)
        buf602 = as_strided(buf590, (8192, 768), (768, 1)); del buf590  # reuse
        extern_kernels.mm(buf601, permute_468, out=buf602)
        del permute_468
        buf603 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf601, (768, 8192), (1, 768)), view_38, out=buf603)
        del view_38
        buf604 = buf599; del buf599  # reuse
        triton__4.run(buf601, buf604, 49152, 128, grid=grid(49152), stream=stream0)
        buf605 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf604, buf605, 768, 64, grid=grid(768), stream=stream0)
        buf606 = buf547; del buf547  # reuse
        buf613 = as_strided(buf601, (16, 512, 768), (393216, 768, 1)); del buf601  # reuse
        buf616 = buf516; del buf516  # reuse
        buf617 = as_strided(buf616, (8192, 768), (768, 1)); del buf616  # reuse
        triton__44.run(buf617, buf575, buf592, buf597, buf602, primals_36, add_17, getitem_9, rsqrt_4, philox_seed_like, buf606, buf613, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_36
        buf609 = as_strided(buf604, (768, 64), (1, 768)); del buf604  # reuse
        buf611 = buf571; del buf571  # reuse
        triton__17.run(buf575, buf592, buf597, buf602, add_17, getitem_9, rsqrt_4, buf609, buf611, 49152, 128, grid=grid(49152), stream=stream0)
        del add_17
        del getitem_9
        del rsqrt_4
        buf614 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf609, buf614, 768, 64, grid=grid(768), stream=stream0)
        buf615 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf611, buf615, 768, 64, grid=grid(768), stream=stream0)
        buf618 = as_strided(buf564, (8192, 3072), (3072, 1)); del buf564  # reuse
        extern_kernels.mm(buf617, permute_472, out=buf618)
        del permute_472
        buf619 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf617, (768, 8192), (1, 768)), view_36, out=buf619)
        del view_36
        buf620 = as_strided(buf611, (1, 768, 64), (49152, 1, 768)); del buf611  # reuse
        triton__4.run(buf617, buf620, 49152, 128, grid=grid(49152), stream=stream0)
        buf621 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf620, buf621, 768, 64, grid=grid(768), stream=stream0)
        buf622 = as_strided(buf618, (16, 512, 3072), (1572864, 3072, 1)); del buf618  # reuse
        buf623 = as_strided(buf622, (8192, 3072), (3072, 1)); del buf622  # reuse
        triton__7.run(buf623, addmm_10, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_10
        buf624 = as_strided(buf617, (8192, 768), (768, 1)); del buf617  # reuse
        extern_kernels.mm(buf623, permute_476, out=buf624)
        del permute_476
        buf625 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf623, (3072, 8192), (1, 3072)), view_34, out=buf625)
        del view_34
        buf626 = buf567; del buf567  # reuse
        triton__8.run(buf623, buf626, 98304, 256, grid=grid(98304), stream=stream0)
        buf627 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf626, buf627, 3072, 32, grid=grid(3072), stream=stream0)
        buf634 = as_strided(buf602, (16, 512, 768), (393216, 768, 1)); del buf602  # reuse
        buf637 = as_strided(buf597, (16, 512, 768), (393216, 768, 1)); del buf597  # reuse
        buf638 = as_strided(buf637, (8192, 768), (768, 1)); del buf637  # reuse
        triton__45.run(buf638, buf613, buf624, primals_30, add_13, getitem_7, rsqrt_3, philox_seed_like, buf634, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_30
        buf630 = as_strided(buf620, (768, 64), (1, 768)); del buf620  # reuse
        buf632 = buf609; del buf609  # reuse
        triton__11.run(buf613, buf624, add_13, getitem_7, rsqrt_3, buf630, buf632, 49152, 128, grid=grid(49152), stream=stream0)
        del add_13
        del getitem_7
        del rsqrt_3
        buf635 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf630, buf635, 768, 64, grid=grid(768), stream=stream0)
        buf636 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf632, buf636, 768, 64, grid=grid(768), stream=stream0)
        buf639 = buf624; del buf624  # reuse
        extern_kernels.mm(buf638, permute_480, out=buf639)
        del permute_480
        buf640 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf638, (768, 8192), (1, 768)), view_32, out=buf640)
        del view_32
        buf641 = as_strided(buf632, (1, 768, 64), (49152, 1, 768)); del buf632  # reuse
        triton__4.run(buf638, buf641, 49152, 128, grid=grid(49152), stream=stream0)
        buf642 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf641, buf642, 768, 64, grid=grid(768), stream=stream0)
        buf643 = as_strided(buf638, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf638  # reuse
        triton__12.run(buf639, buf643, 6291456, grid=grid(6291456), stream=stream0)
        buf644 = as_strided(buf639, (192, 512, 64), (32768, 64, 1)); del buf639  # reuse
        extern_kernels.bmm(permute_485, as_strided(buf643, (192, 512, 64), (32768, 64, 1)), out=buf644)
        del permute_485
        buf645 = as_strided(buf588, (192, 512, 512), (262144, 512, 1)); del buf588  # reuse
        extern_kernels.bmm(as_strided(buf643, (192, 512, 64), (32768, 64, 1)), permute_486, out=buf645)
        del permute_486
        buf647 = as_strided(buf586, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf586  # reuse
        triton__46.run(philox_seed_like, buf645, convert_element_type_16, buf647, 98304, 512, grid=grid(98304), stream=stream0)
        del convert_element_type_16
        buf648 = as_strided(buf643, (192, 64, 512), (32768, 512, 1)); del buf643  # reuse
        extern_kernels.bmm(permute_487, as_strided(buf647, (192, 512, 512), (262144, 512, 1)), out=buf648)
        del permute_487
        buf649 = as_strided(buf613, (192, 512, 64), (32768, 64, 1)); del buf613  # reuse
        extern_kernels.bmm(as_strided(buf647, (192, 512, 512), (262144, 512, 1)), permute_488, out=buf649)
        del permute_488
        buf650 = as_strided(buf592, (8192, 768), (768, 1)); del buf592  # reuse
        triton__14.run(buf644, buf650, 6291456, grid=grid(6291456), stream=stream0)
        buf651 = as_strided(buf644, (8192, 768), (768, 1)); del buf644  # reuse
        extern_kernels.mm(buf650, permute_492, out=buf651)
        del permute_492
        buf652 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf650, (768, 8192), (1, 768)), view_19, out=buf652)
        buf653 = buf641; del buf641  # reuse
        triton__4.run(buf650, buf653, 49152, 128, grid=grid(49152), stream=stream0)
        buf654 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf653, buf654, 768, 64, grid=grid(768), stream=stream0)
        buf655 = buf650; del buf650  # reuse
        triton__15.run(buf648, buf655, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf656 = as_strided(buf648, (8192, 768), (768, 1)); del buf648  # reuse
        extern_kernels.mm(buf655, permute_497, out=buf656)
        del permute_497
        buf657 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf655, (768, 8192), (1, 768)), view_19, out=buf657)
        buf658 = buf653; del buf653  # reuse
        triton__4.run(buf655, buf658, 49152, 128, grid=grid(49152), stream=stream0)
        buf659 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf658, buf659, 768, 64, grid=grid(768), stream=stream0)
        buf660 = buf655; del buf655  # reuse
        triton__14.run(buf649, buf660, 6291456, grid=grid(6291456), stream=stream0)
        buf661 = as_strided(buf649, (8192, 768), (768, 1)); del buf649  # reuse
        extern_kernels.mm(buf660, permute_501, out=buf661)
        del permute_501
        buf662 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf660, (768, 8192), (1, 768)), view_19, out=buf662)
        del view_19
        buf663 = buf658; del buf658  # reuse
        triton__4.run(buf660, buf663, 49152, 128, grid=grid(49152), stream=stream0)
        buf664 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf663, buf664, 768, 64, grid=grid(768), stream=stream0)
        buf665 = buf606; del buf606  # reuse
        buf672 = as_strided(buf660, (16, 512, 768), (393216, 768, 1)); del buf660  # reuse
        buf675 = buf575; del buf575  # reuse
        buf676 = as_strided(buf675, (8192, 768), (768, 1)); del buf675  # reuse
        triton__47.run(buf676, buf634, buf651, buf656, buf661, primals_20, add_9, getitem_5, rsqrt_2, philox_seed_like, buf665, buf672, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_20
        buf668 = as_strided(buf663, (768, 64), (1, 768)); del buf663  # reuse
        buf670 = buf630; del buf630  # reuse
        triton__17.run(buf634, buf651, buf656, buf661, add_9, getitem_5, rsqrt_2, buf668, buf670, 49152, 128, grid=grid(49152), stream=stream0)
        del add_9
        del buf634
        del getitem_5
        del rsqrt_2
        buf673 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf668, buf673, 768, 64, grid=grid(768), stream=stream0)
        buf674 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf670, buf674, 768, 64, grid=grid(768), stream=stream0)
        buf677 = as_strided(buf623, (8192, 3072), (3072, 1)); del buf623  # reuse
        extern_kernels.mm(buf676, permute_505, out=buf677)
        del permute_505
        buf678 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf676, (768, 8192), (1, 768)), view_17, out=buf678)
        del view_17
        buf679 = as_strided(buf670, (1, 768, 64), (49152, 1, 768)); del buf670  # reuse
        triton__4.run(buf676, buf679, 49152, 128, grid=grid(49152), stream=stream0)
        buf680 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf679, buf680, 768, 64, grid=grid(768), stream=stream0)
        buf681 = as_strided(buf677, (16, 512, 3072), (1572864, 3072, 1)); del buf677  # reuse
        buf682 = as_strided(buf681, (8192, 3072), (3072, 1)); del buf681  # reuse
        triton__7.run(buf682, addmm_4, 25165824, grid=grid(25165824), stream=stream0)
        del addmm_4
        buf683 = as_strided(buf676, (8192, 768), (768, 1)); del buf676  # reuse
        extern_kernels.mm(buf682, permute_509, out=buf683)
        del permute_509
        buf684 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf682, (3072, 8192), (1, 3072)), view_15, out=buf684)
        del view_15
        buf685 = buf626; del buf626  # reuse
        triton__8.run(buf682, buf685, 98304, 256, grid=grid(98304), stream=stream0)
        del buf682
        buf686 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__9.run(buf685, buf686, 3072, 32, grid=grid(3072), stream=stream0)
        del buf685
        buf693 = as_strided(buf661, (16, 512, 768), (393216, 768, 1)); del buf661  # reuse
        buf696 = as_strided(buf656, (16, 512, 768), (393216, 768, 1)); del buf656  # reuse
        buf697 = as_strided(buf696, (8192, 768), (768, 1)); del buf696  # reuse
        triton__48.run(buf697, buf672, buf683, primals_14, add_5, getitem_3, rsqrt_1, philox_seed_like, buf693, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_14
        buf689 = as_strided(buf679, (768, 64), (1, 768)); del buf679  # reuse
        buf691 = buf668; del buf668  # reuse
        triton__11.run(buf672, buf683, add_5, getitem_3, rsqrt_1, buf689, buf691, 49152, 128, grid=grid(49152), stream=stream0)
        del add_5
        del getitem_3
        del rsqrt_1
        buf694 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf689, buf694, 768, 64, grid=grid(768), stream=stream0)
        buf695 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf691, buf695, 768, 64, grid=grid(768), stream=stream0)
        buf698 = buf683; del buf683  # reuse
        extern_kernels.mm(buf697, permute_513, out=buf698)
        del permute_513
        buf699 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf697, (768, 8192), (1, 768)), view_13, out=buf699)
        del view_13
        buf700 = as_strided(buf691, (1, 768, 64), (49152, 1, 768)); del buf691  # reuse
        triton__4.run(buf697, buf700, 49152, 128, grid=grid(49152), stream=stream0)
        buf701 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf700, buf701, 768, 64, grid=grid(768), stream=stream0)
        buf702 = as_strided(buf697, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf697  # reuse
        triton__12.run(buf698, buf702, 6291456, grid=grid(6291456), stream=stream0)
        buf703 = as_strided(buf698, (192, 512, 64), (32768, 64, 1)); del buf698  # reuse
        extern_kernels.bmm(permute_518, as_strided(buf702, (192, 512, 64), (32768, 64, 1)), out=buf703)
        del permute_518
        buf704 = as_strided(buf647, (192, 512, 512), (262144, 512, 1)); del buf647  # reuse
        extern_kernels.bmm(as_strided(buf702, (192, 512, 64), (32768, 64, 1)), permute_519, out=buf704)
        del permute_519
        buf706 = as_strided(buf645, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf645  # reuse
        triton__49.run(philox_seed_like, buf704, convert_element_type_5, buf706, 98304, 512, grid=grid(98304), stream=stream0)
        del buf704
        del convert_element_type_5
        buf707 = as_strided(buf702, (192, 64, 512), (32768, 512, 1)); del buf702  # reuse
        extern_kernels.bmm(permute_520, as_strided(buf706, (192, 512, 512), (262144, 512, 1)), out=buf707)
        del permute_520
        buf708 = as_strided(buf672, (192, 512, 64), (32768, 64, 1)); del buf672  # reuse
        extern_kernels.bmm(as_strided(buf706, (192, 512, 512), (262144, 512, 1)), permute_521, out=buf708)
        del buf706
        del permute_521
        buf709 = as_strided(buf651, (8192, 768), (768, 1)); del buf651  # reuse
        triton__14.run(buf703, buf709, 6291456, grid=grid(6291456), stream=stream0)
        buf710 = as_strided(buf703, (8192, 768), (768, 1)); del buf703  # reuse
        extern_kernels.mm(buf709, permute_525, out=buf710)
        del permute_525
        buf711 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf709, (768, 8192), (1, 768)), view, out=buf711)
        buf712 = buf700; del buf700  # reuse
        triton__4.run(buf709, buf712, 49152, 128, grid=grid(49152), stream=stream0)
        buf713 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf712, buf713, 768, 64, grid=grid(768), stream=stream0)
        buf714 = buf709; del buf709  # reuse
        triton__15.run(buf707, buf714, 8192, 768, grid=grid(8192, 768), stream=stream0)
        buf715 = as_strided(buf707, (8192, 768), (768, 1)); del buf707  # reuse
        extern_kernels.mm(buf714, permute_530, out=buf715)
        del permute_530
        buf716 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf714, (768, 8192), (1, 768)), view, out=buf716)
        buf717 = buf712; del buf712  # reuse
        triton__4.run(buf714, buf717, 49152, 128, grid=grid(49152), stream=stream0)
        buf718 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf717, buf718, 768, 64, grid=grid(768), stream=stream0)
        buf719 = buf714; del buf714  # reuse
        triton__14.run(buf708, buf719, 6291456, grid=grid(6291456), stream=stream0)
        buf720 = as_strided(buf708, (8192, 768), (768, 1)); del buf708  # reuse
        extern_kernels.mm(buf719, permute_534, out=buf720)
        del permute_534
        buf721 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(buf719, (768, 8192), (1, 768)), view, out=buf721)
        del view
        buf722 = buf717; del buf717  # reuse
        triton__4.run(buf719, buf722, 49152, 128, grid=grid(49152), stream=stream0)
        buf723 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__5.run(buf722, buf723, 768, 64, grid=grid(768), stream=stream0)
        buf738 = empty_strided((2, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton__50.run(buf738, 1536, grid=grid(1536), stream=stream0)
        buf741 = empty_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton__51.run(buf741, 23440896, grid=grid(23440896), stream=stream0)
        buf724 = buf665; del buf665  # reuse
        buf731 = as_strided(buf719, (16, 512, 768), (393216, 768, 1)); del buf719  # reuse
        triton__52.run(philox_seed_like, buf693, buf710, buf715, buf720, primals_4, add_1, getitem_1, rsqrt, primals_204, convert_element_type_443, buf724, buf731, buf738, buf741, 8192, 768, grid=grid(8192), stream=stream0)
        del buf693
        del buf710
        del buf715
        del buf720
        del convert_element_type_443
        del philox_seed_like
        del primals_204
        del primals_4
        buf727 = as_strided(buf722, (768, 64), (1, 768)); del buf722  # reuse
        buf729 = buf689; del buf689  # reuse
        triton__53.run(buf724, add_1, getitem_1, rsqrt, buf727, buf729, 49152, 128, grid=grid(49152), stream=stream0)
        del add_1
        del buf724
        del getitem_1
        del rsqrt
        buf732 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf727, buf732, 768, 64, grid=grid(768), stream=stream0)
        del buf727
        buf733 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__3.run(buf729, buf733, 768, 64, grid=grid(768), stream=stream0)
        del buf729
        buf735 = empty_strided((512, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton__54.run(buf735, 393216, grid=grid(393216), stream=stream0)
        triton__55.run(buf731, convert_element_type_435, buf735, 393216, 16, grid=grid(393216), stream=stream0)
        del buf731
        del convert_element_type_435
        buf737 = empty_strided((512, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__56.run(buf735, buf737, 393216, grid=grid(393216), stream=stream0)
        del buf735
        buf740 = empty_strided((2, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__57.run(buf738, buf740, 1536, grid=grid(1536), stream=stream0)
        del buf738
        buf743 = empty_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__58.run(buf741, buf743, 23440896, grid=grid(23440896), stream=stream0)
        return (buf743, buf740, buf737, buf732, buf733, as_strided(buf721, (768, 768), (768, 1)), as_strided(buf723, (768, ), (1, )), as_strided(buf716, (768, 768), (768, 1)), as_strided(buf718, (768, ), (1, )), as_strided(buf711, (768, 768), (768, 1)), as_strided(buf713, (768, ), (1, )), as_strided(buf699, (768, 768), (768, 1)), as_strided(buf701, (768, ), (1, )), buf694, buf695, as_strided(buf684, (3072, 768), (768, 1)), as_strided(buf686, (3072, ), (1, )), as_strided(buf678, (768, 3072), (3072, 1)), as_strided(buf680, (768, ), (1, )), buf673, buf674, as_strided(buf662, (768, 768), (768, 1)), as_strided(buf664, (768, ), (1, )), as_strided(buf657, (768, 768), (768, 1)), as_strided(buf659, (768, ), (1, )), as_strided(buf652, (768, 768), (768, 1)), as_strided(buf654, (768, ), (1, )), as_strided(buf640, (768, 768), (768, 1)), as_strided(buf642, (768, ), (1, )), buf635, buf636, as_strided(buf625, (3072, 768), (768, 1)), as_strided(buf627, (3072, ), (1, )), as_strided(buf619, (768, 3072), (3072, 1)), as_strided(buf621, (768, ), (1, )), buf614, buf615, as_strided(buf603, (768, 768), (768, 1)), as_strided(buf605, (768, ), (1, )), as_strided(buf598, (768, 768), (768, 1)), as_strided(buf600, (768, ), (1, )), as_strided(buf593, (768, 768), (768, 1)), as_strided(buf595, (768, ), (1, )), as_strided(buf581, (768, 768), (768, 1)), as_strided(buf583, (768, ), (1, )), buf576, buf577, as_strided(buf566, (3072, 768), (768, 1)), as_strided(buf568, (3072, ), (1, )), as_strided(buf560, (768, 3072), (3072, 1)), as_strided(buf562, (768, ), (1, )), buf555, buf556, as_strided(buf544, (768, 768), (768, 1)), as_strided(buf546, (768, ), (1, )), as_strided(buf539, (768, 768), (768, 1)), as_strided(buf541, (768, ), (1, )), as_strided(buf534, (768, 768), (768, 1)), as_strided(buf536, (768, ), (1, )), as_strided(buf522, (768, 768), (768, 1)), as_strided(buf524, (768, ), (1, )), buf517, buf518, as_strided(buf507, (3072, 768), (768, 1)), as_strided(buf509, (3072, ), (1, )), as_strided(buf501, (768, 3072), (3072, 1)), as_strided(buf503, (768, ), (1, )), buf496, buf497, as_strided(buf485, (768, 768), (768, 1)), as_strided(buf487, (768, ), (1, )), as_strided(buf480, (768, 768), (768, 1)), as_strided(buf482, (768, ), (1, )), as_strided(buf475, (768, 768), (768, 1)), as_strided(buf477, (768, ), (1, )), as_strided(buf463, (768, 768), (768, 1)), as_strided(buf465, (768, ), (1, )), buf458, buf459, as_strided(buf448, (3072, 768), (768, 1)), as_strided(buf450, (3072, ), (1, )), as_strided(buf442, (768, 3072), (3072, 1)), as_strided(buf444, (768, ), (1, )), buf437, buf438, as_strided(buf426, (768, 768), (768, 1)), as_strided(buf428, (768, ), (1, )), as_strided(buf421, (768, 768), (768, 1)), as_strided(buf423, (768, ), (1, )), as_strided(buf416, (768, 768), (768, 1)), as_strided(buf418, (768, ), (1, )), as_strided(buf404, (768, 768), (768, 1)), as_strided(buf406, (768, ), (1, )), buf399, buf400, as_strided(buf389, (3072, 768), (768, 1)), as_strided(buf391, (3072, ), (1, )), as_strided(buf383, (768, 3072), (3072, 1)), as_strided(buf385, (768, ), (1, )), buf378, buf379, as_strided(buf367, (768, 768), (768, 1)), as_strided(buf369, (768, ), (1, )), as_strided(buf362, (768, 768), (768, 1)), as_strided(buf364, (768, ), (1, )), as_strided(buf357, (768, 768), (768, 1)), as_strided(buf359, (768, ), (1, )), as_strided(buf345, (768, 768), (768, 1)), as_strided(buf347, (768, ), (1, )), buf340, buf341, as_strided(buf330, (3072, 768), (768, 1)), as_strided(buf332, (3072, ), (1, )), as_strided(buf324, (768, 3072), (3072, 1)), as_strided(buf326, (768, ), (1, )), buf319, buf320, as_strided(buf308, (768, 768), (768, 1)), as_strided(buf310, (768, ), (1, )), as_strided(buf303, (768, 768), (768, 1)), as_strided(buf305, (768, ), (1, )), as_strided(buf298, (768, 768), (768, 1)), as_strided(buf300, (768, ), (1, )), as_strided(buf286, (768, 768), (768, 1)), as_strided(buf288, (768, ), (1, )), buf281, buf282, as_strided(buf271, (3072, 768), (768, 1)), as_strided(buf273, (3072, ), (1, )), as_strided(buf265, (768, 3072), (3072, 1)), as_strided(buf267, (768, ), (1, )), buf260, buf261, as_strided(buf249, (768, 768), (768, 1)), as_strided(buf251, (768, ), (1, )), as_strided(buf244, (768, 768), (768, 1)), as_strided(buf246, (768, ), (1, )), as_strided(buf239, (768, 768), (768, 1)), as_strided(buf241, (768, ), (1, )), as_strided(buf227, (768, 768), (768, 1)), as_strided(buf229, (768, ), (1, )), buf222, buf223, as_strided(buf212, (3072, 768), (768, 1)), as_strided(buf214, (3072, ), (1, )), as_strided(buf206, (768, 3072), (3072, 1)), as_strided(buf208, (768, ), (1, )), buf201, buf202, as_strided(buf190, (768, 768), (768, 1)), as_strided(buf192, (768, ), (1, )), as_strided(buf185, (768, 768), (768, 1)), as_strided(buf187, (768, ), (1, )), as_strided(buf180, (768, 768), (768, 1)), as_strided(buf182, (768, ), (1, )), as_strided(buf168, (768, 768), (768, 1)), as_strided(buf170, (768, ), (1, )), buf163, buf164, as_strided(buf153, (3072, 768), (768, 1)), as_strided(buf155, (3072, ), (1, )), as_strided(buf147, (768, 3072), (3072, 1)), as_strided(buf149, (768, ), (1, )), buf142, buf143, as_strided(buf131, (768, 768), (768, 1)), as_strided(buf133, (768, ), (1, )), as_strided(buf126, (768, 768), (768, 1)), as_strided(buf128, (768, ), (1, )), as_strided(buf121, (768, 768), (768, 1)), as_strided(buf123, (768, ), (1, )), as_strided(buf109, (768, 768), (768, 1)), as_strided(buf111, (768, ), (1, )), buf104, buf105, as_strided(buf94, (3072, 768), (768, 1)), as_strided(buf96, (3072, ), (1, )), as_strided(buf88, (768, 3072), (3072, 1)), as_strided(buf90, (768, ), (1, )), buf83, buf84, as_strided(buf72, (768, 768), (768, 1)), as_strided(buf74, (768, ), (1, )), as_strided(buf67, (768, 768), (768, 1)), as_strided(buf69, (768, ), (1, )), as_strided(buf62, (768, 768), (768, 1)), as_strided(buf64, (768, ), (1, )), as_strided(buf50, (768, 768), (768, 1)), as_strided(buf52, (768, ), (1, )), buf45, buf46, as_strided(buf35, (3072, 768), (768, 1)), as_strided(buf37, (3072, ), (1, )), as_strided(buf29, (768, 3072), (3072, 1)), as_strided(buf31, (768, ), (1, )), buf24, buf25, as_strided(buf14, (768, 768), (768, 1)), as_strided(buf16, (768, ), (1, )), buf9, buf10, as_strided(buf1, (30522, 768), (768, 1)), as_strided(buf2, (30522, ), (1, )), None, None, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_204 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    add_1 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_1 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    philox_seed_like = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_5 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_13 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_5 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_3 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_4 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_17 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_9 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_5 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_16 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_32 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_13 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_7 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_10 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_36 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_17 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_9 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_4 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_27 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_51 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_21 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_11 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_16 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_55 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_25 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_13 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_38 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_70 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_29 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_15 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_22 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_74 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_33 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_17 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_8 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_76 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_49 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_89 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_37 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_19 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_28 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_93 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_41 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_21 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_60 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_108 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_45 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_23 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_34 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_112 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_49 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_25 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_12 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_114 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_71 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_127 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_53 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_27 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_40 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_131 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_57 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_29 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_14 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_82 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_146 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_61 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_31 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_46 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_150 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_65 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_33 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_93 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_165 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_69 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_35 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_17 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_52 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_169 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_73 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_37 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_171 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_104 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_184 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_77 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_39 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_186 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_58 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_188 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_81 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_41 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_115 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_203 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_85 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_43 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_21 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_205 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_64 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_207 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_89 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_45 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_22 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_209 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_126 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    view_222 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_93 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_47 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_23 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_224 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_70 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    view_226 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    add_97 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_49 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_24 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_228 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    addmm_72 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_137 = rand_strided((16, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    getitem_51 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((16, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_134 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_155 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_156 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_157 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_158 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_167 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_188 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_189 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_190 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_191 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_195 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_200 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_204 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_221 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_222 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_223 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_224 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_233 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_249 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_254 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_255 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_256 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_257 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_261 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_270 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_287 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_288 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_289 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_290 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_299 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_303 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_315 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_320 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_321 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_322 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_323 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_332 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_336 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_348 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_353 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_354 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_355 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_356 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_360 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_365 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_369 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_381 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_386 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_387 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_388 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_389 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_398 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_402 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_414 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_419 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_420 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_421 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_422 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_426 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_431 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_435 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_447 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_452 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_453 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_454 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_455 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_459 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_464 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_468 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_480 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_485 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_486 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_487 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_488 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_492 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_497 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_501 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_513 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_518 = rand_strided((192, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_519 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_520 = rand_strided((192, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float16)
    permute_521 = rand_strided((192, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float16)
    permute_525 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_530 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    permute_534 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_435 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_443 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8192, 30522), (30522, 1), device='cuda:0', dtype=torch.float16)
    print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_204, add_1, getitem_1, rsqrt, philox_seed_like, view, convert_element_type_5, view_13, add_5, getitem_3, rsqrt_1, view_15, addmm_4, view_17, add_9, getitem_5, rsqrt_2, view_19, convert_element_type_16, view_32, add_13, getitem_7, rsqrt_3, view_34, addmm_10, view_36, add_17, getitem_9, rsqrt_4, view_38, convert_element_type_27, view_51, add_21, getitem_11, rsqrt_5, view_53, addmm_16, view_55, add_25, getitem_13, rsqrt_6, view_57, convert_element_type_38, view_70, add_29, getitem_15, rsqrt_7, view_72, addmm_22, view_74, add_33, getitem_17, rsqrt_8, view_76, convert_element_type_49, view_89, add_37, getitem_19, rsqrt_9, view_91, addmm_28, view_93, add_41, getitem_21, rsqrt_10, view_95, convert_element_type_60, view_108, add_45, getitem_23, rsqrt_11, view_110, addmm_34, view_112, add_49, getitem_25, rsqrt_12, view_114, convert_element_type_71, view_127, add_53, getitem_27, rsqrt_13, view_129, addmm_40, view_131, add_57, getitem_29, rsqrt_14, view_133, convert_element_type_82, view_146, add_61, getitem_31, rsqrt_15, view_148, addmm_46, view_150, add_65, getitem_33, rsqrt_16, view_152, convert_element_type_93, view_165, add_69, getitem_35, rsqrt_17, view_167, addmm_52, view_169, add_73, getitem_37, rsqrt_18, view_171, convert_element_type_104, view_184, add_77, getitem_39, rsqrt_19, view_186, addmm_58, view_188, add_81, getitem_41, rsqrt_20, view_190, convert_element_type_115, view_203, add_85, getitem_43, rsqrt_21, view_205, addmm_64, view_207, add_89, getitem_45, rsqrt_22, view_209, convert_element_type_126, view_222, add_93, getitem_47, rsqrt_23, view_224, addmm_70, view_226, add_97, getitem_49, rsqrt_24, view_228, addmm_72, convert_element_type_137, getitem_51, rsqrt_25, view_230, permute_134, permute_138, permute_142, permute_146, permute_150, permute_155, permute_156, permute_157, permute_158, permute_162, permute_167, permute_171, permute_175, permute_179, permute_183, permute_188, permute_189, permute_190, permute_191, permute_195, permute_200, permute_204, permute_208, permute_212, permute_216, permute_221, permute_222, permute_223, permute_224, permute_228, permute_233, permute_237, permute_241, permute_245, permute_249, permute_254, permute_255, permute_256, permute_257, permute_261, permute_266, permute_270, permute_274, permute_278, permute_282, permute_287, permute_288, permute_289, permute_290, permute_294, permute_299, permute_303, permute_307, permute_311, permute_315, permute_320, permute_321, permute_322, permute_323, permute_327, permute_332, permute_336, permute_340, permute_344, permute_348, permute_353, permute_354, permute_355, permute_356, permute_360, permute_365, permute_369, permute_373, permute_377, permute_381, permute_386, permute_387, permute_388, permute_389, permute_393, permute_398, permute_402, permute_406, permute_410, permute_414, permute_419, permute_420, permute_421, permute_422, permute_426, permute_431, permute_435, permute_439, permute_443, permute_447, permute_452, permute_453, permute_454, permute_455, permute_459, permute_464, permute_468, permute_472, permute_476, permute_480, permute_485, permute_486, permute_487, permute_488, permute_492, permute_497, permute_501, permute_505, permute_509, permute_513, permute_518, permute_519, permute_520, permute_521, permute_525, permute_530, permute_534, convert_element_type_435, convert_element_type_443, tangents_1]))
