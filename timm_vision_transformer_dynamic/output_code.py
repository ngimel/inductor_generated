
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

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (75648*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 196
    x1 = (xindex // 196)
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*y2) + (75264*x1)), xmask & ymask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y2), ymask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y2 + (384*x0) + (75648*x1) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp2, xmask & ymask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]})
@triton.jit
def triton__2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 197
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp5 = 384.0
    tmp6 = tmp4 / tmp5
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp6
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp6
        tmp19 = 384.0
        tmp20 = tmp13 / tmp19
        tmp21 = 1e-06
        tmp22 = tmp20 + tmp21
        tmp23 = tl.libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp27 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tl.store(out_ptr1 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp31, rmask & xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 6
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1152*x1) + (226944*x3)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    y2 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (1152*y2) + (226944*x1)), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (y2 + (197*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.125
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        _tmp4 = tl.where(rmask & xmask & (_tmp4 < tmp3), tmp3, _tmp4)
    tmp4 = tl.max(_tmp4, 1)[:, None]
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = 0.125
        tmp7 = tmp5 * tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 - tmp4
        tmp10 = tl.exp(tmp9)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = 0.125
        tmp14 = tmp12 * tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp4
        tmp17 = tl.exp(tmp16)
        tmp18 = tmp17 / tmp11
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (197*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask & xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 6
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (64*x2) + (1152*x1) + (226944*x3)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x2 = (xindex // 384) % 197
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (12608*x1) + (75648*x3)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton__8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 197
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp7 = 384.0
    tmp8 = tmp6 / tmp7
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp8
        tmp16 = tmp15 * tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp20 = tmp18 + tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp8
        tmp25 = 384.0
        tmp26 = tmp17 / tmp25
        tmp27 = 1e-06
        tmp28 = tmp26 + tmp27
        tmp29 = tl.libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr1 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton__9(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = tl.libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]})
@triton.jit
def triton__10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 197
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp9 = 384.0
    tmp10 = tmp8 / tmp9
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp10
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp37 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp40 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp24 = tmp22 + tmp23
        tmp26 = tmp24 + tmp25
        tmp28 = tmp26 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp10
        tmp31 = 384.0
        tmp32 = tmp21 / tmp31
        tmp33 = 1e-06
        tmp34 = tmp32 + tmp33
        tmp35 = tl.libdevice.rsqrt(tmp34)
        tmp36 = tmp30 * tmp35
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp36 * tmp38
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp39 + tmp41
        tmp43 = tmp42.to(tl.float32)
        tl.store(out_ptr1 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton__11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 197
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
        tl.store(in_out_ptr0 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp8, rmask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp11 = 384.0
    tmp12 = tmp10 / tmp11
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp12
        tmp16 = tmp15 * tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp12
        tmp21 = 384.0
        tmp22 = tmp17 / tmp21
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = tl.libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr1 + (r2 + (384*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]})
@triton.jit
def triton__12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        _tmp4 = tl.where(rmask & xmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp5 = 384.0
    tmp6 = tmp4 / tmp5
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp6
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(rmask & xmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp6
        tmp19 = 384.0
        tmp20 = tmp13 / tmp19
        tmp21 = 1e-06
        tmp22 = tmp20 + tmp21
        tmp23 = tl.libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp27 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp31, rmask & xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton__13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp7 = 384.0
    tmp8 = tmp6 / tmp7
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp8
        tmp16 = tmp15 * tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp20 = tmp18 + tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp8
        tmp25 = 384.0
        tmp26 = tmp17 / tmp25
        tmp27 = 1e-06
        tmp28 = tmp26 + tmp27
        tmp29 = tl.libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]})
@triton.jit
def triton__14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp7, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp9 = 384.0
    tmp10 = tmp8 / tmp9
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp10
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(rmask & xmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp37 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp40 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp24 = tmp22 + tmp23
        tmp26 = tmp24 + tmp25
        tmp28 = tmp26 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp10
        tmp31 = 384.0
        tmp32 = tmp21 / tmp31
        tmp33 = 1e-06
        tmp34 = tmp32 + tmp33
        tmp35 = tl.libdevice.rsqrt(tmp34)
        tmp36 = tmp30 * tmp35
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp36 * tmp38
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp39 + tmp41
        tmp43 = tmp42.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp43, rmask & xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton__15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
        tl.store(in_out_ptr0 + (r1 + (384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp8, rmask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp11 = 384.0
    tmp12 = tmp10 / tmp11
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp12
        tmp16 = tmp15 * tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp12
        tmp21 = 384.0
        tmp22 = tmp17 / tmp21
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = tl.libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton__16(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp9, _tmp10)
        tl.store(in_out_ptr0 + (r1 + (384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp8, rmask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp11 = 384.0
    tmp12 = tmp10 / tmp11
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp12
        tmp16 = tmp15 * tmp15
        _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp16, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp12
        tmp21 = 384.0
        tmp22 = tmp17 / tmp21
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = tl.libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (384*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp33, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1 = args
    args.clear()
    arg152_1_size = arg152_1.size()
    s0 = arg152_1_size[0]
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = aten.convolution(arg152_1, arg2_1, None, (16, 16), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf0, (s0, 384, 14, 14), (75264, 196, 14, 1))
        del arg152_1
        del arg2_1
        buf3 = empty_strided((s0, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float16)
        buf1 = as_strided(buf3, (s0, 1, 384), (75648, 384, 1))  # alias
        print('triton__0', 'in_ptr0', 'arg0_1', (arg0_1.sum()/arg0_1.nelement()).item(), arg0_1.amax().item(), arg0_1.amin().item())
        triton__0_xnumel = 384*s0
        stream0 = get_cuda_stream(0)
        triton__0.run(arg0_1, buf1, triton__0_xnumel, grid=grid(triton__0_xnumel), stream=stream0)
        print('triton__0', 'out_ptr0', 'buf1', (buf1.sum()/buf1.nelement()).item(), buf1.amax().item(), buf1.amin().item())
        del arg0_1
        buf2 = as_strided(buf3, (s0, 196, 384), (75648, 384, 1), 384)  # alias
        print('triton__1', 'in_ptr0', 'buf0', (buf0.sum()/buf0.nelement()).item(), buf0.amax().item(), buf0.amin().item())
        print('triton__1', 'in_ptr1', 'arg3_1', (arg3_1.sum()/arg3_1.nelement()).item(), arg3_1.amax().item(), arg3_1.amin().item())
        triton__1_xnumel = 196*s0
        triton__1.run(buf0, arg3_1, buf2, triton__1_xnumel, 384, grid=grid(triton__1_xnumel, 384), stream=stream0)
        print('triton__1', 'out_ptr0', 'buf2', (buf2.sum()/buf2.nelement()).item(), buf2.amax().item(), buf2.amin().item())
        del arg3_1
        del buf0
        buf4 = empty_strided((s0, 197, 1), (197, 1, 197*s0), device='cuda', dtype=torch.float32)
        buf5 = buf4; del buf4  # reuse
        buf7 = empty_strided((s0, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float16)
        print('triton__2', 'in_out_ptr0', 'buf5', (buf5.sum()/buf5.nelement()).item(), buf5.amax().item(), buf5.amin().item())
        print('triton__2', 'in_ptr0', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__2', 'in_ptr1', 'arg1_1', (arg1_1.sum()/arg1_1.nelement()).item(), arg1_1.amax().item(), arg1_1.amin().item())
        print('triton__2', 'in_ptr2', 'arg4_1', (arg4_1.sum()/arg4_1.nelement()).item(), arg4_1.amax().item(), arg4_1.amin().item())
        print('triton__2', 'in_ptr3', 'arg5_1', (arg5_1.sum()/arg5_1.nelement()).item(), arg5_1.amax().item(), arg5_1.amin().item())
        triton__2_xnumel = 197*s0
        triton__2.run(buf5, buf3, arg1_1, arg4_1, arg5_1, buf7, triton__2_xnumel, 384, grid=grid(triton__2_xnumel), stream=stream0)
        print('triton__2', 'in_out_ptr0', 'buf5', (buf5.sum()/buf5.nelement()).item(), buf5.amax().item(), buf5.amin().item())
        print('triton__2', 'out_ptr1', 'buf7', (buf7.sum()/buf7.nelement()).item(), buf7.amax().item(), buf7.amin().item())
        del arg4_1
        del arg5_1
        del buf1
        del buf2
        buf8 = empty_strided((197*s0, 1152), (1152, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(arg7_1, as_strided(buf7, (197*s0, 384), (384, 1)), as_strided(arg6_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf8)
        del arg6_1
        del arg7_1
        buf9 = as_strided(buf7, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf7  # reuse
        print('triton__3', 'in_ptr0', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf8, buf9, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf9', (buf9.sum()/buf9.nelement()).item(), buf9.amax().item(), buf9.amin().item())
        buf10 = empty_strided((s0, 6, 64, 197), (75648, 12608, 197, 1), device='cuda', dtype=torch.float16)
        print('triton__4', 'in_ptr0', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf8, buf10, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf10', (buf10.sum()/buf10.nelement()).item(), buf10.amax().item(), buf10.amin().item())
        buf11 = empty_strided((6*s0, 197, 197), (38809, 197, 1), device='cuda', dtype=torch.float16)
        extern_kernels.bmm(as_strided(buf9, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf10, (6*s0, 64, 197), (12608, 197, 1)), out=buf11)
        buf14 = empty_strided((s0, 6, 197, 197), (232854, 38809, 197, 1), device='cuda', dtype=torch.float16)
        print('triton__5', 'in_ptr0', 'buf11', (buf11.sum()/buf11.nelement()).item(), buf11.amax().item(), buf11.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf11, buf14, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf14', (buf14.sum()/buf14.nelement()).item(), buf14.amax().item(), buf14.amin().item())
        buf15 = buf9; del buf9  # reuse
        print('triton__6', 'in_ptr0', 'buf8', (buf8.sum()/buf8.nelement()).item(), buf8.amax().item(), buf8.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf8, buf15, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf15', (buf15.sum()/buf15.nelement()).item(), buf15.amax().item(), buf15.amin().item())
        buf16 = as_strided(buf10, (6*s0, 197, 64), (12608, 64, 1)); del buf10  # reuse
        extern_kernels.bmm(as_strided(buf14, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf15, (6*s0, 197, 64), (12608, 64, 1)), out=buf16)
        buf17 = as_strided(buf15, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf15  # reuse
        print('triton__7', 'in_ptr0', 'buf16', (buf16.sum()/buf16.nelement()).item(), buf16.amax().item(), buf16.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf16, buf17, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf17', (buf17.sum()/buf17.nelement()).item(), buf17.amax().item(), buf17.amin().item())
        buf18 = as_strided(buf16, (197*s0, 384), (384, 1)); del buf16  # reuse
        extern_kernels.addmm(arg9_1, as_strided(buf17, (197*s0, 384), (384, 1)), as_strided(arg8_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf18)
        del arg8_1
        del arg9_1
        buf19 = buf5; del buf5  # reuse
        buf20 = buf19; del buf19  # reuse
        buf22 = as_strided(buf17, (s0, 197, 384), (75648, 384, 1)); del buf17  # reuse
        print('triton__8', 'in_out_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        print('triton__8', 'in_ptr0', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__8', 'in_ptr1', 'arg1_1', (arg1_1.sum()/arg1_1.nelement()).item(), arg1_1.amax().item(), arg1_1.amin().item())
        print('triton__8', 'in_ptr2', 'buf18', (buf18.sum()/buf18.nelement()).item(), buf18.amax().item(), buf18.amin().item())
        print('triton__8', 'in_ptr3', 'arg10_1', (arg10_1.sum()/arg10_1.nelement()).item(), arg10_1.amax().item(), arg10_1.amin().item())
        print('triton__8', 'in_ptr4', 'arg11_1', (arg11_1.sum()/arg11_1.nelement()).item(), arg11_1.amax().item(), arg11_1.amin().item())
        triton__8_xnumel = 197*s0
        triton__8.run(buf20, buf3, arg1_1, buf18, arg10_1, arg11_1, buf22, triton__8_xnumel, 384, grid=grid(triton__8_xnumel), stream=stream0)
        print('triton__8', 'in_out_ptr0', 'buf20', (buf20.sum()/buf20.nelement()).item(), buf20.amax().item(), buf20.amin().item())
        print('triton__8', 'out_ptr1', 'buf22', (buf22.sum()/buf22.nelement()).item(), buf22.amax().item(), buf22.amin().item())
        del arg10_1
        del arg11_1
        buf23 = empty_strided((197*s0, 1536), (1536, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(arg13_1, as_strided(buf22, (197*s0, 384), (384, 1)), as_strided(arg12_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf23)
        del arg12_1
        del arg13_1
        buf24 = as_strided(buf23, (s0, 197, 1536), (302592, 1536, 1)); del buf23  # reuse
        print('triton__9', 'in_out_ptr0', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf24, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf24', (buf24.sum()/buf24.nelement()).item(), buf24.amax().item(), buf24.amin().item())
        buf25 = as_strided(buf22, (197*s0, 384), (384, 1)); del buf22  # reuse
        extern_kernels.addmm(arg15_1, as_strided(buf24, (197*s0, 1536), (1536, 1)), as_strided(arg14_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf25)
        del arg14_1
        del arg15_1
        buf26 = buf20; del buf20  # reuse
        buf27 = buf26; del buf26  # reuse
        buf29 = empty_strided((s0, 197, 384), (75648, 384, 1), device='cuda', dtype=torch.float16)
        print('triton__10', 'in_out_ptr0', 'buf27', (buf27.sum()/buf27.nelement()).item(), buf27.amax().item(), buf27.amin().item())
        print('triton__10', 'in_ptr0', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__10', 'in_ptr1', 'arg1_1', (arg1_1.sum()/arg1_1.nelement()).item(), arg1_1.amax().item(), arg1_1.amin().item())
        print('triton__10', 'in_ptr2', 'buf18', (buf18.sum()/buf18.nelement()).item(), buf18.amax().item(), buf18.amin().item())
        print('triton__10', 'in_ptr3', 'buf25', (buf25.sum()/buf25.nelement()).item(), buf25.amax().item(), buf25.amin().item())
        print('triton__10', 'in_ptr4', 'arg16_1', (arg16_1.sum()/arg16_1.nelement()).item(), arg16_1.amax().item(), arg16_1.amin().item())
        print('triton__10', 'in_ptr5', 'arg17_1', (arg17_1.sum()/arg17_1.nelement()).item(), arg17_1.amax().item(), arg17_1.amin().item())
        triton__10_xnumel = 197*s0
        triton__10.run(buf27, buf3, arg1_1, buf18, buf25, arg16_1, arg17_1, buf29, triton__10_xnumel, 384, grid=grid(triton__10_xnumel), stream=stream0)
        print('triton__10', 'in_out_ptr0', 'buf27', (buf27.sum()/buf27.nelement()).item(), buf27.amax().item(), buf27.amin().item())
        print('triton__10', 'out_ptr1', 'buf29', (buf29.sum()/buf29.nelement()).item(), buf29.amax().item(), buf29.amin().item())
        del arg16_1
        del arg17_1
        buf30 = buf8; del buf8  # reuse
        extern_kernels.addmm(arg19_1, as_strided(buf29, (197*s0, 384), (384, 1)), as_strided(arg18_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf30)
        del arg18_1
        del arg19_1
        buf31 = as_strided(buf29, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf29  # reuse
        print('triton__3', 'in_ptr0', 'buf30', (buf30.sum()/buf30.nelement()).item(), buf30.amax().item(), buf30.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf30, buf31, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf31', (buf31.sum()/buf31.nelement()).item(), buf31.amax().item(), buf31.amin().item())
        buf32 = empty_strided((s0, 6, 64, 197), (75648, 12608, 197, 1), device='cuda', dtype=torch.float16)
        print('triton__4', 'in_ptr0', 'buf30', (buf30.sum()/buf30.nelement()).item(), buf30.amax().item(), buf30.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf30, buf32, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf32', (buf32.sum()/buf32.nelement()).item(), buf32.amax().item(), buf32.amin().item())
        buf33 = as_strided(buf14, (6*s0, 197, 197), (38809, 197, 1)); del buf14  # reuse
        extern_kernels.bmm(as_strided(buf31, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf32, (6*s0, 64, 197), (12608, 197, 1)), out=buf33)
        buf36 = as_strided(buf11, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf11  # reuse
        print('triton__5', 'in_ptr0', 'buf33', (buf33.sum()/buf33.nelement()).item(), buf33.amax().item(), buf33.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf33, buf36, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf36', (buf36.sum()/buf36.nelement()).item(), buf36.amax().item(), buf36.amin().item())
        buf37 = as_strided(buf32, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf32  # reuse
        print('triton__6', 'in_ptr0', 'buf30', (buf30.sum()/buf30.nelement()).item(), buf30.amax().item(), buf30.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf30, buf37, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf37', (buf37.sum()/buf37.nelement()).item(), buf37.amax().item(), buf37.amin().item())
        buf38 = as_strided(buf31, (6*s0, 197, 64), (12608, 64, 1)); del buf31  # reuse
        extern_kernels.bmm(as_strided(buf36, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf37, (6*s0, 197, 64), (12608, 64, 1)), out=buf38)
        buf39 = as_strided(buf37, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf37  # reuse
        print('triton__7', 'in_ptr0', 'buf38', (buf38.sum()/buf38.nelement()).item(), buf38.amax().item(), buf38.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf38, buf39, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf39', (buf39.sum()/buf39.nelement()).item(), buf39.amax().item(), buf39.amin().item())
        buf40 = as_strided(buf38, (197*s0, 384), (384, 1)); del buf38  # reuse
        extern_kernels.addmm(arg21_1, as_strided(buf39, (197*s0, 384), (384, 1)), as_strided(arg20_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf40)
        del arg20_1
        del arg21_1
        buf41 = as_strided(buf18, (s0, 197, 384), (75648, 384, 1)); del buf18  # reuse
        buf42 = buf27; del buf27  # reuse
        buf43 = buf42; del buf42  # reuse
        buf45 = as_strided(buf39, (s0, 197, 384), (75648, 384, 1)); del buf39  # reuse
        print('triton__11', 'in_out_ptr0', 'buf41', (buf41.sum()/buf41.nelement()).item(), buf41.amax().item(), buf41.amin().item())
        print('triton__11', 'in_out_ptr1', 'buf43', (buf43.sum()/buf43.nelement()).item(), buf43.amax().item(), buf43.amin().item())
        print('triton__11', 'in_ptr0', 'buf3', (buf3.sum()/buf3.nelement()).item(), buf3.amax().item(), buf3.amin().item())
        print('triton__11', 'in_ptr1', 'arg1_1', (arg1_1.sum()/arg1_1.nelement()).item(), arg1_1.amax().item(), arg1_1.amin().item())
        print('triton__11', 'in_ptr2', 'buf25', (buf25.sum()/buf25.nelement()).item(), buf25.amax().item(), buf25.amin().item())
        print('triton__11', 'in_ptr3', 'buf40', (buf40.sum()/buf40.nelement()).item(), buf40.amax().item(), buf40.amin().item())
        print('triton__11', 'in_ptr4', 'arg22_1', (arg22_1.sum()/arg22_1.nelement()).item(), arg22_1.amax().item(), arg22_1.amin().item())
        print('triton__11', 'in_ptr5', 'arg23_1', (arg23_1.sum()/arg23_1.nelement()).item(), arg23_1.amax().item(), arg23_1.amin().item())
        triton__11_xnumel = 197*s0
        triton__11.run(buf41, buf43, buf3, arg1_1, buf25, buf40, arg22_1, arg23_1, buf45, triton__11_xnumel, 384, grid=grid(triton__11_xnumel), stream=stream0)
        print('triton__11', 'in_out_ptr0', 'buf41', (buf41.sum()/buf41.nelement()).item(), buf41.amax().item(), buf41.amin().item())
        print('triton__11', 'in_out_ptr1', 'buf43', (buf43.sum()/buf43.nelement()).item(), buf43.amax().item(), buf43.amin().item())
        print('triton__11', 'out_ptr1', 'buf45', (buf45.sum()/buf45.nelement()).item(), buf45.amax().item(), buf45.amin().item())
        del arg1_1
        del arg22_1
        del arg23_1
        buf46 = as_strided(buf24, (197*s0, 1536), (1536, 1)); del buf24  # reuse
        extern_kernels.addmm(arg25_1, as_strided(buf45, (197*s0, 384), (384, 1)), as_strided(arg24_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf46)
        del arg24_1
        del arg25_1
        buf47 = as_strided(buf46, (s0, 197, 1536), (302592, 1536, 1)); del buf46  # reuse
        print('triton__9', 'in_out_ptr0', 'buf47', (buf47.sum()/buf47.nelement()).item(), buf47.amax().item(), buf47.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf47, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf47', (buf47.sum()/buf47.nelement()).item(), buf47.amax().item(), buf47.amin().item())
        buf48 = as_strided(buf45, (197*s0, 384), (384, 1)); del buf45  # reuse
        extern_kernels.addmm(arg27_1, as_strided(buf47, (197*s0, 1536), (1536, 1)), as_strided(arg26_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf48)
        del arg26_1
        del arg27_1
        buf49 = buf43; del buf43  # reuse
        buf50 = buf49; del buf49  # reuse
        buf52 = as_strided(buf40, (s0, 197, 384), (75648, 384, 1)); del buf40  # reuse
        print('triton__12', 'in_out_ptr0', 'buf50', (buf50.sum()/buf50.nelement()).item(), buf50.amax().item(), buf50.amin().item())
        print('triton__12', 'in_ptr0', 'buf41', (buf41.sum()/buf41.nelement()).item(), buf41.amax().item(), buf41.amin().item())
        print('triton__12', 'in_ptr1', 'buf48', (buf48.sum()/buf48.nelement()).item(), buf48.amax().item(), buf48.amin().item())
        print('triton__12', 'in_ptr2', 'arg28_1', (arg28_1.sum()/arg28_1.nelement()).item(), arg28_1.amax().item(), arg28_1.amin().item())
        print('triton__12', 'in_ptr3', 'arg29_1', (arg29_1.sum()/arg29_1.nelement()).item(), arg29_1.amax().item(), arg29_1.amin().item())
        triton__12_xnumel = 197*s0
        triton__12.run(buf50, buf41, buf48, arg28_1, arg29_1, buf52, triton__12_xnumel, 384, grid=grid(triton__12_xnumel), stream=stream0)
        print('triton__12', 'in_out_ptr0', 'buf50', (buf50.sum()/buf50.nelement()).item(), buf50.amax().item(), buf50.amin().item())
        print('triton__12', 'out_ptr1', 'buf52', (buf52.sum()/buf52.nelement()).item(), buf52.amax().item(), buf52.amin().item())
        del arg28_1
        del arg29_1
        buf53 = buf30; del buf30  # reuse
        extern_kernels.addmm(arg31_1, as_strided(buf52, (197*s0, 384), (384, 1)), as_strided(arg30_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf53)
        del arg30_1
        del arg31_1
        buf54 = as_strided(buf52, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf52  # reuse
        print('triton__3', 'in_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf53, buf54, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf54', (buf54.sum()/buf54.nelement()).item(), buf54.amax().item(), buf54.amin().item())
        buf55 = as_strided(buf3, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf3  # reuse
        print('triton__4', 'in_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf53, buf55, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf55', (buf55.sum()/buf55.nelement()).item(), buf55.amax().item(), buf55.amin().item())
        buf56 = as_strided(buf36, (6*s0, 197, 197), (38809, 197, 1)); del buf36  # reuse
        extern_kernels.bmm(as_strided(buf54, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf55, (6*s0, 64, 197), (12608, 197, 1)), out=buf56)
        buf59 = as_strided(buf33, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf33  # reuse
        print('triton__5', 'in_ptr0', 'buf56', (buf56.sum()/buf56.nelement()).item(), buf56.amax().item(), buf56.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf56, buf59, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf59', (buf59.sum()/buf59.nelement()).item(), buf59.amax().item(), buf59.amin().item())
        buf60 = as_strided(buf55, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf55  # reuse
        print('triton__6', 'in_ptr0', 'buf53', (buf53.sum()/buf53.nelement()).item(), buf53.amax().item(), buf53.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf53, buf60, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf60', (buf60.sum()/buf60.nelement()).item(), buf60.amax().item(), buf60.amin().item())
        buf61 = as_strided(buf54, (6*s0, 197, 64), (12608, 64, 1)); del buf54  # reuse
        extern_kernels.bmm(as_strided(buf59, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf60, (6*s0, 197, 64), (12608, 64, 1)), out=buf61)
        buf62 = as_strided(buf60, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf60  # reuse
        print('triton__7', 'in_ptr0', 'buf61', (buf61.sum()/buf61.nelement()).item(), buf61.amax().item(), buf61.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf61, buf62, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf62', (buf62.sum()/buf62.nelement()).item(), buf62.amax().item(), buf62.amin().item())
        buf63 = as_strided(buf61, (197*s0, 384), (384, 1)); del buf61  # reuse
        extern_kernels.addmm(arg33_1, as_strided(buf62, (197*s0, 384), (384, 1)), as_strided(arg32_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf63)
        del arg32_1
        del arg33_1
        buf64 = buf50; del buf50  # reuse
        buf65 = buf64; del buf64  # reuse
        buf67 = as_strided(buf62, (s0, 197, 384), (75648, 384, 1)); del buf62  # reuse
        print('triton__13', 'in_out_ptr0', 'buf65', (buf65.sum()/buf65.nelement()).item(), buf65.amax().item(), buf65.amin().item())
        print('triton__13', 'in_ptr0', 'buf41', (buf41.sum()/buf41.nelement()).item(), buf41.amax().item(), buf41.amin().item())
        print('triton__13', 'in_ptr1', 'buf48', (buf48.sum()/buf48.nelement()).item(), buf48.amax().item(), buf48.amin().item())
        print('triton__13', 'in_ptr2', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        print('triton__13', 'in_ptr3', 'arg34_1', (arg34_1.sum()/arg34_1.nelement()).item(), arg34_1.amax().item(), arg34_1.amin().item())
        print('triton__13', 'in_ptr4', 'arg35_1', (arg35_1.sum()/arg35_1.nelement()).item(), arg35_1.amax().item(), arg35_1.amin().item())
        triton__13_xnumel = 197*s0
        triton__13.run(buf65, buf41, buf48, buf63, arg34_1, arg35_1, buf67, triton__13_xnumel, 384, grid=grid(triton__13_xnumel), stream=stream0)
        print('triton__13', 'in_out_ptr0', 'buf65', (buf65.sum()/buf65.nelement()).item(), buf65.amax().item(), buf65.amin().item())
        print('triton__13', 'out_ptr1', 'buf67', (buf67.sum()/buf67.nelement()).item(), buf67.amax().item(), buf67.amin().item())
        del arg34_1
        del arg35_1
        buf68 = as_strided(buf47, (197*s0, 1536), (1536, 1)); del buf47  # reuse
        extern_kernels.addmm(arg37_1, as_strided(buf67, (197*s0, 384), (384, 1)), as_strided(arg36_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf68)
        del arg36_1
        del arg37_1
        buf69 = as_strided(buf68, (s0, 197, 1536), (302592, 1536, 1)); del buf68  # reuse
        print('triton__9', 'in_out_ptr0', 'buf69', (buf69.sum()/buf69.nelement()).item(), buf69.amax().item(), buf69.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf69, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf69', (buf69.sum()/buf69.nelement()).item(), buf69.amax().item(), buf69.amin().item())
        buf70 = as_strided(buf67, (197*s0, 384), (384, 1)); del buf67  # reuse
        extern_kernels.addmm(arg39_1, as_strided(buf69, (197*s0, 1536), (1536, 1)), as_strided(arg38_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf70)
        del arg38_1
        del arg39_1
        buf71 = buf65; del buf65  # reuse
        buf72 = buf71; del buf71  # reuse
        buf74 = as_strided(buf25, (s0, 197, 384), (75648, 384, 1)); del buf25  # reuse
        print('triton__14', 'in_out_ptr0', 'buf72', (buf72.sum()/buf72.nelement()).item(), buf72.amax().item(), buf72.amin().item())
        print('triton__14', 'in_ptr0', 'buf41', (buf41.sum()/buf41.nelement()).item(), buf41.amax().item(), buf41.amin().item())
        print('triton__14', 'in_ptr1', 'buf48', (buf48.sum()/buf48.nelement()).item(), buf48.amax().item(), buf48.amin().item())
        print('triton__14', 'in_ptr2', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        print('triton__14', 'in_ptr3', 'buf70', (buf70.sum()/buf70.nelement()).item(), buf70.amax().item(), buf70.amin().item())
        print('triton__14', 'in_ptr4', 'arg40_1', (arg40_1.sum()/arg40_1.nelement()).item(), arg40_1.amax().item(), arg40_1.amin().item())
        print('triton__14', 'in_ptr5', 'arg41_1', (arg41_1.sum()/arg41_1.nelement()).item(), arg41_1.amax().item(), arg41_1.amin().item())
        triton__14_xnumel = 197*s0
        triton__14.run(buf72, buf41, buf48, buf63, buf70, arg40_1, arg41_1, buf74, triton__14_xnumel, 384, grid=grid(triton__14_xnumel), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf72', (buf72.sum()/buf72.nelement()).item(), buf72.amax().item(), buf72.amin().item())
        print('triton__14', 'out_ptr1', 'buf74', (buf74.sum()/buf74.nelement()).item(), buf74.amax().item(), buf74.amin().item())
        del arg40_1
        del arg41_1
        buf75 = buf53; del buf53  # reuse
        extern_kernels.addmm(arg43_1, as_strided(buf74, (197*s0, 384), (384, 1)), as_strided(arg42_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf75)
        del arg42_1
        del arg43_1
        buf76 = as_strided(buf74, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf74  # reuse
        print('triton__3', 'in_ptr0', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf75, buf76, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf76', (buf76.sum()/buf76.nelement()).item(), buf76.amax().item(), buf76.amin().item())
        buf77 = empty_strided((s0, 6, 64, 197), (75648, 12608, 197, 1), device='cuda', dtype=torch.float16)
        print('triton__4', 'in_ptr0', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf75, buf77, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf77', (buf77.sum()/buf77.nelement()).item(), buf77.amax().item(), buf77.amin().item())
        buf78 = as_strided(buf59, (6*s0, 197, 197), (38809, 197, 1)); del buf59  # reuse
        extern_kernels.bmm(as_strided(buf76, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf77, (6*s0, 64, 197), (12608, 197, 1)), out=buf78)
        buf81 = as_strided(buf56, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf56  # reuse
        print('triton__5', 'in_ptr0', 'buf78', (buf78.sum()/buf78.nelement()).item(), buf78.amax().item(), buf78.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf78, buf81, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf81', (buf81.sum()/buf81.nelement()).item(), buf81.amax().item(), buf81.amin().item())
        buf82 = as_strided(buf77, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf77  # reuse
        print('triton__6', 'in_ptr0', 'buf75', (buf75.sum()/buf75.nelement()).item(), buf75.amax().item(), buf75.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf75, buf82, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf82', (buf82.sum()/buf82.nelement()).item(), buf82.amax().item(), buf82.amin().item())
        buf83 = as_strided(buf76, (6*s0, 197, 64), (12608, 64, 1)); del buf76  # reuse
        extern_kernels.bmm(as_strided(buf81, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf82, (6*s0, 197, 64), (12608, 64, 1)), out=buf83)
        buf84 = as_strided(buf82, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf82  # reuse
        print('triton__7', 'in_ptr0', 'buf83', (buf83.sum()/buf83.nelement()).item(), buf83.amax().item(), buf83.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf83, buf84, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf84', (buf84.sum()/buf84.nelement()).item(), buf84.amax().item(), buf84.amin().item())
        buf85 = as_strided(buf83, (197*s0, 384), (384, 1)); del buf83  # reuse
        extern_kernels.addmm(arg45_1, as_strided(buf84, (197*s0, 384), (384, 1)), as_strided(arg44_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf85)
        del arg44_1
        del arg45_1
        buf86 = buf41; del buf41  # reuse
        buf87 = buf72; del buf72  # reuse
        buf88 = buf87; del buf87  # reuse
        buf90 = as_strided(buf84, (s0, 197, 384), (75648, 384, 1)); del buf84  # reuse
        print('triton__15', 'in_out_ptr0', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf88', (buf88.sum()/buf88.nelement()).item(), buf88.amax().item(), buf88.amin().item())
        print('triton__15', 'in_ptr0', 'buf48', (buf48.sum()/buf48.nelement()).item(), buf48.amax().item(), buf48.amin().item())
        print('triton__15', 'in_ptr1', 'buf63', (buf63.sum()/buf63.nelement()).item(), buf63.amax().item(), buf63.amin().item())
        print('triton__15', 'in_ptr2', 'buf70', (buf70.sum()/buf70.nelement()).item(), buf70.amax().item(), buf70.amin().item())
        print('triton__15', 'in_ptr3', 'buf85', (buf85.sum()/buf85.nelement()).item(), buf85.amax().item(), buf85.amin().item())
        print('triton__15', 'in_ptr4', 'arg46_1', (arg46_1.sum()/arg46_1.nelement()).item(), arg46_1.amax().item(), arg46_1.amin().item())
        print('triton__15', 'in_ptr5', 'arg47_1', (arg47_1.sum()/arg47_1.nelement()).item(), arg47_1.amax().item(), arg47_1.amin().item())
        triton__15_xnumel = 197*s0
        triton__15.run(buf86, buf88, buf48, buf63, buf70, buf85, arg46_1, arg47_1, buf90, triton__15_xnumel, 384, grid=grid(triton__15_xnumel), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf88', (buf88.sum()/buf88.nelement()).item(), buf88.amax().item(), buf88.amin().item())
        print('triton__15', 'out_ptr1', 'buf90', (buf90.sum()/buf90.nelement()).item(), buf90.amax().item(), buf90.amin().item())
        del arg46_1
        del arg47_1
        buf91 = as_strided(buf69, (197*s0, 1536), (1536, 1)); del buf69  # reuse
        extern_kernels.addmm(arg49_1, as_strided(buf90, (197*s0, 384), (384, 1)), as_strided(arg48_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf91)
        del arg48_1
        del arg49_1
        buf92 = as_strided(buf91, (s0, 197, 1536), (302592, 1536, 1)); del buf91  # reuse
        print('triton__9', 'in_out_ptr0', 'buf92', (buf92.sum()/buf92.nelement()).item(), buf92.amax().item(), buf92.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf92, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf92', (buf92.sum()/buf92.nelement()).item(), buf92.amax().item(), buf92.amin().item())
        buf93 = as_strided(buf90, (197*s0, 384), (384, 1)); del buf90  # reuse
        extern_kernels.addmm(arg51_1, as_strided(buf92, (197*s0, 1536), (1536, 1)), as_strided(arg50_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf93)
        del arg50_1
        del arg51_1
        buf94 = buf88; del buf88  # reuse
        buf95 = buf94; del buf94  # reuse
        buf97 = as_strided(buf85, (s0, 197, 384), (75648, 384, 1)); del buf85  # reuse
        print('triton__12', 'in_out_ptr0', 'buf95', (buf95.sum()/buf95.nelement()).item(), buf95.amax().item(), buf95.amin().item())
        print('triton__12', 'in_ptr0', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        print('triton__12', 'in_ptr1', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__12', 'in_ptr2', 'arg52_1', (arg52_1.sum()/arg52_1.nelement()).item(), arg52_1.amax().item(), arg52_1.amin().item())
        print('triton__12', 'in_ptr3', 'arg53_1', (arg53_1.sum()/arg53_1.nelement()).item(), arg53_1.amax().item(), arg53_1.amin().item())
        triton__12_xnumel = 197*s0
        triton__12.run(buf95, buf86, buf93, arg52_1, arg53_1, buf97, triton__12_xnumel, 384, grid=grid(triton__12_xnumel), stream=stream0)
        print('triton__12', 'in_out_ptr0', 'buf95', (buf95.sum()/buf95.nelement()).item(), buf95.amax().item(), buf95.amin().item())
        print('triton__12', 'out_ptr1', 'buf97', (buf97.sum()/buf97.nelement()).item(), buf97.amax().item(), buf97.amin().item())
        del arg52_1
        del arg53_1
        buf98 = buf75; del buf75  # reuse
        extern_kernels.addmm(arg55_1, as_strided(buf97, (197*s0, 384), (384, 1)), as_strided(arg54_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf98)
        del arg54_1
        del arg55_1
        buf99 = as_strided(buf97, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf97  # reuse
        print('triton__3', 'in_ptr0', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf98, buf99, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf99', (buf99.sum()/buf99.nelement()).item(), buf99.amax().item(), buf99.amin().item())
        buf100 = as_strided(buf70, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf70  # reuse
        print('triton__4', 'in_ptr0', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf98, buf100, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf100', (buf100.sum()/buf100.nelement()).item(), buf100.amax().item(), buf100.amin().item())
        buf101 = as_strided(buf81, (6*s0, 197, 197), (38809, 197, 1)); del buf81  # reuse
        extern_kernels.bmm(as_strided(buf99, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf100, (6*s0, 64, 197), (12608, 197, 1)), out=buf101)
        buf104 = as_strided(buf78, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf78  # reuse
        print('triton__5', 'in_ptr0', 'buf101', (buf101.sum()/buf101.nelement()).item(), buf101.amax().item(), buf101.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf101, buf104, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf104', (buf104.sum()/buf104.nelement()).item(), buf104.amax().item(), buf104.amin().item())
        buf105 = buf99; del buf99  # reuse
        print('triton__6', 'in_ptr0', 'buf98', (buf98.sum()/buf98.nelement()).item(), buf98.amax().item(), buf98.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf98, buf105, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf105', (buf105.sum()/buf105.nelement()).item(), buf105.amax().item(), buf105.amin().item())
        buf106 = as_strided(buf100, (6*s0, 197, 64), (12608, 64, 1)); del buf100  # reuse
        extern_kernels.bmm(as_strided(buf104, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf105, (6*s0, 197, 64), (12608, 64, 1)), out=buf106)
        buf107 = as_strided(buf105, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf105  # reuse
        print('triton__7', 'in_ptr0', 'buf106', (buf106.sum()/buf106.nelement()).item(), buf106.amax().item(), buf106.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf106, buf107, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf107', (buf107.sum()/buf107.nelement()).item(), buf107.amax().item(), buf107.amin().item())
        buf108 = as_strided(buf106, (197*s0, 384), (384, 1)); del buf106  # reuse
        extern_kernels.addmm(arg57_1, as_strided(buf107, (197*s0, 384), (384, 1)), as_strided(arg56_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf108)
        del arg56_1
        del arg57_1
        buf109 = buf95; del buf95  # reuse
        buf110 = buf109; del buf109  # reuse
        buf112 = as_strided(buf107, (s0, 197, 384), (75648, 384, 1)); del buf107  # reuse
        print('triton__13', 'in_out_ptr0', 'buf110', (buf110.sum()/buf110.nelement()).item(), buf110.amax().item(), buf110.amin().item())
        print('triton__13', 'in_ptr0', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        print('triton__13', 'in_ptr1', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__13', 'in_ptr2', 'buf108', (buf108.sum()/buf108.nelement()).item(), buf108.amax().item(), buf108.amin().item())
        print('triton__13', 'in_ptr3', 'arg58_1', (arg58_1.sum()/arg58_1.nelement()).item(), arg58_1.amax().item(), arg58_1.amin().item())
        print('triton__13', 'in_ptr4', 'arg59_1', (arg59_1.sum()/arg59_1.nelement()).item(), arg59_1.amax().item(), arg59_1.amin().item())
        triton__13_xnumel = 197*s0
        triton__13.run(buf110, buf86, buf93, buf108, arg58_1, arg59_1, buf112, triton__13_xnumel, 384, grid=grid(triton__13_xnumel), stream=stream0)
        print('triton__13', 'in_out_ptr0', 'buf110', (buf110.sum()/buf110.nelement()).item(), buf110.amax().item(), buf110.amin().item())
        print('triton__13', 'out_ptr1', 'buf112', (buf112.sum()/buf112.nelement()).item(), buf112.amax().item(), buf112.amin().item())
        del arg58_1
        del arg59_1
        buf113 = as_strided(buf92, (197*s0, 1536), (1536, 1)); del buf92  # reuse
        extern_kernels.addmm(arg61_1, as_strided(buf112, (197*s0, 384), (384, 1)), as_strided(arg60_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf113)
        del arg60_1
        del arg61_1
        buf114 = as_strided(buf113, (s0, 197, 1536), (302592, 1536, 1)); del buf113  # reuse
        print('triton__9', 'in_out_ptr0', 'buf114', (buf114.sum()/buf114.nelement()).item(), buf114.amax().item(), buf114.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf114, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf114', (buf114.sum()/buf114.nelement()).item(), buf114.amax().item(), buf114.amin().item())
        buf115 = as_strided(buf112, (197*s0, 384), (384, 1)); del buf112  # reuse
        extern_kernels.addmm(arg63_1, as_strided(buf114, (197*s0, 1536), (1536, 1)), as_strided(arg62_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf115)
        del arg62_1
        del arg63_1
        buf116 = buf110; del buf110  # reuse
        buf117 = buf116; del buf116  # reuse
        buf119 = as_strided(buf63, (s0, 197, 384), (75648, 384, 1)); del buf63  # reuse
        print('triton__14', 'in_out_ptr0', 'buf117', (buf117.sum()/buf117.nelement()).item(), buf117.amax().item(), buf117.amin().item())
        print('triton__14', 'in_ptr0', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        print('triton__14', 'in_ptr1', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__14', 'in_ptr2', 'buf108', (buf108.sum()/buf108.nelement()).item(), buf108.amax().item(), buf108.amin().item())
        print('triton__14', 'in_ptr3', 'buf115', (buf115.sum()/buf115.nelement()).item(), buf115.amax().item(), buf115.amin().item())
        print('triton__14', 'in_ptr4', 'arg64_1', (arg64_1.sum()/arg64_1.nelement()).item(), arg64_1.amax().item(), arg64_1.amin().item())
        print('triton__14', 'in_ptr5', 'arg65_1', (arg65_1.sum()/arg65_1.nelement()).item(), arg65_1.amax().item(), arg65_1.amin().item())
        triton__14_xnumel = 197*s0
        triton__14.run(buf117, buf86, buf93, buf108, buf115, arg64_1, arg65_1, buf119, triton__14_xnumel, 384, grid=grid(triton__14_xnumel), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf117', (buf117.sum()/buf117.nelement()).item(), buf117.amax().item(), buf117.amin().item())
        print('triton__14', 'out_ptr1', 'buf119', (buf119.sum()/buf119.nelement()).item(), buf119.amax().item(), buf119.amin().item())
        del arg64_1
        del arg65_1
        buf120 = buf98; del buf98  # reuse
        extern_kernels.addmm(arg67_1, as_strided(buf119, (197*s0, 384), (384, 1)), as_strided(arg66_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf120)
        del arg66_1
        del arg67_1
        buf121 = as_strided(buf119, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf119  # reuse
        print('triton__3', 'in_ptr0', 'buf120', (buf120.sum()/buf120.nelement()).item(), buf120.amax().item(), buf120.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf120, buf121, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf121', (buf121.sum()/buf121.nelement()).item(), buf121.amax().item(), buf121.amin().item())
        buf122 = as_strided(buf48, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf48  # reuse
        print('triton__4', 'in_ptr0', 'buf120', (buf120.sum()/buf120.nelement()).item(), buf120.amax().item(), buf120.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf120, buf122, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf122', (buf122.sum()/buf122.nelement()).item(), buf122.amax().item(), buf122.amin().item())
        buf123 = as_strided(buf104, (6*s0, 197, 197), (38809, 197, 1)); del buf104  # reuse
        extern_kernels.bmm(as_strided(buf121, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf122, (6*s0, 64, 197), (12608, 197, 1)), out=buf123)
        buf126 = as_strided(buf101, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf101  # reuse
        print('triton__5', 'in_ptr0', 'buf123', (buf123.sum()/buf123.nelement()).item(), buf123.amax().item(), buf123.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf123, buf126, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf126', (buf126.sum()/buf126.nelement()).item(), buf126.amax().item(), buf126.amin().item())
        buf127 = as_strided(buf122, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf122  # reuse
        print('triton__6', 'in_ptr0', 'buf120', (buf120.sum()/buf120.nelement()).item(), buf120.amax().item(), buf120.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf120, buf127, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf127', (buf127.sum()/buf127.nelement()).item(), buf127.amax().item(), buf127.amin().item())
        buf128 = as_strided(buf121, (6*s0, 197, 64), (12608, 64, 1)); del buf121  # reuse
        extern_kernels.bmm(as_strided(buf126, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf127, (6*s0, 197, 64), (12608, 64, 1)), out=buf128)
        buf129 = as_strided(buf127, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf127  # reuse
        print('triton__7', 'in_ptr0', 'buf128', (buf128.sum()/buf128.nelement()).item(), buf128.amax().item(), buf128.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf128, buf129, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf129', (buf129.sum()/buf129.nelement()).item(), buf129.amax().item(), buf129.amin().item())
        buf130 = as_strided(buf128, (197*s0, 384), (384, 1)); del buf128  # reuse
        extern_kernels.addmm(arg69_1, as_strided(buf129, (197*s0, 384), (384, 1)), as_strided(arg68_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf130)
        del arg68_1
        del arg69_1
        buf131 = as_strided(buf108, (s0, 197, 384), (75648, 384, 1)); del buf108  # reuse
        buf132 = buf117; del buf117  # reuse
        buf133 = buf132; del buf132  # reuse
        buf135 = as_strided(buf129, (s0, 197, 384), (75648, 384, 1)); del buf129  # reuse
        print('triton__16', 'in_out_ptr0', 'buf131', (buf131.sum()/buf131.nelement()).item(), buf131.amax().item(), buf131.amin().item())
        print('triton__16', 'in_out_ptr1', 'buf133', (buf133.sum()/buf133.nelement()).item(), buf133.amax().item(), buf133.amin().item())
        print('triton__16', 'in_ptr0', 'buf86', (buf86.sum()/buf86.nelement()).item(), buf86.amax().item(), buf86.amin().item())
        print('triton__16', 'in_ptr1', 'buf93', (buf93.sum()/buf93.nelement()).item(), buf93.amax().item(), buf93.amin().item())
        print('triton__16', 'in_ptr2', 'buf115', (buf115.sum()/buf115.nelement()).item(), buf115.amax().item(), buf115.amin().item())
        print('triton__16', 'in_ptr3', 'buf130', (buf130.sum()/buf130.nelement()).item(), buf130.amax().item(), buf130.amin().item())
        print('triton__16', 'in_ptr4', 'arg70_1', (arg70_1.sum()/arg70_1.nelement()).item(), arg70_1.amax().item(), arg70_1.amin().item())
        print('triton__16', 'in_ptr5', 'arg71_1', (arg71_1.sum()/arg71_1.nelement()).item(), arg71_1.amax().item(), arg71_1.amin().item())
        triton__16_xnumel = 197*s0
        triton__16.run(buf131, buf133, buf86, buf93, buf115, buf130, arg70_1, arg71_1, buf135, triton__16_xnumel, 384, grid=grid(triton__16_xnumel), stream=stream0)
        print('triton__16', 'in_out_ptr0', 'buf131', (buf131.sum()/buf131.nelement()).item(), buf131.amax().item(), buf131.amin().item())
        print('triton__16', 'in_out_ptr1', 'buf133', (buf133.sum()/buf133.nelement()).item(), buf133.amax().item(), buf133.amin().item())
        print('triton__16', 'out_ptr1', 'buf135', (buf135.sum()/buf135.nelement()).item(), buf135.amax().item(), buf135.amin().item())
        del arg70_1
        del arg71_1
        buf136 = as_strided(buf114, (197*s0, 1536), (1536, 1)); del buf114  # reuse
        extern_kernels.addmm(arg73_1, as_strided(buf135, (197*s0, 384), (384, 1)), as_strided(arg72_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf136)
        del arg72_1
        del arg73_1
        buf137 = as_strided(buf136, (s0, 197, 1536), (302592, 1536, 1)); del buf136  # reuse
        print('triton__9', 'in_out_ptr0', 'buf137', (buf137.sum()/buf137.nelement()).item(), buf137.amax().item(), buf137.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf137, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf137', (buf137.sum()/buf137.nelement()).item(), buf137.amax().item(), buf137.amin().item())
        buf138 = as_strided(buf135, (197*s0, 384), (384, 1)); del buf135  # reuse
        extern_kernels.addmm(arg75_1, as_strided(buf137, (197*s0, 1536), (1536, 1)), as_strided(arg74_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf138)
        del arg74_1
        del arg75_1
        buf139 = buf133; del buf133  # reuse
        buf140 = buf139; del buf139  # reuse
        buf142 = as_strided(buf93, (s0, 197, 384), (75648, 384, 1)); del buf93  # reuse
        print('triton__12', 'in_out_ptr0', 'buf140', (buf140.sum()/buf140.nelement()).item(), buf140.amax().item(), buf140.amin().item())
        print('triton__12', 'in_ptr0', 'buf131', (buf131.sum()/buf131.nelement()).item(), buf131.amax().item(), buf131.amin().item())
        print('triton__12', 'in_ptr1', 'buf138', (buf138.sum()/buf138.nelement()).item(), buf138.amax().item(), buf138.amin().item())
        print('triton__12', 'in_ptr2', 'arg76_1', (arg76_1.sum()/arg76_1.nelement()).item(), arg76_1.amax().item(), arg76_1.amin().item())
        print('triton__12', 'in_ptr3', 'arg77_1', (arg77_1.sum()/arg77_1.nelement()).item(), arg77_1.amax().item(), arg77_1.amin().item())
        triton__12_xnumel = 197*s0
        triton__12.run(buf140, buf131, buf138, arg76_1, arg77_1, buf142, triton__12_xnumel, 384, grid=grid(triton__12_xnumel), stream=stream0)
        print('triton__12', 'in_out_ptr0', 'buf140', (buf140.sum()/buf140.nelement()).item(), buf140.amax().item(), buf140.amin().item())
        print('triton__12', 'out_ptr1', 'buf142', (buf142.sum()/buf142.nelement()).item(), buf142.amax().item(), buf142.amin().item())
        del arg76_1
        del arg77_1
        buf143 = buf120; del buf120  # reuse
        extern_kernels.addmm(arg79_1, as_strided(buf142, (197*s0, 384), (384, 1)), as_strided(arg78_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf143)
        del arg78_1
        del arg79_1
        buf144 = as_strided(buf142, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf142  # reuse
        print('triton__3', 'in_ptr0', 'buf143', (buf143.sum()/buf143.nelement()).item(), buf143.amax().item(), buf143.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf143, buf144, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf144', (buf144.sum()/buf144.nelement()).item(), buf144.amax().item(), buf144.amin().item())
        buf145 = as_strided(buf86, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf86  # reuse
        print('triton__4', 'in_ptr0', 'buf143', (buf143.sum()/buf143.nelement()).item(), buf143.amax().item(), buf143.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf143, buf145, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf145', (buf145.sum()/buf145.nelement()).item(), buf145.amax().item(), buf145.amin().item())
        buf146 = as_strided(buf126, (6*s0, 197, 197), (38809, 197, 1)); del buf126  # reuse
        extern_kernels.bmm(as_strided(buf144, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf145, (6*s0, 64, 197), (12608, 197, 1)), out=buf146)
        buf149 = as_strided(buf123, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf123  # reuse
        print('triton__5', 'in_ptr0', 'buf146', (buf146.sum()/buf146.nelement()).item(), buf146.amax().item(), buf146.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf146, buf149, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf149', (buf149.sum()/buf149.nelement()).item(), buf149.amax().item(), buf149.amin().item())
        buf150 = as_strided(buf145, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf145  # reuse
        print('triton__6', 'in_ptr0', 'buf143', (buf143.sum()/buf143.nelement()).item(), buf143.amax().item(), buf143.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf143, buf150, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf150', (buf150.sum()/buf150.nelement()).item(), buf150.amax().item(), buf150.amin().item())
        buf151 = as_strided(buf144, (6*s0, 197, 64), (12608, 64, 1)); del buf144  # reuse
        extern_kernels.bmm(as_strided(buf149, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf150, (6*s0, 197, 64), (12608, 64, 1)), out=buf151)
        buf152 = as_strided(buf150, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf150  # reuse
        print('triton__7', 'in_ptr0', 'buf151', (buf151.sum()/buf151.nelement()).item(), buf151.amax().item(), buf151.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf151, buf152, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf152', (buf152.sum()/buf152.nelement()).item(), buf152.amax().item(), buf152.amin().item())
        buf153 = as_strided(buf151, (197*s0, 384), (384, 1)); del buf151  # reuse
        extern_kernels.addmm(arg81_1, as_strided(buf152, (197*s0, 384), (384, 1)), as_strided(arg80_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf153)
        del arg80_1
        del arg81_1
        buf154 = buf140; del buf140  # reuse
        buf155 = buf154; del buf154  # reuse
        buf157 = as_strided(buf152, (s0, 197, 384), (75648, 384, 1)); del buf152  # reuse
        print('triton__13', 'in_out_ptr0', 'buf155', (buf155.sum()/buf155.nelement()).item(), buf155.amax().item(), buf155.amin().item())
        print('triton__13', 'in_ptr0', 'buf131', (buf131.sum()/buf131.nelement()).item(), buf131.amax().item(), buf131.amin().item())
        print('triton__13', 'in_ptr1', 'buf138', (buf138.sum()/buf138.nelement()).item(), buf138.amax().item(), buf138.amin().item())
        print('triton__13', 'in_ptr2', 'buf153', (buf153.sum()/buf153.nelement()).item(), buf153.amax().item(), buf153.amin().item())
        print('triton__13', 'in_ptr3', 'arg82_1', (arg82_1.sum()/arg82_1.nelement()).item(), arg82_1.amax().item(), arg82_1.amin().item())
        print('triton__13', 'in_ptr4', 'arg83_1', (arg83_1.sum()/arg83_1.nelement()).item(), arg83_1.amax().item(), arg83_1.amin().item())
        triton__13_xnumel = 197*s0
        triton__13.run(buf155, buf131, buf138, buf153, arg82_1, arg83_1, buf157, triton__13_xnumel, 384, grid=grid(triton__13_xnumel), stream=stream0)
        print('triton__13', 'in_out_ptr0', 'buf155', (buf155.sum()/buf155.nelement()).item(), buf155.amax().item(), buf155.amin().item())
        print('triton__13', 'out_ptr1', 'buf157', (buf157.sum()/buf157.nelement()).item(), buf157.amax().item(), buf157.amin().item())
        del arg82_1
        del arg83_1
        buf158 = as_strided(buf137, (197*s0, 1536), (1536, 1)); del buf137  # reuse
        extern_kernels.addmm(arg85_1, as_strided(buf157, (197*s0, 384), (384, 1)), as_strided(arg84_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf158)
        del arg84_1
        del arg85_1
        buf159 = as_strided(buf158, (s0, 197, 1536), (302592, 1536, 1)); del buf158  # reuse
        print('triton__9', 'in_out_ptr0', 'buf159', (buf159.sum()/buf159.nelement()).item(), buf159.amax().item(), buf159.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf159, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf159', (buf159.sum()/buf159.nelement()).item(), buf159.amax().item(), buf159.amin().item())
        buf160 = as_strided(buf157, (197*s0, 384), (384, 1)); del buf157  # reuse
        extern_kernels.addmm(arg87_1, as_strided(buf159, (197*s0, 1536), (1536, 1)), as_strided(arg86_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf160)
        del arg86_1
        del arg87_1
        buf161 = buf155; del buf155  # reuse
        buf162 = buf161; del buf161  # reuse
        buf164 = as_strided(buf130, (s0, 197, 384), (75648, 384, 1)); del buf130  # reuse
        print('triton__14', 'in_out_ptr0', 'buf162', (buf162.sum()/buf162.nelement()).item(), buf162.amax().item(), buf162.amin().item())
        print('triton__14', 'in_ptr0', 'buf131', (buf131.sum()/buf131.nelement()).item(), buf131.amax().item(), buf131.amin().item())
        print('triton__14', 'in_ptr1', 'buf138', (buf138.sum()/buf138.nelement()).item(), buf138.amax().item(), buf138.amin().item())
        print('triton__14', 'in_ptr2', 'buf153', (buf153.sum()/buf153.nelement()).item(), buf153.amax().item(), buf153.amin().item())
        print('triton__14', 'in_ptr3', 'buf160', (buf160.sum()/buf160.nelement()).item(), buf160.amax().item(), buf160.amin().item())
        print('triton__14', 'in_ptr4', 'arg88_1', (arg88_1.sum()/arg88_1.nelement()).item(), arg88_1.amax().item(), arg88_1.amin().item())
        print('triton__14', 'in_ptr5', 'arg89_1', (arg89_1.sum()/arg89_1.nelement()).item(), arg89_1.amax().item(), arg89_1.amin().item())
        triton__14_xnumel = 197*s0
        triton__14.run(buf162, buf131, buf138, buf153, buf160, arg88_1, arg89_1, buf164, triton__14_xnumel, 384, grid=grid(triton__14_xnumel), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf162', (buf162.sum()/buf162.nelement()).item(), buf162.amax().item(), buf162.amin().item())
        print('triton__14', 'out_ptr1', 'buf164', (buf164.sum()/buf164.nelement()).item(), buf164.amax().item(), buf164.amin().item())
        del arg88_1
        del arg89_1
        buf165 = buf143; del buf143  # reuse
        extern_kernels.addmm(arg91_1, as_strided(buf164, (197*s0, 384), (384, 1)), as_strided(arg90_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf165)
        del arg90_1
        del arg91_1
        buf166 = as_strided(buf164, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf164  # reuse
        print('triton__3', 'in_ptr0', 'buf165', (buf165.sum()/buf165.nelement()).item(), buf165.amax().item(), buf165.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf165, buf166, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf166', (buf166.sum()/buf166.nelement()).item(), buf166.amax().item(), buf166.amin().item())
        buf167 = as_strided(buf115, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf115  # reuse
        print('triton__4', 'in_ptr0', 'buf165', (buf165.sum()/buf165.nelement()).item(), buf165.amax().item(), buf165.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf165, buf167, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf167', (buf167.sum()/buf167.nelement()).item(), buf167.amax().item(), buf167.amin().item())
        buf168 = as_strided(buf149, (6*s0, 197, 197), (38809, 197, 1)); del buf149  # reuse
        extern_kernels.bmm(as_strided(buf166, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf167, (6*s0, 64, 197), (12608, 197, 1)), out=buf168)
        buf171 = as_strided(buf146, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf146  # reuse
        print('triton__5', 'in_ptr0', 'buf168', (buf168.sum()/buf168.nelement()).item(), buf168.amax().item(), buf168.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf168, buf171, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf171', (buf171.sum()/buf171.nelement()).item(), buf171.amax().item(), buf171.amin().item())
        buf172 = as_strided(buf167, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf167  # reuse
        print('triton__6', 'in_ptr0', 'buf165', (buf165.sum()/buf165.nelement()).item(), buf165.amax().item(), buf165.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf165, buf172, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf172', (buf172.sum()/buf172.nelement()).item(), buf172.amax().item(), buf172.amin().item())
        buf173 = as_strided(buf166, (6*s0, 197, 64), (12608, 64, 1)); del buf166  # reuse
        extern_kernels.bmm(as_strided(buf171, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf172, (6*s0, 197, 64), (12608, 64, 1)), out=buf173)
        buf174 = as_strided(buf172, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf172  # reuse
        print('triton__7', 'in_ptr0', 'buf173', (buf173.sum()/buf173.nelement()).item(), buf173.amax().item(), buf173.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf173, buf174, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf174', (buf174.sum()/buf174.nelement()).item(), buf174.amax().item(), buf174.amin().item())
        buf175 = as_strided(buf173, (197*s0, 384), (384, 1)); del buf173  # reuse
        extern_kernels.addmm(arg93_1, as_strided(buf174, (197*s0, 384), (384, 1)), as_strided(arg92_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf175)
        del arg92_1
        del arg93_1
        buf176 = buf131; del buf131  # reuse
        buf177 = buf162; del buf162  # reuse
        buf178 = buf177; del buf177  # reuse
        buf180 = as_strided(buf174, (s0, 197, 384), (75648, 384, 1)); del buf174  # reuse
        print('triton__15', 'in_out_ptr0', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf178', (buf178.sum()/buf178.nelement()).item(), buf178.amax().item(), buf178.amin().item())
        print('triton__15', 'in_ptr0', 'buf138', (buf138.sum()/buf138.nelement()).item(), buf138.amax().item(), buf138.amin().item())
        print('triton__15', 'in_ptr1', 'buf153', (buf153.sum()/buf153.nelement()).item(), buf153.amax().item(), buf153.amin().item())
        print('triton__15', 'in_ptr2', 'buf160', (buf160.sum()/buf160.nelement()).item(), buf160.amax().item(), buf160.amin().item())
        print('triton__15', 'in_ptr3', 'buf175', (buf175.sum()/buf175.nelement()).item(), buf175.amax().item(), buf175.amin().item())
        print('triton__15', 'in_ptr4', 'arg94_1', (arg94_1.sum()/arg94_1.nelement()).item(), arg94_1.amax().item(), arg94_1.amin().item())
        print('triton__15', 'in_ptr5', 'arg95_1', (arg95_1.sum()/arg95_1.nelement()).item(), arg95_1.amax().item(), arg95_1.amin().item())
        triton__15_xnumel = 197*s0
        triton__15.run(buf176, buf178, buf138, buf153, buf160, buf175, arg94_1, arg95_1, buf180, triton__15_xnumel, 384, grid=grid(triton__15_xnumel), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf178', (buf178.sum()/buf178.nelement()).item(), buf178.amax().item(), buf178.amin().item())
        print('triton__15', 'out_ptr1', 'buf180', (buf180.sum()/buf180.nelement()).item(), buf180.amax().item(), buf180.amin().item())
        del arg94_1
        del arg95_1
        buf181 = as_strided(buf159, (197*s0, 1536), (1536, 1)); del buf159  # reuse
        extern_kernels.addmm(arg97_1, as_strided(buf180, (197*s0, 384), (384, 1)), as_strided(arg96_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf181)
        del arg96_1
        del arg97_1
        buf182 = as_strided(buf181, (s0, 197, 1536), (302592, 1536, 1)); del buf181  # reuse
        print('triton__9', 'in_out_ptr0', 'buf182', (buf182.sum()/buf182.nelement()).item(), buf182.amax().item(), buf182.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf182, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf182', (buf182.sum()/buf182.nelement()).item(), buf182.amax().item(), buf182.amin().item())
        buf183 = as_strided(buf180, (197*s0, 384), (384, 1)); del buf180  # reuse
        extern_kernels.addmm(arg99_1, as_strided(buf182, (197*s0, 1536), (1536, 1)), as_strided(arg98_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf183)
        del arg98_1
        del arg99_1
        buf184 = buf178; del buf178  # reuse
        buf185 = buf184; del buf184  # reuse
        buf187 = as_strided(buf175, (s0, 197, 384), (75648, 384, 1)); del buf175  # reuse
        print('triton__12', 'in_out_ptr0', 'buf185', (buf185.sum()/buf185.nelement()).item(), buf185.amax().item(), buf185.amin().item())
        print('triton__12', 'in_ptr0', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__12', 'in_ptr1', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        print('triton__12', 'in_ptr2', 'arg100_1', (arg100_1.sum()/arg100_1.nelement()).item(), arg100_1.amax().item(), arg100_1.amin().item())
        print('triton__12', 'in_ptr3', 'arg101_1', (arg101_1.sum()/arg101_1.nelement()).item(), arg101_1.amax().item(), arg101_1.amin().item())
        triton__12_xnumel = 197*s0
        triton__12.run(buf185, buf176, buf183, arg100_1, arg101_1, buf187, triton__12_xnumel, 384, grid=grid(triton__12_xnumel), stream=stream0)
        print('triton__12', 'in_out_ptr0', 'buf185', (buf185.sum()/buf185.nelement()).item(), buf185.amax().item(), buf185.amin().item())
        print('triton__12', 'out_ptr1', 'buf187', (buf187.sum()/buf187.nelement()).item(), buf187.amax().item(), buf187.amin().item())
        del arg100_1
        del arg101_1
        buf188 = buf165; del buf165  # reuse
        extern_kernels.addmm(arg103_1, as_strided(buf187, (197*s0, 384), (384, 1)), as_strided(arg102_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf188)
        del arg102_1
        del arg103_1
        buf189 = as_strided(buf187, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf187  # reuse
        print('triton__3', 'in_ptr0', 'buf188', (buf188.sum()/buf188.nelement()).item(), buf188.amax().item(), buf188.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf188, buf189, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf189', (buf189.sum()/buf189.nelement()).item(), buf189.amax().item(), buf189.amin().item())
        buf190 = as_strided(buf160, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf160  # reuse
        print('triton__4', 'in_ptr0', 'buf188', (buf188.sum()/buf188.nelement()).item(), buf188.amax().item(), buf188.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf188, buf190, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf190', (buf190.sum()/buf190.nelement()).item(), buf190.amax().item(), buf190.amin().item())
        buf191 = as_strided(buf171, (6*s0, 197, 197), (38809, 197, 1)); del buf171  # reuse
        extern_kernels.bmm(as_strided(buf189, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf190, (6*s0, 64, 197), (12608, 197, 1)), out=buf191)
        buf194 = as_strided(buf168, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf168  # reuse
        print('triton__5', 'in_ptr0', 'buf191', (buf191.sum()/buf191.nelement()).item(), buf191.amax().item(), buf191.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf191, buf194, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf194', (buf194.sum()/buf194.nelement()).item(), buf194.amax().item(), buf194.amin().item())
        buf195 = as_strided(buf190, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf190  # reuse
        print('triton__6', 'in_ptr0', 'buf188', (buf188.sum()/buf188.nelement()).item(), buf188.amax().item(), buf188.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf188, buf195, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf195', (buf195.sum()/buf195.nelement()).item(), buf195.amax().item(), buf195.amin().item())
        buf196 = as_strided(buf189, (6*s0, 197, 64), (12608, 64, 1)); del buf189  # reuse
        extern_kernels.bmm(as_strided(buf194, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf195, (6*s0, 197, 64), (12608, 64, 1)), out=buf196)
        buf197 = as_strided(buf195, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf195  # reuse
        print('triton__7', 'in_ptr0', 'buf196', (buf196.sum()/buf196.nelement()).item(), buf196.amax().item(), buf196.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf196, buf197, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf197', (buf197.sum()/buf197.nelement()).item(), buf197.amax().item(), buf197.amin().item())
        buf198 = as_strided(buf196, (197*s0, 384), (384, 1)); del buf196  # reuse
        extern_kernels.addmm(arg105_1, as_strided(buf197, (197*s0, 384), (384, 1)), as_strided(arg104_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf198)
        del arg104_1
        del arg105_1
        buf199 = buf185; del buf185  # reuse
        buf200 = buf199; del buf199  # reuse
        buf202 = as_strided(buf197, (s0, 197, 384), (75648, 384, 1)); del buf197  # reuse
        print('triton__13', 'in_out_ptr0', 'buf200', (buf200.sum()/buf200.nelement()).item(), buf200.amax().item(), buf200.amin().item())
        print('triton__13', 'in_ptr0', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__13', 'in_ptr1', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        print('triton__13', 'in_ptr2', 'buf198', (buf198.sum()/buf198.nelement()).item(), buf198.amax().item(), buf198.amin().item())
        print('triton__13', 'in_ptr3', 'arg106_1', (arg106_1.sum()/arg106_1.nelement()).item(), arg106_1.amax().item(), arg106_1.amin().item())
        print('triton__13', 'in_ptr4', 'arg107_1', (arg107_1.sum()/arg107_1.nelement()).item(), arg107_1.amax().item(), arg107_1.amin().item())
        triton__13_xnumel = 197*s0
        triton__13.run(buf200, buf176, buf183, buf198, arg106_1, arg107_1, buf202, triton__13_xnumel, 384, grid=grid(triton__13_xnumel), stream=stream0)
        print('triton__13', 'in_out_ptr0', 'buf200', (buf200.sum()/buf200.nelement()).item(), buf200.amax().item(), buf200.amin().item())
        print('triton__13', 'out_ptr1', 'buf202', (buf202.sum()/buf202.nelement()).item(), buf202.amax().item(), buf202.amin().item())
        del arg106_1
        del arg107_1
        buf203 = as_strided(buf182, (197*s0, 1536), (1536, 1)); del buf182  # reuse
        extern_kernels.addmm(arg109_1, as_strided(buf202, (197*s0, 384), (384, 1)), as_strided(arg108_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf203)
        del arg108_1
        del arg109_1
        buf204 = as_strided(buf203, (s0, 197, 1536), (302592, 1536, 1)); del buf203  # reuse
        print('triton__9', 'in_out_ptr0', 'buf204', (buf204.sum()/buf204.nelement()).item(), buf204.amax().item(), buf204.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf204, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf204', (buf204.sum()/buf204.nelement()).item(), buf204.amax().item(), buf204.amin().item())
        buf205 = as_strided(buf202, (197*s0, 384), (384, 1)); del buf202  # reuse
        extern_kernels.addmm(arg111_1, as_strided(buf204, (197*s0, 1536), (1536, 1)), as_strided(arg110_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf205)
        del arg110_1
        del arg111_1
        buf206 = buf200; del buf200  # reuse
        buf207 = buf206; del buf206  # reuse
        buf209 = as_strided(buf153, (s0, 197, 384), (75648, 384, 1)); del buf153  # reuse
        print('triton__14', 'in_out_ptr0', 'buf207', (buf207.sum()/buf207.nelement()).item(), buf207.amax().item(), buf207.amin().item())
        print('triton__14', 'in_ptr0', 'buf176', (buf176.sum()/buf176.nelement()).item(), buf176.amax().item(), buf176.amin().item())
        print('triton__14', 'in_ptr1', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        print('triton__14', 'in_ptr2', 'buf198', (buf198.sum()/buf198.nelement()).item(), buf198.amax().item(), buf198.amin().item())
        print('triton__14', 'in_ptr3', 'buf205', (buf205.sum()/buf205.nelement()).item(), buf205.amax().item(), buf205.amin().item())
        print('triton__14', 'in_ptr4', 'arg112_1', (arg112_1.sum()/arg112_1.nelement()).item(), arg112_1.amax().item(), arg112_1.amin().item())
        print('triton__14', 'in_ptr5', 'arg113_1', (arg113_1.sum()/arg113_1.nelement()).item(), arg113_1.amax().item(), arg113_1.amin().item())
        triton__14_xnumel = 197*s0
        triton__14.run(buf207, buf176, buf183, buf198, buf205, arg112_1, arg113_1, buf209, triton__14_xnumel, 384, grid=grid(triton__14_xnumel), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf207', (buf207.sum()/buf207.nelement()).item(), buf207.amax().item(), buf207.amin().item())
        print('triton__14', 'out_ptr1', 'buf209', (buf209.sum()/buf209.nelement()).item(), buf209.amax().item(), buf209.amin().item())
        del arg112_1
        del arg113_1
        buf210 = buf188; del buf188  # reuse
        extern_kernels.addmm(arg115_1, as_strided(buf209, (197*s0, 384), (384, 1)), as_strided(arg114_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf210)
        del arg114_1
        del arg115_1
        buf211 = as_strided(buf209, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf209  # reuse
        print('triton__3', 'in_ptr0', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf210, buf211, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf211', (buf211.sum()/buf211.nelement()).item(), buf211.amax().item(), buf211.amin().item())
        buf212 = as_strided(buf138, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf138  # reuse
        print('triton__4', 'in_ptr0', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf210, buf212, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf212', (buf212.sum()/buf212.nelement()).item(), buf212.amax().item(), buf212.amin().item())
        buf213 = as_strided(buf194, (6*s0, 197, 197), (38809, 197, 1)); del buf194  # reuse
        extern_kernels.bmm(as_strided(buf211, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf212, (6*s0, 64, 197), (12608, 197, 1)), out=buf213)
        buf216 = as_strided(buf191, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf191  # reuse
        print('triton__5', 'in_ptr0', 'buf213', (buf213.sum()/buf213.nelement()).item(), buf213.amax().item(), buf213.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf213, buf216, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf216', (buf216.sum()/buf216.nelement()).item(), buf216.amax().item(), buf216.amin().item())
        buf217 = as_strided(buf212, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf212  # reuse
        print('triton__6', 'in_ptr0', 'buf210', (buf210.sum()/buf210.nelement()).item(), buf210.amax().item(), buf210.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf210, buf217, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf217', (buf217.sum()/buf217.nelement()).item(), buf217.amax().item(), buf217.amin().item())
        buf218 = as_strided(buf211, (6*s0, 197, 64), (12608, 64, 1)); del buf211  # reuse
        extern_kernels.bmm(as_strided(buf216, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf217, (6*s0, 197, 64), (12608, 64, 1)), out=buf218)
        buf219 = as_strided(buf217, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf217  # reuse
        print('triton__7', 'in_ptr0', 'buf218', (buf218.sum()/buf218.nelement()).item(), buf218.amax().item(), buf218.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf218, buf219, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf219', (buf219.sum()/buf219.nelement()).item(), buf219.amax().item(), buf219.amin().item())
        buf220 = as_strided(buf218, (197*s0, 384), (384, 1)); del buf218  # reuse
        extern_kernels.addmm(arg117_1, as_strided(buf219, (197*s0, 384), (384, 1)), as_strided(arg116_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf220)
        del arg116_1
        del arg117_1
        buf221 = buf176; del buf176  # reuse
        buf222 = buf207; del buf207  # reuse
        buf223 = buf222; del buf222  # reuse
        buf225 = as_strided(buf219, (s0, 197, 384), (75648, 384, 1)); del buf219  # reuse
        print('triton__15', 'in_out_ptr0', 'buf221', (buf221.sum()/buf221.nelement()).item(), buf221.amax().item(), buf221.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf223', (buf223.sum()/buf223.nelement()).item(), buf223.amax().item(), buf223.amin().item())
        print('triton__15', 'in_ptr0', 'buf183', (buf183.sum()/buf183.nelement()).item(), buf183.amax().item(), buf183.amin().item())
        print('triton__15', 'in_ptr1', 'buf198', (buf198.sum()/buf198.nelement()).item(), buf198.amax().item(), buf198.amin().item())
        print('triton__15', 'in_ptr2', 'buf205', (buf205.sum()/buf205.nelement()).item(), buf205.amax().item(), buf205.amin().item())
        print('triton__15', 'in_ptr3', 'buf220', (buf220.sum()/buf220.nelement()).item(), buf220.amax().item(), buf220.amin().item())
        print('triton__15', 'in_ptr4', 'arg118_1', (arg118_1.sum()/arg118_1.nelement()).item(), arg118_1.amax().item(), arg118_1.amin().item())
        print('triton__15', 'in_ptr5', 'arg119_1', (arg119_1.sum()/arg119_1.nelement()).item(), arg119_1.amax().item(), arg119_1.amin().item())
        triton__15_xnumel = 197*s0
        triton__15.run(buf221, buf223, buf183, buf198, buf205, buf220, arg118_1, arg119_1, buf225, triton__15_xnumel, 384, grid=grid(triton__15_xnumel), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf221', (buf221.sum()/buf221.nelement()).item(), buf221.amax().item(), buf221.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf223', (buf223.sum()/buf223.nelement()).item(), buf223.amax().item(), buf223.amin().item())
        print('triton__15', 'out_ptr1', 'buf225', (buf225.sum()/buf225.nelement()).item(), buf225.amax().item(), buf225.amin().item())
        del arg118_1
        del arg119_1
        buf226 = as_strided(buf204, (197*s0, 1536), (1536, 1)); del buf204  # reuse
        extern_kernels.addmm(arg121_1, as_strided(buf225, (197*s0, 384), (384, 1)), as_strided(arg120_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf226)
        del arg120_1
        del arg121_1
        buf227 = as_strided(buf226, (s0, 197, 1536), (302592, 1536, 1)); del buf226  # reuse
        print('triton__9', 'in_out_ptr0', 'buf227', (buf227.sum()/buf227.nelement()).item(), buf227.amax().item(), buf227.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf227, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf227', (buf227.sum()/buf227.nelement()).item(), buf227.amax().item(), buf227.amin().item())
        buf228 = as_strided(buf225, (197*s0, 384), (384, 1)); del buf225  # reuse
        extern_kernels.addmm(arg123_1, as_strided(buf227, (197*s0, 1536), (1536, 1)), as_strided(arg122_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf228)
        del arg122_1
        del arg123_1
        buf229 = buf223; del buf223  # reuse
        buf230 = buf229; del buf229  # reuse
        buf232 = as_strided(buf220, (s0, 197, 384), (75648, 384, 1)); del buf220  # reuse
        print('triton__12', 'in_out_ptr0', 'buf230', (buf230.sum()/buf230.nelement()).item(), buf230.amax().item(), buf230.amin().item())
        print('triton__12', 'in_ptr0', 'buf221', (buf221.sum()/buf221.nelement()).item(), buf221.amax().item(), buf221.amin().item())
        print('triton__12', 'in_ptr1', 'buf228', (buf228.sum()/buf228.nelement()).item(), buf228.amax().item(), buf228.amin().item())
        print('triton__12', 'in_ptr2', 'arg124_1', (arg124_1.sum()/arg124_1.nelement()).item(), arg124_1.amax().item(), arg124_1.amin().item())
        print('triton__12', 'in_ptr3', 'arg125_1', (arg125_1.sum()/arg125_1.nelement()).item(), arg125_1.amax().item(), arg125_1.amin().item())
        triton__12_xnumel = 197*s0
        triton__12.run(buf230, buf221, buf228, arg124_1, arg125_1, buf232, triton__12_xnumel, 384, grid=grid(triton__12_xnumel), stream=stream0)
        print('triton__12', 'in_out_ptr0', 'buf230', (buf230.sum()/buf230.nelement()).item(), buf230.amax().item(), buf230.amin().item())
        print('triton__12', 'out_ptr1', 'buf232', (buf232.sum()/buf232.nelement()).item(), buf232.amax().item(), buf232.amin().item())
        del arg124_1
        del arg125_1
        buf233 = buf210; del buf210  # reuse
        extern_kernels.addmm(arg127_1, as_strided(buf232, (197*s0, 384), (384, 1)), as_strided(arg126_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf233)
        del arg126_1
        del arg127_1
        buf234 = as_strided(buf232, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf232  # reuse
        print('triton__3', 'in_ptr0', 'buf233', (buf233.sum()/buf233.nelement()).item(), buf233.amax().item(), buf233.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf233, buf234, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf234', (buf234.sum()/buf234.nelement()).item(), buf234.amax().item(), buf234.amin().item())
        buf235 = as_strided(buf205, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf205  # reuse
        print('triton__4', 'in_ptr0', 'buf233', (buf233.sum()/buf233.nelement()).item(), buf233.amax().item(), buf233.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf233, buf235, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf235', (buf235.sum()/buf235.nelement()).item(), buf235.amax().item(), buf235.amin().item())
        buf236 = as_strided(buf216, (6*s0, 197, 197), (38809, 197, 1)); del buf216  # reuse
        extern_kernels.bmm(as_strided(buf234, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf235, (6*s0, 64, 197), (12608, 197, 1)), out=buf236)
        buf239 = as_strided(buf213, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf213  # reuse
        print('triton__5', 'in_ptr0', 'buf236', (buf236.sum()/buf236.nelement()).item(), buf236.amax().item(), buf236.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf236, buf239, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf239', (buf239.sum()/buf239.nelement()).item(), buf239.amax().item(), buf239.amin().item())
        buf240 = as_strided(buf235, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf235  # reuse
        print('triton__6', 'in_ptr0', 'buf233', (buf233.sum()/buf233.nelement()).item(), buf233.amax().item(), buf233.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf233, buf240, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf240', (buf240.sum()/buf240.nelement()).item(), buf240.amax().item(), buf240.amin().item())
        buf241 = as_strided(buf234, (6*s0, 197, 64), (12608, 64, 1)); del buf234  # reuse
        extern_kernels.bmm(as_strided(buf239, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf240, (6*s0, 197, 64), (12608, 64, 1)), out=buf241)
        buf242 = as_strided(buf240, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf240  # reuse
        print('triton__7', 'in_ptr0', 'buf241', (buf241.sum()/buf241.nelement()).item(), buf241.amax().item(), buf241.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf241, buf242, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf242', (buf242.sum()/buf242.nelement()).item(), buf242.amax().item(), buf242.amin().item())
        buf243 = as_strided(buf241, (197*s0, 384), (384, 1)); del buf241  # reuse
        extern_kernels.addmm(arg129_1, as_strided(buf242, (197*s0, 384), (384, 1)), as_strided(arg128_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf243)
        del arg128_1
        del arg129_1
        buf244 = buf230; del buf230  # reuse
        buf245 = buf244; del buf244  # reuse
        buf247 = as_strided(buf242, (s0, 197, 384), (75648, 384, 1)); del buf242  # reuse
        print('triton__13', 'in_out_ptr0', 'buf245', (buf245.sum()/buf245.nelement()).item(), buf245.amax().item(), buf245.amin().item())
        print('triton__13', 'in_ptr0', 'buf221', (buf221.sum()/buf221.nelement()).item(), buf221.amax().item(), buf221.amin().item())
        print('triton__13', 'in_ptr1', 'buf228', (buf228.sum()/buf228.nelement()).item(), buf228.amax().item(), buf228.amin().item())
        print('triton__13', 'in_ptr2', 'buf243', (buf243.sum()/buf243.nelement()).item(), buf243.amax().item(), buf243.amin().item())
        print('triton__13', 'in_ptr3', 'arg130_1', (arg130_1.sum()/arg130_1.nelement()).item(), arg130_1.amax().item(), arg130_1.amin().item())
        print('triton__13', 'in_ptr4', 'arg131_1', (arg131_1.sum()/arg131_1.nelement()).item(), arg131_1.amax().item(), arg131_1.amin().item())
        triton__13_xnumel = 197*s0
        triton__13.run(buf245, buf221, buf228, buf243, arg130_1, arg131_1, buf247, triton__13_xnumel, 384, grid=grid(triton__13_xnumel), stream=stream0)
        print('triton__13', 'in_out_ptr0', 'buf245', (buf245.sum()/buf245.nelement()).item(), buf245.amax().item(), buf245.amin().item())
        print('triton__13', 'out_ptr1', 'buf247', (buf247.sum()/buf247.nelement()).item(), buf247.amax().item(), buf247.amin().item())
        del arg130_1
        del arg131_1
        buf248 = as_strided(buf227, (197*s0, 1536), (1536, 1)); del buf227  # reuse
        extern_kernels.addmm(arg133_1, as_strided(buf247, (197*s0, 384), (384, 1)), as_strided(arg132_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf248)
        del arg132_1
        del arg133_1
        buf249 = as_strided(buf248, (s0, 197, 1536), (302592, 1536, 1)); del buf248  # reuse
        print('triton__9', 'in_out_ptr0', 'buf249', (buf249.sum()/buf249.nelement()).item(), buf249.amax().item(), buf249.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf249, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf249', (buf249.sum()/buf249.nelement()).item(), buf249.amax().item(), buf249.amin().item())
        buf250 = as_strided(buf247, (197*s0, 384), (384, 1)); del buf247  # reuse
        extern_kernels.addmm(arg135_1, as_strided(buf249, (197*s0, 1536), (1536, 1)), as_strided(arg134_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf250)
        del arg134_1
        del arg135_1
        buf251 = buf245; del buf245  # reuse
        buf252 = buf251; del buf251  # reuse
        buf254 = as_strided(buf198, (s0, 197, 384), (75648, 384, 1)); del buf198  # reuse
        print('triton__14', 'in_out_ptr0', 'buf252', (buf252.sum()/buf252.nelement()).item(), buf252.amax().item(), buf252.amin().item())
        print('triton__14', 'in_ptr0', 'buf221', (buf221.sum()/buf221.nelement()).item(), buf221.amax().item(), buf221.amin().item())
        print('triton__14', 'in_ptr1', 'buf228', (buf228.sum()/buf228.nelement()).item(), buf228.amax().item(), buf228.amin().item())
        print('triton__14', 'in_ptr2', 'buf243', (buf243.sum()/buf243.nelement()).item(), buf243.amax().item(), buf243.amin().item())
        print('triton__14', 'in_ptr3', 'buf250', (buf250.sum()/buf250.nelement()).item(), buf250.amax().item(), buf250.amin().item())
        print('triton__14', 'in_ptr4', 'arg136_1', (arg136_1.sum()/arg136_1.nelement()).item(), arg136_1.amax().item(), arg136_1.amin().item())
        print('triton__14', 'in_ptr5', 'arg137_1', (arg137_1.sum()/arg137_1.nelement()).item(), arg137_1.amax().item(), arg137_1.amin().item())
        triton__14_xnumel = 197*s0
        triton__14.run(buf252, buf221, buf228, buf243, buf250, arg136_1, arg137_1, buf254, triton__14_xnumel, 384, grid=grid(triton__14_xnumel), stream=stream0)
        print('triton__14', 'in_out_ptr0', 'buf252', (buf252.sum()/buf252.nelement()).item(), buf252.amax().item(), buf252.amin().item())
        print('triton__14', 'out_ptr1', 'buf254', (buf254.sum()/buf254.nelement()).item(), buf254.amax().item(), buf254.amin().item())
        del arg136_1
        del arg137_1
        buf255 = buf233; del buf233  # reuse
        extern_kernels.addmm(arg139_1, as_strided(buf254, (197*s0, 384), (384, 1)), as_strided(arg138_1, (384, 1152), (1, 384)), alpha=1, beta=1, out=buf255)
        del arg138_1
        del arg139_1
        buf256 = as_strided(buf254, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf254  # reuse
        print('triton__3', 'in_ptr0', 'buf255', (buf255.sum()/buf255.nelement()).item(), buf255.amax().item(), buf255.amin().item())
        triton__3_xnumel = 75648*s0
        triton__3.run(buf255, buf256, triton__3_xnumel, grid=grid(triton__3_xnumel), stream=stream0)
        print('triton__3', 'out_ptr0', 'buf256', (buf256.sum()/buf256.nelement()).item(), buf256.amax().item(), buf256.amin().item())
        buf257 = as_strided(buf183, (s0, 6, 64, 197), (75648, 12608, 197, 1)); del buf183  # reuse
        print('triton__4', 'in_ptr0', 'buf255', (buf255.sum()/buf255.nelement()).item(), buf255.amax().item(), buf255.amin().item())
        triton__4_xnumel = 384*s0
        triton__4.run(buf255, buf257, triton__4_xnumel, 197, grid=grid(triton__4_xnumel, 197), stream=stream0)
        print('triton__4', 'out_ptr0', 'buf257', (buf257.sum()/buf257.nelement()).item(), buf257.amax().item(), buf257.amin().item())
        buf258 = as_strided(buf239, (6*s0, 197, 197), (38809, 197, 1)); del buf239  # reuse
        extern_kernels.bmm(as_strided(buf256, (6*s0, 197, 64), (12608, 64, 1)), as_strided(buf257, (6*s0, 64, 197), (12608, 197, 1)), out=buf258)
        buf261 = as_strided(buf236, (s0, 6, 197, 197), (232854, 38809, 197, 1)); del buf236  # reuse
        print('triton__5', 'in_ptr0', 'buf258', (buf258.sum()/buf258.nelement()).item(), buf258.amax().item(), buf258.amin().item())
        triton__5_xnumel = 1182*s0
        triton__5.run(buf258, buf261, triton__5_xnumel, 197, grid=grid(triton__5_xnumel), stream=stream0)
        print('triton__5', 'out_ptr2', 'buf261', (buf261.sum()/buf261.nelement()).item(), buf261.amax().item(), buf261.amin().item())
        del buf258
        buf262 = as_strided(buf257, (s0, 6, 197, 64), (75648, 12608, 64, 1)); del buf257  # reuse
        print('triton__6', 'in_ptr0', 'buf255', (buf255.sum()/buf255.nelement()).item(), buf255.amax().item(), buf255.amin().item())
        triton__6_xnumel = 75648*s0
        triton__6.run(buf255, buf262, triton__6_xnumel, grid=grid(triton__6_xnumel), stream=stream0)
        print('triton__6', 'out_ptr0', 'buf262', (buf262.sum()/buf262.nelement()).item(), buf262.amax().item(), buf262.amin().item())
        del buf255
        buf263 = as_strided(buf256, (6*s0, 197, 64), (12608, 64, 1)); del buf256  # reuse
        extern_kernels.bmm(as_strided(buf261, (6*s0, 197, 197), (38809, 197, 1)), as_strided(buf262, (6*s0, 197, 64), (12608, 64, 1)), out=buf263)
        del buf261
        buf264 = as_strided(buf262, (s0, 197, 6, 64), (75648, 384, 64, 1)); del buf262  # reuse
        print('triton__7', 'in_ptr0', 'buf263', (buf263.sum()/buf263.nelement()).item(), buf263.amax().item(), buf263.amin().item())
        triton__7_xnumel = 75648*s0
        triton__7.run(buf263, buf264, triton__7_xnumel, grid=grid(triton__7_xnumel), stream=stream0)
        print('triton__7', 'out_ptr0', 'buf264', (buf264.sum()/buf264.nelement()).item(), buf264.amax().item(), buf264.amin().item())
        buf265 = as_strided(buf263, (197*s0, 384), (384, 1)); del buf263  # reuse
        extern_kernels.addmm(arg141_1, as_strided(buf264, (197*s0, 384), (384, 1)), as_strided(arg140_1, (384, 384), (1, 384)), alpha=1, beta=1, out=buf265)
        del arg140_1
        del arg141_1
        buf266 = buf221; del buf221  # reuse
        buf267 = buf252; del buf252  # reuse
        buf268 = buf267; del buf267  # reuse
        buf270 = as_strided(buf264, (s0, 197, 384), (75648, 384, 1)); del buf264  # reuse
        print('triton__15', 'in_out_ptr0', 'buf266', (buf266.sum()/buf266.nelement()).item(), buf266.amax().item(), buf266.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf268', (buf268.sum()/buf268.nelement()).item(), buf268.amax().item(), buf268.amin().item())
        print('triton__15', 'in_ptr0', 'buf228', (buf228.sum()/buf228.nelement()).item(), buf228.amax().item(), buf228.amin().item())
        print('triton__15', 'in_ptr1', 'buf243', (buf243.sum()/buf243.nelement()).item(), buf243.amax().item(), buf243.amin().item())
        print('triton__15', 'in_ptr2', 'buf250', (buf250.sum()/buf250.nelement()).item(), buf250.amax().item(), buf250.amin().item())
        print('triton__15', 'in_ptr3', 'buf265', (buf265.sum()/buf265.nelement()).item(), buf265.amax().item(), buf265.amin().item())
        print('triton__15', 'in_ptr4', 'arg142_1', (arg142_1.sum()/arg142_1.nelement()).item(), arg142_1.amax().item(), arg142_1.amin().item())
        print('triton__15', 'in_ptr5', 'arg143_1', (arg143_1.sum()/arg143_1.nelement()).item(), arg143_1.amax().item(), arg143_1.amin().item())
        triton__15_xnumel = 197*s0
        triton__15.run(buf266, buf268, buf228, buf243, buf250, buf265, arg142_1, arg143_1, buf270, triton__15_xnumel, 384, grid=grid(triton__15_xnumel), stream=stream0)
        print('triton__15', 'in_out_ptr0', 'buf266', (buf266.sum()/buf266.nelement()).item(), buf266.amax().item(), buf266.amin().item())
        print('triton__15', 'in_out_ptr1', 'buf268', (buf268.sum()/buf268.nelement()).item(), buf268.amax().item(), buf268.amin().item())
        print('triton__15', 'out_ptr1', 'buf270', (buf270.sum()/buf270.nelement()).item(), buf270.amax().item(), buf270.amin().item())
        del arg142_1
        del arg143_1
        del buf228
        del buf243
        del buf250
        buf271 = as_strided(buf249, (197*s0, 1536), (1536, 1)); del buf249  # reuse
        extern_kernels.addmm(arg145_1, as_strided(buf270, (197*s0, 384), (384, 1)), as_strided(arg144_1, (384, 1536), (1, 384)), alpha=1, beta=1, out=buf271)
        del arg144_1
        del arg145_1
        buf272 = as_strided(buf271, (s0, 197, 1536), (302592, 1536, 1)); del buf271  # reuse
        print('triton__9', 'in_out_ptr0', 'buf272', (buf272.sum()/buf272.nelement()).item(), buf272.amax().item(), buf272.amin().item())
        triton__9_xnumel = 302592*s0
        triton__9.run(buf272, triton__9_xnumel, grid=grid(triton__9_xnumel), stream=stream0)
        print('triton__9', 'in_out_ptr0', 'buf272', (buf272.sum()/buf272.nelement()).item(), buf272.amax().item(), buf272.amin().item())
        buf273 = as_strided(buf270, (197*s0, 384), (384, 1)); del buf270  # reuse
        extern_kernels.addmm(arg147_1, as_strided(buf272, (197*s0, 1536), (1536, 1)), as_strided(arg146_1, (1536, 384), (1, 1536)), alpha=1, beta=1, out=buf273)
        del arg146_1
        del arg147_1
        del buf272
        buf274 = buf268; del buf268  # reuse
        buf275 = buf274; del buf274  # reuse
        buf277 = as_strided(buf265, (s0, 197, 384), (75648, 384, 1)); del buf265  # reuse
        print('triton__12', 'in_out_ptr0', 'buf275', (buf275.sum()/buf275.nelement()).item(), buf275.amax().item(), buf275.amin().item())
        print('triton__12', 'in_ptr0', 'buf266', (buf266.sum()/buf266.nelement()).item(), buf266.amax().item(), buf266.amin().item())
        print('triton__12', 'in_ptr1', 'buf273', (buf273.sum()/buf273.nelement()).item(), buf273.amax().item(), buf273.amin().item())
        print('triton__12', 'in_ptr2', 'arg148_1', (arg148_1.sum()/arg148_1.nelement()).item(), arg148_1.amax().item(), arg148_1.amin().item())
        print('triton__12', 'in_ptr3', 'arg149_1', (arg149_1.sum()/arg149_1.nelement()).item(), arg149_1.amax().item(), arg149_1.amin().item())
        triton__12_xnumel = 197*s0
        triton__12.run(buf275, buf266, buf273, arg148_1, arg149_1, buf277, triton__12_xnumel, 384, grid=grid(triton__12_xnumel), stream=stream0)
        print('triton__12', 'in_out_ptr0', 'buf275', (buf275.sum()/buf275.nelement()).item(), buf275.amax().item(), buf275.amin().item())
        print('triton__12', 'out_ptr1', 'buf277', (buf277.sum()/buf277.nelement()).item(), buf277.amax().item(), buf277.amin().item())
        del arg148_1
        del arg149_1
        del buf266
        del buf273
        del buf275
        buf278 = empty_strided((s0, 1000), (1000, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(arg151_1, as_strided(buf277, (s0, 384), (75648, 1)), as_strided(arg150_1, (384, 1000), (1, 384)), alpha=1, beta=1, out=buf278)
        del arg150_1
        del arg151_1
        return (buf278, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg50_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg52_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg54_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg55_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg56_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg58_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg59_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg60_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg61_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg62_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg64_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg66_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg67_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg68_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg69_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg72_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg73_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg74_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg76_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg78_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg79_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg80_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg84_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg85_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg86_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg90_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg91_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg92_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg96_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg97_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg98_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg102_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg103_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg104_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg108_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg109_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg110_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg114_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg115_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg116_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg118_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg119_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg120_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg121_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg122_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg124_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg126_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg127_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg128_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg131_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg132_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg133_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg134_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg136_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg138_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg139_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg140_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg142_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg143_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg144_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg145_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg146_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg150_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg151_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg152_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float16)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1]))
