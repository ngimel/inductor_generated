
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

@reduction(size_hints=[1024, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton__0(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp4, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr3 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (x0 + (2048*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = 49.0
        tmp4 = tmp2 / tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        tmp6 = tmp5.to(tl.float32)
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp6 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp11 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask)
    tmp22 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = 49.0
    tmp4 = tmp2 / tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00015943877551020407
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (25088 + r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (25088 + x4 + (50176*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (12544 + r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        tmp6 = tmp5.to(tl.float32)
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp6 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp14 = tl.load(in_ptr5 + (x0), xmask)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (12544 + x4 + (50176*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00015943877551020407
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (50176*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        tmp6 = tmp5.to(tl.float32)
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp6 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp13, xmask)
    tmp14 = tl.load(in_ptr5 + (x0), xmask)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (50176*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00015943877551020407
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (50176*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 12544
    x1 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (50176*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12544
    x1 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (37632 + x0 + (50176*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (50176*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), xmask)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x1), xmask)
    tmp20 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp15 = tl.load(in_ptr5 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr2 + (x0 + (2048*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = 49.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.where(tmp3, tmp1, tmp6)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tmp10.to(tl.float32)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp11 * tmp16
        _tmp18 = tl.where(xmask & rmask, _tmp18 + tmp17, _tmp18)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp12, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp18, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask)
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x4), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x3), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp24 = tl.load(in_ptr8 + (x1), xmask)
    tmp27 = tl.load(in_ptr9 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.00015943877551020407
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp11 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp30, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x2), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x2), xmask).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = tmp0 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp0 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp13, xmask)
    tmp14 = tl.load(in_ptr5 + (x0), xmask)
    tmp16 = tl.load(in_ptr6 + (x0), xmask)
    tmp15 = tmp7 * tmp14
    tmp17 = tmp13 * tmp16
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp17, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp16', 13: '*fp16', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp13 = tl.load(in_ptr5 + (x1), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x3), xmask).to(tl.float32)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp26 = tl.load(in_ptr10 + (x1), xmask)
    tmp32 = tl.load(in_ptr11 + (x1), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 0.00015943877551020407
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp6
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp0 - tmp29
    tmp31 = tmp30 - tmp14
    tmp33 = tmp26 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp35, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x2 = (xindex // 196) % 256
    x3 = (xindex // 50176)
    x7 = xindex % 50176
    tmp0 = (x1 // 2)
    tmp1 = (x0 // 2)
    tmp2 = 1 + (((1 + x1) // 2))
    tmp3 = 1 + (((1 + x0) // 2))
    tmp4 = 0
    tmp5 = tl.where(tmp0 != tmp0, tmp0, tl.where(tmp0 > tmp4, tmp0, tmp4))
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp4, tmp1, tmp4))
    tmp7 = 7
    tmp8 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 < tmp7, tmp2, tmp7))
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp7, tmp3, tmp7))
    tmp10 = tmp5 + tmp4
    tmp11 = tmp6 + tmp4
    tmp12 = 1
    tmp13 = tmp8 - tmp12
    tmp14 = tl.where(tmp10 != tmp10, tmp10, tl.where(tmp10 < tmp13, tmp10, tmp13))
    tmp15 = tmp9 - tmp12
    tmp16 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp15, tmp11, tmp15))
    tmp17 = tl.load(in_ptr0 + (37632 + tmp16 + (7*tmp14) + (49*x2) + (50176*x3)), xmask).to(tl.float32)
    tmp18 = tmp17 / 9
    tmp19 = tmp10 < tmp8
    tmp20 = tmp11 < tmp9
    tmp21 = tmp19 & tmp20
    tmp22 = 0.0
    tmp23 = tl.where(tmp21, tmp18, tmp22)
    tmp24 = tmp6 + tmp12
    tmp25 = tl.where(tmp24 != tmp24, tmp24, tl.where(tmp24 < tmp15, tmp24, tmp15))
    tmp26 = tl.load(in_ptr0 + (37632 + tmp25 + (7*tmp14) + (49*x2) + (50176*x3)), xmask).to(tl.float32)
    tmp27 = tmp26 / 9
    tmp28 = tmp24 < tmp9
    tmp29 = tmp19 & tmp28
    tmp30 = tmp23 + tmp27
    tmp31 = tl.where(tmp29, tmp30, tmp23)
    tmp32 = tmp5 + tmp12
    tmp33 = tl.where(tmp32 != tmp32, tmp32, tl.where(tmp32 < tmp13, tmp32, tmp13))
    tmp34 = tl.load(in_ptr0 + (37632 + tmp16 + (7*tmp33) + (49*x2) + (50176*x3)), xmask).to(tl.float32)
    tmp35 = tmp34 / 9
    tmp36 = tmp32 < tmp8
    tmp37 = tmp36 & tmp20
    tmp38 = tmp31 + tmp35
    tmp39 = tl.where(tmp37, tmp38, tmp31)
    tmp40 = tl.load(in_ptr0 + (37632 + tmp25 + (7*tmp33) + (49*x2) + (50176*x3)), xmask).to(tl.float32)
    tmp41 = tmp40 / 9
    tmp42 = tmp36 & tmp28
    tmp43 = tmp39 + tmp41
    tmp44 = tl.where(tmp42, tmp43, tmp39)
    tl.store(out_ptr0 + (x7 + (200704*x3) + tl.zeros([XBLOCK], tl.int32)), tmp44, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (12544 + r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (12544 + x4 + (50176*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (50176*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 50176
    x1 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (200704*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), xmask)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x1), xmask)
    tmp20 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp7 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp23 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 3.985969387755102e-05
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (50176 + (196*x0) + (100352*(r2 // 196)) + (3211264*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp11, xmask)
''')


triton__34 = async_compile.triton('''
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
def triton__34(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__35 = async_compile.triton('''
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
def triton__35(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (50176 + x4 + (100352*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (25088 + (196*x0) + (100352*(r2 // 196)) + (3211264*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        tmp6 = tmp5.to(tl.float32)
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp6 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp13, xmask)
''')


triton__39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (25088 + x4 + (100352*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 3.985969387755102e-05
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (100352*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + ((196*x0) + (100352*(r2 // 196)) + (3211264*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        tmp6 = tmp5.to(tl.float32)
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp6 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp13, xmask)
''')


triton__41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (100352*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 3.985969387755102e-05
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (100352*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (100352*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (75264 + x0 + (100352*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (100352*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), xmask)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x1), xmask)
    tmp20 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp1 * tmp6
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp2, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 3.985969387755102e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp12, xmask)
''')


triton__50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = tmp0 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp0 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp13, xmask)
    tmp14 = tl.load(in_ptr5 + (x0), xmask)
    tmp16 = tl.load(in_ptr6 + (x0), xmask)
    tmp15 = tmp7 * tmp14
    tmp17 = tmp13 * tmp16
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp17, xmask)
''')


triton__51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp16', 13: '*fp16', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp13 = tl.load(in_ptr5 + (x1), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x3), xmask).to(tl.float32)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp26 = tl.load(in_ptr10 + (x1), xmask)
    tmp32 = tl.load(in_ptr11 + (x1), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 3.985969387755102e-05
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp6
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp0 - tmp29
    tmp31 = tmp30 - tmp14
    tmp33 = tmp26 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp35, xmask)
''')


triton__52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x2 = (xindex // 784) % 128
    x3 = (xindex // 100352)
    x7 = xindex % 100352
    tmp0 = (x1 // 2)
    tmp1 = (x0 // 2)
    tmp2 = 1 + (((1 + x1) // 2))
    tmp3 = 1 + (((1 + x0) // 2))
    tmp4 = 0
    tmp5 = tl.where(tmp0 != tmp0, tmp0, tl.where(tmp0 > tmp4, tmp0, tmp4))
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp4, tmp1, tmp4))
    tmp7 = 14
    tmp8 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 < tmp7, tmp2, tmp7))
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp7, tmp3, tmp7))
    tmp10 = tmp5 + tmp4
    tmp11 = tmp6 + tmp4
    tmp12 = 1
    tmp13 = tmp8 - tmp12
    tmp14 = tl.where(tmp10 != tmp10, tmp10, tl.where(tmp10 < tmp13, tmp10, tmp13))
    tmp15 = tmp9 - tmp12
    tmp16 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp15, tmp11, tmp15))
    tmp17 = tl.load(in_ptr0 + (75264 + tmp16 + (14*tmp14) + (196*x2) + (100352*x3)), xmask).to(tl.float32)
    tmp18 = tmp17 / 9
    tmp19 = tmp10 < tmp8
    tmp20 = tmp11 < tmp9
    tmp21 = tmp19 & tmp20
    tmp22 = 0.0
    tmp23 = tl.where(tmp21, tmp18, tmp22)
    tmp24 = tmp6 + tmp12
    tmp25 = tl.where(tmp24 != tmp24, tmp24, tl.where(tmp24 < tmp15, tmp24, tmp15))
    tmp26 = tl.load(in_ptr0 + (75264 + tmp25 + (14*tmp14) + (196*x2) + (100352*x3)), xmask).to(tl.float32)
    tmp27 = tmp26 / 9
    tmp28 = tmp24 < tmp9
    tmp29 = tmp19 & tmp28
    tmp30 = tmp23 + tmp27
    tmp31 = tl.where(tmp29, tmp30, tmp23)
    tmp32 = tmp5 + tmp12
    tmp33 = tl.where(tmp32 != tmp32, tmp32, tl.where(tmp32 < tmp13, tmp32, tmp13))
    tmp34 = tl.load(in_ptr0 + (75264 + tmp16 + (14*tmp33) + (196*x2) + (100352*x3)), xmask).to(tl.float32)
    tmp35 = tmp34 / 9
    tmp36 = tmp32 < tmp8
    tmp37 = tmp36 & tmp20
    tmp38 = tmp31 + tmp35
    tmp39 = tl.where(tmp37, tmp38, tmp31)
    tmp40 = tl.load(in_ptr0 + (75264 + tmp25 + (14*tmp33) + (196*x2) + (100352*x3)), xmask).to(tl.float32)
    tmp41 = tmp40 / 9
    tmp42 = tmp36 & tmp28
    tmp43 = tmp39 + tmp41
    tmp44 = tl.where(tmp42, tmp43, tmp39)
    tl.store(out_ptr0 + (x7 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp44, xmask)
''')


triton__53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (25088 + (196*x0) + (100352*(r2 // 196)) + (3211264*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp11, xmask)
''')


triton__54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (25088 + x4 + (100352*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + ((196*x0) + (100352*(r2 // 196)) + (3211264*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp11, xmask)
''')


triton__56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (100352*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__57(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (401408*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), xmask)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x1), xmask)
    tmp20 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 9.964923469387754e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__60(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp7 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
''')


triton__62 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp23 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 9.964923469387754e-06
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__63 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__63(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__64 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp4 = 0.0
        tmp5 = tl.load(in_ptr1 + (100352 + (784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.where(tmp2, tmp7, 0)
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        tmp10 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp11 = 0.0
        tmp12 = tl.load(in_ptr1 + (100352 + (784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.load(in_ptr2 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp14 * tmp18
        tmp20 = tl.where(tmp2, tmp19, 0)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp21, xmask)
''')


triton__65 = async_compile.triton('''
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
def triton__65(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__66 = async_compile.triton('''
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
def triton__66(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__67 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (100352 + x4 + (200704*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 9.964923469387754e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__68 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__68(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__69 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp25 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp4 = 0.0
        tmp5 = tl.load(in_ptr1 + (50176 + (784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp3, tmp4, tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp13 = 0.0
        tmp14 = tl.load(in_ptr1 + (50176 + (784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.where(tmp12, tmp13, tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.load(in_ptr3 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.load(in_ptr4 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp18 * tmp22
        tmp24 = tl.where(tmp2, tmp23, 0)
        _tmp25 = tl.where(xmask & rmask, _tmp25 + tmp24, _tmp25)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp25, xmask)
''')


triton__70 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (50176 + x4 + (200704*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 9.964923469387754e-06
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (200704*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__71 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__71(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp25 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp4 = 0.0
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp3, tmp4, tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp13 = 0.0
        tmp14 = tl.load(in_ptr1 + ((784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.where(tmp12, tmp13, tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.load(in_ptr3 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.load(in_ptr4 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp18 * tmp22
        tmp24 = tl.where(tmp2, tmp23, 0)
        _tmp25 = tl.where(xmask & rmask, _tmp25 + tmp24, _tmp25)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp25, xmask)
''')


triton__72 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__72(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (200704*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 9.964923469387754e-06
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (200704*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__73 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__73(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50176
    x1 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (150528 + x0 + (200704*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (200704*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__74 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__74(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__75 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__75(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), xmask)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x1), xmask)
    tmp20 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 9.964923469387754e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__76 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__76(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__77 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__77(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp1 * tmp6
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp2, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__78 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__78(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 9.964923469387754e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__79 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__79(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp12, xmask)
''')


triton__80 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton__80(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = tmp0 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp0 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp13, xmask)
    tmp14 = tl.load(in_ptr5 + (x0), xmask)
    tmp16 = tl.load(in_ptr6 + (x0), xmask)
    tmp15 = tmp7 * tmp14
    tmp17 = tmp13 * tmp16
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp17, xmask)
''')


triton__81 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp16', 13: '*fp16', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton__81(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp13 = tl.load(in_ptr5 + (x1), xmask)
    tmp16 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x3), xmask).to(tl.float32)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp24 = tl.load(in_ptr9 + (x1), xmask)
    tmp26 = tl.load(in_ptr10 + (x1), xmask)
    tmp32 = tl.load(in_ptr11 + (x1), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 9.964923469387754e-06
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp6
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp0 - tmp29
    tmp31 = tmp30 - tmp14
    tmp33 = tmp26 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp35, xmask)
''')


triton__82 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__82(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x2 = (xindex // 3136) % 64
    x3 = (xindex // 200704)
    x7 = xindex % 200704
    tmp0 = (x1 // 2)
    tmp1 = (x0 // 2)
    tmp2 = 1 + (((1 + x1) // 2))
    tmp3 = 1 + (((1 + x0) // 2))
    tmp4 = 0
    tmp5 = tl.where(tmp0 != tmp0, tmp0, tl.where(tmp0 > tmp4, tmp0, tmp4))
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp4, tmp1, tmp4))
    tmp7 = 28
    tmp8 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 < tmp7, tmp2, tmp7))
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp7, tmp3, tmp7))
    tmp10 = tmp5 + tmp4
    tmp11 = tmp6 + tmp4
    tmp12 = 1
    tmp13 = tmp8 - tmp12
    tmp14 = tl.where(tmp10 != tmp10, tmp10, tl.where(tmp10 < tmp13, tmp10, tmp13))
    tmp15 = tmp9 - tmp12
    tmp16 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp15, tmp11, tmp15))
    tmp17 = tl.load(in_ptr0 + (150528 + tmp16 + (28*tmp14) + (784*x2) + (200704*x3)), xmask).to(tl.float32)
    tmp18 = tmp17 / 9
    tmp19 = tmp10 < tmp8
    tmp20 = tmp11 < tmp9
    tmp21 = tmp19 & tmp20
    tmp22 = 0.0
    tmp23 = tl.where(tmp21, tmp18, tmp22)
    tmp24 = tmp6 + tmp12
    tmp25 = tl.where(tmp24 != tmp24, tmp24, tl.where(tmp24 < tmp15, tmp24, tmp15))
    tmp26 = tl.load(in_ptr0 + (150528 + tmp25 + (28*tmp14) + (784*x2) + (200704*x3)), xmask).to(tl.float32)
    tmp27 = tmp26 / 9
    tmp28 = tmp24 < tmp9
    tmp29 = tmp19 & tmp28
    tmp30 = tmp23 + tmp27
    tmp31 = tl.where(tmp29, tmp30, tmp23)
    tmp32 = tmp5 + tmp12
    tmp33 = tl.where(tmp32 != tmp32, tmp32, tl.where(tmp32 < tmp13, tmp32, tmp13))
    tmp34 = tl.load(in_ptr0 + (150528 + tmp16 + (28*tmp33) + (784*x2) + (200704*x3)), xmask).to(tl.float32)
    tmp35 = tmp34 / 9
    tmp36 = tmp32 < tmp8
    tmp37 = tmp36 & tmp20
    tmp38 = tmp31 + tmp35
    tmp39 = tl.where(tmp37, tmp38, tmp31)
    tmp40 = tl.load(in_ptr0 + (150528 + tmp25 + (28*tmp33) + (784*x2) + (200704*x3)), xmask).to(tl.float32)
    tmp41 = tmp40 / 9
    tmp42 = tmp36 & tmp28
    tmp43 = tmp39 + tmp41
    tmp44 = tl.where(tmp42, tmp43, tmp39)
    tl.store(out_ptr0 + (x7 + (802816*x3) + tl.zeros([XBLOCK], tl.int32)), tmp44, xmask)
''')


triton__83 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__83(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp4 = 0.0
        tmp5 = tl.load(in_ptr1 + (50176 + (784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.where(tmp2, tmp7, 0)
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        tmp10 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp11 = 0.0
        tmp12 = tl.load(in_ptr1 + (50176 + (784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.load(in_ptr2 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp14 * tmp18
        tmp20 = tl.where(tmp2, tmp19, 0)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp21, xmask)
''')


triton__84 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__84(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (50176 + x4 + (200704*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 9.964923469387754e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__85 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__85(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp4 = 0.0
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.where(tmp2, tmp7, 0)
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
        tmp10 = tl.load(in_ptr0 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last')
        tmp11 = 0.0
        tmp12 = tl.load(in_ptr1 + ((784*x0) + (200704*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.load(in_ptr2 + ((784*x0) + (50176*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp14 * tmp18
        tmp20 = tl.where(tmp2, tmp19, 0)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp21, xmask)
''')


triton__86 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__86(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (200704*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 9.964923469387754e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__87 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__87(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (802816*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__88 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 524288],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__88(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp11, xmask)
    tmp12 = tl.load(in_ptr4 + (x0), xmask)
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp13, xmask)
''')


triton__89 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__89(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), xmask)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x1), xmask)
    tmp20 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 2.4912308673469386e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__90 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__90(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__91 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 524288],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__91(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp7 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
''')


triton__92 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton__92(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp23 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 2.4912308673469386e-06
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__93 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__93(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__94 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__94(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (200704 + (56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (401408*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp11, xmask)
''')


triton__95 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__95(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 14
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__96 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__96(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 14
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__97 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__97(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (200704 + x4 + (401408*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 2.4912308673469386e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__98 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__98(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__99 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__99(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (100352 + (56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (401408*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        tmp6 = tmp5.to(tl.float32)
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp6 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp13, xmask)
''')


triton__100 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__100(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (100352 + x4 + (401408*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 2.4912308673469386e-06
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (401408*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__101 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__101(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp10 = tl.load(in_ptr4 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (401408*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr3 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = tl.where(tmp0, tmp1, tmp4)
        tmp6 = tmp5.to(tl.float32)
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp6 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp13, xmask)
''')


triton__102 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton__102(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (401408*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp11 = tl.load(in_ptr5 + (x1), xmask)
    tmp14 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x1), xmask)
    tmp22 = tl.load(in_ptr8 + (x1), xmask)
    tmp1 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.where(tmp0, tmp1, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 2.4912308673469386e-06
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
    tl.store(out_ptr2 + (x4 + (401408*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__103 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__103(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (301056 + x0 + (401408*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (401408*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__104 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__104(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (12845056*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (12845056*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (12845056*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp11, xmask)
''')


triton__105 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__105(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), xmask)
    tmp9 = tl.load(in_ptr3 + (x1), xmask)
    tmp12 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr5 + (x1), xmask)
    tmp20 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 2.4912308673469386e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__106 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton__106(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


triton__107 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 524288],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__107(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp1 * tmp6
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp2, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp8, xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, xmask)
''')


triton__108 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__108(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp14 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 2.4912308673469386e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
''')


triton__109 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 524288],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]})
@triton.jit
def triton__109(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp11 = tl.load(in_ptr4 + (x0), xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    tmp17 = tl.load(in_ptr6 + (x0), xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = tl.load(in_ptr5 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp7 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp7 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp20, xmask)
    tmp21 = tl.load(in_ptr7 + (x0), xmask)
    tmp23 = tl.load(in_ptr8 + (x0), xmask)
    tmp22 = tmp14 * tmp21
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp22, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__110 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp16', 15: '*fp16', 16: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]})
@triton.jit
def triton__110(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp23 = tl.load(in_ptr8 + (x1), xmask)
    tmp26 = tl.load(in_ptr9 + (x3), xmask).to(tl.float32)
    tmp28 = tl.load(in_ptr10 + (x1), xmask)
    tmp30 = tl.load(in_ptr11 + (x1), xmask)
    tmp32 = tl.load(in_ptr12 + (x1), xmask)
    tmp38 = tl.load(in_ptr13 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 2.4912308673469386e-06
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp27 - tmp28
    tmp31 = tmp30 * tmp13
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp29 * tmp34
    tmp36 = tmp7 - tmp35
    tmp37 = tmp36 - tmp21
    tmp39 = tmp32 * tmp38
    tmp40 = tmp37 * tmp39
    tmp41 = tmp25.to(tl.float32)
    tmp42 = tmp40.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp41, xmask)
    tl.store(out_ptr3 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp42, xmask)
''')


triton__111 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__111(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__112 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__112(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x2 = (xindex // 3136) % 32
    x3 = (xindex // 100352)
    x7 = xindex % 100352
    tmp0 = (-1) + x1
    tmp1 = (-1) + x0
    tmp2 = 2 + x1
    tmp3 = 2 + x0
    tmp4 = 0
    tmp5 = tl.where(tmp0 != tmp0, tmp0, tl.where(tmp0 > tmp4, tmp0, tmp4))
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp4, tmp1, tmp4))
    tmp7 = 56
    tmp8 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 < tmp7, tmp2, tmp7))
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp7, tmp3, tmp7))
    tmp10 = tmp5 + tmp4
    tmp11 = tmp6 + tmp4
    tmp12 = 1
    tmp13 = tmp8 - tmp12
    tmp14 = tl.where(tmp10 != tmp10, tmp10, tl.where(tmp10 < tmp13, tmp10, tmp13))
    tmp15 = tmp9 - tmp12
    tmp16 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp15, tmp11, tmp15))
    tmp17 = tl.load(in_ptr0 + (301056 + tmp16 + (56*tmp14) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp18 = tmp17 / 9
    tmp19 = tmp10 < tmp8
    tmp20 = tmp11 < tmp9
    tmp21 = tmp19 & tmp20
    tmp22 = 0.0
    tmp23 = tl.where(tmp21, tmp18, tmp22)
    tmp24 = tmp6 + tmp12
    tmp25 = tl.where(tmp24 != tmp24, tmp24, tl.where(tmp24 < tmp15, tmp24, tmp15))
    tmp26 = tl.load(in_ptr0 + (301056 + tmp25 + (56*tmp14) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp27 = tmp26 / 9
    tmp28 = tmp24 < tmp9
    tmp29 = tmp19 & tmp28
    tmp30 = tmp23 + tmp27
    tmp31 = tl.where(tmp29, tmp30, tmp23)
    tmp32 = 2
    tmp33 = tmp6 + tmp32
    tmp34 = tl.where(tmp33 != tmp33, tmp33, tl.where(tmp33 < tmp15, tmp33, tmp15))
    tmp35 = tl.load(in_ptr0 + (301056 + tmp34 + (56*tmp14) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp36 = tmp35 / 9
    tmp37 = tmp33 < tmp9
    tmp38 = tmp19 & tmp37
    tmp39 = tmp31 + tmp36
    tmp40 = tl.where(tmp38, tmp39, tmp31)
    tmp41 = tmp5 + tmp12
    tmp42 = tl.where(tmp41 != tmp41, tmp41, tl.where(tmp41 < tmp13, tmp41, tmp13))
    tmp43 = tl.load(in_ptr0 + (301056 + tmp16 + (56*tmp42) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp44 = tmp43 / 9
    tmp45 = tmp41 < tmp8
    tmp46 = tmp45 & tmp20
    tmp47 = tmp40 + tmp44
    tmp48 = tl.where(tmp46, tmp47, tmp40)
    tmp49 = tl.load(in_ptr0 + (301056 + tmp25 + (56*tmp42) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp28
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp54 = tl.load(in_ptr0 + (301056 + tmp34 + (56*tmp42) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp37
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tmp59 = tmp5 + tmp32
    tmp60 = tl.where(tmp59 != tmp59, tmp59, tl.where(tmp59 < tmp13, tmp59, tmp13))
    tmp61 = tl.load(in_ptr0 + (301056 + tmp16 + (56*tmp60) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp62 = tmp61 / 9
    tmp63 = tmp59 < tmp8
    tmp64 = tmp63 & tmp20
    tmp65 = tmp58 + tmp62
    tmp66 = tl.where(tmp64, tmp65, tmp58)
    tmp67 = tl.load(in_ptr0 + (301056 + tmp25 + (56*tmp60) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp68 = tmp67 / 9
    tmp69 = tmp63 & tmp28
    tmp70 = tmp66 + tmp68
    tmp71 = tl.where(tmp69, tmp70, tmp66)
    tmp72 = tl.load(in_ptr0 + (301056 + tmp34 + (56*tmp60) + (3136*x2) + (401408*x3)), xmask).to(tl.float32)
    tmp73 = tmp72 / 9
    tmp74 = tmp63 & tmp37
    tmp75 = tmp71 + tmp73
    tmp76 = tl.where(tmp74, tmp75, tmp71)
    tl.store(out_ptr0 + (x7 + (401408*x3) + tl.zeros([XBLOCK], tl.int32)), tmp76, xmask)
''')


triton__113 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__113(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (100352 + (56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (401408*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp11, xmask)
''')


triton__114 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__114(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (100352 + x4 + (401408*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 2.4912308673469386e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__115 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__115(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (401408*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tl.load(in_ptr2 + ((56*(((r2 + (28672*x1)) // 56) % 56)) + (3136*x0) + (100352*(((r2 + (28672*x1)) // 3136))) + (r2 % 56)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp3 = tl.where(tmp0, tmp1, tmp2)
        tmp4 = tmp3.to(tl.float32)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp4 * tmp9
        _tmp11 = tl.where(xmask & rmask, _tmp11 + tmp10, _tmp11)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp11, xmask)
''')


triton__116 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__116(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x4 + (401408*x2)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp9 = tl.load(in_ptr4 + (x1), xmask)
    tmp12 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask)
    tmp20 = tl.load(in_ptr7 + (x1), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 2.4912308673469386e-06
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


triton__117 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__117(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__118 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__118(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__119 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__119(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex % 12544
    x1 = (xindex // 112) % 112
    x0 = xindex % 112
    x2 = (xindex // 12544)
    x5 = xindex
    tmp0 = x3
    tmp1 = (x1 // 2)
    tmp2 = (x0 // 2)
    tmp3 = 1 + (((1 + x1) // 2))
    tmp4 = 1 + (((1 + x0) // 2))
    tmp5 = 0
    tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp5, tmp1, tmp5))
    tmp7 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp5, tmp2, tmp5))
    tmp8 = 56
    tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp8, tmp3, tmp8))
    tmp10 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 < tmp8, tmp4, tmp8))
    tmp11 = tmp6 + tmp5
    tmp12 = tmp7 + tmp5
    tmp13 = 1
    tmp14 = tmp9 - tmp13
    tmp15 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp14, tmp11, tmp14))
    tmp16 = tmp10 - tmp13
    tmp17 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 < tmp16, tmp12, tmp16))
    tmp18 = tl.load(in_ptr0 + (tmp17 + (56*tmp15) + (3136*x2)), xmask)
    tmp19 = tl.load(in_ptr1 + (tmp17 + (56*tmp15) + (3136*x2)), xmask).to(tl.float32)
    tmp20 = tmp18 == tmp0
    tmp21 = 0.0
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp7 + tmp13
    tmp24 = tl.where(tmp23 != tmp23, tmp23, tl.where(tmp23 < tmp16, tmp23, tmp16))
    tmp25 = tl.load(in_ptr0 + (tmp24 + (56*tmp15) + (3136*x2)), xmask)
    tmp26 = tl.load(in_ptr1 + (tmp24 + (56*tmp15) + (3136*x2)), xmask).to(tl.float32)
    tmp27 = tmp25 == tmp0
    tmp28 = tmp11 < tmp9
    tmp29 = tmp23 < tmp10
    tmp30 = tmp28 & tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tmp22 + tmp26
    tmp33 = tl.where(tmp31, tmp32, tmp22)
    tmp34 = tmp6 + tmp13
    tmp35 = tl.where(tmp34 != tmp34, tmp34, tl.where(tmp34 < tmp14, tmp34, tmp14))
    tmp36 = tl.load(in_ptr0 + (tmp17 + (56*tmp35) + (3136*x2)), xmask)
    tmp37 = tl.load(in_ptr1 + (tmp17 + (56*tmp35) + (3136*x2)), xmask).to(tl.float32)
    tmp38 = tmp36 == tmp0
    tmp39 = tmp34 < tmp9
    tmp40 = tmp12 < tmp10
    tmp41 = tmp39 & tmp40
    tmp42 = tmp41 & tmp38
    tmp43 = tmp33 + tmp37
    tmp44 = tl.where(tmp42, tmp43, tmp33)
    tmp45 = tl.load(in_ptr0 + (tmp24 + (56*tmp35) + (3136*x2)), xmask)
    tmp46 = tl.load(in_ptr1 + (tmp24 + (56*tmp35) + (3136*x2)), xmask).to(tl.float32)
    tmp47 = tmp45 == tmp0
    tmp48 = tmp39 & tmp29
    tmp49 = tmp48 & tmp47
    tmp50 = tmp44 + tmp46
    tmp51 = tl.where(tmp49, tmp50, tmp44)
    tl.store(out_ptr0 + (x5 + tl.zeros([XBLOCK], tl.int32)), tmp51, xmask)
''')


triton__120 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 262144],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton__120(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (229376*x1)) // 112) % 112)) + (12544*x0) + (802816*(((r2 + (229376*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr1 + ((112*(((r2 + (229376*x1)) // 112) % 112)) + (12544*x0) + (802816*(((r2 + (229376*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp7 = tl.load(in_ptr2 + ((112*(((r2 + (229376*x1)) // 112) % 112)) + (12544*x0) + (802816*(((r2 + (229376*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        _tmp6 = tl.where(xmask & rmask, _tmp6 + tmp5, _tmp6)
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp5 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp12, xmask)
''')


triton__121 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__121(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
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
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__122 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__122(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
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
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__123 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__123(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x1), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (x1), xmask)
    tmp21 = tl.load(in_ptr6 + (x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 6.228077168367346e-07
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


triton__124 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__124(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_5, primals_8, primals_11, primals_14, primals_17, primals_20, primals_23, primals_26, primals_29, primals_32, primals_35, primals_38, primals_41, primals_44, primals_47, primals_50, primals_53, primals_56, primals_59, primals_62, primals_65, primals_68, primals_71, primals_74, primals_77, primals_80, primals_83, primals_86, primals_89, primals_92, primals_95, primals_98, primals_101, primals_104, primals_107, primals_110, primals_113, primals_116, primals_119, primals_122, primals_125, primals_128, primals_131, primals_134, primals_137, primals_140, primals_143, primals_146, primals_149, primals_152, primals_155, primals_158, primals_161, primals_164, primals_167, primals_170, primals_173, primals_176, primals_179, primals_182, primals_185, primals_188, primals_191, primals_194, primals_197, primals_200, primals_203, primals_206, primals_209, primals_212, primals_215, primals_218, primals_221, primals_224, primals_227, primals_230, primals_233, primals_236, primals_239, primals_242, primals_245, primals_248, primals_251, primals_254, convert_element_type, convert_element_type_1, convolution, squeeze_5, relu, getitem, getitem_1, convert_element_type_4, convolution_1, squeeze_14, convert_element_type_7, getitem_6, convolution_2, squeeze_23, convert_element_type_10, getitem_11, convolution_3, squeeze_32, convert_element_type_13, getitem_16, convolution_4, squeeze_41, getitem_21, cat, convert_element_type_16, convolution_5, squeeze_50, convert_element_type_19, convolution_6, squeeze_59, relu_5, convert_element_type_22, convolution_7, squeeze_68, convert_element_type_25, getitem_26, convolution_8, squeeze_77, add_46, convert_element_type_28, convolution_9, squeeze_86, add_52, convert_element_type_31, convolution_10, squeeze_95, cat_1, convert_element_type_34, convolution_11, squeeze_104, relu_10, convert_element_type_37, convolution_12, squeeze_113, convert_element_type_40, getitem_46, convolution_13, squeeze_122, add_74, convert_element_type_43, convolution_14, squeeze_131, add_80, convert_element_type_46, convolution_15, squeeze_140, cat_2, convert_element_type_49, convolution_16, squeeze_149, relu_15, convert_element_type_52, convolution_17, squeeze_158, convert_element_type_55, getitem_66, convolution_18, squeeze_167, convert_element_type_58, getitem_71, convolution_19, squeeze_176, convert_element_type_61, getitem_76, convolution_20, squeeze_185, getitem_81, cat_3, convert_element_type_64, convolution_21, squeeze_194, convert_element_type_67, convolution_22, squeeze_203, relu_20, convert_element_type_70, convolution_23, squeeze_212, convert_element_type_73, getitem_86, convolution_24, squeeze_221, add_133, convert_element_type_76, convolution_25, squeeze_230, add_139, convert_element_type_79, convolution_26, squeeze_239, cat_4, convert_element_type_82, convolution_27, squeeze_248, relu_25, convert_element_type_85, convolution_28, squeeze_257, convert_element_type_88, getitem_106, convolution_29, squeeze_266, add_161, convert_element_type_91, convolution_30, squeeze_275, add_167, convert_element_type_94, convolution_31, squeeze_284, cat_5, convert_element_type_97, convolution_32, squeeze_293, relu_30, convert_element_type_100, convolution_33, squeeze_302, convert_element_type_103, getitem_126, convolution_34, squeeze_311, add_189, convert_element_type_106, convolution_35, squeeze_320, add_195, convert_element_type_109, convolution_36, squeeze_329, cat_6, convert_element_type_112, convolution_37, squeeze_338, relu_35, convert_element_type_115, convolution_38, squeeze_347, convert_element_type_118, getitem_146, convolution_39, squeeze_356, convert_element_type_121, getitem_151, convolution_40, squeeze_365, convert_element_type_124, getitem_156, convolution_41, squeeze_374, getitem_161, cat_7, convert_element_type_127, convolution_42, squeeze_383, convert_element_type_130, convolution_43, squeeze_392, relu_40, convert_element_type_133, convolution_44, squeeze_401, convert_element_type_136, getitem_166, convolution_45, squeeze_410, add_248, convert_element_type_139, convolution_46, squeeze_419, add_254, convert_element_type_142, convolution_47, squeeze_428, cat_8, convert_element_type_145, convolution_48, squeeze_437, relu_45, convert_element_type_148, convolution_49, squeeze_446, convert_element_type_151, getitem_186, convolution_50, squeeze_455, add_276, convert_element_type_154, convolution_51, squeeze_464, add_282, convert_element_type_157, convolution_52, squeeze_473, cat_9, convert_element_type_160, convolution_53, squeeze_482, relu_50, convert_element_type_163, convolution_54, squeeze_491, convert_element_type_166, getitem_206, convolution_55, squeeze_500, add_304, convert_element_type_169, convolution_56, squeeze_509, add_310, convert_element_type_172, convolution_57, squeeze_518, cat_10, convert_element_type_175, convolution_58, squeeze_527, relu_55, convert_element_type_178, convolution_59, squeeze_536, convert_element_type_181, getitem_226, convolution_60, squeeze_545, add_332, convert_element_type_184, convolution_61, squeeze_554, add_338, convert_element_type_187, convolution_62, squeeze_563, cat_11, convert_element_type_190, convolution_63, squeeze_572, relu_60, convert_element_type_193, convolution_64, squeeze_581, convert_element_type_196, getitem_246, convolution_65, squeeze_590, add_360, convert_element_type_199, convolution_66, squeeze_599, add_366, convert_element_type_202, convolution_67, squeeze_608, cat_12, convert_element_type_205, convolution_68, squeeze_617, relu_65, convert_element_type_208, convolution_69, squeeze_626, convert_element_type_211, getitem_266, convolution_70, squeeze_635, convert_element_type_214, getitem_271, convolution_71, squeeze_644, convert_element_type_217, getitem_276, convolution_72, squeeze_653, getitem_281, cat_13, convert_element_type_220, convolution_73, squeeze_662, convert_element_type_223, convolution_74, squeeze_671, relu_70, convert_element_type_226, convolution_75, squeeze_680, convert_element_type_229, getitem_286, convolution_76, squeeze_689, add_419, convert_element_type_232, convolution_77, squeeze_698, add_425, convert_element_type_235, convolution_78, squeeze_707, cat_14, convert_element_type_238, convolution_79, squeeze_716, relu_75, convert_element_type_241, convolution_80, squeeze_725, convert_element_type_244, getitem_306, convolution_81, squeeze_734, add_447, convert_element_type_247, convolution_82, squeeze_743, add_453, convert_element_type_250, convolution_83, squeeze_752, cat_15, convert_element_type_253, convolution_84, squeeze_761, view, permute_1, le, unsqueeze_342, le_1, unsqueeze_354, le_2, unsqueeze_366, le_3, unsqueeze_378, le_4, unsqueeze_390, unsqueeze_402, le_6, unsqueeze_414, le_7, unsqueeze_426, le_8, unsqueeze_438, le_9, unsqueeze_450, unsqueeze_462, unsqueeze_474, le_11, unsqueeze_486, le_12, unsqueeze_498, le_13, unsqueeze_510, le_14, unsqueeze_522, unsqueeze_534, le_16, unsqueeze_546, le_17, unsqueeze_558, le_18, unsqueeze_570, le_19, unsqueeze_582, unsqueeze_594, le_21, unsqueeze_606, le_22, unsqueeze_618, le_23, unsqueeze_630, le_24, unsqueeze_642, unsqueeze_654, le_26, unsqueeze_666, le_27, unsqueeze_678, le_28, unsqueeze_690, le_29, unsqueeze_702, unsqueeze_714, le_31, unsqueeze_726, le_32, unsqueeze_738, le_33, unsqueeze_750, le_34, unsqueeze_762, unsqueeze_774, le_36, unsqueeze_786, le_37, unsqueeze_798, le_38, unsqueeze_810, le_39, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_41, unsqueeze_858, le_42, unsqueeze_870, le_43, unsqueeze_882, le_44, unsqueeze_894, unsqueeze_906, le_46, unsqueeze_918, le_47, unsqueeze_930, le_48, unsqueeze_942, le_49, unsqueeze_954, unsqueeze_966, le_51, unsqueeze_978, le_52, unsqueeze_990, le_53, unsqueeze_1002, le_54, unsqueeze_1014, unsqueeze_1026, le_56, unsqueeze_1038, le_57, unsqueeze_1050, le_58, unsqueeze_1062, le_59, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, le_61, unsqueeze_1110, le_62, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, le_66, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, le_71, unsqueeze_1230, le_72, unsqueeze_1242, le_73, unsqueeze_1254, le_74, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, le_76, unsqueeze_1302, le_77, unsqueeze_1314, le_78, unsqueeze_1326, le_79, unsqueeze_1338, unsqueeze_1350, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160, tangents_161, tangents_162, tangents_163, tangents_164, tangents_165, tangents_166, tangents_167, tangents_168, tangents_169, tangents_170, tangents_171, tangents_172, tangents_173, tangents_174, tangents_175, tangents_176, tangents_177, tangents_178, tangents_179, tangents_180, tangents_181, tangents_182, tangents_183, tangents_184, tangents_185, tangents_186, tangents_187, tangents_188, tangents_189, tangents_190, tangents_191, tangents_192, tangents_193, tangents_194, tangents_195, tangents_196, tangents_197, tangents_198, tangents_199, tangents_200, tangents_201, tangents_202, tangents_203, tangents_204, tangents_205, tangents_206, tangents_207, tangents_208, tangents_209, tangents_210, tangents_211, tangents_212, tangents_213, tangents_214, tangents_215, tangents_216, tangents_217, tangents_218, tangents_219, tangents_220, tangents_221, tangents_222, tangents_223, tangents_224, tangents_225, tangents_226, tangents_227, tangents_228, tangents_229, tangents_230, tangents_231, tangents_232, tangents_233, tangents_234, tangents_235, tangents_236, tangents_237, tangents_238, tangents_239, tangents_240, tangents_241, tangents_242, tangents_243, tangents_244, tangents_245, tangents_246, tangents_247, tangents_248, tangents_249, tangents_250, tangents_251, tangents_252, tangents_253, tangents_254, tangents_255, tangents_256 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((128, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(tangents_256, permute_1, out=buf0)
        del permute_1
        buf1 = empty_strided((1000, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        extern_kernels.mm(as_strided(tangents_256, (1000, 128), (1, 1000)), view, out=buf1)
        del view
        buf2 = empty_strided((1, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        buf4 = as_strided(buf2, (1000, ), (1, )); del buf2  # reuse
        stream0 = get_cuda_stream(0)
        triton__0.run(buf4, tangents_256, 1000, 128, grid=grid(1000), stream=stream0)
        del tangents_256
        buf3 = empty_strided((1000, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf1, buf3, 2048000, grid=grid(2048000), stream=stream0)
        del buf1
        buf5 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        triton__2.run(le, buf0, convolution_84, unsqueeze_342, squeeze_761, buf5, buf6, buf7, 2048, 6272, grid=grid(2048), stream=stream0)
        buf8 = empty_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__3.run(le, buf0, convolution_84, unsqueeze_342, buf6, squeeze_761, buf5, primals_254, buf8, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_84
        del primals_254
        del squeeze_761
        del unsqueeze_342
        buf9 = aten.convolution_backward(buf8, cat_15, convert_element_type_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_15
        del convert_element_type_253
        buf10 = buf9[0]
        assert_size_stride(buf10, (128, 1024, 7, 7), (50176, 49, 7, 1))
        buf11 = buf9[1]
        assert_size_stride(buf11, (2048, 1024, 1, 1), (1024, 1, 1, 1))
        del buf9
        buf12 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf11, buf12, 2097152, grid=grid(2097152), stream=stream0)
        del buf11
        buf13 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__5.run(le_1, buf10, convolution_83, unsqueeze_354, squeeze_752, buf13, buf14, buf15, 256, 6272, grid=grid(256), stream=stream0)
        buf16 = empty_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__6.run(le_1, buf10, convolution_83, unsqueeze_354, buf14, squeeze_752, buf13, primals_251, buf16, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_83
        del le_1
        del primals_251
        del squeeze_752
        del unsqueeze_354
        buf17 = aten.convolution_backward(buf16, add_453, convert_element_type_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_453
        del convert_element_type_250
        buf18 = buf17[0]
        assert_size_stride(buf18, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf19 = buf17[1]
        assert_size_stride(buf19, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf17
        buf20 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf19, buf20, 73728, grid=grid(73728), stream=stream0)
        del buf19
        buf21 = buf14; del buf14  # reuse
        buf22 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf24 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__8.run(le_2, buf10, buf18, convolution_82, unsqueeze_366, squeeze_743, buf21, buf22, buf24, 256, 6272, grid=grid(256), stream=stream0)
        buf25 = buf16; del buf16  # reuse
        buf43 = empty_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf41 = as_strided(buf43, (128, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        triton__9.run(le_2, buf10, buf18, convolution_82, unsqueeze_366, buf22, squeeze_743, buf21, primals_248, buf25, buf41, 1605632, grid=grid(1605632), stream=stream0)
        del buf18
        del convolution_82
        del le_2
        del primals_248
        del squeeze_743
        del unsqueeze_366
        buf26 = aten.convolution_backward(buf25, add_447, convert_element_type_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_447
        del convert_element_type_247
        buf27 = buf26[0]
        assert_size_stride(buf27, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf28 = buf26[1]
        assert_size_stride(buf28, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf26
        buf29 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf28, buf29, 73728, grid=grid(73728), stream=stream0)
        del buf28
        buf30 = buf22; del buf22  # reuse
        buf31 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__10.run(le_3, buf10, buf27, convolution_81, unsqueeze_378, squeeze_734, buf30, buf31, buf33, 256, 6272, grid=grid(256), stream=stream0)
        buf34 = buf25; del buf25  # reuse
        buf40 = as_strided(buf43, (128, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        triton__11.run(le_3, buf10, buf27, convolution_81, unsqueeze_378, buf31, squeeze_734, buf30, primals_245, buf34, buf40, 1605632, grid=grid(1605632), stream=stream0)
        del buf27
        del convolution_81
        del le_3
        del primals_245
        del squeeze_734
        del unsqueeze_378
        buf35 = aten.convolution_backward(buf34, getitem_306, convert_element_type_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf34
        del convert_element_type_244
        del getitem_306
        buf36 = buf35[0]
        assert_size_stride(buf36, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf37 = buf35[1]
        assert_size_stride(buf37, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf35
        buf38 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf37, buf38, 73728, grid=grid(73728), stream=stream0)
        del buf37
        buf39 = as_strided(buf43, (128, 256, 7, 7), (50176, 49, 7, 1))  # alias
        triton__12.run(buf36, buf39, 1605632, grid=grid(1605632), stream=stream0)
        buf42 = as_strided(buf43, (128, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        triton__13.run(buf10, buf42, 1605632, grid=grid(1605632), stream=stream0)
        del buf10
        buf44 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__14.run(le_4, buf43, convolution_80, unsqueeze_390, squeeze_725, buf44, buf45, buf46, 1024, 6272, grid=grid(1024), stream=stream0)
        del buf39
        del buf40
        del buf41
        del buf42
        buf47 = buf43; del buf43  # reuse
        triton__15.run(buf47, le_4, convolution_80, unsqueeze_390, buf45, squeeze_725, buf44, primals_242, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_80
        del le_4
        del primals_242
        del squeeze_725
        del unsqueeze_390
        buf48 = aten.convolution_backward(buf47, relu_75, convert_element_type_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_241
        buf49 = buf48[0]
        assert_size_stride(buf49, (128, 2048, 7, 7), (100352, 49, 7, 1))
        buf50 = buf48[1]
        assert_size_stride(buf50, (1024, 2048, 1, 1), (2048, 1, 1, 1))
        del buf48
        buf51 = empty_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf50, buf51, 2097152, grid=grid(2097152), stream=stream0)
        del buf50
        buf52 = buf6; del buf6  # reuse
        buf53 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf55 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        triton__16.run(relu_75, le, buf0, buf49, convolution_79, unsqueeze_402, squeeze_716, buf52, buf53, buf55, 2048, 6272, grid=grid(2048), stream=stream0)
        buf56 = buf8; del buf8  # reuse
        triton__17.run(relu_75, le, buf0, buf49, convolution_79, unsqueeze_402, buf53, squeeze_716, buf52, primals_239, buf56, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_79
        del primals_239
        del squeeze_716
        del unsqueeze_402
        buf57 = aten.convolution_backward(buf56, cat_14, convert_element_type_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf56
        del cat_14
        del convert_element_type_238
        buf58 = buf57[0]
        assert_size_stride(buf58, (128, 1024, 7, 7), (50176, 49, 7, 1))
        buf59 = buf57[1]
        assert_size_stride(buf59, (2048, 1024, 1, 1), (1024, 1, 1, 1))
        del buf57
        buf60 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf59, buf60, 2097152, grid=grid(2097152), stream=stream0)
        del buf59
        buf61 = buf31; del buf31  # reuse
        buf62 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf63 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__5.run(le_6, buf58, convolution_78, unsqueeze_414, squeeze_707, buf61, buf62, buf63, 256, 6272, grid=grid(256), stream=stream0)
        buf64 = buf36; del buf36  # reuse
        triton__6.run(le_6, buf58, convolution_78, unsqueeze_414, buf62, squeeze_707, buf61, primals_236, buf64, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_78
        del le_6
        del primals_236
        del squeeze_707
        del unsqueeze_414
        buf65 = aten.convolution_backward(buf64, add_425, convert_element_type_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_425
        del convert_element_type_235
        buf66 = buf65[0]
        assert_size_stride(buf66, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf67 = buf65[1]
        assert_size_stride(buf67, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf65
        buf68 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf67, buf68, 73728, grid=grid(73728), stream=stream0)
        del buf67
        buf69 = buf62; del buf62  # reuse
        buf70 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__8.run(le_7, buf58, buf66, convolution_77, unsqueeze_426, squeeze_698, buf69, buf70, buf72, 256, 6272, grid=grid(256), stream=stream0)
        buf73 = buf64; del buf64  # reuse
        buf91 = buf47; del buf47  # reuse
        buf89 = as_strided(buf91, (128, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        triton__9.run(le_7, buf58, buf66, convolution_77, unsqueeze_426, buf70, squeeze_698, buf69, primals_233, buf73, buf89, 1605632, grid=grid(1605632), stream=stream0)
        del buf66
        del convolution_77
        del le_7
        del primals_233
        del squeeze_698
        del unsqueeze_426
        buf74 = aten.convolution_backward(buf73, add_419, convert_element_type_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_419
        del convert_element_type_232
        buf75 = buf74[0]
        assert_size_stride(buf75, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf76 = buf74[1]
        assert_size_stride(buf76, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf74
        buf77 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf76, buf77, 73728, grid=grid(73728), stream=stream0)
        del buf76
        buf78 = buf70; del buf70  # reuse
        buf79 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf81 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__10.run(le_8, buf58, buf75, convolution_76, unsqueeze_438, squeeze_689, buf78, buf79, buf81, 256, 6272, grid=grid(256), stream=stream0)
        buf82 = buf73; del buf73  # reuse
        buf88 = as_strided(buf91, (128, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        triton__11.run(le_8, buf58, buf75, convolution_76, unsqueeze_438, buf79, squeeze_689, buf78, primals_230, buf82, buf88, 1605632, grid=grid(1605632), stream=stream0)
        del buf75
        del convolution_76
        del le_8
        del primals_230
        del squeeze_689
        del unsqueeze_438
        buf83 = aten.convolution_backward(buf82, getitem_286, convert_element_type_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf82
        del convert_element_type_229
        del getitem_286
        buf84 = buf83[0]
        assert_size_stride(buf84, (128, 256, 7, 7), (12544, 49, 7, 1))
        buf85 = buf83[1]
        assert_size_stride(buf85, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf83
        buf86 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf85, buf86, 73728, grid=grid(73728), stream=stream0)
        del buf85
        buf87 = as_strided(buf91, (128, 256, 7, 7), (50176, 49, 7, 1))  # alias
        triton__12.run(buf84, buf87, 1605632, grid=grid(1605632), stream=stream0)
        buf90 = as_strided(buf91, (128, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        triton__13.run(buf58, buf90, 1605632, grid=grid(1605632), stream=stream0)
        del buf58
        buf92 = buf45; del buf45  # reuse
        buf93 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__14.run(le_9, buf91, convolution_75, unsqueeze_450, squeeze_680, buf92, buf93, buf94, 1024, 6272, grid=grid(1024), stream=stream0)
        del buf87
        del buf88
        del buf89
        del buf90
        buf95 = buf91; del buf91  # reuse
        triton__15.run(buf95, le_9, convolution_75, unsqueeze_450, buf93, squeeze_680, buf92, primals_227, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_75
        del le_9
        del primals_227
        del squeeze_680
        del unsqueeze_450
        buf96 = aten.convolution_backward(buf95, relu_70, convert_element_type_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf95
        del convert_element_type_226
        buf97 = buf96[0]
        assert_size_stride(buf97, (128, 2048, 7, 7), (100352, 49, 7, 1))
        buf98 = buf96[1]
        assert_size_stride(buf98, (1024, 2048, 1, 1), (2048, 1, 1, 1))
        del buf96
        buf99 = empty_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf98, buf99, 2097152, grid=grid(2097152), stream=stream0)
        del buf98
        buf100 = empty_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda', dtype=torch.float32)
        triton__18.run(relu_70, relu_75, le, buf0, buf49, buf97, buf100, 12845056, grid=grid(12845056), stream=stream0)
        del buf0
        del le
        del relu_70
        del relu_75
        buf101 = buf53; del buf53  # reuse
        buf102 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf103 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf110 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        triton__19.run(buf100, convolution_74, unsqueeze_462, convolution_73, unsqueeze_474, squeeze_671, squeeze_662, buf101, buf102, buf109, buf103, buf110, 2048, 6272, grid=grid(2048), stream=stream0)
        buf104 = buf97; del buf97  # reuse
        buf111 = buf49; del buf49  # reuse
        triton__20.run(buf100, convolution_74, unsqueeze_462, buf102, squeeze_671, buf101, primals_224, convolution_73, unsqueeze_474, buf109, squeeze_662, primals_221, buf104, buf111, 12845056, grid=grid(12845056), stream=stream0)
        del buf100
        del buf102
        del buf109
        del convolution_73
        del convolution_74
        del primals_221
        del primals_224
        del squeeze_662
        del squeeze_671
        del unsqueeze_462
        del unsqueeze_474
        buf105 = aten.convolution_backward(buf104, relu_65, convert_element_type_223, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf104
        del convert_element_type_223
        buf106 = buf105[0]
        assert_size_stride(buf106, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf107 = buf105[1]
        assert_size_stride(buf107, (2048, 1024, 1, 1), (1024, 1, 1, 1))
        del buf105
        buf108 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf107, buf108, 2097152, grid=grid(2097152), stream=stream0)
        del buf107
        buf112 = aten.convolution_backward(buf111, cat_13, convert_element_type_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_13
        del convert_element_type_220
        buf113 = buf112[0]
        assert_size_stride(buf113, (128, 1024, 7, 7), (50176, 49, 7, 1))
        buf114 = buf112[1]
        assert_size_stride(buf114, (2048, 1024, 1, 1), (1024, 1, 1, 1))
        del buf112
        buf115 = empty_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf114, buf115, 2097152, grid=grid(2097152), stream=stream0)
        del buf114
        buf144 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf116 = as_strided(buf144, (128, 256, 14, 14), (200704, 196, 14, 1), 150528)  # alias
        triton__21.run(buf113, buf116, 6422528, grid=grid(6422528), stream=stream0)
        buf117 = buf79; del buf79  # reuse
        buf118 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__5.run(le_11, buf113, convolution_72, unsqueeze_486, squeeze_653, buf117, buf118, buf119, 256, 6272, grid=grid(256), stream=stream0)
        buf120 = buf84; del buf84  # reuse
        triton__6.run(le_11, buf113, convolution_72, unsqueeze_486, buf118, squeeze_653, buf117, primals_218, buf120, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_72
        del le_11
        del primals_218
        del squeeze_653
        del unsqueeze_486
        buf121 = aten.convolution_backward(buf120, getitem_276, convert_element_type_217, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_217
        del getitem_276
        buf122 = buf121[0]
        assert_size_stride(buf122, (128, 256, 14, 14), (50176, 196, 14, 1))
        buf123 = buf121[1]
        assert_size_stride(buf123, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf121
        buf124 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf123, buf124, 73728, grid=grid(73728), stream=stream0)
        del buf123
        buf125 = buf118; del buf118  # reuse
        buf126 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf127 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__22.run(le_12, buf113, convolution_71, unsqueeze_498, squeeze_644, buf125, buf126, buf127, 256, 6272, grid=grid(256), stream=stream0)
        buf128 = buf120; del buf120  # reuse
        triton__23.run(le_12, buf113, convolution_71, unsqueeze_498, buf126, squeeze_644, buf125, primals_215, buf128, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_71
        del le_12
        del primals_215
        del squeeze_644
        del unsqueeze_498
        buf129 = aten.convolution_backward(buf128, getitem_271, convert_element_type_214, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_214
        del getitem_271
        buf130 = buf129[0]
        assert_size_stride(buf130, (128, 256, 14, 14), (50176, 196, 14, 1))
        buf131 = buf129[1]
        assert_size_stride(buf131, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf129
        buf132 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf131, buf132, 73728, grid=grid(73728), stream=stream0)
        del buf131
        buf133 = buf126; del buf126  # reuse
        buf134 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf135 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__24.run(le_13, buf113, convolution_70, unsqueeze_510, squeeze_635, buf133, buf134, buf135, 256, 6272, grid=grid(256), stream=stream0)
        buf136 = buf128; del buf128  # reuse
        triton__25.run(le_13, buf113, convolution_70, unsqueeze_510, buf134, squeeze_635, buf133, primals_212, buf136, 1605632, grid=grid(1605632), stream=stream0)
        del buf113
        del convolution_70
        del le_13
        del primals_212
        del squeeze_635
        del unsqueeze_510
        buf137 = aten.convolution_backward(buf136, getitem_266, convert_element_type_211, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf136
        del convert_element_type_211
        del getitem_266
        buf138 = buf137[0]
        assert_size_stride(buf138, (128, 256, 14, 14), (50176, 196, 14, 1))
        buf139 = buf137[1]
        assert_size_stride(buf139, (256, 32, 3, 3), (288, 9, 3, 1))
        del buf137
        buf140 = empty_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__7.run(buf139, buf140, 73728, grid=grid(73728), stream=stream0)
        del buf139
        buf141 = as_strided(buf144, (128, 256, 14, 14), (200704, 196, 14, 1))  # alias
        triton__26.run(buf138, buf141, 6422528, grid=grid(6422528), stream=stream0)
        del buf138
        buf142 = as_strided(buf144, (128, 256, 14, 14), (200704, 196, 14, 1), 50176)  # alias
        triton__26.run(buf130, buf142, 6422528, grid=grid(6422528), stream=stream0)
        del buf130
        buf143 = as_strided(buf144, (128, 256, 14, 14), (200704, 196, 14, 1), 100352)  # alias
        triton__26.run(buf122, buf143, 6422528, grid=grid(6422528), stream=stream0)
        buf145 = buf93; del buf93  # reuse
        buf146 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf147 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__27.run(le_14, buf144, convolution_69, unsqueeze_522, squeeze_626, buf145, buf146, buf147, 1024, 25088, grid=grid(1024), stream=stream0)
        del buf116
        del buf141
        del buf142
        del buf143
        buf148 = buf144; del buf144  # reuse
        triton__28.run(buf148, le_14, convolution_69, unsqueeze_522, buf146, squeeze_626, buf145, primals_209, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_69
        del le_14
        del primals_209
        del squeeze_626
        del unsqueeze_522
        buf149 = aten.convolution_backward(buf148, relu_65, convert_element_type_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_208
        buf150 = buf149[0]
        assert_size_stride(buf150, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf151 = buf149[1]
        assert_size_stride(buf151, (1024, 1024, 1, 1), (1024, 1, 1, 1))
        del buf149
        buf152 = empty_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__29.run(buf151, buf152, 1048576, grid=grid(1048576), stream=stream0)
        del buf151
        buf153 = buf146; del buf146  # reuse
        buf154 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf156 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__30.run(relu_65, buf106, buf150, convolution_68, unsqueeze_534, squeeze_617, buf153, buf154, buf156, 1024, 25088, grid=grid(1024), stream=stream0)
        buf157 = buf148; del buf148  # reuse
        triton__31.run(relu_65, buf106, buf150, convolution_68, unsqueeze_534, buf154, squeeze_617, buf153, primals_206, buf157, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_68
        del primals_206
        del squeeze_617
        del unsqueeze_534
        buf158 = aten.convolution_backward(buf157, cat_12, convert_element_type_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf157
        del cat_12
        del convert_element_type_205
        buf159 = buf158[0]
        assert_size_stride(buf159, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf160 = buf158[1]
        assert_size_stride(buf160, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf158
        buf161 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf160, buf161, 524288, grid=grid(524288), stream=stream0)
        del buf160
        buf162 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        triton__33.run(le_16, buf159, convolution_67, unsqueeze_546, buf162, buf164, 512, 6272, grid=grid(512), stream=stream0)
        buf163 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__34.run(buf162, buf163, 128, 4, grid=grid(128), stream=stream0)
        buf165 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf166 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf164, squeeze_608, buf165, buf166, 128, 4, grid=grid(128), stream=stream0)
        buf167 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__36.run(le_16, buf159, convolution_67, unsqueeze_546, buf165, squeeze_608, buf163, primals_203, buf167, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_67
        del le_16
        del primals_203
        del squeeze_608
        del unsqueeze_546
        buf168 = aten.convolution_backward(buf167, add_366, convert_element_type_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_366
        del convert_element_type_202
        buf169 = buf168[0]
        assert_size_stride(buf169, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf170 = buf168[1]
        assert_size_stride(buf170, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf168
        buf171 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf170, buf171, 18432, grid=grid(18432), stream=stream0)
        del buf170
        buf172 = buf164; del buf164  # reuse
        buf174 = buf162; del buf162  # reuse
        triton__38.run(le_17, buf159, buf169, convolution_66, unsqueeze_558, buf172, buf174, 512, 6272, grid=grid(512), stream=stream0)
        buf173 = buf165; del buf165  # reuse
        triton__34.run(buf172, buf173, 128, 4, grid=grid(128), stream=stream0)
        buf175 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf177 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf174, squeeze_599, buf175, buf177, 128, 4, grid=grid(128), stream=stream0)
        buf178 = buf167; del buf167  # reuse
        buf198 = as_strided(buf111, (128, 512, 14, 14), (100352, 196, 14, 1)); del buf111  # reuse
        buf196 = as_strided(buf198, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        triton__39.run(le_17, buf159, buf169, convolution_66, unsqueeze_558, buf175, squeeze_599, buf173, primals_200, buf178, buf196, 3211264, grid=grid(3211264), stream=stream0)
        del buf169
        del convolution_66
        del le_17
        del primals_200
        del squeeze_599
        del unsqueeze_558
        buf179 = aten.convolution_backward(buf178, add_360, convert_element_type_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_360
        del convert_element_type_199
        buf180 = buf179[0]
        assert_size_stride(buf180, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf181 = buf179[1]
        assert_size_stride(buf181, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf179
        buf182 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf181, buf182, 18432, grid=grid(18432), stream=stream0)
        del buf181
        buf183 = buf174; del buf174  # reuse
        buf185 = buf172; del buf172  # reuse
        triton__40.run(le_18, buf159, buf180, convolution_65, unsqueeze_570, buf183, buf185, 512, 6272, grid=grid(512), stream=stream0)
        buf184 = buf175; del buf175  # reuse
        triton__34.run(buf183, buf184, 128, 4, grid=grid(128), stream=stream0)
        buf186 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf185, squeeze_590, buf186, buf188, 128, 4, grid=grid(128), stream=stream0)
        buf189 = buf178; del buf178  # reuse
        buf195 = as_strided(buf198, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        triton__41.run(le_18, buf159, buf180, convolution_65, unsqueeze_570, buf186, squeeze_590, buf184, primals_197, buf189, buf195, 3211264, grid=grid(3211264), stream=stream0)
        del buf180
        del convolution_65
        del le_18
        del primals_197
        del squeeze_590
        del unsqueeze_570
        buf190 = aten.convolution_backward(buf189, getitem_246, convert_element_type_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf189
        del convert_element_type_196
        del getitem_246
        buf191 = buf190[0]
        assert_size_stride(buf191, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf192 = buf190[1]
        assert_size_stride(buf192, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf190
        buf193 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf192, buf193, 18432, grid=grid(18432), stream=stream0)
        del buf192
        buf194 = as_strided(buf198, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        triton__42.run(buf191, buf194, 3211264, grid=grid(3211264), stream=stream0)
        buf197 = as_strided(buf198, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__43.run(buf159, buf197, 3211264, grid=grid(3211264), stream=stream0)
        del buf159
        buf199 = as_strided(buf185, (512, ), (1, )); del buf185  # reuse
        buf200 = as_strided(buf183, (512, ), (1, )); del buf183  # reuse
        buf201 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__44.run(le_19, buf198, convolution_64, unsqueeze_582, squeeze_581, buf199, buf200, buf201, 512, 25088, grid=grid(512), stream=stream0)
        del buf194
        del buf195
        del buf196
        del buf197
        buf202 = buf198; del buf198  # reuse
        triton__45.run(buf202, le_19, convolution_64, unsqueeze_582, buf200, squeeze_581, buf199, primals_194, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_64
        del le_19
        del primals_194
        del squeeze_581
        del unsqueeze_582
        buf203 = aten.convolution_backward(buf202, relu_60, convert_element_type_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_193
        buf204 = buf203[0]
        assert_size_stride(buf204, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf205 = buf203[1]
        assert_size_stride(buf205, (512, 1024, 1, 1), (1024, 1, 1, 1))
        del buf203
        buf206 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf205, buf206, 524288, grid=grid(524288), stream=stream0)
        del buf205
        buf207 = buf106; del buf106  # reuse
        triton__46.run(buf207, relu_60, relu_65, buf150, buf204, 25690112, grid=grid(25690112), stream=stream0)
        del buf150
        del relu_60
        del relu_65
        buf208 = buf154; del buf154  # reuse
        buf209 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf210 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__47.run(buf207, convolution_63, unsqueeze_594, squeeze_572, buf208, buf209, buf210, 1024, 25088, grid=grid(1024), stream=stream0)
        buf211 = buf204; del buf204  # reuse
        triton__48.run(buf207, convolution_63, unsqueeze_594, buf209, squeeze_572, buf208, primals_191, buf211, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_63
        del primals_191
        del squeeze_572
        del unsqueeze_594
        buf212 = aten.convolution_backward(buf211, cat_11, convert_element_type_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_11
        del convert_element_type_190
        buf213 = buf212[0]
        assert_size_stride(buf213, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf214 = buf212[1]
        assert_size_stride(buf214, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf212
        buf215 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf214, buf215, 524288, grid=grid(524288), stream=stream0)
        del buf214
        buf216 = as_strided(buf200, (128, 4), (1, 128)); del buf200  # reuse
        buf218 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        triton__33.run(le_21, buf213, convolution_62, unsqueeze_606, buf216, buf218, 512, 6272, grid=grid(512), stream=stream0)
        buf217 = buf186; del buf186  # reuse
        triton__34.run(buf216, buf217, 128, 4, grid=grid(128), stream=stream0)
        buf219 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf218, squeeze_563, buf219, buf220, 128, 4, grid=grid(128), stream=stream0)
        buf221 = buf191; del buf191  # reuse
        triton__36.run(le_21, buf213, convolution_62, unsqueeze_606, buf219, squeeze_563, buf217, primals_188, buf221, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_62
        del le_21
        del primals_188
        del squeeze_563
        del unsqueeze_606
        buf222 = aten.convolution_backward(buf221, add_338, convert_element_type_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_338
        del convert_element_type_187
        buf223 = buf222[0]
        assert_size_stride(buf223, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf224 = buf222[1]
        assert_size_stride(buf224, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf222
        buf225 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf224, buf225, 18432, grid=grid(18432), stream=stream0)
        del buf224
        buf226 = buf218; del buf218  # reuse
        buf228 = buf216; del buf216  # reuse
        triton__38.run(le_22, buf213, buf223, convolution_61, unsqueeze_618, buf226, buf228, 512, 6272, grid=grid(512), stream=stream0)
        buf227 = buf219; del buf219  # reuse
        triton__34.run(buf226, buf227, 128, 4, grid=grid(128), stream=stream0)
        buf229 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf231 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf228, squeeze_554, buf229, buf231, 128, 4, grid=grid(128), stream=stream0)
        buf232 = buf221; del buf221  # reuse
        buf252 = buf202; del buf202  # reuse
        buf250 = as_strided(buf252, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        triton__39.run(le_22, buf213, buf223, convolution_61, unsqueeze_618, buf229, squeeze_554, buf227, primals_185, buf232, buf250, 3211264, grid=grid(3211264), stream=stream0)
        del buf223
        del convolution_61
        del le_22
        del primals_185
        del squeeze_554
        del unsqueeze_618
        buf233 = aten.convolution_backward(buf232, add_332, convert_element_type_184, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_332
        del convert_element_type_184
        buf234 = buf233[0]
        assert_size_stride(buf234, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf235 = buf233[1]
        assert_size_stride(buf235, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf233
        buf236 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf235, buf236, 18432, grid=grid(18432), stream=stream0)
        del buf235
        buf237 = buf228; del buf228  # reuse
        buf239 = buf226; del buf226  # reuse
        triton__40.run(le_23, buf213, buf234, convolution_60, unsqueeze_630, buf237, buf239, 512, 6272, grid=grid(512), stream=stream0)
        buf238 = buf229; del buf229  # reuse
        triton__34.run(buf237, buf238, 128, 4, grid=grid(128), stream=stream0)
        buf240 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf242 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf239, squeeze_545, buf240, buf242, 128, 4, grid=grid(128), stream=stream0)
        buf243 = buf232; del buf232  # reuse
        buf249 = as_strided(buf252, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        triton__41.run(le_23, buf213, buf234, convolution_60, unsqueeze_630, buf240, squeeze_545, buf238, primals_182, buf243, buf249, 3211264, grid=grid(3211264), stream=stream0)
        del buf234
        del convolution_60
        del le_23
        del primals_182
        del squeeze_545
        del unsqueeze_630
        buf244 = aten.convolution_backward(buf243, getitem_226, convert_element_type_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf243
        del convert_element_type_181
        del getitem_226
        buf245 = buf244[0]
        assert_size_stride(buf245, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf246 = buf244[1]
        assert_size_stride(buf246, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf244
        buf247 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf246, buf247, 18432, grid=grid(18432), stream=stream0)
        del buf246
        buf248 = as_strided(buf252, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        triton__42.run(buf245, buf248, 3211264, grid=grid(3211264), stream=stream0)
        buf251 = as_strided(buf252, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__43.run(buf213, buf251, 3211264, grid=grid(3211264), stream=stream0)
        del buf213
        buf253 = as_strided(buf239, (512, ), (1, )); del buf239  # reuse
        buf254 = as_strided(buf237, (512, ), (1, )); del buf237  # reuse
        buf255 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__44.run(le_24, buf252, convolution_59, unsqueeze_642, squeeze_536, buf253, buf254, buf255, 512, 25088, grid=grid(512), stream=stream0)
        del buf248
        del buf249
        del buf250
        del buf251
        buf256 = buf252; del buf252  # reuse
        triton__45.run(buf256, le_24, convolution_59, unsqueeze_642, buf254, squeeze_536, buf253, primals_179, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_59
        del le_24
        del primals_179
        del squeeze_536
        del unsqueeze_642
        buf257 = aten.convolution_backward(buf256, relu_55, convert_element_type_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_178
        buf258 = buf257[0]
        assert_size_stride(buf258, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf259 = buf257[1]
        assert_size_stride(buf259, (512, 1024, 1, 1), (1024, 1, 1, 1))
        del buf257
        buf260 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf259, buf260, 524288, grid=grid(524288), stream=stream0)
        del buf259
        buf261 = buf209; del buf209  # reuse
        buf262 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__30.run(relu_55, buf207, buf258, convolution_58, unsqueeze_654, squeeze_527, buf261, buf262, buf264, 1024, 25088, grid=grid(1024), stream=stream0)
        buf265 = buf211; del buf211  # reuse
        triton__31.run(relu_55, buf207, buf258, convolution_58, unsqueeze_654, buf262, squeeze_527, buf261, primals_176, buf265, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_58
        del primals_176
        del squeeze_527
        del unsqueeze_654
        buf266 = aten.convolution_backward(buf265, cat_10, convert_element_type_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf265
        del cat_10
        del convert_element_type_175
        buf267 = buf266[0]
        assert_size_stride(buf267, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf268 = buf266[1]
        assert_size_stride(buf268, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf266
        buf269 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf268, buf269, 524288, grid=grid(524288), stream=stream0)
        del buf268
        buf270 = as_strided(buf254, (128, 4), (1, 128)); del buf254  # reuse
        buf272 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        triton__33.run(le_26, buf267, convolution_57, unsqueeze_666, buf270, buf272, 512, 6272, grid=grid(512), stream=stream0)
        buf271 = buf240; del buf240  # reuse
        triton__34.run(buf270, buf271, 128, 4, grid=grid(128), stream=stream0)
        buf273 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf274 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf272, squeeze_518, buf273, buf274, 128, 4, grid=grid(128), stream=stream0)
        buf275 = buf245; del buf245  # reuse
        triton__36.run(le_26, buf267, convolution_57, unsqueeze_666, buf273, squeeze_518, buf271, primals_173, buf275, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_57
        del le_26
        del primals_173
        del squeeze_518
        del unsqueeze_666
        buf276 = aten.convolution_backward(buf275, add_310, convert_element_type_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_310
        del convert_element_type_172
        buf277 = buf276[0]
        assert_size_stride(buf277, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf278 = buf276[1]
        assert_size_stride(buf278, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf276
        buf279 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf278, buf279, 18432, grid=grid(18432), stream=stream0)
        del buf278
        buf280 = buf272; del buf272  # reuse
        buf282 = buf270; del buf270  # reuse
        triton__38.run(le_27, buf267, buf277, convolution_56, unsqueeze_678, buf280, buf282, 512, 6272, grid=grid(512), stream=stream0)
        buf281 = buf273; del buf273  # reuse
        triton__34.run(buf280, buf281, 128, 4, grid=grid(128), stream=stream0)
        buf283 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf285 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf282, squeeze_509, buf283, buf285, 128, 4, grid=grid(128), stream=stream0)
        buf286 = buf275; del buf275  # reuse
        buf306 = buf256; del buf256  # reuse
        buf304 = as_strided(buf306, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        triton__39.run(le_27, buf267, buf277, convolution_56, unsqueeze_678, buf283, squeeze_509, buf281, primals_170, buf286, buf304, 3211264, grid=grid(3211264), stream=stream0)
        del buf277
        del convolution_56
        del le_27
        del primals_170
        del squeeze_509
        del unsqueeze_678
        buf287 = aten.convolution_backward(buf286, add_304, convert_element_type_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_304
        del convert_element_type_169
        buf288 = buf287[0]
        assert_size_stride(buf288, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf289 = buf287[1]
        assert_size_stride(buf289, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf287
        buf290 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf289, buf290, 18432, grid=grid(18432), stream=stream0)
        del buf289
        buf291 = buf282; del buf282  # reuse
        buf293 = buf280; del buf280  # reuse
        triton__40.run(le_28, buf267, buf288, convolution_55, unsqueeze_690, buf291, buf293, 512, 6272, grid=grid(512), stream=stream0)
        buf292 = buf283; del buf283  # reuse
        triton__34.run(buf291, buf292, 128, 4, grid=grid(128), stream=stream0)
        buf294 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf296 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf293, squeeze_500, buf294, buf296, 128, 4, grid=grid(128), stream=stream0)
        buf297 = buf286; del buf286  # reuse
        buf303 = as_strided(buf306, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        triton__41.run(le_28, buf267, buf288, convolution_55, unsqueeze_690, buf294, squeeze_500, buf292, primals_167, buf297, buf303, 3211264, grid=grid(3211264), stream=stream0)
        del buf288
        del convolution_55
        del le_28
        del primals_167
        del squeeze_500
        del unsqueeze_690
        buf298 = aten.convolution_backward(buf297, getitem_206, convert_element_type_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf297
        del convert_element_type_166
        del getitem_206
        buf299 = buf298[0]
        assert_size_stride(buf299, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf300 = buf298[1]
        assert_size_stride(buf300, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf298
        buf301 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf300, buf301, 18432, grid=grid(18432), stream=stream0)
        del buf300
        buf302 = as_strided(buf306, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        triton__42.run(buf299, buf302, 3211264, grid=grid(3211264), stream=stream0)
        buf305 = as_strided(buf306, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__43.run(buf267, buf305, 3211264, grid=grid(3211264), stream=stream0)
        del buf267
        buf307 = as_strided(buf293, (512, ), (1, )); del buf293  # reuse
        buf308 = as_strided(buf291, (512, ), (1, )); del buf291  # reuse
        buf309 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__44.run(le_29, buf306, convolution_54, unsqueeze_702, squeeze_491, buf307, buf308, buf309, 512, 25088, grid=grid(512), stream=stream0)
        del buf302
        del buf303
        del buf304
        del buf305
        buf310 = buf306; del buf306  # reuse
        triton__45.run(buf310, le_29, convolution_54, unsqueeze_702, buf308, squeeze_491, buf307, primals_164, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_54
        del le_29
        del primals_164
        del squeeze_491
        del unsqueeze_702
        buf311 = aten.convolution_backward(buf310, relu_50, convert_element_type_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_163
        buf312 = buf311[0]
        assert_size_stride(buf312, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf313 = buf311[1]
        assert_size_stride(buf313, (512, 1024, 1, 1), (1024, 1, 1, 1))
        del buf311
        buf314 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf313, buf314, 524288, grid=grid(524288), stream=stream0)
        del buf313
        buf315 = buf207; del buf207  # reuse
        triton__46.run(buf315, relu_50, relu_55, buf258, buf312, 25690112, grid=grid(25690112), stream=stream0)
        del buf258
        del relu_50
        del relu_55
        buf316 = buf262; del buf262  # reuse
        buf317 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf318 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__47.run(buf315, convolution_53, unsqueeze_714, squeeze_482, buf316, buf317, buf318, 1024, 25088, grid=grid(1024), stream=stream0)
        buf319 = buf312; del buf312  # reuse
        triton__48.run(buf315, convolution_53, unsqueeze_714, buf317, squeeze_482, buf316, primals_161, buf319, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_53
        del primals_161
        del squeeze_482
        del unsqueeze_714
        buf320 = aten.convolution_backward(buf319, cat_9, convert_element_type_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_9
        del convert_element_type_160
        buf321 = buf320[0]
        assert_size_stride(buf321, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf322 = buf320[1]
        assert_size_stride(buf322, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf320
        buf323 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf322, buf323, 524288, grid=grid(524288), stream=stream0)
        del buf322
        buf324 = as_strided(buf308, (128, 4), (1, 128)); del buf308  # reuse
        buf326 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        triton__33.run(le_31, buf321, convolution_52, unsqueeze_726, buf324, buf326, 512, 6272, grid=grid(512), stream=stream0)
        buf325 = buf294; del buf294  # reuse
        triton__34.run(buf324, buf325, 128, 4, grid=grid(128), stream=stream0)
        buf327 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf328 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf326, squeeze_473, buf327, buf328, 128, 4, grid=grid(128), stream=stream0)
        buf329 = buf299; del buf299  # reuse
        triton__36.run(le_31, buf321, convolution_52, unsqueeze_726, buf327, squeeze_473, buf325, primals_158, buf329, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_52
        del le_31
        del primals_158
        del squeeze_473
        del unsqueeze_726
        buf330 = aten.convolution_backward(buf329, add_282, convert_element_type_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_282
        del convert_element_type_157
        buf331 = buf330[0]
        assert_size_stride(buf331, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf332 = buf330[1]
        assert_size_stride(buf332, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf330
        buf333 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf332, buf333, 18432, grid=grid(18432), stream=stream0)
        del buf332
        buf334 = buf326; del buf326  # reuse
        buf336 = buf324; del buf324  # reuse
        triton__38.run(le_32, buf321, buf331, convolution_51, unsqueeze_738, buf334, buf336, 512, 6272, grid=grid(512), stream=stream0)
        buf335 = buf327; del buf327  # reuse
        triton__34.run(buf334, buf335, 128, 4, grid=grid(128), stream=stream0)
        buf337 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf339 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf336, squeeze_464, buf337, buf339, 128, 4, grid=grid(128), stream=stream0)
        buf340 = buf329; del buf329  # reuse
        buf360 = buf310; del buf310  # reuse
        buf358 = as_strided(buf360, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        triton__39.run(le_32, buf321, buf331, convolution_51, unsqueeze_738, buf337, squeeze_464, buf335, primals_155, buf340, buf358, 3211264, grid=grid(3211264), stream=stream0)
        del buf331
        del convolution_51
        del le_32
        del primals_155
        del squeeze_464
        del unsqueeze_738
        buf341 = aten.convolution_backward(buf340, add_276, convert_element_type_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_276
        del convert_element_type_154
        buf342 = buf341[0]
        assert_size_stride(buf342, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf343 = buf341[1]
        assert_size_stride(buf343, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf341
        buf344 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf343, buf344, 18432, grid=grid(18432), stream=stream0)
        del buf343
        buf345 = buf336; del buf336  # reuse
        buf347 = buf334; del buf334  # reuse
        triton__40.run(le_33, buf321, buf342, convolution_50, unsqueeze_750, buf345, buf347, 512, 6272, grid=grid(512), stream=stream0)
        buf346 = buf337; del buf337  # reuse
        triton__34.run(buf345, buf346, 128, 4, grid=grid(128), stream=stream0)
        buf348 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf350 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf347, squeeze_455, buf348, buf350, 128, 4, grid=grid(128), stream=stream0)
        buf351 = buf340; del buf340  # reuse
        buf357 = as_strided(buf360, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        triton__41.run(le_33, buf321, buf342, convolution_50, unsqueeze_750, buf348, squeeze_455, buf346, primals_152, buf351, buf357, 3211264, grid=grid(3211264), stream=stream0)
        del buf342
        del convolution_50
        del le_33
        del primals_152
        del squeeze_455
        del unsqueeze_750
        buf352 = aten.convolution_backward(buf351, getitem_186, convert_element_type_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf351
        del convert_element_type_151
        del getitem_186
        buf353 = buf352[0]
        assert_size_stride(buf353, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf354 = buf352[1]
        assert_size_stride(buf354, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf352
        buf355 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf354, buf355, 18432, grid=grid(18432), stream=stream0)
        del buf354
        buf356 = as_strided(buf360, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        triton__42.run(buf353, buf356, 3211264, grid=grid(3211264), stream=stream0)
        buf359 = as_strided(buf360, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__43.run(buf321, buf359, 3211264, grid=grid(3211264), stream=stream0)
        del buf321
        buf361 = as_strided(buf347, (512, ), (1, )); del buf347  # reuse
        buf362 = as_strided(buf345, (512, ), (1, )); del buf345  # reuse
        buf363 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__44.run(le_34, buf360, convolution_49, unsqueeze_762, squeeze_446, buf361, buf362, buf363, 512, 25088, grid=grid(512), stream=stream0)
        del buf356
        del buf357
        del buf358
        del buf359
        buf364 = buf360; del buf360  # reuse
        triton__45.run(buf364, le_34, convolution_49, unsqueeze_762, buf362, squeeze_446, buf361, primals_149, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_49
        del le_34
        del primals_149
        del squeeze_446
        del unsqueeze_762
        buf365 = aten.convolution_backward(buf364, relu_45, convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_148
        buf366 = buf365[0]
        assert_size_stride(buf366, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf367 = buf365[1]
        assert_size_stride(buf367, (512, 1024, 1, 1), (1024, 1, 1, 1))
        del buf365
        buf368 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf367, buf368, 524288, grid=grid(524288), stream=stream0)
        del buf367
        buf369 = buf317; del buf317  # reuse
        buf370 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf372 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__30.run(relu_45, buf315, buf366, convolution_48, unsqueeze_774, squeeze_437, buf369, buf370, buf372, 1024, 25088, grid=grid(1024), stream=stream0)
        buf373 = buf319; del buf319  # reuse
        triton__31.run(relu_45, buf315, buf366, convolution_48, unsqueeze_774, buf370, squeeze_437, buf369, primals_146, buf373, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_48
        del primals_146
        del squeeze_437
        del unsqueeze_774
        buf374 = aten.convolution_backward(buf373, cat_8, convert_element_type_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf373
        del cat_8
        del convert_element_type_145
        buf375 = buf374[0]
        assert_size_stride(buf375, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf376 = buf374[1]
        assert_size_stride(buf376, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf374
        buf377 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf376, buf377, 524288, grid=grid(524288), stream=stream0)
        del buf376
        buf378 = as_strided(buf362, (128, 4), (1, 128)); del buf362  # reuse
        buf380 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        triton__33.run(le_36, buf375, convolution_47, unsqueeze_786, buf378, buf380, 512, 6272, grid=grid(512), stream=stream0)
        buf379 = buf348; del buf348  # reuse
        triton__34.run(buf378, buf379, 128, 4, grid=grid(128), stream=stream0)
        buf381 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf382 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf380, squeeze_428, buf381, buf382, 128, 4, grid=grid(128), stream=stream0)
        buf383 = buf353; del buf353  # reuse
        triton__36.run(le_36, buf375, convolution_47, unsqueeze_786, buf381, squeeze_428, buf379, primals_143, buf383, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_47
        del le_36
        del primals_143
        del squeeze_428
        del unsqueeze_786
        buf384 = aten.convolution_backward(buf383, add_254, convert_element_type_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_254
        del convert_element_type_142
        buf385 = buf384[0]
        assert_size_stride(buf385, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf386 = buf384[1]
        assert_size_stride(buf386, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf384
        buf387 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf386, buf387, 18432, grid=grid(18432), stream=stream0)
        del buf386
        buf388 = buf380; del buf380  # reuse
        buf390 = buf378; del buf378  # reuse
        triton__38.run(le_37, buf375, buf385, convolution_46, unsqueeze_798, buf388, buf390, 512, 6272, grid=grid(512), stream=stream0)
        buf389 = buf381; del buf381  # reuse
        triton__34.run(buf388, buf389, 128, 4, grid=grid(128), stream=stream0)
        buf391 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf393 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf390, squeeze_419, buf391, buf393, 128, 4, grid=grid(128), stream=stream0)
        buf394 = buf383; del buf383  # reuse
        buf414 = buf364; del buf364  # reuse
        buf412 = as_strided(buf414, (128, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        triton__39.run(le_37, buf375, buf385, convolution_46, unsqueeze_798, buf391, squeeze_419, buf389, primals_140, buf394, buf412, 3211264, grid=grid(3211264), stream=stream0)
        del buf385
        del convolution_46
        del le_37
        del primals_140
        del squeeze_419
        del unsqueeze_798
        buf395 = aten.convolution_backward(buf394, add_248, convert_element_type_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_248
        del convert_element_type_139
        buf396 = buf395[0]
        assert_size_stride(buf396, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf397 = buf395[1]
        assert_size_stride(buf397, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf395
        buf398 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf397, buf398, 18432, grid=grid(18432), stream=stream0)
        del buf397
        buf399 = buf390; del buf390  # reuse
        buf401 = buf388; del buf388  # reuse
        triton__40.run(le_38, buf375, buf396, convolution_45, unsqueeze_810, buf399, buf401, 512, 6272, grid=grid(512), stream=stream0)
        buf400 = buf391; del buf391  # reuse
        triton__34.run(buf399, buf400, 128, 4, grid=grid(128), stream=stream0)
        buf402 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf404 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf401, squeeze_410, buf402, buf404, 128, 4, grid=grid(128), stream=stream0)
        buf405 = buf394; del buf394  # reuse
        buf411 = as_strided(buf414, (128, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        triton__41.run(le_38, buf375, buf396, convolution_45, unsqueeze_810, buf402, squeeze_410, buf400, primals_137, buf405, buf411, 3211264, grid=grid(3211264), stream=stream0)
        del buf396
        del convolution_45
        del le_38
        del primals_137
        del squeeze_410
        del unsqueeze_810
        buf406 = aten.convolution_backward(buf405, getitem_166, convert_element_type_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf405
        del convert_element_type_136
        del getitem_166
        buf407 = buf406[0]
        assert_size_stride(buf407, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf408 = buf406[1]
        assert_size_stride(buf408, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf406
        buf409 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf408, buf409, 18432, grid=grid(18432), stream=stream0)
        del buf408
        buf410 = as_strided(buf414, (128, 128, 14, 14), (100352, 196, 14, 1))  # alias
        triton__42.run(buf407, buf410, 3211264, grid=grid(3211264), stream=stream0)
        buf413 = as_strided(buf414, (128, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        triton__43.run(buf375, buf413, 3211264, grid=grid(3211264), stream=stream0)
        del buf375
        buf415 = as_strided(buf401, (512, ), (1, )); del buf401  # reuse
        buf416 = as_strided(buf399, (512, ), (1, )); del buf399  # reuse
        buf417 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__44.run(le_39, buf414, convolution_44, unsqueeze_822, squeeze_401, buf415, buf416, buf417, 512, 25088, grid=grid(512), stream=stream0)
        del buf410
        del buf411
        del buf412
        del buf413
        buf418 = buf414; del buf414  # reuse
        triton__45.run(buf418, le_39, convolution_44, unsqueeze_822, buf416, squeeze_401, buf415, primals_134, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_44
        del le_39
        del primals_134
        del squeeze_401
        del unsqueeze_822
        buf419 = aten.convolution_backward(buf418, relu_40, convert_element_type_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf418
        del convert_element_type_133
        buf420 = buf419[0]
        assert_size_stride(buf420, (128, 1024, 14, 14), (200704, 196, 14, 1))
        buf421 = buf419[1]
        assert_size_stride(buf421, (512, 1024, 1, 1), (1024, 1, 1, 1))
        del buf419
        buf422 = empty_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf421, buf422, 524288, grid=grid(524288), stream=stream0)
        del buf421
        buf423 = empty_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda', dtype=torch.float32)
        triton__49.run(relu_40, relu_45, buf315, buf366, buf420, buf423, 25690112, grid=grid(25690112), stream=stream0)
        del buf315
        del relu_40
        del relu_45
        buf424 = buf370; del buf370  # reuse
        buf425 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf432 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf426 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf433 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        triton__50.run(buf423, convolution_43, unsqueeze_834, convolution_42, unsqueeze_846, squeeze_392, squeeze_383, buf424, buf425, buf432, buf426, buf433, 1024, 25088, grid=grid(1024), stream=stream0)
        buf427 = buf420; del buf420  # reuse
        buf434 = buf366; del buf366  # reuse
        triton__51.run(buf423, convolution_43, unsqueeze_834, buf425, squeeze_392, buf424, primals_131, convolution_42, unsqueeze_846, buf432, squeeze_383, primals_128, buf427, buf434, 25690112, grid=grid(25690112), stream=stream0)
        del buf423
        del buf425
        del buf432
        del convolution_42
        del convolution_43
        del primals_128
        del primals_131
        del squeeze_383
        del squeeze_392
        del unsqueeze_834
        del unsqueeze_846
        buf428 = aten.convolution_backward(buf427, relu_35, convert_element_type_130, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf427
        del convert_element_type_130
        buf429 = buf428[0]
        assert_size_stride(buf429, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf430 = buf428[1]
        assert_size_stride(buf430, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf428
        buf431 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf430, buf431, 524288, grid=grid(524288), stream=stream0)
        del buf430
        buf435 = aten.convolution_backward(buf434, cat_7, convert_element_type_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_7
        del convert_element_type_127
        buf436 = buf435[0]
        assert_size_stride(buf436, (128, 512, 14, 14), (100352, 196, 14, 1))
        buf437 = buf435[1]
        assert_size_stride(buf437, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf435
        buf438 = empty_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__32.run(buf437, buf438, 524288, grid=grid(524288), stream=stream0)
        del buf437
        buf473 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf439 = as_strided(buf473, (128, 128, 28, 28), (401408, 784, 28, 1), 301056)  # alias
        triton__52.run(buf436, buf439, 12845056, grid=grid(12845056), stream=stream0)
        buf440 = as_strided(buf416, (128, 4), (1, 128)); del buf416  # reuse
        buf442 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        triton__33.run(le_41, buf436, convolution_41, unsqueeze_858, buf440, buf442, 512, 6272, grid=grid(512), stream=stream0)
        buf441 = buf402; del buf402  # reuse
        triton__34.run(buf440, buf441, 128, 4, grid=grid(128), stream=stream0)
        buf443 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf444 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf442, squeeze_374, buf443, buf444, 128, 4, grid=grid(128), stream=stream0)
        buf445 = buf407; del buf407  # reuse
        triton__36.run(le_41, buf436, convolution_41, unsqueeze_858, buf443, squeeze_374, buf441, primals_125, buf445, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_41
        del le_41
        del primals_125
        del squeeze_374
        del unsqueeze_858
        buf446 = aten.convolution_backward(buf445, getitem_156, convert_element_type_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_124
        del getitem_156
        buf447 = buf446[0]
        assert_size_stride(buf447, (128, 128, 28, 28), (100352, 784, 28, 1))
        buf448 = buf446[1]
        assert_size_stride(buf448, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf446
        buf449 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf448, buf449, 18432, grid=grid(18432), stream=stream0)
        del buf448
        buf450 = buf442; del buf442  # reuse
        buf452 = buf440; del buf440  # reuse
        triton__53.run(le_42, buf436, convolution_40, unsqueeze_870, buf450, buf452, 512, 6272, grid=grid(512), stream=stream0)
        buf451 = buf443; del buf443  # reuse
        triton__34.run(buf450, buf451, 128, 4, grid=grid(128), stream=stream0)
        buf453 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf454 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf452, squeeze_365, buf453, buf454, 128, 4, grid=grid(128), stream=stream0)
        buf455 = buf445; del buf445  # reuse
        triton__54.run(le_42, buf436, convolution_40, unsqueeze_870, buf453, squeeze_365, buf451, primals_122, buf455, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_40
        del le_42
        del primals_122
        del squeeze_365
        del unsqueeze_870
        buf456 = aten.convolution_backward(buf455, getitem_151, convert_element_type_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_121
        del getitem_151
        buf457 = buf456[0]
        assert_size_stride(buf457, (128, 128, 28, 28), (100352, 784, 28, 1))
        buf458 = buf456[1]
        assert_size_stride(buf458, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf456
        buf459 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf458, buf459, 18432, grid=grid(18432), stream=stream0)
        del buf458
        buf460 = buf452; del buf452  # reuse
        buf462 = buf450; del buf450  # reuse
        triton__55.run(le_43, buf436, convolution_39, unsqueeze_882, buf460, buf462, 512, 6272, grid=grid(512), stream=stream0)
        buf461 = buf453; del buf453  # reuse
        triton__34.run(buf460, buf461, 128, 4, grid=grid(128), stream=stream0)
        buf463 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf464 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf462, squeeze_356, buf463, buf464, 128, 4, grid=grid(128), stream=stream0)
        buf465 = buf455; del buf455  # reuse
        triton__56.run(le_43, buf436, convolution_39, unsqueeze_882, buf463, squeeze_356, buf461, primals_119, buf465, 3211264, grid=grid(3211264), stream=stream0)
        del buf436
        del convolution_39
        del le_43
        del primals_119
        del squeeze_356
        del unsqueeze_882
        buf466 = aten.convolution_backward(buf465, getitem_146, convert_element_type_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf465
        del convert_element_type_118
        del getitem_146
        buf467 = buf466[0]
        assert_size_stride(buf467, (128, 128, 28, 28), (100352, 784, 28, 1))
        buf468 = buf466[1]
        assert_size_stride(buf468, (128, 16, 3, 3), (144, 9, 3, 1))
        del buf466
        buf469 = empty_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__37.run(buf468, buf469, 18432, grid=grid(18432), stream=stream0)
        del buf468
        buf470 = as_strided(buf473, (128, 128, 28, 28), (401408, 784, 28, 1))  # alias
        triton__57.run(buf467, buf470, 12845056, grid=grid(12845056), stream=stream0)
        del buf467
        buf471 = as_strided(buf473, (128, 128, 28, 28), (401408, 784, 28, 1), 100352)  # alias
        triton__57.run(buf457, buf471, 12845056, grid=grid(12845056), stream=stream0)
        del buf457
        buf472 = as_strided(buf473, (128, 128, 28, 28), (401408, 784, 28, 1), 200704)  # alias
        triton__57.run(buf447, buf472, 12845056, grid=grid(12845056), stream=stream0)
        buf474 = as_strided(buf462, (512, ), (1, )); del buf462  # reuse
        buf475 = as_strided(buf460, (512, ), (1, )); del buf460  # reuse
        buf476 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__58.run(le_44, buf473, convolution_38, unsqueeze_894, squeeze_347, buf474, buf475, buf476, 512, 100352, grid=grid(512), stream=stream0)
        del buf439
        del buf470
        del buf471
        del buf472
        buf477 = buf473; del buf473  # reuse
        triton__59.run(buf477, le_44, convolution_38, unsqueeze_894, buf475, squeeze_347, buf474, primals_116, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_38
        del le_44
        del primals_116
        del squeeze_347
        del unsqueeze_894
        buf478 = aten.convolution_backward(buf477, relu_35, convert_element_type_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_115
        buf479 = buf478[0]
        assert_size_stride(buf479, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf480 = buf478[1]
        assert_size_stride(buf480, (512, 512, 1, 1), (512, 1, 1, 1))
        del buf478
        buf481 = empty_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__60.run(buf480, buf481, 262144, grid=grid(262144), stream=stream0)
        del buf480
        buf482 = buf475; del buf475  # reuse
        buf483 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf485 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__61.run(relu_35, buf429, buf479, convolution_37, unsqueeze_906, squeeze_338, buf482, buf483, buf485, 512, 100352, grid=grid(512), stream=stream0)
        buf486 = buf477; del buf477  # reuse
        triton__62.run(relu_35, buf429, buf479, convolution_37, unsqueeze_906, buf483, squeeze_338, buf482, primals_113, buf486, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_37
        del primals_113
        del squeeze_338
        del unsqueeze_906
        buf487 = aten.convolution_backward(buf486, cat_6, convert_element_type_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf486
        del cat_6
        del convert_element_type_112
        buf488 = buf487[0]
        assert_size_stride(buf488, (128, 256, 28, 28), (200704, 784, 28, 1))
        buf489 = buf487[1]
        assert_size_stride(buf489, (512, 256, 1, 1), (256, 1, 1, 1))
        del buf487
        buf490 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf489, buf490, 131072, grid=grid(131072), stream=stream0)
        del buf489
        buf491 = empty_strided((64, 13), (1, 64), device='cuda', dtype=torch.float32)
        buf493 = empty_strided((64, 13), (1, 64), device='cuda', dtype=torch.float32)
        triton__64.run(le_46, buf488, convolution_36, unsqueeze_918, buf491, buf493, 832, 7720, grid=grid(832), stream=stream0)
        buf492 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__65.run(buf491, buf492, 64, 13, grid=grid(64), stream=stream0)
        buf494 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf495 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf493, squeeze_329, buf494, buf495, 64, 13, grid=grid(64), stream=stream0)
        buf496 = as_strided(buf122, (128, 64, 28, 28), (50176, 784, 28, 1)); del buf122  # reuse
        triton__67.run(le_46, buf488, convolution_36, unsqueeze_918, buf494, squeeze_329, buf492, primals_110, buf496, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_36
        del le_46
        del primals_110
        del squeeze_329
        del unsqueeze_918
        buf497 = aten.convolution_backward(buf496, add_195, convert_element_type_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_195
        del convert_element_type_109
        buf498 = buf497[0]
        assert_size_stride(buf498, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf499 = buf497[1]
        assert_size_stride(buf499, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf497
        buf500 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf499, buf500, 4608, grid=grid(4608), stream=stream0)
        del buf499
        buf501 = buf493; del buf493  # reuse
        buf503 = buf491; del buf491  # reuse
        triton__69.run(le_47, buf488, buf498, convolution_35, unsqueeze_930, buf501, buf503, 832, 7720, grid=grid(832), stream=stream0)
        buf502 = buf494; del buf494  # reuse
        triton__65.run(buf501, buf502, 64, 13, grid=grid(64), stream=stream0)
        buf504 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf506 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf503, squeeze_320, buf504, buf506, 64, 13, grid=grid(64), stream=stream0)
        buf507 = buf496; del buf496  # reuse
        buf527 = as_strided(buf434, (128, 256, 28, 28), (200704, 784, 28, 1)); del buf434  # reuse
        buf525 = as_strided(buf527, (128, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        triton__70.run(le_47, buf488, buf498, convolution_35, unsqueeze_930, buf504, squeeze_320, buf502, primals_107, buf507, buf525, 6422528, grid=grid(6422528), stream=stream0)
        del buf498
        del convolution_35
        del le_47
        del primals_107
        del squeeze_320
        del unsqueeze_930
        buf508 = aten.convolution_backward(buf507, add_189, convert_element_type_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_189
        del convert_element_type_106
        buf509 = buf508[0]
        assert_size_stride(buf509, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf510 = buf508[1]
        assert_size_stride(buf510, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf508
        buf511 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf510, buf511, 4608, grid=grid(4608), stream=stream0)
        del buf510
        buf512 = buf503; del buf503  # reuse
        buf514 = buf501; del buf501  # reuse
        triton__71.run(le_48, buf488, buf509, convolution_34, unsqueeze_942, buf512, buf514, 832, 7720, grid=grid(832), stream=stream0)
        buf513 = buf504; del buf504  # reuse
        triton__65.run(buf512, buf513, 64, 13, grid=grid(64), stream=stream0)
        buf515 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf517 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf514, squeeze_311, buf515, buf517, 64, 13, grid=grid(64), stream=stream0)
        buf518 = buf507; del buf507  # reuse
        buf524 = as_strided(buf527, (128, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        triton__72.run(le_48, buf488, buf509, convolution_34, unsqueeze_942, buf515, squeeze_311, buf513, primals_104, buf518, buf524, 6422528, grid=grid(6422528), stream=stream0)
        del buf509
        del convolution_34
        del le_48
        del primals_104
        del squeeze_311
        del unsqueeze_942
        buf519 = aten.convolution_backward(buf518, getitem_126, convert_element_type_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf518
        del convert_element_type_103
        del getitem_126
        buf520 = buf519[0]
        assert_size_stride(buf520, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf521 = buf519[1]
        assert_size_stride(buf521, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf519
        buf522 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf521, buf522, 4608, grid=grid(4608), stream=stream0)
        del buf521
        buf523 = as_strided(buf527, (128, 64, 28, 28), (200704, 784, 28, 1))  # alias
        triton__26.run(buf520, buf523, 6422528, grid=grid(6422528), stream=stream0)
        buf526 = as_strided(buf527, (128, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        triton__73.run(buf488, buf526, 6422528, grid=grid(6422528), stream=stream0)
        del buf488
        buf528 = buf134; del buf134  # reuse
        buf529 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf530 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__74.run(le_49, buf527, convolution_33, unsqueeze_954, squeeze_302, buf528, buf529, buf530, 256, 100352, grid=grid(256), stream=stream0)
        del buf523
        del buf524
        del buf525
        del buf526
        buf531 = buf527; del buf527  # reuse
        triton__75.run(buf531, le_49, convolution_33, unsqueeze_954, buf529, squeeze_302, buf528, primals_101, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_33
        del le_49
        del primals_101
        del squeeze_302
        del unsqueeze_954
        buf532 = aten.convolution_backward(buf531, relu_30, convert_element_type_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_100
        buf533 = buf532[0]
        assert_size_stride(buf533, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf534 = buf532[1]
        assert_size_stride(buf534, (256, 512, 1, 1), (512, 1, 1, 1))
        del buf532
        buf535 = empty_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf534, buf535, 131072, grid=grid(131072), stream=stream0)
        del buf534
        buf536 = buf429; del buf429  # reuse
        triton__76.run(buf536, relu_30, relu_35, buf479, buf533, 51380224, grid=grid(51380224), stream=stream0)
        del buf479
        del relu_30
        del relu_35
        buf537 = buf483; del buf483  # reuse
        buf538 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf539 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__77.run(buf536, convolution_32, unsqueeze_966, squeeze_293, buf537, buf538, buf539, 512, 100352, grid=grid(512), stream=stream0)
        buf540 = buf533; del buf533  # reuse
        triton__78.run(buf536, convolution_32, unsqueeze_966, buf538, squeeze_293, buf537, primals_98, buf540, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_32
        del primals_98
        del squeeze_293
        del unsqueeze_966
        buf541 = aten.convolution_backward(buf540, cat_5, convert_element_type_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_5
        del convert_element_type_97
        buf542 = buf541[0]
        assert_size_stride(buf542, (128, 256, 28, 28), (200704, 784, 28, 1))
        buf543 = buf541[1]
        assert_size_stride(buf543, (512, 256, 1, 1), (256, 1, 1, 1))
        del buf541
        buf544 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf543, buf544, 131072, grid=grid(131072), stream=stream0)
        del buf543
        buf545 = buf514; del buf514  # reuse
        buf547 = buf512; del buf512  # reuse
        triton__64.run(le_51, buf542, convolution_31, unsqueeze_978, buf545, buf547, 832, 7720, grid=grid(832), stream=stream0)
        buf546 = buf515; del buf515  # reuse
        triton__65.run(buf545, buf546, 64, 13, grid=grid(64), stream=stream0)
        buf548 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf549 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf547, squeeze_284, buf548, buf549, 64, 13, grid=grid(64), stream=stream0)
        buf550 = buf520; del buf520  # reuse
        triton__67.run(le_51, buf542, convolution_31, unsqueeze_978, buf548, squeeze_284, buf546, primals_95, buf550, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_31
        del le_51
        del primals_95
        del squeeze_284
        del unsqueeze_978
        buf551 = aten.convolution_backward(buf550, add_167, convert_element_type_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_167
        del convert_element_type_94
        buf552 = buf551[0]
        assert_size_stride(buf552, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf553 = buf551[1]
        assert_size_stride(buf553, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf551
        buf554 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf553, buf554, 4608, grid=grid(4608), stream=stream0)
        del buf553
        buf555 = buf547; del buf547  # reuse
        buf557 = buf545; del buf545  # reuse
        triton__69.run(le_52, buf542, buf552, convolution_30, unsqueeze_990, buf555, buf557, 832, 7720, grid=grid(832), stream=stream0)
        buf556 = buf548; del buf548  # reuse
        triton__65.run(buf555, buf556, 64, 13, grid=grid(64), stream=stream0)
        buf558 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf560 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf557, squeeze_275, buf558, buf560, 64, 13, grid=grid(64), stream=stream0)
        buf561 = buf550; del buf550  # reuse
        buf581 = buf531; del buf531  # reuse
        buf579 = as_strided(buf581, (128, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        triton__70.run(le_52, buf542, buf552, convolution_30, unsqueeze_990, buf558, squeeze_275, buf556, primals_92, buf561, buf579, 6422528, grid=grid(6422528), stream=stream0)
        del buf552
        del convolution_30
        del le_52
        del primals_92
        del squeeze_275
        del unsqueeze_990
        buf562 = aten.convolution_backward(buf561, add_161, convert_element_type_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_161
        del convert_element_type_91
        buf563 = buf562[0]
        assert_size_stride(buf563, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf564 = buf562[1]
        assert_size_stride(buf564, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf562
        buf565 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf564, buf565, 4608, grid=grid(4608), stream=stream0)
        del buf564
        buf566 = buf557; del buf557  # reuse
        buf568 = buf555; del buf555  # reuse
        triton__71.run(le_53, buf542, buf563, convolution_29, unsqueeze_1002, buf566, buf568, 832, 7720, grid=grid(832), stream=stream0)
        buf567 = buf558; del buf558  # reuse
        triton__65.run(buf566, buf567, 64, 13, grid=grid(64), stream=stream0)
        buf569 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf571 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf568, squeeze_266, buf569, buf571, 64, 13, grid=grid(64), stream=stream0)
        buf572 = buf561; del buf561  # reuse
        buf578 = as_strided(buf581, (128, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        triton__72.run(le_53, buf542, buf563, convolution_29, unsqueeze_1002, buf569, squeeze_266, buf567, primals_89, buf572, buf578, 6422528, grid=grid(6422528), stream=stream0)
        del buf563
        del convolution_29
        del le_53
        del primals_89
        del squeeze_266
        del unsqueeze_1002
        buf573 = aten.convolution_backward(buf572, getitem_106, convert_element_type_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf572
        del convert_element_type_88
        del getitem_106
        buf574 = buf573[0]
        assert_size_stride(buf574, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf575 = buf573[1]
        assert_size_stride(buf575, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf573
        buf576 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf575, buf576, 4608, grid=grid(4608), stream=stream0)
        del buf575
        buf577 = as_strided(buf581, (128, 64, 28, 28), (200704, 784, 28, 1))  # alias
        triton__26.run(buf574, buf577, 6422528, grid=grid(6422528), stream=stream0)
        buf580 = as_strided(buf581, (128, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        triton__73.run(buf542, buf580, 6422528, grid=grid(6422528), stream=stream0)
        del buf542
        buf582 = buf529; del buf529  # reuse
        buf583 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf584 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__74.run(le_54, buf581, convolution_28, unsqueeze_1014, squeeze_257, buf582, buf583, buf584, 256, 100352, grid=grid(256), stream=stream0)
        del buf577
        del buf578
        del buf579
        del buf580
        buf585 = buf581; del buf581  # reuse
        triton__75.run(buf585, le_54, convolution_28, unsqueeze_1014, buf583, squeeze_257, buf582, primals_86, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_28
        del le_54
        del primals_86
        del squeeze_257
        del unsqueeze_1014
        buf586 = aten.convolution_backward(buf585, relu_25, convert_element_type_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_85
        buf587 = buf586[0]
        assert_size_stride(buf587, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf588 = buf586[1]
        assert_size_stride(buf588, (256, 512, 1, 1), (512, 1, 1, 1))
        del buf586
        buf589 = empty_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf588, buf589, 131072, grid=grid(131072), stream=stream0)
        del buf588
        buf590 = buf538; del buf538  # reuse
        buf591 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf593 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__61.run(relu_25, buf536, buf587, convolution_27, unsqueeze_1026, squeeze_248, buf590, buf591, buf593, 512, 100352, grid=grid(512), stream=stream0)
        buf594 = buf540; del buf540  # reuse
        triton__62.run(relu_25, buf536, buf587, convolution_27, unsqueeze_1026, buf591, squeeze_248, buf590, primals_83, buf594, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_27
        del primals_83
        del squeeze_248
        del unsqueeze_1026
        buf595 = aten.convolution_backward(buf594, cat_4, convert_element_type_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf594
        del cat_4
        del convert_element_type_82
        buf596 = buf595[0]
        assert_size_stride(buf596, (128, 256, 28, 28), (200704, 784, 28, 1))
        buf597 = buf595[1]
        assert_size_stride(buf597, (512, 256, 1, 1), (256, 1, 1, 1))
        del buf595
        buf598 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf597, buf598, 131072, grid=grid(131072), stream=stream0)
        del buf597
        buf599 = buf568; del buf568  # reuse
        buf601 = buf566; del buf566  # reuse
        triton__64.run(le_56, buf596, convolution_26, unsqueeze_1038, buf599, buf601, 832, 7720, grid=grid(832), stream=stream0)
        buf600 = buf569; del buf569  # reuse
        triton__65.run(buf599, buf600, 64, 13, grid=grid(64), stream=stream0)
        buf602 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf603 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf601, squeeze_239, buf602, buf603, 64, 13, grid=grid(64), stream=stream0)
        buf604 = buf574; del buf574  # reuse
        triton__67.run(le_56, buf596, convolution_26, unsqueeze_1038, buf602, squeeze_239, buf600, primals_80, buf604, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_26
        del le_56
        del primals_80
        del squeeze_239
        del unsqueeze_1038
        buf605 = aten.convolution_backward(buf604, add_139, convert_element_type_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_139
        del convert_element_type_79
        buf606 = buf605[0]
        assert_size_stride(buf606, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf607 = buf605[1]
        assert_size_stride(buf607, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf605
        buf608 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf607, buf608, 4608, grid=grid(4608), stream=stream0)
        del buf607
        buf609 = buf601; del buf601  # reuse
        buf611 = buf599; del buf599  # reuse
        triton__69.run(le_57, buf596, buf606, convolution_25, unsqueeze_1050, buf609, buf611, 832, 7720, grid=grid(832), stream=stream0)
        buf610 = buf602; del buf602  # reuse
        triton__65.run(buf609, buf610, 64, 13, grid=grid(64), stream=stream0)
        buf612 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf614 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf611, squeeze_230, buf612, buf614, 64, 13, grid=grid(64), stream=stream0)
        buf615 = buf604; del buf604  # reuse
        buf635 = buf585; del buf585  # reuse
        buf633 = as_strided(buf635, (128, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        triton__70.run(le_57, buf596, buf606, convolution_25, unsqueeze_1050, buf612, squeeze_230, buf610, primals_77, buf615, buf633, 6422528, grid=grid(6422528), stream=stream0)
        del buf606
        del convolution_25
        del le_57
        del primals_77
        del squeeze_230
        del unsqueeze_1050
        buf616 = aten.convolution_backward(buf615, add_133, convert_element_type_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_133
        del convert_element_type_76
        buf617 = buf616[0]
        assert_size_stride(buf617, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf618 = buf616[1]
        assert_size_stride(buf618, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf616
        buf619 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf618, buf619, 4608, grid=grid(4608), stream=stream0)
        del buf618
        buf620 = buf611; del buf611  # reuse
        buf622 = buf609; del buf609  # reuse
        triton__71.run(le_58, buf596, buf617, convolution_24, unsqueeze_1062, buf620, buf622, 832, 7720, grid=grid(832), stream=stream0)
        buf621 = buf612; del buf612  # reuse
        triton__65.run(buf620, buf621, 64, 13, grid=grid(64), stream=stream0)
        buf623 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf625 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf622, squeeze_221, buf623, buf625, 64, 13, grid=grid(64), stream=stream0)
        buf626 = buf615; del buf615  # reuse
        buf632 = as_strided(buf635, (128, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        triton__72.run(le_58, buf596, buf617, convolution_24, unsqueeze_1062, buf623, squeeze_221, buf621, primals_74, buf626, buf632, 6422528, grid=grid(6422528), stream=stream0)
        del buf617
        del convolution_24
        del le_58
        del primals_74
        del squeeze_221
        del unsqueeze_1062
        buf627 = aten.convolution_backward(buf626, getitem_86, convert_element_type_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf626
        del convert_element_type_73
        del getitem_86
        buf628 = buf627[0]
        assert_size_stride(buf628, (128, 64, 28, 28), (50176, 784, 28, 1))
        buf629 = buf627[1]
        assert_size_stride(buf629, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf627
        buf630 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf629, buf630, 4608, grid=grid(4608), stream=stream0)
        del buf629
        buf631 = as_strided(buf635, (128, 64, 28, 28), (200704, 784, 28, 1))  # alias
        triton__26.run(buf628, buf631, 6422528, grid=grid(6422528), stream=stream0)
        buf634 = as_strided(buf635, (128, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        triton__73.run(buf596, buf634, 6422528, grid=grid(6422528), stream=stream0)
        del buf596
        buf636 = buf583; del buf583  # reuse
        buf637 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf638 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__74.run(le_59, buf635, convolution_23, unsqueeze_1074, squeeze_212, buf636, buf637, buf638, 256, 100352, grid=grid(256), stream=stream0)
        del buf631
        del buf632
        del buf633
        del buf634
        buf639 = buf635; del buf635  # reuse
        triton__75.run(buf639, le_59, convolution_23, unsqueeze_1074, buf637, squeeze_212, buf636, primals_71, 25690112, grid=grid(25690112), stream=stream0)
        del convolution_23
        del le_59
        del primals_71
        del squeeze_212
        del unsqueeze_1074
        buf640 = aten.convolution_backward(buf639, relu_20, convert_element_type_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf639
        del convert_element_type_70
        buf641 = buf640[0]
        assert_size_stride(buf641, (128, 512, 28, 28), (401408, 784, 28, 1))
        buf642 = buf640[1]
        assert_size_stride(buf642, (256, 512, 1, 1), (512, 1, 1, 1))
        del buf640
        buf643 = empty_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf642, buf643, 131072, grid=grid(131072), stream=stream0)
        del buf642
        buf644 = empty_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda', dtype=torch.float32)
        triton__79.run(relu_20, relu_25, buf536, buf587, buf641, buf644, 51380224, grid=grid(51380224), stream=stream0)
        del buf536
        del relu_20
        del relu_25
        buf645 = buf591; del buf591  # reuse
        buf646 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf653 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf647 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf654 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        triton__80.run(buf644, convolution_22, unsqueeze_1086, convolution_21, unsqueeze_1098, squeeze_203, squeeze_194, buf645, buf646, buf653, buf647, buf654, 512, 100352, grid=grid(512), stream=stream0)
        buf648 = buf641; del buf641  # reuse
        buf655 = buf587; del buf587  # reuse
        triton__81.run(buf644, convolution_22, unsqueeze_1086, buf646, squeeze_203, buf645, primals_68, convolution_21, unsqueeze_1098, buf653, squeeze_194, primals_65, buf648, buf655, 51380224, grid=grid(51380224), stream=stream0)
        del buf644
        del convolution_21
        del convolution_22
        del primals_65
        del primals_68
        del squeeze_194
        del squeeze_203
        del unsqueeze_1086
        del unsqueeze_1098
        buf649 = aten.convolution_backward(buf648, relu_15, convert_element_type_67, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf648
        del convert_element_type_67
        buf650 = buf649[0]
        assert_size_stride(buf650, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf651 = buf649[1]
        assert_size_stride(buf651, (512, 256, 1, 1), (256, 1, 1, 1))
        del buf649
        buf652 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf651, buf652, 131072, grid=grid(131072), stream=stream0)
        del buf651
        buf656 = aten.convolution_backward(buf655, cat_3, convert_element_type_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_3
        del convert_element_type_64
        buf657 = buf656[0]
        assert_size_stride(buf657, (128, 256, 28, 28), (200704, 784, 28, 1))
        buf658 = buf656[1]
        assert_size_stride(buf658, (512, 256, 1, 1), (256, 1, 1, 1))
        del buf656
        buf659 = empty_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__63.run(buf658, buf659, 131072, grid=grid(131072), stream=stream0)
        del buf658
        buf694 = empty_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf660 = as_strided(buf694, (128, 64, 56, 56), (802816, 3136, 56, 1), 602112)  # alias
        triton__82.run(buf657, buf660, 25690112, grid=grid(25690112), stream=stream0)
        buf661 = buf622; del buf622  # reuse
        buf663 = buf620; del buf620  # reuse
        triton__64.run(le_61, buf657, convolution_20, unsqueeze_1110, buf661, buf663, 832, 7720, grid=grid(832), stream=stream0)
        buf662 = buf623; del buf623  # reuse
        triton__65.run(buf661, buf662, 64, 13, grid=grid(64), stream=stream0)
        buf664 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf665 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf663, squeeze_185, buf664, buf665, 64, 13, grid=grid(64), stream=stream0)
        buf666 = buf628; del buf628  # reuse
        triton__67.run(le_61, buf657, convolution_20, unsqueeze_1110, buf664, squeeze_185, buf662, primals_62, buf666, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_20
        del le_61
        del primals_62
        del squeeze_185
        del unsqueeze_1110
        buf667 = aten.convolution_backward(buf666, getitem_76, convert_element_type_61, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_61
        del getitem_76
        buf668 = buf667[0]
        assert_size_stride(buf668, (128, 64, 56, 56), (200704, 3136, 56, 1))
        buf669 = buf667[1]
        assert_size_stride(buf669, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf667
        buf670 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf669, buf670, 4608, grid=grid(4608), stream=stream0)
        del buf669
        buf671 = buf663; del buf663  # reuse
        buf673 = buf661; del buf661  # reuse
        triton__83.run(le_62, buf657, convolution_19, unsqueeze_1122, buf671, buf673, 832, 7720, grid=grid(832), stream=stream0)
        buf672 = buf664; del buf664  # reuse
        triton__65.run(buf671, buf672, 64, 13, grid=grid(64), stream=stream0)
        buf674 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf675 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf673, squeeze_176, buf674, buf675, 64, 13, grid=grid(64), stream=stream0)
        buf676 = buf666; del buf666  # reuse
        triton__84.run(le_62, buf657, convolution_19, unsqueeze_1122, buf674, squeeze_176, buf672, primals_59, buf676, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_19
        del le_62
        del primals_59
        del squeeze_176
        del unsqueeze_1122
        buf677 = aten.convolution_backward(buf676, getitem_71, convert_element_type_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_58
        del getitem_71
        buf678 = buf677[0]
        assert_size_stride(buf678, (128, 64, 56, 56), (200704, 3136, 56, 1))
        buf679 = buf677[1]
        assert_size_stride(buf679, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf677
        buf680 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf679, buf680, 4608, grid=grid(4608), stream=stream0)
        del buf679
        buf681 = buf673; del buf673  # reuse
        buf683 = buf671; del buf671  # reuse
        triton__85.run(le_63, buf657, convolution_18, unsqueeze_1134, buf681, buf683, 832, 7720, grid=grid(832), stream=stream0)
        buf682 = buf674; del buf674  # reuse
        triton__65.run(buf681, buf682, 64, 13, grid=grid(64), stream=stream0)
        del buf681
        buf684 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf685 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__66.run(buf683, squeeze_167, buf684, buf685, 64, 13, grid=grid(64), stream=stream0)
        del buf683
        buf686 = buf676; del buf676  # reuse
        triton__86.run(le_63, buf657, convolution_18, unsqueeze_1134, buf684, squeeze_167, buf682, primals_56, buf686, 6422528, grid=grid(6422528), stream=stream0)
        del buf657
        del convolution_18
        del le_63
        del primals_56
        del squeeze_167
        del unsqueeze_1134
        buf687 = aten.convolution_backward(buf686, getitem_66, convert_element_type_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf686
        del convert_element_type_55
        del getitem_66
        buf688 = buf687[0]
        assert_size_stride(buf688, (128, 64, 56, 56), (200704, 3136, 56, 1))
        buf689 = buf687[1]
        assert_size_stride(buf689, (64, 8, 3, 3), (72, 9, 3, 1))
        del buf687
        buf690 = empty_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__68.run(buf689, buf690, 4608, grid=grid(4608), stream=stream0)
        del buf689
        buf691 = as_strided(buf694, (128, 64, 56, 56), (802816, 3136, 56, 1))  # alias
        triton__87.run(buf688, buf691, 25690112, grid=grid(25690112), stream=stream0)
        del buf688
        buf692 = as_strided(buf694, (128, 64, 56, 56), (802816, 3136, 56, 1), 200704)  # alias
        triton__87.run(buf678, buf692, 25690112, grid=grid(25690112), stream=stream0)
        del buf678
        buf693 = as_strided(buf694, (128, 64, 56, 56), (802816, 3136, 56, 1), 401408)  # alias
        triton__87.run(buf668, buf693, 25690112, grid=grid(25690112), stream=stream0)
        del buf668
        buf695 = buf637; del buf637  # reuse
        buf696 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf697 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__88.run(le_64, buf694, convolution_17, unsqueeze_1146, squeeze_158, buf695, buf696, buf697, 256, 401408, grid=grid(256), stream=stream0)
        del buf660
        del buf691
        del buf692
        del buf693
        buf698 = buf694; del buf694  # reuse
        triton__89.run(buf698, le_64, convolution_17, unsqueeze_1146, buf696, squeeze_158, buf695, primals_53, 102760448, grid=grid(102760448), stream=stream0)
        del convolution_17
        del le_64
        del primals_53
        del squeeze_158
        del unsqueeze_1146
        buf699 = aten.convolution_backward(buf698, relu_15, convert_element_type_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_52
        buf700 = buf699[0]
        assert_size_stride(buf700, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf701 = buf699[1]
        assert_size_stride(buf701, (256, 256, 1, 1), (256, 1, 1, 1))
        del buf699
        buf702 = empty_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__90.run(buf701, buf702, 65536, grid=grid(65536), stream=stream0)
        del buf701
        buf703 = buf696; del buf696  # reuse
        buf704 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf706 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__91.run(relu_15, buf650, buf700, convolution_16, unsqueeze_1158, squeeze_149, buf703, buf704, buf706, 256, 401408, grid=grid(256), stream=stream0)
        buf707 = buf698; del buf698  # reuse
        triton__92.run(relu_15, buf650, buf700, convolution_16, unsqueeze_1158, buf704, squeeze_149, buf703, primals_50, buf707, 102760448, grid=grid(102760448), stream=stream0)
        del convolution_16
        del primals_50
        del squeeze_149
        del unsqueeze_1158
        buf708 = aten.convolution_backward(buf707, cat_2, convert_element_type_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf707
        del cat_2
        del convert_element_type_49
        buf709 = buf708[0]
        assert_size_stride(buf709, (128, 128, 56, 56), (401408, 3136, 56, 1))
        buf710 = buf708[1]
        assert_size_stride(buf710, (256, 128, 1, 1), (128, 1, 1, 1))
        del buf708
        buf711 = empty_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__93.run(buf710, buf711, 32768, grid=grid(32768), stream=stream0)
        del buf710
        buf712 = empty_strided((32, 14), (1, 32), device='cuda', dtype=torch.float32)
        buf714 = empty_strided((32, 14), (1, 32), device='cuda', dtype=torch.float32)
        triton__94.run(le_66, buf709, convolution_15, unsqueeze_1170, buf712, buf714, 448, 28672, grid=grid(448), stream=stream0)
        buf713 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__95.run(buf712, buf713, 32, 14, grid=grid(32), stream=stream0)
        buf715 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf716 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf714, squeeze_140, buf715, buf716, 32, 14, grid=grid(32), stream=stream0)
        buf717 = as_strided(buf447, (128, 32, 56, 56), (100352, 3136, 56, 1)); del buf447  # reuse
        triton__97.run(le_66, buf709, convolution_15, unsqueeze_1170, buf715, squeeze_140, buf713, primals_47, buf717, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_15
        del le_66
        del primals_47
        del squeeze_140
        del unsqueeze_1170
        buf718 = aten.convolution_backward(buf717, add_80, convert_element_type_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_80
        del convert_element_type_46
        buf719 = buf718[0]
        assert_size_stride(buf719, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf720 = buf718[1]
        assert_size_stride(buf720, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf718
        buf721 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf720, buf721, 1152, grid=grid(1152), stream=stream0)
        del buf720
        buf722 = buf714; del buf714  # reuse
        buf724 = buf712; del buf712  # reuse
        triton__99.run(le_67, buf709, buf719, convolution_14, unsqueeze_1182, buf722, buf724, 448, 28672, grid=grid(448), stream=stream0)
        buf723 = buf715; del buf715  # reuse
        triton__95.run(buf722, buf723, 32, 14, grid=grid(32), stream=stream0)
        buf725 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf727 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf724, squeeze_131, buf725, buf727, 32, 14, grid=grid(32), stream=stream0)
        buf728 = buf717; del buf717  # reuse
        buf748 = as_strided(buf655, (128, 128, 56, 56), (401408, 3136, 56, 1)); del buf655  # reuse
        buf746 = as_strided(buf748, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        triton__100.run(le_67, buf709, buf719, convolution_14, unsqueeze_1182, buf725, squeeze_131, buf723, primals_44, buf728, buf746, 12845056, grid=grid(12845056), stream=stream0)
        del buf719
        del convolution_14
        del le_67
        del primals_44
        del squeeze_131
        del unsqueeze_1182
        buf729 = aten.convolution_backward(buf728, add_74, convert_element_type_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_74
        del convert_element_type_43
        buf730 = buf729[0]
        assert_size_stride(buf730, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf731 = buf729[1]
        assert_size_stride(buf731, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf729
        buf732 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf731, buf732, 1152, grid=grid(1152), stream=stream0)
        del buf731
        buf733 = buf724; del buf724  # reuse
        buf735 = buf722; del buf722  # reuse
        triton__101.run(le_68, buf709, buf730, convolution_13, unsqueeze_1194, buf733, buf735, 448, 28672, grid=grid(448), stream=stream0)
        buf734 = buf725; del buf725  # reuse
        triton__95.run(buf733, buf734, 32, 14, grid=grid(32), stream=stream0)
        buf736 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf738 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf735, squeeze_122, buf736, buf738, 32, 14, grid=grid(32), stream=stream0)
        buf739 = buf728; del buf728  # reuse
        buf745 = as_strided(buf748, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        triton__102.run(le_68, buf709, buf730, convolution_13, unsqueeze_1194, buf736, squeeze_122, buf734, primals_41, buf739, buf745, 12845056, grid=grid(12845056), stream=stream0)
        del buf730
        del convolution_13
        del le_68
        del primals_41
        del squeeze_122
        del unsqueeze_1194
        buf740 = aten.convolution_backward(buf739, getitem_46, convert_element_type_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf739
        del convert_element_type_40
        del getitem_46
        buf741 = buf740[0]
        assert_size_stride(buf741, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf742 = buf740[1]
        assert_size_stride(buf742, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf740
        buf743 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf742, buf743, 1152, grid=grid(1152), stream=stream0)
        del buf742
        buf744 = as_strided(buf748, (128, 32, 56, 56), (401408, 3136, 56, 1))  # alias
        triton__57.run(buf741, buf744, 12845056, grid=grid(12845056), stream=stream0)
        buf747 = as_strided(buf748, (128, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        triton__103.run(buf709, buf747, 12845056, grid=grid(12845056), stream=stream0)
        del buf709
        buf749 = as_strided(buf653, (128, 4), (1, 128)); del buf653  # reuse
        buf751 = as_strided(buf646, (128, 4), (1, 128)); del buf646  # reuse
        triton__104.run(le_69, buf748, convolution_12, unsqueeze_1206, buf749, buf751, 512, 100352, grid=grid(512), stream=stream0)
        del buf744
        del buf745
        del buf746
        del buf747
        buf750 = buf463; del buf463  # reuse
        triton__34.run(buf749, buf750, 128, 4, grid=grid(128), stream=stream0)
        buf752 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf753 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf751, squeeze_113, buf752, buf753, 128, 4, grid=grid(128), stream=stream0)
        buf754 = buf748; del buf748  # reuse
        triton__105.run(buf754, le_69, convolution_12, unsqueeze_1206, buf752, squeeze_113, buf750, primals_38, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_12
        del le_69
        del primals_38
        del squeeze_113
        del unsqueeze_1206
        buf755 = aten.convolution_backward(buf754, relu_10, convert_element_type_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_37
        buf756 = buf755[0]
        assert_size_stride(buf756, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf757 = buf755[1]
        assert_size_stride(buf757, (128, 256, 1, 1), (256, 1, 1, 1))
        del buf755
        buf758 = empty_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__93.run(buf757, buf758, 32768, grid=grid(32768), stream=stream0)
        del buf757
        buf759 = buf650; del buf650  # reuse
        triton__106.run(buf759, relu_10, relu_15, buf700, buf756, 102760448, grid=grid(102760448), stream=stream0)
        del relu_10
        del relu_15
        buf760 = buf704; del buf704  # reuse
        buf761 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf762 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__107.run(buf759, convolution_11, unsqueeze_1218, squeeze_104, buf760, buf761, buf762, 256, 401408, grid=grid(256), stream=stream0)
        buf763 = buf756; del buf756  # reuse
        triton__108.run(buf759, convolution_11, unsqueeze_1218, buf761, squeeze_104, buf760, primals_35, buf763, 102760448, grid=grid(102760448), stream=stream0)
        del convolution_11
        del primals_35
        del squeeze_104
        del unsqueeze_1218
        buf764 = aten.convolution_backward(buf763, cat_1, convert_element_type_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del convert_element_type_34
        buf765 = buf764[0]
        assert_size_stride(buf765, (128, 128, 56, 56), (401408, 3136, 56, 1))
        buf766 = buf764[1]
        assert_size_stride(buf766, (256, 128, 1, 1), (128, 1, 1, 1))
        del buf764
        buf767 = empty_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__93.run(buf766, buf767, 32768, grid=grid(32768), stream=stream0)
        del buf766
        buf768 = buf735; del buf735  # reuse
        buf770 = buf733; del buf733  # reuse
        triton__94.run(le_71, buf765, convolution_10, unsqueeze_1230, buf768, buf770, 448, 28672, grid=grid(448), stream=stream0)
        buf769 = buf736; del buf736  # reuse
        triton__95.run(buf768, buf769, 32, 14, grid=grid(32), stream=stream0)
        buf771 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf772 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf770, squeeze_95, buf771, buf772, 32, 14, grid=grid(32), stream=stream0)
        buf773 = buf741; del buf741  # reuse
        triton__97.run(le_71, buf765, convolution_10, unsqueeze_1230, buf771, squeeze_95, buf769, primals_32, buf773, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_10
        del le_71
        del primals_32
        del squeeze_95
        del unsqueeze_1230
        buf774 = aten.convolution_backward(buf773, add_52, convert_element_type_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_52
        del convert_element_type_31
        buf775 = buf774[0]
        assert_size_stride(buf775, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf776 = buf774[1]
        assert_size_stride(buf776, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf774
        buf777 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf776, buf777, 1152, grid=grid(1152), stream=stream0)
        del buf776
        buf778 = buf770; del buf770  # reuse
        buf780 = buf768; del buf768  # reuse
        triton__99.run(le_72, buf765, buf775, convolution_9, unsqueeze_1242, buf778, buf780, 448, 28672, grid=grid(448), stream=stream0)
        buf779 = buf771; del buf771  # reuse
        triton__95.run(buf778, buf779, 32, 14, grid=grid(32), stream=stream0)
        buf781 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf783 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf780, squeeze_86, buf781, buf783, 32, 14, grid=grid(32), stream=stream0)
        buf784 = buf773; del buf773  # reuse
        buf804 = buf754; del buf754  # reuse
        buf802 = as_strided(buf804, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        triton__100.run(le_72, buf765, buf775, convolution_9, unsqueeze_1242, buf781, squeeze_86, buf779, primals_29, buf784, buf802, 12845056, grid=grid(12845056), stream=stream0)
        del buf775
        del convolution_9
        del le_72
        del primals_29
        del squeeze_86
        del unsqueeze_1242
        buf785 = aten.convolution_backward(buf784, add_46, convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_46
        del convert_element_type_28
        buf786 = buf785[0]
        assert_size_stride(buf786, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf787 = buf785[1]
        assert_size_stride(buf787, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf785
        buf788 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf787, buf788, 1152, grid=grid(1152), stream=stream0)
        del buf787
        buf789 = buf780; del buf780  # reuse
        buf791 = buf778; del buf778  # reuse
        triton__101.run(le_73, buf765, buf786, convolution_8, unsqueeze_1254, buf789, buf791, 448, 28672, grid=grid(448), stream=stream0)
        buf790 = buf781; del buf781  # reuse
        triton__95.run(buf789, buf790, 32, 14, grid=grid(32), stream=stream0)
        buf792 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf794 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf791, squeeze_77, buf792, buf794, 32, 14, grid=grid(32), stream=stream0)
        buf795 = buf784; del buf784  # reuse
        buf801 = as_strided(buf804, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        triton__102.run(le_73, buf765, buf786, convolution_8, unsqueeze_1254, buf792, squeeze_77, buf790, primals_26, buf795, buf801, 12845056, grid=grid(12845056), stream=stream0)
        del buf786
        del convolution_8
        del le_73
        del primals_26
        del squeeze_77
        del unsqueeze_1254
        buf796 = aten.convolution_backward(buf795, getitem_26, convert_element_type_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf795
        del convert_element_type_25
        del getitem_26
        buf797 = buf796[0]
        assert_size_stride(buf797, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf798 = buf796[1]
        assert_size_stride(buf798, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf796
        buf799 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf798, buf799, 1152, grid=grid(1152), stream=stream0)
        del buf798
        buf800 = as_strided(buf804, (128, 32, 56, 56), (401408, 3136, 56, 1))  # alias
        triton__57.run(buf797, buf800, 12845056, grid=grid(12845056), stream=stream0)
        buf803 = as_strided(buf804, (128, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        triton__103.run(buf765, buf803, 12845056, grid=grid(12845056), stream=stream0)
        del buf765
        buf805 = buf751; del buf751  # reuse
        buf807 = buf749; del buf749  # reuse
        triton__104.run(le_74, buf804, convolution_7, unsqueeze_1266, buf805, buf807, 512, 100352, grid=grid(512), stream=stream0)
        del buf800
        del buf801
        del buf802
        del buf803
        buf806 = buf752; del buf752  # reuse
        triton__34.run(buf805, buf806, 128, 4, grid=grid(128), stream=stream0)
        buf808 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf809 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf807, squeeze_68, buf808, buf809, 128, 4, grid=grid(128), stream=stream0)
        buf810 = buf804; del buf804  # reuse
        triton__105.run(buf810, le_74, convolution_7, unsqueeze_1266, buf808, squeeze_68, buf806, primals_23, 51380224, grid=grid(51380224), stream=stream0)
        del convolution_7
        del le_74
        del primals_23
        del squeeze_68
        del unsqueeze_1266
        buf811 = aten.convolution_backward(buf810, relu_5, convert_element_type_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_22
        buf812 = buf811[0]
        assert_size_stride(buf812, (128, 256, 56, 56), (802816, 3136, 56, 1))
        buf813 = buf811[1]
        assert_size_stride(buf813, (128, 256, 1, 1), (256, 1, 1, 1))
        del buf811
        buf814 = empty_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__93.run(buf813, buf814, 32768, grid=grid(32768), stream=stream0)
        del buf813
        buf815 = buf761; del buf761  # reuse
        buf816 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf824 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf818 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf826 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        triton__109.run(relu_5, buf759, buf812, convolution_6, unsqueeze_1278, convolution_5, unsqueeze_1290, squeeze_59, squeeze_50, buf815, buf816, buf824, buf818, buf826, 256, 401408, grid=grid(256), stream=stream0)
        buf819 = buf763; del buf763  # reuse
        buf827 = buf700; del buf700  # reuse
        triton__110.run(relu_5, buf759, buf812, convolution_6, unsqueeze_1278, buf816, squeeze_59, buf815, primals_20, convolution_5, unsqueeze_1290, buf824, squeeze_50, primals_17, buf819, buf827, 102760448, grid=grid(102760448), stream=stream0)
        del buf759
        del buf812
        del buf816
        del buf824
        del convolution_5
        del convolution_6
        del primals_17
        del primals_20
        del relu_5
        del squeeze_50
        del squeeze_59
        del unsqueeze_1278
        del unsqueeze_1290
        buf820 = aten.convolution_backward(buf819, getitem, convert_element_type_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf819
        del convert_element_type_19
        buf821 = buf820[0]
        assert_size_stride(buf821, (128, 64, 56, 56), (200704, 3136, 56, 1))
        buf822 = buf820[1]
        assert_size_stride(buf822, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf820
        buf823 = empty_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__111.run(buf822, buf823, 16384, grid=grid(16384), stream=stream0)
        del buf822
        buf828 = aten.convolution_backward(buf827, cat, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat
        del convert_element_type_16
        buf829 = buf828[0]
        assert_size_stride(buf829, (128, 128, 56, 56), (401408, 3136, 56, 1))
        buf830 = buf828[1]
        assert_size_stride(buf830, (256, 128, 1, 1), (128, 1, 1, 1))
        del buf828
        buf831 = empty_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__93.run(buf830, buf831, 32768, grid=grid(32768), stream=stream0)
        del buf830
        buf866 = buf810; del buf810  # reuse
        buf832 = as_strided(buf866, (128, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        triton__112.run(buf829, buf832, 12845056, grid=grid(12845056), stream=stream0)
        buf833 = buf791; del buf791  # reuse
        buf835 = buf789; del buf789  # reuse
        triton__94.run(le_76, buf829, convolution_4, unsqueeze_1302, buf833, buf835, 448, 28672, grid=grid(448), stream=stream0)
        buf834 = buf792; del buf792  # reuse
        triton__95.run(buf833, buf834, 32, 14, grid=grid(32), stream=stream0)
        buf836 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf837 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf835, squeeze_41, buf836, buf837, 32, 14, grid=grid(32), stream=stream0)
        buf838 = buf797; del buf797  # reuse
        triton__97.run(le_76, buf829, convolution_4, unsqueeze_1302, buf836, squeeze_41, buf834, primals_14, buf838, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_4
        del le_76
        del primals_14
        del squeeze_41
        del unsqueeze_1302
        buf839 = aten.convolution_backward(buf838, getitem_16, convert_element_type_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_13
        del getitem_16
        buf840 = buf839[0]
        assert_size_stride(buf840, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf841 = buf839[1]
        assert_size_stride(buf841, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf839
        buf842 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf841, buf842, 1152, grid=grid(1152), stream=stream0)
        del buf841
        buf843 = buf835; del buf835  # reuse
        buf845 = buf833; del buf833  # reuse
        triton__113.run(le_77, buf829, convolution_3, unsqueeze_1314, buf843, buf845, 448, 28672, grid=grid(448), stream=stream0)
        buf844 = buf836; del buf836  # reuse
        triton__95.run(buf843, buf844, 32, 14, grid=grid(32), stream=stream0)
        buf846 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf847 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf845, squeeze_32, buf846, buf847, 32, 14, grid=grid(32), stream=stream0)
        buf848 = buf838; del buf838  # reuse
        triton__114.run(le_77, buf829, convolution_3, unsqueeze_1314, buf846, squeeze_32, buf844, primals_11, buf848, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_3
        del le_77
        del primals_11
        del squeeze_32
        del unsqueeze_1314
        buf849 = aten.convolution_backward(buf848, getitem_11, convert_element_type_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del convert_element_type_10
        del getitem_11
        buf850 = buf849[0]
        assert_size_stride(buf850, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf851 = buf849[1]
        assert_size_stride(buf851, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf849
        buf852 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf851, buf852, 1152, grid=grid(1152), stream=stream0)
        del buf851
        buf853 = buf845; del buf845  # reuse
        buf855 = buf843; del buf843  # reuse
        triton__115.run(le_78, buf829, convolution_2, unsqueeze_1326, buf853, buf855, 448, 28672, grid=grid(448), stream=stream0)
        buf854 = buf846; del buf846  # reuse
        triton__95.run(buf853, buf854, 32, 14, grid=grid(32), stream=stream0)
        buf856 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf857 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__96.run(buf855, squeeze_23, buf856, buf857, 32, 14, grid=grid(32), stream=stream0)
        buf858 = buf848; del buf848  # reuse
        triton__116.run(le_78, buf829, convolution_2, unsqueeze_1326, buf856, squeeze_23, buf854, primals_8, buf858, 12845056, grid=grid(12845056), stream=stream0)
        del buf829
        del buf856
        del convolution_2
        del le_78
        del primals_8
        del squeeze_23
        del unsqueeze_1326
        buf859 = aten.convolution_backward(buf858, getitem_6, convert_element_type_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf858
        del convert_element_type_7
        del getitem_6
        buf860 = buf859[0]
        assert_size_stride(buf860, (128, 32, 56, 56), (100352, 3136, 56, 1))
        buf861 = buf859[1]
        assert_size_stride(buf861, (32, 4, 3, 3), (36, 9, 3, 1))
        del buf859
        buf862 = empty_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda', dtype=torch.float32)
        triton__98.run(buf861, buf862, 1152, grid=grid(1152), stream=stream0)
        del buf861
        buf863 = as_strided(buf866, (128, 32, 56, 56), (401408, 3136, 56, 1))  # alias
        triton__57.run(buf860, buf863, 12845056, grid=grid(12845056), stream=stream0)
        del buf860
        buf864 = as_strided(buf866, (128, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        triton__57.run(buf850, buf864, 12845056, grid=grid(12845056), stream=stream0)
        del buf850
        buf865 = as_strided(buf866, (128, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        triton__57.run(buf840, buf865, 12845056, grid=grid(12845056), stream=stream0)
        del buf840
        buf867 = buf807; del buf807  # reuse
        buf869 = buf805; del buf805  # reuse
        triton__104.run(le_79, buf866, convolution_1, unsqueeze_1338, buf867, buf869, 512, 100352, grid=grid(512), stream=stream0)
        del buf832
        del buf863
        del buf864
        del buf865
        buf868 = buf808; del buf808  # reuse
        triton__34.run(buf867, buf868, 128, 4, grid=grid(128), stream=stream0)
        del buf867
        buf870 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf871 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__35.run(buf869, squeeze_14, buf870, buf871, 128, 4, grid=grid(128), stream=stream0)
        del buf869
        buf872 = buf866; del buf866  # reuse
        triton__105.run(buf872, le_79, convolution_1, unsqueeze_1338, buf870, squeeze_14, buf868, primals_5, 51380224, grid=grid(51380224), stream=stream0)
        del buf870
        del convolution_1
        del le_79
        del primals_5
        del squeeze_14
        del unsqueeze_1338
        buf873 = aten.convolution_backward(buf872, getitem, convert_element_type_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf872
        del convert_element_type_4
        del getitem
        buf874 = buf873[0]
        assert_size_stride(buf874, (128, 64, 56, 56), (200704, 3136, 56, 1))
        buf875 = buf873[1]
        assert_size_stride(buf875, (128, 64, 1, 1), (64, 1, 1, 1))
        del buf873
        buf876 = empty_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__117.run(buf875, buf876, 8192, grid=grid(8192), stream=stream0)
        del buf875
        buf877 = buf821; del buf821  # reuse
        triton__118.run(buf877, buf874, 25690112, grid=grid(25690112), stream=stream0)
        del buf874
        buf878 = as_strided(buf827, (128, 64, 112, 112), (802816, 12544, 112, 1)); del buf827  # reuse
        triton__119.run(getitem_1, buf877, buf878, 102760448, grid=grid(102760448), stream=stream0)
        del buf877
        del getitem_1
        buf879 = as_strided(buf855, (64, 7), (1, 64)); del buf855  # reuse
        buf881 = as_strided(buf853, (64, 7), (1, 64)); del buf853  # reuse
        triton__120.run(relu, buf878, convolution, unsqueeze_1350, buf879, buf881, 448, 229376, grid=grid(448), stream=stream0)
        buf880 = buf684; del buf684  # reuse
        triton__121.run(buf879, buf880, 64, 7, grid=grid(64), stream=stream0)
        del buf879
        buf882 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf883 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__122.run(buf881, squeeze_5, buf882, buf883, 64, 7, grid=grid(64), stream=stream0)
        del buf881
        buf884 = buf878; del buf878  # reuse
        triton__123.run(buf884, relu, convolution, unsqueeze_1350, buf882, squeeze_5, buf880, primals_2, 102760448, grid=grid(102760448), stream=stream0)
        del buf882
        del convolution
        del primals_2
        del relu
        del squeeze_5
        del unsqueeze_1350
        buf885 = aten.convolution_backward(buf884, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf884
        del convert_element_type
        del convert_element_type_1
        buf886 = buf885[1]
        assert_size_stride(buf886, (64, 3, 7, 7), (147, 49, 7, 1))
        del buf885
        buf887 = empty_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda', dtype=torch.float32)
        triton__124.run(buf886, buf887, 9408, grid=grid(9408), stream=stream0)
        return (buf887, buf883, buf880, buf876, buf871, buf868, buf862, buf857, buf854, buf852, buf847, buf844, buf842, buf837, buf834, buf831, buf826, buf815, buf823, buf818, buf815, buf814, buf809, buf806, buf799, buf794, buf790, buf788, buf783, buf779, buf777, buf772, buf769, buf767, buf762, buf760, buf758, buf753, buf750, buf743, buf738, buf734, buf732, buf727, buf723, buf721, buf716, buf713, buf711, buf706, buf703, buf702, buf697, buf695, buf690, buf685, buf682, buf680, buf675, buf672, buf670, buf665, buf662, buf659, buf654, buf645, buf652, buf647, buf645, buf643, buf638, buf636, buf630, buf625, buf621, buf619, buf614, buf610, buf608, buf603, buf600, buf598, buf593, buf590, buf589, buf584, buf582, buf576, buf571, buf567, buf565, buf560, buf556, buf554, buf549, buf546, buf544, buf539, buf537, buf535, buf530, buf528, buf522, buf517, buf513, buf511, buf506, buf502, buf500, buf495, buf492, buf490, buf485, buf482, buf481, buf476, buf474, buf469, buf464, buf461, buf459, buf454, buf451, buf449, buf444, buf441, buf438, buf433, buf424, buf431, buf426, buf424, buf422, buf417, buf415, buf409, buf404, buf400, buf398, buf393, buf389, buf387, buf382, buf379, buf377, buf372, buf369, buf368, buf363, buf361, buf355, buf350, buf346, buf344, buf339, buf335, buf333, buf328, buf325, buf323, buf318, buf316, buf314, buf309, buf307, buf301, buf296, buf292, buf290, buf285, buf281, buf279, buf274, buf271, buf269, buf264, buf261, buf260, buf255, buf253, buf247, buf242, buf238, buf236, buf231, buf227, buf225, buf220, buf217, buf215, buf210, buf208, buf206, buf201, buf199, buf193, buf188, buf184, buf182, buf177, buf173, buf171, buf166, buf163, buf161, buf156, buf153, buf152, buf147, buf145, buf140, buf135, buf133, buf132, buf127, buf125, buf124, buf119, buf117, buf115, buf110, buf101, buf108, buf103, buf101, buf99, buf94, buf92, buf86, buf81, buf78, buf77, buf72, buf69, buf68, buf63, buf61, buf60, buf55, buf52, buf51, buf46, buf44, buf38, buf33, buf30, buf29, buf24, buf21, buf20, buf15, buf13, buf12, buf7, buf5, buf3, buf4, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_1 = rand_strided((128, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float16)
    convolution = rand_strided((128, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float16)
    squeeze_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((128, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float16)
    getitem = rand_strided((128, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    getitem_1 = rand_strided((128, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.int64)
    convert_element_type_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_1 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_7 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_6 = rand_strided((128, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_2 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_10 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_11 = rand_strided((128, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_3 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_32 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_13 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_16 = rand_strided((128, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_4 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((128, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    cat = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_16 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_5 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_19 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_6 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_22 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_7 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_25 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_26 = rand_strided((128, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_8 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_77 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_28 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_9 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_86 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_52 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_31 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_10 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_11 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_37 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_12 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_40 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_46 = rand_strided((128, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_13 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_122 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_74 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_43 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_14 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_131 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_80 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_46 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_15 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_140 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_49 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_16 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_52 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_17 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    squeeze_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_55 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_66 = rand_strided((128, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_18 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_167 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_58 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_71 = rand_strided((128, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_19 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_176 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_61 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_76 = rand_strided((128, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    convolution_20 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((128, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float16)
    cat_3 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_64 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_21 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_67 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_22 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_23 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_73 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_86 = rand_strided((128, 64, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convolution_24 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_133 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_76 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_25 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_230 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_139 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_79 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_26 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_82 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_27 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_85 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_28 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_257 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_88 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_106 = rand_strided((128, 64, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convolution_29 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_266 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_161 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_91 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_30 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_167 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_94 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_31 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_284 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_97 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_32 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_100 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_33 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_302 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_103 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_126 = rand_strided((128, 64, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convolution_34 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_311 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_189 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_106 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_35 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_320 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_195 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_109 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_36 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_329 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_112 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_37 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_115 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_38 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    squeeze_347 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_118 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_146 = rand_strided((128, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convolution_39 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_356 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_121 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_151 = rand_strided((128, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convolution_40 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_365 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_124 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_156 = rand_strided((128, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    convolution_41 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_374 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_161 = rand_strided((128, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float16)
    cat_7 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_42 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_383 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_130 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_43 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_392 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_133 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_44 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_401 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_136 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_166 = rand_strided((128, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_45 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_410 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_248 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_139 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_46 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_419 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_254 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_142 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_47 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_428 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_145 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_48 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_437 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_148 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_49 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_446 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_151 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_186 = rand_strided((128, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_50 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_455 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_276 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_154 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_51 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_464 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_282 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_157 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_52 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_473 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_160 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_53 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_482 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_163 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_54 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_491 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_166 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_206 = rand_strided((128, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_55 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_500 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_304 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_169 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_56 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_509 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_310 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_172 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_57 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_518 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_175 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_58 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_527 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_178 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_59 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_536 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_181 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_226 = rand_strided((128, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_60 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_545 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_332 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_184 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_61 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_554 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_338 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_187 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_62 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_563 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_190 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_63 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_572 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_193 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_64 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_581 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_196 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_246 = rand_strided((128, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_65 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_590 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_360 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_199 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_66 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_599 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_366 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_202 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_67 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_608 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_205 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_68 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_617 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_208 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_69 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    squeeze_626 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_211 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_266 = rand_strided((128, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_70 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_635 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_214 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_271 = rand_strided((128, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_71 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_644 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_217 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_276 = rand_strided((128, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    convolution_72 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_653 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_281 = rand_strided((128, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float16)
    cat_13 = rand_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_220 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_73 = rand_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_662 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_223 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_74 = rand_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_671 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_226 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_75 = rand_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_680 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_229 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_286 = rand_strided((128, 256, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convolution_76 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_689 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_419 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_232 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_77 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_698 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_425 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_235 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_78 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_707 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_14 = rand_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_238 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_79 = rand_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_716 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_241 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_80 = rand_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_725 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_244 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    getitem_306 = rand_strided((128, 256, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convolution_81 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_734 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_447 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_247 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_82 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_743 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_453 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_250 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    convolution_83 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_752 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_15 = rand_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_253 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    convolution_84 = rand_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float16)
    squeeze_761 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    le = rand_strided((128, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_342 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_354 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_366 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_378 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_414 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_426 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_438 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((128, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_450 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_498 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_13 = rand_strided((128, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_510 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((128, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_522 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_546 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_570 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_19 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_582 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_21 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_618 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_23 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_630 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_24 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_642 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_26 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_27 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_28 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_690 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_29 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_702 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_31 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_726 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_32 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_738 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_33 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_750 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_34 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_762 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_36 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_786 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_37 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_798 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_38 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_810 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_39 = rand_strided((128, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_822 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_41 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_858 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_42 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_870 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_43 = rand_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_882 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_44 = rand_strided((128, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_894 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_46 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_918 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_47 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_930 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_48 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_942 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_49 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_954 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_966 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_51 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_978 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_52 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_990 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_53 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1002 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_54 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1014 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1026 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_56 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1038 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_57 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1050 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_58 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1062 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_59 = rand_strided((128, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1074 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1086 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1098 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_61 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1110 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_62 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1122 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_63 = rand_strided((128, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1134 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_64 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1146 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1158 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_66 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1170 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_67 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1182 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_68 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1194 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_69 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1206 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1218 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_71 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1230 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_72 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1242 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_73 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1254 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_74 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1266 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1278 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1290 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_76 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1302 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_77 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1314 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_78 = rand_strided((128, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1326 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_79 = rand_strided((128, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1338 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1350 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_28 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_36 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_39 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_40 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_42 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_45 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_48 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_54 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_57 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_60 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_69 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_72 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_78 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_81 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_90 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_93 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_99 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_102 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_105 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_106 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_107 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_108 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_114 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_120 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_126 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_130 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_132 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_138 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_144 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_145 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_153 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_160 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_162 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_165 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_168 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_174 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_175 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_180 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_189 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_190 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_204 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_205 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_208 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_220 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_223 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_226 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_238 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_239 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_253 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_254 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_256 = rand_strided((128, 1000), (1000, 1), device='cuda:0', dtype=torch.float16)
    print_performance(lambda: call([primals_2, primals_5, primals_8, primals_11, primals_14, primals_17, primals_20, primals_23, primals_26, primals_29, primals_32, primals_35, primals_38, primals_41, primals_44, primals_47, primals_50, primals_53, primals_56, primals_59, primals_62, primals_65, primals_68, primals_71, primals_74, primals_77, primals_80, primals_83, primals_86, primals_89, primals_92, primals_95, primals_98, primals_101, primals_104, primals_107, primals_110, primals_113, primals_116, primals_119, primals_122, primals_125, primals_128, primals_131, primals_134, primals_137, primals_140, primals_143, primals_146, primals_149, primals_152, primals_155, primals_158, primals_161, primals_164, primals_167, primals_170, primals_173, primals_176, primals_179, primals_182, primals_185, primals_188, primals_191, primals_194, primals_197, primals_200, primals_203, primals_206, primals_209, primals_212, primals_215, primals_218, primals_221, primals_224, primals_227, primals_230, primals_233, primals_236, primals_239, primals_242, primals_245, primals_248, primals_251, primals_254, convert_element_type, convert_element_type_1, convolution, squeeze_5, relu, getitem, getitem_1, convert_element_type_4, convolution_1, squeeze_14, convert_element_type_7, getitem_6, convolution_2, squeeze_23, convert_element_type_10, getitem_11, convolution_3, squeeze_32, convert_element_type_13, getitem_16, convolution_4, squeeze_41, getitem_21, cat, convert_element_type_16, convolution_5, squeeze_50, convert_element_type_19, convolution_6, squeeze_59, relu_5, convert_element_type_22, convolution_7, squeeze_68, convert_element_type_25, getitem_26, convolution_8, squeeze_77, add_46, convert_element_type_28, convolution_9, squeeze_86, add_52, convert_element_type_31, convolution_10, squeeze_95, cat_1, convert_element_type_34, convolution_11, squeeze_104, relu_10, convert_element_type_37, convolution_12, squeeze_113, convert_element_type_40, getitem_46, convolution_13, squeeze_122, add_74, convert_element_type_43, convolution_14, squeeze_131, add_80, convert_element_type_46, convolution_15, squeeze_140, cat_2, convert_element_type_49, convolution_16, squeeze_149, relu_15, convert_element_type_52, convolution_17, squeeze_158, convert_element_type_55, getitem_66, convolution_18, squeeze_167, convert_element_type_58, getitem_71, convolution_19, squeeze_176, convert_element_type_61, getitem_76, convolution_20, squeeze_185, getitem_81, cat_3, convert_element_type_64, convolution_21, squeeze_194, convert_element_type_67, convolution_22, squeeze_203, relu_20, convert_element_type_70, convolution_23, squeeze_212, convert_element_type_73, getitem_86, convolution_24, squeeze_221, add_133, convert_element_type_76, convolution_25, squeeze_230, add_139, convert_element_type_79, convolution_26, squeeze_239, cat_4, convert_element_type_82, convolution_27, squeeze_248, relu_25, convert_element_type_85, convolution_28, squeeze_257, convert_element_type_88, getitem_106, convolution_29, squeeze_266, add_161, convert_element_type_91, convolution_30, squeeze_275, add_167, convert_element_type_94, convolution_31, squeeze_284, cat_5, convert_element_type_97, convolution_32, squeeze_293, relu_30, convert_element_type_100, convolution_33, squeeze_302, convert_element_type_103, getitem_126, convolution_34, squeeze_311, add_189, convert_element_type_106, convolution_35, squeeze_320, add_195, convert_element_type_109, convolution_36, squeeze_329, cat_6, convert_element_type_112, convolution_37, squeeze_338, relu_35, convert_element_type_115, convolution_38, squeeze_347, convert_element_type_118, getitem_146, convolution_39, squeeze_356, convert_element_type_121, getitem_151, convolution_40, squeeze_365, convert_element_type_124, getitem_156, convolution_41, squeeze_374, getitem_161, cat_7, convert_element_type_127, convolution_42, squeeze_383, convert_element_type_130, convolution_43, squeeze_392, relu_40, convert_element_type_133, convolution_44, squeeze_401, convert_element_type_136, getitem_166, convolution_45, squeeze_410, add_248, convert_element_type_139, convolution_46, squeeze_419, add_254, convert_element_type_142, convolution_47, squeeze_428, cat_8, convert_element_type_145, convolution_48, squeeze_437, relu_45, convert_element_type_148, convolution_49, squeeze_446, convert_element_type_151, getitem_186, convolution_50, squeeze_455, add_276, convert_element_type_154, convolution_51, squeeze_464, add_282, convert_element_type_157, convolution_52, squeeze_473, cat_9, convert_element_type_160, convolution_53, squeeze_482, relu_50, convert_element_type_163, convolution_54, squeeze_491, convert_element_type_166, getitem_206, convolution_55, squeeze_500, add_304, convert_element_type_169, convolution_56, squeeze_509, add_310, convert_element_type_172, convolution_57, squeeze_518, cat_10, convert_element_type_175, convolution_58, squeeze_527, relu_55, convert_element_type_178, convolution_59, squeeze_536, convert_element_type_181, getitem_226, convolution_60, squeeze_545, add_332, convert_element_type_184, convolution_61, squeeze_554, add_338, convert_element_type_187, convolution_62, squeeze_563, cat_11, convert_element_type_190, convolution_63, squeeze_572, relu_60, convert_element_type_193, convolution_64, squeeze_581, convert_element_type_196, getitem_246, convolution_65, squeeze_590, add_360, convert_element_type_199, convolution_66, squeeze_599, add_366, convert_element_type_202, convolution_67, squeeze_608, cat_12, convert_element_type_205, convolution_68, squeeze_617, relu_65, convert_element_type_208, convolution_69, squeeze_626, convert_element_type_211, getitem_266, convolution_70, squeeze_635, convert_element_type_214, getitem_271, convolution_71, squeeze_644, convert_element_type_217, getitem_276, convolution_72, squeeze_653, getitem_281, cat_13, convert_element_type_220, convolution_73, squeeze_662, convert_element_type_223, convolution_74, squeeze_671, relu_70, convert_element_type_226, convolution_75, squeeze_680, convert_element_type_229, getitem_286, convolution_76, squeeze_689, add_419, convert_element_type_232, convolution_77, squeeze_698, add_425, convert_element_type_235, convolution_78, squeeze_707, cat_14, convert_element_type_238, convolution_79, squeeze_716, relu_75, convert_element_type_241, convolution_80, squeeze_725, convert_element_type_244, getitem_306, convolution_81, squeeze_734, add_447, convert_element_type_247, convolution_82, squeeze_743, add_453, convert_element_type_250, convolution_83, squeeze_752, cat_15, convert_element_type_253, convolution_84, squeeze_761, view, permute_1, le, unsqueeze_342, le_1, unsqueeze_354, le_2, unsqueeze_366, le_3, unsqueeze_378, le_4, unsqueeze_390, unsqueeze_402, le_6, unsqueeze_414, le_7, unsqueeze_426, le_8, unsqueeze_438, le_9, unsqueeze_450, unsqueeze_462, unsqueeze_474, le_11, unsqueeze_486, le_12, unsqueeze_498, le_13, unsqueeze_510, le_14, unsqueeze_522, unsqueeze_534, le_16, unsqueeze_546, le_17, unsqueeze_558, le_18, unsqueeze_570, le_19, unsqueeze_582, unsqueeze_594, le_21, unsqueeze_606, le_22, unsqueeze_618, le_23, unsqueeze_630, le_24, unsqueeze_642, unsqueeze_654, le_26, unsqueeze_666, le_27, unsqueeze_678, le_28, unsqueeze_690, le_29, unsqueeze_702, unsqueeze_714, le_31, unsqueeze_726, le_32, unsqueeze_738, le_33, unsqueeze_750, le_34, unsqueeze_762, unsqueeze_774, le_36, unsqueeze_786, le_37, unsqueeze_798, le_38, unsqueeze_810, le_39, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_41, unsqueeze_858, le_42, unsqueeze_870, le_43, unsqueeze_882, le_44, unsqueeze_894, unsqueeze_906, le_46, unsqueeze_918, le_47, unsqueeze_930, le_48, unsqueeze_942, le_49, unsqueeze_954, unsqueeze_966, le_51, unsqueeze_978, le_52, unsqueeze_990, le_53, unsqueeze_1002, le_54, unsqueeze_1014, unsqueeze_1026, le_56, unsqueeze_1038, le_57, unsqueeze_1050, le_58, unsqueeze_1062, le_59, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, le_61, unsqueeze_1110, le_62, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, le_66, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, le_71, unsqueeze_1230, le_72, unsqueeze_1242, le_73, unsqueeze_1254, le_74, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, le_76, unsqueeze_1302, le_77, unsqueeze_1314, le_78, unsqueeze_1326, le_79, unsqueeze_1338, unsqueeze_1350, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160, tangents_161, tangents_162, tangents_163, tangents_164, tangents_165, tangents_166, tangents_167, tangents_168, tangents_169, tangents_170, tangents_171, tangents_172, tangents_173, tangents_174, tangents_175, tangents_176, tangents_177, tangents_178, tangents_179, tangents_180, tangents_181, tangents_182, tangents_183, tangents_184, tangents_185, tangents_186, tangents_187, tangents_188, tangents_189, tangents_190, tangents_191, tangents_192, tangents_193, tangents_194, tangents_195, tangents_196, tangents_197, tangents_198, tangents_199, tangents_200, tangents_201, tangents_202, tangents_203, tangents_204, tangents_205, tangents_206, tangents_207, tangents_208, tangents_209, tangents_210, tangents_211, tangents_212, tangents_213, tangents_214, tangents_215, tangents_216, tangents_217, tangents_218, tangents_219, tangents_220, tangents_221, tangents_222, tangents_223, tangents_224, tangents_225, tangents_226, tangents_227, tangents_228, tangents_229, tangents_230, tangents_231, tangents_232, tangents_233, tangents_234, tangents_235, tangents_236, tangents_237, tangents_238, tangents_239, tangents_240, tangents_241, tangents_242, tangents_243, tangents_244, tangents_245, tangents_246, tangents_247, tangents_248, tangents_249, tangents_250, tangents_251, tangents_252, tangents_253, tangents_254, tangents_255, tangents_256]))
