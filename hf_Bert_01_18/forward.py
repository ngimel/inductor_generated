
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
seed_cuda_0 = None  # 12bf87036c8e625335a9db42dcf50de0c1ec952294785adced537424d5733e17


triton__0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp16', 4: '*i64', 5: '*fp16', 6: '*i64', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*i64', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: 'i32', 15: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]})
@triton.jit
def triton__0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, seed8, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    x0 = xindex % 512
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp5 = tl.load(in_ptr4 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp1 = tl.load(in_ptr1 + (r2 + (768*tmp0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr3 + (r2 + (768*tmp2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp6 = tl.load(in_ptr5 + (r2 + (768*tmp5)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp7 = tmp4 + tmp6
        tl.store(out_ptr0 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp7, rmask & xmask)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp9, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = 768.0
        tmp14 = tmp10 / tmp13
        tmp15 = tmp12 - tmp14
        tmp16 = tmp15 * tmp15
        _tmp17 = tl.where(xmask & rmask, _tmp17 + tmp16, _tmp17)
        _tmp18 = tl.where(xmask & rmask, _tmp18 + tmp12, _tmp18)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 768.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp17 / tmp19
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = tl.libdevice.rsqrt(tmp23)
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp20, xmask)
    tl.store(in_out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp29 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp32 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 - tmp20
        tmp28 = tmp27 * tmp24
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp35, rmask & xmask)
    tmp36_load = tl.load(seed8 + (0))
    tmp36 = tl.broadcast_to(tmp36_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp42 = tl.load(out_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp37 = r2 + (768*x3)
        tmp38 = tl.rand(tmp36, tmp37)
        tmp39 = 0.1
        tmp40 = tmp38 > tmp39
        tmp41 = tmp40.to(tl.float32)
        tmp43 = tmp41 * tmp42
        tmp44 = 1.1111111111111112
        tmp45 = tmp43 * tmp44
        tl.store(out_ptr3 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, out_ptr0, out_ptr1, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 12288
    ynumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    y2 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*y2) + (393216*x1)), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (y2 + (512*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y2 + (512*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 6291456 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__5(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 56623104 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp10 = r1 + (768*x0)
        tmp11 = tl.rand(tmp0, tmp10)
        tmp12 = tmp11 > tmp3
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 * tmp14
        tmp16 = tmp15 * tmp8
        tmp17 = tmp9 + tmp16
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp17, rmask & xmask)
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    _tmp27 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp28 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = 768.0
        tmp24 = tmp20 / tmp23
        tmp25 = tmp22 - tmp24
        tmp26 = tmp25 * tmp25
        _tmp27 = tl.where(xmask & rmask, _tmp27 + tmp26, _tmp27)
        _tmp28 = tl.where(xmask & rmask, _tmp28 + tmp22, _tmp28)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp29 = 768.0
    tmp30 = tmp28 / tmp29
    tmp31 = tmp27 / tmp29
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = tl.libdevice.rsqrt(tmp33)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp30, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp34, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp35 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp42 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp36 - tmp30
        tmp38 = tmp37 * tmp34
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp38 * tmp40
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp41 + tmp43
        tmp45 = tmp44.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp45, rmask & xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__7(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 62914560 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 69206016 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__9(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 119537664 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__10(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 125829120 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 132120576 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__12(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 182452224 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__13(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 188743680 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__14(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 195035136 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__15(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 245366784 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__16(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 251658240 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__17(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 257949696 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__18(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 308281344 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__19(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 314572800 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__20(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 320864256 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__21(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 371195904 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__22(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 377487360 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__23(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 383778816 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__24(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 434110464 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__25(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 440401920 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__26(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 446693376 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__27(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 497025024 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__28(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 503316480 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__29(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 509607936 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__30(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 559939584 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__31(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 566231040 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__32(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 572522496 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__33(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 622854144 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__34(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 629145600 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__35(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 635437056 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__36(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 685768704 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__37(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 692060160 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton__38(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp6 = -65504.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.max(_tmp10, 1)[:, None]
    _tmp23 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1.0
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp17 = -65504.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        _tmp23 = tl.where(xmask & rmask, _tmp23 + tmp22, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = 8.0
        tmp26 = tmp24 / tmp25
        tmp27 = 1.0
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp27 - tmp28
        tmp30 = -65504.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp10
        tmp35 = tl.exp(tmp34)
        tmp36 = tmp35 / tmp23
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask & xmask)
    tmp38_load = tl.load(seed1 + (0))
    tmp38 = tl.broadcast_to(tmp38_load, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp44 = tl.load(out_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp39 = 698351616 + r1 + (512*x0)
        tmp40 = tl.rand(tmp38, tmp39)
        tmp41 = 0.1
        tmp42 = tmp40 > tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tmp46 = 1.1111111111111112
        tmp47 = tmp45 * tmp46
        tl.store(out_ptr3 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, rmask & xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(out_ptr3 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr4 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp48, rmask & xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__39(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 748683264 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
''')


triton__40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton__40(in_out_ptr0, in_out_ptr1, in_out_ptr2, seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0_load = tl.load(seed0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = 754974720 + r1 + (768*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 + tmp10
        tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp11, rmask & xmask)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp22 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 768.0
        tmp18 = tmp14 / tmp17
        tmp19 = tmp16 - tmp18
        tmp20 = tmp19 * tmp19
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
        _tmp22 = tl.where(xmask & rmask, _tmp22 + tmp16, _tmp22)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 / tmp23
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.libdevice.rsqrt(tmp27)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
    tl.store(in_out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp24
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp39, rmask & xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton__41(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp10, rmask & xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    _tmp20 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 768.0
        tmp17 = tmp13 / tmp16
        tmp18 = tmp15 - tmp17
        tmp19 = tmp18 * tmp18
        _tmp20 = tl.where(xmask & rmask, _tmp20 + tmp19, _tmp20)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp15, _tmp21)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp22 = 768.0
    tmp23 = tmp21 / tmp22
    tmp24 = tmp20 / tmp22
    tmp25 = 1e-12
    tmp26 = tmp24 + tmp25
    tmp27 = tl.libdevice.rsqrt(tmp26)
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(in_out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp27, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(out_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp32 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp23
        tmp31 = tmp30 * tmp27
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 * tmp33
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp34 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp38, rmask & xmask)
''')


triton__42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206 = args
    args.clear()
    torch.randint(2**31, size=(), dtype=torch.int64, out=seed_cuda_0)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        buf2 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf4 = as_strided(buf3, (16, 512, 1), (512, 1, 1)); del buf3  # reuse
        buf5 = as_strided(buf2, (16, 512, 1), (512, 1, 1)); del buf2  # reuse
        buf6 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        buf7 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        stream0 = get_cuda_stream(0)
        triton__0.run(buf4, buf5, primals_206, primals_1, primals_204, primals_2, primals_205, primals_3, primals_4, primals_5, seed_cuda_0, buf0, buf6, buf7, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        buf8 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_7, buf7, as_strided(primals_6, (768, 768), (1, 768)), alpha=1, beta=1, out=buf8)
        del primals_7
        buf9 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_9, buf7, as_strided(primals_8, (768, 768), (1, 768)), alpha=1, beta=1, out=buf9)
        del primals_9
        buf10 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_11, buf7, as_strided(primals_10, (768, 768), (1, 768)), alpha=1, beta=1, out=buf10)
        del primals_11
        buf11 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf435 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf8, buf11, buf435, 6291456, grid=grid(6291456), stream=stream0)
        buf12 = as_strided(buf8, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf8  # reuse
        buf436 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf9, buf12, buf436, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf13 = empty_strided((192, 512, 512), (262144, 512, 1), device='cuda', dtype=torch.float16)
        extern_kernels.bmm(as_strided(buf11, (192, 512, 64), (32768, 64, 1)), as_strided(buf12, (192, 64, 512), (32768, 512, 1)), out=buf13)
        buf16 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf17 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf433 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__3.run(buf13, seed_cuda_0, buf16, buf17, buf433, 98304, 512, grid=grid(98304), stream=stream0)
        buf18 = as_strided(buf12, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf12  # reuse
        buf434 = as_strided(buf11, (192, 64, 512), (32768, 1, 64)); del buf11  # reuse
        triton__1.run(buf10, buf18, buf434, 6291456, grid=grid(6291456), stream=stream0)
        buf19 = as_strided(buf10, (192, 512, 64), (32768, 64, 1)); del buf10  # reuse
        extern_kernels.bmm(as_strided(buf17, (192, 512, 512), (262144, 512, 1)), as_strided(buf18, (192, 512, 64), (32768, 64, 1)), out=buf19)
        buf20 = as_strided(buf18, (8192, 768), (768, 1)); del buf18  # reuse
        triton__4.run(buf19, buf20, 6291456, grid=grid(6291456), stream=stream0)
        buf21 = as_strided(buf19, (8192, 768), (768, 1)); del buf19  # reuse
        extern_kernels.addmm(primals_13, buf20, as_strided(primals_12, (768, 768), (1, 768)), alpha=1, beta=1, out=buf21)
        del primals_13
        buf22 = as_strided(buf21, (16, 512, 768), (393216, 768, 1)); del buf21  # reuse
        buf24 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf26 = as_strided(buf25, (16, 512, 1), (512, 1, 1)); del buf25  # reuse
        buf27 = as_strided(buf24, (16, 512, 1), (512, 1, 1)); del buf24  # reuse
        buf28 = as_strided(buf9, (16, 512, 768), (393216, 768, 1)); del buf9  # reuse
        triton__5.run(buf22, buf26, buf27, seed_cuda_0, buf6, primals_14, primals_15, buf28, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_15
        buf29 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_17, as_strided(buf28, (8192, 768), (768, 1)), as_strided(primals_16, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf29)
        del primals_17
        buf30 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf29, buf30, 25165824, grid=grid(25165824), stream=stream0)
        buf31 = as_strided(buf6, (8192, 768), (768, 1)); del buf6  # reuse
        extern_kernels.addmm(primals_19, buf30, as_strided(primals_18, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf31)
        del primals_19
        buf32 = as_strided(buf31, (16, 512, 768), (393216, 768, 1)); del buf31  # reuse
        buf34 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf36 = as_strided(buf35, (16, 512, 1), (512, 1, 1)); del buf35  # reuse
        buf37 = as_strided(buf34, (16, 512, 1), (512, 1, 1)); del buf34  # reuse
        buf38 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__7.run(buf32, buf36, buf37, seed_cuda_0, buf28, primals_20, primals_21, buf38, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_21
        buf39 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_23, as_strided(buf38, (8192, 768), (768, 1)), as_strided(primals_22, (768, 768), (1, 768)), alpha=1, beta=1, out=buf39)
        del primals_23
        buf40 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_25, as_strided(buf38, (8192, 768), (768, 1)), as_strided(primals_24, (768, 768), (1, 768)), alpha=1, beta=1, out=buf40)
        del primals_25
        buf41 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_27, as_strided(buf38, (8192, 768), (768, 1)), as_strided(primals_26, (768, 768), (1, 768)), alpha=1, beta=1, out=buf41)
        del primals_27
        buf42 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf431 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf39, buf42, buf431, 6291456, grid=grid(6291456), stream=stream0)
        buf43 = as_strided(buf39, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf39  # reuse
        buf432 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf40, buf43, buf432, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf44 = as_strided(buf17, (192, 512, 512), (262144, 512, 1)); del buf17  # reuse
        extern_kernels.bmm(as_strided(buf42, (192, 512, 64), (32768, 64, 1)), as_strided(buf43, (192, 64, 512), (32768, 512, 1)), out=buf44)
        buf47 = as_strided(buf13, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf13  # reuse
        buf48 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf429 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__8.run(buf44, seed_cuda_0, buf47, buf48, buf429, 98304, 512, grid=grid(98304), stream=stream0)
        buf49 = as_strided(buf43, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf43  # reuse
        buf430 = as_strided(buf42, (192, 64, 512), (32768, 1, 64)); del buf42  # reuse
        triton__1.run(buf41, buf49, buf430, 6291456, grid=grid(6291456), stream=stream0)
        buf50 = as_strided(buf41, (192, 512, 64), (32768, 64, 1)); del buf41  # reuse
        extern_kernels.bmm(as_strided(buf48, (192, 512, 512), (262144, 512, 1)), as_strided(buf49, (192, 512, 64), (32768, 64, 1)), out=buf50)
        buf51 = as_strided(buf49, (8192, 768), (768, 1)); del buf49  # reuse
        triton__4.run(buf50, buf51, 6291456, grid=grid(6291456), stream=stream0)
        buf52 = as_strided(buf50, (8192, 768), (768, 1)); del buf50  # reuse
        extern_kernels.addmm(primals_29, buf51, as_strided(primals_28, (768, 768), (1, 768)), alpha=1, beta=1, out=buf52)
        del primals_29
        buf53 = as_strided(buf52, (16, 512, 768), (393216, 768, 1)); del buf52  # reuse
        buf55 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf57 = as_strided(buf56, (16, 512, 1), (512, 1, 1)); del buf56  # reuse
        buf58 = as_strided(buf55, (16, 512, 1), (512, 1, 1)); del buf55  # reuse
        buf59 = as_strided(buf40, (16, 512, 768), (393216, 768, 1)); del buf40  # reuse
        triton__9.run(buf53, buf57, buf58, seed_cuda_0, buf38, primals_30, primals_31, buf59, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_31
        buf60 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_33, as_strided(buf59, (8192, 768), (768, 1)), as_strided(primals_32, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf60)
        del primals_33
        buf61 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf60, buf61, 25165824, grid=grid(25165824), stream=stream0)
        buf62 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_35, buf61, as_strided(primals_34, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf62)
        del primals_35
        buf63 = as_strided(buf62, (16, 512, 768), (393216, 768, 1)); del buf62  # reuse
        buf65 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf66 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf67 = as_strided(buf66, (16, 512, 1), (512, 1, 1)); del buf66  # reuse
        buf68 = as_strided(buf65, (16, 512, 1), (512, 1, 1)); del buf65  # reuse
        buf69 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__10.run(buf63, buf67, buf68, seed_cuda_0, buf59, primals_36, primals_37, buf69, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_37
        buf70 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_39, as_strided(buf69, (8192, 768), (768, 1)), as_strided(primals_38, (768, 768), (1, 768)), alpha=1, beta=1, out=buf70)
        del primals_39
        buf71 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_41, as_strided(buf69, (8192, 768), (768, 1)), as_strided(primals_40, (768, 768), (1, 768)), alpha=1, beta=1, out=buf71)
        del primals_41
        buf72 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_43, as_strided(buf69, (8192, 768), (768, 1)), as_strided(primals_42, (768, 768), (1, 768)), alpha=1, beta=1, out=buf72)
        del primals_43
        buf73 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf427 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf70, buf73, buf427, 6291456, grid=grid(6291456), stream=stream0)
        buf74 = as_strided(buf70, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf70  # reuse
        buf428 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf71, buf74, buf428, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf75 = as_strided(buf48, (192, 512, 512), (262144, 512, 1)); del buf48  # reuse
        extern_kernels.bmm(as_strided(buf73, (192, 512, 64), (32768, 64, 1)), as_strided(buf74, (192, 64, 512), (32768, 512, 1)), out=buf75)
        buf78 = as_strided(buf44, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf44  # reuse
        buf79 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf425 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__11.run(buf75, seed_cuda_0, buf78, buf79, buf425, 98304, 512, grid=grid(98304), stream=stream0)
        buf80 = as_strided(buf74, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf74  # reuse
        buf426 = as_strided(buf73, (192, 64, 512), (32768, 1, 64)); del buf73  # reuse
        triton__1.run(buf72, buf80, buf426, 6291456, grid=grid(6291456), stream=stream0)
        buf81 = as_strided(buf72, (192, 512, 64), (32768, 64, 1)); del buf72  # reuse
        extern_kernels.bmm(as_strided(buf79, (192, 512, 512), (262144, 512, 1)), as_strided(buf80, (192, 512, 64), (32768, 64, 1)), out=buf81)
        buf82 = as_strided(buf80, (8192, 768), (768, 1)); del buf80  # reuse
        triton__4.run(buf81, buf82, 6291456, grid=grid(6291456), stream=stream0)
        buf83 = as_strided(buf81, (8192, 768), (768, 1)); del buf81  # reuse
        extern_kernels.addmm(primals_45, buf82, as_strided(primals_44, (768, 768), (1, 768)), alpha=1, beta=1, out=buf83)
        del primals_45
        buf84 = as_strided(buf83, (16, 512, 768), (393216, 768, 1)); del buf83  # reuse
        buf86 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf88 = as_strided(buf87, (16, 512, 1), (512, 1, 1)); del buf87  # reuse
        buf89 = as_strided(buf86, (16, 512, 1), (512, 1, 1)); del buf86  # reuse
        buf90 = as_strided(buf71, (16, 512, 768), (393216, 768, 1)); del buf71  # reuse
        triton__12.run(buf84, buf88, buf89, seed_cuda_0, buf69, primals_46, primals_47, buf90, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_47
        buf91 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_49, as_strided(buf90, (8192, 768), (768, 1)), as_strided(primals_48, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf91)
        del primals_49
        buf92 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf91, buf92, 25165824, grid=grid(25165824), stream=stream0)
        buf93 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_51, buf92, as_strided(primals_50, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf93)
        del primals_51
        buf94 = as_strided(buf93, (16, 512, 768), (393216, 768, 1)); del buf93  # reuse
        buf96 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf97 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf98 = as_strided(buf97, (16, 512, 1), (512, 1, 1)); del buf97  # reuse
        buf99 = as_strided(buf96, (16, 512, 1), (512, 1, 1)); del buf96  # reuse
        buf100 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__13.run(buf94, buf98, buf99, seed_cuda_0, buf90, primals_52, primals_53, buf100, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_53
        buf101 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_55, as_strided(buf100, (8192, 768), (768, 1)), as_strided(primals_54, (768, 768), (1, 768)), alpha=1, beta=1, out=buf101)
        del primals_55
        buf102 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_57, as_strided(buf100, (8192, 768), (768, 1)), as_strided(primals_56, (768, 768), (1, 768)), alpha=1, beta=1, out=buf102)
        del primals_57
        buf103 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_59, as_strided(buf100, (8192, 768), (768, 1)), as_strided(primals_58, (768, 768), (1, 768)), alpha=1, beta=1, out=buf103)
        del primals_59
        buf104 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf423 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf101, buf104, buf423, 6291456, grid=grid(6291456), stream=stream0)
        buf105 = as_strided(buf101, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf101  # reuse
        buf424 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf102, buf105, buf424, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf106 = as_strided(buf79, (192, 512, 512), (262144, 512, 1)); del buf79  # reuse
        extern_kernels.bmm(as_strided(buf104, (192, 512, 64), (32768, 64, 1)), as_strided(buf105, (192, 64, 512), (32768, 512, 1)), out=buf106)
        buf109 = as_strided(buf75, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf75  # reuse
        buf110 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf421 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__14.run(buf106, seed_cuda_0, buf109, buf110, buf421, 98304, 512, grid=grid(98304), stream=stream0)
        buf111 = as_strided(buf105, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf105  # reuse
        buf422 = as_strided(buf104, (192, 64, 512), (32768, 1, 64)); del buf104  # reuse
        triton__1.run(buf103, buf111, buf422, 6291456, grid=grid(6291456), stream=stream0)
        buf112 = as_strided(buf103, (192, 512, 64), (32768, 64, 1)); del buf103  # reuse
        extern_kernels.bmm(as_strided(buf110, (192, 512, 512), (262144, 512, 1)), as_strided(buf111, (192, 512, 64), (32768, 64, 1)), out=buf112)
        buf113 = as_strided(buf111, (8192, 768), (768, 1)); del buf111  # reuse
        triton__4.run(buf112, buf113, 6291456, grid=grid(6291456), stream=stream0)
        buf114 = as_strided(buf112, (8192, 768), (768, 1)); del buf112  # reuse
        extern_kernels.addmm(primals_61, buf113, as_strided(primals_60, (768, 768), (1, 768)), alpha=1, beta=1, out=buf114)
        del primals_61
        buf115 = as_strided(buf114, (16, 512, 768), (393216, 768, 1)); del buf114  # reuse
        buf117 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf119 = as_strided(buf118, (16, 512, 1), (512, 1, 1)); del buf118  # reuse
        buf120 = as_strided(buf117, (16, 512, 1), (512, 1, 1)); del buf117  # reuse
        buf121 = as_strided(buf102, (16, 512, 768), (393216, 768, 1)); del buf102  # reuse
        triton__15.run(buf115, buf119, buf120, seed_cuda_0, buf100, primals_62, primals_63, buf121, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_63
        buf122 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_65, as_strided(buf121, (8192, 768), (768, 1)), as_strided(primals_64, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf122)
        del primals_65
        buf123 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf122, buf123, 25165824, grid=grid(25165824), stream=stream0)
        buf124 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_67, buf123, as_strided(primals_66, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf124)
        del primals_67
        buf125 = as_strided(buf124, (16, 512, 768), (393216, 768, 1)); del buf124  # reuse
        buf127 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf128 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf129 = as_strided(buf128, (16, 512, 1), (512, 1, 1)); del buf128  # reuse
        buf130 = as_strided(buf127, (16, 512, 1), (512, 1, 1)); del buf127  # reuse
        buf131 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__16.run(buf125, buf129, buf130, seed_cuda_0, buf121, primals_68, primals_69, buf131, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_69
        buf132 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_71, as_strided(buf131, (8192, 768), (768, 1)), as_strided(primals_70, (768, 768), (1, 768)), alpha=1, beta=1, out=buf132)
        del primals_71
        buf133 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_73, as_strided(buf131, (8192, 768), (768, 1)), as_strided(primals_72, (768, 768), (1, 768)), alpha=1, beta=1, out=buf133)
        del primals_73
        buf134 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_75, as_strided(buf131, (8192, 768), (768, 1)), as_strided(primals_74, (768, 768), (1, 768)), alpha=1, beta=1, out=buf134)
        del primals_75
        buf135 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf419 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf132, buf135, buf419, 6291456, grid=grid(6291456), stream=stream0)
        buf136 = as_strided(buf132, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf132  # reuse
        buf420 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf133, buf136, buf420, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf137 = as_strided(buf110, (192, 512, 512), (262144, 512, 1)); del buf110  # reuse
        extern_kernels.bmm(as_strided(buf135, (192, 512, 64), (32768, 64, 1)), as_strided(buf136, (192, 64, 512), (32768, 512, 1)), out=buf137)
        buf140 = as_strided(buf106, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf106  # reuse
        buf141 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf417 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__17.run(buf137, seed_cuda_0, buf140, buf141, buf417, 98304, 512, grid=grid(98304), stream=stream0)
        buf142 = as_strided(buf136, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf136  # reuse
        buf418 = as_strided(buf135, (192, 64, 512), (32768, 1, 64)); del buf135  # reuse
        triton__1.run(buf134, buf142, buf418, 6291456, grid=grid(6291456), stream=stream0)
        buf143 = as_strided(buf134, (192, 512, 64), (32768, 64, 1)); del buf134  # reuse
        extern_kernels.bmm(as_strided(buf141, (192, 512, 512), (262144, 512, 1)), as_strided(buf142, (192, 512, 64), (32768, 64, 1)), out=buf143)
        buf144 = as_strided(buf142, (8192, 768), (768, 1)); del buf142  # reuse
        triton__4.run(buf143, buf144, 6291456, grid=grid(6291456), stream=stream0)
        buf145 = as_strided(buf143, (8192, 768), (768, 1)); del buf143  # reuse
        extern_kernels.addmm(primals_77, buf144, as_strided(primals_76, (768, 768), (1, 768)), alpha=1, beta=1, out=buf145)
        del primals_77
        buf146 = as_strided(buf145, (16, 512, 768), (393216, 768, 1)); del buf145  # reuse
        buf148 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf149 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf150 = as_strided(buf149, (16, 512, 1), (512, 1, 1)); del buf149  # reuse
        buf151 = as_strided(buf148, (16, 512, 1), (512, 1, 1)); del buf148  # reuse
        buf152 = as_strided(buf133, (16, 512, 768), (393216, 768, 1)); del buf133  # reuse
        triton__18.run(buf146, buf150, buf151, seed_cuda_0, buf131, primals_78, primals_79, buf152, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_79
        buf153 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_81, as_strided(buf152, (8192, 768), (768, 1)), as_strided(primals_80, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf153)
        del primals_81
        buf154 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf153, buf154, 25165824, grid=grid(25165824), stream=stream0)
        buf155 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_83, buf154, as_strided(primals_82, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf155)
        del primals_83
        buf156 = as_strided(buf155, (16, 512, 768), (393216, 768, 1)); del buf155  # reuse
        buf158 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf159 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf160 = as_strided(buf159, (16, 512, 1), (512, 1, 1)); del buf159  # reuse
        buf161 = as_strided(buf158, (16, 512, 1), (512, 1, 1)); del buf158  # reuse
        buf162 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__19.run(buf156, buf160, buf161, seed_cuda_0, buf152, primals_84, primals_85, buf162, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_85
        buf163 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_87, as_strided(buf162, (8192, 768), (768, 1)), as_strided(primals_86, (768, 768), (1, 768)), alpha=1, beta=1, out=buf163)
        del primals_87
        buf164 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_89, as_strided(buf162, (8192, 768), (768, 1)), as_strided(primals_88, (768, 768), (1, 768)), alpha=1, beta=1, out=buf164)
        del primals_89
        buf165 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_91, as_strided(buf162, (8192, 768), (768, 1)), as_strided(primals_90, (768, 768), (1, 768)), alpha=1, beta=1, out=buf165)
        del primals_91
        buf166 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf415 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf163, buf166, buf415, 6291456, grid=grid(6291456), stream=stream0)
        buf167 = as_strided(buf163, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf163  # reuse
        buf416 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf164, buf167, buf416, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf168 = as_strided(buf141, (192, 512, 512), (262144, 512, 1)); del buf141  # reuse
        extern_kernels.bmm(as_strided(buf166, (192, 512, 64), (32768, 64, 1)), as_strided(buf167, (192, 64, 512), (32768, 512, 1)), out=buf168)
        buf171 = as_strided(buf137, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf137  # reuse
        buf172 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf413 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__20.run(buf168, seed_cuda_0, buf171, buf172, buf413, 98304, 512, grid=grid(98304), stream=stream0)
        buf173 = as_strided(buf167, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf167  # reuse
        buf414 = as_strided(buf166, (192, 64, 512), (32768, 1, 64)); del buf166  # reuse
        triton__1.run(buf165, buf173, buf414, 6291456, grid=grid(6291456), stream=stream0)
        buf174 = as_strided(buf165, (192, 512, 64), (32768, 64, 1)); del buf165  # reuse
        extern_kernels.bmm(as_strided(buf172, (192, 512, 512), (262144, 512, 1)), as_strided(buf173, (192, 512, 64), (32768, 64, 1)), out=buf174)
        buf175 = as_strided(buf173, (8192, 768), (768, 1)); del buf173  # reuse
        triton__4.run(buf174, buf175, 6291456, grid=grid(6291456), stream=stream0)
        buf176 = as_strided(buf174, (8192, 768), (768, 1)); del buf174  # reuse
        extern_kernels.addmm(primals_93, buf175, as_strided(primals_92, (768, 768), (1, 768)), alpha=1, beta=1, out=buf176)
        del primals_93
        buf177 = as_strided(buf176, (16, 512, 768), (393216, 768, 1)); del buf176  # reuse
        buf179 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf181 = as_strided(buf180, (16, 512, 1), (512, 1, 1)); del buf180  # reuse
        buf182 = as_strided(buf179, (16, 512, 1), (512, 1, 1)); del buf179  # reuse
        buf183 = as_strided(buf164, (16, 512, 768), (393216, 768, 1)); del buf164  # reuse
        triton__21.run(buf177, buf181, buf182, seed_cuda_0, buf162, primals_94, primals_95, buf183, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_95
        buf184 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_97, as_strided(buf183, (8192, 768), (768, 1)), as_strided(primals_96, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf184)
        del primals_97
        buf185 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf184, buf185, 25165824, grid=grid(25165824), stream=stream0)
        buf186 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_99, buf185, as_strided(primals_98, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf186)
        del primals_99
        buf187 = as_strided(buf186, (16, 512, 768), (393216, 768, 1)); del buf186  # reuse
        buf189 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf190 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf191 = as_strided(buf190, (16, 512, 1), (512, 1, 1)); del buf190  # reuse
        buf192 = as_strided(buf189, (16, 512, 1), (512, 1, 1)); del buf189  # reuse
        buf193 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__22.run(buf187, buf191, buf192, seed_cuda_0, buf183, primals_100, primals_101, buf193, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_101
        buf194 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_103, as_strided(buf193, (8192, 768), (768, 1)), as_strided(primals_102, (768, 768), (1, 768)), alpha=1, beta=1, out=buf194)
        del primals_103
        buf195 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_105, as_strided(buf193, (8192, 768), (768, 1)), as_strided(primals_104, (768, 768), (1, 768)), alpha=1, beta=1, out=buf195)
        del primals_105
        buf196 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_107, as_strided(buf193, (8192, 768), (768, 1)), as_strided(primals_106, (768, 768), (1, 768)), alpha=1, beta=1, out=buf196)
        del primals_107
        buf197 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf411 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf194, buf197, buf411, 6291456, grid=grid(6291456), stream=stream0)
        buf198 = as_strided(buf194, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf194  # reuse
        buf412 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf195, buf198, buf412, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf199 = as_strided(buf172, (192, 512, 512), (262144, 512, 1)); del buf172  # reuse
        extern_kernels.bmm(as_strided(buf197, (192, 512, 64), (32768, 64, 1)), as_strided(buf198, (192, 64, 512), (32768, 512, 1)), out=buf199)
        buf202 = as_strided(buf168, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf168  # reuse
        buf203 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf409 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__23.run(buf199, seed_cuda_0, buf202, buf203, buf409, 98304, 512, grid=grid(98304), stream=stream0)
        buf204 = as_strided(buf198, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf198  # reuse
        buf410 = as_strided(buf197, (192, 64, 512), (32768, 1, 64)); del buf197  # reuse
        triton__1.run(buf196, buf204, buf410, 6291456, grid=grid(6291456), stream=stream0)
        buf205 = as_strided(buf196, (192, 512, 64), (32768, 64, 1)); del buf196  # reuse
        extern_kernels.bmm(as_strided(buf203, (192, 512, 512), (262144, 512, 1)), as_strided(buf204, (192, 512, 64), (32768, 64, 1)), out=buf205)
        buf206 = as_strided(buf204, (8192, 768), (768, 1)); del buf204  # reuse
        triton__4.run(buf205, buf206, 6291456, grid=grid(6291456), stream=stream0)
        buf207 = as_strided(buf205, (8192, 768), (768, 1)); del buf205  # reuse
        extern_kernels.addmm(primals_109, buf206, as_strided(primals_108, (768, 768), (1, 768)), alpha=1, beta=1, out=buf207)
        del primals_109
        buf208 = as_strided(buf207, (16, 512, 768), (393216, 768, 1)); del buf207  # reuse
        buf210 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf211 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf212 = as_strided(buf211, (16, 512, 1), (512, 1, 1)); del buf211  # reuse
        buf213 = as_strided(buf210, (16, 512, 1), (512, 1, 1)); del buf210  # reuse
        buf214 = as_strided(buf195, (16, 512, 768), (393216, 768, 1)); del buf195  # reuse
        triton__24.run(buf208, buf212, buf213, seed_cuda_0, buf193, primals_110, primals_111, buf214, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_111
        buf215 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_113, as_strided(buf214, (8192, 768), (768, 1)), as_strided(primals_112, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf215)
        del primals_113
        buf216 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf215, buf216, 25165824, grid=grid(25165824), stream=stream0)
        buf217 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_115, buf216, as_strided(primals_114, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf217)
        del primals_115
        buf218 = as_strided(buf217, (16, 512, 768), (393216, 768, 1)); del buf217  # reuse
        buf220 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf221 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf222 = as_strided(buf221, (16, 512, 1), (512, 1, 1)); del buf221  # reuse
        buf223 = as_strided(buf220, (16, 512, 1), (512, 1, 1)); del buf220  # reuse
        buf224 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__25.run(buf218, buf222, buf223, seed_cuda_0, buf214, primals_116, primals_117, buf224, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_117
        buf225 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_119, as_strided(buf224, (8192, 768), (768, 1)), as_strided(primals_118, (768, 768), (1, 768)), alpha=1, beta=1, out=buf225)
        del primals_119
        buf226 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_121, as_strided(buf224, (8192, 768), (768, 1)), as_strided(primals_120, (768, 768), (1, 768)), alpha=1, beta=1, out=buf226)
        del primals_121
        buf227 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_123, as_strided(buf224, (8192, 768), (768, 1)), as_strided(primals_122, (768, 768), (1, 768)), alpha=1, beta=1, out=buf227)
        del primals_123
        buf228 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf407 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf225, buf228, buf407, 6291456, grid=grid(6291456), stream=stream0)
        buf229 = as_strided(buf225, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf225  # reuse
        buf408 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf226, buf229, buf408, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf230 = as_strided(buf203, (192, 512, 512), (262144, 512, 1)); del buf203  # reuse
        extern_kernels.bmm(as_strided(buf228, (192, 512, 64), (32768, 64, 1)), as_strided(buf229, (192, 64, 512), (32768, 512, 1)), out=buf230)
        buf233 = as_strided(buf199, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf199  # reuse
        buf234 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf405 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__26.run(buf230, seed_cuda_0, buf233, buf234, buf405, 98304, 512, grid=grid(98304), stream=stream0)
        buf235 = as_strided(buf229, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf229  # reuse
        buf406 = as_strided(buf228, (192, 64, 512), (32768, 1, 64)); del buf228  # reuse
        triton__1.run(buf227, buf235, buf406, 6291456, grid=grid(6291456), stream=stream0)
        buf236 = as_strided(buf227, (192, 512, 64), (32768, 64, 1)); del buf227  # reuse
        extern_kernels.bmm(as_strided(buf234, (192, 512, 512), (262144, 512, 1)), as_strided(buf235, (192, 512, 64), (32768, 64, 1)), out=buf236)
        buf237 = as_strided(buf235, (8192, 768), (768, 1)); del buf235  # reuse
        triton__4.run(buf236, buf237, 6291456, grid=grid(6291456), stream=stream0)
        buf238 = as_strided(buf236, (8192, 768), (768, 1)); del buf236  # reuse
        extern_kernels.addmm(primals_125, buf237, as_strided(primals_124, (768, 768), (1, 768)), alpha=1, beta=1, out=buf238)
        del primals_125
        buf239 = as_strided(buf238, (16, 512, 768), (393216, 768, 1)); del buf238  # reuse
        buf241 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf242 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf243 = as_strided(buf242, (16, 512, 1), (512, 1, 1)); del buf242  # reuse
        buf244 = as_strided(buf241, (16, 512, 1), (512, 1, 1)); del buf241  # reuse
        buf245 = as_strided(buf226, (16, 512, 768), (393216, 768, 1)); del buf226  # reuse
        triton__27.run(buf239, buf243, buf244, seed_cuda_0, buf224, primals_126, primals_127, buf245, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_127
        buf246 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_129, as_strided(buf245, (8192, 768), (768, 1)), as_strided(primals_128, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf246)
        del primals_129
        buf247 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf246, buf247, 25165824, grid=grid(25165824), stream=stream0)
        buf248 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_131, buf247, as_strided(primals_130, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf248)
        del primals_131
        buf249 = as_strided(buf248, (16, 512, 768), (393216, 768, 1)); del buf248  # reuse
        buf251 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf252 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf253 = as_strided(buf252, (16, 512, 1), (512, 1, 1)); del buf252  # reuse
        buf254 = as_strided(buf251, (16, 512, 1), (512, 1, 1)); del buf251  # reuse
        buf255 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__28.run(buf249, buf253, buf254, seed_cuda_0, buf245, primals_132, primals_133, buf255, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_133
        buf256 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_135, as_strided(buf255, (8192, 768), (768, 1)), as_strided(primals_134, (768, 768), (1, 768)), alpha=1, beta=1, out=buf256)
        del primals_135
        buf257 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_137, as_strided(buf255, (8192, 768), (768, 1)), as_strided(primals_136, (768, 768), (1, 768)), alpha=1, beta=1, out=buf257)
        del primals_137
        buf258 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_139, as_strided(buf255, (8192, 768), (768, 1)), as_strided(primals_138, (768, 768), (1, 768)), alpha=1, beta=1, out=buf258)
        del primals_139
        buf259 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf403 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf256, buf259, buf403, 6291456, grid=grid(6291456), stream=stream0)
        buf260 = as_strided(buf256, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf256  # reuse
        buf404 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf257, buf260, buf404, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf261 = as_strided(buf234, (192, 512, 512), (262144, 512, 1)); del buf234  # reuse
        extern_kernels.bmm(as_strided(buf259, (192, 512, 64), (32768, 64, 1)), as_strided(buf260, (192, 64, 512), (32768, 512, 1)), out=buf261)
        buf264 = as_strided(buf230, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf230  # reuse
        buf265 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf401 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__29.run(buf261, seed_cuda_0, buf264, buf265, buf401, 98304, 512, grid=grid(98304), stream=stream0)
        buf266 = as_strided(buf260, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf260  # reuse
        buf402 = as_strided(buf259, (192, 64, 512), (32768, 1, 64)); del buf259  # reuse
        triton__1.run(buf258, buf266, buf402, 6291456, grid=grid(6291456), stream=stream0)
        buf267 = as_strided(buf258, (192, 512, 64), (32768, 64, 1)); del buf258  # reuse
        extern_kernels.bmm(as_strided(buf265, (192, 512, 512), (262144, 512, 1)), as_strided(buf266, (192, 512, 64), (32768, 64, 1)), out=buf267)
        buf268 = as_strided(buf266, (8192, 768), (768, 1)); del buf266  # reuse
        triton__4.run(buf267, buf268, 6291456, grid=grid(6291456), stream=stream0)
        buf269 = as_strided(buf267, (8192, 768), (768, 1)); del buf267  # reuse
        extern_kernels.addmm(primals_141, buf268, as_strided(primals_140, (768, 768), (1, 768)), alpha=1, beta=1, out=buf269)
        del primals_141
        buf270 = as_strided(buf269, (16, 512, 768), (393216, 768, 1)); del buf269  # reuse
        buf272 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf273 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf274 = as_strided(buf273, (16, 512, 1), (512, 1, 1)); del buf273  # reuse
        buf275 = as_strided(buf272, (16, 512, 1), (512, 1, 1)); del buf272  # reuse
        buf276 = as_strided(buf257, (16, 512, 768), (393216, 768, 1)); del buf257  # reuse
        triton__30.run(buf270, buf274, buf275, seed_cuda_0, buf255, primals_142, primals_143, buf276, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_143
        buf277 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_145, as_strided(buf276, (8192, 768), (768, 1)), as_strided(primals_144, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf277)
        del primals_145
        buf278 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf277, buf278, 25165824, grid=grid(25165824), stream=stream0)
        buf279 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_147, buf278, as_strided(primals_146, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf279)
        del primals_147
        buf280 = as_strided(buf279, (16, 512, 768), (393216, 768, 1)); del buf279  # reuse
        buf282 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf283 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf284 = as_strided(buf283, (16, 512, 1), (512, 1, 1)); del buf283  # reuse
        buf285 = as_strided(buf282, (16, 512, 1), (512, 1, 1)); del buf282  # reuse
        buf286 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__31.run(buf280, buf284, buf285, seed_cuda_0, buf276, primals_148, primals_149, buf286, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_149
        buf287 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_151, as_strided(buf286, (8192, 768), (768, 1)), as_strided(primals_150, (768, 768), (1, 768)), alpha=1, beta=1, out=buf287)
        del primals_151
        buf288 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_153, as_strided(buf286, (8192, 768), (768, 1)), as_strided(primals_152, (768, 768), (1, 768)), alpha=1, beta=1, out=buf288)
        del primals_153
        buf289 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_155, as_strided(buf286, (8192, 768), (768, 1)), as_strided(primals_154, (768, 768), (1, 768)), alpha=1, beta=1, out=buf289)
        del primals_155
        buf290 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf399 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf287, buf290, buf399, 6291456, grid=grid(6291456), stream=stream0)
        buf291 = as_strided(buf287, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf287  # reuse
        buf400 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf288, buf291, buf400, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf292 = as_strided(buf265, (192, 512, 512), (262144, 512, 1)); del buf265  # reuse
        extern_kernels.bmm(as_strided(buf290, (192, 512, 64), (32768, 64, 1)), as_strided(buf291, (192, 64, 512), (32768, 512, 1)), out=buf292)
        buf295 = as_strided(buf261, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf261  # reuse
        buf296 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf397 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__32.run(buf292, seed_cuda_0, buf295, buf296, buf397, 98304, 512, grid=grid(98304), stream=stream0)
        buf297 = as_strided(buf291, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf291  # reuse
        buf398 = as_strided(buf290, (192, 64, 512), (32768, 1, 64)); del buf290  # reuse
        triton__1.run(buf289, buf297, buf398, 6291456, grid=grid(6291456), stream=stream0)
        buf298 = as_strided(buf289, (192, 512, 64), (32768, 64, 1)); del buf289  # reuse
        extern_kernels.bmm(as_strided(buf296, (192, 512, 512), (262144, 512, 1)), as_strided(buf297, (192, 512, 64), (32768, 64, 1)), out=buf298)
        buf299 = as_strided(buf297, (8192, 768), (768, 1)); del buf297  # reuse
        triton__4.run(buf298, buf299, 6291456, grid=grid(6291456), stream=stream0)
        buf300 = as_strided(buf298, (8192, 768), (768, 1)); del buf298  # reuse
        extern_kernels.addmm(primals_157, buf299, as_strided(primals_156, (768, 768), (1, 768)), alpha=1, beta=1, out=buf300)
        del primals_157
        buf301 = as_strided(buf300, (16, 512, 768), (393216, 768, 1)); del buf300  # reuse
        buf303 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf304 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf305 = as_strided(buf304, (16, 512, 1), (512, 1, 1)); del buf304  # reuse
        buf306 = as_strided(buf303, (16, 512, 1), (512, 1, 1)); del buf303  # reuse
        buf307 = as_strided(buf288, (16, 512, 768), (393216, 768, 1)); del buf288  # reuse
        triton__33.run(buf301, buf305, buf306, seed_cuda_0, buf286, primals_158, primals_159, buf307, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_159
        buf308 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_161, as_strided(buf307, (8192, 768), (768, 1)), as_strided(primals_160, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf308)
        del primals_161
        buf309 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf308, buf309, 25165824, grid=grid(25165824), stream=stream0)
        buf310 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_163, buf309, as_strided(primals_162, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf310)
        del primals_163
        buf311 = as_strided(buf310, (16, 512, 768), (393216, 768, 1)); del buf310  # reuse
        buf313 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf314 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf315 = as_strided(buf314, (16, 512, 1), (512, 1, 1)); del buf314  # reuse
        buf316 = as_strided(buf313, (16, 512, 1), (512, 1, 1)); del buf313  # reuse
        buf317 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__34.run(buf311, buf315, buf316, seed_cuda_0, buf307, primals_164, primals_165, buf317, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_165
        buf318 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_167, as_strided(buf317, (8192, 768), (768, 1)), as_strided(primals_166, (768, 768), (1, 768)), alpha=1, beta=1, out=buf318)
        del primals_167
        buf319 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_169, as_strided(buf317, (8192, 768), (768, 1)), as_strided(primals_168, (768, 768), (1, 768)), alpha=1, beta=1, out=buf319)
        del primals_169
        buf320 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_171, as_strided(buf317, (8192, 768), (768, 1)), as_strided(primals_170, (768, 768), (1, 768)), alpha=1, beta=1, out=buf320)
        del primals_171
        buf321 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf395 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf318, buf321, buf395, 6291456, grid=grid(6291456), stream=stream0)
        buf322 = as_strided(buf318, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf318  # reuse
        buf396 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf319, buf322, buf396, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf323 = as_strided(buf296, (192, 512, 512), (262144, 512, 1)); del buf296  # reuse
        extern_kernels.bmm(as_strided(buf321, (192, 512, 64), (32768, 64, 1)), as_strided(buf322, (192, 64, 512), (32768, 512, 1)), out=buf323)
        buf326 = as_strided(buf292, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf292  # reuse
        buf327 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf393 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__35.run(buf323, seed_cuda_0, buf326, buf327, buf393, 98304, 512, grid=grid(98304), stream=stream0)
        buf328 = as_strided(buf322, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf322  # reuse
        buf394 = as_strided(buf321, (192, 64, 512), (32768, 1, 64)); del buf321  # reuse
        triton__1.run(buf320, buf328, buf394, 6291456, grid=grid(6291456), stream=stream0)
        buf329 = as_strided(buf320, (192, 512, 64), (32768, 64, 1)); del buf320  # reuse
        extern_kernels.bmm(as_strided(buf327, (192, 512, 512), (262144, 512, 1)), as_strided(buf328, (192, 512, 64), (32768, 64, 1)), out=buf329)
        buf330 = as_strided(buf328, (8192, 768), (768, 1)); del buf328  # reuse
        triton__4.run(buf329, buf330, 6291456, grid=grid(6291456), stream=stream0)
        buf331 = as_strided(buf329, (8192, 768), (768, 1)); del buf329  # reuse
        extern_kernels.addmm(primals_173, buf330, as_strided(primals_172, (768, 768), (1, 768)), alpha=1, beta=1, out=buf331)
        del primals_173
        buf332 = as_strided(buf331, (16, 512, 768), (393216, 768, 1)); del buf331  # reuse
        buf334 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf335 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf336 = as_strided(buf335, (16, 512, 1), (512, 1, 1)); del buf335  # reuse
        buf337 = as_strided(buf334, (16, 512, 1), (512, 1, 1)); del buf334  # reuse
        buf338 = as_strided(buf319, (16, 512, 768), (393216, 768, 1)); del buf319  # reuse
        triton__36.run(buf332, buf336, buf337, seed_cuda_0, buf317, primals_174, primals_175, buf338, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_175
        buf339 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_177, as_strided(buf338, (8192, 768), (768, 1)), as_strided(primals_176, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf339)
        del primals_177
        buf340 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf339, buf340, 25165824, grid=grid(25165824), stream=stream0)
        buf341 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_179, buf340, as_strided(primals_178, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf341)
        del primals_179
        buf342 = as_strided(buf341, (16, 512, 768), (393216, 768, 1)); del buf341  # reuse
        buf344 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf345 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf346 = as_strided(buf345, (16, 512, 1), (512, 1, 1)); del buf345  # reuse
        buf347 = as_strided(buf344, (16, 512, 1), (512, 1, 1)); del buf344  # reuse
        buf348 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        triton__37.run(buf342, buf346, buf347, seed_cuda_0, buf338, primals_180, primals_181, buf348, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_181
        buf349 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_183, as_strided(buf348, (8192, 768), (768, 1)), as_strided(primals_182, (768, 768), (1, 768)), alpha=1, beta=1, out=buf349)
        del primals_183
        buf350 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_185, as_strided(buf348, (8192, 768), (768, 1)), as_strided(primals_184, (768, 768), (1, 768)), alpha=1, beta=1, out=buf350)
        del primals_185
        buf351 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_187, as_strided(buf348, (8192, 768), (768, 1)), as_strided(primals_186, (768, 768), (1, 768)), alpha=1, beta=1, out=buf351)
        del primals_187
        buf352 = empty_strided((16, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float16)
        buf391 = empty_strided((192, 64, 512), (32768, 1, 64), device='cuda', dtype=torch.float16)
        triton__1.run(buf349, buf352, buf391, 6291456, grid=grid(6291456), stream=stream0)
        buf353 = as_strided(buf349, (16, 12, 64, 512), (393216, 32768, 512, 1)); del buf349  # reuse
        buf392 = empty_strided((192, 512, 64), (32768, 1, 512), device='cuda', dtype=torch.float16)
        triton__2.run(buf350, buf353, buf392, 12288, 512, grid=grid(12288, 512), stream=stream0)
        buf354 = as_strided(buf327, (192, 512, 512), (262144, 512, 1)); del buf327  # reuse
        extern_kernels.bmm(as_strided(buf352, (192, 512, 64), (32768, 64, 1)), as_strided(buf353, (192, 64, 512), (32768, 512, 1)), out=buf354)
        buf357 = as_strided(buf323, (16, 12, 512, 512), (3145728, 262144, 512, 1)); del buf323  # reuse
        buf358 = empty_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float16)
        buf389 = empty_strided((192, 512, 512), (262144, 1, 512), device='cuda', dtype=torch.float16)
        triton__38.run(buf354, seed_cuda_0, buf357, buf358, buf389, 98304, 512, grid=grid(98304), stream=stream0)
        del buf354
        buf359 = as_strided(buf353, (16, 12, 512, 64), (393216, 32768, 64, 1)); del buf353  # reuse
        buf390 = as_strided(buf352, (192, 64, 512), (32768, 1, 64)); del buf352  # reuse
        triton__1.run(buf351, buf359, buf390, 6291456, grid=grid(6291456), stream=stream0)
        buf360 = as_strided(buf351, (192, 512, 64), (32768, 64, 1)); del buf351  # reuse
        extern_kernels.bmm(as_strided(buf358, (192, 512, 512), (262144, 512, 1)), as_strided(buf359, (192, 512, 64), (32768, 64, 1)), out=buf360)
        del buf358
        buf361 = as_strided(buf359, (8192, 768), (768, 1)); del buf359  # reuse
        triton__4.run(buf360, buf361, 6291456, grid=grid(6291456), stream=stream0)
        buf362 = as_strided(buf360, (8192, 768), (768, 1)); del buf360  # reuse
        extern_kernels.addmm(primals_189, buf361, as_strided(primals_188, (768, 768), (1, 768)), alpha=1, beta=1, out=buf362)
        del primals_189
        buf363 = as_strided(buf362, (16, 512, 768), (393216, 768, 1)); del buf362  # reuse
        buf365 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf366 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf367 = as_strided(buf366, (16, 512, 1), (512, 1, 1)); del buf366  # reuse
        buf368 = as_strided(buf365, (16, 512, 1), (512, 1, 1)); del buf365  # reuse
        buf369 = as_strided(buf350, (16, 512, 768), (393216, 768, 1)); del buf350  # reuse
        triton__39.run(buf363, buf367, buf368, seed_cuda_0, buf348, primals_190, primals_191, buf369, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_191
        buf370 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_193, as_strided(buf369, (8192, 768), (768, 1)), as_strided(primals_192, (768, 3072), (1, 768)), alpha=1, beta=1, out=buf370)
        del primals_193
        buf371 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float16)
        triton__6.run(buf370, buf371, 25165824, grid=grid(25165824), stream=stream0)
        buf372 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_195, buf371, as_strided(primals_194, (3072, 768), (1, 3072)), alpha=1, beta=1, out=buf372)
        del primals_195
        buf373 = as_strided(buf372, (16, 512, 768), (393216, 768, 1)); del buf372  # reuse
        buf375 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf377 = as_strided(buf376, (16, 512, 1), (512, 1, 1)); del buf376  # reuse
        buf378 = as_strided(buf375, (16, 512, 1), (512, 1, 1)); del buf375  # reuse
        buf379 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__40.run(buf373, buf377, buf378, seed_cuda_0, buf369, primals_196, primals_197, buf379, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_197
        buf380 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_199, buf379, as_strided(primals_198, (768, 768), (1, 768)), alpha=1, beta=1, out=buf380)
        del primals_199
        buf381 = empty_strided((16, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float16)
        buf383 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf384 = empty_strided((16, 512, 1), (512, 1, 8192), device='cuda', dtype=torch.float32)
        buf385 = as_strided(buf384, (16, 512, 1), (512, 1, 1)); del buf384  # reuse
        buf386 = as_strided(buf383, (16, 512, 1), (512, 1, 1)); del buf383  # reuse
        buf387 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float16)
        triton__41.run(buf385, buf386, buf380, primals_200, primals_201, buf381, buf387, 8192, 768, grid=grid(8192), stream=stream0)
        del primals_201
        buf388 = empty_strided((8192, 30522), (30522, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(primals_203, buf387, as_strided(primals_202, (768, 30522), (1, 768)), alpha=1, beta=1, out=buf388)
        del primals_203
        buf437 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__42.run(primals_205, buf437, 512, grid=grid(512), stream=stream0)
        del primals_205
        buf438 = empty_strided((16, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__43.run(primals_206, buf438, 8192, grid=grid(8192), stream=stream0)
        del primals_206
        return (as_strided(buf388, (16, 512, 30522), (15627264, 30522, 1)), buf388, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_204, buf0, buf4, buf5, seed_cuda_0.clone(), buf7, buf16, buf20, buf22, buf26, buf27, as_strided(buf28, (8192, 768), (768, 1)), buf29, buf30, buf32, buf36, buf37, as_strided(buf38, (8192, 768), (768, 1)), buf47, buf51, buf53, buf57, buf58, as_strided(buf59, (8192, 768), (768, 1)), buf60, buf61, buf63, buf67, buf68, as_strided(buf69, (8192, 768), (768, 1)), buf78, buf82, buf84, buf88, buf89, as_strided(buf90, (8192, 768), (768, 1)), buf91, buf92, buf94, buf98, buf99, as_strided(buf100, (8192, 768), (768, 1)), buf109, buf113, buf115, buf119, buf120, as_strided(buf121, (8192, 768), (768, 1)), buf122, buf123, buf125, buf129, buf130, as_strided(buf131, (8192, 768), (768, 1)), buf140, buf144, buf146, buf150, buf151, as_strided(buf152, (8192, 768), (768, 1)), buf153, buf154, buf156, buf160, buf161, as_strided(buf162, (8192, 768), (768, 1)), buf171, buf175, buf177, buf181, buf182, as_strided(buf183, (8192, 768), (768, 1)), buf184, buf185, buf187, buf191, buf192, as_strided(buf193, (8192, 768), (768, 1)), buf202, buf206, buf208, buf212, buf213, as_strided(buf214, (8192, 768), (768, 1)), buf215, buf216, buf218, buf222, buf223, as_strided(buf224, (8192, 768), (768, 1)), buf233, buf237, buf239, buf243, buf244, as_strided(buf245, (8192, 768), (768, 1)), buf246, buf247, buf249, buf253, buf254, as_strided(buf255, (8192, 768), (768, 1)), buf264, buf268, buf270, buf274, buf275, as_strided(buf276, (8192, 768), (768, 1)), buf277, buf278, buf280, buf284, buf285, as_strided(buf286, (8192, 768), (768, 1)), buf295, buf299, buf301, buf305, buf306, as_strided(buf307, (8192, 768), (768, 1)), buf308, buf309, buf311, buf315, buf316, as_strided(buf317, (8192, 768), (768, 1)), buf326, buf330, buf332, buf336, buf337, as_strided(buf338, (8192, 768), (768, 1)), buf339, buf340, buf342, buf346, buf347, as_strided(buf348, (8192, 768), (768, 1)), buf357, buf361, buf363, buf367, buf368, as_strided(buf369, (8192, 768), (768, 1)), buf370, buf371, buf373, buf377, buf378, buf379, buf380, buf381, buf385, buf386, buf387, as_strided(primals_202, (30522, 768), (768, 1)), as_strided(primals_198, (768, 768), (768, 1)), as_strided(primals_194, (768, 3072), (3072, 1)), as_strided(primals_192, (3072, 768), (768, 1)), as_strided(primals_188, (768, 768), (768, 1)), buf389, buf390, buf391, buf392, as_strided(primals_186, (768, 768), (768, 1)), as_strided(primals_184, (768, 768), (768, 1)), as_strided(primals_182, (768, 768), (768, 1)), as_strided(primals_178, (768, 3072), (3072, 1)), as_strided(primals_176, (3072, 768), (768, 1)), as_strided(primals_172, (768, 768), (768, 1)), buf393, buf394, buf395, buf396, as_strided(primals_170, (768, 768), (768, 1)), as_strided(primals_168, (768, 768), (768, 1)), as_strided(primals_166, (768, 768), (768, 1)), as_strided(primals_162, (768, 3072), (3072, 1)), as_strided(primals_160, (3072, 768), (768, 1)), as_strided(primals_156, (768, 768), (768, 1)), buf397, buf398, buf399, buf400, as_strided(primals_154, (768, 768), (768, 1)), as_strided(primals_152, (768, 768), (768, 1)), as_strided(primals_150, (768, 768), (768, 1)), as_strided(primals_146, (768, 3072), (3072, 1)), as_strided(primals_144, (3072, 768), (768, 1)), as_strided(primals_140, (768, 768), (768, 1)), buf401, buf402, buf403, buf404, as_strided(primals_138, (768, 768), (768, 1)), as_strided(primals_136, (768, 768), (768, 1)), as_strided(primals_134, (768, 768), (768, 1)), as_strided(primals_130, (768, 3072), (3072, 1)), as_strided(primals_128, (3072, 768), (768, 1)), as_strided(primals_124, (768, 768), (768, 1)), buf405, buf406, buf407, buf408, as_strided(primals_122, (768, 768), (768, 1)), as_strided(primals_120, (768, 768), (768, 1)), as_strided(primals_118, (768, 768), (768, 1)), as_strided(primals_114, (768, 3072), (3072, 1)), as_strided(primals_112, (3072, 768), (768, 1)), as_strided(primals_108, (768, 768), (768, 1)), buf409, buf410, buf411, buf412, as_strided(primals_106, (768, 768), (768, 1)), as_strided(primals_104, (768, 768), (768, 1)), as_strided(primals_102, (768, 768), (768, 1)), as_strided(primals_98, (768, 3072), (3072, 1)), as_strided(primals_96, (3072, 768), (768, 1)), as_strided(primals_92, (768, 768), (768, 1)), buf413, buf414, buf415, buf416, as_strided(primals_90, (768, 768), (768, 1)), as_strided(primals_88, (768, 768), (768, 1)), as_strided(primals_86, (768, 768), (768, 1)), as_strided(primals_82, (768, 3072), (3072, 1)), as_strided(primals_80, (3072, 768), (768, 1)), as_strided(primals_76, (768, 768), (768, 1)), buf417, buf418, buf419, buf420, as_strided(primals_74, (768, 768), (768, 1)), as_strided(primals_72, (768, 768), (768, 1)), as_strided(primals_70, (768, 768), (768, 1)), as_strided(primals_66, (768, 3072), (3072, 1)), as_strided(primals_64, (3072, 768), (768, 1)), as_strided(primals_60, (768, 768), (768, 1)), buf421, buf422, buf423, buf424, as_strided(primals_58, (768, 768), (768, 1)), as_strided(primals_56, (768, 768), (768, 1)), as_strided(primals_54, (768, 768), (768, 1)), as_strided(primals_50, (768, 3072), (3072, 1)), as_strided(primals_48, (3072, 768), (768, 1)), as_strided(primals_44, (768, 768), (768, 1)), buf425, buf426, buf427, buf428, as_strided(primals_42, (768, 768), (768, 1)), as_strided(primals_40, (768, 768), (768, 1)), as_strided(primals_38, (768, 768), (768, 1)), as_strided(primals_34, (768, 3072), (3072, 1)), as_strided(primals_32, (3072, 768), (768, 1)), as_strided(primals_28, (768, 768), (768, 1)), buf429, buf430, buf431, buf432, as_strided(primals_26, (768, 768), (768, 1)), as_strided(primals_24, (768, 768), (768, 1)), as_strided(primals_22, (768, 768), (768, 1)), as_strided(primals_18, (768, 3072), (3072, 1)), as_strided(primals_16, (3072, 768), (768, 1)), as_strided(primals_12, (768, 768), (768, 1)), buf433, buf434, buf435, buf436, as_strided(primals_10, (768, 768), (768, 1)), as_strided(primals_8, (768, 768), (768, 1)), as_strided(primals_6, (768, 768), (768, 1)), buf437, buf438, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    seed_cuda_0 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_2 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_3 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_8 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_16 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_17 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_26 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_33 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_34 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_48 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_49 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_54 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_64 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_65 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_66 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_70 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_72 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_81 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_82 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_86 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_88 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_90 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_96 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_97 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_98 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_102 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_104 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_108 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_112 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_113 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_114 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_120 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_122 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_124 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_128 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_129 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_130 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_134 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_136 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_140 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_144 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_145 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_146 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_152 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_154 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_160 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_161 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_162 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_165 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_168 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_170 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_172 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_176 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_177 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_178 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_182 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_184 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_186 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_188 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_189 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_192 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_193 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_194 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_198 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_202 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    primals_203 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_204 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206]))
