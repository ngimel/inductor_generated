
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

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
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

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 114688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (114688*x1)) // 112) % 112)) + (12544*x0) + (401408*(((r2 + (114688*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__3 = async_compile.triton('''
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
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 114688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (114688*x1)) // 112) % 112)) + (12544*x0) + (401408*(((r2 + (114688*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 1605632.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000006228081046
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1605632.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1605632.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 0.0
    tmp19 = tl.where(tmp17 != tmp17, tmp17, tl.where(tmp17 > tmp18, tmp17, tmp18))
    tmp20 = 6.0
    tmp21 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 < tmp20, tmp19, tmp20))
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp16 <= tmp18
    tmp24 = tmp16 >= tmp20
    tmp25 = tmp23 | tmp24
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 65536],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (57344*x1)) // 112) % 112)) + (12544*x0) + (200704*(((r2 + (57344*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16, 32],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 28
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
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 65536],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (57344*x1)) // 112) % 112)) + (12544*x0) + (200704*(((r2 + (57344*x1)) // 12544))) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 1605632.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16, 32],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 28
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
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000006228081046
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16, 32],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 28
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
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1605632.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1248
    rnumel = 123511
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (123511*x1)
        tmp1 = 1605632
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x0) + (1204224*(((r2 + (123511*x1)) // 12544) % 128)) + ((r2 + (123511*x1)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1248
    rnumel = 123511
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (123511*x1)
        tmp1 = 1605632
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x0) + (1204224*(((r2 + (123511*x1)) // 12544) % 128)) + ((r2 + (123511*x1)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp6 = 1605632.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp4 - tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((12544*x0) + (1204224*(((r2 + (123511*x1)) // 12544) % 128)) + ((r2 + (123511*x1)) % 12544) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp15, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000006228081046
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1605632.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 154140672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 96
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1605632.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 80282
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (80282*x1)
        tmp1 = 401408
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x0) + (301056*(((r2 + (80282*x1)) // 3136) % 128)) + ((r2 + (80282*x1)) % 3136) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 5
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 80282
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (80282*x1)
        tmp1 = 401408
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x0) + (301056*(((r2 + (80282*x1)) // 3136) % 128)) + ((r2 + (80282*x1)) % 3136) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp6 = 401408.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp4 - tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((3136*x0) + (301056*(((r2 + (80282*x1)) // 3136) % 128)) + ((r2 + (80282*x1)) % 3136) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp15, xmask)
''')


triton__27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 5
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000024912370735
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 8],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 5
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38535168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 96
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 0.0
    tmp19 = tl.where(tmp17 != tmp17, tmp17, tl.where(tmp17 > tmp18, tmp17, tmp18))
    tmp20 = 6.0
    tmp21 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 < tmp20, tmp19, tmp20))
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp16 <= tmp18
    tmp24 = tmp16 >= tmp20
    tmp25 = tmp23 | tmp24
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
''')


triton__30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 432
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 27
    x1 = (xindex // 27)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (84672*(r2 // 3136)) + (677376*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27
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
        tmp0 = tl.load(in_ptr0 + (x0 + (27*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 432
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 27
    x1 = (xindex // 27)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (84672*(r2 // 3136)) + (677376*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 401408.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27
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
        tmp0 = tl.load(in_ptr0 + (x0 + (27*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000024912370735
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27
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
        tmp0 = tl.load(in_ptr0 + (x0 + (27*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10838016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 27
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4374
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
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

@reduction(size_hints=[1024, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 648
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 162
    x1 = (xindex // 162)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (508032*(r2 // 3136)) + (16257024*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 162
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
        tmp0 = tl.load(in_ptr0 + (x0 + (162*r1)), rmask & xmask, eviction_policy='evict_last')
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

@reduction(size_hints=[1024, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 648
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 162
    x1 = (xindex // 162)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (508032*(r2 // 3136)) + (16257024*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 401408.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 162
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
        tmp0 = tl.load(in_ptr0 + (x0 + (162*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000024912370735
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 162
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
        tmp0 = tl.load(in_ptr0 + (x0 + (162*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65028096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 162
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1458
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65028096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 162
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 0.0
    tmp19 = tl.where(tmp17 != tmp17, tmp17, tl.where(tmp17 > tmp18, tmp17, tmp18))
    tmp20 = 6.0
    tmp21 = tl.where(tmp19 != tmp19, tmp19, tl.where(tmp19 < tmp20, tmp19, tmp20))
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp16 <= tmp18
    tmp24 = tmp16 >= tmp20
    tmp25 = tmp23 | tmp24
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
''')


triton__46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6156
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 494
    rnumel = 30878
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 38)
    x0 = xindex % 38
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (30878*x1)
        tmp1 = 401408
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x0) + (119168*(((r2 + (30878*x1)) // 3136) % 128)) + ((r2 + (30878*x1)) % 3136) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38
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
        tmp0 = tl.load(in_ptr0 + (x0 + (38*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 494
    rnumel = 30878
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 38)
    x0 = xindex % 38
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (30878*x1)
        tmp1 = 401408
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x0) + (119168*(((r2 + (30878*x1)) // 3136) % 128)) + ((r2 + (30878*x1)) % 3136) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp6 = 401408.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp4 - tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((3136*x0) + (119168*(((r2 + (30878*x1)) // 3136) % 128)) + ((r2 + (30878*x1)) % 3136) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp15, xmask)
''')


triton__50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38
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
        tmp0 = tl.load(in_ptr0 + (x0 + (38*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000024912370735
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38
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
        tmp0 = tl.load(in_ptr0 + (x0 + (38*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 401408.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15253504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 38
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10838016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 84672
    x1 = (xindex // 84672)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (119168*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (119168*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4415488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 34496
    x1 = (xindex // 34496)
    tmp0 = tl.load(in_ptr0 + (84672 + x0 + (119168*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (119168*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 524288],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 228
    rnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (715008*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp3 = tl.load(in_ptr0 + (r1 + (3136*x0) + (715008*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 401408.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 401408.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000024912370735
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 91521024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 228
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 401408.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2052
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 228
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (178752*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp3 = tl.load(in_ptr0 + (r1 + (784*x0) + (178752*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 100352.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 100352.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.00000996502277
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32768, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 29184
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 228
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (784*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 100352.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (784*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 784.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4332
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__62 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__63 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 19
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__64 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19
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
        tmp0 = tl.load(in_ptr0 + (x0 + (19*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (19*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__65 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 19
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__66 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 228
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__67 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 228
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__68 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22880256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__69 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__70 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 650
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 50)
    x0 = xindex % 50
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (39200*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__71 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50
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
        tmp0 = tl.load(in_ptr0 + (x0 + (50*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__72 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 650
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 50)
    x0 = xindex % 50
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (39200*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp6 = 100352.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp4 - tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((784*x0) + (39200*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp15, xmask)
''')


triton__73 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50
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
        tmp0 = tl.load(in_ptr0 + (x0 + (50*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 100352.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.00000996502277
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__74 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50
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
        tmp0 = tl.load(in_ptr0 + (x0 + (50*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 100352.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__75 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 50
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__76 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
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
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 300
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (235200*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp3 = tl.load(in_ptr0 + (r1 + (784*x0) + (235200*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 100352.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 100352.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.00000996502277
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__78 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30105600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 300
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__79 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2700
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__80 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 300
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (784*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 100352.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (784*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 784.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__81 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7500
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__82 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__83 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__84 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25
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
        tmp0 = tl.load(in_ptr0 + (x0 + (25*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (25*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__85 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__86 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 300
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__87 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 300
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__88 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30105600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__89 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18300
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__90 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 793
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 61)
    x0 = xindex % 61
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (47824*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        _tmp6 = tl.where(rmask & xmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp6, xmask)
''')


triton__91 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 61
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
        tmp0 = tl.load(in_ptr0 + (x0 + (61*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__92 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 793
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 61)
    x0 = xindex % 61
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x1)
        tmp1 = 100352
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (47824*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp6 = 100352.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp4 - tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.where(tmp2, tmp9, 0)
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
        tmp12 = tl.load(in_ptr0 + ((784*x0) + (47824*(((r2 + (7720*x1)) // 784) % 128)) + ((r2 + (7720*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.where(tmp2, tmp13, 0)
        _tmp15 = tl.where(rmask & xmask, _tmp15 + tmp14, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp15, xmask)
''')


triton__93 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 61
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
        tmp0 = tl.load(in_ptr0 + (x0 + (61*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 100352.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.00000996502277
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__94 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 16],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 61
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
        tmp0 = tl.load(in_ptr0 + (x0 + (61*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 100352.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__95 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6121472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 61
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__96 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 39200
    x1 = (xindex // 39200)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (47824*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (47824*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__97 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1103872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8624
    x1 = (xindex // 8624)
    tmp0 = tl.load(in_ptr0 + (39200 + x0 + (47824*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (47824*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__98 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22326
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
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

@reduction(size_hints=[512, 131072],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 366
    rnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (286944*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp3 = tl.load(in_ptr0 + (r1 + (784*x0) + (286944*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 100352.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 100352.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.00000996502277
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__100 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36728832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 366
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 100352.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__101 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3294
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__102 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 366
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (71736*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (71736*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__103 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 46848
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 366
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (196*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__104 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10980
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__105 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__106 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 30
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__107 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (30*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__108 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 30
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__109 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 366
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__110 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 366
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__111 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9182208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__112 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__113 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (14112*(r2 // 196)) + (451584*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__114 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__115 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (14112*(r2 // 196)) + (451584*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__116 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000398612827361
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__117 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__118 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__119 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__120 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 432
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (84672*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (84672*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__121 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10838016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 432
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__122 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__123 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 55296
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 432
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (196*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__124 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__125 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__126 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__127 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 36
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
        tmp0 = tl.load(in_ptr0 + (x0 + (36*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (36*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__128 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__129 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__130 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__131 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10838016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__132 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__133 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 84
    x1 = (xindex // 84)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (16464*(r2 // 196)) + (526848*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__134 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 84
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
        tmp0 = tl.load(in_ptr0 + (x0 + (84*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__135 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 84
    x1 = (xindex // 84)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (16464*(r2 // 196)) + (526848*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__136 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 84
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
        tmp0 = tl.load(in_ptr0 + (x0 + (84*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000398612827361
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__137 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 84
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
        tmp0 = tl.load(in_ptr0 + (x0 + (84*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__138 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 84
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__139 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14112
    x1 = (xindex // 14112)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16464*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (16464*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__140 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2352
    x1 = (xindex // 2352)
    tmp0 = tl.load(in_ptr0 + (14112 + x0 + (16464*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (16464*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__141 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__142 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 504
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (98784*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (98784*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__143 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12644352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 504
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__144 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__145 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64512
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 504
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (196*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__146 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__147 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__148 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 42
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__149 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 42
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
        tmp0 = tl.load(in_ptr0 + (x0 + (42*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (42*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__150 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 42
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__151 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__152 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 504
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__153 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12644352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__154 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 47880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__155 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 380
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 95
    x1 = (xindex // 95)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (18620*(r2 // 196)) + (595840*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__156 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 95
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
        tmp0 = tl.load(in_ptr0 + (x0 + (95*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__157 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 380
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 95
    x1 = (xindex // 95)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (18620*(r2 // 196)) + (595840*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__158 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 95
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
        tmp0 = tl.load(in_ptr0 + (x0 + (95*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000398612827361
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__159 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 95
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
        tmp0 = tl.load(in_ptr0 + (x0 + (95*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__160 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2383360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 95
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__161 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16464
    x1 = (xindex // 16464)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (18620*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (18620*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__162 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 275968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2156
    x1 = (xindex // 2156)
    tmp0 = tl.load(in_ptr0 + (16464 + x0 + (18620*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (18620*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__163 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54150
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__164 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 570
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (111720*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (111720*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__165 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14300160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 570
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__166 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5130
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__167 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72960
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 570
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (196*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__168 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26790
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__169 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 47
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__170 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 47
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__171 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 47
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
        tmp0 = tl.load(in_ptr0 + (x0 + (47*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (47*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__172 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 47
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__173 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 570
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__174 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 570
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__175 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14300160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__176 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 60420
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__177 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 424
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 106
    x1 = (xindex // 106)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (20776*(r2 // 196)) + (664832*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__178 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 106
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
        tmp0 = tl.load(in_ptr0 + (x0 + (106*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__179 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 424
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 106
    x1 = (xindex // 106)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (20776*(r2 // 196)) + (664832*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__180 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 106
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
        tmp0 = tl.load(in_ptr0 + (x0 + (106*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000398612827361
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__181 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 106
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
        tmp0 = tl.load(in_ptr0 + (x0 + (106*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__182 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2659328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 106
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__183 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2383360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 18620
    x1 = (xindex // 18620)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (20776*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (20776*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__184 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 275968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2156
    x1 = (xindex // 2156)
    tmp0 = tl.load(in_ptr0 + (18620 + x0 + (20776*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (20776*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__185 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__186 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 636
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (124656*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (124656*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__187 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15955968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 636
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__188 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5724
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__189 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 81408
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 636
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (196*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__190 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33708
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__191 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__192 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 53
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__193 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 53
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
        tmp0 = tl.load(in_ptr0 + (x0 + (53*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (53*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__194 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 53
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__195 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 636
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__196 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 636
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__197 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15955968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__198 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 74412
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__199 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 468
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 117
    x1 = (xindex // 117)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (22932*(r2 // 196)) + (733824*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__200 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 117
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
        tmp0 = tl.load(in_ptr0 + (x0 + (117*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__201 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 468
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 117
    x1 = (xindex // 117)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (22932*(r2 // 196)) + (733824*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__202 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 117
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
        tmp0 = tl.load(in_ptr0 + (x0 + (117*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000398612827361
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__203 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 117
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
        tmp0 = tl.load(in_ptr0 + (x0 + (117*r1)), rmask & xmask, eviction_policy='evict_last')
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__204 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2935296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 117
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__205 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2659328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 20776
    x1 = (xindex // 20776)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (22932*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (22932*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__206 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 275968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2156
    x1 = (xindex // 2156)
    tmp0 = tl.load(in_ptr0 + (20776 + x0 + (22932*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (22932*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__207 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 82134
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__208 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 702
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (137592*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (137592*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__209 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17611776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 702
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__210 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6318
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__211 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 89856
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 702
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (196*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__212 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40716
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__213 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 58
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__214 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__215 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 58
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
        tmp0 = tl.load(in_ptr0 + (x0 + (58*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (58*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__216 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__217 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__218 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 89856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 702
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__219 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17611776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__220 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 89856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__221 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp2, xmask)
''')


triton__222 = async_compile.triton('''
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
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


triton__223 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (25088*(r2 // 196)) + (802816*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 25088.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp6 = tmp5 * tmp5
        _tmp7 = tl.where(rmask & xmask, _tmp7 + tmp6, _tmp7)
        _tmp8 = tl.where(rmask & xmask, _tmp8 + tmp1, _tmp8)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp7, xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp8, xmask)
''')


triton__224 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.libdevice.rsqrt(tmp5)
    tmp7 = 1.0000398612827361
    tmp8 = tmp3 * tmp7
    tmp9 = 0.1
    tmp10 = tmp8 * tmp9
    tmp12 = 0.9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp6, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp14, xmask)
''')


triton__225 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 4],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 25088.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = 0.9
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp9, xmask)
    tl.store(out_ptr2 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp3, xmask)
''')


triton__226 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__227 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2935296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 22932
    x1 = (xindex // 22932)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25088*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (25088*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__228 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 275968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2156
    x1 = (xindex // 2156)
    tmp0 = tl.load(in_ptr0 + (22932 + x0 + (25088*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (25088*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__229 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__230 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 25088.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 25088.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0000398612827361
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__231 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 25088.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__232 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__233 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__234 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 768
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 6272.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__235 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__236 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__237 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__238 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[64, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__239 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__240 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__241 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__242 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__243 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 107520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__244 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 140
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (6860*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (6860*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__245 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 878080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 140
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__246 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 117600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__247 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 840
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (41160*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (41160*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__248 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5268480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 840
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__249 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__250 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 107520
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 840
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 6272.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__251 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 58800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__252 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 70
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__253 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 70
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__254 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 70
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
        tmp0 = tl.load(in_ptr0 + (x0 + (70*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (70*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__255 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 70
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__256 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__257 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 107520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 840
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__258 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5268480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__259 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 126840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__260 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 151
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7399*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (7399*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__261 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 947072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 151
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__262 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 878080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6860
    x1 = (xindex // 6860)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (7399*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (7399*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__263 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 539
    x1 = (xindex // 539)
    tmp0 = tl.load(in_ptr0 + (6860 + x0 + (7399*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (7399*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__264 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 136806
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__265 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 906
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (44394*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (44394*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__266 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5682432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 906
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__267 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8154
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__268 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 115968
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 906
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 6272.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__269 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67950
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__270 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__271 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 75
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__272 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75
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
        tmp0 = tl.load(in_ptr0 + (x0 + (75*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (75*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__273 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 75
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__274 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 906
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__275 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 906
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__276 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5682432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__277 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 146772
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__278 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 162
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7938*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (7938*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__279 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1016064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 162
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__280 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 947072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7399
    x1 = (xindex // 7399)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (7938*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (7938*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__281 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 539
    x1 = (xindex // 539)
    tmp0 = tl.load(in_ptr0 + (7399 + x0 + (7938*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (7938*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__282 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 157464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__283 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 972
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (47628*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (47628*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__284 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6096384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 972
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__285 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8748
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__286 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 124416
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 972
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 6272.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__287 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78732
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__288 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__289 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 81
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__290 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 81
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
        tmp0 = tl.load(in_ptr0 + (x0 + (81*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (81*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__291 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 81
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__292 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 972
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__293 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 124416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 972
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__294 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6096384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__295 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 169128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__296 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 174
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (8526*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (8526*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__297 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1091328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 174
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__298 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1016064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7938
    x1 = (xindex // 7938)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8526*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (8526*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__299 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 588
    x1 = (xindex // 588)
    tmp0 = tl.load(in_ptr0 + (7938 + x0 + (8526*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (8526*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__300 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__301 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1044
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (51156*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (51156*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__302 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6547968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1044
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.sigmoid(tmp16)
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tmp16 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp20, xmask)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp26, xmask)
''')


triton__303 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9396
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__304 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[262144, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 133632
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1044
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp18 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 6272.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        _tmp18 = tl.where(rmask & xmask, _tmp18 + tmp17, _tmp18)
        tl.store(out_ptr0 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp16, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp21, xmask)
''')


triton__305 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90828
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__306 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__307 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 87
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__308 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[128, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 87
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
        tmp0 = tl.load(in_ptr0 + (x0 + (87*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (87*r1)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 128.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 128.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0078740157480315
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__309 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 87
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 128.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(0 != 0, 0, tl.where(0 > tmp16, 0, tmp16))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp17, xmask)
''')


triton__310 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1044
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__311 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 133632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1044
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__312 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6547968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp5, tmp4, tmp5))
    tmp7 = 6.0
    tmp8 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp7, tmp6, tmp7))
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


triton__313 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 193140
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__314 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 185
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__315 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1160320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 185
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp14 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 6272.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 - tmp4
    tmp7 = tmp6 / tmp3
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.libdevice.rsqrt(tmp9)
    tmp11 = tmp5 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__316 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1091328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8526
    x1 = (xindex // 8526)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (9065*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (9065*x1) + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


triton__317 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 539
    x1 = (xindex // 539)
    tmp0 = tl.load(in_ptr0 + (8526 + x0 + (9065*x1)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (9065*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__318 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 236800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__319 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (62720*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        _tmp2 = tl.where(rmask & xmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 49
        r2 = (rindex // 49)
        tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (62720*r2)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 6272.0
        tmp6 = tmp2 / tmp5
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        _tmp9 = tl.where(rmask & xmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask & xmask, _tmp10 + tmp4, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + x0, tmp9, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tmp20 = tl.load(in_ptr1 + (x0), xmask)
    tmp26 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.libdevice.rsqrt(tmp14)
    tmp16 = 1.0001594642002871
    tmp17 = tmp12 * tmp16
    tmp18 = 0.1
    tmp19 = tmp17 * tmp18
    tmp21 = 0.9
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tmp10 / tmp11
    tmp25 = tmp24 * tmp18
    tmp27 = tmp26 * tmp21
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr3 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp15, xmask)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp23, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp24, xmask)
''')


triton__320 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[262144, 64],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 163840
    rnumel = 49
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1280
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp12 = tl.load(in_ptr3 + (x0), xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask)
    _tmp28 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 6272.0
        tmp4 = tmp2 / tmp3
        tmp5 = tmp1 - tmp4
        tmp7 = tmp6 / tmp3
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.libdevice.rsqrt(tmp9)
        tmp11 = tmp5 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.sigmoid(tmp16)
        tmp18 = 1.0
        tmp19 = tmp18 - tmp17
        tmp20 = tmp16 * tmp19
        tmp21 = tmp20 + tmp18
        tmp22 = tmp17 * tmp21
        tmp23 = tmp16.to(tl.float32)
        tmp24 = tl.sigmoid(tmp23)
        tmp25 = tmp23 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        _tmp28 = tl.where(rmask & xmask, _tmp28 + tmp27, _tmp28)
        tl.store(out_ptr1 + (r2 + (49*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp22, rmask & xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp29 = 49.0
    tmp30 = tmp28 / tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr3 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp31, xmask)
''')


triton__321 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__322 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


triton__323 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda', dtype=torch.float16)
        stream0 = get_cuda_stream(0)
        triton__0.run(primals_1, buf0, 864, grid=grid(864), stream=stream0)
        del primals_1
        buf1 = empty_strided((128, 3, 224, 224), (150528, 50176, 224, 1), device='cuda', dtype=torch.float16)
        triton__1.run(primals_169, buf1, 19267584, grid=grid(19267584), stream=stream0)
        del primals_169
        buf2 = aten.convolution(buf1, buf0, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf2, (128, 32, 112, 112), (401408, 12544, 112, 1))
        buf3 = empty_strided((1, 32, 1, 1, 14), (448, 1, 448, 448, 32), device='cuda', dtype=torch.float32)
        triton__2.run(buf2, buf3, 448, 114688, grid=grid(448), stream=stream0)
        buf4 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        triton__3.run(buf3, buf4, 32, 14, grid=grid(32), stream=stream0)
        buf5 = buf3; del buf3  # reuse
        buf7 = empty_strided((1, 32, 1, 1, 14), (448, 1, 448, 448, 32), device='cuda', dtype=torch.float32)
        triton__4.run(buf2, buf4, buf5, buf7, 448, 114688, grid=grid(448), stream=stream0)
        buf6 = buf4; del buf4  # reuse
        buf9 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__5.run(buf5, primals_172, buf6, buf9, buf11, 32, 14, grid=grid(32), stream=stream0)
        del primals_172
        buf8 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf883 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf7, primals_171, buf8, buf10, buf883, 32, 14, grid=grid(32), stream=stream0)
        del primals_171
        buf13 = empty_strided((128, 32, 112, 112), (401408, 12544, 112, 1), device='cuda', dtype=torch.float16)
        buf882 = empty_strided((128, 32, 112, 112), (401408, 12544, 112, 1), device='cuda', dtype=torch.float16)
        triton__7.run(buf2, buf8, buf6, primals_173, primals_174, buf13, buf882, 51380224, grid=grid(51380224), stream=stream0)
        del primals_174
        buf14 = empty_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__8.run(primals_2, buf14, 288, grid=grid(288), stream=stream0)
        del primals_2
        buf15 = aten.convolution(buf13, buf14, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 32)
        assert_size_stride(buf15, (128, 32, 112, 112), (401408, 12544, 112, 1))
        buf16 = buf7; del buf7  # reuse
        triton__2.run(buf15, buf16, 448, 114688, grid=grid(448), stream=stream0)
        buf17 = buf8; del buf8  # reuse
        triton__3.run(buf16, buf17, 32, 14, grid=grid(32), stream=stream0)
        buf18 = buf16; del buf16  # reuse
        buf20 = buf5; del buf5  # reuse
        triton__4.run(buf15, buf17, buf18, buf20, 448, 114688, grid=grid(448), stream=stream0)
        buf19 = buf17; del buf17  # reuse
        buf22 = as_strided(buf6, (32, ), (1, )); del buf6  # reuse
        buf24 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__5.run(buf18, primals_177, buf19, buf22, buf24, 32, 14, grid=grid(32), stream=stream0)
        del primals_177
        buf21 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        buf880 = empty_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__6.run(buf20, primals_176, buf21, buf23, buf880, 32, 14, grid=grid(32), stream=stream0)
        del primals_176
        buf26 = empty_strided((128, 32, 112, 112), (401408, 12544, 112, 1), device='cuda', dtype=torch.float16)
        buf879 = empty_strided((128, 32, 112, 112), (401408, 12544, 112, 1), device='cuda', dtype=torch.bool)
        triton__9.run(buf15, buf21, buf19, primals_178, primals_179, buf26, buf879, 51380224, grid=grid(51380224), stream=stream0)
        del buf19
        del buf21
        del primals_179
        buf27 = empty_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__10.run(primals_3, buf27, 512, grid=grid(512), stream=stream0)
        del primals_3
        buf28 = aten.convolution(buf26, buf27, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf28, (128, 16, 112, 112), (200704, 12544, 112, 1))
        buf29 = as_strided(buf20, (1, 16, 1, 1, 28), (448, 1, 448, 448, 16)); del buf20  # reuse
        triton__11.run(buf28, buf29, 448, 57344, grid=grid(448), stream=stream0)
        buf30 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        triton__12.run(buf29, buf30, 16, 28, grid=grid(16), stream=stream0)
        buf31 = buf29; del buf29  # reuse
        buf33 = as_strided(buf18, (1, 16, 1, 1, 28), (448, 1, 448, 448, 16)); del buf18  # reuse
        triton__13.run(buf28, buf30, buf31, buf33, 448, 57344, grid=grid(448), stream=stream0)
        buf32 = buf30; del buf30  # reuse
        buf35 = empty_strided((16, ), (1, ), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((16, ), (1, ), device='cuda', dtype=torch.float32)
        triton__14.run(buf31, primals_182, buf32, buf35, buf37, 16, 28, grid=grid(16), stream=stream0)
        del buf31
        del primals_182
        buf34 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((16, ), (1, ), device='cuda', dtype=torch.float32)
        buf878 = empty_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__15.run(buf33, primals_181, buf34, buf36, buf878, 16, 28, grid=grid(16), stream=stream0)
        del buf33
        del primals_181
        buf38 = empty_strided((128, 16, 112, 112), (200704, 12544, 112, 1), device='cuda', dtype=torch.float16)
        triton__16.run(buf28, buf34, buf32, primals_183, primals_184, buf38, 25690112, grid=grid(25690112), stream=stream0)
        del buf32
        del buf34
        del primals_184
        buf39 = empty_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__17.run(primals_4, buf39, 1536, grid=grid(1536), stream=stream0)
        del primals_4
        buf40 = aten.convolution(buf38, buf39, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf40, (128, 96, 112, 112), (1204224, 12544, 112, 1))
        buf41 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        triton__18.run(buf40, buf41, 1248, 123511, grid=grid(1248), stream=stream0)
        buf42 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        triton__19.run(buf41, buf42, 96, 13, grid=grid(96), stream=stream0)
        buf43 = buf41; del buf41  # reuse
        buf45 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        triton__20.run(buf40, buf42, buf43, buf45, 1248, 123511, grid=grid(1248), stream=stream0)
        buf44 = buf42; del buf42  # reuse
        buf47 = empty_strided((96, ), (1, ), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((96, ), (1, ), device='cuda', dtype=torch.float32)
        triton__21.run(buf43, primals_187, buf44, buf47, buf49, 96, 13, grid=grid(96), stream=stream0)
        del buf43
        del primals_187
        buf46 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((96, ), (1, ), device='cuda', dtype=torch.float32)
        buf877 = empty_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__22.run(buf45, primals_186, buf46, buf48, buf877, 96, 13, grid=grid(96), stream=stream0)
        del buf45
        del primals_186
        buf51 = empty_strided((128, 96, 112, 112), (1204224, 12544, 112, 1), device='cuda', dtype=torch.float16)
        buf876 = empty_strided((128, 96, 112, 112), (1204224, 12544, 112, 1), device='cuda', dtype=torch.float16)
        triton__23.run(buf40, buf46, buf44, primals_188, primals_189, buf51, buf876, 154140672, grid=grid(154140672), stream=stream0)
        del primals_189
        buf52 = empty_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__0.run(primals_5, buf52, 864, grid=grid(864), stream=stream0)
        del primals_5
        buf53 = aten.convolution(buf51, buf52, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 96)
        assert_size_stride(buf53, (128, 96, 56, 56), (301056, 3136, 56, 1))
        buf54 = empty_strided((1, 96, 1, 1, 5), (480, 1, 480, 480, 96), device='cuda', dtype=torch.float32)
        triton__24.run(buf53, buf54, 480, 80282, grid=grid(480), stream=stream0)
        buf55 = buf46; del buf46  # reuse
        triton__25.run(buf54, buf55, 96, 5, grid=grid(96), stream=stream0)
        buf56 = buf54; del buf54  # reuse
        buf58 = empty_strided((1, 96, 1, 1, 5), (480, 1, 480, 480, 96), device='cuda', dtype=torch.float32)
        triton__26.run(buf53, buf55, buf56, buf58, 480, 80282, grid=grid(480), stream=stream0)
        buf57 = buf55; del buf55  # reuse
        buf60 = as_strided(buf44, (96, ), (1, )); del buf44  # reuse
        buf62 = empty_strided((96, ), (1, ), device='cuda', dtype=torch.float32)
        triton__27.run(buf56, primals_192, buf57, buf60, buf62, 96, 5, grid=grid(96), stream=stream0)
        del buf56
        del primals_192
        buf59 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((96, ), (1, ), device='cuda', dtype=torch.float32)
        buf874 = empty_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__28.run(buf58, primals_191, buf59, buf61, buf874, 96, 5, grid=grid(96), stream=stream0)
        del buf58
        del primals_191
        buf64 = empty_strided((128, 96, 56, 56), (301056, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf873 = empty_strided((128, 96, 56, 56), (301056, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__29.run(buf53, buf59, buf57, primals_193, primals_194, buf64, buf873, 38535168, grid=grid(38535168), stream=stream0)
        del buf57
        del buf59
        del primals_194
        buf65 = empty_strided((27, 96, 1, 1), (96, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__30.run(primals_6, buf65, 2592, grid=grid(2592), stream=stream0)
        del primals_6
        buf66 = aten.convolution(buf64, buf65, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf66, (128, 27, 56, 56), (84672, 3136, 56, 1))
        buf67 = empty_strided((1, 27, 1, 1, 16), (432, 1, 432, 432, 27), device='cuda', dtype=torch.float32)
        triton__31.run(buf66, buf67, 432, 25088, grid=grid(432), stream=stream0)
        buf68 = empty_strided((1, 27, 1, 1), (27, 1, 27, 27), device='cuda', dtype=torch.float32)
        triton__32.run(buf67, buf68, 27, 16, grid=grid(27), stream=stream0)
        buf69 = buf67; del buf67  # reuse
        buf71 = empty_strided((1, 27, 1, 1, 16), (432, 1, 432, 432, 27), device='cuda', dtype=torch.float32)
        triton__33.run(buf66, buf68, buf69, buf71, 432, 25088, grid=grid(432), stream=stream0)
        buf70 = buf68; del buf68  # reuse
        buf73 = empty_strided((27, ), (1, ), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((27, ), (1, ), device='cuda', dtype=torch.float32)
        triton__34.run(buf69, primals_197, buf70, buf73, buf75, 27, 16, grid=grid(27), stream=stream0)
        del primals_197
        buf72 = empty_strided((1, 27, 1, 1), (27, 1, 27, 27), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((27, ), (1, ), device='cuda', dtype=torch.float32)
        buf872 = empty_strided((1, 27, 1, 1), (27, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__35.run(buf71, primals_196, buf72, buf74, buf872, 27, 16, grid=grid(27), stream=stream0)
        del primals_196
        buf76 = empty_strided((128, 27, 56, 56), (84672, 3136, 56, 1), device='cuda', dtype=torch.float16)
        triton__36.run(buf66, buf72, buf70, primals_198, primals_199, buf76, 10838016, grid=grid(10838016), stream=stream0)
        del buf70
        del buf72
        del primals_199
        buf77 = empty_strided((162, 27, 1, 1), (27, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__37.run(primals_7, buf77, 4374, grid=grid(4374), stream=stream0)
        del primals_7
        buf78 = aten.convolution(buf76, buf77, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf78, (128, 162, 56, 56), (508032, 3136, 56, 1))
        buf79 = empty_strided((1, 162, 1, 1, 4), (648, 1, 648, 648, 162), device='cuda', dtype=torch.float32)
        triton__38.run(buf78, buf79, 648, 100352, grid=grid(648), stream=stream0)
        buf80 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cuda', dtype=torch.float32)
        triton__39.run(buf79, buf80, 162, 4, grid=grid(162), stream=stream0)
        buf81 = buf79; del buf79  # reuse
        buf83 = empty_strided((1, 162, 1, 1, 4), (648, 1, 648, 648, 162), device='cuda', dtype=torch.float32)
        triton__40.run(buf78, buf80, buf81, buf83, 648, 100352, grid=grid(648), stream=stream0)
        buf82 = buf80; del buf80  # reuse
        buf85 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        triton__41.run(buf81, primals_202, buf82, buf85, buf87, 162, 4, grid=grid(162), stream=stream0)
        del primals_202
        buf84 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cuda', dtype=torch.float32)
        buf86 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        buf871 = empty_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__42.run(buf83, primals_201, buf84, buf86, buf871, 162, 4, grid=grid(162), stream=stream0)
        del primals_201
        buf89 = empty_strided((128, 162, 56, 56), (508032, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf870 = empty_strided((128, 162, 56, 56), (508032, 3136, 56, 1), device='cuda', dtype=torch.float16)
        triton__43.run(buf78, buf84, buf82, primals_203, primals_204, buf89, buf870, 65028096, grid=grid(65028096), stream=stream0)
        del primals_204
        buf90 = empty_strided((162, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__44.run(primals_8, buf90, 1458, grid=grid(1458), stream=stream0)
        del primals_8
        buf91 = aten.convolution(buf89, buf90, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 162)
        assert_size_stride(buf91, (128, 162, 56, 56), (508032, 3136, 56, 1))
        buf92 = buf83; del buf83  # reuse
        triton__38.run(buf91, buf92, 648, 100352, grid=grid(648), stream=stream0)
        buf93 = buf84; del buf84  # reuse
        triton__39.run(buf92, buf93, 162, 4, grid=grid(162), stream=stream0)
        buf94 = buf92; del buf92  # reuse
        buf96 = buf81; del buf81  # reuse
        triton__40.run(buf91, buf93, buf94, buf96, 648, 100352, grid=grid(648), stream=stream0)
        buf95 = buf93; del buf93  # reuse
        buf98 = as_strided(buf82, (162, ), (1, )); del buf82  # reuse
        buf100 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        triton__41.run(buf94, primals_207, buf95, buf98, buf100, 162, 4, grid=grid(162), stream=stream0)
        del buf94
        del primals_207
        buf97 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cuda', dtype=torch.float32)
        buf99 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        buf868 = empty_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__42.run(buf96, primals_206, buf97, buf99, buf868, 162, 4, grid=grid(162), stream=stream0)
        del buf96
        del primals_206
        buf102 = empty_strided((128, 162, 56, 56), (508032, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf867 = empty_strided((128, 162, 56, 56), (508032, 3136, 56, 1), device='cuda', dtype=torch.bool)
        triton__45.run(buf91, buf97, buf95, primals_208, primals_209, buf102, buf867, 65028096, grid=grid(65028096), stream=stream0)
        del primals_209
        buf103 = empty_strided((38, 162, 1, 1), (162, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__46.run(primals_9, buf103, 6156, grid=grid(6156), stream=stream0)
        del primals_9
        buf104 = aten.convolution(buf102, buf103, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf104, (128, 38, 56, 56), (119168, 3136, 56, 1))
        buf105 = empty_strided((1, 38, 1, 1, 13), (494, 1, 494, 494, 38), device='cuda', dtype=torch.float32)
        triton__47.run(buf104, buf105, 494, 30878, grid=grid(494), stream=stream0)
        buf106 = empty_strided((1, 38, 1, 1), (38, 1, 38, 38), device='cuda', dtype=torch.float32)
        triton__48.run(buf105, buf106, 38, 13, grid=grid(38), stream=stream0)
        buf107 = buf105; del buf105  # reuse
        buf109 = empty_strided((1, 38, 1, 1, 13), (494, 1, 494, 494, 38), device='cuda', dtype=torch.float32)
        triton__49.run(buf104, buf106, buf107, buf109, 494, 30878, grid=grid(494), stream=stream0)
        buf108 = buf106; del buf106  # reuse
        buf111 = empty_strided((38, ), (1, ), device='cuda', dtype=torch.float32)
        buf113 = empty_strided((38, ), (1, ), device='cuda', dtype=torch.float32)
        triton__50.run(buf107, primals_212, buf108, buf111, buf113, 38, 13, grid=grid(38), stream=stream0)
        del buf107
        del primals_212
        buf110 = empty_strided((1, 38, 1, 1), (38, 1, 38, 38), device='cuda', dtype=torch.float32)
        buf112 = empty_strided((38, ), (1, ), device='cuda', dtype=torch.float32)
        buf866 = empty_strided((1, 38, 1, 1), (38, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__51.run(buf109, primals_211, buf110, buf112, buf866, 38, 13, grid=grid(38), stream=stream0)
        del buf109
        del primals_211
        buf114 = empty_strided((128, 38, 56, 56), (119168, 3136, 56, 1), device='cuda', dtype=torch.float16)
        triton__52.run(buf104, buf110, buf108, primals_213, primals_214, buf114, 15253504, grid=grid(15253504), stream=stream0)
        del buf108
        del buf110
        del primals_214
        buf117 = empty_strided((128, 38, 56, 56), (119168, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf115 = as_strided(buf117, (128, 27, 56, 56), (119168, 3136, 56, 1))  # alias
        triton__53.run(buf114, buf76, buf115, 10838016, grid=grid(10838016), stream=stream0)
        buf116 = as_strided(buf117, (128, 11, 56, 56), (119168, 3136, 56, 1), 84672)  # alias
        triton__54.run(buf114, buf116, 4415488, grid=grid(4415488), stream=stream0)
        del buf114
        buf118 = empty_strided((228, 38, 1, 1), (38, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__55.run(primals_10, buf118, 8664, grid=grid(8664), stream=stream0)
        del primals_10
        buf119 = aten.convolution(buf117, buf118, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf119, (128, 228, 56, 56), (715008, 3136, 56, 1))
        buf121 = empty_strided((1, 228, 1, 1), (228, 1, 228, 228), device='cuda', dtype=torch.float32)
        buf122 = empty_strided((1, 228, 1, 1), (228, 1, 228, 228), device='cuda', dtype=torch.float32)
        buf123 = empty_strided((228, ), (1, ), device='cuda', dtype=torch.float32)
        buf125 = empty_strided((228, ), (1, ), device='cuda', dtype=torch.float32)
        buf124 = empty_strided((228, ), (1, ), device='cuda', dtype=torch.float32)
        buf865 = empty_strided((1, 228, 1, 1), (228, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__56.run(buf119, primals_217, primals_216, buf121, buf122, buf123, buf125, buf124, buf865, 228, 401408, grid=grid(228), stream=stream0)
        del primals_216
        del primals_217
        buf127 = empty_strided((128, 228, 56, 56), (715008, 3136, 56, 1), device='cuda', dtype=torch.float16)
        buf864 = empty_strided((128, 228, 56, 56), (715008, 3136, 56, 1), device='cuda', dtype=torch.float16)
        triton__57.run(buf119, buf122, buf121, primals_218, primals_219, buf127, buf864, 91521024, grid=grid(91521024), stream=stream0)
        del primals_219
        buf128 = empty_strided((228, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__58.run(primals_11, buf128, 2052, grid=grid(2052), stream=stream0)
        del primals_11
        buf129 = aten.convolution(buf127, buf128, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 228)
        assert_size_stride(buf129, (128, 228, 28, 28), (178752, 784, 28, 1))
        buf131 = buf122; del buf122  # reuse
        buf132 = buf121; del buf121  # reuse
        buf133 = empty_strided((228, ), (1, ), device='cuda', dtype=torch.float32)
        buf135 = empty_strided((228, ), (1, ), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((228, ), (1, ), device='cuda', dtype=torch.float32)
        buf862 = empty_strided((1, 228, 1, 1), (228, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__59.run(buf129, primals_222, primals_221, buf131, buf132, buf133, buf135, buf134, buf862, 228, 100352, grid=grid(228), stream=stream0)
        del primals_221
        del primals_222
        buf136 = empty_strided((128, 228, 28, 28), (178752, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf138 = empty_strided((128, 228, 1, 1), (228, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__60.run(buf129, buf132, buf131, primals_223, primals_224, buf136, buf138, 29184, 784, grid=grid(29184), stream=stream0)
        del buf131
        del buf132
        del primals_224
        buf139 = empty_strided((19, 228, 1, 1), (228, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__61.run(primals_12, buf139, 4332, grid=grid(4332), stream=stream0)
        del primals_12
        buf140 = empty_strided((19, ), (1, ), device='cuda', dtype=torch.float16)
        triton__62.run(primals_13, buf140, 19, grid=grid(19), stream=stream0)
        del primals_13
        buf141 = aten.convolution(buf138, buf139, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf141, (128, 19, 1, 1), (19, 1, 1, 1))
        buf142 = buf141; del buf141  # reuse
        triton__63.run(buf142, buf140, 2432, grid=grid(2432), stream=stream0)
        del buf140
        buf144 = empty_strided((1, 19, 1, 1), (19, 1, 19, 19), device='cuda', dtype=torch.float32)
        buf145 = empty_strided((1, 19, 1, 1), (19, 1, 19, 19), device='cuda', dtype=torch.float32)
        buf146 = empty_strided((19, ), (1, ), device='cuda', dtype=torch.float32)
        buf148 = empty_strided((19, ), (1, ), device='cuda', dtype=torch.float32)
        buf147 = empty_strided((19, ), (1, ), device='cuda', dtype=torch.float32)
        buf861 = empty_strided((1, 19, 1, 1), (19, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__64.run(buf142, primals_131, primals_130, buf144, buf145, buf146, buf148, buf147, buf861, 19, 128, grid=grid(19), stream=stream0)
        del primals_130
        del primals_131
        buf149 = empty_strided((128, 19, 1, 1), (19, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__65.run(buf142, buf145, buf144, primals_14, primals_15, buf149, 2432, grid=grid(2432), stream=stream0)
        del buf144
        del buf145
        del primals_15
        buf150 = empty_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__61.run(primals_16, buf150, 4332, grid=grid(4332), stream=stream0)
        del primals_16
        buf151 = empty_strided((228, ), (1, ), device='cuda', dtype=torch.float16)
        triton__66.run(primals_17, buf151, 228, grid=grid(228), stream=stream0)
        del primals_17
        buf152 = aten.convolution(buf149, buf150, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf152, (128, 228, 1, 1), (228, 1, 1, 1))
        buf153 = buf152; del buf152  # reuse
        triton__67.run(buf153, buf151, 29184, grid=grid(29184), stream=stream0)
        del buf151
        buf154 = empty_strided((128, 228, 28, 28), (178752, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__68.run(buf136, buf153, buf154, 22880256, grid=grid(22880256), stream=stream0)
        buf155 = empty_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__69.run(primals_18, buf155, 11400, grid=grid(11400), stream=stream0)
        del primals_18
        buf156 = aten.convolution(buf154, buf155, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf156, (128, 50, 28, 28), (39200, 784, 28, 1))
        buf157 = empty_strided((1, 50, 1, 1, 13), (650, 1, 650, 650, 50), device='cuda', dtype=torch.float32)
        triton__70.run(buf156, buf157, 650, 7720, grid=grid(650), stream=stream0)
        buf158 = empty_strided((1, 50, 1, 1), (50, 1, 50, 50), device='cuda', dtype=torch.float32)
        triton__71.run(buf157, buf158, 50, 13, grid=grid(50), stream=stream0)
        buf159 = buf157; del buf157  # reuse
        buf161 = empty_strided((1, 50, 1, 1, 13), (650, 1, 650, 650, 50), device='cuda', dtype=torch.float32)
        triton__72.run(buf156, buf158, buf159, buf161, 650, 7720, grid=grid(650), stream=stream0)
        buf160 = buf158; del buf158  # reuse
        buf163 = empty_strided((50, ), (1, ), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((50, ), (1, ), device='cuda', dtype=torch.float32)
        triton__73.run(buf159, primals_227, buf160, buf163, buf165, 50, 13, grid=grid(50), stream=stream0)
        del buf159
        del primals_227
        buf162 = empty_strided((1, 50, 1, 1), (50, 1, 50, 50), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((50, ), (1, ), device='cuda', dtype=torch.float32)
        buf860 = empty_strided((1, 50, 1, 1), (50, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__74.run(buf161, primals_226, buf162, buf164, buf860, 50, 13, grid=grid(50), stream=stream0)
        del buf161
        del primals_226
        buf166 = empty_strided((128, 50, 28, 28), (39200, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__75.run(buf156, buf162, buf160, primals_228, primals_229, buf166, 5017600, grid=grid(5017600), stream=stream0)
        del buf160
        del buf162
        del primals_229
        buf167 = empty_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__76.run(primals_19, buf167, 15000, grid=grid(15000), stream=stream0)
        del primals_19
        buf168 = aten.convolution(buf166, buf167, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf168, (128, 300, 28, 28), (235200, 784, 28, 1))
        buf170 = empty_strided((1, 300, 1, 1), (300, 1, 300, 300), device='cuda', dtype=torch.float32)
        buf171 = empty_strided((1, 300, 1, 1), (300, 1, 300, 300), device='cuda', dtype=torch.float32)
        buf172 = empty_strided((300, ), (1, ), device='cuda', dtype=torch.float32)
        buf174 = empty_strided((300, ), (1, ), device='cuda', dtype=torch.float32)
        buf173 = empty_strided((300, ), (1, ), device='cuda', dtype=torch.float32)
        buf859 = empty_strided((1, 300, 1, 1), (300, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__77.run(buf168, primals_232, primals_231, buf170, buf171, buf172, buf174, buf173, buf859, 300, 100352, grid=grid(300), stream=stream0)
        del primals_231
        del primals_232
        buf176 = empty_strided((128, 300, 28, 28), (235200, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf858 = empty_strided((128, 300, 28, 28), (235200, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__78.run(buf168, buf171, buf170, primals_233, primals_234, buf176, buf858, 30105600, grid=grid(30105600), stream=stream0)
        del primals_234
        buf177 = empty_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__79.run(primals_20, buf177, 2700, grid=grid(2700), stream=stream0)
        del primals_20
        buf178 = aten.convolution(buf176, buf177, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 300)
        assert_size_stride(buf178, (128, 300, 28, 28), (235200, 784, 28, 1))
        buf180 = buf171; del buf171  # reuse
        buf181 = buf170; del buf170  # reuse
        buf182 = empty_strided((300, ), (1, ), device='cuda', dtype=torch.float32)
        buf184 = empty_strided((300, ), (1, ), device='cuda', dtype=torch.float32)
        buf183 = empty_strided((300, ), (1, ), device='cuda', dtype=torch.float32)
        buf856 = empty_strided((1, 300, 1, 1), (300, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__77.run(buf178, primals_237, primals_236, buf180, buf181, buf182, buf184, buf183, buf856, 300, 100352, grid=grid(300), stream=stream0)
        del primals_236
        del primals_237
        buf185 = empty_strided((128, 300, 28, 28), (235200, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf187 = empty_strided((128, 300, 1, 1), (300, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__80.run(buf178, buf181, buf180, primals_238, primals_239, buf185, buf187, 38400, 784, grid=grid(38400), stream=stream0)
        del buf180
        del buf181
        del primals_239
        buf188 = empty_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__81.run(primals_21, buf188, 7500, grid=grid(7500), stream=stream0)
        del primals_21
        buf189 = empty_strided((25, ), (1, ), device='cuda', dtype=torch.float16)
        triton__82.run(primals_22, buf189, 25, grid=grid(25), stream=stream0)
        del primals_22
        buf190 = aten.convolution(buf187, buf188, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf190, (128, 25, 1, 1), (25, 1, 1, 1))
        buf191 = buf190; del buf190  # reuse
        triton__83.run(buf191, buf189, 3200, grid=grid(3200), stream=stream0)
        del buf189
        buf193 = empty_strided((1, 25, 1, 1), (25, 1, 25, 25), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((1, 25, 1, 1), (25, 1, 25, 25), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((25, ), (1, ), device='cuda', dtype=torch.float32)
        buf197 = empty_strided((25, ), (1, ), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((25, ), (1, ), device='cuda', dtype=torch.float32)
        buf855 = empty_strided((1, 25, 1, 1), (25, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__84.run(buf191, primals_134, primals_133, buf193, buf194, buf195, buf197, buf196, buf855, 25, 128, grid=grid(25), stream=stream0)
        del primals_133
        del primals_134
        buf198 = empty_strided((128, 25, 1, 1), (25, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__85.run(buf191, buf194, buf193, primals_23, primals_24, buf198, 3200, grid=grid(3200), stream=stream0)
        del buf193
        del buf194
        del primals_24
        buf199 = empty_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__81.run(primals_25, buf199, 7500, grid=grid(7500), stream=stream0)
        del primals_25
        buf200 = empty_strided((300, ), (1, ), device='cuda', dtype=torch.float16)
        triton__86.run(primals_26, buf200, 300, grid=grid(300), stream=stream0)
        del primals_26
        buf201 = aten.convolution(buf198, buf199, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf201, (128, 300, 1, 1), (300, 1, 1, 1))
        buf202 = buf201; del buf201  # reuse
        triton__87.run(buf202, buf200, 38400, grid=grid(38400), stream=stream0)
        del buf200
        buf203 = empty_strided((128, 300, 28, 28), (235200, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__88.run(buf185, buf202, buf203, 30105600, grid=grid(30105600), stream=stream0)
        buf204 = empty_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__89.run(primals_27, buf204, 18300, grid=grid(18300), stream=stream0)
        del primals_27
        buf205 = aten.convolution(buf203, buf204, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf205, (128, 61, 28, 28), (47824, 784, 28, 1))
        buf206 = empty_strided((1, 61, 1, 1, 13), (793, 1, 793, 793, 61), device='cuda', dtype=torch.float32)
        triton__90.run(buf205, buf206, 793, 7720, grid=grid(793), stream=stream0)
        buf207 = empty_strided((1, 61, 1, 1), (61, 1, 61, 61), device='cuda', dtype=torch.float32)
        triton__91.run(buf206, buf207, 61, 13, grid=grid(61), stream=stream0)
        buf208 = buf206; del buf206  # reuse
        buf210 = empty_strided((1, 61, 1, 1, 13), (793, 1, 793, 793, 61), device='cuda', dtype=torch.float32)
        triton__92.run(buf205, buf207, buf208, buf210, 793, 7720, grid=grid(793), stream=stream0)
        buf209 = buf207; del buf207  # reuse
        buf212 = empty_strided((61, ), (1, ), device='cuda', dtype=torch.float32)
        buf214 = empty_strided((61, ), (1, ), device='cuda', dtype=torch.float32)
        triton__93.run(buf208, primals_242, buf209, buf212, buf214, 61, 13, grid=grid(61), stream=stream0)
        del buf208
        del primals_242
        buf211 = empty_strided((1, 61, 1, 1), (61, 1, 61, 61), device='cuda', dtype=torch.float32)
        buf213 = empty_strided((61, ), (1, ), device='cuda', dtype=torch.float32)
        buf854 = empty_strided((1, 61, 1, 1), (61, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__94.run(buf210, primals_241, buf211, buf213, buf854, 61, 13, grid=grid(61), stream=stream0)
        del buf210
        del primals_241
        buf215 = empty_strided((128, 61, 28, 28), (47824, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__95.run(buf205, buf211, buf209, primals_243, primals_244, buf215, 6121472, grid=grid(6121472), stream=stream0)
        del buf209
        del buf211
        del primals_244
        buf218 = empty_strided((128, 61, 28, 28), (47824, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf216 = as_strided(buf218, (128, 50, 28, 28), (47824, 784, 28, 1))  # alias
        triton__96.run(buf215, buf166, buf216, 5017600, grid=grid(5017600), stream=stream0)
        buf217 = as_strided(buf218, (128, 11, 28, 28), (47824, 784, 28, 1), 39200)  # alias
        triton__97.run(buf215, buf217, 1103872, grid=grid(1103872), stream=stream0)
        del buf215
        buf219 = empty_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__98.run(primals_28, buf219, 22326, grid=grid(22326), stream=stream0)
        del primals_28
        buf220 = aten.convolution(buf218, buf219, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf220, (128, 366, 28, 28), (286944, 784, 28, 1))
        buf222 = empty_strided((1, 366, 1, 1), (366, 1, 366, 366), device='cuda', dtype=torch.float32)
        buf223 = empty_strided((1, 366, 1, 1), (366, 1, 366, 366), device='cuda', dtype=torch.float32)
        buf224 = empty_strided((366, ), (1, ), device='cuda', dtype=torch.float32)
        buf226 = empty_strided((366, ), (1, ), device='cuda', dtype=torch.float32)
        buf225 = empty_strided((366, ), (1, ), device='cuda', dtype=torch.float32)
        buf853 = empty_strided((1, 366, 1, 1), (366, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__99.run(buf220, primals_247, primals_246, buf222, buf223, buf224, buf226, buf225, buf853, 366, 100352, grid=grid(366), stream=stream0)
        del primals_246
        del primals_247
        buf228 = empty_strided((128, 366, 28, 28), (286944, 784, 28, 1), device='cuda', dtype=torch.float16)
        buf852 = empty_strided((128, 366, 28, 28), (286944, 784, 28, 1), device='cuda', dtype=torch.float16)
        triton__100.run(buf220, buf223, buf222, primals_248, primals_249, buf228, buf852, 36728832, grid=grid(36728832), stream=stream0)
        del primals_249
        buf229 = empty_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__101.run(primals_29, buf229, 3294, grid=grid(3294), stream=stream0)
        del primals_29
        buf230 = aten.convolution(buf228, buf229, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 366)
        assert_size_stride(buf230, (128, 366, 14, 14), (71736, 196, 14, 1))
        buf232 = buf223; del buf223  # reuse
        buf233 = buf222; del buf222  # reuse
        buf234 = empty_strided((366, ), (1, ), device='cuda', dtype=torch.float32)
        buf236 = empty_strided((366, ), (1, ), device='cuda', dtype=torch.float32)
        buf235 = empty_strided((366, ), (1, ), device='cuda', dtype=torch.float32)
        buf850 = empty_strided((1, 366, 1, 1), (366, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__102.run(buf230, primals_252, primals_251, buf232, buf233, buf234, buf236, buf235, buf850, 366, 25088, grid=grid(366), stream=stream0)
        del primals_251
        del primals_252
        buf237 = empty_strided((128, 366, 14, 14), (71736, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf239 = empty_strided((128, 366, 1, 1), (366, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__103.run(buf230, buf233, buf232, primals_253, primals_254, buf237, buf239, 46848, 196, grid=grid(46848), stream=stream0)
        del buf232
        del buf233
        del primals_254
        buf240 = empty_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__104.run(primals_30, buf240, 10980, grid=grid(10980), stream=stream0)
        del primals_30
        buf241 = empty_strided((30, ), (1, ), device='cuda', dtype=torch.float16)
        triton__105.run(primals_31, buf241, 30, grid=grid(30), stream=stream0)
        del primals_31
        buf242 = aten.convolution(buf239, buf240, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf242, (128, 30, 1, 1), (30, 1, 1, 1))
        buf243 = buf242; del buf242  # reuse
        triton__106.run(buf243, buf241, 3840, grid=grid(3840), stream=stream0)
        del buf241
        buf245 = empty_strided((1, 30, 1, 1), (30, 1, 30, 30), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((1, 30, 1, 1), (30, 1, 30, 30), device='cuda', dtype=torch.float32)
        buf247 = empty_strided((30, ), (1, ), device='cuda', dtype=torch.float32)
        buf249 = empty_strided((30, ), (1, ), device='cuda', dtype=torch.float32)
        buf248 = empty_strided((30, ), (1, ), device='cuda', dtype=torch.float32)
        buf849 = empty_strided((1, 30, 1, 1), (30, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__107.run(buf243, primals_137, primals_136, buf245, buf246, buf247, buf249, buf248, buf849, 30, 128, grid=grid(30), stream=stream0)
        del primals_136
        del primals_137
        buf250 = empty_strided((128, 30, 1, 1), (30, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__108.run(buf243, buf246, buf245, primals_32, primals_33, buf250, 3840, grid=grid(3840), stream=stream0)
        del buf245
        del buf246
        del primals_33
        buf251 = empty_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__104.run(primals_34, buf251, 10980, grid=grid(10980), stream=stream0)
        del primals_34
        buf252 = empty_strided((366, ), (1, ), device='cuda', dtype=torch.float16)
        triton__109.run(primals_35, buf252, 366, grid=grid(366), stream=stream0)
        del primals_35
        buf253 = aten.convolution(buf250, buf251, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf253, (128, 366, 1, 1), (366, 1, 1, 1))
        buf254 = buf253; del buf253  # reuse
        triton__110.run(buf254, buf252, 46848, grid=grid(46848), stream=stream0)
        del buf252
        buf255 = empty_strided((128, 366, 14, 14), (71736, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__111.run(buf237, buf254, buf255, 9182208, grid=grid(9182208), stream=stream0)
        buf256 = empty_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__112.run(primals_36, buf256, 26352, grid=grid(26352), stream=stream0)
        del primals_36
        buf257 = aten.convolution(buf255, buf256, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf257, (128, 72, 14, 14), (14112, 196, 14, 1))
        buf258 = empty_strided((1, 72, 1, 1, 4), (288, 1, 288, 288, 72), device='cuda', dtype=torch.float32)
        triton__113.run(buf257, buf258, 288, 6272, grid=grid(288), stream=stream0)
        buf259 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        triton__114.run(buf258, buf259, 72, 4, grid=grid(72), stream=stream0)
        buf260 = buf258; del buf258  # reuse
        buf262 = empty_strided((1, 72, 1, 1, 4), (288, 1, 288, 288, 72), device='cuda', dtype=torch.float32)
        triton__115.run(buf257, buf259, buf260, buf262, 288, 6272, grid=grid(288), stream=stream0)
        buf261 = buf259; del buf259  # reuse
        buf264 = empty_strided((72, ), (1, ), device='cuda', dtype=torch.float32)
        buf266 = empty_strided((72, ), (1, ), device='cuda', dtype=torch.float32)
        triton__116.run(buf260, primals_257, buf261, buf264, buf266, 72, 4, grid=grid(72), stream=stream0)
        del buf260
        del primals_257
        buf263 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((72, ), (1, ), device='cuda', dtype=torch.float32)
        buf848 = empty_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__117.run(buf262, primals_256, buf263, buf265, buf848, 72, 4, grid=grid(72), stream=stream0)
        del buf262
        del primals_256
        buf267 = empty_strided((128, 72, 14, 14), (14112, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__118.run(buf257, buf263, buf261, primals_258, primals_259, buf267, 1806336, grid=grid(1806336), stream=stream0)
        del buf261
        del buf263
        del primals_259
        buf268 = empty_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__119.run(primals_37, buf268, 31104, grid=grid(31104), stream=stream0)
        del primals_37
        buf269 = aten.convolution(buf267, buf268, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf269, (128, 432, 14, 14), (84672, 196, 14, 1))
        buf271 = as_strided(buf71, (1, 432, 1, 1), (432, 1, 432, 432)); del buf71  # reuse
        buf272 = as_strided(buf69, (1, 432, 1, 1), (432, 1, 432, 432)); del buf69  # reuse
        buf273 = empty_strided((432, ), (1, ), device='cuda', dtype=torch.float32)
        buf275 = empty_strided((432, ), (1, ), device='cuda', dtype=torch.float32)
        buf274 = empty_strided((432, ), (1, ), device='cuda', dtype=torch.float32)
        buf847 = empty_strided((1, 432, 1, 1), (432, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__120.run(buf269, primals_262, primals_261, buf271, buf272, buf273, buf275, buf274, buf847, 432, 25088, grid=grid(432), stream=stream0)
        del primals_261
        del primals_262
        buf277 = empty_strided((128, 432, 14, 14), (84672, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf846 = empty_strided((128, 432, 14, 14), (84672, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__121.run(buf269, buf272, buf271, primals_263, primals_264, buf277, buf846, 10838016, grid=grid(10838016), stream=stream0)
        del primals_264
        buf278 = empty_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__122.run(primals_38, buf278, 3888, grid=grid(3888), stream=stream0)
        del primals_38
        buf279 = aten.convolution(buf277, buf278, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 432)
        assert_size_stride(buf279, (128, 432, 14, 14), (84672, 196, 14, 1))
        buf281 = buf272; del buf272  # reuse
        buf282 = buf271; del buf271  # reuse
        buf283 = empty_strided((432, ), (1, ), device='cuda', dtype=torch.float32)
        buf285 = empty_strided((432, ), (1, ), device='cuda', dtype=torch.float32)
        buf284 = empty_strided((432, ), (1, ), device='cuda', dtype=torch.float32)
        buf844 = empty_strided((1, 432, 1, 1), (432, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__120.run(buf279, primals_267, primals_266, buf281, buf282, buf283, buf285, buf284, buf844, 432, 25088, grid=grid(432), stream=stream0)
        del primals_266
        del primals_267
        buf286 = empty_strided((128, 432, 14, 14), (84672, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf288 = empty_strided((128, 432, 1, 1), (432, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__123.run(buf279, buf282, buf281, primals_268, primals_269, buf286, buf288, 55296, 196, grid=grid(55296), stream=stream0)
        del buf281
        del buf282
        del primals_269
        buf289 = empty_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__124.run(primals_39, buf289, 15552, grid=grid(15552), stream=stream0)
        del primals_39
        buf290 = empty_strided((36, ), (1, ), device='cuda', dtype=torch.float16)
        triton__125.run(primals_40, buf290, 36, grid=grid(36), stream=stream0)
        del primals_40
        buf291 = aten.convolution(buf288, buf289, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf291, (128, 36, 1, 1), (36, 1, 1, 1))
        buf292 = buf291; del buf291  # reuse
        triton__126.run(buf292, buf290, 4608, grid=grid(4608), stream=stream0)
        del buf290
        buf294 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf295 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf296 = empty_strided((36, ), (1, ), device='cuda', dtype=torch.float32)
        buf298 = empty_strided((36, ), (1, ), device='cuda', dtype=torch.float32)
        buf297 = empty_strided((36, ), (1, ), device='cuda', dtype=torch.float32)
        buf843 = empty_strided((1, 36, 1, 1), (36, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__127.run(buf292, primals_140, primals_139, buf294, buf295, buf296, buf298, buf297, buf843, 36, 128, grid=grid(36), stream=stream0)
        del primals_139
        del primals_140
        buf299 = empty_strided((128, 36, 1, 1), (36, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__128.run(buf292, buf295, buf294, primals_41, primals_42, buf299, 4608, grid=grid(4608), stream=stream0)
        del buf294
        del buf295
        del primals_42
        buf300 = empty_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__124.run(primals_43, buf300, 15552, grid=grid(15552), stream=stream0)
        del primals_43
        buf301 = empty_strided((432, ), (1, ), device='cuda', dtype=torch.float16)
        triton__129.run(primals_44, buf301, 432, grid=grid(432), stream=stream0)
        del primals_44
        buf302 = aten.convolution(buf299, buf300, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf302, (128, 432, 1, 1), (432, 1, 1, 1))
        buf303 = buf302; del buf302  # reuse
        triton__130.run(buf303, buf301, 55296, grid=grid(55296), stream=stream0)
        del buf301
        buf304 = empty_strided((128, 432, 14, 14), (84672, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__131.run(buf286, buf303, buf304, 10838016, grid=grid(10838016), stream=stream0)
        buf305 = empty_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__132.run(primals_45, buf305, 36288, grid=grid(36288), stream=stream0)
        del primals_45
        buf306 = aten.convolution(buf304, buf305, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf306, (128, 84, 14, 14), (16464, 196, 14, 1))
        buf307 = empty_strided((1, 84, 1, 1, 4), (336, 1, 336, 336, 84), device='cuda', dtype=torch.float32)
        triton__133.run(buf306, buf307, 336, 6272, grid=grid(336), stream=stream0)
        buf308 = empty_strided((1, 84, 1, 1), (84, 1, 84, 84), device='cuda', dtype=torch.float32)
        triton__134.run(buf307, buf308, 84, 4, grid=grid(84), stream=stream0)
        buf309 = buf307; del buf307  # reuse
        buf311 = empty_strided((1, 84, 1, 1, 4), (336, 1, 336, 336, 84), device='cuda', dtype=torch.float32)
        triton__135.run(buf306, buf308, buf309, buf311, 336, 6272, grid=grid(336), stream=stream0)
        buf310 = buf308; del buf308  # reuse
        buf313 = empty_strided((84, ), (1, ), device='cuda', dtype=torch.float32)
        buf315 = empty_strided((84, ), (1, ), device='cuda', dtype=torch.float32)
        triton__136.run(buf309, primals_272, buf310, buf313, buf315, 84, 4, grid=grid(84), stream=stream0)
        del buf309
        del primals_272
        buf312 = empty_strided((1, 84, 1, 1), (84, 1, 84, 84), device='cuda', dtype=torch.float32)
        buf314 = empty_strided((84, ), (1, ), device='cuda', dtype=torch.float32)
        buf842 = empty_strided((1, 84, 1, 1), (84, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__137.run(buf311, primals_271, buf312, buf314, buf842, 84, 4, grid=grid(84), stream=stream0)
        del buf311
        del primals_271
        buf316 = empty_strided((128, 84, 14, 14), (16464, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__138.run(buf306, buf312, buf310, primals_273, primals_274, buf316, 2107392, grid=grid(2107392), stream=stream0)
        del buf310
        del buf312
        del primals_274
        buf319 = empty_strided((128, 84, 14, 14), (16464, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf317 = as_strided(buf319, (128, 72, 14, 14), (16464, 196, 14, 1))  # alias
        triton__139.run(buf316, buf267, buf317, 1806336, grid=grid(1806336), stream=stream0)
        buf318 = as_strided(buf319, (128, 12, 14, 14), (16464, 196, 14, 1), 14112)  # alias
        triton__140.run(buf316, buf318, 301056, grid=grid(301056), stream=stream0)
        del buf316
        buf320 = empty_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__141.run(primals_46, buf320, 42336, grid=grid(42336), stream=stream0)
        del primals_46
        buf321 = aten.convolution(buf319, buf320, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf321, (128, 504, 14, 14), (98784, 196, 14, 1))
        buf323 = empty_strided((1, 504, 1, 1), (504, 1, 504, 504), device='cuda', dtype=torch.float32)
        buf324 = empty_strided((1, 504, 1, 1), (504, 1, 504, 504), device='cuda', dtype=torch.float32)
        buf325 = empty_strided((504, ), (1, ), device='cuda', dtype=torch.float32)
        buf327 = empty_strided((504, ), (1, ), device='cuda', dtype=torch.float32)
        buf326 = empty_strided((504, ), (1, ), device='cuda', dtype=torch.float32)
        buf841 = empty_strided((1, 504, 1, 1), (504, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__142.run(buf321, primals_277, primals_276, buf323, buf324, buf325, buf327, buf326, buf841, 504, 25088, grid=grid(504), stream=stream0)
        del primals_276
        del primals_277
        buf329 = empty_strided((128, 504, 14, 14), (98784, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf840 = empty_strided((128, 504, 14, 14), (98784, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__143.run(buf321, buf324, buf323, primals_278, primals_279, buf329, buf840, 12644352, grid=grid(12644352), stream=stream0)
        del primals_279
        buf330 = empty_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__144.run(primals_47, buf330, 4536, grid=grid(4536), stream=stream0)
        del primals_47
        buf331 = aten.convolution(buf329, buf330, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 504)
        assert_size_stride(buf331, (128, 504, 14, 14), (98784, 196, 14, 1))
        buf333 = buf324; del buf324  # reuse
        buf334 = buf323; del buf323  # reuse
        buf335 = empty_strided((504, ), (1, ), device='cuda', dtype=torch.float32)
        buf337 = empty_strided((504, ), (1, ), device='cuda', dtype=torch.float32)
        buf336 = empty_strided((504, ), (1, ), device='cuda', dtype=torch.float32)
        buf838 = empty_strided((1, 504, 1, 1), (504, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__142.run(buf331, primals_282, primals_281, buf333, buf334, buf335, buf337, buf336, buf838, 504, 25088, grid=grid(504), stream=stream0)
        del primals_281
        del primals_282
        buf338 = empty_strided((128, 504, 14, 14), (98784, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf340 = empty_strided((128, 504, 1, 1), (504, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__145.run(buf331, buf334, buf333, primals_283, primals_284, buf338, buf340, 64512, 196, grid=grid(64512), stream=stream0)
        del buf333
        del buf334
        del primals_284
        buf341 = empty_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__146.run(primals_48, buf341, 21168, grid=grid(21168), stream=stream0)
        del primals_48
        buf342 = empty_strided((42, ), (1, ), device='cuda', dtype=torch.float16)
        triton__147.run(primals_49, buf342, 42, grid=grid(42), stream=stream0)
        del primals_49
        buf343 = aten.convolution(buf340, buf341, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf343, (128, 42, 1, 1), (42, 1, 1, 1))
        buf344 = buf343; del buf343  # reuse
        triton__148.run(buf344, buf342, 5376, grid=grid(5376), stream=stream0)
        del buf342
        buf346 = empty_strided((1, 42, 1, 1), (42, 1, 42, 42), device='cuda', dtype=torch.float32)
        buf347 = empty_strided((1, 42, 1, 1), (42, 1, 42, 42), device='cuda', dtype=torch.float32)
        buf348 = empty_strided((42, ), (1, ), device='cuda', dtype=torch.float32)
        buf350 = empty_strided((42, ), (1, ), device='cuda', dtype=torch.float32)
        buf349 = empty_strided((42, ), (1, ), device='cuda', dtype=torch.float32)
        buf837 = empty_strided((1, 42, 1, 1), (42, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__149.run(buf344, primals_143, primals_142, buf346, buf347, buf348, buf350, buf349, buf837, 42, 128, grid=grid(42), stream=stream0)
        del primals_142
        del primals_143
        buf351 = empty_strided((128, 42, 1, 1), (42, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__150.run(buf344, buf347, buf346, primals_50, primals_51, buf351, 5376, grid=grid(5376), stream=stream0)
        del buf346
        del buf347
        del primals_51
        buf352 = empty_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__146.run(primals_52, buf352, 21168, grid=grid(21168), stream=stream0)
        del primals_52
        buf353 = empty_strided((504, ), (1, ), device='cuda', dtype=torch.float16)
        triton__151.run(primals_53, buf353, 504, grid=grid(504), stream=stream0)
        del primals_53
        buf354 = aten.convolution(buf351, buf352, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf354, (128, 504, 1, 1), (504, 1, 1, 1))
        buf355 = buf354; del buf354  # reuse
        triton__152.run(buf355, buf353, 64512, grid=grid(64512), stream=stream0)
        del buf353
        buf356 = empty_strided((128, 504, 14, 14), (98784, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__153.run(buf338, buf355, buf356, 12644352, grid=grid(12644352), stream=stream0)
        buf357 = empty_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__154.run(primals_54, buf357, 47880, grid=grid(47880), stream=stream0)
        del primals_54
        buf358 = aten.convolution(buf356, buf357, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf358, (128, 95, 14, 14), (18620, 196, 14, 1))
        buf359 = empty_strided((1, 95, 1, 1, 4), (380, 1, 380, 380, 95), device='cuda', dtype=torch.float32)
        triton__155.run(buf358, buf359, 380, 6272, grid=grid(380), stream=stream0)
        buf360 = empty_strided((1, 95, 1, 1), (95, 1, 95, 95), device='cuda', dtype=torch.float32)
        triton__156.run(buf359, buf360, 95, 4, grid=grid(95), stream=stream0)
        buf361 = buf359; del buf359  # reuse
        buf363 = empty_strided((1, 95, 1, 1, 4), (380, 1, 380, 380, 95), device='cuda', dtype=torch.float32)
        triton__157.run(buf358, buf360, buf361, buf363, 380, 6272, grid=grid(380), stream=stream0)
        buf362 = buf360; del buf360  # reuse
        buf365 = empty_strided((95, ), (1, ), device='cuda', dtype=torch.float32)
        buf367 = empty_strided((95, ), (1, ), device='cuda', dtype=torch.float32)
        triton__158.run(buf361, primals_287, buf362, buf365, buf367, 95, 4, grid=grid(95), stream=stream0)
        del buf361
        del primals_287
        buf364 = empty_strided((1, 95, 1, 1), (95, 1, 95, 95), device='cuda', dtype=torch.float32)
        buf366 = empty_strided((95, ), (1, ), device='cuda', dtype=torch.float32)
        buf836 = empty_strided((1, 95, 1, 1), (95, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__159.run(buf363, primals_286, buf364, buf366, buf836, 95, 4, grid=grid(95), stream=stream0)
        del buf363
        del primals_286
        buf368 = empty_strided((128, 95, 14, 14), (18620, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__160.run(buf358, buf364, buf362, primals_288, primals_289, buf368, 2383360, grid=grid(2383360), stream=stream0)
        del buf362
        del buf364
        del primals_289
        buf371 = empty_strided((128, 95, 14, 14), (18620, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf369 = as_strided(buf371, (128, 84, 14, 14), (18620, 196, 14, 1))  # alias
        triton__161.run(buf368, buf319, buf369, 2107392, grid=grid(2107392), stream=stream0)
        buf370 = as_strided(buf371, (128, 11, 14, 14), (18620, 196, 14, 1), 16464)  # alias
        triton__162.run(buf368, buf370, 275968, grid=grid(275968), stream=stream0)
        del buf368
        buf372 = empty_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__163.run(primals_55, buf372, 54150, grid=grid(54150), stream=stream0)
        del primals_55
        buf373 = aten.convolution(buf371, buf372, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf373, (128, 570, 14, 14), (111720, 196, 14, 1))
        buf375 = empty_strided((1, 570, 1, 1), (570, 1, 570, 570), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((1, 570, 1, 1), (570, 1, 570, 570), device='cuda', dtype=torch.float32)
        buf377 = empty_strided((570, ), (1, ), device='cuda', dtype=torch.float32)
        buf379 = empty_strided((570, ), (1, ), device='cuda', dtype=torch.float32)
        buf378 = empty_strided((570, ), (1, ), device='cuda', dtype=torch.float32)
        buf835 = empty_strided((1, 570, 1, 1), (570, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__164.run(buf373, primals_292, primals_291, buf375, buf376, buf377, buf379, buf378, buf835, 570, 25088, grid=grid(570), stream=stream0)
        del primals_291
        del primals_292
        buf381 = empty_strided((128, 570, 14, 14), (111720, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf834 = empty_strided((128, 570, 14, 14), (111720, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__165.run(buf373, buf376, buf375, primals_293, primals_294, buf381, buf834, 14300160, grid=grid(14300160), stream=stream0)
        del primals_294
        buf382 = empty_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__166.run(primals_56, buf382, 5130, grid=grid(5130), stream=stream0)
        del primals_56
        buf383 = aten.convolution(buf381, buf382, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 570)
        assert_size_stride(buf383, (128, 570, 14, 14), (111720, 196, 14, 1))
        buf385 = buf376; del buf376  # reuse
        buf386 = buf375; del buf375  # reuse
        buf387 = empty_strided((570, ), (1, ), device='cuda', dtype=torch.float32)
        buf389 = empty_strided((570, ), (1, ), device='cuda', dtype=torch.float32)
        buf388 = empty_strided((570, ), (1, ), device='cuda', dtype=torch.float32)
        buf832 = empty_strided((1, 570, 1, 1), (570, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__164.run(buf383, primals_297, primals_296, buf385, buf386, buf387, buf389, buf388, buf832, 570, 25088, grid=grid(570), stream=stream0)
        del primals_296
        del primals_297
        buf390 = empty_strided((128, 570, 14, 14), (111720, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf392 = empty_strided((128, 570, 1, 1), (570, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__167.run(buf383, buf386, buf385, primals_298, primals_299, buf390, buf392, 72960, 196, grid=grid(72960), stream=stream0)
        del buf385
        del buf386
        del primals_299
        buf393 = empty_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__168.run(primals_57, buf393, 26790, grid=grid(26790), stream=stream0)
        del primals_57
        buf394 = empty_strided((47, ), (1, ), device='cuda', dtype=torch.float16)
        triton__169.run(primals_58, buf394, 47, grid=grid(47), stream=stream0)
        del primals_58
        buf395 = aten.convolution(buf392, buf393, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf395, (128, 47, 1, 1), (47, 1, 1, 1))
        buf396 = buf395; del buf395  # reuse
        triton__170.run(buf396, buf394, 6016, grid=grid(6016), stream=stream0)
        del buf394
        buf398 = empty_strided((1, 47, 1, 1), (47, 1, 47, 47), device='cuda', dtype=torch.float32)
        buf399 = empty_strided((1, 47, 1, 1), (47, 1, 47, 47), device='cuda', dtype=torch.float32)
        buf400 = empty_strided((47, ), (1, ), device='cuda', dtype=torch.float32)
        buf402 = empty_strided((47, ), (1, ), device='cuda', dtype=torch.float32)
        buf401 = empty_strided((47, ), (1, ), device='cuda', dtype=torch.float32)
        buf831 = empty_strided((1, 47, 1, 1), (47, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__171.run(buf396, primals_146, primals_145, buf398, buf399, buf400, buf402, buf401, buf831, 47, 128, grid=grid(47), stream=stream0)
        del primals_145
        del primals_146
        buf403 = empty_strided((128, 47, 1, 1), (47, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__172.run(buf396, buf399, buf398, primals_59, primals_60, buf403, 6016, grid=grid(6016), stream=stream0)
        del buf398
        del buf399
        del primals_60
        buf404 = empty_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__168.run(primals_61, buf404, 26790, grid=grid(26790), stream=stream0)
        del primals_61
        buf405 = empty_strided((570, ), (1, ), device='cuda', dtype=torch.float16)
        triton__173.run(primals_62, buf405, 570, grid=grid(570), stream=stream0)
        del primals_62
        buf406 = aten.convolution(buf403, buf404, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf406, (128, 570, 1, 1), (570, 1, 1, 1))
        buf407 = buf406; del buf406  # reuse
        triton__174.run(buf407, buf405, 72960, grid=grid(72960), stream=stream0)
        del buf405
        buf408 = empty_strided((128, 570, 14, 14), (111720, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__175.run(buf390, buf407, buf408, 14300160, grid=grid(14300160), stream=stream0)
        buf409 = empty_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__176.run(primals_63, buf409, 60420, grid=grid(60420), stream=stream0)
        del primals_63
        buf410 = aten.convolution(buf408, buf409, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf410, (128, 106, 14, 14), (20776, 196, 14, 1))
        buf411 = empty_strided((1, 106, 1, 1, 4), (424, 1, 424, 424, 106), device='cuda', dtype=torch.float32)
        triton__177.run(buf410, buf411, 424, 6272, grid=grid(424), stream=stream0)
        buf412 = empty_strided((1, 106, 1, 1), (106, 1, 106, 106), device='cuda', dtype=torch.float32)
        triton__178.run(buf411, buf412, 106, 4, grid=grid(106), stream=stream0)
        buf413 = buf411; del buf411  # reuse
        buf415 = empty_strided((1, 106, 1, 1, 4), (424, 1, 424, 424, 106), device='cuda', dtype=torch.float32)
        triton__179.run(buf410, buf412, buf413, buf415, 424, 6272, grid=grid(424), stream=stream0)
        buf414 = buf412; del buf412  # reuse
        buf417 = empty_strided((106, ), (1, ), device='cuda', dtype=torch.float32)
        buf419 = empty_strided((106, ), (1, ), device='cuda', dtype=torch.float32)
        triton__180.run(buf413, primals_302, buf414, buf417, buf419, 106, 4, grid=grid(106), stream=stream0)
        del buf413
        del primals_302
        buf416 = empty_strided((1, 106, 1, 1), (106, 1, 106, 106), device='cuda', dtype=torch.float32)
        buf418 = empty_strided((106, ), (1, ), device='cuda', dtype=torch.float32)
        buf830 = empty_strided((1, 106, 1, 1), (106, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__181.run(buf415, primals_301, buf416, buf418, buf830, 106, 4, grid=grid(106), stream=stream0)
        del buf415
        del primals_301
        buf420 = empty_strided((128, 106, 14, 14), (20776, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__182.run(buf410, buf416, buf414, primals_303, primals_304, buf420, 2659328, grid=grid(2659328), stream=stream0)
        del buf414
        del buf416
        del primals_304
        buf423 = empty_strided((128, 106, 14, 14), (20776, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf421 = as_strided(buf423, (128, 95, 14, 14), (20776, 196, 14, 1))  # alias
        triton__183.run(buf420, buf371, buf421, 2383360, grid=grid(2383360), stream=stream0)
        buf422 = as_strided(buf423, (128, 11, 14, 14), (20776, 196, 14, 1), 18620)  # alias
        triton__184.run(buf420, buf422, 275968, grid=grid(275968), stream=stream0)
        del buf420
        buf424 = empty_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__185.run(primals_64, buf424, 67416, grid=grid(67416), stream=stream0)
        del primals_64
        buf425 = aten.convolution(buf423, buf424, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf425, (128, 636, 14, 14), (124656, 196, 14, 1))
        buf427 = empty_strided((1, 636, 1, 1), (636, 1, 636, 636), device='cuda', dtype=torch.float32)
        buf428 = empty_strided((1, 636, 1, 1), (636, 1, 636, 636), device='cuda', dtype=torch.float32)
        buf429 = empty_strided((636, ), (1, ), device='cuda', dtype=torch.float32)
        buf431 = empty_strided((636, ), (1, ), device='cuda', dtype=torch.float32)
        buf430 = empty_strided((636, ), (1, ), device='cuda', dtype=torch.float32)
        buf829 = empty_strided((1, 636, 1, 1), (636, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__186.run(buf425, primals_307, primals_306, buf427, buf428, buf429, buf431, buf430, buf829, 636, 25088, grid=grid(636), stream=stream0)
        del primals_306
        del primals_307
        buf433 = empty_strided((128, 636, 14, 14), (124656, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf828 = empty_strided((128, 636, 14, 14), (124656, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__187.run(buf425, buf428, buf427, primals_308, primals_309, buf433, buf828, 15955968, grid=grid(15955968), stream=stream0)
        del primals_309
        buf434 = empty_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__188.run(primals_65, buf434, 5724, grid=grid(5724), stream=stream0)
        del primals_65
        buf435 = aten.convolution(buf433, buf434, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 636)
        assert_size_stride(buf435, (128, 636, 14, 14), (124656, 196, 14, 1))
        buf437 = buf428; del buf428  # reuse
        buf438 = buf427; del buf427  # reuse
        buf439 = empty_strided((636, ), (1, ), device='cuda', dtype=torch.float32)
        buf441 = empty_strided((636, ), (1, ), device='cuda', dtype=torch.float32)
        buf440 = empty_strided((636, ), (1, ), device='cuda', dtype=torch.float32)
        buf826 = empty_strided((1, 636, 1, 1), (636, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__186.run(buf435, primals_312, primals_311, buf437, buf438, buf439, buf441, buf440, buf826, 636, 25088, grid=grid(636), stream=stream0)
        del primals_311
        del primals_312
        buf442 = empty_strided((128, 636, 14, 14), (124656, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf444 = empty_strided((128, 636, 1, 1), (636, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__189.run(buf435, buf438, buf437, primals_313, primals_314, buf442, buf444, 81408, 196, grid=grid(81408), stream=stream0)
        del buf437
        del buf438
        del primals_314
        buf445 = empty_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__190.run(primals_66, buf445, 33708, grid=grid(33708), stream=stream0)
        del primals_66
        buf446 = empty_strided((53, ), (1, ), device='cuda', dtype=torch.float16)
        triton__191.run(primals_67, buf446, 53, grid=grid(53), stream=stream0)
        del primals_67
        buf447 = aten.convolution(buf444, buf445, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf447, (128, 53, 1, 1), (53, 1, 1, 1))
        buf448 = buf447; del buf447  # reuse
        triton__192.run(buf448, buf446, 6784, grid=grid(6784), stream=stream0)
        del buf446
        buf450 = empty_strided((1, 53, 1, 1), (53, 1, 53, 53), device='cuda', dtype=torch.float32)
        buf451 = empty_strided((1, 53, 1, 1), (53, 1, 53, 53), device='cuda', dtype=torch.float32)
        buf452 = empty_strided((53, ), (1, ), device='cuda', dtype=torch.float32)
        buf454 = empty_strided((53, ), (1, ), device='cuda', dtype=torch.float32)
        buf453 = empty_strided((53, ), (1, ), device='cuda', dtype=torch.float32)
        buf825 = empty_strided((1, 53, 1, 1), (53, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__193.run(buf448, primals_149, primals_148, buf450, buf451, buf452, buf454, buf453, buf825, 53, 128, grid=grid(53), stream=stream0)
        del primals_148
        del primals_149
        buf455 = empty_strided((128, 53, 1, 1), (53, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__194.run(buf448, buf451, buf450, primals_68, primals_69, buf455, 6784, grid=grid(6784), stream=stream0)
        del buf450
        del buf451
        del primals_69
        buf456 = empty_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__190.run(primals_70, buf456, 33708, grid=grid(33708), stream=stream0)
        del primals_70
        buf457 = empty_strided((636, ), (1, ), device='cuda', dtype=torch.float16)
        triton__195.run(primals_71, buf457, 636, grid=grid(636), stream=stream0)
        del primals_71
        buf458 = aten.convolution(buf455, buf456, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf458, (128, 636, 1, 1), (636, 1, 1, 1))
        buf459 = buf458; del buf458  # reuse
        triton__196.run(buf459, buf457, 81408, grid=grid(81408), stream=stream0)
        del buf457
        buf460 = empty_strided((128, 636, 14, 14), (124656, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__197.run(buf442, buf459, buf460, 15955968, grid=grid(15955968), stream=stream0)
        buf461 = empty_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__198.run(primals_72, buf461, 74412, grid=grid(74412), stream=stream0)
        del primals_72
        buf462 = aten.convolution(buf460, buf461, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf462, (128, 117, 14, 14), (22932, 196, 14, 1))
        buf463 = empty_strided((1, 117, 1, 1, 4), (468, 1, 468, 468, 117), device='cuda', dtype=torch.float32)
        triton__199.run(buf462, buf463, 468, 6272, grid=grid(468), stream=stream0)
        buf464 = empty_strided((1, 117, 1, 1), (117, 1, 117, 117), device='cuda', dtype=torch.float32)
        triton__200.run(buf463, buf464, 117, 4, grid=grid(117), stream=stream0)
        buf465 = buf463; del buf463  # reuse
        buf467 = empty_strided((1, 117, 1, 1, 4), (468, 1, 468, 468, 117), device='cuda', dtype=torch.float32)
        triton__201.run(buf462, buf464, buf465, buf467, 468, 6272, grid=grid(468), stream=stream0)
        buf466 = buf464; del buf464  # reuse
        buf469 = empty_strided((117, ), (1, ), device='cuda', dtype=torch.float32)
        buf471 = empty_strided((117, ), (1, ), device='cuda', dtype=torch.float32)
        triton__202.run(buf465, primals_317, buf466, buf469, buf471, 117, 4, grid=grid(117), stream=stream0)
        del buf465
        del primals_317
        buf468 = empty_strided((1, 117, 1, 1), (117, 1, 117, 117), device='cuda', dtype=torch.float32)
        buf470 = empty_strided((117, ), (1, ), device='cuda', dtype=torch.float32)
        buf824 = empty_strided((1, 117, 1, 1), (117, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__203.run(buf467, primals_316, buf468, buf470, buf824, 117, 4, grid=grid(117), stream=stream0)
        del buf467
        del primals_316
        buf472 = empty_strided((128, 117, 14, 14), (22932, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__204.run(buf462, buf468, buf466, primals_318, primals_319, buf472, 2935296, grid=grid(2935296), stream=stream0)
        del buf466
        del buf468
        del primals_319
        buf475 = empty_strided((128, 117, 14, 14), (22932, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf473 = as_strided(buf475, (128, 106, 14, 14), (22932, 196, 14, 1))  # alias
        triton__205.run(buf472, buf423, buf473, 2659328, grid=grid(2659328), stream=stream0)
        buf474 = as_strided(buf475, (128, 11, 14, 14), (22932, 196, 14, 1), 20776)  # alias
        triton__206.run(buf472, buf474, 275968, grid=grid(275968), stream=stream0)
        del buf472
        buf476 = empty_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__207.run(primals_73, buf476, 82134, grid=grid(82134), stream=stream0)
        del primals_73
        buf477 = aten.convolution(buf475, buf476, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf477, (128, 702, 14, 14), (137592, 196, 14, 1))
        buf479 = empty_strided((1, 702, 1, 1), (702, 1, 702, 702), device='cuda', dtype=torch.float32)
        buf480 = empty_strided((1, 702, 1, 1), (702, 1, 702, 702), device='cuda', dtype=torch.float32)
        buf481 = empty_strided((702, ), (1, ), device='cuda', dtype=torch.float32)
        buf483 = empty_strided((702, ), (1, ), device='cuda', dtype=torch.float32)
        buf482 = empty_strided((702, ), (1, ), device='cuda', dtype=torch.float32)
        buf823 = empty_strided((1, 702, 1, 1), (702, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__208.run(buf477, primals_322, primals_321, buf479, buf480, buf481, buf483, buf482, buf823, 702, 25088, grid=grid(702), stream=stream0)
        del primals_321
        del primals_322
        buf485 = empty_strided((128, 702, 14, 14), (137592, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf822 = empty_strided((128, 702, 14, 14), (137592, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__209.run(buf477, buf480, buf479, primals_323, primals_324, buf485, buf822, 17611776, grid=grid(17611776), stream=stream0)
        del primals_324
        buf486 = empty_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__210.run(primals_74, buf486, 6318, grid=grid(6318), stream=stream0)
        del primals_74
        buf487 = aten.convolution(buf485, buf486, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 702)
        assert_size_stride(buf487, (128, 702, 14, 14), (137592, 196, 14, 1))
        buf489 = buf480; del buf480  # reuse
        buf490 = buf479; del buf479  # reuse
        buf491 = empty_strided((702, ), (1, ), device='cuda', dtype=torch.float32)
        buf493 = empty_strided((702, ), (1, ), device='cuda', dtype=torch.float32)
        buf492 = empty_strided((702, ), (1, ), device='cuda', dtype=torch.float32)
        buf820 = empty_strided((1, 702, 1, 1), (702, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__208.run(buf487, primals_327, primals_326, buf489, buf490, buf491, buf493, buf492, buf820, 702, 25088, grid=grid(702), stream=stream0)
        del primals_326
        del primals_327
        buf494 = empty_strided((128, 702, 14, 14), (137592, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf496 = empty_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__211.run(buf487, buf490, buf489, primals_328, primals_329, buf494, buf496, 89856, 196, grid=grid(89856), stream=stream0)
        del buf489
        del buf490
        del primals_329
        buf497 = empty_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__212.run(primals_75, buf497, 40716, grid=grid(40716), stream=stream0)
        del primals_75
        buf498 = empty_strided((58, ), (1, ), device='cuda', dtype=torch.float16)
        triton__213.run(primals_76, buf498, 58, grid=grid(58), stream=stream0)
        del primals_76
        buf499 = aten.convolution(buf496, buf497, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf499, (128, 58, 1, 1), (58, 1, 1, 1))
        buf500 = buf499; del buf499  # reuse
        triton__214.run(buf500, buf498, 7424, grid=grid(7424), stream=stream0)
        del buf498
        buf502 = empty_strided((1, 58, 1, 1), (58, 1, 58, 58), device='cuda', dtype=torch.float32)
        buf503 = empty_strided((1, 58, 1, 1), (58, 1, 58, 58), device='cuda', dtype=torch.float32)
        buf504 = empty_strided((58, ), (1, ), device='cuda', dtype=torch.float32)
        buf506 = empty_strided((58, ), (1, ), device='cuda', dtype=torch.float32)
        buf505 = empty_strided((58, ), (1, ), device='cuda', dtype=torch.float32)
        buf819 = empty_strided((1, 58, 1, 1), (58, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__215.run(buf500, primals_152, primals_151, buf502, buf503, buf504, buf506, buf505, buf819, 58, 128, grid=grid(58), stream=stream0)
        del primals_151
        del primals_152
        buf507 = empty_strided((128, 58, 1, 1), (58, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__216.run(buf500, buf503, buf502, primals_77, primals_78, buf507, 7424, grid=grid(7424), stream=stream0)
        del buf502
        del buf503
        del primals_78
        buf508 = empty_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__212.run(primals_79, buf508, 40716, grid=grid(40716), stream=stream0)
        del primals_79
        buf509 = empty_strided((702, ), (1, ), device='cuda', dtype=torch.float16)
        triton__217.run(primals_80, buf509, 702, grid=grid(702), stream=stream0)
        del primals_80
        buf510 = aten.convolution(buf507, buf508, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf510, (128, 702, 1, 1), (702, 1, 1, 1))
        buf511 = buf510; del buf510  # reuse
        triton__218.run(buf511, buf509, 89856, grid=grid(89856), stream=stream0)
        del buf509
        buf512 = empty_strided((128, 702, 14, 14), (137592, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__219.run(buf494, buf511, buf512, 17611776, grid=grid(17611776), stream=stream0)
        buf513 = empty_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__220.run(primals_81, buf513, 89856, grid=grid(89856), stream=stream0)
        del primals_81
        buf514 = aten.convolution(buf512, buf513, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf514, (128, 128, 14, 14), (25088, 196, 14, 1))
        buf515 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        triton__221.run(buf514, buf515, 512, 6272, grid=grid(512), stream=stream0)
        buf516 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        triton__222.run(buf515, buf516, 128, 4, grid=grid(128), stream=stream0)
        buf517 = buf515; del buf515  # reuse
        buf519 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        triton__223.run(buf514, buf516, buf517, buf519, 512, 6272, grid=grid(512), stream=stream0)
        buf518 = buf516; del buf516  # reuse
        buf521 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf523 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        triton__224.run(buf517, primals_332, buf518, buf521, buf523, 128, 4, grid=grid(128), stream=stream0)
        del buf517
        del primals_332
        buf520 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf522 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf818 = empty_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__225.run(buf519, primals_331, buf520, buf522, buf818, 128, 4, grid=grid(128), stream=stream0)
        del buf519
        del primals_331
        buf524 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__226.run(buf514, buf520, buf518, primals_333, primals_334, buf524, 3211264, grid=grid(3211264), stream=stream0)
        del buf518
        del buf520
        del primals_334
        buf527 = empty_strided((128, 128, 14, 14), (25088, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf525 = as_strided(buf527, (128, 117, 14, 14), (25088, 196, 14, 1))  # alias
        triton__227.run(buf524, buf475, buf525, 2935296, grid=grid(2935296), stream=stream0)
        buf526 = as_strided(buf527, (128, 11, 14, 14), (25088, 196, 14, 1), 22932)  # alias
        triton__228.run(buf524, buf526, 275968, grid=grid(275968), stream=stream0)
        del buf524
        buf528 = empty_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__229.run(primals_82, buf528, 98304, grid=grid(98304), stream=stream0)
        del primals_82
        buf529 = aten.convolution(buf527, buf528, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf529, (128, 768, 14, 14), (150528, 196, 14, 1))
        buf531 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf532 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf533 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf535 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf534 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf817 = empty_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__230.run(buf529, primals_337, primals_336, buf531, buf532, buf533, buf535, buf534, buf817, 768, 25088, grid=grid(768), stream=stream0)
        del primals_336
        del primals_337
        buf537 = empty_strided((128, 768, 14, 14), (150528, 196, 14, 1), device='cuda', dtype=torch.float16)
        buf816 = empty_strided((128, 768, 14, 14), (150528, 196, 14, 1), device='cuda', dtype=torch.float16)
        triton__231.run(buf529, buf532, buf531, primals_338, primals_339, buf537, buf816, 19267584, grid=grid(19267584), stream=stream0)
        del primals_339
        buf538 = empty_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__232.run(primals_83, buf538, 6912, grid=grid(6912), stream=stream0)
        del primals_83
        buf539 = aten.convolution(buf537, buf538, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 768)
        assert_size_stride(buf539, (128, 768, 7, 7), (37632, 49, 7, 1))
        buf541 = buf532; del buf532  # reuse
        buf542 = buf531; del buf531  # reuse
        buf543 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf545 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf544 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        buf814 = empty_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__233.run(buf539, primals_342, primals_341, buf541, buf542, buf543, buf545, buf544, buf814, 768, 6272, grid=grid(768), stream=stream0)
        del primals_341
        del primals_342
        buf546 = empty_strided((128, 768, 7, 7), (37632, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf548 = empty_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__234.run(buf539, buf542, buf541, primals_343, primals_344, buf546, buf548, 98304, 49, grid=grid(98304), stream=stream0)
        del buf541
        del buf542
        del primals_344
        buf549 = empty_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__235.run(primals_84, buf549, 49152, grid=grid(49152), stream=stream0)
        del primals_84
        buf550 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float16)
        triton__236.run(primals_85, buf550, 64, grid=grid(64), stream=stream0)
        del primals_85
        buf551 = aten.convolution(buf548, buf549, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf551, (128, 64, 1, 1), (64, 1, 1, 1))
        buf552 = buf551; del buf551  # reuse
        triton__237.run(buf552, buf550, 8192, grid=grid(8192), stream=stream0)
        del buf550
        buf554 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf555 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf556 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf558 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf557 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf813 = empty_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__238.run(buf552, primals_155, primals_154, buf554, buf555, buf556, buf558, buf557, buf813, 64, 128, grid=grid(64), stream=stream0)
        del primals_154
        del primals_155
        buf559 = empty_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__239.run(buf552, buf555, buf554, primals_86, primals_87, buf559, 8192, grid=grid(8192), stream=stream0)
        del buf554
        del buf555
        del primals_87
        buf560 = empty_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__235.run(primals_88, buf560, 49152, grid=grid(49152), stream=stream0)
        del primals_88
        buf561 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float16)
        triton__240.run(primals_89, buf561, 768, grid=grid(768), stream=stream0)
        del primals_89
        buf562 = aten.convolution(buf559, buf560, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf562, (128, 768, 1, 1), (768, 1, 1, 1))
        buf563 = buf562; del buf562  # reuse
        triton__241.run(buf563, buf561, 98304, grid=grid(98304), stream=stream0)
        del buf561
        buf564 = empty_strided((128, 768, 7, 7), (37632, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__242.run(buf546, buf563, buf564, 4816896, grid=grid(4816896), stream=stream0)
        buf565 = empty_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__243.run(primals_90, buf565, 107520, grid=grid(107520), stream=stream0)
        del primals_90
        buf566 = aten.convolution(buf564, buf565, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf566, (128, 140, 7, 7), (6860, 49, 7, 1))
        buf568 = empty_strided((1, 140, 1, 1), (140, 1, 140, 140), device='cuda', dtype=torch.float32)
        buf569 = empty_strided((1, 140, 1, 1), (140, 1, 140, 140), device='cuda', dtype=torch.float32)
        buf570 = empty_strided((140, ), (1, ), device='cuda', dtype=torch.float32)
        buf572 = empty_strided((140, ), (1, ), device='cuda', dtype=torch.float32)
        buf571 = empty_strided((140, ), (1, ), device='cuda', dtype=torch.float32)
        buf812 = empty_strided((1, 140, 1, 1), (140, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__244.run(buf566, primals_347, primals_346, buf568, buf569, buf570, buf572, buf571, buf812, 140, 6272, grid=grid(140), stream=stream0)
        del primals_346
        del primals_347
        buf573 = empty_strided((128, 140, 7, 7), (6860, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__245.run(buf566, buf569, buf568, primals_348, primals_349, buf573, 878080, grid=grid(878080), stream=stream0)
        del buf568
        del buf569
        del primals_349
        buf574 = empty_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__246.run(primals_91, buf574, 117600, grid=grid(117600), stream=stream0)
        del primals_91
        buf575 = aten.convolution(buf573, buf574, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf575, (128, 840, 7, 7), (41160, 49, 7, 1))
        buf577 = empty_strided((1, 840, 1, 1), (840, 1, 840, 840), device='cuda', dtype=torch.float32)
        buf578 = empty_strided((1, 840, 1, 1), (840, 1, 840, 840), device='cuda', dtype=torch.float32)
        buf579 = empty_strided((840, ), (1, ), device='cuda', dtype=torch.float32)
        buf581 = empty_strided((840, ), (1, ), device='cuda', dtype=torch.float32)
        buf580 = empty_strided((840, ), (1, ), device='cuda', dtype=torch.float32)
        buf811 = empty_strided((1, 840, 1, 1), (840, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__247.run(buf575, primals_352, primals_351, buf577, buf578, buf579, buf581, buf580, buf811, 840, 6272, grid=grid(840), stream=stream0)
        del primals_351
        del primals_352
        buf583 = empty_strided((128, 840, 7, 7), (41160, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf810 = empty_strided((128, 840, 7, 7), (41160, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__248.run(buf575, buf578, buf577, primals_353, primals_354, buf583, buf810, 5268480, grid=grid(5268480), stream=stream0)
        del primals_354
        buf584 = empty_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__249.run(primals_92, buf584, 7560, grid=grid(7560), stream=stream0)
        del primals_92
        buf585 = aten.convolution(buf583, buf584, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 840)
        assert_size_stride(buf585, (128, 840, 7, 7), (41160, 49, 7, 1))
        buf587 = buf578; del buf578  # reuse
        buf588 = buf577; del buf577  # reuse
        buf589 = empty_strided((840, ), (1, ), device='cuda', dtype=torch.float32)
        buf591 = empty_strided((840, ), (1, ), device='cuda', dtype=torch.float32)
        buf590 = empty_strided((840, ), (1, ), device='cuda', dtype=torch.float32)
        buf808 = empty_strided((1, 840, 1, 1), (840, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__247.run(buf585, primals_357, primals_356, buf587, buf588, buf589, buf591, buf590, buf808, 840, 6272, grid=grid(840), stream=stream0)
        del primals_356
        del primals_357
        buf592 = empty_strided((128, 840, 7, 7), (41160, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf594 = empty_strided((128, 840, 1, 1), (840, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__250.run(buf585, buf588, buf587, primals_358, primals_359, buf592, buf594, 107520, 49, grid=grid(107520), stream=stream0)
        del buf587
        del buf588
        del primals_359
        buf595 = empty_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__251.run(primals_93, buf595, 58800, grid=grid(58800), stream=stream0)
        del primals_93
        buf596 = empty_strided((70, ), (1, ), device='cuda', dtype=torch.float16)
        triton__252.run(primals_94, buf596, 70, grid=grid(70), stream=stream0)
        del primals_94
        buf597 = aten.convolution(buf594, buf595, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf597, (128, 70, 1, 1), (70, 1, 1, 1))
        buf598 = buf597; del buf597  # reuse
        triton__253.run(buf598, buf596, 8960, grid=grid(8960), stream=stream0)
        del buf596
        buf600 = empty_strided((1, 70, 1, 1), (70, 1, 70, 70), device='cuda', dtype=torch.float32)
        buf601 = empty_strided((1, 70, 1, 1), (70, 1, 70, 70), device='cuda', dtype=torch.float32)
        buf602 = empty_strided((70, ), (1, ), device='cuda', dtype=torch.float32)
        buf604 = empty_strided((70, ), (1, ), device='cuda', dtype=torch.float32)
        buf603 = empty_strided((70, ), (1, ), device='cuda', dtype=torch.float32)
        buf807 = empty_strided((1, 70, 1, 1), (70, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__254.run(buf598, primals_158, primals_157, buf600, buf601, buf602, buf604, buf603, buf807, 70, 128, grid=grid(70), stream=stream0)
        del primals_157
        del primals_158
        buf605 = empty_strided((128, 70, 1, 1), (70, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__255.run(buf598, buf601, buf600, primals_95, primals_96, buf605, 8960, grid=grid(8960), stream=stream0)
        del buf600
        del buf601
        del primals_96
        buf606 = empty_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__251.run(primals_97, buf606, 58800, grid=grid(58800), stream=stream0)
        del primals_97
        buf607 = empty_strided((840, ), (1, ), device='cuda', dtype=torch.float16)
        triton__256.run(primals_98, buf607, 840, grid=grid(840), stream=stream0)
        del primals_98
        buf608 = aten.convolution(buf605, buf606, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf608, (128, 840, 1, 1), (840, 1, 1, 1))
        buf609 = buf608; del buf608  # reuse
        triton__257.run(buf609, buf607, 107520, grid=grid(107520), stream=stream0)
        del buf607
        buf610 = empty_strided((128, 840, 7, 7), (41160, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__258.run(buf592, buf609, buf610, 5268480, grid=grid(5268480), stream=stream0)
        buf611 = empty_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__259.run(primals_99, buf611, 126840, grid=grid(126840), stream=stream0)
        del primals_99
        buf612 = aten.convolution(buf610, buf611, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf612, (128, 151, 7, 7), (7399, 49, 7, 1))
        buf614 = empty_strided((1, 151, 1, 1), (151, 1, 151, 151), device='cuda', dtype=torch.float32)
        buf615 = empty_strided((1, 151, 1, 1), (151, 1, 151, 151), device='cuda', dtype=torch.float32)
        buf616 = empty_strided((151, ), (1, ), device='cuda', dtype=torch.float32)
        buf618 = empty_strided((151, ), (1, ), device='cuda', dtype=torch.float32)
        buf617 = empty_strided((151, ), (1, ), device='cuda', dtype=torch.float32)
        buf806 = empty_strided((1, 151, 1, 1), (151, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__260.run(buf612, primals_362, primals_361, buf614, buf615, buf616, buf618, buf617, buf806, 151, 6272, grid=grid(151), stream=stream0)
        del primals_361
        del primals_362
        buf619 = empty_strided((128, 151, 7, 7), (7399, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__261.run(buf612, buf615, buf614, primals_363, primals_364, buf619, 947072, grid=grid(947072), stream=stream0)
        del buf614
        del buf615
        del primals_364
        buf622 = empty_strided((128, 151, 7, 7), (7399, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf620 = as_strided(buf622, (128, 140, 7, 7), (7399, 49, 7, 1))  # alias
        triton__262.run(buf619, buf573, buf620, 878080, grid=grid(878080), stream=stream0)
        buf621 = as_strided(buf622, (128, 11, 7, 7), (7399, 49, 7, 1), 6860)  # alias
        triton__263.run(buf619, buf621, 68992, grid=grid(68992), stream=stream0)
        del buf619
        buf623 = empty_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__264.run(primals_100, buf623, 136806, grid=grid(136806), stream=stream0)
        del primals_100
        buf624 = aten.convolution(buf622, buf623, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf624, (128, 906, 7, 7), (44394, 49, 7, 1))
        buf626 = empty_strided((1, 906, 1, 1), (906, 1, 906, 906), device='cuda', dtype=torch.float32)
        buf627 = empty_strided((1, 906, 1, 1), (906, 1, 906, 906), device='cuda', dtype=torch.float32)
        buf628 = empty_strided((906, ), (1, ), device='cuda', dtype=torch.float32)
        buf630 = empty_strided((906, ), (1, ), device='cuda', dtype=torch.float32)
        buf629 = empty_strided((906, ), (1, ), device='cuda', dtype=torch.float32)
        buf805 = empty_strided((1, 906, 1, 1), (906, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__265.run(buf624, primals_367, primals_366, buf626, buf627, buf628, buf630, buf629, buf805, 906, 6272, grid=grid(906), stream=stream0)
        del primals_366
        del primals_367
        buf632 = empty_strided((128, 906, 7, 7), (44394, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf804 = empty_strided((128, 906, 7, 7), (44394, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__266.run(buf624, buf627, buf626, primals_368, primals_369, buf632, buf804, 5682432, grid=grid(5682432), stream=stream0)
        del primals_369
        buf633 = empty_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__267.run(primals_101, buf633, 8154, grid=grid(8154), stream=stream0)
        del primals_101
        buf634 = aten.convolution(buf632, buf633, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 906)
        assert_size_stride(buf634, (128, 906, 7, 7), (44394, 49, 7, 1))
        buf636 = buf627; del buf627  # reuse
        buf637 = buf626; del buf626  # reuse
        buf638 = empty_strided((906, ), (1, ), device='cuda', dtype=torch.float32)
        buf640 = empty_strided((906, ), (1, ), device='cuda', dtype=torch.float32)
        buf639 = empty_strided((906, ), (1, ), device='cuda', dtype=torch.float32)
        buf802 = empty_strided((1, 906, 1, 1), (906, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__265.run(buf634, primals_372, primals_371, buf636, buf637, buf638, buf640, buf639, buf802, 906, 6272, grid=grid(906), stream=stream0)
        del primals_371
        del primals_372
        buf641 = empty_strided((128, 906, 7, 7), (44394, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf643 = empty_strided((128, 906, 1, 1), (906, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__268.run(buf634, buf637, buf636, primals_373, primals_374, buf641, buf643, 115968, 49, grid=grid(115968), stream=stream0)
        del buf636
        del buf637
        del primals_374
        buf644 = empty_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__269.run(primals_102, buf644, 67950, grid=grid(67950), stream=stream0)
        del primals_102
        buf645 = empty_strided((75, ), (1, ), device='cuda', dtype=torch.float16)
        triton__270.run(primals_103, buf645, 75, grid=grid(75), stream=stream0)
        del primals_103
        buf646 = aten.convolution(buf643, buf644, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf646, (128, 75, 1, 1), (75, 1, 1, 1))
        buf647 = buf646; del buf646  # reuse
        triton__271.run(buf647, buf645, 9600, grid=grid(9600), stream=stream0)
        del buf645
        buf649 = empty_strided((1, 75, 1, 1), (75, 1, 75, 75), device='cuda', dtype=torch.float32)
        buf650 = empty_strided((1, 75, 1, 1), (75, 1, 75, 75), device='cuda', dtype=torch.float32)
        buf651 = empty_strided((75, ), (1, ), device='cuda', dtype=torch.float32)
        buf653 = empty_strided((75, ), (1, ), device='cuda', dtype=torch.float32)
        buf652 = empty_strided((75, ), (1, ), device='cuda', dtype=torch.float32)
        buf801 = empty_strided((1, 75, 1, 1), (75, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__272.run(buf647, primals_161, primals_160, buf649, buf650, buf651, buf653, buf652, buf801, 75, 128, grid=grid(75), stream=stream0)
        del primals_160
        del primals_161
        buf654 = empty_strided((128, 75, 1, 1), (75, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__273.run(buf647, buf650, buf649, primals_104, primals_105, buf654, 9600, grid=grid(9600), stream=stream0)
        del buf649
        del buf650
        del primals_105
        buf655 = empty_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__269.run(primals_106, buf655, 67950, grid=grid(67950), stream=stream0)
        del primals_106
        buf656 = empty_strided((906, ), (1, ), device='cuda', dtype=torch.float16)
        triton__274.run(primals_107, buf656, 906, grid=grid(906), stream=stream0)
        del primals_107
        buf657 = aten.convolution(buf654, buf655, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf657, (128, 906, 1, 1), (906, 1, 1, 1))
        buf658 = buf657; del buf657  # reuse
        triton__275.run(buf658, buf656, 115968, grid=grid(115968), stream=stream0)
        del buf656
        buf659 = empty_strided((128, 906, 7, 7), (44394, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__276.run(buf641, buf658, buf659, 5682432, grid=grid(5682432), stream=stream0)
        buf660 = empty_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__277.run(primals_108, buf660, 146772, grid=grid(146772), stream=stream0)
        del primals_108
        buf661 = aten.convolution(buf659, buf660, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf661, (128, 162, 7, 7), (7938, 49, 7, 1))
        buf663 = buf97; del buf97  # reuse
        buf664 = buf95; del buf95  # reuse
        buf665 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        buf667 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        buf666 = empty_strided((162, ), (1, ), device='cuda', dtype=torch.float32)
        buf800 = empty_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__278.run(buf661, primals_377, primals_376, buf663, buf664, buf665, buf667, buf666, buf800, 162, 6272, grid=grid(162), stream=stream0)
        del primals_376
        del primals_377
        buf668 = empty_strided((128, 162, 7, 7), (7938, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__279.run(buf661, buf664, buf663, primals_378, primals_379, buf668, 1016064, grid=grid(1016064), stream=stream0)
        del buf663
        del buf664
        del primals_379
        buf671 = empty_strided((128, 162, 7, 7), (7938, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf669 = as_strided(buf671, (128, 151, 7, 7), (7938, 49, 7, 1))  # alias
        triton__280.run(buf668, buf622, buf669, 947072, grid=grid(947072), stream=stream0)
        buf670 = as_strided(buf671, (128, 11, 7, 7), (7938, 49, 7, 1), 7399)  # alias
        triton__281.run(buf668, buf670, 68992, grid=grid(68992), stream=stream0)
        del buf668
        buf672 = empty_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__282.run(primals_109, buf672, 157464, grid=grid(157464), stream=stream0)
        del primals_109
        buf673 = aten.convolution(buf671, buf672, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf673, (128, 972, 7, 7), (47628, 49, 7, 1))
        buf675 = empty_strided((1, 972, 1, 1), (972, 1, 972, 972), device='cuda', dtype=torch.float32)
        buf676 = empty_strided((1, 972, 1, 1), (972, 1, 972, 972), device='cuda', dtype=torch.float32)
        buf677 = empty_strided((972, ), (1, ), device='cuda', dtype=torch.float32)
        buf679 = empty_strided((972, ), (1, ), device='cuda', dtype=torch.float32)
        buf678 = empty_strided((972, ), (1, ), device='cuda', dtype=torch.float32)
        buf799 = empty_strided((1, 972, 1, 1), (972, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__283.run(buf673, primals_382, primals_381, buf675, buf676, buf677, buf679, buf678, buf799, 972, 6272, grid=grid(972), stream=stream0)
        del primals_381
        del primals_382
        buf681 = empty_strided((128, 972, 7, 7), (47628, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf798 = empty_strided((128, 972, 7, 7), (47628, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__284.run(buf673, buf676, buf675, primals_383, primals_384, buf681, buf798, 6096384, grid=grid(6096384), stream=stream0)
        del primals_384
        buf682 = empty_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__285.run(primals_110, buf682, 8748, grid=grid(8748), stream=stream0)
        del primals_110
        buf683 = aten.convolution(buf681, buf682, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 972)
        assert_size_stride(buf683, (128, 972, 7, 7), (47628, 49, 7, 1))
        buf685 = buf676; del buf676  # reuse
        buf686 = buf675; del buf675  # reuse
        buf687 = empty_strided((972, ), (1, ), device='cuda', dtype=torch.float32)
        buf689 = empty_strided((972, ), (1, ), device='cuda', dtype=torch.float32)
        buf688 = empty_strided((972, ), (1, ), device='cuda', dtype=torch.float32)
        buf796 = empty_strided((1, 972, 1, 1), (972, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__283.run(buf683, primals_387, primals_386, buf685, buf686, buf687, buf689, buf688, buf796, 972, 6272, grid=grid(972), stream=stream0)
        del primals_386
        del primals_387
        buf690 = empty_strided((128, 972, 7, 7), (47628, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf692 = empty_strided((128, 972, 1, 1), (972, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__286.run(buf683, buf686, buf685, primals_388, primals_389, buf690, buf692, 124416, 49, grid=grid(124416), stream=stream0)
        del buf685
        del buf686
        del primals_389
        buf693 = empty_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__287.run(primals_111, buf693, 78732, grid=grid(78732), stream=stream0)
        del primals_111
        buf694 = empty_strided((81, ), (1, ), device='cuda', dtype=torch.float16)
        triton__288.run(primals_112, buf694, 81, grid=grid(81), stream=stream0)
        del primals_112
        buf695 = aten.convolution(buf692, buf693, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf695, (128, 81, 1, 1), (81, 1, 1, 1))
        buf696 = buf695; del buf695  # reuse
        triton__289.run(buf696, buf694, 10368, grid=grid(10368), stream=stream0)
        del buf694
        buf698 = empty_strided((1, 81, 1, 1), (81, 1, 81, 81), device='cuda', dtype=torch.float32)
        buf699 = empty_strided((1, 81, 1, 1), (81, 1, 81, 81), device='cuda', dtype=torch.float32)
        buf700 = empty_strided((81, ), (1, ), device='cuda', dtype=torch.float32)
        buf702 = empty_strided((81, ), (1, ), device='cuda', dtype=torch.float32)
        buf701 = empty_strided((81, ), (1, ), device='cuda', dtype=torch.float32)
        buf795 = empty_strided((1, 81, 1, 1), (81, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__290.run(buf696, primals_164, primals_163, buf698, buf699, buf700, buf702, buf701, buf795, 81, 128, grid=grid(81), stream=stream0)
        del primals_163
        del primals_164
        buf703 = empty_strided((128, 81, 1, 1), (81, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__291.run(buf696, buf699, buf698, primals_113, primals_114, buf703, 10368, grid=grid(10368), stream=stream0)
        del buf698
        del buf699
        del primals_114
        buf704 = empty_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__287.run(primals_115, buf704, 78732, grid=grid(78732), stream=stream0)
        del primals_115
        buf705 = empty_strided((972, ), (1, ), device='cuda', dtype=torch.float16)
        triton__292.run(primals_116, buf705, 972, grid=grid(972), stream=stream0)
        del primals_116
        buf706 = aten.convolution(buf703, buf704, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf706, (128, 972, 1, 1), (972, 1, 1, 1))
        buf707 = buf706; del buf706  # reuse
        triton__293.run(buf707, buf705, 124416, grid=grid(124416), stream=stream0)
        del buf705
        buf708 = empty_strided((128, 972, 7, 7), (47628, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__294.run(buf690, buf707, buf708, 6096384, grid=grid(6096384), stream=stream0)
        buf709 = empty_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__295.run(primals_117, buf709, 169128, grid=grid(169128), stream=stream0)
        del primals_117
        buf710 = aten.convolution(buf708, buf709, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf710, (128, 174, 7, 7), (8526, 49, 7, 1))
        buf712 = empty_strided((1, 174, 1, 1), (174, 1, 174, 174), device='cuda', dtype=torch.float32)
        buf713 = empty_strided((1, 174, 1, 1), (174, 1, 174, 174), device='cuda', dtype=torch.float32)
        buf714 = empty_strided((174, ), (1, ), device='cuda', dtype=torch.float32)
        buf716 = empty_strided((174, ), (1, ), device='cuda', dtype=torch.float32)
        buf715 = empty_strided((174, ), (1, ), device='cuda', dtype=torch.float32)
        buf794 = empty_strided((1, 174, 1, 1), (174, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__296.run(buf710, primals_392, primals_391, buf712, buf713, buf714, buf716, buf715, buf794, 174, 6272, grid=grid(174), stream=stream0)
        del primals_391
        del primals_392
        buf717 = empty_strided((128, 174, 7, 7), (8526, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__297.run(buf710, buf713, buf712, primals_393, primals_394, buf717, 1091328, grid=grid(1091328), stream=stream0)
        del buf712
        del buf713
        del primals_394
        buf720 = empty_strided((128, 174, 7, 7), (8526, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf718 = as_strided(buf720, (128, 162, 7, 7), (8526, 49, 7, 1))  # alias
        triton__298.run(buf717, buf671, buf718, 1016064, grid=grid(1016064), stream=stream0)
        buf719 = as_strided(buf720, (128, 12, 7, 7), (8526, 49, 7, 1), 7938)  # alias
        triton__299.run(buf717, buf719, 75264, grid=grid(75264), stream=stream0)
        del buf717
        buf721 = empty_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__300.run(primals_118, buf721, 181656, grid=grid(181656), stream=stream0)
        del primals_118
        buf722 = aten.convolution(buf720, buf721, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf722, (128, 1044, 7, 7), (51156, 49, 7, 1))
        buf724 = empty_strided((1, 1044, 1, 1), (1044, 1, 1044, 1044), device='cuda', dtype=torch.float32)
        buf725 = empty_strided((1, 1044, 1, 1), (1044, 1, 1044, 1044), device='cuda', dtype=torch.float32)
        buf726 = empty_strided((1044, ), (1, ), device='cuda', dtype=torch.float32)
        buf728 = empty_strided((1044, ), (1, ), device='cuda', dtype=torch.float32)
        buf727 = empty_strided((1044, ), (1, ), device='cuda', dtype=torch.float32)
        buf793 = empty_strided((1, 1044, 1, 1), (1044, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__301.run(buf722, primals_397, primals_396, buf724, buf725, buf726, buf728, buf727, buf793, 1044, 6272, grid=grid(1044), stream=stream0)
        del primals_396
        del primals_397
        buf730 = empty_strided((128, 1044, 7, 7), (51156, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf792 = empty_strided((128, 1044, 7, 7), (51156, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__302.run(buf722, buf725, buf724, primals_398, primals_399, buf730, buf792, 6547968, grid=grid(6547968), stream=stream0)
        del primals_399
        buf731 = empty_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cuda', dtype=torch.float16)
        triton__303.run(primals_119, buf731, 9396, grid=grid(9396), stream=stream0)
        del primals_119
        buf732 = aten.convolution(buf730, buf731, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1044)
        assert_size_stride(buf732, (128, 1044, 7, 7), (51156, 49, 7, 1))
        buf734 = buf725; del buf725  # reuse
        buf735 = buf724; del buf724  # reuse
        buf736 = empty_strided((1044, ), (1, ), device='cuda', dtype=torch.float32)
        buf738 = empty_strided((1044, ), (1, ), device='cuda', dtype=torch.float32)
        buf737 = empty_strided((1044, ), (1, ), device='cuda', dtype=torch.float32)
        buf790 = empty_strided((1, 1044, 1, 1), (1044, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__301.run(buf732, primals_402, primals_401, buf734, buf735, buf736, buf738, buf737, buf790, 1044, 6272, grid=grid(1044), stream=stream0)
        del primals_401
        del primals_402
        buf739 = empty_strided((128, 1044, 7, 7), (51156, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf741 = empty_strided((128, 1044, 1, 1), (1044, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__304.run(buf732, buf735, buf734, primals_403, primals_404, buf739, buf741, 133632, 49, grid=grid(133632), stream=stream0)
        del buf734
        del buf735
        del primals_404
        buf742 = empty_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__305.run(primals_120, buf742, 90828, grid=grid(90828), stream=stream0)
        del primals_120
        buf743 = empty_strided((87, ), (1, ), device='cuda', dtype=torch.float16)
        triton__306.run(primals_121, buf743, 87, grid=grid(87), stream=stream0)
        del primals_121
        buf744 = aten.convolution(buf741, buf742, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf744, (128, 87, 1, 1), (87, 1, 1, 1))
        buf745 = buf744; del buf744  # reuse
        triton__307.run(buf745, buf743, 11136, grid=grid(11136), stream=stream0)
        del buf743
        buf747 = empty_strided((1, 87, 1, 1), (87, 1, 87, 87), device='cuda', dtype=torch.float32)
        buf748 = empty_strided((1, 87, 1, 1), (87, 1, 87, 87), device='cuda', dtype=torch.float32)
        buf749 = empty_strided((87, ), (1, ), device='cuda', dtype=torch.float32)
        buf751 = empty_strided((87, ), (1, ), device='cuda', dtype=torch.float32)
        buf750 = empty_strided((87, ), (1, ), device='cuda', dtype=torch.float32)
        buf789 = empty_strided((1, 87, 1, 1), (87, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__308.run(buf745, primals_167, primals_166, buf747, buf748, buf749, buf751, buf750, buf789, 87, 128, grid=grid(87), stream=stream0)
        del primals_166
        del primals_167
        buf752 = empty_strided((128, 87, 1, 1), (87, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__309.run(buf745, buf748, buf747, primals_122, primals_123, buf752, 11136, grid=grid(11136), stream=stream0)
        del buf747
        del buf748
        del primals_123
        buf753 = empty_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__305.run(primals_124, buf753, 90828, grid=grid(90828), stream=stream0)
        del primals_124
        buf754 = empty_strided((1044, ), (1, ), device='cuda', dtype=torch.float16)
        triton__310.run(primals_125, buf754, 1044, grid=grid(1044), stream=stream0)
        del primals_125
        buf755 = aten.convolution(buf752, buf753, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf755, (128, 1044, 1, 1), (1044, 1, 1, 1))
        buf756 = buf755; del buf755  # reuse
        triton__311.run(buf756, buf754, 133632, grid=grid(133632), stream=stream0)
        del buf754
        buf757 = empty_strided((128, 1044, 7, 7), (51156, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__312.run(buf739, buf756, buf757, 6547968, grid=grid(6547968), stream=stream0)
        buf758 = empty_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__313.run(primals_126, buf758, 193140, grid=grid(193140), stream=stream0)
        del primals_126
        buf759 = aten.convolution(buf757, buf758, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf759, (128, 185, 7, 7), (9065, 49, 7, 1))
        buf761 = empty_strided((1, 185, 1, 1), (185, 1, 185, 185), device='cuda', dtype=torch.float32)
        buf762 = empty_strided((1, 185, 1, 1), (185, 1, 185, 185), device='cuda', dtype=torch.float32)
        buf763 = empty_strided((185, ), (1, ), device='cuda', dtype=torch.float32)
        buf765 = empty_strided((185, ), (1, ), device='cuda', dtype=torch.float32)
        buf764 = empty_strided((185, ), (1, ), device='cuda', dtype=torch.float32)
        buf788 = empty_strided((1, 185, 1, 1), (185, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__314.run(buf759, primals_407, primals_406, buf761, buf762, buf763, buf765, buf764, buf788, 185, 6272, grid=grid(185), stream=stream0)
        del primals_406
        del primals_407
        buf766 = empty_strided((128, 185, 7, 7), (9065, 49, 7, 1), device='cuda', dtype=torch.float16)
        triton__315.run(buf759, buf762, buf761, primals_408, primals_409, buf766, 1160320, grid=grid(1160320), stream=stream0)
        del buf761
        del buf762
        del primals_409
        buf769 = empty_strided((128, 185, 7, 7), (9065, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf767 = as_strided(buf769, (128, 174, 7, 7), (9065, 49, 7, 1))  # alias
        triton__316.run(buf766, buf720, buf767, 1091328, grid=grid(1091328), stream=stream0)
        buf768 = as_strided(buf769, (128, 11, 7, 7), (9065, 49, 7, 1), 8526)  # alias
        triton__317.run(buf766, buf768, 68992, grid=grid(68992), stream=stream0)
        del buf766
        buf770 = empty_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cuda', dtype=torch.float16)
        triton__318.run(primals_127, buf770, 236800, grid=grid(236800), stream=stream0)
        del primals_127
        buf771 = aten.convolution(buf769, buf770, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf771, (128, 1280, 7, 7), (62720, 49, 7, 1))
        buf773 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf774 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf775 = empty_strided((1280, ), (1, ), device='cuda', dtype=torch.float32)
        buf777 = empty_strided((1280, ), (1, ), device='cuda', dtype=torch.float32)
        buf776 = empty_strided((1280, ), (1, ), device='cuda', dtype=torch.float32)
        buf787 = empty_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton__319.run(buf771, primals_412, primals_411, buf773, buf774, buf775, buf777, buf776, buf787, 1280, 6272, grid=grid(1280), stream=stream0)
        del primals_411
        del primals_412
        buf786 = empty_strided((128, 1280, 7, 7), (62720, 49, 7, 1), device='cuda', dtype=torch.float16)
        buf780 = empty_strided((128, 1280), (1280, 1), device='cuda', dtype=torch.float16)
        triton__320.run(buf771, buf774, buf773, primals_413, primals_414, buf786, buf780, 163840, 49, grid=grid(163840), stream=stream0)
        del buf773
        del buf774
        del primals_414
        buf781 = empty_strided((1000, 1280), (1280, 1), device='cuda', dtype=torch.float16)
        buf784 = empty_strided((1000, 1280), (1280, 1), device='cuda', dtype=torch.float16)
        triton__321.run(primals_128, buf781, buf784, 1280000, grid=grid(1280000), stream=stream0)
        del primals_128
        buf782 = empty_strided((1000, ), (1, ), device='cuda', dtype=torch.float16)
        triton__322.run(primals_129, buf782, 1000, grid=grid(1000), stream=stream0)
        del primals_129
        buf783 = empty_strided((128, 1000), (1000, 1), device='cuda', dtype=torch.float16)
        extern_kernels.addmm(buf782, buf780, as_strided(buf781, (1280, 1000), (1, 1280)), alpha=1, beta=1, out=buf783)
        del buf781
        del buf782
        buf884 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_132, buf884, 1, grid=grid(1), stream=stream0)
        del primals_132
        buf885 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_135, buf885, 1, grid=grid(1), stream=stream0)
        del primals_135
        buf886 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_138, buf886, 1, grid=grid(1), stream=stream0)
        del primals_138
        buf887 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_141, buf887, 1, grid=grid(1), stream=stream0)
        del primals_141
        buf888 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_144, buf888, 1, grid=grid(1), stream=stream0)
        del primals_144
        buf889 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_147, buf889, 1, grid=grid(1), stream=stream0)
        del primals_147
        buf890 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_150, buf890, 1, grid=grid(1), stream=stream0)
        del primals_150
        buf891 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_153, buf891, 1, grid=grid(1), stream=stream0)
        del primals_153
        buf892 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_156, buf892, 1, grid=grid(1), stream=stream0)
        del primals_156
        buf893 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_159, buf893, 1, grid=grid(1), stream=stream0)
        del primals_159
        buf894 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_162, buf894, 1, grid=grid(1), stream=stream0)
        del primals_162
        buf895 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_165, buf895, 1, grid=grid(1), stream=stream0)
        del primals_165
        buf896 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_168, buf896, 1, grid=grid(1), stream=stream0)
        del primals_168
        buf897 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_170, buf897, 1, grid=grid(1), stream=stream0)
        del primals_170
        buf898 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_175, buf898, 1, grid=grid(1), stream=stream0)
        del primals_175
        buf899 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_180, buf899, 1, grid=grid(1), stream=stream0)
        del primals_180
        buf900 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_185, buf900, 1, grid=grid(1), stream=stream0)
        del primals_185
        buf901 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_190, buf901, 1, grid=grid(1), stream=stream0)
        del primals_190
        buf902 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_195, buf902, 1, grid=grid(1), stream=stream0)
        del primals_195
        buf903 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_200, buf903, 1, grid=grid(1), stream=stream0)
        del primals_200
        buf904 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_205, buf904, 1, grid=grid(1), stream=stream0)
        del primals_205
        buf905 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_210, buf905, 1, grid=grid(1), stream=stream0)
        del primals_210
        buf906 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_215, buf906, 1, grid=grid(1), stream=stream0)
        del primals_215
        buf907 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_220, buf907, 1, grid=grid(1), stream=stream0)
        del primals_220
        buf908 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_225, buf908, 1, grid=grid(1), stream=stream0)
        del primals_225
        buf909 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_230, buf909, 1, grid=grid(1), stream=stream0)
        del primals_230
        buf910 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_235, buf910, 1, grid=grid(1), stream=stream0)
        del primals_235
        buf911 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_240, buf911, 1, grid=grid(1), stream=stream0)
        del primals_240
        buf912 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_245, buf912, 1, grid=grid(1), stream=stream0)
        del primals_245
        buf913 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_250, buf913, 1, grid=grid(1), stream=stream0)
        del primals_250
        buf914 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_255, buf914, 1, grid=grid(1), stream=stream0)
        del primals_255
        buf915 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_260, buf915, 1, grid=grid(1), stream=stream0)
        del primals_260
        buf916 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_265, buf916, 1, grid=grid(1), stream=stream0)
        del primals_265
        buf917 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_270, buf917, 1, grid=grid(1), stream=stream0)
        del primals_270
        buf918 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_275, buf918, 1, grid=grid(1), stream=stream0)
        del primals_275
        buf919 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_280, buf919, 1, grid=grid(1), stream=stream0)
        del primals_280
        buf920 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_285, buf920, 1, grid=grid(1), stream=stream0)
        del primals_285
        buf921 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_290, buf921, 1, grid=grid(1), stream=stream0)
        del primals_290
        buf922 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_295, buf922, 1, grid=grid(1), stream=stream0)
        del primals_295
        buf923 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_300, buf923, 1, grid=grid(1), stream=stream0)
        del primals_300
        buf924 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_305, buf924, 1, grid=grid(1), stream=stream0)
        del primals_305
        buf925 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_310, buf925, 1, grid=grid(1), stream=stream0)
        del primals_310
        buf926 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_315, buf926, 1, grid=grid(1), stream=stream0)
        del primals_315
        buf927 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_320, buf927, 1, grid=grid(1), stream=stream0)
        del primals_320
        buf928 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_325, buf928, 1, grid=grid(1), stream=stream0)
        del primals_325
        buf929 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_330, buf929, 1, grid=grid(1), stream=stream0)
        del primals_330
        buf930 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_335, buf930, 1, grid=grid(1), stream=stream0)
        del primals_335
        buf931 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_340, buf931, 1, grid=grid(1), stream=stream0)
        del primals_340
        buf932 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_345, buf932, 1, grid=grid(1), stream=stream0)
        del primals_345
        buf933 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_350, buf933, 1, grid=grid(1), stream=stream0)
        del primals_350
        buf934 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_355, buf934, 1, grid=grid(1), stream=stream0)
        del primals_355
        buf935 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_360, buf935, 1, grid=grid(1), stream=stream0)
        del primals_360
        buf936 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_365, buf936, 1, grid=grid(1), stream=stream0)
        del primals_365
        buf937 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_370, buf937, 1, grid=grid(1), stream=stream0)
        del primals_370
        buf938 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_375, buf938, 1, grid=grid(1), stream=stream0)
        del primals_375
        buf939 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_380, buf939, 1, grid=grid(1), stream=stream0)
        del primals_380
        buf940 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_385, buf940, 1, grid=grid(1), stream=stream0)
        del primals_385
        buf941 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_390, buf941, 1, grid=grid(1), stream=stream0)
        del primals_390
        buf942 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_395, buf942, 1, grid=grid(1), stream=stream0)
        del primals_395
        buf943 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_400, buf943, 1, grid=grid(1), stream=stream0)
        del primals_400
        buf944 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_405, buf944, 1, grid=grid(1), stream=stream0)
        del primals_405
        buf945 = empty_strided((), (), device='cuda', dtype=torch.int64)
        triton__323.run(primals_410, buf945, 1, grid=grid(1), stream=stream0)
        del primals_410
        return (buf147, buf148, buf884, buf196, buf197, buf885, buf248, buf249, buf886, buf297, buf298, buf887, buf349, buf350, buf888, buf401, buf402, buf889, buf453, buf454, buf890, buf505, buf506, buf891, buf557, buf558, buf892, buf603, buf604, buf893, buf652, buf653, buf894, buf701, buf702, buf895, buf750, buf751, buf896, buf10, buf11, buf23, buf24, buf36, buf37, buf48, buf49, buf61, buf62, buf74, buf75, buf86, buf87, buf99, buf100, buf112, buf113, buf124, buf125, buf134, buf135, buf164, buf165, buf173, buf174, buf183, buf184, buf213, buf214, buf225, buf226, buf235, buf236, buf265, buf266, buf274, buf275, buf284, buf285, buf314, buf315, buf326, buf327, buf336, buf337, buf366, buf367, buf378, buf379, buf388, buf389, buf418, buf419, buf430, buf431, buf440, buf441, buf470, buf471, buf482, buf483, buf492, buf493, buf522, buf523, buf534, buf535, buf544, buf545, buf571, buf572, buf580, buf581, buf590, buf591, buf617, buf618, buf629, buf630, buf639, buf640, buf666, buf667, buf678, buf679, buf688, buf689, buf715, buf716, buf727, buf728, buf737, buf738, buf764, buf765, buf776, buf777, buf783, buf897, buf898, buf899, buf900, buf901, buf902, buf903, buf904, buf905, buf906, buf907, buf908, buf909, buf910, buf911, buf912, buf913, buf914, buf915, buf916, buf917, buf918, buf919, buf920, buf921, buf922, buf923, buf924, buf925, buf926, buf927, buf928, buf929, buf930, buf931, buf932, buf933, buf934, buf935, buf936, buf937, buf938, buf939, buf940, buf941, buf942, buf943, buf944, buf945, primals_14, primals_23, primals_32, primals_41, primals_50, primals_59, primals_68, primals_77, primals_86, primals_95, primals_104, primals_113, primals_122, primals_173, primals_178, primals_183, primals_188, primals_193, primals_198, primals_203, primals_208, primals_213, primals_218, primals_223, primals_228, primals_233, primals_238, primals_243, primals_248, primals_253, primals_258, primals_263, primals_268, primals_273, primals_278, primals_283, primals_288, primals_293, primals_298, primals_303, primals_308, primals_313, primals_318, primals_323, primals_328, primals_333, primals_338, primals_343, primals_348, primals_353, primals_358, primals_363, primals_368, primals_373, primals_378, primals_383, primals_388, primals_393, primals_398, primals_403, primals_408, primals_413, buf0, buf1, buf2, buf9, buf13, buf14, buf15, buf22, buf26, buf27, buf28, buf35, buf38, buf39, buf40, buf47, buf51, buf52, buf53, buf60, buf64, buf65, buf66, buf73, buf76, buf77, buf78, buf85, buf89, buf90, buf91, buf98, buf102, buf103, buf104, buf111, buf117, buf118, buf119, buf123, buf127, buf128, buf129, buf133, buf136, buf138, buf139, buf142, buf146, buf149, buf150, buf153, buf154, buf155, buf156, buf163, buf166, buf167, buf168, buf172, buf176, buf177, buf178, buf182, buf185, buf187, buf188, buf191, buf195, buf198, buf199, buf202, buf203, buf204, buf205, buf212, buf218, buf219, buf220, buf224, buf228, buf229, buf230, buf234, buf237, buf239, buf240, buf243, buf247, buf250, buf251, buf254, buf255, buf256, buf257, buf264, buf267, buf268, buf269, buf273, buf277, buf278, buf279, buf283, buf286, buf288, buf289, buf292, buf296, buf299, buf300, buf303, buf304, buf305, buf306, buf313, buf319, buf320, buf321, buf325, buf329, buf330, buf331, buf335, buf338, buf340, buf341, buf344, buf348, buf351, buf352, buf355, buf356, buf357, buf358, buf365, buf371, buf372, buf373, buf377, buf381, buf382, buf383, buf387, buf390, buf392, buf393, buf396, buf400, buf403, buf404, buf407, buf408, buf409, buf410, buf417, buf423, buf424, buf425, buf429, buf433, buf434, buf435, buf439, buf442, buf444, buf445, buf448, buf452, buf455, buf456, buf459, buf460, buf461, buf462, buf469, buf475, buf476, buf477, buf481, buf485, buf486, buf487, buf491, buf494, buf496, buf497, buf500, buf504, buf507, buf508, buf511, buf512, buf513, buf514, buf521, buf527, buf528, buf529, buf533, buf537, buf538, buf539, buf543, buf546, buf548, buf549, buf552, buf556, buf559, buf560, buf563, buf564, buf565, buf566, buf570, buf573, buf574, buf575, buf579, buf583, buf584, buf585, buf589, buf592, buf594, buf595, buf598, buf602, buf605, buf606, buf609, buf610, buf611, buf612, buf616, buf622, buf623, buf624, buf628, buf632, buf633, buf634, buf638, buf641, buf643, buf644, buf647, buf651, buf654, buf655, buf658, buf659, buf660, buf661, buf665, buf671, buf672, buf673, buf677, buf681, buf682, buf683, buf687, buf690, buf692, buf693, buf696, buf700, buf703, buf704, buf707, buf708, buf709, buf710, buf714, buf720, buf721, buf722, buf726, buf730, buf731, buf732, buf736, buf739, buf741, buf742, buf745, buf749, buf752, buf753, buf756, buf757, buf758, buf759, buf763, buf769, buf770, buf771, buf775, buf780, buf784, buf786, buf787, buf788, buf789, buf790, buf792, buf793, buf794, buf795, buf796, buf798, buf799, buf800, buf801, buf802, buf804, buf805, buf806, buf807, buf808, buf810, buf811, buf812, buf813, buf814, buf816, buf817, buf818, buf819, buf820, buf822, buf823, buf824, buf825, buf826, buf828, buf829, buf830, buf831, buf832, buf834, buf835, buf836, buf837, buf838, buf840, buf841, buf842, buf843, buf844, buf846, buf847, buf848, buf849, buf850, buf852, buf853, buf854, buf855, buf856, buf858, buf859, buf860, buf861, buf862, buf864, buf865, buf866, buf867, buf868, buf870, buf871, buf872, buf873, buf874, buf876, buf877, buf878, buf879, buf880, buf882, buf883, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((27, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((162, 27, 1, 1), (27, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((162, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((38, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((228, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((228, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((19, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_133 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_136 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_139 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_142 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_145 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_148 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_151 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_157 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_160 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_163 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_166 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_169 = rand_strided((128, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_176 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_181 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_191 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_196 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_221 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_236 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_251 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_266 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_281 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_326 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_341 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_346 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_356 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_361 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_366 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_371 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_376 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_381 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_386 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_391 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_396 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_401 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_406 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_411 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414]))
