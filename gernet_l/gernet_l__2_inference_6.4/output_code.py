
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

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__12(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1228800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__22(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


triton__23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__23(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = -0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        print('triton__0', 'in_ptr0', 'arg0_1', (arg0_1.sum()/arg0_1.nelement()).item(), arg0_1.amax().item(), arg0_1.amin().item())
        print('triton__0', 'in_ptr1', 'arg173_1', (arg173_1.sum()/arg173_1.nelement()).item(), arg173_1.amax().item(), arg173_1.amin().item())
        stream0 = get_cuda_stream(0)
        triton__0.run(arg0_1, arg173_1, arg0_1, 864, grid=grid(864), stream=stream0)
        print('triton__0', 'out_ptr0', 'arg0_1', (arg0_1.sum()/arg0_1.nelement()).item(), arg0_1.amax().item(), arg0_1.amin().item())
        del arg0_1
        del arg173_1
        print('triton__1', 'in_ptr0', 'arg1_1', (arg1_1.sum()/arg1_1.nelement()).item(), arg1_1.amax().item(), arg1_1.amin().item())
        print('triton__1', 'in_ptr1', 'arg174_1', (arg174_1.sum()/arg174_1.nelement()).item(), arg174_1.amax().item(), arg174_1.amin().item())
        triton__1.run(arg1_1, arg174_1, arg1_1, 32, grid=grid(32), stream=stream0)
        print('triton__1', 'out_ptr0', 'arg1_1', (arg1_1.sum()/arg1_1.nelement()).item(), arg1_1.amax().item(), arg1_1.amin().item())
        del arg174_1
        del arg1_1
        print('triton__1', 'in_ptr0', 'arg2_1', (arg2_1.sum()/arg2_1.nelement()).item(), arg2_1.amax().item(), arg2_1.amin().item())
        print('triton__1', 'in_ptr1', 'arg175_1', (arg175_1.sum()/arg175_1.nelement()).item(), arg175_1.amax().item(), arg175_1.amin().item())
        triton__1.run(arg2_1, arg175_1, arg2_1, 32, grid=grid(32), stream=stream0)
        print('triton__1', 'out_ptr0', 'arg2_1', (arg2_1.sum()/arg2_1.nelement()).item(), arg2_1.amax().item(), arg2_1.amin().item())
        del arg175_1
        del arg2_1
        print('triton__2', 'in_ptr0', 'arg3_1', (arg3_1.sum()/arg3_1.nelement()).item(), arg3_1.amax().item(), arg3_1.amin().item())
        print('triton__2', 'in_ptr1', 'arg176_1', (arg176_1.sum()/arg176_1.nelement()).item(), arg176_1.amax().item(), arg176_1.amin().item())
        triton__2.run(arg3_1, arg176_1, arg3_1, 4096, grid=grid(4096), stream=stream0)
        print('triton__2', 'out_ptr0', 'arg3_1', (arg3_1.sum()/arg3_1.nelement()).item(), arg3_1.amax().item(), arg3_1.amin().item())
        del arg176_1
        del arg3_1
        print('triton__3', 'in_ptr0', 'arg4_1', (arg4_1.sum()/arg4_1.nelement()).item(), arg4_1.amax().item(), arg4_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg177_1', (arg177_1.sum()/arg177_1.nelement()).item(), arg177_1.amax().item(), arg177_1.amin().item())
        triton__3.run(arg4_1, arg177_1, arg4_1, 128, grid=grid(128), stream=stream0)
        print('triton__3', 'out_ptr0', 'arg4_1', (arg4_1.sum()/arg4_1.nelement()).item(), arg4_1.amax().item(), arg4_1.amin().item())
        del arg177_1
        del arg4_1
        print('triton__3', 'in_ptr0', 'arg5_1', (arg5_1.sum()/arg5_1.nelement()).item(), arg5_1.amax().item(), arg5_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg178_1', (arg178_1.sum()/arg178_1.nelement()).item(), arg178_1.amax().item(), arg178_1.amin().item())
        triton__3.run(arg5_1, arg178_1, arg5_1, 128, grid=grid(128), stream=stream0)
        print('triton__3', 'out_ptr0', 'arg5_1', (arg5_1.sum()/arg5_1.nelement()).item(), arg5_1.amax().item(), arg5_1.amin().item())
        del arg178_1
        del arg5_1
        print('triton__4', 'in_ptr0', 'arg6_1', (arg6_1.sum()/arg6_1.nelement()).item(), arg6_1.amax().item(), arg6_1.amin().item())
        print('triton__4', 'in_ptr1', 'arg179_1', (arg179_1.sum()/arg179_1.nelement()).item(), arg179_1.amax().item(), arg179_1.amin().item())
        triton__4.run(arg6_1, arg179_1, arg6_1, 36864, grid=grid(36864), stream=stream0)
        print('triton__4', 'out_ptr0', 'arg6_1', (arg6_1.sum()/arg6_1.nelement()).item(), arg6_1.amax().item(), arg6_1.amin().item())
        del arg179_1
        del arg6_1
        print('triton__3', 'in_ptr0', 'arg7_1', (arg7_1.sum()/arg7_1.nelement()).item(), arg7_1.amax().item(), arg7_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg180_1', (arg180_1.sum()/arg180_1.nelement()).item(), arg180_1.amax().item(), arg180_1.amin().item())
        triton__3.run(arg7_1, arg180_1, arg7_1, 128, grid=grid(128), stream=stream0)
        print('triton__3', 'out_ptr0', 'arg7_1', (arg7_1.sum()/arg7_1.nelement()).item(), arg7_1.amax().item(), arg7_1.amin().item())
        del arg180_1
        del arg7_1
        print('triton__3', 'in_ptr0', 'arg8_1', (arg8_1.sum()/arg8_1.nelement()).item(), arg8_1.amax().item(), arg8_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg181_1', (arg181_1.sum()/arg181_1.nelement()).item(), arg181_1.amax().item(), arg181_1.amin().item())
        triton__3.run(arg8_1, arg181_1, arg8_1, 128, grid=grid(128), stream=stream0)
        print('triton__3', 'out_ptr0', 'arg8_1', (arg8_1.sum()/arg8_1.nelement()).item(), arg8_1.amax().item(), arg8_1.amin().item())
        del arg181_1
        del arg8_1
        print('triton__5', 'in_ptr0', 'arg9_1', (arg9_1.sum()/arg9_1.nelement()).item(), arg9_1.amax().item(), arg9_1.amin().item())
        print('triton__5', 'in_ptr1', 'arg182_1', (arg182_1.sum()/arg182_1.nelement()).item(), arg182_1.amax().item(), arg182_1.amin().item())
        triton__5.run(arg9_1, arg182_1, arg9_1, 147456, grid=grid(147456), stream=stream0)
        print('triton__5', 'out_ptr0', 'arg9_1', (arg9_1.sum()/arg9_1.nelement()).item(), arg9_1.amax().item(), arg9_1.amin().item())
        del arg182_1
        del arg9_1
        print('triton__3', 'in_ptr0', 'arg10_1', (arg10_1.sum()/arg10_1.nelement()).item(), arg10_1.amax().item(), arg10_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg183_1', (arg183_1.sum()/arg183_1.nelement()).item(), arg183_1.amax().item(), arg183_1.amin().item())
        triton__3.run(arg10_1, arg183_1, arg10_1, 128, grid=grid(128), stream=stream0)
        print('triton__3', 'out_ptr0', 'arg10_1', (arg10_1.sum()/arg10_1.nelement()).item(), arg10_1.amax().item(), arg10_1.amin().item())
        del arg10_1
        del arg183_1
        print('triton__3', 'in_ptr0', 'arg11_1', (arg11_1.sum()/arg11_1.nelement()).item(), arg11_1.amax().item(), arg11_1.amin().item())
        print('triton__3', 'in_ptr1', 'arg184_1', (arg184_1.sum()/arg184_1.nelement()).item(), arg184_1.amax().item(), arg184_1.amin().item())
        triton__3.run(arg11_1, arg184_1, arg11_1, 128, grid=grid(128), stream=stream0)
        print('triton__3', 'out_ptr0', 'arg11_1', (arg11_1.sum()/arg11_1.nelement()).item(), arg11_1.amax().item(), arg11_1.amin().item())
        del arg11_1
        del arg184_1
        print('triton__6', 'in_ptr0', 'arg12_1', (arg12_1.sum()/arg12_1.nelement()).item(), arg12_1.amax().item(), arg12_1.amin().item())
        print('triton__6', 'in_ptr1', 'arg185_1', (arg185_1.sum()/arg185_1.nelement()).item(), arg185_1.amax().item(), arg185_1.amin().item())
        triton__6.run(arg12_1, arg185_1, arg12_1, 24576, grid=grid(24576), stream=stream0)
        print('triton__6', 'out_ptr0', 'arg12_1', (arg12_1.sum()/arg12_1.nelement()).item(), arg12_1.amax().item(), arg12_1.amin().item())
        del arg12_1
        del arg185_1
        print('triton__7', 'in_ptr0', 'arg13_1', (arg13_1.sum()/arg13_1.nelement()).item(), arg13_1.amax().item(), arg13_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg186_1', (arg186_1.sum()/arg186_1.nelement()).item(), arg186_1.amax().item(), arg186_1.amin().item())
        triton__7.run(arg13_1, arg186_1, arg13_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg13_1', (arg13_1.sum()/arg13_1.nelement()).item(), arg13_1.amax().item(), arg13_1.amin().item())
        del arg13_1
        del arg186_1
        print('triton__7', 'in_ptr0', 'arg14_1', (arg14_1.sum()/arg14_1.nelement()).item(), arg14_1.amax().item(), arg14_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg187_1', (arg187_1.sum()/arg187_1.nelement()).item(), arg187_1.amax().item(), arg187_1.amin().item())
        triton__7.run(arg14_1, arg187_1, arg14_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg14_1', (arg14_1.sum()/arg14_1.nelement()).item(), arg14_1.amax().item(), arg14_1.amin().item())
        del arg14_1
        del arg187_1
        print('triton__8', 'in_ptr0', 'arg15_1', (arg15_1.sum()/arg15_1.nelement()).item(), arg15_1.amax().item(), arg15_1.amin().item())
        print('triton__8', 'in_ptr1', 'arg188_1', (arg188_1.sum()/arg188_1.nelement()).item(), arg188_1.amax().item(), arg188_1.amin().item())
        triton__8.run(arg15_1, arg188_1, arg15_1, 221184, grid=grid(221184), stream=stream0)
        print('triton__8', 'out_ptr0', 'arg15_1', (arg15_1.sum()/arg15_1.nelement()).item(), arg15_1.amax().item(), arg15_1.amin().item())
        del arg15_1
        del arg188_1
        print('triton__7', 'in_ptr0', 'arg16_1', (arg16_1.sum()/arg16_1.nelement()).item(), arg16_1.amax().item(), arg16_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg189_1', (arg189_1.sum()/arg189_1.nelement()).item(), arg189_1.amax().item(), arg189_1.amin().item())
        triton__7.run(arg16_1, arg189_1, arg16_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg16_1', (arg16_1.sum()/arg16_1.nelement()).item(), arg16_1.amax().item(), arg16_1.amin().item())
        del arg16_1
        del arg189_1
        print('triton__7', 'in_ptr0', 'arg17_1', (arg17_1.sum()/arg17_1.nelement()).item(), arg17_1.amax().item(), arg17_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg190_1', (arg190_1.sum()/arg190_1.nelement()).item(), arg190_1.amax().item(), arg190_1.amin().item())
        triton__7.run(arg17_1, arg190_1, arg17_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg17_1', (arg17_1.sum()/arg17_1.nelement()).item(), arg17_1.amax().item(), arg17_1.amin().item())
        del arg17_1
        del arg190_1
        print('triton__9', 'in_ptr0', 'arg18_1', (arg18_1.sum()/arg18_1.nelement()).item(), arg18_1.amax().item(), arg18_1.amin().item())
        print('triton__9', 'in_ptr1', 'arg191_1', (arg191_1.sum()/arg191_1.nelement()).item(), arg191_1.amax().item(), arg191_1.amin().item())
        triton__9.run(arg18_1, arg191_1, arg18_1, 331776, grid=grid(331776), stream=stream0)
        print('triton__9', 'out_ptr0', 'arg18_1', (arg18_1.sum()/arg18_1.nelement()).item(), arg18_1.amax().item(), arg18_1.amin().item())
        del arg18_1
        del arg191_1
        print('triton__7', 'in_ptr0', 'arg19_1', (arg19_1.sum()/arg19_1.nelement()).item(), arg19_1.amax().item(), arg19_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg192_1', (arg192_1.sum()/arg192_1.nelement()).item(), arg192_1.amax().item(), arg192_1.amin().item())
        triton__7.run(arg19_1, arg192_1, arg19_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg19_1', (arg19_1.sum()/arg19_1.nelement()).item(), arg19_1.amax().item(), arg19_1.amin().item())
        del arg192_1
        del arg19_1
        print('triton__7', 'in_ptr0', 'arg20_1', (arg20_1.sum()/arg20_1.nelement()).item(), arg20_1.amax().item(), arg20_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg193_1', (arg193_1.sum()/arg193_1.nelement()).item(), arg193_1.amax().item(), arg193_1.amin().item())
        triton__7.run(arg20_1, arg193_1, arg20_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg20_1', (arg20_1.sum()/arg20_1.nelement()).item(), arg20_1.amax().item(), arg20_1.amin().item())
        del arg193_1
        del arg20_1
        print('triton__9', 'in_ptr0', 'arg21_1', (arg21_1.sum()/arg21_1.nelement()).item(), arg21_1.amax().item(), arg21_1.amin().item())
        print('triton__9', 'in_ptr1', 'arg194_1', (arg194_1.sum()/arg194_1.nelement()).item(), arg194_1.amax().item(), arg194_1.amin().item())
        triton__9.run(arg21_1, arg194_1, arg21_1, 331776, grid=grid(331776), stream=stream0)
        print('triton__9', 'out_ptr0', 'arg21_1', (arg21_1.sum()/arg21_1.nelement()).item(), arg21_1.amax().item(), arg21_1.amin().item())
        del arg194_1
        del arg21_1
        print('triton__7', 'in_ptr0', 'arg22_1', (arg22_1.sum()/arg22_1.nelement()).item(), arg22_1.amax().item(), arg22_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg195_1', (arg195_1.sum()/arg195_1.nelement()).item(), arg195_1.amax().item(), arg195_1.amin().item())
        triton__7.run(arg22_1, arg195_1, arg22_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg22_1', (arg22_1.sum()/arg22_1.nelement()).item(), arg22_1.amax().item(), arg22_1.amin().item())
        del arg195_1
        del arg22_1
        print('triton__7', 'in_ptr0', 'arg23_1', (arg23_1.sum()/arg23_1.nelement()).item(), arg23_1.amax().item(), arg23_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg196_1', (arg196_1.sum()/arg196_1.nelement()).item(), arg196_1.amax().item(), arg196_1.amin().item())
        triton__7.run(arg23_1, arg196_1, arg23_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg23_1', (arg23_1.sum()/arg23_1.nelement()).item(), arg23_1.amax().item(), arg23_1.amin().item())
        del arg196_1
        del arg23_1
        print('triton__9', 'in_ptr0', 'arg24_1', (arg24_1.sum()/arg24_1.nelement()).item(), arg24_1.amax().item(), arg24_1.amin().item())
        print('triton__9', 'in_ptr1', 'arg197_1', (arg197_1.sum()/arg197_1.nelement()).item(), arg197_1.amax().item(), arg197_1.amin().item())
        triton__9.run(arg24_1, arg197_1, arg24_1, 331776, grid=grid(331776), stream=stream0)
        print('triton__9', 'out_ptr0', 'arg24_1', (arg24_1.sum()/arg24_1.nelement()).item(), arg24_1.amax().item(), arg24_1.amin().item())
        del arg197_1
        del arg24_1
        print('triton__7', 'in_ptr0', 'arg25_1', (arg25_1.sum()/arg25_1.nelement()).item(), arg25_1.amax().item(), arg25_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg198_1', (arg198_1.sum()/arg198_1.nelement()).item(), arg198_1.amax().item(), arg198_1.amin().item())
        triton__7.run(arg25_1, arg198_1, arg25_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg25_1', (arg25_1.sum()/arg25_1.nelement()).item(), arg25_1.amax().item(), arg25_1.amin().item())
        del arg198_1
        del arg25_1
        print('triton__7', 'in_ptr0', 'arg26_1', (arg26_1.sum()/arg26_1.nelement()).item(), arg26_1.amax().item(), arg26_1.amin().item())
        print('triton__7', 'in_ptr1', 'arg199_1', (arg199_1.sum()/arg199_1.nelement()).item(), arg199_1.amax().item(), arg199_1.amin().item())
        triton__7.run(arg26_1, arg199_1, arg26_1, 192, grid=grid(192), stream=stream0)
        print('triton__7', 'out_ptr0', 'arg26_1', (arg26_1.sum()/arg26_1.nelement()).item(), arg26_1.amax().item(), arg26_1.amin().item())
        del arg199_1
        del arg26_1
        print('triton__10', 'in_ptr0', 'arg27_1', (arg27_1.sum()/arg27_1.nelement()).item(), arg27_1.amax().item(), arg27_1.amin().item())
        print('triton__10', 'in_ptr1', 'arg200_1', (arg200_1.sum()/arg200_1.nelement()).item(), arg200_1.amax().item(), arg200_1.amin().item())
        triton__10.run(arg27_1, arg200_1, arg27_1, 122880, grid=grid(122880), stream=stream0)
        print('triton__10', 'out_ptr0', 'arg27_1', (arg27_1.sum()/arg27_1.nelement()).item(), arg27_1.amax().item(), arg27_1.amin().item())
        del arg200_1
        del arg27_1
        print('triton__11', 'in_ptr0', 'arg28_1', (arg28_1.sum()/arg28_1.nelement()).item(), arg28_1.amax().item(), arg28_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg201_1', (arg201_1.sum()/arg201_1.nelement()).item(), arg201_1.amax().item(), arg201_1.amin().item())
        triton__11.run(arg28_1, arg201_1, arg28_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg28_1', (arg28_1.sum()/arg28_1.nelement()).item(), arg28_1.amax().item(), arg28_1.amin().item())
        del arg201_1
        del arg28_1
        print('triton__11', 'in_ptr0', 'arg29_1', (arg29_1.sum()/arg29_1.nelement()).item(), arg29_1.amax().item(), arg29_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg202_1', (arg202_1.sum()/arg202_1.nelement()).item(), arg202_1.amax().item(), arg202_1.amin().item())
        triton__11.run(arg29_1, arg202_1, arg29_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg29_1', (arg29_1.sum()/arg29_1.nelement()).item(), arg29_1.amax().item(), arg29_1.amin().item())
        del arg202_1
        del arg29_1
        print('triton__12', 'in_ptr0', 'arg30_1', (arg30_1.sum()/arg30_1.nelement()).item(), arg30_1.amax().item(), arg30_1.amin().item())
        print('triton__12', 'in_ptr1', 'arg203_1', (arg203_1.sum()/arg203_1.nelement()).item(), arg203_1.amax().item(), arg203_1.amin().item())
        triton__12.run(arg30_1, arg203_1, arg30_1, 30720, grid=grid(30720), stream=stream0)
        print('triton__12', 'out_ptr0', 'arg30_1', (arg30_1.sum()/arg30_1.nelement()).item(), arg30_1.amax().item(), arg30_1.amin().item())
        del arg203_1
        del arg30_1
        print('triton__13', 'in_ptr0', 'arg31_1', (arg31_1.sum()/arg31_1.nelement()).item(), arg31_1.amax().item(), arg31_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg204_1', (arg204_1.sum()/arg204_1.nelement()).item(), arg204_1.amax().item(), arg204_1.amin().item())
        triton__13.run(arg31_1, arg204_1, arg31_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg31_1', (arg31_1.sum()/arg31_1.nelement()).item(), arg31_1.amax().item(), arg31_1.amin().item())
        del arg204_1
        del arg31_1
        print('triton__13', 'in_ptr0', 'arg32_1', (arg32_1.sum()/arg32_1.nelement()).item(), arg32_1.amax().item(), arg32_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg205_1', (arg205_1.sum()/arg205_1.nelement()).item(), arg205_1.amax().item(), arg205_1.amin().item())
        triton__13.run(arg32_1, arg205_1, arg32_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg32_1', (arg32_1.sum()/arg32_1.nelement()).item(), arg32_1.amax().item(), arg32_1.amin().item())
        del arg205_1
        del arg32_1
        print('triton__14', 'in_ptr0', 'arg33_1', (arg33_1.sum()/arg33_1.nelement()).item(), arg33_1.amax().item(), arg33_1.amin().item())
        print('triton__14', 'in_ptr1', 'arg206_1', (arg206_1.sum()/arg206_1.nelement()).item(), arg206_1.amax().item(), arg206_1.amin().item())
        triton__14.run(arg33_1, arg206_1, arg33_1, 230400, grid=grid(230400), stream=stream0)
        print('triton__14', 'out_ptr0', 'arg33_1', (arg33_1.sum()/arg33_1.nelement()).item(), arg33_1.amax().item(), arg33_1.amin().item())
        del arg206_1
        del arg33_1
        print('triton__13', 'in_ptr0', 'arg34_1', (arg34_1.sum()/arg34_1.nelement()).item(), arg34_1.amax().item(), arg34_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg207_1', (arg207_1.sum()/arg207_1.nelement()).item(), arg207_1.amax().item(), arg207_1.amin().item())
        triton__13.run(arg34_1, arg207_1, arg34_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg34_1', (arg34_1.sum()/arg34_1.nelement()).item(), arg34_1.amax().item(), arg34_1.amin().item())
        del arg207_1
        del arg34_1
        print('triton__13', 'in_ptr0', 'arg35_1', (arg35_1.sum()/arg35_1.nelement()).item(), arg35_1.amax().item(), arg35_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg208_1', (arg208_1.sum()/arg208_1.nelement()).item(), arg208_1.amax().item(), arg208_1.amin().item())
        triton__13.run(arg35_1, arg208_1, arg35_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg35_1', (arg35_1.sum()/arg35_1.nelement()).item(), arg35_1.amax().item(), arg35_1.amin().item())
        del arg208_1
        del arg35_1
        print('triton__15', 'in_ptr0', 'arg36_1', (arg36_1.sum()/arg36_1.nelement()).item(), arg36_1.amax().item(), arg36_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg209_1', (arg209_1.sum()/arg209_1.nelement()).item(), arg209_1.amax().item(), arg209_1.amin().item())
        triton__15.run(arg36_1, arg209_1, arg36_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg36_1', (arg36_1.sum()/arg36_1.nelement()).item(), arg36_1.amax().item(), arg36_1.amin().item())
        del arg209_1
        del arg36_1
        print('triton__11', 'in_ptr0', 'arg37_1', (arg37_1.sum()/arg37_1.nelement()).item(), arg37_1.amax().item(), arg37_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg210_1', (arg210_1.sum()/arg210_1.nelement()).item(), arg210_1.amax().item(), arg210_1.amin().item())
        triton__11.run(arg37_1, arg210_1, arg37_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg37_1', (arg37_1.sum()/arg37_1.nelement()).item(), arg37_1.amax().item(), arg37_1.amin().item())
        del arg210_1
        del arg37_1
        print('triton__11', 'in_ptr0', 'arg38_1', (arg38_1.sum()/arg38_1.nelement()).item(), arg38_1.amax().item(), arg38_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg211_1', (arg211_1.sum()/arg211_1.nelement()).item(), arg211_1.amax().item(), arg211_1.amin().item())
        triton__11.run(arg38_1, arg211_1, arg38_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg38_1', (arg38_1.sum()/arg38_1.nelement()).item(), arg38_1.amax().item(), arg38_1.amin().item())
        del arg211_1
        del arg38_1
        print('triton__15', 'in_ptr0', 'arg39_1', (arg39_1.sum()/arg39_1.nelement()).item(), arg39_1.amax().item(), arg39_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg212_1', (arg212_1.sum()/arg212_1.nelement()).item(), arg212_1.amax().item(), arg212_1.amin().item())
        triton__15.run(arg39_1, arg212_1, arg39_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg39_1', (arg39_1.sum()/arg39_1.nelement()).item(), arg39_1.amax().item(), arg39_1.amin().item())
        del arg212_1
        del arg39_1
        print('triton__13', 'in_ptr0', 'arg40_1', (arg40_1.sum()/arg40_1.nelement()).item(), arg40_1.amax().item(), arg40_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg213_1', (arg213_1.sum()/arg213_1.nelement()).item(), arg213_1.amax().item(), arg213_1.amin().item())
        triton__13.run(arg40_1, arg213_1, arg40_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg40_1', (arg40_1.sum()/arg40_1.nelement()).item(), arg40_1.amax().item(), arg40_1.amin().item())
        del arg213_1
        del arg40_1
        print('triton__13', 'in_ptr0', 'arg41_1', (arg41_1.sum()/arg41_1.nelement()).item(), arg41_1.amax().item(), arg41_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg214_1', (arg214_1.sum()/arg214_1.nelement()).item(), arg214_1.amax().item(), arg214_1.amin().item())
        triton__13.run(arg41_1, arg214_1, arg41_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg41_1', (arg41_1.sum()/arg41_1.nelement()).item(), arg41_1.amax().item(), arg41_1.amin().item())
        del arg214_1
        del arg41_1
        print('triton__14', 'in_ptr0', 'arg42_1', (arg42_1.sum()/arg42_1.nelement()).item(), arg42_1.amax().item(), arg42_1.amin().item())
        print('triton__14', 'in_ptr1', 'arg215_1', (arg215_1.sum()/arg215_1.nelement()).item(), arg215_1.amax().item(), arg215_1.amin().item())
        triton__14.run(arg42_1, arg215_1, arg42_1, 230400, grid=grid(230400), stream=stream0)
        print('triton__14', 'out_ptr0', 'arg42_1', (arg42_1.sum()/arg42_1.nelement()).item(), arg42_1.amax().item(), arg42_1.amin().item())
        del arg215_1
        del arg42_1
        print('triton__13', 'in_ptr0', 'arg43_1', (arg43_1.sum()/arg43_1.nelement()).item(), arg43_1.amax().item(), arg43_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg216_1', (arg216_1.sum()/arg216_1.nelement()).item(), arg216_1.amax().item(), arg216_1.amin().item())
        triton__13.run(arg43_1, arg216_1, arg43_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg43_1', (arg43_1.sum()/arg43_1.nelement()).item(), arg43_1.amax().item(), arg43_1.amin().item())
        del arg216_1
        del arg43_1
        print('triton__13', 'in_ptr0', 'arg44_1', (arg44_1.sum()/arg44_1.nelement()).item(), arg44_1.amax().item(), arg44_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg217_1', (arg217_1.sum()/arg217_1.nelement()).item(), arg217_1.amax().item(), arg217_1.amin().item())
        triton__13.run(arg44_1, arg217_1, arg44_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg44_1', (arg44_1.sum()/arg44_1.nelement()).item(), arg44_1.amax().item(), arg44_1.amin().item())
        del arg217_1
        del arg44_1
        print('triton__15', 'in_ptr0', 'arg45_1', (arg45_1.sum()/arg45_1.nelement()).item(), arg45_1.amax().item(), arg45_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg218_1', (arg218_1.sum()/arg218_1.nelement()).item(), arg218_1.amax().item(), arg218_1.amin().item())
        triton__15.run(arg45_1, arg218_1, arg45_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg45_1', (arg45_1.sum()/arg45_1.nelement()).item(), arg45_1.amax().item(), arg45_1.amin().item())
        del arg218_1
        del arg45_1
        print('triton__11', 'in_ptr0', 'arg46_1', (arg46_1.sum()/arg46_1.nelement()).item(), arg46_1.amax().item(), arg46_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg219_1', (arg219_1.sum()/arg219_1.nelement()).item(), arg219_1.amax().item(), arg219_1.amin().item())
        triton__11.run(arg46_1, arg219_1, arg46_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg46_1', (arg46_1.sum()/arg46_1.nelement()).item(), arg46_1.amax().item(), arg46_1.amin().item())
        del arg219_1
        del arg46_1
        print('triton__11', 'in_ptr0', 'arg47_1', (arg47_1.sum()/arg47_1.nelement()).item(), arg47_1.amax().item(), arg47_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg220_1', (arg220_1.sum()/arg220_1.nelement()).item(), arg220_1.amax().item(), arg220_1.amin().item())
        triton__11.run(arg47_1, arg220_1, arg47_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg47_1', (arg47_1.sum()/arg47_1.nelement()).item(), arg47_1.amax().item(), arg47_1.amin().item())
        del arg220_1
        del arg47_1
        print('triton__15', 'in_ptr0', 'arg48_1', (arg48_1.sum()/arg48_1.nelement()).item(), arg48_1.amax().item(), arg48_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg221_1', (arg221_1.sum()/arg221_1.nelement()).item(), arg221_1.amax().item(), arg221_1.amin().item())
        triton__15.run(arg48_1, arg221_1, arg48_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg48_1', (arg48_1.sum()/arg48_1.nelement()).item(), arg48_1.amax().item(), arg48_1.amin().item())
        del arg221_1
        del arg48_1
        print('triton__13', 'in_ptr0', 'arg49_1', (arg49_1.sum()/arg49_1.nelement()).item(), arg49_1.amax().item(), arg49_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg222_1', (arg222_1.sum()/arg222_1.nelement()).item(), arg222_1.amax().item(), arg222_1.amin().item())
        triton__13.run(arg49_1, arg222_1, arg49_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg49_1', (arg49_1.sum()/arg49_1.nelement()).item(), arg49_1.amax().item(), arg49_1.amin().item())
        del arg222_1
        del arg49_1
        print('triton__13', 'in_ptr0', 'arg50_1', (arg50_1.sum()/arg50_1.nelement()).item(), arg50_1.amax().item(), arg50_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg223_1', (arg223_1.sum()/arg223_1.nelement()).item(), arg223_1.amax().item(), arg223_1.amin().item())
        triton__13.run(arg50_1, arg223_1, arg50_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg50_1', (arg50_1.sum()/arg50_1.nelement()).item(), arg50_1.amax().item(), arg50_1.amin().item())
        del arg223_1
        del arg50_1
        print('triton__14', 'in_ptr0', 'arg51_1', (arg51_1.sum()/arg51_1.nelement()).item(), arg51_1.amax().item(), arg51_1.amin().item())
        print('triton__14', 'in_ptr1', 'arg224_1', (arg224_1.sum()/arg224_1.nelement()).item(), arg224_1.amax().item(), arg224_1.amin().item())
        triton__14.run(arg51_1, arg224_1, arg51_1, 230400, grid=grid(230400), stream=stream0)
        print('triton__14', 'out_ptr0', 'arg51_1', (arg51_1.sum()/arg51_1.nelement()).item(), arg51_1.amax().item(), arg51_1.amin().item())
        del arg224_1
        del arg51_1
        print('triton__13', 'in_ptr0', 'arg52_1', (arg52_1.sum()/arg52_1.nelement()).item(), arg52_1.amax().item(), arg52_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg225_1', (arg225_1.sum()/arg225_1.nelement()).item(), arg225_1.amax().item(), arg225_1.amin().item())
        triton__13.run(arg52_1, arg225_1, arg52_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg52_1', (arg52_1.sum()/arg52_1.nelement()).item(), arg52_1.amax().item(), arg52_1.amin().item())
        del arg225_1
        del arg52_1
        print('triton__13', 'in_ptr0', 'arg53_1', (arg53_1.sum()/arg53_1.nelement()).item(), arg53_1.amax().item(), arg53_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg226_1', (arg226_1.sum()/arg226_1.nelement()).item(), arg226_1.amax().item(), arg226_1.amin().item())
        triton__13.run(arg53_1, arg226_1, arg53_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg53_1', (arg53_1.sum()/arg53_1.nelement()).item(), arg53_1.amax().item(), arg53_1.amin().item())
        del arg226_1
        del arg53_1
        print('triton__15', 'in_ptr0', 'arg54_1', (arg54_1.sum()/arg54_1.nelement()).item(), arg54_1.amax().item(), arg54_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg227_1', (arg227_1.sum()/arg227_1.nelement()).item(), arg227_1.amax().item(), arg227_1.amin().item())
        triton__15.run(arg54_1, arg227_1, arg54_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg54_1', (arg54_1.sum()/arg54_1.nelement()).item(), arg54_1.amax().item(), arg54_1.amin().item())
        del arg227_1
        del arg54_1
        print('triton__11', 'in_ptr0', 'arg55_1', (arg55_1.sum()/arg55_1.nelement()).item(), arg55_1.amax().item(), arg55_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg228_1', (arg228_1.sum()/arg228_1.nelement()).item(), arg228_1.amax().item(), arg228_1.amin().item())
        triton__11.run(arg55_1, arg228_1, arg55_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg55_1', (arg55_1.sum()/arg55_1.nelement()).item(), arg55_1.amax().item(), arg55_1.amin().item())
        del arg228_1
        del arg55_1
        print('triton__11', 'in_ptr0', 'arg56_1', (arg56_1.sum()/arg56_1.nelement()).item(), arg56_1.amax().item(), arg56_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg229_1', (arg229_1.sum()/arg229_1.nelement()).item(), arg229_1.amax().item(), arg229_1.amin().item())
        triton__11.run(arg56_1, arg229_1, arg56_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg56_1', (arg56_1.sum()/arg56_1.nelement()).item(), arg56_1.amax().item(), arg56_1.amin().item())
        del arg229_1
        del arg56_1
        print('triton__15', 'in_ptr0', 'arg57_1', (arg57_1.sum()/arg57_1.nelement()).item(), arg57_1.amax().item(), arg57_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg230_1', (arg230_1.sum()/arg230_1.nelement()).item(), arg230_1.amax().item(), arg230_1.amin().item())
        triton__15.run(arg57_1, arg230_1, arg57_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg57_1', (arg57_1.sum()/arg57_1.nelement()).item(), arg57_1.amax().item(), arg57_1.amin().item())
        del arg230_1
        del arg57_1
        print('triton__13', 'in_ptr0', 'arg58_1', (arg58_1.sum()/arg58_1.nelement()).item(), arg58_1.amax().item(), arg58_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg231_1', (arg231_1.sum()/arg231_1.nelement()).item(), arg231_1.amax().item(), arg231_1.amin().item())
        triton__13.run(arg58_1, arg231_1, arg58_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg58_1', (arg58_1.sum()/arg58_1.nelement()).item(), arg58_1.amax().item(), arg58_1.amin().item())
        del arg231_1
        del arg58_1
        print('triton__13', 'in_ptr0', 'arg59_1', (arg59_1.sum()/arg59_1.nelement()).item(), arg59_1.amax().item(), arg59_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg232_1', (arg232_1.sum()/arg232_1.nelement()).item(), arg232_1.amax().item(), arg232_1.amin().item())
        triton__13.run(arg59_1, arg232_1, arg59_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg59_1', (arg59_1.sum()/arg59_1.nelement()).item(), arg59_1.amax().item(), arg59_1.amin().item())
        del arg232_1
        del arg59_1
        print('triton__14', 'in_ptr0', 'arg60_1', (arg60_1.sum()/arg60_1.nelement()).item(), arg60_1.amax().item(), arg60_1.amin().item())
        print('triton__14', 'in_ptr1', 'arg233_1', (arg233_1.sum()/arg233_1.nelement()).item(), arg233_1.amax().item(), arg233_1.amin().item())
        triton__14.run(arg60_1, arg233_1, arg60_1, 230400, grid=grid(230400), stream=stream0)
        print('triton__14', 'out_ptr0', 'arg60_1', (arg60_1.sum()/arg60_1.nelement()).item(), arg60_1.amax().item(), arg60_1.amin().item())
        del arg233_1
        del arg60_1
        print('triton__13', 'in_ptr0', 'arg61_1', (arg61_1.sum()/arg61_1.nelement()).item(), arg61_1.amax().item(), arg61_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg234_1', (arg234_1.sum()/arg234_1.nelement()).item(), arg234_1.amax().item(), arg234_1.amin().item())
        triton__13.run(arg61_1, arg234_1, arg61_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg61_1', (arg61_1.sum()/arg61_1.nelement()).item(), arg61_1.amax().item(), arg61_1.amin().item())
        del arg234_1
        del arg61_1
        print('triton__13', 'in_ptr0', 'arg62_1', (arg62_1.sum()/arg62_1.nelement()).item(), arg62_1.amax().item(), arg62_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg235_1', (arg235_1.sum()/arg235_1.nelement()).item(), arg235_1.amax().item(), arg235_1.amin().item())
        triton__13.run(arg62_1, arg235_1, arg62_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg62_1', (arg62_1.sum()/arg62_1.nelement()).item(), arg62_1.amax().item(), arg62_1.amin().item())
        del arg235_1
        del arg62_1
        print('triton__15', 'in_ptr0', 'arg63_1', (arg63_1.sum()/arg63_1.nelement()).item(), arg63_1.amax().item(), arg63_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg236_1', (arg236_1.sum()/arg236_1.nelement()).item(), arg236_1.amax().item(), arg236_1.amin().item())
        triton__15.run(arg63_1, arg236_1, arg63_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg63_1', (arg63_1.sum()/arg63_1.nelement()).item(), arg63_1.amax().item(), arg63_1.amin().item())
        del arg236_1
        del arg63_1
        print('triton__11', 'in_ptr0', 'arg64_1', (arg64_1.sum()/arg64_1.nelement()).item(), arg64_1.amax().item(), arg64_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg237_1', (arg237_1.sum()/arg237_1.nelement()).item(), arg237_1.amax().item(), arg237_1.amin().item())
        triton__11.run(arg64_1, arg237_1, arg64_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg64_1', (arg64_1.sum()/arg64_1.nelement()).item(), arg64_1.amax().item(), arg64_1.amin().item())
        del arg237_1
        del arg64_1
        print('triton__11', 'in_ptr0', 'arg65_1', (arg65_1.sum()/arg65_1.nelement()).item(), arg65_1.amax().item(), arg65_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg238_1', (arg238_1.sum()/arg238_1.nelement()).item(), arg238_1.amax().item(), arg238_1.amin().item())
        triton__11.run(arg65_1, arg238_1, arg65_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg65_1', (arg65_1.sum()/arg65_1.nelement()).item(), arg65_1.amax().item(), arg65_1.amin().item())
        del arg238_1
        del arg65_1
        print('triton__15', 'in_ptr0', 'arg66_1', (arg66_1.sum()/arg66_1.nelement()).item(), arg66_1.amax().item(), arg66_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg239_1', (arg239_1.sum()/arg239_1.nelement()).item(), arg239_1.amax().item(), arg239_1.amin().item())
        triton__15.run(arg66_1, arg239_1, arg66_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg66_1', (arg66_1.sum()/arg66_1.nelement()).item(), arg66_1.amax().item(), arg66_1.amin().item())
        del arg239_1
        del arg66_1
        print('triton__13', 'in_ptr0', 'arg67_1', (arg67_1.sum()/arg67_1.nelement()).item(), arg67_1.amax().item(), arg67_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg240_1', (arg240_1.sum()/arg240_1.nelement()).item(), arg240_1.amax().item(), arg240_1.amin().item())
        triton__13.run(arg67_1, arg240_1, arg67_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg67_1', (arg67_1.sum()/arg67_1.nelement()).item(), arg67_1.amax().item(), arg67_1.amin().item())
        del arg240_1
        del arg67_1
        print('triton__13', 'in_ptr0', 'arg68_1', (arg68_1.sum()/arg68_1.nelement()).item(), arg68_1.amax().item(), arg68_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg241_1', (arg241_1.sum()/arg241_1.nelement()).item(), arg241_1.amax().item(), arg241_1.amin().item())
        triton__13.run(arg68_1, arg241_1, arg68_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg68_1', (arg68_1.sum()/arg68_1.nelement()).item(), arg68_1.amax().item(), arg68_1.amin().item())
        del arg241_1
        del arg68_1
        print('triton__14', 'in_ptr0', 'arg69_1', (arg69_1.sum()/arg69_1.nelement()).item(), arg69_1.amax().item(), arg69_1.amin().item())
        print('triton__14', 'in_ptr1', 'arg242_1', (arg242_1.sum()/arg242_1.nelement()).item(), arg242_1.amax().item(), arg242_1.amin().item())
        triton__14.run(arg69_1, arg242_1, arg69_1, 230400, grid=grid(230400), stream=stream0)
        print('triton__14', 'out_ptr0', 'arg69_1', (arg69_1.sum()/arg69_1.nelement()).item(), arg69_1.amax().item(), arg69_1.amin().item())
        del arg242_1
        del arg69_1
        print('triton__13', 'in_ptr0', 'arg70_1', (arg70_1.sum()/arg70_1.nelement()).item(), arg70_1.amax().item(), arg70_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg243_1', (arg243_1.sum()/arg243_1.nelement()).item(), arg243_1.amax().item(), arg243_1.amin().item())
        triton__13.run(arg70_1, arg243_1, arg70_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg70_1', (arg70_1.sum()/arg70_1.nelement()).item(), arg70_1.amax().item(), arg70_1.amin().item())
        del arg243_1
        del arg70_1
        print('triton__13', 'in_ptr0', 'arg71_1', (arg71_1.sum()/arg71_1.nelement()).item(), arg71_1.amax().item(), arg71_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg244_1', (arg244_1.sum()/arg244_1.nelement()).item(), arg244_1.amax().item(), arg244_1.amin().item())
        triton__13.run(arg71_1, arg244_1, arg71_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg71_1', (arg71_1.sum()/arg71_1.nelement()).item(), arg71_1.amax().item(), arg71_1.amin().item())
        del arg244_1
        del arg71_1
        print('triton__15', 'in_ptr0', 'arg72_1', (arg72_1.sum()/arg72_1.nelement()).item(), arg72_1.amax().item(), arg72_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg245_1', (arg245_1.sum()/arg245_1.nelement()).item(), arg245_1.amax().item(), arg245_1.amin().item())
        triton__15.run(arg72_1, arg245_1, arg72_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg72_1', (arg72_1.sum()/arg72_1.nelement()).item(), arg72_1.amax().item(), arg72_1.amin().item())
        del arg245_1
        del arg72_1
        print('triton__11', 'in_ptr0', 'arg73_1', (arg73_1.sum()/arg73_1.nelement()).item(), arg73_1.amax().item(), arg73_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg246_1', (arg246_1.sum()/arg246_1.nelement()).item(), arg246_1.amax().item(), arg246_1.amin().item())
        triton__11.run(arg73_1, arg246_1, arg73_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg73_1', (arg73_1.sum()/arg73_1.nelement()).item(), arg73_1.amax().item(), arg73_1.amin().item())
        del arg246_1
        del arg73_1
        print('triton__11', 'in_ptr0', 'arg74_1', (arg74_1.sum()/arg74_1.nelement()).item(), arg74_1.amax().item(), arg74_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg247_1', (arg247_1.sum()/arg247_1.nelement()).item(), arg247_1.amax().item(), arg247_1.amin().item())
        triton__11.run(arg74_1, arg247_1, arg74_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg74_1', (arg74_1.sum()/arg74_1.nelement()).item(), arg74_1.amax().item(), arg74_1.amin().item())
        del arg247_1
        del arg74_1
        print('triton__15', 'in_ptr0', 'arg75_1', (arg75_1.sum()/arg75_1.nelement()).item(), arg75_1.amax().item(), arg75_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg248_1', (arg248_1.sum()/arg248_1.nelement()).item(), arg248_1.amax().item(), arg248_1.amin().item())
        triton__15.run(arg75_1, arg248_1, arg75_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg75_1', (arg75_1.sum()/arg75_1.nelement()).item(), arg75_1.amax().item(), arg75_1.amin().item())
        del arg248_1
        del arg75_1
        print('triton__13', 'in_ptr0', 'arg76_1', (arg76_1.sum()/arg76_1.nelement()).item(), arg76_1.amax().item(), arg76_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg249_1', (arg249_1.sum()/arg249_1.nelement()).item(), arg249_1.amax().item(), arg249_1.amin().item())
        triton__13.run(arg76_1, arg249_1, arg76_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg76_1', (arg76_1.sum()/arg76_1.nelement()).item(), arg76_1.amax().item(), arg76_1.amin().item())
        del arg249_1
        del arg76_1
        print('triton__13', 'in_ptr0', 'arg77_1', (arg77_1.sum()/arg77_1.nelement()).item(), arg77_1.amax().item(), arg77_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg250_1', (arg250_1.sum()/arg250_1.nelement()).item(), arg250_1.amax().item(), arg250_1.amin().item())
        triton__13.run(arg77_1, arg250_1, arg77_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg77_1', (arg77_1.sum()/arg77_1.nelement()).item(), arg77_1.amax().item(), arg77_1.amin().item())
        del arg250_1
        del arg77_1
        print('triton__14', 'in_ptr0', 'arg78_1', (arg78_1.sum()/arg78_1.nelement()).item(), arg78_1.amax().item(), arg78_1.amin().item())
        print('triton__14', 'in_ptr1', 'arg251_1', (arg251_1.sum()/arg251_1.nelement()).item(), arg251_1.amax().item(), arg251_1.amin().item())
        triton__14.run(arg78_1, arg251_1, arg78_1, 230400, grid=grid(230400), stream=stream0)
        print('triton__14', 'out_ptr0', 'arg78_1', (arg78_1.sum()/arg78_1.nelement()).item(), arg78_1.amax().item(), arg78_1.amin().item())
        del arg251_1
        del arg78_1
        print('triton__13', 'in_ptr0', 'arg79_1', (arg79_1.sum()/arg79_1.nelement()).item(), arg79_1.amax().item(), arg79_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg252_1', (arg252_1.sum()/arg252_1.nelement()).item(), arg252_1.amax().item(), arg252_1.amin().item())
        triton__13.run(arg79_1, arg252_1, arg79_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg79_1', (arg79_1.sum()/arg79_1.nelement()).item(), arg79_1.amax().item(), arg79_1.amin().item())
        del arg252_1
        del arg79_1
        print('triton__13', 'in_ptr0', 'arg80_1', (arg80_1.sum()/arg80_1.nelement()).item(), arg80_1.amax().item(), arg80_1.amin().item())
        print('triton__13', 'in_ptr1', 'arg253_1', (arg253_1.sum()/arg253_1.nelement()).item(), arg253_1.amax().item(), arg253_1.amin().item())
        triton__13.run(arg80_1, arg253_1, arg80_1, 160, grid=grid(160), stream=stream0)
        print('triton__13', 'out_ptr0', 'arg80_1', (arg80_1.sum()/arg80_1.nelement()).item(), arg80_1.amax().item(), arg80_1.amin().item())
        del arg253_1
        del arg80_1
        print('triton__15', 'in_ptr0', 'arg81_1', (arg81_1.sum()/arg81_1.nelement()).item(), arg81_1.amax().item(), arg81_1.amin().item())
        print('triton__15', 'in_ptr1', 'arg254_1', (arg254_1.sum()/arg254_1.nelement()).item(), arg254_1.amax().item(), arg254_1.amin().item())
        triton__15.run(arg81_1, arg254_1, arg81_1, 102400, grid=grid(102400), stream=stream0)
        print('triton__15', 'out_ptr0', 'arg81_1', (arg81_1.sum()/arg81_1.nelement()).item(), arg81_1.amax().item(), arg81_1.amin().item())
        del arg254_1
        del arg81_1
        print('triton__11', 'in_ptr0', 'arg82_1', (arg82_1.sum()/arg82_1.nelement()).item(), arg82_1.amax().item(), arg82_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg255_1', (arg255_1.sum()/arg255_1.nelement()).item(), arg255_1.amax().item(), arg255_1.amin().item())
        triton__11.run(arg82_1, arg255_1, arg82_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg82_1', (arg82_1.sum()/arg82_1.nelement()).item(), arg82_1.amax().item(), arg82_1.amin().item())
        del arg255_1
        del arg82_1
        print('triton__11', 'in_ptr0', 'arg83_1', (arg83_1.sum()/arg83_1.nelement()).item(), arg83_1.amax().item(), arg83_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg256_1', (arg256_1.sum()/arg256_1.nelement()).item(), arg256_1.amax().item(), arg256_1.amin().item())
        triton__11.run(arg83_1, arg256_1, arg83_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg83_1', (arg83_1.sum()/arg83_1.nelement()).item(), arg83_1.amax().item(), arg83_1.amin().item())
        del arg256_1
        del arg83_1
        print('triton__16', 'in_ptr0', 'arg84_1', (arg84_1.sum()/arg84_1.nelement()).item(), arg84_1.amax().item(), arg84_1.amin().item())
        print('triton__16', 'in_ptr1', 'arg257_1', (arg257_1.sum()/arg257_1.nelement()).item(), arg257_1.amax().item(), arg257_1.amin().item())
        triton__16.run(arg84_1, arg257_1, arg84_1, 409600, grid=grid(409600), stream=stream0)
        print('triton__16', 'out_ptr0', 'arg84_1', (arg84_1.sum()/arg84_1.nelement()).item(), arg84_1.amax().item(), arg84_1.amin().item())
        del arg257_1
        del arg84_1
        print('triton__11', 'in_ptr0', 'arg85_1', (arg85_1.sum()/arg85_1.nelement()).item(), arg85_1.amax().item(), arg85_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg258_1', (arg258_1.sum()/arg258_1.nelement()).item(), arg258_1.amax().item(), arg258_1.amin().item())
        triton__11.run(arg85_1, arg258_1, arg85_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg85_1', (arg85_1.sum()/arg85_1.nelement()).item(), arg85_1.amax().item(), arg85_1.amin().item())
        del arg258_1
        del arg85_1
        print('triton__11', 'in_ptr0', 'arg86_1', (arg86_1.sum()/arg86_1.nelement()).item(), arg86_1.amax().item(), arg86_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg259_1', (arg259_1.sum()/arg259_1.nelement()).item(), arg259_1.amax().item(), arg259_1.amin().item())
        triton__11.run(arg86_1, arg259_1, arg86_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg86_1', (arg86_1.sum()/arg86_1.nelement()).item(), arg86_1.amax().item(), arg86_1.amin().item())
        del arg259_1
        del arg86_1
        print('triton__17', 'in_ptr0', 'arg87_1', (arg87_1.sum()/arg87_1.nelement()).item(), arg87_1.amax().item(), arg87_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg260_1', (arg260_1.sum()/arg260_1.nelement()).item(), arg260_1.amax().item(), arg260_1.amin().item())
        triton__17.run(arg87_1, arg260_1, arg87_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg87_1', (arg87_1.sum()/arg87_1.nelement()).item(), arg87_1.amax().item(), arg87_1.amin().item())
        del arg260_1
        del arg87_1
        print('triton__18', 'in_ptr0', 'arg88_1', (arg88_1.sum()/arg88_1.nelement()).item(), arg88_1.amax().item(), arg88_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg261_1', (arg261_1.sum()/arg261_1.nelement()).item(), arg261_1.amax().item(), arg261_1.amin().item())
        triton__18.run(arg88_1, arg261_1, arg88_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg88_1', (arg88_1.sum()/arg88_1.nelement()).item(), arg88_1.amax().item(), arg88_1.amin().item())
        del arg261_1
        del arg88_1
        print('triton__18', 'in_ptr0', 'arg89_1', (arg89_1.sum()/arg89_1.nelement()).item(), arg89_1.amax().item(), arg89_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg262_1', (arg262_1.sum()/arg262_1.nelement()).item(), arg262_1.amax().item(), arg262_1.amin().item())
        triton__18.run(arg89_1, arg262_1, arg89_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg89_1', (arg89_1.sum()/arg89_1.nelement()).item(), arg89_1.amax().item(), arg89_1.amin().item())
        del arg262_1
        del arg89_1
        print('triton__19', 'in_ptr0', 'arg90_1', (arg90_1.sum()/arg90_1.nelement()).item(), arg90_1.amax().item(), arg90_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg263_1', (arg263_1.sum()/arg263_1.nelement()).item(), arg263_1.amax().item(), arg263_1.amin().item())
        triton__19.run(arg90_1, arg263_1, arg90_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg90_1', (arg90_1.sum()/arg90_1.nelement()).item(), arg90_1.amax().item(), arg90_1.amin().item())
        del arg263_1
        del arg90_1
        print('triton__18', 'in_ptr0', 'arg91_1', (arg91_1.sum()/arg91_1.nelement()).item(), arg91_1.amax().item(), arg91_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg264_1', (arg264_1.sum()/arg264_1.nelement()).item(), arg264_1.amax().item(), arg264_1.amin().item())
        triton__18.run(arg91_1, arg264_1, arg91_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg91_1', (arg91_1.sum()/arg91_1.nelement()).item(), arg91_1.amax().item(), arg91_1.amin().item())
        del arg264_1
        del arg91_1
        print('triton__18', 'in_ptr0', 'arg92_1', (arg92_1.sum()/arg92_1.nelement()).item(), arg92_1.amax().item(), arg92_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg265_1', (arg265_1.sum()/arg265_1.nelement()).item(), arg265_1.amax().item(), arg265_1.amin().item())
        triton__18.run(arg92_1, arg265_1, arg92_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg92_1', (arg92_1.sum()/arg92_1.nelement()).item(), arg92_1.amax().item(), arg92_1.amin().item())
        del arg265_1
        del arg92_1
        print('triton__17', 'in_ptr0', 'arg93_1', (arg93_1.sum()/arg93_1.nelement()).item(), arg93_1.amax().item(), arg93_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg266_1', (arg266_1.sum()/arg266_1.nelement()).item(), arg266_1.amax().item(), arg266_1.amin().item())
        triton__17.run(arg93_1, arg266_1, arg93_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg93_1', (arg93_1.sum()/arg93_1.nelement()).item(), arg93_1.amax().item(), arg93_1.amin().item())
        del arg266_1
        del arg93_1
        print('triton__11', 'in_ptr0', 'arg94_1', (arg94_1.sum()/arg94_1.nelement()).item(), arg94_1.amax().item(), arg94_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg267_1', (arg267_1.sum()/arg267_1.nelement()).item(), arg267_1.amax().item(), arg267_1.amin().item())
        triton__11.run(arg94_1, arg267_1, arg94_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg94_1', (arg94_1.sum()/arg94_1.nelement()).item(), arg94_1.amax().item(), arg94_1.amin().item())
        del arg267_1
        del arg94_1
        print('triton__11', 'in_ptr0', 'arg95_1', (arg95_1.sum()/arg95_1.nelement()).item(), arg95_1.amax().item(), arg95_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg268_1', (arg268_1.sum()/arg268_1.nelement()).item(), arg268_1.amax().item(), arg268_1.amin().item())
        triton__11.run(arg95_1, arg268_1, arg95_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg95_1', (arg95_1.sum()/arg95_1.nelement()).item(), arg95_1.amax().item(), arg95_1.amin().item())
        del arg268_1
        del arg95_1
        print('triton__17', 'in_ptr0', 'arg96_1', (arg96_1.sum()/arg96_1.nelement()).item(), arg96_1.amax().item(), arg96_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg269_1', (arg269_1.sum()/arg269_1.nelement()).item(), arg269_1.amax().item(), arg269_1.amin().item())
        triton__17.run(arg96_1, arg269_1, arg96_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg96_1', (arg96_1.sum()/arg96_1.nelement()).item(), arg96_1.amax().item(), arg96_1.amin().item())
        del arg269_1
        del arg96_1
        print('triton__18', 'in_ptr0', 'arg97_1', (arg97_1.sum()/arg97_1.nelement()).item(), arg97_1.amax().item(), arg97_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg270_1', (arg270_1.sum()/arg270_1.nelement()).item(), arg270_1.amax().item(), arg270_1.amin().item())
        triton__18.run(arg97_1, arg270_1, arg97_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg97_1', (arg97_1.sum()/arg97_1.nelement()).item(), arg97_1.amax().item(), arg97_1.amin().item())
        del arg270_1
        del arg97_1
        print('triton__18', 'in_ptr0', 'arg98_1', (arg98_1.sum()/arg98_1.nelement()).item(), arg98_1.amax().item(), arg98_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg271_1', (arg271_1.sum()/arg271_1.nelement()).item(), arg271_1.amax().item(), arg271_1.amin().item())
        triton__18.run(arg98_1, arg271_1, arg98_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg98_1', (arg98_1.sum()/arg98_1.nelement()).item(), arg98_1.amax().item(), arg98_1.amin().item())
        del arg271_1
        del arg98_1
        print('triton__19', 'in_ptr0', 'arg99_1', (arg99_1.sum()/arg99_1.nelement()).item(), arg99_1.amax().item(), arg99_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg272_1', (arg272_1.sum()/arg272_1.nelement()).item(), arg272_1.amax().item(), arg272_1.amin().item())
        triton__19.run(arg99_1, arg272_1, arg99_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg99_1', (arg99_1.sum()/arg99_1.nelement()).item(), arg99_1.amax().item(), arg99_1.amin().item())
        del arg272_1
        del arg99_1
        print('triton__18', 'in_ptr0', 'arg100_1', (arg100_1.sum()/arg100_1.nelement()).item(), arg100_1.amax().item(), arg100_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg273_1', (arg273_1.sum()/arg273_1.nelement()).item(), arg273_1.amax().item(), arg273_1.amin().item())
        triton__18.run(arg100_1, arg273_1, arg100_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg100_1', (arg100_1.sum()/arg100_1.nelement()).item(), arg100_1.amax().item(), arg100_1.amin().item())
        del arg100_1
        del arg273_1
        print('triton__18', 'in_ptr0', 'arg101_1', (arg101_1.sum()/arg101_1.nelement()).item(), arg101_1.amax().item(), arg101_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg274_1', (arg274_1.sum()/arg274_1.nelement()).item(), arg274_1.amax().item(), arg274_1.amin().item())
        triton__18.run(arg101_1, arg274_1, arg101_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg101_1', (arg101_1.sum()/arg101_1.nelement()).item(), arg101_1.amax().item(), arg101_1.amin().item())
        del arg101_1
        del arg274_1
        print('triton__17', 'in_ptr0', 'arg102_1', (arg102_1.sum()/arg102_1.nelement()).item(), arg102_1.amax().item(), arg102_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg275_1', (arg275_1.sum()/arg275_1.nelement()).item(), arg275_1.amax().item(), arg275_1.amin().item())
        triton__17.run(arg102_1, arg275_1, arg102_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg102_1', (arg102_1.sum()/arg102_1.nelement()).item(), arg102_1.amax().item(), arg102_1.amin().item())
        del arg102_1
        del arg275_1
        print('triton__11', 'in_ptr0', 'arg103_1', (arg103_1.sum()/arg103_1.nelement()).item(), arg103_1.amax().item(), arg103_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg276_1', (arg276_1.sum()/arg276_1.nelement()).item(), arg276_1.amax().item(), arg276_1.amin().item())
        triton__11.run(arg103_1, arg276_1, arg103_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg103_1', (arg103_1.sum()/arg103_1.nelement()).item(), arg103_1.amax().item(), arg103_1.amin().item())
        del arg103_1
        del arg276_1
        print('triton__11', 'in_ptr0', 'arg104_1', (arg104_1.sum()/arg104_1.nelement()).item(), arg104_1.amax().item(), arg104_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg277_1', (arg277_1.sum()/arg277_1.nelement()).item(), arg277_1.amax().item(), arg277_1.amin().item())
        triton__11.run(arg104_1, arg277_1, arg104_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg104_1', (arg104_1.sum()/arg104_1.nelement()).item(), arg104_1.amax().item(), arg104_1.amin().item())
        del arg104_1
        del arg277_1
        print('triton__17', 'in_ptr0', 'arg105_1', (arg105_1.sum()/arg105_1.nelement()).item(), arg105_1.amax().item(), arg105_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg278_1', (arg278_1.sum()/arg278_1.nelement()).item(), arg278_1.amax().item(), arg278_1.amin().item())
        triton__17.run(arg105_1, arg278_1, arg105_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg105_1', (arg105_1.sum()/arg105_1.nelement()).item(), arg105_1.amax().item(), arg105_1.amin().item())
        del arg105_1
        del arg278_1
        print('triton__18', 'in_ptr0', 'arg106_1', (arg106_1.sum()/arg106_1.nelement()).item(), arg106_1.amax().item(), arg106_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg279_1', (arg279_1.sum()/arg279_1.nelement()).item(), arg279_1.amax().item(), arg279_1.amin().item())
        triton__18.run(arg106_1, arg279_1, arg106_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg106_1', (arg106_1.sum()/arg106_1.nelement()).item(), arg106_1.amax().item(), arg106_1.amin().item())
        del arg106_1
        del arg279_1
        print('triton__18', 'in_ptr0', 'arg107_1', (arg107_1.sum()/arg107_1.nelement()).item(), arg107_1.amax().item(), arg107_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg280_1', (arg280_1.sum()/arg280_1.nelement()).item(), arg280_1.amax().item(), arg280_1.amin().item())
        triton__18.run(arg107_1, arg280_1, arg107_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg107_1', (arg107_1.sum()/arg107_1.nelement()).item(), arg107_1.amax().item(), arg107_1.amin().item())
        del arg107_1
        del arg280_1
        print('triton__19', 'in_ptr0', 'arg108_1', (arg108_1.sum()/arg108_1.nelement()).item(), arg108_1.amax().item(), arg108_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg281_1', (arg281_1.sum()/arg281_1.nelement()).item(), arg281_1.amax().item(), arg281_1.amin().item())
        triton__19.run(arg108_1, arg281_1, arg108_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg108_1', (arg108_1.sum()/arg108_1.nelement()).item(), arg108_1.amax().item(), arg108_1.amin().item())
        del arg108_1
        del arg281_1
        print('triton__18', 'in_ptr0', 'arg109_1', (arg109_1.sum()/arg109_1.nelement()).item(), arg109_1.amax().item(), arg109_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg282_1', (arg282_1.sum()/arg282_1.nelement()).item(), arg282_1.amax().item(), arg282_1.amin().item())
        triton__18.run(arg109_1, arg282_1, arg109_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg109_1', (arg109_1.sum()/arg109_1.nelement()).item(), arg109_1.amax().item(), arg109_1.amin().item())
        del arg109_1
        del arg282_1
        print('triton__18', 'in_ptr0', 'arg110_1', (arg110_1.sum()/arg110_1.nelement()).item(), arg110_1.amax().item(), arg110_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg283_1', (arg283_1.sum()/arg283_1.nelement()).item(), arg283_1.amax().item(), arg283_1.amin().item())
        triton__18.run(arg110_1, arg283_1, arg110_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg110_1', (arg110_1.sum()/arg110_1.nelement()).item(), arg110_1.amax().item(), arg110_1.amin().item())
        del arg110_1
        del arg283_1
        print('triton__17', 'in_ptr0', 'arg111_1', (arg111_1.sum()/arg111_1.nelement()).item(), arg111_1.amax().item(), arg111_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg284_1', (arg284_1.sum()/arg284_1.nelement()).item(), arg284_1.amax().item(), arg284_1.amin().item())
        triton__17.run(arg111_1, arg284_1, arg111_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg111_1', (arg111_1.sum()/arg111_1.nelement()).item(), arg111_1.amax().item(), arg111_1.amin().item())
        del arg111_1
        del arg284_1
        print('triton__11', 'in_ptr0', 'arg112_1', (arg112_1.sum()/arg112_1.nelement()).item(), arg112_1.amax().item(), arg112_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg285_1', (arg285_1.sum()/arg285_1.nelement()).item(), arg285_1.amax().item(), arg285_1.amin().item())
        triton__11.run(arg112_1, arg285_1, arg112_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg112_1', (arg112_1.sum()/arg112_1.nelement()).item(), arg112_1.amax().item(), arg112_1.amin().item())
        del arg112_1
        del arg285_1
        print('triton__11', 'in_ptr0', 'arg113_1', (arg113_1.sum()/arg113_1.nelement()).item(), arg113_1.amax().item(), arg113_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg286_1', (arg286_1.sum()/arg286_1.nelement()).item(), arg286_1.amax().item(), arg286_1.amin().item())
        triton__11.run(arg113_1, arg286_1, arg113_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg113_1', (arg113_1.sum()/arg113_1.nelement()).item(), arg113_1.amax().item(), arg113_1.amin().item())
        del arg113_1
        del arg286_1
        print('triton__17', 'in_ptr0', 'arg114_1', (arg114_1.sum()/arg114_1.nelement()).item(), arg114_1.amax().item(), arg114_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg287_1', (arg287_1.sum()/arg287_1.nelement()).item(), arg287_1.amax().item(), arg287_1.amin().item())
        triton__17.run(arg114_1, arg287_1, arg114_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg114_1', (arg114_1.sum()/arg114_1.nelement()).item(), arg114_1.amax().item(), arg114_1.amin().item())
        del arg114_1
        del arg287_1
        print('triton__18', 'in_ptr0', 'arg115_1', (arg115_1.sum()/arg115_1.nelement()).item(), arg115_1.amax().item(), arg115_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg288_1', (arg288_1.sum()/arg288_1.nelement()).item(), arg288_1.amax().item(), arg288_1.amin().item())
        triton__18.run(arg115_1, arg288_1, arg115_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg115_1', (arg115_1.sum()/arg115_1.nelement()).item(), arg115_1.amax().item(), arg115_1.amin().item())
        del arg115_1
        del arg288_1
        print('triton__18', 'in_ptr0', 'arg116_1', (arg116_1.sum()/arg116_1.nelement()).item(), arg116_1.amax().item(), arg116_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg289_1', (arg289_1.sum()/arg289_1.nelement()).item(), arg289_1.amax().item(), arg289_1.amin().item())
        triton__18.run(arg116_1, arg289_1, arg116_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg116_1', (arg116_1.sum()/arg116_1.nelement()).item(), arg116_1.amax().item(), arg116_1.amin().item())
        del arg116_1
        del arg289_1
        print('triton__19', 'in_ptr0', 'arg117_1', (arg117_1.sum()/arg117_1.nelement()).item(), arg117_1.amax().item(), arg117_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg290_1', (arg290_1.sum()/arg290_1.nelement()).item(), arg290_1.amax().item(), arg290_1.amin().item())
        triton__19.run(arg117_1, arg290_1, arg117_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg117_1', (arg117_1.sum()/arg117_1.nelement()).item(), arg117_1.amax().item(), arg117_1.amin().item())
        del arg117_1
        del arg290_1
        print('triton__18', 'in_ptr0', 'arg118_1', (arg118_1.sum()/arg118_1.nelement()).item(), arg118_1.amax().item(), arg118_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg291_1', (arg291_1.sum()/arg291_1.nelement()).item(), arg291_1.amax().item(), arg291_1.amin().item())
        triton__18.run(arg118_1, arg291_1, arg118_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg118_1', (arg118_1.sum()/arg118_1.nelement()).item(), arg118_1.amax().item(), arg118_1.amin().item())
        del arg118_1
        del arg291_1
        print('triton__18', 'in_ptr0', 'arg119_1', (arg119_1.sum()/arg119_1.nelement()).item(), arg119_1.amax().item(), arg119_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg292_1', (arg292_1.sum()/arg292_1.nelement()).item(), arg292_1.amax().item(), arg292_1.amin().item())
        triton__18.run(arg119_1, arg292_1, arg119_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg119_1', (arg119_1.sum()/arg119_1.nelement()).item(), arg119_1.amax().item(), arg119_1.amin().item())
        del arg119_1
        del arg292_1
        print('triton__17', 'in_ptr0', 'arg120_1', (arg120_1.sum()/arg120_1.nelement()).item(), arg120_1.amax().item(), arg120_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg293_1', (arg293_1.sum()/arg293_1.nelement()).item(), arg293_1.amax().item(), arg293_1.amin().item())
        triton__17.run(arg120_1, arg293_1, arg120_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg120_1', (arg120_1.sum()/arg120_1.nelement()).item(), arg120_1.amax().item(), arg120_1.amin().item())
        del arg120_1
        del arg293_1
        print('triton__11', 'in_ptr0', 'arg121_1', (arg121_1.sum()/arg121_1.nelement()).item(), arg121_1.amax().item(), arg121_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg294_1', (arg294_1.sum()/arg294_1.nelement()).item(), arg294_1.amax().item(), arg294_1.amin().item())
        triton__11.run(arg121_1, arg294_1, arg121_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg121_1', (arg121_1.sum()/arg121_1.nelement()).item(), arg121_1.amax().item(), arg121_1.amin().item())
        del arg121_1
        del arg294_1
        print('triton__11', 'in_ptr0', 'arg122_1', (arg122_1.sum()/arg122_1.nelement()).item(), arg122_1.amax().item(), arg122_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg295_1', (arg295_1.sum()/arg295_1.nelement()).item(), arg295_1.amax().item(), arg295_1.amin().item())
        triton__11.run(arg122_1, arg295_1, arg122_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg122_1', (arg122_1.sum()/arg122_1.nelement()).item(), arg122_1.amax().item(), arg122_1.amin().item())
        del arg122_1
        del arg295_1
        print('triton__17', 'in_ptr0', 'arg123_1', (arg123_1.sum()/arg123_1.nelement()).item(), arg123_1.amax().item(), arg123_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg296_1', (arg296_1.sum()/arg296_1.nelement()).item(), arg296_1.amax().item(), arg296_1.amin().item())
        triton__17.run(arg123_1, arg296_1, arg123_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg123_1', (arg123_1.sum()/arg123_1.nelement()).item(), arg123_1.amax().item(), arg123_1.amin().item())
        del arg123_1
        del arg296_1
        print('triton__18', 'in_ptr0', 'arg124_1', (arg124_1.sum()/arg124_1.nelement()).item(), arg124_1.amax().item(), arg124_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg297_1', (arg297_1.sum()/arg297_1.nelement()).item(), arg297_1.amax().item(), arg297_1.amin().item())
        triton__18.run(arg124_1, arg297_1, arg124_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg124_1', (arg124_1.sum()/arg124_1.nelement()).item(), arg124_1.amax().item(), arg124_1.amin().item())
        del arg124_1
        del arg297_1
        print('triton__18', 'in_ptr0', 'arg125_1', (arg125_1.sum()/arg125_1.nelement()).item(), arg125_1.amax().item(), arg125_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg298_1', (arg298_1.sum()/arg298_1.nelement()).item(), arg298_1.amax().item(), arg298_1.amin().item())
        triton__18.run(arg125_1, arg298_1, arg125_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg125_1', (arg125_1.sum()/arg125_1.nelement()).item(), arg125_1.amax().item(), arg125_1.amin().item())
        del arg125_1
        del arg298_1
        print('triton__19', 'in_ptr0', 'arg126_1', (arg126_1.sum()/arg126_1.nelement()).item(), arg126_1.amax().item(), arg126_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg299_1', (arg299_1.sum()/arg299_1.nelement()).item(), arg299_1.amax().item(), arg299_1.amin().item())
        triton__19.run(arg126_1, arg299_1, arg126_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg126_1', (arg126_1.sum()/arg126_1.nelement()).item(), arg126_1.amax().item(), arg126_1.amin().item())
        del arg126_1
        del arg299_1
        print('triton__18', 'in_ptr0', 'arg127_1', (arg127_1.sum()/arg127_1.nelement()).item(), arg127_1.amax().item(), arg127_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg300_1', (arg300_1.sum()/arg300_1.nelement()).item(), arg300_1.amax().item(), arg300_1.amin().item())
        triton__18.run(arg127_1, arg300_1, arg127_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg127_1', (arg127_1.sum()/arg127_1.nelement()).item(), arg127_1.amax().item(), arg127_1.amin().item())
        del arg127_1
        del arg300_1
        print('triton__18', 'in_ptr0', 'arg128_1', (arg128_1.sum()/arg128_1.nelement()).item(), arg128_1.amax().item(), arg128_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg301_1', (arg301_1.sum()/arg301_1.nelement()).item(), arg301_1.amax().item(), arg301_1.amin().item())
        triton__18.run(arg128_1, arg301_1, arg128_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg128_1', (arg128_1.sum()/arg128_1.nelement()).item(), arg128_1.amax().item(), arg128_1.amin().item())
        del arg128_1
        del arg301_1
        print('triton__17', 'in_ptr0', 'arg129_1', (arg129_1.sum()/arg129_1.nelement()).item(), arg129_1.amax().item(), arg129_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg302_1', (arg302_1.sum()/arg302_1.nelement()).item(), arg302_1.amax().item(), arg302_1.amin().item())
        triton__17.run(arg129_1, arg302_1, arg129_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg129_1', (arg129_1.sum()/arg129_1.nelement()).item(), arg129_1.amax().item(), arg129_1.amin().item())
        del arg129_1
        del arg302_1
        print('triton__11', 'in_ptr0', 'arg130_1', (arg130_1.sum()/arg130_1.nelement()).item(), arg130_1.amax().item(), arg130_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg303_1', (arg303_1.sum()/arg303_1.nelement()).item(), arg303_1.amax().item(), arg303_1.amin().item())
        triton__11.run(arg130_1, arg303_1, arg130_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg130_1', (arg130_1.sum()/arg130_1.nelement()).item(), arg130_1.amax().item(), arg130_1.amin().item())
        del arg130_1
        del arg303_1
        print('triton__11', 'in_ptr0', 'arg131_1', (arg131_1.sum()/arg131_1.nelement()).item(), arg131_1.amax().item(), arg131_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg304_1', (arg304_1.sum()/arg304_1.nelement()).item(), arg304_1.amax().item(), arg304_1.amin().item())
        triton__11.run(arg131_1, arg304_1, arg131_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg131_1', (arg131_1.sum()/arg131_1.nelement()).item(), arg131_1.amax().item(), arg131_1.amin().item())
        del arg131_1
        del arg304_1
        print('triton__17', 'in_ptr0', 'arg132_1', (arg132_1.sum()/arg132_1.nelement()).item(), arg132_1.amax().item(), arg132_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg305_1', (arg305_1.sum()/arg305_1.nelement()).item(), arg305_1.amax().item(), arg305_1.amin().item())
        triton__17.run(arg132_1, arg305_1, arg132_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg132_1', (arg132_1.sum()/arg132_1.nelement()).item(), arg132_1.amax().item(), arg132_1.amin().item())
        del arg132_1
        del arg305_1
        print('triton__18', 'in_ptr0', 'arg133_1', (arg133_1.sum()/arg133_1.nelement()).item(), arg133_1.amax().item(), arg133_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg306_1', (arg306_1.sum()/arg306_1.nelement()).item(), arg306_1.amax().item(), arg306_1.amin().item())
        triton__18.run(arg133_1, arg306_1, arg133_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg133_1', (arg133_1.sum()/arg133_1.nelement()).item(), arg133_1.amax().item(), arg133_1.amin().item())
        del arg133_1
        del arg306_1
        print('triton__18', 'in_ptr0', 'arg134_1', (arg134_1.sum()/arg134_1.nelement()).item(), arg134_1.amax().item(), arg134_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg307_1', (arg307_1.sum()/arg307_1.nelement()).item(), arg307_1.amax().item(), arg307_1.amin().item())
        triton__18.run(arg134_1, arg307_1, arg134_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg134_1', (arg134_1.sum()/arg134_1.nelement()).item(), arg134_1.amax().item(), arg134_1.amin().item())
        del arg134_1
        del arg307_1
        print('triton__19', 'in_ptr0', 'arg135_1', (arg135_1.sum()/arg135_1.nelement()).item(), arg135_1.amax().item(), arg135_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg308_1', (arg308_1.sum()/arg308_1.nelement()).item(), arg308_1.amax().item(), arg308_1.amin().item())
        triton__19.run(arg135_1, arg308_1, arg135_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg135_1', (arg135_1.sum()/arg135_1.nelement()).item(), arg135_1.amax().item(), arg135_1.amin().item())
        del arg135_1
        del arg308_1
        print('triton__18', 'in_ptr0', 'arg136_1', (arg136_1.sum()/arg136_1.nelement()).item(), arg136_1.amax().item(), arg136_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg309_1', (arg309_1.sum()/arg309_1.nelement()).item(), arg309_1.amax().item(), arg309_1.amin().item())
        triton__18.run(arg136_1, arg309_1, arg136_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg136_1', (arg136_1.sum()/arg136_1.nelement()).item(), arg136_1.amax().item(), arg136_1.amin().item())
        del arg136_1
        del arg309_1
        print('triton__18', 'in_ptr0', 'arg137_1', (arg137_1.sum()/arg137_1.nelement()).item(), arg137_1.amax().item(), arg137_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg310_1', (arg310_1.sum()/arg310_1.nelement()).item(), arg310_1.amax().item(), arg310_1.amin().item())
        triton__18.run(arg137_1, arg310_1, arg137_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg137_1', (arg137_1.sum()/arg137_1.nelement()).item(), arg137_1.amax().item(), arg137_1.amin().item())
        del arg137_1
        del arg310_1
        print('triton__17', 'in_ptr0', 'arg138_1', (arg138_1.sum()/arg138_1.nelement()).item(), arg138_1.amax().item(), arg138_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg311_1', (arg311_1.sum()/arg311_1.nelement()).item(), arg311_1.amax().item(), arg311_1.amin().item())
        triton__17.run(arg138_1, arg311_1, arg138_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg138_1', (arg138_1.sum()/arg138_1.nelement()).item(), arg138_1.amax().item(), arg138_1.amin().item())
        del arg138_1
        del arg311_1
        print('triton__11', 'in_ptr0', 'arg139_1', (arg139_1.sum()/arg139_1.nelement()).item(), arg139_1.amax().item(), arg139_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg312_1', (arg312_1.sum()/arg312_1.nelement()).item(), arg312_1.amax().item(), arg312_1.amin().item())
        triton__11.run(arg139_1, arg312_1, arg139_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg139_1', (arg139_1.sum()/arg139_1.nelement()).item(), arg139_1.amax().item(), arg139_1.amin().item())
        del arg139_1
        del arg312_1
        print('triton__11', 'in_ptr0', 'arg140_1', (arg140_1.sum()/arg140_1.nelement()).item(), arg140_1.amax().item(), arg140_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg313_1', (arg313_1.sum()/arg313_1.nelement()).item(), arg313_1.amax().item(), arg313_1.amin().item())
        triton__11.run(arg140_1, arg313_1, arg140_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg140_1', (arg140_1.sum()/arg140_1.nelement()).item(), arg140_1.amax().item(), arg140_1.amin().item())
        del arg140_1
        del arg313_1
        print('triton__17', 'in_ptr0', 'arg141_1', (arg141_1.sum()/arg141_1.nelement()).item(), arg141_1.amax().item(), arg141_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg314_1', (arg314_1.sum()/arg314_1.nelement()).item(), arg314_1.amax().item(), arg314_1.amin().item())
        triton__17.run(arg141_1, arg314_1, arg141_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg141_1', (arg141_1.sum()/arg141_1.nelement()).item(), arg141_1.amax().item(), arg141_1.amin().item())
        del arg141_1
        del arg314_1
        print('triton__18', 'in_ptr0', 'arg142_1', (arg142_1.sum()/arg142_1.nelement()).item(), arg142_1.amax().item(), arg142_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg315_1', (arg315_1.sum()/arg315_1.nelement()).item(), arg315_1.amax().item(), arg315_1.amin().item())
        triton__18.run(arg142_1, arg315_1, arg142_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg142_1', (arg142_1.sum()/arg142_1.nelement()).item(), arg142_1.amax().item(), arg142_1.amin().item())
        del arg142_1
        del arg315_1
        print('triton__18', 'in_ptr0', 'arg143_1', (arg143_1.sum()/arg143_1.nelement()).item(), arg143_1.amax().item(), arg143_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg316_1', (arg316_1.sum()/arg316_1.nelement()).item(), arg316_1.amax().item(), arg316_1.amin().item())
        triton__18.run(arg143_1, arg316_1, arg143_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg143_1', (arg143_1.sum()/arg143_1.nelement()).item(), arg143_1.amax().item(), arg143_1.amin().item())
        del arg143_1
        del arg316_1
        print('triton__19', 'in_ptr0', 'arg144_1', (arg144_1.sum()/arg144_1.nelement()).item(), arg144_1.amax().item(), arg144_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg317_1', (arg317_1.sum()/arg317_1.nelement()).item(), arg317_1.amax().item(), arg317_1.amin().item())
        triton__19.run(arg144_1, arg317_1, arg144_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg144_1', (arg144_1.sum()/arg144_1.nelement()).item(), arg144_1.amax().item(), arg144_1.amin().item())
        del arg144_1
        del arg317_1
        print('triton__18', 'in_ptr0', 'arg145_1', (arg145_1.sum()/arg145_1.nelement()).item(), arg145_1.amax().item(), arg145_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg318_1', (arg318_1.sum()/arg318_1.nelement()).item(), arg318_1.amax().item(), arg318_1.amin().item())
        triton__18.run(arg145_1, arg318_1, arg145_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg145_1', (arg145_1.sum()/arg145_1.nelement()).item(), arg145_1.amax().item(), arg145_1.amin().item())
        del arg145_1
        del arg318_1
        print('triton__18', 'in_ptr0', 'arg146_1', (arg146_1.sum()/arg146_1.nelement()).item(), arg146_1.amax().item(), arg146_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg319_1', (arg319_1.sum()/arg319_1.nelement()).item(), arg319_1.amax().item(), arg319_1.amin().item())
        triton__18.run(arg146_1, arg319_1, arg146_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg146_1', (arg146_1.sum()/arg146_1.nelement()).item(), arg146_1.amax().item(), arg146_1.amin().item())
        del arg146_1
        del arg319_1
        print('triton__17', 'in_ptr0', 'arg147_1', (arg147_1.sum()/arg147_1.nelement()).item(), arg147_1.amax().item(), arg147_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg320_1', (arg320_1.sum()/arg320_1.nelement()).item(), arg320_1.amax().item(), arg320_1.amin().item())
        triton__17.run(arg147_1, arg320_1, arg147_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg147_1', (arg147_1.sum()/arg147_1.nelement()).item(), arg147_1.amax().item(), arg147_1.amin().item())
        del arg147_1
        del arg320_1
        print('triton__11', 'in_ptr0', 'arg148_1', (arg148_1.sum()/arg148_1.nelement()).item(), arg148_1.amax().item(), arg148_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg321_1', (arg321_1.sum()/arg321_1.nelement()).item(), arg321_1.amax().item(), arg321_1.amin().item())
        triton__11.run(arg148_1, arg321_1, arg148_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg148_1', (arg148_1.sum()/arg148_1.nelement()).item(), arg148_1.amax().item(), arg148_1.amin().item())
        del arg148_1
        del arg321_1
        print('triton__11', 'in_ptr0', 'arg149_1', (arg149_1.sum()/arg149_1.nelement()).item(), arg149_1.amax().item(), arg149_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg322_1', (arg322_1.sum()/arg322_1.nelement()).item(), arg322_1.amax().item(), arg322_1.amin().item())
        triton__11.run(arg149_1, arg322_1, arg149_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg149_1', (arg149_1.sum()/arg149_1.nelement()).item(), arg149_1.amax().item(), arg149_1.amin().item())
        del arg149_1
        del arg322_1
        print('triton__17', 'in_ptr0', 'arg150_1', (arg150_1.sum()/arg150_1.nelement()).item(), arg150_1.amax().item(), arg150_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg323_1', (arg323_1.sum()/arg323_1.nelement()).item(), arg323_1.amax().item(), arg323_1.amin().item())
        triton__17.run(arg150_1, arg323_1, arg150_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg150_1', (arg150_1.sum()/arg150_1.nelement()).item(), arg150_1.amax().item(), arg150_1.amin().item())
        del arg150_1
        del arg323_1
        print('triton__18', 'in_ptr0', 'arg151_1', (arg151_1.sum()/arg151_1.nelement()).item(), arg151_1.amax().item(), arg151_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg324_1', (arg324_1.sum()/arg324_1.nelement()).item(), arg324_1.amax().item(), arg324_1.amin().item())
        triton__18.run(arg151_1, arg324_1, arg151_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg151_1', (arg151_1.sum()/arg151_1.nelement()).item(), arg151_1.amax().item(), arg151_1.amin().item())
        del arg151_1
        del arg324_1
        print('triton__18', 'in_ptr0', 'arg152_1', (arg152_1.sum()/arg152_1.nelement()).item(), arg152_1.amax().item(), arg152_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg325_1', (arg325_1.sum()/arg325_1.nelement()).item(), arg325_1.amax().item(), arg325_1.amin().item())
        triton__18.run(arg152_1, arg325_1, arg152_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg152_1', (arg152_1.sum()/arg152_1.nelement()).item(), arg152_1.amax().item(), arg152_1.amin().item())
        del arg152_1
        del arg325_1
        print('triton__19', 'in_ptr0', 'arg153_1', (arg153_1.sum()/arg153_1.nelement()).item(), arg153_1.amax().item(), arg153_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg326_1', (arg326_1.sum()/arg326_1.nelement()).item(), arg326_1.amax().item(), arg326_1.amin().item())
        triton__19.run(arg153_1, arg326_1, arg153_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg153_1', (arg153_1.sum()/arg153_1.nelement()).item(), arg153_1.amax().item(), arg153_1.amin().item())
        del arg153_1
        del arg326_1
        print('triton__18', 'in_ptr0', 'arg154_1', (arg154_1.sum()/arg154_1.nelement()).item(), arg154_1.amax().item(), arg154_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg327_1', (arg327_1.sum()/arg327_1.nelement()).item(), arg327_1.amax().item(), arg327_1.amin().item())
        triton__18.run(arg154_1, arg327_1, arg154_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg154_1', (arg154_1.sum()/arg154_1.nelement()).item(), arg154_1.amax().item(), arg154_1.amin().item())
        del arg154_1
        del arg327_1
        print('triton__18', 'in_ptr0', 'arg155_1', (arg155_1.sum()/arg155_1.nelement()).item(), arg155_1.amax().item(), arg155_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg328_1', (arg328_1.sum()/arg328_1.nelement()).item(), arg328_1.amax().item(), arg328_1.amin().item())
        triton__18.run(arg155_1, arg328_1, arg155_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg155_1', (arg155_1.sum()/arg155_1.nelement()).item(), arg155_1.amax().item(), arg155_1.amin().item())
        del arg155_1
        del arg328_1
        print('triton__17', 'in_ptr0', 'arg156_1', (arg156_1.sum()/arg156_1.nelement()).item(), arg156_1.amax().item(), arg156_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg329_1', (arg329_1.sum()/arg329_1.nelement()).item(), arg329_1.amax().item(), arg329_1.amin().item())
        triton__17.run(arg156_1, arg329_1, arg156_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg156_1', (arg156_1.sum()/arg156_1.nelement()).item(), arg156_1.amax().item(), arg156_1.amin().item())
        del arg156_1
        del arg329_1
        print('triton__11', 'in_ptr0', 'arg157_1', (arg157_1.sum()/arg157_1.nelement()).item(), arg157_1.amax().item(), arg157_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg330_1', (arg330_1.sum()/arg330_1.nelement()).item(), arg330_1.amax().item(), arg330_1.amin().item())
        triton__11.run(arg157_1, arg330_1, arg157_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg157_1', (arg157_1.sum()/arg157_1.nelement()).item(), arg157_1.amax().item(), arg157_1.amin().item())
        del arg157_1
        del arg330_1
        print('triton__11', 'in_ptr0', 'arg158_1', (arg158_1.sum()/arg158_1.nelement()).item(), arg158_1.amax().item(), arg158_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg331_1', (arg331_1.sum()/arg331_1.nelement()).item(), arg331_1.amax().item(), arg331_1.amin().item())
        triton__11.run(arg158_1, arg331_1, arg158_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg158_1', (arg158_1.sum()/arg158_1.nelement()).item(), arg158_1.amax().item(), arg158_1.amin().item())
        del arg158_1
        del arg331_1
        print('triton__17', 'in_ptr0', 'arg159_1', (arg159_1.sum()/arg159_1.nelement()).item(), arg159_1.amax().item(), arg159_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg332_1', (arg332_1.sum()/arg332_1.nelement()).item(), arg332_1.amax().item(), arg332_1.amin().item())
        triton__17.run(arg159_1, arg332_1, arg159_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg159_1', (arg159_1.sum()/arg159_1.nelement()).item(), arg159_1.amax().item(), arg159_1.amin().item())
        del arg159_1
        del arg332_1
        print('triton__18', 'in_ptr0', 'arg160_1', (arg160_1.sum()/arg160_1.nelement()).item(), arg160_1.amax().item(), arg160_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg333_1', (arg333_1.sum()/arg333_1.nelement()).item(), arg333_1.amax().item(), arg333_1.amin().item())
        triton__18.run(arg160_1, arg333_1, arg160_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg160_1', (arg160_1.sum()/arg160_1.nelement()).item(), arg160_1.amax().item(), arg160_1.amin().item())
        del arg160_1
        del arg333_1
        print('triton__18', 'in_ptr0', 'arg161_1', (arg161_1.sum()/arg161_1.nelement()).item(), arg161_1.amax().item(), arg161_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg334_1', (arg334_1.sum()/arg334_1.nelement()).item(), arg334_1.amax().item(), arg334_1.amin().item())
        triton__18.run(arg161_1, arg334_1, arg161_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg161_1', (arg161_1.sum()/arg161_1.nelement()).item(), arg161_1.amax().item(), arg161_1.amin().item())
        del arg161_1
        del arg334_1
        print('triton__19', 'in_ptr0', 'arg162_1', (arg162_1.sum()/arg162_1.nelement()).item(), arg162_1.amax().item(), arg162_1.amin().item())
        print('triton__19', 'in_ptr1', 'arg335_1', (arg335_1.sum()/arg335_1.nelement()).item(), arg335_1.amax().item(), arg335_1.amin().item())
        triton__19.run(arg162_1, arg335_1, arg162_1, 17280, grid=grid(17280), stream=stream0)
        print('triton__19', 'out_ptr0', 'arg162_1', (arg162_1.sum()/arg162_1.nelement()).item(), arg162_1.amax().item(), arg162_1.amin().item())
        del arg162_1
        del arg335_1
        print('triton__18', 'in_ptr0', 'arg163_1', (arg163_1.sum()/arg163_1.nelement()).item(), arg163_1.amax().item(), arg163_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg336_1', (arg336_1.sum()/arg336_1.nelement()).item(), arg336_1.amax().item(), arg336_1.amin().item())
        triton__18.run(arg163_1, arg336_1, arg163_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg163_1', (arg163_1.sum()/arg163_1.nelement()).item(), arg163_1.amax().item(), arg163_1.amin().item())
        del arg163_1
        del arg336_1
        print('triton__18', 'in_ptr0', 'arg164_1', (arg164_1.sum()/arg164_1.nelement()).item(), arg164_1.amax().item(), arg164_1.amin().item())
        print('triton__18', 'in_ptr1', 'arg337_1', (arg337_1.sum()/arg337_1.nelement()).item(), arg337_1.amax().item(), arg337_1.amin().item())
        triton__18.run(arg164_1, arg337_1, arg164_1, 1920, grid=grid(1920), stream=stream0)
        print('triton__18', 'out_ptr0', 'arg164_1', (arg164_1.sum()/arg164_1.nelement()).item(), arg164_1.amax().item(), arg164_1.amin().item())
        del arg164_1
        del arg337_1
        print('triton__17', 'in_ptr0', 'arg165_1', (arg165_1.sum()/arg165_1.nelement()).item(), arg165_1.amax().item(), arg165_1.amin().item())
        print('triton__17', 'in_ptr1', 'arg338_1', (arg338_1.sum()/arg338_1.nelement()).item(), arg338_1.amax().item(), arg338_1.amin().item())
        triton__17.run(arg165_1, arg338_1, arg165_1, 1228800, grid=grid(1228800), stream=stream0)
        print('triton__17', 'out_ptr0', 'arg165_1', (arg165_1.sum()/arg165_1.nelement()).item(), arg165_1.amax().item(), arg165_1.amin().item())
        del arg165_1
        del arg338_1
        print('triton__11', 'in_ptr0', 'arg166_1', (arg166_1.sum()/arg166_1.nelement()).item(), arg166_1.amax().item(), arg166_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg339_1', (arg339_1.sum()/arg339_1.nelement()).item(), arg339_1.amax().item(), arg339_1.amin().item())
        triton__11.run(arg166_1, arg339_1, arg166_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg166_1', (arg166_1.sum()/arg166_1.nelement()).item(), arg166_1.amax().item(), arg166_1.amin().item())
        del arg166_1
        del arg339_1
        print('triton__11', 'in_ptr0', 'arg167_1', (arg167_1.sum()/arg167_1.nelement()).item(), arg167_1.amax().item(), arg167_1.amin().item())
        print('triton__11', 'in_ptr1', 'arg340_1', (arg340_1.sum()/arg340_1.nelement()).item(), arg340_1.amax().item(), arg340_1.amin().item())
        triton__11.run(arg167_1, arg340_1, arg167_1, 640, grid=grid(640), stream=stream0)
        print('triton__11', 'out_ptr0', 'arg167_1', (arg167_1.sum()/arg167_1.nelement()).item(), arg167_1.amax().item(), arg167_1.amin().item())
        del arg167_1
        del arg340_1
        print('triton__20', 'in_ptr0', 'arg168_1', (arg168_1.sum()/arg168_1.nelement()).item(), arg168_1.amax().item(), arg168_1.amin().item())
        print('triton__20', 'in_ptr1', 'arg341_1', (arg341_1.sum()/arg341_1.nelement()).item(), arg341_1.amax().item(), arg341_1.amin().item())
        triton__20.run(arg168_1, arg341_1, arg168_1, 1638400, grid=grid(1638400), stream=stream0)
        print('triton__20', 'out_ptr0', 'arg168_1', (arg168_1.sum()/arg168_1.nelement()).item(), arg168_1.amax().item(), arg168_1.amin().item())
        del arg168_1
        del arg341_1
        print('triton__21', 'in_ptr0', 'arg169_1', (arg169_1.sum()/arg169_1.nelement()).item(), arg169_1.amax().item(), arg169_1.amin().item())
        print('triton__21', 'in_ptr1', 'arg342_1', (arg342_1.sum()/arg342_1.nelement()).item(), arg342_1.amax().item(), arg342_1.amin().item())
        triton__21.run(arg169_1, arg342_1, arg169_1, 2560, grid=grid(2560), stream=stream0)
        print('triton__21', 'out_ptr0', 'arg169_1', (arg169_1.sum()/arg169_1.nelement()).item(), arg169_1.amax().item(), arg169_1.amin().item())
        del arg169_1
        del arg342_1
        print('triton__21', 'in_ptr0', 'arg170_1', (arg170_1.sum()/arg170_1.nelement()).item(), arg170_1.amax().item(), arg170_1.amin().item())
        print('triton__21', 'in_ptr1', 'arg343_1', (arg343_1.sum()/arg343_1.nelement()).item(), arg343_1.amax().item(), arg343_1.amin().item())
        triton__21.run(arg170_1, arg343_1, arg170_1, 2560, grid=grid(2560), stream=stream0)
        print('triton__21', 'out_ptr0', 'arg170_1', (arg170_1.sum()/arg170_1.nelement()).item(), arg170_1.amax().item(), arg170_1.amin().item())
        del arg170_1
        del arg343_1
        print('triton__22', 'in_ptr0', 'arg171_1', (arg171_1.sum()/arg171_1.nelement()).item(), arg171_1.amax().item(), arg171_1.amin().item())
        print('triton__22', 'in_ptr1', 'arg344_1', (arg344_1.sum()/arg344_1.nelement()).item(), arg344_1.amax().item(), arg344_1.amin().item())
        triton__22.run(arg171_1, arg344_1, arg171_1, 2560000, grid=grid(2560000), stream=stream0)
        print('triton__22', 'out_ptr0', 'arg171_1', (arg171_1.sum()/arg171_1.nelement()).item(), arg171_1.amax().item(), arg171_1.amin().item())
        del arg171_1
        del arg344_1
        print('triton__23', 'in_ptr0', 'arg172_1', (arg172_1.sum()/arg172_1.nelement()).item(), arg172_1.amax().item(), arg172_1.amin().item())
        print('triton__23', 'in_ptr1', 'arg345_1', (arg345_1.sum()/arg345_1.nelement()).item(), arg345_1.amax().item(), arg345_1.amin().item())
        triton__23.run(arg172_1, arg345_1, arg172_1, 1000, grid=grid(1000), stream=stream0)
        print('triton__23', 'out_ptr0', 'arg172_1', (arg172_1.sum()/arg172_1.nelement()).item(), arg172_1.amax().item(), arg172_1.amin().item())
        del arg172_1
        del arg345_1
        return ()


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1]))
