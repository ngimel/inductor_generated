
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

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840000
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

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
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

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton__9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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

@pointwise(size_hints=[2], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton__10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        stream0 = get_cuda_stream(0)
        triton__0.run(arg0_1, arg25_1, arg0_1, 3840000, grid=grid(3840000), stream=stream0)
        del arg0_1
        del arg25_1
        triton__1.run(arg1_1, arg26_1, arg1_1, 65536, grid=grid(65536), stream=stream0)
        del arg1_1
        del arg26_1
        triton__2.run(arg2_1, arg27_1, arg2_1, 256, grid=grid(256), stream=stream0)
        del arg27_1
        del arg2_1
        triton__3.run(arg3_1, arg28_1, arg3_1, 128, grid=grid(128), stream=stream0)
        del arg28_1
        del arg3_1
        triton__3.run(arg4_1, arg29_1, arg4_1, 128, grid=grid(128), stream=stream0)
        del arg29_1
        del arg4_1
        triton__4.run(arg5_1, arg30_1, arg5_1, 524288, grid=grid(524288), stream=stream0)
        del arg30_1
        del arg5_1
        triton__5.run(arg6_1, arg31_1, arg6_1, 4096, grid=grid(4096), stream=stream0)
        del arg31_1
        del arg6_1
        triton__5.run(arg7_1, arg32_1, arg7_1, 4096, grid=grid(4096), stream=stream0)
        del arg32_1
        del arg7_1
        triton__5.run(arg8_1, arg33_1, arg8_1, 4096, grid=grid(4096), stream=stream0)
        del arg33_1
        del arg8_1
        triton__6.run(arg9_1, arg34_1, arg9_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg34_1
        del arg9_1
        triton__5.run(arg10_1, arg35_1, arg10_1, 4096, grid=grid(4096), stream=stream0)
        del arg10_1
        del arg35_1
        triton__6.run(arg11_1, arg36_1, arg11_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg11_1
        del arg36_1
        triton__5.run(arg12_1, arg37_1, arg12_1, 4096, grid=grid(4096), stream=stream0)
        del arg12_1
        del arg37_1
        triton__6.run(arg13_1, arg38_1, arg13_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg13_1
        del arg38_1
        triton__5.run(arg14_1, arg39_1, arg14_1, 4096, grid=grid(4096), stream=stream0)
        del arg14_1
        del arg39_1
        triton__6.run(arg15_1, arg40_1, arg15_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg15_1
        del arg40_1
        triton__5.run(arg16_1, arg41_1, arg16_1, 4096, grid=grid(4096), stream=stream0)
        del arg16_1
        del arg41_1
        triton__5.run(arg17_1, arg42_1, arg17_1, 4096, grid=grid(4096), stream=stream0)
        del arg17_1
        del arg42_1
        triton__5.run(arg18_1, arg43_1, arg18_1, 4096, grid=grid(4096), stream=stream0)
        del arg18_1
        del arg43_1
        triton__7.run(arg19_1, arg44_1, arg19_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg19_1
        del arg44_1
        triton__8.run(arg20_1, arg45_1, arg20_1, 16384, grid=grid(16384), stream=stream0)
        del arg20_1
        del arg45_1
        triton__7.run(arg21_1, arg46_1, arg21_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg21_1
        del arg46_1
        triton__5.run(arg22_1, arg47_1, arg22_1, 4096, grid=grid(4096), stream=stream0)
        del arg22_1
        del arg47_1
        triton__9.run(arg23_1, arg48_1, arg23_1, 8192, grid=grid(8192), stream=stream0)
        del arg23_1
        del arg48_1
        triton__10.run(arg24_1, arg49_1, arg24_1, 2, grid=grid(2), stream=stream0)
        del arg24_1
        del arg49_1
        return ()


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((2, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1]))
