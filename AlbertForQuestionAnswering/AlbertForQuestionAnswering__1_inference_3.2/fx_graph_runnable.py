import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x88X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\r\x00\x00\x00torch._decompq\x18X\r\x00\x00\x00torch.testingq\x19X\x0b\x00\x00\x00torch._refsq\x1aX\x13\x00\x00\x00torch.distributionsq\x1bX\x0c\x00\x00\x00torch._primsq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x18\x00\x00\x00error_on_nested_fx_traceq&\x88X\t\x00\x00\x00allow_rnnq\'\x89X\x08\x00\x00\x00base_dirq(X\x1c\x00\x00\x00/scratch/ngimel/work/pytorchq)X\x0e\x00\x00\x00debug_dir_rootq*X0\x00\x00\x00/scratch/ngimel/work/pytorch/torch_compile_debugq+X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq,\x89X\x13\x00\x00\x00_save_config_ignoreq-h\r]q.(X\x0b\x00\x00\x00repro_levelq/X\x0b\x00\x00\x00repro_afterq0X!\x00\x00\x00skipfiles_inline_module_allowlistq1X\x12\x00\x00\x00constant_functionsq2e\x85q3Rq4u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x17\x00\x00\x00realize_reads_thresholdq\x10K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x11M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x12K\x08X\x0f\x00\x00\x00fallback_randomq\x13\x88X\x12\x00\x00\x00implicit_fallbacksq\x14\x88X\r\x00\x00\x00prefuse_nodesq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x0f\x00\x00\x00compile_threadsq\x1bK X\x13\x00\x00\x00kernel_name_max_opsq\x1cK\nX\r\x00\x00\x00shape_paddingq\x1d\x89X\x0e\x00\x00\x00permute_fusionq\x1e\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq\x1f\x89X\x0b\x00\x00\x00cpp.threadsq J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq!\x89X\x0b\x00\x00\x00cpp.simdlenq"NX\x12\x00\x00\x00cpp.min_chunk_sizeq#M\x00\x10X\x07\x00\x00\x00cpp.cxxq$NX\x03\x00\x00\x00g++q%\x86q&X\x19\x00\x00\x00cpp.enable_kernel_profileq\'\x89X\x12\x00\x00\x00cpp.weight_prepackq(\x88X\x11\x00\x00\x00triton.cudagraphsq)\x88X\x17\x00\x00\x00triton.debug_sync_graphq*\x89X\x18\x00\x00\x00triton.debug_sync_kernelq+\x89X\x12\x00\x00\x00triton.convolutionq,X\x04\x00\x00\x00atenq-X\x15\x00\x00\x00triton.dense_indexingq.\x89X\x10\x00\x00\x00triton.max_tilesq/K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq0\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq1\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq2\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq3\x88X\x1f\x00\x00\x00triton.descriptive_kernel_namesq4\x89X\r\x00\x00\x00trace.enabledq5\x88X\x0f\x00\x00\x00trace.debug_logq6\x88X\x0e\x00\x00\x00trace.info_logq7\x89X\x0e\x00\x00\x00trace.fx_graphq8\x88X\x1a\x00\x00\x00trace.fx_graph_transformedq9\x88X\x13\x00\x00\x00trace.ir_pre_fusionq:\x88X\x14\x00\x00\x00trace.ir_post_fusionq;\x88X\x11\x00\x00\x00trace.output_codeq<\x88X\x13\x00\x00\x00trace.graph_diagramq=\x89X\x15\x00\x00\x00trace.compile_profileq>\x89X\x10\x00\x00\x00trace.upload_tarq?Nu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x12\x00\x00\x00use_dynamic_shapesq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x03\x00\x00\x00cseq\x08\x88X\x10\x00\x00\x00max_dist_from_bwq\tK\x03X\x0b\x00\x00\x00debug_jointq\n\x88X\x0c\x00\x00\x00debug_graphsq\x0b\x88X\x11\x00\x00\x00debug_partitionerq\x0c\x88X\t\x00\x00\x00log_levelq\rK\nu.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.0a0+git05397b1
# torch cuda version: 11.6
# torch git version: 05397b12505f4fd1bc98af562e103f4162993c1a


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Thu_Feb_10_18:23:41_PST_2022 
# Cuda compilation tools, release 11.6, V11.6.112 
# Build cuda_11.6.r11.6/compiler.30978841_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1):
        mul = torch.ops.aten.mul.Tensor(arg25_1, -0.01);  arg25_1 = None
        add_ = torch.ops.aten.add_.Tensor(arg0_1, mul);  arg0_1 = mul = None
        mul_1 = torch.ops.aten.mul.Tensor(arg26_1, -0.01);  arg26_1 = None
        add__1 = torch.ops.aten.add_.Tensor(arg1_1, mul_1);  arg1_1 = mul_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(arg27_1, -0.01);  arg27_1 = None
        add__2 = torch.ops.aten.add_.Tensor(arg2_1, mul_2);  arg2_1 = mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(arg28_1, -0.01);  arg28_1 = None
        add__3 = torch.ops.aten.add_.Tensor(arg3_1, mul_3);  arg3_1 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(arg29_1, -0.01);  arg29_1 = None
        add__4 = torch.ops.aten.add_.Tensor(arg4_1, mul_4);  arg4_1 = mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(arg30_1, -0.01);  arg30_1 = None
        add__5 = torch.ops.aten.add_.Tensor(arg5_1, mul_5);  arg5_1 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(arg31_1, -0.01);  arg31_1 = None
        add__6 = torch.ops.aten.add_.Tensor(arg6_1, mul_6);  arg6_1 = mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(arg32_1, -0.01);  arg32_1 = None
        add__7 = torch.ops.aten.add_.Tensor(arg7_1, mul_7);  arg7_1 = mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(arg33_1, -0.01);  arg33_1 = None
        add__8 = torch.ops.aten.add_.Tensor(arg8_1, mul_8);  arg8_1 = mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(arg34_1, -0.01);  arg34_1 = None
        add__9 = torch.ops.aten.add_.Tensor(arg9_1, mul_9);  arg9_1 = mul_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(arg35_1, -0.01);  arg35_1 = None
        add__10 = torch.ops.aten.add_.Tensor(arg10_1, mul_10);  arg10_1 = mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(arg36_1, -0.01);  arg36_1 = None
        add__11 = torch.ops.aten.add_.Tensor(arg11_1, mul_11);  arg11_1 = mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(arg37_1, -0.01);  arg37_1 = None
        add__12 = torch.ops.aten.add_.Tensor(arg12_1, mul_12);  arg12_1 = mul_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(arg38_1, -0.01);  arg38_1 = None
        add__13 = torch.ops.aten.add_.Tensor(arg13_1, mul_13);  arg13_1 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(arg39_1, -0.01);  arg39_1 = None
        add__14 = torch.ops.aten.add_.Tensor(arg14_1, mul_14);  arg14_1 = mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(arg40_1, -0.01);  arg40_1 = None
        add__15 = torch.ops.aten.add_.Tensor(arg15_1, mul_15);  arg15_1 = mul_15 = None
        mul_16 = torch.ops.aten.mul.Tensor(arg41_1, -0.01);  arg41_1 = None
        add__16 = torch.ops.aten.add_.Tensor(arg16_1, mul_16);  arg16_1 = mul_16 = None
        mul_17 = torch.ops.aten.mul.Tensor(arg42_1, -0.01);  arg42_1 = None
        add__17 = torch.ops.aten.add_.Tensor(arg17_1, mul_17);  arg17_1 = mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(arg43_1, -0.01);  arg43_1 = None
        add__18 = torch.ops.aten.add_.Tensor(arg18_1, mul_18);  arg18_1 = mul_18 = None
        mul_19 = torch.ops.aten.mul.Tensor(arg44_1, -0.01);  arg44_1 = None
        add__19 = torch.ops.aten.add_.Tensor(arg19_1, mul_19);  arg19_1 = mul_19 = None
        mul_20 = torch.ops.aten.mul.Tensor(arg45_1, -0.01);  arg45_1 = None
        add__20 = torch.ops.aten.add_.Tensor(arg20_1, mul_20);  arg20_1 = mul_20 = None
        mul_21 = torch.ops.aten.mul.Tensor(arg46_1, -0.01);  arg46_1 = None
        add__21 = torch.ops.aten.add_.Tensor(arg21_1, mul_21);  arg21_1 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(arg47_1, -0.01);  arg47_1 = None
        add__22 = torch.ops.aten.add_.Tensor(arg22_1, mul_22);  arg22_1 = mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(arg48_1, -0.01);  arg48_1 = None
        add__23 = torch.ops.aten.add_.Tensor(arg23_1, mul_23);  arg23_1 = mul_23 = None
        mul_24 = torch.ops.aten.mul.Tensor(arg49_1, -0.01);  arg49_1 = None
        add__24 = torch.ops.aten.add_.Tensor(arg24_1, mul_24);  arg24_1 = mul_24 = None
        return ()
        
args = [((30000, 128), (128, 1), torch.float32, 'cuda'), ((512, 128), (128, 1), torch.float32, 'cuda'), ((2, 128), (128, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((4096, 128), (128, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((16384, 4096), (4096, 1), torch.float32, 'cuda'), ((16384,), (1,), torch.float32, 'cuda'), ((4096, 16384), (16384, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((2, 4096), (4096, 1), torch.float32, 'cuda'), ((2,), (1,), torch.float32, 'cuda'), ((30000, 128), (128, 1), torch.float32, 'cuda'), ((512, 128), (128, 1), torch.float32, 'cuda'), ((2, 128), (128, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((4096, 128), (128, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((16384, 4096), (4096, 1), torch.float32, 'cuda'), ((16384,), (1,), torch.float32, 'cuda'), ((4096, 16384), (16384, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((2, 4096), (4096, 1), torch.float32, 'cuda'), ((2,), (1,), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
