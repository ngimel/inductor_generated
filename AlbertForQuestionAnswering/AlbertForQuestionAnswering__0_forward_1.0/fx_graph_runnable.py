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
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x88X\x0c\x00\x00\x00debug_graphsq\x07\x88X\x0b\x00\x00\x00debug_jointq\x08\x88X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\nu.')


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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30):
        full = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1 = torch.ops.aten.slice.Tensor(primals_26, 0, 0, 9223372036854775807);  primals_26 = None
        expand = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, -10000.0);  sub = None
        slice_2 = torch.ops.aten.slice.Tensor(primals_27, 0, 0, 9223372036854775807);  primals_27 = None
        embedding = torch.ops.aten.embedding.default(primals_1, primals_28, 0);  primals_1 = None
        embedding_1 = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(primals_3, slice_2);  primals_3 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, primals_4)
        add_3 = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
        permute = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        view = torch.ops.aten.view.default(add_3, [512, 128]);  add_3 = None
        addmm = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
        view_1 = torch.ops.aten.view.default(addmm, [1, 512, 4096]);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        view_2 = torch.ops.aten.view.default(view_1, [512, 4096])
        addmm_1 = torch.ops.aten.addmm.default(primals_9, view_2, permute_1)
        view_3 = torch.ops.aten.view.default(addmm_1, [1, 512, 4096]);  addmm_1 = None
        permute_2 = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_11, view_2, permute_2)
        view_5 = torch.ops.aten.view.default(addmm_2, [1, 512, 4096]);  addmm_2 = None
        permute_3 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_13, view_2, permute_3)
        view_7 = torch.ops.aten.view.default(addmm_3, [1, 512, 4096]);  addmm_3 = None
        view_8 = torch.ops.aten.view.default(view_3, [1, 512, 64, 64]);  view_3 = None
        permute_4 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        view_9 = torch.ops.aten.view.default(view_5, [1, 512, 64, 64]);  view_5 = None
        permute_5 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        view_10 = torch.ops.aten.view.default(view_7, [1, 512, 64, 64]);  view_7 = None
        permute_6 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
        expand_1 = torch.ops.aten.expand.default(permute_4, [1, 64, 512, 64]);  permute_4 = None
        view_11 = torch.ops.aten.view.default(expand_1, [64, 512, 64]);  expand_1 = None
        expand_2 = torch.ops.aten.expand.default(permute_7, [1, 64, 64, 512]);  permute_7 = None
        view_12 = torch.ops.aten.view.default(expand_2, [64, 64, 512]);  expand_2 = None
        bmm = torch.ops.aten.bmm.default(view_11, view_12)
        view_13 = torch.ops.aten.view.default(bmm, [1, 64, 512, 512]);  bmm = None
        div = torch.ops.aten.div.Tensor(view_13, 8.0);  view_13 = None
        add_4 = torch.ops.aten.add.Tensor(div, mul);  div = None
        amax = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_3 = torch.ops.aten.expand.default(div_1, [1, 64, 512, 512])
        view_14 = torch.ops.aten.view.default(expand_3, [64, 512, 512]);  expand_3 = None
        expand_4 = torch.ops.aten.expand.default(permute_6, [1, 64, 512, 64]);  permute_6 = None
        view_15 = torch.ops.aten.view.default(expand_4, [64, 512, 64]);  expand_4 = None
        bmm_1 = torch.ops.aten.bmm.default(view_14, view_15)
        view_16 = torch.ops.aten.view.default(bmm_1, [1, 64, 512, 64]);  bmm_1 = None
        permute_8 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [1, 512, 4096]);  clone = None
        permute_9 = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        view_17 = torch.ops.aten.view.default(_unsafe_view, [512, 4096]);  _unsafe_view = None
        addmm_4 = torch.ops.aten.addmm.default(primals_15, view_17, permute_9)
        view_18 = torch.ops.aten.view.default(addmm_4, [1, 512, 4096]);  addmm_4 = None
        add_5 = torch.ops.aten.add.Tensor(view_1, view_18);  view_1 = view_18 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, primals_16)
        add_7 = torch.ops.aten.add.Tensor(mul_4, primals_17);  mul_4 = None
        permute_10 = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        view_19 = torch.ops.aten.view.default(add_7, [512, 4096])
        addmm_5 = torch.ops.aten.addmm.default(primals_19, view_19, permute_10)
        view_20 = torch.ops.aten.view.default(addmm_5, [1, 512, 16384])
        mul_5 = torch.ops.aten.mul.Tensor(view_20, 0.5)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_20, 3.0)
        mul_6 = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_8 = torch.ops.aten.add.Tensor(view_20, mul_6);  view_20 = mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh = torch.ops.aten.tanh.default(mul_7);  mul_7 = None
        add_9 = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_8 = torch.ops.aten.mul.Tensor(mul_5, add_9);  mul_5 = add_9 = None
        permute_11 = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        view_21 = torch.ops.aten.view.default(mul_8, [512, 16384]);  mul_8 = None
        addmm_6 = torch.ops.aten.addmm.default(primals_21, view_21, permute_11)
        view_22 = torch.ops.aten.view.default(addmm_6, [1, 512, 4096]);  addmm_6 = None
        add_10 = torch.ops.aten.add.Tensor(view_22, add_7);  view_22 = add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_22)
        add_12 = torch.ops.aten.add.Tensor(mul_10, primals_23);  mul_10 = None
        view_23 = torch.ops.aten.view.default(add_12, [512, 4096])
        addmm_7 = torch.ops.aten.addmm.default(primals_9, view_23, permute_1)
        view_24 = torch.ops.aten.view.default(addmm_7, [1, 512, 4096]);  addmm_7 = None
        addmm_8 = torch.ops.aten.addmm.default(primals_11, view_23, permute_2)
        view_26 = torch.ops.aten.view.default(addmm_8, [1, 512, 4096]);  addmm_8 = None
        addmm_9 = torch.ops.aten.addmm.default(primals_13, view_23, permute_3)
        view_28 = torch.ops.aten.view.default(addmm_9, [1, 512, 4096]);  addmm_9 = None
        view_29 = torch.ops.aten.view.default(view_24, [1, 512, 64, 64]);  view_24 = None
        permute_15 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        view_30 = torch.ops.aten.view.default(view_26, [1, 512, 64, 64]);  view_26 = None
        permute_16 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        view_31 = torch.ops.aten.view.default(view_28, [1, 512, 64, 64]);  view_28 = None
        permute_17 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        permute_18 = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_5 = torch.ops.aten.expand.default(permute_15, [1, 64, 512, 64]);  permute_15 = None
        view_32 = torch.ops.aten.view.default(expand_5, [64, 512, 64]);  expand_5 = None
        expand_6 = torch.ops.aten.expand.default(permute_18, [1, 64, 64, 512]);  permute_18 = None
        view_33 = torch.ops.aten.view.default(expand_6, [64, 64, 512]);  expand_6 = None
        bmm_2 = torch.ops.aten.bmm.default(view_32, view_33)
        view_34 = torch.ops.aten.view.default(bmm_2, [1, 64, 512, 512]);  bmm_2 = None
        div_2 = torch.ops.aten.div.Tensor(view_34, 8.0);  view_34 = None
        add_13 = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
        amax_1 = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_5 = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        expand_7 = torch.ops.aten.expand.default(div_3, [1, 64, 512, 512])
        view_35 = torch.ops.aten.view.default(expand_7, [64, 512, 512]);  expand_7 = None
        expand_8 = torch.ops.aten.expand.default(permute_17, [1, 64, 512, 64]);  permute_17 = None
        view_36 = torch.ops.aten.view.default(expand_8, [64, 512, 64]);  expand_8 = None
        bmm_3 = torch.ops.aten.bmm.default(view_35, view_36)
        view_37 = torch.ops.aten.view.default(bmm_3, [1, 64, 512, 64]);  bmm_3 = None
        permute_19 = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        clone_1 = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_1, [1, 512, 4096]);  clone_1 = None
        view_38 = torch.ops.aten.view.default(_unsafe_view_1, [512, 4096]);  _unsafe_view_1 = None
        addmm_10 = torch.ops.aten.addmm.default(primals_15, view_38, permute_9)
        view_39 = torch.ops.aten.view.default(addmm_10, [1, 512, 4096]);  addmm_10 = None
        add_14 = torch.ops.aten.add.Tensor(add_12, view_39);  add_12 = view_39 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_14, getitem_7);  add_14 = getitem_7 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, primals_16)
        add_16 = torch.ops.aten.add.Tensor(mul_12, primals_17);  mul_12 = None
        view_40 = torch.ops.aten.view.default(add_16, [512, 4096])
        addmm_11 = torch.ops.aten.addmm.default(primals_19, view_40, permute_10)
        view_41 = torch.ops.aten.view.default(addmm_11, [1, 512, 16384])
        mul_13 = torch.ops.aten.mul.Tensor(view_41, 0.5)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_14 = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17 = torch.ops.aten.add.Tensor(view_41, mul_14);  view_41 = mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_15);  mul_15 = None
        add_18 = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_16 = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = add_18 = None
        view_42 = torch.ops.aten.view.default(mul_16, [512, 16384]);  mul_16 = None
        addmm_12 = torch.ops.aten.addmm.default(primals_21, view_42, permute_11)
        view_43 = torch.ops.aten.view.default(addmm_12, [1, 512, 4096]);  addmm_12 = None
        add_19 = torch.ops.aten.add.Tensor(view_43, add_16);  view_43 = add_16 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, primals_22)
        add_21 = torch.ops.aten.add.Tensor(mul_18, primals_23);  mul_18 = None
        view_44 = torch.ops.aten.view.default(add_21, [512, 4096])
        addmm_13 = torch.ops.aten.addmm.default(primals_9, view_44, permute_1)
        view_45 = torch.ops.aten.view.default(addmm_13, [1, 512, 4096]);  addmm_13 = None
        addmm_14 = torch.ops.aten.addmm.default(primals_11, view_44, permute_2)
        view_47 = torch.ops.aten.view.default(addmm_14, [1, 512, 4096]);  addmm_14 = None
        addmm_15 = torch.ops.aten.addmm.default(primals_13, view_44, permute_3)
        view_49 = torch.ops.aten.view.default(addmm_15, [1, 512, 4096]);  addmm_15 = None
        view_50 = torch.ops.aten.view.default(view_45, [1, 512, 64, 64]);  view_45 = None
        permute_26 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        view_51 = torch.ops.aten.view.default(view_47, [1, 512, 64, 64]);  view_47 = None
        permute_27 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        view_52 = torch.ops.aten.view.default(view_49, [1, 512, 64, 64]);  view_49 = None
        permute_28 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        permute_29 = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2]);  permute_27 = None
        expand_9 = torch.ops.aten.expand.default(permute_26, [1, 64, 512, 64]);  permute_26 = None
        view_53 = torch.ops.aten.view.default(expand_9, [64, 512, 64]);  expand_9 = None
        expand_10 = torch.ops.aten.expand.default(permute_29, [1, 64, 64, 512]);  permute_29 = None
        view_54 = torch.ops.aten.view.default(expand_10, [64, 64, 512]);  expand_10 = None
        bmm_4 = torch.ops.aten.bmm.default(view_53, view_54)
        view_55 = torch.ops.aten.view.default(bmm_4, [1, 64, 512, 512]);  bmm_4 = None
        div_4 = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
        add_22 = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
        amax_2 = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_8 = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        expand_11 = torch.ops.aten.expand.default(div_5, [1, 64, 512, 512])
        view_56 = torch.ops.aten.view.default(expand_11, [64, 512, 512]);  expand_11 = None
        expand_12 = torch.ops.aten.expand.default(permute_28, [1, 64, 512, 64]);  permute_28 = None
        view_57 = torch.ops.aten.view.default(expand_12, [64, 512, 64]);  expand_12 = None
        bmm_5 = torch.ops.aten.bmm.default(view_56, view_57)
        view_58 = torch.ops.aten.view.default(bmm_5, [1, 64, 512, 64]);  bmm_5 = None
        permute_30 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_2 = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_2, [1, 512, 4096]);  clone_2 = None
        view_59 = torch.ops.aten.view.default(_unsafe_view_2, [512, 4096]);  _unsafe_view_2 = None
        addmm_16 = torch.ops.aten.addmm.default(primals_15, view_59, permute_9)
        view_60 = torch.ops.aten.view.default(addmm_16, [1, 512, 4096]);  addmm_16 = None
        add_23 = torch.ops.aten.add.Tensor(add_21, view_60);  add_21 = view_60 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, primals_16)
        add_25 = torch.ops.aten.add.Tensor(mul_20, primals_17);  mul_20 = None
        view_61 = torch.ops.aten.view.default(add_25, [512, 4096])
        addmm_17 = torch.ops.aten.addmm.default(primals_19, view_61, permute_10)
        view_62 = torch.ops.aten.view.default(addmm_17, [1, 512, 16384])
        mul_21 = torch.ops.aten.mul.Tensor(view_62, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_62, 3.0)
        mul_22 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_26 = torch.ops.aten.add.Tensor(view_62, mul_22);  view_62 = mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_23);  mul_23 = None
        add_27 = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_24 = torch.ops.aten.mul.Tensor(mul_21, add_27);  mul_21 = add_27 = None
        view_63 = torch.ops.aten.view.default(mul_24, [512, 16384]);  mul_24 = None
        addmm_18 = torch.ops.aten.addmm.default(primals_21, view_63, permute_11)
        view_64 = torch.ops.aten.view.default(addmm_18, [1, 512, 4096]);  addmm_18 = None
        add_28 = torch.ops.aten.add.Tensor(view_64, add_25);  view_64 = add_25 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_28, getitem_13);  add_28 = getitem_13 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, primals_22)
        add_30 = torch.ops.aten.add.Tensor(mul_26, primals_23);  mul_26 = None
        view_65 = torch.ops.aten.view.default(add_30, [512, 4096])
        addmm_19 = torch.ops.aten.addmm.default(primals_9, view_65, permute_1)
        view_66 = torch.ops.aten.view.default(addmm_19, [1, 512, 4096]);  addmm_19 = None
        addmm_20 = torch.ops.aten.addmm.default(primals_11, view_65, permute_2)
        view_68 = torch.ops.aten.view.default(addmm_20, [1, 512, 4096]);  addmm_20 = None
        addmm_21 = torch.ops.aten.addmm.default(primals_13, view_65, permute_3)
        view_70 = torch.ops.aten.view.default(addmm_21, [1, 512, 4096]);  addmm_21 = None
        view_71 = torch.ops.aten.view.default(view_66, [1, 512, 64, 64]);  view_66 = None
        permute_37 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        view_72 = torch.ops.aten.view.default(view_68, [1, 512, 64, 64]);  view_68 = None
        permute_38 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        view_73 = torch.ops.aten.view.default(view_70, [1, 512, 64, 64]);  view_70 = None
        permute_39 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        permute_40 = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
        expand_13 = torch.ops.aten.expand.default(permute_37, [1, 64, 512, 64]);  permute_37 = None
        view_74 = torch.ops.aten.view.default(expand_13, [64, 512, 64]);  expand_13 = None
        expand_14 = torch.ops.aten.expand.default(permute_40, [1, 64, 64, 512]);  permute_40 = None
        view_75 = torch.ops.aten.view.default(expand_14, [64, 64, 512]);  expand_14 = None
        bmm_6 = torch.ops.aten.bmm.default(view_74, view_75)
        view_76 = torch.ops.aten.view.default(bmm_6, [1, 64, 512, 512]);  bmm_6 = None
        div_6 = torch.ops.aten.div.Tensor(view_76, 8.0);  view_76 = None
        add_31 = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
        amax_3 = torch.ops.aten.amax.default(add_31, [-1], True)
        sub_11 = torch.ops.aten.sub.Tensor(add_31, amax_3);  add_31 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        expand_15 = torch.ops.aten.expand.default(div_7, [1, 64, 512, 512])
        view_77 = torch.ops.aten.view.default(expand_15, [64, 512, 512]);  expand_15 = None
        expand_16 = torch.ops.aten.expand.default(permute_39, [1, 64, 512, 64]);  permute_39 = None
        view_78 = torch.ops.aten.view.default(expand_16, [64, 512, 64]);  expand_16 = None
        bmm_7 = torch.ops.aten.bmm.default(view_77, view_78)
        view_79 = torch.ops.aten.view.default(bmm_7, [1, 64, 512, 64]);  bmm_7 = None
        permute_41 = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        clone_3 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_3, [1, 512, 4096]);  clone_3 = None
        view_80 = torch.ops.aten.view.default(_unsafe_view_3, [512, 4096]);  _unsafe_view_3 = None
        addmm_22 = torch.ops.aten.addmm.default(primals_15, view_80, permute_9)
        view_81 = torch.ops.aten.view.default(addmm_22, [1, 512, 4096]);  addmm_22 = None
        add_32 = torch.ops.aten.add.Tensor(add_30, view_81);  add_30 = view_81 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_32, getitem_15);  add_32 = getitem_15 = None
        mul_27 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, primals_16)
        add_34 = torch.ops.aten.add.Tensor(mul_28, primals_17);  mul_28 = None
        view_82 = torch.ops.aten.view.default(add_34, [512, 4096])
        addmm_23 = torch.ops.aten.addmm.default(primals_19, view_82, permute_10)
        view_83 = torch.ops.aten.view.default(addmm_23, [1, 512, 16384])
        mul_29 = torch.ops.aten.mul.Tensor(view_83, 0.5)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_83, 3.0)
        mul_30 = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_35 = torch.ops.aten.add.Tensor(view_83, mul_30);  view_83 = mul_30 = None
        mul_31 = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
        add_36 = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_32 = torch.ops.aten.mul.Tensor(mul_29, add_36);  mul_29 = add_36 = None
        view_84 = torch.ops.aten.view.default(mul_32, [512, 16384]);  mul_32 = None
        addmm_24 = torch.ops.aten.addmm.default(primals_21, view_84, permute_11)
        view_85 = torch.ops.aten.view.default(addmm_24, [1, 512, 4096]);  addmm_24 = None
        add_37 = torch.ops.aten.add.Tensor(view_85, add_34);  view_85 = add_34 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_37, getitem_17);  add_37 = getitem_17 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, primals_22)
        add_39 = torch.ops.aten.add.Tensor(mul_34, primals_23);  mul_34 = None
        view_86 = torch.ops.aten.view.default(add_39, [512, 4096])
        addmm_25 = torch.ops.aten.addmm.default(primals_9, view_86, permute_1)
        view_87 = torch.ops.aten.view.default(addmm_25, [1, 512, 4096]);  addmm_25 = None
        addmm_26 = torch.ops.aten.addmm.default(primals_11, view_86, permute_2)
        view_89 = torch.ops.aten.view.default(addmm_26, [1, 512, 4096]);  addmm_26 = None
        addmm_27 = torch.ops.aten.addmm.default(primals_13, view_86, permute_3)
        view_91 = torch.ops.aten.view.default(addmm_27, [1, 512, 4096]);  addmm_27 = None
        view_92 = torch.ops.aten.view.default(view_87, [1, 512, 64, 64]);  view_87 = None
        permute_48 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        view_93 = torch.ops.aten.view.default(view_89, [1, 512, 64, 64]);  view_89 = None
        permute_49 = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        view_94 = torch.ops.aten.view.default(view_91, [1, 512, 64, 64]);  view_91 = None
        permute_50 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        permute_51 = torch.ops.aten.permute.default(permute_49, [0, 1, 3, 2]);  permute_49 = None
        expand_17 = torch.ops.aten.expand.default(permute_48, [1, 64, 512, 64]);  permute_48 = None
        view_95 = torch.ops.aten.view.default(expand_17, [64, 512, 64]);  expand_17 = None
        expand_18 = torch.ops.aten.expand.default(permute_51, [1, 64, 64, 512]);  permute_51 = None
        view_96 = torch.ops.aten.view.default(expand_18, [64, 64, 512]);  expand_18 = None
        bmm_8 = torch.ops.aten.bmm.default(view_95, view_96)
        view_97 = torch.ops.aten.view.default(bmm_8, [1, 64, 512, 512]);  bmm_8 = None
        div_8 = torch.ops.aten.div.Tensor(view_97, 8.0);  view_97 = None
        add_40 = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
        amax_4 = torch.ops.aten.amax.default(add_40, [-1], True)
        sub_14 = torch.ops.aten.sub.Tensor(add_40, amax_4);  add_40 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        expand_19 = torch.ops.aten.expand.default(div_9, [1, 64, 512, 512])
        view_98 = torch.ops.aten.view.default(expand_19, [64, 512, 512]);  expand_19 = None
        expand_20 = torch.ops.aten.expand.default(permute_50, [1, 64, 512, 64]);  permute_50 = None
        view_99 = torch.ops.aten.view.default(expand_20, [64, 512, 64]);  expand_20 = None
        bmm_9 = torch.ops.aten.bmm.default(view_98, view_99)
        view_100 = torch.ops.aten.view.default(bmm_9, [1, 64, 512, 64]);  bmm_9 = None
        permute_52 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        clone_4 = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        _unsafe_view_4 = torch.ops.aten._unsafe_view.default(clone_4, [1, 512, 4096]);  clone_4 = None
        view_101 = torch.ops.aten.view.default(_unsafe_view_4, [512, 4096]);  _unsafe_view_4 = None
        addmm_28 = torch.ops.aten.addmm.default(primals_15, view_101, permute_9)
        view_102 = torch.ops.aten.view.default(addmm_28, [1, 512, 4096]);  addmm_28 = None
        add_41 = torch.ops.aten.add.Tensor(add_39, view_102);  add_39 = view_102 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_41, getitem_19);  add_41 = getitem_19 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, primals_16)
        add_43 = torch.ops.aten.add.Tensor(mul_36, primals_17);  mul_36 = None
        view_103 = torch.ops.aten.view.default(add_43, [512, 4096])
        addmm_29 = torch.ops.aten.addmm.default(primals_19, view_103, permute_10)
        view_104 = torch.ops.aten.view.default(addmm_29, [1, 512, 16384])
        mul_37 = torch.ops.aten.mul.Tensor(view_104, 0.5)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_104, 3.0)
        mul_38 = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_44 = torch.ops.aten.add.Tensor(view_104, mul_38);  view_104 = mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_39);  mul_39 = None
        add_45 = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_40 = torch.ops.aten.mul.Tensor(mul_37, add_45);  mul_37 = add_45 = None
        view_105 = torch.ops.aten.view.default(mul_40, [512, 16384]);  mul_40 = None
        addmm_30 = torch.ops.aten.addmm.default(primals_21, view_105, permute_11)
        view_106 = torch.ops.aten.view.default(addmm_30, [1, 512, 4096]);  addmm_30 = None
        add_46 = torch.ops.aten.add.Tensor(view_106, add_43);  view_106 = add_43 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_46, getitem_21);  add_46 = getitem_21 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, primals_22)
        add_48 = torch.ops.aten.add.Tensor(mul_42, primals_23);  mul_42 = None
        view_107 = torch.ops.aten.view.default(add_48, [512, 4096])
        addmm_31 = torch.ops.aten.addmm.default(primals_9, view_107, permute_1)
        view_108 = torch.ops.aten.view.default(addmm_31, [1, 512, 4096]);  addmm_31 = None
        addmm_32 = torch.ops.aten.addmm.default(primals_11, view_107, permute_2)
        view_110 = torch.ops.aten.view.default(addmm_32, [1, 512, 4096]);  addmm_32 = None
        addmm_33 = torch.ops.aten.addmm.default(primals_13, view_107, permute_3)
        view_112 = torch.ops.aten.view.default(addmm_33, [1, 512, 4096]);  addmm_33 = None
        view_113 = torch.ops.aten.view.default(view_108, [1, 512, 64, 64]);  view_108 = None
        permute_59 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        view_114 = torch.ops.aten.view.default(view_110, [1, 512, 64, 64]);  view_110 = None
        permute_60 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        view_115 = torch.ops.aten.view.default(view_112, [1, 512, 64, 64]);  view_112 = None
        permute_61 = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        permute_62 = torch.ops.aten.permute.default(permute_60, [0, 1, 3, 2]);  permute_60 = None
        expand_21 = torch.ops.aten.expand.default(permute_59, [1, 64, 512, 64]);  permute_59 = None
        view_116 = torch.ops.aten.view.default(expand_21, [64, 512, 64]);  expand_21 = None
        expand_22 = torch.ops.aten.expand.default(permute_62, [1, 64, 64, 512]);  permute_62 = None
        view_117 = torch.ops.aten.view.default(expand_22, [64, 64, 512]);  expand_22 = None
        bmm_10 = torch.ops.aten.bmm.default(view_116, view_117)
        view_118 = torch.ops.aten.view.default(bmm_10, [1, 64, 512, 512]);  bmm_10 = None
        div_10 = torch.ops.aten.div.Tensor(view_118, 8.0);  view_118 = None
        add_49 = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
        amax_5 = torch.ops.aten.amax.default(add_49, [-1], True)
        sub_17 = torch.ops.aten.sub.Tensor(add_49, amax_5);  add_49 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        expand_23 = torch.ops.aten.expand.default(div_11, [1, 64, 512, 512])
        view_119 = torch.ops.aten.view.default(expand_23, [64, 512, 512]);  expand_23 = None
        expand_24 = torch.ops.aten.expand.default(permute_61, [1, 64, 512, 64]);  permute_61 = None
        view_120 = torch.ops.aten.view.default(expand_24, [64, 512, 64]);  expand_24 = None
        bmm_11 = torch.ops.aten.bmm.default(view_119, view_120)
        view_121 = torch.ops.aten.view.default(bmm_11, [1, 64, 512, 64]);  bmm_11 = None
        permute_63 = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_5 = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        _unsafe_view_5 = torch.ops.aten._unsafe_view.default(clone_5, [1, 512, 4096]);  clone_5 = None
        view_122 = torch.ops.aten.view.default(_unsafe_view_5, [512, 4096]);  _unsafe_view_5 = None
        addmm_34 = torch.ops.aten.addmm.default(primals_15, view_122, permute_9)
        view_123 = torch.ops.aten.view.default(addmm_34, [1, 512, 4096]);  addmm_34 = None
        add_50 = torch.ops.aten.add.Tensor(add_48, view_123);  add_48 = view_123 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_50, getitem_23);  add_50 = getitem_23 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, primals_16)
        add_52 = torch.ops.aten.add.Tensor(mul_44, primals_17);  mul_44 = None
        view_124 = torch.ops.aten.view.default(add_52, [512, 4096])
        addmm_35 = torch.ops.aten.addmm.default(primals_19, view_124, permute_10)
        view_125 = torch.ops.aten.view.default(addmm_35, [1, 512, 16384])
        mul_45 = torch.ops.aten.mul.Tensor(view_125, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
        mul_46 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_53 = torch.ops.aten.add.Tensor(view_125, mul_46);  view_125 = mul_46 = None
        mul_47 = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_47);  mul_47 = None
        add_54 = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_48 = torch.ops.aten.mul.Tensor(mul_45, add_54);  mul_45 = add_54 = None
        view_126 = torch.ops.aten.view.default(mul_48, [512, 16384]);  mul_48 = None
        addmm_36 = torch.ops.aten.addmm.default(primals_21, view_126, permute_11)
        view_127 = torch.ops.aten.view.default(addmm_36, [1, 512, 4096]);  addmm_36 = None
        add_55 = torch.ops.aten.add.Tensor(view_127, add_52);  view_127 = add_52 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, primals_22)
        add_57 = torch.ops.aten.add.Tensor(mul_50, primals_23);  mul_50 = None
        view_128 = torch.ops.aten.view.default(add_57, [512, 4096])
        addmm_37 = torch.ops.aten.addmm.default(primals_9, view_128, permute_1)
        view_129 = torch.ops.aten.view.default(addmm_37, [1, 512, 4096]);  addmm_37 = None
        addmm_38 = torch.ops.aten.addmm.default(primals_11, view_128, permute_2)
        view_131 = torch.ops.aten.view.default(addmm_38, [1, 512, 4096]);  addmm_38 = None
        addmm_39 = torch.ops.aten.addmm.default(primals_13, view_128, permute_3)
        view_133 = torch.ops.aten.view.default(addmm_39, [1, 512, 4096]);  addmm_39 = None
        view_134 = torch.ops.aten.view.default(view_129, [1, 512, 64, 64]);  view_129 = None
        permute_70 = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        view_135 = torch.ops.aten.view.default(view_131, [1, 512, 64, 64]);  view_131 = None
        permute_71 = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        view_136 = torch.ops.aten.view.default(view_133, [1, 512, 64, 64]);  view_133 = None
        permute_72 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        permute_73 = torch.ops.aten.permute.default(permute_71, [0, 1, 3, 2]);  permute_71 = None
        expand_25 = torch.ops.aten.expand.default(permute_70, [1, 64, 512, 64]);  permute_70 = None
        view_137 = torch.ops.aten.view.default(expand_25, [64, 512, 64]);  expand_25 = None
        expand_26 = torch.ops.aten.expand.default(permute_73, [1, 64, 64, 512]);  permute_73 = None
        view_138 = torch.ops.aten.view.default(expand_26, [64, 64, 512]);  expand_26 = None
        bmm_12 = torch.ops.aten.bmm.default(view_137, view_138)
        view_139 = torch.ops.aten.view.default(bmm_12, [1, 64, 512, 512]);  bmm_12 = None
        div_12 = torch.ops.aten.div.Tensor(view_139, 8.0);  view_139 = None
        add_58 = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
        amax_6 = torch.ops.aten.amax.default(add_58, [-1], True)
        sub_20 = torch.ops.aten.sub.Tensor(add_58, amax_6);  add_58 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        expand_27 = torch.ops.aten.expand.default(div_13, [1, 64, 512, 512])
        view_140 = torch.ops.aten.view.default(expand_27, [64, 512, 512]);  expand_27 = None
        expand_28 = torch.ops.aten.expand.default(permute_72, [1, 64, 512, 64]);  permute_72 = None
        view_141 = torch.ops.aten.view.default(expand_28, [64, 512, 64]);  expand_28 = None
        bmm_13 = torch.ops.aten.bmm.default(view_140, view_141)
        view_142 = torch.ops.aten.view.default(bmm_13, [1, 64, 512, 64]);  bmm_13 = None
        permute_74 = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_6 = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_6, [1, 512, 4096]);  clone_6 = None
        view_143 = torch.ops.aten.view.default(_unsafe_view_6, [512, 4096]);  _unsafe_view_6 = None
        addmm_40 = torch.ops.aten.addmm.default(primals_15, view_143, permute_9)
        view_144 = torch.ops.aten.view.default(addmm_40, [1, 512, 4096]);  addmm_40 = None
        add_59 = torch.ops.aten.add.Tensor(add_57, view_144);  add_57 = view_144 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, primals_16)
        add_61 = torch.ops.aten.add.Tensor(mul_52, primals_17);  mul_52 = None
        view_145 = torch.ops.aten.view.default(add_61, [512, 4096])
        addmm_41 = torch.ops.aten.addmm.default(primals_19, view_145, permute_10)
        view_146 = torch.ops.aten.view.default(addmm_41, [1, 512, 16384])
        mul_53 = torch.ops.aten.mul.Tensor(view_146, 0.5)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(view_146, 3.0)
        mul_54 = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_62 = torch.ops.aten.add.Tensor(view_146, mul_54);  view_146 = mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_55);  mul_55 = None
        add_63 = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_56 = torch.ops.aten.mul.Tensor(mul_53, add_63);  mul_53 = add_63 = None
        view_147 = torch.ops.aten.view.default(mul_56, [512, 16384]);  mul_56 = None
        addmm_42 = torch.ops.aten.addmm.default(primals_21, view_147, permute_11)
        view_148 = torch.ops.aten.view.default(addmm_42, [1, 512, 4096]);  addmm_42 = None
        add_64 = torch.ops.aten.add.Tensor(view_148, add_61);  view_148 = add_61 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_64, getitem_29);  add_64 = getitem_29 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, primals_22)
        add_66 = torch.ops.aten.add.Tensor(mul_58, primals_23);  mul_58 = None
        view_149 = torch.ops.aten.view.default(add_66, [512, 4096])
        addmm_43 = torch.ops.aten.addmm.default(primals_9, view_149, permute_1)
        view_150 = torch.ops.aten.view.default(addmm_43, [1, 512, 4096]);  addmm_43 = None
        addmm_44 = torch.ops.aten.addmm.default(primals_11, view_149, permute_2)
        view_152 = torch.ops.aten.view.default(addmm_44, [1, 512, 4096]);  addmm_44 = None
        addmm_45 = torch.ops.aten.addmm.default(primals_13, view_149, permute_3)
        view_154 = torch.ops.aten.view.default(addmm_45, [1, 512, 4096]);  addmm_45 = None
        view_155 = torch.ops.aten.view.default(view_150, [1, 512, 64, 64]);  view_150 = None
        permute_81 = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        view_156 = torch.ops.aten.view.default(view_152, [1, 512, 64, 64]);  view_152 = None
        permute_82 = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        view_157 = torch.ops.aten.view.default(view_154, [1, 512, 64, 64]);  view_154 = None
        permute_83 = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        permute_84 = torch.ops.aten.permute.default(permute_82, [0, 1, 3, 2]);  permute_82 = None
        expand_29 = torch.ops.aten.expand.default(permute_81, [1, 64, 512, 64]);  permute_81 = None
        view_158 = torch.ops.aten.view.default(expand_29, [64, 512, 64]);  expand_29 = None
        expand_30 = torch.ops.aten.expand.default(permute_84, [1, 64, 64, 512]);  permute_84 = None
        view_159 = torch.ops.aten.view.default(expand_30, [64, 64, 512]);  expand_30 = None
        bmm_14 = torch.ops.aten.bmm.default(view_158, view_159)
        view_160 = torch.ops.aten.view.default(bmm_14, [1, 64, 512, 512]);  bmm_14 = None
        div_14 = torch.ops.aten.div.Tensor(view_160, 8.0);  view_160 = None
        add_67 = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
        amax_7 = torch.ops.aten.amax.default(add_67, [-1], True)
        sub_23 = torch.ops.aten.sub.Tensor(add_67, amax_7);  add_67 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        expand_31 = torch.ops.aten.expand.default(div_15, [1, 64, 512, 512])
        view_161 = torch.ops.aten.view.default(expand_31, [64, 512, 512]);  expand_31 = None
        expand_32 = torch.ops.aten.expand.default(permute_83, [1, 64, 512, 64]);  permute_83 = None
        view_162 = torch.ops.aten.view.default(expand_32, [64, 512, 64]);  expand_32 = None
        bmm_15 = torch.ops.aten.bmm.default(view_161, view_162)
        view_163 = torch.ops.aten.view.default(bmm_15, [1, 64, 512, 64]);  bmm_15 = None
        permute_85 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        clone_7 = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(clone_7, [1, 512, 4096]);  clone_7 = None
        view_164 = torch.ops.aten.view.default(_unsafe_view_7, [512, 4096]);  _unsafe_view_7 = None
        addmm_46 = torch.ops.aten.addmm.default(primals_15, view_164, permute_9)
        view_165 = torch.ops.aten.view.default(addmm_46, [1, 512, 4096]);  addmm_46 = None
        add_68 = torch.ops.aten.add.Tensor(add_66, view_165);  add_66 = view_165 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_68, getitem_31);  add_68 = getitem_31 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, primals_16)
        add_70 = torch.ops.aten.add.Tensor(mul_60, primals_17);  mul_60 = None
        view_166 = torch.ops.aten.view.default(add_70, [512, 4096])
        addmm_47 = torch.ops.aten.addmm.default(primals_19, view_166, permute_10)
        view_167 = torch.ops.aten.view.default(addmm_47, [1, 512, 16384])
        mul_61 = torch.ops.aten.mul.Tensor(view_167, 0.5)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(view_167, 3.0)
        mul_62 = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_71 = torch.ops.aten.add.Tensor(view_167, mul_62);  view_167 = mul_62 = None
        mul_63 = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_63);  mul_63 = None
        add_72 = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_64 = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
        view_168 = torch.ops.aten.view.default(mul_64, [512, 16384]);  mul_64 = None
        addmm_48 = torch.ops.aten.addmm.default(primals_21, view_168, permute_11)
        view_169 = torch.ops.aten.view.default(addmm_48, [1, 512, 4096]);  addmm_48 = None
        add_73 = torch.ops.aten.add.Tensor(view_169, add_70);  view_169 = add_70 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_73, getitem_33);  add_73 = getitem_33 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, primals_22)
        add_75 = torch.ops.aten.add.Tensor(mul_66, primals_23);  mul_66 = None
        view_170 = torch.ops.aten.view.default(add_75, [512, 4096])
        addmm_49 = torch.ops.aten.addmm.default(primals_9, view_170, permute_1)
        view_171 = torch.ops.aten.view.default(addmm_49, [1, 512, 4096]);  addmm_49 = None
        addmm_50 = torch.ops.aten.addmm.default(primals_11, view_170, permute_2)
        view_173 = torch.ops.aten.view.default(addmm_50, [1, 512, 4096]);  addmm_50 = None
        addmm_51 = torch.ops.aten.addmm.default(primals_13, view_170, permute_3)
        view_175 = torch.ops.aten.view.default(addmm_51, [1, 512, 4096]);  addmm_51 = None
        view_176 = torch.ops.aten.view.default(view_171, [1, 512, 64, 64]);  view_171 = None
        permute_92 = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
        view_177 = torch.ops.aten.view.default(view_173, [1, 512, 64, 64]);  view_173 = None
        permute_93 = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        view_178 = torch.ops.aten.view.default(view_175, [1, 512, 64, 64]);  view_175 = None
        permute_94 = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        permute_95 = torch.ops.aten.permute.default(permute_93, [0, 1, 3, 2]);  permute_93 = None
        expand_33 = torch.ops.aten.expand.default(permute_92, [1, 64, 512, 64]);  permute_92 = None
        view_179 = torch.ops.aten.view.default(expand_33, [64, 512, 64]);  expand_33 = None
        expand_34 = torch.ops.aten.expand.default(permute_95, [1, 64, 64, 512]);  permute_95 = None
        view_180 = torch.ops.aten.view.default(expand_34, [64, 64, 512]);  expand_34 = None
        bmm_16 = torch.ops.aten.bmm.default(view_179, view_180)
        view_181 = torch.ops.aten.view.default(bmm_16, [1, 64, 512, 512]);  bmm_16 = None
        div_16 = torch.ops.aten.div.Tensor(view_181, 8.0);  view_181 = None
        add_76 = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
        amax_8 = torch.ops.aten.amax.default(add_76, [-1], True)
        sub_26 = torch.ops.aten.sub.Tensor(add_76, amax_8);  add_76 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        expand_35 = torch.ops.aten.expand.default(div_17, [1, 64, 512, 512])
        view_182 = torch.ops.aten.view.default(expand_35, [64, 512, 512]);  expand_35 = None
        expand_36 = torch.ops.aten.expand.default(permute_94, [1, 64, 512, 64]);  permute_94 = None
        view_183 = torch.ops.aten.view.default(expand_36, [64, 512, 64]);  expand_36 = None
        bmm_17 = torch.ops.aten.bmm.default(view_182, view_183)
        view_184 = torch.ops.aten.view.default(bmm_17, [1, 64, 512, 64]);  bmm_17 = None
        permute_96 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        clone_8 = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(clone_8, [1, 512, 4096]);  clone_8 = None
        view_185 = torch.ops.aten.view.default(_unsafe_view_8, [512, 4096]);  _unsafe_view_8 = None
        addmm_52 = torch.ops.aten.addmm.default(primals_15, view_185, permute_9)
        view_186 = torch.ops.aten.view.default(addmm_52, [1, 512, 4096]);  addmm_52 = None
        add_77 = torch.ops.aten.add.Tensor(add_75, view_186);  add_75 = view_186 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_77, getitem_35);  add_77 = getitem_35 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
        mul_68 = torch.ops.aten.mul.Tensor(mul_67, primals_16)
        add_79 = torch.ops.aten.add.Tensor(mul_68, primals_17);  mul_68 = None
        view_187 = torch.ops.aten.view.default(add_79, [512, 4096])
        addmm_53 = torch.ops.aten.addmm.default(primals_19, view_187, permute_10)
        view_188 = torch.ops.aten.view.default(addmm_53, [1, 512, 16384])
        mul_69 = torch.ops.aten.mul.Tensor(view_188, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(view_188, 3.0)
        mul_70 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_80 = torch.ops.aten.add.Tensor(view_188, mul_70);  view_188 = mul_70 = None
        mul_71 = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_71);  mul_71 = None
        add_81 = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_72 = torch.ops.aten.mul.Tensor(mul_69, add_81);  mul_69 = add_81 = None
        view_189 = torch.ops.aten.view.default(mul_72, [512, 16384]);  mul_72 = None
        addmm_54 = torch.ops.aten.addmm.default(primals_21, view_189, permute_11)
        view_190 = torch.ops.aten.view.default(addmm_54, [1, 512, 4096]);  addmm_54 = None
        add_82 = torch.ops.aten.add.Tensor(view_190, add_79);  view_190 = add_79 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_83 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_82, getitem_37);  add_82 = getitem_37 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, primals_22)
        add_84 = torch.ops.aten.add.Tensor(mul_74, primals_23);  mul_74 = None
        view_191 = torch.ops.aten.view.default(add_84, [512, 4096])
        addmm_55 = torch.ops.aten.addmm.default(primals_9, view_191, permute_1)
        view_192 = torch.ops.aten.view.default(addmm_55, [1, 512, 4096]);  addmm_55 = None
        addmm_56 = torch.ops.aten.addmm.default(primals_11, view_191, permute_2)
        view_194 = torch.ops.aten.view.default(addmm_56, [1, 512, 4096]);  addmm_56 = None
        addmm_57 = torch.ops.aten.addmm.default(primals_13, view_191, permute_3)
        view_196 = torch.ops.aten.view.default(addmm_57, [1, 512, 4096]);  addmm_57 = None
        view_197 = torch.ops.aten.view.default(view_192, [1, 512, 64, 64]);  view_192 = None
        permute_103 = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
        view_198 = torch.ops.aten.view.default(view_194, [1, 512, 64, 64]);  view_194 = None
        permute_104 = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
        view_199 = torch.ops.aten.view.default(view_196, [1, 512, 64, 64]);  view_196 = None
        permute_105 = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        permute_106 = torch.ops.aten.permute.default(permute_104, [0, 1, 3, 2]);  permute_104 = None
        expand_37 = torch.ops.aten.expand.default(permute_103, [1, 64, 512, 64]);  permute_103 = None
        view_200 = torch.ops.aten.view.default(expand_37, [64, 512, 64]);  expand_37 = None
        expand_38 = torch.ops.aten.expand.default(permute_106, [1, 64, 64, 512]);  permute_106 = None
        view_201 = torch.ops.aten.view.default(expand_38, [64, 64, 512]);  expand_38 = None
        bmm_18 = torch.ops.aten.bmm.default(view_200, view_201)
        view_202 = torch.ops.aten.view.default(bmm_18, [1, 64, 512, 512]);  bmm_18 = None
        div_18 = torch.ops.aten.div.Tensor(view_202, 8.0);  view_202 = None
        add_85 = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
        amax_9 = torch.ops.aten.amax.default(add_85, [-1], True)
        sub_29 = torch.ops.aten.sub.Tensor(add_85, amax_9);  add_85 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        expand_39 = torch.ops.aten.expand.default(div_19, [1, 64, 512, 512])
        view_203 = torch.ops.aten.view.default(expand_39, [64, 512, 512]);  expand_39 = None
        expand_40 = torch.ops.aten.expand.default(permute_105, [1, 64, 512, 64]);  permute_105 = None
        view_204 = torch.ops.aten.view.default(expand_40, [64, 512, 64]);  expand_40 = None
        bmm_19 = torch.ops.aten.bmm.default(view_203, view_204)
        view_205 = torch.ops.aten.view.default(bmm_19, [1, 64, 512, 64]);  bmm_19 = None
        permute_107 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_9 = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(clone_9, [1, 512, 4096]);  clone_9 = None
        view_206 = torch.ops.aten.view.default(_unsafe_view_9, [512, 4096]);  _unsafe_view_9 = None
        addmm_58 = torch.ops.aten.addmm.default(primals_15, view_206, permute_9)
        view_207 = torch.ops.aten.view.default(addmm_58, [1, 512, 4096]);  addmm_58 = None
        add_86 = torch.ops.aten.add.Tensor(add_84, view_207);  add_84 = view_207 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_87 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_86, getitem_39);  add_86 = getitem_39 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, primals_16)
        add_88 = torch.ops.aten.add.Tensor(mul_76, primals_17);  mul_76 = None
        view_208 = torch.ops.aten.view.default(add_88, [512, 4096])
        addmm_59 = torch.ops.aten.addmm.default(primals_19, view_208, permute_10)
        view_209 = torch.ops.aten.view.default(addmm_59, [1, 512, 16384])
        mul_77 = torch.ops.aten.mul.Tensor(view_209, 0.5)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(view_209, 3.0)
        mul_78 = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_89 = torch.ops.aten.add.Tensor(view_209, mul_78);  view_209 = mul_78 = None
        mul_79 = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_79);  mul_79 = None
        add_90 = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_80 = torch.ops.aten.mul.Tensor(mul_77, add_90);  mul_77 = add_90 = None
        view_210 = torch.ops.aten.view.default(mul_80, [512, 16384]);  mul_80 = None
        addmm_60 = torch.ops.aten.addmm.default(primals_21, view_210, permute_11)
        view_211 = torch.ops.aten.view.default(addmm_60, [1, 512, 4096]);  addmm_60 = None
        add_91 = torch.ops.aten.add.Tensor(view_211, add_88);  view_211 = add_88 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_92 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_91, getitem_41);  add_91 = getitem_41 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, primals_22)
        add_93 = torch.ops.aten.add.Tensor(mul_82, primals_23);  mul_82 = None
        view_212 = torch.ops.aten.view.default(add_93, [512, 4096])
        addmm_61 = torch.ops.aten.addmm.default(primals_9, view_212, permute_1)
        view_213 = torch.ops.aten.view.default(addmm_61, [1, 512, 4096]);  addmm_61 = None
        addmm_62 = torch.ops.aten.addmm.default(primals_11, view_212, permute_2)
        view_215 = torch.ops.aten.view.default(addmm_62, [1, 512, 4096]);  addmm_62 = None
        addmm_63 = torch.ops.aten.addmm.default(primals_13, view_212, permute_3)
        view_217 = torch.ops.aten.view.default(addmm_63, [1, 512, 4096]);  addmm_63 = None
        view_218 = torch.ops.aten.view.default(view_213, [1, 512, 64, 64]);  view_213 = None
        permute_114 = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        view_219 = torch.ops.aten.view.default(view_215, [1, 512, 64, 64]);  view_215 = None
        permute_115 = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None
        view_220 = torch.ops.aten.view.default(view_217, [1, 512, 64, 64]);  view_217 = None
        permute_116 = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        permute_117 = torch.ops.aten.permute.default(permute_115, [0, 1, 3, 2]);  permute_115 = None
        expand_41 = torch.ops.aten.expand.default(permute_114, [1, 64, 512, 64]);  permute_114 = None
        view_221 = torch.ops.aten.view.default(expand_41, [64, 512, 64]);  expand_41 = None
        expand_42 = torch.ops.aten.expand.default(permute_117, [1, 64, 64, 512]);  permute_117 = None
        view_222 = torch.ops.aten.view.default(expand_42, [64, 64, 512]);  expand_42 = None
        bmm_20 = torch.ops.aten.bmm.default(view_221, view_222)
        view_223 = torch.ops.aten.view.default(bmm_20, [1, 64, 512, 512]);  bmm_20 = None
        div_20 = torch.ops.aten.div.Tensor(view_223, 8.0);  view_223 = None
        add_94 = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
        amax_10 = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_32 = torch.ops.aten.sub.Tensor(add_94, amax_10);  add_94 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        expand_43 = torch.ops.aten.expand.default(div_21, [1, 64, 512, 512])
        view_224 = torch.ops.aten.view.default(expand_43, [64, 512, 512]);  expand_43 = None
        expand_44 = torch.ops.aten.expand.default(permute_116, [1, 64, 512, 64]);  permute_116 = None
        view_225 = torch.ops.aten.view.default(expand_44, [64, 512, 64]);  expand_44 = None
        bmm_21 = torch.ops.aten.bmm.default(view_224, view_225)
        view_226 = torch.ops.aten.view.default(bmm_21, [1, 64, 512, 64]);  bmm_21 = None
        permute_118 = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_10 = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(clone_10, [1, 512, 4096]);  clone_10 = None
        view_227 = torch.ops.aten.view.default(_unsafe_view_10, [512, 4096]);  _unsafe_view_10 = None
        addmm_64 = torch.ops.aten.addmm.default(primals_15, view_227, permute_9)
        view_228 = torch.ops.aten.view.default(addmm_64, [1, 512, 4096]);  addmm_64 = None
        add_95 = torch.ops.aten.add.Tensor(add_93, view_228);  add_93 = view_228 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_95, getitem_43);  add_95 = getitem_43 = None
        mul_83 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, primals_16)
        add_97 = torch.ops.aten.add.Tensor(mul_84, primals_17);  mul_84 = None
        view_229 = torch.ops.aten.view.default(add_97, [512, 4096])
        addmm_65 = torch.ops.aten.addmm.default(primals_19, view_229, permute_10)
        view_230 = torch.ops.aten.view.default(addmm_65, [1, 512, 16384])
        mul_85 = torch.ops.aten.mul.Tensor(view_230, 0.5)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(view_230, 3.0)
        mul_86 = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_98 = torch.ops.aten.add.Tensor(view_230, mul_86);  view_230 = mul_86 = None
        mul_87 = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_87);  mul_87 = None
        add_99 = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_88 = torch.ops.aten.mul.Tensor(mul_85, add_99);  mul_85 = add_99 = None
        view_231 = torch.ops.aten.view.default(mul_88, [512, 16384]);  mul_88 = None
        addmm_66 = torch.ops.aten.addmm.default(primals_21, view_231, permute_11)
        view_232 = torch.ops.aten.view.default(addmm_66, [1, 512, 4096]);  addmm_66 = None
        add_100 = torch.ops.aten.add.Tensor(view_232, add_97);  view_232 = add_97 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_101 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_100, getitem_45);  add_100 = getitem_45 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, primals_22)
        add_102 = torch.ops.aten.add.Tensor(mul_90, primals_23);  mul_90 = None
        view_233 = torch.ops.aten.view.default(add_102, [512, 4096])
        addmm_67 = torch.ops.aten.addmm.default(primals_9, view_233, permute_1);  primals_9 = None
        view_234 = torch.ops.aten.view.default(addmm_67, [1, 512, 4096]);  addmm_67 = None
        addmm_68 = torch.ops.aten.addmm.default(primals_11, view_233, permute_2);  primals_11 = None
        view_236 = torch.ops.aten.view.default(addmm_68, [1, 512, 4096]);  addmm_68 = None
        addmm_69 = torch.ops.aten.addmm.default(primals_13, view_233, permute_3);  primals_13 = None
        view_238 = torch.ops.aten.view.default(addmm_69, [1, 512, 4096]);  addmm_69 = None
        view_239 = torch.ops.aten.view.default(view_234, [1, 512, 64, 64]);  view_234 = None
        permute_125 = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
        view_240 = torch.ops.aten.view.default(view_236, [1, 512, 64, 64]);  view_236 = None
        permute_126 = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        view_241 = torch.ops.aten.view.default(view_238, [1, 512, 64, 64]);  view_238 = None
        permute_127 = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
        permute_128 = torch.ops.aten.permute.default(permute_126, [0, 1, 3, 2]);  permute_126 = None
        expand_45 = torch.ops.aten.expand.default(permute_125, [1, 64, 512, 64]);  permute_125 = None
        view_242 = torch.ops.aten.view.default(expand_45, [64, 512, 64]);  expand_45 = None
        expand_46 = torch.ops.aten.expand.default(permute_128, [1, 64, 64, 512]);  permute_128 = None
        view_243 = torch.ops.aten.view.default(expand_46, [64, 64, 512]);  expand_46 = None
        bmm_22 = torch.ops.aten.bmm.default(view_242, view_243)
        view_244 = torch.ops.aten.view.default(bmm_22, [1, 64, 512, 512]);  bmm_22 = None
        div_22 = torch.ops.aten.div.Tensor(view_244, 8.0);  view_244 = None
        add_103 = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
        amax_11 = torch.ops.aten.amax.default(add_103, [-1], True)
        sub_35 = torch.ops.aten.sub.Tensor(add_103, amax_11);  add_103 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        expand_47 = torch.ops.aten.expand.default(div_23, [1, 64, 512, 512])
        view_245 = torch.ops.aten.view.default(expand_47, [64, 512, 512]);  expand_47 = None
        expand_48 = torch.ops.aten.expand.default(permute_127, [1, 64, 512, 64]);  permute_127 = None
        view_246 = torch.ops.aten.view.default(expand_48, [64, 512, 64]);  expand_48 = None
        bmm_23 = torch.ops.aten.bmm.default(view_245, view_246)
        view_247 = torch.ops.aten.view.default(bmm_23, [1, 64, 512, 64]);  bmm_23 = None
        permute_129 = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        clone_11 = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        _unsafe_view_11 = torch.ops.aten._unsafe_view.default(clone_11, [1, 512, 4096]);  clone_11 = None
        view_248 = torch.ops.aten.view.default(_unsafe_view_11, [512, 4096]);  _unsafe_view_11 = None
        addmm_70 = torch.ops.aten.addmm.default(primals_15, view_248, permute_9);  primals_15 = None
        view_249 = torch.ops.aten.view.default(addmm_70, [1, 512, 4096]);  addmm_70 = None
        add_104 = torch.ops.aten.add.Tensor(add_102, view_249);  add_102 = view_249 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_105 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_104, getitem_47);  add_104 = getitem_47 = None
        mul_91 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_91, primals_16)
        add_106 = torch.ops.aten.add.Tensor(mul_92, primals_17);  mul_92 = primals_17 = None
        view_250 = torch.ops.aten.view.default(add_106, [512, 4096])
        addmm_71 = torch.ops.aten.addmm.default(primals_19, view_250, permute_10);  primals_19 = None
        view_251 = torch.ops.aten.view.default(addmm_71, [1, 512, 16384])
        mul_93 = torch.ops.aten.mul.Tensor(view_251, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
        mul_94 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_107 = torch.ops.aten.add.Tensor(view_251, mul_94);  view_251 = mul_94 = None
        mul_95 = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_95);  mul_95 = None
        add_108 = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_96 = torch.ops.aten.mul.Tensor(mul_93, add_108);  mul_93 = add_108 = None
        view_252 = torch.ops.aten.view.default(mul_96, [512, 16384]);  mul_96 = None
        addmm_72 = torch.ops.aten.addmm.default(primals_21, view_252, permute_11);  primals_21 = None
        view_253 = torch.ops.aten.view.default(addmm_72, [1, 512, 4096]);  addmm_72 = None
        add_109 = torch.ops.aten.add.Tensor(view_253, add_106);  view_253 = add_106 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_109, getitem_49);  add_109 = getitem_49 = None
        mul_97 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, primals_22)
        add_111 = torch.ops.aten.add.Tensor(mul_98, primals_23);  mul_98 = primals_23 = None
        permute_133 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        view_254 = torch.ops.aten.view.default(add_111, [512, 4096]);  add_111 = None
        addmm_73 = torch.ops.aten.addmm.default(primals_25, view_254, permute_133);  primals_25 = None
        view_255 = torch.ops.aten.view.default(addmm_73, [1, 512, 2]);  addmm_73 = None
        split = torch.ops.aten.split.Tensor(view_255, 1, -1);  view_255 = None
        getitem_50 = split[0]
        getitem_51 = split[1];  split = None
        squeeze = torch.ops.aten.squeeze.dim(getitem_50, -1);  getitem_50 = None
        clone_12 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        squeeze_1 = torch.ops.aten.squeeze.dim(getitem_51, -1);  getitem_51 = None
        clone_13 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(primals_29, 0);  primals_29 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(primals_30, 0);  primals_30 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
        amax_12 = torch.ops.aten.amax.default(clone_12, [1], True)
        sub_38 = torch.ops.aten.sub.Tensor(clone_12, amax_12);  amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_38)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(clamp_max, 1)
        gather = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2)
        squeeze_2 = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        ne = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(ne, neg, scalar_tensor);  neg = None
        sum_14 = torch.ops.aten.sum.default(ne)
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15 = torch.ops.aten.sum.default(where);  where = None
        div_24 = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
        amax_13 = torch.ops.aten.amax.default(clone_13, [1], True)
        sub_40 = torch.ops.aten.sub.Tensor(clone_13, amax_13);  amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_40)
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
        log_1 = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_41 = torch.ops.aten.sub.Tensor(sub_40, log_1);  sub_40 = log_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(clamp_max_1, 1)
        gather_1 = torch.ops.aten.gather.default(sub_41, 1, unsqueeze_3)
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        ne_2 = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
        where_1 = torch.ops.aten.where.self(ne_2, neg_1, scalar_tensor);  neg_1 = scalar_tensor = None
        sum_17 = torch.ops.aten.sum.default(ne_2)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
        sum_18 = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_25 = torch.ops.aten.div.Tensor(sum_18, convert_element_type_1);  sum_18 = convert_element_type_1 = None
        add_112 = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None
        div_26 = torch.ops.aten.div.Tensor(add_112, 2);  add_112 = None
        permute_134 = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        div_30 = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None
        permute_138 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_142 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        div_31 = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None
        permute_146 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_151 = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
        permute_152 = torch.ops.aten.permute.default(view_246, [0, 2, 1]);  view_246 = None
        permute_153 = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
        permute_154 = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
        permute_159 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_163 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_167 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        div_33 = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None
        div_34 = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None
        permute_184 = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
        permute_185 = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
        permute_186 = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
        permute_187 = torch.ops.aten.permute.default(view_222, [0, 2, 1]);  view_222 = None
        div_36 = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None
        div_37 = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None
        permute_217 = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
        permute_218 = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
        permute_219 = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
        permute_220 = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
        div_39 = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None
        div_40 = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None
        permute_250 = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
        permute_251 = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        permute_252 = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_253 = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
        div_42 = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None
        div_43 = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None
        permute_283 = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
        permute_284 = torch.ops.aten.permute.default(view_162, [0, 2, 1]);  view_162 = None
        permute_285 = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
        permute_286 = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
        div_45 = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None
        div_46 = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None
        permute_316 = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
        permute_317 = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_318 = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
        permute_319 = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
        div_48 = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None
        div_49 = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None
        permute_349 = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_350 = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
        permute_351 = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
        permute_352 = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
        div_51 = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None
        div_52 = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None
        permute_382 = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
        permute_383 = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_384 = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
        permute_385 = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
        div_54 = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None
        div_55 = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None
        permute_415 = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_416 = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_417 = torch.ops.aten.permute.default(view_74, [0, 2, 1]);  view_74 = None
        permute_418 = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        div_57 = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None
        div_58 = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None
        permute_448 = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_449 = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_450 = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_451 = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
        div_60 = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None
        div_61 = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None
        permute_481 = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_482 = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        permute_483 = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_484 = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        div_63 = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None
        div_64 = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None
        permute_514 = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_515 = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_516 = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_517 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_534 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        div_66 = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(slice_2, torch.float32);  slice_2 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(primals_28, torch.float32);  primals_28 = None
        return [div_26, clone_12, clone_13, primals_4, primals_16, primals_22, mul_1, view, view_2, div_1, view_17, mul_3, view_19, addmm_5, tanh, view_21, mul_9, view_23, div_3, view_38, mul_11, view_40, addmm_11, tanh_1, view_42, mul_17, view_44, div_5, view_59, mul_19, view_61, addmm_17, tanh_2, view_63, mul_25, view_65, div_7, view_80, mul_27, view_82, addmm_23, tanh_3, view_84, mul_33, view_86, div_9, view_101, mul_35, view_103, addmm_29, tanh_4, view_105, mul_41, view_107, div_11, view_122, mul_43, view_124, addmm_35, tanh_5, view_126, mul_49, view_128, div_13, view_143, mul_51, view_145, addmm_41, tanh_6, view_147, mul_57, view_149, div_15, view_164, mul_59, view_166, addmm_47, tanh_7, view_168, mul_65, view_170, div_17, view_185, mul_67, view_187, addmm_53, tanh_8, view_189, mul_73, view_191, div_19, view_206, mul_75, view_208, addmm_59, tanh_9, view_210, mul_81, view_212, div_21, view_227, mul_83, view_229, addmm_65, tanh_10, view_231, mul_89, view_233, div_23, view_248, mul_91, view_250, addmm_71, tanh_11, view_252, mul_97, view_254, sub_39, unsqueeze_2, ne, sub_41, unsqueeze_3, ne_2, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, permute_186, permute_187, div_36, div_37, permute_217, permute_218, permute_219, permute_220, div_39, div_40, permute_250, permute_251, permute_252, permute_253, div_42, div_43, permute_283, permute_284, permute_285, permute_286, div_45, div_46, permute_316, permute_317, permute_318, permute_319, div_48, div_49, permute_349, permute_350, permute_351, permute_352, div_51, div_52, permute_382, permute_383, permute_384, permute_385, div_54, div_55, permute_415, permute_416, permute_417, permute_418, div_57, div_58, permute_448, permute_449, permute_450, permute_451, div_60, div_61, permute_481, permute_482, permute_483, permute_484, div_63, div_64, permute_514, permute_515, permute_516, permute_517, permute_534, div_66, convert_element_type_2, convert_element_type_4, convert_element_type_6]
        
args = [((30000, 128), (128, 1), torch.float32, 'cuda'), ((2, 128), (128, 1), torch.float32, 'cuda'), ((512, 128), (128, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((4096, 128), (128, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((16384, 4096), (4096, 1), torch.float32, 'cuda'), ((16384,), (1,), torch.float32, 'cuda'), ((4096, 16384), (16384, 1), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((2, 4096), (4096, 1), torch.float32, 'cuda'), ((2,), (1,), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.int64, 'cuda'), ((1, 512), (512, 1), torch.int64, 'cuda'), ((1, 512), (512, 1), torch.int64, 'cuda'), ((1,), (1,), torch.int64, 'cuda'), ((1,), (1,), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
