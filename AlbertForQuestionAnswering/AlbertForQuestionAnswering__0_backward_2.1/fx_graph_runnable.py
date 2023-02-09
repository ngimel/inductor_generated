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
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x12\x00\x00\x00use_dynamic_shapesq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x03\x00\x00\x00cseq\x08\x88X\x10\x00\x00\x00max_dist_from_bwq\tK\x03X\x0b\x00\x00\x00debug_jointq\n\x89X\x0c\x00\x00\x00debug_graphsq\x0b\x89X\x11\x00\x00\x00debug_partitionerq\x0c\x89X\t\x00\x00\x00log_levelq\rK\x14u.')


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

    
    
    def forward(self, primals_4, primals_16, primals_22, mul_1, view, view_2, div_1, view_17, mul_3, view_19, addmm_5, tanh, view_21, mul_9, view_23, div_3, view_38, mul_11, view_40, addmm_11, tanh_1, view_42, mul_17, view_44, div_5, view_59, mul_19, view_61, addmm_17, tanh_2, view_63, mul_25, view_65, div_7, view_80, mul_27, view_82, addmm_23, tanh_3, view_84, mul_33, view_86, div_9, view_101, mul_35, view_103, addmm_29, tanh_4, view_105, mul_41, view_107, div_11, view_122, mul_43, view_124, addmm_35, tanh_5, view_126, mul_49, view_128, div_13, view_143, mul_51, view_145, addmm_41, tanh_6, view_147, mul_57, view_149, div_15, view_164, mul_59, view_166, addmm_47, tanh_7, view_168, mul_65, view_170, div_17, view_185, mul_67, view_187, addmm_53, tanh_8, view_189, mul_73, view_191, div_19, view_206, mul_75, view_208, addmm_59, tanh_9, view_210, mul_81, view_212, div_21, view_227, mul_83, view_229, addmm_65, tanh_10, view_231, mul_89, view_233, div_23, view_248, mul_91, view_250, addmm_71, tanh_11, view_252, mul_97, view_254, sub_39, unsqueeze_2, ne, sub_41, unsqueeze_3, ne_2, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, permute_186, permute_187, div_36, div_37, permute_217, permute_218, permute_219, permute_220, div_39, div_40, permute_250, permute_251, permute_252, permute_253, div_42, div_43, permute_283, permute_284, permute_285, permute_286, div_45, div_46, permute_316, permute_317, permute_318, permute_319, div_48, div_49, permute_349, permute_350, permute_351, permute_352, div_51, div_52, permute_382, permute_383, permute_384, permute_385, div_54, div_55, permute_415, permute_416, permute_417, permute_418, div_57, div_58, permute_448, permute_449, permute_450, permute_451, div_60, div_61, permute_481, permute_482, permute_483, permute_484, div_63, div_64, permute_514, permute_515, permute_516, permute_517, permute_534, div_66, convert_element_type_2, convert_element_type_4, convert_element_type_6, tangents_1, tangents_2, tangents_3):
        view_20 = torch.ops.aten.view.default(addmm_5, [1, 512, 16384]);  addmm_5 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_20, 0.5)
        add_9 = torch.ops.aten.add.Tensor(tanh, 1.0)
        view_41 = torch.ops.aten.view.default(addmm_11, [1, 512, 16384]);  addmm_11 = None
        mul_13 = torch.ops.aten.mul.Tensor(view_41, 0.5)
        add_18 = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        view_62 = torch.ops.aten.view.default(addmm_17, [1, 512, 16384]);  addmm_17 = None
        mul_21 = torch.ops.aten.mul.Tensor(view_62, 0.5)
        add_27 = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        view_83 = torch.ops.aten.view.default(addmm_23, [1, 512, 16384]);  addmm_23 = None
        mul_29 = torch.ops.aten.mul.Tensor(view_83, 0.5)
        add_36 = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        view_104 = torch.ops.aten.view.default(addmm_29, [1, 512, 16384]);  addmm_29 = None
        mul_37 = torch.ops.aten.mul.Tensor(view_104, 0.5)
        add_45 = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        view_125 = torch.ops.aten.view.default(addmm_35, [1, 512, 16384]);  addmm_35 = None
        mul_45 = torch.ops.aten.mul.Tensor(view_125, 0.5)
        add_54 = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        view_146 = torch.ops.aten.view.default(addmm_41, [1, 512, 16384]);  addmm_41 = None
        mul_53 = torch.ops.aten.mul.Tensor(view_146, 0.5)
        add_63 = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        view_167 = torch.ops.aten.view.default(addmm_47, [1, 512, 16384]);  addmm_47 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_167, 0.5)
        add_72 = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        view_188 = torch.ops.aten.view.default(addmm_53, [1, 512, 16384]);  addmm_53 = None
        mul_69 = torch.ops.aten.mul.Tensor(view_188, 0.5)
        add_81 = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        view_209 = torch.ops.aten.view.default(addmm_59, [1, 512, 16384]);  addmm_59 = None
        mul_77 = torch.ops.aten.mul.Tensor(view_209, 0.5)
        add_90 = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        view_230 = torch.ops.aten.view.default(addmm_65, [1, 512, 16384]);  addmm_65 = None
        mul_85 = torch.ops.aten.mul.Tensor(view_230, 0.5)
        add_99 = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        view_251 = torch.ops.aten.view.default(addmm_71, [1, 512, 16384]);  addmm_71 = None
        mul_93 = torch.ops.aten.mul.Tensor(view_251, 0.5)
        add_108 = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        sum_14 = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_17 = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
        div_27 = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
        div_28 = torch.ops.aten.div.Tensor(div_27, convert_element_type_1);  convert_element_type_1 = None
        full_like = torch.ops.aten.full_like.default(sub_41, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False, memory_format = torch.preserve_format)
        scatter = torch.ops.aten.scatter.value(full_like, 1, unsqueeze_3, -1.0);  full_like = None
        ne_4 = torch.ops.aten.ne.Scalar(unsqueeze_3, 512);  unsqueeze_3 = None
        where_2 = torch.ops.aten.where.self(ne_4, div_28, scalar_tensor);  ne_4 = div_28 = None
        mul_99 = torch.ops.aten.mul.Tensor(scatter, where_2);  scatter = where_2 = None
        exp_14 = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_99, [1], True)
        mul_100 = torch.ops.aten.mul.Tensor(exp_14, sum_19);  exp_14 = sum_19 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        add_113 = torch.ops.aten.add.Tensor(tangents_3, sub_42);  tangents_3 = sub_42 = None
        div_29 = torch.ops.aten.div.Tensor(div_27, convert_element_type);  div_27 = convert_element_type = None
        full_like_1 = torch.ops.aten.full_like.default(sub_39, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False, memory_format = torch.preserve_format)
        scatter_1 = torch.ops.aten.scatter.value(full_like_1, 1, unsqueeze_2, -1.0);  full_like_1 = None
        ne_5 = torch.ops.aten.ne.Scalar(unsqueeze_2, 512);  unsqueeze_2 = None
        where_3 = torch.ops.aten.where.self(ne_5, div_29, scalar_tensor);  ne_5 = div_29 = None
        mul_101 = torch.ops.aten.mul.Tensor(scatter_1, where_3);  scatter_1 = where_3 = None
        exp_15 = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_101, [1], True)
        mul_102 = torch.ops.aten.mul.Tensor(exp_15, sum_20);  exp_15 = sum_20 = None
        sub_43 = torch.ops.aten.sub.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
        add_114 = torch.ops.aten.add.Tensor(tangents_2, sub_43);  tangents_2 = sub_43 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(add_113, 2);  add_113 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(add_114, 2);  add_114 = None
        cat = torch.ops.aten.cat.default([unsqueeze_7, unsqueeze_6], 2);  unsqueeze_7 = unsqueeze_6 = None
        view_256 = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
        mm = torch.ops.aten.mm.default(view_256, permute_134);  permute_134 = None
        permute_135 = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_135, view_254);  permute_135 = view_254 = None
        permute_136 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
        view_257 = torch.ops.aten.view.default(sum_21, [2]);  sum_21 = None
        view_258 = torch.ops.aten.view.default(mm, [1, 512, 4096]);  mm = None
        permute_137 = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        mul_104 = torch.ops.aten.mul.Tensor(view_258, primals_22)
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, 4096)
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_104, [2], True)
        mul_106 = torch.ops.aten.mul.Tensor(mul_104, mul_97);  mul_104 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_106, [2], True);  mul_106 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_97, sum_23);  sum_23 = None
        sub_45 = torch.ops.aten.sub.Tensor(mul_105, sum_22);  mul_105 = sum_22 = None
        sub_46 = torch.ops.aten.sub.Tensor(sub_45, mul_107);  sub_45 = mul_107 = None
        mul_108 = torch.ops.aten.mul.Tensor(div_30, sub_46);  div_30 = sub_46 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_258, mul_97);  mul_97 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_109, [0, 1]);  mul_109 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(view_258, [0, 1]);  view_258 = None
        view_259 = torch.ops.aten.view.default(mul_108, [512, 4096])
        mm_2 = torch.ops.aten.mm.default(view_259, permute_138)
        permute_139 = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_139, view_252);  permute_139 = view_252 = None
        permute_140 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(view_259, [0], True);  view_259 = None
        view_260 = torch.ops.aten.view.default(sum_26, [4096]);  sum_26 = None
        view_261 = torch.ops.aten.view.default(mm_2, [1, 512, 16384]);  mm_2 = None
        permute_141 = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mul_110 = torch.ops.aten.mul.Tensor(view_261, mul_93);  mul_93 = None
        mul_111 = torch.ops.aten.mul.Tensor(view_261, add_108);  view_261 = add_108 = None
        mul_112 = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_47 = torch.ops.aten.sub.Tensor(1, mul_112);  mul_112 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_110, sub_47);  mul_110 = sub_47 = None
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, 0.7978845608028654);  mul_113 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, 0.044715)
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(view_251, 2.0);  view_251 = None
        mul_116 = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
        add_115 = torch.ops.aten.add.Tensor(mul_114, mul_117);  mul_114 = mul_117 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_111, 0.5);  mul_111 = None
        add_116 = torch.ops.aten.add.Tensor(add_115, mul_118);  add_115 = mul_118 = None
        view_262 = torch.ops.aten.view.default(add_116, [512, 16384]);  add_116 = None
        mm_4 = torch.ops.aten.mm.default(view_262, permute_142)
        permute_143 = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_143, view_250);  permute_143 = view_250 = None
        permute_144 = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(view_262, [0], True);  view_262 = None
        view_263 = torch.ops.aten.view.default(sum_27, [16384]);  sum_27 = None
        view_264 = torch.ops.aten.view.default(mm_4, [1, 512, 4096]);  mm_4 = None
        add_117 = torch.ops.aten.add.Tensor(mul_108, view_264);  mul_108 = view_264 = None
        permute_145 = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mul_120 = torch.ops.aten.mul.Tensor(add_117, primals_16)
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, 4096)
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
        mul_122 = torch.ops.aten.mul.Tensor(mul_120, mul_91);  mul_120 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_91, sum_29);  sum_29 = None
        sub_49 = torch.ops.aten.sub.Tensor(mul_121, sum_28);  mul_121 = sum_28 = None
        sub_50 = torch.ops.aten.sub.Tensor(sub_49, mul_123);  sub_49 = mul_123 = None
        mul_124 = torch.ops.aten.mul.Tensor(div_31, sub_50);  div_31 = sub_50 = None
        mul_125 = torch.ops.aten.mul.Tensor(add_117, mul_91);  mul_91 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
        view_265 = torch.ops.aten.view.default(mul_124, [512, 4096])
        mm_6 = torch.ops.aten.mm.default(view_265, permute_146)
        permute_147 = torch.ops.aten.permute.default(view_265, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_147, view_248);  permute_147 = view_248 = None
        permute_148 = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
        view_266 = torch.ops.aten.view.default(sum_32, [4096]);  sum_32 = None
        view_267 = torch.ops.aten.view.default(mm_6, [1, 512, 4096]);  mm_6 = None
        permute_149 = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        view_268 = torch.ops.aten.view.default(view_267, [1, 512, 64, 64]);  view_267 = None
        permute_150 = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        view_269 = torch.ops.aten.view.default(permute_150, [64, 512, 64]);  permute_150 = None
        bmm_24 = torch.ops.aten.bmm.default(permute_151, view_269);  permute_151 = None
        bmm_25 = torch.ops.aten.bmm.default(view_269, permute_152);  view_269 = permute_152 = None
        view_270 = torch.ops.aten.view.default(bmm_24, [1, 64, 512, 64]);  bmm_24 = None
        view_271 = torch.ops.aten.view.default(bmm_25, [1, 64, 512, 512]);  bmm_25 = None
        mul_126 = torch.ops.aten.mul.Tensor(view_271, div_23);  view_271 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_126, [-1], True)
        mul_127 = torch.ops.aten.mul.Tensor(div_23, sum_33);  div_23 = sum_33 = None
        sub_51 = torch.ops.aten.sub.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
        div_32 = torch.ops.aten.div.Tensor(sub_51, 8.0);  sub_51 = None
        view_272 = torch.ops.aten.view.default(div_32, [64, 512, 512]);  div_32 = None
        bmm_26 = torch.ops.aten.bmm.default(permute_153, view_272);  permute_153 = None
        bmm_27 = torch.ops.aten.bmm.default(view_272, permute_154);  view_272 = permute_154 = None
        view_273 = torch.ops.aten.view.default(bmm_26, [1, 64, 64, 512]);  bmm_26 = None
        view_274 = torch.ops.aten.view.default(bmm_27, [1, 64, 512, 64]);  bmm_27 = None
        permute_155 = torch.ops.aten.permute.default(view_273, [0, 1, 3, 2]);  view_273 = None
        permute_156 = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_14 = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        _unsafe_view_12 = torch.ops.aten._unsafe_view.default(clone_14, [1, 512, 4096]);  clone_14 = None
        permute_157 = torch.ops.aten.permute.default(permute_155, [0, 2, 1, 3]);  permute_155 = None
        view_275 = torch.ops.aten.view.default(permute_157, [1, 512, 4096]);  permute_157 = None
        permute_158 = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_15 = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        _unsafe_view_13 = torch.ops.aten._unsafe_view.default(clone_15, [1, 512, 4096]);  clone_15 = None
        view_276 = torch.ops.aten.view.default(_unsafe_view_12, [512, 4096]);  _unsafe_view_12 = None
        mm_8 = torch.ops.aten.mm.default(view_276, permute_159)
        permute_160 = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_160, view_233);  permute_160 = None
        permute_161 = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        view_277 = torch.ops.aten.view.default(sum_34, [4096]);  sum_34 = None
        view_278 = torch.ops.aten.view.default(mm_8, [1, 512, 4096]);  mm_8 = None
        add_118 = torch.ops.aten.add.Tensor(mul_124, view_278);  mul_124 = view_278 = None
        permute_162 = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        view_279 = torch.ops.aten.view.default(view_275, [512, 4096]);  view_275 = None
        mm_10 = torch.ops.aten.mm.default(view_279, permute_163)
        permute_164 = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_164, view_233);  permute_164 = None
        permute_165 = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        view_280 = torch.ops.aten.view.default(sum_35, [4096]);  sum_35 = None
        view_281 = torch.ops.aten.view.default(mm_10, [1, 512, 4096]);  mm_10 = None
        add_119 = torch.ops.aten.add.Tensor(add_118, view_281);  add_118 = view_281 = None
        permute_166 = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        view_282 = torch.ops.aten.view.default(_unsafe_view_13, [512, 4096]);  _unsafe_view_13 = None
        mm_12 = torch.ops.aten.mm.default(view_282, permute_167)
        permute_168 = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_13 = torch.ops.aten.mm.default(permute_168, view_233);  permute_168 = view_233 = None
        permute_169 = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        view_283 = torch.ops.aten.view.default(sum_36, [4096]);  sum_36 = None
        view_284 = torch.ops.aten.view.default(mm_12, [1, 512, 4096]);  mm_12 = None
        add_120 = torch.ops.aten.add.Tensor(add_119, view_284);  add_119 = view_284 = None
        permute_170 = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        mul_129 = torch.ops.aten.mul.Tensor(add_120, primals_22)
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, 4096)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
        mul_131 = torch.ops.aten.mul.Tensor(mul_129, mul_89);  mul_129 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_89, sum_38);  sum_38 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_130, sum_37);  mul_130 = sum_37 = None
        sub_54 = torch.ops.aten.sub.Tensor(sub_53, mul_132);  sub_53 = mul_132 = None
        mul_133 = torch.ops.aten.mul.Tensor(div_33, sub_54);  div_33 = sub_54 = None
        mul_134 = torch.ops.aten.mul.Tensor(add_120, mul_89);  mul_89 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
        add_121 = torch.ops.aten.add.Tensor(sum_24, sum_39);  sum_24 = sum_39 = None
        add_122 = torch.ops.aten.add.Tensor(sum_25, sum_40);  sum_25 = sum_40 = None
        view_285 = torch.ops.aten.view.default(mul_133, [512, 4096])
        mm_14 = torch.ops.aten.mm.default(view_285, permute_138)
        permute_172 = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_15 = torch.ops.aten.mm.default(permute_172, view_231);  permute_172 = view_231 = None
        permute_173 = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        view_286 = torch.ops.aten.view.default(sum_41, [4096]);  sum_41 = None
        add_123 = torch.ops.aten.add.Tensor(view_260, view_286);  view_260 = view_286 = None
        view_287 = torch.ops.aten.view.default(mm_14, [1, 512, 16384]);  mm_14 = None
        permute_174 = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        add_124 = torch.ops.aten.add.Tensor(permute_141, permute_174);  permute_141 = permute_174 = None
        mul_135 = torch.ops.aten.mul.Tensor(view_287, mul_85);  mul_85 = None
        mul_136 = torch.ops.aten.mul.Tensor(view_287, add_99);  view_287 = add_99 = None
        mul_137 = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_55 = torch.ops.aten.sub.Tensor(1, mul_137);  mul_137 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_135, sub_55);  mul_135 = sub_55 = None
        mul_139 = torch.ops.aten.mul.Tensor(mul_138, 0.7978845608028654);  mul_138 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, 0.044715)
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(view_230, 2.0);  view_230 = None
        mul_141 = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_142 = torch.ops.aten.mul.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        add_125 = torch.ops.aten.add.Tensor(mul_139, mul_142);  mul_139 = mul_142 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_136, 0.5);  mul_136 = None
        add_126 = torch.ops.aten.add.Tensor(add_125, mul_143);  add_125 = mul_143 = None
        view_288 = torch.ops.aten.view.default(add_126, [512, 16384]);  add_126 = None
        mm_16 = torch.ops.aten.mm.default(view_288, permute_142)
        permute_176 = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_17 = torch.ops.aten.mm.default(permute_176, view_229);  permute_176 = view_229 = None
        permute_177 = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        view_289 = torch.ops.aten.view.default(sum_42, [16384]);  sum_42 = None
        add_127 = torch.ops.aten.add.Tensor(view_263, view_289);  view_263 = view_289 = None
        view_290 = torch.ops.aten.view.default(mm_16, [1, 512, 4096]);  mm_16 = None
        add_128 = torch.ops.aten.add.Tensor(mul_133, view_290);  mul_133 = view_290 = None
        permute_178 = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        add_129 = torch.ops.aten.add.Tensor(permute_145, permute_178);  permute_145 = permute_178 = None
        mul_145 = torch.ops.aten.mul.Tensor(add_128, primals_16)
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, 4096)
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
        mul_147 = torch.ops.aten.mul.Tensor(mul_145, mul_83);  mul_145 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_83, sum_44);  sum_44 = None
        sub_57 = torch.ops.aten.sub.Tensor(mul_146, sum_43);  mul_146 = sum_43 = None
        sub_58 = torch.ops.aten.sub.Tensor(sub_57, mul_148);  sub_57 = mul_148 = None
        mul_149 = torch.ops.aten.mul.Tensor(div_34, sub_58);  div_34 = sub_58 = None
        mul_150 = torch.ops.aten.mul.Tensor(add_128, mul_83);  mul_83 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
        add_130 = torch.ops.aten.add.Tensor(sum_30, sum_45);  sum_30 = sum_45 = None
        add_131 = torch.ops.aten.add.Tensor(sum_31, sum_46);  sum_31 = sum_46 = None
        view_291 = torch.ops.aten.view.default(mul_149, [512, 4096])
        mm_18 = torch.ops.aten.mm.default(view_291, permute_146)
        permute_180 = torch.ops.aten.permute.default(view_291, [1, 0])
        mm_19 = torch.ops.aten.mm.default(permute_180, view_227);  permute_180 = view_227 = None
        permute_181 = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
        view_292 = torch.ops.aten.view.default(sum_47, [4096]);  sum_47 = None
        add_132 = torch.ops.aten.add.Tensor(view_266, view_292);  view_266 = view_292 = None
        view_293 = torch.ops.aten.view.default(mm_18, [1, 512, 4096]);  mm_18 = None
        permute_182 = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        add_133 = torch.ops.aten.add.Tensor(permute_149, permute_182);  permute_149 = permute_182 = None
        view_294 = torch.ops.aten.view.default(view_293, [1, 512, 64, 64]);  view_293 = None
        permute_183 = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        view_295 = torch.ops.aten.view.default(permute_183, [64, 512, 64]);  permute_183 = None
        bmm_28 = torch.ops.aten.bmm.default(permute_184, view_295);  permute_184 = None
        bmm_29 = torch.ops.aten.bmm.default(view_295, permute_185);  view_295 = permute_185 = None
        view_296 = torch.ops.aten.view.default(bmm_28, [1, 64, 512, 64]);  bmm_28 = None
        view_297 = torch.ops.aten.view.default(bmm_29, [1, 64, 512, 512]);  bmm_29 = None
        mul_151 = torch.ops.aten.mul.Tensor(view_297, div_21);  view_297 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(mul_151, [-1], True)
        mul_152 = torch.ops.aten.mul.Tensor(div_21, sum_48);  div_21 = sum_48 = None
        sub_59 = torch.ops.aten.sub.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        div_35 = torch.ops.aten.div.Tensor(sub_59, 8.0);  sub_59 = None
        view_298 = torch.ops.aten.view.default(div_35, [64, 512, 512]);  div_35 = None
        bmm_30 = torch.ops.aten.bmm.default(permute_186, view_298);  permute_186 = None
        bmm_31 = torch.ops.aten.bmm.default(view_298, permute_187);  view_298 = permute_187 = None
        view_299 = torch.ops.aten.view.default(bmm_30, [1, 64, 64, 512]);  bmm_30 = None
        view_300 = torch.ops.aten.view.default(bmm_31, [1, 64, 512, 64]);  bmm_31 = None
        permute_188 = torch.ops.aten.permute.default(view_299, [0, 1, 3, 2]);  view_299 = None
        permute_189 = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_16 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        _unsafe_view_14 = torch.ops.aten._unsafe_view.default(clone_16, [1, 512, 4096]);  clone_16 = None
        permute_190 = torch.ops.aten.permute.default(permute_188, [0, 2, 1, 3]);  permute_188 = None
        view_301 = torch.ops.aten.view.default(permute_190, [1, 512, 4096]);  permute_190 = None
        permute_191 = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_17 = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        _unsafe_view_15 = torch.ops.aten._unsafe_view.default(clone_17, [1, 512, 4096]);  clone_17 = None
        view_302 = torch.ops.aten.view.default(_unsafe_view_14, [512, 4096]);  _unsafe_view_14 = None
        mm_20 = torch.ops.aten.mm.default(view_302, permute_159)
        permute_193 = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_21 = torch.ops.aten.mm.default(permute_193, view_212);  permute_193 = None
        permute_194 = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        view_303 = torch.ops.aten.view.default(sum_49, [4096]);  sum_49 = None
        add_134 = torch.ops.aten.add.Tensor(view_277, view_303);  view_277 = view_303 = None
        view_304 = torch.ops.aten.view.default(mm_20, [1, 512, 4096]);  mm_20 = None
        add_135 = torch.ops.aten.add.Tensor(mul_149, view_304);  mul_149 = view_304 = None
        permute_195 = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        add_136 = torch.ops.aten.add.Tensor(permute_162, permute_195);  permute_162 = permute_195 = None
        view_305 = torch.ops.aten.view.default(view_301, [512, 4096]);  view_301 = None
        mm_22 = torch.ops.aten.mm.default(view_305, permute_163)
        permute_197 = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_23 = torch.ops.aten.mm.default(permute_197, view_212);  permute_197 = None
        permute_198 = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        view_306 = torch.ops.aten.view.default(sum_50, [4096]);  sum_50 = None
        add_137 = torch.ops.aten.add.Tensor(view_280, view_306);  view_280 = view_306 = None
        view_307 = torch.ops.aten.view.default(mm_22, [1, 512, 4096]);  mm_22 = None
        add_138 = torch.ops.aten.add.Tensor(add_135, view_307);  add_135 = view_307 = None
        permute_199 = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        add_139 = torch.ops.aten.add.Tensor(permute_166, permute_199);  permute_166 = permute_199 = None
        view_308 = torch.ops.aten.view.default(_unsafe_view_15, [512, 4096]);  _unsafe_view_15 = None
        mm_24 = torch.ops.aten.mm.default(view_308, permute_167)
        permute_201 = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_25 = torch.ops.aten.mm.default(permute_201, view_212);  permute_201 = view_212 = None
        permute_202 = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
        view_309 = torch.ops.aten.view.default(sum_51, [4096]);  sum_51 = None
        add_140 = torch.ops.aten.add.Tensor(view_283, view_309);  view_283 = view_309 = None
        view_310 = torch.ops.aten.view.default(mm_24, [1, 512, 4096]);  mm_24 = None
        add_141 = torch.ops.aten.add.Tensor(add_138, view_310);  add_138 = view_310 = None
        permute_203 = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        add_142 = torch.ops.aten.add.Tensor(permute_170, permute_203);  permute_170 = permute_203 = None
        mul_154 = torch.ops.aten.mul.Tensor(add_141, primals_22)
        mul_155 = torch.ops.aten.mul.Tensor(mul_154, 4096)
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
        mul_156 = torch.ops.aten.mul.Tensor(mul_154, mul_81);  mul_154 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_81, sum_53);  sum_53 = None
        sub_61 = torch.ops.aten.sub.Tensor(mul_155, sum_52);  mul_155 = sum_52 = None
        sub_62 = torch.ops.aten.sub.Tensor(sub_61, mul_157);  sub_61 = mul_157 = None
        mul_158 = torch.ops.aten.mul.Tensor(div_36, sub_62);  div_36 = sub_62 = None
        mul_159 = torch.ops.aten.mul.Tensor(add_141, mul_81);  mul_81 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
        add_143 = torch.ops.aten.add.Tensor(add_121, sum_54);  add_121 = sum_54 = None
        add_144 = torch.ops.aten.add.Tensor(add_122, sum_55);  add_122 = sum_55 = None
        view_311 = torch.ops.aten.view.default(mul_158, [512, 4096])
        mm_26 = torch.ops.aten.mm.default(view_311, permute_138)
        permute_205 = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_27 = torch.ops.aten.mm.default(permute_205, view_210);  permute_205 = view_210 = None
        permute_206 = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_312 = torch.ops.aten.view.default(sum_56, [4096]);  sum_56 = None
        add_145 = torch.ops.aten.add.Tensor(add_123, view_312);  add_123 = view_312 = None
        view_313 = torch.ops.aten.view.default(mm_26, [1, 512, 16384]);  mm_26 = None
        permute_207 = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        add_146 = torch.ops.aten.add.Tensor(add_124, permute_207);  add_124 = permute_207 = None
        mul_160 = torch.ops.aten.mul.Tensor(view_313, mul_77);  mul_77 = None
        mul_161 = torch.ops.aten.mul.Tensor(view_313, add_90);  view_313 = add_90 = None
        mul_162 = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_63 = torch.ops.aten.sub.Tensor(1, mul_162);  mul_162 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_160, sub_63);  mul_160 = sub_63 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, 0.7978845608028654);  mul_163 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 0.044715)
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(view_209, 2.0);  view_209 = None
        mul_166 = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        add_147 = torch.ops.aten.add.Tensor(mul_164, mul_167);  mul_164 = mul_167 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_161, 0.5);  mul_161 = None
        add_148 = torch.ops.aten.add.Tensor(add_147, mul_168);  add_147 = mul_168 = None
        view_314 = torch.ops.aten.view.default(add_148, [512, 16384]);  add_148 = None
        mm_28 = torch.ops.aten.mm.default(view_314, permute_142)
        permute_209 = torch.ops.aten.permute.default(view_314, [1, 0])
        mm_29 = torch.ops.aten.mm.default(permute_209, view_208);  permute_209 = view_208 = None
        permute_210 = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
        view_315 = torch.ops.aten.view.default(sum_57, [16384]);  sum_57 = None
        add_149 = torch.ops.aten.add.Tensor(add_127, view_315);  add_127 = view_315 = None
        view_316 = torch.ops.aten.view.default(mm_28, [1, 512, 4096]);  mm_28 = None
        add_150 = torch.ops.aten.add.Tensor(mul_158, view_316);  mul_158 = view_316 = None
        permute_211 = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        add_151 = torch.ops.aten.add.Tensor(add_129, permute_211);  add_129 = permute_211 = None
        mul_170 = torch.ops.aten.mul.Tensor(add_150, primals_16)
        mul_171 = torch.ops.aten.mul.Tensor(mul_170, 4096)
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
        mul_172 = torch.ops.aten.mul.Tensor(mul_170, mul_75);  mul_170 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_75, sum_59);  sum_59 = None
        sub_65 = torch.ops.aten.sub.Tensor(mul_171, sum_58);  mul_171 = sum_58 = None
        sub_66 = torch.ops.aten.sub.Tensor(sub_65, mul_173);  sub_65 = mul_173 = None
        mul_174 = torch.ops.aten.mul.Tensor(div_37, sub_66);  div_37 = sub_66 = None
        mul_175 = torch.ops.aten.mul.Tensor(add_150, mul_75);  mul_75 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
        add_152 = torch.ops.aten.add.Tensor(add_130, sum_60);  add_130 = sum_60 = None
        add_153 = torch.ops.aten.add.Tensor(add_131, sum_61);  add_131 = sum_61 = None
        view_317 = torch.ops.aten.view.default(mul_174, [512, 4096])
        mm_30 = torch.ops.aten.mm.default(view_317, permute_146)
        permute_213 = torch.ops.aten.permute.default(view_317, [1, 0])
        mm_31 = torch.ops.aten.mm.default(permute_213, view_206);  permute_213 = view_206 = None
        permute_214 = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
        view_318 = torch.ops.aten.view.default(sum_62, [4096]);  sum_62 = None
        add_154 = torch.ops.aten.add.Tensor(add_132, view_318);  add_132 = view_318 = None
        view_319 = torch.ops.aten.view.default(mm_30, [1, 512, 4096]);  mm_30 = None
        permute_215 = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        add_155 = torch.ops.aten.add.Tensor(add_133, permute_215);  add_133 = permute_215 = None
        view_320 = torch.ops.aten.view.default(view_319, [1, 512, 64, 64]);  view_319 = None
        permute_216 = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        view_321 = torch.ops.aten.view.default(permute_216, [64, 512, 64]);  permute_216 = None
        bmm_32 = torch.ops.aten.bmm.default(permute_217, view_321);  permute_217 = None
        bmm_33 = torch.ops.aten.bmm.default(view_321, permute_218);  view_321 = permute_218 = None
        view_322 = torch.ops.aten.view.default(bmm_32, [1, 64, 512, 64]);  bmm_32 = None
        view_323 = torch.ops.aten.view.default(bmm_33, [1, 64, 512, 512]);  bmm_33 = None
        mul_176 = torch.ops.aten.mul.Tensor(view_323, div_19);  view_323 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_176, [-1], True)
        mul_177 = torch.ops.aten.mul.Tensor(div_19, sum_63);  div_19 = sum_63 = None
        sub_67 = torch.ops.aten.sub.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        div_38 = torch.ops.aten.div.Tensor(sub_67, 8.0);  sub_67 = None
        view_324 = torch.ops.aten.view.default(div_38, [64, 512, 512]);  div_38 = None
        bmm_34 = torch.ops.aten.bmm.default(permute_219, view_324);  permute_219 = None
        bmm_35 = torch.ops.aten.bmm.default(view_324, permute_220);  view_324 = permute_220 = None
        view_325 = torch.ops.aten.view.default(bmm_34, [1, 64, 64, 512]);  bmm_34 = None
        view_326 = torch.ops.aten.view.default(bmm_35, [1, 64, 512, 64]);  bmm_35 = None
        permute_221 = torch.ops.aten.permute.default(view_325, [0, 1, 3, 2]);  view_325 = None
        permute_222 = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_18 = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        _unsafe_view_16 = torch.ops.aten._unsafe_view.default(clone_18, [1, 512, 4096]);  clone_18 = None
        permute_223 = torch.ops.aten.permute.default(permute_221, [0, 2, 1, 3]);  permute_221 = None
        view_327 = torch.ops.aten.view.default(permute_223, [1, 512, 4096]);  permute_223 = None
        permute_224 = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        clone_19 = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        _unsafe_view_17 = torch.ops.aten._unsafe_view.default(clone_19, [1, 512, 4096]);  clone_19 = None
        view_328 = torch.ops.aten.view.default(_unsafe_view_16, [512, 4096]);  _unsafe_view_16 = None
        mm_32 = torch.ops.aten.mm.default(view_328, permute_159)
        permute_226 = torch.ops.aten.permute.default(view_328, [1, 0])
        mm_33 = torch.ops.aten.mm.default(permute_226, view_191);  permute_226 = None
        permute_227 = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
        view_329 = torch.ops.aten.view.default(sum_64, [4096]);  sum_64 = None
        add_156 = torch.ops.aten.add.Tensor(add_134, view_329);  add_134 = view_329 = None
        view_330 = torch.ops.aten.view.default(mm_32, [1, 512, 4096]);  mm_32 = None
        add_157 = torch.ops.aten.add.Tensor(mul_174, view_330);  mul_174 = view_330 = None
        permute_228 = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
        add_158 = torch.ops.aten.add.Tensor(add_136, permute_228);  add_136 = permute_228 = None
        view_331 = torch.ops.aten.view.default(view_327, [512, 4096]);  view_327 = None
        mm_34 = torch.ops.aten.mm.default(view_331, permute_163)
        permute_230 = torch.ops.aten.permute.default(view_331, [1, 0])
        mm_35 = torch.ops.aten.mm.default(permute_230, view_191);  permute_230 = None
        permute_231 = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
        view_332 = torch.ops.aten.view.default(sum_65, [4096]);  sum_65 = None
        add_159 = torch.ops.aten.add.Tensor(add_137, view_332);  add_137 = view_332 = None
        view_333 = torch.ops.aten.view.default(mm_34, [1, 512, 4096]);  mm_34 = None
        add_160 = torch.ops.aten.add.Tensor(add_157, view_333);  add_157 = view_333 = None
        permute_232 = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        add_161 = torch.ops.aten.add.Tensor(add_139, permute_232);  add_139 = permute_232 = None
        view_334 = torch.ops.aten.view.default(_unsafe_view_17, [512, 4096]);  _unsafe_view_17 = None
        mm_36 = torch.ops.aten.mm.default(view_334, permute_167)
        permute_234 = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_37 = torch.ops.aten.mm.default(permute_234, view_191);  permute_234 = view_191 = None
        permute_235 = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        view_335 = torch.ops.aten.view.default(sum_66, [4096]);  sum_66 = None
        add_162 = torch.ops.aten.add.Tensor(add_140, view_335);  add_140 = view_335 = None
        view_336 = torch.ops.aten.view.default(mm_36, [1, 512, 4096]);  mm_36 = None
        add_163 = torch.ops.aten.add.Tensor(add_160, view_336);  add_160 = view_336 = None
        permute_236 = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        add_164 = torch.ops.aten.add.Tensor(add_142, permute_236);  add_142 = permute_236 = None
        mul_179 = torch.ops.aten.mul.Tensor(add_163, primals_22)
        mul_180 = torch.ops.aten.mul.Tensor(mul_179, 4096)
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
        mul_181 = torch.ops.aten.mul.Tensor(mul_179, mul_73);  mul_179 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_73, sum_68);  sum_68 = None
        sub_69 = torch.ops.aten.sub.Tensor(mul_180, sum_67);  mul_180 = sum_67 = None
        sub_70 = torch.ops.aten.sub.Tensor(sub_69, mul_182);  sub_69 = mul_182 = None
        mul_183 = torch.ops.aten.mul.Tensor(div_39, sub_70);  div_39 = sub_70 = None
        mul_184 = torch.ops.aten.mul.Tensor(add_163, mul_73);  mul_73 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(add_163, [0, 1]);  add_163 = None
        add_165 = torch.ops.aten.add.Tensor(add_143, sum_69);  add_143 = sum_69 = None
        add_166 = torch.ops.aten.add.Tensor(add_144, sum_70);  add_144 = sum_70 = None
        view_337 = torch.ops.aten.view.default(mul_183, [512, 4096])
        mm_38 = torch.ops.aten.mm.default(view_337, permute_138)
        permute_238 = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_39 = torch.ops.aten.mm.default(permute_238, view_189);  permute_238 = view_189 = None
        permute_239 = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        view_338 = torch.ops.aten.view.default(sum_71, [4096]);  sum_71 = None
        add_167 = torch.ops.aten.add.Tensor(add_145, view_338);  add_145 = view_338 = None
        view_339 = torch.ops.aten.view.default(mm_38, [1, 512, 16384]);  mm_38 = None
        permute_240 = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        add_168 = torch.ops.aten.add.Tensor(add_146, permute_240);  add_146 = permute_240 = None
        mul_185 = torch.ops.aten.mul.Tensor(view_339, mul_69);  mul_69 = None
        mul_186 = torch.ops.aten.mul.Tensor(view_339, add_81);  view_339 = add_81 = None
        mul_187 = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_71 = torch.ops.aten.sub.Tensor(1, mul_187);  mul_187 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_185, sub_71);  mul_185 = sub_71 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, 0.7978845608028654);  mul_188 = None
        mul_190 = torch.ops.aten.mul.Tensor(mul_189, 0.044715)
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(view_188, 2.0);  view_188 = None
        mul_191 = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        add_169 = torch.ops.aten.add.Tensor(mul_189, mul_192);  mul_189 = mul_192 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_186, 0.5);  mul_186 = None
        add_170 = torch.ops.aten.add.Tensor(add_169, mul_193);  add_169 = mul_193 = None
        view_340 = torch.ops.aten.view.default(add_170, [512, 16384]);  add_170 = None
        mm_40 = torch.ops.aten.mm.default(view_340, permute_142)
        permute_242 = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_41 = torch.ops.aten.mm.default(permute_242, view_187);  permute_242 = view_187 = None
        permute_243 = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
        view_341 = torch.ops.aten.view.default(sum_72, [16384]);  sum_72 = None
        add_171 = torch.ops.aten.add.Tensor(add_149, view_341);  add_149 = view_341 = None
        view_342 = torch.ops.aten.view.default(mm_40, [1, 512, 4096]);  mm_40 = None
        add_172 = torch.ops.aten.add.Tensor(mul_183, view_342);  mul_183 = view_342 = None
        permute_244 = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        add_173 = torch.ops.aten.add.Tensor(add_151, permute_244);  add_151 = permute_244 = None
        mul_195 = torch.ops.aten.mul.Tensor(add_172, primals_16)
        mul_196 = torch.ops.aten.mul.Tensor(mul_195, 4096)
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
        mul_197 = torch.ops.aten.mul.Tensor(mul_195, mul_67);  mul_195 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_67, sum_74);  sum_74 = None
        sub_73 = torch.ops.aten.sub.Tensor(mul_196, sum_73);  mul_196 = sum_73 = None
        sub_74 = torch.ops.aten.sub.Tensor(sub_73, mul_198);  sub_73 = mul_198 = None
        mul_199 = torch.ops.aten.mul.Tensor(div_40, sub_74);  div_40 = sub_74 = None
        mul_200 = torch.ops.aten.mul.Tensor(add_172, mul_67);  mul_67 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(add_172, [0, 1]);  add_172 = None
        add_174 = torch.ops.aten.add.Tensor(add_152, sum_75);  add_152 = sum_75 = None
        add_175 = torch.ops.aten.add.Tensor(add_153, sum_76);  add_153 = sum_76 = None
        view_343 = torch.ops.aten.view.default(mul_199, [512, 4096])
        mm_42 = torch.ops.aten.mm.default(view_343, permute_146)
        permute_246 = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_43 = torch.ops.aten.mm.default(permute_246, view_185);  permute_246 = view_185 = None
        permute_247 = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        view_344 = torch.ops.aten.view.default(sum_77, [4096]);  sum_77 = None
        add_176 = torch.ops.aten.add.Tensor(add_154, view_344);  add_154 = view_344 = None
        view_345 = torch.ops.aten.view.default(mm_42, [1, 512, 4096]);  mm_42 = None
        permute_248 = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        add_177 = torch.ops.aten.add.Tensor(add_155, permute_248);  add_155 = permute_248 = None
        view_346 = torch.ops.aten.view.default(view_345, [1, 512, 64, 64]);  view_345 = None
        permute_249 = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        view_347 = torch.ops.aten.view.default(permute_249, [64, 512, 64]);  permute_249 = None
        bmm_36 = torch.ops.aten.bmm.default(permute_250, view_347);  permute_250 = None
        bmm_37 = torch.ops.aten.bmm.default(view_347, permute_251);  view_347 = permute_251 = None
        view_348 = torch.ops.aten.view.default(bmm_36, [1, 64, 512, 64]);  bmm_36 = None
        view_349 = torch.ops.aten.view.default(bmm_37, [1, 64, 512, 512]);  bmm_37 = None
        mul_201 = torch.ops.aten.mul.Tensor(view_349, div_17);  view_349 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_201, [-1], True)
        mul_202 = torch.ops.aten.mul.Tensor(div_17, sum_78);  div_17 = sum_78 = None
        sub_75 = torch.ops.aten.sub.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
        div_41 = torch.ops.aten.div.Tensor(sub_75, 8.0);  sub_75 = None
        view_350 = torch.ops.aten.view.default(div_41, [64, 512, 512]);  div_41 = None
        bmm_38 = torch.ops.aten.bmm.default(permute_252, view_350);  permute_252 = None
        bmm_39 = torch.ops.aten.bmm.default(view_350, permute_253);  view_350 = permute_253 = None
        view_351 = torch.ops.aten.view.default(bmm_38, [1, 64, 64, 512]);  bmm_38 = None
        view_352 = torch.ops.aten.view.default(bmm_39, [1, 64, 512, 64]);  bmm_39 = None
        permute_254 = torch.ops.aten.permute.default(view_351, [0, 1, 3, 2]);  view_351 = None
        permute_255 = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
        clone_20 = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
        _unsafe_view_18 = torch.ops.aten._unsafe_view.default(clone_20, [1, 512, 4096]);  clone_20 = None
        permute_256 = torch.ops.aten.permute.default(permute_254, [0, 2, 1, 3]);  permute_254 = None
        view_353 = torch.ops.aten.view.default(permute_256, [1, 512, 4096]);  permute_256 = None
        permute_257 = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_21 = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        _unsafe_view_19 = torch.ops.aten._unsafe_view.default(clone_21, [1, 512, 4096]);  clone_21 = None
        view_354 = torch.ops.aten.view.default(_unsafe_view_18, [512, 4096]);  _unsafe_view_18 = None
        mm_44 = torch.ops.aten.mm.default(view_354, permute_159)
        permute_259 = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_45 = torch.ops.aten.mm.default(permute_259, view_170);  permute_259 = None
        permute_260 = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(view_354, [0], True);  view_354 = None
        view_355 = torch.ops.aten.view.default(sum_79, [4096]);  sum_79 = None
        add_178 = torch.ops.aten.add.Tensor(add_156, view_355);  add_156 = view_355 = None
        view_356 = torch.ops.aten.view.default(mm_44, [1, 512, 4096]);  mm_44 = None
        add_179 = torch.ops.aten.add.Tensor(mul_199, view_356);  mul_199 = view_356 = None
        permute_261 = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
        add_180 = torch.ops.aten.add.Tensor(add_158, permute_261);  add_158 = permute_261 = None
        view_357 = torch.ops.aten.view.default(view_353, [512, 4096]);  view_353 = None
        mm_46 = torch.ops.aten.mm.default(view_357, permute_163)
        permute_263 = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_47 = torch.ops.aten.mm.default(permute_263, view_170);  permute_263 = None
        permute_264 = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
        view_358 = torch.ops.aten.view.default(sum_80, [4096]);  sum_80 = None
        add_181 = torch.ops.aten.add.Tensor(add_159, view_358);  add_159 = view_358 = None
        view_359 = torch.ops.aten.view.default(mm_46, [1, 512, 4096]);  mm_46 = None
        add_182 = torch.ops.aten.add.Tensor(add_179, view_359);  add_179 = view_359 = None
        permute_265 = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        add_183 = torch.ops.aten.add.Tensor(add_161, permute_265);  add_161 = permute_265 = None
        view_360 = torch.ops.aten.view.default(_unsafe_view_19, [512, 4096]);  _unsafe_view_19 = None
        mm_48 = torch.ops.aten.mm.default(view_360, permute_167)
        permute_267 = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_49 = torch.ops.aten.mm.default(permute_267, view_170);  permute_267 = view_170 = None
        permute_268 = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
        view_361 = torch.ops.aten.view.default(sum_81, [4096]);  sum_81 = None
        add_184 = torch.ops.aten.add.Tensor(add_162, view_361);  add_162 = view_361 = None
        view_362 = torch.ops.aten.view.default(mm_48, [1, 512, 4096]);  mm_48 = None
        add_185 = torch.ops.aten.add.Tensor(add_182, view_362);  add_182 = view_362 = None
        permute_269 = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
        add_186 = torch.ops.aten.add.Tensor(add_164, permute_269);  add_164 = permute_269 = None
        mul_204 = torch.ops.aten.mul.Tensor(add_185, primals_22)
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, 4096)
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_204, [2], True)
        mul_206 = torch.ops.aten.mul.Tensor(mul_204, mul_65);  mul_204 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_206, [2], True);  mul_206 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_65, sum_83);  sum_83 = None
        sub_77 = torch.ops.aten.sub.Tensor(mul_205, sum_82);  mul_205 = sum_82 = None
        sub_78 = torch.ops.aten.sub.Tensor(sub_77, mul_207);  sub_77 = mul_207 = None
        mul_208 = torch.ops.aten.mul.Tensor(div_42, sub_78);  div_42 = sub_78 = None
        mul_209 = torch.ops.aten.mul.Tensor(add_185, mul_65);  mul_65 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_209, [0, 1]);  mul_209 = None
        sum_85 = torch.ops.aten.sum.dim_IntList(add_185, [0, 1]);  add_185 = None
        add_187 = torch.ops.aten.add.Tensor(add_165, sum_84);  add_165 = sum_84 = None
        add_188 = torch.ops.aten.add.Tensor(add_166, sum_85);  add_166 = sum_85 = None
        view_363 = torch.ops.aten.view.default(mul_208, [512, 4096])
        mm_50 = torch.ops.aten.mm.default(view_363, permute_138)
        permute_271 = torch.ops.aten.permute.default(view_363, [1, 0])
        mm_51 = torch.ops.aten.mm.default(permute_271, view_168);  permute_271 = view_168 = None
        permute_272 = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        view_364 = torch.ops.aten.view.default(sum_86, [4096]);  sum_86 = None
        add_189 = torch.ops.aten.add.Tensor(add_167, view_364);  add_167 = view_364 = None
        view_365 = torch.ops.aten.view.default(mm_50, [1, 512, 16384]);  mm_50 = None
        permute_273 = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        add_190 = torch.ops.aten.add.Tensor(add_168, permute_273);  add_168 = permute_273 = None
        mul_210 = torch.ops.aten.mul.Tensor(view_365, mul_61);  mul_61 = None
        mul_211 = torch.ops.aten.mul.Tensor(view_365, add_72);  view_365 = add_72 = None
        mul_212 = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_79 = torch.ops.aten.sub.Tensor(1, mul_212);  mul_212 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_210, sub_79);  mul_210 = sub_79 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, 0.7978845608028654);  mul_213 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, 0.044715)
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(view_167, 2.0);  view_167 = None
        mul_216 = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
        add_191 = torch.ops.aten.add.Tensor(mul_214, mul_217);  mul_214 = mul_217 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_211, 0.5);  mul_211 = None
        add_192 = torch.ops.aten.add.Tensor(add_191, mul_218);  add_191 = mul_218 = None
        view_366 = torch.ops.aten.view.default(add_192, [512, 16384]);  add_192 = None
        mm_52 = torch.ops.aten.mm.default(view_366, permute_142)
        permute_275 = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_53 = torch.ops.aten.mm.default(permute_275, view_166);  permute_275 = view_166 = None
        permute_276 = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
        sum_87 = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        view_367 = torch.ops.aten.view.default(sum_87, [16384]);  sum_87 = None
        add_193 = torch.ops.aten.add.Tensor(add_171, view_367);  add_171 = view_367 = None
        view_368 = torch.ops.aten.view.default(mm_52, [1, 512, 4096]);  mm_52 = None
        add_194 = torch.ops.aten.add.Tensor(mul_208, view_368);  mul_208 = view_368 = None
        permute_277 = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        add_195 = torch.ops.aten.add.Tensor(add_173, permute_277);  add_173 = permute_277 = None
        mul_220 = torch.ops.aten.mul.Tensor(add_194, primals_16)
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, 4096)
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
        mul_222 = torch.ops.aten.mul.Tensor(mul_220, mul_59);  mul_220 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_59, sum_89);  sum_89 = None
        sub_81 = torch.ops.aten.sub.Tensor(mul_221, sum_88);  mul_221 = sum_88 = None
        sub_82 = torch.ops.aten.sub.Tensor(sub_81, mul_223);  sub_81 = mul_223 = None
        mul_224 = torch.ops.aten.mul.Tensor(div_43, sub_82);  div_43 = sub_82 = None
        mul_225 = torch.ops.aten.mul.Tensor(add_194, mul_59);  mul_59 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(add_194, [0, 1]);  add_194 = None
        add_196 = torch.ops.aten.add.Tensor(add_174, sum_90);  add_174 = sum_90 = None
        add_197 = torch.ops.aten.add.Tensor(add_175, sum_91);  add_175 = sum_91 = None
        view_369 = torch.ops.aten.view.default(mul_224, [512, 4096])
        mm_54 = torch.ops.aten.mm.default(view_369, permute_146)
        permute_279 = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_55 = torch.ops.aten.mm.default(permute_279, view_164);  permute_279 = view_164 = None
        permute_280 = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        view_370 = torch.ops.aten.view.default(sum_92, [4096]);  sum_92 = None
        add_198 = torch.ops.aten.add.Tensor(add_176, view_370);  add_176 = view_370 = None
        view_371 = torch.ops.aten.view.default(mm_54, [1, 512, 4096]);  mm_54 = None
        permute_281 = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
        add_199 = torch.ops.aten.add.Tensor(add_177, permute_281);  add_177 = permute_281 = None
        view_372 = torch.ops.aten.view.default(view_371, [1, 512, 64, 64]);  view_371 = None
        permute_282 = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        view_373 = torch.ops.aten.view.default(permute_282, [64, 512, 64]);  permute_282 = None
        bmm_40 = torch.ops.aten.bmm.default(permute_283, view_373);  permute_283 = None
        bmm_41 = torch.ops.aten.bmm.default(view_373, permute_284);  view_373 = permute_284 = None
        view_374 = torch.ops.aten.view.default(bmm_40, [1, 64, 512, 64]);  bmm_40 = None
        view_375 = torch.ops.aten.view.default(bmm_41, [1, 64, 512, 512]);  bmm_41 = None
        mul_226 = torch.ops.aten.mul.Tensor(view_375, div_15);  view_375 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_226, [-1], True)
        mul_227 = torch.ops.aten.mul.Tensor(div_15, sum_93);  div_15 = sum_93 = None
        sub_83 = torch.ops.aten.sub.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
        div_44 = torch.ops.aten.div.Tensor(sub_83, 8.0);  sub_83 = None
        view_376 = torch.ops.aten.view.default(div_44, [64, 512, 512]);  div_44 = None
        bmm_42 = torch.ops.aten.bmm.default(permute_285, view_376);  permute_285 = None
        bmm_43 = torch.ops.aten.bmm.default(view_376, permute_286);  view_376 = permute_286 = None
        view_377 = torch.ops.aten.view.default(bmm_42, [1, 64, 64, 512]);  bmm_42 = None
        view_378 = torch.ops.aten.view.default(bmm_43, [1, 64, 512, 64]);  bmm_43 = None
        permute_287 = torch.ops.aten.permute.default(view_377, [0, 1, 3, 2]);  view_377 = None
        permute_288 = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_22 = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        _unsafe_view_20 = torch.ops.aten._unsafe_view.default(clone_22, [1, 512, 4096]);  clone_22 = None
        permute_289 = torch.ops.aten.permute.default(permute_287, [0, 2, 1, 3]);  permute_287 = None
        view_379 = torch.ops.aten.view.default(permute_289, [1, 512, 4096]);  permute_289 = None
        permute_290 = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_23 = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(clone_23, [1, 512, 4096]);  clone_23 = None
        view_380 = torch.ops.aten.view.default(_unsafe_view_20, [512, 4096]);  _unsafe_view_20 = None
        mm_56 = torch.ops.aten.mm.default(view_380, permute_159)
        permute_292 = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_57 = torch.ops.aten.mm.default(permute_292, view_149);  permute_292 = None
        permute_293 = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        view_381 = torch.ops.aten.view.default(sum_94, [4096]);  sum_94 = None
        add_200 = torch.ops.aten.add.Tensor(add_178, view_381);  add_178 = view_381 = None
        view_382 = torch.ops.aten.view.default(mm_56, [1, 512, 4096]);  mm_56 = None
        add_201 = torch.ops.aten.add.Tensor(mul_224, view_382);  mul_224 = view_382 = None
        permute_294 = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
        add_202 = torch.ops.aten.add.Tensor(add_180, permute_294);  add_180 = permute_294 = None
        view_383 = torch.ops.aten.view.default(view_379, [512, 4096]);  view_379 = None
        mm_58 = torch.ops.aten.mm.default(view_383, permute_163)
        permute_296 = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_59 = torch.ops.aten.mm.default(permute_296, view_149);  permute_296 = None
        permute_297 = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
        view_384 = torch.ops.aten.view.default(sum_95, [4096]);  sum_95 = None
        add_203 = torch.ops.aten.add.Tensor(add_181, view_384);  add_181 = view_384 = None
        view_385 = torch.ops.aten.view.default(mm_58, [1, 512, 4096]);  mm_58 = None
        add_204 = torch.ops.aten.add.Tensor(add_201, view_385);  add_201 = view_385 = None
        permute_298 = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
        add_205 = torch.ops.aten.add.Tensor(add_183, permute_298);  add_183 = permute_298 = None
        view_386 = torch.ops.aten.view.default(_unsafe_view_21, [512, 4096]);  _unsafe_view_21 = None
        mm_60 = torch.ops.aten.mm.default(view_386, permute_167)
        permute_300 = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_61 = torch.ops.aten.mm.default(permute_300, view_149);  permute_300 = view_149 = None
        permute_301 = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
        view_387 = torch.ops.aten.view.default(sum_96, [4096]);  sum_96 = None
        add_206 = torch.ops.aten.add.Tensor(add_184, view_387);  add_184 = view_387 = None
        view_388 = torch.ops.aten.view.default(mm_60, [1, 512, 4096]);  mm_60 = None
        add_207 = torch.ops.aten.add.Tensor(add_204, view_388);  add_204 = view_388 = None
        permute_302 = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
        add_208 = torch.ops.aten.add.Tensor(add_186, permute_302);  add_186 = permute_302 = None
        mul_229 = torch.ops.aten.mul.Tensor(add_207, primals_22)
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, 4096)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
        mul_231 = torch.ops.aten.mul.Tensor(mul_229, mul_57);  mul_229 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_57, sum_98);  sum_98 = None
        sub_85 = torch.ops.aten.sub.Tensor(mul_230, sum_97);  mul_230 = sum_97 = None
        sub_86 = torch.ops.aten.sub.Tensor(sub_85, mul_232);  sub_85 = mul_232 = None
        mul_233 = torch.ops.aten.mul.Tensor(div_45, sub_86);  div_45 = sub_86 = None
        mul_234 = torch.ops.aten.mul.Tensor(add_207, mul_57);  mul_57 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(add_207, [0, 1]);  add_207 = None
        add_209 = torch.ops.aten.add.Tensor(add_187, sum_99);  add_187 = sum_99 = None
        add_210 = torch.ops.aten.add.Tensor(add_188, sum_100);  add_188 = sum_100 = None
        view_389 = torch.ops.aten.view.default(mul_233, [512, 4096])
        mm_62 = torch.ops.aten.mm.default(view_389, permute_138)
        permute_304 = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_63 = torch.ops.aten.mm.default(permute_304, view_147);  permute_304 = view_147 = None
        permute_305 = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
        view_390 = torch.ops.aten.view.default(sum_101, [4096]);  sum_101 = None
        add_211 = torch.ops.aten.add.Tensor(add_189, view_390);  add_189 = view_390 = None
        view_391 = torch.ops.aten.view.default(mm_62, [1, 512, 16384]);  mm_62 = None
        permute_306 = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        add_212 = torch.ops.aten.add.Tensor(add_190, permute_306);  add_190 = permute_306 = None
        mul_235 = torch.ops.aten.mul.Tensor(view_391, mul_53);  mul_53 = None
        mul_236 = torch.ops.aten.mul.Tensor(view_391, add_63);  view_391 = add_63 = None
        mul_237 = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_87 = torch.ops.aten.sub.Tensor(1, mul_237);  mul_237 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_235, sub_87);  mul_235 = sub_87 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, 0.7978845608028654);  mul_238 = None
        mul_240 = torch.ops.aten.mul.Tensor(mul_239, 0.044715)
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(view_146, 2.0);  view_146 = None
        mul_241 = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
        add_213 = torch.ops.aten.add.Tensor(mul_239, mul_242);  mul_239 = mul_242 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_236, 0.5);  mul_236 = None
        add_214 = torch.ops.aten.add.Tensor(add_213, mul_243);  add_213 = mul_243 = None
        view_392 = torch.ops.aten.view.default(add_214, [512, 16384]);  add_214 = None
        mm_64 = torch.ops.aten.mm.default(view_392, permute_142)
        permute_308 = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_65 = torch.ops.aten.mm.default(permute_308, view_145);  permute_308 = view_145 = None
        permute_309 = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
        view_393 = torch.ops.aten.view.default(sum_102, [16384]);  sum_102 = None
        add_215 = torch.ops.aten.add.Tensor(add_193, view_393);  add_193 = view_393 = None
        view_394 = torch.ops.aten.view.default(mm_64, [1, 512, 4096]);  mm_64 = None
        add_216 = torch.ops.aten.add.Tensor(mul_233, view_394);  mul_233 = view_394 = None
        permute_310 = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
        add_217 = torch.ops.aten.add.Tensor(add_195, permute_310);  add_195 = permute_310 = None
        mul_245 = torch.ops.aten.mul.Tensor(add_216, primals_16)
        mul_246 = torch.ops.aten.mul.Tensor(mul_245, 4096)
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
        mul_247 = torch.ops.aten.mul.Tensor(mul_245, mul_51);  mul_245 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_51, sum_104);  sum_104 = None
        sub_89 = torch.ops.aten.sub.Tensor(mul_246, sum_103);  mul_246 = sum_103 = None
        sub_90 = torch.ops.aten.sub.Tensor(sub_89, mul_248);  sub_89 = mul_248 = None
        mul_249 = torch.ops.aten.mul.Tensor(div_46, sub_90);  div_46 = sub_90 = None
        mul_250 = torch.ops.aten.mul.Tensor(add_216, mul_51);  mul_51 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(add_216, [0, 1]);  add_216 = None
        add_218 = torch.ops.aten.add.Tensor(add_196, sum_105);  add_196 = sum_105 = None
        add_219 = torch.ops.aten.add.Tensor(add_197, sum_106);  add_197 = sum_106 = None
        view_395 = torch.ops.aten.view.default(mul_249, [512, 4096])
        mm_66 = torch.ops.aten.mm.default(view_395, permute_146)
        permute_312 = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_67 = torch.ops.aten.mm.default(permute_312, view_143);  permute_312 = view_143 = None
        permute_313 = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
        view_396 = torch.ops.aten.view.default(sum_107, [4096]);  sum_107 = None
        add_220 = torch.ops.aten.add.Tensor(add_198, view_396);  add_198 = view_396 = None
        view_397 = torch.ops.aten.view.default(mm_66, [1, 512, 4096]);  mm_66 = None
        permute_314 = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
        add_221 = torch.ops.aten.add.Tensor(add_199, permute_314);  add_199 = permute_314 = None
        view_398 = torch.ops.aten.view.default(view_397, [1, 512, 64, 64]);  view_397 = None
        permute_315 = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        view_399 = torch.ops.aten.view.default(permute_315, [64, 512, 64]);  permute_315 = None
        bmm_44 = torch.ops.aten.bmm.default(permute_316, view_399);  permute_316 = None
        bmm_45 = torch.ops.aten.bmm.default(view_399, permute_317);  view_399 = permute_317 = None
        view_400 = torch.ops.aten.view.default(bmm_44, [1, 64, 512, 64]);  bmm_44 = None
        view_401 = torch.ops.aten.view.default(bmm_45, [1, 64, 512, 512]);  bmm_45 = None
        mul_251 = torch.ops.aten.mul.Tensor(view_401, div_13);  view_401 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
        mul_252 = torch.ops.aten.mul.Tensor(div_13, sum_108);  div_13 = sum_108 = None
        sub_91 = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
        div_47 = torch.ops.aten.div.Tensor(sub_91, 8.0);  sub_91 = None
        view_402 = torch.ops.aten.view.default(div_47, [64, 512, 512]);  div_47 = None
        bmm_46 = torch.ops.aten.bmm.default(permute_318, view_402);  permute_318 = None
        bmm_47 = torch.ops.aten.bmm.default(view_402, permute_319);  view_402 = permute_319 = None
        view_403 = torch.ops.aten.view.default(bmm_46, [1, 64, 64, 512]);  bmm_46 = None
        view_404 = torch.ops.aten.view.default(bmm_47, [1, 64, 512, 64]);  bmm_47 = None
        permute_320 = torch.ops.aten.permute.default(view_403, [0, 1, 3, 2]);  view_403 = None
        permute_321 = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        clone_24 = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(clone_24, [1, 512, 4096]);  clone_24 = None
        permute_322 = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
        view_405 = torch.ops.aten.view.default(permute_322, [1, 512, 4096]);  permute_322 = None
        permute_323 = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        clone_25 = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        _unsafe_view_23 = torch.ops.aten._unsafe_view.default(clone_25, [1, 512, 4096]);  clone_25 = None
        view_406 = torch.ops.aten.view.default(_unsafe_view_22, [512, 4096]);  _unsafe_view_22 = None
        mm_68 = torch.ops.aten.mm.default(view_406, permute_159)
        permute_325 = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_69 = torch.ops.aten.mm.default(permute_325, view_128);  permute_325 = None
        permute_326 = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
        view_407 = torch.ops.aten.view.default(sum_109, [4096]);  sum_109 = None
        add_222 = torch.ops.aten.add.Tensor(add_200, view_407);  add_200 = view_407 = None
        view_408 = torch.ops.aten.view.default(mm_68, [1, 512, 4096]);  mm_68 = None
        add_223 = torch.ops.aten.add.Tensor(mul_249, view_408);  mul_249 = view_408 = None
        permute_327 = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
        add_224 = torch.ops.aten.add.Tensor(add_202, permute_327);  add_202 = permute_327 = None
        view_409 = torch.ops.aten.view.default(view_405, [512, 4096]);  view_405 = None
        mm_70 = torch.ops.aten.mm.default(view_409, permute_163)
        permute_329 = torch.ops.aten.permute.default(view_409, [1, 0])
        mm_71 = torch.ops.aten.mm.default(permute_329, view_128);  permute_329 = None
        permute_330 = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
        sum_110 = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
        view_410 = torch.ops.aten.view.default(sum_110, [4096]);  sum_110 = None
        add_225 = torch.ops.aten.add.Tensor(add_203, view_410);  add_203 = view_410 = None
        view_411 = torch.ops.aten.view.default(mm_70, [1, 512, 4096]);  mm_70 = None
        add_226 = torch.ops.aten.add.Tensor(add_223, view_411);  add_223 = view_411 = None
        permute_331 = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
        add_227 = torch.ops.aten.add.Tensor(add_205, permute_331);  add_205 = permute_331 = None
        view_412 = torch.ops.aten.view.default(_unsafe_view_23, [512, 4096]);  _unsafe_view_23 = None
        mm_72 = torch.ops.aten.mm.default(view_412, permute_167)
        permute_333 = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_73 = torch.ops.aten.mm.default(permute_333, view_128);  permute_333 = view_128 = None
        permute_334 = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
        view_413 = torch.ops.aten.view.default(sum_111, [4096]);  sum_111 = None
        add_228 = torch.ops.aten.add.Tensor(add_206, view_413);  add_206 = view_413 = None
        view_414 = torch.ops.aten.view.default(mm_72, [1, 512, 4096]);  mm_72 = None
        add_229 = torch.ops.aten.add.Tensor(add_226, view_414);  add_226 = view_414 = None
        permute_335 = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
        add_230 = torch.ops.aten.add.Tensor(add_208, permute_335);  add_208 = permute_335 = None
        mul_254 = torch.ops.aten.mul.Tensor(add_229, primals_22)
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, 4096)
        sum_112 = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
        mul_256 = torch.ops.aten.mul.Tensor(mul_254, mul_49);  mul_254 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_49, sum_113);  sum_113 = None
        sub_93 = torch.ops.aten.sub.Tensor(mul_255, sum_112);  mul_255 = sum_112 = None
        sub_94 = torch.ops.aten.sub.Tensor(sub_93, mul_257);  sub_93 = mul_257 = None
        mul_258 = torch.ops.aten.mul.Tensor(div_48, sub_94);  div_48 = sub_94 = None
        mul_259 = torch.ops.aten.mul.Tensor(add_229, mul_49);  mul_49 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(add_229, [0, 1]);  add_229 = None
        add_231 = torch.ops.aten.add.Tensor(add_209, sum_114);  add_209 = sum_114 = None
        add_232 = torch.ops.aten.add.Tensor(add_210, sum_115);  add_210 = sum_115 = None
        view_415 = torch.ops.aten.view.default(mul_258, [512, 4096])
        mm_74 = torch.ops.aten.mm.default(view_415, permute_138)
        permute_337 = torch.ops.aten.permute.default(view_415, [1, 0])
        mm_75 = torch.ops.aten.mm.default(permute_337, view_126);  permute_337 = view_126 = None
        permute_338 = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
        view_416 = torch.ops.aten.view.default(sum_116, [4096]);  sum_116 = None
        add_233 = torch.ops.aten.add.Tensor(add_211, view_416);  add_211 = view_416 = None
        view_417 = torch.ops.aten.view.default(mm_74, [1, 512, 16384]);  mm_74 = None
        permute_339 = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        add_234 = torch.ops.aten.add.Tensor(add_212, permute_339);  add_212 = permute_339 = None
        mul_260 = torch.ops.aten.mul.Tensor(view_417, mul_45);  mul_45 = None
        mul_261 = torch.ops.aten.mul.Tensor(view_417, add_54);  view_417 = add_54 = None
        mul_262 = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_95 = torch.ops.aten.sub.Tensor(1, mul_262);  mul_262 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_260, sub_95);  mul_260 = sub_95 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_263, 0.7978845608028654);  mul_263 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, 0.044715)
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(view_125, 2.0);  view_125 = None
        mul_266 = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
        add_235 = torch.ops.aten.add.Tensor(mul_264, mul_267);  mul_264 = mul_267 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_261, 0.5);  mul_261 = None
        add_236 = torch.ops.aten.add.Tensor(add_235, mul_268);  add_235 = mul_268 = None
        view_418 = torch.ops.aten.view.default(add_236, [512, 16384]);  add_236 = None
        mm_76 = torch.ops.aten.mm.default(view_418, permute_142)
        permute_341 = torch.ops.aten.permute.default(view_418, [1, 0])
        mm_77 = torch.ops.aten.mm.default(permute_341, view_124);  permute_341 = view_124 = None
        permute_342 = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(view_418, [0], True);  view_418 = None
        view_419 = torch.ops.aten.view.default(sum_117, [16384]);  sum_117 = None
        add_237 = torch.ops.aten.add.Tensor(add_215, view_419);  add_215 = view_419 = None
        view_420 = torch.ops.aten.view.default(mm_76, [1, 512, 4096]);  mm_76 = None
        add_238 = torch.ops.aten.add.Tensor(mul_258, view_420);  mul_258 = view_420 = None
        permute_343 = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
        add_239 = torch.ops.aten.add.Tensor(add_217, permute_343);  add_217 = permute_343 = None
        mul_270 = torch.ops.aten.mul.Tensor(add_238, primals_16)
        mul_271 = torch.ops.aten.mul.Tensor(mul_270, 4096)
        sum_118 = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
        mul_272 = torch.ops.aten.mul.Tensor(mul_270, mul_43);  mul_270 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_43, sum_119);  sum_119 = None
        sub_97 = torch.ops.aten.sub.Tensor(mul_271, sum_118);  mul_271 = sum_118 = None
        sub_98 = torch.ops.aten.sub.Tensor(sub_97, mul_273);  sub_97 = mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(div_49, sub_98);  div_49 = sub_98 = None
        mul_275 = torch.ops.aten.mul.Tensor(add_238, mul_43);  mul_43 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(add_238, [0, 1]);  add_238 = None
        add_240 = torch.ops.aten.add.Tensor(add_218, sum_120);  add_218 = sum_120 = None
        add_241 = torch.ops.aten.add.Tensor(add_219, sum_121);  add_219 = sum_121 = None
        view_421 = torch.ops.aten.view.default(mul_274, [512, 4096])
        mm_78 = torch.ops.aten.mm.default(view_421, permute_146)
        permute_345 = torch.ops.aten.permute.default(view_421, [1, 0])
        mm_79 = torch.ops.aten.mm.default(permute_345, view_122);  permute_345 = view_122 = None
        permute_346 = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(view_421, [0], True);  view_421 = None
        view_422 = torch.ops.aten.view.default(sum_122, [4096]);  sum_122 = None
        add_242 = torch.ops.aten.add.Tensor(add_220, view_422);  add_220 = view_422 = None
        view_423 = torch.ops.aten.view.default(mm_78, [1, 512, 4096]);  mm_78 = None
        permute_347 = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
        add_243 = torch.ops.aten.add.Tensor(add_221, permute_347);  add_221 = permute_347 = None
        view_424 = torch.ops.aten.view.default(view_423, [1, 512, 64, 64]);  view_423 = None
        permute_348 = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        view_425 = torch.ops.aten.view.default(permute_348, [64, 512, 64]);  permute_348 = None
        bmm_48 = torch.ops.aten.bmm.default(permute_349, view_425);  permute_349 = None
        bmm_49 = torch.ops.aten.bmm.default(view_425, permute_350);  view_425 = permute_350 = None
        view_426 = torch.ops.aten.view.default(bmm_48, [1, 64, 512, 64]);  bmm_48 = None
        view_427 = torch.ops.aten.view.default(bmm_49, [1, 64, 512, 512]);  bmm_49 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_427, div_11);  view_427 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        mul_277 = torch.ops.aten.mul.Tensor(div_11, sum_123);  div_11 = sum_123 = None
        sub_99 = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        div_50 = torch.ops.aten.div.Tensor(sub_99, 8.0);  sub_99 = None
        view_428 = torch.ops.aten.view.default(div_50, [64, 512, 512]);  div_50 = None
        bmm_50 = torch.ops.aten.bmm.default(permute_351, view_428);  permute_351 = None
        bmm_51 = torch.ops.aten.bmm.default(view_428, permute_352);  view_428 = permute_352 = None
        view_429 = torch.ops.aten.view.default(bmm_50, [1, 64, 64, 512]);  bmm_50 = None
        view_430 = torch.ops.aten.view.default(bmm_51, [1, 64, 512, 64]);  bmm_51 = None
        permute_353 = torch.ops.aten.permute.default(view_429, [0, 1, 3, 2]);  view_429 = None
        permute_354 = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        clone_26 = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
        _unsafe_view_24 = torch.ops.aten._unsafe_view.default(clone_26, [1, 512, 4096]);  clone_26 = None
        permute_355 = torch.ops.aten.permute.default(permute_353, [0, 2, 1, 3]);  permute_353 = None
        view_431 = torch.ops.aten.view.default(permute_355, [1, 512, 4096]);  permute_355 = None
        permute_356 = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_27 = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
        _unsafe_view_25 = torch.ops.aten._unsafe_view.default(clone_27, [1, 512, 4096]);  clone_27 = None
        view_432 = torch.ops.aten.view.default(_unsafe_view_24, [512, 4096]);  _unsafe_view_24 = None
        mm_80 = torch.ops.aten.mm.default(view_432, permute_159)
        permute_358 = torch.ops.aten.permute.default(view_432, [1, 0])
        mm_81 = torch.ops.aten.mm.default(permute_358, view_107);  permute_358 = None
        permute_359 = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(view_432, [0], True);  view_432 = None
        view_433 = torch.ops.aten.view.default(sum_124, [4096]);  sum_124 = None
        add_244 = torch.ops.aten.add.Tensor(add_222, view_433);  add_222 = view_433 = None
        view_434 = torch.ops.aten.view.default(mm_80, [1, 512, 4096]);  mm_80 = None
        add_245 = torch.ops.aten.add.Tensor(mul_274, view_434);  mul_274 = view_434 = None
        permute_360 = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
        add_246 = torch.ops.aten.add.Tensor(add_224, permute_360);  add_224 = permute_360 = None
        view_435 = torch.ops.aten.view.default(view_431, [512, 4096]);  view_431 = None
        mm_82 = torch.ops.aten.mm.default(view_435, permute_163)
        permute_362 = torch.ops.aten.permute.default(view_435, [1, 0])
        mm_83 = torch.ops.aten.mm.default(permute_362, view_107);  permute_362 = None
        permute_363 = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
        sum_125 = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
        view_436 = torch.ops.aten.view.default(sum_125, [4096]);  sum_125 = None
        add_247 = torch.ops.aten.add.Tensor(add_225, view_436);  add_225 = view_436 = None
        view_437 = torch.ops.aten.view.default(mm_82, [1, 512, 4096]);  mm_82 = None
        add_248 = torch.ops.aten.add.Tensor(add_245, view_437);  add_245 = view_437 = None
        permute_364 = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
        add_249 = torch.ops.aten.add.Tensor(add_227, permute_364);  add_227 = permute_364 = None
        view_438 = torch.ops.aten.view.default(_unsafe_view_25, [512, 4096]);  _unsafe_view_25 = None
        mm_84 = torch.ops.aten.mm.default(view_438, permute_167)
        permute_366 = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_85 = torch.ops.aten.mm.default(permute_366, view_107);  permute_366 = view_107 = None
        permute_367 = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
        view_439 = torch.ops.aten.view.default(sum_126, [4096]);  sum_126 = None
        add_250 = torch.ops.aten.add.Tensor(add_228, view_439);  add_228 = view_439 = None
        view_440 = torch.ops.aten.view.default(mm_84, [1, 512, 4096]);  mm_84 = None
        add_251 = torch.ops.aten.add.Tensor(add_248, view_440);  add_248 = view_440 = None
        permute_368 = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
        add_252 = torch.ops.aten.add.Tensor(add_230, permute_368);  add_230 = permute_368 = None
        mul_279 = torch.ops.aten.mul.Tensor(add_251, primals_22)
        mul_280 = torch.ops.aten.mul.Tensor(mul_279, 4096)
        sum_127 = torch.ops.aten.sum.dim_IntList(mul_279, [2], True)
        mul_281 = torch.ops.aten.mul.Tensor(mul_279, mul_41);  mul_279 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(mul_281, [2], True);  mul_281 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_41, sum_128);  sum_128 = None
        sub_101 = torch.ops.aten.sub.Tensor(mul_280, sum_127);  mul_280 = sum_127 = None
        sub_102 = torch.ops.aten.sub.Tensor(sub_101, mul_282);  sub_101 = mul_282 = None
        mul_283 = torch.ops.aten.mul.Tensor(div_51, sub_102);  div_51 = sub_102 = None
        mul_284 = torch.ops.aten.mul.Tensor(add_251, mul_41);  mul_41 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(mul_284, [0, 1]);  mul_284 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
        add_253 = torch.ops.aten.add.Tensor(add_231, sum_129);  add_231 = sum_129 = None
        add_254 = torch.ops.aten.add.Tensor(add_232, sum_130);  add_232 = sum_130 = None
        view_441 = torch.ops.aten.view.default(mul_283, [512, 4096])
        mm_86 = torch.ops.aten.mm.default(view_441, permute_138)
        permute_370 = torch.ops.aten.permute.default(view_441, [1, 0])
        mm_87 = torch.ops.aten.mm.default(permute_370, view_105);  permute_370 = view_105 = None
        permute_371 = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
        view_442 = torch.ops.aten.view.default(sum_131, [4096]);  sum_131 = None
        add_255 = torch.ops.aten.add.Tensor(add_233, view_442);  add_233 = view_442 = None
        view_443 = torch.ops.aten.view.default(mm_86, [1, 512, 16384]);  mm_86 = None
        permute_372 = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
        add_256 = torch.ops.aten.add.Tensor(add_234, permute_372);  add_234 = permute_372 = None
        mul_285 = torch.ops.aten.mul.Tensor(view_443, mul_37);  mul_37 = None
        mul_286 = torch.ops.aten.mul.Tensor(view_443, add_45);  view_443 = add_45 = None
        mul_287 = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_103 = torch.ops.aten.sub.Tensor(1, mul_287);  mul_287 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_285, sub_103);  mul_285 = sub_103 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_288, 0.7978845608028654);  mul_288 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, 0.044715)
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(view_104, 2.0);  view_104 = None
        mul_291 = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
        add_257 = torch.ops.aten.add.Tensor(mul_289, mul_292);  mul_289 = mul_292 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_286, 0.5);  mul_286 = None
        add_258 = torch.ops.aten.add.Tensor(add_257, mul_293);  add_257 = mul_293 = None
        view_444 = torch.ops.aten.view.default(add_258, [512, 16384]);  add_258 = None
        mm_88 = torch.ops.aten.mm.default(view_444, permute_142)
        permute_374 = torch.ops.aten.permute.default(view_444, [1, 0])
        mm_89 = torch.ops.aten.mm.default(permute_374, view_103);  permute_374 = view_103 = None
        permute_375 = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
        view_445 = torch.ops.aten.view.default(sum_132, [16384]);  sum_132 = None
        add_259 = torch.ops.aten.add.Tensor(add_237, view_445);  add_237 = view_445 = None
        view_446 = torch.ops.aten.view.default(mm_88, [1, 512, 4096]);  mm_88 = None
        add_260 = torch.ops.aten.add.Tensor(mul_283, view_446);  mul_283 = view_446 = None
        permute_376 = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
        add_261 = torch.ops.aten.add.Tensor(add_239, permute_376);  add_239 = permute_376 = None
        mul_295 = torch.ops.aten.mul.Tensor(add_260, primals_16)
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, 4096)
        sum_133 = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
        mul_297 = torch.ops.aten.mul.Tensor(mul_295, mul_35);  mul_295 = None
        sum_134 = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
        mul_298 = torch.ops.aten.mul.Tensor(mul_35, sum_134);  sum_134 = None
        sub_105 = torch.ops.aten.sub.Tensor(mul_296, sum_133);  mul_296 = sum_133 = None
        sub_106 = torch.ops.aten.sub.Tensor(sub_105, mul_298);  sub_105 = mul_298 = None
        mul_299 = torch.ops.aten.mul.Tensor(div_52, sub_106);  div_52 = sub_106 = None
        mul_300 = torch.ops.aten.mul.Tensor(add_260, mul_35);  mul_35 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
        sum_136 = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
        add_262 = torch.ops.aten.add.Tensor(add_240, sum_135);  add_240 = sum_135 = None
        add_263 = torch.ops.aten.add.Tensor(add_241, sum_136);  add_241 = sum_136 = None
        view_447 = torch.ops.aten.view.default(mul_299, [512, 4096])
        mm_90 = torch.ops.aten.mm.default(view_447, permute_146)
        permute_378 = torch.ops.aten.permute.default(view_447, [1, 0])
        mm_91 = torch.ops.aten.mm.default(permute_378, view_101);  permute_378 = view_101 = None
        permute_379 = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
        sum_137 = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        view_448 = torch.ops.aten.view.default(sum_137, [4096]);  sum_137 = None
        add_264 = torch.ops.aten.add.Tensor(add_242, view_448);  add_242 = view_448 = None
        view_449 = torch.ops.aten.view.default(mm_90, [1, 512, 4096]);  mm_90 = None
        permute_380 = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
        add_265 = torch.ops.aten.add.Tensor(add_243, permute_380);  add_243 = permute_380 = None
        view_450 = torch.ops.aten.view.default(view_449, [1, 512, 64, 64]);  view_449 = None
        permute_381 = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        view_451 = torch.ops.aten.view.default(permute_381, [64, 512, 64]);  permute_381 = None
        bmm_52 = torch.ops.aten.bmm.default(permute_382, view_451);  permute_382 = None
        bmm_53 = torch.ops.aten.bmm.default(view_451, permute_383);  view_451 = permute_383 = None
        view_452 = torch.ops.aten.view.default(bmm_52, [1, 64, 512, 64]);  bmm_52 = None
        view_453 = torch.ops.aten.view.default(bmm_53, [1, 64, 512, 512]);  bmm_53 = None
        mul_301 = torch.ops.aten.mul.Tensor(view_453, div_9);  view_453 = None
        sum_138 = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
        mul_302 = torch.ops.aten.mul.Tensor(div_9, sum_138);  div_9 = sum_138 = None
        sub_107 = torch.ops.aten.sub.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
        div_53 = torch.ops.aten.div.Tensor(sub_107, 8.0);  sub_107 = None
        view_454 = torch.ops.aten.view.default(div_53, [64, 512, 512]);  div_53 = None
        bmm_54 = torch.ops.aten.bmm.default(permute_384, view_454);  permute_384 = None
        bmm_55 = torch.ops.aten.bmm.default(view_454, permute_385);  view_454 = permute_385 = None
        view_455 = torch.ops.aten.view.default(bmm_54, [1, 64, 64, 512]);  bmm_54 = None
        view_456 = torch.ops.aten.view.default(bmm_55, [1, 64, 512, 64]);  bmm_55 = None
        permute_386 = torch.ops.aten.permute.default(view_455, [0, 1, 3, 2]);  view_455 = None
        permute_387 = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        clone_28 = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
        _unsafe_view_26 = torch.ops.aten._unsafe_view.default(clone_28, [1, 512, 4096]);  clone_28 = None
        permute_388 = torch.ops.aten.permute.default(permute_386, [0, 2, 1, 3]);  permute_386 = None
        view_457 = torch.ops.aten.view.default(permute_388, [1, 512, 4096]);  permute_388 = None
        permute_389 = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_29 = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
        _unsafe_view_27 = torch.ops.aten._unsafe_view.default(clone_29, [1, 512, 4096]);  clone_29 = None
        view_458 = torch.ops.aten.view.default(_unsafe_view_26, [512, 4096]);  _unsafe_view_26 = None
        mm_92 = torch.ops.aten.mm.default(view_458, permute_159)
        permute_391 = torch.ops.aten.permute.default(view_458, [1, 0])
        mm_93 = torch.ops.aten.mm.default(permute_391, view_86);  permute_391 = None
        permute_392 = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
        sum_139 = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
        view_459 = torch.ops.aten.view.default(sum_139, [4096]);  sum_139 = None
        add_266 = torch.ops.aten.add.Tensor(add_244, view_459);  add_244 = view_459 = None
        view_460 = torch.ops.aten.view.default(mm_92, [1, 512, 4096]);  mm_92 = None
        add_267 = torch.ops.aten.add.Tensor(mul_299, view_460);  mul_299 = view_460 = None
        permute_393 = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
        add_268 = torch.ops.aten.add.Tensor(add_246, permute_393);  add_246 = permute_393 = None
        view_461 = torch.ops.aten.view.default(view_457, [512, 4096]);  view_457 = None
        mm_94 = torch.ops.aten.mm.default(view_461, permute_163)
        permute_395 = torch.ops.aten.permute.default(view_461, [1, 0])
        mm_95 = torch.ops.aten.mm.default(permute_395, view_86);  permute_395 = None
        permute_396 = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
        view_462 = torch.ops.aten.view.default(sum_140, [4096]);  sum_140 = None
        add_269 = torch.ops.aten.add.Tensor(add_247, view_462);  add_247 = view_462 = None
        view_463 = torch.ops.aten.view.default(mm_94, [1, 512, 4096]);  mm_94 = None
        add_270 = torch.ops.aten.add.Tensor(add_267, view_463);  add_267 = view_463 = None
        permute_397 = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
        add_271 = torch.ops.aten.add.Tensor(add_249, permute_397);  add_249 = permute_397 = None
        view_464 = torch.ops.aten.view.default(_unsafe_view_27, [512, 4096]);  _unsafe_view_27 = None
        mm_96 = torch.ops.aten.mm.default(view_464, permute_167)
        permute_399 = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_97 = torch.ops.aten.mm.default(permute_399, view_86);  permute_399 = view_86 = None
        permute_400 = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        view_465 = torch.ops.aten.view.default(sum_141, [4096]);  sum_141 = None
        add_272 = torch.ops.aten.add.Tensor(add_250, view_465);  add_250 = view_465 = None
        view_466 = torch.ops.aten.view.default(mm_96, [1, 512, 4096]);  mm_96 = None
        add_273 = torch.ops.aten.add.Tensor(add_270, view_466);  add_270 = view_466 = None
        permute_401 = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
        add_274 = torch.ops.aten.add.Tensor(add_252, permute_401);  add_252 = permute_401 = None
        mul_304 = torch.ops.aten.mul.Tensor(add_273, primals_22)
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, 4096)
        sum_142 = torch.ops.aten.sum.dim_IntList(mul_304, [2], True)
        mul_306 = torch.ops.aten.mul.Tensor(mul_304, mul_33);  mul_304 = None
        sum_143 = torch.ops.aten.sum.dim_IntList(mul_306, [2], True);  mul_306 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_33, sum_143);  sum_143 = None
        sub_109 = torch.ops.aten.sub.Tensor(mul_305, sum_142);  mul_305 = sum_142 = None
        sub_110 = torch.ops.aten.sub.Tensor(sub_109, mul_307);  sub_109 = mul_307 = None
        mul_308 = torch.ops.aten.mul.Tensor(div_54, sub_110);  div_54 = sub_110 = None
        mul_309 = torch.ops.aten.mul.Tensor(add_273, mul_33);  mul_33 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1]);  mul_309 = None
        sum_145 = torch.ops.aten.sum.dim_IntList(add_273, [0, 1]);  add_273 = None
        add_275 = torch.ops.aten.add.Tensor(add_253, sum_144);  add_253 = sum_144 = None
        add_276 = torch.ops.aten.add.Tensor(add_254, sum_145);  add_254 = sum_145 = None
        view_467 = torch.ops.aten.view.default(mul_308, [512, 4096])
        mm_98 = torch.ops.aten.mm.default(view_467, permute_138)
        permute_403 = torch.ops.aten.permute.default(view_467, [1, 0])
        mm_99 = torch.ops.aten.mm.default(permute_403, view_84);  permute_403 = view_84 = None
        permute_404 = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
        view_468 = torch.ops.aten.view.default(sum_146, [4096]);  sum_146 = None
        add_277 = torch.ops.aten.add.Tensor(add_255, view_468);  add_255 = view_468 = None
        view_469 = torch.ops.aten.view.default(mm_98, [1, 512, 16384]);  mm_98 = None
        permute_405 = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
        add_278 = torch.ops.aten.add.Tensor(add_256, permute_405);  add_256 = permute_405 = None
        mul_310 = torch.ops.aten.mul.Tensor(view_469, mul_29);  mul_29 = None
        mul_311 = torch.ops.aten.mul.Tensor(view_469, add_36);  view_469 = add_36 = None
        mul_312 = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_111 = torch.ops.aten.sub.Tensor(1, mul_312);  mul_312 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_310, sub_111);  mul_310 = sub_111 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, 0.7978845608028654);  mul_313 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, 0.044715)
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(view_83, 2.0);  view_83 = None
        mul_316 = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        add_279 = torch.ops.aten.add.Tensor(mul_314, mul_317);  mul_314 = mul_317 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_311, 0.5);  mul_311 = None
        add_280 = torch.ops.aten.add.Tensor(add_279, mul_318);  add_279 = mul_318 = None
        view_470 = torch.ops.aten.view.default(add_280, [512, 16384]);  add_280 = None
        mm_100 = torch.ops.aten.mm.default(view_470, permute_142)
        permute_407 = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_101 = torch.ops.aten.mm.default(permute_407, view_82);  permute_407 = view_82 = None
        permute_408 = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        view_471 = torch.ops.aten.view.default(sum_147, [16384]);  sum_147 = None
        add_281 = torch.ops.aten.add.Tensor(add_259, view_471);  add_259 = view_471 = None
        view_472 = torch.ops.aten.view.default(mm_100, [1, 512, 4096]);  mm_100 = None
        add_282 = torch.ops.aten.add.Tensor(mul_308, view_472);  mul_308 = view_472 = None
        permute_409 = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
        add_283 = torch.ops.aten.add.Tensor(add_261, permute_409);  add_261 = permute_409 = None
        mul_320 = torch.ops.aten.mul.Tensor(add_282, primals_16)
        mul_321 = torch.ops.aten.mul.Tensor(mul_320, 4096)
        sum_148 = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
        mul_322 = torch.ops.aten.mul.Tensor(mul_320, mul_27);  mul_320 = None
        sum_149 = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_27, sum_149);  sum_149 = None
        sub_113 = torch.ops.aten.sub.Tensor(mul_321, sum_148);  mul_321 = sum_148 = None
        sub_114 = torch.ops.aten.sub.Tensor(sub_113, mul_323);  sub_113 = mul_323 = None
        mul_324 = torch.ops.aten.mul.Tensor(div_55, sub_114);  div_55 = sub_114 = None
        mul_325 = torch.ops.aten.mul.Tensor(add_282, mul_27);  mul_27 = None
        sum_150 = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
        sum_151 = torch.ops.aten.sum.dim_IntList(add_282, [0, 1]);  add_282 = None
        add_284 = torch.ops.aten.add.Tensor(add_262, sum_150);  add_262 = sum_150 = None
        add_285 = torch.ops.aten.add.Tensor(add_263, sum_151);  add_263 = sum_151 = None
        view_473 = torch.ops.aten.view.default(mul_324, [512, 4096])
        mm_102 = torch.ops.aten.mm.default(view_473, permute_146)
        permute_411 = torch.ops.aten.permute.default(view_473, [1, 0])
        mm_103 = torch.ops.aten.mm.default(permute_411, view_80);  permute_411 = view_80 = None
        permute_412 = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
        sum_152 = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
        view_474 = torch.ops.aten.view.default(sum_152, [4096]);  sum_152 = None
        add_286 = torch.ops.aten.add.Tensor(add_264, view_474);  add_264 = view_474 = None
        view_475 = torch.ops.aten.view.default(mm_102, [1, 512, 4096]);  mm_102 = None
        permute_413 = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
        add_287 = torch.ops.aten.add.Tensor(add_265, permute_413);  add_265 = permute_413 = None
        view_476 = torch.ops.aten.view.default(view_475, [1, 512, 64, 64]);  view_475 = None
        permute_414 = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        view_477 = torch.ops.aten.view.default(permute_414, [64, 512, 64]);  permute_414 = None
        bmm_56 = torch.ops.aten.bmm.default(permute_415, view_477);  permute_415 = None
        bmm_57 = torch.ops.aten.bmm.default(view_477, permute_416);  view_477 = permute_416 = None
        view_478 = torch.ops.aten.view.default(bmm_56, [1, 64, 512, 64]);  bmm_56 = None
        view_479 = torch.ops.aten.view.default(bmm_57, [1, 64, 512, 512]);  bmm_57 = None
        mul_326 = torch.ops.aten.mul.Tensor(view_479, div_7);  view_479 = None
        sum_153 = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
        mul_327 = torch.ops.aten.mul.Tensor(div_7, sum_153);  div_7 = sum_153 = None
        sub_115 = torch.ops.aten.sub.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        div_56 = torch.ops.aten.div.Tensor(sub_115, 8.0);  sub_115 = None
        view_480 = torch.ops.aten.view.default(div_56, [64, 512, 512]);  div_56 = None
        bmm_58 = torch.ops.aten.bmm.default(permute_417, view_480);  permute_417 = None
        bmm_59 = torch.ops.aten.bmm.default(view_480, permute_418);  view_480 = permute_418 = None
        view_481 = torch.ops.aten.view.default(bmm_58, [1, 64, 64, 512]);  bmm_58 = None
        view_482 = torch.ops.aten.view.default(bmm_59, [1, 64, 512, 64]);  bmm_59 = None
        permute_419 = torch.ops.aten.permute.default(view_481, [0, 1, 3, 2]);  view_481 = None
        permute_420 = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        clone_30 = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
        _unsafe_view_28 = torch.ops.aten._unsafe_view.default(clone_30, [1, 512, 4096]);  clone_30 = None
        permute_421 = torch.ops.aten.permute.default(permute_419, [0, 2, 1, 3]);  permute_419 = None
        view_483 = torch.ops.aten.view.default(permute_421, [1, 512, 4096]);  permute_421 = None
        permute_422 = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
        clone_31 = torch.ops.aten.clone.default(permute_422, memory_format = torch.contiguous_format);  permute_422 = None
        _unsafe_view_29 = torch.ops.aten._unsafe_view.default(clone_31, [1, 512, 4096]);  clone_31 = None
        view_484 = torch.ops.aten.view.default(_unsafe_view_28, [512, 4096]);  _unsafe_view_28 = None
        mm_104 = torch.ops.aten.mm.default(view_484, permute_159)
        permute_424 = torch.ops.aten.permute.default(view_484, [1, 0])
        mm_105 = torch.ops.aten.mm.default(permute_424, view_65);  permute_424 = None
        permute_425 = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
        sum_154 = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
        view_485 = torch.ops.aten.view.default(sum_154, [4096]);  sum_154 = None
        add_288 = torch.ops.aten.add.Tensor(add_266, view_485);  add_266 = view_485 = None
        view_486 = torch.ops.aten.view.default(mm_104, [1, 512, 4096]);  mm_104 = None
        add_289 = torch.ops.aten.add.Tensor(mul_324, view_486);  mul_324 = view_486 = None
        permute_426 = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
        add_290 = torch.ops.aten.add.Tensor(add_268, permute_426);  add_268 = permute_426 = None
        view_487 = torch.ops.aten.view.default(view_483, [512, 4096]);  view_483 = None
        mm_106 = torch.ops.aten.mm.default(view_487, permute_163)
        permute_428 = torch.ops.aten.permute.default(view_487, [1, 0])
        mm_107 = torch.ops.aten.mm.default(permute_428, view_65);  permute_428 = None
        permute_429 = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
        sum_155 = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
        view_488 = torch.ops.aten.view.default(sum_155, [4096]);  sum_155 = None
        add_291 = torch.ops.aten.add.Tensor(add_269, view_488);  add_269 = view_488 = None
        view_489 = torch.ops.aten.view.default(mm_106, [1, 512, 4096]);  mm_106 = None
        add_292 = torch.ops.aten.add.Tensor(add_289, view_489);  add_289 = view_489 = None
        permute_430 = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
        add_293 = torch.ops.aten.add.Tensor(add_271, permute_430);  add_271 = permute_430 = None
        view_490 = torch.ops.aten.view.default(_unsafe_view_29, [512, 4096]);  _unsafe_view_29 = None
        mm_108 = torch.ops.aten.mm.default(view_490, permute_167)
        permute_432 = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_109 = torch.ops.aten.mm.default(permute_432, view_65);  permute_432 = view_65 = None
        permute_433 = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
        sum_156 = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
        view_491 = torch.ops.aten.view.default(sum_156, [4096]);  sum_156 = None
        add_294 = torch.ops.aten.add.Tensor(add_272, view_491);  add_272 = view_491 = None
        view_492 = torch.ops.aten.view.default(mm_108, [1, 512, 4096]);  mm_108 = None
        add_295 = torch.ops.aten.add.Tensor(add_292, view_492);  add_292 = view_492 = None
        permute_434 = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
        add_296 = torch.ops.aten.add.Tensor(add_274, permute_434);  add_274 = permute_434 = None
        mul_329 = torch.ops.aten.mul.Tensor(add_295, primals_22)
        mul_330 = torch.ops.aten.mul.Tensor(mul_329, 4096)
        sum_157 = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
        mul_331 = torch.ops.aten.mul.Tensor(mul_329, mul_25);  mul_329 = None
        sum_158 = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_25, sum_158);  sum_158 = None
        sub_117 = torch.ops.aten.sub.Tensor(mul_330, sum_157);  mul_330 = sum_157 = None
        sub_118 = torch.ops.aten.sub.Tensor(sub_117, mul_332);  sub_117 = mul_332 = None
        mul_333 = torch.ops.aten.mul.Tensor(div_57, sub_118);  div_57 = sub_118 = None
        mul_334 = torch.ops.aten.mul.Tensor(add_295, mul_25);  mul_25 = None
        sum_159 = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
        sum_160 = torch.ops.aten.sum.dim_IntList(add_295, [0, 1]);  add_295 = None
        add_297 = torch.ops.aten.add.Tensor(add_275, sum_159);  add_275 = sum_159 = None
        add_298 = torch.ops.aten.add.Tensor(add_276, sum_160);  add_276 = sum_160 = None
        view_493 = torch.ops.aten.view.default(mul_333, [512, 4096])
        mm_110 = torch.ops.aten.mm.default(view_493, permute_138)
        permute_436 = torch.ops.aten.permute.default(view_493, [1, 0])
        mm_111 = torch.ops.aten.mm.default(permute_436, view_63);  permute_436 = view_63 = None
        permute_437 = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
        view_494 = torch.ops.aten.view.default(sum_161, [4096]);  sum_161 = None
        add_299 = torch.ops.aten.add.Tensor(add_277, view_494);  add_277 = view_494 = None
        view_495 = torch.ops.aten.view.default(mm_110, [1, 512, 16384]);  mm_110 = None
        permute_438 = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
        add_300 = torch.ops.aten.add.Tensor(add_278, permute_438);  add_278 = permute_438 = None
        mul_335 = torch.ops.aten.mul.Tensor(view_495, mul_21);  mul_21 = None
        mul_336 = torch.ops.aten.mul.Tensor(view_495, add_27);  view_495 = add_27 = None
        mul_337 = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_119 = torch.ops.aten.sub.Tensor(1, mul_337);  mul_337 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_335, sub_119);  mul_335 = sub_119 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, 0.7978845608028654);  mul_338 = None
        mul_340 = torch.ops.aten.mul.Tensor(mul_339, 0.044715)
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(view_62, 2.0);  view_62 = None
        mul_341 = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        add_301 = torch.ops.aten.add.Tensor(mul_339, mul_342);  mul_339 = mul_342 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_336, 0.5);  mul_336 = None
        add_302 = torch.ops.aten.add.Tensor(add_301, mul_343);  add_301 = mul_343 = None
        view_496 = torch.ops.aten.view.default(add_302, [512, 16384]);  add_302 = None
        mm_112 = torch.ops.aten.mm.default(view_496, permute_142)
        permute_440 = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_113 = torch.ops.aten.mm.default(permute_440, view_61);  permute_440 = view_61 = None
        permute_441 = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
        sum_162 = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        view_497 = torch.ops.aten.view.default(sum_162, [16384]);  sum_162 = None
        add_303 = torch.ops.aten.add.Tensor(add_281, view_497);  add_281 = view_497 = None
        view_498 = torch.ops.aten.view.default(mm_112, [1, 512, 4096]);  mm_112 = None
        add_304 = torch.ops.aten.add.Tensor(mul_333, view_498);  mul_333 = view_498 = None
        permute_442 = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
        add_305 = torch.ops.aten.add.Tensor(add_283, permute_442);  add_283 = permute_442 = None
        mul_345 = torch.ops.aten.mul.Tensor(add_304, primals_16)
        mul_346 = torch.ops.aten.mul.Tensor(mul_345, 4096)
        sum_163 = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
        mul_347 = torch.ops.aten.mul.Tensor(mul_345, mul_19);  mul_345 = None
        sum_164 = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_19, sum_164);  sum_164 = None
        sub_121 = torch.ops.aten.sub.Tensor(mul_346, sum_163);  mul_346 = sum_163 = None
        sub_122 = torch.ops.aten.sub.Tensor(sub_121, mul_348);  sub_121 = mul_348 = None
        mul_349 = torch.ops.aten.mul.Tensor(div_58, sub_122);  div_58 = sub_122 = None
        mul_350 = torch.ops.aten.mul.Tensor(add_304, mul_19);  mul_19 = None
        sum_165 = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
        sum_166 = torch.ops.aten.sum.dim_IntList(add_304, [0, 1]);  add_304 = None
        add_306 = torch.ops.aten.add.Tensor(add_284, sum_165);  add_284 = sum_165 = None
        add_307 = torch.ops.aten.add.Tensor(add_285, sum_166);  add_285 = sum_166 = None
        view_499 = torch.ops.aten.view.default(mul_349, [512, 4096])
        mm_114 = torch.ops.aten.mm.default(view_499, permute_146)
        permute_444 = torch.ops.aten.permute.default(view_499, [1, 0])
        mm_115 = torch.ops.aten.mm.default(permute_444, view_59);  permute_444 = view_59 = None
        permute_445 = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
        sum_167 = torch.ops.aten.sum.dim_IntList(view_499, [0], True);  view_499 = None
        view_500 = torch.ops.aten.view.default(sum_167, [4096]);  sum_167 = None
        add_308 = torch.ops.aten.add.Tensor(add_286, view_500);  add_286 = view_500 = None
        view_501 = torch.ops.aten.view.default(mm_114, [1, 512, 4096]);  mm_114 = None
        permute_446 = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
        add_309 = torch.ops.aten.add.Tensor(add_287, permute_446);  add_287 = permute_446 = None
        view_502 = torch.ops.aten.view.default(view_501, [1, 512, 64, 64]);  view_501 = None
        permute_447 = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
        view_503 = torch.ops.aten.view.default(permute_447, [64, 512, 64]);  permute_447 = None
        bmm_60 = torch.ops.aten.bmm.default(permute_448, view_503);  permute_448 = None
        bmm_61 = torch.ops.aten.bmm.default(view_503, permute_449);  view_503 = permute_449 = None
        view_504 = torch.ops.aten.view.default(bmm_60, [1, 64, 512, 64]);  bmm_60 = None
        view_505 = torch.ops.aten.view.default(bmm_61, [1, 64, 512, 512]);  bmm_61 = None
        mul_351 = torch.ops.aten.mul.Tensor(view_505, div_5);  view_505 = None
        sum_168 = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
        mul_352 = torch.ops.aten.mul.Tensor(div_5, sum_168);  div_5 = sum_168 = None
        sub_123 = torch.ops.aten.sub.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        div_59 = torch.ops.aten.div.Tensor(sub_123, 8.0);  sub_123 = None
        view_506 = torch.ops.aten.view.default(div_59, [64, 512, 512]);  div_59 = None
        bmm_62 = torch.ops.aten.bmm.default(permute_450, view_506);  permute_450 = None
        bmm_63 = torch.ops.aten.bmm.default(view_506, permute_451);  view_506 = permute_451 = None
        view_507 = torch.ops.aten.view.default(bmm_62, [1, 64, 64, 512]);  bmm_62 = None
        view_508 = torch.ops.aten.view.default(bmm_63, [1, 64, 512, 64]);  bmm_63 = None
        permute_452 = torch.ops.aten.permute.default(view_507, [0, 1, 3, 2]);  view_507 = None
        permute_453 = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        clone_32 = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        _unsafe_view_30 = torch.ops.aten._unsafe_view.default(clone_32, [1, 512, 4096]);  clone_32 = None
        permute_454 = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None
        view_509 = torch.ops.aten.view.default(permute_454, [1, 512, 4096]);  permute_454 = None
        permute_455 = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        clone_33 = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        _unsafe_view_31 = torch.ops.aten._unsafe_view.default(clone_33, [1, 512, 4096]);  clone_33 = None
        view_510 = torch.ops.aten.view.default(_unsafe_view_30, [512, 4096]);  _unsafe_view_30 = None
        mm_116 = torch.ops.aten.mm.default(view_510, permute_159)
        permute_457 = torch.ops.aten.permute.default(view_510, [1, 0])
        mm_117 = torch.ops.aten.mm.default(permute_457, view_44);  permute_457 = None
        permute_458 = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
        sum_169 = torch.ops.aten.sum.dim_IntList(view_510, [0], True);  view_510 = None
        view_511 = torch.ops.aten.view.default(sum_169, [4096]);  sum_169 = None
        add_310 = torch.ops.aten.add.Tensor(add_288, view_511);  add_288 = view_511 = None
        view_512 = torch.ops.aten.view.default(mm_116, [1, 512, 4096]);  mm_116 = None
        add_311 = torch.ops.aten.add.Tensor(mul_349, view_512);  mul_349 = view_512 = None
        permute_459 = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
        add_312 = torch.ops.aten.add.Tensor(add_290, permute_459);  add_290 = permute_459 = None
        view_513 = torch.ops.aten.view.default(view_509, [512, 4096]);  view_509 = None
        mm_118 = torch.ops.aten.mm.default(view_513, permute_163)
        permute_461 = torch.ops.aten.permute.default(view_513, [1, 0])
        mm_119 = torch.ops.aten.mm.default(permute_461, view_44);  permute_461 = None
        permute_462 = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
        sum_170 = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
        view_514 = torch.ops.aten.view.default(sum_170, [4096]);  sum_170 = None
        add_313 = torch.ops.aten.add.Tensor(add_291, view_514);  add_291 = view_514 = None
        view_515 = torch.ops.aten.view.default(mm_118, [1, 512, 4096]);  mm_118 = None
        add_314 = torch.ops.aten.add.Tensor(add_311, view_515);  add_311 = view_515 = None
        permute_463 = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
        add_315 = torch.ops.aten.add.Tensor(add_293, permute_463);  add_293 = permute_463 = None
        view_516 = torch.ops.aten.view.default(_unsafe_view_31, [512, 4096]);  _unsafe_view_31 = None
        mm_120 = torch.ops.aten.mm.default(view_516, permute_167)
        permute_465 = torch.ops.aten.permute.default(view_516, [1, 0])
        mm_121 = torch.ops.aten.mm.default(permute_465, view_44);  permute_465 = view_44 = None
        permute_466 = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
        sum_171 = torch.ops.aten.sum.dim_IntList(view_516, [0], True);  view_516 = None
        view_517 = torch.ops.aten.view.default(sum_171, [4096]);  sum_171 = None
        add_316 = torch.ops.aten.add.Tensor(add_294, view_517);  add_294 = view_517 = None
        view_518 = torch.ops.aten.view.default(mm_120, [1, 512, 4096]);  mm_120 = None
        add_317 = torch.ops.aten.add.Tensor(add_314, view_518);  add_314 = view_518 = None
        permute_467 = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
        add_318 = torch.ops.aten.add.Tensor(add_296, permute_467);  add_296 = permute_467 = None
        mul_354 = torch.ops.aten.mul.Tensor(add_317, primals_22)
        mul_355 = torch.ops.aten.mul.Tensor(mul_354, 4096)
        sum_172 = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
        mul_356 = torch.ops.aten.mul.Tensor(mul_354, mul_17);  mul_354 = None
        sum_173 = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_17, sum_173);  sum_173 = None
        sub_125 = torch.ops.aten.sub.Tensor(mul_355, sum_172);  mul_355 = sum_172 = None
        sub_126 = torch.ops.aten.sub.Tensor(sub_125, mul_357);  sub_125 = mul_357 = None
        mul_358 = torch.ops.aten.mul.Tensor(div_60, sub_126);  div_60 = sub_126 = None
        mul_359 = torch.ops.aten.mul.Tensor(add_317, mul_17);  mul_17 = None
        sum_174 = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
        sum_175 = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None
        add_319 = torch.ops.aten.add.Tensor(add_297, sum_174);  add_297 = sum_174 = None
        add_320 = torch.ops.aten.add.Tensor(add_298, sum_175);  add_298 = sum_175 = None
        view_519 = torch.ops.aten.view.default(mul_358, [512, 4096])
        mm_122 = torch.ops.aten.mm.default(view_519, permute_138)
        permute_469 = torch.ops.aten.permute.default(view_519, [1, 0])
        mm_123 = torch.ops.aten.mm.default(permute_469, view_42);  permute_469 = view_42 = None
        permute_470 = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
        sum_176 = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
        view_520 = torch.ops.aten.view.default(sum_176, [4096]);  sum_176 = None
        add_321 = torch.ops.aten.add.Tensor(add_299, view_520);  add_299 = view_520 = None
        view_521 = torch.ops.aten.view.default(mm_122, [1, 512, 16384]);  mm_122 = None
        permute_471 = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
        add_322 = torch.ops.aten.add.Tensor(add_300, permute_471);  add_300 = permute_471 = None
        mul_360 = torch.ops.aten.mul.Tensor(view_521, mul_13);  mul_13 = None
        mul_361 = torch.ops.aten.mul.Tensor(view_521, add_18);  view_521 = add_18 = None
        mul_362 = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_127 = torch.ops.aten.sub.Tensor(1, mul_362);  mul_362 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_360, sub_127);  mul_360 = sub_127 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, 0.7978845608028654);  mul_363 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, 0.044715)
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(view_41, 2.0);  view_41 = None
        mul_366 = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        add_323 = torch.ops.aten.add.Tensor(mul_364, mul_367);  mul_364 = mul_367 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_361, 0.5);  mul_361 = None
        add_324 = torch.ops.aten.add.Tensor(add_323, mul_368);  add_323 = mul_368 = None
        view_522 = torch.ops.aten.view.default(add_324, [512, 16384]);  add_324 = None
        mm_124 = torch.ops.aten.mm.default(view_522, permute_142)
        permute_473 = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_125 = torch.ops.aten.mm.default(permute_473, view_40);  permute_473 = view_40 = None
        permute_474 = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
        sum_177 = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
        view_523 = torch.ops.aten.view.default(sum_177, [16384]);  sum_177 = None
        add_325 = torch.ops.aten.add.Tensor(add_303, view_523);  add_303 = view_523 = None
        view_524 = torch.ops.aten.view.default(mm_124, [1, 512, 4096]);  mm_124 = None
        add_326 = torch.ops.aten.add.Tensor(mul_358, view_524);  mul_358 = view_524 = None
        permute_475 = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
        add_327 = torch.ops.aten.add.Tensor(add_305, permute_475);  add_305 = permute_475 = None
        mul_370 = torch.ops.aten.mul.Tensor(add_326, primals_16)
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, 4096)
        sum_178 = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
        mul_372 = torch.ops.aten.mul.Tensor(mul_370, mul_11);  mul_370 = None
        sum_179 = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
        mul_373 = torch.ops.aten.mul.Tensor(mul_11, sum_179);  sum_179 = None
        sub_129 = torch.ops.aten.sub.Tensor(mul_371, sum_178);  mul_371 = sum_178 = None
        sub_130 = torch.ops.aten.sub.Tensor(sub_129, mul_373);  sub_129 = mul_373 = None
        mul_374 = torch.ops.aten.mul.Tensor(div_61, sub_130);  div_61 = sub_130 = None
        mul_375 = torch.ops.aten.mul.Tensor(add_326, mul_11);  mul_11 = None
        sum_180 = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
        sum_181 = torch.ops.aten.sum.dim_IntList(add_326, [0, 1]);  add_326 = None
        add_328 = torch.ops.aten.add.Tensor(add_306, sum_180);  add_306 = sum_180 = None
        add_329 = torch.ops.aten.add.Tensor(add_307, sum_181);  add_307 = sum_181 = None
        view_525 = torch.ops.aten.view.default(mul_374, [512, 4096])
        mm_126 = torch.ops.aten.mm.default(view_525, permute_146)
        permute_477 = torch.ops.aten.permute.default(view_525, [1, 0])
        mm_127 = torch.ops.aten.mm.default(permute_477, view_38);  permute_477 = view_38 = None
        permute_478 = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
        sum_182 = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
        view_526 = torch.ops.aten.view.default(sum_182, [4096]);  sum_182 = None
        add_330 = torch.ops.aten.add.Tensor(add_308, view_526);  add_308 = view_526 = None
        view_527 = torch.ops.aten.view.default(mm_126, [1, 512, 4096]);  mm_126 = None
        permute_479 = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
        add_331 = torch.ops.aten.add.Tensor(add_309, permute_479);  add_309 = permute_479 = None
        view_528 = torch.ops.aten.view.default(view_527, [1, 512, 64, 64]);  view_527 = None
        permute_480 = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
        view_529 = torch.ops.aten.view.default(permute_480, [64, 512, 64]);  permute_480 = None
        bmm_64 = torch.ops.aten.bmm.default(permute_481, view_529);  permute_481 = None
        bmm_65 = torch.ops.aten.bmm.default(view_529, permute_482);  view_529 = permute_482 = None
        view_530 = torch.ops.aten.view.default(bmm_64, [1, 64, 512, 64]);  bmm_64 = None
        view_531 = torch.ops.aten.view.default(bmm_65, [1, 64, 512, 512]);  bmm_65 = None
        mul_376 = torch.ops.aten.mul.Tensor(view_531, div_3);  view_531 = None
        sum_183 = torch.ops.aten.sum.dim_IntList(mul_376, [-1], True)
        mul_377 = torch.ops.aten.mul.Tensor(div_3, sum_183);  div_3 = sum_183 = None
        sub_131 = torch.ops.aten.sub.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
        div_62 = torch.ops.aten.div.Tensor(sub_131, 8.0);  sub_131 = None
        view_532 = torch.ops.aten.view.default(div_62, [64, 512, 512]);  div_62 = None
        bmm_66 = torch.ops.aten.bmm.default(permute_483, view_532);  permute_483 = None
        bmm_67 = torch.ops.aten.bmm.default(view_532, permute_484);  view_532 = permute_484 = None
        view_533 = torch.ops.aten.view.default(bmm_66, [1, 64, 64, 512]);  bmm_66 = None
        view_534 = torch.ops.aten.view.default(bmm_67, [1, 64, 512, 64]);  bmm_67 = None
        permute_485 = torch.ops.aten.permute.default(view_533, [0, 1, 3, 2]);  view_533 = None
        permute_486 = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
        clone_34 = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
        _unsafe_view_32 = torch.ops.aten._unsafe_view.default(clone_34, [1, 512, 4096]);  clone_34 = None
        permute_487 = torch.ops.aten.permute.default(permute_485, [0, 2, 1, 3]);  permute_485 = None
        view_535 = torch.ops.aten.view.default(permute_487, [1, 512, 4096]);  permute_487 = None
        permute_488 = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        clone_35 = torch.ops.aten.clone.default(permute_488, memory_format = torch.contiguous_format);  permute_488 = None
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(clone_35, [1, 512, 4096]);  clone_35 = None
        view_536 = torch.ops.aten.view.default(_unsafe_view_32, [512, 4096]);  _unsafe_view_32 = None
        mm_128 = torch.ops.aten.mm.default(view_536, permute_159)
        permute_490 = torch.ops.aten.permute.default(view_536, [1, 0])
        mm_129 = torch.ops.aten.mm.default(permute_490, view_23);  permute_490 = None
        permute_491 = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
        sum_184 = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
        view_537 = torch.ops.aten.view.default(sum_184, [4096]);  sum_184 = None
        add_332 = torch.ops.aten.add.Tensor(add_310, view_537);  add_310 = view_537 = None
        view_538 = torch.ops.aten.view.default(mm_128, [1, 512, 4096]);  mm_128 = None
        add_333 = torch.ops.aten.add.Tensor(mul_374, view_538);  mul_374 = view_538 = None
        permute_492 = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
        add_334 = torch.ops.aten.add.Tensor(add_312, permute_492);  add_312 = permute_492 = None
        view_539 = torch.ops.aten.view.default(view_535, [512, 4096]);  view_535 = None
        mm_130 = torch.ops.aten.mm.default(view_539, permute_163)
        permute_494 = torch.ops.aten.permute.default(view_539, [1, 0])
        mm_131 = torch.ops.aten.mm.default(permute_494, view_23);  permute_494 = None
        permute_495 = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
        sum_185 = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
        view_540 = torch.ops.aten.view.default(sum_185, [4096]);  sum_185 = None
        add_335 = torch.ops.aten.add.Tensor(add_313, view_540);  add_313 = view_540 = None
        view_541 = torch.ops.aten.view.default(mm_130, [1, 512, 4096]);  mm_130 = None
        add_336 = torch.ops.aten.add.Tensor(add_333, view_541);  add_333 = view_541 = None
        permute_496 = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
        add_337 = torch.ops.aten.add.Tensor(add_315, permute_496);  add_315 = permute_496 = None
        view_542 = torch.ops.aten.view.default(_unsafe_view_33, [512, 4096]);  _unsafe_view_33 = None
        mm_132 = torch.ops.aten.mm.default(view_542, permute_167)
        permute_498 = torch.ops.aten.permute.default(view_542, [1, 0])
        mm_133 = torch.ops.aten.mm.default(permute_498, view_23);  permute_498 = view_23 = None
        permute_499 = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
        sum_186 = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
        view_543 = torch.ops.aten.view.default(sum_186, [4096]);  sum_186 = None
        add_338 = torch.ops.aten.add.Tensor(add_316, view_543);  add_316 = view_543 = None
        view_544 = torch.ops.aten.view.default(mm_132, [1, 512, 4096]);  mm_132 = None
        add_339 = torch.ops.aten.add.Tensor(add_336, view_544);  add_336 = view_544 = None
        permute_500 = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
        add_340 = torch.ops.aten.add.Tensor(add_318, permute_500);  add_318 = permute_500 = None
        mul_379 = torch.ops.aten.mul.Tensor(add_339, primals_22);  primals_22 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, 4096)
        sum_187 = torch.ops.aten.sum.dim_IntList(mul_379, [2], True)
        mul_381 = torch.ops.aten.mul.Tensor(mul_379, mul_9);  mul_379 = None
        sum_188 = torch.ops.aten.sum.dim_IntList(mul_381, [2], True);  mul_381 = None
        mul_382 = torch.ops.aten.mul.Tensor(mul_9, sum_188);  sum_188 = None
        sub_133 = torch.ops.aten.sub.Tensor(mul_380, sum_187);  mul_380 = sum_187 = None
        sub_134 = torch.ops.aten.sub.Tensor(sub_133, mul_382);  sub_133 = mul_382 = None
        mul_383 = torch.ops.aten.mul.Tensor(div_63, sub_134);  div_63 = sub_134 = None
        mul_384 = torch.ops.aten.mul.Tensor(add_339, mul_9);  mul_9 = None
        sum_189 = torch.ops.aten.sum.dim_IntList(mul_384, [0, 1]);  mul_384 = None
        sum_190 = torch.ops.aten.sum.dim_IntList(add_339, [0, 1]);  add_339 = None
        add_341 = torch.ops.aten.add.Tensor(add_319, sum_189);  add_319 = sum_189 = None
        add_342 = torch.ops.aten.add.Tensor(add_320, sum_190);  add_320 = sum_190 = None
        view_545 = torch.ops.aten.view.default(mul_383, [512, 4096])
        mm_134 = torch.ops.aten.mm.default(view_545, permute_138);  permute_138 = None
        permute_502 = torch.ops.aten.permute.default(view_545, [1, 0])
        mm_135 = torch.ops.aten.mm.default(permute_502, view_21);  permute_502 = view_21 = None
        permute_503 = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
        sum_191 = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
        view_546 = torch.ops.aten.view.default(sum_191, [4096]);  sum_191 = None
        add_343 = torch.ops.aten.add.Tensor(add_321, view_546);  add_321 = view_546 = None
        view_547 = torch.ops.aten.view.default(mm_134, [1, 512, 16384]);  mm_134 = None
        permute_504 = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
        add_344 = torch.ops.aten.add.Tensor(add_322, permute_504);  add_322 = permute_504 = None
        mul_385 = torch.ops.aten.mul.Tensor(view_547, mul_5);  mul_5 = None
        mul_386 = torch.ops.aten.mul.Tensor(view_547, add_9);  view_547 = add_9 = None
        mul_387 = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_135 = torch.ops.aten.sub.Tensor(1, mul_387);  mul_387 = None
        mul_388 = torch.ops.aten.mul.Tensor(mul_385, sub_135);  mul_385 = sub_135 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, 0.7978845608028654);  mul_388 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_389, 0.044715)
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(view_20, 2.0);  view_20 = None
        mul_391 = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
        add_345 = torch.ops.aten.add.Tensor(mul_389, mul_392);  mul_389 = mul_392 = None
        mul_393 = torch.ops.aten.mul.Tensor(mul_386, 0.5);  mul_386 = None
        add_346 = torch.ops.aten.add.Tensor(add_345, mul_393);  add_345 = mul_393 = None
        view_548 = torch.ops.aten.view.default(add_346, [512, 16384]);  add_346 = None
        mm_136 = torch.ops.aten.mm.default(view_548, permute_142);  permute_142 = None
        permute_506 = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_137 = torch.ops.aten.mm.default(permute_506, view_19);  permute_506 = view_19 = None
        permute_507 = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
        sum_192 = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        view_549 = torch.ops.aten.view.default(sum_192, [16384]);  sum_192 = None
        add_347 = torch.ops.aten.add.Tensor(add_325, view_549);  add_325 = view_549 = None
        view_550 = torch.ops.aten.view.default(mm_136, [1, 512, 4096]);  mm_136 = None
        add_348 = torch.ops.aten.add.Tensor(mul_383, view_550);  mul_383 = view_550 = None
        permute_508 = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
        add_349 = torch.ops.aten.add.Tensor(add_327, permute_508);  add_327 = permute_508 = None
        mul_395 = torch.ops.aten.mul.Tensor(add_348, primals_16);  primals_16 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_395, 4096)
        sum_193 = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397 = torch.ops.aten.mul.Tensor(mul_395, mul_3);  mul_395 = None
        sum_194 = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_3, sum_194);  sum_194 = None
        sub_137 = torch.ops.aten.sub.Tensor(mul_396, sum_193);  mul_396 = sum_193 = None
        sub_138 = torch.ops.aten.sub.Tensor(sub_137, mul_398);  sub_137 = mul_398 = None
        mul_399 = torch.ops.aten.mul.Tensor(div_64, sub_138);  div_64 = sub_138 = None
        mul_400 = torch.ops.aten.mul.Tensor(add_348, mul_3);  mul_3 = None
        sum_195 = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_196 = torch.ops.aten.sum.dim_IntList(add_348, [0, 1]);  add_348 = None
        add_350 = torch.ops.aten.add.Tensor(add_328, sum_195);  add_328 = sum_195 = None
        add_351 = torch.ops.aten.add.Tensor(add_329, sum_196);  add_329 = sum_196 = None
        view_551 = torch.ops.aten.view.default(mul_399, [512, 4096])
        mm_138 = torch.ops.aten.mm.default(view_551, permute_146);  permute_146 = None
        permute_510 = torch.ops.aten.permute.default(view_551, [1, 0])
        mm_139 = torch.ops.aten.mm.default(permute_510, view_17);  permute_510 = view_17 = None
        permute_511 = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
        sum_197 = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
        view_552 = torch.ops.aten.view.default(sum_197, [4096]);  sum_197 = None
        add_352 = torch.ops.aten.add.Tensor(add_330, view_552);  add_330 = view_552 = None
        view_553 = torch.ops.aten.view.default(mm_138, [1, 512, 4096]);  mm_138 = None
        permute_512 = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
        add_353 = torch.ops.aten.add.Tensor(add_331, permute_512);  add_331 = permute_512 = None
        view_554 = torch.ops.aten.view.default(view_553, [1, 512, 64, 64]);  view_553 = None
        permute_513 = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        view_555 = torch.ops.aten.view.default(permute_513, [64, 512, 64]);  permute_513 = None
        bmm_68 = torch.ops.aten.bmm.default(permute_514, view_555);  permute_514 = None
        bmm_69 = torch.ops.aten.bmm.default(view_555, permute_515);  view_555 = permute_515 = None
        view_556 = torch.ops.aten.view.default(bmm_68, [1, 64, 512, 64]);  bmm_68 = None
        view_557 = torch.ops.aten.view.default(bmm_69, [1, 64, 512, 512]);  bmm_69 = None
        mul_401 = torch.ops.aten.mul.Tensor(view_557, div_1);  view_557 = None
        sum_198 = torch.ops.aten.sum.dim_IntList(mul_401, [-1], True)
        mul_402 = torch.ops.aten.mul.Tensor(div_1, sum_198);  div_1 = sum_198 = None
        sub_139 = torch.ops.aten.sub.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
        div_65 = torch.ops.aten.div.Tensor(sub_139, 8.0);  sub_139 = None
        view_558 = torch.ops.aten.view.default(div_65, [64, 512, 512]);  div_65 = None
        bmm_70 = torch.ops.aten.bmm.default(permute_516, view_558);  permute_516 = None
        bmm_71 = torch.ops.aten.bmm.default(view_558, permute_517);  view_558 = permute_517 = None
        view_559 = torch.ops.aten.view.default(bmm_70, [1, 64, 64, 512]);  bmm_70 = None
        view_560 = torch.ops.aten.view.default(bmm_71, [1, 64, 512, 64]);  bmm_71 = None
        permute_518 = torch.ops.aten.permute.default(view_559, [0, 1, 3, 2]);  view_559 = None
        permute_519 = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        clone_36 = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(clone_36, [1, 512, 4096]);  clone_36 = None
        permute_520 = torch.ops.aten.permute.default(permute_518, [0, 2, 1, 3]);  permute_518 = None
        view_561 = torch.ops.aten.view.default(permute_520, [1, 512, 4096]);  permute_520 = None
        permute_521 = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
        clone_37 = torch.ops.aten.clone.default(permute_521, memory_format = torch.contiguous_format);  permute_521 = None
        _unsafe_view_35 = torch.ops.aten._unsafe_view.default(clone_37, [1, 512, 4096]);  clone_37 = None
        view_562 = torch.ops.aten.view.default(_unsafe_view_34, [512, 4096]);  _unsafe_view_34 = None
        mm_140 = torch.ops.aten.mm.default(view_562, permute_159);  permute_159 = None
        permute_523 = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_141 = torch.ops.aten.mm.default(permute_523, view_2);  permute_523 = None
        permute_524 = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
        sum_199 = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        view_563 = torch.ops.aten.view.default(sum_199, [4096]);  sum_199 = None
        add_354 = torch.ops.aten.add.Tensor(add_332, view_563);  add_332 = view_563 = None
        view_564 = torch.ops.aten.view.default(mm_140, [1, 512, 4096]);  mm_140 = None
        add_355 = torch.ops.aten.add.Tensor(mul_399, view_564);  mul_399 = view_564 = None
        permute_525 = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
        add_356 = torch.ops.aten.add.Tensor(add_334, permute_525);  add_334 = permute_525 = None
        view_565 = torch.ops.aten.view.default(view_561, [512, 4096]);  view_561 = None
        mm_142 = torch.ops.aten.mm.default(view_565, permute_163);  permute_163 = None
        permute_527 = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_143 = torch.ops.aten.mm.default(permute_527, view_2);  permute_527 = None
        permute_528 = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
        sum_200 = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        view_566 = torch.ops.aten.view.default(sum_200, [4096]);  sum_200 = None
        add_357 = torch.ops.aten.add.Tensor(add_335, view_566);  add_335 = view_566 = None
        view_567 = torch.ops.aten.view.default(mm_142, [1, 512, 4096]);  mm_142 = None
        add_358 = torch.ops.aten.add.Tensor(add_355, view_567);  add_355 = view_567 = None
        permute_529 = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
        add_359 = torch.ops.aten.add.Tensor(add_337, permute_529);  add_337 = permute_529 = None
        view_568 = torch.ops.aten.view.default(_unsafe_view_35, [512, 4096]);  _unsafe_view_35 = None
        mm_144 = torch.ops.aten.mm.default(view_568, permute_167);  permute_167 = None
        permute_531 = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_145 = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = view_2 = None
        permute_532 = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
        sum_201 = torch.ops.aten.sum.dim_IntList(view_568, [0], True);  view_568 = None
        view_569 = torch.ops.aten.view.default(sum_201, [4096]);  sum_201 = None
        add_360 = torch.ops.aten.add.Tensor(add_338, view_569);  add_338 = view_569 = None
        view_570 = torch.ops.aten.view.default(mm_144, [1, 512, 4096]);  mm_144 = None
        add_361 = torch.ops.aten.add.Tensor(add_358, view_570);  add_358 = view_570 = None
        permute_533 = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
        add_362 = torch.ops.aten.add.Tensor(add_340, permute_533);  add_340 = permute_533 = None
        view_571 = torch.ops.aten.view.default(add_361, [512, 4096]);  add_361 = None
        mm_146 = torch.ops.aten.mm.default(view_571, permute_534);  permute_534 = None
        permute_535 = torch.ops.aten.permute.default(view_571, [1, 0])
        mm_147 = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
        permute_536 = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
        sum_202 = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
        view_572 = torch.ops.aten.view.default(sum_202, [4096]);  sum_202 = None
        view_573 = torch.ops.aten.view.default(mm_146, [1, 512, 128]);  mm_146 = None
        permute_537 = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
        mul_404 = torch.ops.aten.mul.Tensor(view_573, primals_4);  primals_4 = None
        mul_405 = torch.ops.aten.mul.Tensor(mul_404, 128)
        sum_203 = torch.ops.aten.sum.dim_IntList(mul_404, [2], True)
        mul_406 = torch.ops.aten.mul.Tensor(mul_404, mul_1);  mul_404 = None
        sum_204 = torch.ops.aten.sum.dim_IntList(mul_406, [2], True);  mul_406 = None
        mul_407 = torch.ops.aten.mul.Tensor(mul_1, sum_204);  sum_204 = None
        sub_141 = torch.ops.aten.sub.Tensor(mul_405, sum_203);  mul_405 = sum_203 = None
        sub_142 = torch.ops.aten.sub.Tensor(sub_141, mul_407);  sub_141 = mul_407 = None
        mul_408 = torch.ops.aten.mul.Tensor(div_66, sub_142);  div_66 = sub_142 = None
        mul_409 = torch.ops.aten.mul.Tensor(view_573, mul_1);  mul_1 = None
        sum_205 = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1]);  mul_409 = None
        sum_206 = torch.ops.aten.sum.dim_IntList(view_573, [0, 1]);  view_573 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.int64);  convert_element_type_2 = None
        eq = torch.ops.aten.eq.Scalar(convert_element_type_3, -1)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_4 = torch.ops.aten.where.self(unsqueeze_8, scalar_tensor, mul_408);  unsqueeze_8 = None
        full_1 = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full_1, [convert_element_type_3], where_4, True);  full_1 = convert_element_type_3 = where_4 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.int64);  convert_element_type_4 = None
        eq_1 = torch.ops.aten.eq.Scalar(convert_element_type_5, -1)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_5 = torch.ops.aten.where.self(unsqueeze_9, scalar_tensor, mul_408);  unsqueeze_9 = None
        full_2 = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1 = torch.ops.aten.index_put.default(full_2, [convert_element_type_5], where_5, True);  full_2 = convert_element_type_5 = where_5 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.int64);  convert_element_type_6 = None
        eq_2 = torch.ops.aten.eq.Scalar(convert_element_type_7, 0)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_6 = torch.ops.aten.where.self(unsqueeze_10, scalar_tensor, mul_408);  unsqueeze_10 = scalar_tensor = mul_408 = None
        full_3 = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2 = torch.ops.aten.index_put.default(full_3, [convert_element_type_7], where_6, True);  full_3 = convert_element_type_7 = where_6 = None
        return [index_put_2, index_put_1, index_put, sum_205, sum_206, permute_537, view_572, add_362, add_360, add_359, add_357, add_356, add_354, add_353, add_352, add_350, add_351, add_349, add_347, add_344, add_343, add_341, add_342, permute_137, view_257, None, None, None, None, None]
        
args = [((128,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((4096,), (1,), torch.float32, 'cuda'), ((1, 512, 128), (65536, 128, 1), torch.float32, 'cuda'), ((512, 128), (128, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 64, 512, 512), (16777216, 262144, 512, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 16384), (8388608, 16384, 1), torch.float32, 'cuda'), ((512, 16384), (16384, 1), torch.float32, 'cuda'), ((1, 512, 4096), (2097152, 4096, 1), torch.float32, 'cuda'), ((512, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.float32, 'cuda'), ((1, 1), (1, 1), torch.int64, 'cuda'), ((1,), (1,), torch.bool, 'cuda'), ((1, 512), (512, 1), torch.float32, 'cuda'), ((1, 1), (1, 1), torch.int64, 'cuda'), ((1,), (1,), torch.bool, 'cuda'), ((2, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((4096, 16384), (16384, 1), torch.float32, 'cuda'), ((16384, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((4096, 4096), (4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((64, 512, 512), (262144, 1, 512), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 64, 512), (64, 1, 4096), torch.float32, 'cuda'), ((64, 512, 64), (64, 4096, 1), torch.float32, 'cuda'), ((4096, 128), (128, 1), torch.float32, 'cuda'), ((1, 512, 1), (512, 1, 1), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.float32, 'cuda'), ((), (), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
