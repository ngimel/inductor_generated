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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x88X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\x13\x00\x00\x00torch.distributionsq\x18X\x0b\x00\x00\x00torch._refsq\x19X\r\x00\x00\x00torch.testingq\x1aX\r\x00\x00\x00torch._decompq\x1bX\x0c\x00\x00\x00torch._primsq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x18\x00\x00\x00error_on_nested_fx_traceq&\x88X\t\x00\x00\x00allow_rnnq\'\x89X\x08\x00\x00\x00base_dirq(X\x1c\x00\x00\x00/scratch/ngimel/work/pytorchq)X\x0e\x00\x00\x00debug_dir_rootq*X0\x00\x00\x00/scratch/ngimel/work/pytorch/torch_compile_debugq+X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq,\x89X\x13\x00\x00\x00_save_config_ignoreq-h\r]q.(X!\x00\x00\x00skipfiles_inline_module_allowlistq/X\x0b\x00\x00\x00repro_levelq0X\x12\x00\x00\x00constant_functionsq1X\x0b\x00\x00\x00repro_afterq2e\x85q3Rq4u.')
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_48, primals_52, primals_57, primals_62, primals_67, primals_72, primals_77, primals_82, primals_87, primals_92, primals_97, primals_102, primals_107, primals_112, primals_117, primals_122, primals_127, primals_132, primals_137, primals_142, primals_147, primals_152, primals_157, primals_162, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean_8, div, mul_56, getitem, getitem_1, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_14, div_1, mul_92, getitem_2, getitem_3, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_20, div_2, mul_128, getitem_4, getitem_5, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_26, div_3, view, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70):
        mm = torch.ops.aten.mm.default(tangents_47, permute_1);  permute_1 = None
        permute_2 = torch.ops.aten.permute.default(tangents_47, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        permute_3 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_47, [0], True);  tangents_47 = None
        view_1 = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        permute_4 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        view_2 = torch.ops.aten.view.default(mm, [8, 1024, 1, 1]);  mm = None
        expand = torch.ops.aten.expand.default(view_2, [8, 1024, 7, 7]);  view_2 = None
        div_4 = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        mul_165 = torch.ops.aten.mul.Tensor(div_4, relu_22)
        mul_166 = torch.ops.aten.mul.Tensor(div_4, div_3);  div_4 = div_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_165, [2, 3], True);  mul_165 = None
        mul_167 = torch.ops.aten.mul.Tensor(sum_2, 0.16666666666666666);  sum_2 = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(bitwise_and, mul_167, scalar_tensor);  bitwise_and = mul_167 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(where, mean_26, primals_44, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_26 = primals_44 = None
        getitem_6 = convolution_backward[0]
        getitem_7 = convolution_backward[1];  convolution_backward = None
        expand_1 = torch.ops.aten.expand.default(getitem_6, [8, 1024, 7, 7]);  getitem_6 = None
        div_5 = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
        add_119 = torch.ops.aten.add.Tensor(mul_166, div_5);  mul_166 = div_5 = None
        le = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_1 = torch.ops.aten.where.self(le, scalar_tensor, add_119);  le = add_119 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_23 = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_94);  convolution_39 = unsqueeze_94 = None
        mul_168 = torch.ops.aten.mul.Tensor(where_1, sub_23)
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_168, [0, 2, 3]);  mul_168 = None
        mul_169 = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        mul_170 = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
        mul_171 = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_172 = torch.ops.aten.mul.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        mul_173 = torch.ops.aten.mul.Tensor(squeeze_67, primals_162);  primals_162 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(mul_173, 0);  mul_173 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
        mul_174 = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_100);  sub_23 = unsqueeze_100 = None
        sub_25 = torch.ops.aten.sub.Tensor(where_1, mul_174);  where_1 = mul_174 = None
        sub_26 = torch.ops.aten.sub.Tensor(sub_25, unsqueeze_97);  sub_25 = unsqueeze_97 = None
        mul_175 = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_103);  sub_26 = unsqueeze_103 = None
        mul_176 = torch.ops.aten.mul.Tensor(sum_5, squeeze_67);  sum_5 = squeeze_67 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_175, cat_3, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_175 = cat_3 = primals_43 = None
        getitem_9 = convolution_backward_1[0]
        getitem_10 = convolution_backward_1[1];  convolution_backward_1 = None
        slice_1 = torch.ops.aten.slice.Tensor(getitem_9, 1, 0, 768)
        slice_2 = torch.ops.aten.slice.Tensor(getitem_9, 1, 768, 992)
        slice_3 = torch.ops.aten.slice.Tensor(getitem_9, 1, 992, 1216)
        slice_4 = torch.ops.aten.slice.Tensor(getitem_9, 1, 1216, 1440);  getitem_9 = None
        where_2 = torch.ops.aten.where.self(le_1, scalar_tensor, slice_4);  le_1 = slice_4 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_27 = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_106);  convolution_38 = unsqueeze_106 = None
        mul_177 = torch.ops.aten.mul.Tensor(where_2, sub_27)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_177, [0, 2, 3]);  mul_177 = None
        mul_178 = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(mul_178, 0);  mul_178 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, 3);  unsqueeze_108 = None
        mul_179 = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
        mul_180 = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_181 = torch.ops.aten.mul.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
        mul_182 = torch.ops.aten.mul.Tensor(squeeze_64, primals_157);  primals_157 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(mul_182, 0);  mul_182 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_112);  sub_27 = unsqueeze_112 = None
        sub_29 = torch.ops.aten.sub.Tensor(where_2, mul_183);  where_2 = mul_183 = None
        sub_30 = torch.ops.aten.sub.Tensor(sub_29, unsqueeze_109);  sub_29 = unsqueeze_109 = None
        mul_184 = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_115);  sub_30 = unsqueeze_115 = None
        mul_185 = torch.ops.aten.mul.Tensor(sum_7, squeeze_64);  sum_7 = squeeze_64 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_184, convolution_37, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_184 = convolution_37 = primals_42 = None
        getitem_12 = convolution_backward_2[0]
        getitem_13 = convolution_backward_2[1];  convolution_backward_2 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_12, relu_20, primals_41, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_12 = primals_41 = None
        getitem_15 = convolution_backward_3[0]
        getitem_16 = convolution_backward_3[1];  convolution_backward_3 = None
        add_120 = torch.ops.aten.add.Tensor(slice_3, getitem_15);  slice_3 = getitem_15 = None
        le_2 = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_3 = torch.ops.aten.where.self(le_2, scalar_tensor, add_120);  le_2 = add_120 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_31 = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_118);  convolution_36 = unsqueeze_118 = None
        mul_186 = torch.ops.aten.mul.Tensor(where_3, sub_31)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_186, [0, 2, 3]);  mul_186 = None
        mul_187 = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(mul_187, 0);  mul_187 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, 3);  unsqueeze_120 = None
        mul_188 = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
        mul_189 = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_190 = torch.ops.aten.mul.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(mul_190, 0);  mul_190 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(unsqueeze_123, 3);  unsqueeze_123 = None
        mul_191 = torch.ops.aten.mul.Tensor(squeeze_61, primals_152);  primals_152 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(mul_191, 0);  mul_191 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
        mul_192 = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_124);  sub_31 = unsqueeze_124 = None
        sub_33 = torch.ops.aten.sub.Tensor(where_3, mul_192);  where_3 = mul_192 = None
        sub_34 = torch.ops.aten.sub.Tensor(sub_33, unsqueeze_121);  sub_33 = unsqueeze_121 = None
        mul_193 = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_127);  sub_34 = unsqueeze_127 = None
        mul_194 = torch.ops.aten.mul.Tensor(sum_9, squeeze_61);  sum_9 = squeeze_61 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_193, convolution_35, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_193 = convolution_35 = primals_40 = None
        getitem_18 = convolution_backward_4[0]
        getitem_19 = convolution_backward_4[1];  convolution_backward_4 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_18, relu_19, primals_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_18 = primals_39 = None
        getitem_21 = convolution_backward_5[0]
        getitem_22 = convolution_backward_5[1];  convolution_backward_5 = None
        add_121 = torch.ops.aten.add.Tensor(slice_2, getitem_21);  slice_2 = getitem_21 = None
        le_3 = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_4 = torch.ops.aten.where.self(le_3, scalar_tensor, add_121);  le_3 = add_121 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_35 = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130);  convolution_34 = unsqueeze_130 = None
        mul_195 = torch.ops.aten.mul.Tensor(where_4, sub_35)
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_195, [0, 2, 3]);  mul_195 = None
        mul_196 = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(mul_196, 0);  mul_196 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
        mul_197 = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
        mul_198 = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_199 = torch.ops.aten.mul.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(mul_199, 0);  mul_199 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
        mul_200 = torch.ops.aten.mul.Tensor(squeeze_58, primals_147);  primals_147 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(mul_200, 0);  mul_200 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
        mul_201 = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_136);  sub_35 = unsqueeze_136 = None
        sub_37 = torch.ops.aten.sub.Tensor(where_4, mul_201);  where_4 = mul_201 = None
        sub_38 = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_133);  sub_37 = unsqueeze_133 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_139);  sub_38 = unsqueeze_139 = None
        mul_203 = torch.ops.aten.mul.Tensor(sum_11, squeeze_58);  sum_11 = squeeze_58 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_202, convolution_33, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_202 = convolution_33 = primals_38 = None
        getitem_24 = convolution_backward_6[0]
        getitem_25 = convolution_backward_6[1];  convolution_backward_6 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_24, relu_18, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_24 = primals_37 = None
        getitem_27 = convolution_backward_7[0]
        getitem_28 = convolution_backward_7[1];  convolution_backward_7 = None
        le_4 = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_5 = torch.ops.aten.where.self(le_4, scalar_tensor, getitem_27);  le_4 = getitem_27 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_39 = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_142);  convolution_32 = unsqueeze_142 = None
        mul_204 = torch.ops.aten.mul.Tensor(where_5, sub_39)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
        mul_205 = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(mul_205, 0);  mul_205 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
        mul_206 = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
        mul_207 = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_208 = torch.ops.aten.mul.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(mul_208, 0);  mul_208 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
        mul_209 = torch.ops.aten.mul.Tensor(squeeze_55, primals_142);  primals_142 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_148);  sub_39 = unsqueeze_148 = None
        sub_41 = torch.ops.aten.sub.Tensor(where_5, mul_210);  where_5 = mul_210 = None
        sub_42 = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_145);  sub_41 = unsqueeze_145 = None
        mul_211 = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_151);  sub_42 = unsqueeze_151 = None
        mul_212 = torch.ops.aten.mul.Tensor(sum_13, squeeze_55);  sum_13 = squeeze_55 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_211, getitem_4, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = getitem_4 = primals_36 = None
        getitem_30 = convolution_backward_8[0]
        getitem_31 = convolution_backward_8[1];  convolution_backward_8 = None
        add_122 = torch.ops.aten.add.Tensor(slice_1, getitem_30);  slice_1 = getitem_30 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(add_122, mul_128, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_5);  add_122 = mul_128 = getitem_5 = None
        mul_213 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, relu_17)
        mul_214 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, div_2);  max_pool2d_with_indices_backward = div_2 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_213, [2, 3], True);  mul_213 = None
        mul_215 = torch.ops.aten.mul.Tensor(sum_14, 0.16666666666666666);  sum_14 = None
        where_6 = torch.ops.aten.where.self(bitwise_and_1, mul_215, scalar_tensor);  bitwise_and_1 = mul_215 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_6, mean_20, primals_34, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_20 = primals_34 = None
        getitem_33 = convolution_backward_9[0]
        getitem_34 = convolution_backward_9[1];  convolution_backward_9 = None
        expand_2 = torch.ops.aten.expand.default(getitem_33, [8, 768, 14, 14]);  getitem_33 = None
        div_6 = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None
        add_123 = torch.ops.aten.add.Tensor(mul_214, div_6);  mul_214 = div_6 = None
        le_5 = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_7 = torch.ops.aten.where.self(le_5, scalar_tensor, add_123);  le_5 = add_123 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_43 = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_154);  convolution_30 = unsqueeze_154 = None
        mul_216 = torch.ops.aten.mul.Tensor(where_7, sub_43)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_216, [0, 2, 3]);  mul_216 = None
        mul_217 = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
        mul_218 = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
        mul_219 = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_220 = torch.ops.aten.mul.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(mul_220, 0);  mul_220 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
        mul_221 = torch.ops.aten.mul.Tensor(squeeze_52, primals_137);  primals_137 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(mul_221, 0);  mul_221 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_160);  sub_43 = unsqueeze_160 = None
        sub_45 = torch.ops.aten.sub.Tensor(where_7, mul_222);  where_7 = mul_222 = None
        sub_46 = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_157);  sub_45 = unsqueeze_157 = None
        mul_223 = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_163);  sub_46 = unsqueeze_163 = None
        mul_224 = torch.ops.aten.mul.Tensor(sum_17, squeeze_52);  sum_17 = squeeze_52 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_223, cat_2, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_223 = cat_2 = primals_33 = None
        getitem_36 = convolution_backward_10[0]
        getitem_37 = convolution_backward_10[1];  convolution_backward_10 = None
        slice_5 = torch.ops.aten.slice.Tensor(getitem_36, 1, 0, 512)
        slice_6 = torch.ops.aten.slice.Tensor(getitem_36, 1, 512, 704)
        slice_7 = torch.ops.aten.slice.Tensor(getitem_36, 1, 704, 896)
        slice_8 = torch.ops.aten.slice.Tensor(getitem_36, 1, 896, 1088);  getitem_36 = None
        where_8 = torch.ops.aten.where.self(le_6, scalar_tensor, slice_8);  le_6 = slice_8 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_47 = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_166);  convolution_29 = unsqueeze_166 = None
        mul_225 = torch.ops.aten.mul.Tensor(where_8, sub_47)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3]);  mul_225 = None
        mul_226 = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
        mul_227 = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
        mul_228 = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_229 = torch.ops.aten.mul.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(mul_229, 0);  mul_229 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
        mul_230 = torch.ops.aten.mul.Tensor(squeeze_49, primals_132);  primals_132 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
        mul_231 = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_172);  sub_47 = unsqueeze_172 = None
        sub_49 = torch.ops.aten.sub.Tensor(where_8, mul_231);  where_8 = mul_231 = None
        sub_50 = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_169);  sub_49 = unsqueeze_169 = None
        mul_232 = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_175);  sub_50 = unsqueeze_175 = None
        mul_233 = torch.ops.aten.mul.Tensor(sum_19, squeeze_49);  sum_19 = squeeze_49 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_232, convolution_28, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_232 = convolution_28 = primals_32 = None
        getitem_39 = convolution_backward_11[0]
        getitem_40 = convolution_backward_11[1];  convolution_backward_11 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(getitem_39, relu_15, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_39 = primals_31 = None
        getitem_42 = convolution_backward_12[0]
        getitem_43 = convolution_backward_12[1];  convolution_backward_12 = None
        add_124 = torch.ops.aten.add.Tensor(slice_7, getitem_42);  slice_7 = getitem_42 = None
        le_7 = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_9 = torch.ops.aten.where.self(le_7, scalar_tensor, add_124);  le_7 = add_124 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_51 = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_178);  convolution_27 = unsqueeze_178 = None
        mul_234 = torch.ops.aten.mul.Tensor(where_9, sub_51)
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_234, [0, 2, 3]);  mul_234 = None
        mul_235 = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
        mul_236 = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
        mul_237 = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_238 = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
        mul_239 = torch.ops.aten.mul.Tensor(squeeze_46, primals_127);  primals_127 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(mul_239, 0);  mul_239 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        mul_240 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_184);  sub_51 = unsqueeze_184 = None
        sub_53 = torch.ops.aten.sub.Tensor(where_9, mul_240);  where_9 = mul_240 = None
        sub_54 = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_181);  sub_53 = unsqueeze_181 = None
        mul_241 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_187);  sub_54 = unsqueeze_187 = None
        mul_242 = torch.ops.aten.mul.Tensor(sum_21, squeeze_46);  sum_21 = squeeze_46 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_241, convolution_26, primals_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_241 = convolution_26 = primals_30 = None
        getitem_45 = convolution_backward_13[0]
        getitem_46 = convolution_backward_13[1];  convolution_backward_13 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(getitem_45, relu_14, primals_29, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_45 = primals_29 = None
        getitem_48 = convolution_backward_14[0]
        getitem_49 = convolution_backward_14[1];  convolution_backward_14 = None
        add_125 = torch.ops.aten.add.Tensor(slice_6, getitem_48);  slice_6 = getitem_48 = None
        le_8 = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_10 = torch.ops.aten.where.self(le_8, scalar_tensor, add_125);  le_8 = add_125 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_55 = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_190);  convolution_25 = unsqueeze_190 = None
        mul_243 = torch.ops.aten.mul.Tensor(where_10, sub_55)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_243, [0, 2, 3]);  mul_243 = None
        mul_244 = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
        mul_245 = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
        mul_246 = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_247 = torch.ops.aten.mul.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(mul_247, 0);  mul_247 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
        mul_248 = torch.ops.aten.mul.Tensor(squeeze_43, primals_122);  primals_122 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(mul_248, 0);  mul_248 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
        mul_249 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_196);  sub_55 = unsqueeze_196 = None
        sub_57 = torch.ops.aten.sub.Tensor(where_10, mul_249);  where_10 = mul_249 = None
        sub_58 = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_193);  sub_57 = unsqueeze_193 = None
        mul_250 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_199);  sub_58 = unsqueeze_199 = None
        mul_251 = torch.ops.aten.mul.Tensor(sum_23, squeeze_43);  sum_23 = squeeze_43 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_250, convolution_24, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_250 = convolution_24 = primals_28 = None
        getitem_51 = convolution_backward_15[0]
        getitem_52 = convolution_backward_15[1];  convolution_backward_15 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(getitem_51, relu_13, primals_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_51 = primals_27 = None
        getitem_54 = convolution_backward_16[0]
        getitem_55 = convolution_backward_16[1];  convolution_backward_16 = None
        le_9 = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_11 = torch.ops.aten.where.self(le_9, scalar_tensor, getitem_54);  le_9 = getitem_54 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_59 = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_202);  convolution_23 = unsqueeze_202 = None
        mul_252 = torch.ops.aten.mul.Tensor(where_11, sub_59)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
        mul_253 = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
        mul_254 = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
        mul_255 = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_256 = torch.ops.aten.mul.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
        mul_257 = torch.ops.aten.mul.Tensor(squeeze_40, primals_117);  primals_117 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_208);  sub_59 = unsqueeze_208 = None
        sub_61 = torch.ops.aten.sub.Tensor(where_11, mul_258);  where_11 = mul_258 = None
        sub_62 = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_205);  sub_61 = unsqueeze_205 = None
        mul_259 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_211);  sub_62 = unsqueeze_211 = None
        mul_260 = torch.ops.aten.mul.Tensor(sum_25, squeeze_40);  sum_25 = squeeze_40 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_259, getitem_2, primals_26, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_259 = getitem_2 = primals_26 = None
        getitem_57 = convolution_backward_17[0]
        getitem_58 = convolution_backward_17[1];  convolution_backward_17 = None
        add_126 = torch.ops.aten.add.Tensor(slice_5, getitem_57);  slice_5 = getitem_57 = None
        max_pool2d_with_indices_backward_1 = torch.ops.aten.max_pool2d_with_indices_backward.default(add_126, mul_92, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_3);  add_126 = mul_92 = getitem_3 = None
        mul_261 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_1, relu_12)
        mul_262 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_1, div_1);  max_pool2d_with_indices_backward_1 = div_1 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_261, [2, 3], True);  mul_261 = None
        mul_263 = torch.ops.aten.mul.Tensor(sum_26, 0.16666666666666666);  sum_26 = None
        where_12 = torch.ops.aten.where.self(bitwise_and_2, mul_263, scalar_tensor);  bitwise_and_2 = mul_263 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_12, mean_14, primals_24, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_12 = mean_14 = primals_24 = None
        getitem_60 = convolution_backward_18[0]
        getitem_61 = convolution_backward_18[1];  convolution_backward_18 = None
        expand_3 = torch.ops.aten.expand.default(getitem_60, [8, 512, 28, 28]);  getitem_60 = None
        div_7 = torch.ops.aten.div.Scalar(expand_3, 784);  expand_3 = None
        add_127 = torch.ops.aten.add.Tensor(mul_262, div_7);  mul_262 = div_7 = None
        le_10 = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_13 = torch.ops.aten.where.self(le_10, scalar_tensor, add_127);  le_10 = add_127 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_63 = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_214);  convolution_21 = unsqueeze_214 = None
        mul_264 = torch.ops.aten.mul.Tensor(where_13, sub_63)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3]);  mul_264 = None
        mul_265 = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_266 = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
        mul_267 = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_268 = torch.ops.aten.mul.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
        mul_269 = torch.ops.aten.mul.Tensor(squeeze_37, primals_112);  primals_112 = None
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
        mul_270 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_220);  sub_63 = unsqueeze_220 = None
        sub_65 = torch.ops.aten.sub.Tensor(where_13, mul_270);  where_13 = mul_270 = None
        sub_66 = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_217);  sub_65 = unsqueeze_217 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_223);  sub_66 = unsqueeze_223 = None
        mul_272 = torch.ops.aten.mul.Tensor(sum_29, squeeze_37);  sum_29 = squeeze_37 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_271, cat_1, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_271 = cat_1 = primals_23 = None
        getitem_63 = convolution_backward_19[0]
        getitem_64 = convolution_backward_19[1];  convolution_backward_19 = None
        slice_9 = torch.ops.aten.slice.Tensor(getitem_63, 1, 0, 256)
        slice_10 = torch.ops.aten.slice.Tensor(getitem_63, 1, 256, 416)
        slice_11 = torch.ops.aten.slice.Tensor(getitem_63, 1, 416, 576)
        slice_12 = torch.ops.aten.slice.Tensor(getitem_63, 1, 576, 736);  getitem_63 = None
        where_14 = torch.ops.aten.where.self(le_11, scalar_tensor, slice_12);  le_11 = slice_12 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_67 = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_226);  convolution_20 = unsqueeze_226 = None
        mul_273 = torch.ops.aten.mul.Tensor(where_14, sub_67)
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3]);  mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
        mul_275 = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        mul_276 = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_277 = torch.ops.aten.mul.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(mul_277, 0);  mul_277 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
        mul_278 = torch.ops.aten.mul.Tensor(squeeze_34, primals_107);  primals_107 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
        mul_279 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_232);  sub_67 = unsqueeze_232 = None
        sub_69 = torch.ops.aten.sub.Tensor(where_14, mul_279);  where_14 = mul_279 = None
        sub_70 = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_229);  sub_69 = unsqueeze_229 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_235);  sub_70 = unsqueeze_235 = None
        mul_281 = torch.ops.aten.mul.Tensor(sum_31, squeeze_34);  sum_31 = squeeze_34 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_280, convolution_19, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_280 = convolution_19 = primals_22 = None
        getitem_66 = convolution_backward_20[0]
        getitem_67 = convolution_backward_20[1];  convolution_backward_20 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_66, relu_10, primals_21, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_66 = primals_21 = None
        getitem_69 = convolution_backward_21[0]
        getitem_70 = convolution_backward_21[1];  convolution_backward_21 = None
        add_128 = torch.ops.aten.add.Tensor(slice_11, getitem_69);  slice_11 = getitem_69 = None
        le_12 = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_15 = torch.ops.aten.where.self(le_12, scalar_tensor, add_128);  le_12 = add_128 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_71 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_238);  convolution_18 = unsqueeze_238 = None
        mul_282 = torch.ops.aten.mul.Tensor(where_15, sub_71)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
        mul_283 = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
        mul_284 = torch.ops.aten.mul.Tensor(sum_33, 0.00015943877551020407)
        mul_285 = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_286 = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
        mul_287 = torch.ops.aten.mul.Tensor(squeeze_31, primals_102);  primals_102 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
        mul_288 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_244);  sub_71 = unsqueeze_244 = None
        sub_73 = torch.ops.aten.sub.Tensor(where_15, mul_288);  where_15 = mul_288 = None
        sub_74 = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_241);  sub_73 = unsqueeze_241 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_247);  sub_74 = unsqueeze_247 = None
        mul_290 = torch.ops.aten.mul.Tensor(sum_33, squeeze_31);  sum_33 = squeeze_31 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_289, convolution_17, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = convolution_17 = primals_20 = None
        getitem_72 = convolution_backward_22[0]
        getitem_73 = convolution_backward_22[1];  convolution_backward_22 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_72, relu_9, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_72 = primals_19 = None
        getitem_75 = convolution_backward_23[0]
        getitem_76 = convolution_backward_23[1];  convolution_backward_23 = None
        add_129 = torch.ops.aten.add.Tensor(slice_10, getitem_75);  slice_10 = getitem_75 = None
        le_13 = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_16 = torch.ops.aten.where.self(le_13, scalar_tensor, add_129);  le_13 = add_129 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_75 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_250);  convolution_16 = unsqueeze_250 = None
        mul_291 = torch.ops.aten.mul.Tensor(where_16, sub_75)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_291, [0, 2, 3]);  mul_291 = None
        mul_292 = torch.ops.aten.mul.Tensor(sum_34, 0.00015943877551020407)
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_293 = torch.ops.aten.mul.Tensor(sum_35, 0.00015943877551020407)
        mul_294 = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_295 = torch.ops.aten.mul.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_296 = torch.ops.aten.mul.Tensor(squeeze_28, primals_97);  primals_97 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(mul_296, 0);  mul_296 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_297 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_256);  sub_75 = unsqueeze_256 = None
        sub_77 = torch.ops.aten.sub.Tensor(where_16, mul_297);  where_16 = mul_297 = None
        sub_78 = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_253);  sub_77 = unsqueeze_253 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_259);  sub_78 = unsqueeze_259 = None
        mul_299 = torch.ops.aten.mul.Tensor(sum_35, squeeze_28);  sum_35 = squeeze_28 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_298, convolution_15, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_298 = convolution_15 = primals_18 = None
        getitem_78 = convolution_backward_24[0]
        getitem_79 = convolution_backward_24[1];  convolution_backward_24 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(getitem_78, relu_8, primals_17, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_78 = primals_17 = None
        getitem_81 = convolution_backward_25[0]
        getitem_82 = convolution_backward_25[1];  convolution_backward_25 = None
        le_14 = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_17 = torch.ops.aten.where.self(le_14, scalar_tensor, getitem_81);  le_14 = getitem_81 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_79 = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_262);  convolution_14 = unsqueeze_262 = None
        mul_300 = torch.ops.aten.mul.Tensor(where_17, sub_79)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
        mul_301 = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_302 = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
        mul_303 = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_304 = torch.ops.aten.mul.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_305 = torch.ops.aten.mul.Tensor(squeeze_25, primals_92);  primals_92 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_268);  sub_79 = unsqueeze_268 = None
        sub_81 = torch.ops.aten.sub.Tensor(where_17, mul_306);  where_17 = mul_306 = None
        sub_82 = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_265);  sub_81 = unsqueeze_265 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_271);  sub_82 = unsqueeze_271 = None
        mul_308 = torch.ops.aten.mul.Tensor(sum_37, squeeze_25);  sum_37 = squeeze_25 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_307, getitem, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_307 = getitem = primals_16 = None
        getitem_84 = convolution_backward_26[0]
        getitem_85 = convolution_backward_26[1];  convolution_backward_26 = None
        add_130 = torch.ops.aten.add.Tensor(slice_9, getitem_84);  slice_9 = getitem_84 = None
        max_pool2d_with_indices_backward_2 = torch.ops.aten.max_pool2d_with_indices_backward.default(add_130, mul_56, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_1);  add_130 = mul_56 = getitem_1 = None
        mul_309 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_2, relu_7)
        mul_310 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_2, div);  max_pool2d_with_indices_backward_2 = div = None
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_309, [2, 3], True);  mul_309 = None
        mul_311 = torch.ops.aten.mul.Tensor(sum_38, 0.16666666666666666);  sum_38 = None
        where_18 = torch.ops.aten.where.self(bitwise_and_3, mul_311, scalar_tensor);  bitwise_and_3 = mul_311 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_18, mean_8, primals_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = mean_8 = primals_14 = None
        getitem_87 = convolution_backward_27[0]
        getitem_88 = convolution_backward_27[1];  convolution_backward_27 = None
        expand_4 = torch.ops.aten.expand.default(getitem_87, [8, 256, 56, 56]);  getitem_87 = None
        div_8 = torch.ops.aten.div.Scalar(expand_4, 3136);  expand_4 = None
        add_131 = torch.ops.aten.add.Tensor(mul_310, div_8);  mul_310 = div_8 = None
        le_15 = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_19 = torch.ops.aten.where.self(le_15, scalar_tensor, add_131);  le_15 = add_131 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_83 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_274);  convolution_12 = unsqueeze_274 = None
        mul_312 = torch.ops.aten.mul.Tensor(where_19, sub_83)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_312, [0, 2, 3]);  mul_312 = None
        mul_313 = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(mul_313, 0);  mul_313 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_314 = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
        mul_315 = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_316 = torch.ops.aten.mul.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_317 = torch.ops.aten.mul.Tensor(squeeze_22, primals_87);  primals_87 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_318 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_280);  sub_83 = unsqueeze_280 = None
        sub_85 = torch.ops.aten.sub.Tensor(where_19, mul_318);  where_19 = mul_318 = None
        sub_86 = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_277);  sub_85 = unsqueeze_277 = None
        mul_319 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_283);  sub_86 = unsqueeze_283 = None
        mul_320 = torch.ops.aten.mul.Tensor(sum_41, squeeze_22);  sum_41 = squeeze_22 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_319, cat, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_319 = cat = primals_13 = None
        getitem_90 = convolution_backward_28[0]
        getitem_91 = convolution_backward_28[1];  convolution_backward_28 = None
        slice_13 = torch.ops.aten.slice.Tensor(getitem_90, 1, 0, 64)
        slice_14 = torch.ops.aten.slice.Tensor(getitem_90, 1, 64, 192)
        slice_15 = torch.ops.aten.slice.Tensor(getitem_90, 1, 192, 320)
        slice_16 = torch.ops.aten.slice.Tensor(getitem_90, 1, 320, 448);  getitem_90 = None
        where_20 = torch.ops.aten.where.self(le_16, scalar_tensor, slice_16);  le_16 = slice_16 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_87 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_286);  convolution_11 = unsqueeze_286 = None
        mul_321 = torch.ops.aten.mul.Tensor(where_20, sub_87)
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_321, [0, 2, 3]);  mul_321 = None
        mul_322 = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_323 = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
        mul_324 = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_325 = torch.ops.aten.mul.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_326 = torch.ops.aten.mul.Tensor(squeeze_19, primals_82);  primals_82 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(mul_326, 0);  mul_326 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_327 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_292);  sub_87 = unsqueeze_292 = None
        sub_89 = torch.ops.aten.sub.Tensor(where_20, mul_327);  where_20 = mul_327 = None
        sub_90 = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_289);  sub_89 = unsqueeze_289 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_295);  sub_90 = unsqueeze_295 = None
        mul_329 = torch.ops.aten.mul.Tensor(sum_43, squeeze_19);  sum_43 = squeeze_19 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_328, convolution_10, primals_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = convolution_10 = primals_12 = None
        getitem_93 = convolution_backward_29[0]
        getitem_94 = convolution_backward_29[1];  convolution_backward_29 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(getitem_93, relu_5, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_93 = primals_11 = None
        getitem_96 = convolution_backward_30[0]
        getitem_97 = convolution_backward_30[1];  convolution_backward_30 = None
        add_132 = torch.ops.aten.add.Tensor(slice_15, getitem_96);  slice_15 = getitem_96 = None
        le_17 = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_21 = torch.ops.aten.where.self(le_17, scalar_tensor, add_132);  le_17 = add_132 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_91 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_298);  convolution_9 = unsqueeze_298 = None
        mul_330 = torch.ops.aten.mul.Tensor(where_21, sub_91)
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
        mul_331 = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_332 = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
        mul_333 = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_334 = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_335 = torch.ops.aten.mul.Tensor(squeeze_16, primals_77);  primals_77 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_336 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_304);  sub_91 = unsqueeze_304 = None
        sub_93 = torch.ops.aten.sub.Tensor(where_21, mul_336);  where_21 = mul_336 = None
        sub_94 = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_301);  sub_93 = unsqueeze_301 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_307);  sub_94 = unsqueeze_307 = None
        mul_338 = torch.ops.aten.mul.Tensor(sum_45, squeeze_16);  sum_45 = squeeze_16 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_337, convolution_8, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = convolution_8 = primals_10 = None
        getitem_99 = convolution_backward_31[0]
        getitem_100 = convolution_backward_31[1];  convolution_backward_31 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(getitem_99, relu_4, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_99 = primals_9 = None
        getitem_102 = convolution_backward_32[0]
        getitem_103 = convolution_backward_32[1];  convolution_backward_32 = None
        add_133 = torch.ops.aten.add.Tensor(slice_14, getitem_102);  slice_14 = getitem_102 = None
        le_18 = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_22 = torch.ops.aten.where.self(le_18, scalar_tensor, add_133);  le_18 = add_133 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_95 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_310);  convolution_7 = unsqueeze_310 = None
        mul_339 = torch.ops.aten.mul.Tensor(where_22, sub_95)
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_339, [0, 2, 3]);  mul_339 = None
        mul_340 = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_341 = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        mul_342 = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_343 = torch.ops.aten.mul.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_344 = torch.ops.aten.mul.Tensor(squeeze_13, primals_72);  primals_72 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(mul_344, 0);  mul_344 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_345 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_316);  sub_95 = unsqueeze_316 = None
        sub_97 = torch.ops.aten.sub.Tensor(where_22, mul_345);  where_22 = mul_345 = None
        sub_98 = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_313);  sub_97 = unsqueeze_313 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_319);  sub_98 = unsqueeze_319 = None
        mul_347 = torch.ops.aten.mul.Tensor(sum_47, squeeze_13);  sum_47 = squeeze_13 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_346, convolution_6, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_346 = convolution_6 = primals_8 = None
        getitem_105 = convolution_backward_33[0]
        getitem_106 = convolution_backward_33[1];  convolution_backward_33 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(getitem_105, relu_3, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_105 = primals_7 = None
        getitem_108 = convolution_backward_34[0]
        getitem_109 = convolution_backward_34[1];  convolution_backward_34 = None
        le_19 = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_23 = torch.ops.aten.where.self(le_19, scalar_tensor, getitem_108);  le_19 = getitem_108 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_99 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_322);  convolution_5 = unsqueeze_322 = None
        mul_348 = torch.ops.aten.mul.Tensor(where_23, sub_99)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
        mul_349 = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_350 = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        mul_351 = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_352 = torch.ops.aten.mul.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(mul_352, 0);  mul_352 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_353 = torch.ops.aten.mul.Tensor(squeeze_10, primals_67);  primals_67 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_328);  sub_99 = unsqueeze_328 = None
        sub_101 = torch.ops.aten.sub.Tensor(where_23, mul_354);  where_23 = mul_354 = None
        sub_102 = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_325);  sub_101 = unsqueeze_325 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_331);  sub_102 = unsqueeze_331 = None
        mul_356 = torch.ops.aten.mul.Tensor(sum_49, squeeze_10);  sum_49 = squeeze_10 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_355, relu_2, primals_6, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_355 = primals_6 = None
        getitem_111 = convolution_backward_35[0]
        getitem_112 = convolution_backward_35[1];  convolution_backward_35 = None
        add_134 = torch.ops.aten.add.Tensor(slice_13, getitem_111);  slice_13 = getitem_111 = None
        le_20 = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_24 = torch.ops.aten.where.self(le_20, scalar_tensor, add_134);  le_20 = add_134 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_103 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_334);  convolution_4 = unsqueeze_334 = None
        mul_357 = torch.ops.aten.mul.Tensor(where_24, sub_103)
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
        mul_358 = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_359 = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
        mul_360 = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_361 = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_362 = torch.ops.aten.mul.Tensor(squeeze_7, primals_62);  primals_62 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_363 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_340);  sub_103 = unsqueeze_340 = None
        sub_105 = torch.ops.aten.sub.Tensor(where_24, mul_363);  where_24 = mul_363 = None
        sub_106 = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_337);  sub_105 = unsqueeze_337 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_343);  sub_106 = unsqueeze_343 = None
        mul_365 = torch.ops.aten.mul.Tensor(sum_51, squeeze_7);  sum_51 = squeeze_7 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_364, convolution_3, primals_5, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_364 = convolution_3 = primals_5 = None
        getitem_114 = convolution_backward_36[0]
        getitem_115 = convolution_backward_36[1];  convolution_backward_36 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(getitem_114, relu_1, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  getitem_114 = primals_4 = None
        getitem_117 = convolution_backward_37[0]
        getitem_118 = convolution_backward_37[1];  convolution_backward_37 = None
        le_21 = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_25 = torch.ops.aten.where.self(le_21, scalar_tensor, getitem_117);  le_21 = getitem_117 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_107 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_346);  convolution_2 = unsqueeze_346 = None
        mul_366 = torch.ops.aten.mul.Tensor(where_25, sub_107)
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_366, [0, 2, 3]);  mul_366 = None
        mul_367 = torch.ops.aten.mul.Tensor(sum_52, 9.964923469387754e-06)
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        mul_368 = torch.ops.aten.mul.Tensor(sum_53, 9.964923469387754e-06)
        mul_369 = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_370 = torch.ops.aten.mul.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_371 = torch.ops.aten.mul.Tensor(squeeze_4, primals_57);  primals_57 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_372 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_352);  sub_107 = unsqueeze_352 = None
        sub_109 = torch.ops.aten.sub.Tensor(where_25, mul_372);  where_25 = mul_372 = None
        sub_110 = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_349);  sub_109 = unsqueeze_349 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_355);  sub_110 = unsqueeze_355 = None
        mul_374 = torch.ops.aten.mul.Tensor(sum_53, squeeze_4);  sum_53 = squeeze_4 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_373, convolution_1, primals_3, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_373 = convolution_1 = primals_3 = None
        getitem_120 = convolution_backward_38[0]
        getitem_121 = convolution_backward_38[1];  convolution_backward_38 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(getitem_120, relu, primals_2, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  getitem_120 = primals_2 = None
        getitem_123 = convolution_backward_39[0]
        getitem_124 = convolution_backward_39[1];  convolution_backward_39 = None
        le_22 = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_26 = torch.ops.aten.where.self(le_22, scalar_tensor, getitem_123);  le_22 = scalar_tensor = getitem_123 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_111 = torch.ops.aten.sub.Tensor(convolution, unsqueeze_358);  convolution = unsqueeze_358 = None
        mul_375 = torch.ops.aten.mul.Tensor(where_26, sub_111)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
        mul_376 = torch.ops.aten.mul.Tensor(sum_54, 9.964923469387754e-06)
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
        mul_377 = torch.ops.aten.mul.Tensor(sum_55, 9.964923469387754e-06)
        mul_378 = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_379 = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        mul_380 = torch.ops.aten.mul.Tensor(squeeze_1, primals_52);  primals_52 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        mul_381 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_364);  sub_111 = unsqueeze_364 = None
        sub_113 = torch.ops.aten.sub.Tensor(where_26, mul_381);  where_26 = mul_381 = None
        sub_114 = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_361);  sub_113 = unsqueeze_361 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_367);  sub_114 = unsqueeze_367 = None
        mul_383 = torch.ops.aten.mul.Tensor(sum_55, squeeze_1);  sum_55 = squeeze_1 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_382, primals_48, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_382 = primals_48 = primals_1 = None
        getitem_127 = convolution_backward_40[1];  convolution_backward_40 = None
        return [getitem_127, getitem_124, getitem_121, getitem_118, getitem_115, getitem_112, getitem_109, getitem_106, getitem_103, getitem_100, getitem_97, getitem_94, getitem_91, getitem_88, sum_39, getitem_85, getitem_82, getitem_79, getitem_76, getitem_73, getitem_70, getitem_67, getitem_64, getitem_61, sum_27, getitem_58, getitem_55, getitem_52, getitem_49, getitem_46, getitem_43, getitem_40, getitem_37, getitem_34, sum_15, getitem_31, getitem_28, getitem_25, getitem_22, getitem_19, getitem_16, getitem_13, getitem_10, getitem_7, sum_3, permute_4, view_1, None, None, None, None, mul_383, sum_54, None, None, None, mul_374, sum_52, None, None, None, mul_365, sum_50, None, None, None, mul_356, sum_48, None, None, None, mul_347, sum_46, None, None, None, mul_338, sum_44, None, None, None, mul_329, sum_42, None, None, None, mul_320, sum_40, None, None, None, mul_308, sum_36, None, None, None, mul_299, sum_34, None, None, None, mul_290, sum_32, None, None, None, mul_281, sum_30, None, None, None, mul_272, sum_28, None, None, None, mul_260, sum_24, None, None, None, mul_251, sum_22, None, None, None, mul_242, sum_20, None, None, None, mul_233, sum_18, None, None, None, mul_224, sum_16, None, None, None, mul_212, sum_12, None, None, None, mul_203, sum_10, None, None, None, mul_194, sum_8, None, None, None, mul_185, sum_6, None, None, None, mul_176, sum_4]
        
args = [((64, 3, 3, 3), (27, 9, 3, 1), torch.float32, 'cuda'), ((64, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((64, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((64, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((64, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((128, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((128, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((128, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((128, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((128, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((128, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((128, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((256, 448, 1, 1), (448, 1, 1, 1), torch.float32, 'cuda'), ((256, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((160, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((160, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((160, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((160, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((160, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((512, 736, 1, 1), (736, 1, 1, 1), torch.float32, 'cuda'), ((512, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((192, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((192, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((192, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((192, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((192, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((192, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((192, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((768, 1088, 1, 1), (1088, 1, 1, 1), torch.float32, 'cuda'), ((768, 768, 1, 1), (768, 1, 1, 1), torch.float32, 'cuda'), ((224, 768, 1, 1), (768, 1, 1, 1), torch.float32, 'cuda'), ((224, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((224, 224, 1, 1), (224, 1, 1, 1), torch.float32, 'cuda'), ((224, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((224, 224, 1, 1), (224, 1, 1, 1), torch.float32, 'cuda'), ((224, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((224, 224, 1, 1), (224, 1, 1, 1), torch.float32, 'cuda'), ((1024, 1440, 1, 1), (1440, 1, 1, 1), torch.float32, 'cuda'), ((1024, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((8, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((8, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((8, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((8, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((8, 64, 56, 56), (200704, 3136, 56, 1), torch.float32, 'cuda'), ((8, 64, 56, 56), (200704, 3136, 56, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 56, 56), (200704, 3136, 56, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 448, 56, 56), (1404928, 3136, 56, 1), torch.float32, 'cuda'), ((8, 256, 56, 56), (802816, 3136, 56, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 56, 56), (802816, 3136, 56, 1), torch.float32, 'cuda'), ((8, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((8, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((8, 256, 56, 56), (802816, 3136, 56, 1), torch.float32, 'cuda'), ((8, 256, 28, 28), (200704, 784, 28, 1), torch.float32, 'cuda'), ((8, 256, 28, 28), (200704, 784, 28, 1), torch.int64, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 736, 28, 28), (577024, 784, 28, 1), torch.float32, 'cuda'), ((8, 512, 28, 28), (401408, 784, 28, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 28, 28), (401408, 784, 28, 1), torch.float32, 'cuda'), ((8, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((8, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((8, 512, 28, 28), (401408, 784, 28, 1), torch.float32, 'cuda'), ((8, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((8, 512, 14, 14), (100352, 196, 14, 1), torch.int64, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 1088, 14, 14), (213248, 196, 14, 1), torch.float32, 'cuda'), ((8, 768, 14, 14), (150528, 196, 14, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((8, 768, 14, 14), (150528, 196, 14, 1), torch.float32, 'cuda'), ((8, 768, 1, 1), (768, 1, 1, 1), torch.float32, 'cuda'), ((8, 768, 1, 1), (768, 1, 1, 1), torch.float32, 'cuda'), ((8, 768, 14, 14), (150528, 196, 14, 1), torch.float32, 'cuda'), ((8, 768, 7, 7), (37632, 49, 7, 1), torch.float32, 'cuda'), ((8, 768, 7, 7), (37632, 49, 7, 1), torch.int64, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((8, 1440, 7, 7), (70560, 49, 7, 1), torch.float32, 'cuda'), ((8, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((8, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((8, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((8, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((8, 1024), (1024, 1), torch.float32, 'cuda'), ((1000, 1024), (1024, 1), torch.float32, 'cuda'), ((8, 1024, 1, 1), (1024, 1, 1, 1), torch.bool, 'cuda'), ((1, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((8, 224, 7, 7), (10976, 49, 7, 1), torch.bool, 'cuda'), ((1, 224, 1, 1), (224, 1, 1, 1), torch.float32, 'cuda'), ((1, 224, 1, 1), (224, 1, 1, 1), torch.float32, 'cuda'), ((1, 224, 1, 1), (224, 1, 1, 1), torch.float32, 'cuda'), ((1, 224, 1, 1), (224, 1, 1, 1), torch.float32, 'cuda'), ((8, 768, 1, 1), (768, 1, 1, 1), torch.bool, 'cuda'), ((1, 768, 1, 1), (768, 1, 1, 1), torch.float32, 'cuda'), ((8, 192, 14, 14), (37632, 196, 14, 1), torch.bool, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((8, 512, 1, 1), (512, 1, 1, 1), torch.bool, 'cuda'), ((1, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((8, 160, 28, 28), (125440, 784, 28, 1), torch.bool, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((8, 256, 1, 1), (256, 1, 1, 1), torch.bool, 'cuda'), ((1, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((8, 128, 56, 56), (401408, 3136, 56, 1), torch.bool, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((224,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((8, 1000), (1000, 1), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
