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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x88X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\r\x00\x00\x00torch.testingq\x18X\x13\x00\x00\x00torch.distributionsq\x19X\x0c\x00\x00\x00torch._primsq\x1aX\x0b\x00\x00\x00torch._refsq\x1bX\r\x00\x00\x00torch._decompq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x18\x00\x00\x00error_on_nested_fx_traceq&\x88X\t\x00\x00\x00allow_rnnq\'\x89X\x08\x00\x00\x00base_dirq(X\x1c\x00\x00\x00/scratch/ngimel/work/pytorchq)X\x0e\x00\x00\x00debug_dir_rootq*X0\x00\x00\x00/scratch/ngimel/work/pytorch/torch_compile_debugq+X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq,\x89X\x13\x00\x00\x00_save_config_ignoreq-h\r]q.(X!\x00\x00\x00skipfiles_inline_module_allowlistq/X\x0b\x00\x00\x00repro_afterq0X\x0b\x00\x00\x00repro_levelq1X\x12\x00\x00\x00constant_functionsq2e\x85q3Rq4u.')
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

    
    
    def forward(self, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_45, primals_49, primals_54, primals_59, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem, getitem_1, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, _unsafe_view_3, _unsafe_view_4, div, squeeze_67, mul_186, convolution_28, squeeze_70, mul_194, convolution_29, squeeze_73, mul_202, _unsafe_view_10, _unsafe_view_11, div_1, _unsafe_view_13, avg_pool2d, squeeze_76, mul_211, convolution_31, squeeze_79, convolution_32, squeeze_82, mul_226, convolution_33, squeeze_85, mul_234, _unsafe_view_17, _unsafe_view_18, div_2, squeeze_88, mul_243, convolution_35, squeeze_91, view_61, permute_25, mul_253, unsqueeze_126, mul_265, sub_40, permute_30, permute_31, permute_35, permute_41, permute_43, permute_44, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, unsqueeze_186, permute_48, permute_49, permute_53, permute_59, permute_61, permute_62, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, sub_76, permute_66, permute_67, permute_71, permute_77, permute_79, permute_80, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94):
        clone_66 = torch.ops.aten.clone.default(add_24)
        sigmoid_4 = torch.ops.aten.sigmoid.default(add_24)
        mul_39 = torch.ops.aten.mul.Tensor(add_24, sigmoid_4);  add_24 = sigmoid_4 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(convolution_5);  convolution_5 = None
        view_1 = torch.ops.aten.view.default(sigmoid_5, [8, -1, 1, 1])
        expand = torch.ops.aten.expand.default(view_1, [8, 64, 64, 64]);  view_1 = None
        clone_69 = torch.ops.aten.clone.default(add_45)
        sigmoid_8 = torch.ops.aten.sigmoid.default(add_45)
        mul_71 = torch.ops.aten.mul.Tensor(add_45, sigmoid_8);  add_45 = sigmoid_8 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
        view_3 = torch.ops.aten.view.default(sigmoid_9, [8, -1, 1, 1])
        expand_1 = torch.ops.aten.expand.default(view_3, [8, 64, 64, 64]);  view_3 = None
        clone_72 = torch.ops.aten.clone.default(add_61)
        sigmoid_12 = torch.ops.aten.sigmoid.default(add_61)
        mul_96 = torch.ops.aten.mul.Tensor(add_61, sigmoid_12);  add_61 = sigmoid_12 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convolution_14);  convolution_14 = None
        view_5 = torch.ops.aten.view.default(sigmoid_13, [8, -1, 1, 1])
        expand_2 = torch.ops.aten.expand.default(view_5, [8, 128, 32, 32]);  view_5 = None
        clone_75 = torch.ops.aten.clone.default(add_82)
        sigmoid_16 = torch.ops.aten.sigmoid.default(add_82)
        mul_128 = torch.ops.aten.mul.Tensor(add_82, sigmoid_16);  add_82 = sigmoid_16 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(convolution_19);  convolution_19 = None
        view_7 = torch.ops.aten.view.default(sigmoid_17, [8, -1, 1, 1])
        expand_3 = torch.ops.aten.expand.default(view_7, [8, 128, 32, 32]);  view_7 = None
        clone_78 = torch.ops.aten.clone.default(add_98)
        sigmoid_20 = torch.ops.aten.sigmoid.default(add_98)
        mul_153 = torch.ops.aten.mul.Tensor(add_98, sigmoid_20);  add_98 = sigmoid_20 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
        view_9 = torch.ops.aten.view.default(sigmoid_21, [8, -1, 1, 1])
        expand_4 = torch.ops.aten.expand.default(view_9, [8, 256, 16, 16]);  view_9 = None
        mm_6 = torch.ops.aten.mm.default(tangents_63, permute_25);  permute_25 = None
        permute_26 = torch.ops.aten.permute.default(tangents_63, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_26, view_61);  permute_26 = view_61 = None
        permute_27 = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(tangents_63, [0], True);  tangents_63 = None
        view_62 = torch.ops.aten.view.default(sum_4, [1000]);  sum_4 = None
        permute_28 = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        view_63 = torch.ops.aten.view.default(mm_6, [8, 2048, 1, 1]);  mm_6 = None
        expand_23 = torch.ops.aten.expand.default(view_63, [8, 2048, 8, 8]);  view_63 = None
        div_3 = torch.ops.aten.div.Scalar(expand_23, 64);  expand_23 = None
        mul_254 = torch.ops.aten.mul.Tensor(div_3, mul_253);  div_3 = mul_253 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_254, [0, 2, 3])
        sub_35 = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_126);  convolution_35 = unsqueeze_126 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, sub_35)
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3]);  mul_255 = None
        mul_256 = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
        mul_257 = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
        mul_258 = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_259 = torch.ops.aten.mul.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(mul_259, 0);  mul_259 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
        mul_260 = torch.ops.aten.mul.Tensor(squeeze_91, primals_199);  primals_199 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
        mul_261 = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_132);  sub_35 = unsqueeze_132 = None
        sub_37 = torch.ops.aten.sub.Tensor(mul_254, mul_261);  mul_261 = None
        sub_38 = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_129);  sub_37 = unsqueeze_129 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_135);  sub_38 = unsqueeze_135 = None
        mul_263 = torch.ops.aten.mul.Tensor(sum_6, squeeze_91);  sum_6 = squeeze_91 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_262, mul_243, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_262 = mul_243 = primals_42 = None
        getitem_11 = convolution_backward[0]
        getitem_12 = convolution_backward[1];  convolution_backward = None
        mul_266 = torch.ops.aten.mul.Tensor(getitem_11, mul_265);  getitem_11 = mul_265 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3])
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, sub_40)
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_267, [0, 2, 3]);  mul_267 = None
        mul_268 = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        mul_269 = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
        mul_270 = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_271 = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        mul_272 = torch.ops.aten.mul.Tensor(squeeze_88, primals_194);  primals_194 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        mul_273 = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_144);  sub_40 = unsqueeze_144 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_266, mul_273);  mul_266 = mul_273 = None
        sub_43 = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_141);  sub_42 = unsqueeze_141 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_147);  sub_43 = unsqueeze_147 = None
        mul_275 = torch.ops.aten.mul.Tensor(sum_8, squeeze_88);  sum_8 = squeeze_88 = None
        view_64 = torch.ops.aten.view.default(mul_274, [32, 128, 64]);  mul_274 = None
        permute_29 = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        view_65 = torch.ops.aten.view.default(permute_29, [32, 64, 128]);  permute_29 = None
        bmm_6 = torch.ops.aten.bmm.default(permute_30, view_65);  permute_30 = None
        bmm_7 = torch.ops.aten.bmm.default(view_65, permute_31);  view_65 = permute_31 = None
        view_66 = torch.ops.aten.view.default(bmm_6, [32, 64, 128]);  bmm_6 = None
        view_67 = torch.ops.aten.view.default(bmm_7, [32, 64, 64]);  bmm_7 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_67, div_2);  view_67 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        mul_277 = torch.ops.aten.mul.Tensor(div_2, sum_9);  div_2 = sum_9 = None
        sub_44 = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        view_68 = torch.ops.aten.view.default(sub_44, [32, 8, 8, 8, 8])
        permute_32 = torch.ops.aten.permute.default(view_68, [0, 2, 4, 1, 3])
        sum_10 = torch.ops.aten.sum.dim_IntList(permute_32, [2], True);  permute_32 = None
        view_69 = torch.ops.aten.view.default(sum_10, [256, 8, 8]);  sum_10 = None
        full = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter = torch.ops.aten.slice_scatter.default(full, view_69, 2, 7, 9223372036854775807);  view_69 = None
        full_1 = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(full_1, slice_scatter, 1, 0, 8);  slice_scatter = None
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_1, 0, 0, 9223372036854775807);  slice_scatter_1 = None
        view_70 = torch.ops.aten.view.default(slice_scatter_2, [256, 135]);  slice_scatter_2 = None
        constant_pad_nd_12 = torch.ops.aten.constant_pad_nd.default(view_70, [0, -7]);  view_70 = None
        view_71 = torch.ops.aten.view.default(constant_pad_nd_12, [256, 8, 16]);  constant_pad_nd_12 = None
        constant_pad_nd_13 = torch.ops.aten.constant_pad_nd.default(view_71, [0, -1]);  view_71 = None
        view_72 = torch.ops.aten.view.default(constant_pad_nd_13, [32, 8, 8, 15]);  constant_pad_nd_13 = None
        view_73 = torch.ops.aten.view.default(view_72, [2048, 15]);  view_72 = None
        permute_33 = torch.ops.aten.permute.default(view_73, [1, 0])
        mm_8 = torch.ops.aten.mm.default(permute_33, _unsafe_view_18);  permute_33 = _unsafe_view_18 = None
        permute_34 = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
        mm_9 = torch.ops.aten.mm.default(view_73, permute_35);  view_73 = permute_35 = None
        view_74 = torch.ops.aten.view.default(mm_9, [32, 8, 8, 16]);  mm_9 = None
        permute_36 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_37 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_38 = torch.ops.aten.permute.default(view_68, [0, 1, 3, 2, 4]);  view_68 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(permute_38, [2], True);  permute_38 = None
        view_75 = torch.ops.aten.view.default(sum_11, [256, 8, 8]);  sum_11 = None
        slice_scatter_3 = torch.ops.aten.slice_scatter.default(full, view_75, 2, 7, 9223372036854775807);  full = view_75 = None
        slice_scatter_4 = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_3, 1, 0, 8);  slice_scatter_3 = None
        slice_scatter_5 = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_4, 0, 0, 9223372036854775807);  full_1 = slice_scatter_4 = None
        view_76 = torch.ops.aten.view.default(slice_scatter_5, [256, 135]);  slice_scatter_5 = None
        constant_pad_nd_14 = torch.ops.aten.constant_pad_nd.default(view_76, [0, -7]);  view_76 = None
        view_77 = torch.ops.aten.view.default(constant_pad_nd_14, [256, 8, 16]);  constant_pad_nd_14 = None
        constant_pad_nd_15 = torch.ops.aten.constant_pad_nd.default(view_77, [0, -1]);  view_77 = None
        view_78 = torch.ops.aten.view.default(constant_pad_nd_15, [32, 8, 8, 15]);  constant_pad_nd_15 = None
        view_79 = torch.ops.aten.view.default(view_78, [2048, 15]);  view_78 = None
        permute_39 = torch.ops.aten.permute.default(view_79, [1, 0])
        mm_10 = torch.ops.aten.mm.default(permute_39, _unsafe_view_17);  permute_39 = _unsafe_view_17 = None
        permute_40 = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
        mm_11 = torch.ops.aten.mm.default(view_79, permute_41);  view_79 = permute_41 = None
        view_80 = torch.ops.aten.view.default(mm_11, [32, 8, 8, 16]);  mm_11 = None
        add_171 = torch.ops.aten.add.Tensor(permute_37, view_80);  permute_37 = view_80 = None
        permute_42 = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        clone_110 = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format);  add_171 = None
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(clone_110, [32, 64, 16]);  clone_110 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_44, 0.25);  sub_44 = None
        view_81 = torch.ops.aten.view.default(mul_278, [32, 64, 64]);  mul_278 = None
        bmm_8 = torch.ops.aten.bmm.default(permute_43, view_81);  permute_43 = None
        bmm_9 = torch.ops.aten.bmm.default(view_81, permute_44);  view_81 = permute_44 = None
        view_82 = torch.ops.aten.view.default(bmm_8, [32, 16, 64]);  bmm_8 = None
        view_83 = torch.ops.aten.view.default(bmm_9, [32, 64, 16]);  bmm_9 = None
        add_172 = torch.ops.aten.add.Tensor(_unsafe_view_21, view_83);  _unsafe_view_21 = view_83 = None
        permute_45 = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        clone_111 = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(clone_111, [8, 512, 8, 8]);  clone_111 = None
        view_84 = torch.ops.aten.view.default(view_82, [8, 64, 8, 8]);  view_82 = None
        permute_46 = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
        clone_112 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        _unsafe_view_23 = torch.ops.aten._unsafe_view.default(clone_112, [8, 64, 8, 8]);  clone_112 = None
        cat = torch.ops.aten.cat.default([_unsafe_view_23, view_84, _unsafe_view_22], 1);  _unsafe_view_23 = view_84 = _unsafe_view_22 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(cat, mul_234, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat = mul_234 = primals_41 = None
        getitem_14 = convolution_backward_1[0]
        getitem_15 = convolution_backward_1[1];  convolution_backward_1 = None
        mul_281 = torch.ops.aten.mul.Tensor(getitem_14, mul_280);  getitem_14 = mul_280 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3])
        sub_46 = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_150);  convolution_33 = unsqueeze_150 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_281, sub_46)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
        mul_283 = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        mul_284 = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
        mul_285 = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_286 = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        mul_287 = torch.ops.aten.mul.Tensor(squeeze_85, primals_189);  primals_189 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        mul_288 = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_156);  sub_46 = unsqueeze_156 = None
        sub_48 = torch.ops.aten.sub.Tensor(mul_281, mul_288);  mul_281 = mul_288 = None
        sub_49 = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_153);  sub_48 = unsqueeze_153 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_159);  sub_49 = unsqueeze_159 = None
        mul_290 = torch.ops.aten.mul.Tensor(sum_13, squeeze_85);  sum_13 = squeeze_85 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_289, mul_226, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = mul_226 = primals_40 = None
        getitem_17 = convolution_backward_2[0]
        getitem_18 = convolution_backward_2[1];  convolution_backward_2 = None
        add_174 = torch.ops.aten.add.Tensor(mul_254, getitem_17);  mul_254 = getitem_17 = None
        mul_293 = torch.ops.aten.mul.Tensor(add_174, mul_292);  add_174 = mul_292 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3])
        sub_51 = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_162);  convolution_32 = unsqueeze_162 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_293, sub_51)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_294, [0, 2, 3]);  mul_294 = None
        mul_295 = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_296 = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
        mul_297 = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_298 = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        mul_299 = torch.ops.aten.mul.Tensor(squeeze_82, primals_184);  primals_184 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_300 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_168);  sub_51 = unsqueeze_168 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_293, mul_300);  mul_300 = None
        sub_54 = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_165);  sub_53 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_171);  sub_54 = unsqueeze_171 = None
        mul_302 = torch.ops.aten.mul.Tensor(sum_15, squeeze_82);  sum_15 = squeeze_82 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_301, mul_194, primals_39, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_301 = primals_39 = None
        getitem_20 = convolution_backward_3[0]
        getitem_21 = convolution_backward_3[1];  convolution_backward_3 = None
        sub_55 = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_174);  convolution_31 = unsqueeze_174 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_293, sub_55)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
        mul_305 = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
        mul_306 = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_307 = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        mul_308 = torch.ops.aten.mul.Tensor(squeeze_79, primals_179);  primals_179 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_180);  sub_55 = unsqueeze_180 = None
        sub_57 = torch.ops.aten.sub.Tensor(mul_293, mul_309);  mul_293 = mul_309 = None
        sub_58 = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_165);  sub_57 = unsqueeze_165 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_183);  sub_58 = unsqueeze_183 = None
        mul_311 = torch.ops.aten.mul.Tensor(sum_17, squeeze_79);  sum_17 = squeeze_79 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_310, mul_211, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_310 = mul_211 = primals_38 = None
        getitem_23 = convolution_backward_4[0]
        getitem_24 = convolution_backward_4[1];  convolution_backward_4 = None
        mul_314 = torch.ops.aten.mul.Tensor(getitem_23, mul_313);  getitem_23 = mul_313 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
        sub_60 = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_186);  avg_pool2d = unsqueeze_186 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, sub_60)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
        mul_316 = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
        mul_317 = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
        mul_318 = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_319 = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
        mul_320 = torch.ops.aten.mul.Tensor(squeeze_76, primals_174);  primals_174 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
        mul_321 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_192);  sub_60 = unsqueeze_192 = None
        sub_62 = torch.ops.aten.sub.Tensor(mul_314, mul_321);  mul_314 = mul_321 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_189);  sub_62 = unsqueeze_189 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_195);  sub_63 = unsqueeze_195 = None
        mul_323 = torch.ops.aten.mul.Tensor(sum_19, squeeze_76);  sum_19 = squeeze_76 = None
        avg_pool2d_backward = torch.ops.aten.avg_pool2d_backward.default(mul_322, _unsafe_view_13, [2, 2], [2, 2], [0, 0], False, True, None);  mul_322 = _unsafe_view_13 = None
        view_85 = torch.ops.aten.view.default(avg_pool2d_backward, [32, 128, 256]);  avg_pool2d_backward = None
        permute_47 = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        view_86 = torch.ops.aten.view.default(permute_47, [32, 256, 128]);  permute_47 = None
        bmm_10 = torch.ops.aten.bmm.default(permute_48, view_86);  permute_48 = None
        bmm_11 = torch.ops.aten.bmm.default(view_86, permute_49);  view_86 = permute_49 = None
        view_87 = torch.ops.aten.view.default(bmm_10, [32, 256, 128]);  bmm_10 = None
        view_88 = torch.ops.aten.view.default(bmm_11, [32, 256, 256]);  bmm_11 = None
        mul_324 = torch.ops.aten.mul.Tensor(view_88, div_1);  view_88 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_324, [-1], True)
        mul_325 = torch.ops.aten.mul.Tensor(div_1, sum_20);  div_1 = sum_20 = None
        sub_64 = torch.ops.aten.sub.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
        view_89 = torch.ops.aten.view.default(sub_64, [32, 16, 16, 16, 16])
        permute_50 = torch.ops.aten.permute.default(view_89, [0, 2, 4, 1, 3])
        sum_21 = torch.ops.aten.sum.dim_IntList(permute_50, [2], True);  permute_50 = None
        view_90 = torch.ops.aten.view.default(sum_21, [512, 16, 16]);  sum_21 = None
        full_6 = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_6 = torch.ops.aten.slice_scatter.default(full_6, view_90, 2, 15, 9223372036854775807);  view_90 = None
        full_7 = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_7 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_6, 1, 0, 16);  slice_scatter_6 = None
        slice_scatter_8 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_7 = None
        view_91 = torch.ops.aten.view.default(slice_scatter_8, [512, 527]);  slice_scatter_8 = None
        constant_pad_nd_16 = torch.ops.aten.constant_pad_nd.default(view_91, [0, -15]);  view_91 = None
        view_92 = torch.ops.aten.view.default(constant_pad_nd_16, [512, 16, 32]);  constant_pad_nd_16 = None
        constant_pad_nd_17 = torch.ops.aten.constant_pad_nd.default(view_92, [0, -1]);  view_92 = None
        view_93 = torch.ops.aten.view.default(constant_pad_nd_17, [32, 16, 16, 31]);  constant_pad_nd_17 = None
        view_94 = torch.ops.aten.view.default(view_93, [8192, 31]);  view_93 = None
        permute_51 = torch.ops.aten.permute.default(view_94, [1, 0])
        mm_12 = torch.ops.aten.mm.default(permute_51, _unsafe_view_11);  permute_51 = _unsafe_view_11 = None
        permute_52 = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
        mm_13 = torch.ops.aten.mm.default(view_94, permute_53);  view_94 = permute_53 = None
        view_95 = torch.ops.aten.view.default(mm_13, [32, 16, 16, 16]);  mm_13 = None
        permute_54 = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_55 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        permute_56 = torch.ops.aten.permute.default(view_89, [0, 1, 3, 2, 4]);  view_89 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(permute_56, [2], True);  permute_56 = None
        view_96 = torch.ops.aten.view.default(sum_22, [512, 16, 16]);  sum_22 = None
        slice_scatter_9 = torch.ops.aten.slice_scatter.default(full_6, view_96, 2, 15, 9223372036854775807);  view_96 = None
        slice_scatter_10 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_9, 1, 0, 16);  slice_scatter_9 = None
        slice_scatter_11 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_10, 0, 0, 9223372036854775807);  slice_scatter_10 = None
        view_97 = torch.ops.aten.view.default(slice_scatter_11, [512, 527]);  slice_scatter_11 = None
        constant_pad_nd_18 = torch.ops.aten.constant_pad_nd.default(view_97, [0, -15]);  view_97 = None
        view_98 = torch.ops.aten.view.default(constant_pad_nd_18, [512, 16, 32]);  constant_pad_nd_18 = None
        constant_pad_nd_19 = torch.ops.aten.constant_pad_nd.default(view_98, [0, -1]);  view_98 = None
        view_99 = torch.ops.aten.view.default(constant_pad_nd_19, [32, 16, 16, 31]);  constant_pad_nd_19 = None
        view_100 = torch.ops.aten.view.default(view_99, [8192, 31]);  view_99 = None
        permute_57 = torch.ops.aten.permute.default(view_100, [1, 0])
        mm_14 = torch.ops.aten.mm.default(permute_57, _unsafe_view_10);  permute_57 = _unsafe_view_10 = None
        permute_58 = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
        mm_15 = torch.ops.aten.mm.default(view_100, permute_59);  view_100 = permute_59 = None
        view_101 = torch.ops.aten.view.default(mm_15, [32, 16, 16, 16]);  mm_15 = None
        add_177 = torch.ops.aten.add.Tensor(permute_55, view_101);  permute_55 = view_101 = None
        permute_60 = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        clone_113 = torch.ops.aten.clone.default(add_177, memory_format = torch.contiguous_format);  add_177 = None
        _unsafe_view_24 = torch.ops.aten._unsafe_view.default(clone_113, [32, 256, 16]);  clone_113 = None
        mul_326 = torch.ops.aten.mul.Tensor(sub_64, 0.25);  sub_64 = None
        view_102 = torch.ops.aten.view.default(mul_326, [32, 256, 256]);  mul_326 = None
        bmm_12 = torch.ops.aten.bmm.default(permute_61, view_102);  permute_61 = None
        bmm_13 = torch.ops.aten.bmm.default(view_102, permute_62);  view_102 = permute_62 = None
        view_103 = torch.ops.aten.view.default(bmm_12, [32, 16, 256]);  bmm_12 = None
        view_104 = torch.ops.aten.view.default(bmm_13, [32, 256, 16]);  bmm_13 = None
        add_178 = torch.ops.aten.add.Tensor(_unsafe_view_24, view_104);  _unsafe_view_24 = view_104 = None
        permute_63 = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        clone_114 = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        _unsafe_view_25 = torch.ops.aten._unsafe_view.default(clone_114, [8, 512, 16, 16]);  clone_114 = None
        view_105 = torch.ops.aten.view.default(view_103, [8, 64, 16, 16]);  view_103 = None
        permute_64 = torch.ops.aten.permute.default(add_178, [0, 2, 1]);  add_178 = None
        clone_115 = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        _unsafe_view_26 = torch.ops.aten._unsafe_view.default(clone_115, [8, 64, 16, 16]);  clone_115 = None
        cat_1 = torch.ops.aten.cat.default([_unsafe_view_26, view_105, _unsafe_view_25], 1);  _unsafe_view_26 = view_105 = _unsafe_view_25 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(cat_1, mul_202, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_1 = mul_202 = primals_37 = None
        getitem_26 = convolution_backward_5[0]
        getitem_27 = convolution_backward_5[1];  convolution_backward_5 = None
        mul_329 = torch.ops.aten.mul.Tensor(getitem_26, mul_328);  getitem_26 = mul_328 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3])
        sub_66 = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_198);  convolution_29 = unsqueeze_198 = None
        mul_330 = torch.ops.aten.mul.Tensor(mul_329, sub_66)
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
        mul_331 = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_332 = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
        mul_333 = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_334 = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_335 = torch.ops.aten.mul.Tensor(squeeze_73, primals_169);  primals_169 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_336 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_204);  sub_66 = unsqueeze_204 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_329, mul_336);  mul_329 = mul_336 = None
        sub_69 = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_201);  sub_68 = unsqueeze_201 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_207);  sub_69 = unsqueeze_207 = None
        mul_338 = torch.ops.aten.mul.Tensor(sum_24, squeeze_73);  sum_24 = squeeze_73 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_337, mul_194, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = mul_194 = primals_36 = None
        getitem_29 = convolution_backward_6[0]
        getitem_30 = convolution_backward_6[1];  convolution_backward_6 = None
        add_180 = torch.ops.aten.add.Tensor(getitem_20, getitem_29);  getitem_20 = getitem_29 = None
        mul_341 = torch.ops.aten.mul.Tensor(add_180, mul_340);  add_180 = mul_340 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3])
        sub_71 = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_210);  convolution_28 = unsqueeze_210 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_341, sub_71)
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_342, [0, 2, 3]);  mul_342 = None
        mul_343 = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_344 = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
        mul_345 = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_346 = torch.ops.aten.mul.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_347 = torch.ops.aten.mul.Tensor(squeeze_70, primals_164);  primals_164 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_348 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_216);  sub_71 = unsqueeze_216 = None
        sub_73 = torch.ops.aten.sub.Tensor(mul_341, mul_348);  mul_348 = None
        sub_74 = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_213);  sub_73 = unsqueeze_213 = None
        mul_349 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_219);  sub_74 = unsqueeze_219 = None
        mul_350 = torch.ops.aten.mul.Tensor(sum_26, squeeze_70);  sum_26 = squeeze_70 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_349, mul_186, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_349 = mul_186 = primals_35 = None
        getitem_32 = convolution_backward_7[0]
        getitem_33 = convolution_backward_7[1];  convolution_backward_7 = None
        mul_353 = torch.ops.aten.mul.Tensor(getitem_32, mul_352);  getitem_32 = mul_352 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_353, [0, 2, 3])
        mul_354 = torch.ops.aten.mul.Tensor(mul_353, sub_76)
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_354, [0, 2, 3]);  mul_354 = None
        mul_355 = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_356 = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
        mul_357 = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_358 = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_359 = torch.ops.aten.mul.Tensor(squeeze_67, primals_159);  primals_159 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_360 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_228);  sub_76 = unsqueeze_228 = None
        sub_78 = torch.ops.aten.sub.Tensor(mul_353, mul_360);  mul_353 = mul_360 = None
        sub_79 = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_225);  sub_78 = unsqueeze_225 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_231);  sub_79 = unsqueeze_231 = None
        mul_362 = torch.ops.aten.mul.Tensor(sum_28, squeeze_67);  sum_28 = squeeze_67 = None
        view_106 = torch.ops.aten.view.default(mul_361, [32, 64, 256]);  mul_361 = None
        permute_65 = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        view_107 = torch.ops.aten.view.default(permute_65, [32, 256, 64]);  permute_65 = None
        bmm_14 = torch.ops.aten.bmm.default(permute_66, view_107);  permute_66 = None
        bmm_15 = torch.ops.aten.bmm.default(view_107, permute_67);  view_107 = permute_67 = None
        view_108 = torch.ops.aten.view.default(bmm_14, [32, 256, 64]);  bmm_14 = None
        view_109 = torch.ops.aten.view.default(bmm_15, [32, 256, 256]);  bmm_15 = None
        mul_363 = torch.ops.aten.mul.Tensor(view_109, div);  view_109 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
        mul_364 = torch.ops.aten.mul.Tensor(div, sum_29);  div = sum_29 = None
        sub_80 = torch.ops.aten.sub.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
        view_110 = torch.ops.aten.view.default(sub_80, [32, 16, 16, 16, 16])
        permute_68 = torch.ops.aten.permute.default(view_110, [0, 2, 4, 1, 3])
        sum_30 = torch.ops.aten.sum.dim_IntList(permute_68, [2], True);  permute_68 = None
        view_111 = torch.ops.aten.view.default(sum_30, [512, 16, 16]);  sum_30 = None
        slice_scatter_12 = torch.ops.aten.slice_scatter.default(full_6, view_111, 2, 15, 9223372036854775807);  view_111 = None
        slice_scatter_13 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_12, 1, 0, 16);  slice_scatter_12 = None
        slice_scatter_14 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_13 = None
        view_112 = torch.ops.aten.view.default(slice_scatter_14, [512, 527]);  slice_scatter_14 = None
        constant_pad_nd_20 = torch.ops.aten.constant_pad_nd.default(view_112, [0, -15]);  view_112 = None
        view_113 = torch.ops.aten.view.default(constant_pad_nd_20, [512, 16, 32]);  constant_pad_nd_20 = None
        constant_pad_nd_21 = torch.ops.aten.constant_pad_nd.default(view_113, [0, -1]);  view_113 = None
        view_114 = torch.ops.aten.view.default(constant_pad_nd_21, [32, 16, 16, 31]);  constant_pad_nd_21 = None
        view_115 = torch.ops.aten.view.default(view_114, [8192, 31]);  view_114 = None
        permute_69 = torch.ops.aten.permute.default(view_115, [1, 0])
        mm_16 = torch.ops.aten.mm.default(permute_69, _unsafe_view_4);  permute_69 = _unsafe_view_4 = None
        permute_70 = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
        mm_17 = torch.ops.aten.mm.default(view_115, permute_71);  view_115 = permute_71 = None
        view_116 = torch.ops.aten.view.default(mm_17, [32, 16, 16, 16]);  mm_17 = None
        permute_72 = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_73 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        permute_74 = torch.ops.aten.permute.default(view_110, [0, 1, 3, 2, 4]);  view_110 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(permute_74, [2], True);  permute_74 = None
        view_117 = torch.ops.aten.view.default(sum_31, [512, 16, 16]);  sum_31 = None
        slice_scatter_15 = torch.ops.aten.slice_scatter.default(full_6, view_117, 2, 15, 9223372036854775807);  full_6 = view_117 = None
        slice_scatter_16 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_15, 1, 0, 16);  slice_scatter_15 = None
        slice_scatter_17 = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_16, 0, 0, 9223372036854775807);  full_7 = slice_scatter_16 = None
        view_118 = torch.ops.aten.view.default(slice_scatter_17, [512, 527]);  slice_scatter_17 = None
        constant_pad_nd_22 = torch.ops.aten.constant_pad_nd.default(view_118, [0, -15]);  view_118 = None
        view_119 = torch.ops.aten.view.default(constant_pad_nd_22, [512, 16, 32]);  constant_pad_nd_22 = None
        constant_pad_nd_23 = torch.ops.aten.constant_pad_nd.default(view_119, [0, -1]);  view_119 = None
        view_120 = torch.ops.aten.view.default(constant_pad_nd_23, [32, 16, 16, 31]);  constant_pad_nd_23 = None
        view_121 = torch.ops.aten.view.default(view_120, [8192, 31]);  view_120 = None
        permute_75 = torch.ops.aten.permute.default(view_121, [1, 0])
        mm_18 = torch.ops.aten.mm.default(permute_75, _unsafe_view_3);  permute_75 = _unsafe_view_3 = None
        permute_76 = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
        mm_19 = torch.ops.aten.mm.default(view_121, permute_77);  view_121 = permute_77 = None
        view_122 = torch.ops.aten.view.default(mm_19, [32, 16, 16, 16]);  mm_19 = None
        add_183 = torch.ops.aten.add.Tensor(permute_73, view_122);  permute_73 = view_122 = None
        permute_78 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        clone_116 = torch.ops.aten.clone.default(add_183, memory_format = torch.contiguous_format);  add_183 = None
        _unsafe_view_27 = torch.ops.aten._unsafe_view.default(clone_116, [32, 256, 16]);  clone_116 = None
        mul_365 = torch.ops.aten.mul.Tensor(sub_80, 0.25);  sub_80 = None
        view_123 = torch.ops.aten.view.default(mul_365, [32, 256, 256]);  mul_365 = None
        bmm_16 = torch.ops.aten.bmm.default(permute_79, view_123);  permute_79 = None
        bmm_17 = torch.ops.aten.bmm.default(view_123, permute_80);  view_123 = permute_80 = None
        view_124 = torch.ops.aten.view.default(bmm_16, [32, 16, 256]);  bmm_16 = None
        view_125 = torch.ops.aten.view.default(bmm_17, [32, 256, 16]);  bmm_17 = None
        add_184 = torch.ops.aten.add.Tensor(_unsafe_view_27, view_125);  _unsafe_view_27 = view_125 = None
        permute_81 = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        clone_117 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        _unsafe_view_28 = torch.ops.aten._unsafe_view.default(clone_117, [8, 256, 16, 16]);  clone_117 = None
        view_126 = torch.ops.aten.view.default(view_124, [8, 64, 16, 16]);  view_124 = None
        permute_82 = torch.ops.aten.permute.default(add_184, [0, 2, 1]);  add_184 = None
        clone_118 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        _unsafe_view_29 = torch.ops.aten._unsafe_view.default(clone_118, [8, 64, 16, 16]);  clone_118 = None
        cat_2 = torch.ops.aten.cat.default([_unsafe_view_29, view_126, _unsafe_view_28], 1);  _unsafe_view_29 = view_126 = _unsafe_view_28 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(cat_2, mul_177, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_2 = mul_177 = primals_34 = None
        getitem_35 = convolution_backward_8[0]
        getitem_36 = convolution_backward_8[1];  convolution_backward_8 = None
        mul_368 = torch.ops.aten.mul.Tensor(getitem_35, mul_367);  getitem_35 = mul_367 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3])
        sub_82 = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_234);  convolution_26 = unsqueeze_234 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_368, sub_82)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_369, [0, 2, 3]);  mul_369 = None
        mul_370 = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_371 = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
        mul_372 = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_373 = torch.ops.aten.mul.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_374 = torch.ops.aten.mul.Tensor(squeeze_64, primals_154);  primals_154 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_375 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_240);  sub_82 = unsqueeze_240 = None
        sub_84 = torch.ops.aten.sub.Tensor(mul_368, mul_375);  mul_368 = mul_375 = None
        sub_85 = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_237);  sub_84 = unsqueeze_237 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_243);  sub_85 = unsqueeze_243 = None
        mul_377 = torch.ops.aten.mul.Tensor(sum_33, squeeze_64);  sum_33 = squeeze_64 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_376, mul_169, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_376 = mul_169 = primals_33 = None
        getitem_38 = convolution_backward_9[0]
        getitem_39 = convolution_backward_9[1];  convolution_backward_9 = None
        add_186 = torch.ops.aten.add.Tensor(mul_341, getitem_38);  mul_341 = getitem_38 = None
        mul_380 = torch.ops.aten.mul.Tensor(add_186, mul_379);  add_186 = mul_379 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3])
        sub_87 = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_246);  convolution_25 = unsqueeze_246 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_380, sub_87)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
        mul_382 = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_383 = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
        mul_384 = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_385 = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_386 = torch.ops.aten.mul.Tensor(squeeze_61, primals_149);  primals_149 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_387 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_252);  sub_87 = unsqueeze_252 = None
        sub_89 = torch.ops.aten.sub.Tensor(mul_380, mul_387);  mul_387 = None
        sub_90 = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_249);  sub_89 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_255);  sub_90 = unsqueeze_255 = None
        mul_389 = torch.ops.aten.mul.Tensor(sum_35, squeeze_61);  sum_35 = squeeze_61 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_388, mul_137, primals_32, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_388 = primals_32 = None
        getitem_41 = convolution_backward_10[0]
        getitem_42 = convolution_backward_10[1];  convolution_backward_10 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_258);  convolution_24 = unsqueeze_258 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_380, sub_91)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_390, [0, 2, 3]);  mul_390 = None
        mul_392 = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
        mul_393 = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_394 = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_395 = torch.ops.aten.mul.Tensor(squeeze_58, primals_144);  primals_144 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_396 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_264);  sub_91 = unsqueeze_264 = None
        sub_93 = torch.ops.aten.sub.Tensor(mul_380, mul_396);  mul_380 = mul_396 = None
        sub_94 = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_249);  sub_93 = unsqueeze_249 = None
        mul_397 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_267);  sub_94 = unsqueeze_267 = None
        mul_398 = torch.ops.aten.mul.Tensor(sum_37, squeeze_58);  sum_37 = squeeze_58 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_397, mul_154, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_397 = mul_154 = primals_31 = None
        getitem_44 = convolution_backward_11[0]
        getitem_45 = convolution_backward_11[1];  convolution_backward_11 = None
        mul_399 = torch.ops.aten.mul.Tensor(getitem_44, mul_153);  mul_153 = None
        mul_400 = torch.ops.aten.mul.Tensor(getitem_44, expand_4);  getitem_44 = expand_4 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_399, [2, 3], True);  mul_399 = None
        view_127 = torch.ops.aten.view.default(sum_38, [8, 1, 256]);  sum_38 = None
        sub_95 = torch.ops.aten.sub.Tensor(1, sigmoid_21)
        mul_401 = torch.ops.aten.mul.Tensor(sigmoid_21, sub_95);  sigmoid_21 = sub_95 = None
        mul_402 = torch.ops.aten.mul.Tensor(view_127, mul_401);  view_127 = mul_401 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_402, view_8, primals_30, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_402 = view_8 = primals_30 = None
        getitem_47 = convolution_backward_12[0]
        getitem_48 = convolution_backward_12[1];  convolution_backward_12 = None
        view_128 = torch.ops.aten.view.default(getitem_47, [8, 256]);  getitem_47 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(view_128, 2);  view_128 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
        expand_24 = torch.ops.aten.expand.default(unsqueeze_269, [8, 256, 16, 16]);  unsqueeze_269 = None
        div_4 = torch.ops.aten.div.Scalar(expand_24, 256);  expand_24 = None
        add_188 = torch.ops.aten.add.Tensor(mul_400, div_4);  mul_400 = div_4 = None
        sigmoid_42 = torch.ops.aten.sigmoid.default(clone_78)
        empty_like_10 = torch.ops.aten.empty_like.default(sigmoid_42, memory_format = torch.preserve_format)
        full_like_10 = torch.ops.aten.full_like.default(empty_like_10, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_10 = None
        sub_96 = torch.ops.aten.sub.Tensor(full_like_10, sigmoid_42);  full_like_10 = None
        mul_403 = torch.ops.aten.mul.Tensor(clone_78, sub_96);  clone_78 = sub_96 = None
        add_189 = torch.ops.aten.add.Scalar(mul_403, 1);  mul_403 = None
        mul_404 = torch.ops.aten.mul.Tensor(sigmoid_42, add_189);  sigmoid_42 = add_189 = None
        mul_405 = torch.ops.aten.mul.Tensor(add_188, mul_404);  add_188 = mul_404 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3])
        sub_97 = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_272);  convolution_22 = unsqueeze_272 = None
        mul_406 = torch.ops.aten.mul.Tensor(mul_405, sub_97)
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3]);  mul_406 = None
        mul_407 = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
        mul_408 = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
        mul_409 = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_410 = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
        mul_411 = torch.ops.aten.mul.Tensor(squeeze_55, primals_139);  primals_139 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_278);  sub_97 = unsqueeze_278 = None
        sub_99 = torch.ops.aten.sub.Tensor(mul_405, mul_412);  mul_405 = mul_412 = None
        sub_100 = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_275);  sub_99 = unsqueeze_275 = None
        mul_413 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_281);  sub_100 = unsqueeze_281 = None
        mul_414 = torch.ops.aten.mul.Tensor(sum_40, squeeze_55);  sum_40 = squeeze_55 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_413, mul_145, primals_29, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_413 = mul_145 = primals_29 = None
        getitem_50 = convolution_backward_13[0]
        getitem_51 = convolution_backward_13[1];  convolution_backward_13 = None
        mul_417 = torch.ops.aten.mul.Tensor(getitem_50, mul_416);  getitem_50 = mul_416 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3])
        sub_102 = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_284);  convolution_21 = unsqueeze_284 = None
        mul_418 = torch.ops.aten.mul.Tensor(mul_417, sub_102)
        sum_42 = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
        mul_419 = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        mul_420 = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
        mul_421 = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_422 = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
        mul_423 = torch.ops.aten.mul.Tensor(squeeze_52, primals_134);  primals_134 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_290);  sub_102 = unsqueeze_290 = None
        sub_104 = torch.ops.aten.sub.Tensor(mul_417, mul_424);  mul_417 = mul_424 = None
        sub_105 = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_287);  sub_104 = unsqueeze_287 = None
        mul_425 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_293);  sub_105 = unsqueeze_293 = None
        mul_426 = torch.ops.aten.mul.Tensor(sum_42, squeeze_52);  sum_42 = squeeze_52 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_425, mul_137, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = mul_137 = primals_28 = None
        getitem_53 = convolution_backward_14[0]
        getitem_54 = convolution_backward_14[1];  convolution_backward_14 = None
        add_191 = torch.ops.aten.add.Tensor(getitem_41, getitem_53);  getitem_41 = getitem_53 = None
        mul_429 = torch.ops.aten.mul.Tensor(add_191, mul_428);  add_191 = mul_428 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3])
        sub_107 = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_296);  convolution_20 = unsqueeze_296 = None
        mul_430 = torch.ops.aten.mul.Tensor(mul_429, sub_107)
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
        mul_431 = torch.ops.aten.mul.Tensor(sum_43, 0.0001220703125)
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
        mul_432 = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
        mul_433 = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_434 = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
        mul_435 = torch.ops.aten.mul.Tensor(squeeze_49, primals_129);  primals_129 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
        mul_436 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_302);  sub_107 = unsqueeze_302 = None
        sub_109 = torch.ops.aten.sub.Tensor(mul_429, mul_436);  mul_436 = None
        sub_110 = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_299);  sub_109 = unsqueeze_299 = None
        mul_437 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_305);  sub_110 = unsqueeze_305 = None
        mul_438 = torch.ops.aten.mul.Tensor(sum_44, squeeze_49);  sum_44 = squeeze_49 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_437, mul_129, primals_27, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = mul_129 = primals_27 = None
        getitem_56 = convolution_backward_15[0]
        getitem_57 = convolution_backward_15[1];  convolution_backward_15 = None
        mul_439 = torch.ops.aten.mul.Tensor(getitem_56, mul_128);  mul_128 = None
        mul_440 = torch.ops.aten.mul.Tensor(getitem_56, expand_3);  getitem_56 = expand_3 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_439, [2, 3], True);  mul_439 = None
        view_129 = torch.ops.aten.view.default(sum_45, [8, 1, 128]);  sum_45 = None
        sub_111 = torch.ops.aten.sub.Tensor(1, sigmoid_17)
        mul_441 = torch.ops.aten.mul.Tensor(sigmoid_17, sub_111);  sigmoid_17 = sub_111 = None
        mul_442 = torch.ops.aten.mul.Tensor(view_129, mul_441);  view_129 = mul_441 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_442, view_6, primals_26, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_442 = view_6 = primals_26 = None
        getitem_59 = convolution_backward_16[0]
        getitem_60 = convolution_backward_16[1];  convolution_backward_16 = None
        view_130 = torch.ops.aten.view.default(getitem_59, [8, 128]);  getitem_59 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(view_130, 2);  view_130 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        expand_25 = torch.ops.aten.expand.default(unsqueeze_307, [8, 128, 32, 32]);  unsqueeze_307 = None
        div_5 = torch.ops.aten.div.Scalar(expand_25, 1024);  expand_25 = None
        add_193 = torch.ops.aten.add.Tensor(mul_440, div_5);  mul_440 = div_5 = None
        sigmoid_45 = torch.ops.aten.sigmoid.default(clone_75)
        empty_like_13 = torch.ops.aten.empty_like.default(sigmoid_45, memory_format = torch.preserve_format)
        full_like_13 = torch.ops.aten.full_like.default(empty_like_13, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_13 = None
        sub_112 = torch.ops.aten.sub.Tensor(full_like_13, sigmoid_45);  full_like_13 = None
        mul_443 = torch.ops.aten.mul.Tensor(clone_75, sub_112);  clone_75 = sub_112 = None
        add_194 = torch.ops.aten.add.Scalar(mul_443, 1);  mul_443 = None
        mul_444 = torch.ops.aten.mul.Tensor(sigmoid_45, add_194);  sigmoid_45 = add_194 = None
        mul_445 = torch.ops.aten.mul.Tensor(add_193, mul_444);  add_193 = mul_444 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3])
        sub_113 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_310);  convolution_18 = unsqueeze_310 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, sub_113)
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3]);  mul_446 = None
        mul_447 = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_448 = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
        mul_449 = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_450 = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_451 = torch.ops.aten.mul.Tensor(squeeze_46, primals_124);  primals_124 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_452 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_316);  sub_113 = unsqueeze_316 = None
        sub_115 = torch.ops.aten.sub.Tensor(mul_445, mul_452);  mul_445 = mul_452 = None
        sub_116 = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_313);  sub_115 = unsqueeze_313 = None
        mul_453 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_319);  sub_116 = unsqueeze_319 = None
        mul_454 = torch.ops.aten.mul.Tensor(sum_47, squeeze_46);  sum_47 = squeeze_46 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_453, mul_120, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_453 = mul_120 = primals_25 = None
        getitem_62 = convolution_backward_17[0]
        getitem_63 = convolution_backward_17[1];  convolution_backward_17 = None
        mul_457 = torch.ops.aten.mul.Tensor(getitem_62, mul_456);  getitem_62 = mul_456 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3])
        sub_118 = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_322);  convolution_17 = unsqueeze_322 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_457, sub_118)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
        mul_459 = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_460 = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
        mul_461 = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_462 = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_463 = torch.ops.aten.mul.Tensor(squeeze_43, primals_119);  primals_119 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_464 = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_328);  sub_118 = unsqueeze_328 = None
        sub_120 = torch.ops.aten.sub.Tensor(mul_457, mul_464);  mul_457 = mul_464 = None
        sub_121 = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_325);  sub_120 = unsqueeze_325 = None
        mul_465 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_331);  sub_121 = unsqueeze_331 = None
        mul_466 = torch.ops.aten.mul.Tensor(sum_49, squeeze_43);  sum_49 = squeeze_43 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_465, mul_112, primals_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = mul_112 = primals_24 = None
        getitem_65 = convolution_backward_18[0]
        getitem_66 = convolution_backward_18[1];  convolution_backward_18 = None
        add_196 = torch.ops.aten.add.Tensor(mul_429, getitem_65);  mul_429 = getitem_65 = None
        mul_469 = torch.ops.aten.mul.Tensor(add_196, mul_468);  add_196 = mul_468 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_469, [0, 2, 3])
        sub_123 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_334);  convolution_16 = unsqueeze_334 = None
        mul_470 = torch.ops.aten.mul.Tensor(mul_469, sub_123)
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
        mul_471 = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_472 = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
        mul_473 = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_474 = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_475 = torch.ops.aten.mul.Tensor(squeeze_40, primals_114);  primals_114 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_476 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_340);  sub_123 = unsqueeze_340 = None
        sub_125 = torch.ops.aten.sub.Tensor(mul_469, mul_476);  mul_476 = None
        sub_126 = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_337);  sub_125 = None
        mul_477 = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_343);  sub_126 = unsqueeze_343 = None
        mul_478 = torch.ops.aten.mul.Tensor(sum_51, squeeze_40);  sum_51 = squeeze_40 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_477, mul_80, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = primals_23 = None
        getitem_68 = convolution_backward_19[0]
        getitem_69 = convolution_backward_19[1];  convolution_backward_19 = None
        sub_127 = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_346);  convolution_15 = unsqueeze_346 = None
        mul_479 = torch.ops.aten.mul.Tensor(mul_469, sub_127)
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
        mul_481 = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
        mul_482 = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_483 = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_484 = torch.ops.aten.mul.Tensor(squeeze_37, primals_109);  primals_109 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_485 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_352);  sub_127 = unsqueeze_352 = None
        sub_129 = torch.ops.aten.sub.Tensor(mul_469, mul_485);  mul_469 = mul_485 = None
        sub_130 = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_337);  sub_129 = unsqueeze_337 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_355);  sub_130 = unsqueeze_355 = None
        mul_487 = torch.ops.aten.mul.Tensor(sum_53, squeeze_37);  sum_53 = squeeze_37 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_486, mul_97, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_97 = primals_22 = None
        getitem_71 = convolution_backward_20[0]
        getitem_72 = convolution_backward_20[1];  convolution_backward_20 = None
        mul_488 = torch.ops.aten.mul.Tensor(getitem_71, mul_96);  mul_96 = None
        mul_489 = torch.ops.aten.mul.Tensor(getitem_71, expand_2);  getitem_71 = expand_2 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_488, [2, 3], True);  mul_488 = None
        view_131 = torch.ops.aten.view.default(sum_54, [8, 1, 128]);  sum_54 = None
        sub_131 = torch.ops.aten.sub.Tensor(1, sigmoid_13)
        mul_490 = torch.ops.aten.mul.Tensor(sigmoid_13, sub_131);  sigmoid_13 = sub_131 = None
        mul_491 = torch.ops.aten.mul.Tensor(view_131, mul_490);  view_131 = mul_490 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_491, view_4, primals_21, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_491 = view_4 = primals_21 = None
        getitem_74 = convolution_backward_21[0]
        getitem_75 = convolution_backward_21[1];  convolution_backward_21 = None
        view_132 = torch.ops.aten.view.default(getitem_74, [8, 128]);  getitem_74 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(view_132, 2);  view_132 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        expand_26 = torch.ops.aten.expand.default(unsqueeze_357, [8, 128, 32, 32]);  unsqueeze_357 = None
        div_6 = torch.ops.aten.div.Scalar(expand_26, 1024);  expand_26 = None
        add_198 = torch.ops.aten.add.Tensor(mul_489, div_6);  mul_489 = div_6 = None
        sigmoid_48 = torch.ops.aten.sigmoid.default(clone_72)
        empty_like_16 = torch.ops.aten.empty_like.default(sigmoid_48, memory_format = torch.preserve_format)
        full_like_16 = torch.ops.aten.full_like.default(empty_like_16, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_16 = None
        sub_132 = torch.ops.aten.sub.Tensor(full_like_16, sigmoid_48);  full_like_16 = None
        mul_492 = torch.ops.aten.mul.Tensor(clone_72, sub_132);  clone_72 = sub_132 = None
        add_199 = torch.ops.aten.add.Scalar(mul_492, 1);  mul_492 = None
        mul_493 = torch.ops.aten.mul.Tensor(sigmoid_48, add_199);  sigmoid_48 = add_199 = None
        mul_494 = torch.ops.aten.mul.Tensor(add_198, mul_493);  add_198 = mul_493 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_494, [0, 2, 3])
        sub_133 = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_360);  convolution_13 = unsqueeze_360 = None
        mul_495 = torch.ops.aten.mul.Tensor(mul_494, sub_133)
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
        mul_496 = torch.ops.aten.mul.Tensor(sum_55, 0.0001220703125)
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_497 = torch.ops.aten.mul.Tensor(sum_56, 0.0001220703125)
        mul_498 = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_499 = torch.ops.aten.mul.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        mul_500 = torch.ops.aten.mul.Tensor(squeeze_34, primals_104);  primals_104 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_501 = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_366);  sub_133 = unsqueeze_366 = None
        sub_135 = torch.ops.aten.sub.Tensor(mul_494, mul_501);  mul_494 = mul_501 = None
        sub_136 = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_363);  sub_135 = unsqueeze_363 = None
        mul_502 = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_369);  sub_136 = unsqueeze_369 = None
        mul_503 = torch.ops.aten.mul.Tensor(sum_56, squeeze_34);  sum_56 = squeeze_34 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_502, mul_88, primals_20, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_502 = mul_88 = primals_20 = None
        getitem_77 = convolution_backward_22[0]
        getitem_78 = convolution_backward_22[1];  convolution_backward_22 = None
        mul_506 = torch.ops.aten.mul.Tensor(getitem_77, mul_505);  getitem_77 = mul_505 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3])
        sub_138 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_372);  convolution_12 = unsqueeze_372 = None
        mul_507 = torch.ops.aten.mul.Tensor(mul_506, sub_138)
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
        mul_508 = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_509 = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
        mul_510 = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_511 = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        mul_512 = torch.ops.aten.mul.Tensor(squeeze_31, primals_99);  primals_99 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_513 = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_378);  sub_138 = unsqueeze_378 = None
        sub_140 = torch.ops.aten.sub.Tensor(mul_506, mul_513);  mul_506 = mul_513 = None
        sub_141 = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_375);  sub_140 = unsqueeze_375 = None
        mul_514 = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_381);  sub_141 = unsqueeze_381 = None
        mul_515 = torch.ops.aten.mul.Tensor(sum_58, squeeze_31);  sum_58 = squeeze_31 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_514, mul_80, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = mul_80 = primals_19 = None
        getitem_80 = convolution_backward_23[0]
        getitem_81 = convolution_backward_23[1];  convolution_backward_23 = None
        add_201 = torch.ops.aten.add.Tensor(getitem_68, getitem_80);  getitem_68 = getitem_80 = None
        mul_518 = torch.ops.aten.mul.Tensor(add_201, mul_517);  add_201 = mul_517 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3])
        sub_143 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_384);  convolution_11 = unsqueeze_384 = None
        mul_519 = torch.ops.aten.mul.Tensor(mul_518, sub_143)
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
        mul_520 = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_521 = torch.ops.aten.mul.Tensor(sum_60, 3.0517578125e-05)
        mul_522 = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_523 = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        mul_524 = torch.ops.aten.mul.Tensor(squeeze_28, primals_94);  primals_94 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_525 = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_390);  sub_143 = unsqueeze_390 = None
        sub_145 = torch.ops.aten.sub.Tensor(mul_518, mul_525);  mul_525 = None
        sub_146 = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_387);  sub_145 = unsqueeze_387 = None
        mul_526 = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_393);  sub_146 = unsqueeze_393 = None
        mul_527 = torch.ops.aten.mul.Tensor(sum_60, squeeze_28);  sum_60 = squeeze_28 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_526, mul_72, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_526 = mul_72 = primals_18 = None
        getitem_83 = convolution_backward_24[0]
        getitem_84 = convolution_backward_24[1];  convolution_backward_24 = None
        mul_528 = torch.ops.aten.mul.Tensor(getitem_83, mul_71);  mul_71 = None
        mul_529 = torch.ops.aten.mul.Tensor(getitem_83, expand_1);  getitem_83 = expand_1 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_528, [2, 3], True);  mul_528 = None
        view_133 = torch.ops.aten.view.default(sum_61, [8, 1, 64]);  sum_61 = None
        sub_147 = torch.ops.aten.sub.Tensor(1, sigmoid_9)
        mul_530 = torch.ops.aten.mul.Tensor(sigmoid_9, sub_147);  sigmoid_9 = sub_147 = None
        mul_531 = torch.ops.aten.mul.Tensor(view_133, mul_530);  view_133 = mul_530 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_531, view_2, primals_17, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_531 = view_2 = primals_17 = None
        getitem_86 = convolution_backward_25[0]
        getitem_87 = convolution_backward_25[1];  convolution_backward_25 = None
        view_134 = torch.ops.aten.view.default(getitem_86, [8, 64]);  getitem_86 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(view_134, 2);  view_134 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
        expand_27 = torch.ops.aten.expand.default(unsqueeze_395, [8, 64, 64, 64]);  unsqueeze_395 = None
        div_7 = torch.ops.aten.div.Scalar(expand_27, 4096);  expand_27 = None
        add_203 = torch.ops.aten.add.Tensor(mul_529, div_7);  mul_529 = div_7 = None
        sigmoid_51 = torch.ops.aten.sigmoid.default(clone_69)
        empty_like_19 = torch.ops.aten.empty_like.default(sigmoid_51, memory_format = torch.preserve_format)
        full_like_19 = torch.ops.aten.full_like.default(empty_like_19, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_19 = None
        sub_148 = torch.ops.aten.sub.Tensor(full_like_19, sigmoid_51);  full_like_19 = None
        mul_532 = torch.ops.aten.mul.Tensor(clone_69, sub_148);  clone_69 = sub_148 = None
        add_204 = torch.ops.aten.add.Scalar(mul_532, 1);  mul_532 = None
        mul_533 = torch.ops.aten.mul.Tensor(sigmoid_51, add_204);  sigmoid_51 = add_204 = None
        mul_534 = torch.ops.aten.mul.Tensor(add_203, mul_533);  add_203 = mul_533 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3])
        sub_149 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_398);  convolution_9 = unsqueeze_398 = None
        mul_535 = torch.ops.aten.mul.Tensor(mul_534, sub_149)
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
        mul_536 = torch.ops.aten.mul.Tensor(sum_62, 3.0517578125e-05)
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
        mul_537 = torch.ops.aten.mul.Tensor(sum_63, 3.0517578125e-05)
        mul_538 = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_539 = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
        mul_540 = torch.ops.aten.mul.Tensor(squeeze_25, primals_89);  primals_89 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        mul_541 = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_404);  sub_149 = unsqueeze_404 = None
        sub_151 = torch.ops.aten.sub.Tensor(mul_534, mul_541);  mul_534 = mul_541 = None
        sub_152 = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_401);  sub_151 = unsqueeze_401 = None
        mul_542 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_407);  sub_152 = unsqueeze_407 = None
        mul_543 = torch.ops.aten.mul.Tensor(sum_63, squeeze_25);  sum_63 = squeeze_25 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_542, mul_63, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_542 = mul_63 = primals_16 = None
        getitem_89 = convolution_backward_26[0]
        getitem_90 = convolution_backward_26[1];  convolution_backward_26 = None
        mul_546 = torch.ops.aten.mul.Tensor(getitem_89, mul_545);  getitem_89 = mul_545 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2, 3])
        sub_154 = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_410);  convolution_8 = unsqueeze_410 = None
        mul_547 = torch.ops.aten.mul.Tensor(mul_546, sub_154)
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
        mul_548 = torch.ops.aten.mul.Tensor(sum_64, 3.0517578125e-05)
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
        mul_549 = torch.ops.aten.mul.Tensor(sum_65, 3.0517578125e-05)
        mul_550 = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_551 = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
        mul_552 = torch.ops.aten.mul.Tensor(squeeze_22, primals_84);  primals_84 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
        mul_553 = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_416);  sub_154 = unsqueeze_416 = None
        sub_156 = torch.ops.aten.sub.Tensor(mul_546, mul_553);  mul_546 = mul_553 = None
        sub_157 = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_413);  sub_156 = unsqueeze_413 = None
        mul_554 = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_419);  sub_157 = unsqueeze_419 = None
        mul_555 = torch.ops.aten.mul.Tensor(sum_65, squeeze_22);  sum_65 = squeeze_22 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_554, mul_55, primals_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = mul_55 = primals_15 = None
        getitem_92 = convolution_backward_27[0]
        getitem_93 = convolution_backward_27[1];  convolution_backward_27 = None
        add_206 = torch.ops.aten.add.Tensor(mul_518, getitem_92);  mul_518 = getitem_92 = None
        mul_558 = torch.ops.aten.mul.Tensor(add_206, mul_557);  add_206 = mul_557 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3])
        sub_159 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_422);  convolution_7 = unsqueeze_422 = None
        mul_559 = torch.ops.aten.mul.Tensor(mul_558, sub_159)
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
        mul_560 = torch.ops.aten.mul.Tensor(sum_66, 3.0517578125e-05)
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
        mul_561 = torch.ops.aten.mul.Tensor(sum_67, 3.0517578125e-05)
        mul_562 = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_563 = torch.ops.aten.mul.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
        mul_564 = torch.ops.aten.mul.Tensor(squeeze_19, primals_79);  primals_79 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_565 = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_428);  sub_159 = unsqueeze_428 = None
        sub_161 = torch.ops.aten.sub.Tensor(mul_558, mul_565);  mul_565 = None
        sub_162 = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_425);  sub_161 = None
        mul_566 = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_431);  sub_162 = unsqueeze_431 = None
        mul_567 = torch.ops.aten.mul.Tensor(sum_67, squeeze_19);  sum_67 = squeeze_19 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_566, getitem, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_566 = primals_14 = None
        getitem_95 = convolution_backward_28[0]
        getitem_96 = convolution_backward_28[1];  convolution_backward_28 = None
        sub_163 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_434);  convolution_6 = unsqueeze_434 = None
        mul_568 = torch.ops.aten.mul.Tensor(mul_558, sub_163)
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_568, [0, 2, 3]);  mul_568 = None
        mul_570 = torch.ops.aten.mul.Tensor(sum_69, 3.0517578125e-05)
        mul_571 = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_572 = torch.ops.aten.mul.Tensor(mul_570, mul_571);  mul_570 = mul_571 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        mul_573 = torch.ops.aten.mul.Tensor(squeeze_16, primals_74);  primals_74 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_574 = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_440);  sub_163 = unsqueeze_440 = None
        sub_165 = torch.ops.aten.sub.Tensor(mul_558, mul_574);  mul_558 = mul_574 = None
        sub_166 = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_425);  sub_165 = unsqueeze_425 = None
        mul_575 = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_443);  sub_166 = unsqueeze_443 = None
        mul_576 = torch.ops.aten.mul.Tensor(sum_69, squeeze_16);  sum_69 = squeeze_16 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_575, mul_40, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_575 = mul_40 = primals_13 = None
        getitem_98 = convolution_backward_29[0]
        getitem_99 = convolution_backward_29[1];  convolution_backward_29 = None
        mul_577 = torch.ops.aten.mul.Tensor(getitem_98, mul_39);  mul_39 = None
        mul_578 = torch.ops.aten.mul.Tensor(getitem_98, expand);  getitem_98 = expand = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_577, [2, 3], True);  mul_577 = None
        view_135 = torch.ops.aten.view.default(sum_70, [8, 1, 64]);  sum_70 = None
        sub_167 = torch.ops.aten.sub.Tensor(1, sigmoid_5)
        mul_579 = torch.ops.aten.mul.Tensor(sigmoid_5, sub_167);  sigmoid_5 = sub_167 = None
        mul_580 = torch.ops.aten.mul.Tensor(view_135, mul_579);  view_135 = mul_579 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_580, view, primals_12, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_580 = view = primals_12 = None
        getitem_101 = convolution_backward_30[0]
        getitem_102 = convolution_backward_30[1];  convolution_backward_30 = None
        view_136 = torch.ops.aten.view.default(getitem_101, [8, 64]);  getitem_101 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(view_136, 2);  view_136 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        expand_28 = torch.ops.aten.expand.default(unsqueeze_445, [8, 64, 64, 64]);  unsqueeze_445 = None
        div_8 = torch.ops.aten.div.Scalar(expand_28, 4096);  expand_28 = None
        add_208 = torch.ops.aten.add.Tensor(mul_578, div_8);  mul_578 = div_8 = None
        sigmoid_54 = torch.ops.aten.sigmoid.default(clone_66)
        empty_like_22 = torch.ops.aten.empty_like.default(sigmoid_54, memory_format = torch.preserve_format)
        full_like_22 = torch.ops.aten.full_like.default(empty_like_22, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_22 = None
        sub_168 = torch.ops.aten.sub.Tensor(full_like_22, sigmoid_54);  full_like_22 = None
        mul_581 = torch.ops.aten.mul.Tensor(clone_66, sub_168);  clone_66 = sub_168 = None
        add_209 = torch.ops.aten.add.Scalar(mul_581, 1);  mul_581 = None
        mul_582 = torch.ops.aten.mul.Tensor(sigmoid_54, add_209);  sigmoid_54 = add_209 = None
        mul_583 = torch.ops.aten.mul.Tensor(add_208, mul_582);  add_208 = mul_582 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_583, [0, 2, 3])
        sub_169 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_448);  convolution_4 = unsqueeze_448 = None
        mul_584 = torch.ops.aten.mul.Tensor(mul_583, sub_169)
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
        mul_585 = torch.ops.aten.mul.Tensor(sum_71, 3.0517578125e-05)
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_586 = torch.ops.aten.mul.Tensor(sum_72, 3.0517578125e-05)
        mul_587 = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_588 = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        mul_589 = torch.ops.aten.mul.Tensor(squeeze_13, primals_69);  primals_69 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_590 = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_454);  sub_169 = unsqueeze_454 = None
        sub_171 = torch.ops.aten.sub.Tensor(mul_583, mul_590);  mul_583 = mul_590 = None
        sub_172 = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_451);  sub_171 = unsqueeze_451 = None
        mul_591 = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_457);  sub_172 = unsqueeze_457 = None
        mul_592 = torch.ops.aten.mul.Tensor(sum_72, squeeze_13);  sum_72 = squeeze_13 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_591, mul_31, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_591 = mul_31 = primals_11 = None
        getitem_104 = convolution_backward_31[0]
        getitem_105 = convolution_backward_31[1];  convolution_backward_31 = None
        mul_595 = torch.ops.aten.mul.Tensor(getitem_104, mul_594);  getitem_104 = mul_594 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3])
        sub_174 = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_460);  convolution_3 = unsqueeze_460 = None
        mul_596 = torch.ops.aten.mul.Tensor(mul_595, sub_174)
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
        mul_597 = torch.ops.aten.mul.Tensor(sum_73, 3.0517578125e-05)
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_598 = torch.ops.aten.mul.Tensor(sum_74, 3.0517578125e-05)
        mul_599 = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_600 = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
        mul_601 = torch.ops.aten.mul.Tensor(squeeze_10, primals_64);  primals_64 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_602 = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_466);  sub_174 = unsqueeze_466 = None
        sub_176 = torch.ops.aten.sub.Tensor(mul_595, mul_602);  mul_595 = mul_602 = None
        sub_177 = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_463);  sub_176 = unsqueeze_463 = None
        mul_603 = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_469);  sub_177 = unsqueeze_469 = None
        mul_604 = torch.ops.aten.mul.Tensor(sum_74, squeeze_10);  sum_74 = squeeze_10 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_603, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = getitem = primals_10 = None
        getitem_107 = convolution_backward_32[0]
        getitem_108 = convolution_backward_32[1];  convolution_backward_32 = None
        add_211 = torch.ops.aten.add.Tensor(getitem_95, getitem_107);  getitem_95 = getitem_107 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(add_211, mul_23, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_211 = mul_23 = getitem_1 = None
        mul_607 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, mul_606);  max_pool2d_with_indices_backward = mul_606 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3])
        sub_179 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_472);  convolution_2 = unsqueeze_472 = None
        mul_608 = torch.ops.aten.mul.Tensor(mul_607, sub_179)
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
        mul_609 = torch.ops.aten.mul.Tensor(sum_75, 7.62939453125e-06)
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_610 = torch.ops.aten.mul.Tensor(sum_76, 7.62939453125e-06)
        mul_611 = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_612 = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
        mul_613 = torch.ops.aten.mul.Tensor(squeeze_7, primals_59);  primals_59 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_614 = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_478);  sub_179 = unsqueeze_478 = None
        sub_181 = torch.ops.aten.sub.Tensor(mul_607, mul_614);  mul_607 = mul_614 = None
        sub_182 = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_475);  sub_181 = unsqueeze_475 = None
        mul_615 = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_481);  sub_182 = unsqueeze_481 = None
        mul_616 = torch.ops.aten.mul.Tensor(sum_76, squeeze_7);  sum_76 = squeeze_7 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_615, mul_15, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_615 = mul_15 = primals_9 = None
        getitem_110 = convolution_backward_33[0]
        getitem_111 = convolution_backward_33[1];  convolution_backward_33 = None
        mul_619 = torch.ops.aten.mul.Tensor(getitem_110, mul_618);  getitem_110 = mul_618 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3])
        sub_184 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_484);  convolution_1 = unsqueeze_484 = None
        mul_620 = torch.ops.aten.mul.Tensor(mul_619, sub_184)
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
        mul_621 = torch.ops.aten.mul.Tensor(sum_77, 7.62939453125e-06)
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_622 = torch.ops.aten.mul.Tensor(sum_78, 7.62939453125e-06)
        mul_623 = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_624 = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
        mul_625 = torch.ops.aten.mul.Tensor(squeeze_4, primals_54);  primals_54 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_626 = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_490);  sub_184 = unsqueeze_490 = None
        sub_186 = torch.ops.aten.sub.Tensor(mul_619, mul_626);  mul_619 = mul_626 = None
        sub_187 = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_487);  sub_186 = unsqueeze_487 = None
        mul_627 = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_493);  sub_187 = unsqueeze_493 = None
        mul_628 = torch.ops.aten.mul.Tensor(sum_78, squeeze_4);  sum_78 = squeeze_4 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_627, mul_7, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = mul_7 = primals_8 = None
        getitem_113 = convolution_backward_34[0]
        getitem_114 = convolution_backward_34[1];  convolution_backward_34 = None
        mul_631 = torch.ops.aten.mul.Tensor(getitem_113, mul_630);  getitem_113 = mul_630 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3])
        sub_189 = torch.ops.aten.sub.Tensor(convolution, unsqueeze_496);  convolution = unsqueeze_496 = None
        mul_632 = torch.ops.aten.mul.Tensor(mul_631, sub_189)
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_632, [0, 2, 3]);  mul_632 = None
        mul_633 = torch.ops.aten.mul.Tensor(sum_79, 7.62939453125e-06)
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_634 = torch.ops.aten.mul.Tensor(sum_80, 7.62939453125e-06)
        mul_635 = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_636 = torch.ops.aten.mul.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        mul_637 = torch.ops.aten.mul.Tensor(squeeze_1, primals_49);  primals_49 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_638 = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_502);  sub_189 = unsqueeze_502 = None
        sub_191 = torch.ops.aten.sub.Tensor(mul_631, mul_638);  mul_631 = mul_638 = None
        sub_192 = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_499);  sub_191 = unsqueeze_499 = None
        mul_639 = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_505);  sub_192 = unsqueeze_505 = None
        mul_640 = torch.ops.aten.mul.Tensor(sum_80, squeeze_1);  sum_80 = squeeze_1 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_639, primals_45, primals_7, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_639 = primals_45 = primals_7 = None
        getitem_117 = convolution_backward_35[1];  convolution_backward_35 = None
        return [permute_78, permute_72, permute_60, permute_54, permute_42, permute_36, getitem_117, getitem_114, getitem_111, getitem_108, getitem_105, getitem_102, getitem_99, getitem_96, getitem_93, getitem_90, getitem_87, getitem_84, getitem_81, getitem_78, getitem_75, getitem_72, getitem_69, getitem_66, getitem_63, getitem_60, getitem_57, getitem_54, getitem_51, getitem_48, getitem_45, getitem_42, getitem_39, getitem_36, getitem_33, getitem_30, getitem_27, getitem_24, getitem_21, getitem_18, getitem_15, getitem_12, permute_28, view_62, None, None, None, None, mul_640, sum_79, None, None, None, mul_628, sum_77, None, None, None, mul_616, sum_75, None, None, None, mul_604, sum_73, None, None, None, mul_592, sum_71, None, None, None, mul_576, sum_66, None, None, None, mul_567, sum_66, None, None, None, mul_555, sum_64, None, None, None, mul_543, sum_62, None, None, None, mul_527, sum_59, None, None, None, mul_515, sum_57, None, None, None, mul_503, sum_55, None, None, None, mul_487, sum_50, None, None, None, mul_478, sum_50, None, None, None, mul_466, sum_48, None, None, None, mul_454, sum_46, None, None, None, mul_438, sum_43, None, None, None, mul_426, sum_41, None, None, None, mul_414, sum_39, None, None, None, mul_398, sum_34, None, None, None, mul_389, sum_34, None, None, None, mul_377, sum_32, None, None, None, mul_362, sum_27, None, None, None, mul_350, sum_25, None, None, None, mul_338, sum_23, None, None, None, mul_323, sum_18, None, None, None, mul_311, sum_14, None, None, None, mul_302, sum_14, None, None, None, mul_290, sum_12, None, None, None, mul_275, sum_7, None, None, None, mul_263, sum_5]
        
args = [((24, 3, 3, 3), (27, 9, 3, 1), torch.float32, 'cuda'), ((32, 24, 3, 3), (216, 9, 3, 1), torch.float32, 'cuda'), ((64, 32, 3, 3), (288, 9, 3, 1), torch.float32, 'cuda'), ((64, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((64, 16, 3, 3), (144, 9, 3, 1), torch.float32, 'cuda'), ((1, 1, 3), (3, 3, 1), torch.float32, 'cuda'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((64, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((64, 16, 3, 3), (144, 9, 3, 1), torch.float32, 'cuda'), ((1, 1, 3), (3, 3, 1), torch.float32, 'cuda'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((128, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((128, 16, 3, 3), (144, 9, 3, 1), torch.float32, 'cuda'), ((1, 1, 5), (5, 5, 1), torch.float32, 'cuda'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((512, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((128, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((128, 16, 3, 3), (144, 9, 3, 1), torch.float32, 'cuda'), ((1, 1, 5), (5, 5, 1), torch.float32, 'cuda'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((256, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((256, 16, 3, 3), (144, 9, 3, 1), torch.float32, 'cuda'), ((1, 1, 5), (5, 5, 1), torch.float32, 'cuda'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((1024, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((384, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((512, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((640, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((2048, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cuda'), ((640, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((8, 3, 256, 256), (196608, 65536, 256, 1), torch.float32, 'cuda'), ((24,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((8, 24, 128, 128), (393216, 16384, 128, 1), torch.float32, 'cuda'), ((24,), (1,), torch.float32, 'cuda'), ((8, 24, 128, 128), (393216, 16384, 128, 1), torch.float32, 'cuda'), ((8, 32, 128, 128), (524288, 16384, 128, 1), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((8, 32, 128, 128), (524288, 16384, 128, 1), torch.float32, 'cuda'), ((8, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.int64, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((8, 1, 64), (64, 64, 1), torch.float32, 'cuda'), ((8, 1, 64), (64, 64, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((8, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((8, 1, 64), (64, 64, 1), torch.float32, 'cuda'), ((8, 1, 64), (64, 64, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((8, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((8, 1, 128), (128, 128, 1), torch.float32, 'cuda'), ((8, 1, 128), (128, 128, 1), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((8, 512, 32, 32), (524288, 1024, 32, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 32, 32), (524288, 1024, 32, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 32, 32), (524288, 1024, 32, 1), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((8, 1, 128), (128, 128, 1), torch.float32, 'cuda'), ((8, 1, 128), (128, 128, 1), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((8, 512, 32, 32), (524288, 1024, 32, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 32, 32), (524288, 1024, 32, 1), torch.float32, 'cuda'), ((8, 256, 32, 32), (262144, 1024, 32, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 32, 32), (262144, 1024, 32, 1), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((8, 1, 256), (256, 256, 1), torch.float32, 'cuda'), ((8, 1, 256), (256, 256, 1), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((8, 1024, 16, 16), (262144, 256, 16, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((8, 1024, 16, 16), (262144, 256, 16, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((8, 1024, 16, 16), (262144, 256, 16, 1), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((8192, 16), (16, 1), torch.float32, 'cuda'), ((8192, 16), (16, 1), torch.float32, 'cuda'), ((32, 256, 256), (65536, 256, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((8, 1024, 16, 16), (262144, 256, 16, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((8, 1024, 16, 16), (262144, 256, 16, 1), torch.float32, 'cuda'), ((8, 512, 16, 16), (131072, 256, 16, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 16, 16), (131072, 256, 16, 1), torch.float32, 'cuda'), ((8192, 16), (16, 1), torch.float32, 'cuda'), ((8192, 16), (16, 1), torch.float32, 'cuda'), ((32, 256, 256), (65536, 256, 1), torch.float32, 'cuda'), ((8, 512, 16, 16), (131072, 256, 16, 1), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((8, 2048, 8, 8), (131072, 64, 8, 1), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((8, 2048, 8, 8), (131072, 64, 8, 1), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((8, 2048, 8, 8), (131072, 64, 8, 1), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((2048, 16), (16, 1), torch.float32, 'cuda'), ((2048, 16), (16, 1), torch.float32, 'cuda'), ((32, 64, 64), (4096, 64, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((8, 2048, 8, 8), (131072, 64, 8, 1), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((8, 2048), (2048, 1), torch.float32, 'cuda'), ((1000, 2048), (2048, 1), torch.float32, 'cuda'), ((8, 2048, 8, 8), (131072, 64, 8, 1), torch.float32, 'cuda'), ((1, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((32, 64, 64), (4096, 1, 64), torch.float32, 'cuda'), ((32, 128, 64), (8192, 64, 1), torch.float32, 'cuda'), ((15, 16), (16, 1), torch.float32, 'cuda'), ((15, 16), (16, 1), torch.float32, 'cuda'), ((32, 16, 64), (1024, 64, 1), torch.float32, 'cuda'), ((32, 64, 16), (1024, 1, 64), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((1, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((8, 2048, 8, 8), (131072, 64, 8, 1), torch.float32, 'cuda'), ((1, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cuda'), ((1, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cuda'), ((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32, 'cuda'), ((1, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((32, 256, 256), (65536, 1, 256), torch.float32, 'cuda'), ((32, 128, 256), (32768, 256, 1), torch.float32, 'cuda'), ((31, 16), (16, 1), torch.float32, 'cuda'), ((31, 16), (16, 1), torch.float32, 'cuda'), ((32, 16, 256), (4096, 256, 1), torch.float32, 'cuda'), ((32, 256, 16), (4096, 1, 256), torch.float32, 'cuda'), ((8, 512, 16, 16), (131072, 256, 16, 1), torch.float32, 'cuda'), ((1, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((8, 1024, 16, 16), (262144, 256, 16, 1), torch.float32, 'cuda'), ((1, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((32, 256, 256), (65536, 1, 256), torch.float32, 'cuda'), ((32, 64, 256), (16384, 256, 1), torch.float32, 'cuda'), ((31, 16), (16, 1), torch.float32, 'cuda'), ((31, 16), (16, 1), torch.float32, 'cuda'), ((32, 16, 256), (4096, 256, 1), torch.float32, 'cuda'), ((32, 256, 16), (4096, 1, 256), torch.float32, 'cuda'), ((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32, 'cuda'), ((1, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((8, 1024, 16, 16), (262144, 256, 16, 1), torch.float32, 'cuda'), ((1, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((1, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cuda'), ((1, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((8, 256, 32, 32), (262144, 1024, 32, 1), torch.float32, 'cuda'), ((1, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((8, 512, 32, 32), (524288, 1024, 32, 1), torch.float32, 'cuda'), ((1, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((8, 512, 32, 32), (524288, 1024, 32, 1), torch.float32, 'cuda'), ((1, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((1, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((8, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32, 'cuda'), ((1, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((8, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32, 'cuda'), ((1, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((1, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((8, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((8, 32, 128, 128), (524288, 16384, 128, 1), torch.float32, 'cuda'), ((1, 32, 1, 1), (32, 1, 1, 1), torch.float32, 'cuda'), ((8, 24, 128, 128), (393216, 16384, 128, 1), torch.float32, 'cuda'), ((1, 24, 1, 1), (24, 1, 1, 1), torch.float32, 'cuda'), ((24,), (1,), torch.float32, 'cuda'), ((24,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((2048,), (1,), torch.float32, 'cuda'), ((8, 1000), (1000, 1), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
