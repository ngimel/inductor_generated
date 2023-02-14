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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x88X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\x13\x00\x00\x00torch.distributionsq\x18X\x0c\x00\x00\x00torch._primsq\x19X\x0b\x00\x00\x00torch._refsq\x1aX\r\x00\x00\x00torch._decompq\x1bX\r\x00\x00\x00torch.testingq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x18\x00\x00\x00error_on_nested_fx_traceq&\x88X\t\x00\x00\x00allow_rnnq\'\x89X\x08\x00\x00\x00base_dirq(X\x1c\x00\x00\x00/scratch/ngimel/work/pytorchq)X\x0e\x00\x00\x00debug_dir_rootq*X0\x00\x00\x00/scratch/ngimel/work/pytorch/torch_compile_debugq+X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq,\x89X\x13\x00\x00\x00_save_config_ignoreq-h\r]q.(X\x0b\x00\x00\x00repro_afterq/X\x0b\x00\x00\x00repro_levelq0X!\x00\x00\x00skipfiles_inline_module_allowlistq1X\x12\x00\x00\x00constant_functionsq2e\x85q3Rq4u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x12\x00\x00\x00developer_warningsq\x02\x88X\x10\x00\x00\x00disable_progressq\x03\x88X\x10\x00\x00\x00verbose_progressq\x04\x89X\x0b\x00\x00\x00cpp_wrapperq\x05\x89X\x03\x00\x00\x00dceq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x0c\x00\x00\x00size_assertsq\x08\x88X\x10\x00\x00\x00pick_loop_ordersq\t\x88X\x0f\x00\x00\x00inplace_buffersq\n\x88X\x11\x00\x00\x00benchmark_harnessq\x0b\x88X\x0f\x00\x00\x00epilogue_fusionq\x0c\x89X\x15\x00\x00\x00epilogue_fusion_firstq\r\x89X\x0f\x00\x00\x00pattern_matcherq\x0e\x88X\n\x00\x00\x00reorderingq\x0f\x89X\x0c\x00\x00\x00max_autotuneq\x10\x89X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x88X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x0f\x00\x00\x00compile_threadsq\x1bK X\x13\x00\x00\x00kernel_name_max_opsq\x1cK\nX\r\x00\x00\x00shape_paddingq\x1d\x89X\x0e\x00\x00\x00permute_fusionq\x1e\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq\x1f\x89X\x0b\x00\x00\x00cpp.threadsq J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq!\x89X\x0b\x00\x00\x00cpp.simdlenq"NX\x12\x00\x00\x00cpp.min_chunk_sizeq#M\x00\x10X\x07\x00\x00\x00cpp.cxxq$NX\x03\x00\x00\x00g++q%\x86q&X\x19\x00\x00\x00cpp.enable_kernel_profileq\'\x89X\x12\x00\x00\x00cpp.weight_prepackq(\x88X\x11\x00\x00\x00triton.cudagraphsq)\x89X\x17\x00\x00\x00triton.debug_sync_graphq*\x89X\x18\x00\x00\x00triton.debug_sync_kernelq+\x89X\x12\x00\x00\x00triton.convolutionq,X\x04\x00\x00\x00atenq-X\x15\x00\x00\x00triton.dense_indexingq.\x89X\x10\x00\x00\x00triton.max_tilesq/K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq0\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq1\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq2\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq3\x88X\x1f\x00\x00\x00triton.descriptive_kernel_namesq4\x89X\x1c\x00\x00\x00triton.persistent_reductionsq5\x89X\r\x00\x00\x00trace.enabledq6\x88X\x0f\x00\x00\x00trace.debug_logq7\x88X\x0e\x00\x00\x00trace.info_logq8\x89X\x0e\x00\x00\x00trace.fx_graphq9\x88X\x1a\x00\x00\x00trace.fx_graph_transformedq:\x88X\x13\x00\x00\x00trace.ir_pre_fusionq;\x88X\x14\x00\x00\x00trace.ir_post_fusionq<\x88X\x11\x00\x00\x00trace.output_codeq=\x88X\x13\x00\x00\x00trace.graph_diagramq>\x89X\x15\x00\x00\x00trace.compile_profileq?\x89X\x10\x00\x00\x00trace.upload_tarq@Nu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x12\x00\x00\x00use_dynamic_shapesq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x03\x00\x00\x00cseq\x08\x88X\x10\x00\x00\x00max_dist_from_bwq\tK\x03X\x0b\x00\x00\x00debug_jointq\n\x89X\x0c\x00\x00\x00debug_graphsq\x0b\x89X\x11\x00\x00\x00debug_partitionerq\x0c\x89X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.0a0+git728dfee
# torch cuda version: 11.6
# torch git version: 728dfeee486fbd965710ccbb225fd275dd7bd35c


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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_60, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, primals_204, primals_209, primals_214, primals_219, primals_224, primals_229, primals_234, primals_239, primals_244, primals_249, primals_254, primals_259, primals_264, primals_269, primals_274, primals_279, primals_284, primals_289, primals_294, primals_299, primals_304, primals_309, primals_314, primals_319, primals_324, primals_329, primals_334, primals_339, primals_344, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, view, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160, tangents_161, tangents_162, tangents_163, tangents_164, tangents_165, tangents_166, tangents_167, tangents_168, tangents_169, tangents_170, tangents_171, tangents_172):
        mm = torch.ops.aten.mm.default(tangents_115, permute_1);  permute_1 = None
        permute_2 = torch.ops.aten.permute.default(tangents_115, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        permute_3 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_115, [0], True);  tangents_115 = None
        view_1 = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        permute_4 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        view_2 = torch.ops.aten.view.default(mm, [8, 2560, 1, 1]);  mm = None
        expand = torch.ops.aten.expand.default(view_2, [8, 2560, 8, 8]);  view_2 = None
        div = torch.ops.aten.div.Scalar(expand, 64);  expand = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(le, scalar_tensor, div);  le = div = None
        sum_2 = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_57 = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_230);  convolution_56 = unsqueeze_230 = None
        mul_399 = torch.ops.aten.mul.Tensor(where, sub_57)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3]);  mul_399 = None
        mul_400 = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
        mul_401 = torch.ops.aten.mul.Tensor(sum_3, 0.001953125)
        mul_402 = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_403 = torch.ops.aten.mul.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(mul_403, 0);  mul_403 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
        mul_404 = torch.ops.aten.mul.Tensor(squeeze_169, primals_344);  primals_344 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
        mul_405 = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_236);  sub_57 = unsqueeze_236 = None
        sub_59 = torch.ops.aten.sub.Tensor(where, mul_405);  where = mul_405 = None
        sub_60 = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_233);  sub_59 = unsqueeze_233 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_239);  sub_60 = unsqueeze_239 = None
        mul_407 = torch.ops.aten.mul.Tensor(sum_3, squeeze_169);  sum_3 = squeeze_169 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_406, relu_51, primals_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = primals_57 = None
        getitem_114 = convolution_backward[0]
        getitem_115 = convolution_backward[1];  convolution_backward = None
        le_1 = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        where_1 = torch.ops.aten.where.self(le_1, scalar_tensor, getitem_114);  le_1 = getitem_114 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_61 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_242);  convolution_55 = unsqueeze_242 = None
        mul_408 = torch.ops.aten.mul.Tensor(where_1, sub_61)
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2, 3]);  mul_408 = None
        mul_409 = torch.ops.aten.mul.Tensor(sum_4, 0.001953125)
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
        mul_410 = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
        mul_411 = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_412 = torch.ops.aten.mul.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
        mul_413 = torch.ops.aten.mul.Tensor(squeeze_166, primals_339);  primals_339 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_248);  sub_61 = unsqueeze_248 = None
        sub_63 = torch.ops.aten.sub.Tensor(where_1, mul_414);  mul_414 = None
        sub_64 = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_245);  sub_63 = unsqueeze_245 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_251);  sub_64 = unsqueeze_251 = None
        mul_416 = torch.ops.aten.mul.Tensor(sum_5, squeeze_166);  sum_5 = squeeze_166 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_415, relu_50, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_415 = primals_56 = None
        getitem_117 = convolution_backward_1[0]
        getitem_118 = convolution_backward_1[1];  convolution_backward_1 = None
        le_2 = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        where_2 = torch.ops.aten.where.self(le_2, scalar_tensor, getitem_117);  le_2 = getitem_117 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_65 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_254);  convolution_54 = unsqueeze_254 = None
        mul_417 = torch.ops.aten.mul.Tensor(where_2, sub_65)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3]);  mul_417 = None
        mul_418 = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
        mul_419 = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
        mul_420 = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_421 = torch.ops.aten.mul.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
        mul_422 = torch.ops.aten.mul.Tensor(squeeze_163, primals_334);  primals_334 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
        mul_423 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_260);  sub_65 = unsqueeze_260 = None
        sub_67 = torch.ops.aten.sub.Tensor(where_2, mul_423);  where_2 = mul_423 = None
        sub_68 = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_257);  sub_67 = unsqueeze_257 = None
        mul_424 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_263);  sub_68 = unsqueeze_263 = None
        mul_425 = torch.ops.aten.mul.Tensor(sum_7, squeeze_163);  sum_7 = squeeze_163 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_424, relu_49, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_424 = primals_55 = None
        getitem_120 = convolution_backward_2[0]
        getitem_121 = convolution_backward_2[1];  convolution_backward_2 = None
        le_3 = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        where_3 = torch.ops.aten.where.self(le_3, scalar_tensor, getitem_120);  le_3 = getitem_120 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_69 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_266);  convolution_53 = unsqueeze_266 = None
        mul_426 = torch.ops.aten.mul.Tensor(where_3, sub_69)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
        mul_427 = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
        mul_428 = torch.ops.aten.mul.Tensor(sum_9, 0.001953125)
        mul_429 = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_430 = torch.ops.aten.mul.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
        mul_431 = torch.ops.aten.mul.Tensor(squeeze_160, primals_329);  primals_329 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
        mul_432 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_272);  sub_69 = unsqueeze_272 = None
        sub_71 = torch.ops.aten.sub.Tensor(where_3, mul_432);  where_3 = mul_432 = None
        sub_72 = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_269);  sub_71 = unsqueeze_269 = None
        mul_433 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_275);  sub_72 = unsqueeze_275 = None
        mul_434 = torch.ops.aten.mul.Tensor(sum_9, squeeze_160);  sum_9 = squeeze_160 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_433, relu_48, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_433 = primals_54 = None
        getitem_123 = convolution_backward_3[0]
        getitem_124 = convolution_backward_3[1];  convolution_backward_3 = None
        add_303 = torch.ops.aten.add.Tensor(where_1, getitem_123);  where_1 = getitem_123 = None
        le_4 = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        where_4 = torch.ops.aten.where.self(le_4, scalar_tensor, add_303);  le_4 = add_303 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_73 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_278);  convolution_52 = unsqueeze_278 = None
        mul_435 = torch.ops.aten.mul.Tensor(where_4, sub_73)
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_435, [0, 2, 3]);  mul_435 = None
        mul_436 = torch.ops.aten.mul.Tensor(sum_10, 0.001953125)
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
        mul_437 = torch.ops.aten.mul.Tensor(sum_11, 0.001953125)
        mul_438 = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_439 = torch.ops.aten.mul.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
        mul_440 = torch.ops.aten.mul.Tensor(squeeze_157, primals_324);  primals_324 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        mul_441 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_284);  sub_73 = unsqueeze_284 = None
        sub_75 = torch.ops.aten.sub.Tensor(where_4, mul_441);  mul_441 = None
        sub_76 = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_281);  sub_75 = unsqueeze_281 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_287);  sub_76 = unsqueeze_287 = None
        mul_443 = torch.ops.aten.mul.Tensor(sum_11, squeeze_157);  sum_11 = squeeze_157 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_442, relu_47, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_442 = primals_53 = None
        getitem_126 = convolution_backward_4[0]
        getitem_127 = convolution_backward_4[1];  convolution_backward_4 = None
        le_5 = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_5 = torch.ops.aten.where.self(le_5, scalar_tensor, getitem_126);  le_5 = getitem_126 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_77 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_290);  convolution_51 = unsqueeze_290 = None
        mul_444 = torch.ops.aten.mul.Tensor(where_5, sub_77)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_444, [0, 2, 3]);  mul_444 = None
        mul_445 = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
        mul_446 = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
        mul_447 = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_448 = torch.ops.aten.mul.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
        mul_449 = torch.ops.aten.mul.Tensor(squeeze_154, primals_319);  primals_319 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
        mul_450 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_296);  sub_77 = unsqueeze_296 = None
        sub_79 = torch.ops.aten.sub.Tensor(where_5, mul_450);  where_5 = mul_450 = None
        sub_80 = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_293);  sub_79 = unsqueeze_293 = None
        mul_451 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_299);  sub_80 = unsqueeze_299 = None
        mul_452 = torch.ops.aten.mul.Tensor(sum_13, squeeze_154);  sum_13 = squeeze_154 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_451, relu_46, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_451 = primals_52 = None
        getitem_129 = convolution_backward_5[0]
        getitem_130 = convolution_backward_5[1];  convolution_backward_5 = None
        le_6 = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_6 = torch.ops.aten.where.self(le_6, scalar_tensor, getitem_129);  le_6 = getitem_129 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_81 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_302);  convolution_50 = unsqueeze_302 = None
        mul_453 = torch.ops.aten.mul.Tensor(where_6, sub_81)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3]);  mul_453 = None
        mul_454 = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
        mul_455 = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
        mul_456 = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_457 = torch.ops.aten.mul.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
        mul_458 = torch.ops.aten.mul.Tensor(squeeze_151, primals_314);  primals_314 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
        mul_459 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_308);  sub_81 = unsqueeze_308 = None
        sub_83 = torch.ops.aten.sub.Tensor(where_6, mul_459);  where_6 = mul_459 = None
        sub_84 = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_305);  sub_83 = unsqueeze_305 = None
        mul_460 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_311);  sub_84 = unsqueeze_311 = None
        mul_461 = torch.ops.aten.mul.Tensor(sum_15, squeeze_151);  sum_15 = squeeze_151 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_460, relu_45, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_460 = primals_51 = None
        getitem_132 = convolution_backward_6[0]
        getitem_133 = convolution_backward_6[1];  convolution_backward_6 = None
        add_304 = torch.ops.aten.add.Tensor(where_4, getitem_132);  where_4 = getitem_132 = None
        le_7 = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_7 = torch.ops.aten.where.self(le_7, scalar_tensor, add_304);  le_7 = add_304 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_85 = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_314);  convolution_49 = unsqueeze_314 = None
        mul_462 = torch.ops.aten.mul.Tensor(where_7, sub_85)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_462, [0, 2, 3]);  mul_462 = None
        mul_463 = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
        mul_464 = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
        mul_465 = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_466 = torch.ops.aten.mul.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
        mul_467 = torch.ops.aten.mul.Tensor(squeeze_148, primals_309);  primals_309 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
        mul_468 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_320);  sub_85 = unsqueeze_320 = None
        sub_87 = torch.ops.aten.sub.Tensor(where_7, mul_468);  mul_468 = None
        sub_88 = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_317);  sub_87 = unsqueeze_317 = None
        mul_469 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_323);  sub_88 = unsqueeze_323 = None
        mul_470 = torch.ops.aten.mul.Tensor(sum_17, squeeze_148);  sum_17 = squeeze_148 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_469, relu_44, primals_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_469 = primals_50 = None
        getitem_135 = convolution_backward_7[0]
        getitem_136 = convolution_backward_7[1];  convolution_backward_7 = None
        le_8 = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_8 = torch.ops.aten.where.self(le_8, scalar_tensor, getitem_135);  le_8 = getitem_135 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_89 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_326);  convolution_48 = unsqueeze_326 = None
        mul_471 = torch.ops.aten.mul.Tensor(where_8, sub_89)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_471, [0, 2, 3]);  mul_471 = None
        mul_472 = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
        mul_473 = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
        mul_474 = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_475 = torch.ops.aten.mul.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
        mul_476 = torch.ops.aten.mul.Tensor(squeeze_145, primals_304);  primals_304 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
        mul_477 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_332);  sub_89 = unsqueeze_332 = None
        sub_91 = torch.ops.aten.sub.Tensor(where_8, mul_477);  where_8 = mul_477 = None
        sub_92 = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_329);  sub_91 = unsqueeze_329 = None
        mul_478 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_335);  sub_92 = unsqueeze_335 = None
        mul_479 = torch.ops.aten.mul.Tensor(sum_19, squeeze_145);  sum_19 = squeeze_145 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_478, relu_43, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_478 = primals_49 = None
        getitem_138 = convolution_backward_8[0]
        getitem_139 = convolution_backward_8[1];  convolution_backward_8 = None
        le_9 = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_9 = torch.ops.aten.where.self(le_9, scalar_tensor, getitem_138);  le_9 = getitem_138 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_93 = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_338);  convolution_47 = unsqueeze_338 = None
        mul_480 = torch.ops.aten.mul.Tensor(where_9, sub_93)
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3]);  mul_480 = None
        mul_481 = torch.ops.aten.mul.Tensor(sum_20, 0.001953125)
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
        mul_482 = torch.ops.aten.mul.Tensor(sum_21, 0.001953125)
        mul_483 = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_484 = torch.ops.aten.mul.Tensor(mul_482, mul_483);  mul_482 = mul_483 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
        mul_485 = torch.ops.aten.mul.Tensor(squeeze_142, primals_299);  primals_299 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_344);  sub_93 = unsqueeze_344 = None
        sub_95 = torch.ops.aten.sub.Tensor(where_9, mul_486);  where_9 = mul_486 = None
        sub_96 = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_341);  sub_95 = unsqueeze_341 = None
        mul_487 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_347);  sub_96 = unsqueeze_347 = None
        mul_488 = torch.ops.aten.mul.Tensor(sum_21, squeeze_142);  sum_21 = squeeze_142 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_487, relu_42, primals_48, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_487 = primals_48 = None
        getitem_141 = convolution_backward_9[0]
        getitem_142 = convolution_backward_9[1];  convolution_backward_9 = None
        add_305 = torch.ops.aten.add.Tensor(where_7, getitem_141);  where_7 = getitem_141 = None
        le_10 = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_10 = torch.ops.aten.where.self(le_10, scalar_tensor, add_305);  le_10 = add_305 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_97 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_350);  convolution_46 = unsqueeze_350 = None
        mul_489 = torch.ops.aten.mul.Tensor(where_10, sub_97)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3]);  mul_489 = None
        mul_490 = torch.ops.aten.mul.Tensor(sum_22, 0.001953125)
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
        mul_491 = torch.ops.aten.mul.Tensor(sum_23, 0.001953125)
        mul_492 = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_493 = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
        mul_494 = torch.ops.aten.mul.Tensor(squeeze_139, primals_294);  primals_294 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
        mul_495 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_356);  sub_97 = unsqueeze_356 = None
        sub_99 = torch.ops.aten.sub.Tensor(where_10, mul_495);  mul_495 = None
        sub_100 = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_353);  sub_99 = unsqueeze_353 = None
        mul_496 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_359);  sub_100 = unsqueeze_359 = None
        mul_497 = torch.ops.aten.mul.Tensor(sum_23, squeeze_139);  sum_23 = squeeze_139 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_496, relu_41, primals_47, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_496 = primals_47 = None
        getitem_144 = convolution_backward_10[0]
        getitem_145 = convolution_backward_10[1];  convolution_backward_10 = None
        le_11 = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_11 = torch.ops.aten.where.self(le_11, scalar_tensor, getitem_144);  le_11 = getitem_144 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_101 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_362);  convolution_45 = unsqueeze_362 = None
        mul_498 = torch.ops.aten.mul.Tensor(where_11, sub_101)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_498, [0, 2, 3]);  mul_498 = None
        mul_499 = torch.ops.aten.mul.Tensor(sum_24, 0.001953125)
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
        mul_500 = torch.ops.aten.mul.Tensor(sum_25, 0.001953125)
        mul_501 = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_502 = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
        mul_503 = torch.ops.aten.mul.Tensor(squeeze_136, primals_289);  primals_289 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
        mul_504 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_368);  sub_101 = unsqueeze_368 = None
        sub_103 = torch.ops.aten.sub.Tensor(where_11, mul_504);  where_11 = mul_504 = None
        sub_104 = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_365);  sub_103 = unsqueeze_365 = None
        mul_505 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_371);  sub_104 = unsqueeze_371 = None
        mul_506 = torch.ops.aten.mul.Tensor(sum_25, squeeze_136);  sum_25 = squeeze_136 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_505, relu_40, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_505 = primals_46 = None
        getitem_147 = convolution_backward_11[0]
        getitem_148 = convolution_backward_11[1];  convolution_backward_11 = None
        le_12 = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_12 = torch.ops.aten.where.self(le_12, scalar_tensor, getitem_147);  le_12 = getitem_147 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_105 = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_374);  convolution_44 = unsqueeze_374 = None
        mul_507 = torch.ops.aten.mul.Tensor(where_12, sub_105)
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
        mul_508 = torch.ops.aten.mul.Tensor(sum_26, 0.001953125)
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
        mul_509 = torch.ops.aten.mul.Tensor(sum_27, 0.001953125)
        mul_510 = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_511 = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
        mul_512 = torch.ops.aten.mul.Tensor(squeeze_133, primals_284);  primals_284 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
        mul_513 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_380);  sub_105 = unsqueeze_380 = None
        sub_107 = torch.ops.aten.sub.Tensor(where_12, mul_513);  where_12 = mul_513 = None
        sub_108 = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_377);  sub_107 = unsqueeze_377 = None
        mul_514 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_383);  sub_108 = unsqueeze_383 = None
        mul_515 = torch.ops.aten.mul.Tensor(sum_27, squeeze_133);  sum_27 = squeeze_133 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_514, relu_39, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = primals_45 = None
        getitem_150 = convolution_backward_12[0]
        getitem_151 = convolution_backward_12[1];  convolution_backward_12 = None
        add_306 = torch.ops.aten.add.Tensor(where_10, getitem_150);  where_10 = getitem_150 = None
        le_13 = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_13 = torch.ops.aten.where.self(le_13, scalar_tensor, add_306);  le_13 = add_306 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_109 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_386);  convolution_43 = unsqueeze_386 = None
        mul_516 = torch.ops.aten.mul.Tensor(where_13, sub_109)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
        mul_517 = torch.ops.aten.mul.Tensor(sum_28, 0.001953125)
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
        mul_518 = torch.ops.aten.mul.Tensor(sum_29, 0.001953125)
        mul_519 = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_520 = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
        mul_521 = torch.ops.aten.mul.Tensor(squeeze_130, primals_279);  primals_279 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
        mul_522 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_392);  sub_109 = unsqueeze_392 = None
        sub_111 = torch.ops.aten.sub.Tensor(where_13, mul_522);  mul_522 = None
        sub_112 = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_389);  sub_111 = unsqueeze_389 = None
        mul_523 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_395);  sub_112 = unsqueeze_395 = None
        mul_524 = torch.ops.aten.mul.Tensor(sum_29, squeeze_130);  sum_29 = squeeze_130 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_523, relu_38, primals_44, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_523 = primals_44 = None
        getitem_153 = convolution_backward_13[0]
        getitem_154 = convolution_backward_13[1];  convolution_backward_13 = None
        le_14 = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_14 = torch.ops.aten.where.self(le_14, scalar_tensor, getitem_153);  le_14 = getitem_153 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_113 = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_398);  convolution_42 = unsqueeze_398 = None
        mul_525 = torch.ops.aten.mul.Tensor(where_14, sub_113)
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_525, [0, 2, 3]);  mul_525 = None
        mul_526 = torch.ops.aten.mul.Tensor(sum_30, 0.001953125)
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
        mul_527 = torch.ops.aten.mul.Tensor(sum_31, 0.001953125)
        mul_528 = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_529 = torch.ops.aten.mul.Tensor(mul_527, mul_528);  mul_527 = mul_528 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
        mul_530 = torch.ops.aten.mul.Tensor(squeeze_127, primals_274);  primals_274 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        mul_531 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_404);  sub_113 = unsqueeze_404 = None
        sub_115 = torch.ops.aten.sub.Tensor(where_14, mul_531);  where_14 = mul_531 = None
        sub_116 = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_401);  sub_115 = unsqueeze_401 = None
        mul_532 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_407);  sub_116 = unsqueeze_407 = None
        mul_533 = torch.ops.aten.mul.Tensor(sum_31, squeeze_127);  sum_31 = squeeze_127 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_532, relu_37, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_532 = primals_43 = None
        getitem_156 = convolution_backward_14[0]
        getitem_157 = convolution_backward_14[1];  convolution_backward_14 = None
        le_15 = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_15 = torch.ops.aten.where.self(le_15, scalar_tensor, getitem_156);  le_15 = getitem_156 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_117 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_410);  convolution_41 = unsqueeze_410 = None
        mul_534 = torch.ops.aten.mul.Tensor(where_15, sub_117)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3]);  mul_534 = None
        mul_535 = torch.ops.aten.mul.Tensor(sum_32, 0.001953125)
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
        mul_536 = torch.ops.aten.mul.Tensor(sum_33, 0.001953125)
        mul_537 = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_538 = torch.ops.aten.mul.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
        mul_539 = torch.ops.aten.mul.Tensor(squeeze_124, primals_269);  primals_269 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
        mul_540 = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_416);  sub_117 = unsqueeze_416 = None
        sub_119 = torch.ops.aten.sub.Tensor(where_15, mul_540);  where_15 = mul_540 = None
        sub_120 = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_413);  sub_119 = unsqueeze_413 = None
        mul_541 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_419);  sub_120 = unsqueeze_419 = None
        mul_542 = torch.ops.aten.mul.Tensor(sum_33, squeeze_124);  sum_33 = squeeze_124 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_541, relu_36, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_541 = primals_42 = None
        getitem_159 = convolution_backward_15[0]
        getitem_160 = convolution_backward_15[1];  convolution_backward_15 = None
        add_307 = torch.ops.aten.add.Tensor(where_13, getitem_159);  where_13 = getitem_159 = None
        le_16 = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_16 = torch.ops.aten.where.self(le_16, scalar_tensor, add_307);  le_16 = add_307 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_121 = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_422);  convolution_40 = unsqueeze_422 = None
        mul_543 = torch.ops.aten.mul.Tensor(where_16, sub_121)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_543, [0, 2, 3]);  mul_543 = None
        mul_544 = torch.ops.aten.mul.Tensor(sum_34, 0.001953125)
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
        mul_545 = torch.ops.aten.mul.Tensor(sum_35, 0.001953125)
        mul_546 = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_547 = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
        mul_548 = torch.ops.aten.mul.Tensor(squeeze_121, primals_264);  primals_264 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_549 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_428);  sub_121 = unsqueeze_428 = None
        sub_123 = torch.ops.aten.sub.Tensor(where_16, mul_549);  mul_549 = None
        sub_124 = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_425);  sub_123 = unsqueeze_425 = None
        mul_550 = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_431);  sub_124 = unsqueeze_431 = None
        mul_551 = torch.ops.aten.mul.Tensor(sum_35, squeeze_121);  sum_35 = squeeze_121 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_550, relu_35, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_550 = primals_41 = None
        getitem_162 = convolution_backward_16[0]
        getitem_163 = convolution_backward_16[1];  convolution_backward_16 = None
        le_17 = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_17 = torch.ops.aten.where.self(le_17, scalar_tensor, getitem_162);  le_17 = getitem_162 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_125 = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_434);  convolution_39 = unsqueeze_434 = None
        mul_552 = torch.ops.aten.mul.Tensor(where_17, sub_125)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3]);  mul_552 = None
        mul_553 = torch.ops.aten.mul.Tensor(sum_36, 0.001953125)
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
        mul_554 = torch.ops.aten.mul.Tensor(sum_37, 0.001953125)
        mul_555 = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_556 = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        mul_557 = torch.ops.aten.mul.Tensor(squeeze_118, primals_259);  primals_259 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_558 = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_440);  sub_125 = unsqueeze_440 = None
        sub_127 = torch.ops.aten.sub.Tensor(where_17, mul_558);  where_17 = mul_558 = None
        sub_128 = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_437);  sub_127 = unsqueeze_437 = None
        mul_559 = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_443);  sub_128 = unsqueeze_443 = None
        mul_560 = torch.ops.aten.mul.Tensor(sum_37, squeeze_118);  sum_37 = squeeze_118 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_559, relu_34, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_559 = primals_40 = None
        getitem_165 = convolution_backward_17[0]
        getitem_166 = convolution_backward_17[1];  convolution_backward_17 = None
        le_18 = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_18 = torch.ops.aten.where.self(le_18, scalar_tensor, getitem_165);  le_18 = getitem_165 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_129 = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_446);  convolution_38 = unsqueeze_446 = None
        mul_561 = torch.ops.aten.mul.Tensor(where_18, sub_129)
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3]);  mul_561 = None
        mul_562 = torch.ops.aten.mul.Tensor(sum_38, 0.001953125)
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
        mul_563 = torch.ops.aten.mul.Tensor(sum_39, 0.001953125)
        mul_564 = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_565 = torch.ops.aten.mul.Tensor(mul_563, mul_564);  mul_563 = mul_564 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
        mul_566 = torch.ops.aten.mul.Tensor(squeeze_115, primals_254);  primals_254 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
        mul_567 = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_452);  sub_129 = unsqueeze_452 = None
        sub_131 = torch.ops.aten.sub.Tensor(where_18, mul_567);  where_18 = mul_567 = None
        sub_132 = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_449);  sub_131 = unsqueeze_449 = None
        mul_568 = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_455);  sub_132 = unsqueeze_455 = None
        mul_569 = torch.ops.aten.mul.Tensor(sum_39, squeeze_115);  sum_39 = squeeze_115 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_568, relu_33, primals_39, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_568 = primals_39 = None
        getitem_168 = convolution_backward_18[0]
        getitem_169 = convolution_backward_18[1];  convolution_backward_18 = None
        add_308 = torch.ops.aten.add.Tensor(where_16, getitem_168);  where_16 = getitem_168 = None
        le_19 = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_19 = torch.ops.aten.where.self(le_19, scalar_tensor, add_308);  le_19 = add_308 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_133 = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_458);  convolution_37 = unsqueeze_458 = None
        mul_570 = torch.ops.aten.mul.Tensor(where_19, sub_133)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_570, [0, 2, 3]);  mul_570 = None
        mul_571 = torch.ops.aten.mul.Tensor(sum_40, 0.001953125)
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
        mul_572 = torch.ops.aten.mul.Tensor(sum_41, 0.001953125)
        mul_573 = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_574 = torch.ops.aten.mul.Tensor(mul_572, mul_573);  mul_572 = mul_573 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
        mul_575 = torch.ops.aten.mul.Tensor(squeeze_112, primals_249);  primals_249 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
        mul_576 = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_464);  sub_133 = unsqueeze_464 = None
        sub_135 = torch.ops.aten.sub.Tensor(where_19, mul_576);  mul_576 = None
        sub_136 = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_461);  sub_135 = unsqueeze_461 = None
        mul_577 = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_467);  sub_136 = unsqueeze_467 = None
        mul_578 = torch.ops.aten.mul.Tensor(sum_41, squeeze_112);  sum_41 = squeeze_112 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_577, relu_32, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_577 = primals_38 = None
        getitem_171 = convolution_backward_19[0]
        getitem_172 = convolution_backward_19[1];  convolution_backward_19 = None
        le_20 = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_20 = torch.ops.aten.where.self(le_20, scalar_tensor, getitem_171);  le_20 = getitem_171 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_137 = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_470);  convolution_36 = unsqueeze_470 = None
        mul_579 = torch.ops.aten.mul.Tensor(where_20, sub_137)
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_579, [0, 2, 3]);  mul_579 = None
        mul_580 = torch.ops.aten.mul.Tensor(sum_42, 0.001953125)
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
        mul_581 = torch.ops.aten.mul.Tensor(sum_43, 0.001953125)
        mul_582 = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_583 = torch.ops.aten.mul.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
        mul_584 = torch.ops.aten.mul.Tensor(squeeze_109, primals_244);  primals_244 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
        mul_585 = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_476);  sub_137 = unsqueeze_476 = None
        sub_139 = torch.ops.aten.sub.Tensor(where_20, mul_585);  where_20 = mul_585 = None
        sub_140 = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_473);  sub_139 = unsqueeze_473 = None
        mul_586 = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_479);  sub_140 = unsqueeze_479 = None
        mul_587 = torch.ops.aten.mul.Tensor(sum_43, squeeze_109);  sum_43 = squeeze_109 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_586, relu_31, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_586 = primals_37 = None
        getitem_174 = convolution_backward_20[0]
        getitem_175 = convolution_backward_20[1];  convolution_backward_20 = None
        le_21 = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_21 = torch.ops.aten.where.self(le_21, scalar_tensor, getitem_174);  le_21 = getitem_174 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_141 = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_482);  convolution_35 = unsqueeze_482 = None
        mul_588 = torch.ops.aten.mul.Tensor(where_21, sub_141)
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_588, [0, 2, 3]);  mul_588 = None
        mul_589 = torch.ops.aten.mul.Tensor(sum_44, 0.001953125)
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
        mul_590 = torch.ops.aten.mul.Tensor(sum_45, 0.001953125)
        mul_591 = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_592 = torch.ops.aten.mul.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
        mul_593 = torch.ops.aten.mul.Tensor(squeeze_106, primals_239);  primals_239 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
        mul_594 = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_488);  sub_141 = unsqueeze_488 = None
        sub_143 = torch.ops.aten.sub.Tensor(where_21, mul_594);  where_21 = mul_594 = None
        sub_144 = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_485);  sub_143 = unsqueeze_485 = None
        mul_595 = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_491);  sub_144 = unsqueeze_491 = None
        mul_596 = torch.ops.aten.mul.Tensor(sum_45, squeeze_106);  sum_45 = squeeze_106 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_595, relu_30, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_595 = primals_36 = None
        getitem_177 = convolution_backward_21[0]
        getitem_178 = convolution_backward_21[1];  convolution_backward_21 = None
        add_309 = torch.ops.aten.add.Tensor(where_19, getitem_177);  where_19 = getitem_177 = None
        le_22 = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_22 = torch.ops.aten.where.self(le_22, scalar_tensor, add_309);  le_22 = add_309 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_145 = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_494);  convolution_34 = unsqueeze_494 = None
        mul_597 = torch.ops.aten.mul.Tensor(where_22, sub_145)
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
        mul_598 = torch.ops.aten.mul.Tensor(sum_46, 0.001953125)
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
        mul_599 = torch.ops.aten.mul.Tensor(sum_47, 0.001953125)
        mul_600 = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_601 = torch.ops.aten.mul.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
        mul_602 = torch.ops.aten.mul.Tensor(squeeze_103, primals_234);  primals_234 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
        mul_603 = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_500);  sub_145 = unsqueeze_500 = None
        sub_147 = torch.ops.aten.sub.Tensor(where_22, mul_603);  mul_603 = None
        sub_148 = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_497);  sub_147 = unsqueeze_497 = None
        mul_604 = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_503);  sub_148 = unsqueeze_503 = None
        mul_605 = torch.ops.aten.mul.Tensor(sum_47, squeeze_103);  sum_47 = squeeze_103 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_604, relu_29, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_604 = primals_35 = None
        getitem_180 = convolution_backward_22[0]
        getitem_181 = convolution_backward_22[1];  convolution_backward_22 = None
        le_23 = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_23 = torch.ops.aten.where.self(le_23, scalar_tensor, getitem_180);  le_23 = getitem_180 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_149 = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_506);  convolution_33 = unsqueeze_506 = None
        mul_606 = torch.ops.aten.mul.Tensor(where_23, sub_149)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2, 3]);  mul_606 = None
        mul_607 = torch.ops.aten.mul.Tensor(sum_48, 0.001953125)
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
        mul_608 = torch.ops.aten.mul.Tensor(sum_49, 0.001953125)
        mul_609 = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_610 = torch.ops.aten.mul.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
        mul_611 = torch.ops.aten.mul.Tensor(squeeze_100, primals_229);  primals_229 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
        mul_612 = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_512);  sub_149 = unsqueeze_512 = None
        sub_151 = torch.ops.aten.sub.Tensor(where_23, mul_612);  where_23 = mul_612 = None
        sub_152 = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_509);  sub_151 = unsqueeze_509 = None
        mul_613 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_515);  sub_152 = unsqueeze_515 = None
        mul_614 = torch.ops.aten.mul.Tensor(sum_49, squeeze_100);  sum_49 = squeeze_100 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_613, relu_28, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_613 = primals_34 = None
        getitem_183 = convolution_backward_23[0]
        getitem_184 = convolution_backward_23[1];  convolution_backward_23 = None
        le_24 = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_24 = torch.ops.aten.where.self(le_24, scalar_tensor, getitem_183);  le_24 = getitem_183 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_153 = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_518);  convolution_32 = unsqueeze_518 = None
        mul_615 = torch.ops.aten.mul.Tensor(where_24, sub_153)
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_615, [0, 2, 3]);  mul_615 = None
        mul_616 = torch.ops.aten.mul.Tensor(sum_50, 0.001953125)
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
        mul_617 = torch.ops.aten.mul.Tensor(sum_51, 0.001953125)
        mul_618 = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_619 = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
        mul_620 = torch.ops.aten.mul.Tensor(squeeze_97, primals_224);  primals_224 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
        mul_621 = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_524);  sub_153 = unsqueeze_524 = None
        sub_155 = torch.ops.aten.sub.Tensor(where_24, mul_621);  where_24 = mul_621 = None
        sub_156 = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_521);  sub_155 = unsqueeze_521 = None
        mul_622 = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_527);  sub_156 = unsqueeze_527 = None
        mul_623 = torch.ops.aten.mul.Tensor(sum_51, squeeze_97);  sum_51 = squeeze_97 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_622, relu_27, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_622 = primals_33 = None
        getitem_186 = convolution_backward_24[0]
        getitem_187 = convolution_backward_24[1];  convolution_backward_24 = None
        add_310 = torch.ops.aten.add.Tensor(where_22, getitem_186);  where_22 = getitem_186 = None
        le_25 = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_25 = torch.ops.aten.where.self(le_25, scalar_tensor, add_310);  le_25 = add_310 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_157 = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_530);  convolution_31 = unsqueeze_530 = None
        mul_624 = torch.ops.aten.mul.Tensor(where_25, sub_157)
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_624, [0, 2, 3]);  mul_624 = None
        mul_625 = torch.ops.aten.mul.Tensor(sum_52, 0.001953125)
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
        mul_626 = torch.ops.aten.mul.Tensor(sum_53, 0.001953125)
        mul_627 = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_628 = torch.ops.aten.mul.Tensor(mul_626, mul_627);  mul_626 = mul_627 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
        mul_629 = torch.ops.aten.mul.Tensor(squeeze_94, primals_219);  primals_219 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
        mul_630 = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_536);  sub_157 = unsqueeze_536 = None
        sub_159 = torch.ops.aten.sub.Tensor(where_25, mul_630);  mul_630 = None
        sub_160 = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_533);  sub_159 = None
        mul_631 = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_539);  sub_160 = unsqueeze_539 = None
        mul_632 = torch.ops.aten.mul.Tensor(sum_53, squeeze_94);  sum_53 = squeeze_94 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_631, relu_24, primals_32, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_631 = primals_32 = None
        getitem_189 = convolution_backward_25[0]
        getitem_190 = convolution_backward_25[1];  convolution_backward_25 = None
        sub_161 = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_542);  convolution_30 = unsqueeze_542 = None
        mul_633 = torch.ops.aten.mul.Tensor(where_25, sub_161)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3]);  mul_633 = None
        mul_635 = torch.ops.aten.mul.Tensor(sum_55, 0.001953125)
        mul_636 = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_637 = torch.ops.aten.mul.Tensor(mul_635, mul_636);  mul_635 = mul_636 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
        mul_638 = torch.ops.aten.mul.Tensor(squeeze_91, primals_214);  primals_214 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
        mul_639 = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_548);  sub_161 = unsqueeze_548 = None
        sub_163 = torch.ops.aten.sub.Tensor(where_25, mul_639);  where_25 = mul_639 = None
        sub_164 = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_533);  sub_163 = unsqueeze_533 = None
        mul_640 = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_551);  sub_164 = unsqueeze_551 = None
        mul_641 = torch.ops.aten.mul.Tensor(sum_55, squeeze_91);  sum_55 = squeeze_91 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_640, relu_26, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_640 = primals_31 = None
        getitem_192 = convolution_backward_26[0]
        getitem_193 = convolution_backward_26[1];  convolution_backward_26 = None
        le_26 = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_26 = torch.ops.aten.where.self(le_26, scalar_tensor, getitem_192);  le_26 = getitem_192 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_165 = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_554);  convolution_29 = unsqueeze_554 = None
        mul_642 = torch.ops.aten.mul.Tensor(where_26, sub_165)
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3]);  mul_642 = None
        mul_643 = torch.ops.aten.mul.Tensor(sum_56, 0.001953125)
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
        mul_644 = torch.ops.aten.mul.Tensor(sum_57, 0.001953125)
        mul_645 = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_646 = torch.ops.aten.mul.Tensor(mul_644, mul_645);  mul_644 = mul_645 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
        mul_647 = torch.ops.aten.mul.Tensor(squeeze_88, primals_209);  primals_209 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
        mul_648 = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_560);  sub_165 = unsqueeze_560 = None
        sub_167 = torch.ops.aten.sub.Tensor(where_26, mul_648);  where_26 = mul_648 = None
        sub_168 = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_557);  sub_167 = unsqueeze_557 = None
        mul_649 = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_563);  sub_168 = unsqueeze_563 = None
        mul_650 = torch.ops.aten.mul.Tensor(sum_57, squeeze_88);  sum_57 = squeeze_88 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_649, relu_25, primals_30, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_649 = primals_30 = None
        getitem_195 = convolution_backward_27[0]
        getitem_196 = convolution_backward_27[1];  convolution_backward_27 = None
        le_27 = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_27 = torch.ops.aten.where.self(le_27, scalar_tensor, getitem_195);  le_27 = getitem_195 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_169 = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_566);  convolution_28 = unsqueeze_566 = None
        mul_651 = torch.ops.aten.mul.Tensor(where_27, sub_169)
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
        mul_652 = torch.ops.aten.mul.Tensor(sum_58, 0.00048828125)
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
        mul_653 = torch.ops.aten.mul.Tensor(sum_59, 0.00048828125)
        mul_654 = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_655 = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
        mul_656 = torch.ops.aten.mul.Tensor(squeeze_85, primals_204);  primals_204 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
        mul_657 = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_572);  sub_169 = unsqueeze_572 = None
        sub_171 = torch.ops.aten.sub.Tensor(where_27, mul_657);  where_27 = mul_657 = None
        sub_172 = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_569);  sub_171 = unsqueeze_569 = None
        mul_658 = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_575);  sub_172 = unsqueeze_575 = None
        mul_659 = torch.ops.aten.mul.Tensor(sum_59, squeeze_85);  sum_59 = squeeze_85 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_658, relu_24, primals_29, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_658 = primals_29 = None
        getitem_198 = convolution_backward_28[0]
        getitem_199 = convolution_backward_28[1];  convolution_backward_28 = None
        add_311 = torch.ops.aten.add.Tensor(getitem_189, getitem_198);  getitem_189 = getitem_198 = None
        le_28 = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_28 = torch.ops.aten.where.self(le_28, scalar_tensor, add_311);  le_28 = add_311 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_173 = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_578);  convolution_27 = unsqueeze_578 = None
        mul_660 = torch.ops.aten.mul.Tensor(where_28, sub_173)
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_660, [0, 2, 3]);  mul_660 = None
        mul_661 = torch.ops.aten.mul.Tensor(sum_60, 0.00048828125)
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
        mul_662 = torch.ops.aten.mul.Tensor(sum_61, 0.00048828125)
        mul_663 = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_664 = torch.ops.aten.mul.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
        mul_665 = torch.ops.aten.mul.Tensor(squeeze_82, primals_199);  primals_199 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
        mul_666 = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_584);  sub_173 = unsqueeze_584 = None
        sub_175 = torch.ops.aten.sub.Tensor(where_28, mul_666);  mul_666 = None
        sub_176 = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_581);  sub_175 = unsqueeze_581 = None
        mul_667 = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_587);  sub_176 = unsqueeze_587 = None
        mul_668 = torch.ops.aten.mul.Tensor(sum_61, squeeze_82);  sum_61 = squeeze_82 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_667, relu_23, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = primals_28 = None
        getitem_201 = convolution_backward_29[0]
        getitem_202 = convolution_backward_29[1];  convolution_backward_29 = None
        le_29 = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_29 = torch.ops.aten.where.self(le_29, scalar_tensor, getitem_201);  le_29 = getitem_201 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_177 = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_590);  convolution_26 = unsqueeze_590 = None
        mul_669 = torch.ops.aten.mul.Tensor(where_29, sub_177)
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_669, [0, 2, 3]);  mul_669 = None
        mul_670 = torch.ops.aten.mul.Tensor(sum_62, 0.00048828125)
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(mul_670, 0);  mul_670 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
        mul_671 = torch.ops.aten.mul.Tensor(sum_63, 0.00048828125)
        mul_672 = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_673 = torch.ops.aten.mul.Tensor(mul_671, mul_672);  mul_671 = mul_672 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
        mul_674 = torch.ops.aten.mul.Tensor(squeeze_79, primals_194);  primals_194 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
        mul_675 = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_596);  sub_177 = unsqueeze_596 = None
        sub_179 = torch.ops.aten.sub.Tensor(where_29, mul_675);  where_29 = mul_675 = None
        sub_180 = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_593);  sub_179 = unsqueeze_593 = None
        mul_676 = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_599);  sub_180 = unsqueeze_599 = None
        mul_677 = torch.ops.aten.mul.Tensor(sum_63, squeeze_79);  sum_63 = squeeze_79 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_676, relu_22, primals_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_676 = primals_27 = None
        getitem_204 = convolution_backward_30[0]
        getitem_205 = convolution_backward_30[1];  convolution_backward_30 = None
        le_30 = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_30 = torch.ops.aten.where.self(le_30, scalar_tensor, getitem_204);  le_30 = getitem_204 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_181 = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_602);  convolution_25 = unsqueeze_602 = None
        mul_678 = torch.ops.aten.mul.Tensor(where_30, sub_181)
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_678, [0, 2, 3]);  mul_678 = None
        mul_679 = torch.ops.aten.mul.Tensor(sum_64, 0.00048828125)
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
        mul_680 = torch.ops.aten.mul.Tensor(sum_65, 0.00048828125)
        mul_681 = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_682 = torch.ops.aten.mul.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
        mul_683 = torch.ops.aten.mul.Tensor(squeeze_76, primals_189);  primals_189 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
        mul_684 = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_608);  sub_181 = unsqueeze_608 = None
        sub_183 = torch.ops.aten.sub.Tensor(where_30, mul_684);  where_30 = mul_684 = None
        sub_184 = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_605);  sub_183 = unsqueeze_605 = None
        mul_685 = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_611);  sub_184 = unsqueeze_611 = None
        mul_686 = torch.ops.aten.mul.Tensor(sum_65, squeeze_76);  sum_65 = squeeze_76 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_685, relu_21, primals_26, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_685 = primals_26 = None
        getitem_207 = convolution_backward_31[0]
        getitem_208 = convolution_backward_31[1];  convolution_backward_31 = None
        add_312 = torch.ops.aten.add.Tensor(where_28, getitem_207);  where_28 = getitem_207 = None
        le_31 = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_31 = torch.ops.aten.where.self(le_31, scalar_tensor, add_312);  le_31 = add_312 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_185 = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_614);  convolution_24 = unsqueeze_614 = None
        mul_687 = torch.ops.aten.mul.Tensor(where_31, sub_185)
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_687, [0, 2, 3]);  mul_687 = None
        mul_688 = torch.ops.aten.mul.Tensor(sum_66, 0.00048828125)
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
        mul_689 = torch.ops.aten.mul.Tensor(sum_67, 0.00048828125)
        mul_690 = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_691 = torch.ops.aten.mul.Tensor(mul_689, mul_690);  mul_689 = mul_690 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
        mul_692 = torch.ops.aten.mul.Tensor(squeeze_73, primals_184);  primals_184 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
        mul_693 = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_620);  sub_185 = unsqueeze_620 = None
        sub_187 = torch.ops.aten.sub.Tensor(where_31, mul_693);  mul_693 = None
        sub_188 = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_617);  sub_187 = unsqueeze_617 = None
        mul_694 = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_623);  sub_188 = unsqueeze_623 = None
        mul_695 = torch.ops.aten.mul.Tensor(sum_67, squeeze_73);  sum_67 = squeeze_73 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_694, relu_20, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_694 = primals_25 = None
        getitem_210 = convolution_backward_32[0]
        getitem_211 = convolution_backward_32[1];  convolution_backward_32 = None
        le_32 = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_32 = torch.ops.aten.where.self(le_32, scalar_tensor, getitem_210);  le_32 = getitem_210 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_189 = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_626);  convolution_23 = unsqueeze_626 = None
        mul_696 = torch.ops.aten.mul.Tensor(where_32, sub_189)
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_696, [0, 2, 3]);  mul_696 = None
        mul_697 = torch.ops.aten.mul.Tensor(sum_68, 0.00048828125)
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
        mul_698 = torch.ops.aten.mul.Tensor(sum_69, 0.00048828125)
        mul_699 = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_700 = torch.ops.aten.mul.Tensor(mul_698, mul_699);  mul_698 = mul_699 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
        mul_701 = torch.ops.aten.mul.Tensor(squeeze_70, primals_179);  primals_179 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
        mul_702 = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_632);  sub_189 = unsqueeze_632 = None
        sub_191 = torch.ops.aten.sub.Tensor(where_32, mul_702);  where_32 = mul_702 = None
        sub_192 = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_629);  sub_191 = unsqueeze_629 = None
        mul_703 = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_635);  sub_192 = unsqueeze_635 = None
        mul_704 = torch.ops.aten.mul.Tensor(sum_69, squeeze_70);  sum_69 = squeeze_70 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_703, relu_19, primals_24, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_703 = primals_24 = None
        getitem_213 = convolution_backward_33[0]
        getitem_214 = convolution_backward_33[1];  convolution_backward_33 = None
        le_33 = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_33 = torch.ops.aten.where.self(le_33, scalar_tensor, getitem_213);  le_33 = getitem_213 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_193 = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_638);  convolution_22 = unsqueeze_638 = None
        mul_705 = torch.ops.aten.mul.Tensor(where_33, sub_193)
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_705, [0, 2, 3]);  mul_705 = None
        mul_706 = torch.ops.aten.mul.Tensor(sum_70, 0.00048828125)
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
        mul_707 = torch.ops.aten.mul.Tensor(sum_71, 0.00048828125)
        mul_708 = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_709 = torch.ops.aten.mul.Tensor(mul_707, mul_708);  mul_707 = mul_708 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
        mul_710 = torch.ops.aten.mul.Tensor(squeeze_67, primals_174);  primals_174 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
        mul_711 = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_644);  sub_193 = unsqueeze_644 = None
        sub_195 = torch.ops.aten.sub.Tensor(where_33, mul_711);  where_33 = mul_711 = None
        sub_196 = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_641);  sub_195 = unsqueeze_641 = None
        mul_712 = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_647);  sub_196 = unsqueeze_647 = None
        mul_713 = torch.ops.aten.mul.Tensor(sum_71, squeeze_67);  sum_71 = squeeze_67 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_712, relu_18, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_712 = primals_23 = None
        getitem_216 = convolution_backward_34[0]
        getitem_217 = convolution_backward_34[1];  convolution_backward_34 = None
        add_313 = torch.ops.aten.add.Tensor(where_31, getitem_216);  where_31 = getitem_216 = None
        le_34 = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_34 = torch.ops.aten.where.self(le_34, scalar_tensor, add_313);  le_34 = add_313 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_197 = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_650);  convolution_21 = unsqueeze_650 = None
        mul_714 = torch.ops.aten.mul.Tensor(where_34, sub_197)
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_714, [0, 2, 3]);  mul_714 = None
        mul_715 = torch.ops.aten.mul.Tensor(sum_72, 0.00048828125)
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
        mul_716 = torch.ops.aten.mul.Tensor(sum_73, 0.00048828125)
        mul_717 = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_718 = torch.ops.aten.mul.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
        mul_719 = torch.ops.aten.mul.Tensor(squeeze_64, primals_169);  primals_169 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
        mul_720 = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_656);  sub_197 = unsqueeze_656 = None
        sub_199 = torch.ops.aten.sub.Tensor(where_34, mul_720);  mul_720 = None
        sub_200 = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_653);  sub_199 = unsqueeze_653 = None
        mul_721 = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_659);  sub_200 = unsqueeze_659 = None
        mul_722 = torch.ops.aten.mul.Tensor(sum_73, squeeze_64);  sum_73 = squeeze_64 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_721, relu_17, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_721 = primals_22 = None
        getitem_219 = convolution_backward_35[0]
        getitem_220 = convolution_backward_35[1];  convolution_backward_35 = None
        le_35 = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_35 = torch.ops.aten.where.self(le_35, scalar_tensor, getitem_219);  le_35 = getitem_219 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        sub_201 = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_662);  convolution_20 = unsqueeze_662 = None
        mul_723 = torch.ops.aten.mul.Tensor(where_35, sub_201)
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_723, [0, 2, 3]);  mul_723 = None
        mul_724 = torch.ops.aten.mul.Tensor(sum_74, 0.00048828125)
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
        mul_725 = torch.ops.aten.mul.Tensor(sum_75, 0.00048828125)
        mul_726 = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_727 = torch.ops.aten.mul.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
        mul_728 = torch.ops.aten.mul.Tensor(squeeze_61, primals_164);  primals_164 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
        mul_729 = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_668);  sub_201 = unsqueeze_668 = None
        sub_203 = torch.ops.aten.sub.Tensor(where_35, mul_729);  where_35 = mul_729 = None
        sub_204 = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_665);  sub_203 = unsqueeze_665 = None
        mul_730 = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_671);  sub_204 = unsqueeze_671 = None
        mul_731 = torch.ops.aten.mul.Tensor(sum_75, squeeze_61);  sum_75 = squeeze_61 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_730, relu_16, primals_21, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_730 = primals_21 = None
        getitem_222 = convolution_backward_36[0]
        getitem_223 = convolution_backward_36[1];  convolution_backward_36 = None
        le_36 = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_36 = torch.ops.aten.where.self(le_36, scalar_tensor, getitem_222);  le_36 = getitem_222 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_205 = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_674);  convolution_19 = unsqueeze_674 = None
        mul_732 = torch.ops.aten.mul.Tensor(where_36, sub_205)
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3]);  mul_732 = None
        mul_733 = torch.ops.aten.mul.Tensor(sum_76, 0.00048828125)
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
        mul_734 = torch.ops.aten.mul.Tensor(sum_77, 0.00048828125)
        mul_735 = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_736 = torch.ops.aten.mul.Tensor(mul_734, mul_735);  mul_734 = mul_735 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
        mul_737 = torch.ops.aten.mul.Tensor(squeeze_58, primals_159);  primals_159 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
        mul_738 = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_680);  sub_205 = unsqueeze_680 = None
        sub_207 = torch.ops.aten.sub.Tensor(where_36, mul_738);  where_36 = mul_738 = None
        sub_208 = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_677);  sub_207 = unsqueeze_677 = None
        mul_739 = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_683);  sub_208 = unsqueeze_683 = None
        mul_740 = torch.ops.aten.mul.Tensor(sum_77, squeeze_58);  sum_77 = squeeze_58 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_739, relu_15, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_739 = primals_20 = None
        getitem_225 = convolution_backward_37[0]
        getitem_226 = convolution_backward_37[1];  convolution_backward_37 = None
        add_314 = torch.ops.aten.add.Tensor(where_34, getitem_225);  where_34 = getitem_225 = None
        le_37 = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_37 = torch.ops.aten.where.self(le_37, scalar_tensor, add_314);  le_37 = add_314 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        sub_209 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_686);  convolution_18 = unsqueeze_686 = None
        mul_741 = torch.ops.aten.mul.Tensor(where_37, sub_209)
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3]);  mul_741 = None
        mul_742 = torch.ops.aten.mul.Tensor(sum_78, 0.00048828125)
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
        mul_743 = torch.ops.aten.mul.Tensor(sum_79, 0.00048828125)
        mul_744 = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_745 = torch.ops.aten.mul.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
        mul_746 = torch.ops.aten.mul.Tensor(squeeze_55, primals_154);  primals_154 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
        mul_747 = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_692);  sub_209 = unsqueeze_692 = None
        sub_211 = torch.ops.aten.sub.Tensor(where_37, mul_747);  mul_747 = None
        sub_212 = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_689);  sub_211 = unsqueeze_689 = None
        mul_748 = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_695);  sub_212 = unsqueeze_695 = None
        mul_749 = torch.ops.aten.mul.Tensor(sum_79, squeeze_55);  sum_79 = squeeze_55 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_748, relu_14, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_748 = primals_19 = None
        getitem_228 = convolution_backward_38[0]
        getitem_229 = convolution_backward_38[1];  convolution_backward_38 = None
        le_38 = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_38 = torch.ops.aten.where.self(le_38, scalar_tensor, getitem_228);  le_38 = getitem_228 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
        sub_213 = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_698);  convolution_17 = unsqueeze_698 = None
        mul_750 = torch.ops.aten.mul.Tensor(where_38, sub_213)
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3]);  mul_750 = None
        mul_751 = torch.ops.aten.mul.Tensor(sum_80, 0.00048828125)
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
        mul_752 = torch.ops.aten.mul.Tensor(sum_81, 0.00048828125)
        mul_753 = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_754 = torch.ops.aten.mul.Tensor(mul_752, mul_753);  mul_752 = mul_753 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
        mul_755 = torch.ops.aten.mul.Tensor(squeeze_52, primals_149);  primals_149 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
        mul_756 = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_704);  sub_213 = unsqueeze_704 = None
        sub_215 = torch.ops.aten.sub.Tensor(where_38, mul_756);  where_38 = mul_756 = None
        sub_216 = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_701);  sub_215 = unsqueeze_701 = None
        mul_757 = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_707);  sub_216 = unsqueeze_707 = None
        mul_758 = torch.ops.aten.mul.Tensor(sum_81, squeeze_52);  sum_81 = squeeze_52 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_757, relu_13, primals_18, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_757 = primals_18 = None
        getitem_231 = convolution_backward_39[0]
        getitem_232 = convolution_backward_39[1];  convolution_backward_39 = None
        le_39 = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_39 = torch.ops.aten.where.self(le_39, scalar_tensor, getitem_231);  le_39 = getitem_231 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
        sub_217 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_710);  convolution_16 = unsqueeze_710 = None
        mul_759 = torch.ops.aten.mul.Tensor(where_39, sub_217)
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
        mul_760 = torch.ops.aten.mul.Tensor(sum_82, 0.00048828125)
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
        mul_761 = torch.ops.aten.mul.Tensor(sum_83, 0.00048828125)
        mul_762 = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_763 = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
        mul_764 = torch.ops.aten.mul.Tensor(squeeze_49, primals_144);  primals_144 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
        mul_765 = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_716);  sub_217 = unsqueeze_716 = None
        sub_219 = torch.ops.aten.sub.Tensor(where_39, mul_765);  where_39 = mul_765 = None
        sub_220 = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_713);  sub_219 = unsqueeze_713 = None
        mul_766 = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_719);  sub_220 = unsqueeze_719 = None
        mul_767 = torch.ops.aten.mul.Tensor(sum_83, squeeze_49);  sum_83 = squeeze_49 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_766, relu_12, primals_17, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_766 = primals_17 = None
        getitem_234 = convolution_backward_40[0]
        getitem_235 = convolution_backward_40[1];  convolution_backward_40 = None
        add_315 = torch.ops.aten.add.Tensor(where_37, getitem_234);  where_37 = getitem_234 = None
        le_40 = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_40 = torch.ops.aten.where.self(le_40, scalar_tensor, add_315);  le_40 = add_315 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
        sub_221 = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_722);  convolution_15 = unsqueeze_722 = None
        mul_768 = torch.ops.aten.mul.Tensor(where_40, sub_221)
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_768, [0, 2, 3]);  mul_768 = None
        mul_769 = torch.ops.aten.mul.Tensor(sum_84, 0.00048828125)
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
        mul_770 = torch.ops.aten.mul.Tensor(sum_85, 0.00048828125)
        mul_771 = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_772 = torch.ops.aten.mul.Tensor(mul_770, mul_771);  mul_770 = mul_771 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
        mul_773 = torch.ops.aten.mul.Tensor(squeeze_46, primals_139);  primals_139 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
        mul_774 = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_728);  sub_221 = unsqueeze_728 = None
        sub_223 = torch.ops.aten.sub.Tensor(where_40, mul_774);  mul_774 = None
        sub_224 = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_725);  sub_223 = unsqueeze_725 = None
        mul_775 = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_731);  sub_224 = unsqueeze_731 = None
        mul_776 = torch.ops.aten.mul.Tensor(sum_85, squeeze_46);  sum_85 = squeeze_46 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_775, relu_11, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_775 = primals_16 = None
        getitem_237 = convolution_backward_41[0]
        getitem_238 = convolution_backward_41[1];  convolution_backward_41 = None
        le_41 = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_41 = torch.ops.aten.where.self(le_41, scalar_tensor, getitem_237);  le_41 = getitem_237 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
        sub_225 = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_734);  convolution_14 = unsqueeze_734 = None
        mul_777 = torch.ops.aten.mul.Tensor(where_41, sub_225)
        sum_87 = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3]);  mul_777 = None
        mul_778 = torch.ops.aten.mul.Tensor(sum_86, 0.00048828125)
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
        mul_779 = torch.ops.aten.mul.Tensor(sum_87, 0.00048828125)
        mul_780 = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_781 = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
        mul_782 = torch.ops.aten.mul.Tensor(squeeze_43, primals_134);  primals_134 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
        mul_783 = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_740);  sub_225 = unsqueeze_740 = None
        sub_227 = torch.ops.aten.sub.Tensor(where_41, mul_783);  where_41 = mul_783 = None
        sub_228 = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_737);  sub_227 = unsqueeze_737 = None
        mul_784 = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_743);  sub_228 = unsqueeze_743 = None
        mul_785 = torch.ops.aten.mul.Tensor(sum_87, squeeze_43);  sum_87 = squeeze_43 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_784, relu_10, primals_15, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = primals_15 = None
        getitem_240 = convolution_backward_42[0]
        getitem_241 = convolution_backward_42[1];  convolution_backward_42 = None
        le_42 = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_42 = torch.ops.aten.where.self(le_42, scalar_tensor, getitem_240);  le_42 = getitem_240 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
        sub_229 = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_746);  convolution_13 = unsqueeze_746 = None
        mul_786 = torch.ops.aten.mul.Tensor(where_42, sub_229)
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_786, [0, 2, 3]);  mul_786 = None
        mul_787 = torch.ops.aten.mul.Tensor(sum_88, 0.00048828125)
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
        mul_788 = torch.ops.aten.mul.Tensor(sum_89, 0.00048828125)
        mul_789 = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_790 = torch.ops.aten.mul.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
        mul_791 = torch.ops.aten.mul.Tensor(squeeze_40, primals_129);  primals_129 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
        mul_792 = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_752);  sub_229 = unsqueeze_752 = None
        sub_231 = torch.ops.aten.sub.Tensor(where_42, mul_792);  where_42 = mul_792 = None
        sub_232 = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_749);  sub_231 = unsqueeze_749 = None
        mul_793 = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_755);  sub_232 = unsqueeze_755 = None
        mul_794 = torch.ops.aten.mul.Tensor(sum_89, squeeze_40);  sum_89 = squeeze_40 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_793, relu_9, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_793 = primals_14 = None
        getitem_243 = convolution_backward_43[0]
        getitem_244 = convolution_backward_43[1];  convolution_backward_43 = None
        add_316 = torch.ops.aten.add.Tensor(where_40, getitem_243);  where_40 = getitem_243 = None
        le_43 = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_43 = torch.ops.aten.where.self(le_43, scalar_tensor, add_316);  le_43 = add_316 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
        sub_233 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_758);  convolution_12 = unsqueeze_758 = None
        mul_795 = torch.ops.aten.mul.Tensor(where_43, sub_233)
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3]);  mul_795 = None
        mul_796 = torch.ops.aten.mul.Tensor(sum_90, 0.00048828125)
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
        mul_797 = torch.ops.aten.mul.Tensor(sum_91, 0.00048828125)
        mul_798 = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_799 = torch.ops.aten.mul.Tensor(mul_797, mul_798);  mul_797 = mul_798 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
        mul_800 = torch.ops.aten.mul.Tensor(squeeze_37, primals_124);  primals_124 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
        mul_801 = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_764);  sub_233 = unsqueeze_764 = None
        sub_235 = torch.ops.aten.sub.Tensor(where_43, mul_801);  mul_801 = None
        sub_236 = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_761);  sub_235 = None
        mul_802 = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_767);  sub_236 = unsqueeze_767 = None
        mul_803 = torch.ops.aten.mul.Tensor(sum_91, squeeze_37);  sum_91 = squeeze_37 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_802, relu_6, primals_13, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_802 = primals_13 = None
        getitem_246 = convolution_backward_44[0]
        getitem_247 = convolution_backward_44[1];  convolution_backward_44 = None
        sub_237 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_770);  convolution_11 = unsqueeze_770 = None
        mul_804 = torch.ops.aten.mul.Tensor(where_43, sub_237)
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_804, [0, 2, 3]);  mul_804 = None
        mul_806 = torch.ops.aten.mul.Tensor(sum_93, 0.00048828125)
        mul_807 = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_808 = torch.ops.aten.mul.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
        mul_809 = torch.ops.aten.mul.Tensor(squeeze_34, primals_119);  primals_119 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
        mul_810 = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_776);  sub_237 = unsqueeze_776 = None
        sub_239 = torch.ops.aten.sub.Tensor(where_43, mul_810);  where_43 = mul_810 = None
        sub_240 = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_761);  sub_239 = unsqueeze_761 = None
        mul_811 = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_779);  sub_240 = unsqueeze_779 = None
        mul_812 = torch.ops.aten.mul.Tensor(sum_93, squeeze_34);  sum_93 = squeeze_34 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_811, relu_8, primals_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_811 = primals_12 = None
        getitem_249 = convolution_backward_45[0]
        getitem_250 = convolution_backward_45[1];  convolution_backward_45 = None
        le_44 = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_44 = torch.ops.aten.where.self(le_44, scalar_tensor, getitem_249);  le_44 = getitem_249 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
        sub_241 = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_782);  convolution_10 = unsqueeze_782 = None
        mul_813 = torch.ops.aten.mul.Tensor(where_44, sub_241)
        sum_95 = torch.ops.aten.sum.dim_IntList(mul_813, [0, 2, 3]);  mul_813 = None
        mul_814 = torch.ops.aten.mul.Tensor(sum_94, 0.00048828125)
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(mul_814, 0);  mul_814 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
        mul_815 = torch.ops.aten.mul.Tensor(sum_95, 0.00048828125)
        mul_816 = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_817 = torch.ops.aten.mul.Tensor(mul_815, mul_816);  mul_815 = mul_816 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
        mul_818 = torch.ops.aten.mul.Tensor(squeeze_31, primals_114);  primals_114 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
        mul_819 = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_788);  sub_241 = unsqueeze_788 = None
        sub_243 = torch.ops.aten.sub.Tensor(where_44, mul_819);  where_44 = mul_819 = None
        sub_244 = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_785);  sub_243 = unsqueeze_785 = None
        mul_820 = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_791);  sub_244 = unsqueeze_791 = None
        mul_821 = torch.ops.aten.mul.Tensor(sum_95, squeeze_31);  sum_95 = squeeze_31 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_820, relu_7, primals_11, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_820 = primals_11 = None
        getitem_252 = convolution_backward_46[0]
        getitem_253 = convolution_backward_46[1];  convolution_backward_46 = None
        le_45 = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_45 = torch.ops.aten.where.self(le_45, scalar_tensor, getitem_252);  le_45 = getitem_252 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
        sub_245 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_794);  convolution_9 = unsqueeze_794 = None
        mul_822 = torch.ops.aten.mul.Tensor(where_45, sub_245)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_822, [0, 2, 3]);  mul_822 = None
        mul_823 = torch.ops.aten.mul.Tensor(sum_96, 0.0001220703125)
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
        mul_824 = torch.ops.aten.mul.Tensor(sum_97, 0.0001220703125)
        mul_825 = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_826 = torch.ops.aten.mul.Tensor(mul_824, mul_825);  mul_824 = mul_825 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
        mul_827 = torch.ops.aten.mul.Tensor(squeeze_28, primals_109);  primals_109 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
        mul_828 = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_800);  sub_245 = unsqueeze_800 = None
        sub_247 = torch.ops.aten.sub.Tensor(where_45, mul_828);  where_45 = mul_828 = None
        sub_248 = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_797);  sub_247 = unsqueeze_797 = None
        mul_829 = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_803);  sub_248 = unsqueeze_803 = None
        mul_830 = torch.ops.aten.mul.Tensor(sum_97, squeeze_28);  sum_97 = squeeze_28 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_829, relu_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_829 = primals_10 = None
        getitem_255 = convolution_backward_47[0]
        getitem_256 = convolution_backward_47[1];  convolution_backward_47 = None
        add_317 = torch.ops.aten.add.Tensor(getitem_246, getitem_255);  getitem_246 = getitem_255 = None
        le_46 = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_46 = torch.ops.aten.where.self(le_46, scalar_tensor, add_317);  le_46 = add_317 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
        sub_249 = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_806);  convolution_8 = unsqueeze_806 = None
        mul_831 = torch.ops.aten.mul.Tensor(where_46, sub_249)
        sum_99 = torch.ops.aten.sum.dim_IntList(mul_831, [0, 2, 3]);  mul_831 = None
        mul_832 = torch.ops.aten.mul.Tensor(sum_98, 0.0001220703125)
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
        mul_833 = torch.ops.aten.mul.Tensor(sum_99, 0.0001220703125)
        mul_834 = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_835 = torch.ops.aten.mul.Tensor(mul_833, mul_834);  mul_833 = mul_834 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
        mul_836 = torch.ops.aten.mul.Tensor(squeeze_25, primals_104);  primals_104 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
        mul_837 = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_812);  sub_249 = unsqueeze_812 = None
        sub_251 = torch.ops.aten.sub.Tensor(where_46, mul_837);  mul_837 = None
        sub_252 = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_809);  sub_251 = unsqueeze_809 = None
        mul_838 = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_815);  sub_252 = unsqueeze_815 = None
        mul_839 = torch.ops.aten.mul.Tensor(sum_99, squeeze_25);  sum_99 = squeeze_25 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_838, relu_5, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_838 = primals_9 = None
        getitem_258 = convolution_backward_48[0]
        getitem_259 = convolution_backward_48[1];  convolution_backward_48 = None
        le_47 = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_47 = torch.ops.aten.where.self(le_47, scalar_tensor, getitem_258);  le_47 = getitem_258 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
        sub_253 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_818);  convolution_7 = unsqueeze_818 = None
        mul_840 = torch.ops.aten.mul.Tensor(where_47, sub_253)
        sum_101 = torch.ops.aten.sum.dim_IntList(mul_840, [0, 2, 3]);  mul_840 = None
        mul_841 = torch.ops.aten.mul.Tensor(sum_100, 0.0001220703125)
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
        mul_842 = torch.ops.aten.mul.Tensor(sum_101, 0.0001220703125)
        mul_843 = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_844 = torch.ops.aten.mul.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
        mul_845 = torch.ops.aten.mul.Tensor(squeeze_22, primals_99);  primals_99 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
        mul_846 = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_824);  sub_253 = unsqueeze_824 = None
        sub_255 = torch.ops.aten.sub.Tensor(where_47, mul_846);  where_47 = mul_846 = None
        sub_256 = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_821);  sub_255 = unsqueeze_821 = None
        mul_847 = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_827);  sub_256 = unsqueeze_827 = None
        mul_848 = torch.ops.aten.mul.Tensor(sum_101, squeeze_22);  sum_101 = squeeze_22 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_847, relu_4, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_847 = primals_8 = None
        getitem_261 = convolution_backward_49[0]
        getitem_262 = convolution_backward_49[1];  convolution_backward_49 = None
        add_318 = torch.ops.aten.add.Tensor(where_46, getitem_261);  where_46 = getitem_261 = None
        le_48 = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_48 = torch.ops.aten.where.self(le_48, scalar_tensor, add_318);  le_48 = add_318 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
        sub_257 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_830);  convolution_6 = unsqueeze_830 = None
        mul_849 = torch.ops.aten.mul.Tensor(where_48, sub_257)
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_849, [0, 2, 3]);  mul_849 = None
        mul_850 = torch.ops.aten.mul.Tensor(sum_102, 0.0001220703125)
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
        mul_851 = torch.ops.aten.mul.Tensor(sum_103, 0.0001220703125)
        mul_852 = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_853 = torch.ops.aten.mul.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
        mul_854 = torch.ops.aten.mul.Tensor(squeeze_19, primals_94);  primals_94 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
        mul_855 = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_836);  sub_257 = unsqueeze_836 = None
        sub_259 = torch.ops.aten.sub.Tensor(where_48, mul_855);  mul_855 = None
        sub_260 = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_833);  sub_259 = None
        mul_856 = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_839);  sub_260 = unsqueeze_839 = None
        mul_857 = torch.ops.aten.mul.Tensor(sum_103, squeeze_19);  sum_103 = squeeze_19 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_856, relu_2, primals_7, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_856 = primals_7 = None
        getitem_264 = convolution_backward_50[0]
        getitem_265 = convolution_backward_50[1];  convolution_backward_50 = None
        sub_261 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_842);  convolution_5 = unsqueeze_842 = None
        mul_858 = torch.ops.aten.mul.Tensor(where_48, sub_261)
        sum_105 = torch.ops.aten.sum.dim_IntList(mul_858, [0, 2, 3]);  mul_858 = None
        mul_860 = torch.ops.aten.mul.Tensor(sum_105, 0.0001220703125)
        mul_861 = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_862 = torch.ops.aten.mul.Tensor(mul_860, mul_861);  mul_860 = mul_861 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
        mul_863 = torch.ops.aten.mul.Tensor(squeeze_16, primals_89);  primals_89 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
        mul_864 = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_848);  sub_261 = unsqueeze_848 = None
        sub_263 = torch.ops.aten.sub.Tensor(where_48, mul_864);  where_48 = mul_864 = None
        sub_264 = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_833);  sub_263 = unsqueeze_833 = None
        mul_865 = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_851);  sub_264 = unsqueeze_851 = None
        mul_866 = torch.ops.aten.mul.Tensor(sum_105, squeeze_16);  sum_105 = squeeze_16 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_865, relu_3, primals_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_865 = primals_6 = None
        getitem_267 = convolution_backward_51[0]
        getitem_268 = convolution_backward_51[1];  convolution_backward_51 = None
        le_49 = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_49 = torch.ops.aten.where.self(le_49, scalar_tensor, getitem_267);  le_49 = getitem_267 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
        sub_265 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_854);  convolution_4 = unsqueeze_854 = None
        mul_867 = torch.ops.aten.mul.Tensor(where_49, sub_265)
        sum_107 = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3]);  mul_867 = None
        mul_868 = torch.ops.aten.mul.Tensor(sum_106, 0.0001220703125)
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
        mul_869 = torch.ops.aten.mul.Tensor(sum_107, 0.0001220703125)
        mul_870 = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_871 = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
        mul_872 = torch.ops.aten.mul.Tensor(squeeze_13, primals_84);  primals_84 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
        mul_873 = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_860);  sub_265 = unsqueeze_860 = None
        sub_267 = torch.ops.aten.sub.Tensor(where_49, mul_873);  where_49 = mul_873 = None
        sub_268 = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_857);  sub_267 = unsqueeze_857 = None
        mul_874 = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_863);  sub_268 = unsqueeze_863 = None
        mul_875 = torch.ops.aten.mul.Tensor(sum_107, squeeze_13);  sum_107 = squeeze_13 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_874, relu_2, primals_5, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_874 = primals_5 = None
        getitem_270 = convolution_backward_52[0]
        getitem_271 = convolution_backward_52[1];  convolution_backward_52 = None
        add_319 = torch.ops.aten.add.Tensor(getitem_264, getitem_270);  getitem_264 = getitem_270 = None
        le_50 = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_50 = torch.ops.aten.where.self(le_50, scalar_tensor, add_319);  le_50 = add_319 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
        sub_269 = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_866);  convolution_3 = unsqueeze_866 = None
        mul_876 = torch.ops.aten.mul.Tensor(where_50, sub_269)
        sum_109 = torch.ops.aten.sum.dim_IntList(mul_876, [0, 2, 3]);  mul_876 = None
        mul_877 = torch.ops.aten.mul.Tensor(sum_108, 3.0517578125e-05)
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(mul_877, 0);  mul_877 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
        mul_878 = torch.ops.aten.mul.Tensor(sum_109, 3.0517578125e-05)
        mul_879 = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_880 = torch.ops.aten.mul.Tensor(mul_878, mul_879);  mul_878 = mul_879 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
        mul_881 = torch.ops.aten.mul.Tensor(squeeze_10, primals_79);  primals_79 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
        mul_882 = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_872);  sub_269 = unsqueeze_872 = None
        sub_271 = torch.ops.aten.sub.Tensor(where_50, mul_882);  mul_882 = None
        sub_272 = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_869);  sub_271 = None
        mul_883 = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_875);  sub_272 = unsqueeze_875 = None
        mul_884 = torch.ops.aten.mul.Tensor(sum_109, squeeze_10);  sum_109 = squeeze_10 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_883, relu, primals_4, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_883 = primals_4 = None
        getitem_273 = convolution_backward_53[0]
        getitem_274 = convolution_backward_53[1];  convolution_backward_53 = None
        sub_273 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_878);  convolution_2 = unsqueeze_878 = None
        mul_885 = torch.ops.aten.mul.Tensor(where_50, sub_273)
        sum_111 = torch.ops.aten.sum.dim_IntList(mul_885, [0, 2, 3]);  mul_885 = None
        mul_887 = torch.ops.aten.mul.Tensor(sum_111, 3.0517578125e-05)
        mul_888 = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_889 = torch.ops.aten.mul.Tensor(mul_887, mul_888);  mul_887 = mul_888 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
        mul_890 = torch.ops.aten.mul.Tensor(squeeze_7, primals_74);  primals_74 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
        mul_891 = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_884);  sub_273 = unsqueeze_884 = None
        sub_275 = torch.ops.aten.sub.Tensor(where_50, mul_891);  where_50 = mul_891 = None
        sub_276 = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_869);  sub_275 = unsqueeze_869 = None
        mul_892 = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_887);  sub_276 = unsqueeze_887 = None
        mul_893 = torch.ops.aten.mul.Tensor(sum_111, squeeze_7);  sum_111 = squeeze_7 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_892, relu_1, primals_3, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_892 = primals_3 = None
        getitem_276 = convolution_backward_54[0]
        getitem_277 = convolution_backward_54[1];  convolution_backward_54 = None
        le_51 = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_51 = torch.ops.aten.where.self(le_51, scalar_tensor, getitem_276);  le_51 = getitem_276 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
        sub_277 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_890);  convolution_1 = unsqueeze_890 = None
        mul_894 = torch.ops.aten.mul.Tensor(where_51, sub_277)
        sum_113 = torch.ops.aten.sum.dim_IntList(mul_894, [0, 2, 3]);  mul_894 = None
        mul_895 = torch.ops.aten.mul.Tensor(sum_112, 3.0517578125e-05)
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
        mul_896 = torch.ops.aten.mul.Tensor(sum_113, 3.0517578125e-05)
        mul_897 = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_898 = torch.ops.aten.mul.Tensor(mul_896, mul_897);  mul_896 = mul_897 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
        mul_899 = torch.ops.aten.mul.Tensor(squeeze_4, primals_69);  primals_69 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
        mul_900 = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_896);  sub_277 = unsqueeze_896 = None
        sub_279 = torch.ops.aten.sub.Tensor(where_51, mul_900);  where_51 = mul_900 = None
        sub_280 = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_893);  sub_279 = unsqueeze_893 = None
        mul_901 = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_899);  sub_280 = unsqueeze_899 = None
        mul_902 = torch.ops.aten.mul.Tensor(sum_113, squeeze_4);  sum_113 = squeeze_4 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_901, relu, primals_2, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_901 = primals_2 = None
        getitem_279 = convolution_backward_55[0]
        getitem_280 = convolution_backward_55[1];  convolution_backward_55 = None
        add_320 = torch.ops.aten.add.Tensor(getitem_273, getitem_279);  getitem_273 = getitem_279 = None
        le_52 = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_52 = torch.ops.aten.where.self(le_52, scalar_tensor, add_320);  le_52 = scalar_tensor = add_320 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
        sub_281 = torch.ops.aten.sub.Tensor(convolution, unsqueeze_902);  convolution = unsqueeze_902 = None
        mul_903 = torch.ops.aten.mul.Tensor(where_52, sub_281)
        sum_115 = torch.ops.aten.sum.dim_IntList(mul_903, [0, 2, 3]);  mul_903 = None
        mul_904 = torch.ops.aten.mul.Tensor(sum_114, 7.62939453125e-06)
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
        mul_905 = torch.ops.aten.mul.Tensor(sum_115, 7.62939453125e-06)
        mul_906 = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_907 = torch.ops.aten.mul.Tensor(mul_905, mul_906);  mul_905 = mul_906 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
        mul_908 = torch.ops.aten.mul.Tensor(squeeze_1, primals_64);  primals_64 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
        mul_909 = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_908);  sub_281 = unsqueeze_908 = None
        sub_283 = torch.ops.aten.sub.Tensor(where_52, mul_909);  where_52 = mul_909 = None
        sub_284 = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_905);  sub_283 = unsqueeze_905 = None
        mul_910 = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_911);  sub_284 = unsqueeze_911 = None
        mul_911 = torch.ops.aten.mul.Tensor(sum_115, squeeze_1);  sum_115 = squeeze_1 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_910, primals_60, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_910 = primals_60 = primals_1 = None
        getitem_283 = convolution_backward_56[1];  convolution_backward_56 = None
        return [getitem_283, getitem_280, getitem_277, getitem_274, getitem_271, getitem_268, getitem_265, getitem_262, getitem_259, getitem_256, getitem_253, getitem_250, getitem_247, getitem_244, getitem_241, getitem_238, getitem_235, getitem_232, getitem_229, getitem_226, getitem_223, getitem_220, getitem_217, getitem_214, getitem_211, getitem_208, getitem_205, getitem_202, getitem_199, getitem_196, getitem_193, getitem_190, getitem_187, getitem_184, getitem_181, getitem_178, getitem_175, getitem_172, getitem_169, getitem_166, getitem_163, getitem_160, getitem_157, getitem_154, getitem_151, getitem_148, getitem_145, getitem_142, getitem_139, getitem_136, getitem_133, getitem_130, getitem_127, getitem_124, getitem_121, getitem_118, getitem_115, permute_4, view_1, None, None, None, None, mul_911, sum_114, None, None, None, mul_902, sum_112, None, None, None, mul_893, sum_108, None, None, None, mul_884, sum_108, None, None, None, mul_875, sum_106, None, None, None, mul_866, sum_102, None, None, None, mul_857, sum_102, None, None, None, mul_848, sum_100, None, None, None, mul_839, sum_98, None, None, None, mul_830, sum_96, None, None, None, mul_821, sum_94, None, None, None, mul_812, sum_90, None, None, None, mul_803, sum_90, None, None, None, mul_794, sum_88, None, None, None, mul_785, sum_86, None, None, None, mul_776, sum_84, None, None, None, mul_767, sum_82, None, None, None, mul_758, sum_80, None, None, None, mul_749, sum_78, None, None, None, mul_740, sum_76, None, None, None, mul_731, sum_74, None, None, None, mul_722, sum_72, None, None, None, mul_713, sum_70, None, None, None, mul_704, sum_68, None, None, None, mul_695, sum_66, None, None, None, mul_686, sum_64, None, None, None, mul_677, sum_62, None, None, None, mul_668, sum_60, None, None, None, mul_659, sum_58, None, None, None, mul_650, sum_56, None, None, None, mul_641, sum_52, None, None, None, mul_632, sum_52, None, None, None, mul_623, sum_50, None, None, None, mul_614, sum_48, None, None, None, mul_605, sum_46, None, None, None, mul_596, sum_44, None, None, None, mul_587, sum_42, None, None, None, mul_578, sum_40, None, None, None, mul_569, sum_38, None, None, None, mul_560, sum_36, None, None, None, mul_551, sum_34, None, None, None, mul_542, sum_32, None, None, None, mul_533, sum_30, None, None, None, mul_524, sum_28, None, None, None, mul_515, sum_26, None, None, None, mul_506, sum_24, None, None, None, mul_497, sum_22, None, None, None, mul_488, sum_20, None, None, None, mul_479, sum_18, None, None, None, mul_470, sum_16, None, None, None, mul_461, sum_14, None, None, None, mul_452, sum_12, None, None, None, mul_443, sum_10, None, None, None, mul_434, sum_8, None, None, None, mul_425, sum_6, None, None, None, mul_416, sum_4, None, None, None, mul_407, sum_2]
        
args = [((32, 3, 3, 3), (27, 9, 3, 1), torch.float32, 'cuda'), ((128, 32, 3, 3), (288, 9, 3, 1), torch.float32, 'cuda'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((128, 32, 1, 1), (32, 1, 1, 1), torch.float32, 'cuda'), ((192, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((160, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((2560, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((8, 3, 256, 256), (196608, 65536, 256, 1), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((8, 32, 128, 128), (524288, 16384, 128, 1), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((8, 32, 128, 128), (524288, 16384, 128, 1), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((8, 192, 32, 32), (196608, 1024, 32, 1), torch.float32, 'cuda'), ((8, 160, 32, 32), (163840, 1024, 32, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 32, 32), (163840, 1024, 32, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((8, 160, 16, 16), (40960, 256, 16, 1), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 16, 16), (163840, 256, 16, 1), torch.float32, 'cuda'), ((8, 1920, 16, 16), (491520, 256, 16, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 16, 16), (491520, 256, 16, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((8, 1920, 8, 8), (122880, 64, 8, 1), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((8, 640, 8, 8), (40960, 64, 8, 1), torch.float32, 'cuda'), ((8, 2560, 8, 8), (163840, 64, 8, 1), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((8, 2560), (2560, 1), torch.float32, 'cuda'), ((1000, 2560), (2560, 1), torch.float32, 'cuda'), ((8, 2560, 8, 8), (163840, 64, 8, 1), torch.bool, 'cuda'), ((1, 2560, 1, 1), (2560, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 32, 1, 1), (32, 1, 1, 1), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((8, 1000), (1000, 1), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda'), ((), (), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
