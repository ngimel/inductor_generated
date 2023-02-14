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
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x88X\x0c\x00\x00\x00debug_graphsq\x07\x88X\x0b\x00\x00\x00debug_jointq\x08\x88X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\nu.')


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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345):
        clone = torch.ops.aten.clone.default(primals_62);  primals_62 = None
        clone_1 = torch.ops.aten.clone.default(primals_63);  primals_63 = None
        clone_2 = torch.ops.aten.clone.default(primals_67);  primals_67 = None
        clone_3 = torch.ops.aten.clone.default(primals_68);  primals_68 = None
        clone_4 = torch.ops.aten.clone.default(primals_72);  primals_72 = None
        clone_5 = torch.ops.aten.clone.default(primals_73);  primals_73 = None
        clone_6 = torch.ops.aten.clone.default(primals_77);  primals_77 = None
        clone_7 = torch.ops.aten.clone.default(primals_78);  primals_78 = None
        clone_8 = torch.ops.aten.clone.default(primals_82);  primals_82 = None
        clone_9 = torch.ops.aten.clone.default(primals_83);  primals_83 = None
        clone_10 = torch.ops.aten.clone.default(primals_87);  primals_87 = None
        clone_11 = torch.ops.aten.clone.default(primals_88);  primals_88 = None
        clone_12 = torch.ops.aten.clone.default(primals_92);  primals_92 = None
        clone_13 = torch.ops.aten.clone.default(primals_93);  primals_93 = None
        clone_14 = torch.ops.aten.clone.default(primals_97);  primals_97 = None
        clone_15 = torch.ops.aten.clone.default(primals_98);  primals_98 = None
        clone_16 = torch.ops.aten.clone.default(primals_102);  primals_102 = None
        clone_17 = torch.ops.aten.clone.default(primals_103);  primals_103 = None
        clone_18 = torch.ops.aten.clone.default(primals_107);  primals_107 = None
        clone_19 = torch.ops.aten.clone.default(primals_108);  primals_108 = None
        clone_20 = torch.ops.aten.clone.default(primals_112);  primals_112 = None
        clone_21 = torch.ops.aten.clone.default(primals_113);  primals_113 = None
        clone_22 = torch.ops.aten.clone.default(primals_117);  primals_117 = None
        clone_23 = torch.ops.aten.clone.default(primals_118);  primals_118 = None
        clone_24 = torch.ops.aten.clone.default(primals_122);  primals_122 = None
        clone_25 = torch.ops.aten.clone.default(primals_123);  primals_123 = None
        clone_26 = torch.ops.aten.clone.default(primals_127);  primals_127 = None
        clone_27 = torch.ops.aten.clone.default(primals_128);  primals_128 = None
        clone_28 = torch.ops.aten.clone.default(primals_132);  primals_132 = None
        clone_29 = torch.ops.aten.clone.default(primals_133);  primals_133 = None
        clone_30 = torch.ops.aten.clone.default(primals_137);  primals_137 = None
        clone_31 = torch.ops.aten.clone.default(primals_138);  primals_138 = None
        clone_32 = torch.ops.aten.clone.default(primals_142);  primals_142 = None
        clone_33 = torch.ops.aten.clone.default(primals_143);  primals_143 = None
        clone_34 = torch.ops.aten.clone.default(primals_147);  primals_147 = None
        clone_35 = torch.ops.aten.clone.default(primals_148);  primals_148 = None
        clone_36 = torch.ops.aten.clone.default(primals_152);  primals_152 = None
        clone_37 = torch.ops.aten.clone.default(primals_153);  primals_153 = None
        clone_38 = torch.ops.aten.clone.default(primals_157);  primals_157 = None
        clone_39 = torch.ops.aten.clone.default(primals_158);  primals_158 = None
        clone_40 = torch.ops.aten.clone.default(primals_162);  primals_162 = None
        clone_41 = torch.ops.aten.clone.default(primals_163);  primals_163 = None
        clone_42 = torch.ops.aten.clone.default(primals_167);  primals_167 = None
        clone_43 = torch.ops.aten.clone.default(primals_168);  primals_168 = None
        clone_44 = torch.ops.aten.clone.default(primals_172);  primals_172 = None
        clone_45 = torch.ops.aten.clone.default(primals_173);  primals_173 = None
        clone_46 = torch.ops.aten.clone.default(primals_177);  primals_177 = None
        clone_47 = torch.ops.aten.clone.default(primals_178);  primals_178 = None
        clone_48 = torch.ops.aten.clone.default(primals_182);  primals_182 = None
        clone_49 = torch.ops.aten.clone.default(primals_183);  primals_183 = None
        clone_50 = torch.ops.aten.clone.default(primals_187);  primals_187 = None
        clone_51 = torch.ops.aten.clone.default(primals_188);  primals_188 = None
        clone_52 = torch.ops.aten.clone.default(primals_192);  primals_192 = None
        clone_53 = torch.ops.aten.clone.default(primals_193);  primals_193 = None
        clone_54 = torch.ops.aten.clone.default(primals_197);  primals_197 = None
        clone_55 = torch.ops.aten.clone.default(primals_198);  primals_198 = None
        clone_56 = torch.ops.aten.clone.default(primals_202);  primals_202 = None
        clone_57 = torch.ops.aten.clone.default(primals_203);  primals_203 = None
        clone_58 = torch.ops.aten.clone.default(primals_207);  primals_207 = None
        clone_59 = torch.ops.aten.clone.default(primals_208);  primals_208 = None
        clone_60 = torch.ops.aten.clone.default(primals_212);  primals_212 = None
        clone_61 = torch.ops.aten.clone.default(primals_213);  primals_213 = None
        clone_62 = torch.ops.aten.clone.default(primals_217);  primals_217 = None
        clone_63 = torch.ops.aten.clone.default(primals_218);  primals_218 = None
        clone_64 = torch.ops.aten.clone.default(primals_222);  primals_222 = None
        clone_65 = torch.ops.aten.clone.default(primals_223);  primals_223 = None
        clone_66 = torch.ops.aten.clone.default(primals_227);  primals_227 = None
        clone_67 = torch.ops.aten.clone.default(primals_228);  primals_228 = None
        clone_68 = torch.ops.aten.clone.default(primals_232);  primals_232 = None
        clone_69 = torch.ops.aten.clone.default(primals_233);  primals_233 = None
        clone_70 = torch.ops.aten.clone.default(primals_237);  primals_237 = None
        clone_71 = torch.ops.aten.clone.default(primals_238);  primals_238 = None
        clone_72 = torch.ops.aten.clone.default(primals_242);  primals_242 = None
        clone_73 = torch.ops.aten.clone.default(primals_243);  primals_243 = None
        clone_74 = torch.ops.aten.clone.default(primals_247);  primals_247 = None
        clone_75 = torch.ops.aten.clone.default(primals_248);  primals_248 = None
        clone_76 = torch.ops.aten.clone.default(primals_252);  primals_252 = None
        clone_77 = torch.ops.aten.clone.default(primals_253);  primals_253 = None
        clone_78 = torch.ops.aten.clone.default(primals_257);  primals_257 = None
        clone_79 = torch.ops.aten.clone.default(primals_258);  primals_258 = None
        clone_80 = torch.ops.aten.clone.default(primals_262);  primals_262 = None
        clone_81 = torch.ops.aten.clone.default(primals_263);  primals_263 = None
        clone_82 = torch.ops.aten.clone.default(primals_267);  primals_267 = None
        clone_83 = torch.ops.aten.clone.default(primals_268);  primals_268 = None
        clone_84 = torch.ops.aten.clone.default(primals_272);  primals_272 = None
        clone_85 = torch.ops.aten.clone.default(primals_273);  primals_273 = None
        clone_86 = torch.ops.aten.clone.default(primals_277);  primals_277 = None
        clone_87 = torch.ops.aten.clone.default(primals_278);  primals_278 = None
        clone_88 = torch.ops.aten.clone.default(primals_282);  primals_282 = None
        clone_89 = torch.ops.aten.clone.default(primals_283);  primals_283 = None
        clone_90 = torch.ops.aten.clone.default(primals_287);  primals_287 = None
        clone_91 = torch.ops.aten.clone.default(primals_288);  primals_288 = None
        clone_92 = torch.ops.aten.clone.default(primals_292);  primals_292 = None
        clone_93 = torch.ops.aten.clone.default(primals_293);  primals_293 = None
        clone_94 = torch.ops.aten.clone.default(primals_297);  primals_297 = None
        clone_95 = torch.ops.aten.clone.default(primals_298);  primals_298 = None
        clone_96 = torch.ops.aten.clone.default(primals_302);  primals_302 = None
        clone_97 = torch.ops.aten.clone.default(primals_303);  primals_303 = None
        clone_98 = torch.ops.aten.clone.default(primals_307);  primals_307 = None
        clone_99 = torch.ops.aten.clone.default(primals_308);  primals_308 = None
        clone_100 = torch.ops.aten.clone.default(primals_312);  primals_312 = None
        clone_101 = torch.ops.aten.clone.default(primals_313);  primals_313 = None
        clone_102 = torch.ops.aten.clone.default(primals_317);  primals_317 = None
        clone_103 = torch.ops.aten.clone.default(primals_318);  primals_318 = None
        clone_104 = torch.ops.aten.clone.default(primals_322);  primals_322 = None
        clone_105 = torch.ops.aten.clone.default(primals_323);  primals_323 = None
        clone_106 = torch.ops.aten.clone.default(primals_327);  primals_327 = None
        clone_107 = torch.ops.aten.clone.default(primals_328);  primals_328 = None
        clone_108 = torch.ops.aten.clone.default(primals_332);  primals_332 = None
        clone_109 = torch.ops.aten.clone.default(primals_333);  primals_333 = None
        clone_110 = torch.ops.aten.clone.default(primals_337);  primals_337 = None
        clone_111 = torch.ops.aten.clone.default(primals_338);  primals_338 = None
        clone_112 = torch.ops.aten.clone.default(primals_342);  primals_342 = None
        clone_113 = torch.ops.aten.clone.default(primals_343);  primals_343 = None
        convolution = torch.ops.aten.convolution.default(primals_60, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add = torch.ops.aten.add.Tensor(primals_61, 1);  primals_61 = None
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2 = torch.ops.aten.mul.Tensor(clone, 0.9);  clone = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(clone_1, 0.9);  clone_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_64, -1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_65, -1);  primals_65 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu = torch.ops.aten.relu.default(add_4);  add_4 = None
        convolution_1 = torch.ops.aten.convolution.default(relu, primals_2, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_5 = torch.ops.aten.add.Tensor(primals_66, 1);  primals_66 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3 = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4 = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9 = torch.ops.aten.mul.Tensor(clone_2, 0.9);  clone_2 = None
        add_7 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(squeeze_5, 1.000030518509476);  squeeze_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(clone_3, 0.9);  clone_3 = None
        add_8 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(primals_69, -1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        add_9 = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1 = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, primals_3, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_10 = torch.ops.aten.add.Tensor(primals_71, 1);  primals_71 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6 = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7 = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15 = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16 = torch.ops.aten.mul.Tensor(clone_4, 0.9);  clone_4 = None
        add_12 = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8 = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(squeeze_8, 1.000030518509476);  squeeze_8 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(clone_5, 0.9);  clone_5 = None
        add_13 = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(primals_74, -1)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        add_14 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convolution_3 = torch.ops.aten.convolution.default(relu, primals_4, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_15 = torch.ops.aten.add.Tensor(primals_76, 1);  primals_76 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9 = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10 = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22 = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23 = torch.ops.aten.mul.Tensor(clone_6, 0.9);  clone_6 = None
        add_17 = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11 = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24 = torch.ops.aten.mul.Tensor(squeeze_11, 1.000030518509476);  squeeze_11 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26 = torch.ops.aten.mul.Tensor(clone_7, 0.9);  clone_7 = None
        add_18 = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(primals_79, -1)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        add_19 = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        add_20 = torch.ops.aten.add.Tensor(add_14, add_19);  add_14 = add_19 = None
        relu_2 = torch.ops.aten.relu.default(add_20);  add_20 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_2, primals_5, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_21 = torch.ops.aten.add.Tensor(primals_81, 1);  primals_81 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12 = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13 = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29 = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30 = torch.ops.aten.mul.Tensor(clone_8, 0.9);  clone_8 = None
        add_23 = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14 = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31 = torch.ops.aten.mul.Tensor(squeeze_14, 1.0001220852154804);  squeeze_14 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33 = torch.ops.aten.mul.Tensor(clone_9, 0.9);  clone_9 = None
        add_24 = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        add_25 = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        relu_3 = torch.ops.aten.relu.default(add_25);  add_25 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_3, primals_6, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_26 = torch.ops.aten.add.Tensor(primals_86, 1);  primals_86 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5 = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15 = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16 = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36 = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37 = torch.ops.aten.mul.Tensor(clone_10, 0.9);  clone_10 = None
        add_28 = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17 = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38 = torch.ops.aten.mul.Tensor(squeeze_17, 1.0001220852154804);  squeeze_17 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40 = torch.ops.aten.mul.Tensor(clone_11, 0.9);  clone_11 = None
        add_29 = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(primals_89, -1)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        add_30 = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_2, primals_7, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_31 = torch.ops.aten.add.Tensor(primals_91, 1);  primals_91 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_6 = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18 = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19 = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43 = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44 = torch.ops.aten.mul.Tensor(clone_12, 0.9);  clone_12 = None
        add_33 = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20 = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_45 = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001220852154804);  squeeze_20 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47 = torch.ops.aten.mul.Tensor(clone_13, 0.9);  clone_13 = None
        add_34 = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(primals_94, -1)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        add_35 = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        add_36 = torch.ops.aten.add.Tensor(add_30, add_35);  add_30 = add_35 = None
        relu_4 = torch.ops.aten.relu.default(add_36);  add_36 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_4, primals_8, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_37 = torch.ops.aten.add.Tensor(primals_96, 1);  primals_96 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_7 = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21 = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22 = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50 = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51 = torch.ops.aten.mul.Tensor(clone_14, 0.9);  clone_14 = None
        add_39 = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23 = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_52 = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001220852154804);  squeeze_23 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54 = torch.ops.aten.mul.Tensor(clone_15, 0.9);  clone_15 = None
        add_40 = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(primals_99, -1)
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        add_41 = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        relu_5 = torch.ops.aten.relu.default(add_41);  add_41 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_5, primals_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_42 = torch.ops.aten.add.Tensor(primals_101, 1);  primals_101 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8 = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24 = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25 = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57 = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58 = torch.ops.aten.mul.Tensor(clone_16, 0.9);  clone_16 = None
        add_44 = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26 = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_59 = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001220852154804);  squeeze_26 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61 = torch.ops.aten.mul.Tensor(clone_17, 0.9);  clone_17 = None
        add_45 = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(primals_104, -1)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        add_46 = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        add_47 = torch.ops.aten.add.Tensor(add_46, relu_4);  add_46 = None
        relu_6 = torch.ops.aten.relu.default(add_47);  add_47 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_6, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_48 = torch.ops.aten.add.Tensor(primals_106, 1);  primals_106 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_9 = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_63 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27 = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28 = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64 = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65 = torch.ops.aten.mul.Tensor(clone_18, 0.9);  clone_18 = None
        add_50 = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29 = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_66 = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001220852154804);  squeeze_29 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68 = torch.ops.aten.mul.Tensor(clone_19, 0.9);  clone_19 = None
        add_51 = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(primals_109, -1)
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        add_52 = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        relu_7 = torch.ops.aten.relu.default(add_52);  add_52 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_7, primals_11, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_53 = torch.ops.aten.add.Tensor(primals_111, 1);  primals_111 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_10 = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30 = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_31 = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71 = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72 = torch.ops.aten.mul.Tensor(clone_20, 0.9);  clone_20 = None
        add_55 = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32 = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_73 = torch.ops.aten.mul.Tensor(squeeze_32, 1.0004885197850513);  squeeze_32 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75 = torch.ops.aten.mul.Tensor(clone_21, 0.9);  clone_21 = None
        add_56 = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        add_57 = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        relu_8 = torch.ops.aten.relu.default(add_57);  add_57 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_8, primals_12, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_58 = torch.ops.aten.add.Tensor(primals_116, 1);  primals_116 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_59 = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_11 = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_77 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33 = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34 = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78 = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79 = torch.ops.aten.mul.Tensor(clone_22, 0.9);  clone_22 = None
        add_60 = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35 = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_80 = torch.ops.aten.mul.Tensor(squeeze_35, 1.0004885197850513);  squeeze_35 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82 = torch.ops.aten.mul.Tensor(clone_23, 0.9);  clone_23 = None
        add_61 = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(primals_119, -1)
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        add_62 = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_6, primals_13, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_63 = torch.ops.aten.add.Tensor(primals_121, 1);  primals_121 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12 = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
        mul_84 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36 = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37 = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85 = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86 = torch.ops.aten.mul.Tensor(clone_24, 0.9);  clone_24 = None
        add_65 = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38 = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_87 = torch.ops.aten.mul.Tensor(squeeze_38, 1.0004885197850513);  squeeze_38 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89 = torch.ops.aten.mul.Tensor(clone_25, 0.9);  clone_25 = None
        add_66 = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(primals_124, -1)
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(primals_125, -1);  primals_125 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        add_67 = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        add_68 = torch.ops.aten.add.Tensor(add_62, add_67);  add_62 = add_67 = None
        relu_9 = torch.ops.aten.relu.default(add_68);  add_68 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_9, primals_14, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_69 = torch.ops.aten.add.Tensor(primals_126, 1);  primals_126 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_13 = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39 = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_40 = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92 = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93 = torch.ops.aten.mul.Tensor(clone_26, 0.9);  clone_26 = None
        add_71 = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41 = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_94 = torch.ops.aten.mul.Tensor(squeeze_41, 1.0004885197850513);  squeeze_41 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96 = torch.ops.aten.mul.Tensor(clone_27, 0.9);  clone_27 = None
        add_72 = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(primals_129, -1)
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        add_73 = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        relu_10 = torch.ops.aten.relu.default(add_73);  add_73 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_10, primals_15, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_74 = torch.ops.aten.add.Tensor(primals_131, 1);  primals_131 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_14 = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
        mul_98 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42 = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43 = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99 = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100 = torch.ops.aten.mul.Tensor(clone_28, 0.9);  clone_28 = None
        add_76 = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44 = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_101 = torch.ops.aten.mul.Tensor(squeeze_44, 1.0004885197850513);  squeeze_44 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103 = torch.ops.aten.mul.Tensor(clone_29, 0.9);  clone_29 = None
        add_77 = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(primals_134, -1)
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        add_78 = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        relu_11 = torch.ops.aten.relu.default(add_78);  add_78 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_11, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_79 = torch.ops.aten.add.Tensor(primals_136, 1);  primals_136 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_15 = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
        mul_105 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45 = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46 = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106 = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107 = torch.ops.aten.mul.Tensor(clone_30, 0.9);  clone_30 = None
        add_81 = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47 = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_108 = torch.ops.aten.mul.Tensor(squeeze_47, 1.0004885197850513);  squeeze_47 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110 = torch.ops.aten.mul.Tensor(clone_31, 0.9);  clone_31 = None
        add_82 = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(primals_139, -1)
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(primals_140, -1);  primals_140 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        add_83 = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        add_84 = torch.ops.aten.add.Tensor(add_83, relu_9);  add_83 = None
        relu_12 = torch.ops.aten.relu.default(add_84);  add_84 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_12, primals_17, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_85 = torch.ops.aten.add.Tensor(primals_141, 1);  primals_141 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_16 = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48 = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_49 = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113 = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114 = torch.ops.aten.mul.Tensor(clone_32, 0.9);  clone_32 = None
        add_87 = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50 = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_115 = torch.ops.aten.mul.Tensor(squeeze_50, 1.0004885197850513);  squeeze_50 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117 = torch.ops.aten.mul.Tensor(clone_33, 0.9);  clone_33 = None
        add_88 = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        add_89 = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        relu_13 = torch.ops.aten.relu.default(add_89);  add_89 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_13, primals_18, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_90 = torch.ops.aten.add.Tensor(primals_146, 1);  primals_146 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_91 = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_17 = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
        mul_119 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51 = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_52 = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120 = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121 = torch.ops.aten.mul.Tensor(clone_34, 0.9);  clone_34 = None
        add_92 = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53 = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_122 = torch.ops.aten.mul.Tensor(squeeze_53, 1.0004885197850513);  squeeze_53 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124 = torch.ops.aten.mul.Tensor(clone_35, 0.9);  clone_35 = None
        add_93 = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(primals_149, -1)
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        add_94 = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        relu_14 = torch.ops.aten.relu.default(add_94);  add_94 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_14, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_95 = torch.ops.aten.add.Tensor(primals_151, 1);  primals_151 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_18 = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
        mul_126 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54 = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_55 = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127 = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128 = torch.ops.aten.mul.Tensor(clone_36, 0.9);  clone_36 = None
        add_97 = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56 = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_129 = torch.ops.aten.mul.Tensor(squeeze_56, 1.0004885197850513);  squeeze_56 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131 = torch.ops.aten.mul.Tensor(clone_37, 0.9);  clone_37 = None
        add_98 = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(primals_154, -1)
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(primals_155, -1);  primals_155 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        add_99 = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        add_100 = torch.ops.aten.add.Tensor(add_99, relu_12);  add_99 = None
        relu_15 = torch.ops.aten.relu.default(add_100);  add_100 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_15, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_101 = torch.ops.aten.add.Tensor(primals_156, 1);  primals_156 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_19 = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
        mul_133 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57 = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_58 = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134 = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135 = torch.ops.aten.mul.Tensor(clone_38, 0.9);  clone_38 = None
        add_103 = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59 = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_136 = torch.ops.aten.mul.Tensor(squeeze_59, 1.0004885197850513);  squeeze_59 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138 = torch.ops.aten.mul.Tensor(clone_39, 0.9);  clone_39 = None
        add_104 = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(primals_159, -1)
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        mul_139 = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        add_105 = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        relu_16 = torch.ops.aten.relu.default(add_105);  add_105 = None
        convolution_20 = torch.ops.aten.convolution.default(relu_16, primals_21, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_106 = torch.ops.aten.add.Tensor(primals_161, 1);  primals_161 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_107 = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_20 = torch.ops.aten.sub.Tensor(convolution_20, getitem_41)
        mul_140 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60 = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_61 = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141 = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142 = torch.ops.aten.mul.Tensor(clone_40, 0.9);  clone_40 = None
        add_108 = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62 = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_143 = torch.ops.aten.mul.Tensor(squeeze_62, 1.0004885197850513);  squeeze_62 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145 = torch.ops.aten.mul.Tensor(clone_41, 0.9);  clone_41 = None
        add_109 = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(primals_164, -1)
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        add_110 = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        relu_17 = torch.ops.aten.relu.default(add_110);  add_110 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_17, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_111 = torch.ops.aten.add.Tensor(primals_166, 1);  primals_166 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_112 = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_21 = torch.ops.aten.sub.Tensor(convolution_21, getitem_43)
        mul_147 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63 = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_64 = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148 = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149 = torch.ops.aten.mul.Tensor(clone_42, 0.9);  clone_42 = None
        add_113 = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65 = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_150 = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152 = torch.ops.aten.mul.Tensor(clone_43, 0.9);  clone_43 = None
        add_114 = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(primals_169, -1)
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(primals_170, -1);  primals_170 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        add_115 = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        add_116 = torch.ops.aten.add.Tensor(add_115, relu_15);  add_115 = None
        relu_18 = torch.ops.aten.relu.default(add_116);  add_116 = None
        convolution_22 = torch.ops.aten.convolution.default(relu_18, primals_23, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_117 = torch.ops.aten.add.Tensor(primals_171, 1);  primals_171 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_118 = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_22 = torch.ops.aten.sub.Tensor(convolution_22, getitem_45)
        mul_154 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66 = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_67 = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155 = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156 = torch.ops.aten.mul.Tensor(clone_44, 0.9);  clone_44 = None
        add_119 = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68 = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_157 = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159 = torch.ops.aten.mul.Tensor(clone_45, 0.9);  clone_45 = None
        add_120 = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        add_121 = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        relu_19 = torch.ops.aten.relu.default(add_121);  add_121 = None
        convolution_23 = torch.ops.aten.convolution.default(relu_19, primals_24, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_122 = torch.ops.aten.add.Tensor(primals_176, 1);  primals_176 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_23 = torch.ops.aten.sub.Tensor(convolution_23, getitem_47)
        mul_161 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69 = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_70 = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162 = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163 = torch.ops.aten.mul.Tensor(clone_46, 0.9);  clone_46 = None
        add_124 = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71 = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_164 = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166 = torch.ops.aten.mul.Tensor(clone_47, 0.9);  clone_47 = None
        add_125 = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(primals_179, -1)
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        add_126 = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        relu_20 = torch.ops.aten.relu.default(add_126);  add_126 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_20, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_127 = torch.ops.aten.add.Tensor(primals_181, 1);  primals_181 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_128 = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24 = torch.ops.aten.sub.Tensor(convolution_24, getitem_49)
        mul_168 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72 = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_73 = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169 = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170 = torch.ops.aten.mul.Tensor(clone_48, 0.9);  clone_48 = None
        add_129 = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74 = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_171 = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173 = torch.ops.aten.mul.Tensor(clone_49, 0.9);  clone_49 = None
        add_130 = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(primals_184, -1)
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(primals_185, -1);  primals_185 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        add_131 = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        add_132 = torch.ops.aten.add.Tensor(add_131, relu_18);  add_131 = None
        relu_21 = torch.ops.aten.relu.default(add_132);  add_132 = None
        convolution_25 = torch.ops.aten.convolution.default(relu_21, primals_26, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_133 = torch.ops.aten.add.Tensor(primals_186, 1);  primals_186 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_25 = torch.ops.aten.sub.Tensor(convolution_25, getitem_51)
        mul_175 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75 = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_76 = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_176 = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_177 = torch.ops.aten.mul.Tensor(clone_50, 0.9);  clone_50 = None
        add_135 = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77 = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_178 = torch.ops.aten.mul.Tensor(squeeze_77, 1.0004885197850513);  squeeze_77 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180 = torch.ops.aten.mul.Tensor(clone_51, 0.9);  clone_51 = None
        add_136 = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(primals_189, -1)
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(primals_190, -1);  primals_190 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        add_137 = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        relu_22 = torch.ops.aten.relu.default(add_137);  add_137 = None
        convolution_26 = torch.ops.aten.convolution.default(relu_22, primals_27, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_138 = torch.ops.aten.add.Tensor(primals_191, 1);  primals_191 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_139 = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        sub_26 = torch.ops.aten.sub.Tensor(convolution_26, getitem_53)
        mul_182 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78 = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_79 = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183 = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_184 = torch.ops.aten.mul.Tensor(clone_52, 0.9);  clone_52 = None
        add_140 = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80 = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_185 = torch.ops.aten.mul.Tensor(squeeze_80, 1.0004885197850513);  squeeze_80 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187 = torch.ops.aten.mul.Tensor(clone_53, 0.9);  clone_53 = None
        add_141 = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(primals_194, -1)
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        add_142 = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        relu_23 = torch.ops.aten.relu.default(add_142);  add_142 = None
        convolution_27 = torch.ops.aten.convolution.default(relu_23, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_143 = torch.ops.aten.add.Tensor(primals_196, 1);  primals_196 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_144 = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_27 = torch.ops.aten.sub.Tensor(convolution_27, getitem_55)
        mul_189 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81 = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_82 = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190 = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191 = torch.ops.aten.mul.Tensor(clone_54, 0.9);  clone_54 = None
        add_145 = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83 = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_192 = torch.ops.aten.mul.Tensor(squeeze_83, 1.0004885197850513);  squeeze_83 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194 = torch.ops.aten.mul.Tensor(clone_55, 0.9);  clone_55 = None
        add_146 = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(primals_199, -1)
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(primals_200, -1);  primals_200 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        add_147 = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        add_148 = torch.ops.aten.add.Tensor(add_147, relu_21);  add_147 = None
        relu_24 = torch.ops.aten.relu.default(add_148);  add_148 = None
        convolution_28 = torch.ops.aten.convolution.default(relu_24, primals_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_149 = torch.ops.aten.add.Tensor(primals_201, 1);  primals_201 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_150 = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_28 = torch.ops.aten.sub.Tensor(convolution_28, getitem_57)
        mul_196 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84 = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_85 = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197 = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198 = torch.ops.aten.mul.Tensor(clone_56, 0.9);  clone_56 = None
        add_151 = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86 = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_199 = torch.ops.aten.mul.Tensor(squeeze_86, 1.0004885197850513);  squeeze_86 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201 = torch.ops.aten.mul.Tensor(clone_57, 0.9);  clone_57 = None
        add_152 = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        add_153 = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        relu_25 = torch.ops.aten.relu.default(add_153);  add_153 = None
        convolution_29 = torch.ops.aten.convolution.default(relu_25, primals_30, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1920)
        add_154 = torch.ops.aten.add.Tensor(primals_206, 1);  primals_206 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_155 = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_29 = torch.ops.aten.sub.Tensor(convolution_29, getitem_59)
        mul_203 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87 = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_88 = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204 = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205 = torch.ops.aten.mul.Tensor(clone_58, 0.9);  clone_58 = None
        add_156 = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89 = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_206 = torch.ops.aten.mul.Tensor(squeeze_89, 1.0019569471624266);  squeeze_89 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208 = torch.ops.aten.mul.Tensor(clone_59, 0.9);  clone_59 = None
        add_157 = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(primals_209, -1)
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        add_158 = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        relu_26 = torch.ops.aten.relu.default(add_158);  add_158 = None
        convolution_30 = torch.ops.aten.convolution.default(relu_26, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_159 = torch.ops.aten.add.Tensor(primals_211, 1);  primals_211 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_160 = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_30 = torch.ops.aten.sub.Tensor(convolution_30, getitem_61)
        mul_210 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90 = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_91 = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211 = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212 = torch.ops.aten.mul.Tensor(clone_60, 0.9);  clone_60 = None
        add_161 = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92 = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_213 = torch.ops.aten.mul.Tensor(squeeze_92, 1.0019569471624266);  squeeze_92 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215 = torch.ops.aten.mul.Tensor(clone_61, 0.9);  clone_61 = None
        add_162 = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(primals_214, -1)
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(primals_215, -1);  primals_215 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        add_163 = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        convolution_31 = torch.ops.aten.convolution.default(relu_24, primals_32, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_164 = torch.ops.aten.add.Tensor(primals_216, 1);  primals_216 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_31 = torch.ops.aten.sub.Tensor(convolution_31, getitem_63)
        mul_217 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93 = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_94 = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218 = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_219 = torch.ops.aten.mul.Tensor(clone_62, 0.9);  clone_62 = None
        add_166 = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95 = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_220 = torch.ops.aten.mul.Tensor(squeeze_95, 1.0019569471624266);  squeeze_95 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222 = torch.ops.aten.mul.Tensor(clone_63, 0.9);  clone_63 = None
        add_167 = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(primals_219, -1)
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(primals_220, -1);  primals_220 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        add_168 = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        add_169 = torch.ops.aten.add.Tensor(add_163, add_168);  add_163 = add_168 = None
        relu_27 = torch.ops.aten.relu.default(add_169);  add_169 = None
        convolution_32 = torch.ops.aten.convolution.default(relu_27, primals_33, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_170 = torch.ops.aten.add.Tensor(primals_221, 1);  primals_221 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64 = var_mean_32[0]
        getitem_65 = var_mean_32[1];  var_mean_32 = None
        add_171 = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_32 = torch.ops.aten.sub.Tensor(convolution_32, getitem_65)
        mul_224 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96 = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_97 = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225 = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226 = torch.ops.aten.mul.Tensor(clone_64, 0.9);  clone_64 = None
        add_172 = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98 = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_227 = torch.ops.aten.mul.Tensor(squeeze_98, 1.0019569471624266);  squeeze_98 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229 = torch.ops.aten.mul.Tensor(clone_65, 0.9);  clone_65 = None
        add_173 = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(primals_224, -1)
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(primals_225, -1);  primals_225 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        add_174 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        relu_28 = torch.ops.aten.relu.default(add_174);  add_174 = None
        convolution_33 = torch.ops.aten.convolution.default(relu_28, primals_34, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_175 = torch.ops.aten.add.Tensor(primals_226, 1);  primals_226 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66 = var_mean_33[0]
        getitem_67 = var_mean_33[1];  var_mean_33 = None
        add_176 = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_33 = torch.ops.aten.sub.Tensor(convolution_33, getitem_67)
        mul_231 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99 = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_100 = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232 = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233 = torch.ops.aten.mul.Tensor(clone_66, 0.9);  clone_66 = None
        add_177 = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101 = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_234 = torch.ops.aten.mul.Tensor(squeeze_101, 1.0019569471624266);  squeeze_101 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236 = torch.ops.aten.mul.Tensor(clone_67, 0.9);  clone_67 = None
        add_178 = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(primals_229, -1)
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(primals_230, -1);  primals_230 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        add_179 = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        relu_29 = torch.ops.aten.relu.default(add_179);  add_179 = None
        convolution_34 = torch.ops.aten.convolution.default(relu_29, primals_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_180 = torch.ops.aten.add.Tensor(primals_231, 1);  primals_231 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68 = var_mean_34[0]
        getitem_69 = var_mean_34[1];  var_mean_34 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_34 = torch.ops.aten.sub.Tensor(convolution_34, getitem_69)
        mul_238 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102 = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_103 = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239 = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240 = torch.ops.aten.mul.Tensor(clone_68, 0.9);  clone_68 = None
        add_182 = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104 = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_241 = torch.ops.aten.mul.Tensor(squeeze_104, 1.0019569471624266);  squeeze_104 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243 = torch.ops.aten.mul.Tensor(clone_69, 0.9);  clone_69 = None
        add_183 = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        add_184 = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        add_185 = torch.ops.aten.add.Tensor(add_184, relu_27);  add_184 = None
        relu_30 = torch.ops.aten.relu.default(add_185);  add_185 = None
        convolution_35 = torch.ops.aten.convolution.default(relu_30, primals_36, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_186 = torch.ops.aten.add.Tensor(primals_236, 1);  primals_236 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70 = var_mean_35[0]
        getitem_71 = var_mean_35[1];  var_mean_35 = None
        add_187 = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
        sub_35 = torch.ops.aten.sub.Tensor(convolution_35, getitem_71)
        mul_245 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105 = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_106 = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246 = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247 = torch.ops.aten.mul.Tensor(clone_70, 0.9);  clone_70 = None
        add_188 = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107 = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_248 = torch.ops.aten.mul.Tensor(squeeze_107, 1.0019569471624266);  squeeze_107 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250 = torch.ops.aten.mul.Tensor(clone_71, 0.9);  clone_71 = None
        add_189 = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(primals_239, -1)
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        add_190 = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        relu_31 = torch.ops.aten.relu.default(add_190);  add_190 = None
        convolution_36 = torch.ops.aten.convolution.default(relu_31, primals_37, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_191 = torch.ops.aten.add.Tensor(primals_241, 1);  primals_241 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72 = var_mean_36[0]
        getitem_73 = var_mean_36[1];  var_mean_36 = None
        add_192 = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_36 = torch.ops.aten.sub.Tensor(convolution_36, getitem_73)
        mul_252 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108 = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_109 = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253 = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254 = torch.ops.aten.mul.Tensor(clone_72, 0.9);  clone_72 = None
        add_193 = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110 = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_255 = torch.ops.aten.mul.Tensor(squeeze_110, 1.0019569471624266);  squeeze_110 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257 = torch.ops.aten.mul.Tensor(clone_73, 0.9);  clone_73 = None
        add_194 = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(primals_244, -1)
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(primals_245, -1);  primals_245 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        add_195 = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        relu_32 = torch.ops.aten.relu.default(add_195);  add_195 = None
        convolution_37 = torch.ops.aten.convolution.default(relu_32, primals_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_196 = torch.ops.aten.add.Tensor(primals_246, 1);  primals_246 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74 = var_mean_37[0]
        getitem_75 = var_mean_37[1];  var_mean_37 = None
        add_197 = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_37 = torch.ops.aten.sub.Tensor(convolution_37, getitem_75)
        mul_259 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111 = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_112 = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260 = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_261 = torch.ops.aten.mul.Tensor(clone_74, 0.9);  clone_74 = None
        add_198 = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113 = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_262 = torch.ops.aten.mul.Tensor(squeeze_113, 1.0019569471624266);  squeeze_113 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264 = torch.ops.aten.mul.Tensor(clone_75, 0.9);  clone_75 = None
        add_199 = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(primals_249, -1)
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(primals_250, -1);  primals_250 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        add_200 = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        add_201 = torch.ops.aten.add.Tensor(add_200, relu_30);  add_200 = None
        relu_33 = torch.ops.aten.relu.default(add_201);  add_201 = None
        convolution_38 = torch.ops.aten.convolution.default(relu_33, primals_39, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_202 = torch.ops.aten.add.Tensor(primals_251, 1);  primals_251 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76 = var_mean_38[0]
        getitem_77 = var_mean_38[1];  var_mean_38 = None
        add_203 = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_38 = torch.ops.aten.sub.Tensor(convolution_38, getitem_77)
        mul_266 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114 = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_115 = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267 = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_268 = torch.ops.aten.mul.Tensor(clone_76, 0.9);  clone_76 = None
        add_204 = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116 = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_269 = torch.ops.aten.mul.Tensor(squeeze_116, 1.0019569471624266);  squeeze_116 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271 = torch.ops.aten.mul.Tensor(clone_77, 0.9);  clone_77 = None
        add_205 = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(primals_254, -1)
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(primals_255, -1);  primals_255 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        add_206 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        relu_34 = torch.ops.aten.relu.default(add_206);  add_206 = None
        convolution_39 = torch.ops.aten.convolution.default(relu_34, primals_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_207 = torch.ops.aten.add.Tensor(primals_256, 1);  primals_256 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78 = var_mean_39[0]
        getitem_79 = var_mean_39[1];  var_mean_39 = None
        add_208 = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        sub_39 = torch.ops.aten.sub.Tensor(convolution_39, getitem_79)
        mul_273 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117 = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_118 = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274 = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_275 = torch.ops.aten.mul.Tensor(clone_78, 0.9);  clone_78 = None
        add_209 = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119 = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_276 = torch.ops.aten.mul.Tensor(squeeze_119, 1.0019569471624266);  squeeze_119 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278 = torch.ops.aten.mul.Tensor(clone_79, 0.9);  clone_79 = None
        add_210 = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(primals_259, -1)
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(primals_260, -1);  primals_260 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        add_211 = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        relu_35 = torch.ops.aten.relu.default(add_211);  add_211 = None
        convolution_40 = torch.ops.aten.convolution.default(relu_35, primals_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_212 = torch.ops.aten.add.Tensor(primals_261, 1);  primals_261 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80 = var_mean_40[0]
        getitem_81 = var_mean_40[1];  var_mean_40 = None
        add_213 = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_40 = torch.ops.aten.sub.Tensor(convolution_40, getitem_81)
        mul_280 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120 = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_121 = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_281 = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_282 = torch.ops.aten.mul.Tensor(clone_80, 0.9);  clone_80 = None
        add_214 = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122 = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_283 = torch.ops.aten.mul.Tensor(squeeze_122, 1.0019569471624266);  squeeze_122 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285 = torch.ops.aten.mul.Tensor(clone_81, 0.9);  clone_81 = None
        add_215 = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        add_216 = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        add_217 = torch.ops.aten.add.Tensor(add_216, relu_33);  add_216 = None
        relu_36 = torch.ops.aten.relu.default(add_217);  add_217 = None
        convolution_41 = torch.ops.aten.convolution.default(relu_36, primals_42, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_218 = torch.ops.aten.add.Tensor(primals_266, 1);  primals_266 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82 = var_mean_41[0]
        getitem_83 = var_mean_41[1];  var_mean_41 = None
        add_219 = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_41, getitem_83)
        mul_287 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123 = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_124 = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288 = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289 = torch.ops.aten.mul.Tensor(clone_82, 0.9);  clone_82 = None
        add_220 = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125 = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_290 = torch.ops.aten.mul.Tensor(squeeze_125, 1.0019569471624266);  squeeze_125 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292 = torch.ops.aten.mul.Tensor(clone_83, 0.9);  clone_83 = None
        add_221 = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(primals_269, -1)
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(primals_270, -1);  primals_270 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        add_222 = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        relu_37 = torch.ops.aten.relu.default(add_222);  add_222 = None
        convolution_42 = torch.ops.aten.convolution.default(relu_37, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_223 = torch.ops.aten.add.Tensor(primals_271, 1);  primals_271 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_84 = var_mean_42[0]
        getitem_85 = var_mean_42[1];  var_mean_42 = None
        add_224 = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_42, getitem_85)
        mul_294 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126 = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_127 = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295 = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_296 = torch.ops.aten.mul.Tensor(clone_84, 0.9);  clone_84 = None
        add_225 = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128 = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_297 = torch.ops.aten.mul.Tensor(squeeze_128, 1.0019569471624266);  squeeze_128 = None
        mul_298 = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299 = torch.ops.aten.mul.Tensor(clone_85, 0.9);  clone_85 = None
        add_226 = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(primals_274, -1)
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(primals_275, -1);  primals_275 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        add_227 = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        relu_38 = torch.ops.aten.relu.default(add_227);  add_227 = None
        convolution_43 = torch.ops.aten.convolution.default(relu_38, primals_44, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_228 = torch.ops.aten.add.Tensor(primals_276, 1);  primals_276 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_86 = var_mean_43[0]
        getitem_87 = var_mean_43[1];  var_mean_43 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_43, getitem_87)
        mul_301 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129 = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_130 = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302 = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_303 = torch.ops.aten.mul.Tensor(clone_86, 0.9);  clone_86 = None
        add_230 = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131 = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_304 = torch.ops.aten.mul.Tensor(squeeze_131, 1.0019569471624266);  squeeze_131 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306 = torch.ops.aten.mul.Tensor(clone_87, 0.9);  clone_87 = None
        add_231 = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(primals_279, -1)
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(primals_280, -1);  primals_280 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        add_232 = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        add_233 = torch.ops.aten.add.Tensor(add_232, relu_36);  add_232 = None
        relu_39 = torch.ops.aten.relu.default(add_233);  add_233 = None
        convolution_44 = torch.ops.aten.convolution.default(relu_39, primals_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_234 = torch.ops.aten.add.Tensor(primals_281, 1);  primals_281 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
        getitem_88 = var_mean_44[0]
        getitem_89 = var_mean_44[1];  var_mean_44 = None
        add_235 = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_44, getitem_89)
        mul_308 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132 = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_133 = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309 = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_310 = torch.ops.aten.mul.Tensor(clone_88, 0.9);  clone_88 = None
        add_236 = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134 = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_311 = torch.ops.aten.mul.Tensor(squeeze_134, 1.0019569471624266);  squeeze_134 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313 = torch.ops.aten.mul.Tensor(clone_89, 0.9);  clone_89 = None
        add_237 = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(primals_284, -1)
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(primals_285, -1);  primals_285 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        add_238 = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        relu_40 = torch.ops.aten.relu.default(add_238);  add_238 = None
        convolution_45 = torch.ops.aten.convolution.default(relu_40, primals_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_239 = torch.ops.aten.add.Tensor(primals_286, 1);  primals_286 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_90 = var_mean_45[0]
        getitem_91 = var_mean_45[1];  var_mean_45 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_45, getitem_91)
        mul_315 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135 = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_136 = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_316 = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_317 = torch.ops.aten.mul.Tensor(clone_90, 0.9);  clone_90 = None
        add_241 = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137 = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_318 = torch.ops.aten.mul.Tensor(squeeze_137, 1.0019569471624266);  squeeze_137 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320 = torch.ops.aten.mul.Tensor(clone_91, 0.9);  clone_91 = None
        add_242 = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(primals_289, -1)
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(primals_290, -1);  primals_290 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        add_243 = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        relu_41 = torch.ops.aten.relu.default(add_243);  add_243 = None
        convolution_46 = torch.ops.aten.convolution.default(relu_41, primals_47, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_244 = torch.ops.aten.add.Tensor(primals_291, 1);  primals_291 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem_92 = var_mean_46[0]
        getitem_93 = var_mean_46[1];  var_mean_46 = None
        add_245 = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        sub_46 = torch.ops.aten.sub.Tensor(convolution_46, getitem_93)
        mul_322 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138 = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_139 = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323 = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_324 = torch.ops.aten.mul.Tensor(clone_92, 0.9);  clone_92 = None
        add_246 = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140 = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_325 = torch.ops.aten.mul.Tensor(squeeze_140, 1.0019569471624266);  squeeze_140 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327 = torch.ops.aten.mul.Tensor(clone_93, 0.9);  clone_93 = None
        add_247 = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        add_248 = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        add_249 = torch.ops.aten.add.Tensor(add_248, relu_39);  add_248 = None
        relu_42 = torch.ops.aten.relu.default(add_249);  add_249 = None
        convolution_47 = torch.ops.aten.convolution.default(relu_42, primals_48, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_250 = torch.ops.aten.add.Tensor(primals_296, 1);  primals_296 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem_94 = var_mean_47[0]
        getitem_95 = var_mean_47[1];  var_mean_47 = None
        add_251 = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_47 = torch.ops.aten.sub.Tensor(convolution_47, getitem_95)
        mul_329 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141 = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_142 = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330 = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_331 = torch.ops.aten.mul.Tensor(clone_94, 0.9);  clone_94 = None
        add_252 = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143 = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_332 = torch.ops.aten.mul.Tensor(squeeze_143, 1.0019569471624266);  squeeze_143 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334 = torch.ops.aten.mul.Tensor(clone_95, 0.9);  clone_95 = None
        add_253 = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(primals_299, -1)
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(primals_300, -1);  primals_300 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        add_254 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        relu_43 = torch.ops.aten.relu.default(add_254);  add_254 = None
        convolution_48 = torch.ops.aten.convolution.default(relu_43, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_255 = torch.ops.aten.add.Tensor(primals_301, 1);  primals_301 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96 = var_mean_48[0]
        getitem_97 = var_mean_48[1];  var_mean_48 = None
        add_256 = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        sub_48 = torch.ops.aten.sub.Tensor(convolution_48, getitem_97)
        mul_336 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144 = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_145 = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_337 = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_338 = torch.ops.aten.mul.Tensor(clone_96, 0.9);  clone_96 = None
        add_257 = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146 = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_339 = torch.ops.aten.mul.Tensor(squeeze_146, 1.0019569471624266);  squeeze_146 = None
        mul_340 = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341 = torch.ops.aten.mul.Tensor(clone_97, 0.9);  clone_97 = None
        add_258 = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(primals_304, -1)
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(primals_305, -1);  primals_305 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        add_259 = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        relu_44 = torch.ops.aten.relu.default(add_259);  add_259 = None
        convolution_49 = torch.ops.aten.convolution.default(relu_44, primals_50, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_260 = torch.ops.aten.add.Tensor(primals_306, 1);  primals_306 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem_98 = var_mean_49[0]
        getitem_99 = var_mean_49[1];  var_mean_49 = None
        add_261 = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_49, getitem_99)
        mul_343 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147 = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_148 = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_344 = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_345 = torch.ops.aten.mul.Tensor(clone_98, 0.9);  clone_98 = None
        add_262 = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149 = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_346 = torch.ops.aten.mul.Tensor(squeeze_149, 1.0019569471624266);  squeeze_149 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348 = torch.ops.aten.mul.Tensor(clone_99, 0.9);  clone_99 = None
        add_263 = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(primals_309, -1)
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(primals_310, -1);  primals_310 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        add_264 = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        add_265 = torch.ops.aten.add.Tensor(add_264, relu_42);  add_264 = None
        relu_45 = torch.ops.aten.relu.default(add_265);  add_265 = None
        convolution_50 = torch.ops.aten.convolution.default(relu_45, primals_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_266 = torch.ops.aten.add.Tensor(primals_311, 1);  primals_311 = None
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_100 = var_mean_50[0]
        getitem_101 = var_mean_50[1];  var_mean_50 = None
        add_267 = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_50, getitem_101)
        mul_350 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150 = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_151 = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_351 = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_352 = torch.ops.aten.mul.Tensor(clone_100, 0.9);  clone_100 = None
        add_268 = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152 = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_353 = torch.ops.aten.mul.Tensor(squeeze_152, 1.0019569471624266);  squeeze_152 = None
        mul_354 = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355 = torch.ops.aten.mul.Tensor(clone_101, 0.9);  clone_101 = None
        add_269 = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(primals_314, -1)
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(primals_315, -1);  primals_315 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        add_270 = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        relu_46 = torch.ops.aten.relu.default(add_270);  add_270 = None
        convolution_51 = torch.ops.aten.convolution.default(relu_46, primals_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_271 = torch.ops.aten.add.Tensor(primals_316, 1);  primals_316 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem_102 = var_mean_51[0]
        getitem_103 = var_mean_51[1];  var_mean_51 = None
        add_272 = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_51, getitem_103)
        mul_357 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153 = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_154 = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_358 = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_359 = torch.ops.aten.mul.Tensor(clone_102, 0.9);  clone_102 = None
        add_273 = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155 = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_360 = torch.ops.aten.mul.Tensor(squeeze_155, 1.0019569471624266);  squeeze_155 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362 = torch.ops.aten.mul.Tensor(clone_103, 0.9);  clone_103 = None
        add_274 = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(primals_319, -1)
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(primals_320, -1);  primals_320 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        add_275 = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        relu_47 = torch.ops.aten.relu.default(add_275);  add_275 = None
        convolution_52 = torch.ops.aten.convolution.default(relu_47, primals_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_276 = torch.ops.aten.add.Tensor(primals_321, 1);  primals_321 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
        getitem_104 = var_mean_52[0]
        getitem_105 = var_mean_52[1];  var_mean_52 = None
        add_277 = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_52, getitem_105)
        mul_364 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156 = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_157 = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365 = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_366 = torch.ops.aten.mul.Tensor(clone_104, 0.9);  clone_104 = None
        add_278 = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158 = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_367 = torch.ops.aten.mul.Tensor(squeeze_158, 1.0019569471624266);  squeeze_158 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369 = torch.ops.aten.mul.Tensor(clone_105, 0.9);  clone_105 = None
        add_279 = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(primals_324, -1)
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(primals_325, -1);  primals_325 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_370 = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        add_280 = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        add_281 = torch.ops.aten.add.Tensor(add_280, relu_45);  add_280 = None
        relu_48 = torch.ops.aten.relu.default(add_281);  add_281 = None
        convolution_53 = torch.ops.aten.convolution.default(relu_48, primals_54, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_282 = torch.ops.aten.add.Tensor(primals_326, 1);  primals_326 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
        getitem_106 = var_mean_53[0]
        getitem_107 = var_mean_53[1];  var_mean_53 = None
        add_283 = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        sub_53 = torch.ops.aten.sub.Tensor(convolution_53, getitem_107)
        mul_371 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_159 = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_160 = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_372 = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
        mul_373 = torch.ops.aten.mul.Tensor(clone_106, 0.9);  clone_106 = None
        add_284 = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
        squeeze_161 = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_374 = torch.ops.aten.mul.Tensor(squeeze_161, 1.0019569471624266);  squeeze_161 = None
        mul_375 = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
        mul_376 = torch.ops.aten.mul.Tensor(clone_107, 0.9);  clone_107 = None
        add_285 = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(primals_329, -1)
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(primals_330, -1);  primals_330 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
        add_286 = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
        relu_49 = torch.ops.aten.relu.default(add_286);  add_286 = None
        convolution_54 = torch.ops.aten.convolution.default(relu_49, primals_55, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
        add_287 = torch.ops.aten.add.Tensor(primals_331, 1);  primals_331 = None
        var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
        getitem_108 = var_mean_54[0]
        getitem_109 = var_mean_54[1];  var_mean_54 = None
        add_288 = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
        sub_54 = torch.ops.aten.sub.Tensor(convolution_54, getitem_109)
        mul_378 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_162 = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_163 = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_379 = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
        mul_380 = torch.ops.aten.mul.Tensor(clone_108, 0.9);  clone_108 = None
        add_289 = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        squeeze_164 = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_381 = torch.ops.aten.mul.Tensor(squeeze_164, 1.0019569471624266);  squeeze_164 = None
        mul_382 = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
        mul_383 = torch.ops.aten.mul.Tensor(clone_109, 0.9);  clone_109 = None
        add_290 = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(primals_334, -1)
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(primals_335, -1);  primals_335 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
        add_291 = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
        relu_50 = torch.ops.aten.relu.default(add_291);  add_291 = None
        convolution_55 = torch.ops.aten.convolution.default(relu_50, primals_56, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_292 = torch.ops.aten.add.Tensor(primals_336, 1);  primals_336 = None
        var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
        getitem_110 = var_mean_55[0]
        getitem_111 = var_mean_55[1];  var_mean_55 = None
        add_293 = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        sub_55 = torch.ops.aten.sub.Tensor(convolution_55, getitem_111)
        mul_385 = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_165 = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_166 = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_386 = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
        mul_387 = torch.ops.aten.mul.Tensor(clone_110, 0.9);  clone_110 = None
        add_294 = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        squeeze_167 = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_388 = torch.ops.aten.mul.Tensor(squeeze_167, 1.0019569471624266);  squeeze_167 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
        mul_390 = torch.ops.aten.mul.Tensor(clone_111, 0.9);  clone_111 = None
        add_295 = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(primals_339, -1)
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(primals_340, -1);  primals_340 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        mul_391 = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
        add_296 = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
        add_297 = torch.ops.aten.add.Tensor(add_296, relu_48);  add_296 = None
        relu_51 = torch.ops.aten.relu.default(add_297);  add_297 = None
        convolution_56 = torch.ops.aten.convolution.default(relu_51, primals_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_298 = torch.ops.aten.add.Tensor(primals_341, 1);  primals_341 = None
        var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
        getitem_112 = var_mean_56[0]
        getitem_113 = var_mean_56[1];  var_mean_56 = None
        add_299 = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_56 = torch.ops.aten.sub.Tensor(convolution_56, getitem_113)
        mul_392 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        squeeze_168 = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_169 = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_393 = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
        mul_394 = torch.ops.aten.mul.Tensor(clone_112, 0.9);  clone_112 = None
        add_300 = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
        squeeze_170 = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_395 = torch.ops.aten.mul.Tensor(squeeze_170, 1.0019569471624266);  squeeze_170 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
        mul_397 = torch.ops.aten.mul.Tensor(clone_113, 0.9);  clone_113 = None
        add_301 = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(primals_344, -1)
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(primals_345, -1);  primals_345 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_398 = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
        add_302 = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
        relu_52 = torch.ops.aten.relu.default(add_302);  add_302 = None
        mean = torch.ops.aten.mean.dim(relu_52, [-1, -2], True)
        view = torch.ops.aten.view.default(mean, [8, 2560]);  mean = None
        permute = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm = torch.ops.aten.addmm.default(primals_59, view, permute);  primals_59 = None
        permute_1 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        le = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, 2);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(unsqueeze_865, 3);  unsqueeze_865 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, 2);  unsqueeze_876 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(unsqueeze_877, 3);  unsqueeze_877 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, 2);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(unsqueeze_889, 3);  unsqueeze_889 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
        return [add_2, add_3, add_7, add_8, add_12, add_13, add_17, add_18, add_23, add_24, add_28, add_29, add_33, add_34, add_39, add_40, add_44, add_45, add_50, add_51, add_55, add_56, add_60, add_61, add_65, add_66, add_71, add_72, add_76, add_77, add_81, add_82, add_87, add_88, add_92, add_93, add_97, add_98, add_103, add_104, add_108, add_109, add_113, add_114, add_119, add_120, add_124, add_125, add_129, add_130, add_135, add_136, add_140, add_141, add_145, add_146, add_151, add_152, add_156, add_157, add_161, add_162, add_166, add_167, add_172, add_173, add_177, add_178, add_182, add_183, add_188, add_189, add_193, add_194, add_198, add_199, add_204, add_205, add_209, add_210, add_214, add_215, add_220, add_221, add_225, add_226, add_230, add_231, add_236, add_237, add_241, add_242, add_246, add_247, add_252, add_253, add_257, add_258, add_262, add_263, add_268, add_269, add_273, add_274, add_278, add_279, add_284, add_285, add_289, add_290, add_294, add_295, add_300, add_301, addmm, add, add_5, add_10, add_15, add_21, add_26, add_31, add_37, add_42, add_48, add_53, add_58, add_63, add_69, add_74, add_79, add_85, add_90, add_95, add_101, add_106, add_111, add_117, add_122, add_127, add_133, add_138, add_143, add_149, add_154, add_159, add_164, add_170, add_175, add_180, add_186, add_191, add_196, add_202, add_207, add_212, add_218, add_223, add_228, add_234, add_239, add_244, add_250, add_255, add_260, add_266, add_271, add_276, add_282, add_287, add_292, add_298, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_60, primals_64, primals_69, primals_74, primals_79, primals_84, primals_89, primals_94, primals_99, primals_104, primals_109, primals_114, primals_119, primals_124, primals_129, primals_134, primals_139, primals_144, primals_149, primals_154, primals_159, primals_164, primals_169, primals_174, primals_179, primals_184, primals_189, primals_194, primals_199, primals_204, primals_209, primals_214, primals_219, primals_224, primals_229, primals_234, primals_239, primals_244, primals_249, primals_254, primals_259, primals_264, primals_269, primals_274, primals_279, primals_284, primals_289, primals_294, primals_299, primals_304, primals_309, primals_314, primals_319, primals_324, primals_329, primals_334, primals_339, primals_344, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, view, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902]
        
args = [((32, 3, 3, 3), (27, 9, 3, 1), torch.float32, 'cuda'), ((128, 32, 3, 3), (288, 9, 3, 1), torch.float32, 'cuda'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((128, 32, 1, 1), (32, 1, 1, 1), torch.float32, 'cuda'), ((192, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((160, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((2560, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1000, 2560), (2560, 1), torch.float32, 'cuda'), ((1000,), (1,), torch.float32, 'cuda'), ((8, 3, 256, 256), (196608, 65536, 256, 1), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
