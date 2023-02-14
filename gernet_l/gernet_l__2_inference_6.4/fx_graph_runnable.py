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
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x12\x00\x00\x00use_dynamic_shapesq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x03\x00\x00\x00cseq\x08\x88X\x10\x00\x00\x00max_dist_from_bwq\tK\x03X\x0b\x00\x00\x00debug_jointq\n\x88X\x0c\x00\x00\x00debug_graphsq\x0b\x88X\x11\x00\x00\x00debug_partitionerq\x0c\x88X\t\x00\x00\x00log_levelq\rK\nu.')


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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1):
        mul = torch.ops.aten.mul.Tensor(arg173_1, -0.01);  arg173_1 = None
        add_ = torch.ops.aten.add_.Tensor(arg0_1, mul);  arg0_1 = mul = None
        mul_1 = torch.ops.aten.mul.Tensor(arg174_1, -0.01);  arg174_1 = None
        add__1 = torch.ops.aten.add_.Tensor(arg1_1, mul_1);  arg1_1 = mul_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(arg175_1, -0.01);  arg175_1 = None
        add__2 = torch.ops.aten.add_.Tensor(arg2_1, mul_2);  arg2_1 = mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(arg176_1, -0.01);  arg176_1 = None
        add__3 = torch.ops.aten.add_.Tensor(arg3_1, mul_3);  arg3_1 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(arg177_1, -0.01);  arg177_1 = None
        add__4 = torch.ops.aten.add_.Tensor(arg4_1, mul_4);  arg4_1 = mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(arg178_1, -0.01);  arg178_1 = None
        add__5 = torch.ops.aten.add_.Tensor(arg5_1, mul_5);  arg5_1 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(arg179_1, -0.01);  arg179_1 = None
        add__6 = torch.ops.aten.add_.Tensor(arg6_1, mul_6);  arg6_1 = mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(arg180_1, -0.01);  arg180_1 = None
        add__7 = torch.ops.aten.add_.Tensor(arg7_1, mul_7);  arg7_1 = mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(arg181_1, -0.01);  arg181_1 = None
        add__8 = torch.ops.aten.add_.Tensor(arg8_1, mul_8);  arg8_1 = mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(arg182_1, -0.01);  arg182_1 = None
        add__9 = torch.ops.aten.add_.Tensor(arg9_1, mul_9);  arg9_1 = mul_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(arg183_1, -0.01);  arg183_1 = None
        add__10 = torch.ops.aten.add_.Tensor(arg10_1, mul_10);  arg10_1 = mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(arg184_1, -0.01);  arg184_1 = None
        add__11 = torch.ops.aten.add_.Tensor(arg11_1, mul_11);  arg11_1 = mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(arg185_1, -0.01);  arg185_1 = None
        add__12 = torch.ops.aten.add_.Tensor(arg12_1, mul_12);  arg12_1 = mul_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(arg186_1, -0.01);  arg186_1 = None
        add__13 = torch.ops.aten.add_.Tensor(arg13_1, mul_13);  arg13_1 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(arg187_1, -0.01);  arg187_1 = None
        add__14 = torch.ops.aten.add_.Tensor(arg14_1, mul_14);  arg14_1 = mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(arg188_1, -0.01);  arg188_1 = None
        add__15 = torch.ops.aten.add_.Tensor(arg15_1, mul_15);  arg15_1 = mul_15 = None
        mul_16 = torch.ops.aten.mul.Tensor(arg189_1, -0.01);  arg189_1 = None
        add__16 = torch.ops.aten.add_.Tensor(arg16_1, mul_16);  arg16_1 = mul_16 = None
        mul_17 = torch.ops.aten.mul.Tensor(arg190_1, -0.01);  arg190_1 = None
        add__17 = torch.ops.aten.add_.Tensor(arg17_1, mul_17);  arg17_1 = mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(arg191_1, -0.01);  arg191_1 = None
        add__18 = torch.ops.aten.add_.Tensor(arg18_1, mul_18);  arg18_1 = mul_18 = None
        mul_19 = torch.ops.aten.mul.Tensor(arg192_1, -0.01);  arg192_1 = None
        add__19 = torch.ops.aten.add_.Tensor(arg19_1, mul_19);  arg19_1 = mul_19 = None
        mul_20 = torch.ops.aten.mul.Tensor(arg193_1, -0.01);  arg193_1 = None
        add__20 = torch.ops.aten.add_.Tensor(arg20_1, mul_20);  arg20_1 = mul_20 = None
        mul_21 = torch.ops.aten.mul.Tensor(arg194_1, -0.01);  arg194_1 = None
        add__21 = torch.ops.aten.add_.Tensor(arg21_1, mul_21);  arg21_1 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(arg195_1, -0.01);  arg195_1 = None
        add__22 = torch.ops.aten.add_.Tensor(arg22_1, mul_22);  arg22_1 = mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(arg196_1, -0.01);  arg196_1 = None
        add__23 = torch.ops.aten.add_.Tensor(arg23_1, mul_23);  arg23_1 = mul_23 = None
        mul_24 = torch.ops.aten.mul.Tensor(arg197_1, -0.01);  arg197_1 = None
        add__24 = torch.ops.aten.add_.Tensor(arg24_1, mul_24);  arg24_1 = mul_24 = None
        mul_25 = torch.ops.aten.mul.Tensor(arg198_1, -0.01);  arg198_1 = None
        add__25 = torch.ops.aten.add_.Tensor(arg25_1, mul_25);  arg25_1 = mul_25 = None
        mul_26 = torch.ops.aten.mul.Tensor(arg199_1, -0.01);  arg199_1 = None
        add__26 = torch.ops.aten.add_.Tensor(arg26_1, mul_26);  arg26_1 = mul_26 = None
        mul_27 = torch.ops.aten.mul.Tensor(arg200_1, -0.01);  arg200_1 = None
        add__27 = torch.ops.aten.add_.Tensor(arg27_1, mul_27);  arg27_1 = mul_27 = None
        mul_28 = torch.ops.aten.mul.Tensor(arg201_1, -0.01);  arg201_1 = None
        add__28 = torch.ops.aten.add_.Tensor(arg28_1, mul_28);  arg28_1 = mul_28 = None
        mul_29 = torch.ops.aten.mul.Tensor(arg202_1, -0.01);  arg202_1 = None
        add__29 = torch.ops.aten.add_.Tensor(arg29_1, mul_29);  arg29_1 = mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(arg203_1, -0.01);  arg203_1 = None
        add__30 = torch.ops.aten.add_.Tensor(arg30_1, mul_30);  arg30_1 = mul_30 = None
        mul_31 = torch.ops.aten.mul.Tensor(arg204_1, -0.01);  arg204_1 = None
        add__31 = torch.ops.aten.add_.Tensor(arg31_1, mul_31);  arg31_1 = mul_31 = None
        mul_32 = torch.ops.aten.mul.Tensor(arg205_1, -0.01);  arg205_1 = None
        add__32 = torch.ops.aten.add_.Tensor(arg32_1, mul_32);  arg32_1 = mul_32 = None
        mul_33 = torch.ops.aten.mul.Tensor(arg206_1, -0.01);  arg206_1 = None
        add__33 = torch.ops.aten.add_.Tensor(arg33_1, mul_33);  arg33_1 = mul_33 = None
        mul_34 = torch.ops.aten.mul.Tensor(arg207_1, -0.01);  arg207_1 = None
        add__34 = torch.ops.aten.add_.Tensor(arg34_1, mul_34);  arg34_1 = mul_34 = None
        mul_35 = torch.ops.aten.mul.Tensor(arg208_1, -0.01);  arg208_1 = None
        add__35 = torch.ops.aten.add_.Tensor(arg35_1, mul_35);  arg35_1 = mul_35 = None
        mul_36 = torch.ops.aten.mul.Tensor(arg209_1, -0.01);  arg209_1 = None
        add__36 = torch.ops.aten.add_.Tensor(arg36_1, mul_36);  arg36_1 = mul_36 = None
        mul_37 = torch.ops.aten.mul.Tensor(arg210_1, -0.01);  arg210_1 = None
        add__37 = torch.ops.aten.add_.Tensor(arg37_1, mul_37);  arg37_1 = mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(arg211_1, -0.01);  arg211_1 = None
        add__38 = torch.ops.aten.add_.Tensor(arg38_1, mul_38);  arg38_1 = mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(arg212_1, -0.01);  arg212_1 = None
        add__39 = torch.ops.aten.add_.Tensor(arg39_1, mul_39);  arg39_1 = mul_39 = None
        mul_40 = torch.ops.aten.mul.Tensor(arg213_1, -0.01);  arg213_1 = None
        add__40 = torch.ops.aten.add_.Tensor(arg40_1, mul_40);  arg40_1 = mul_40 = None
        mul_41 = torch.ops.aten.mul.Tensor(arg214_1, -0.01);  arg214_1 = None
        add__41 = torch.ops.aten.add_.Tensor(arg41_1, mul_41);  arg41_1 = mul_41 = None
        mul_42 = torch.ops.aten.mul.Tensor(arg215_1, -0.01);  arg215_1 = None
        add__42 = torch.ops.aten.add_.Tensor(arg42_1, mul_42);  arg42_1 = mul_42 = None
        mul_43 = torch.ops.aten.mul.Tensor(arg216_1, -0.01);  arg216_1 = None
        add__43 = torch.ops.aten.add_.Tensor(arg43_1, mul_43);  arg43_1 = mul_43 = None
        mul_44 = torch.ops.aten.mul.Tensor(arg217_1, -0.01);  arg217_1 = None
        add__44 = torch.ops.aten.add_.Tensor(arg44_1, mul_44);  arg44_1 = mul_44 = None
        mul_45 = torch.ops.aten.mul.Tensor(arg218_1, -0.01);  arg218_1 = None
        add__45 = torch.ops.aten.add_.Tensor(arg45_1, mul_45);  arg45_1 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(arg219_1, -0.01);  arg219_1 = None
        add__46 = torch.ops.aten.add_.Tensor(arg46_1, mul_46);  arg46_1 = mul_46 = None
        mul_47 = torch.ops.aten.mul.Tensor(arg220_1, -0.01);  arg220_1 = None
        add__47 = torch.ops.aten.add_.Tensor(arg47_1, mul_47);  arg47_1 = mul_47 = None
        mul_48 = torch.ops.aten.mul.Tensor(arg221_1, -0.01);  arg221_1 = None
        add__48 = torch.ops.aten.add_.Tensor(arg48_1, mul_48);  arg48_1 = mul_48 = None
        mul_49 = torch.ops.aten.mul.Tensor(arg222_1, -0.01);  arg222_1 = None
        add__49 = torch.ops.aten.add_.Tensor(arg49_1, mul_49);  arg49_1 = mul_49 = None
        mul_50 = torch.ops.aten.mul.Tensor(arg223_1, -0.01);  arg223_1 = None
        add__50 = torch.ops.aten.add_.Tensor(arg50_1, mul_50);  arg50_1 = mul_50 = None
        mul_51 = torch.ops.aten.mul.Tensor(arg224_1, -0.01);  arg224_1 = None
        add__51 = torch.ops.aten.add_.Tensor(arg51_1, mul_51);  arg51_1 = mul_51 = None
        mul_52 = torch.ops.aten.mul.Tensor(arg225_1, -0.01);  arg225_1 = None
        add__52 = torch.ops.aten.add_.Tensor(arg52_1, mul_52);  arg52_1 = mul_52 = None
        mul_53 = torch.ops.aten.mul.Tensor(arg226_1, -0.01);  arg226_1 = None
        add__53 = torch.ops.aten.add_.Tensor(arg53_1, mul_53);  arg53_1 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(arg227_1, -0.01);  arg227_1 = None
        add__54 = torch.ops.aten.add_.Tensor(arg54_1, mul_54);  arg54_1 = mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(arg228_1, -0.01);  arg228_1 = None
        add__55 = torch.ops.aten.add_.Tensor(arg55_1, mul_55);  arg55_1 = mul_55 = None
        mul_56 = torch.ops.aten.mul.Tensor(arg229_1, -0.01);  arg229_1 = None
        add__56 = torch.ops.aten.add_.Tensor(arg56_1, mul_56);  arg56_1 = mul_56 = None
        mul_57 = torch.ops.aten.mul.Tensor(arg230_1, -0.01);  arg230_1 = None
        add__57 = torch.ops.aten.add_.Tensor(arg57_1, mul_57);  arg57_1 = mul_57 = None
        mul_58 = torch.ops.aten.mul.Tensor(arg231_1, -0.01);  arg231_1 = None
        add__58 = torch.ops.aten.add_.Tensor(arg58_1, mul_58);  arg58_1 = mul_58 = None
        mul_59 = torch.ops.aten.mul.Tensor(arg232_1, -0.01);  arg232_1 = None
        add__59 = torch.ops.aten.add_.Tensor(arg59_1, mul_59);  arg59_1 = mul_59 = None
        mul_60 = torch.ops.aten.mul.Tensor(arg233_1, -0.01);  arg233_1 = None
        add__60 = torch.ops.aten.add_.Tensor(arg60_1, mul_60);  arg60_1 = mul_60 = None
        mul_61 = torch.ops.aten.mul.Tensor(arg234_1, -0.01);  arg234_1 = None
        add__61 = torch.ops.aten.add_.Tensor(arg61_1, mul_61);  arg61_1 = mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(arg235_1, -0.01);  arg235_1 = None
        add__62 = torch.ops.aten.add_.Tensor(arg62_1, mul_62);  arg62_1 = mul_62 = None
        mul_63 = torch.ops.aten.mul.Tensor(arg236_1, -0.01);  arg236_1 = None
        add__63 = torch.ops.aten.add_.Tensor(arg63_1, mul_63);  arg63_1 = mul_63 = None
        mul_64 = torch.ops.aten.mul.Tensor(arg237_1, -0.01);  arg237_1 = None
        add__64 = torch.ops.aten.add_.Tensor(arg64_1, mul_64);  arg64_1 = mul_64 = None
        mul_65 = torch.ops.aten.mul.Tensor(arg238_1, -0.01);  arg238_1 = None
        add__65 = torch.ops.aten.add_.Tensor(arg65_1, mul_65);  arg65_1 = mul_65 = None
        mul_66 = torch.ops.aten.mul.Tensor(arg239_1, -0.01);  arg239_1 = None
        add__66 = torch.ops.aten.add_.Tensor(arg66_1, mul_66);  arg66_1 = mul_66 = None
        mul_67 = torch.ops.aten.mul.Tensor(arg240_1, -0.01);  arg240_1 = None
        add__67 = torch.ops.aten.add_.Tensor(arg67_1, mul_67);  arg67_1 = mul_67 = None
        mul_68 = torch.ops.aten.mul.Tensor(arg241_1, -0.01);  arg241_1 = None
        add__68 = torch.ops.aten.add_.Tensor(arg68_1, mul_68);  arg68_1 = mul_68 = None
        mul_69 = torch.ops.aten.mul.Tensor(arg242_1, -0.01);  arg242_1 = None
        add__69 = torch.ops.aten.add_.Tensor(arg69_1, mul_69);  arg69_1 = mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(arg243_1, -0.01);  arg243_1 = None
        add__70 = torch.ops.aten.add_.Tensor(arg70_1, mul_70);  arg70_1 = mul_70 = None
        mul_71 = torch.ops.aten.mul.Tensor(arg244_1, -0.01);  arg244_1 = None
        add__71 = torch.ops.aten.add_.Tensor(arg71_1, mul_71);  arg71_1 = mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(arg245_1, -0.01);  arg245_1 = None
        add__72 = torch.ops.aten.add_.Tensor(arg72_1, mul_72);  arg72_1 = mul_72 = None
        mul_73 = torch.ops.aten.mul.Tensor(arg246_1, -0.01);  arg246_1 = None
        add__73 = torch.ops.aten.add_.Tensor(arg73_1, mul_73);  arg73_1 = mul_73 = None
        mul_74 = torch.ops.aten.mul.Tensor(arg247_1, -0.01);  arg247_1 = None
        add__74 = torch.ops.aten.add_.Tensor(arg74_1, mul_74);  arg74_1 = mul_74 = None
        mul_75 = torch.ops.aten.mul.Tensor(arg248_1, -0.01);  arg248_1 = None
        add__75 = torch.ops.aten.add_.Tensor(arg75_1, mul_75);  arg75_1 = mul_75 = None
        mul_76 = torch.ops.aten.mul.Tensor(arg249_1, -0.01);  arg249_1 = None
        add__76 = torch.ops.aten.add_.Tensor(arg76_1, mul_76);  arg76_1 = mul_76 = None
        mul_77 = torch.ops.aten.mul.Tensor(arg250_1, -0.01);  arg250_1 = None
        add__77 = torch.ops.aten.add_.Tensor(arg77_1, mul_77);  arg77_1 = mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(arg251_1, -0.01);  arg251_1 = None
        add__78 = torch.ops.aten.add_.Tensor(arg78_1, mul_78);  arg78_1 = mul_78 = None
        mul_79 = torch.ops.aten.mul.Tensor(arg252_1, -0.01);  arg252_1 = None
        add__79 = torch.ops.aten.add_.Tensor(arg79_1, mul_79);  arg79_1 = mul_79 = None
        mul_80 = torch.ops.aten.mul.Tensor(arg253_1, -0.01);  arg253_1 = None
        add__80 = torch.ops.aten.add_.Tensor(arg80_1, mul_80);  arg80_1 = mul_80 = None
        mul_81 = torch.ops.aten.mul.Tensor(arg254_1, -0.01);  arg254_1 = None
        add__81 = torch.ops.aten.add_.Tensor(arg81_1, mul_81);  arg81_1 = mul_81 = None
        mul_82 = torch.ops.aten.mul.Tensor(arg255_1, -0.01);  arg255_1 = None
        add__82 = torch.ops.aten.add_.Tensor(arg82_1, mul_82);  arg82_1 = mul_82 = None
        mul_83 = torch.ops.aten.mul.Tensor(arg256_1, -0.01);  arg256_1 = None
        add__83 = torch.ops.aten.add_.Tensor(arg83_1, mul_83);  arg83_1 = mul_83 = None
        mul_84 = torch.ops.aten.mul.Tensor(arg257_1, -0.01);  arg257_1 = None
        add__84 = torch.ops.aten.add_.Tensor(arg84_1, mul_84);  arg84_1 = mul_84 = None
        mul_85 = torch.ops.aten.mul.Tensor(arg258_1, -0.01);  arg258_1 = None
        add__85 = torch.ops.aten.add_.Tensor(arg85_1, mul_85);  arg85_1 = mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(arg259_1, -0.01);  arg259_1 = None
        add__86 = torch.ops.aten.add_.Tensor(arg86_1, mul_86);  arg86_1 = mul_86 = None
        mul_87 = torch.ops.aten.mul.Tensor(arg260_1, -0.01);  arg260_1 = None
        add__87 = torch.ops.aten.add_.Tensor(arg87_1, mul_87);  arg87_1 = mul_87 = None
        mul_88 = torch.ops.aten.mul.Tensor(arg261_1, -0.01);  arg261_1 = None
        add__88 = torch.ops.aten.add_.Tensor(arg88_1, mul_88);  arg88_1 = mul_88 = None
        mul_89 = torch.ops.aten.mul.Tensor(arg262_1, -0.01);  arg262_1 = None
        add__89 = torch.ops.aten.add_.Tensor(arg89_1, mul_89);  arg89_1 = mul_89 = None
        mul_90 = torch.ops.aten.mul.Tensor(arg263_1, -0.01);  arg263_1 = None
        add__90 = torch.ops.aten.add_.Tensor(arg90_1, mul_90);  arg90_1 = mul_90 = None
        mul_91 = torch.ops.aten.mul.Tensor(arg264_1, -0.01);  arg264_1 = None
        add__91 = torch.ops.aten.add_.Tensor(arg91_1, mul_91);  arg91_1 = mul_91 = None
        mul_92 = torch.ops.aten.mul.Tensor(arg265_1, -0.01);  arg265_1 = None
        add__92 = torch.ops.aten.add_.Tensor(arg92_1, mul_92);  arg92_1 = mul_92 = None
        mul_93 = torch.ops.aten.mul.Tensor(arg266_1, -0.01);  arg266_1 = None
        add__93 = torch.ops.aten.add_.Tensor(arg93_1, mul_93);  arg93_1 = mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(arg267_1, -0.01);  arg267_1 = None
        add__94 = torch.ops.aten.add_.Tensor(arg94_1, mul_94);  arg94_1 = mul_94 = None
        mul_95 = torch.ops.aten.mul.Tensor(arg268_1, -0.01);  arg268_1 = None
        add__95 = torch.ops.aten.add_.Tensor(arg95_1, mul_95);  arg95_1 = mul_95 = None
        mul_96 = torch.ops.aten.mul.Tensor(arg269_1, -0.01);  arg269_1 = None
        add__96 = torch.ops.aten.add_.Tensor(arg96_1, mul_96);  arg96_1 = mul_96 = None
        mul_97 = torch.ops.aten.mul.Tensor(arg270_1, -0.01);  arg270_1 = None
        add__97 = torch.ops.aten.add_.Tensor(arg97_1, mul_97);  arg97_1 = mul_97 = None
        mul_98 = torch.ops.aten.mul.Tensor(arg271_1, -0.01);  arg271_1 = None
        add__98 = torch.ops.aten.add_.Tensor(arg98_1, mul_98);  arg98_1 = mul_98 = None
        mul_99 = torch.ops.aten.mul.Tensor(arg272_1, -0.01);  arg272_1 = None
        add__99 = torch.ops.aten.add_.Tensor(arg99_1, mul_99);  arg99_1 = mul_99 = None
        mul_100 = torch.ops.aten.mul.Tensor(arg273_1, -0.01);  arg273_1 = None
        add__100 = torch.ops.aten.add_.Tensor(arg100_1, mul_100);  arg100_1 = mul_100 = None
        mul_101 = torch.ops.aten.mul.Tensor(arg274_1, -0.01);  arg274_1 = None
        add__101 = torch.ops.aten.add_.Tensor(arg101_1, mul_101);  arg101_1 = mul_101 = None
        mul_102 = torch.ops.aten.mul.Tensor(arg275_1, -0.01);  arg275_1 = None
        add__102 = torch.ops.aten.add_.Tensor(arg102_1, mul_102);  arg102_1 = mul_102 = None
        mul_103 = torch.ops.aten.mul.Tensor(arg276_1, -0.01);  arg276_1 = None
        add__103 = torch.ops.aten.add_.Tensor(arg103_1, mul_103);  arg103_1 = mul_103 = None
        mul_104 = torch.ops.aten.mul.Tensor(arg277_1, -0.01);  arg277_1 = None
        add__104 = torch.ops.aten.add_.Tensor(arg104_1, mul_104);  arg104_1 = mul_104 = None
        mul_105 = torch.ops.aten.mul.Tensor(arg278_1, -0.01);  arg278_1 = None
        add__105 = torch.ops.aten.add_.Tensor(arg105_1, mul_105);  arg105_1 = mul_105 = None
        mul_106 = torch.ops.aten.mul.Tensor(arg279_1, -0.01);  arg279_1 = None
        add__106 = torch.ops.aten.add_.Tensor(arg106_1, mul_106);  arg106_1 = mul_106 = None
        mul_107 = torch.ops.aten.mul.Tensor(arg280_1, -0.01);  arg280_1 = None
        add__107 = torch.ops.aten.add_.Tensor(arg107_1, mul_107);  arg107_1 = mul_107 = None
        mul_108 = torch.ops.aten.mul.Tensor(arg281_1, -0.01);  arg281_1 = None
        add__108 = torch.ops.aten.add_.Tensor(arg108_1, mul_108);  arg108_1 = mul_108 = None
        mul_109 = torch.ops.aten.mul.Tensor(arg282_1, -0.01);  arg282_1 = None
        add__109 = torch.ops.aten.add_.Tensor(arg109_1, mul_109);  arg109_1 = mul_109 = None
        mul_110 = torch.ops.aten.mul.Tensor(arg283_1, -0.01);  arg283_1 = None
        add__110 = torch.ops.aten.add_.Tensor(arg110_1, mul_110);  arg110_1 = mul_110 = None
        mul_111 = torch.ops.aten.mul.Tensor(arg284_1, -0.01);  arg284_1 = None
        add__111 = torch.ops.aten.add_.Tensor(arg111_1, mul_111);  arg111_1 = mul_111 = None
        mul_112 = torch.ops.aten.mul.Tensor(arg285_1, -0.01);  arg285_1 = None
        add__112 = torch.ops.aten.add_.Tensor(arg112_1, mul_112);  arg112_1 = mul_112 = None
        mul_113 = torch.ops.aten.mul.Tensor(arg286_1, -0.01);  arg286_1 = None
        add__113 = torch.ops.aten.add_.Tensor(arg113_1, mul_113);  arg113_1 = mul_113 = None
        mul_114 = torch.ops.aten.mul.Tensor(arg287_1, -0.01);  arg287_1 = None
        add__114 = torch.ops.aten.add_.Tensor(arg114_1, mul_114);  arg114_1 = mul_114 = None
        mul_115 = torch.ops.aten.mul.Tensor(arg288_1, -0.01);  arg288_1 = None
        add__115 = torch.ops.aten.add_.Tensor(arg115_1, mul_115);  arg115_1 = mul_115 = None
        mul_116 = torch.ops.aten.mul.Tensor(arg289_1, -0.01);  arg289_1 = None
        add__116 = torch.ops.aten.add_.Tensor(arg116_1, mul_116);  arg116_1 = mul_116 = None
        mul_117 = torch.ops.aten.mul.Tensor(arg290_1, -0.01);  arg290_1 = None
        add__117 = torch.ops.aten.add_.Tensor(arg117_1, mul_117);  arg117_1 = mul_117 = None
        mul_118 = torch.ops.aten.mul.Tensor(arg291_1, -0.01);  arg291_1 = None
        add__118 = torch.ops.aten.add_.Tensor(arg118_1, mul_118);  arg118_1 = mul_118 = None
        mul_119 = torch.ops.aten.mul.Tensor(arg292_1, -0.01);  arg292_1 = None
        add__119 = torch.ops.aten.add_.Tensor(arg119_1, mul_119);  arg119_1 = mul_119 = None
        mul_120 = torch.ops.aten.mul.Tensor(arg293_1, -0.01);  arg293_1 = None
        add__120 = torch.ops.aten.add_.Tensor(arg120_1, mul_120);  arg120_1 = mul_120 = None
        mul_121 = torch.ops.aten.mul.Tensor(arg294_1, -0.01);  arg294_1 = None
        add__121 = torch.ops.aten.add_.Tensor(arg121_1, mul_121);  arg121_1 = mul_121 = None
        mul_122 = torch.ops.aten.mul.Tensor(arg295_1, -0.01);  arg295_1 = None
        add__122 = torch.ops.aten.add_.Tensor(arg122_1, mul_122);  arg122_1 = mul_122 = None
        mul_123 = torch.ops.aten.mul.Tensor(arg296_1, -0.01);  arg296_1 = None
        add__123 = torch.ops.aten.add_.Tensor(arg123_1, mul_123);  arg123_1 = mul_123 = None
        mul_124 = torch.ops.aten.mul.Tensor(arg297_1, -0.01);  arg297_1 = None
        add__124 = torch.ops.aten.add_.Tensor(arg124_1, mul_124);  arg124_1 = mul_124 = None
        mul_125 = torch.ops.aten.mul.Tensor(arg298_1, -0.01);  arg298_1 = None
        add__125 = torch.ops.aten.add_.Tensor(arg125_1, mul_125);  arg125_1 = mul_125 = None
        mul_126 = torch.ops.aten.mul.Tensor(arg299_1, -0.01);  arg299_1 = None
        add__126 = torch.ops.aten.add_.Tensor(arg126_1, mul_126);  arg126_1 = mul_126 = None
        mul_127 = torch.ops.aten.mul.Tensor(arg300_1, -0.01);  arg300_1 = None
        add__127 = torch.ops.aten.add_.Tensor(arg127_1, mul_127);  arg127_1 = mul_127 = None
        mul_128 = torch.ops.aten.mul.Tensor(arg301_1, -0.01);  arg301_1 = None
        add__128 = torch.ops.aten.add_.Tensor(arg128_1, mul_128);  arg128_1 = mul_128 = None
        mul_129 = torch.ops.aten.mul.Tensor(arg302_1, -0.01);  arg302_1 = None
        add__129 = torch.ops.aten.add_.Tensor(arg129_1, mul_129);  arg129_1 = mul_129 = None
        mul_130 = torch.ops.aten.mul.Tensor(arg303_1, -0.01);  arg303_1 = None
        add__130 = torch.ops.aten.add_.Tensor(arg130_1, mul_130);  arg130_1 = mul_130 = None
        mul_131 = torch.ops.aten.mul.Tensor(arg304_1, -0.01);  arg304_1 = None
        add__131 = torch.ops.aten.add_.Tensor(arg131_1, mul_131);  arg131_1 = mul_131 = None
        mul_132 = torch.ops.aten.mul.Tensor(arg305_1, -0.01);  arg305_1 = None
        add__132 = torch.ops.aten.add_.Tensor(arg132_1, mul_132);  arg132_1 = mul_132 = None
        mul_133 = torch.ops.aten.mul.Tensor(arg306_1, -0.01);  arg306_1 = None
        add__133 = torch.ops.aten.add_.Tensor(arg133_1, mul_133);  arg133_1 = mul_133 = None
        mul_134 = torch.ops.aten.mul.Tensor(arg307_1, -0.01);  arg307_1 = None
        add__134 = torch.ops.aten.add_.Tensor(arg134_1, mul_134);  arg134_1 = mul_134 = None
        mul_135 = torch.ops.aten.mul.Tensor(arg308_1, -0.01);  arg308_1 = None
        add__135 = torch.ops.aten.add_.Tensor(arg135_1, mul_135);  arg135_1 = mul_135 = None
        mul_136 = torch.ops.aten.mul.Tensor(arg309_1, -0.01);  arg309_1 = None
        add__136 = torch.ops.aten.add_.Tensor(arg136_1, mul_136);  arg136_1 = mul_136 = None
        mul_137 = torch.ops.aten.mul.Tensor(arg310_1, -0.01);  arg310_1 = None
        add__137 = torch.ops.aten.add_.Tensor(arg137_1, mul_137);  arg137_1 = mul_137 = None
        mul_138 = torch.ops.aten.mul.Tensor(arg311_1, -0.01);  arg311_1 = None
        add__138 = torch.ops.aten.add_.Tensor(arg138_1, mul_138);  arg138_1 = mul_138 = None
        mul_139 = torch.ops.aten.mul.Tensor(arg312_1, -0.01);  arg312_1 = None
        add__139 = torch.ops.aten.add_.Tensor(arg139_1, mul_139);  arg139_1 = mul_139 = None
        mul_140 = torch.ops.aten.mul.Tensor(arg313_1, -0.01);  arg313_1 = None
        add__140 = torch.ops.aten.add_.Tensor(arg140_1, mul_140);  arg140_1 = mul_140 = None
        mul_141 = torch.ops.aten.mul.Tensor(arg314_1, -0.01);  arg314_1 = None
        add__141 = torch.ops.aten.add_.Tensor(arg141_1, mul_141);  arg141_1 = mul_141 = None
        mul_142 = torch.ops.aten.mul.Tensor(arg315_1, -0.01);  arg315_1 = None
        add__142 = torch.ops.aten.add_.Tensor(arg142_1, mul_142);  arg142_1 = mul_142 = None
        mul_143 = torch.ops.aten.mul.Tensor(arg316_1, -0.01);  arg316_1 = None
        add__143 = torch.ops.aten.add_.Tensor(arg143_1, mul_143);  arg143_1 = mul_143 = None
        mul_144 = torch.ops.aten.mul.Tensor(arg317_1, -0.01);  arg317_1 = None
        add__144 = torch.ops.aten.add_.Tensor(arg144_1, mul_144);  arg144_1 = mul_144 = None
        mul_145 = torch.ops.aten.mul.Tensor(arg318_1, -0.01);  arg318_1 = None
        add__145 = torch.ops.aten.add_.Tensor(arg145_1, mul_145);  arg145_1 = mul_145 = None
        mul_146 = torch.ops.aten.mul.Tensor(arg319_1, -0.01);  arg319_1 = None
        add__146 = torch.ops.aten.add_.Tensor(arg146_1, mul_146);  arg146_1 = mul_146 = None
        mul_147 = torch.ops.aten.mul.Tensor(arg320_1, -0.01);  arg320_1 = None
        add__147 = torch.ops.aten.add_.Tensor(arg147_1, mul_147);  arg147_1 = mul_147 = None
        mul_148 = torch.ops.aten.mul.Tensor(arg321_1, -0.01);  arg321_1 = None
        add__148 = torch.ops.aten.add_.Tensor(arg148_1, mul_148);  arg148_1 = mul_148 = None
        mul_149 = torch.ops.aten.mul.Tensor(arg322_1, -0.01);  arg322_1 = None
        add__149 = torch.ops.aten.add_.Tensor(arg149_1, mul_149);  arg149_1 = mul_149 = None
        mul_150 = torch.ops.aten.mul.Tensor(arg323_1, -0.01);  arg323_1 = None
        add__150 = torch.ops.aten.add_.Tensor(arg150_1, mul_150);  arg150_1 = mul_150 = None
        mul_151 = torch.ops.aten.mul.Tensor(arg324_1, -0.01);  arg324_1 = None
        add__151 = torch.ops.aten.add_.Tensor(arg151_1, mul_151);  arg151_1 = mul_151 = None
        mul_152 = torch.ops.aten.mul.Tensor(arg325_1, -0.01);  arg325_1 = None
        add__152 = torch.ops.aten.add_.Tensor(arg152_1, mul_152);  arg152_1 = mul_152 = None
        mul_153 = torch.ops.aten.mul.Tensor(arg326_1, -0.01);  arg326_1 = None
        add__153 = torch.ops.aten.add_.Tensor(arg153_1, mul_153);  arg153_1 = mul_153 = None
        mul_154 = torch.ops.aten.mul.Tensor(arg327_1, -0.01);  arg327_1 = None
        add__154 = torch.ops.aten.add_.Tensor(arg154_1, mul_154);  arg154_1 = mul_154 = None
        mul_155 = torch.ops.aten.mul.Tensor(arg328_1, -0.01);  arg328_1 = None
        add__155 = torch.ops.aten.add_.Tensor(arg155_1, mul_155);  arg155_1 = mul_155 = None
        mul_156 = torch.ops.aten.mul.Tensor(arg329_1, -0.01);  arg329_1 = None
        add__156 = torch.ops.aten.add_.Tensor(arg156_1, mul_156);  arg156_1 = mul_156 = None
        mul_157 = torch.ops.aten.mul.Tensor(arg330_1, -0.01);  arg330_1 = None
        add__157 = torch.ops.aten.add_.Tensor(arg157_1, mul_157);  arg157_1 = mul_157 = None
        mul_158 = torch.ops.aten.mul.Tensor(arg331_1, -0.01);  arg331_1 = None
        add__158 = torch.ops.aten.add_.Tensor(arg158_1, mul_158);  arg158_1 = mul_158 = None
        mul_159 = torch.ops.aten.mul.Tensor(arg332_1, -0.01);  arg332_1 = None
        add__159 = torch.ops.aten.add_.Tensor(arg159_1, mul_159);  arg159_1 = mul_159 = None
        mul_160 = torch.ops.aten.mul.Tensor(arg333_1, -0.01);  arg333_1 = None
        add__160 = torch.ops.aten.add_.Tensor(arg160_1, mul_160);  arg160_1 = mul_160 = None
        mul_161 = torch.ops.aten.mul.Tensor(arg334_1, -0.01);  arg334_1 = None
        add__161 = torch.ops.aten.add_.Tensor(arg161_1, mul_161);  arg161_1 = mul_161 = None
        mul_162 = torch.ops.aten.mul.Tensor(arg335_1, -0.01);  arg335_1 = None
        add__162 = torch.ops.aten.add_.Tensor(arg162_1, mul_162);  arg162_1 = mul_162 = None
        mul_163 = torch.ops.aten.mul.Tensor(arg336_1, -0.01);  arg336_1 = None
        add__163 = torch.ops.aten.add_.Tensor(arg163_1, mul_163);  arg163_1 = mul_163 = None
        mul_164 = torch.ops.aten.mul.Tensor(arg337_1, -0.01);  arg337_1 = None
        add__164 = torch.ops.aten.add_.Tensor(arg164_1, mul_164);  arg164_1 = mul_164 = None
        mul_165 = torch.ops.aten.mul.Tensor(arg338_1, -0.01);  arg338_1 = None
        add__165 = torch.ops.aten.add_.Tensor(arg165_1, mul_165);  arg165_1 = mul_165 = None
        mul_166 = torch.ops.aten.mul.Tensor(arg339_1, -0.01);  arg339_1 = None
        add__166 = torch.ops.aten.add_.Tensor(arg166_1, mul_166);  arg166_1 = mul_166 = None
        mul_167 = torch.ops.aten.mul.Tensor(arg340_1, -0.01);  arg340_1 = None
        add__167 = torch.ops.aten.add_.Tensor(arg167_1, mul_167);  arg167_1 = mul_167 = None
        mul_168 = torch.ops.aten.mul.Tensor(arg341_1, -0.01);  arg341_1 = None
        add__168 = torch.ops.aten.add_.Tensor(arg168_1, mul_168);  arg168_1 = mul_168 = None
        mul_169 = torch.ops.aten.mul.Tensor(arg342_1, -0.01);  arg342_1 = None
        add__169 = torch.ops.aten.add_.Tensor(arg169_1, mul_169);  arg169_1 = mul_169 = None
        mul_170 = torch.ops.aten.mul.Tensor(arg343_1, -0.01);  arg343_1 = None
        add__170 = torch.ops.aten.add_.Tensor(arg170_1, mul_170);  arg170_1 = mul_170 = None
        mul_171 = torch.ops.aten.mul.Tensor(arg344_1, -0.01);  arg344_1 = None
        add__171 = torch.ops.aten.add_.Tensor(arg171_1, mul_171);  arg171_1 = mul_171 = None
        mul_172 = torch.ops.aten.mul.Tensor(arg345_1, -0.01);  arg345_1 = None
        add__172 = torch.ops.aten.add_.Tensor(arg172_1, mul_172);  arg172_1 = mul_172 = None
        return ()
        
args = [((32, 3, 3, 3), (27, 9, 3, 1), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((128, 32, 1, 1), (32, 1, 1, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128, 32, 3, 3), (288, 9, 3, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((192, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((640, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((2560, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((1000, 2560), (2560, 1), torch.float32, 'cuda'), ((1000,), (1,), torch.float32, 'cuda'), ((32, 3, 3, 3), (27, 9, 3, 1), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((32,), (1,), torch.float32, 'cuda'), ((128, 32, 1, 1), (32, 1, 1, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128, 32, 3, 3), (288, 9, 3, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((192, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192, 192, 3, 3), (1728, 9, 3, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((640, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 192, 1, 1), (192, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((160, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160, 160, 3, 3), (1440, 9, 3, 1), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((160,), (1,), torch.float32, 'cuda'), ((640, 160, 1, 1), (160, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((1920, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((1920,), (1,), torch.float32, 'cuda'), ((640, 1920, 1, 1), (1920, 1, 1, 1), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((640,), (1,), torch.float32, 'cuda'), ((2560, 640, 1, 1), (640, 1, 1, 1), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((2560,), (1,), torch.float32, 'cuda'), ((1000, 2560), (2560, 1), torch.float32, 'cuda'), ((1000,), (1,), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced
