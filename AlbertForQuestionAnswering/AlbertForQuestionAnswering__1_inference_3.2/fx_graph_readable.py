class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[30000, 128], arg1_1: f32[512, 128], arg2_1: f32[2, 128], arg3_1: f32[128], arg4_1: f32[128], arg5_1: f32[4096, 128], arg6_1: f32[4096], arg7_1: f32[4096], arg8_1: f32[4096], arg9_1: f32[4096, 4096], arg10_1: f32[4096], arg11_1: f32[4096, 4096], arg12_1: f32[4096], arg13_1: f32[4096, 4096], arg14_1: f32[4096], arg15_1: f32[4096, 4096], arg16_1: f32[4096], arg17_1: f32[4096], arg18_1: f32[4096], arg19_1: f32[16384, 4096], arg20_1: f32[16384], arg21_1: f32[4096, 16384], arg22_1: f32[4096], arg23_1: f32[2, 4096], arg24_1: f32[2], arg25_1: f32[30000, 128], arg26_1: f32[512, 128], arg27_1: f32[2, 128], arg28_1: f32[128], arg29_1: f32[128], arg30_1: f32[4096, 128], arg31_1: f32[4096], arg32_1: f32[4096], arg33_1: f32[4096], arg34_1: f32[4096, 4096], arg35_1: f32[4096], arg36_1: f32[4096, 4096], arg37_1: f32[4096], arg38_1: f32[4096, 4096], arg39_1: f32[4096], arg40_1: f32[4096, 4096], arg41_1: f32[4096], arg42_1: f32[4096], arg43_1: f32[4096], arg44_1: f32[16384, 4096], arg45_1: f32[16384], arg46_1: f32[4096, 16384], arg47_1: f32[4096], arg48_1: f32[2, 4096], arg49_1: f32[2]):
        # File: /scratch/ngimel/work/pytorch/torch/optim/sgd.py:266, code: param.add_(d_p, alpha=-lr)
        mul: f32[30000, 128] = torch.ops.aten.mul.Tensor(arg25_1, -0.01);  arg25_1 = None
        add_: f32[30000, 128] = torch.ops.aten.add_.Tensor(arg0_1, mul);  arg0_1 = mul = None
        mul_1: f32[512, 128] = torch.ops.aten.mul.Tensor(arg26_1, -0.01);  arg26_1 = None
        add__1: f32[512, 128] = torch.ops.aten.add_.Tensor(arg1_1, mul_1);  arg1_1 = mul_1 = None
        mul_2: f32[2, 128] = torch.ops.aten.mul.Tensor(arg27_1, -0.01);  arg27_1 = None
        add__2: f32[2, 128] = torch.ops.aten.add_.Tensor(arg2_1, mul_2);  arg2_1 = mul_2 = None
        mul_3: f32[128] = torch.ops.aten.mul.Tensor(arg28_1, -0.01);  arg28_1 = None
        add__3: f32[128] = torch.ops.aten.add_.Tensor(arg3_1, mul_3);  arg3_1 = mul_3 = None
        mul_4: f32[128] = torch.ops.aten.mul.Tensor(arg29_1, -0.01);  arg29_1 = None
        add__4: f32[128] = torch.ops.aten.add_.Tensor(arg4_1, mul_4);  arg4_1 = mul_4 = None
        mul_5: f32[4096, 128] = torch.ops.aten.mul.Tensor(arg30_1, -0.01);  arg30_1 = None
        add__5: f32[4096, 128] = torch.ops.aten.add_.Tensor(arg5_1, mul_5);  arg5_1 = mul_5 = None
        mul_6: f32[4096] = torch.ops.aten.mul.Tensor(arg31_1, -0.01);  arg31_1 = None
        add__6: f32[4096] = torch.ops.aten.add_.Tensor(arg6_1, mul_6);  arg6_1 = mul_6 = None
        mul_7: f32[4096] = torch.ops.aten.mul.Tensor(arg32_1, -0.01);  arg32_1 = None
        add__7: f32[4096] = torch.ops.aten.add_.Tensor(arg7_1, mul_7);  arg7_1 = mul_7 = None
        mul_8: f32[4096] = torch.ops.aten.mul.Tensor(arg33_1, -0.01);  arg33_1 = None
        add__8: f32[4096] = torch.ops.aten.add_.Tensor(arg8_1, mul_8);  arg8_1 = mul_8 = None
        mul_9: f32[4096, 4096] = torch.ops.aten.mul.Tensor(arg34_1, -0.01);  arg34_1 = None
        add__9: f32[4096, 4096] = torch.ops.aten.add_.Tensor(arg9_1, mul_9);  arg9_1 = mul_9 = None
        mul_10: f32[4096] = torch.ops.aten.mul.Tensor(arg35_1, -0.01);  arg35_1 = None
        add__10: f32[4096] = torch.ops.aten.add_.Tensor(arg10_1, mul_10);  arg10_1 = mul_10 = None
        mul_11: f32[4096, 4096] = torch.ops.aten.mul.Tensor(arg36_1, -0.01);  arg36_1 = None
        add__11: f32[4096, 4096] = torch.ops.aten.add_.Tensor(arg11_1, mul_11);  arg11_1 = mul_11 = None
        mul_12: f32[4096] = torch.ops.aten.mul.Tensor(arg37_1, -0.01);  arg37_1 = None
        add__12: f32[4096] = torch.ops.aten.add_.Tensor(arg12_1, mul_12);  arg12_1 = mul_12 = None
        mul_13: f32[4096, 4096] = torch.ops.aten.mul.Tensor(arg38_1, -0.01);  arg38_1 = None
        add__13: f32[4096, 4096] = torch.ops.aten.add_.Tensor(arg13_1, mul_13);  arg13_1 = mul_13 = None
        mul_14: f32[4096] = torch.ops.aten.mul.Tensor(arg39_1, -0.01);  arg39_1 = None
        add__14: f32[4096] = torch.ops.aten.add_.Tensor(arg14_1, mul_14);  arg14_1 = mul_14 = None
        mul_15: f32[4096, 4096] = torch.ops.aten.mul.Tensor(arg40_1, -0.01);  arg40_1 = None
        add__15: f32[4096, 4096] = torch.ops.aten.add_.Tensor(arg15_1, mul_15);  arg15_1 = mul_15 = None
        mul_16: f32[4096] = torch.ops.aten.mul.Tensor(arg41_1, -0.01);  arg41_1 = None
        add__16: f32[4096] = torch.ops.aten.add_.Tensor(arg16_1, mul_16);  arg16_1 = mul_16 = None
        mul_17: f32[4096] = torch.ops.aten.mul.Tensor(arg42_1, -0.01);  arg42_1 = None
        add__17: f32[4096] = torch.ops.aten.add_.Tensor(arg17_1, mul_17);  arg17_1 = mul_17 = None
        mul_18: f32[4096] = torch.ops.aten.mul.Tensor(arg43_1, -0.01);  arg43_1 = None
        add__18: f32[4096] = torch.ops.aten.add_.Tensor(arg18_1, mul_18);  arg18_1 = mul_18 = None
        mul_19: f32[16384, 4096] = torch.ops.aten.mul.Tensor(arg44_1, -0.01);  arg44_1 = None
        add__19: f32[16384, 4096] = torch.ops.aten.add_.Tensor(arg19_1, mul_19);  arg19_1 = mul_19 = None
        mul_20: f32[16384] = torch.ops.aten.mul.Tensor(arg45_1, -0.01);  arg45_1 = None
        add__20: f32[16384] = torch.ops.aten.add_.Tensor(arg20_1, mul_20);  arg20_1 = mul_20 = None
        mul_21: f32[4096, 16384] = torch.ops.aten.mul.Tensor(arg46_1, -0.01);  arg46_1 = None
        add__21: f32[4096, 16384] = torch.ops.aten.add_.Tensor(arg21_1, mul_21);  arg21_1 = mul_21 = None
        mul_22: f32[4096] = torch.ops.aten.mul.Tensor(arg47_1, -0.01);  arg47_1 = None
        add__22: f32[4096] = torch.ops.aten.add_.Tensor(arg22_1, mul_22);  arg22_1 = mul_22 = None
        mul_23: f32[2, 4096] = torch.ops.aten.mul.Tensor(arg48_1, -0.01);  arg48_1 = None
        add__23: f32[2, 4096] = torch.ops.aten.add_.Tensor(arg23_1, mul_23);  arg23_1 = mul_23 = None
        mul_24: f32[2] = torch.ops.aten.mul.Tensor(arg49_1, -0.01);  arg49_1 = None
        add__24: f32[2] = torch.ops.aten.add_.Tensor(arg24_1, mul_24);  arg24_1 = mul_24 = None
        return ()
        