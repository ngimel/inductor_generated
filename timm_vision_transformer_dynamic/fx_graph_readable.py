class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f16[1, 1, 384], arg1_1: f16[1, 197, 384], arg2_1: f16[384, 3, 16, 16], arg3_1: f16[384], arg4_1: f16[384], arg5_1: f16[384], arg6_1: f16[1152, 384], arg7_1: f16[1152], arg8_1: f16[384, 384], arg9_1: f16[384], arg10_1: f16[384], arg11_1: f16[384], arg12_1: f16[1536, 384], arg13_1: f16[1536], arg14_1: f16[384, 1536], arg15_1: f16[384], arg16_1: f16[384], arg17_1: f16[384], arg18_1: f16[1152, 384], arg19_1: f16[1152], arg20_1: f16[384, 384], arg21_1: f16[384], arg22_1: f16[384], arg23_1: f16[384], arg24_1: f16[1536, 384], arg25_1: f16[1536], arg26_1: f16[384, 1536], arg27_1: f16[384], arg28_1: f16[384], arg29_1: f16[384], arg30_1: f16[1152, 384], arg31_1: f16[1152], arg32_1: f16[384, 384], arg33_1: f16[384], arg34_1: f16[384], arg35_1: f16[384], arg36_1: f16[1536, 384], arg37_1: f16[1536], arg38_1: f16[384, 1536], arg39_1: f16[384], arg40_1: f16[384], arg41_1: f16[384], arg42_1: f16[1152, 384], arg43_1: f16[1152], arg44_1: f16[384, 384], arg45_1: f16[384], arg46_1: f16[384], arg47_1: f16[384], arg48_1: f16[1536, 384], arg49_1: f16[1536], arg50_1: f16[384, 1536], arg51_1: f16[384], arg52_1: f16[384], arg53_1: f16[384], arg54_1: f16[1152, 384], arg55_1: f16[1152], arg56_1: f16[384, 384], arg57_1: f16[384], arg58_1: f16[384], arg59_1: f16[384], arg60_1: f16[1536, 384], arg61_1: f16[1536], arg62_1: f16[384, 1536], arg63_1: f16[384], arg64_1: f16[384], arg65_1: f16[384], arg66_1: f16[1152, 384], arg67_1: f16[1152], arg68_1: f16[384, 384], arg69_1: f16[384], arg70_1: f16[384], arg71_1: f16[384], arg72_1: f16[1536, 384], arg73_1: f16[1536], arg74_1: f16[384, 1536], arg75_1: f16[384], arg76_1: f16[384], arg77_1: f16[384], arg78_1: f16[1152, 384], arg79_1: f16[1152], arg80_1: f16[384, 384], arg81_1: f16[384], arg82_1: f16[384], arg83_1: f16[384], arg84_1: f16[1536, 384], arg85_1: f16[1536], arg86_1: f16[384, 1536], arg87_1: f16[384], arg88_1: f16[384], arg89_1: f16[384], arg90_1: f16[1152, 384], arg91_1: f16[1152], arg92_1: f16[384, 384], arg93_1: f16[384], arg94_1: f16[384], arg95_1: f16[384], arg96_1: f16[1536, 384], arg97_1: f16[1536], arg98_1: f16[384, 1536], arg99_1: f16[384], arg100_1: f16[384], arg101_1: f16[384], arg102_1: f16[1152, 384], arg103_1: f16[1152], arg104_1: f16[384, 384], arg105_1: f16[384], arg106_1: f16[384], arg107_1: f16[384], arg108_1: f16[1536, 384], arg109_1: f16[1536], arg110_1: f16[384, 1536], arg111_1: f16[384], arg112_1: f16[384], arg113_1: f16[384], arg114_1: f16[1152, 384], arg115_1: f16[1152], arg116_1: f16[384, 384], arg117_1: f16[384], arg118_1: f16[384], arg119_1: f16[384], arg120_1: f16[1536, 384], arg121_1: f16[1536], arg122_1: f16[384, 1536], arg123_1: f16[384], arg124_1: f16[384], arg125_1: f16[384], arg126_1: f16[1152, 384], arg127_1: f16[1152], arg128_1: f16[384, 384], arg129_1: f16[384], arg130_1: f16[384], arg131_1: f16[384], arg132_1: f16[1536, 384], arg133_1: f16[1536], arg134_1: f16[384, 1536], arg135_1: f16[384], arg136_1: f16[384], arg137_1: f16[384], arg138_1: f16[1152, 384], arg139_1: f16[1152], arg140_1: f16[384, 384], arg141_1: f16[384], arg142_1: f16[384], arg143_1: f16[384], arg144_1: f16[1536, 384], arg145_1: f16[1536], arg146_1: f16[384, 1536], arg147_1: f16[384], arg148_1: f16[384], arg149_1: f16[384], arg150_1: f16[1000, 384], arg151_1: f16[1000], arg152_1: f16[s0, 3, 224, 224]):
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/patch_embed.py:35, code: x = self.proj(x)
        convolution: f16[s0, 384, 14, 14] = torch.ops.aten.convolution.default(arg152_1, arg2_1, arg3_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg3_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/patch_embed.py:37, code: x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        sym_size: Sym(s0) = torch.ops.aten.sym_size(arg152_1, 0);  arg152_1 = None
        sym_size_1: Sym(14) = torch.ops.aten.sym_size(convolution, 2)
        sym_size_2: Sym(14) = torch.ops.aten.sym_size(convolution, 3)
        
        # No stacktrace found for following nodes
        mul: Sym(196) = sym_size_1 * sym_size_2;  sym_size_1 = sym_size_2 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/patch_embed.py:37, code: x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        view: f16[s0, 384, 196] = torch.ops.aten.view.default(convolution, [sym_size, 384, mul]);  convolution = mul = None
        permute: f16[s0, 196, 384] = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:434, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        expand: f16[s0, 1, 384] = torch.ops.aten.expand.default(arg0_1, [sym_size, -1, -1]);  arg0_1 = None
        cat: f16[s0, 197, 384] = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:435, code: x = self.pos_drop(x + self.pos_embed)
        add: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(cat, arg1_1);  arg1_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True);  convert_element_type = None
        getitem: f32[s0, 197, 1] = var_mean[0]
        getitem_1: f32[s0, 197, 1] = var_mean[1];  var_mean = None
        add_1: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul_1: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        add_2: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        convert_element_type_1: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_2, torch.float16);  add_2 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_1: f16[384, 1152] = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        sym_size_3: Sym(197) = torch.ops.aten.sym_size(cat, 1);  cat = None
        
        # No stacktrace found for following nodes
        mul_3: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_1: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_1, [mul_3, 384]);  convert_element_type_1 = mul_3 = None
        addmm: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg7_1, view_1, permute_1);  arg7_1 = view_1 = permute_1 = None
        view_2: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm, [sym_size, sym_size_3, 1152]);  addmm = None
        view_3: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_2, [sym_size, sym_size_3, 3, 6, 64])
        permute_2: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: f16[s0, 6, 197, 64] = unbind[0]
        getitem_3: f16[s0, 6, 197, 64] = unbind[1]
        getitem_4: f16[s0, 6, 197, 64] = unbind[2];  unbind = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_3: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_3, [0, 1, 3, 2]);  getitem_3 = None
        sym_size_4: Sym(197) = torch.ops.aten.sym_size(view_2, 1);  view_2 = None
        expand_1: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_2, [sym_size, 6, sym_size_4, 64]);  getitem_2 = None
        clone: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        
        # No stacktrace found for following nodes
        mul_4: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone, [mul_4, sym_size_4, 64]);  clone = None
        expand_2: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_3, [sym_size, 6, 64, sym_size_4]);  permute_3 = None
        clone_1: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        _unsafe_view_1: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_1, [mul_4, 64, sym_size_4]);  clone_1 = mul_4 = None
        bmm: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1);  _unsafe_view = _unsafe_view_1 = None
        view_4: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm, [sym_size, 6, sym_size_4, sym_size_4]);  bmm = None
        mul_5: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_4, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_2: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_5, torch.float32);  mul_5 = None
        amax: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_5: Sym(6) = torch.ops.aten.sym_size(view_4, 1);  view_4 = None
        expand_3: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_3, [sym_size, sym_size_5, sym_size_4, sym_size_4]);  convert_element_type_3 = None
        
        # No stacktrace found for following nodes
        mul_6: Sym(6*s0) = sym_size * sym_size_5
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_5: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_3, [mul_6, sym_size_4, sym_size_4]);  expand_3 = None
        expand_4: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_4, [sym_size, sym_size_5, sym_size_4, 64]);  getitem_4 = None
        clone_2: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        _unsafe_view_2: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_2, [mul_6, sym_size_4, 64]);  clone_2 = mul_6 = None
        bmm_1: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_5, _unsafe_view_2);  view_5 = _unsafe_view_2 = None
        view_6: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_1, [sym_size, sym_size_5, sym_size_4, 64]);  bmm_1 = sym_size_5 = None
        permute_4: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_3: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        _unsafe_view_3: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_3, [sym_size, sym_size_3, 384]);  clone_3 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_5: f16[384, 384] = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        
        # No stacktrace found for following nodes
        mul_7: Sym(197*s0) = sym_size * sym_size_4
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_6: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_3, 2)
        view_7: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_3, [mul_7, sym_size_6]);  _unsafe_view_3 = mul_7 = sym_size_6 = None
        addmm_1: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg9_1, view_7, permute_5);  arg9_1 = view_7 = permute_5 = None
        view_8: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_1, [sym_size, sym_size_4, 384]);  addmm_1 = sym_size_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_3: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add, view_8);  add = view_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_4: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True);  convert_element_type_4 = None
        getitem_5: f32[s0, 197, 1] = var_mean_1[0]
        getitem_6: f32[s0, 197, 1] = var_mean_1[1];  var_mean_1 = None
        add_4: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
        rsqrt_1: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
        mul_8: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_9: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_8, arg10_1);  mul_8 = arg10_1 = None
        add_5: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_9, arg11_1);  mul_9 = arg11_1 = None
        convert_element_type_5: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_6: f16[384, 1536] = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        
        # No stacktrace found for following nodes
        mul_10: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_9: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_5, [mul_10, 384]);  convert_element_type_5 = mul_10 = None
        addmm_2: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg13_1, view_9, permute_6);  arg13_1 = view_9 = permute_6 = None
        view_10: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_2, [sym_size, sym_size_3, 1536]);  addmm_2 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_6: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_10, torch.float32)
        mul_11: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_12: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_6: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_11, add_6);  mul_11 = add_6 = None
        convert_element_type_7: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_7: f16[1536, 384] = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        sym_size_7: Sym(197) = torch.ops.aten.sym_size(view_10, 1);  view_10 = None
        
        # No stacktrace found for following nodes
        mul_14: Sym(197*s0) = sym_size * sym_size_7
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_11: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_7, [mul_14, 1536]);  convert_element_type_7 = mul_14 = None
        addmm_3: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg15_1, view_11, permute_7);  arg15_1 = view_11 = permute_7 = None
        view_12: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_3, [sym_size, sym_size_7, 384]);  addmm_3 = sym_size_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_7: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_3, view_12);  add_3 = view_12 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_8: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_7: f32[s0, 197, 1] = var_mean_2[0]
        getitem_8: f32[s0, 197, 1] = var_mean_2[1];  var_mean_2 = None
        add_8: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
        rsqrt_2: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_7, getitem_8);  getitem_8 = None
        mul_15: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_16: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_15, arg16_1);  mul_15 = arg16_1 = None
        add_9: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_16, arg17_1);  mul_16 = arg17_1 = None
        convert_element_type_9: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_8: f16[384, 1152] = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        
        # No stacktrace found for following nodes
        mul_17: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_13: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_9, [mul_17, 384]);  convert_element_type_9 = mul_17 = None
        addmm_4: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg19_1, view_13, permute_8);  arg19_1 = view_13 = permute_8 = None
        view_14: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_4, [sym_size, sym_size_3, 1152]);  addmm_4 = None
        view_15: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_14, [sym_size, sym_size_3, 3, 6, 64])
        permute_9: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
        getitem_9: f16[s0, 6, 197, 64] = unbind_1[0]
        getitem_10: f16[s0, 6, 197, 64] = unbind_1[1]
        getitem_11: f16[s0, 6, 197, 64] = unbind_1[2];  unbind_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_10: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
        sym_size_8: Sym(197) = torch.ops.aten.sym_size(view_14, 1);  view_14 = None
        expand_5: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_9, [sym_size, 6, sym_size_8, 64]);  getitem_9 = None
        clone_4: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        
        # No stacktrace found for following nodes
        mul_18: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_4: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_4, [mul_18, sym_size_8, 64]);  clone_4 = None
        expand_6: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_10, [sym_size, 6, 64, sym_size_8]);  permute_10 = None
        clone_5: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        _unsafe_view_5: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_5, [mul_18, 64, sym_size_8]);  clone_5 = mul_18 = None
        bmm_2: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_4, _unsafe_view_5);  _unsafe_view_4 = _unsafe_view_5 = None
        view_16: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_2, [sym_size, 6, sym_size_8, sym_size_8]);  bmm_2 = None
        mul_19: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_16, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_10: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_19, torch.float32);  mul_19 = None
        amax_1: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_4: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = amax_1 = None
        exp_1: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_11: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_9: Sym(6) = torch.ops.aten.sym_size(view_16, 1);  view_16 = None
        expand_7: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_11, [sym_size, sym_size_9, sym_size_8, sym_size_8]);  convert_element_type_11 = None
        
        # No stacktrace found for following nodes
        mul_20: Sym(6*s0) = sym_size * sym_size_9
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_17: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_7, [mul_20, sym_size_8, sym_size_8]);  expand_7 = None
        expand_8: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_11, [sym_size, sym_size_9, sym_size_8, 64]);  getitem_11 = None
        clone_6: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        _unsafe_view_6: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_6, [mul_20, sym_size_8, 64]);  clone_6 = mul_20 = None
        bmm_3: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_17, _unsafe_view_6);  view_17 = _unsafe_view_6 = None
        view_18: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_3, [sym_size, sym_size_9, sym_size_8, 64]);  bmm_3 = sym_size_9 = None
        permute_11: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_7: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        _unsafe_view_7: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_7, [sym_size, sym_size_3, 384]);  clone_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_12: f16[384, 384] = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        
        # No stacktrace found for following nodes
        mul_21: Sym(197*s0) = sym_size * sym_size_8
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_10: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_7, 2)
        view_19: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_7, [mul_21, sym_size_10]);  _unsafe_view_7 = mul_21 = sym_size_10 = None
        addmm_5: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg21_1, view_19, permute_12);  arg21_1 = view_19 = permute_12 = None
        view_20: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_5, [sym_size, sym_size_8, 384]);  addmm_5 = sym_size_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_10: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_7, view_20);  add_7 = view_20 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_12: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [2], correction = 0, keepdim = True);  convert_element_type_12 = None
        getitem_12: f32[s0, 197, 1] = var_mean_3[0]
        getitem_13: f32[s0, 197, 1] = var_mean_3[1];  var_mean_3 = None
        add_11: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_3: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_5: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_10, getitem_13);  getitem_13 = None
        mul_22: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_23: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_22, arg22_1);  mul_22 = arg22_1 = None
        add_12: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_23, arg23_1);  mul_23 = arg23_1 = None
        convert_element_type_13: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_13: f16[384, 1536] = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        
        # No stacktrace found for following nodes
        mul_24: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_21: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_13, [mul_24, 384]);  convert_element_type_13 = mul_24 = None
        addmm_6: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg25_1, view_21, permute_13);  arg25_1 = view_21 = permute_13 = None
        view_22: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_6, [sym_size, sym_size_3, 1536]);  addmm_6 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_14: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_22, torch.float32)
        mul_25: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_26: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_1: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_13: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_27: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_25, add_13);  mul_25 = add_13 = None
        convert_element_type_15: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_27, torch.float16);  mul_27 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_14: f16[1536, 384] = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        sym_size_11: Sym(197) = torch.ops.aten.sym_size(view_22, 1);  view_22 = None
        
        # No stacktrace found for following nodes
        mul_28: Sym(197*s0) = sym_size * sym_size_11
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_23: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_15, [mul_28, 1536]);  convert_element_type_15 = mul_28 = None
        addmm_7: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg27_1, view_23, permute_14);  arg27_1 = view_23 = permute_14 = None
        view_24: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_7, [sym_size, sym_size_11, 384]);  addmm_7 = sym_size_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_14: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_10, view_24);  add_10 = view_24 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_16: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True);  convert_element_type_16 = None
        getitem_14: f32[s0, 197, 1] = var_mean_4[0]
        getitem_15: f32[s0, 197, 1] = var_mean_4[1];  var_mean_4 = None
        add_15: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_4: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_6: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_14, getitem_15);  getitem_15 = None
        mul_29: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_30: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_29, arg28_1);  mul_29 = arg28_1 = None
        add_16: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_30, arg29_1);  mul_30 = arg29_1 = None
        convert_element_type_17: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_16, torch.float16);  add_16 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_15: f16[384, 1152] = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        
        # No stacktrace found for following nodes
        mul_31: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_25: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_17, [mul_31, 384]);  convert_element_type_17 = mul_31 = None
        addmm_8: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg31_1, view_25, permute_15);  arg31_1 = view_25 = permute_15 = None
        view_26: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_8, [sym_size, sym_size_3, 1152]);  addmm_8 = None
        view_27: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_26, [sym_size, sym_size_3, 3, 6, 64])
        permute_16: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_27, [2, 0, 3, 1, 4]);  view_27 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_2 = torch.ops.aten.unbind.int(permute_16);  permute_16 = None
        getitem_16: f16[s0, 6, 197, 64] = unbind_2[0]
        getitem_17: f16[s0, 6, 197, 64] = unbind_2[1]
        getitem_18: f16[s0, 6, 197, 64] = unbind_2[2];  unbind_2 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_17: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_17, [0, 1, 3, 2]);  getitem_17 = None
        sym_size_12: Sym(197) = torch.ops.aten.sym_size(view_26, 1);  view_26 = None
        expand_9: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_16, [sym_size, 6, sym_size_12, 64]);  getitem_16 = None
        clone_8: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        
        # No stacktrace found for following nodes
        mul_32: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_8: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_8, [mul_32, sym_size_12, 64]);  clone_8 = None
        expand_10: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_17, [sym_size, 6, 64, sym_size_12]);  permute_17 = None
        clone_9: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        _unsafe_view_9: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_9, [mul_32, 64, sym_size_12]);  clone_9 = mul_32 = None
        bmm_4: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_8, _unsafe_view_9);  _unsafe_view_8 = _unsafe_view_9 = None
        view_28: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_4, [sym_size, 6, sym_size_12, sym_size_12]);  bmm_4 = None
        mul_33: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_28, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_18: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_33, torch.float32);  mul_33 = None
        amax_2: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_7: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_2);  convert_element_type_18 = amax_2 = None
        exp_2: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_19: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_2, torch.float16);  div_2 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_13: Sym(6) = torch.ops.aten.sym_size(view_28, 1);  view_28 = None
        expand_11: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_19, [sym_size, sym_size_13, sym_size_12, sym_size_12]);  convert_element_type_19 = None
        
        # No stacktrace found for following nodes
        mul_34: Sym(6*s0) = sym_size * sym_size_13
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_29: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_11, [mul_34, sym_size_12, sym_size_12]);  expand_11 = None
        expand_12: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_18, [sym_size, sym_size_13, sym_size_12, 64]);  getitem_18 = None
        clone_10: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        _unsafe_view_10: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_10, [mul_34, sym_size_12, 64]);  clone_10 = mul_34 = None
        bmm_5: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_29, _unsafe_view_10);  view_29 = _unsafe_view_10 = None
        view_30: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_5, [sym_size, sym_size_13, sym_size_12, 64]);  bmm_5 = sym_size_13 = None
        permute_18: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_11: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        _unsafe_view_11: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_11, [sym_size, sym_size_3, 384]);  clone_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_19: f16[384, 384] = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        
        # No stacktrace found for following nodes
        mul_35: Sym(197*s0) = sym_size * sym_size_12
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_14: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_11, 2)
        view_31: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_11, [mul_35, sym_size_14]);  _unsafe_view_11 = mul_35 = sym_size_14 = None
        addmm_9: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg33_1, view_31, permute_19);  arg33_1 = view_31 = permute_19 = None
        view_32: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_9, [sym_size, sym_size_12, 384]);  addmm_9 = sym_size_12 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_17: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_14, view_32);  add_14 = view_32 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_20: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True);  convert_element_type_20 = None
        getitem_19: f32[s0, 197, 1] = var_mean_5[0]
        getitem_20: f32[s0, 197, 1] = var_mean_5[1];  var_mean_5 = None
        add_18: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_19, 1e-06);  getitem_19 = None
        rsqrt_5: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_17, getitem_20);  getitem_20 = None
        mul_36: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_37: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_36, arg34_1);  mul_36 = arg34_1 = None
        add_19: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_37, arg35_1);  mul_37 = arg35_1 = None
        convert_element_type_21: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_20: f16[384, 1536] = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        
        # No stacktrace found for following nodes
        mul_38: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_33: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_21, [mul_38, 384]);  convert_element_type_21 = mul_38 = None
        addmm_10: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg37_1, view_33, permute_20);  arg37_1 = view_33 = permute_20 = None
        view_34: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_10, [sym_size, sym_size_3, 1536]);  addmm_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_22: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_34, torch.float32)
        mul_39: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_40: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_20: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_41: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_39, add_20);  mul_39 = add_20 = None
        convert_element_type_23: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_41, torch.float16);  mul_41 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_21: f16[1536, 384] = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        sym_size_15: Sym(197) = torch.ops.aten.sym_size(view_34, 1);  view_34 = None
        
        # No stacktrace found for following nodes
        mul_42: Sym(197*s0) = sym_size * sym_size_15
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_35: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_23, [mul_42, 1536]);  convert_element_type_23 = mul_42 = None
        addmm_11: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg39_1, view_35, permute_21);  arg39_1 = view_35 = permute_21 = None
        view_36: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_11, [sym_size, sym_size_15, 384]);  addmm_11 = sym_size_15 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_21: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_17, view_36);  add_17 = view_36 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_24: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True);  convert_element_type_24 = None
        getitem_21: f32[s0, 197, 1] = var_mean_6[0]
        getitem_22: f32[s0, 197, 1] = var_mean_6[1];  var_mean_6 = None
        add_22: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_21, 1e-06);  getitem_21 = None
        rsqrt_6: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_21, getitem_22);  getitem_22 = None
        mul_43: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_44: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_43, arg40_1);  mul_43 = arg40_1 = None
        add_23: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_44, arg41_1);  mul_44 = arg41_1 = None
        convert_element_type_25: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_23, torch.float16);  add_23 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_22: f16[384, 1152] = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        
        # No stacktrace found for following nodes
        mul_45: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_37: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_25, [mul_45, 384]);  convert_element_type_25 = mul_45 = None
        addmm_12: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg43_1, view_37, permute_22);  arg43_1 = view_37 = permute_22 = None
        view_38: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_12, [sym_size, sym_size_3, 1152]);  addmm_12 = None
        view_39: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_38, [sym_size, sym_size_3, 3, 6, 64])
        permute_23: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_39, [2, 0, 3, 1, 4]);  view_39 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_3 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
        getitem_23: f16[s0, 6, 197, 64] = unbind_3[0]
        getitem_24: f16[s0, 6, 197, 64] = unbind_3[1]
        getitem_25: f16[s0, 6, 197, 64] = unbind_3[2];  unbind_3 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_24: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_24, [0, 1, 3, 2]);  getitem_24 = None
        sym_size_16: Sym(197) = torch.ops.aten.sym_size(view_38, 1);  view_38 = None
        expand_13: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_23, [sym_size, 6, sym_size_16, 64]);  getitem_23 = None
        clone_12: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        
        # No stacktrace found for following nodes
        mul_46: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_12: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_12, [mul_46, sym_size_16, 64]);  clone_12 = None
        expand_14: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_24, [sym_size, 6, 64, sym_size_16]);  permute_24 = None
        clone_13: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        _unsafe_view_13: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_13, [mul_46, 64, sym_size_16]);  clone_13 = mul_46 = None
        bmm_6: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_12, _unsafe_view_13);  _unsafe_view_12 = _unsafe_view_13 = None
        view_40: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_6, [sym_size, 6, sym_size_16, sym_size_16]);  bmm_6 = None
        mul_47: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_40, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_26: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_47, torch.float32);  mul_47 = None
        amax_3: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_26, [-1], True)
        sub_10: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_3);  convert_element_type_26 = amax_3 = None
        exp_3: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_27: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_3, torch.float16);  div_3 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_17: Sym(6) = torch.ops.aten.sym_size(view_40, 1);  view_40 = None
        expand_15: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_27, [sym_size, sym_size_17, sym_size_16, sym_size_16]);  convert_element_type_27 = None
        
        # No stacktrace found for following nodes
        mul_48: Sym(6*s0) = sym_size * sym_size_17
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_41: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_15, [mul_48, sym_size_16, sym_size_16]);  expand_15 = None
        expand_16: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_25, [sym_size, sym_size_17, sym_size_16, 64]);  getitem_25 = None
        clone_14: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        _unsafe_view_14: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_14, [mul_48, sym_size_16, 64]);  clone_14 = mul_48 = None
        bmm_7: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_41, _unsafe_view_14);  view_41 = _unsafe_view_14 = None
        view_42: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_7, [sym_size, sym_size_17, sym_size_16, 64]);  bmm_7 = sym_size_17 = None
        permute_25: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_15: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        _unsafe_view_15: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_15, [sym_size, sym_size_3, 384]);  clone_15 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_26: f16[384, 384] = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        
        # No stacktrace found for following nodes
        mul_49: Sym(197*s0) = sym_size * sym_size_16
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_18: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_15, 2)
        view_43: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_15, [mul_49, sym_size_18]);  _unsafe_view_15 = mul_49 = sym_size_18 = None
        addmm_13: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg45_1, view_43, permute_26);  arg45_1 = view_43 = permute_26 = None
        view_44: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_13, [sym_size, sym_size_16, 384]);  addmm_13 = sym_size_16 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_24: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_21, view_44);  add_21 = view_44 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_28: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_28, [2], correction = 0, keepdim = True);  convert_element_type_28 = None
        getitem_26: f32[s0, 197, 1] = var_mean_7[0]
        getitem_27: f32[s0, 197, 1] = var_mean_7[1];  var_mean_7 = None
        add_25: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_7: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_11: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_24, getitem_27);  getitem_27 = None
        mul_50: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_51: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_50, arg46_1);  mul_50 = arg46_1 = None
        add_26: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_51, arg47_1);  mul_51 = arg47_1 = None
        convert_element_type_29: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_27: f16[384, 1536] = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        
        # No stacktrace found for following nodes
        mul_52: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_45: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_29, [mul_52, 384]);  convert_element_type_29 = mul_52 = None
        addmm_14: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg49_1, view_45, permute_27);  arg49_1 = view_45 = permute_27 = None
        view_46: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_14, [sym_size, sym_size_3, 1536]);  addmm_14 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_30: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_46, torch.float32)
        mul_53: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.5)
        mul_54: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.7071067811865476);  convert_element_type_30 = None
        erf_3: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_27: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_55: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_53, add_27);  mul_53 = add_27 = None
        convert_element_type_31: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_55, torch.float16);  mul_55 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_28: f16[1536, 384] = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        sym_size_19: Sym(197) = torch.ops.aten.sym_size(view_46, 1);  view_46 = None
        
        # No stacktrace found for following nodes
        mul_56: Sym(197*s0) = sym_size * sym_size_19
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_47: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_31, [mul_56, 1536]);  convert_element_type_31 = mul_56 = None
        addmm_15: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg51_1, view_47, permute_28);  arg51_1 = view_47 = permute_28 = None
        view_48: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_15, [sym_size, sym_size_19, 384]);  addmm_15 = sym_size_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_28: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_24, view_48);  add_24 = view_48 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_32: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_32, [2], correction = 0, keepdim = True);  convert_element_type_32 = None
        getitem_28: f32[s0, 197, 1] = var_mean_8[0]
        getitem_29: f32[s0, 197, 1] = var_mean_8[1];  var_mean_8 = None
        add_29: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_8: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_12: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_28, getitem_29);  getitem_29 = None
        mul_57: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_58: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_57, arg52_1);  mul_57 = arg52_1 = None
        add_30: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_58, arg53_1);  mul_58 = arg53_1 = None
        convert_element_type_33: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_29: f16[384, 1152] = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        
        # No stacktrace found for following nodes
        mul_59: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_49: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_33, [mul_59, 384]);  convert_element_type_33 = mul_59 = None
        addmm_16: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg55_1, view_49, permute_29);  arg55_1 = view_49 = permute_29 = None
        view_50: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_16, [sym_size, sym_size_3, 1152]);  addmm_16 = None
        view_51: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_50, [sym_size, sym_size_3, 3, 6, 64])
        permute_30: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_4 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
        getitem_30: f16[s0, 6, 197, 64] = unbind_4[0]
        getitem_31: f16[s0, 6, 197, 64] = unbind_4[1]
        getitem_32: f16[s0, 6, 197, 64] = unbind_4[2];  unbind_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_31: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
        sym_size_20: Sym(197) = torch.ops.aten.sym_size(view_50, 1);  view_50 = None
        expand_17: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_30, [sym_size, 6, sym_size_20, 64]);  getitem_30 = None
        clone_16: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        
        # No stacktrace found for following nodes
        mul_60: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_16: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_16, [mul_60, sym_size_20, 64]);  clone_16 = None
        expand_18: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_31, [sym_size, 6, 64, sym_size_20]);  permute_31 = None
        clone_17: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        _unsafe_view_17: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_17, [mul_60, 64, sym_size_20]);  clone_17 = mul_60 = None
        bmm_8: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_16, _unsafe_view_17);  _unsafe_view_16 = _unsafe_view_17 = None
        view_52: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_8, [sym_size, 6, sym_size_20, sym_size_20]);  bmm_8 = None
        mul_61: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_52, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_34: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_61, torch.float32);  mul_61 = None
        amax_4: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_34, [-1], True)
        sub_13: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_4);  convert_element_type_34 = amax_4 = None
        exp_4: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_35: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_4, torch.float16);  div_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_21: Sym(6) = torch.ops.aten.sym_size(view_52, 1);  view_52 = None
        expand_19: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_35, [sym_size, sym_size_21, sym_size_20, sym_size_20]);  convert_element_type_35 = None
        
        # No stacktrace found for following nodes
        mul_62: Sym(6*s0) = sym_size * sym_size_21
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_53: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_19, [mul_62, sym_size_20, sym_size_20]);  expand_19 = None
        expand_20: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_32, [sym_size, sym_size_21, sym_size_20, 64]);  getitem_32 = None
        clone_18: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        _unsafe_view_18: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_18, [mul_62, sym_size_20, 64]);  clone_18 = mul_62 = None
        bmm_9: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_53, _unsafe_view_18);  view_53 = _unsafe_view_18 = None
        view_54: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_9, [sym_size, sym_size_21, sym_size_20, 64]);  bmm_9 = sym_size_21 = None
        permute_32: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_19: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        _unsafe_view_19: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_19, [sym_size, sym_size_3, 384]);  clone_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_33: f16[384, 384] = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        
        # No stacktrace found for following nodes
        mul_63: Sym(197*s0) = sym_size * sym_size_20
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_22: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_19, 2)
        view_55: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_19, [mul_63, sym_size_22]);  _unsafe_view_19 = mul_63 = sym_size_22 = None
        addmm_17: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg57_1, view_55, permute_33);  arg57_1 = view_55 = permute_33 = None
        view_56: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_17, [sym_size, sym_size_20, 384]);  addmm_17 = sym_size_20 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_31: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_28, view_56);  add_28 = view_56 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_36: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True);  convert_element_type_36 = None
        getitem_33: f32[s0, 197, 1] = var_mean_9[0]
        getitem_34: f32[s0, 197, 1] = var_mean_9[1];  var_mean_9 = None
        add_32: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
        rsqrt_9: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_14: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_31, getitem_34);  getitem_34 = None
        mul_64: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_65: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_64, arg58_1);  mul_64 = arg58_1 = None
        add_33: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_65, arg59_1);  mul_65 = arg59_1 = None
        convert_element_type_37: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_34: f16[384, 1536] = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        
        # No stacktrace found for following nodes
        mul_66: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_57: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_37, [mul_66, 384]);  convert_element_type_37 = mul_66 = None
        addmm_18: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg61_1, view_57, permute_34);  arg61_1 = view_57 = permute_34 = None
        view_58: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_18, [sym_size, sym_size_3, 1536]);  addmm_18 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_38: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_58, torch.float32)
        mul_67: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_68: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_4: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_34: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_69: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_67, add_34);  mul_67 = add_34 = None
        convert_element_type_39: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_35: f16[1536, 384] = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        sym_size_23: Sym(197) = torch.ops.aten.sym_size(view_58, 1);  view_58 = None
        
        # No stacktrace found for following nodes
        mul_70: Sym(197*s0) = sym_size * sym_size_23
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_59: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_39, [mul_70, 1536]);  convert_element_type_39 = mul_70 = None
        addmm_19: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg63_1, view_59, permute_35);  arg63_1 = view_59 = permute_35 = None
        view_60: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_19, [sym_size, sym_size_23, 384]);  addmm_19 = sym_size_23 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_35: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_31, view_60);  add_31 = view_60 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_40: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True);  convert_element_type_40 = None
        getitem_35: f32[s0, 197, 1] = var_mean_10[0]
        getitem_36: f32[s0, 197, 1] = var_mean_10[1];  var_mean_10 = None
        add_36: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_35, 1e-06);  getitem_35 = None
        rsqrt_10: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_15: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_35, getitem_36);  getitem_36 = None
        mul_71: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_72: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_71, arg64_1);  mul_71 = arg64_1 = None
        add_37: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_72, arg65_1);  mul_72 = arg65_1 = None
        convert_element_type_41: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_36: f16[384, 1152] = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        
        # No stacktrace found for following nodes
        mul_73: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_61: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_41, [mul_73, 384]);  convert_element_type_41 = mul_73 = None
        addmm_20: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg67_1, view_61, permute_36);  arg67_1 = view_61 = permute_36 = None
        view_62: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_20, [sym_size, sym_size_3, 1152]);  addmm_20 = None
        view_63: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_62, [sym_size, sym_size_3, 3, 6, 64])
        permute_37: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
        getitem_37: f16[s0, 6, 197, 64] = unbind_5[0]
        getitem_38: f16[s0, 6, 197, 64] = unbind_5[1]
        getitem_39: f16[s0, 6, 197, 64] = unbind_5[2];  unbind_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_38: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_38, [0, 1, 3, 2]);  getitem_38 = None
        sym_size_24: Sym(197) = torch.ops.aten.sym_size(view_62, 1);  view_62 = None
        expand_21: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_37, [sym_size, 6, sym_size_24, 64]);  getitem_37 = None
        clone_20: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        
        # No stacktrace found for following nodes
        mul_74: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_20: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_20, [mul_74, sym_size_24, 64]);  clone_20 = None
        expand_22: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_38, [sym_size, 6, 64, sym_size_24]);  permute_38 = None
        clone_21: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        _unsafe_view_21: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_21, [mul_74, 64, sym_size_24]);  clone_21 = mul_74 = None
        bmm_10: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_20, _unsafe_view_21);  _unsafe_view_20 = _unsafe_view_21 = None
        view_64: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_10, [sym_size, 6, sym_size_24, sym_size_24]);  bmm_10 = None
        mul_75: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_64, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_42: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_75, torch.float32);  mul_75 = None
        amax_5: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_42, [-1], True)
        sub_16: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_42, amax_5);  convert_element_type_42 = amax_5 = None
        exp_5: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_43: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_5, torch.float16);  div_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_25: Sym(6) = torch.ops.aten.sym_size(view_64, 1);  view_64 = None
        expand_23: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_43, [sym_size, sym_size_25, sym_size_24, sym_size_24]);  convert_element_type_43 = None
        
        # No stacktrace found for following nodes
        mul_76: Sym(6*s0) = sym_size * sym_size_25
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_65: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_23, [mul_76, sym_size_24, sym_size_24]);  expand_23 = None
        expand_24: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_39, [sym_size, sym_size_25, sym_size_24, 64]);  getitem_39 = None
        clone_22: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        _unsafe_view_22: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_22, [mul_76, sym_size_24, 64]);  clone_22 = mul_76 = None
        bmm_11: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_65, _unsafe_view_22);  view_65 = _unsafe_view_22 = None
        view_66: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_11, [sym_size, sym_size_25, sym_size_24, 64]);  bmm_11 = sym_size_25 = None
        permute_39: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        clone_23: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        _unsafe_view_23: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_23, [sym_size, sym_size_3, 384]);  clone_23 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_40: f16[384, 384] = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        
        # No stacktrace found for following nodes
        mul_77: Sym(197*s0) = sym_size * sym_size_24
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_26: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_23, 2)
        view_67: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_23, [mul_77, sym_size_26]);  _unsafe_view_23 = mul_77 = sym_size_26 = None
        addmm_21: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg69_1, view_67, permute_40);  arg69_1 = view_67 = permute_40 = None
        view_68: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_21, [sym_size, sym_size_24, 384]);  addmm_21 = sym_size_24 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_38: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_35, view_68);  add_35 = view_68 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_44: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True);  convert_element_type_44 = None
        getitem_40: f32[s0, 197, 1] = var_mean_11[0]
        getitem_41: f32[s0, 197, 1] = var_mean_11[1];  var_mean_11 = None
        add_39: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        rsqrt_11: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_17: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_38, getitem_41);  getitem_41 = None
        mul_78: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_79: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_78, arg70_1);  mul_78 = arg70_1 = None
        add_40: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_79, arg71_1);  mul_79 = arg71_1 = None
        convert_element_type_45: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_40, torch.float16);  add_40 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_41: f16[384, 1536] = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        
        # No stacktrace found for following nodes
        mul_80: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_69: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_45, [mul_80, 384]);  convert_element_type_45 = mul_80 = None
        addmm_22: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg73_1, view_69, permute_41);  arg73_1 = view_69 = permute_41 = None
        view_70: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_22, [sym_size, sym_size_3, 1536]);  addmm_22 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_46: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_70, torch.float32)
        mul_81: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.5)
        mul_82: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.7071067811865476);  convert_element_type_46 = None
        erf_5: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_41: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_83: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_81, add_41);  mul_81 = add_41 = None
        convert_element_type_47: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_83, torch.float16);  mul_83 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_42: f16[1536, 384] = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        sym_size_27: Sym(197) = torch.ops.aten.sym_size(view_70, 1);  view_70 = None
        
        # No stacktrace found for following nodes
        mul_84: Sym(197*s0) = sym_size * sym_size_27
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_71: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_47, [mul_84, 1536]);  convert_element_type_47 = mul_84 = None
        addmm_23: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg75_1, view_71, permute_42);  arg75_1 = view_71 = permute_42 = None
        view_72: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_23, [sym_size, sym_size_27, 384]);  addmm_23 = sym_size_27 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_42: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_38, view_72);  add_38 = view_72 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_48: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True);  convert_element_type_48 = None
        getitem_42: f32[s0, 197, 1] = var_mean_12[0]
        getitem_43: f32[s0, 197, 1] = var_mean_12[1];  var_mean_12 = None
        add_43: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        rsqrt_12: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_18: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_42, getitem_43);  getitem_43 = None
        mul_85: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_86: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_85, arg76_1);  mul_85 = arg76_1 = None
        add_44: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_86, arg77_1);  mul_86 = arg77_1 = None
        convert_element_type_49: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_43: f16[384, 1152] = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        
        # No stacktrace found for following nodes
        mul_87: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_73: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_49, [mul_87, 384]);  convert_element_type_49 = mul_87 = None
        addmm_24: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg79_1, view_73, permute_43);  arg79_1 = view_73 = permute_43 = None
        view_74: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_24, [sym_size, sym_size_3, 1152]);  addmm_24 = None
        view_75: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_74, [sym_size, sym_size_3, 3, 6, 64])
        permute_44: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_75, [2, 0, 3, 1, 4]);  view_75 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_6 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
        getitem_44: f16[s0, 6, 197, 64] = unbind_6[0]
        getitem_45: f16[s0, 6, 197, 64] = unbind_6[1]
        getitem_46: f16[s0, 6, 197, 64] = unbind_6[2];  unbind_6 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_45: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_45, [0, 1, 3, 2]);  getitem_45 = None
        sym_size_28: Sym(197) = torch.ops.aten.sym_size(view_74, 1);  view_74 = None
        expand_25: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_44, [sym_size, 6, sym_size_28, 64]);  getitem_44 = None
        clone_24: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        
        # No stacktrace found for following nodes
        mul_88: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_24: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_24, [mul_88, sym_size_28, 64]);  clone_24 = None
        expand_26: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_45, [sym_size, 6, 64, sym_size_28]);  permute_45 = None
        clone_25: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        _unsafe_view_25: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_25, [mul_88, 64, sym_size_28]);  clone_25 = mul_88 = None
        bmm_12: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_24, _unsafe_view_25);  _unsafe_view_24 = _unsafe_view_25 = None
        view_76: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_12, [sym_size, 6, sym_size_28, sym_size_28]);  bmm_12 = None
        mul_89: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_76, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_50: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_89, torch.float32);  mul_89 = None
        amax_6: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_50, [-1], True)
        sub_19: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_50, amax_6);  convert_element_type_50 = amax_6 = None
        exp_6: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_51: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_6, torch.float16);  div_6 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_29: Sym(6) = torch.ops.aten.sym_size(view_76, 1);  view_76 = None
        expand_27: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_51, [sym_size, sym_size_29, sym_size_28, sym_size_28]);  convert_element_type_51 = None
        
        # No stacktrace found for following nodes
        mul_90: Sym(6*s0) = sym_size * sym_size_29
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_77: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_27, [mul_90, sym_size_28, sym_size_28]);  expand_27 = None
        expand_28: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_46, [sym_size, sym_size_29, sym_size_28, 64]);  getitem_46 = None
        clone_26: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        _unsafe_view_26: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_26, [mul_90, sym_size_28, 64]);  clone_26 = mul_90 = None
        bmm_13: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_77, _unsafe_view_26);  view_77 = _unsafe_view_26 = None
        view_78: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_13, [sym_size, sym_size_29, sym_size_28, 64]);  bmm_13 = sym_size_29 = None
        permute_46: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        clone_27: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        _unsafe_view_27: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_27, [sym_size, sym_size_3, 384]);  clone_27 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_47: f16[384, 384] = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        
        # No stacktrace found for following nodes
        mul_91: Sym(197*s0) = sym_size * sym_size_28
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_30: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_27, 2)
        view_79: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_27, [mul_91, sym_size_30]);  _unsafe_view_27 = mul_91 = sym_size_30 = None
        addmm_25: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg81_1, view_79, permute_47);  arg81_1 = view_79 = permute_47 = None
        view_80: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_25, [sym_size, sym_size_28, 384]);  addmm_25 = sym_size_28 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_45: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_42, view_80);  add_42 = view_80 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_52: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_52, [2], correction = 0, keepdim = True);  convert_element_type_52 = None
        getitem_47: f32[s0, 197, 1] = var_mean_13[0]
        getitem_48: f32[s0, 197, 1] = var_mean_13[1];  var_mean_13 = None
        add_46: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
        rsqrt_13: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_20: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_45, getitem_48);  getitem_48 = None
        mul_92: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_93: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_92, arg82_1);  mul_92 = arg82_1 = None
        add_47: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_93, arg83_1);  mul_93 = arg83_1 = None
        convert_element_type_53: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_47, torch.float16);  add_47 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_48: f16[384, 1536] = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        
        # No stacktrace found for following nodes
        mul_94: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_81: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_53, [mul_94, 384]);  convert_element_type_53 = mul_94 = None
        addmm_26: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg85_1, view_81, permute_48);  arg85_1 = view_81 = permute_48 = None
        view_82: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_26, [sym_size, sym_size_3, 1536]);  addmm_26 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_54: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_82, torch.float32)
        mul_95: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.5)
        mul_96: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.7071067811865476);  convert_element_type_54 = None
        erf_6: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_48: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_97: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_95, add_48);  mul_95 = add_48 = None
        convert_element_type_55: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_97, torch.float16);  mul_97 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_49: f16[1536, 384] = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        sym_size_31: Sym(197) = torch.ops.aten.sym_size(view_82, 1);  view_82 = None
        
        # No stacktrace found for following nodes
        mul_98: Sym(197*s0) = sym_size * sym_size_31
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_83: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_55, [mul_98, 1536]);  convert_element_type_55 = mul_98 = None
        addmm_27: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg87_1, view_83, permute_49);  arg87_1 = view_83 = permute_49 = None
        view_84: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_27, [sym_size, sym_size_31, 384]);  addmm_27 = sym_size_31 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_49: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_45, view_84);  add_45 = view_84 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_56: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_56, [2], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_49: f32[s0, 197, 1] = var_mean_14[0]
        getitem_50: f32[s0, 197, 1] = var_mean_14[1];  var_mean_14 = None
        add_50: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_49, 1e-06);  getitem_49 = None
        rsqrt_14: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_21: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_49, getitem_50);  getitem_50 = None
        mul_99: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_100: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_99, arg88_1);  mul_99 = arg88_1 = None
        add_51: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_100, arg89_1);  mul_100 = arg89_1 = None
        convert_element_type_57: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_51, torch.float16);  add_51 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_50: f16[384, 1152] = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        
        # No stacktrace found for following nodes
        mul_101: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_85: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_57, [mul_101, 384]);  convert_element_type_57 = mul_101 = None
        addmm_28: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg91_1, view_85, permute_50);  arg91_1 = view_85 = permute_50 = None
        view_86: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_28, [sym_size, sym_size_3, 1152]);  addmm_28 = None
        view_87: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_86, [sym_size, sym_size_3, 3, 6, 64])
        permute_51: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_7 = torch.ops.aten.unbind.int(permute_51);  permute_51 = None
        getitem_51: f16[s0, 6, 197, 64] = unbind_7[0]
        getitem_52: f16[s0, 6, 197, 64] = unbind_7[1]
        getitem_53: f16[s0, 6, 197, 64] = unbind_7[2];  unbind_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_52: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_52, [0, 1, 3, 2]);  getitem_52 = None
        sym_size_32: Sym(197) = torch.ops.aten.sym_size(view_86, 1);  view_86 = None
        expand_29: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_51, [sym_size, 6, sym_size_32, 64]);  getitem_51 = None
        clone_28: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        
        # No stacktrace found for following nodes
        mul_102: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_28: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_28, [mul_102, sym_size_32, 64]);  clone_28 = None
        expand_30: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_52, [sym_size, 6, 64, sym_size_32]);  permute_52 = None
        clone_29: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        _unsafe_view_29: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_29, [mul_102, 64, sym_size_32]);  clone_29 = mul_102 = None
        bmm_14: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_28, _unsafe_view_29);  _unsafe_view_28 = _unsafe_view_29 = None
        view_88: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_14, [sym_size, 6, sym_size_32, sym_size_32]);  bmm_14 = None
        mul_103: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_88, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_58: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_103, torch.float32);  mul_103 = None
        amax_7: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_58, [-1], True)
        sub_22: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_58, amax_7);  convert_element_type_58 = amax_7 = None
        exp_7: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_59: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_7, torch.float16);  div_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_33: Sym(6) = torch.ops.aten.sym_size(view_88, 1);  view_88 = None
        expand_31: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_59, [sym_size, sym_size_33, sym_size_32, sym_size_32]);  convert_element_type_59 = None
        
        # No stacktrace found for following nodes
        mul_104: Sym(6*s0) = sym_size * sym_size_33
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_89: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_31, [mul_104, sym_size_32, sym_size_32]);  expand_31 = None
        expand_32: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_53, [sym_size, sym_size_33, sym_size_32, 64]);  getitem_53 = None
        clone_30: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        _unsafe_view_30: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_30, [mul_104, sym_size_32, 64]);  clone_30 = mul_104 = None
        bmm_15: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_89, _unsafe_view_30);  view_89 = _unsafe_view_30 = None
        view_90: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_15, [sym_size, sym_size_33, sym_size_32, 64]);  bmm_15 = sym_size_33 = None
        permute_53: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_31: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        _unsafe_view_31: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_31, [sym_size, sym_size_3, 384]);  clone_31 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_54: f16[384, 384] = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        
        # No stacktrace found for following nodes
        mul_105: Sym(197*s0) = sym_size * sym_size_32
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_34: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_31, 2)
        view_91: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_31, [mul_105, sym_size_34]);  _unsafe_view_31 = mul_105 = sym_size_34 = None
        addmm_29: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg93_1, view_91, permute_54);  arg93_1 = view_91 = permute_54 = None
        view_92: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_29, [sym_size, sym_size_32, 384]);  addmm_29 = sym_size_32 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_52: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_49, view_92);  add_49 = view_92 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_60: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True);  convert_element_type_60 = None
        getitem_54: f32[s0, 197, 1] = var_mean_15[0]
        getitem_55: f32[s0, 197, 1] = var_mean_15[1];  var_mean_15 = None
        add_53: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_15: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_23: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_52, getitem_55);  getitem_55 = None
        mul_106: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_107: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_106, arg94_1);  mul_106 = arg94_1 = None
        add_54: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_107, arg95_1);  mul_107 = arg95_1 = None
        convert_element_type_61: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_55: f16[384, 1536] = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        
        # No stacktrace found for following nodes
        mul_108: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_93: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_61, [mul_108, 384]);  convert_element_type_61 = mul_108 = None
        addmm_30: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg97_1, view_93, permute_55);  arg97_1 = view_93 = permute_55 = None
        view_94: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_30, [sym_size, sym_size_3, 1536]);  addmm_30 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_62: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_94, torch.float32)
        mul_109: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.5)
        mul_110: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.7071067811865476);  convert_element_type_62 = None
        erf_7: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_110);  mul_110 = None
        add_55: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_111: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_109, add_55);  mul_109 = add_55 = None
        convert_element_type_63: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_111, torch.float16);  mul_111 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_56: f16[1536, 384] = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        sym_size_35: Sym(197) = torch.ops.aten.sym_size(view_94, 1);  view_94 = None
        
        # No stacktrace found for following nodes
        mul_112: Sym(197*s0) = sym_size * sym_size_35
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_95: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_63, [mul_112, 1536]);  convert_element_type_63 = mul_112 = None
        addmm_31: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg99_1, view_95, permute_56);  arg99_1 = view_95 = permute_56 = None
        view_96: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_31, [sym_size, sym_size_35, 384]);  addmm_31 = sym_size_35 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_56: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_52, view_96);  add_52 = view_96 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_64: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True);  convert_element_type_64 = None
        getitem_56: f32[s0, 197, 1] = var_mean_16[0]
        getitem_57: f32[s0, 197, 1] = var_mean_16[1];  var_mean_16 = None
        add_57: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_16: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_24: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_56, getitem_57);  getitem_57 = None
        mul_113: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_114: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_113, arg100_1);  mul_113 = arg100_1 = None
        add_58: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_114, arg101_1);  mul_114 = arg101_1 = None
        convert_element_type_65: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_58, torch.float16);  add_58 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_57: f16[384, 1152] = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        
        # No stacktrace found for following nodes
        mul_115: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_97: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_65, [mul_115, 384]);  convert_element_type_65 = mul_115 = None
        addmm_32: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg103_1, view_97, permute_57);  arg103_1 = view_97 = permute_57 = None
        view_98: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_32, [sym_size, sym_size_3, 1152]);  addmm_32 = None
        view_99: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_98, [sym_size, sym_size_3, 3, 6, 64])
        permute_58: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_99, [2, 0, 3, 1, 4]);  view_99 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_8 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
        getitem_58: f16[s0, 6, 197, 64] = unbind_8[0]
        getitem_59: f16[s0, 6, 197, 64] = unbind_8[1]
        getitem_60: f16[s0, 6, 197, 64] = unbind_8[2];  unbind_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_59: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_59, [0, 1, 3, 2]);  getitem_59 = None
        sym_size_36: Sym(197) = torch.ops.aten.sym_size(view_98, 1);  view_98 = None
        expand_33: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_58, [sym_size, 6, sym_size_36, 64]);  getitem_58 = None
        clone_32: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        
        # No stacktrace found for following nodes
        mul_116: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_32: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_32, [mul_116, sym_size_36, 64]);  clone_32 = None
        expand_34: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_59, [sym_size, 6, 64, sym_size_36]);  permute_59 = None
        clone_33: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        _unsafe_view_33: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_33, [mul_116, 64, sym_size_36]);  clone_33 = mul_116 = None
        bmm_16: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_32, _unsafe_view_33);  _unsafe_view_32 = _unsafe_view_33 = None
        view_100: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_16, [sym_size, 6, sym_size_36, sym_size_36]);  bmm_16 = None
        mul_117: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_100, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_66: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_117, torch.float32);  mul_117 = None
        amax_8: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_66, [-1], True)
        sub_25: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_66, amax_8);  convert_element_type_66 = amax_8 = None
        exp_8: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_67: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_8, torch.float16);  div_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_37: Sym(6) = torch.ops.aten.sym_size(view_100, 1);  view_100 = None
        expand_35: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_67, [sym_size, sym_size_37, sym_size_36, sym_size_36]);  convert_element_type_67 = None
        
        # No stacktrace found for following nodes
        mul_118: Sym(6*s0) = sym_size * sym_size_37
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_101: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_35, [mul_118, sym_size_36, sym_size_36]);  expand_35 = None
        expand_36: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_60, [sym_size, sym_size_37, sym_size_36, 64]);  getitem_60 = None
        clone_34: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        _unsafe_view_34: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_34, [mul_118, sym_size_36, 64]);  clone_34 = mul_118 = None
        bmm_17: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_101, _unsafe_view_34);  view_101 = _unsafe_view_34 = None
        view_102: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_17, [sym_size, sym_size_37, sym_size_36, 64]);  bmm_17 = sym_size_37 = None
        permute_60: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_35: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        _unsafe_view_35: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_35, [sym_size, sym_size_3, 384]);  clone_35 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_61: f16[384, 384] = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        
        # No stacktrace found for following nodes
        mul_119: Sym(197*s0) = sym_size * sym_size_36
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_38: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_35, 2)
        view_103: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_35, [mul_119, sym_size_38]);  _unsafe_view_35 = mul_119 = sym_size_38 = None
        addmm_33: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg105_1, view_103, permute_61);  arg105_1 = view_103 = permute_61 = None
        view_104: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_33, [sym_size, sym_size_36, 384]);  addmm_33 = sym_size_36 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_59: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_56, view_104);  add_56 = view_104 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_68: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_68, [2], correction = 0, keepdim = True);  convert_element_type_68 = None
        getitem_61: f32[s0, 197, 1] = var_mean_17[0]
        getitem_62: f32[s0, 197, 1] = var_mean_17[1];  var_mean_17 = None
        add_60: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_61, 1e-06);  getitem_61 = None
        rsqrt_17: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_26: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_59, getitem_62);  getitem_62 = None
        mul_120: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_121: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_120, arg106_1);  mul_120 = arg106_1 = None
        add_61: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_121, arg107_1);  mul_121 = arg107_1 = None
        convert_element_type_69: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_61, torch.float16);  add_61 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_62: f16[384, 1536] = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        
        # No stacktrace found for following nodes
        mul_122: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_105: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_69, [mul_122, 384]);  convert_element_type_69 = mul_122 = None
        addmm_34: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg109_1, view_105, permute_62);  arg109_1 = view_105 = permute_62 = None
        view_106: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_34, [sym_size, sym_size_3, 1536]);  addmm_34 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_70: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_106, torch.float32)
        mul_123: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.5)
        mul_124: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.7071067811865476);  convert_element_type_70 = None
        erf_8: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_124);  mul_124 = None
        add_62: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_125: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_123, add_62);  mul_123 = add_62 = None
        convert_element_type_71: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_125, torch.float16);  mul_125 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_63: f16[1536, 384] = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        sym_size_39: Sym(197) = torch.ops.aten.sym_size(view_106, 1);  view_106 = None
        
        # No stacktrace found for following nodes
        mul_126: Sym(197*s0) = sym_size * sym_size_39
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_107: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_71, [mul_126, 1536]);  convert_element_type_71 = mul_126 = None
        addmm_35: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg111_1, view_107, permute_63);  arg111_1 = view_107 = permute_63 = None
        view_108: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_35, [sym_size, sym_size_39, 384]);  addmm_35 = sym_size_39 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_63: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_59, view_108);  add_59 = view_108 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_72: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True);  convert_element_type_72 = None
        getitem_63: f32[s0, 197, 1] = var_mean_18[0]
        getitem_64: f32[s0, 197, 1] = var_mean_18[1];  var_mean_18 = None
        add_64: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_63, 1e-06);  getitem_63 = None
        rsqrt_18: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_27: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_63, getitem_64);  getitem_64 = None
        mul_127: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_128: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_127, arg112_1);  mul_127 = arg112_1 = None
        add_65: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_128, arg113_1);  mul_128 = arg113_1 = None
        convert_element_type_73: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_64: f16[384, 1152] = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        
        # No stacktrace found for following nodes
        mul_129: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_109: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_73, [mul_129, 384]);  convert_element_type_73 = mul_129 = None
        addmm_36: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg115_1, view_109, permute_64);  arg115_1 = view_109 = permute_64 = None
        view_110: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_36, [sym_size, sym_size_3, 1152]);  addmm_36 = None
        view_111: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_110, [sym_size, sym_size_3, 3, 6, 64])
        permute_65: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_9 = torch.ops.aten.unbind.int(permute_65);  permute_65 = None
        getitem_65: f16[s0, 6, 197, 64] = unbind_9[0]
        getitem_66: f16[s0, 6, 197, 64] = unbind_9[1]
        getitem_67: f16[s0, 6, 197, 64] = unbind_9[2];  unbind_9 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_66: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_66, [0, 1, 3, 2]);  getitem_66 = None
        sym_size_40: Sym(197) = torch.ops.aten.sym_size(view_110, 1);  view_110 = None
        expand_37: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_65, [sym_size, 6, sym_size_40, 64]);  getitem_65 = None
        clone_36: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        
        # No stacktrace found for following nodes
        mul_130: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_36: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_36, [mul_130, sym_size_40, 64]);  clone_36 = None
        expand_38: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_66, [sym_size, 6, 64, sym_size_40]);  permute_66 = None
        clone_37: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        _unsafe_view_37: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_37, [mul_130, 64, sym_size_40]);  clone_37 = mul_130 = None
        bmm_18: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_36, _unsafe_view_37);  _unsafe_view_36 = _unsafe_view_37 = None
        view_112: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_18, [sym_size, 6, sym_size_40, sym_size_40]);  bmm_18 = None
        mul_131: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_112, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_74: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_131, torch.float32);  mul_131 = None
        amax_9: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_74, [-1], True)
        sub_28: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_74, amax_9);  convert_element_type_74 = amax_9 = None
        exp_9: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_75: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_9, torch.float16);  div_9 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_41: Sym(6) = torch.ops.aten.sym_size(view_112, 1);  view_112 = None
        expand_39: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_75, [sym_size, sym_size_41, sym_size_40, sym_size_40]);  convert_element_type_75 = None
        
        # No stacktrace found for following nodes
        mul_132: Sym(6*s0) = sym_size * sym_size_41
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_113: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_39, [mul_132, sym_size_40, sym_size_40]);  expand_39 = None
        expand_40: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_67, [sym_size, sym_size_41, sym_size_40, 64]);  getitem_67 = None
        clone_38: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        _unsafe_view_38: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_38, [mul_132, sym_size_40, 64]);  clone_38 = mul_132 = None
        bmm_19: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_113, _unsafe_view_38);  view_113 = _unsafe_view_38 = None
        view_114: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_19, [sym_size, sym_size_41, sym_size_40, 64]);  bmm_19 = sym_size_41 = None
        permute_67: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        clone_39: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        _unsafe_view_39: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_39, [sym_size, sym_size_3, 384]);  clone_39 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_68: f16[384, 384] = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        
        # No stacktrace found for following nodes
        mul_133: Sym(197*s0) = sym_size * sym_size_40
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_42: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_39, 2)
        view_115: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_39, [mul_133, sym_size_42]);  _unsafe_view_39 = mul_133 = sym_size_42 = None
        addmm_37: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg117_1, view_115, permute_68);  arg117_1 = view_115 = permute_68 = None
        view_116: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_37, [sym_size, sym_size_40, 384]);  addmm_37 = sym_size_40 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_66: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_63, view_116);  add_63 = view_116 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_76: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True);  convert_element_type_76 = None
        getitem_68: f32[s0, 197, 1] = var_mean_19[0]
        getitem_69: f32[s0, 197, 1] = var_mean_19[1];  var_mean_19 = None
        add_67: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
        rsqrt_19: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_29: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_66, getitem_69);  getitem_69 = None
        mul_134: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_135: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_134, arg118_1);  mul_134 = arg118_1 = None
        add_68: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_135, arg119_1);  mul_135 = arg119_1 = None
        convert_element_type_77: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_68, torch.float16);  add_68 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_69: f16[384, 1536] = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        
        # No stacktrace found for following nodes
        mul_136: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_117: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_77, [mul_136, 384]);  convert_element_type_77 = mul_136 = None
        addmm_38: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg121_1, view_117, permute_69);  arg121_1 = view_117 = permute_69 = None
        view_118: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_38, [sym_size, sym_size_3, 1536]);  addmm_38 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_78: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_118, torch.float32)
        mul_137: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.5)
        mul_138: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.7071067811865476);  convert_element_type_78 = None
        erf_9: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_138);  mul_138 = None
        add_69: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_139: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_137, add_69);  mul_137 = add_69 = None
        convert_element_type_79: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_139, torch.float16);  mul_139 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_70: f16[1536, 384] = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        sym_size_43: Sym(197) = torch.ops.aten.sym_size(view_118, 1);  view_118 = None
        
        # No stacktrace found for following nodes
        mul_140: Sym(197*s0) = sym_size * sym_size_43
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_119: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_79, [mul_140, 1536]);  convert_element_type_79 = mul_140 = None
        addmm_39: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg123_1, view_119, permute_70);  arg123_1 = view_119 = permute_70 = None
        view_120: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_39, [sym_size, sym_size_43, 384]);  addmm_39 = sym_size_43 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_70: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_66, view_120);  add_66 = view_120 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_80: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True);  convert_element_type_80 = None
        getitem_70: f32[s0, 197, 1] = var_mean_20[0]
        getitem_71: f32[s0, 197, 1] = var_mean_20[1];  var_mean_20 = None
        add_71: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
        rsqrt_20: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_30: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_70, getitem_71);  getitem_71 = None
        mul_141: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_142: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_141, arg124_1);  mul_141 = arg124_1 = None
        add_72: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_142, arg125_1);  mul_142 = arg125_1 = None
        convert_element_type_81: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_72, torch.float16);  add_72 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_71: f16[384, 1152] = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        
        # No stacktrace found for following nodes
        mul_143: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_121: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_81, [mul_143, 384]);  convert_element_type_81 = mul_143 = None
        addmm_40: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg127_1, view_121, permute_71);  arg127_1 = view_121 = permute_71 = None
        view_122: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_40, [sym_size, sym_size_3, 1152]);  addmm_40 = None
        view_123: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_122, [sym_size, sym_size_3, 3, 6, 64])
        permute_72: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_10 = torch.ops.aten.unbind.int(permute_72);  permute_72 = None
        getitem_72: f16[s0, 6, 197, 64] = unbind_10[0]
        getitem_73: f16[s0, 6, 197, 64] = unbind_10[1]
        getitem_74: f16[s0, 6, 197, 64] = unbind_10[2];  unbind_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_73: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_73, [0, 1, 3, 2]);  getitem_73 = None
        sym_size_44: Sym(197) = torch.ops.aten.sym_size(view_122, 1);  view_122 = None
        expand_41: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_72, [sym_size, 6, sym_size_44, 64]);  getitem_72 = None
        clone_40: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        
        # No stacktrace found for following nodes
        mul_144: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_40: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_40, [mul_144, sym_size_44, 64]);  clone_40 = None
        expand_42: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_73, [sym_size, 6, 64, sym_size_44]);  permute_73 = None
        clone_41: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        _unsafe_view_41: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_41, [mul_144, 64, sym_size_44]);  clone_41 = mul_144 = None
        bmm_20: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_40, _unsafe_view_41);  _unsafe_view_40 = _unsafe_view_41 = None
        view_124: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_20, [sym_size, 6, sym_size_44, sym_size_44]);  bmm_20 = None
        mul_145: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_124, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_82: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_145, torch.float32);  mul_145 = None
        amax_10: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_82, [-1], True)
        sub_31: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_82, amax_10);  convert_element_type_82 = amax_10 = None
        exp_10: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_83: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_10, torch.float16);  div_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_45: Sym(6) = torch.ops.aten.sym_size(view_124, 1);  view_124 = None
        expand_43: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_83, [sym_size, sym_size_45, sym_size_44, sym_size_44]);  convert_element_type_83 = None
        
        # No stacktrace found for following nodes
        mul_146: Sym(6*s0) = sym_size * sym_size_45
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_125: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_43, [mul_146, sym_size_44, sym_size_44]);  expand_43 = None
        expand_44: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_74, [sym_size, sym_size_45, sym_size_44, 64]);  getitem_74 = None
        clone_42: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        _unsafe_view_42: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_42, [mul_146, sym_size_44, 64]);  clone_42 = mul_146 = None
        bmm_21: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_125, _unsafe_view_42);  view_125 = _unsafe_view_42 = None
        view_126: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_21, [sym_size, sym_size_45, sym_size_44, 64]);  bmm_21 = sym_size_45 = None
        permute_74: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_43: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        _unsafe_view_43: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_43, [sym_size, sym_size_3, 384]);  clone_43 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_75: f16[384, 384] = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        
        # No stacktrace found for following nodes
        mul_147: Sym(197*s0) = sym_size * sym_size_44
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_46: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_43, 2)
        view_127: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_43, [mul_147, sym_size_46]);  _unsafe_view_43 = mul_147 = sym_size_46 = None
        addmm_41: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg129_1, view_127, permute_75);  arg129_1 = view_127 = permute_75 = None
        view_128: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_41, [sym_size, sym_size_44, 384]);  addmm_41 = sym_size_44 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_73: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_70, view_128);  add_70 = view_128 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_84: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_84, [2], correction = 0, keepdim = True);  convert_element_type_84 = None
        getitem_75: f32[s0, 197, 1] = var_mean_21[0]
        getitem_76: f32[s0, 197, 1] = var_mean_21[1];  var_mean_21 = None
        add_74: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
        rsqrt_21: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_32: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_73, getitem_76);  getitem_76 = None
        mul_148: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_149: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_148, arg130_1);  mul_148 = arg130_1 = None
        add_75: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_149, arg131_1);  mul_149 = arg131_1 = None
        convert_element_type_85: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_75, torch.float16);  add_75 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_76: f16[384, 1536] = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        
        # No stacktrace found for following nodes
        mul_150: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_129: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_85, [mul_150, 384]);  convert_element_type_85 = mul_150 = None
        addmm_42: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg133_1, view_129, permute_76);  arg133_1 = view_129 = permute_76 = None
        view_130: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_42, [sym_size, sym_size_3, 1536]);  addmm_42 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_86: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_130, torch.float32)
        mul_151: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.5)
        mul_152: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.7071067811865476);  convert_element_type_86 = None
        erf_10: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_76: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_153: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_151, add_76);  mul_151 = add_76 = None
        convert_element_type_87: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_153, torch.float16);  mul_153 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_77: f16[1536, 384] = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        sym_size_47: Sym(197) = torch.ops.aten.sym_size(view_130, 1);  view_130 = None
        
        # No stacktrace found for following nodes
        mul_154: Sym(197*s0) = sym_size * sym_size_47
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_131: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_87, [mul_154, 1536]);  convert_element_type_87 = mul_154 = None
        addmm_43: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg135_1, view_131, permute_77);  arg135_1 = view_131 = permute_77 = None
        view_132: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_43, [sym_size, sym_size_47, 384]);  addmm_43 = sym_size_47 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_77: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_73, view_132);  add_73 = view_132 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        convert_element_type_88: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_77, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_88, [2], correction = 0, keepdim = True);  convert_element_type_88 = None
        getitem_77: f32[s0, 197, 1] = var_mean_22[0]
        getitem_78: f32[s0, 197, 1] = var_mean_22[1];  var_mean_22 = None
        add_78: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
        rsqrt_22: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_33: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_77, getitem_78);  getitem_78 = None
        mul_155: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_156: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_155, arg136_1);  mul_155 = arg136_1 = None
        add_79: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_156, arg137_1);  mul_156 = arg137_1 = None
        convert_element_type_89: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_79, torch.float16);  add_79 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_78: f16[384, 1152] = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        
        # No stacktrace found for following nodes
        mul_157: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:201, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_133: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_89, [mul_157, 384]);  convert_element_type_89 = mul_157 = None
        addmm_44: f16[197*s0, 1152] = torch.ops.aten.addmm.default(arg139_1, view_133, permute_78);  arg139_1 = view_133 = permute_78 = None
        view_134: f16[s0, 197, 1152] = torch.ops.aten.view.default(addmm_44, [sym_size, sym_size_3, 1152]);  addmm_44 = None
        view_135: f16[s0, 197, 3, 6, 64] = torch.ops.aten.view.default(view_134, [sym_size, sym_size_3, 3, 6, 64])
        permute_79: f16[3, s0, 6, 197, 64] = torch.ops.aten.permute.default(view_135, [2, 0, 3, 1, 4]);  view_135 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:202, code: q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_11 = torch.ops.aten.unbind.int(permute_79);  permute_79 = None
        getitem_79: f16[s0, 6, 197, 64] = unbind_11[0]
        getitem_80: f16[s0, 6, 197, 64] = unbind_11[1]
        getitem_81: f16[s0, 6, 197, 64] = unbind_11[2];  unbind_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_80: f16[s0, 6, 64, 197] = torch.ops.aten.permute.default(getitem_80, [0, 1, 3, 2]);  getitem_80 = None
        sym_size_48: Sym(197) = torch.ops.aten.sym_size(view_134, 1);  view_134 = None
        expand_45: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_79, [sym_size, 6, sym_size_48, 64]);  getitem_79 = None
        clone_44: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        
        # No stacktrace found for following nodes
        mul_158: Sym(6*s0) = sym_size * 6
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:204, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        _unsafe_view_44: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_44, [mul_158, sym_size_48, 64]);  clone_44 = None
        expand_46: f16[s0, 6, 64, 197] = torch.ops.aten.expand.default(permute_80, [sym_size, 6, 64, sym_size_48]);  permute_80 = None
        clone_45: f16[s0, 6, 64, 197] = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        _unsafe_view_45: f16[6*s0, 64, 197] = torch.ops.aten._unsafe_view.default(clone_45, [mul_158, 64, sym_size_48]);  clone_45 = mul_158 = None
        bmm_22: f16[6*s0, 197, 197] = torch.ops.aten.bmm.default(_unsafe_view_44, _unsafe_view_45);  _unsafe_view_44 = _unsafe_view_45 = None
        view_136: f16[s0, 6, 197, 197] = torch.ops.aten.view.default(bmm_22, [sym_size, 6, sym_size_48, sym_size_48]);  bmm_22 = None
        mul_159: f16[s0, 6, 197, 197] = torch.ops.aten.mul.Tensor(view_136, 0.125)
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:205, code: attn = attn.softmax(dim=-1)
        convert_element_type_90: f32[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(mul_159, torch.float32);  mul_159 = None
        amax_11: f32[s0, 6, 197, 1] = torch.ops.aten.amax.default(convert_element_type_90, [-1], True)
        sub_34: f32[s0, 6, 197, 197] = torch.ops.aten.sub.Tensor(convert_element_type_90, amax_11);  convert_element_type_90 = amax_11 = None
        exp_11: f32[s0, 6, 197, 197] = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: f32[s0, 6, 197, 1] = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: f32[s0, 6, 197, 197] = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_91: f16[s0, 6, 197, 197] = torch.ops.prims.convert_element_type.default(div_11, torch.float16);  div_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        sym_size_49: Sym(6) = torch.ops.aten.sym_size(view_136, 1);  view_136 = None
        expand_47: f16[s0, 6, 197, 197] = torch.ops.aten.expand.default(convert_element_type_91, [sym_size, sym_size_49, sym_size_48, sym_size_48]);  convert_element_type_91 = None
        
        # No stacktrace found for following nodes
        mul_160: Sym(6*s0) = sym_size * sym_size_49
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:208, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        view_137: f16[6*s0, 197, 197] = torch.ops.aten.view.default(expand_47, [mul_160, sym_size_48, sym_size_48]);  expand_47 = None
        expand_48: f16[s0, 6, 197, 64] = torch.ops.aten.expand.default(getitem_81, [sym_size, sym_size_49, sym_size_48, 64]);  getitem_81 = None
        clone_46: f16[s0, 6, 197, 64] = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        _unsafe_view_46: f16[6*s0, 197, 64] = torch.ops.aten._unsafe_view.default(clone_46, [mul_160, sym_size_48, 64]);  clone_46 = mul_160 = None
        bmm_23: f16[6*s0, 197, 64] = torch.ops.aten.bmm.default(view_137, _unsafe_view_46);  view_137 = _unsafe_view_46 = None
        view_138: f16[s0, 6, 197, 64] = torch.ops.aten.view.default(bmm_23, [sym_size, sym_size_49, sym_size_48, 64]);  bmm_23 = sym_size_49 = None
        permute_81: f16[s0, 197, 6, 64] = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_47: f16[s0, 197, 6, 64] = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        _unsafe_view_47: f16[s0, 197, 384] = torch.ops.aten._unsafe_view.default(clone_47, [sym_size, sym_size_3, 384]);  clone_47 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        permute_82: f16[384, 384] = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        
        # No stacktrace found for following nodes
        mul_161: Sym(197*s0) = sym_size * sym_size_48
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:209, code: x = self.proj(x)
        sym_size_50: Sym(384) = torch.ops.aten.sym_size(_unsafe_view_47, 2)
        view_139: f16[197*s0, 384] = torch.ops.aten.view.default(_unsafe_view_47, [mul_161, sym_size_50]);  _unsafe_view_47 = mul_161 = sym_size_50 = None
        addmm_45: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg141_1, view_139, permute_82);  arg141_1 = view_139 = permute_82 = None
        view_140: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_45, [sym_size, sym_size_48, 384]);  addmm_45 = sym_size_48 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:242, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_80: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_77, view_140);  add_77 = view_140 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_92: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_80, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_92, [2], correction = 0, keepdim = True);  convert_element_type_92 = None
        getitem_82: f32[s0, 197, 1] = var_mean_23[0]
        getitem_83: f32[s0, 197, 1] = var_mean_23[1];  var_mean_23 = None
        add_81: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
        rsqrt_23: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_35: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_80, getitem_83);  getitem_83 = None
        mul_162: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_163: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_162, arg142_1);  mul_162 = arg142_1 = None
        add_82: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_163, arg143_1);  mul_163 = arg143_1 = None
        convert_element_type_93: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_82, torch.float16);  add_82 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        permute_83: f16[384, 1536] = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        
        # No stacktrace found for following nodes
        mul_164: Sym(197*s0) = sym_size * sym_size_3
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:26, code: x = self.fc1(x)
        view_141: f16[197*s0, 384] = torch.ops.aten.view.default(convert_element_type_93, [mul_164, 384]);  convert_element_type_93 = mul_164 = None
        addmm_46: f16[197*s0, 1536] = torch.ops.aten.addmm.default(arg145_1, view_141, permute_83);  arg145_1 = view_141 = permute_83 = None
        view_142: f16[s0, 197, 1536] = torch.ops.aten.view.default(addmm_46, [sym_size, sym_size_3, 1536]);  addmm_46 = sym_size_3 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:27, code: x = self.act(x)
        convert_element_type_94: f32[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(view_142, torch.float32)
        mul_165: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5)
        mul_166: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476);  convert_element_type_94 = None
        erf_11: f32[s0, 197, 1536] = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_83: f32[s0, 197, 1536] = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_167: f32[s0, 197, 1536] = torch.ops.aten.mul.Tensor(mul_165, add_83);  mul_165 = add_83 = None
        convert_element_type_95: f16[s0, 197, 1536] = torch.ops.prims.convert_element_type.default(mul_167, torch.float16);  mul_167 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        permute_84: f16[1536, 384] = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        sym_size_51: Sym(197) = torch.ops.aten.sym_size(view_142, 1);  view_142 = None
        
        # No stacktrace found for following nodes
        mul_168: Sym(197*s0) = sym_size * sym_size_51
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/mlp.py:29, code: x = self.fc2(x)
        view_143: f16[197*s0, 1536] = torch.ops.aten.view.default(convert_element_type_95, [mul_168, 1536]);  convert_element_type_95 = mul_168 = None
        addmm_47: f16[197*s0, 384] = torch.ops.aten.addmm.default(arg147_1, view_143, permute_84);  arg147_1 = view_143 = permute_84 = None
        view_144: f16[s0, 197, 384] = torch.ops.aten.view.default(addmm_47, [sym_size, sym_size_51, 384]);  addmm_47 = sym_size = sym_size_51 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:243, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_84: f16[s0, 197, 384] = torch.ops.aten.add.Tensor(add_80, view_144);  add_80 = view_144 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:440, code: x = self.norm(x)
        convert_element_type_96: f32[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_84, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True);  convert_element_type_96 = None
        getitem_84: f32[s0, 197, 1] = var_mean_24[0]
        getitem_85: f32[s0, 197, 1] = var_mean_24[1];  var_mean_24 = None
        add_85: f32[s0, 197, 1] = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
        rsqrt_24: f32[s0, 197, 1] = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_36: f32[s0, 197, 384] = torch.ops.aten.sub.Tensor(add_84, getitem_85);  add_84 = getitem_85 = None
        mul_169: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_170: f32[s0, 197, 384] = torch.ops.aten.mul.Tensor(mul_169, arg148_1);  mul_169 = arg148_1 = None
        add_86: f32[s0, 197, 384] = torch.ops.aten.add.Tensor(mul_170, arg149_1);  mul_170 = arg149_1 = None
        convert_element_type_97: f16[s0, 197, 384] = torch.ops.prims.convert_element_type.default(add_86, torch.float16);  add_86 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:445, code: x = x[:, self.num_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        slice_1: f16[s0, 197, 384] = torch.ops.aten.slice.Tensor(convert_element_type_97, 0, 0, 9223372036854775807);  convert_element_type_97 = None
        select: f16[s0, 384] = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/vision_transformer.py:447, code: return x if pre_logits else self.head(x)
        permute_85: f16[384, 1000] = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_48: f16[s0, 1000] = torch.ops.aten.addmm.default(arg151_1, select, permute_85);  arg151_1 = select = permute_85 = None
        return (addmm_48,)
        