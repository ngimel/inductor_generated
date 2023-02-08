class GraphModule(torch.nn.Module):
    def forward(self, primals_7: f32[24, 3, 3, 3], primals_8: f32[32, 24, 3, 3], primals_9: f32[64, 32, 3, 3], primals_10: f32[64, 64, 1, 1], primals_11: f32[64, 16, 3, 3], primals_12: f32[1, 1, 3], primals_13: f32[256, 64, 1, 1], primals_14: f32[256, 64, 1, 1], primals_15: f32[64, 256, 1, 1], primals_16: f32[64, 16, 3, 3], primals_17: f32[1, 1, 3], primals_18: f32[256, 64, 1, 1], primals_19: f32[128, 256, 1, 1], primals_20: f32[128, 16, 3, 3], primals_21: f32[1, 1, 5], primals_22: f32[512, 128, 1, 1], primals_23: f32[512, 256, 1, 1], primals_24: f32[128, 512, 1, 1], primals_25: f32[128, 16, 3, 3], primals_26: f32[1, 1, 5], primals_27: f32[512, 128, 1, 1], primals_28: f32[256, 512, 1, 1], primals_29: f32[256, 16, 3, 3], primals_30: f32[1, 1, 5], primals_31: f32[1024, 256, 1, 1], primals_32: f32[1024, 512, 1, 1], primals_33: f32[256, 1024, 1, 1], primals_34: f32[384, 256, 1, 1], primals_35: f32[1024, 256, 1, 1], primals_36: f32[512, 1024, 1, 1], primals_37: f32[640, 512, 1, 1], primals_38: f32[2048, 512, 1, 1], primals_39: f32[2048, 1024, 1, 1], primals_40: f32[512, 2048, 1, 1], primals_41: f32[640, 512, 1, 1], primals_42: f32[2048, 512, 1, 1], primals_45: f32[8, 3, 256, 256], primals_49: f32[24], primals_54: f32[32], primals_59: f32[64], primals_64: f32[64], primals_69: f32[64], primals_74: f32[256], primals_79: f32[256], primals_84: f32[64], primals_89: f32[64], primals_94: f32[256], primals_99: f32[128], primals_104: f32[128], primals_109: f32[512], primals_114: f32[512], primals_119: f32[128], primals_124: f32[128], primals_129: f32[512], primals_134: f32[256], primals_139: f32[256], primals_144: f32[1024], primals_149: f32[1024], primals_154: f32[256], primals_159: f32[256], primals_164: f32[1024], primals_169: f32[512], primals_174: f32[512], primals_179: f32[2048], primals_184: f32[2048], primals_189: f32[512], primals_194: f32[512], primals_199: f32[2048], convolution: f32[8, 24, 128, 128], squeeze_1: f32[24], mul_7: f32[8, 24, 128, 128], convolution_1: f32[8, 32, 128, 128], squeeze_4: f32[32], mul_15: f32[8, 32, 128, 128], convolution_2: f32[8, 64, 128, 128], squeeze_7: f32[64], mul_23: f32[8, 64, 128, 128], getitem: f32[8, 64, 64, 64], getitem_1: i64[8, 64, 64, 64], convolution_3: f32[8, 64, 64, 64], squeeze_10: f32[64], mul_31: f32[8, 64, 64, 64], convolution_4: f32[8, 64, 64, 64], squeeze_13: f32[64], add_24: f32[8, 64, 64, 64], view: f32[8, 1, 64], convolution_5: f32[8, 1, 64], mul_40: f32[8, 64, 64, 64], convolution_6: f32[8, 256, 64, 64], squeeze_16: f32[256], convolution_7: f32[8, 256, 64, 64], squeeze_19: f32[256], mul_55: f32[8, 256, 64, 64], convolution_8: f32[8, 64, 64, 64], squeeze_22: f32[64], mul_63: f32[8, 64, 64, 64], convolution_9: f32[8, 64, 64, 64], squeeze_25: f32[64], add_45: f32[8, 64, 64, 64], view_2: f32[8, 1, 64], convolution_10: f32[8, 1, 64], mul_72: f32[8, 64, 64, 64], convolution_11: f32[8, 256, 64, 64], squeeze_28: f32[256], mul_80: f32[8, 256, 64, 64], convolution_12: f32[8, 128, 64, 64], squeeze_31: f32[128], mul_88: f32[8, 128, 64, 64], convolution_13: f32[8, 128, 32, 32], squeeze_34: f32[128], add_61: f32[8, 128, 32, 32], view_4: f32[8, 1, 128], convolution_14: f32[8, 1, 128], mul_97: f32[8, 128, 32, 32], convolution_15: f32[8, 512, 32, 32], squeeze_37: f32[512], convolution_16: f32[8, 512, 32, 32], squeeze_40: f32[512], mul_112: f32[8, 512, 32, 32], convolution_17: f32[8, 128, 32, 32], squeeze_43: f32[128], mul_120: f32[8, 128, 32, 32], convolution_18: f32[8, 128, 32, 32], squeeze_46: f32[128], add_82: f32[8, 128, 32, 32], view_6: f32[8, 1, 128], convolution_19: f32[8, 1, 128], mul_129: f32[8, 128, 32, 32], convolution_20: f32[8, 512, 32, 32], squeeze_49: f32[512], mul_137: f32[8, 512, 32, 32], convolution_21: f32[8, 256, 32, 32], squeeze_52: f32[256], mul_145: f32[8, 256, 32, 32], convolution_22: f32[8, 256, 16, 16], squeeze_55: f32[256], add_98: f32[8, 256, 16, 16], view_8: f32[8, 1, 256], convolution_23: f32[8, 1, 256], mul_154: f32[8, 256, 16, 16], convolution_24: f32[8, 1024, 16, 16], squeeze_58: f32[1024], convolution_25: f32[8, 1024, 16, 16], squeeze_61: f32[1024], mul_169: f32[8, 1024, 16, 16], convolution_26: f32[8, 256, 16, 16], squeeze_64: f32[256], mul_177: f32[8, 256, 16, 16], _unsafe_view_3: f32[8192, 16], _unsafe_view_4: f32[8192, 16], div: f32[32, 256, 256], squeeze_67: f32[256], mul_186: f32[8, 256, 16, 16], convolution_28: f32[8, 1024, 16, 16], squeeze_70: f32[1024], mul_194: f32[8, 1024, 16, 16], convolution_29: f32[8, 512, 16, 16], squeeze_73: f32[512], mul_202: f32[8, 512, 16, 16], _unsafe_view_10: f32[8192, 16], _unsafe_view_11: f32[8192, 16], div_1: f32[32, 256, 256], _unsafe_view_13: f32[8, 512, 16, 16], avg_pool2d: f32[8, 512, 8, 8], squeeze_76: f32[512], mul_211: f32[8, 512, 8, 8], convolution_31: f32[8, 2048, 8, 8], squeeze_79: f32[2048], convolution_32: f32[8, 2048, 8, 8], squeeze_82: f32[2048], mul_226: f32[8, 2048, 8, 8], convolution_33: f32[8, 512, 8, 8], squeeze_85: f32[512], mul_234: f32[8, 512, 8, 8], _unsafe_view_17: f32[2048, 16], _unsafe_view_18: f32[2048, 16], div_2: f32[32, 64, 64], squeeze_88: f32[512], mul_243: f32[8, 512, 8, 8], convolution_35: f32[8, 2048, 8, 8], squeeze_91: f32[2048], view_61: f32[8, 2048], permute_25: f32[1000, 2048], mul_253: f32[8, 2048, 8, 8], unsqueeze_126: f32[1, 2048, 1, 1], mul_265: f32[8, 512, 8, 8], sub_40: f32[8, 512, 8, 8], permute_30: f32[32, 64, 64], permute_31: f32[32, 128, 64], permute_35: f32[15, 16], permute_41: f32[15, 16], permute_43: f32[32, 16, 64], permute_44: f32[32, 64, 16], mul_280: f32[8, 512, 8, 8], unsqueeze_150: f32[1, 512, 1, 1], mul_292: f32[8, 2048, 8, 8], unsqueeze_162: f32[1, 2048, 1, 1], unsqueeze_174: f32[1, 2048, 1, 1], mul_313: f32[8, 512, 8, 8], unsqueeze_186: f32[1, 512, 1, 1], permute_48: f32[32, 256, 256], permute_49: f32[32, 128, 256], permute_53: f32[31, 16], permute_59: f32[31, 16], permute_61: f32[32, 16, 256], permute_62: f32[32, 256, 16], mul_328: f32[8, 512, 16, 16], unsqueeze_198: f32[1, 512, 1, 1], mul_340: f32[8, 1024, 16, 16], unsqueeze_210: f32[1, 1024, 1, 1], mul_352: f32[8, 256, 16, 16], sub_76: f32[8, 256, 16, 16], permute_66: f32[32, 256, 256], permute_67: f32[32, 64, 256], permute_71: f32[31, 16], permute_77: f32[31, 16], permute_79: f32[32, 16, 256], permute_80: f32[32, 256, 16], mul_367: f32[8, 256, 16, 16], unsqueeze_234: f32[1, 256, 1, 1], mul_379: f32[8, 1024, 16, 16], unsqueeze_246: f32[1, 1024, 1, 1], unsqueeze_258: f32[1, 1024, 1, 1], unsqueeze_272: f32[1, 256, 1, 1], mul_416: f32[8, 256, 32, 32], unsqueeze_284: f32[1, 256, 1, 1], mul_428: f32[8, 512, 32, 32], unsqueeze_296: f32[1, 512, 1, 1], unsqueeze_310: f32[1, 128, 1, 1], mul_456: f32[8, 128, 32, 32], unsqueeze_322: f32[1, 128, 1, 1], mul_468: f32[8, 512, 32, 32], unsqueeze_334: f32[1, 512, 1, 1], unsqueeze_346: f32[1, 512, 1, 1], unsqueeze_360: f32[1, 128, 1, 1], mul_505: f32[8, 128, 64, 64], unsqueeze_372: f32[1, 128, 1, 1], mul_517: f32[8, 256, 64, 64], unsqueeze_384: f32[1, 256, 1, 1], unsqueeze_398: f32[1, 64, 1, 1], mul_545: f32[8, 64, 64, 64], unsqueeze_410: f32[1, 64, 1, 1], mul_557: f32[8, 256, 64, 64], unsqueeze_422: f32[1, 256, 1, 1], unsqueeze_434: f32[1, 256, 1, 1], unsqueeze_448: f32[1, 64, 1, 1], mul_594: f32[8, 64, 64, 64], unsqueeze_460: f32[1, 64, 1, 1], mul_606: f32[8, 64, 128, 128], unsqueeze_472: f32[1, 64, 1, 1], mul_618: f32[8, 32, 128, 128], unsqueeze_484: f32[1, 32, 1, 1], mul_630: f32[8, 24, 128, 128], unsqueeze_496: f32[1, 24, 1, 1], tangents_1: f32[24], tangents_2: f32[24], tangents_3: f32[32], tangents_4: f32[32], tangents_5: f32[64], tangents_6: f32[64], tangents_7: f32[64], tangents_8: f32[64], tangents_9: f32[64], tangents_10: f32[64], tangents_11: f32[256], tangents_12: f32[256], tangents_13: f32[256], tangents_14: f32[256], tangents_15: f32[64], tangents_16: f32[64], tangents_17: f32[64], tangents_18: f32[64], tangents_19: f32[256], tangents_20: f32[256], tangents_21: f32[128], tangents_22: f32[128], tangents_23: f32[128], tangents_24: f32[128], tangents_25: f32[512], tangents_26: f32[512], tangents_27: f32[512], tangents_28: f32[512], tangents_29: f32[128], tangents_30: f32[128], tangents_31: f32[128], tangents_32: f32[128], tangents_33: f32[512], tangents_34: f32[512], tangents_35: f32[256], tangents_36: f32[256], tangents_37: f32[256], tangents_38: f32[256], tangents_39: f32[1024], tangents_40: f32[1024], tangents_41: f32[1024], tangents_42: f32[1024], tangents_43: f32[256], tangents_44: f32[256], tangents_45: f32[256], tangents_46: f32[256], tangents_47: f32[1024], tangents_48: f32[1024], tangents_49: f32[512], tangents_50: f32[512], tangents_51: f32[512], tangents_52: f32[512], tangents_53: f32[2048], tangents_54: f32[2048], tangents_55: f32[2048], tangents_56: f32[2048], tangents_57: f32[512], tangents_58: f32[512], tangents_59: f32[512], tangents_60: f32[512], tangents_61: f32[2048], tangents_62: f32[2048], tangents_63: f32[8, 1000], tangents_64: i64[], tangents_65: i64[], tangents_66: i64[], tangents_67: i64[], tangents_68: i64[], tangents_69: i64[], tangents_70: i64[], tangents_71: i64[], tangents_72: i64[], tangents_73: i64[], tangents_74: i64[], tangents_75: i64[], tangents_76: i64[], tangents_77: i64[], tangents_78: i64[], tangents_79: i64[], tangents_80: i64[], tangents_81: i64[], tangents_82: i64[], tangents_83: i64[], tangents_84: i64[], tangents_85: i64[], tangents_86: i64[], tangents_87: i64[], tangents_88: i64[], tangents_89: i64[], tangents_90: i64[], tangents_91: i64[], tangents_92: i64[], tangents_93: i64[], tangents_94: i64[]):
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        clone_66: f32[8, 64, 64, 64] = torch.ops.aten.clone.default(add_24)
        sigmoid_4: f32[8, 64, 64, 64] = torch.ops.aten.sigmoid.default(add_24)
        mul_39: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(add_24, sigmoid_4);  add_24 = sigmoid_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: f32[8, 1, 64] = torch.ops.aten.sigmoid.default(convolution_5);  convolution_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_1: f32[8, 64, 1, 1] = torch.ops.aten.view.default(sigmoid_5, [8, -1, 1, 1])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        expand: f32[8, 64, 64, 64] = torch.ops.aten.expand.default(view_1, [8, 64, 64, 64]);  view_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        clone_69: f32[8, 64, 64, 64] = torch.ops.aten.clone.default(add_45)
        sigmoid_8: f32[8, 64, 64, 64] = torch.ops.aten.sigmoid.default(add_45)
        mul_71: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(add_45, sigmoid_8);  add_45 = sigmoid_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: f32[8, 1, 64] = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_3: f32[8, 64, 1, 1] = torch.ops.aten.view.default(sigmoid_9, [8, -1, 1, 1])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        expand_1: f32[8, 64, 64, 64] = torch.ops.aten.expand.default(view_3, [8, 64, 64, 64]);  view_3 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        clone_72: f32[8, 128, 32, 32] = torch.ops.aten.clone.default(add_61)
        sigmoid_12: f32[8, 128, 32, 32] = torch.ops.aten.sigmoid.default(add_61)
        mul_96: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(add_61, sigmoid_12);  add_61 = sigmoid_12 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_13: f32[8, 1, 128] = torch.ops.aten.sigmoid.default(convolution_14);  convolution_14 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_5: f32[8, 128, 1, 1] = torch.ops.aten.view.default(sigmoid_13, [8, -1, 1, 1])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        expand_2: f32[8, 128, 32, 32] = torch.ops.aten.expand.default(view_5, [8, 128, 32, 32]);  view_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        clone_75: f32[8, 128, 32, 32] = torch.ops.aten.clone.default(add_82)
        sigmoid_16: f32[8, 128, 32, 32] = torch.ops.aten.sigmoid.default(add_82)
        mul_128: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(add_82, sigmoid_16);  add_82 = sigmoid_16 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_17: f32[8, 1, 128] = torch.ops.aten.sigmoid.default(convolution_19);  convolution_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_7: f32[8, 128, 1, 1] = torch.ops.aten.view.default(sigmoid_17, [8, -1, 1, 1])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        expand_3: f32[8, 128, 32, 32] = torch.ops.aten.expand.default(view_7, [8, 128, 32, 32]);  view_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        clone_78: f32[8, 256, 16, 16] = torch.ops.aten.clone.default(add_98)
        sigmoid_20: f32[8, 256, 16, 16] = torch.ops.aten.sigmoid.default(add_98)
        mul_153: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(add_98, sigmoid_20);  add_98 = sigmoid_20 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_21: f32[8, 1, 256] = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_9: f32[8, 256, 1, 1] = torch.ops.aten.view.default(sigmoid_21, [8, -1, 1, 1])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        expand_4: f32[8, 256, 16, 16] = torch.ops.aten.expand.default(view_9, [8, 256, 16, 16]);  view_9 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/classifier.py:55, code: x = self.fc(x)
        mm_6: f32[8, 2048] = torch.ops.aten.mm.default(tangents_63, permute_25);  permute_25 = None
        permute_26: f32[1000, 8] = torch.ops.aten.permute.default(tangents_63, [1, 0])
        mm_7: f32[1000, 2048] = torch.ops.aten.mm.default(permute_26, view_61);  permute_26 = view_61 = None
        permute_27: f32[2048, 1000] = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_4: f32[1, 1000] = torch.ops.aten.sum.dim_IntList(tangents_63, [0], True);  tangents_63 = None
        view_62: f32[1000] = torch.ops.aten.view.default(sum_4, [1000]);  sum_4 = None
        permute_28: f32[1000, 2048] = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/adaptive_avgmax_pool.py:108, code: x = self.flatten(x)
        view_63: f32[8, 2048, 1, 1] = torch.ops.aten.view.default(mm_6, [8, 2048, 1, 1]);  mm_6 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/adaptive_avgmax_pool.py:107, code: x = self.pool(x)
        expand_23: f32[8, 2048, 8, 8] = torch.ops.aten.expand.default(view_63, [8, 2048, 8, 8]);  view_63 = None
        div_3: f32[8, 2048, 8, 8] = torch.ops.aten.div.Scalar(expand_23, 64);  expand_23 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1251, code: return self.act(x)
        mul_254: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(div_3, mul_253);  div_3 = mul_253 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_5: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_254, [0, 2, 3])
        sub_35: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_126);  convolution_35 = unsqueeze_126 = None
        mul_255: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(mul_254, sub_35)
        sum_6: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3]);  mul_255 = None
        mul_256: f32[2048] = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
        unsqueeze_127: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
        unsqueeze_128: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        unsqueeze_129: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
        mul_257: f32[2048] = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
        mul_258: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_259: f32[2048] = torch.ops.aten.mul.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
        unsqueeze_130: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_259, 0);  mul_259 = None
        unsqueeze_131: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
        unsqueeze_132: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
        mul_260: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_91, primals_199);  primals_199 = None
        unsqueeze_133: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
        unsqueeze_134: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
        unsqueeze_135: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
        mul_261: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_132);  sub_35 = unsqueeze_132 = None
        sub_37: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(mul_254, mul_261);  mul_261 = None
        sub_38: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_129);  sub_37 = unsqueeze_129 = None
        mul_262: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_135);  sub_38 = unsqueeze_135 = None
        mul_263: f32[2048] = torch.ops.aten.mul.Tensor(sum_6, squeeze_91);  sum_6 = squeeze_91 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_262, mul_243, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_262 = mul_243 = primals_42 = None
        getitem_11: f32[8, 512, 8, 8] = convolution_backward[0]
        getitem_12: f32[2048, 512, 1, 1] = convolution_backward[1];  convolution_backward = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_266: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(getitem_11, mul_265);  getitem_11 = mul_265 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_7: f32[512] = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3])
        mul_267: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(mul_266, sub_40)
        sum_8: f32[512] = torch.ops.aten.sum.dim_IntList(mul_267, [0, 2, 3]);  mul_267 = None
        mul_268: f32[512] = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
        unsqueeze_139: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
        unsqueeze_140: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        mul_269: f32[512] = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
        mul_270: f32[512] = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_271: f32[512] = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
        unsqueeze_142: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_143: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        mul_272: f32[512] = torch.ops.aten.mul.Tensor(squeeze_88, primals_194);  primals_194 = None
        unsqueeze_145: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
        unsqueeze_146: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        mul_273: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_144);  sub_40 = unsqueeze_144 = None
        sub_42: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(mul_266, mul_273);  mul_266 = mul_273 = None
        sub_43: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_141);  sub_42 = unsqueeze_141 = None
        mul_274: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_147);  sub_43 = unsqueeze_147 = None
        mul_275: f32[512] = torch.ops.aten.mul.Tensor(sum_8, squeeze_88);  sum_8 = squeeze_88 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        view_64: f32[32, 128, 64] = torch.ops.aten.view.default(mul_274, [32, 128, 64]);  mul_274 = None
        permute_29: f32[32, 64, 128] = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        view_65: f32[32, 64, 128] = torch.ops.aten.view.default(permute_29, [32, 64, 128]);  permute_29 = None
        bmm_6: f32[32, 64, 128] = torch.ops.aten.bmm.default(permute_30, view_65);  permute_30 = None
        bmm_7: f32[32, 64, 64] = torch.ops.aten.bmm.default(view_65, permute_31);  view_65 = permute_31 = None
        view_66: f32[32, 64, 128] = torch.ops.aten.view.default(bmm_6, [32, 64, 128]);  bmm_6 = None
        view_67: f32[32, 64, 64] = torch.ops.aten.view.default(bmm_7, [32, 64, 64]);  bmm_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
        mul_276: f32[32, 64, 64] = torch.ops.aten.mul.Tensor(view_67, div_2);  view_67 = None
        sum_9: f32[32, 64, 1] = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        mul_277: f32[32, 64, 64] = torch.ops.aten.mul.Tensor(div_2, sum_9);  div_2 = sum_9 = None
        sub_44: f32[32, 64, 64] = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
        view_68: f32[32, 8, 8, 8, 8] = torch.ops.aten.view.default(sub_44, [32, 8, 8, 8, 8])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
        permute_32: f32[32, 8, 8, 8, 8] = torch.ops.aten.permute.default(view_68, [0, 2, 4, 1, 3])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        sum_10: f32[32, 8, 1, 8, 8] = torch.ops.aten.sum.dim_IntList(permute_32, [2], True);  permute_32 = None
        view_69: f32[256, 8, 8] = torch.ops.aten.view.default(sum_10, [256, 8, 8]);  sum_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
        full: f32[256, 8, 15] = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: f32[256, 8, 15] = torch.ops.aten.slice_scatter.default(full, view_69, 2, 7, 9223372036854775807);  view_69 = None
        full_1: f32[256, 9, 15] = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1: f32[256, 9, 15] = torch.ops.aten.slice_scatter.default(full_1, slice_scatter, 1, 0, 8);  slice_scatter = None
        slice_scatter_2: f32[256, 9, 15] = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_1, 0, 0, 9223372036854775807);  slice_scatter_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_70: f32[256, 135] = torch.ops.aten.view.default(slice_scatter_2, [256, 135]);  slice_scatter_2 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
        constant_pad_nd_12: f32[256, 128] = torch.ops.aten.constant_pad_nd.default(view_70, [0, -7]);  view_70 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_71: f32[256, 8, 16] = torch.ops.aten.view.default(constant_pad_nd_12, [256, 8, 16]);  constant_pad_nd_12 = None
        constant_pad_nd_13: f32[256, 8, 15] = torch.ops.aten.constant_pad_nd.default(view_71, [0, -1]);  view_71 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
        view_72: f32[32, 8, 8, 15] = torch.ops.aten.view.default(constant_pad_nd_13, [32, 8, 8, 15]);  constant_pad_nd_13 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        view_73: f32[2048, 15] = torch.ops.aten.view.default(view_72, [2048, 15]);  view_72 = None
        permute_33: f32[15, 2048] = torch.ops.aten.permute.default(view_73, [1, 0])
        mm_8: f32[15, 16] = torch.ops.aten.mm.default(permute_33, _unsafe_view_18);  permute_33 = _unsafe_view_18 = None
        permute_34: f32[16, 15] = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
        mm_9: f32[2048, 16] = torch.ops.aten.mm.default(view_73, permute_35);  view_73 = permute_35 = None
        view_74: f32[32, 8, 8, 16] = torch.ops.aten.view.default(mm_9, [32, 8, 8, 16]);  mm_9 = None
        permute_36: f32[15, 16] = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
        permute_37: f32[32, 8, 8, 16] = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
        permute_38: f32[32, 8, 8, 8, 8] = torch.ops.aten.permute.default(view_68, [0, 1, 3, 2, 4]);  view_68 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        sum_11: f32[32, 8, 1, 8, 8] = torch.ops.aten.sum.dim_IntList(permute_38, [2], True);  permute_38 = None
        view_75: f32[256, 8, 8] = torch.ops.aten.view.default(sum_11, [256, 8, 8]);  sum_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
        slice_scatter_3: f32[256, 8, 15] = torch.ops.aten.slice_scatter.default(full, view_75, 2, 7, 9223372036854775807);  full = view_75 = None
        slice_scatter_4: f32[256, 9, 15] = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_3, 1, 0, 8);  slice_scatter_3 = None
        slice_scatter_5: f32[256, 9, 15] = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_4, 0, 0, 9223372036854775807);  full_1 = slice_scatter_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_76: f32[256, 135] = torch.ops.aten.view.default(slice_scatter_5, [256, 135]);  slice_scatter_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
        constant_pad_nd_14: f32[256, 128] = torch.ops.aten.constant_pad_nd.default(view_76, [0, -7]);  view_76 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_77: f32[256, 8, 16] = torch.ops.aten.view.default(constant_pad_nd_14, [256, 8, 16]);  constant_pad_nd_14 = None
        constant_pad_nd_15: f32[256, 8, 15] = torch.ops.aten.constant_pad_nd.default(view_77, [0, -1]);  view_77 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
        view_78: f32[32, 8, 8, 15] = torch.ops.aten.view.default(constant_pad_nd_15, [32, 8, 8, 15]);  constant_pad_nd_15 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        view_79: f32[2048, 15] = torch.ops.aten.view.default(view_78, [2048, 15]);  view_78 = None
        permute_39: f32[15, 2048] = torch.ops.aten.permute.default(view_79, [1, 0])
        mm_10: f32[15, 16] = torch.ops.aten.mm.default(permute_39, _unsafe_view_17);  permute_39 = _unsafe_view_17 = None
        permute_40: f32[16, 15] = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
        mm_11: f32[2048, 16] = torch.ops.aten.mm.default(view_79, permute_41);  view_79 = permute_41 = None
        view_80: f32[32, 8, 8, 16] = torch.ops.aten.view.default(mm_11, [32, 8, 8, 16]);  mm_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        add_171: f32[32, 8, 8, 16] = torch.ops.aten.add.Tensor(permute_37, view_80);  permute_37 = view_80 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        permute_42: f32[15, 16] = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
        clone_110: f32[32, 8, 8, 16] = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format);  add_171 = None
        _unsafe_view_21: f32[32, 64, 16] = torch.ops.aten._unsafe_view.default(clone_110, [32, 64, 16]);  clone_110 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        mul_278: f32[32, 64, 64] = torch.ops.aten.mul.Tensor(sub_44, 0.25);  sub_44 = None
        view_81: f32[32, 64, 64] = torch.ops.aten.view.default(mul_278, [32, 64, 64]);  mul_278 = None
        bmm_8: f32[32, 16, 64] = torch.ops.aten.bmm.default(permute_43, view_81);  permute_43 = None
        bmm_9: f32[32, 64, 16] = torch.ops.aten.bmm.default(view_81, permute_44);  view_81 = permute_44 = None
        view_82: f32[32, 16, 64] = torch.ops.aten.view.default(bmm_8, [32, 16, 64]);  bmm_8 = None
        view_83: f32[32, 64, 16] = torch.ops.aten.view.default(bmm_9, [32, 64, 16]);  bmm_9 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_172: f32[32, 64, 16] = torch.ops.aten.add.Tensor(_unsafe_view_21, view_83);  _unsafe_view_21 = view_83 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        permute_45: f32[32, 128, 64] = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        clone_111: f32[32, 128, 64] = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        _unsafe_view_22: f32[8, 512, 8, 8] = torch.ops.aten._unsafe_view.default(clone_111, [8, 512, 8, 8]);  clone_111 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        view_84: f32[8, 64, 8, 8] = torch.ops.aten.view.default(view_82, [8, 64, 8, 8]);  view_82 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        permute_46: f32[32, 16, 64] = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
        clone_112: f32[32, 16, 64] = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        _unsafe_view_23: f32[8, 64, 8, 8] = torch.ops.aten._unsafe_view.default(clone_112, [8, 64, 8, 8]);  clone_112 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        cat: f32[8, 640, 8, 8] = torch.ops.aten.cat.default([_unsafe_view_23, view_84, _unsafe_view_22], 1);  _unsafe_view_23 = view_84 = _unsafe_view_22 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(cat, mul_234, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat = mul_234 = primals_41 = None
        getitem_14: f32[8, 512, 8, 8] = convolution_backward_1[0]
        getitem_15: f32[640, 512, 1, 1] = convolution_backward_1[1];  convolution_backward_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_281: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(getitem_14, mul_280);  getitem_14 = mul_280 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_12: f32[512] = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3])
        sub_46: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_150);  convolution_33 = unsqueeze_150 = None
        mul_282: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(mul_281, sub_46)
        sum_13: f32[512] = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
        mul_283: f32[512] = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
        unsqueeze_151: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
        unsqueeze_152: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        mul_284: f32[512] = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
        mul_285: f32[512] = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_286: f32[512] = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_154: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_155: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        mul_287: f32[512] = torch.ops.aten.mul.Tensor(squeeze_85, primals_189);  primals_189 = None
        unsqueeze_157: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_158: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        mul_288: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_156);  sub_46 = unsqueeze_156 = None
        sub_48: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(mul_281, mul_288);  mul_281 = mul_288 = None
        sub_49: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_153);  sub_48 = unsqueeze_153 = None
        mul_289: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_159);  sub_49 = unsqueeze_159 = None
        mul_290: f32[512] = torch.ops.aten.mul.Tensor(sum_13, squeeze_85);  sum_13 = squeeze_85 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_289, mul_226, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = mul_226 = primals_40 = None
        getitem_17: f32[8, 2048, 8, 8] = convolution_backward_2[0]
        getitem_18: f32[512, 2048, 1, 1] = convolution_backward_2[1];  convolution_backward_2 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_174: f32[8, 2048, 8, 8] = torch.ops.aten.add.Tensor(mul_254, getitem_17);  mul_254 = getitem_17 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1251, code: return self.act(x)
        mul_293: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(add_174, mul_292);  add_174 = mul_292 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_14: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3])
        sub_51: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_162);  convolution_32 = unsqueeze_162 = None
        mul_294: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(mul_293, sub_51)
        sum_15: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_294, [0, 2, 3]);  mul_294 = None
        mul_295: f32[2048] = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
        unsqueeze_163: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
        unsqueeze_164: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_296: f32[2048] = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
        mul_297: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_298: f32[2048] = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        unsqueeze_166: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_167: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        mul_299: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_82, primals_184);  primals_184 = None
        unsqueeze_169: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
        unsqueeze_170: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_300: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_168);  sub_51 = unsqueeze_168 = None
        sub_53: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(mul_293, mul_300);  mul_300 = None
        sub_54: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_165);  sub_53 = None
        mul_301: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_171);  sub_54 = unsqueeze_171 = None
        mul_302: f32[2048] = torch.ops.aten.mul.Tensor(sum_15, squeeze_82);  sum_15 = squeeze_82 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_301, mul_194, primals_39, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_301 = primals_39 = None
        getitem_20: f32[8, 1024, 16, 16] = convolution_backward_3[0]
        getitem_21: f32[2048, 1024, 1, 1] = convolution_backward_3[1];  convolution_backward_3 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sub_55: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_174);  convolution_31 = unsqueeze_174 = None
        mul_303: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(mul_293, sub_55)
        sum_17: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
        mul_305: f32[2048] = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
        mul_306: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_307: f32[2048] = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_178: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
        unsqueeze_179: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        mul_308: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_79, primals_179);  primals_179 = None
        unsqueeze_181: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
        unsqueeze_182: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
        unsqueeze_183: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
        mul_309: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_180);  sub_55 = unsqueeze_180 = None
        sub_57: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(mul_293, mul_309);  mul_293 = mul_309 = None
        sub_58: f32[8, 2048, 8, 8] = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_165);  sub_57 = unsqueeze_165 = None
        mul_310: f32[8, 2048, 8, 8] = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_183);  sub_58 = unsqueeze_183 = None
        mul_311: f32[2048] = torch.ops.aten.mul.Tensor(sum_17, squeeze_79);  sum_17 = squeeze_79 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_310, mul_211, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_310 = mul_211 = primals_38 = None
        getitem_23: f32[8, 512, 8, 8] = convolution_backward_4[0]
        getitem_24: f32[2048, 512, 1, 1] = convolution_backward_4[1];  convolution_backward_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_314: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(getitem_23, mul_313);  getitem_23 = mul_313 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_18: f32[512] = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
        sub_60: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_186);  avg_pool2d = unsqueeze_186 = None
        mul_315: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(mul_314, sub_60)
        sum_19: f32[512] = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
        mul_316: f32[512] = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
        unsqueeze_187: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
        unsqueeze_188: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        unsqueeze_189: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
        mul_317: f32[512] = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
        mul_318: f32[512] = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_319: f32[512] = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        unsqueeze_190: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
        unsqueeze_191: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
        unsqueeze_192: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
        mul_320: f32[512] = torch.ops.aten.mul.Tensor(squeeze_76, primals_174);  primals_174 = None
        unsqueeze_193: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
        unsqueeze_194: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
        unsqueeze_195: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
        mul_321: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_192);  sub_60 = unsqueeze_192 = None
        sub_62: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(mul_314, mul_321);  mul_314 = mul_321 = None
        sub_63: f32[8, 512, 8, 8] = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_189);  sub_62 = unsqueeze_189 = None
        mul_322: f32[8, 512, 8, 8] = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_195);  sub_63 = unsqueeze_195 = None
        mul_323: f32[512] = torch.ops.aten.mul.Tensor(sum_19, squeeze_76);  sum_19 = squeeze_76 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:156, code: out = self.pool(out)
        avg_pool2d_backward: f32[8, 512, 16, 16] = torch.ops.aten.avg_pool2d_backward.default(mul_322, _unsafe_view_13, [2, 2], [2, 2], [0, 0], False, True, None);  mul_322 = _unsafe_view_13 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        view_85: f32[32, 128, 256] = torch.ops.aten.view.default(avg_pool2d_backward, [32, 128, 256]);  avg_pool2d_backward = None
        permute_47: f32[32, 256, 128] = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        view_86: f32[32, 256, 128] = torch.ops.aten.view.default(permute_47, [32, 256, 128]);  permute_47 = None
        bmm_10: f32[32, 256, 128] = torch.ops.aten.bmm.default(permute_48, view_86);  permute_48 = None
        bmm_11: f32[32, 256, 256] = torch.ops.aten.bmm.default(view_86, permute_49);  view_86 = permute_49 = None
        view_87: f32[32, 256, 128] = torch.ops.aten.view.default(bmm_10, [32, 256, 128]);  bmm_10 = None
        view_88: f32[32, 256, 256] = torch.ops.aten.view.default(bmm_11, [32, 256, 256]);  bmm_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
        mul_324: f32[32, 256, 256] = torch.ops.aten.mul.Tensor(view_88, div_1);  view_88 = None
        sum_20: f32[32, 256, 1] = torch.ops.aten.sum.dim_IntList(mul_324, [-1], True)
        mul_325: f32[32, 256, 256] = torch.ops.aten.mul.Tensor(div_1, sum_20);  div_1 = sum_20 = None
        sub_64: f32[32, 256, 256] = torch.ops.aten.sub.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
        view_89: f32[32, 16, 16, 16, 16] = torch.ops.aten.view.default(sub_64, [32, 16, 16, 16, 16])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
        permute_50: f32[32, 16, 16, 16, 16] = torch.ops.aten.permute.default(view_89, [0, 2, 4, 1, 3])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        sum_21: f32[32, 16, 1, 16, 16] = torch.ops.aten.sum.dim_IntList(permute_50, [2], True);  permute_50 = None
        view_90: f32[512, 16, 16] = torch.ops.aten.view.default(sum_21, [512, 16, 16]);  sum_21 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
        full_6: f32[512, 16, 31] = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_6: f32[512, 16, 31] = torch.ops.aten.slice_scatter.default(full_6, view_90, 2, 15, 9223372036854775807);  view_90 = None
        full_7: f32[512, 17, 31] = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_7: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_6, 1, 0, 16);  slice_scatter_6 = None
        slice_scatter_8: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_91: f32[512, 527] = torch.ops.aten.view.default(slice_scatter_8, [512, 527]);  slice_scatter_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
        constant_pad_nd_16: f32[512, 512] = torch.ops.aten.constant_pad_nd.default(view_91, [0, -15]);  view_91 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_92: f32[512, 16, 32] = torch.ops.aten.view.default(constant_pad_nd_16, [512, 16, 32]);  constant_pad_nd_16 = None
        constant_pad_nd_17: f32[512, 16, 31] = torch.ops.aten.constant_pad_nd.default(view_92, [0, -1]);  view_92 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
        view_93: f32[32, 16, 16, 31] = torch.ops.aten.view.default(constant_pad_nd_17, [32, 16, 16, 31]);  constant_pad_nd_17 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        view_94: f32[8192, 31] = torch.ops.aten.view.default(view_93, [8192, 31]);  view_93 = None
        permute_51: f32[31, 8192] = torch.ops.aten.permute.default(view_94, [1, 0])
        mm_12: f32[31, 16] = torch.ops.aten.mm.default(permute_51, _unsafe_view_11);  permute_51 = _unsafe_view_11 = None
        permute_52: f32[16, 31] = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
        mm_13: f32[8192, 16] = torch.ops.aten.mm.default(view_94, permute_53);  view_94 = permute_53 = None
        view_95: f32[32, 16, 16, 16] = torch.ops.aten.view.default(mm_13, [32, 16, 16, 16]);  mm_13 = None
        permute_54: f32[31, 16] = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
        permute_55: f32[32, 16, 16, 16] = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
        permute_56: f32[32, 16, 16, 16, 16] = torch.ops.aten.permute.default(view_89, [0, 1, 3, 2, 4]);  view_89 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        sum_22: f32[32, 16, 1, 16, 16] = torch.ops.aten.sum.dim_IntList(permute_56, [2], True);  permute_56 = None
        view_96: f32[512, 16, 16] = torch.ops.aten.view.default(sum_22, [512, 16, 16]);  sum_22 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
        slice_scatter_9: f32[512, 16, 31] = torch.ops.aten.slice_scatter.default(full_6, view_96, 2, 15, 9223372036854775807);  view_96 = None
        slice_scatter_10: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_9, 1, 0, 16);  slice_scatter_9 = None
        slice_scatter_11: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_10, 0, 0, 9223372036854775807);  slice_scatter_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_97: f32[512, 527] = torch.ops.aten.view.default(slice_scatter_11, [512, 527]);  slice_scatter_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
        constant_pad_nd_18: f32[512, 512] = torch.ops.aten.constant_pad_nd.default(view_97, [0, -15]);  view_97 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_98: f32[512, 16, 32] = torch.ops.aten.view.default(constant_pad_nd_18, [512, 16, 32]);  constant_pad_nd_18 = None
        constant_pad_nd_19: f32[512, 16, 31] = torch.ops.aten.constant_pad_nd.default(view_98, [0, -1]);  view_98 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
        view_99: f32[32, 16, 16, 31] = torch.ops.aten.view.default(constant_pad_nd_19, [32, 16, 16, 31]);  constant_pad_nd_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        view_100: f32[8192, 31] = torch.ops.aten.view.default(view_99, [8192, 31]);  view_99 = None
        permute_57: f32[31, 8192] = torch.ops.aten.permute.default(view_100, [1, 0])
        mm_14: f32[31, 16] = torch.ops.aten.mm.default(permute_57, _unsafe_view_10);  permute_57 = _unsafe_view_10 = None
        permute_58: f32[16, 31] = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
        mm_15: f32[8192, 16] = torch.ops.aten.mm.default(view_100, permute_59);  view_100 = permute_59 = None
        view_101: f32[32, 16, 16, 16] = torch.ops.aten.view.default(mm_15, [32, 16, 16, 16]);  mm_15 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        add_177: f32[32, 16, 16, 16] = torch.ops.aten.add.Tensor(permute_55, view_101);  permute_55 = view_101 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        permute_60: f32[31, 16] = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
        clone_113: f32[32, 16, 16, 16] = torch.ops.aten.clone.default(add_177, memory_format = torch.contiguous_format);  add_177 = None
        _unsafe_view_24: f32[32, 256, 16] = torch.ops.aten._unsafe_view.default(clone_113, [32, 256, 16]);  clone_113 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        mul_326: f32[32, 256, 256] = torch.ops.aten.mul.Tensor(sub_64, 0.25);  sub_64 = None
        view_102: f32[32, 256, 256] = torch.ops.aten.view.default(mul_326, [32, 256, 256]);  mul_326 = None
        bmm_12: f32[32, 16, 256] = torch.ops.aten.bmm.default(permute_61, view_102);  permute_61 = None
        bmm_13: f32[32, 256, 16] = torch.ops.aten.bmm.default(view_102, permute_62);  view_102 = permute_62 = None
        view_103: f32[32, 16, 256] = torch.ops.aten.view.default(bmm_12, [32, 16, 256]);  bmm_12 = None
        view_104: f32[32, 256, 16] = torch.ops.aten.view.default(bmm_13, [32, 256, 16]);  bmm_13 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_178: f32[32, 256, 16] = torch.ops.aten.add.Tensor(_unsafe_view_24, view_104);  _unsafe_view_24 = view_104 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        permute_63: f32[32, 128, 256] = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        clone_114: f32[32, 128, 256] = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        _unsafe_view_25: f32[8, 512, 16, 16] = torch.ops.aten._unsafe_view.default(clone_114, [8, 512, 16, 16]);  clone_114 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        view_105: f32[8, 64, 16, 16] = torch.ops.aten.view.default(view_103, [8, 64, 16, 16]);  view_103 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        permute_64: f32[32, 16, 256] = torch.ops.aten.permute.default(add_178, [0, 2, 1]);  add_178 = None
        clone_115: f32[32, 16, 256] = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        _unsafe_view_26: f32[8, 64, 16, 16] = torch.ops.aten._unsafe_view.default(clone_115, [8, 64, 16, 16]);  clone_115 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        cat_1: f32[8, 640, 16, 16] = torch.ops.aten.cat.default([_unsafe_view_26, view_105, _unsafe_view_25], 1);  _unsafe_view_26 = view_105 = _unsafe_view_25 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(cat_1, mul_202, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_1 = mul_202 = primals_37 = None
        getitem_26: f32[8, 512, 16, 16] = convolution_backward_5[0]
        getitem_27: f32[640, 512, 1, 1] = convolution_backward_5[1];  convolution_backward_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_329: f32[8, 512, 16, 16] = torch.ops.aten.mul.Tensor(getitem_26, mul_328);  getitem_26 = mul_328 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_23: f32[512] = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3])
        sub_66: f32[8, 512, 16, 16] = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_198);  convolution_29 = unsqueeze_198 = None
        mul_330: f32[8, 512, 16, 16] = torch.ops.aten.mul.Tensor(mul_329, sub_66)
        sum_24: f32[512] = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
        mul_331: f32[512] = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
        unsqueeze_199: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
        unsqueeze_200: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_332: f32[512] = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
        mul_333: f32[512] = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_334: f32[512] = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
        unsqueeze_202: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
        unsqueeze_203: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_335: f32[512] = torch.ops.aten.mul.Tensor(squeeze_73, primals_169);  primals_169 = None
        unsqueeze_205: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
        unsqueeze_206: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_336: f32[8, 512, 16, 16] = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_204);  sub_66 = unsqueeze_204 = None
        sub_68: f32[8, 512, 16, 16] = torch.ops.aten.sub.Tensor(mul_329, mul_336);  mul_329 = mul_336 = None
        sub_69: f32[8, 512, 16, 16] = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_201);  sub_68 = unsqueeze_201 = None
        mul_337: f32[8, 512, 16, 16] = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_207);  sub_69 = unsqueeze_207 = None
        mul_338: f32[512] = torch.ops.aten.mul.Tensor(sum_24, squeeze_73);  sum_24 = squeeze_73 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_337, mul_194, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = mul_194 = primals_36 = None
        getitem_29: f32[8, 1024, 16, 16] = convolution_backward_6[0]
        getitem_30: f32[512, 1024, 1, 1] = convolution_backward_6[1];  convolution_backward_6 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_180: f32[8, 1024, 16, 16] = torch.ops.aten.add.Tensor(getitem_20, getitem_29);  getitem_20 = getitem_29 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1251, code: return self.act(x)
        mul_341: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(add_180, mul_340);  add_180 = mul_340 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_25: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3])
        sub_71: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_210);  convolution_28 = unsqueeze_210 = None
        mul_342: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(mul_341, sub_71)
        sum_26: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_342, [0, 2, 3]);  mul_342 = None
        mul_343: f32[1024] = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
        unsqueeze_211: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
        unsqueeze_212: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_344: f32[1024] = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
        mul_345: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_346: f32[1024] = torch.ops.aten.mul.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        unsqueeze_214: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
        unsqueeze_215: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_347: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_70, primals_164);  primals_164 = None
        unsqueeze_217: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
        unsqueeze_218: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_348: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_216);  sub_71 = unsqueeze_216 = None
        sub_73: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(mul_341, mul_348);  mul_348 = None
        sub_74: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_213);  sub_73 = unsqueeze_213 = None
        mul_349: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_219);  sub_74 = unsqueeze_219 = None
        mul_350: f32[1024] = torch.ops.aten.mul.Tensor(sum_26, squeeze_70);  sum_26 = squeeze_70 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_349, mul_186, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_349 = mul_186 = primals_35 = None
        getitem_32: f32[8, 256, 16, 16] = convolution_backward_7[0]
        getitem_33: f32[1024, 256, 1, 1] = convolution_backward_7[1];  convolution_backward_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_353: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(getitem_32, mul_352);  getitem_32 = mul_352 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_27: f32[256] = torch.ops.aten.sum.dim_IntList(mul_353, [0, 2, 3])
        mul_354: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(mul_353, sub_76)
        sum_28: f32[256] = torch.ops.aten.sum.dim_IntList(mul_354, [0, 2, 3]);  mul_354 = None
        mul_355: f32[256] = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
        unsqueeze_223: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
        unsqueeze_224: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_356: f32[256] = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
        mul_357: f32[256] = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_358: f32[256] = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
        unsqueeze_226: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
        unsqueeze_227: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_359: f32[256] = torch.ops.aten.mul.Tensor(squeeze_67, primals_159);  primals_159 = None
        unsqueeze_229: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
        unsqueeze_230: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_360: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_228);  sub_76 = unsqueeze_228 = None
        sub_78: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(mul_353, mul_360);  mul_353 = mul_360 = None
        sub_79: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_225);  sub_78 = unsqueeze_225 = None
        mul_361: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_231);  sub_79 = unsqueeze_231 = None
        mul_362: f32[256] = torch.ops.aten.mul.Tensor(sum_28, squeeze_67);  sum_28 = squeeze_67 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        view_106: f32[32, 64, 256] = torch.ops.aten.view.default(mul_361, [32, 64, 256]);  mul_361 = None
        permute_65: f32[32, 256, 64] = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        view_107: f32[32, 256, 64] = torch.ops.aten.view.default(permute_65, [32, 256, 64]);  permute_65 = None
        bmm_14: f32[32, 256, 64] = torch.ops.aten.bmm.default(permute_66, view_107);  permute_66 = None
        bmm_15: f32[32, 256, 256] = torch.ops.aten.bmm.default(view_107, permute_67);  view_107 = permute_67 = None
        view_108: f32[32, 256, 64] = torch.ops.aten.view.default(bmm_14, [32, 256, 64]);  bmm_14 = None
        view_109: f32[32, 256, 256] = torch.ops.aten.view.default(bmm_15, [32, 256, 256]);  bmm_15 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
        mul_363: f32[32, 256, 256] = torch.ops.aten.mul.Tensor(view_109, div);  view_109 = None
        sum_29: f32[32, 256, 1] = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
        mul_364: f32[32, 256, 256] = torch.ops.aten.mul.Tensor(div, sum_29);  div = sum_29 = None
        sub_80: f32[32, 256, 256] = torch.ops.aten.sub.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
        view_110: f32[32, 16, 16, 16, 16] = torch.ops.aten.view.default(sub_80, [32, 16, 16, 16, 16])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
        permute_68: f32[32, 16, 16, 16, 16] = torch.ops.aten.permute.default(view_110, [0, 2, 4, 1, 3])
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        sum_30: f32[32, 16, 1, 16, 16] = torch.ops.aten.sum.dim_IntList(permute_68, [2], True);  permute_68 = None
        view_111: f32[512, 16, 16] = torch.ops.aten.view.default(sum_30, [512, 16, 16]);  sum_30 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
        slice_scatter_12: f32[512, 16, 31] = torch.ops.aten.slice_scatter.default(full_6, view_111, 2, 15, 9223372036854775807);  view_111 = None
        slice_scatter_13: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_12, 1, 0, 16);  slice_scatter_12 = None
        slice_scatter_14: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_13 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_112: f32[512, 527] = torch.ops.aten.view.default(slice_scatter_14, [512, 527]);  slice_scatter_14 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
        constant_pad_nd_20: f32[512, 512] = torch.ops.aten.constant_pad_nd.default(view_112, [0, -15]);  view_112 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_113: f32[512, 16, 32] = torch.ops.aten.view.default(constant_pad_nd_20, [512, 16, 32]);  constant_pad_nd_20 = None
        constant_pad_nd_21: f32[512, 16, 31] = torch.ops.aten.constant_pad_nd.default(view_113, [0, -1]);  view_113 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
        view_114: f32[32, 16, 16, 31] = torch.ops.aten.view.default(constant_pad_nd_21, [32, 16, 16, 31]);  constant_pad_nd_21 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        view_115: f32[8192, 31] = torch.ops.aten.view.default(view_114, [8192, 31]);  view_114 = None
        permute_69: f32[31, 8192] = torch.ops.aten.permute.default(view_115, [1, 0])
        mm_16: f32[31, 16] = torch.ops.aten.mm.default(permute_69, _unsafe_view_4);  permute_69 = _unsafe_view_4 = None
        permute_70: f32[16, 31] = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
        mm_17: f32[8192, 16] = torch.ops.aten.mm.default(view_115, permute_71);  view_115 = permute_71 = None
        view_116: f32[32, 16, 16, 16] = torch.ops.aten.view.default(mm_17, [32, 16, 16, 16]);  mm_17 = None
        permute_72: f32[31, 16] = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
        permute_73: f32[32, 16, 16, 16] = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
        permute_74: f32[32, 16, 16, 16, 16] = torch.ops.aten.permute.default(view_110, [0, 1, 3, 2, 4]);  view_110 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        sum_31: f32[32, 16, 1, 16, 16] = torch.ops.aten.sum.dim_IntList(permute_74, [2], True);  permute_74 = None
        view_117: f32[512, 16, 16] = torch.ops.aten.view.default(sum_31, [512, 16, 16]);  sum_31 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
        slice_scatter_15: f32[512, 16, 31] = torch.ops.aten.slice_scatter.default(full_6, view_117, 2, 15, 9223372036854775807);  full_6 = view_117 = None
        slice_scatter_16: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_15, 1, 0, 16);  slice_scatter_15 = None
        slice_scatter_17: f32[512, 17, 31] = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_16, 0, 0, 9223372036854775807);  full_7 = slice_scatter_16 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_118: f32[512, 527] = torch.ops.aten.view.default(slice_scatter_17, [512, 527]);  slice_scatter_17 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
        constant_pad_nd_22: f32[512, 512] = torch.ops.aten.constant_pad_nd.default(view_118, [0, -15]);  view_118 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_119: f32[512, 16, 32] = torch.ops.aten.view.default(constant_pad_nd_22, [512, 16, 32]);  constant_pad_nd_22 = None
        constant_pad_nd_23: f32[512, 16, 31] = torch.ops.aten.constant_pad_nd.default(view_119, [0, -1]);  view_119 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
        view_120: f32[32, 16, 16, 31] = torch.ops.aten.view.default(constant_pad_nd_23, [32, 16, 16, 31]);  constant_pad_nd_23 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        view_121: f32[8192, 31] = torch.ops.aten.view.default(view_120, [8192, 31]);  view_120 = None
        permute_75: f32[31, 8192] = torch.ops.aten.permute.default(view_121, [1, 0])
        mm_18: f32[31, 16] = torch.ops.aten.mm.default(permute_75, _unsafe_view_3);  permute_75 = _unsafe_view_3 = None
        permute_76: f32[16, 31] = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
        mm_19: f32[8192, 16] = torch.ops.aten.mm.default(view_121, permute_77);  view_121 = permute_77 = None
        view_122: f32[32, 16, 16, 16] = torch.ops.aten.view.default(mm_19, [32, 16, 16, 16]);  mm_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        add_183: f32[32, 16, 16, 16] = torch.ops.aten.add.Tensor(permute_73, view_122);  permute_73 = view_122 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
        permute_78: f32[31, 16] = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
        clone_116: f32[32, 16, 16, 16] = torch.ops.aten.clone.default(add_183, memory_format = torch.contiguous_format);  add_183 = None
        _unsafe_view_27: f32[32, 256, 16] = torch.ops.aten._unsafe_view.default(clone_116, [32, 256, 16]);  clone_116 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        mul_365: f32[32, 256, 256] = torch.ops.aten.mul.Tensor(sub_80, 0.25);  sub_80 = None
        view_123: f32[32, 256, 256] = torch.ops.aten.view.default(mul_365, [32, 256, 256]);  mul_365 = None
        bmm_16: f32[32, 16, 256] = torch.ops.aten.bmm.default(permute_79, view_123);  permute_79 = None
        bmm_17: f32[32, 256, 16] = torch.ops.aten.bmm.default(view_123, permute_80);  view_123 = permute_80 = None
        view_124: f32[32, 16, 256] = torch.ops.aten.view.default(bmm_16, [32, 16, 256]);  bmm_16 = None
        view_125: f32[32, 256, 16] = torch.ops.aten.view.default(bmm_17, [32, 256, 16]);  bmm_17 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_184: f32[32, 256, 16] = torch.ops.aten.add.Tensor(_unsafe_view_27, view_125);  _unsafe_view_27 = view_125 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        permute_81: f32[32, 64, 256] = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        clone_117: f32[32, 64, 256] = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        _unsafe_view_28: f32[8, 256, 16, 16] = torch.ops.aten._unsafe_view.default(clone_117, [8, 256, 16, 16]);  clone_117 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        view_126: f32[8, 64, 16, 16] = torch.ops.aten.view.default(view_124, [8, 64, 16, 16]);  view_124 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        permute_82: f32[32, 16, 256] = torch.ops.aten.permute.default(add_184, [0, 2, 1]);  add_184 = None
        clone_118: f32[32, 16, 256] = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        _unsafe_view_29: f32[8, 64, 16, 16] = torch.ops.aten._unsafe_view.default(clone_118, [8, 64, 16, 16]);  clone_118 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        cat_2: f32[8, 384, 16, 16] = torch.ops.aten.cat.default([_unsafe_view_29, view_126, _unsafe_view_28], 1);  _unsafe_view_29 = view_126 = _unsafe_view_28 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(cat_2, mul_177, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_2 = mul_177 = primals_34 = None
        getitem_35: f32[8, 256, 16, 16] = convolution_backward_8[0]
        getitem_36: f32[384, 256, 1, 1] = convolution_backward_8[1];  convolution_backward_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_368: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(getitem_35, mul_367);  getitem_35 = mul_367 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_32: f32[256] = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3])
        sub_82: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_234);  convolution_26 = unsqueeze_234 = None
        mul_369: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(mul_368, sub_82)
        sum_33: f32[256] = torch.ops.aten.sum.dim_IntList(mul_369, [0, 2, 3]);  mul_369 = None
        mul_370: f32[256] = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
        unsqueeze_235: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
        unsqueeze_236: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_371: f32[256] = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
        mul_372: f32[256] = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_373: f32[256] = torch.ops.aten.mul.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        unsqueeze_238: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_239: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_374: f32[256] = torch.ops.aten.mul.Tensor(squeeze_64, primals_154);  primals_154 = None
        unsqueeze_241: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
        unsqueeze_242: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_375: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_240);  sub_82 = unsqueeze_240 = None
        sub_84: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(mul_368, mul_375);  mul_368 = mul_375 = None
        sub_85: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_237);  sub_84 = unsqueeze_237 = None
        mul_376: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_243);  sub_85 = unsqueeze_243 = None
        mul_377: f32[256] = torch.ops.aten.mul.Tensor(sum_33, squeeze_64);  sum_33 = squeeze_64 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_376, mul_169, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_376 = mul_169 = primals_33 = None
        getitem_38: f32[8, 1024, 16, 16] = convolution_backward_9[0]
        getitem_39: f32[256, 1024, 1, 1] = convolution_backward_9[1];  convolution_backward_9 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_186: f32[8, 1024, 16, 16] = torch.ops.aten.add.Tensor(mul_341, getitem_38);  mul_341 = getitem_38 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1051, code: return self.act(x)
        mul_380: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(add_186, mul_379);  add_186 = mul_379 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_34: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3])
        sub_87: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_246);  convolution_25 = unsqueeze_246 = None
        mul_381: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(mul_380, sub_87)
        sum_35: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
        mul_382: f32[1024] = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
        unsqueeze_247: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
        unsqueeze_248: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_383: f32[1024] = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
        mul_384: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_385: f32[1024] = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
        unsqueeze_250: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
        unsqueeze_251: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_386: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_61, primals_149);  primals_149 = None
        unsqueeze_253: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
        unsqueeze_254: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_387: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_252);  sub_87 = unsqueeze_252 = None
        sub_89: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(mul_380, mul_387);  mul_387 = None
        sub_90: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_249);  sub_89 = None
        mul_388: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_255);  sub_90 = unsqueeze_255 = None
        mul_389: f32[1024] = torch.ops.aten.mul.Tensor(sum_35, squeeze_61);  sum_35 = squeeze_61 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_388, mul_137, primals_32, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_388 = primals_32 = None
        getitem_41: f32[8, 512, 32, 32] = convolution_backward_10[0]
        getitem_42: f32[1024, 512, 1, 1] = convolution_backward_10[1];  convolution_backward_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sub_91: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_258);  convolution_24 = unsqueeze_258 = None
        mul_390: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(mul_380, sub_91)
        sum_37: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_390, [0, 2, 3]);  mul_390 = None
        mul_392: f32[1024] = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
        mul_393: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_394: f32[1024] = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        unsqueeze_262: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_263: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_395: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_58, primals_144);  primals_144 = None
        unsqueeze_265: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
        unsqueeze_266: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_396: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_264);  sub_91 = unsqueeze_264 = None
        sub_93: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(mul_380, mul_396);  mul_380 = mul_396 = None
        sub_94: f32[8, 1024, 16, 16] = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_249);  sub_93 = unsqueeze_249 = None
        mul_397: f32[8, 1024, 16, 16] = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_267);  sub_94 = unsqueeze_267 = None
        mul_398: f32[1024] = torch.ops.aten.mul.Tensor(sum_37, squeeze_58);  sum_37 = squeeze_58 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_397, mul_154, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_397 = mul_154 = primals_31 = None
        getitem_44: f32[8, 256, 16, 16] = convolution_backward_11[0]
        getitem_45: f32[1024, 256, 1, 1] = convolution_backward_11[1];  convolution_backward_11 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        mul_399: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(getitem_44, mul_153);  mul_153 = None
        mul_400: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(getitem_44, expand_4);  getitem_44 = expand_4 = None
        sum_38: f32[8, 256, 1, 1] = torch.ops.aten.sum.dim_IntList(mul_399, [2, 3], True);  mul_399 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_127: f32[8, 1, 256] = torch.ops.aten.view.default(sum_38, [8, 1, 256]);  sum_38 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_95: f32[8, 1, 256] = torch.ops.aten.sub.Tensor(1, sigmoid_21)
        mul_401: f32[8, 1, 256] = torch.ops.aten.mul.Tensor(sigmoid_21, sub_95);  sigmoid_21 = sub_95 = None
        mul_402: f32[8, 1, 256] = torch.ops.aten.mul.Tensor(view_127, mul_401);  view_127 = mul_401 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:86, code: y = self.conv(y)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_402, view_8, primals_30, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_402 = view_8 = primals_30 = None
        getitem_47: f32[8, 1, 256] = convolution_backward_12[0]
        getitem_48: f32[1, 1, 5] = convolution_backward_12[1];  convolution_backward_12 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        view_128: f32[8, 256] = torch.ops.aten.view.default(getitem_47, [8, 256]);  getitem_47 = None
        unsqueeze_268: f32[8, 256, 1] = torch.ops.aten.unsqueeze.default(view_128, 2);  view_128 = None
        unsqueeze_269: f32[8, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
        expand_24: f32[8, 256, 16, 16] = torch.ops.aten.expand.default(unsqueeze_269, [8, 256, 16, 16]);  unsqueeze_269 = None
        div_4: f32[8, 256, 16, 16] = torch.ops.aten.div.Scalar(expand_24, 256);  expand_24 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        add_188: f32[8, 256, 16, 16] = torch.ops.aten.add.Tensor(mul_400, div_4);  mul_400 = div_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        sigmoid_42: f32[8, 256, 16, 16] = torch.ops.aten.sigmoid.default(clone_78)
        empty_like_10: f32[8, 256, 16, 16] = torch.ops.aten.empty_like.default(sigmoid_42, memory_format = torch.preserve_format)
        full_like_10: f32[8, 256, 16, 16] = torch.ops.aten.full_like.default(empty_like_10, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_10 = None
        sub_96: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(full_like_10, sigmoid_42);  full_like_10 = None
        mul_403: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(clone_78, sub_96);  clone_78 = sub_96 = None
        add_189: f32[8, 256, 16, 16] = torch.ops.aten.add.Scalar(mul_403, 1);  mul_403 = None
        mul_404: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(sigmoid_42, add_189);  sigmoid_42 = add_189 = None
        mul_405: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(add_188, mul_404);  add_188 = mul_404 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_39: f32[256] = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3])
        sub_97: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_272);  convolution_22 = unsqueeze_272 = None
        mul_406: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(mul_405, sub_97)
        sum_40: f32[256] = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3]);  mul_406 = None
        mul_407: f32[256] = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
        unsqueeze_273: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
        unsqueeze_274: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_275: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
        mul_408: f32[256] = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
        mul_409: f32[256] = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_410: f32[256] = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
        unsqueeze_276: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_277: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
        unsqueeze_278: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
        mul_411: f32[256] = torch.ops.aten.mul.Tensor(squeeze_55, primals_139);  primals_139 = None
        unsqueeze_279: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_280: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        unsqueeze_281: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
        mul_412: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_278);  sub_97 = unsqueeze_278 = None
        sub_99: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(mul_405, mul_412);  mul_405 = mul_412 = None
        sub_100: f32[8, 256, 16, 16] = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_275);  sub_99 = unsqueeze_275 = None
        mul_413: f32[8, 256, 16, 16] = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_281);  sub_100 = unsqueeze_281 = None
        mul_414: f32[256] = torch.ops.aten.mul.Tensor(sum_40, squeeze_55);  sum_40 = squeeze_55 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_413, mul_145, primals_29, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_413 = mul_145 = primals_29 = None
        getitem_50: f32[8, 256, 32, 32] = convolution_backward_13[0]
        getitem_51: f32[256, 16, 3, 3] = convolution_backward_13[1];  convolution_backward_13 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_417: f32[8, 256, 32, 32] = torch.ops.aten.mul.Tensor(getitem_50, mul_416);  getitem_50 = mul_416 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_41: f32[256] = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3])
        sub_102: f32[8, 256, 32, 32] = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_284);  convolution_21 = unsqueeze_284 = None
        mul_418: f32[8, 256, 32, 32] = torch.ops.aten.mul.Tensor(mul_417, sub_102)
        sum_42: f32[256] = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
        mul_419: f32[256] = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
        unsqueeze_285: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
        unsqueeze_286: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
        unsqueeze_287: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        mul_420: f32[256] = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
        mul_421: f32[256] = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_422: f32[256] = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
        unsqueeze_288: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
        unsqueeze_289: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
        unsqueeze_290: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
        mul_423: f32[256] = torch.ops.aten.mul.Tensor(squeeze_52, primals_134);  primals_134 = None
        unsqueeze_291: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_292: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        unsqueeze_293: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
        mul_424: f32[8, 256, 32, 32] = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_290);  sub_102 = unsqueeze_290 = None
        sub_104: f32[8, 256, 32, 32] = torch.ops.aten.sub.Tensor(mul_417, mul_424);  mul_417 = mul_424 = None
        sub_105: f32[8, 256, 32, 32] = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_287);  sub_104 = unsqueeze_287 = None
        mul_425: f32[8, 256, 32, 32] = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_293);  sub_105 = unsqueeze_293 = None
        mul_426: f32[256] = torch.ops.aten.mul.Tensor(sum_42, squeeze_52);  sum_42 = squeeze_52 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_425, mul_137, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = mul_137 = primals_28 = None
        getitem_53: f32[8, 512, 32, 32] = convolution_backward_14[0]
        getitem_54: f32[256, 512, 1, 1] = convolution_backward_14[1];  convolution_backward_14 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_191: f32[8, 512, 32, 32] = torch.ops.aten.add.Tensor(getitem_41, getitem_53);  getitem_41 = getitem_53 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1051, code: return self.act(x)
        mul_429: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(add_191, mul_428);  add_191 = mul_428 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_43: f32[512] = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3])
        sub_107: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_296);  convolution_20 = unsqueeze_296 = None
        mul_430: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(mul_429, sub_107)
        sum_44: f32[512] = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
        mul_431: f32[512] = torch.ops.aten.mul.Tensor(sum_43, 0.0001220703125)
        unsqueeze_297: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_298: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_299: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
        mul_432: f32[512] = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
        mul_433: f32[512] = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_434: f32[512] = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
        unsqueeze_300: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
        unsqueeze_301: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
        unsqueeze_302: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
        mul_435: f32[512] = torch.ops.aten.mul.Tensor(squeeze_49, primals_129);  primals_129 = None
        unsqueeze_303: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
        unsqueeze_304: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
        mul_436: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_302);  sub_107 = unsqueeze_302 = None
        sub_109: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(mul_429, mul_436);  mul_436 = None
        sub_110: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_299);  sub_109 = unsqueeze_299 = None
        mul_437: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_305);  sub_110 = unsqueeze_305 = None
        mul_438: f32[512] = torch.ops.aten.mul.Tensor(sum_44, squeeze_49);  sum_44 = squeeze_49 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_437, mul_129, primals_27, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = mul_129 = primals_27 = None
        getitem_56: f32[8, 128, 32, 32] = convolution_backward_15[0]
        getitem_57: f32[512, 128, 1, 1] = convolution_backward_15[1];  convolution_backward_15 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        mul_439: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(getitem_56, mul_128);  mul_128 = None
        mul_440: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(getitem_56, expand_3);  getitem_56 = expand_3 = None
        sum_45: f32[8, 128, 1, 1] = torch.ops.aten.sum.dim_IntList(mul_439, [2, 3], True);  mul_439 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_129: f32[8, 1, 128] = torch.ops.aten.view.default(sum_45, [8, 1, 128]);  sum_45 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_111: f32[8, 1, 128] = torch.ops.aten.sub.Tensor(1, sigmoid_17)
        mul_441: f32[8, 1, 128] = torch.ops.aten.mul.Tensor(sigmoid_17, sub_111);  sigmoid_17 = sub_111 = None
        mul_442: f32[8, 1, 128] = torch.ops.aten.mul.Tensor(view_129, mul_441);  view_129 = mul_441 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:86, code: y = self.conv(y)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_442, view_6, primals_26, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_442 = view_6 = primals_26 = None
        getitem_59: f32[8, 1, 128] = convolution_backward_16[0]
        getitem_60: f32[1, 1, 5] = convolution_backward_16[1];  convolution_backward_16 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        view_130: f32[8, 128] = torch.ops.aten.view.default(getitem_59, [8, 128]);  getitem_59 = None
        unsqueeze_306: f32[8, 128, 1] = torch.ops.aten.unsqueeze.default(view_130, 2);  view_130 = None
        unsqueeze_307: f32[8, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        expand_25: f32[8, 128, 32, 32] = torch.ops.aten.expand.default(unsqueeze_307, [8, 128, 32, 32]);  unsqueeze_307 = None
        div_5: f32[8, 128, 32, 32] = torch.ops.aten.div.Scalar(expand_25, 1024);  expand_25 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        add_193: f32[8, 128, 32, 32] = torch.ops.aten.add.Tensor(mul_440, div_5);  mul_440 = div_5 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        sigmoid_45: f32[8, 128, 32, 32] = torch.ops.aten.sigmoid.default(clone_75)
        empty_like_13: f32[8, 128, 32, 32] = torch.ops.aten.empty_like.default(sigmoid_45, memory_format = torch.preserve_format)
        full_like_13: f32[8, 128, 32, 32] = torch.ops.aten.full_like.default(empty_like_13, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_13 = None
        sub_112: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(full_like_13, sigmoid_45);  full_like_13 = None
        mul_443: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(clone_75, sub_112);  clone_75 = sub_112 = None
        add_194: f32[8, 128, 32, 32] = torch.ops.aten.add.Scalar(mul_443, 1);  mul_443 = None
        mul_444: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sigmoid_45, add_194);  sigmoid_45 = add_194 = None
        mul_445: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(add_193, mul_444);  add_193 = mul_444 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_46: f32[128] = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3])
        sub_113: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_310);  convolution_18 = unsqueeze_310 = None
        mul_446: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(mul_445, sub_113)
        sum_47: f32[128] = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3]);  mul_446 = None
        mul_447: f32[128] = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
        unsqueeze_311: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
        unsqueeze_312: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_448: f32[128] = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
        mul_449: f32[128] = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_450: f32[128] = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
        unsqueeze_314: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_315: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_451: f32[128] = torch.ops.aten.mul.Tensor(squeeze_46, primals_124);  primals_124 = None
        unsqueeze_317: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
        unsqueeze_318: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_452: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_316);  sub_113 = unsqueeze_316 = None
        sub_115: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(mul_445, mul_452);  mul_445 = mul_452 = None
        sub_116: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_313);  sub_115 = unsqueeze_313 = None
        mul_453: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_319);  sub_116 = unsqueeze_319 = None
        mul_454: f32[128] = torch.ops.aten.mul.Tensor(sum_47, squeeze_46);  sum_47 = squeeze_46 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_453, mul_120, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_453 = mul_120 = primals_25 = None
        getitem_62: f32[8, 128, 32, 32] = convolution_backward_17[0]
        getitem_63: f32[128, 16, 3, 3] = convolution_backward_17[1];  convolution_backward_17 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_457: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(getitem_62, mul_456);  getitem_62 = mul_456 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_48: f32[128] = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3])
        sub_118: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_322);  convolution_17 = unsqueeze_322 = None
        mul_458: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(mul_457, sub_118)
        sum_49: f32[128] = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
        mul_459: f32[128] = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
        unsqueeze_323: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_324: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_460: f32[128] = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
        mul_461: f32[128] = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_462: f32[128] = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
        unsqueeze_326: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
        unsqueeze_327: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_463: f32[128] = torch.ops.aten.mul.Tensor(squeeze_43, primals_119);  primals_119 = None
        unsqueeze_329: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
        unsqueeze_330: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_464: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_328);  sub_118 = unsqueeze_328 = None
        sub_120: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(mul_457, mul_464);  mul_457 = mul_464 = None
        sub_121: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_325);  sub_120 = unsqueeze_325 = None
        mul_465: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_331);  sub_121 = unsqueeze_331 = None
        mul_466: f32[128] = torch.ops.aten.mul.Tensor(sum_49, squeeze_43);  sum_49 = squeeze_43 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_465, mul_112, primals_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = mul_112 = primals_24 = None
        getitem_65: f32[8, 512, 32, 32] = convolution_backward_18[0]
        getitem_66: f32[128, 512, 1, 1] = convolution_backward_18[1];  convolution_backward_18 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_196: f32[8, 512, 32, 32] = torch.ops.aten.add.Tensor(mul_429, getitem_65);  mul_429 = getitem_65 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1051, code: return self.act(x)
        mul_469: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(add_196, mul_468);  add_196 = mul_468 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_50: f32[512] = torch.ops.aten.sum.dim_IntList(mul_469, [0, 2, 3])
        sub_123: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_334);  convolution_16 = unsqueeze_334 = None
        mul_470: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(mul_469, sub_123)
        sum_51: f32[512] = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
        mul_471: f32[512] = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
        unsqueeze_335: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
        unsqueeze_336: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_472: f32[512] = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
        mul_473: f32[512] = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_474: f32[512] = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
        unsqueeze_338: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_339: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_475: f32[512] = torch.ops.aten.mul.Tensor(squeeze_40, primals_114);  primals_114 = None
        unsqueeze_341: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_342: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_476: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_340);  sub_123 = unsqueeze_340 = None
        sub_125: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(mul_469, mul_476);  mul_476 = None
        sub_126: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_337);  sub_125 = None
        mul_477: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_343);  sub_126 = unsqueeze_343 = None
        mul_478: f32[512] = torch.ops.aten.mul.Tensor(sum_51, squeeze_40);  sum_51 = squeeze_40 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_477, mul_80, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = primals_23 = None
        getitem_68: f32[8, 256, 64, 64] = convolution_backward_19[0]
        getitem_69: f32[512, 256, 1, 1] = convolution_backward_19[1];  convolution_backward_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sub_127: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_346);  convolution_15 = unsqueeze_346 = None
        mul_479: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(mul_469, sub_127)
        sum_53: f32[512] = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
        mul_481: f32[512] = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
        mul_482: f32[512] = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_483: f32[512] = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
        unsqueeze_350: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_351: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_484: f32[512] = torch.ops.aten.mul.Tensor(squeeze_37, primals_109);  primals_109 = None
        unsqueeze_353: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_354: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_485: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_352);  sub_127 = unsqueeze_352 = None
        sub_129: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(mul_469, mul_485);  mul_469 = mul_485 = None
        sub_130: f32[8, 512, 32, 32] = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_337);  sub_129 = unsqueeze_337 = None
        mul_486: f32[8, 512, 32, 32] = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_355);  sub_130 = unsqueeze_355 = None
        mul_487: f32[512] = torch.ops.aten.mul.Tensor(sum_53, squeeze_37);  sum_53 = squeeze_37 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_486, mul_97, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_97 = primals_22 = None
        getitem_71: f32[8, 128, 32, 32] = convolution_backward_20[0]
        getitem_72: f32[512, 128, 1, 1] = convolution_backward_20[1];  convolution_backward_20 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        mul_488: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(getitem_71, mul_96);  mul_96 = None
        mul_489: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(getitem_71, expand_2);  getitem_71 = expand_2 = None
        sum_54: f32[8, 128, 1, 1] = torch.ops.aten.sum.dim_IntList(mul_488, [2, 3], True);  mul_488 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_131: f32[8, 1, 128] = torch.ops.aten.view.default(sum_54, [8, 1, 128]);  sum_54 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_131: f32[8, 1, 128] = torch.ops.aten.sub.Tensor(1, sigmoid_13)
        mul_490: f32[8, 1, 128] = torch.ops.aten.mul.Tensor(sigmoid_13, sub_131);  sigmoid_13 = sub_131 = None
        mul_491: f32[8, 1, 128] = torch.ops.aten.mul.Tensor(view_131, mul_490);  view_131 = mul_490 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:86, code: y = self.conv(y)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_491, view_4, primals_21, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_491 = view_4 = primals_21 = None
        getitem_74: f32[8, 1, 128] = convolution_backward_21[0]
        getitem_75: f32[1, 1, 5] = convolution_backward_21[1];  convolution_backward_21 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        view_132: f32[8, 128] = torch.ops.aten.view.default(getitem_74, [8, 128]);  getitem_74 = None
        unsqueeze_356: f32[8, 128, 1] = torch.ops.aten.unsqueeze.default(view_132, 2);  view_132 = None
        unsqueeze_357: f32[8, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        expand_26: f32[8, 128, 32, 32] = torch.ops.aten.expand.default(unsqueeze_357, [8, 128, 32, 32]);  unsqueeze_357 = None
        div_6: f32[8, 128, 32, 32] = torch.ops.aten.div.Scalar(expand_26, 1024);  expand_26 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        add_198: f32[8, 128, 32, 32] = torch.ops.aten.add.Tensor(mul_489, div_6);  mul_489 = div_6 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        sigmoid_48: f32[8, 128, 32, 32] = torch.ops.aten.sigmoid.default(clone_72)
        empty_like_16: f32[8, 128, 32, 32] = torch.ops.aten.empty_like.default(sigmoid_48, memory_format = torch.preserve_format)
        full_like_16: f32[8, 128, 32, 32] = torch.ops.aten.full_like.default(empty_like_16, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_16 = None
        sub_132: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(full_like_16, sigmoid_48);  full_like_16 = None
        mul_492: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(clone_72, sub_132);  clone_72 = sub_132 = None
        add_199: f32[8, 128, 32, 32] = torch.ops.aten.add.Scalar(mul_492, 1);  mul_492 = None
        mul_493: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sigmoid_48, add_199);  sigmoid_48 = add_199 = None
        mul_494: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(add_198, mul_493);  add_198 = mul_493 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_55: f32[128] = torch.ops.aten.sum.dim_IntList(mul_494, [0, 2, 3])
        sub_133: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_360);  convolution_13 = unsqueeze_360 = None
        mul_495: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(mul_494, sub_133)
        sum_56: f32[128] = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
        mul_496: f32[128] = torch.ops.aten.mul.Tensor(sum_55, 0.0001220703125)
        unsqueeze_361: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_362: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_497: f32[128] = torch.ops.aten.mul.Tensor(sum_56, 0.0001220703125)
        mul_498: f32[128] = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_499: f32[128] = torch.ops.aten.mul.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
        unsqueeze_364: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
        unsqueeze_365: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        mul_500: f32[128] = torch.ops.aten.mul.Tensor(squeeze_34, primals_104);  primals_104 = None
        unsqueeze_367: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
        unsqueeze_368: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_501: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_366);  sub_133 = unsqueeze_366 = None
        sub_135: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(mul_494, mul_501);  mul_494 = mul_501 = None
        sub_136: f32[8, 128, 32, 32] = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_363);  sub_135 = unsqueeze_363 = None
        mul_502: f32[8, 128, 32, 32] = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_369);  sub_136 = unsqueeze_369 = None
        mul_503: f32[128] = torch.ops.aten.mul.Tensor(sum_56, squeeze_34);  sum_56 = squeeze_34 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_502, mul_88, primals_20, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_502 = mul_88 = primals_20 = None
        getitem_77: f32[8, 128, 64, 64] = convolution_backward_22[0]
        getitem_78: f32[128, 16, 3, 3] = convolution_backward_22[1];  convolution_backward_22 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_506: f32[8, 128, 64, 64] = torch.ops.aten.mul.Tensor(getitem_77, mul_505);  getitem_77 = mul_505 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_57: f32[128] = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3])
        sub_138: f32[8, 128, 64, 64] = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_372);  convolution_12 = unsqueeze_372 = None
        mul_507: f32[8, 128, 64, 64] = torch.ops.aten.mul.Tensor(mul_506, sub_138)
        sum_58: f32[128] = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
        mul_508: f32[128] = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
        unsqueeze_373: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
        unsqueeze_374: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_509: f32[128] = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
        mul_510: f32[128] = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_511: f32[128] = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
        unsqueeze_376: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_377: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        mul_512: f32[128] = torch.ops.aten.mul.Tensor(squeeze_31, primals_99);  primals_99 = None
        unsqueeze_379: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
        unsqueeze_380: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_513: f32[8, 128, 64, 64] = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_378);  sub_138 = unsqueeze_378 = None
        sub_140: f32[8, 128, 64, 64] = torch.ops.aten.sub.Tensor(mul_506, mul_513);  mul_506 = mul_513 = None
        sub_141: f32[8, 128, 64, 64] = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_375);  sub_140 = unsqueeze_375 = None
        mul_514: f32[8, 128, 64, 64] = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_381);  sub_141 = unsqueeze_381 = None
        mul_515: f32[128] = torch.ops.aten.mul.Tensor(sum_58, squeeze_31);  sum_58 = squeeze_31 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_514, mul_80, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = mul_80 = primals_19 = None
        getitem_80: f32[8, 256, 64, 64] = convolution_backward_23[0]
        getitem_81: f32[128, 256, 1, 1] = convolution_backward_23[1];  convolution_backward_23 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_201: f32[8, 256, 64, 64] = torch.ops.aten.add.Tensor(getitem_68, getitem_80);  getitem_68 = getitem_80 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1051, code: return self.act(x)
        mul_518: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(add_201, mul_517);  add_201 = mul_517 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_59: f32[256] = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3])
        sub_143: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_384);  convolution_11 = unsqueeze_384 = None
        mul_519: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(mul_518, sub_143)
        sum_60: f32[256] = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
        mul_520: f32[256] = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
        unsqueeze_385: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_386: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_521: f32[256] = torch.ops.aten.mul.Tensor(sum_60, 3.0517578125e-05)
        mul_522: f32[256] = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_523: f32[256] = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
        unsqueeze_388: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_389: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        mul_524: f32[256] = torch.ops.aten.mul.Tensor(squeeze_28, primals_94);  primals_94 = None
        unsqueeze_391: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
        unsqueeze_392: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_525: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_390);  sub_143 = unsqueeze_390 = None
        sub_145: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(mul_518, mul_525);  mul_525 = None
        sub_146: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_387);  sub_145 = unsqueeze_387 = None
        mul_526: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_393);  sub_146 = unsqueeze_393 = None
        mul_527: f32[256] = torch.ops.aten.mul.Tensor(sum_60, squeeze_28);  sum_60 = squeeze_28 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_526, mul_72, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_526 = mul_72 = primals_18 = None
        getitem_83: f32[8, 64, 64, 64] = convolution_backward_24[0]
        getitem_84: f32[256, 64, 1, 1] = convolution_backward_24[1];  convolution_backward_24 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        mul_528: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(getitem_83, mul_71);  mul_71 = None
        mul_529: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(getitem_83, expand_1);  getitem_83 = expand_1 = None
        sum_61: f32[8, 64, 1, 1] = torch.ops.aten.sum.dim_IntList(mul_528, [2, 3], True);  mul_528 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_133: f32[8, 1, 64] = torch.ops.aten.view.default(sum_61, [8, 1, 64]);  sum_61 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_147: f32[8, 1, 64] = torch.ops.aten.sub.Tensor(1, sigmoid_9)
        mul_530: f32[8, 1, 64] = torch.ops.aten.mul.Tensor(sigmoid_9, sub_147);  sigmoid_9 = sub_147 = None
        mul_531: f32[8, 1, 64] = torch.ops.aten.mul.Tensor(view_133, mul_530);  view_133 = mul_530 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:86, code: y = self.conv(y)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_531, view_2, primals_17, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_531 = view_2 = primals_17 = None
        getitem_86: f32[8, 1, 64] = convolution_backward_25[0]
        getitem_87: f32[1, 1, 3] = convolution_backward_25[1];  convolution_backward_25 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        view_134: f32[8, 64] = torch.ops.aten.view.default(getitem_86, [8, 64]);  getitem_86 = None
        unsqueeze_394: f32[8, 64, 1] = torch.ops.aten.unsqueeze.default(view_134, 2);  view_134 = None
        unsqueeze_395: f32[8, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
        expand_27: f32[8, 64, 64, 64] = torch.ops.aten.expand.default(unsqueeze_395, [8, 64, 64, 64]);  unsqueeze_395 = None
        div_7: f32[8, 64, 64, 64] = torch.ops.aten.div.Scalar(expand_27, 4096);  expand_27 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        add_203: f32[8, 64, 64, 64] = torch.ops.aten.add.Tensor(mul_529, div_7);  mul_529 = div_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        sigmoid_51: f32[8, 64, 64, 64] = torch.ops.aten.sigmoid.default(clone_69)
        empty_like_19: f32[8, 64, 64, 64] = torch.ops.aten.empty_like.default(sigmoid_51, memory_format = torch.preserve_format)
        full_like_19: f32[8, 64, 64, 64] = torch.ops.aten.full_like.default(empty_like_19, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_19 = None
        sub_148: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(full_like_19, sigmoid_51);  full_like_19 = None
        mul_532: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(clone_69, sub_148);  clone_69 = sub_148 = None
        add_204: f32[8, 64, 64, 64] = torch.ops.aten.add.Scalar(mul_532, 1);  mul_532 = None
        mul_533: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sigmoid_51, add_204);  sigmoid_51 = add_204 = None
        mul_534: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(add_203, mul_533);  add_203 = mul_533 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_62: f32[64] = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3])
        sub_149: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_398);  convolution_9 = unsqueeze_398 = None
        mul_535: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(mul_534, sub_149)
        sum_63: f32[64] = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
        mul_536: f32[64] = torch.ops.aten.mul.Tensor(sum_62, 3.0517578125e-05)
        unsqueeze_399: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
        unsqueeze_400: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        unsqueeze_401: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
        mul_537: f32[64] = torch.ops.aten.mul.Tensor(sum_63, 3.0517578125e-05)
        mul_538: f32[64] = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_539: f32[64] = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
        unsqueeze_402: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_403: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
        unsqueeze_404: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
        mul_540: f32[64] = torch.ops.aten.mul.Tensor(squeeze_25, primals_89);  primals_89 = None
        unsqueeze_405: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_406: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        mul_541: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_404);  sub_149 = unsqueeze_404 = None
        sub_151: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(mul_534, mul_541);  mul_534 = mul_541 = None
        sub_152: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_401);  sub_151 = unsqueeze_401 = None
        mul_542: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_407);  sub_152 = unsqueeze_407 = None
        mul_543: f32[64] = torch.ops.aten.mul.Tensor(sum_63, squeeze_25);  sum_63 = squeeze_25 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_542, mul_63, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_542 = mul_63 = primals_16 = None
        getitem_89: f32[8, 64, 64, 64] = convolution_backward_26[0]
        getitem_90: f32[64, 16, 3, 3] = convolution_backward_26[1];  convolution_backward_26 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_546: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(getitem_89, mul_545);  getitem_89 = mul_545 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_64: f32[64] = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2, 3])
        sub_154: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_410);  convolution_8 = unsqueeze_410 = None
        mul_547: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(mul_546, sub_154)
        sum_65: f32[64] = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
        mul_548: f32[64] = torch.ops.aten.mul.Tensor(sum_64, 3.0517578125e-05)
        unsqueeze_411: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_412: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        unsqueeze_413: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
        mul_549: f32[64] = torch.ops.aten.mul.Tensor(sum_65, 3.0517578125e-05)
        mul_550: f32[64] = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_551: f32[64] = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
        unsqueeze_414: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
        unsqueeze_415: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
        unsqueeze_416: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
        mul_552: f32[64] = torch.ops.aten.mul.Tensor(squeeze_22, primals_84);  primals_84 = None
        unsqueeze_417: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
        unsqueeze_418: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_419: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
        mul_553: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_416);  sub_154 = unsqueeze_416 = None
        sub_156: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(mul_546, mul_553);  mul_546 = mul_553 = None
        sub_157: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_413);  sub_156 = unsqueeze_413 = None
        mul_554: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_419);  sub_157 = unsqueeze_419 = None
        mul_555: f32[64] = torch.ops.aten.mul.Tensor(sum_65, squeeze_22);  sum_65 = squeeze_22 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_554, mul_55, primals_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = mul_55 = primals_15 = None
        getitem_92: f32[8, 256, 64, 64] = convolution_backward_27[0]
        getitem_93: f32[64, 256, 1, 1] = convolution_backward_27[1];  convolution_backward_27 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_206: f32[8, 256, 64, 64] = torch.ops.aten.add.Tensor(mul_518, getitem_92);  mul_518 = getitem_92 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1051, code: return self.act(x)
        mul_558: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(add_206, mul_557);  add_206 = mul_557 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_66: f32[256] = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3])
        sub_159: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_422);  convolution_7 = unsqueeze_422 = None
        mul_559: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(mul_558, sub_159)
        sum_67: f32[256] = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
        mul_560: f32[256] = torch.ops.aten.mul.Tensor(sum_66, 3.0517578125e-05)
        unsqueeze_423: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
        unsqueeze_424: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        unsqueeze_425: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
        mul_561: f32[256] = torch.ops.aten.mul.Tensor(sum_67, 3.0517578125e-05)
        mul_562: f32[256] = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_563: f32[256] = torch.ops.aten.mul.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
        unsqueeze_426: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
        unsqueeze_427: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
        unsqueeze_428: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
        mul_564: f32[256] = torch.ops.aten.mul.Tensor(squeeze_19, primals_79);  primals_79 = None
        unsqueeze_429: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
        unsqueeze_430: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_565: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_428);  sub_159 = unsqueeze_428 = None
        sub_161: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(mul_558, mul_565);  mul_565 = None
        sub_162: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_425);  sub_161 = None
        mul_566: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_431);  sub_162 = unsqueeze_431 = None
        mul_567: f32[256] = torch.ops.aten.mul.Tensor(sum_67, squeeze_19);  sum_67 = squeeze_19 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_566, getitem, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_566 = primals_14 = None
        getitem_95: f32[8, 64, 64, 64] = convolution_backward_28[0]
        getitem_96: f32[256, 64, 1, 1] = convolution_backward_28[1];  convolution_backward_28 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sub_163: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_434);  convolution_6 = unsqueeze_434 = None
        mul_568: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(mul_558, sub_163)
        sum_69: f32[256] = torch.ops.aten.sum.dim_IntList(mul_568, [0, 2, 3]);  mul_568 = None
        mul_570: f32[256] = torch.ops.aten.mul.Tensor(sum_69, 3.0517578125e-05)
        mul_571: f32[256] = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_572: f32[256] = torch.ops.aten.mul.Tensor(mul_570, mul_571);  mul_570 = mul_571 = None
        unsqueeze_438: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_439: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        mul_573: f32[256] = torch.ops.aten.mul.Tensor(squeeze_16, primals_74);  primals_74 = None
        unsqueeze_441: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_442: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_574: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_440);  sub_163 = unsqueeze_440 = None
        sub_165: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(mul_558, mul_574);  mul_558 = mul_574 = None
        sub_166: f32[8, 256, 64, 64] = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_425);  sub_165 = unsqueeze_425 = None
        mul_575: f32[8, 256, 64, 64] = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_443);  sub_166 = unsqueeze_443 = None
        mul_576: f32[256] = torch.ops.aten.mul.Tensor(sum_69, squeeze_16);  sum_69 = squeeze_16 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_575, mul_40, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_575 = mul_40 = primals_13 = None
        getitem_98: f32[8, 64, 64, 64] = convolution_backward_29[0]
        getitem_99: f32[256, 64, 1, 1] = convolution_backward_29[1];  convolution_backward_29 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:91, code: return x * y.expand_as(x)
        mul_577: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(getitem_98, mul_39);  mul_39 = None
        mul_578: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(getitem_98, expand);  getitem_98 = expand = None
        sum_70: f32[8, 64, 1, 1] = torch.ops.aten.sum.dim_IntList(mul_577, [2, 3], True);  mul_577 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_135: f32[8, 1, 64] = torch.ops.aten.view.default(sum_70, [8, 1, 64]);  sum_70 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_167: f32[8, 1, 64] = torch.ops.aten.sub.Tensor(1, sigmoid_5)
        mul_579: f32[8, 1, 64] = torch.ops.aten.mul.Tensor(sigmoid_5, sub_167);  sigmoid_5 = sub_167 = None
        mul_580: f32[8, 1, 64] = torch.ops.aten.mul.Tensor(view_135, mul_579);  view_135 = mul_579 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:86, code: y = self.conv(y)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_580, view, primals_12, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_580 = view = primals_12 = None
        getitem_101: f32[8, 1, 64] = convolution_backward_30[0]
        getitem_102: f32[1, 1, 3] = convolution_backward_30[1];  convolution_backward_30 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        view_136: f32[8, 64] = torch.ops.aten.view.default(getitem_101, [8, 64]);  getitem_101 = None
        unsqueeze_444: f32[8, 64, 1] = torch.ops.aten.unsqueeze.default(view_136, 2);  view_136 = None
        unsqueeze_445: f32[8, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        expand_28: f32[8, 64, 64, 64] = torch.ops.aten.expand.default(unsqueeze_445, [8, 64, 64, 64]);  unsqueeze_445 = None
        div_8: f32[8, 64, 64, 64] = torch.ops.aten.div.Scalar(expand_28, 4096);  expand_28 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        add_208: f32[8, 64, 64, 64] = torch.ops.aten.add.Tensor(mul_578, div_8);  mul_578 = div_8 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        sigmoid_54: f32[8, 64, 64, 64] = torch.ops.aten.sigmoid.default(clone_66)
        empty_like_22: f32[8, 64, 64, 64] = torch.ops.aten.empty_like.default(sigmoid_54, memory_format = torch.preserve_format)
        full_like_22: f32[8, 64, 64, 64] = torch.ops.aten.full_like.default(empty_like_22, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_like_22 = None
        sub_168: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(full_like_22, sigmoid_54);  full_like_22 = None
        mul_581: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(clone_66, sub_168);  clone_66 = sub_168 = None
        add_209: f32[8, 64, 64, 64] = torch.ops.aten.add.Scalar(mul_581, 1);  mul_581 = None
        mul_582: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sigmoid_54, add_209);  sigmoid_54 = add_209 = None
        mul_583: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(add_208, mul_582);  add_208 = mul_582 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_71: f32[64] = torch.ops.aten.sum.dim_IntList(mul_583, [0, 2, 3])
        sub_169: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_448);  convolution_4 = unsqueeze_448 = None
        mul_584: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(mul_583, sub_169)
        sum_72: f32[64] = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
        mul_585: f32[64] = torch.ops.aten.mul.Tensor(sum_71, 3.0517578125e-05)
        unsqueeze_449: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_450: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_586: f32[64] = torch.ops.aten.mul.Tensor(sum_72, 3.0517578125e-05)
        mul_587: f32[64] = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_588: f32[64] = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
        unsqueeze_452: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
        unsqueeze_453: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        mul_589: f32[64] = torch.ops.aten.mul.Tensor(squeeze_13, primals_69);  primals_69 = None
        unsqueeze_455: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
        unsqueeze_456: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_590: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_454);  sub_169 = unsqueeze_454 = None
        sub_171: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(mul_583, mul_590);  mul_583 = mul_590 = None
        sub_172: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_451);  sub_171 = unsqueeze_451 = None
        mul_591: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_457);  sub_172 = unsqueeze_457 = None
        mul_592: f32[64] = torch.ops.aten.mul.Tensor(sum_72, squeeze_13);  sum_72 = squeeze_13 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_591, mul_31, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_591 = mul_31 = primals_11 = None
        getitem_104: f32[8, 64, 64, 64] = convolution_backward_31[0]
        getitem_105: f32[64, 16, 3, 3] = convolution_backward_31[1];  convolution_backward_31 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_595: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(getitem_104, mul_594);  getitem_104 = mul_594 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_73: f32[64] = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3])
        sub_174: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_460);  convolution_3 = unsqueeze_460 = None
        mul_596: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(mul_595, sub_174)
        sum_74: f32[64] = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
        mul_597: f32[64] = torch.ops.aten.mul.Tensor(sum_73, 3.0517578125e-05)
        unsqueeze_461: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_462: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_598: f32[64] = torch.ops.aten.mul.Tensor(sum_74, 3.0517578125e-05)
        mul_599: f32[64] = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_600: f32[64] = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
        unsqueeze_464: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_465: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
        mul_601: f32[64] = torch.ops.aten.mul.Tensor(squeeze_10, primals_64);  primals_64 = None
        unsqueeze_467: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_468: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_602: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_466);  sub_174 = unsqueeze_466 = None
        sub_176: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(mul_595, mul_602);  mul_595 = mul_602 = None
        sub_177: f32[8, 64, 64, 64] = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_463);  sub_176 = unsqueeze_463 = None
        mul_603: f32[8, 64, 64, 64] = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_469);  sub_177 = unsqueeze_469 = None
        mul_604: f32[64] = torch.ops.aten.mul.Tensor(sum_74, squeeze_10);  sum_74 = squeeze_10 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_603, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = getitem = primals_10 = None
        getitem_107: f32[8, 64, 64, 64] = convolution_backward_32[0]
        getitem_108: f32[64, 64, 1, 1] = convolution_backward_32[1];  convolution_backward_32 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        add_211: f32[8, 64, 64, 64] = torch.ops.aten.add.Tensor(getitem_95, getitem_107);  getitem_95 = getitem_107 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/byobnet.py:1547, code: x = self.stem(x)
        max_pool2d_with_indices_backward: f32[8, 64, 128, 128] = torch.ops.aten.max_pool2d_with_indices_backward.default(add_211, mul_23, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_211 = mul_23 = getitem_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_607: f32[8, 64, 128, 128] = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, mul_606);  max_pool2d_with_indices_backward = mul_606 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_75: f32[64] = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3])
        sub_179: f32[8, 64, 128, 128] = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_472);  convolution_2 = unsqueeze_472 = None
        mul_608: f32[8, 64, 128, 128] = torch.ops.aten.mul.Tensor(mul_607, sub_179)
        sum_76: f32[64] = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
        mul_609: f32[64] = torch.ops.aten.mul.Tensor(sum_75, 7.62939453125e-06)
        unsqueeze_473: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_474: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_610: f32[64] = torch.ops.aten.mul.Tensor(sum_76, 7.62939453125e-06)
        mul_611: f32[64] = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_612: f32[64] = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        unsqueeze_476: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_477: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
        mul_613: f32[64] = torch.ops.aten.mul.Tensor(squeeze_7, primals_59);  primals_59 = None
        unsqueeze_479: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
        unsqueeze_480: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_614: f32[8, 64, 128, 128] = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_478);  sub_179 = unsqueeze_478 = None
        sub_181: f32[8, 64, 128, 128] = torch.ops.aten.sub.Tensor(mul_607, mul_614);  mul_607 = mul_614 = None
        sub_182: f32[8, 64, 128, 128] = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_475);  sub_181 = unsqueeze_475 = None
        mul_615: f32[8, 64, 128, 128] = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_481);  sub_182 = unsqueeze_481 = None
        mul_616: f32[64] = torch.ops.aten.mul.Tensor(sum_76, squeeze_7);  sum_76 = squeeze_7 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_615, mul_15, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_615 = mul_15 = primals_9 = None
        getitem_110: f32[8, 32, 128, 128] = convolution_backward_33[0]
        getitem_111: f32[64, 32, 3, 3] = convolution_backward_33[1];  convolution_backward_33 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_619: f32[8, 32, 128, 128] = torch.ops.aten.mul.Tensor(getitem_110, mul_618);  getitem_110 = mul_618 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_77: f32[32] = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3])
        sub_184: f32[8, 32, 128, 128] = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_484);  convolution_1 = unsqueeze_484 = None
        mul_620: f32[8, 32, 128, 128] = torch.ops.aten.mul.Tensor(mul_619, sub_184)
        sum_78: f32[32] = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
        mul_621: f32[32] = torch.ops.aten.mul.Tensor(sum_77, 7.62939453125e-06)
        unsqueeze_485: f32[1, 32] = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_486: f32[1, 32, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487: f32[1, 32, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_622: f32[32] = torch.ops.aten.mul.Tensor(sum_78, 7.62939453125e-06)
        mul_623: f32[32] = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_624: f32[32] = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
        unsqueeze_488: f32[1, 32] = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
        unsqueeze_489: f32[1, 32, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490: f32[1, 32, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
        mul_625: f32[32] = torch.ops.aten.mul.Tensor(squeeze_4, primals_54);  primals_54 = None
        unsqueeze_491: f32[1, 32] = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
        unsqueeze_492: f32[1, 32, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493: f32[1, 32, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_626: f32[8, 32, 128, 128] = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_490);  sub_184 = unsqueeze_490 = None
        sub_186: f32[8, 32, 128, 128] = torch.ops.aten.sub.Tensor(mul_619, mul_626);  mul_619 = mul_626 = None
        sub_187: f32[8, 32, 128, 128] = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_487);  sub_186 = unsqueeze_487 = None
        mul_627: f32[8, 32, 128, 128] = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_493);  sub_187 = unsqueeze_493 = None
        mul_628: f32[32] = torch.ops.aten.mul.Tensor(sum_78, squeeze_4);  sum_78 = squeeze_4 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_627, mul_7, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = mul_7 = primals_8 = None
        getitem_113: f32[8, 24, 128, 128] = convolution_backward_34[0]
        getitem_114: f32[32, 24, 3, 3] = convolution_backward_34[1];  convolution_backward_34 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:80, code: x = self.act(x)
        mul_631: f32[8, 24, 128, 128] = torch.ops.aten.mul.Tensor(getitem_113, mul_630);  getitem_113 = mul_630 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/norm_act.py:68, code: x = F.batch_norm(
        sum_79: f32[24] = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3])
        sub_189: f32[8, 24, 128, 128] = torch.ops.aten.sub.Tensor(convolution, unsqueeze_496);  convolution = unsqueeze_496 = None
        mul_632: f32[8, 24, 128, 128] = torch.ops.aten.mul.Tensor(mul_631, sub_189)
        sum_80: f32[24] = torch.ops.aten.sum.dim_IntList(mul_632, [0, 2, 3]);  mul_632 = None
        mul_633: f32[24] = torch.ops.aten.mul.Tensor(sum_79, 7.62939453125e-06)
        unsqueeze_497: f32[1, 24] = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
        unsqueeze_498: f32[1, 24, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499: f32[1, 24, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_634: f32[24] = torch.ops.aten.mul.Tensor(sum_80, 7.62939453125e-06)
        mul_635: f32[24] = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_636: f32[24] = torch.ops.aten.mul.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        unsqueeze_500: f32[1, 24] = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
        unsqueeze_501: f32[1, 24, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502: f32[1, 24, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        mul_637: f32[24] = torch.ops.aten.mul.Tensor(squeeze_1, primals_49);  primals_49 = None
        unsqueeze_503: f32[1, 24] = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
        unsqueeze_504: f32[1, 24, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505: f32[1, 24, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_638: f32[8, 24, 128, 128] = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_502);  sub_189 = unsqueeze_502 = None
        sub_191: f32[8, 24, 128, 128] = torch.ops.aten.sub.Tensor(mul_631, mul_638);  mul_631 = mul_638 = None
        sub_192: f32[8, 24, 128, 128] = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_499);  sub_191 = unsqueeze_499 = None
        mul_639: f32[8, 24, 128, 128] = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_505);  sub_192 = unsqueeze_505 = None
        mul_640: f32[24] = torch.ops.aten.mul.Tensor(sum_80, squeeze_1);  sum_80 = squeeze_1 = None
        
        # File: /scratch/ngimel/work/env/lib/python3.9/site-packages/timm/models/layers/conv_bn_act.py:35, code: x = self.conv(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_639, primals_45, primals_7, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_639 = primals_45 = primals_7 = None
        getitem_117: f32[24, 3, 3, 3] = convolution_backward_35[1];  convolution_backward_35 = None
        return [permute_78, permute_72, permute_60, permute_54, permute_42, permute_36, getitem_117, getitem_114, getitem_111, getitem_108, getitem_105, getitem_102, getitem_99, getitem_96, getitem_93, getitem_90, getitem_87, getitem_84, getitem_81, getitem_78, getitem_75, getitem_72, getitem_69, getitem_66, getitem_63, getitem_60, getitem_57, getitem_54, getitem_51, getitem_48, getitem_45, getitem_42, getitem_39, getitem_36, getitem_33, getitem_30, getitem_27, getitem_24, getitem_21, getitem_18, getitem_15, getitem_12, permute_28, view_62, None, None, None, None, mul_640, sum_79, None, None, None, mul_628, sum_77, None, None, None, mul_616, sum_75, None, None, None, mul_604, sum_73, None, None, None, mul_592, sum_71, None, None, None, mul_576, sum_66, None, None, None, mul_567, sum_66, None, None, None, mul_555, sum_64, None, None, None, mul_543, sum_62, None, None, None, mul_527, sum_59, None, None, None, mul_515, sum_57, None, None, None, mul_503, sum_55, None, None, None, mul_487, sum_50, None, None, None, mul_478, sum_50, None, None, None, mul_466, sum_48, None, None, None, mul_454, sum_46, None, None, None, mul_438, sum_43, None, None, None, mul_426, sum_41, None, None, None, mul_414, sum_39, None, None, None, mul_398, sum_34, None, None, None, mul_389, sum_34, None, None, None, mul_377, sum_32, None, None, None, mul_362, sum_27, None, None, None, mul_350, sum_25, None, None, None, mul_338, sum_23, None, None, None, mul_323, sum_18, None, None, None, mul_311, sum_14, None, None, None, mul_302, sum_14, None, None, None, mul_290, sum_12, None, None, None, mul_275, sum_7, None, None, None, mul_263, sum_5]
        