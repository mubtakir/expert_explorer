
# -*- coding: utf-8 -*-
'''
لازال في مرحلة تطوير
'''

"""
كل الأفكار المبتكرة تعود بمطلقها للمطور: باسل يحيى عبدالله
مع أكواد أولية قدمها للنماذج الذكاء الصطناعي لانتاج اكواد أكبر وأنضج.
هذا العمل (وكثير غيره) لازال في مرحلة التطوير وأجعله مفتوح المصدر بترخيص MIT
بشرط ذكر المصدر الأساس عند تطويره والاستفادة منه، ومع احتفاظ المطور (باسل يحيى عبدالله) بحقوق الملكية الفكرية
وهو غير مسؤول عن أي مخاطر نتيجة استخدام وتجريب هذا الكود أو غيره مما أنتجه
"""

"""
==============================================================================
 نظام  / RL-E⁴: وكيل تعلم موحد مع معادلة قابلة للتكيف 
 و ShapeExtractor مع تحسين بايزي () في نفس الملف
==============================================================================

 **الوصف:**
 هذا الملف يحتوي على تعريفات فئات نظام التعلم المعزز المتقدم RL-E⁴،
 بالإضافة إلى فئات ShapeExtractor و ShapePlotter2D، ووظيفة مقارنة SSIM،
 والجزء الرئيسي (__main__) يقوم بتشغيل حلقة التحسين البايزي لـ ShapeExtractor.

 (الإصدار 1.1.1 من RL-E⁴: إصلاح NameError، دمج RiskProgressEvaluator)
 (الإصدار 1.1.0 من حلقة التحسين: يستخدم Bayesian Optimization، يصلح أخطاء)

 يلتزم الكود بقاعدة "تعليمة واحدة لكل سطر".

 **المتطلبات:**
 - Python 3.x, NumPy, PyTorch, Gymnasium, Matplotlib, Scikit-optimize, Scikit-image, Pyparsing.
 - (Optional) PyBullet, Pillow.

 **كيفية الاستخدام (لحلقة التحسين):**
 1.  تأكد من تثبيت جميع المكتبات.
 2.  **هام:** عدّل `external_image_path` في قسم `if __name__ == "__main__":`.
 3.  (اختياري) عدّل `n_calls_optimizer` و `search_space_definitions`.
 4.  قم بتشغيل الكود.
"""

# --- 1. Imports ---
import logging
import os
import math
import re
import time
import sys
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Callable
from collections import defaultdict, deque, OrderedDict
import traceback
import random
import copy
import inspect
import warnings
# import json # Not currently used

# --- 2. Library Availability Checks ---
try:
    import cv2
    CV_AVAILABLE = True # <-- ** تم إضافة التعريف هنا **
except ImportError:
    cv2 = None; CV_AVAILABLE = False; print("ERROR: OpenCV required.")
    sys.exit(1) # Exit if OpenCV is essential for ShapeExtractor

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None; PIL_AVAILABLE = False; print("INFO: Pillow not available.")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    ssim = None; SKIMAGE_AVAILABLE = False; print("WARNING: scikit-image not found. SSIM comparison/optimization disabled.")

try:
    from pyparsing import (Word, alphas, alphanums, nums, hexnums,
                           Suppress, Optional as ppOptional, Group,
                           delimitedList, Literal, Combine, CaselessLiteral,
                           ParseException, StringEnd)
    PYPARSING_AVAILABLE = True
except ImportError:
    Word, alphas, alphanums, nums, hexnums = None, None, None, None, None
    Suppress, ppOptional, Group, delimitedList = None, None, None, None
    Literal, Combine, CaselessLiteral = None, None, None
    ParseException, StringEnd = None, None
    PYPARSING_AVAILABLE = False
    print("ERROR: pyparsing library is required for ShapePlotter2D.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib, plt, LinearSegmentedColormap, LineCollection = None, None, None, None
    MATPLOTLIB_AVAILABLE = False
    print("ERROR: Matplotlib is required for plotting.")
    sys.exit(1)
except Exception as e_mpl:
    matplotlib, plt, LinearSegmentedColormap, LineCollection = None, None, None, None
    MATPLOTLIB_AVAILABLE = False
    print(f"ERROR: Failed to initialize Matplotlib: {e_mpl}")
    sys.exit(1)

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    gp_minimize, Real, Integer, Categorical, use_named_args = None, None, None, None, None
    SKOPT_AVAILABLE = False
    print("WARNING: scikit-optimize not found. Bayesian Optimization disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal # Needed for PPO example if merged
    TORCH_AVAILABLE = True
except ImportError:
    torch, nn, F, optim, Normal = None, None, None, None, None
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required for RL Agent components.")
    # Decide if exit is needed depending on which part is run
    # sys.exit(1) # Exit if running RL part

try:
    import gymnasium as gym # Use gymnasium
    from gymnasium import wrappers
    GYMNASIUM_AVAILABLE = True
except ImportError:
    gym = None; wrappers = None
    GYMNASIUM_AVAILABLE = False
    print("ERROR: Gymnasium is required for RL Agent components.")
    # sys.exit(1) # Exit if running RL part


# --- 3. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'
)
logger = logging.getLogger("OmniMind_ShapeOpt") # More descriptive name

# --- 4. Type Definitions ---
PointInt = Tuple[int, int]
PointFloat = Tuple[float, float]
ColorBGR = Tuple[int, int, int]
ShapeData = Dict[str, Any]

# --- 5. Evolving Equation & Evolution Engine (from RL-E4) ---
class EvolvingEquation(nn.Module):
    """معادلة متطورة (v1.1.1)."""
    def __init__(self, input_dim: int, init_complexity: int = 3, output_dim: int = 1,
                 complexity_limit: int = 15, min_complexity: int = 2,
                 output_activation: Optional[Callable] = None,
                 exp_clamp_min: float = 0.1, exp_clamp_max: float = 4.0,
                 term_clamp: float = 1e4, output_clamp: float = 1e5):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0: raise ValueError("input_dim positive int.")
        if not isinstance(output_dim, int) or output_dim <= 0: raise ValueError("output_dim positive int.")
        if not isinstance(init_complexity, int) or init_complexity <= 0: raise ValueError("init_complexity positive int.")
        if not isinstance(min_complexity, int) or min_complexity <= 0: raise ValueError("min_complexity positive int.")
        if not isinstance(complexity_limit, int) or complexity_limit < min_complexity: raise ValueError(f"complexity_limit >= min_complexity required.")
        self.input_dim = input_dim; self.output_dim = output_dim
        self.complexity = max(min_complexity, min(init_complexity, complexity_limit))
        self.complexity_limit = complexity_limit; self.min_complexity = min_complexity
        self.output_activation = output_activation; self.exp_clamp_min = exp_clamp_min
        self.exp_clamp_max = exp_clamp_max; self.term_clamp = term_clamp; self.output_clamp = output_clamp
        self._initialize_components()
        self._func_repr_map = OrderedDict([
            ('sin',{'shape':'sine','color':'#FF6347'}), ('cos',{'shape':'cosine','color':'#4682B4'}),
            ('tanh',{'shape':'wave','color':'#32CD32'}), ('sigmoid',{'shape':'step','color':'#FFD700'}),
            ('relu',{'shape':'ramp','color':'#DC143C'}), ('leaky_relu',{'shape':'ramp','color':'#8A2BE2'}),
            ('gelu',{'shape':'smoothramp','color':'#00CED1'}), ('<lambda>',{'shape':'line','color':'#A9A9A9'}),
            ('pow',{'shape':'parabola','color':'#6A5ACD'}), ('exp',{'shape':'decay','color':'#FF8C00'}),
            ('sqrt',{'shape':'root','color':'#2E8B57'}), ('clamp',{'shape':'plateau','color':'#DAA520'}),
            ('*',{'shape':'swishlike','color':'#D2691E'}) ])

    def _initialize_components(self):
        self.input_transform = nn.Linear(self.input_dim, self.complexity)
        nn.init.xavier_uniform_(self.input_transform.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.input_transform.bias)
        self.coefficients = nn.ParameterList()
        self.exponents = nn.ParameterList()
        i = 0
        while i < self.complexity:
            coeff = nn.Parameter(torch.randn(1) * 0.05, requires_grad=True)
            self.coefficients.append(coeff)
            exp = nn.Parameter(torch.abs(torch.randn(1) * 0.1) + 1.0, requires_grad=True)
            self.exponents.append(exp)
            i += 1
        self.function_library: List[Callable] = [torch.sin, torch.cos, torch.tanh, torch.sigmoid, F.relu, F.leaky_relu, F.gelu, lambda x: x, lambda x: torch.pow(x, 2), lambda x: torch.exp(-torch.abs(x)), lambda x: torch.sqrt(torch.abs(x) + 1e-6), lambda x: torch.clamp(x, -3.0, 3.0), lambda x: x * torch.sigmoid(x),]
        if not self.function_library: raise ValueError("Function library empty.")
        self.functions: List[Callable] = []
        k = 0
        while k < self.complexity: self.functions.append(self._select_function()); k += 1
        self.output_layer = nn.Linear(self.complexity, self.output_dim)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.output_layer.bias)

    def _select_function(self) -> Callable: return random.choice(self.function_library)
    def _safe_pow(self, base: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(base); base_abs = torch.abs(base) + 1e-8; powered = torch.pow(base_abs, exp)
        return sign * powered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            if x.shape[0] == self.input_dim: x = x.unsqueeze(0)
            else: raise ValueError("1D Input dim mismatch")
        elif x.dim() > 2:
             orig_shape = x.shape; x = x.view(x.shape[0], -1)
             if x.shape[1] != self.input_dim: raise ValueError("Flattened dim mismatch")
             # warnings.warn(f"Input > 2D flattened.", RuntimeWarning) # Reduce noise
        elif x.dim() == 2:
            if x.shape[1] != self.input_dim: raise ValueError("2D Input dim mismatch")
        else: raise ValueError("Input tensor must be >= 1D.")
        try: dev = next(self.parameters()).device
        except StopIteration: dev = torch.device("cpu")
        if x.device != dev: x = x.to(dev)
        if torch.isnan(x).any(): x = torch.nan_to_num(x, nan=0.0)
        transformed: Optional[torch.Tensor] = None
        try:
            transformed = self.input_transform(x)
            if torch.isnan(transformed).any(): transformed = torch.nan_to_num(transformed, nan=0.0)
            transformed = torch.clamp(transformed, -self.term_clamp, self.term_clamp)
        except Exception: return torch.zeros(x.shape[0], self.output_dim, device=dev)
        terms = torch.zeros(x.shape[0], self.complexity, device=dev)
        if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')): return torch.zeros(x.shape[0], self.output_dim, device=dev)
        len_c = len(self.coefficients); len_e = len(self.exponents); len_f = len(self.functions)
        current_comp = min(len_c, len_e, len_f)
        if current_comp != self.complexity: self.complexity = current_comp # Try to self-correct
        if self.complexity == 0: return torch.zeros(x.shape[0], self.output_dim, device=dev)
        idx = 0
        while idx < self.complexity:
            try:
                feat = transformed[:, idx]; exp = torch.clamp(self.exponents[idx], self.exp_clamp_min, self.exp_clamp_max)
                powered = self._safe_pow(feat, exp); powered = torch.clamp(powered, -self.term_clamp, self.term_clamp)
                if torch.isnan(powered).any(): powered = torch.zeros_like(feat)
                activated = self.functions[idx](powered); activated = torch.clamp(activated, -self.term_clamp, self.term_clamp)
                if torch.isnan(activated).any(): activated = torch.zeros_like(feat)
                term = self.coefficients[idx] * activated; terms[:, idx] = torch.clamp(term, -self.term_clamp, self.term_clamp)
            except Exception: terms[:, idx] = 0.0
            idx += 1
        if torch.isnan(terms).any(): terms = torch.nan_to_num(terms, nan=0.0)
        output: Optional[torch.Tensor] = None
        try:
            out_raw = self.output_layer(terms[:, :self.complexity]) # Use current complexity
            out_clamp = torch.clamp(out_raw, -self.output_clamp, self.output_clamp)
            if torch.isnan(out_clamp).any(): out_clamp = torch.nan_to_num(out_clamp, nan=0.0)
            if self.output_activation: out_act = self.output_activation(out_clamp); output = torch.nan_to_num(out_act, nan=0.0) if torch.isnan(out_act).any() else out_act
            else: output = out_clamp
        except Exception: output = torch.zeros(x.shape[0], self.output_dim, device=dev)
        if output is None: output = torch.zeros(x.shape[0], self.output_dim, device=dev)
        elif torch.isnan(output).any(): output = torch.zeros_like(output)
        return output

    def to_shape_engine_string(self) -> str:
        """تمثيل المعادلة كنص لـ ShapePlotter2D."""
        parts = []; scale_p = 5.0; scale_e = 2.0
        if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')): return "eq_not_init"
        current_comp = min(len(self.coefficients), len(self.exponents), len(self.functions))
        if current_comp != self.complexity: return "eq_mismatch"
        idx = 0
        while idx < current_comp:
             coeff = self.coefficients[idx]; exp = self.exponents[idx]; func = self.functions[idx]
             coeff_v = round(coeff.item(), 3); exp_v_clamp = torch.clamp(exp, self.exp_clamp_min, self.exp_clamp_max); exp_v = round(exp_v_clamp.item(), 3)
             f_name = getattr(func, '__name__', '<lambda>')
             if f_name == '<lambda>':
                 f_repr = repr(func)
                 if 'pow' in f_repr and '** 2' in f_repr: f_name = 'pow'
                 elif '*' in f_repr and 'sigmoid' in f_repr: f_name = '*'
                 elif 'sqrt' in f_repr: f_name = 'sqrt'
                 elif 'exp' in f_repr and 'abs' in f_repr: f_name = 'exp'
                 elif 'clamp' in f_repr: f_name = 'clamp'
                 elif 'x' in f_repr and len(f_repr) < 25: f_name = '<lambda>'
             repr_info = self._func_repr_map.get(f_name, self._func_repr_map['<lambda>']); s_type = repr_info['shape']
             p1 = round(idx*scale_p*0.5 + coeff_v*scale_p*0.2, 2); p2 = round(coeff_v*scale_p, 2); p3 = round(abs(exp_v)*scale_e, 2)
             params_s = f"{p1},{p2},{p3}"
             styles: Dict[str, Any] = {}; styles['color'] = repr_info['color']; styles['linewidth'] = round(1.0 + abs(exp_v - 1.0)*1.5, 2)
             styles['opacity'] = round(np.clip(0.4 + abs(coeff_v)*0.5, 0.2, 0.9), 2)
             if coeff_v < -0.01: styles['fill'] = 'True'
             if f_name in ['cos', 'relu', 'leaky_relu']: styles['dash'] = '--'
             styles_s_list: List[str] = [];
             for k, v in styles.items(): styles_s_list.append(f"{k}={v}")
             styles_s = ",".join(styles_s_list)
             term_s = f"{s_type}({params_s}){{{styles_s}}}"
             parts.append(term_s); idx += 1
        return " + ".join(parts) if parts else "empty"

    def add_term(self) -> bool:
        """إضافة حد جديد."""
        if self.complexity >= self.complexity_limit: return False
        new_comp = self.complexity + 1;
        try: dev = next(self.parameters()).device
        except StopIteration: dev = torch.device("cpu")
        try: old_iw=self.input_transform.weight.data.clone(); old_ib=self.input_transform.bias.data.clone(); old_ow=self.output_layer.weight.data.clone(); old_ob=self.output_layer.bias.data.clone()
        except AttributeError: return False
        new_c = nn.Parameter(torch.randn(1, device=dev)*0.01, requires_grad=True); self.coefficients.append(new_c)
        new_e = nn.Parameter(torch.abs(torch.randn(1, device=dev)*0.05)+1.0, requires_grad=True); self.exponents.append(new_e)
        new_f = self._select_function(); self.functions.append(new_f)
        new_in = nn.Linear(self.input_dim, new_comp, device=dev)
        with torch.no_grad():
            new_in.weight.data[:self.complexity,:] = old_iw; new_in.bias.data[:self.complexity] = old_ib
            nn.init.xavier_uniform_(new_in.weight.data[self.complexity:], gain=nn.init.calculate_gain('relu'))
            new_in.weight.data[self.complexity:] *= 0.01; nn.init.zeros_(new_in.bias.data[self.complexity:])
        self.input_transform = new_in
        new_out = nn.Linear(new_comp, self.output_dim, device=dev)
        with torch.no_grad():
            new_out.weight.data[:,:self.complexity] = old_ow
            nn.init.xavier_uniform_(new_out.weight.data[:,self.complexity:], gain=nn.init.calculate_gain('linear'))
            new_out.weight.data[:,self.complexity:] *= 0.01; new_out.bias.data.copy_(old_ob)
        self.output_layer = new_out
        self.complexity = new_comp; return True

    def prune_term(self, aggressive: bool = False) -> bool:
        """إزالة حد."""
        if self.complexity <= self.min_complexity: return False
        new_comp = self.complexity - 1; idx_prune = -1
        try: dev = next(self.parameters()).device
        except StopIteration: dev = torch.device("cpu")
        try: # Select index
            if aggressive and self.complexity > 1:
                 with torch.no_grad():
                    if not hasattr(self, 'coefficients') or not self.coefficients: idx_prune = random.randint(0, self.complexity - 1)
                    else:
                        coeffs_cpu = torch.tensor([torch.abs(c.data).item() for c in self.coefficients], device='cpu')
                        in_w = self.input_transform.weight; out_w = self.output_layer.weight
                        in_w_cpu = in_w.data.cpu() if in_w is not None else None; out_w_cpu = out_w.data.cpu() if out_w is not None else None
                        in_norm = torch.norm(in_w_cpu, p=1, dim=1) if in_w_cpu is not None else torch.zeros(self.complexity, device='cpu')
                        out_norm = torch.norm(out_w_cpu, p=1, dim=0) if out_w_cpu is not None else torch.zeros(self.complexity, device='cpu')
                        match = (coeffs_cpu.shape[0]==self.complexity and in_norm.shape[0]==self.complexity and out_norm.shape[0]==self.complexity)
                        if match: imp = (coeffs_cpu*(in_norm+out_norm))+1e-9; idx_prune = torch.argmin(imp).item() if not torch.isnan(imp).any() else random.randint(0, self.complexity - 1)
                        else: idx_prune = random.randint(0, self.complexity - 1)
            else: idx_prune = random.randint(0, self.complexity - 1)
            if not (0 <= idx_prune < self.complexity): return False
        except Exception: return False
        try: # Backup weights
            old_iw=self.input_transform.weight.data.clone(); old_ib=self.input_transform.bias.data.clone()
            old_ow=self.output_layer.weight.data.clone(); old_ob=self.output_layer.bias.data.clone()
        except AttributeError: return False
        try: # Remove components
            if hasattr(self,'coefficients') and len(self.coefficients)>idx_prune: del self.coefficients[idx_prune]
            if hasattr(self,'exponents') and len(self.exponents)>idx_prune: del self.exponents[idx_prune]
            if hasattr(self,'functions') and len(self.functions)>idx_prune: self.functions.pop(idx_prune)
        except (IndexError,Exception): return False
        # Shrink layers
        new_in = nn.Linear(self.input_dim, new_comp, device=dev)
        with torch.no_grad():
            new_wi = torch.cat([old_iw[:idx_prune], old_iw[idx_prune+1:]], dim=0); new_bi = torch.cat([old_ib[:idx_prune], old_ib[idx_prune+1:]])
            if new_wi.shape[0]!=new_comp or new_bi.shape[0]!=new_comp: return False
            new_in.weight.data.copy_(new_wi); new_in.bias.data.copy_(new_bi)
        self.input_transform = new_in
        new_out = nn.Linear(new_comp, self.output_dim, device=dev)
        with torch.no_grad():
            new_wo = torch.cat([old_ow[:,:idx_prune], old_ow[:,idx_prune+1:]], dim=1)
            if new_wo.shape[1]!=new_comp: return False
            new_out.weight.data.copy_(new_wo); new_out.bias.data.copy_(old_ob)
        self.output_layer = new_out
        self.complexity = new_comp; return True


# --- 4. Core Component: Evolution Engine ---
class EvolutionEngine:
    """محرك التطور: يدير تطور `EvolvingEquation`."""
    def __init__(self, mutation_power: float = 0.03, history_size: int = 50, cooldown_period: int = 30,
                 add_term_threshold: float = 0.85, prune_term_threshold: float = 0.20,
                 add_term_prob: float = 0.15, prune_term_prob: float = 0.25, swap_func_prob: float = 0.02):
        # Validation omitted for brevity, assumed correct from previous versions
        self.base_mutation_power = mutation_power; self.performance_history: deque = deque(maxlen=history_size)
        self.cooldown_period = cooldown_period; self.term_change_cooldown = 0
        self.add_term_threshold = add_term_threshold; self.prune_term_threshold = prune_term_threshold
        self.add_term_prob = add_term_prob; self.prune_term_prob = prune_term_prob; self.swap_func_prob = swap_func_prob

    def _calculate_percentile(self, current_reward: float) -> float:
        if math.isnan(current_reward): return 0.5
        valid_history = [r for r in self.performance_history if not math.isnan(r)];
        if not valid_history: return 0.5
        history_array = np.array(valid_history)
        percentile = 0.5
        try: percentile_score = stats.percentileofscore(history_array, current_reward, kind='mean') / 100.0; percentile = percentile_score
        except Exception: percentile = np.mean(history_array < current_reward)
        return np.clip(percentile, 0.0, 1.0)

    def _dynamic_mutation_scale(self, percentile: float) -> float:
        if math.isnan(percentile): percentile = 0.5
        scale = 1.0
        if percentile > 0.9: scale = 0.5
        elif percentile < 0.1: scale = 1.5
        else: scale = 1.5 - (percentile - 0.1) * (1.0 / 0.8)
        return max(0.1, min(scale, 2.0))

    def evolve_equation(self, equation: EvolvingEquation, reward: float, step: int) -> bool:
        structure_changed = False
        if not isinstance(equation, EvolvingEquation): return False
        if not math.isnan(reward): self.performance_history.append(reward); perf_percentile = self._calculate_percentile(reward)
        else: valid_hist = [r for r in self.performance_history if not math.isnan(r)]; perf_percentile = self._calculate_percentile(valid_hist[-1]) if valid_hist else 0.5

        if self.term_change_cooldown > 0: self.term_change_cooldown -= 1

        can_change_struct = (self.term_change_cooldown == 0 and len(self.performance_history) >= max(10, self.performance_history.maxlen // 4))
        if can_change_struct:
            rand_val = random.random(); action_desc = "None"
            try_add = perf_percentile > self.add_term_threshold and rand_val < self.add_term_prob
            if try_add:
                if equation.add_term(): self.term_change_cooldown = self.cooldown_period; structure_changed = True; action_desc = f"Add (->{equation.complexity})"
            else:
                try_prune = perf_percentile < self.prune_term_threshold and rand_val < self.prune_term_prob
                if try_prune:
                    if equation.prune_term(aggressive=True): self.term_change_cooldown = self.cooldown_period; structure_changed = True; action_desc = f"Prune (->{equation.complexity})"
            if structure_changed: logger.info(f"EVO: Step {step}: Eq '{type(equation).__name__}' -> {action_desc} | Pct: {perf_percentile:.2f}") # Use main logger

        can_swap = (not structure_changed and self.term_change_cooldown == 0 and random.random() < self.swap_func_prob)
        if can_swap:
            swapped_ok = self.swap_function(equation)
            if swapped_ok: self.term_change_cooldown = max(self.term_change_cooldown, 2)

        mut_scale = self._dynamic_mutation_scale(perf_percentile)
        self._mutate_parameters(equation, mut_scale, step)
        return structure_changed

    def _mutate_parameters(self, equation: EvolvingEquation, mutation_scale: float, step: int):
        if not isinstance(equation, EvolvingEquation): return
        cooling = max(0.5, 1.0 - step / 2000000.0); power = self.base_mutation_power * mutation_scale * cooling
        if power < 1e-9: return
        try:
            with torch.no_grad():
                try: device = next(equation.parameters()).device
                except StopIteration: return
                if hasattr(equation, 'coefficients') and equation.coefficients:
                    idx = 0; L = len(equation.coefficients)
                    while idx < L: coeff = equation.coefficients[idx]; noise = torch.randn_like(coeff.data)*power; coeff.data.add_(noise); idx+=1
                if hasattr(equation, 'exponents') and equation.exponents:
                    idx = 0; L = len(equation.exponents); exp_power = power * 0.1
                    while idx < L: exp = equation.exponents[idx]; noise = torch.randn_like(exp.data)*exp_power; exp.data.add_(noise); exp.data.clamp_(min=equation.exp_clamp_min, max=equation.exp_clamp_max); idx+=1
                if hasattr(equation, 'input_transform') and isinstance(equation.input_transform, nn.Linear):
                    layer = equation.input_transform; ps = 0.3; noise_w = torch.randn_like(layer.weight.data)*power*ps; layer.weight.data.add_(noise_w)
                    if layer.bias is not None: noise_b = torch.randn_like(layer.bias.data)*power*ps*0.5; layer.bias.data.add_(noise_b)
                if hasattr(equation, 'output_layer') and isinstance(equation.output_layer, nn.Linear):
                    layer = equation.output_layer; ps = 0.3; noise_w = torch.randn_like(layer.weight.data)*power*ps; layer.weight.data.add_(noise_w)
                    if layer.bias is not None: noise_b = torch.randn_like(layer.bias.data)*power*ps*0.5; layer.bias.data.add_(noise_b)
        except Exception as e: warnings.warn(f"Mutation error step {step}: {e}", RuntimeWarning)

    def swap_function(self, equation: EvolvingEquation) -> bool:
        if not isinstance(equation, EvolvingEquation): return False
        if equation.complexity <= 0: return False
        if not hasattr(equation, 'functions') or not equation.functions: return False
        if not hasattr(equation, 'function_library') or not equation.function_library: return False
        try:
            idx = random.randint(0, equation.complexity - 1); old_f = equation.functions[idx]
            attempts=0; max_att=len(equation.function_library)*2; new_f=old_f
            while attempts < max_att:
                cand_f = equation._select_function(); cand_repr = getattr(cand_f, '__name__', repr(cand_f)); old_repr = getattr(old_f, '__name__', repr(old_f))
                is_diff = (cand_repr != old_repr) or (cand_repr == '<lambda>' and cand_f is not old_f) or (len(equation.function_library) == 1)
                if is_diff or attempts == max_att-1: new_f=cand_f; break
                attempts += 1
            equation.functions[idx] = new_f; return True
        except (IndexError, Exception) as e: warnings.warn(f"Function swap error: {e}", RuntimeWarning); return False


# --- 5. Core Component: Replay Buffer ---
class ReplayBuffer:
    """ذاكرة تخزين مؤقت للتجارب مع فحص NaN."""
    def __init__(self, capacity=100000):
        if not isinstance(capacity, int) or capacity <= 0: raise ValueError("Capacity must be pos int.")
        self.capacity = capacity; self.buffer = deque(maxlen=capacity)
        self._push_nan_warnings = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}; self._sample_nan_warning = 0
    def push(self, state, action, reward, next_state, done):
        skip = False; nan_src = None
        try:
            if isinstance(reward, (int, float)) and (math.isnan(reward) or math.isinf(reward)): reward = 0.0; self._push_nan_warnings['reward'] += 1; nan_src = 'reward(fixed)'
            s_arr = np.asarray(state, dtype=np.float32); ns_arr = np.asarray(next_state, dtype=np.float32); a_arr = np.asarray(action, dtype=np.float32)
            if np.any(np.isnan(s_arr)) or np.any(np.isinf(s_arr)): skip = True; self._push_nan_warnings['state'] += 1; nan_src = 'state'
            if not skip and (np.any(np.isnan(ns_arr)) or np.any(np.isinf(ns_arr))): skip = True; self._push_nan_warnings['next_state'] += 1; nan_src = 'next_state'
            if not skip and (np.any(np.isnan(a_arr)) or np.any(np.isinf(a_arr))): skip = True; self._push_nan_warnings['action'] += 1; nan_src = 'action'
            if skip:
                tot_warn = sum(self._push_nan_warnings.values())
                if tot_warn % 500 == 1: warnings.warn(f"Skip exp NaN/Inf in '{nan_src}'. Skips(S/A/R/S'):{self._push_nan_warnings['state']}/{self._push_nan_warnings['action']}/{self._push_nan_warnings['reward']}/{self._push_nan_warnings['next_state']}", RuntimeWarning)
                return
            done_float = float(done)
            experience = (s_arr, a_arr, float(reward), ns_arr, done_float)
            self.buffer.append(experience)
        except (TypeError, ValueError) as e: warnings.warn(f"Could not store experience: {e}. Skip.", RuntimeWarning)
    def sample(self, batch_size):
        current_size = len(self.buffer)
        if current_size < batch_size: return None
        try: indices = np.random.choice(current_size, batch_size, replace=False); batch = [self.buffer[i] for i in indices]
        except (ValueError, Exception) as e: warnings.warn(f"Error sampling buffer: {e}.", RuntimeWarning); return None
        try:
            states, actions, rewards, next_states, dones = zip(*batch)
            s_np=np.array(states,dtype=np.float32); a_np=np.array(actions,dtype=np.float32); r_np=np.array(rewards,dtype=np.float32).reshape(-1,1)
            ns_np=np.array(next_states,dtype=np.float32); d_np=np.array(dones,dtype=np.float32).reshape(-1,1)
            is_nan_inf = (np.isnan(s_np).any() or np.isnan(a_np).any() or np.isnan(r_np).any() or np.isnan(ns_np).any() or np.isnan(d_np).any() or
                          np.isinf(s_np).any() or np.isinf(a_np).any() or np.isinf(r_np).any() or np.isinf(ns_np).any() or np.isinf(d_np).any())
            if is_nan_inf:
                self._sample_nan_warning += 1
                if self._sample_nan_warning % 100 == 1: warnings.warn(f"NaN/Inf sampled batch. Skips: {self._sample_nan_warning}. None.", RuntimeWarning)
                return None
            s_t=torch.from_numpy(s_np); a_t=torch.from_numpy(a_np); r_t=torch.from_numpy(r_np)
            ns_t=torch.from_numpy(ns_np); d_t=torch.from_numpy(d_np)
        except (ValueError, TypeError, Exception) as e: warnings.warn(f"Failed convert batch tensors: {e}. None.", RuntimeWarning); return None
        return s_t, a_t, r_t, ns_t, d_t
    def __len__(self): return len(self.buffer)


# --- 6. المكونات الجديدة: مكونات المعادلة الموحدة ---
class EquationComponent(nn.Module):
    """الفئة الأساسية للمكونات."""
    def __init__(self, component_id: str, tags: Optional[Set[str]] = None,
                 required_context: Optional[Set[str]] = None, provides: Optional[Set[str]] = None):
        super().__init__()
        self.component_id = component_id
        self.tags = tags if tags else set()
        self.required_context = required_context if required_context else set()
        self.provides = provides if provides else set()
        # --- حفظ وسائط المنشئ ---
        frame = inspect.currentframe().f_back
        if frame: # Check if frame exists
            args_info = inspect.getargvalues(frame)
            args_names = args_info.args
            args_values = args_info.locals
            self._init_args = {arg_name: args_values[arg_name] for arg_name in args_names if arg_name != 'self'}
            # Convert sets to lists for serialization
            if 'tags' in self._init_args and self._init_args['tags'] is not None: self._init_args['tags'] = list(self._init_args['tags'])
            if 'required_context' in self._init_args and self._init_args['required_context'] is not None: self._init_args['required_context'] = list(self._init_args['required_context'])
            if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self._init_args['provides'])
        else: self._init_args = {'component_id': component_id} # Fallback

    def is_active(self, context: Dict[str, Any]) -> bool:
        if not self.required_context: return True
        return self.required_context.issubset(context.keys())

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement forward.")

    def evolve(self, engine: EvolutionEngine, reward: float, step: int) -> bool: return False
    def to_string(self) -> str:
        req_str = ','.join(self.required_context) if self.required_context else 'Any'
        prov_str = ','.join(self.provides) if self.provides else 'None'
        tag_str = ','.join(self.tags) if self.tags else 'None'
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.component_id}, tags={{{tag_str}}}, req={{{req_str}}}, prov={{{prov_str}}})"
    def get_complexity(self) -> float: return 1.0


class NumericTransformComponent(EquationComponent):
    """مكون يغلف EvolvingEquation."""
    def __init__(self, component_id: str, equation: EvolvingEquation,
                 activation: Optional[Callable] = None, tags: Optional[Set[str]] = None,
                 required_context: Optional[Set[str]] = None, provides: Optional[Set[str]] = None,
                 input_key: str = 'features', output_key: str = 'features'):
        provides = provides if provides is not None else {output_key}
        super().__init__(component_id, tags, required_context, provides)
        if not isinstance(equation, EvolvingEquation): raise TypeError("Eq must be EvolvingEquation.")
        self.equation = equation; self.activation = activation
        self.input_key = input_key; self.output_key = output_key
        self._init_args['equation_config'] = { 'input_dim': equation.input_dim, 'init_complexity': equation.complexity, 'output_dim': equation.output_dim, 'complexity_limit': equation.complexity_limit, 'min_complexity': equation.min_complexity, 'exp_clamp_min': equation.exp_clamp_min, 'exp_clamp_max': equation.exp_clamp_max, 'term_clamp': equation.term_clamp, 'output_clamp': equation.output_clamp, }
        activation_name = 'None';
        if activation and hasattr(activation, '__name__'): activation_name = activation.__name__
        self._init_args['activation_name'] = activation_name
        self._init_args.pop('equation', None); self._init_args.pop('activation', None)
        self._init_args['input_key'] = input_key; self._init_args['output_key'] = output_key
        self.provides = self.provides.union({output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = list(self.provides)

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.input_key not in data: return {}
        input_tensor = data[self.input_key]
        if not isinstance(input_tensor, torch.Tensor): return {}
        if input_tensor.shape[-1] != self.equation.input_dim: return {}
        try:
            transformed = self.equation(input_tensor)
            if torch.isnan(transformed).any(): transformed = torch.zeros_like(transformed)
            if self.activation:
                activated = self.activation(transformed)
                if torch.isnan(activated).any(): activated = torch.zeros_like(activated)
                update_dict = {self.output_key: activated}; return update_dict
            else: update_dict = {self.output_key: transformed}; return update_dict
        except Exception as e: warnings.warn(f"Error NTC '{self.component_id}': {e}. Skip.", RuntimeWarning); return {}

    def evolve(self, engine: EvolutionEngine, reward: float, step: int) -> bool:
        structure_changed = engine.evolve_equation(self.equation, reward, step)
        if structure_changed:
            if 'equation_config' in self._init_args and isinstance(self._init_args['equation_config'], dict):
                self._init_args['equation_config']['init_complexity'] = self.equation.complexity
        return structure_changed

    def to_string(self) -> str:
        base = super().to_string().replace("Component","NumericTransform")[:-1]
        act = self._init_args.get('activation_name', 'None')
        eq_str = self.equation.to_shape_engine_string() if hasattr(self.equation, 'to_shape_engine_string') else "N/A"
        comp = self.get_complexity()
        info = f", EqComp={comp:.1f}, Act={act}, In='{self.input_key}', Out='{self.output_key}', Eq='{eq_str}'"
        return base + info + ")"

    def get_complexity(self) -> float: return getattr(self.equation, 'complexity', 1.0)


class PolicyDecisionComponent(EquationComponent):
    """مكون لتطبيق Tanh."""
    def __init__(self, component_id: str, action_dim: int,
                 tags: Optional[Set[str]] = None, required_context: Optional[Set[str]] = None,
                 provides: Optional[Set[str]] = None,
                 input_key: str = 'policy_features', output_key: str = 'action_raw'):
        tags = tags if tags is not None else {'actor'}
        req_ctx = required_context if required_context is not None else {'request_action'}
        prov = provides if provides is not None else {output_key}
        super().__init__(component_id, tags, req_ctx, prov)
        self.action_dim = action_dim; self.input_key = input_key; self.output_key = output_key
        self._init_args['action_dim'] = action_dim; self._init_args['input_key'] = input_key; self._init_args['output_key'] = output_key
        self.provides = self.provides.union({output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = list(self.provides)

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.input_key not in data: return {}
        policy_features = data[self.input_key]
        if not isinstance(policy_features, torch.Tensor): return {}
        action_tanh = torch.tanh(policy_features)
        if torch.isnan(action_tanh).any(): action_tanh = torch.zeros_like(action_tanh)
        update_dict = {self.output_key: action_tanh}; return update_dict

    def to_string(self) -> str:
        base = super().to_string().replace("Component","PolicyDecision")[:-1]
        info = f", Dim={self.action_dim}, In='{self.input_key}', Out='{self.output_key}'"
        return base + info + ")"


class ValueEstimationComponent(EquationComponent):
    """مكون لضمان مخرج قيمة واحد."""
    def __init__(self, component_id: str,
                 tags: Optional[Set[str]] = None, required_context: Optional[Set[str]] = None,
                 provides: Optional[Set[str]] = None,
                 input_key: str = 'value_features', output_key: str = 'q_value'):
        tags = tags if tags is not None else {'critic', 'evaluator'}
        req_ctx = required_context if required_context is not None else {'request_q_value'}
        prov = provides if provides is not None else {output_key}
        super().__init__(component_id, tags, req_ctx, prov)
        self.input_key = input_key; self.output_key = output_key
        self._init_args['input_key'] = input_key; self._init_args['output_key'] = output_key
        self.provides = self.provides.union({output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = list(self.provides)

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.input_key not in data: return {}
        value_features = data[self.input_key]
        if not isinstance(value_features, torch.Tensor): return {}
        final_value: torch.Tensor
        if value_features.shape[-1] != 1: final_value = value_features.mean(dim=-1, keepdim=True)
        else: final_value = value_features
        if torch.isnan(final_value).any(): final_value = torch.zeros_like(final_value)
        update_dict = {self.output_key: final_value}; return update_dict

    def to_string(self) -> str:
        base = super().to_string().replace("Component","ValueEstimation")[:-1]
        info = f", In='{self.input_key}', Out='{self.output_key}'"; return base + info + ")"


class RiskProgressEvaluator(EquationComponent):
    """مكون لتقدير المخاطر والتقدم."""
    def __init__(self, component_id: str, input_dim: int, hidden_dim: int = 64,
                 tags: Optional[Set[str]] = None, required_context: Optional[Set[str]] = None,
                 provides: Optional[Set[str]] = None, input_key: str = 'features',
                 risk_output_key: str = 'estimated_risk', progress_output_key: str = 'estimated_progress'):
        tags = tags if tags is not None else {'expert', 'evaluator'}
        req_ctx = required_context if required_context is not None else {'request_evaluation', 'task:rl'}
        prov = provides if provides is not None else {risk_output_key, progress_output_key}
        super().__init__(component_id=component_id, tags=tags, required_context=req_ctx, provides=prov)
        if not isinstance(input_dim, int) or input_dim <= 0: raise ValueError("RPEval: input_dim err.")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0: raise ValueError("RPEval: hidden_dim err.")
        self.input_key = input_key; self.risk_output_key = risk_output_key; self.progress_output_key = progress_output_key
        self.risk_network = nn.Sequential( nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Softplus() )
        self._init_weights(self.risk_network)
        self.progress_network = nn.Sequential( nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1) )
        self._init_weights(self.progress_network)
        self._init_args['input_dim'] = input_dim; self._init_args['hidden_dim'] = hidden_dim
        self._init_args['input_key'] = input_key; self._init_args['risk_output_key'] = risk_output_key; self._init_args['progress_output_key'] = progress_output_key
        self.provides = self.provides.union({risk_output_key, progress_output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = list(self.provides)

    def _init_weights(self, module_seq: nn.Sequential):
        for layer in module_seq:
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'));
            if layer.bias is not None: nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm): nn.init.ones_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.input_key not in data: return {}
        input_features = data[self.input_key]
        if not isinstance(input_features, torch.Tensor): return {}
        expected_dim = self.risk_network[0].in_features
        if input_features.dim() < 2 or input_features.shape[-1] != expected_dim: return {}
        output_updates: Dict[str, torch.Tensor] = {}
        try:
            est_risk = self.risk_network(input_features)
            if torch.isnan(est_risk).any(): est_risk = torch.zeros_like(est_risk)
            output_updates[self.risk_output_key] = est_risk
        except Exception as e_risk: logger.error(f"Risk err '{self.component_id}': {e_risk}", exc_info=False) # Use main logger
        try:
            est_prog = self.progress_network(input_features)
            if torch.isnan(est_prog).any(): est_prog = torch.zeros_like(est_prog)
            output_updates[self.progress_output_key] = est_prog
        except Exception as e_prog: logger.error(f"Prog err '{self.component_id}': {e_prog}", exc_info=False) # Use main logger
        return output_updates

    def evolve(self, engine: EvolutionEngine, reward: float, step: int) -> bool: return False
    def to_string(self) -> str:
        base = super().to_string().replace("Component","RiskProgressEval")[:-1]
        info = f", In='{self.input_key}', Risk='{self.risk_output_key}', Prog='{self.progress_output_key}'"
        return base + info + ")"
    def get_complexity(self) -> float:
        risk_p = sum(p.numel() for p in self.risk_network.parameters()); prog_p = sum(p.numel() for p in self.progress_network.parameters())
        return (risk_p + prog_p) / 10000.0


# --- 7. المعادلة الموحدة ---
class UnifiedAdaptableEquation(nn.Module):
    """المعادلة الموحدة."""
    def __init__(self, equation_id: str):
        super().__init__()
        self.equation_id = equation_id
        self.components = nn.ModuleDict()
        self._execution_order: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}[{equation_id}]")
        self.logger.setLevel(logging.WARNING)

    def add_component(self, component: EquationComponent, execute_after: Optional[str] = None):
        component_id = component.component_id
        if component_id in self.components: raise ValueError(f"Comp id '{component_id}' exists.")
        self.components[component_id] = component
        if execute_after is None or not self._execution_order: self._execution_order.insert(0, component_id)
        else:
            try: idx = self._execution_order.index(execute_after); self._execution_order.insert(idx + 1, component_id)
            except ValueError: self.logger.warning(f"Exec after ID '{execute_after}' not found. Append '{component_id}'."); self._execution_order.append(component_id)

    def forward(self, initial_data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        current_data = initial_data.copy()
        active_ids: Set[str] = set()
        for cid, c in self.components.items():
            if c.is_active(context): active_ids.add(cid)
        if not active_ids: return current_data

        exec_queue = [cid for cid in self._execution_order if cid in active_ids]
        pending = set(exec_queue); executed: Set[str] = set()
        max_p = len(self.components) + 2; current_p = 0

        while current_p < max_p:
            if not pending: break
            progress = False; next_q: List[str] = []; current_q = exec_queue[:]
            exec_queue = []

            comp_q_idx = 0
            while comp_q_idx < len(current_q):
                 cid = current_q[comp_q_idx]
                 if cid in pending:
                     comp = self.components[cid]
                     in_key = getattr(comp, 'input_key', None)
                     ready = (in_key is None or in_key in current_data)
                     if ready:
                         try:
                             update = comp(current_data, context)
                             if not isinstance(update, dict): update = {}
                             current_data.update(update)
                             executed.add(cid); pending.remove(cid); progress = True
                             for key_upd, tensor_upd in update.items():
                                 if isinstance(tensor_upd, torch.Tensor):
                                     if torch.isnan(tensor_upd).any() or torch.isinf(tensor_upd).any():
                                         # self.logger.warning(f"NaN/Inf key '{key_upd}' comp '{cid}'. Clamp.")
                                         current_data[key_upd] = torch.nan_to_num(tensor_upd, nan=0.0, posinf=1.0, neginf=-1.0)
                         except Exception as e_fwd:
                             self.logger.error(f"Error comp '{cid}': {e_fwd}", exc_info=False)
                             if cid in pending: pending.remove(cid)
                     else: next_q.append(cid)
                 comp_q_idx += 1

            exec_queue = next_q
            if not progress and pending:
                 self.logger.warning(f"Exec stalled pass {current_p+1}. Pending:{pending}")
                 break
            current_p += 1

        remaining = active_ids - executed
        if remaining: self.logger.warning(f"Not all active executed. Remaining: {remaining}")
        return current_data

    def evolve(self, evolution_engine: EvolutionEngine, reward: float, step: int) -> List[Tuple[str, Set[str]]]:
        changed_info: List[Tuple[str, Set[str]]] = []
        for cid, comp in self.components.items():
            if hasattr(comp, 'evolve') and callable(comp.evolve):
                try:
                    changed = comp.evolve(evolution_engine, reward, step)
                    if changed: tags = getattr(comp, 'tags', set()); changed_info.append((cid, tags))
                except Exception as e_evo: self.logger.warning(f"Error evolving comp '{cid}': {e_evo}", exc_info=False)
        return changed_info

    def get_total_complexity(self) -> float:
        total: float = 0.0
        for component in self.components.values(): total = total + component.get_complexity()
        return total

    def to_string(self) -> str:
        lines: List[str] = []; eq_id = self.equation_id; total_comp = self.get_total_complexity()
        lines.append(f"UnifiedAdaptableEquation(id={eq_id}, TotalComplexity={total_comp:.2f})")
        lines.append(" Execution Order & Components:")
        details = {cid: comp.to_string() for cid, comp in self.components.items()}
        i_order = 0
        while i_order < len(self._execution_order):
            cid_ord = self._execution_order[i_order]
            detail_str = details.get(cid_ord, f"ERROR: ID '{cid_ord}' not found")
            lines.append(f"  {i_order+1}. {detail_str}")
            i_order += 1
        ordered_set = set(self._execution_order)
        unordered: List[str] = []
        for cid_unord, comp_unord in self.components.items():
             if cid_unord not in ordered_set: unordered.append(f"  - {comp_unord.to_string()}")
        if unordered: lines.append(" Unordered Components:"); lines.extend(unordered)
        return "\n".join(lines)

    def get_parameters(self): return self.parameters()

    def get_tagged_parameters(self, required_tags: Set[str]) -> List[nn.Parameter]:
        tagged_list: List[nn.Parameter] = []; processed_ids: Set[int] = set()
        for component in self.components.values():
             comp_tags = getattr(component, 'tags', set())
             needs_inclusion = 'shared' in comp_tags or not required_tags.isdisjoint(comp_tags)
             if needs_inclusion:
                  try:
                       component_params_list = list(component.parameters())
                       for param_item in component_params_list:
                            param_mem_id = id(param_item)
                            if param_mem_id not in processed_ids: tagged_list.append(param_item); processed_ids.add(param_mem_id)
                  except Exception as e: comp_id_str = getattr(component,'component_id','Unknown'); warnings.warn(f"Could not get params comp '{comp_id_str}': {e}", RuntimeWarning)
        return tagged_list


# --- 8. Core Component: Unified Learning Agent ---
class UnifiedLearningAgent(nn.Module):
    """وكيل تعلم موحد (الإصدار 1.1.1)."""
    def __init__(self,
                 state_dim: int, action_dim: int, action_bounds: Tuple[float, float],
                 unified_equation: UnifiedAdaptableEquation, evolution_engine: EvolutionEngine,
                 gamma: float = 0.99, tau: float = 0.005, actor_lr: float = 1e-4, critic_lr: float = 3e-4,
                 buffer_capacity: int = int(1e6), batch_size: int = 128,
                 exploration_noise_std: float = 0.3, noise_decay_rate: float = 0.9999, min_exploration_noise: float = 0.1,
                 weight_decay: float = 1e-5, grad_clip_norm: Optional[float] = 1.0, grad_clip_value: Optional[float] = None,
                 risk_penalty_weight: float = 0.05, progress_bonus_weight: float = 0.05):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"UnifiedLearningAgent(v1.1.1) using device: {self.device}")
        self.state_dim = state_dim; self.action_dim = action_dim; self.batch_size = batch_size
        self.gamma = gamma; self.tau = tau
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm is not None and grad_clip_norm > 0 else None
        self.grad_clip_value = grad_clip_value if grad_clip_value is not None and grad_clip_value > 0 else None
        if action_bounds is None or len(action_bounds) != 2: raise ValueError("action_bounds needed.")
        self.action_low = float(action_bounds[0]); self.action_high = float(action_bounds[1])
        if self.action_low >= self.action_high: raise ValueError("action low < high needed.")
        action_scale_t = torch.tensor((self.action_high - self.action_low) / 2.0, dtype=torch.float32)
        action_bias_t = torch.tensor((self.action_high + self.action_low) / 2.0, dtype=torch.float32)
        if (action_scale_t <= 1e-6).any(): warnings.warn("Action scale near zero."); action_scale_t.clamp_(min=1e-6)
        self.register_buffer('action_scale', action_scale_t); self.register_buffer('action_bias', action_bias_t)
        if not isinstance(unified_equation, nn.Module): raise TypeError("unified_equation must be nn.Module.")
        self.unified_eq = unified_equation.to(self.device)
        if hasattr(self.unified_eq, '_execution_order'): self._execution_order = self.unified_eq._execution_order[:]
        else: warnings.warn("Unified eq missing '_execution_order'."); self._execution_order = []
        self.target_unified_eq: Optional[UnifiedAdaptableEquation] = None; self._initialize_target_network()
        logger.info("\n--- Initial Unified Adaptable Equation ---"); logger.info(self.unified_eq.to_string()); logger.info("-" * 50 + "\n")
        if not isinstance(evolution_engine, EvolutionEngine): raise TypeError("evolution_engine needed.")
        self.evolution_engine = evolution_engine
        self.actor_lr = actor_lr; self.critic_lr = critic_lr; self.weight_decay = weight_decay
        self.actor_optimizer = self._create_optimizer(actor_lr, weight_decay, {'actor', 'shared'})
        self.critic_optimizer = self._create_optimizer(critic_lr, weight_decay, {'critic', 'shared', 'expert'})
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.exploration_noise_std = exploration_noise_std; self.min_exploration_noise = min_exploration_noise
        self.noise_decay_rate = noise_decay_rate; self.current_noise_std = exploration_noise_std
        self.risk_penalty_weight = risk_penalty_weight; self.progress_bonus_weight = progress_bonus_weight
        self.total_updates = 0; self.total_evolutions = 0

    def _initialize_target_network(self):
        try:
            self.target_unified_eq = copy.deepcopy(self.unified_eq)
            if self.target_unified_eq is None: raise RuntimeError("Deepcopy failed.")
            target_params = list(self.target_unified_eq.parameters())
            for param in target_params: param.requires_grad = False
        except Exception as e: logger.error(f"CRITICAL ERROR init target net: {e}"); raise RuntimeError("Failed init target net") from e

    def _get_tagged_parameters(self, required_tags: Set[str]):
        return self.unified_eq.get_tagged_parameters(required_tags)

    def _create_optimizer(self, lr: float, wd: float, tags: Set[str]) -> Optional[optim.Optimizer]:
        params = self._get_tagged_parameters(tags)
        if params:
             num_params = sum(p.numel() for p in params)
             logger.info(f"Creating AdamW tags {tags} ({len(params)} tensors, {num_params:,} params) LR={lr:.1e}, WD={wd:.1e}")
             optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
             return optimizer
        else: logger.warning(f"No parameters for tags {tags}. Optimizer not created."); return None

    def _set_requires_grad(self, tags: Set[str], requires_grad: bool):
        params = self._get_tagged_parameters(tags)
        for param in params: param.requires_grad = requires_grad

    def _update_target_network(self):
        if self.target_unified_eq is None: return
        with torch.no_grad():
            try:
                main_params = list(self.unified_eq.parameters()); target_params = list(self.target_unified_eq.parameters())
                if len(main_params) != len(target_params): logger.warning("Target mismatch. Syncing."); self._sync_target_network(); return
                i_param = 0
                while i_param < len(target_params):
                    target_p = target_params[i_param]; main_p = main_params[i_param]
                    target_p.data.mul_(1.0 - self.tau); target_p.data.add_(self.tau * main_p.data)
                    i_param += 1
            except Exception as e: logger.warning(f"Soft target update error: {e}. Syncing."); self._sync_target_network()

    def _sync_target_network(self):
        try:
            if self.target_unified_eq is None: self._initialize_target_network()
            if self.target_unified_eq is not None:
                 self.target_unified_eq.load_state_dict(self.unified_eq.state_dict())
                 target_params = list(self.target_unified_eq.parameters())
                 for param in target_params: param.requires_grad = False
            else: logger.error("Failed to initialize or sync target network.")
        except RuntimeError as e: logger.warning(f"Failed sync target: {e}.")
        except AttributeError: logger.warning("Target net missing or not init for sync.")
        except Exception as e_sync: logger.error(f"Unexpected sync error: {e_sync}", exc_info=True)


    def get_action(self, state, explore=True) -> np.ndarray:
        try:
            state_np = np.asarray(state, dtype=np.float32);
            if np.isnan(state_np).any(): state_np = np.zeros_like(state_np)
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        except Exception as e: return np.zeros(self.action_dim, dtype=np.float32)

        self.unified_eq.eval(); final_action = np.zeros(self.action_dim, dtype=np.float32)
        try:
            with torch.no_grad():
                context = {'task': 'rl', 'request_action': True, 'mode': 'eval' if not explore else 'explore'}
                initial_data = {'state': state_tensor, 'features': state_tensor}
                output_data = self.unified_eq(initial_data, context)
                if 'action_raw' not in output_data: warnings.warn("Key 'action_raw' missing. Zero action.", RuntimeWarning)
                else:
                    action_pre_tanh = output_data['action_raw']
                    if torch.isnan(action_pre_tanh).any(): action_pre_tanh = torch.zeros_like(action_pre_tanh)
                    action_tanh = torch.tanh(action_pre_tanh)
                    if explore:
                        noise = torch.randn(self.action_dim, device=self.device) * self.current_noise_std
                        action_noisy = action_tanh + noise
                        action_final_tanh = torch.clamp(action_noisy, -1.0, 1.0)
                    else: action_final_tanh = action_tanh
                    action_scaled = action_final_tanh * self.action_scale + self.action_bias
                    action_clipped = torch.clamp(action_scaled, self.action_low, self.action_high)
                    if torch.isnan(action_clipped).any(): action_clipped = torch.full_like(action_clipped, (self.action_high + self.action_low) / 2.0)
                    final_action = action_clipped.squeeze(0).cpu().numpy()
        except Exception as e: warnings.warn(f"Exception get_action: {e}. Zero.", RuntimeWarning); traceback.print_exc()
        finally: self.unified_eq.train()
        return final_action.reshape(self.action_dim)

    def update(self, step):
        """تنفيذ خطوة تحديث."""
        if len(self.replay_buffer) < self.batch_size: return None, None, 0.0
        self.total_updates += 1
        sample = self.replay_buffer.sample(self.batch_size)
        if sample is None: return None, None, 0.0
        try: states, actions, rewards, next_states, dones = [t.to(self.device) for t in sample]
        except Exception as e: warnings.warn(f"Failed move batch device: {e}.", RuntimeWarning); return None, None, 0.0
        avg_reward = rewards.mean().item();
        if math.isnan(avg_reward) or math.isinf(avg_reward): avg_reward = 0.0
        q_loss_val, policy_loss_val = None, None; q_target = None

        # --- 1. Critic Update ---
        try:
            # أ. حساب الهدف Q
            with torch.no_grad():
                if self.target_unified_eq is None: raise RuntimeError("Tgt net None.")
                self.target_unified_eq.eval()
                tgt_act_ctx = {'task': 'rl', 'request_action': True, 'mode': 'target'}
                tgt_act_data = {'state': next_states, 'features': next_states}
                tgt_policy_out = self.target_unified_eq(tgt_act_data, tgt_act_ctx)
                if 'action_raw' not in tgt_policy_out: raise ValueError("Tgt missing 'action_raw'")
                next_actions_tanh = torch.tanh(tgt_policy_out['action_raw'])
                next_actions_scaled = torch.clamp(next_actions_tanh * self.action_scale + self.action_bias, self.action_low, self.action_high)
                if torch.isnan(next_actions_scaled).any(): next_actions_scaled = torch.zeros_like(next_actions_scaled)
                tgt_q_ctx = {'task': 'rl', 'request_q_value': True, 'mode': 'target'}
                tgt_critic_in = torch.cat([next_states, next_actions_scaled], dim=1)
                tgt_q_data = {'state': next_states, 'action': next_actions_scaled, 'critic_input': tgt_critic_in, 'features': next_states}
                tgt_critic_out = self.target_unified_eq(tgt_q_data, tgt_q_ctx)
                if 'q_value' not in tgt_critic_out: raise ValueError("Tgt missing 'q_value'")
                target_q_values = tgt_critic_out['q_value']
                if torch.isnan(target_q_values).any(): target_q_values = torch.zeros_like(target_q_values)
                q_target = rewards + self.gamma * target_q_values * (1.0 - dones)
                if torch.isnan(q_target).any(): raise ValueError("NaN/Inf q_target")
                self.target_unified_eq.train()

            # ب. تحديث المقيّم الرئيسي
            if self.critic_optimizer and q_target is not None:
                self.unified_eq.train(); self._set_requires_grad({'actor'}, False); self._set_requires_grad({'critic', 'shared', 'expert'}, True)
                curr_q_ctx = {'task': 'rl', 'request_q_value': True, 'mode': 'train_critic'}
                curr_critic_in = torch.cat([states, actions], dim=1)
                curr_q_data = {'state': states, 'action': actions, 'critic_input': curr_critic_in, 'features': states}
                critic_output = self.unified_eq(curr_q_data, curr_q_ctx)
                if 'q_value' not in critic_output: raise ValueError("Main missing 'q_value' critic")
                current_q_values = critic_output['q_value']
                if torch.isnan(current_q_values).any(): raise ValueError("NaN/Inf current_q")
                q_loss = F.mse_loss(current_q_values, q_target.detach())
                if torch.isnan(q_loss): raise ValueError(f"NaN Q-loss")
                self.critic_optimizer.zero_grad(); q_loss.backward()
                critic_params = self._get_tagged_parameters({'critic', 'shared', 'expert'})
                if critic_params:
                    if self.grad_clip_norm: torch.nn.utils.clip_grad_norm_(critic_params, self.grad_clip_norm)
                    if self.grad_clip_value: torch.nn.utils.clip_grad_value_(critic_params, self.grad_clip_value)
                self.critic_optimizer.step(); q_loss_val = q_loss.item()
        except Exception as e_critic:
            self.logger.warning(f"Exception Critic update step {step}: {e_critic}", exc_info=False)
            return q_loss_val, policy_loss_val, avg_reward
        finally: self._set_requires_grad({'actor'}, True)

        # --- 2. Actor Update ---
        if self.actor_optimizer:
            try:
                # تجميد المقيّم والخبير
                self._set_requires_grad({'critic', 'expert'}, False); self._set_requires_grad({'actor', 'shared'}, True)
                self.unified_eq.train() # Ensure actor is in train mode for grad

                # أ. حساب الإجراءات المقترحة
                policy_act_ctx = {'task': 'rl', 'request_action': True, 'mode': 'train_actor'}
                policy_act_data = {'state': states, 'features': states}
                policy_output = self.unified_eq(policy_act_data, policy_act_ctx)
                if 'action_raw' not in policy_output: raise ValueError("Main missing 'action_raw' policy")
                actions_pred_tanh = torch.tanh(policy_output['action_raw'])
                actions_pred_scaled = actions_pred_tanh * self.action_scale + self.action_bias

                # ب. تقييم الإجراءات بواسطة المقيّم (المجمد)
                policy_q_ctx = {'task': 'rl', 'request_q_value': True, 'mode': 'eval_policy'}
                policy_critic_in = torch.cat([states, actions_pred_scaled], dim=1)
                policy_q_data = {'state': states, 'action': actions_pred_scaled, 'critic_input': policy_critic_in, 'features': states}
                policy_critic_output = self.unified_eq(policy_q_data, policy_q_ctx)
                if 'q_value' not in policy_critic_output: raise ValueError("Main missing 'q_value' policy eval")
                policy_q_values = policy_critic_output['q_value']

                # ج. حساب تقديرات المخاطر/التقدم (لا يتطلب تدرج هنا)
                eval_ctx = {'task': 'rl', 'request_evaluation': True, 'mode': 'eval_policy'}
                eval_data = {'state': states, 'features': states}
                risk_progress_output = self.unified_eq(eval_data, eval_ctx)
                estimated_risk = risk_progress_output.get('estimated_risk', torch.zeros_like(policy_q_values)).detach()
                estimated_progress = risk_progress_output.get('estimated_progress', torch.zeros_like(policy_q_values)).detach()

                # د. حساب خسارة السياسة المعدلة
                q_term_loss = -policy_q_values.mean()
                risk_term_loss = self.risk_penalty_weight * estimated_risk.mean()
                progress_term_loss = -self.progress_bonus_weight * estimated_progress.mean()
                policy_loss = q_term_loss + risk_term_loss + progress_term_loss

                if torch.isnan(policy_loss): raise ValueError(f"NaN Policy loss")

                # هـ. تحديث السياسة
                self.actor_optimizer.zero_grad(); policy_loss.backward()
                actor_params = self._get_tagged_parameters({'actor', 'shared'})
                if actor_params:
                    if self.grad_clip_norm: torch.nn.utils.clip_grad_norm_(actor_params, self.grad_clip_norm)
                    if self.grad_clip_value: torch.nn.utils.clip_grad_value_(actor_params, self.grad_clip_value)
                self.actor_optimizer.step(); policy_loss_val = policy_loss.item()

            except Exception as e_actor:
                self.logger.warning(f"Exception Actor update step {step}: {e_actor}", exc_info=False)
            finally:
                # إعادة تفعيل تدرجات المقيّم والخبير
                self._set_requires_grad({'critic', 'expert'}, True)

        # --- 3. تحديث الهدف ---
        try: self._update_target_network()
        except Exception as e_target: self.logger.warning(f"Exception target update step {step}: {e_target}")

        # --- 4. التطور ---
        try:
            changed_info = self.unified_eq.evolve(self.evolution_engine, avg_reward, self.total_updates)
            if changed_info:
                 self.total_evolutions += 1; logger.info(f"Unified eq structure changed. Reinit optim & sync target. Changed: {changed_info}") # Use main logger
                 actor_reset = any('shared' in tags or 'actor' in tags for _, tags in changed_info)
                 critic_reset = any('shared' in tags or 'critic' in tags or 'expert' in tags for _, tags in changed_info)
                 wd = self.weight_decay
                 if actor_reset and self.actor_optimizer: self.actor_optimizer = self._create_optimizer(self.actor_lr, wd, {'actor', 'shared'})
                 if critic_reset and self.critic_optimizer: self.critic_optimizer = self._create_optimizer(self.critic_lr, wd, {'critic', 'shared', 'expert'})
                 self._sync_target_network()
        except Exception as e_evolve: self.logger.warning(f"Exception Unified Eq evolution step {self.total_updates}: {e_evolve}", exc_info=True)

        # --- 5. تقليل الضوضاء ---
        self.current_noise_std = max(self.min_exploration_noise, self.current_noise_std * self.noise_decay_rate)

        return q_loss_val, policy_loss_val, avg_reward

    def evaluate(self, env, episodes=5):
        """تقييم أداء الوكيل."""
        total_rewards_eval = []
        max_episode_steps = getattr(env.spec, 'max_episode_steps', 1000) if hasattr(env, 'spec') else 1000
        ep_idx_eval = 0
        while ep_idx_eval < episodes:
            ep_reward_eval = 0.0; steps_eval = 0; state_eval, info_eval = None, None
            try: eval_seed = random.randint(0, 100000); state_eval, info_eval = env.reset(seed=eval_seed)
            except Exception as e: self.logger.warning(f"Eval reset failed: {e}. Skip ep."); ep_idx_eval+=1; continue
            state_eval = np.asarray(state_eval, dtype=np.float32); terminated_eval, truncated_eval = False, False
            while not (terminated_eval or truncated_eval):
                if steps_eval >= max_episode_steps: truncated_eval = True; break
                try:
                    action_eval = self.get_action(state_eval, explore=False)
                    if action_eval is None: action_eval = np.zeros(self.action_dim, dtype=np.float32)
                    elif np.isnan(action_eval).any(): action_eval = np.zeros(self.action_dim, dtype=np.float32)
                    elif np.isinf(action_eval).any(): action_eval = np.zeros(self.action_dim, dtype=np.float32)
                    next_state_eval, reward_eval, terminated_eval, truncated_eval, info_eval = env.step(action_eval)
                    if math.isnan(reward_eval): reward_eval = 0.0
                    if math.isinf(reward_eval): reward_eval = 0.0
                    ep_reward_eval += reward_eval; state_eval = np.asarray(next_state_eval, dtype=np.float32); steps_eval += 1
                except (gym.error.Error, Exception) as e: self.logger.warning(f"Exception eval step ep {ep_idx_eval+1}: {e}. End."); terminated_eval = True; break
            total_rewards_eval.append(ep_reward_eval)
            ep_idx_eval += 1

        if not total_rewards_eval: return -np.inf
        mean_reward_final_eval = np.mean(total_rewards_eval)
        return mean_reward_final_eval

    def save_model(self, filename="unified_agent_checkpoint_v111.pt"):
        """حفظ حالة الوكيل."""
        self.unified_eq.to('cpu')
        try:
            component_configs_save = {}
            for comp_id, comp in self.unified_eq.components.items():
                 init_args_comp = getattr(comp, '_init_args', {}).copy()
                 if hasattr(comp, 'equation') and isinstance(comp.equation, EvolvingEquation):
                      init_args_comp['equation_state_dict'] = comp.equation.state_dict()
                      if 'equation_config' in init_args_comp and isinstance(init_args_comp['equation_config'], dict):
                          init_args_comp['equation_config']['init_complexity'] = comp.equation.complexity
                 else: init_args_comp['equation_state_dict'] = None
                 act_func = getattr(comp, 'activation', None)
                 init_args_comp['activation_name'] = act_func.__name__ if act_func and hasattr(act_func, '__name__') else 'None'
                 init_args_comp.pop('equation', None); init_args_comp.pop('activation', None)
                 component_configs_save[comp_id] = {'class_name': type(comp).__name__, 'init_args': init_args_comp, 'state_dict': comp.state_dict()}

            equation_structure_save = {'equation_id': self.unified_eq.equation_id, 'execution_order': self._execution_order, 'component_configs': component_configs_save}
            save_data_dict = {
                'metadata': {'description': 'Unified Learning Agent (v1.1.1)', 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")},
                'agent_config': {'state_dim': self.state_dim, 'action_dim': self.action_dim, 'action_bounds': (self.action_low, self.action_high),'gamma': self.gamma, 'tau': self.tau, 'batch_size': self.batch_size,'actor_lr': self.actor_lr, 'critic_lr': self.critic_lr, 'weight_decay': self.weight_decay, 'risk_penalty': self.risk_penalty_weight, 'progress_bonus': self.progress_bonus_weight},
                'unified_equation_structure': equation_structure_save,
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict() if self.actor_optimizer else None,
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict() if self.critic_optimizer else None,
                'training_state': {'total_updates': self.total_updates, 'current_noise_std': self.current_noise_std, 'total_evolutions': self.total_evolutions},
                'evolution_engine_state': {'performance_history': list(self.evolution_engine.performance_history), 'term_change_cooldown': self.evolution_engine.term_change_cooldown,}
            }
            save_dir_path = os.path.dirname(filename)
            if save_dir_path: os.makedirs(save_dir_path, exist_ok=True)
            torch.save(save_data_dict, filename)
            self.logger.info(f"Unified Agent state saved to '{filename}'")
        except Exception as e: self.logger.error(f"Failed save state: {e}", exc_info=True)
        finally: self.unified_eq.to(self.device)

    def load_model(self, filename="unified_agent_checkpoint_v111.pt"):
        """تحميل حالة الوكيل."""
        try:
            self.logger.info(f"Attempting load Unified agent state from '{filename}'...")
            if not os.path.exists(filename): self.logger.warning(f"Checkpoint not found: '{filename}'"); return False
            checkpoint = torch.load(filename, map_location=self.device)

            required_keys_list = ['agent_config', 'unified_equation_structure', 'actor_optimizer_state_dict', 'critic_optimizer_state_dict', 'training_state']
            if not all(k in checkpoint for k in required_keys_list): missing = [k for k in required_keys_list if k not in checkpoint]; self.logger.error(f"Checkpoint incomplete. Missing: {missing}"); return False

            cfg_load = checkpoint['agent_config']; train_state_load = checkpoint['training_state']; eq_struct_load = checkpoint['unified_equation_structure']

            if cfg_load.get('state_dim')!=self.state_dim or cfg_load.get('action_dim')!=self.action_dim: self.logger.error("Dim mismatch."); return False
            saved_bounds_load = cfg_load.get('action_bounds')
            if saved_bounds_load and (abs(saved_bounds_load[0]-self.action_low)>1e-6 or abs(saved_bounds_load[1]-self.action_high)>1e-6): self.logger.warning("Action bounds mismatch.")

            self.gamma = cfg_load.get('gamma', self.gamma); self.tau = cfg_load.get('tau', self.tau); self.batch_size = cfg_load.get('batch_size', self.batch_size)
            actor_lr_load = cfg_load.get('actor_lr', self.actor_lr); critic_lr_load = cfg_load.get('critic_lr', self.critic_lr); wd_load = cfg_load.get('weight_decay', self.weight_decay)
            self.risk_penalty_weight = cfg_load.get('risk_penalty', self.risk_penalty_weight)
            self.progress_bonus_weight = cfg_load.get('progress_bonus', self.progress_bonus_weight)

            self.logger.info("  Rebuilding Unified Equation from saved structure...")
            new_unified_eq = UnifiedAdaptableEquation(eq_struct_load.get('equation_id', 'loaded_eq'))
            activation_map = {'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'None': None, 'identity': lambda x: x, 'gelu': F.gelu, '<lambda>': None}
            component_classes_map = {'NumericTransformComponent': NumericTransformComponent, 'PolicyDecisionComponent': PolicyDecisionComponent, 'ValueEstimationComponent': ValueEstimationComponent, 'RiskProgressEvaluator': RiskProgressEvaluator}
            rebuilt_components_dict = {}; component_state_dicts_load = {}

            for comp_id_load, comp_config_load in eq_struct_load.get('component_configs', {}).items():
                comp_class_name_load = comp_config_load.get('class_name')
                if comp_class_name_load in component_classes_map:
                    CompClassRef = component_classes_map[comp_class_name_load]
                    init_args_load = comp_config_load.get('init_args', {}).copy()
                    comp_state_dict_load = comp_config_load.get('state_dict')
                    if comp_class_name_load == 'NumericTransformComponent':
                         eq_config_load = init_args_load.pop('equation_config', None)
                         eq_state_dict_load = init_args_load.pop('equation_state_dict', None)
                         activation_name_load = init_args_load.pop('activation_name', 'None')
                         init_args_load['activation'] = activation_map.get(activation_name_load)
                         if eq_config_load:
                              try:
                                  init_complexity_val = eq_config_load.get('init_complexity', 3)
                                  evol_eq_load = EvolvingEquation(**{**eq_config_load, 'init_complexity': init_complexity_val})
                                  if eq_state_dict_load is not None:
                                      if evol_eq_load.complexity != init_complexity_val: evol_eq_load = EvolvingEquation(**{**eq_config_load, 'init_complexity': init_complexity_val})
                                      evol_eq_load.load_state_dict(eq_state_dict_load)
                                  init_args_load['equation'] = evol_eq_load
                              except Exception as eq_rebuild_e: self.logger.warning(f"Failed rebuild EvolvingEq '{comp_id_load}': {eq_rebuild_e}"); continue
                         else: self.logger.warning(f"Eq config missing '{comp_id_load}'"); continue
                    if 'tags' in init_args_load and isinstance(init_args_load['tags'], list): init_args_load['tags'] = set(init_args_load['tags'])
                    if 'required_context' in init_args_load and isinstance(init_args_load['required_context'], list): init_args_load['required_context'] = set(init_args_load['required_context'])
                    if 'provides' in init_args_load and isinstance(init_args_load['provides'], list): init_args_load['provides'] = set(init_args_load['provides'])
                    try:
                        expected_args_list = list(inspect.signature(CompClassRef.__init__).parameters.keys())
                        if 'self' in expected_args_list: expected_args_list.remove('self')
                        filtered_init_args = {k: v for k, v in init_args_load.items() if k in expected_args_list}
                        component_instance = CompClassRef(**filtered_init_args)
                        rebuilt_components_dict[comp_id_load] = component_instance
                        component_state_dicts_load[comp_id_load] = comp_state_dict_load
                    except Exception as build_e: self.logger.warning(f"Failed init comp '{comp_id_load}' class '{comp_class_name_load}': {build_e}")
                else: self.logger.warning(f"Unknown component class name '{comp_class_name_load}' for id '{comp_id_load}'. Skipping.")

            saved_exec_order_list = eq_struct_load.get('execution_order', [])
            new_unified_eq._execution_order = saved_exec_order_list
            for comp_id_exec in saved_exec_order_list:
                if comp_id_exec in rebuilt_components_dict: new_unified_eq.components[comp_id_exec] = rebuilt_components_dict[comp_id_exec]
                else: self.logger.warning(f"Comp '{comp_id_exec}' from saved exec order missing.")
            for comp_id_state, state_dict_comp in component_state_dicts_load.items():
                if comp_id_state in new_unified_eq.components and state_dict_comp:
                    try: load_info = new_unified_eq.components[comp_id_state].load_state_dict(state_dict_comp, strict=False);
                    except Exception as load_comp_e: self.logger.warning(f"Failed load state comp '{comp_id_state}': {load_comp_e}")

            self.unified_eq = new_unified_eq.to(self.device); self._execution_order = self.unified_eq._execution_order[:]

            actor_optim_state = checkpoint.get('actor_optimizer_state_dict'); critic_optim_state = checkpoint.get('critic_optimizer_state_dict')
            self.actor_optimizer = self._create_optimizer(actor_lr_load, wd_load, {'actor', 'shared'})
            if self.actor_optimizer and actor_optim_state:
                try: self.actor_optimizer.load_state_dict(actor_optim_state)
                except Exception as ao_load_e: self.logger.warning(f"Failed load actor optim state: {ao_load_e}. Reset."); self.actor_optimizer = self._create_optimizer(actor_lr_load, wd_load, {'actor', 'shared'})
            self.critic_optimizer = self._create_optimizer(critic_lr_load, wd_load, {'critic', 'shared', 'expert'})
            if self.critic_optimizer and critic_optim_state:
                 try: self.critic_optimizer.load_state_dict(critic_optim_state)
                 except Exception as co_load_e: self.logger.warning(f"Failed load critic optim state: {co_load_e}. Reset."); self.critic_optimizer = self._create_optimizer(critic_lr_load, wd_load, {'critic', 'shared', 'expert'})

            self._sync_target_network()

            self.total_updates = train_state_load.get('total_updates', 0)
            self.current_noise_std = train_state_load.get('current_noise_std', self.exploration_noise_std)
            self.total_evolutions = train_state_load.get('total_evolutions', 0)
            if 'evolution_engine_state' in checkpoint:
                 evo_state_load = checkpoint['evolution_engine_state']
                 evo_hist_load = evo_state_load.get('performance_history', [])
                 self.evolution_engine.performance_history = deque(evo_hist_load, maxlen=self.evolution_engine.performance_history.maxlen)
                 self.evolution_engine.term_change_cooldown = evo_state_load.get('term_change_cooldown', 0)

            self.logger.info(f"Unified Agent state loaded successfully from '{filename}'.")
            self.logger.info(f"  Updates: {self.total_updates}, Evols: {self.total_evolutions}")
            return True

        except Exception as e: self.logger.error(f"Unexpected error loading agent state: {e}", exc_info=True); return False


# --- 9. Main Training Function ---
def train_unified_agent(env_name="Pendulum-v1", max_steps=100000, batch_size=128,
                        eval_frequency=5000, start_learning_steps=5000, update_frequency=1,
                        hidden_dims=[64, 64], eq_init_complexity=4, eq_complexity_limit=10,
                        actor_lr=1e-4, critic_lr=3e-4, tau=0.005, weight_decay=1e-5,
                        evolution_mutation_power=0.02, evolution_cooldown=60,
                        exploration_noise=0.3, noise_decay=0.9999, min_noise=0.1,
                        risk_penalty=0.05, progress_bonus=0.05,
                        save_best=True, save_periodically=True, save_interval=25000,
                        render_eval=False, eval_episodes=10, grad_clip_norm=1.0, grad_clip_value=None,
                        resume_from_checkpoint=None, results_dir="unified_agent_results_v111"):
    """الدالة الرئيسية لتدريب وكيل التعلم الموحد."""
    start_time = time.time()
    print("\n" + "="*60); print(f"=== Starting Unified Agent Training (v1.1.1) for {env_name} ==="); print(f"=== Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ==="); print("="*60)
    print(f"\n--- Hyperparameters ---"); print(f"  Env: {env_name}, Max Steps: {max_steps:,}"); print(f"  Unified Eq: Hidden={hidden_dims}, InitComp={eq_init_complexity}, Limit={eq_complexity_limit}"); print(f"  Batch: {batch_size}, StartLearn: {start_learning_steps:,}, UpdateFreq: {update_frequency}"); print(f"  Eval: Freq={eval_frequency:,}, Eps={eval_episodes}"); print(f"  Learning: ActorLR={actor_lr:.1e}, CriticLR={critic_lr:.1e}, Tau={tau:.3f}, WD={weight_decay:.1e}, ClipN={grad_clip_norm}, ClipV={grad_clip_value}"); print(f"  Evolution: Mut={evolution_mutation_power:.3f}, CD={evolution_cooldown}"); print(f"  Exploration: Start={exploration_noise:.2f}, Decay={noise_decay:.4f}, Min={min_noise:.2f}"); print(f"  Policy Loss Weights: RiskPenalty={risk_penalty:.3f}, ProgressBonus={progress_bonus:.3f}"); print(f"  Saving: Best={save_best}, Periodic={save_periodically} (Interval={save_interval:,})"); print(f"  Resume: {resume_from_checkpoint if resume_from_checkpoint else 'None'}"); print(f"  Results Dir: {results_dir}"); print("-" * 50 + "\n")
    # --- Initialize Environment ---
    try:
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_name), deque_size=50)
        eval_render_mode = "human" if render_eval else None; eval_env = None
        try: eval_env = gym.make(env_name, render_mode=eval_render_mode)
        except Exception:
            try: eval_env = gym.make(env_name)
            except Exception as e_fallback: print(f"CRITICAL ERROR: Create eval env failed: {e_fallback}"); return None, []
        env_seed = RANDOM_SEED; eval_env_seed = RANDOM_SEED + 1
        env.reset(seed=env_seed); env.action_space.seed(env_seed)
        eval_env.reset(seed=eval_env_seed); eval_env.action_space.seed(eval_env_seed)
        state_dim_env = env.observation_space.shape[0]; action_dim_env = env.action_space.shape[0]
        action_low_env = env.action_space.low; action_high_env = env.action_space.high
        if np.any(np.isinf(action_low_env)) or np.any(np.isinf(action_high_env)): action_bounds_env = (-1.0, 1.0)
        else: action_bounds_env = (float(action_low_env.min()), float(action_high_env.max()))
        logger.info(f"Env Details: State={state_dim_env}, Action={action_dim_env}, Bounds={action_bounds_env}")
    except Exception as e: print(f"CRITICAL ERROR initializing env: {e}"); return None, []

    # --- Build Unified Equation Structure ---
    try:
        logger.info("Building example RL Unified Equation structure...")
        unified_eq_agent = UnifiedAdaptableEquation(equation_id="rl_policy_value_net_v111")
        current_dim_build = state_dim_env; last_comp_id_build: Optional[str] = None; features_key_build = 'state'; shared_last_comp_id_build = last_comp_id_build
        # Shared Layers
        hidden_layer_idx = 0
        while hidden_layer_idx < len(hidden_dims):
             h_dim_build = hidden_dims[hidden_layer_idx]
             comp_id_build = f'shared_hidden_{hidden_layer_idx}'; eq_shared_build = EvolvingEquation(current_dim_build, eq_init_complexity, h_dim_build, eq_complexity_limit)
             comp_shared = NumericTransformComponent(comp_id_build, eq_shared_build, F.relu, {'shared'}, {'task:rl'}, {'features'}, features_key_build, 'features')
             unified_eq_agent.add_component(comp_shared, execute_after=shared_last_comp_id_build); current_dim_build = h_dim_build; shared_last_comp_id_build = comp_id_build; features_key_build = 'features'
             hidden_layer_idx += 1
        logger.info(f"  Shared layers added. Last shared output dim: {current_dim_build}")
        # Risk/Progress Evaluator
        evaluator_input_dim_build = current_dim_build
        eval_comp_id_build = 'risk_prog_eval'; risk_prog_evaluator_build = RiskProgressEvaluator(eval_comp_id_build, evaluator_input_dim_build, hidden_dim=max(32, evaluator_input_dim_build // 2), input_key='features', required_context={'request_evaluation'})
        unified_eq_agent.add_component(risk_prog_evaluator_build, execute_after=shared_last_comp_id_build)
        logger.info("RiskProgressEvaluator component added.")
        # Actor Path
        actor_output_eq_build = EvolvingEquation(current_dim_build, eq_init_complexity, action_dim_env, eq_complexity_limit)
        actor_output_comp_build = NumericTransformComponent('actor_output_layer', actor_output_eq_build, None, {'actor'}, {'request_action'}, {'policy_features'}, 'features', 'policy_features')
        unified_eq_agent.add_component(actor_output_comp_build, execute_after=shared_last_comp_id_build)
        policy_decision_comp_build = PolicyDecisionComponent('policy_decision', action_dim_env, {'actor'}, {'request_action'}, {'action_raw'}, 'policy_features', 'action_raw')
        unified_eq_agent.add_component(policy_decision_comp_build, execute_after='actor_output_layer')
        logger.info("Actor path added.")
        # Critic Path
        critic_input_dim_build = current_dim_build + action_dim_env # Features + Action
        critic_hidden_dim_build = hidden_dims[-1]
        critic_hidden_eq_build = EvolvingEquation(critic_input_dim_build, eq_init_complexity, critic_hidden_dim_build, eq_complexity_limit)
        # تأكد من أن input_key هنا يستقبل الحالة + الفعل المدمجين
        critic_hidden_comp_build = NumericTransformComponent('critic_hidden_layer', critic_hidden_eq_build, F.relu, {'critic','expert'}, {'request_q_value'}, {'value_features'}, 'critic_input', 'value_features')
        unified_eq_agent.add_component(critic_hidden_comp_build, execute_after=shared_last_comp_id_build) # يعتمد على الميزات المشتركة (التي ستدمج مع الفعل لاحقًا)
        critic_output_eq_build = EvolvingEquation(critic_hidden_dim_build, eq_init_complexity, 1, eq_complexity_limit)
        critic_output_comp_build = NumericTransformComponent('critic_output_layer', critic_output_eq_build, None, {'critic','expert'}, {'request_q_value'}, {'value_features_final'}, 'value_features', 'value_features_final')
        unified_eq_agent.add_component(critic_output_comp_build, execute_after='critic_hidden_layer')
        value_estimation_comp_build = ValueEstimationComponent('value_estimation', {'critic','expert'}, {'request_q_value'}, {'q_value'}, 'value_features_final', 'q_value')
        unified_eq_agent.add_component(value_estimation_comp_build, execute_after='critic_output_layer')
        logger.info("Critic path added.")
        logger.info("Unified Equation structure built successfully.")
    except Exception as e: print(f"CRITICAL ERROR building unified eq: {e}"); traceback.print_exc(); return None, []

    # --- Initialize Agent ---
    try:
        evo_engine = EvolutionEngine(mutation_power=evolution_mutation_power, cooldown_period=evolution_cooldown)
        agent = UnifiedLearningAgent(
            state_dim_env, action_dim_env, action_bounds_env, unified_equation=unified_eq_agent, evolution_engine=evo_engine,
            gamma=gamma, tau=tau, actor_lr=actor_lr, critic_lr=critic_lr,
            buffer_capacity=int(1e6), batch_size=batch_size,
            exploration_noise_std=exploration_noise, noise_decay_rate=noise_decay, min_exploration_noise=min_noise,
            weight_decay=weight_decay, grad_clip_norm=grad_clip_norm, grad_clip_value=grad_clip_value,
            risk_penalty_weight=risk_penalty, progress_bonus_weight=progress_bonus
        )
    except Exception as e: print(f"CRITICAL ERROR initializing agent: {e}"); return None, []

    # --- Initialize tracking variables ---
    evaluation_rewards_list: List[float] = []
    steps_history_list: List[int] = []
    loss_buffer_size_calc = max(500, eval_frequency * 2 // update_frequency);
    q_loss_history: deque = deque(maxlen=loss_buffer_size_calc)
    policy_loss_history: deque = deque(maxlen=loss_buffer_size_calc)
    best_eval_metric_value = -np.inf
    start_step_num = 0
    total_episodes_count = 0

    # --- Create results directory ---
    try: os.makedirs(results_dir, exist_ok=True)
    except OSError as e: save_best=False; save_periodically=False; logger.warning(f"Cannot create results dir: {e}")

    # --- Resume from checkpoint if specified ---
    if resume_from_checkpoint:
        logger.info(f"\n--- Resuming Training from: {resume_from_checkpoint} ---")
        if agent.load_model(resume_from_checkpoint):
             start_step_num = agent.total_updates # Resume based on agent updates
             logger.info(f"Resumed from update step: {agent.total_updates:,}")
        else: logger.warning("Failed load checkpoint. Start scratch."); start_step_num = 0; agent.total_updates = 0

    # --- Initial environment reset ---
    try:
        current_state, current_info = env.reset(seed=RANDOM_SEED + start_step_num)
        current_state = np.asarray(current_state, dtype=np.float32)
    except Exception as e: print(f"CRITICAL ERROR resetting env: {e}"); env.close(); eval_env.close(); return None, []

    # --- Main Training Loop ---
    logger.info("\n--- Starting Training Loop (Unified Agent v1.1.1) ---")
    pbar = tqdm(range(start_step_num, max_steps), initial=start_step_num, total=max_steps, desc="Training", unit="step", ncols=120)
    current_episode_reward = 0.0
    current_episode_steps = 0

    for current_env_step_num in pbar:
        # --- Environment Interaction ---
        action_to_take: np.ndarray
        if current_env_step_num < start_learning_steps:
            action_to_take = env.action_space.sample()
        else:
            action_to_take = agent.get_action(current_state, explore=True)

        try:
            next_obs, reward_val, terminated_flag, truncated_flag, info_dict = env.step(action_to_take)
            is_done = terminated_flag or truncated_flag
            if math.isnan(reward_val): reward_val = 0.0
            if math.isinf(reward_val): reward_val = 0.0 # Handle Inf rewards too
            next_obs_processed = np.asarray(next_obs, dtype=np.float32)
            agent.replay_buffer.push(current_state, action_to_take, reward_val, next_obs_processed, float(is_done))
            current_state = next_obs_processed
            current_episode_reward += reward_val
            current_episode_steps += 1

            # --- Episode End Handling ---
            if is_done:
                total_episodes_count += 1
                if 'episode' in info_dict:
                    # Access stats safely
                    ep_reward = info_dict['episode'].get('r', 0.0) # Default to 0.0 if key missing
                    ep_length = info_dict['episode'].get('l', 0)   # Default to 0 if key missing
                    # Calculate moving averages safely
                    avg_rew_val = np.mean(env.return_queue) if env.return_queue else ep_reward # Use current if queue empty
                    avg_len_val = np.mean(env.length_queue) if env.length_queue else ep_length # Use current if queue empty
                    eq_complexity_val = agent.unified_eq.get_total_complexity()
                    # Build postfix dictionary for tqdm
                    postfix_dict = {"Ep": total_episodes_count, "Rew(r)": f"{avg_rew_val:.1f}", "Len(r)": f"{avg_len_val:.0f}", "Noise": f"{agent.current_noise_std:.3f}", "Eq_C": f"{eq_complexity_val:.1f}"}
                    if q_loss_history: postfix_dict["QL(r)"] = f"{np.mean(q_loss_history):.2f}"
                    if policy_loss_history: postfix_dict["PL(r)"] = f"{np.mean(policy_loss_history):.2f}"
                    pbar.set_postfix(postfix_dict)

                # Reset environment
                reset_env_seed = RANDOM_SEED + current_env_step_num + total_episodes_count
                current_state, current_info = env.reset(seed=reset_env_seed)
                current_state = np.asarray(current_state, dtype=np.float32)
                current_episode_reward = 0.0
                current_episode_steps = 0

        except (gym.error.Error, Exception) as e:
            logger.warning(f"\nException env step {current_env_step_num}: {e}", exc_info=False)
            try:
                 reset_seed_after_err = RANDOM_SEED + current_env_step_num + total_episodes_count + 1
                 current_state, current_info = env.reset(seed=reset_seed_after_err)
                 current_state = np.asarray(current_state, dtype=np.float32); logger.info("Env reset OK after error.")
                 current_episode_reward = 0.0; current_episode_steps = 0
            except Exception as e2: logger.error(f"CRITICAL ERROR: Failed reset after error: {e2}. Stop."); break
            continue

        # --- Agent Update ---
        should_update_agent = current_env_step_num >= start_learning_steps and current_env_step_num % update_frequency == 0
        if should_update_agent:
            q_loss, policy_loss, batch_avg_reward = agent.update(step=agent.total_updates)
            if q_loss is not None: q_loss_history.append(q_loss)
            if policy_loss is not None: policy_loss_history.append(policy_loss)

        # --- Periodic Evaluation ---
        should_evaluate_agent = current_env_step_num > 0 and current_env_step_num % eval_frequency == 0 and current_env_step_num >= start_learning_steps
        if should_evaluate_agent:
            pbar.write("\n" + "-"*40 + f" Evaluating at Env Step {current_env_step_num:,} (Agent Updates: {agent.total_updates:,}) " + "-"*40)
            eval_avg_reward_val = agent.evaluate(eval_env, episodes=eval_episodes)
            evaluation_rewards_list.append(eval_avg_reward_val); steps_history_list.append(current_env_step_num)
            pbar.write(f"  Avg Eval Reward ({eval_episodes} eps): {eval_avg_reward_val:.2f}"); pbar.write(f"  Evolutions: {agent.total_evolutions}")
            avg_q_loss_str = f"{np.mean(q_loss_history):.4f}" if q_loss_history else "N/A"; avg_p_loss_str = f"{np.mean(policy_loss_history):.4f}" if policy_loss_history else "N/A"
            pbar.write(f"  Avg Losses (Q): {avg_q_loss_str} | (P): {avg_p_loss_str}")
            try: pbar.write(f"--- Current Unified Equation ---"); pbar.write(agent.unified_eq.to_string())
            except Exception as repr_e: pbar.write(f"  Error repr eq: {repr_e}")
            pbar.write("-" * 100)
            # Save best model
            if save_best:
                 is_better = not math.isinf(eval_avg_reward_val) and not math.isnan(eval_avg_reward_val) and eval_avg_reward_val > best_eval_metric_value
                 if is_better:
                     old_best_str = f"{best_eval_metric_value:.2f}" if not math.isinf(best_eval_metric_value) else "-inf"
                     pbar.write(f"  ** New best reward ({eval_avg_reward_val:.2f} > {old_best_str})! Saving model... **")
                     best_eval_metric_value = eval_avg_reward_val
                     best_model_filename = os.path.join(results_dir, f"unified_best_{env_name.replace('-','_')}.pt")
                     agent.save_model(filename=best_model_filename)

        # --- Periodic Saving ---
        should_save_periodically = save_periodically and current_env_step_num > 0 and current_env_step_num % save_interval == 0 and current_env_step_num > start_step_num
        if should_save_periodically:
             periodic_filename = os.path.join(results_dir, f"unified_step_{current_env_step_num}_{env_name.replace('-','_')}.pt")
             pbar.write(f"\n--- Saving Periodic Checkpoint at Env Step {current_env_step_num:,} ---"); agent.save_model(filename=periodic_filename)

    # --- End of Training Loop ---
    progress_bar.close();
    try: env.close() # Close the training environment
    except Exception: pass
    try: eval_env.close() # Close the evaluation environment
    except Exception: pass
    # Save final model
    final_model_filename = os.path.join(results_dir, f"unified_final_step_{max_steps}_{env_name.replace('-','_')}.pt")
    logger.info(f"\n--- Saving Final Model at Env Step {max_steps:,} ---"); agent.save_model(filename=final_model_filename)
    # Print summary
    end_time = time.time(); total_time = end_time - start_time
    print("\n" + "="*60); print(f"=== Training Finished (Unified Agent v1.1.1) ==="); print(f"=== Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ==="); print("="*60)
    print(f"Total Env Steps: {max_steps:,}"); print(f"Total Episodes: {total_episodes_count:,}"); print(f"Total Agent Updates: {agent.total_updates:,}"); print(f"Total Evolutions: {agent.total_evolutions}")
    print(f"Total Training Time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    if env.return_queue: print(f"Avg reward last {len(env.return_queue)} train eps: {np.mean(env.return_queue):.2f}")
    print(f"Final Equation Complexity: {agent.unified_eq.get_total_complexity():.2f}")
    best_str = f"{best_eval_metric_value:.2f}" if not math.isinf(best_eval_metric_value) else "N/A"; print(f"Best Eval Reward: {best_str}")

    # --- Generate Plots ---
    if steps_history_list and evaluation_rewards_list:
        logger.info("\n--- Generating Training Plots ---")
        try:
            # Set style and create figure/axes
            plt.style.use('ggplot'); fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True); fig.suptitle(f'Unified Agent Training ({env_name} - v1.1.1)', fontsize=16)
            # Plot Reward
            ax1 = axes[0]; ax1.plot(steps_history_list, evaluation_rewards_list, marker='.', linestyle='-', color='dodgerblue', label='Avg Eval Reward')
            if len(evaluation_rewards_list) >= 5: mv_avg = np.convolve(evaluation_rewards_list, np.ones(5)/5, mode='valid'); ax1.plot(steps_history_list[4:], mv_avg, linestyle='--', color='orangered', label='Moving Avg (5 evals)')
            ax1.set_ylabel('Avg Eval Reward'); ax1.set_title('Evaluation Reward'); ax1.legend(); ax1.grid(True, ls='--', lw=0.5)
            # Plot Q-Loss
            ax2 = axes[1];
            if q_loss_history:
                q_loss_list_plot = [ql for ql in q_loss_history if ql is not None] # Filter None
                update_steps_est_q = np.linspace(start_learning_steps, current_env_step_num, len(q_loss_list_plot), dtype=int)
                if len(update_steps_est_q) == len(q_loss_list_plot): # Ensure lengths match
                    ax2.plot(update_steps_est_q, q_loss_list_plot, label='Q-Loss (Raw)', alpha=0.4, c='darkorange', lw=0.8)
                    if len(q_loss_list_plot) >= 20: ql_ma = np.convolve(q_loss_list_plot, np.ones(20)/20, mode='valid'); ax2.plot(update_steps_est_q[19:], ql_ma, label='Q-Loss (MA-20)', c='red', lw=1.2)
                    if all(ql > 1e-9 for ql in q_loss_list_plot): # Check > 0
                         try: ax2.set_yscale('log'); ax2.set_ylabel('Q-Loss (Log)')
                         except ValueError: ax2.set_ylabel('Q-Loss')
                    else: ax2.set_ylabel('Q-Loss')
                else: ax2.text(0.5, 0.5, 'Q-Loss Data Mismatch', ha='center', va='center', transform=ax2.transAxes); ax2.set_ylabel('Q-Loss')
            else: ax2.text(0.5, 0.5, 'No Q-Loss Data', ha='center', va='center', transform=ax2.transAxes); ax2.set_ylabel('Q-Loss')
            ax2.set_title('Critic Loss'); ax2.legend(); ax2.grid(True, ls='--', lw=0.5);
            # Plot Policy Loss
            ax3 = axes[2];
            if policy_loss_history:
                p_loss_list_plot = [pl for pl in policy_loss_history if pl is not None] # Filter None
                update_steps_est_p = np.linspace(start_learning_steps, current_env_step_num, len(p_loss_list_plot), dtype=int)
                if len(update_steps_est_p) == len(p_loss_list_plot): # Ensure lengths match
                    ax3.plot(update_steps_est_p, p_loss_list_plot, label='P Loss (Raw)', alpha=0.4, c='forestgreen', lw=0.8)
                    if len(p_loss_list_plot) >= 20: pl_ma = np.convolve(p_loss_list_plot, np.ones(20)/20, mode='valid'); ax3.plot(update_steps_est_p[19:], pl_ma, label='P Loss (MA-20)', c='darkgreen', lw=1.2)
                else: ax3.text(0.5, 0.5, 'P-Loss Data Mismatch', ha='center', va='center', transform=ax3.transAxes)
            else: ax3.text(0.5, 0.5, 'No Policy Loss Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_ylabel('Policy Loss (-Q+Risk-Prog)'); ax3.set_title('Actor Loss'); ax3.legend(); ax3.grid(True, ls='--', lw=0.5)
            # Final plot settings
            ax3.set_xlabel('Environment Steps'); fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            # Save plot
            plot_filename = os.path.join(results_dir, f"unified_plots_{env_name.replace('-','_')}.png")
            plt.savefig(plot_filename, dpi=300); logger.info(f"Plots saved: '{plot_filename}'"); plt.close(fig)
        except Exception as e_plot: warnings.warn(f"Could not generate plots: {e_plot}", RuntimeWarning)
    else: logger.info("No evaluation data, skipping plots.")

    # --- Final Evaluation ---
    if trained_agent:
        logger.info("\n" + "="*60); logger.info("=== Final Evaluation ==="); logger.info("="*60)
        final_eval_env = None
        try:
            logger.info("Evaluating final agent (rendering if possible)...")
            final_eval_env = gym.make(ENVIRONMENT_NAME, render_mode="human")
            final_performance = trained_agent.evaluate(final_eval_env, episodes=5)
            logger.info(f"Final Agent Avg Perf (5 eps, rendered): {final_performance:.2f}")
        except Exception as e_render:
            logger.warning(f"  * Render eval failed: {e_render}. Evaluating without render...")
            try:
                final_eval_env = gym.make(ENVIRONMENT_NAME)
                final_performance = trained_agent.evaluate(final_eval_env, episodes=10)
                logger.info(f"  * Final Agent Avg Perf (10 eps, no render): {final_performance:.2f}")
            except Exception as e2_render: logger.error(f"  * ERROR: Final evaluation failed: {e2_render}")
        finally:
             if final_eval_env:
                 try: final_eval_env.close()
                 except Exception: pass
        # Evaluate best saved model
        if SAVE_BEST_MODEL and os.path.exists(results_dir):
             best_model_file = os.path.join(results_dir, f"unified_best_{env_name.replace('-','_')}.pt")
             logger.info("\n" + "="*60); logger.info(f"=== Evaluating Best Saved Unified Agent ==="); logger.info("="*60)
             if os.path.exists(best_model_file):
                 try:
                    logger.info("Loading best model for evaluation...")
                    # Rebuild structure for loading
                    eval_eq_structure_best = UnifiedAdaptableEquation("eval_best_eq")
                    eval_curr_dim_best = trained_agent.state_dim; eval_last_id_best: Optional[str] = None; eval_feat_key = 'state'; eval_shared_last_id = None
                    i_hidden_best = 0
                    while i_hidden_best < len(hidden_dims):
                        h_dim_best = hidden_dims[i_hidden_best]
                        comp_id_best = f'shared_hidden_{i_hidden_best}'; eq_s_best = EvolvingEquation(eval_curr_dim_best, eq_init_complexity, h_dim_best, eq_complexity_limit)
                        comp_s = NumericTransformComponent(comp_id_best, eq_s_best, F.relu, {'shared'}, {'task:rl'}, {'features'}, eval_feat_key, 'features')
                        eval_eq_structure_best.add_component(comp_s, execute_after=eval_shared_last_id); eval_curr_dim_best = h_dim_best; eval_shared_last_id = comp_id_best; eval_feat_key = 'features'
                        i_hidden_best += 1
                    eval_id_risk = 'risk_prog_eval'; eval_risk_prog_inst = RiskProgressEvaluator(eval_id_risk, eval_curr_dim_best, max(32,eval_curr_dim_best//2), input_key='features', required_context={'request_evaluation'})
                    eval_eq_structure_best.add_component(eval_risk_prog_inst, execute_after=eval_shared_last_id)
                    actor_out_eq_best = EvolvingEquation(eval_curr_dim_best, eq_init_complexity, trained_agent.action_dim, eq_complexity_limit)
                    actor_out_comp_best = NumericTransformComponent('actor_output_layer', actor_out_eq_best, None, {'actor'}, {'request_action'}, {'policy_features'}, 'features', 'policy_features')
                    eval_eq_structure_best.add_component(actor_out_comp_best, execute_after=eval_shared_last_id)
                    policy_dec_comp_best = PolicyDecisionComponent('policy_decision', trained_agent.action_dim, {'actor'}, {'request_action'}, {'action_raw'}, 'policy_features', 'action_raw')
                    eval_eq_structure_best.add_component(policy_dec_comp_best, execute_after='actor_output_layer')
                    critic_in_dim_best = eval_curr_dim_best + trained_agent.action_dim
                    critic_hid_dim_best = hidden_dims[-1]
                    critic_hid_eq_best = EvolvingEquation(critic_in_dim_best, eq_init_complexity, critic_hid_dim_best, eq_complexity_limit)
                    critic_hid_comp_best = NumericTransformComponent('critic_hidden_layer', critic_hid_eq_best, F.relu, {'critic','expert'}, {'request_q_value'}, {'value_features'}, 'critic_input', 'value_features')
                    eval_eq_structure_best.add_component(critic_hid_comp_best, execute_after=eval_shared_last_id)
                    critic_out_eq_best = EvolvingEquation(critic_hid_dim_best, eq_init_complexity, 1, eq_complexity_limit)
                    critic_out_comp_best = NumericTransformComponent('critic_output_layer', critic_out_eq_best, None, {'critic','expert'}, {'request_q_value'}, {'value_features_final'}, 'value_features', 'value_features_final')
                    eval_eq_structure_best.add_component(critic_out_comp_best, execute_after='critic_hidden_layer')
                    value_est_comp_best = ValueEstimationComponent('value_estimation', {'critic','expert'}, {'request_q_value'}, {'q_value'}, 'value_features_final', 'q_value')
                    eval_eq_structure_best.add_component(value_est_comp_best, execute_after='critic_output_layer')
                    # Initialize evaluation agent
                    eval_best_agent = UnifiedLearningAgent(
                        trained_agent.state_dim, trained_agent.action_dim, (trained_agent.action_low, trained_agent.action_high),
                        eval_eq_structure_best, evolution_engine=None, # No evolution during eval
                        gamma=trained_agent.gamma, tau=trained_agent.tau, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                        buffer_capacity=100, batch_size=BATCH_SIZE, exploration_noise_std=0, noise_decay_rate=1, min_exploration_noise=0,
                        weight_decay=WEIGHT_DECAY, grad_clip_norm=None, grad_clip_value=None,
                        risk_penalty_weight=trained_agent.risk_penalty_weight, progress_bonus_weight=trained_agent.progress_bonus_weight)
                    # Load model
                    if eval_best_agent.load_model(best_model_file):
                        logger.info("Best model loaded successfully. Evaluating...")
                        best_eval_env_final = None
                        try:
                             best_eval_env_final = gym.make(ENVIRONMENT_NAME)
                             best_performance = eval_best_agent.evaluate(best_eval_env_final, episodes=10)
                             logger.info(f"Best Saved Agent Avg Perf (10 eps): {best_performance:.2f}")
                             logger.info(f"  Equation Complexity: {eval_best_agent.unified_eq.get_total_complexity():.2f}")
                             logger.info("\n--- Best Model Equation Representation ---")
                             logger.info(eval_best_agent.unified_eq.to_string())
                        except Exception as eval_e: logger.error(f"Error evaluating best model: {eval_e}")
                        finally:
                             if best_eval_env_final:
                                 try: best_eval_env_final.close()
                                 except Exception: pass
                    else: logger.warning("Skipping eval of best model (load failed).")
                 except Exception as e_load_best: logger.error(f"ERROR loading/evaluating best model: {e_load_best}"); traceback.print_exc()
             else: logger.warning(f"Best model file not found: '{best_model_file}'. Skipping.")
    else: logger.info("\nTraining failed or agent init failed. No final evaluation.")

    print("\n===================================================")
    print("===      Unified Learning Agent Execution End       ===")
    print("===================================================")

# --- End of Main Execution Block ---

# ============================================================== #
# ===================== ShapePlotter2D Class =================== #
# ============================================================== #
# Note: ShapePlotter2D is included here for completeness,
#       as it might be used with the to_shape_engine_string method
#       of EvolvingEquation for visualization or analysis.
class ShapePlotter2D:
    """محرك رسم الأشكال ثنائي الأبعاد (v1.1)."""
    # ... (الكود الكامل لـ ShapePlotter2D v1.1 يوضع هنا) ...
    def __init__(self):
        self.xp = np
        self.components: List[Dict] = []
        self.current_style: Dict[str, Any] = {'color': '#000000','linewidth': 1.5,'fill': False,'gradient': None,'dash': None,'opacity': 1.0,}
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.parser = None
        if PYPARSING_AVAILABLE: self._setup_parser()
        else: logger.error("Pyparsing unavailable for plotter.")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.WARNING)

    def _setup_parser(self):
        if not PYPARSING_AVAILABLE: return
        left_paren=Suppress('('); right_paren=Suppress(')'); left_bracket=Suppress('[')
        right_bracket=Suppress(']'); left_brace=Suppress('{'); right_brace=Suppress('}')
        equals_sign=Suppress('='); colon=Suppress(':'); comma=Suppress(',')
        point_lit=Literal('.'); exponent_lit=CaselessLiteral('E'); plus_minus_lit=Literal('+')|Literal('-')
        number_literal=Combine(ppOptional(plus_minus_lit)+Word(nums)+ppOptional(point_lit+ppOptional(Word(nums)))+ppOptional(exponent_lit+ppOptional(plus_minus_lit)+Word(nums)))
        number_literal.setParseAction(lambda tokens: float(tokens[0])).setName("number")
        identifier=Word(alphas, alphanums+"_").setName("identifier")
        param_value=number_literal|identifier
        param_list=ppOptional(delimitedList(Group(param_value),delim=comma)).setParseAction(lambda t: t if t else []).setName("parameters")("params")
        func_name=identifier.copy().setName("function_name")("func")
        range_expr=Group(left_bracket+number_literal("min")+colon+number_literal("max")+right_bracket).setName("range")("range")
        style_key=identifier.copy().setName("style_key")("key")
        hex_color_literal=Combine(Literal('#')+Word(hexnums,exact=6)).setName("hex_color")
        bool_true=CaselessLiteral("true")|CaselessLiteral("yes")|CaselessLiteral("on")
        bool_false=CaselessLiteral("false")|CaselessLiteral("no")|CaselessLiteral("off")|CaselessLiteral("none")
        bool_literal=(bool_true.copy().setParseAction(lambda: True)|bool_false.copy().setParseAction(lambda: False)).setName("boolean")
        string_value=Word(alphanums+"-_./\\:").setName("string_value")
        simple_style_value=number_literal|hex_color_literal|bool_literal|identifier|string_value
        tuple_element=simple_style_value|hex_color_literal
        tuple_value=Group(left_paren+delimitedList(tuple_element,delim=comma)+right_paren).setName("tuple_value")
        list_of_tuples_value=Group(left_bracket+delimitedList(tuple_value,delim=comma)+right_bracket).setName("list_of_tuples")("list_value")
        style_value=list_of_tuples_value|simple_style_value; style_value.setName("style_value")
        style_assignment=Group(style_key+equals_sign+style_value).setName("style_assignment")
        style_expr=Group(left_brace+ppOptional(delimitedList(style_assignment,delim=comma))+right_brace).setParseAction(lambda t: t[0] if t else []).setName("style_block")("style")
        shape_component_expr=( func_name+left_paren+param_list+right_paren+ppOptional(range_expr)+ppOptional(style_expr) ).setName("shape_component")
        self.parser = shape_component_expr + StringEnd()

    def _parse_style(self, style_tokens: Optional[List]) -> Dict:
        style_output_dict: Dict[str, Any] = {}
        if style_tokens is None: return style_output_dict
        for style_item_group in style_tokens:
            style_key_str = style_item_group['key']; value_parsed_token = style_item_group[1]
            if 'list_value' in style_item_group:
                list_of_parsed_tuples = style_item_group['list_value']; processed_tuple_list = []
                for parsed_tuple_group in list_of_parsed_tuples:
                    current_processed_tuple = tuple(val for val in parsed_tuple_group)
                    processed_tuple_list.append(current_processed_tuple)
                if style_key_str == 'gradient':
                    gradient_colors: List[str] = []; gradient_positions: List[float] = []; is_gradient_valid = True
                    for gradient_tuple in processed_tuple_list:
                        is_valid_tuple = (len(gradient_tuple) == 2 and isinstance(gradient_tuple[0], str) and isinstance(gradient_tuple[1], (float, int)))
                        if is_valid_tuple: gradient_colors.append(gradient_tuple[0]); gradient_positions.append(float(gradient_tuple[1]))
                        else: self.logger.warning(f"Invalid grad stop: {gradient_tuple}"); is_gradient_valid = False; break
                    if is_gradient_valid and gradient_colors:
                        sorted_gradient_data = sorted(zip(gradient_positions, gradient_colors))
                        gradient_positions = [pos for pos, col in sorted_gradient_data]; gradient_colors = [col for pos, col in sorted_gradient_data]
                        if not gradient_positions or gradient_positions[0] > 1e-6: first_color = gradient_colors[0] if gradient_colors else '#000'; gradient_positions.insert(0, 0.0); gradient_colors.insert(0, first_color)
                        if gradient_positions[-1] < 1.0 - 1e-6 : last_color = gradient_colors[-1] if gradient_colors else '#FFF'; gradient_positions.append(1.0); gradient_colors.append(last_color)
                        style_output_dict[style_key_str] = (gradient_colors, gradient_positions)
                elif style_key_str == 'dash':
                    dash_tuple_valid = (processed_tuple_list and isinstance(processed_tuple_list[0], tuple) and all(isinstance(n, (int, float)) for n in processed_tuple_list[0]))
                    if dash_tuple_valid:
                        try: float_values = [float(x) for x in processed_tuple_list[0]]; dash_string = ",".join(map(str, float_values)); style_output_dict[style_key_str] = dash_string
                        except Exception as e: self.logger.warning(f"Invalid dash list: {e}"); style_output_dict[style_key_str] = None
                    else: self.logger.warning(f"Invalid dash format: {processed_tuple_list}"); style_output_dict[style_key_str] = None
                else: style_output_dict[style_key_str] = processed_tuple_list
            else: style_output_dict[style_key_str] = value_parsed_token
        current_dash_value = style_output_dict.get('dash')
        if current_dash_value == '--': style_output_dict['dash'] = '5,5'
        if 'linewidth' in style_output_dict:
            lw_val = style_output_dict['linewidth']
            if not isinstance(lw_val, (int, float)):
                try: style_output_dict['linewidth'] = float(lw_val)
                except ValueError: self.logger.warning(f"Invalid lw: '{lw_val}'"); style_output_dict.pop('linewidth', None)
        if 'opacity' in style_output_dict:
            op_val = style_output_dict['opacity']
            if not isinstance(op_val, (int, float)):
                try: style_output_dict['opacity'] = float(op_val)
                except ValueError: self.logger.warning(f"Invalid op: '{op_val}'"); style_output_dict.pop('opacity', None)
        return style_output_dict

    def set_style(self, **kwargs):
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.current_style.update(valid_kwargs)
        # self.logger.info(f"Default style updated: {self.current_style}")

    def parse_equation(self, equation: str):
        if not self.parser: self.logger.error("Parser not initialized."); return self
        # self.logger.info(f"\n--- [Plotter] Parsing: {equation[:50]}... ---")
        equation_parts = re.split(r'\s*[\+\&\|\-]\s*', equation)
        newly_parsed_components: List[Dict] = []
        part_index = 0; total_parts = len(equation_parts)
        while part_index < total_parts:
            part_string = equation_parts[part_index].strip()
            if not part_string: part_index += 1; continue
            try:
                parsed_result = self.parser.parseString(part_string, parseAll=True)
                function_name = parsed_result.func.lower()
                raw_params_list = parsed_result.params if 'params' in parsed_result else []
                processed_params: List[Union[float, str]] = []
                param_group_index = 0
                while param_group_index < len(raw_params_list):
                     param_group = raw_params_list[param_group_index]; value_in_group = param_group[0]
                     if isinstance(value_in_group, str):
                         try: float_value = float(value_in_group); processed_params.append(float_value)
                         except ValueError: processed_params.append(value_in_group)
                     else: processed_params.append(value_in_group)
                     param_group_index += 1
                component_dict = self._create_shape_2d(function_name, processed_params)
                style_tokens_parsed = parsed_result.style if 'style' in parsed_result else None
                shape_specific_style = self._parse_style(style_tokens_parsed)
                final_shape_style = {**self.current_style, **shape_specific_style}
                component_dict['style'] = final_shape_style
                if 'range' in parsed_result:
                    range_value_list = parsed_result.range.asList()
                    if len(range_value_list) == 2:
                        try: range_min = float(range_value_list[0]); range_max = float(range_value_list[1]); component_dict['range'] = (range_min, range_max)
                        except (ValueError, TypeError) as e: self.logger.warning(f" Invalid range: {e}")
                component_dict['name'] = function_name
                component_dict['original_params'] = list(processed_params)
                newly_parsed_components.append(component_dict)
            except ParseException as parse_error: print(f"!!!! Plotter Parse Error: '{part_string}' -> {parse_error.explain()} !!!!")
            except ValueError as value_error: print(f"!!!! Plotter Value/Param Error: '{part_string}' -> {value_error} !!!!")
            except Exception as general_error: print(f"!!!! Plotter Unexpected Error: '{part_string}' -> {general_error} !!!!"); traceback.print_exc()
            part_index += 1
        self.components.extend(newly_parsed_components)
        return self

    def _create_shape_2d(self, func_name: str, params: List[Union[float, str]]) -> Dict:
        processed_float_params: List[float] = []
        i = 0
        while i < len(params):
            p = params[i]
            if isinstance(p, (int, float)): processed_float_params.append(float(p))
            else: raise ValueError(f"Param {i+1} ('{p}') for '{func_name}' must be numeric.")
            i += 1
        shapes_2d_registry = {
            'line': (self._create_line, 4), 'circle': (self._create_circle, 3),
            'bezier': (self._create_bezier, lambda p_lst: len(p_lst) >= 4 and len(p_lst) % 2 == 0),
            'sine': (self._create_sine, 3), 'exp': (self._create_exp, 3),
            'polygon': (self._create_polygon, lambda p_lst: len(p_lst) >= 6 and len(p_lst) % 2 == 0)
        }
        if func_name not in shapes_2d_registry: raise ValueError(f"Unsupported shape: '{func_name}'")
        creator_func, param_check = shapes_2d_registry[func_name]
        num_params = len(processed_float_params)
        valid = False; expected = 'N/A'
        if isinstance(param_check, int): expected = f"{param_check}"; valid = (num_params == param_check)
        elif callable(param_check): expected = "specific format"; valid = param_check(processed_float_params)
        else: raise TypeError("Invalid param check.")
        if not valid: raise ValueError(f"Param error for '{func_name}'. Expected: {expected}, Got: {num_params}.")
        try: shape_dict = creator_func(*processed_float_params); shape_dict['type'] = '2d'; return shape_dict
        except TypeError as e: raise ValueError(f"Creator func type error for '{func_name}': {e}")

    def _create_line(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            _x1,_y1,_x2,_y2 = p; dx = _x2-_x1
            if abs(dx)<1e-9: return xp.where(xp.abs(x-_x1)<1e-9, (_y1+_y2)/2.0, xp.nan)
            m = (_y2-_y1)/dx; c = _y1-m*_x1; return m*x+c
        dr = (min(x1,x2), max(x1,x2)); return {'func':func_impl, 'params':[x1,y1,x2,y2], 'range':dr, 'parametric':False}

    def _create_circle(self, x0: float, y0: float, r: float) -> Dict:
        if r < 0: r = abs(r)
        def func_impl(t: np.ndarray, p: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            _x0,_y0,_r = p; x = _x0+_r*xp.cos(t); y = _y0+_r*xp.sin(t); return x, y
        dr = (0, 2*np.pi); return {'func':func_impl, 'params':[x0,y0,r], 'range':dr, 'parametric':True, 'is_polygon':True}

    def _create_bezier(self, *params_flat: float) -> Dict:
        if not PYPARSING_AVAILABLE: raise ImportError("math.comb unavailable.")
        from math import comb as math_comb
        def func_impl(t: np.ndarray, p_in: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            pts = xp.array(p_in).reshape(-1, 2); n = len(pts)-1
            if n < 0: return xp.array([]), xp.array([])
            coeffs = xp.array([math_comb(n, k) for k in range(n+1)])
            t_col = xp.asarray(t).reshape(-1, 1); k_rng = xp.arange(n+1)
            t_pow = t_col**k_rng; omt_pow = (1.0-t_col)**(n-k_rng)
            bernstein = coeffs*t_pow*omt_pow; coords = bernstein@pts
            return coords[:,0], coords[:,1]
        dr = (0.0, 1.0); return {'func':func_impl, 'params':list(params_flat), 'range':dr, 'parametric':True}

    def _create_sine(self, A: float, freq: float, phase: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            _A,_f,_p = p; return _A*xp.sin(_f*x+_p) if abs(_f)>1e-9 else xp.full_like(x, _A*xp.sin(_p))
        period = 2*np.pi/abs(freq) if abs(freq)>1e-9 else 10.0; dr=(0,period)
        return {'func':func_impl, 'params':[A,freq,phase], 'range':dr, 'parametric':False}

    def _create_exp(self, A: float, k: float, x0: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            _A,_k,_x0 = p; return xp.full_like(x, _A) if abs(_k)<1e-9 else _A*xp.exp(xp.clip(-_k*(x-_x0),-700,700))
        rw = 5.0/abs(k) if abs(k)>1e-9 else 5.0; dr=(x0-rw, x0+rw)
        return {'func':func_impl, 'params':[A,k,x0], 'range':dr, 'parametric':False}

    def _create_polygon(self, *params_flat: float) -> Dict:
        def func_impl(t: np.ndarray, p_in: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            pts = list(zip(p_in[0::2], p_in[1::2])); closed = pts + [pts[0]]
            segs = xp.array(closed); n_segs = len(pts)
            if n_segs == 0: return xp.array([]), xp.array([])
            diffs = xp.diff(segs, axis=0); lengths = xp.sqrt(xp.sum(diffs**2, axis=1))
            total_len = xp.sum(lengths)
            if total_len < 1e-9: return xp.full_like(t, segs[0,0]), xp.full_like(t, segs[0,1])
            cum_norm = xp.concatenate((xp.array([0.0]), xp.cumsum(lengths))) / total_len
            t_clip = xp.clip(t, 0.0, 1.0); x_res, y_res = xp.zeros_like(t_clip), xp.zeros_like(t_clip)
            i_seg = 0
            while i_seg < n_segs:
                 s_n, e_n = cum_norm[i_seg], cum_norm[i_seg+1]
                 mask = (t_clip >= s_n) & (t_clip <= e_n)
                 if not xp.any(mask): i_seg += 1; continue
                 seg_len_n = e_n - s_n
                 seg_t = xp.where(seg_len_n > 1e-9, (t_clip[mask] - s_n) / seg_len_n, 0.0)
                 s_pt, e_pt = segs[i_seg], segs[i_seg+1]
                 x_res[mask] = s_pt[0] + (e_pt[0] - s_pt[0]) * seg_t
                 y_res[mask] = s_pt[1] + (e_pt[1] - s_pt[1]) * seg_t
                 i_seg += 1
            x_res[t_clip >= 1.0] = segs[-1, 0]; y_res[t_clip >= 1.0] = segs[-1, 1]
            return x_res, y_res
        dr = (0.0, 1.0); return {'func':func_impl, 'params':list(params_flat), 'range':dr, 'parametric':True, 'is_polygon':True}

    def _create_gradient(self, colors: List[str], positions: List[float]) -> Optional[LinearSegmentedColormap]:
        if not MATPLOTLIB_AVAILABLE: return None
        if not colors or not positions or len(colors) != len(positions): return None
        try:
            sorted_data = sorted(zip(positions, colors))
            norm_pos = [max(0.0, min(1.0, p)) for p, c in sorted_data]
            sorted_cols = [c for p, c in sorted_data]
            cdict = {'red': [], 'green': [], 'blue': []}; valid = False
            i = 0
            while i < len(norm_pos):
                 pos, color = norm_pos[i], sorted_cols[i]
                 try: rgb = plt.cm.colors.to_rgb(color); valid=True
                 except ValueError: i+=1; continue
                 cdict['red'].append((pos, rgb[0], rgb[0]))
                 cdict['green'].append((pos, rgb[1], rgb[1]))
                 cdict['blue'].append((pos, rgb[2], rgb[2]))
                 i+=1
            if not valid: return None
            name = f"custom_{id(colors)}_{int(time.time()*1000)}"
            cmap = LinearSegmentedColormap(name, cdict)
            return cmap
        except Exception as e: self.logger.error(f"Gradient creation error: {e}"); return None

    def plot(self, resolution: int = 500, title: str = "2D Plot", figsize: Tuple[float, float] = (8, 8),
             ax: Optional[plt.Axes] = None, show_plot: bool = True, save_path: Optional[str] = None,
             clear_before_plot: bool = True):
        if not MATPLOTLIB_AVAILABLE: self.logger.error("Matplotlib not available."); return

        current_ax = ax if ax is not None else self.ax
        current_fig: Optional[plt.Figure] = None
        setup_new_internal = False

        if current_ax is None:
            if self.fig is None or self.ax is None:
                 self.fig, self.ax = plt.subplots(figsize=figsize); setup_new_internal = True
            current_ax = self.ax; current_fig = self.fig
        elif ax is not None: current_ax = ax; current_fig = ax.figure
        else: current_ax = self.ax; current_fig = self.fig

        if current_ax is None: self.logger.error("Failed to get Axes."); return
        if current_fig is None and current_ax is not None: current_fig = current_ax.figure

        if clear_before_plot: current_ax.clear()

        min_x, max_x = float('inf'), float('-inf'); min_y, max_y = float('inf'), float('-inf')
        has_drawable = False; plot_data_cache: List[Dict] = []
        i = 0
        while i < len(self.components):
            comp = self.components[i]
            is_valid = (comp.get('type') == '2d' and 'func' in comp and 'range' in comp and 'params' in comp)
            if not is_valid: i += 1; continue
            comp_name = comp.get('name', f'Comp {i}')
            params = comp['params']; comp_range = comp['range']; is_para = comp.get('parametric', False)
            try:
                xp = self.xp
                if is_para: t = xp.linspace(comp_range[0], comp_range[1], resolution); x_calc, y_calc = comp['func'](t, params, xp)
                else: x_calc = xp.linspace(comp_range[0], comp_range[1], resolution); y_calc = comp['func'](x_calc, params, xp)
                valid_mask = ~xp.isnan(x_calc) & ~xp.isnan(y_calc)
                x_plot, y_plot = x_calc[valid_mask], y_calc[valid_mask]
                if x_plot.size > 0:
                    min_x = min(min_x, xp.min(x_plot)); max_x = max(max_x, xp.max(x_plot))
                    min_y = min(min_y, xp.min(y_plot)); max_y = max(max_y, xp.max(y_plot))
                    plot_data_cache.append({'x': x_plot, 'y': y_plot, 'comp': comp}); has_drawable = True
            except Exception as e: self.logger.error(f" Calc error for {comp_name}: {e}", exc_info=False)
            i += 1

        if has_drawable:
            if not np.isfinite(min_x): min_x = -1.0
            if not np.isfinite(max_x): max_x = 1.0
            if not np.isfinite(min_y): min_y = -1.0
            if not np.isfinite(max_y): max_y = 1.0
            xr = max_x - min_x; yr = max_y - min_y
            px = xr * 0.1 + (0.1 if xr < 1e-6 else 0); py = yr * 0.1 + (0.1 if yr < 1e-6 else 0)
            if px < 1e-6: px = 1.0;
            if py < 1e-6: py = 1.0;
            xlim_min = min_x - px; xlim_max = max_x + px; ylim_min = min_y - py; ylim_max = max_y + py
            if not np.isfinite(xlim_min): xlim_min = -10.0
            if not np.isfinite(xlim_max): xlim_max = 10.0
            if not np.isfinite(ylim_min): ylim_min = -10.0
            if not np.isfinite(ylim_max): ylim_max = 10.0
            current_ax.set_xlim(xlim_min, xlim_max); current_ax.set_ylim(ylim_min, ylim_max)
            current_ax.set_aspect('equal', adjustable='box')
        else:
            current_ax.set_xlim(-10, 10); current_ax.set_ylim(-10, 10); current_ax.set_aspect('equal', adjustable='box')

        for data in plot_data_cache:
            x_p, y_p = data['x'], data['y']; comp = data['comp']; style = comp.get('style', self.current_style)
            is_poly = comp.get('is_polygon', False); comp_nm = comp.get('name', '?')
            min_pts = 1 if is_poly and style.get('fill') else 2
            if x_p.size < min_pts: continue
            color = style.get('color', '#000'); lw = style.get('linewidth', 1.0); alpha = style.get('opacity', 1.0)
            fill = style.get('fill', False); gradient = style.get('gradient'); dash = style.get('dash')
            ls = '-';
            if dash:
                ls_map = {'-':'-', '--':'--', ':':':', '-.':'-.'};
                if dash in ls_map: ls = ls_map[dash]
                elif isinstance(dash, str) and re.match(r'^[\d\s,.]+$', dash):
                    try: dt = tuple(map(float, re.findall(r"(\d+\.?\d*)", dash))); ls = (0, dt) if dt else '-'
                    except ValueError: pass
            if gradient:
                cmap = self._create_gradient(gradient[0], gradient[1])
                if cmap:
                    pts = np.array([x_p, y_p]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    if len(segs)>0:
                        norm = plt.Normalize(0, 1); lc_colors = cmap(norm(np.linspace(0, 1, len(segs)))); lc_colors[:, 3] = alpha
                        lc = LineCollection(segs, colors=lc_colors, linewidths=lw, linestyle=ls); current_ax.add_collection(lc)
                        if fill:
                            fill_c = cmap(0.5); fill_a = alpha * 0.4; fill_final = (*fill_c[:3], fill_a)
                            if is_poly: current_ax.fill(x_p, y_p, color=fill_final, closed=True)
                            else: current_ax.fill_between(x_p, y_p, color=fill_final, interpolate=True)
                else: # Fallback
                    current_ax.plot(x_p, y_p, color=color, lw=lw, linestyle=ls, alpha=alpha)
                    if fill: fill_a = alpha*0.3; current_ax.fill(x_p, y_p, color=color, alpha=fill_a, closed=is_poly) if is_poly else current_ax.fill_between(x_p,y_p,color=color,alpha=fill_a)
            else: # No gradient
                current_ax.plot(x_p, y_p, color=color, lw=lw, linestyle=ls, alpha=alpha)
                if fill:
                    fill_a = alpha * 0.3
                    if is_poly: current_ax.fill(x_p, y_p, color=color, alpha=fill_a, closed=True)
                    elif x_p.ndim==1 and y_p.ndim==1 and x_p.shape==y_p.shape: current_ax.fill_between(x_p, y_p, color=color, alpha=fill_a, interpolate=True)

        current_ax.set_title(title); current_ax.set_xlabel("X-Axis"); current_ax.set_ylabel("Y-Axis")
        current_ax.grid(True, linestyle='--', alpha=0.6)
        if current_fig:
             try: current_fig.tight_layout()
             except Exception: pass

        if save_path and current_fig:
            try:
                 save_dir = os.path.dirname(save_path)
                 if save_dir: os.makedirs(save_dir, exist_ok=True)
                 current_fig.savefig(save_path, dpi=90, bbox_inches='tight', pad_inches=0.1)
                 # self.logger.info(f"Plot saved to: {save_path}")
            except Exception as e:
                 self.logger.error(f"Failed save plot to '{save_path}': {e}")

        if show_plot:
            try: plt.show()
            except Exception as e: self.logger.error(f"Error displaying plot: {e}")
        elif setup_new_internal and current_fig and not save_path and ax is None:
             plt.close(current_fig)
             self.fig = None; self.ax = None


# ============================================================== #
# ==================== COMPARISON FUNCTION ===================== #
# ============================================================== #

def compare_images_ssim(image_path_a: str, image_path_b: str) -> Optional[float]:
    """
    تحسب درجة التشابه الهيكلي (SSIM) بين صورتين.
    """
    # استخدام logger المعرف عامًا
    if not SKIMAGE_AVAILABLE or ssim is None: logger.error("SSIM requires scikit-image."); return None
    if not CV_AVAILABLE: logger.error("SSIM requires OpenCV."); return None
    try:
        img_a = cv2.imread(image_path_a)
        img_b = cv2.imread(image_path_b)
        if img_a is None: logger.error(f"SSIM fail: Cannot read A: {image_path_a}"); return None
        if img_b is None: logger.error(f"SSIM fail: Cannot read B: {image_path_b}"); return None
        if img_a.shape != img_b.shape:
            target_h, target_w = img_a.shape[:2]
            # logger.warning(f"Resizing B ({img_b.shape}) to A ({img_a.shape}) for SSIM.")
            img_b = cv2.resize(img_b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if img_a.shape != img_b.shape: logger.error("Resize failed."); return None
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        data_range_val = float(gray_a.max() - gray_a.min())
        if data_range_val < 1e-6: return 1.0 if np.array_equal(gray_a, gray_b) else 0.0
        min_dim = min(gray_a.shape[0], gray_a.shape[1])
        win_size_val = min(7, min_dim)
        if win_size_val % 2 == 0: win_size_val -= 1
        win_size_val = max(3, win_size_val)
        score_val = ssim(gray_a, gray_b, data_range=data_range_val, win_size=win_size_val)
        return float(score_val)
    except cv2.error as cv_err: logger.error(f"OpenCV error during compare: {cv_err}"); return None
    except Exception as e: logger.error(f"SSIM comparison error: {e}", exc_info=False); return None

# ===================== ShapeExtractor Class =================== #
class ShapeExtractor:
    """Extractor لاستخراج المعادلات من الصور باستخدام خوارزميات الرؤية الحاسوبية (v1.1.0)."""
    
    # التهيئة الافتراضية للمعلمات
    DEFAULT_CONFIG = {
        'canny_threshold1': 50,
        'canny_threshold2': 150,
        'approx_poly_epsilon_factor': 0.02,
        'contour_min_area': 500,
        'line_polygon_distance_tolerance': 5.0,
        'min_final_line_length': 30.0,
        'hough_lines_threshold': 50,
        'line_polygon_angle_tolerance_deg': 5.0,
        'deduplication_method': 'distance',
        'remove_lines_within_polygons': True,
        'merge_corner_lines': True
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.DEFAULT_CONFIG.copy()
        self.validation_errors = []
        self.validate_config()

    def validate_config(self):
        """التحقق من صحة البارامترات."""
        for key, value in self.config.items():
            if key not in self.DEFAULT_CONFIG:
                self.validation_errors.append(f"Unknown parameter: {key}")
            # إضافة تحقق من النوع إذا لزم الأمر

    def extract_equation(self, image_path: str) -> str:
        """استخراج المعادلة من الصورة."""
        if not CV_AVAILABLE:
            raise ImportError("OpenCV required for extraction.")
        
        # منطق الاستخراج (مثال مبسط)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(image, 
                self.config['canny_threshold1'], 
                self.config['canny_threshold2']
            )
            # ... (إضافة منطق استخراج الأشكال الهندسية)
            return "extracted_equation_representation"
        except Exception as e:
            raise RuntimeError(f"Extraction failed: {e}")


# ============================================================== #
# ===================== OPTIMIZATION LOOP ====================== #
# ============================================================== #
if __name__ == "__main__":

    best_config_global = ShapeExtractor.DEFAULT_CONFIG.copy()

    print("*" * 70)
    print(" Shape Extractor Optimization Loop (v1.1.0 using Bayesian Optimization)")
    print("*" * 70)

    # --- التحقق من المكتبات ---
    # ** تم إصلاح الخطأ هنا **
    if not CV_AVAILABLE: print("\nERROR: OpenCV required."); sys.exit(1)
    if not PYPARSING_AVAILABLE: print("\nERROR: Pyparsing required."); sys.exit(1)
    if not MATPLOTLIB_AVAILABLE: print("\nERROR: Matplotlib required."); sys.exit(1)
    if not SKIMAGE_AVAILABLE: print("\nERROR: scikit-image required for comparison."); sys.exit(1)
    if not SKOPT_AVAILABLE: print("\nERROR: scikit-optimize required for optimization."); sys.exit(1)

    # --- الإعدادات ---
    # !!! هام: يجب تعديل هذا المسار !!!
    external_image_path = "tt.png" # <-- ** تأكد من أن هذا المسار صحيح لصورتك **
    reconstructed_image_path = "_temp_reconstructed_opt.png"
    n_calls_optimizer = 30
    n_initial_points_optimizer = 10

    # --- تعريف فضاء البحث (نقل التعريف إلى هنا) ---
    params_to_tune = {
        "canny_threshold2":        {"min": 100, "max": 250, "type": int,   "step": 10},
        "approx_poly_epsilon_factor": {"min": 0.01, "max": 0.03, "type": float, "step": 0.002},
        "contour_min_area":        {"min": 100, "max": 1000, "type": int,   "step": 50},
        "line_polygon_distance_tolerance": {"min": 1.0, "max": 8.0, "type": float, "step": 0.5},
        "min_final_line_length":   {"min": 10.0, "max": 50.0, "type": float, "step": 5.0},
        "hough_lines_threshold":   {"min": 30,  "max": 100, "type": int,   "step": 5},
        "line_polygon_angle_tolerance_deg": {"min": 3.0, "max": 10.0, "type": float, "step": 0.5},
    }
    search_space_definitions = []
    for name, settings in params_to_tune.items():
        param_type = settings["type"]
        param_min = settings["min"]
        param_max = settings["max"]
        if param_type == int:
            space = Integer(param_min, param_max, name=name)
            search_space_definitions.append(space)
        elif param_type == float:
            space = Real(param_min, param_max, prior='uniform', name=name)
            search_space_definitions.append(space)
    dimension_names_list = [dim.name for dim in search_space_definitions] # يجب أن يكون معرفًا هنا

    # التحقق من وجود الصورة
    if not os.path.exists(external_image_path):
        print(f"\nERROR: Image file not found: '{external_image_path}'"); sys.exit(1)
    else:
        print(f"\nUsing external image: '{external_image_path}'")

    # --- تهيئة أفضل النتائج والراسم والشكل ---
    best_ssim_score_global: float = -1.1
    best_config_global = ShapeExtractor.DEFAULT_CONFIG.copy()
    best_equation_global: Optional[str] = None
    plotter_instance = ShapePlotter2D()
    reusable_fig, reusable_ax = plt.subplots(figsize=(6, 6))

    # --- دالة الهدف (Objective Function) ---
    @use_named_args(search_space_definitions)
    def objective_function(**params_dict) -> float:
        global best_ssim_score_global, best_config_global, best_equation_global
        trial_config = ShapeExtractor.DEFAULT_CONFIG.copy()
        trial_config.update(params_dict)
        current_params_str_list: List[str] = []
        for k_param, v_param in params_dict.items():
            param_str = f"{k_param}={v_param:.3f}" if isinstance(v_param,float) else f"{k_param}={v_param}"
            current_params_str_list.append(param_str)
        current_params_str = ", ".join(current_params_str_list)
        logger.info(f"--- Running Trial with Params: {current_params_str} ---")

        current_trial_ssim: float = -1.1
        extracted_eq_trial: Optional[str] = None
        try:
            extractor_trial = ShapeExtractor(config=trial_config)
            extracted_equation_trial = extractor_trial.extract_equation(external_image_path)
            if extracted_equation_trial:
                plotter_instance.components = []
                if plotter_instance.parser:
                    plotter_instance.parse_equation(extracted_equation_trial)
                    plot_fig_size = (6, 6)
                    plotter_instance.plot(ax=reusable_ax, show_plot=False, save_path=reconstructed_image_path, clear_before_plot=True)
                    if os.path.exists(reconstructed_image_path):
                         ssim_result = compare_images_ssim(external_image_path, reconstructed_image_path)
                         if ssim_result is not None: current_trial_ssim = ssim_result
                         else: current_trial_ssim = -1.1
                    else: current_trial_ssim = -1.1
                else: current_trial_ssim = -1.1
            else: current_trial_ssim = -1.1
        except Exception as trial_exception:
            logger.error(f"Error during objective function trial: {trial_exception}", exc_info=False)
            current_trial_ssim = -1.1

        logger.info(f"  Trial SSIM = {current_trial_ssim:.4f}")
        is_improvement = current_trial_ssim > best_ssim_score_global
        if is_improvement:
            logger.info(f"*** New Best SSIM Found: {current_trial_ssim:.4f} (Prev: {best_ssim_score_global:.4f}) ***")
            logger.info(f"   Achieved with: {current_params_str}")
            best_ssim_score_global = current_trial_ssim
            best_config_global = trial_config.copy()
            best_equation_global = extracted_equation_trial

        return -current_trial_ssim

    # --- تشغيل التحسين البايزي ---
    optimization_result = None
    if SKOPT_AVAILABLE and gp_minimize:
        print(f"\n--- Starting Bayesian Optimization ({n_calls_optimizer} calls, {n_initial_points_optimizer} initial) ---")
        LOCAL_RANDOM_SEED = 42 # <-- ** تعريف البذرة محليًا **
        try:
            optimization_result = gp_minimize(
                func=objective_function,
                dimensions=search_space_definitions,
                n_calls=n_calls_optimizer,
                n_initial_points=n_initial_points_optimizer,
                acq_func='EI',
                random_state=LOCAL_RANDOM_SEED, # <-- استخدام البذرة المحلية
                n_jobs=-1,
            )
        except Exception as opt_err:
            logger.error(f"Bayesian optimization failed: {opt_err}", exc_info=True)
            optimization_result = None
    else:
         logger.warning("\nOptimization skipped (scikit-optimize unavailable).")
         if best_ssim_score_global < -1.0: # Run baseline if needed
             logger.info("\n--- Running Baseline Extraction Only ---")
             try:
                 baseline_extractor = ShapeExtractor(config=best_config_global)
                 best_equation_global = baseline_extractor.extract_equation(external_image_path)
                 if best_equation_global and plotter_instance.parser:
                     plotter_instance.components = []
                     plotter_instance.parse_equation(best_equation_global)
                     baseline_figsize=(6,6)
                     plotter_instance.plot(ax=reusable_ax, show_plot=False, save_path=reconstructed_image_path, clear_before_plot=True)
                     if os.path.exists(reconstructed_image_path) and SKIMAGE_AVAILABLE:
                         ssim_base = compare_images_ssim(external_image_path, reconstructed_image_path)
                         if ssim_base is not None: best_ssim_score_global = ssim_base
             except Exception as base_err: logger.error(f"Error during baseline only run: {base_err}")

    # --- النتائج النهائية ---
    print("\n--- Optimization Finished ---")
    if optimization_result:
        best_params_values = optimization_result.x
        best_objective_value = optimization_result.fun
        best_ssim_from_optimizer = -best_objective_value
        print(f"Best SSIM score found by optimization: {best_ssim_from_optimizer:.4f}")
        print("Best Configuration Found by Optimizer:")
        best_config_from_optimizer = ShapeExtractor.DEFAULT_CONFIG.copy()
        param_idx = 0
        while param_idx < len(dimension_names_list):
            param_name_opt = dimension_names_list[param_idx]
            param_value_opt = best_params_values[param_idx]
            current_dimension_def = search_space_definitions[param_idx]
            if isinstance(current_dimension_def, Integer):
                 best_config_from_optimizer[param_name_opt] = int(round(param_value_opt))
            else:
                 best_config_from_optimizer[param_name_opt] = param_value_opt
            param_idx += 1
        key_idx_opt = 0
        config_keys_list_opt = list(best_config_from_optimizer.keys())
        while key_idx_opt < len(config_keys_list_opt):
             key_opt = config_keys_list_opt[key_idx_opt]
             value_opt = best_config_from_optimizer[key_opt]
             # ** تم نقل تعريف params_to_tune إلى الأعلى **
             is_tuned_opt = key_opt in params_to_tune
             is_relevant_opt = key_opt in ["deduplication_method", "remove_lines_within_polygons", "merge_corner_lines"]
             if is_tuned_opt or is_relevant_opt:
                  print(f"  {key_opt}: {value_opt}")
             key_idx_opt += 1
        print("\nRe-extracting equation with overall best found configuration...")
        try:
             final_extractor = ShapeExtractor(config=best_config_global) # Use overall best config
             final_equation_result = final_extractor.extract_equation(external_image_path)
             print("\nOverall Best Extracted Equation:")
             if final_equation_result: print(final_equation_result)
             else:
                  print("Extraction failed with overall best config.")
                  if best_equation_global: print("\nBest equation during trials:"); print(best_equation_global)
                  else: print("No valid equation found.")
        except Exception as final_extract_err:
             print(f"Error re-extracting: {final_extract_err}")
             print("\nBest Extracted Equation (from earlier iteration):")
             if best_equation_global: print(best_equation_global)
             else: print("No valid equation generated.")
    else:
        print(f"Bayesian optimization did not run or failed.")
        print(f"Best SSIM score from baseline/manual runs: {best_ssim_score_global:.4f}")
        print("Best Configuration Found (baseline):")
        key_idx_base = 0
        config_keys_base_list = list(best_config_global.keys())
        while key_idx_base < len(config_keys_base_list):
             key_base = config_keys_base_list[key_idx_base]
             value_base = best_config_global[key_base]
             # ** تم نقل تعريف params_to_tune إلى الأعلى **
             is_tuned_base = key_base in params_to_tune
             is_relevant_base = key_base in ["deduplication_method", "remove_lines_within_polygons", "merge_corner_lines"]
             if is_tuned_base or is_relevant_base: print(f"  {key_base}: {value_base}")
             key_idx_base += 1
        print("\nBest Extracted Equation (baseline):")
        if best_equation_global: print(best_equation_global)
        else: print("No valid equation generated.")

    # --- التنظيف النهائي ---
    if reusable_fig: plt.close(reusable_fig)
    if os.path.exists(reconstructed_image_path):
        try: os.remove(reconstructed_image_path); print(f"\nRemoved temp image: '{reconstructed_image_path}'")
        except OSError as e_remove: logger.warning(f"Cannot remove temp image: {e_remove}")

    print("\n" + "*" * 70)
    print(" Shape Extractor Optimization Loop Complete")
    print("*" * 70)
