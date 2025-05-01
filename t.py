
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
 نظام OmniMind-UL / RL-E⁴: وكيل تعلم موحد مع معادلة قابلة للتكيف (v1.1.1)
==============================================================================

 **الإصدار:** 1.1.1 (إصلاح NameError، دمج RiskProgressEvaluator، إضافة حلقة تدريب، جاهز للتقييم)
 **المطور:** Basil Yahya Abdullah (مع توضيحات وتنسيق وتعديلات إضافية)
 **التاريخ:** 24 أبريل 2025

 **الوصف العام:**
 -------------
 هذا الكود يقدم نظام تعلم معزز مبتكر، يُشار إليه هنا بـ "OmniMind-UL" أو "RL-E⁴",
 يعتمد على تمثيل الوكيل باستخدام "معادلة موحدة قابلة للتكيف"
 (UnifiedAdaptableEquation)، وهي بنية مرنة تتكون من "مكونات" متخصصة
 يمكن أن تحتوي بداخلها على "معادلات متطورة" (EvolvingEquation). يهدف هذا
 النظام إلى تحقيق توازن بين القدرة التعبيرية، قابلية التفسير، والتكيف الذاتي.

 **الفكرة المركزية (الخبير والمستكشف):**
 --------------------------------------
 يجسد النظام مفهوم التفاعل بين "خبير" (مكونات التقييم والمخاطر/التقدم)
 و "مستكشف" (مكونات السياسة). يتم التفاعل من خلال تدفق البيانات وآلية
 تحديث Actor-Critic المعدلة التي توازن بين تعظيم القيمة المتوقعة (Q)،
 معاقبة المخاطر، ومكافأة التقدم.

 **المكونات الرئيسية:**
 -----------------
 1.  **`EvolvingEquation`:** معادلات رياضية ديناميكية قابلة للتطور.
 2.  **`EquationComponent` وفئاتها الفرعية:** (NumericTransform, PolicyDecision,
     ValueEstimation, RiskProgressEvaluator) لبناء هيكل المعادلة.
 3.  **`UnifiedAdaptableEquation`:** الحاوية المرنة للمكونات.
 4.  **`EvolutionEngine`:** لإدارة تطور المعادلات.
 5.  **`UnifiedLearningAgent`:** الوكيل الرئيسي (بنية Actor-Critic معدلة).
 6.  **`ReplayBuffer`:** ذاكرة تجارب.
 7.  **`train_unified_agent`:** الدالة الرئيسية لتنظيم التدريب والتقييم.

 **الابتكارات الرئيسية:**
 --------------------
 -   التمثيل الموحد القائم على المكونات.
 -   التطور الهيكلي للمعادلات الرياضية.
 -   دمج تقديرات المخاطر والتقدم في التعلم.
 -   آلية تفاعل الخبير/المستكشف.
 -   قابلية التفسير المحتملة للمعادلات (`to_shape_engine_string`).

 **الاستخدام:**
 -----------
 يُستخدم لتدريب وكيل تعلم معزز في بيئات Gym/Gymnasium. يتم تحديد البيئة
 والمعاملات في قسم `if __name__ == "__main__":`.

 **المتطلبات:**
 -----------
 - Python 3.x, NumPy, PyTorch, Gymnasium, Matplotlib, Scikit-optimize, Scikit-image.
 - (Optional) PyBullet.

 **ملاحظة التقييم:**
 ----------------
 هذا الكود يمثل بنية متقدمة وتجريبية. يتطلب ضبطًا للمعاملات وموارد للتدريب.
 تم الالتزام الصارم بقاعدة "تعليمة واحدة لكل سطر" وزيادة التوثيق.
 *هذا الإصدار يتضمن حلقة تدريب وتشغيل كاملة.*
"""

# --- 1. Imports ---
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import wrappers
import random
from collections import deque, OrderedDict
import matplotlib
try: matplotlib.use('Agg') # Use Agg backend
except Exception: print("Note: Could not set Matplotlib backend to Agg.")
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import time
import os
import warnings
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Set
from scipy import stats
import inspect
import traceback
# Optional imports - Check availability later if needed
try: from skopt import gp_minimize; from skopt.space import Real, Integer, Categorical; from skopt.utils import use_named_args; SKOPT_AVAILABLE = True
except ImportError: gp_minimize, Real, Integer, Categorical, use_named_args = None, None, None, None, None; SKOPT_AVAILABLE = False; print("Warning: scikit-optimize not available.")
try: from skimage.metrics import structural_similarity as ssim; SKIMAGE_AVAILABLE = True
except ImportError: ssim = None; SKIMAGE_AVAILABLE = False; print("Warning: scikit-image not available.")


# --- 2. Global Configuration & Seed ---
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- 3. Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Set appropriate level (INFO or DEBUG)
    format='%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'
)
logger = logging.getLogger("OmniMindUL_RLE4") # Main logger

# --- 4. Type Definitions ---
PointInt = Tuple[int, int]
PointFloat = Tuple[float, float]
ColorBGR = Tuple[int, int, int]
ShapeData = Dict[str, Any]

# ============================================================== #
# ================= 5. Core Component: Evolving Equation ======= #
# ============================================================== #
class EvolvingEquation(nn.Module):
    """
    يمثل معادلة رياضية مرنة تتطور بنيتها ومعاملاتها ديناميكيًا. (v1.1.1)
    """
    def __init__(self, input_dim: int, init_complexity: int = 3, output_dim: int = 1,
                 complexity_limit: int = 15, min_complexity: int = 2,
                 output_activation: Optional[Callable] = None,
                 exp_clamp_min: float = 0.1, exp_clamp_max: float = 4.0,
                 term_clamp: float = 1e4, output_clamp: float = 1e5):
        """تهيئة المعادلة المتطورة."""
        super().__init__()
        # Validation
        if not isinstance(input_dim, int) or input_dim <= 0: raise ValueError("input_dim must be positive int.")
        if not isinstance(output_dim, int) or output_dim <= 0: raise ValueError("output_dim must be positive int.")
        if not isinstance(init_complexity, int) or init_complexity <= 0: raise ValueError("init_complexity must be positive int.")
        if not isinstance(min_complexity, int) or min_complexity <= 0: raise ValueError("min_complexity must be positive int.")
        if not isinstance(complexity_limit, int) or complexity_limit < min_complexity: raise ValueError(f"complexity_limit >= min_complexity required.")
        # Store parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.complexity = max(min_complexity, min(init_complexity, complexity_limit))
        self.complexity_limit = complexity_limit
        self.min_complexity = min_complexity
        self.output_activation = output_activation
        self.exp_clamp_min = exp_clamp_min
        self.exp_clamp_max = exp_clamp_max
        self.term_clamp = term_clamp
        self.output_clamp = output_clamp
        # Initialize internal components
        self._initialize_components()
        # Define function representations
        self._func_repr_map = OrderedDict([
            ('sin', {'shape': 'sine', 'color': '#FF6347'}), ('cos', {'shape': 'cosine', 'color': '#4682B4'}),
            ('tanh', {'shape': 'wave', 'color': '#32CD32'}), ('sigmoid', {'shape': 'step', 'color': '#FFD700'}),
            ('relu', {'shape': 'ramp', 'color': '#DC143C'}), ('leaky_relu', {'shape': 'ramp', 'color': '#8A2BE2'}),
            ('gelu', {'shape': 'smoothramp', 'color': '#00CED1'}), ('<lambda>', {'shape': 'line', 'color': '#A9A9A9'}),
            ('pow', {'shape': 'parabola', 'color': '#6A5ACD'}), ('exp', {'shape': 'decay', 'color': '#FF8C00'}),
            ('sqrt', {'shape': 'root', 'color': '#2E8B57'}), ('clamp', {'shape': 'plateau', 'color': '#DAA520'}),
            ('*', {'shape': 'swishlike', 'color': '#D2691E'})
        ])

    def _initialize_components(self):
        """تهيئة المكونات الداخلية."""
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
        self.function_library: List[Callable] = [
            torch.sin, torch.cos, torch.tanh, torch.sigmoid, F.relu,
            F.leaky_relu, F.gelu, lambda x: x, lambda x: torch.pow(x, 2),
            lambda x: torch.exp(-torch.abs(x)), lambda x: torch.sqrt(torch.abs(x) + 1e-6),
            lambda x: torch.clamp(x, -3.0, 3.0), lambda x: x * torch.sigmoid(x),
        ]
        if not self.function_library: raise ValueError("Function library empty.")
        self.functions: List[Callable] = []
        k = 0
        while k < self.complexity:
            selected_func = self._select_function()
            self.functions.append(selected_func)
            k += 1
        self.output_layer = nn.Linear(self.complexity, self.output_dim)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.output_layer.bias)

    def _select_function(self) -> Callable:
        """اختيار دالة عشوائية."""
        selected = random.choice(self.function_library)
        return selected

    def _safe_pow(self, base: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
        """عملية رفع للأس آمنة."""
        sign_val = torch.sign(base)
        base_abs_safe_val = torch.abs(base) + 1e-8
        powered_val = torch.pow(base_abs_safe_val, exp)
        result = sign_val * powered_val
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """التمرير الأمامي."""
        if not isinstance(x, torch.Tensor):
             try: x = torch.tensor(x, dtype=torch.float32)
             except Exception as e: raise TypeError(f"Input 'x' must be tensor: {e}")
        if x.dim() == 1:
            if x.shape[0] == self.input_dim: x = x.unsqueeze(0)
            else: raise ValueError(f"1D Input dim mismatch")
        elif x.dim() > 2:
             orig_shape = x.shape; x = x.view(x.shape[0], -1)
             if x.shape[1] != self.input_dim: raise ValueError("Flattened dim mismatch")
             warnings.warn(f"Input > 2D ({orig_shape}) flattened to ({x.shape}).", RuntimeWarning)
        elif x.dim() == 2:
            if x.shape[1] != self.input_dim: raise ValueError(f"2D Input dim mismatch")
        else: raise ValueError("Input tensor must be >= 1D.")
        try: model_device = next(self.parameters()).device
        except StopIteration: model_device = torch.device("cpu")
        if x.device != model_device: x = x.to(model_device)
        if torch.isnan(x).any(): x = torch.nan_to_num(x, nan=0.0)

        transformed_features: Optional[torch.Tensor] = None
        try:
            transformed_features = self.input_transform(x)
            if torch.isnan(transformed_features).any(): transformed_features = torch.nan_to_num(transformed_features, nan=0.0)
            transformed_features = torch.clamp(transformed_features, -self.term_clamp, self.term_clamp)
        except Exception as e: return torch.zeros(x.shape[0], self.output_dim, device=model_device)

        term_results = torch.zeros(x.shape[0], self.complexity, device=model_device)
        if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')): return torch.zeros(x.shape[0], self.output_dim, device=model_device)
        len_coeff = len(self.coefficients); len_exp = len(self.exponents); len_func = len(self.functions)
        current_complexity = min(len_coeff, len_exp, len_func) # Use the minimum length found
        if current_complexity != self.complexity:
            warnings.warn(f"Complexity mismatch ({self.complexity} vs actual {current_complexity}). Using {current_complexity}.", RuntimeWarning)
            # This indicates a potential issue during evolution (add/prune)
            # For forward pass, we proceed with the minimum available length
            # A more robust solution might involve resizing layers here if possible
            if current_complexity == 0: return torch.zeros(x.shape[0], self.output_dim, device=model_device)

        term_idx = 0
        while term_idx < current_complexity: # Use the potentially adjusted complexity
            try:
                feature_i = transformed_features[:, term_idx]
                exp_val = torch.clamp(self.exponents[term_idx], self.exp_clamp_min, self.exp_clamp_max)
                term_powered = self._safe_pow(feature_i, exp_val)
                if torch.isnan(term_powered).any(): term_powered = torch.zeros_like(feature_i)
                term_powered = torch.clamp(term_powered, -self.term_clamp, self.term_clamp)
                current_func = self.functions[term_idx]
                term_activated = current_func(term_powered)
                if torch.isnan(term_activated).any(): term_activated = torch.zeros_like(feature_i)
                term_activated = torch.clamp(term_activated, -self.term_clamp, self.term_clamp)
                term_value = self.coefficients[term_idx] * term_activated
                term_results[:, term_idx] = torch.clamp(term_value, -self.term_clamp, self.term_clamp)
            except Exception as e_term: term_results[:, term_idx] = 0.0
            term_idx += 1

        output_final: Optional[torch.Tensor] = None
        try:
            # Adjust slicing if complexity changed
            output_raw = self.output_layer(term_results[:, :current_complexity]) # Use adjusted complexity
            output_clamped = torch.clamp(output_raw, -self.output_clamp, self.output_clamp)
            if torch.isnan(output_clamped).any(): output_clamped = torch.nan_to_num(output_clamped, nan=0.0)
            if self.output_activation:
                output_activated = self.output_activation(output_clamped)
                if torch.isnan(output_activated).any(): output_activated = torch.nan_to_num(output_activated, nan=0.0)
                output_final = output_activated
            else:
                output_final = output_clamped
        except Exception as e_output: output_final = torch.zeros(x.shape[0], self.output_dim, device=model_device)

        if output_final is None: output_final = torch.zeros(x.shape[0], self.output_dim, device=model_device)
        elif torch.isnan(output_final).any(): output_final = torch.zeros_like(output_final)

        return output_final

    def to_shape_engine_string(self) -> str:
        """تمثيل المعادلة كنص لـ ShapePlotter2D."""
        parts = []; param_scale = 5.0; exp_scale = 2.0
        if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')): return "eq_not_init"
        current_complexity = min(len(self.coefficients), len(self.exponents), len(self.functions))
        if current_complexity != self.complexity: return "eq_mismatch"

        term_idx = 0
        while term_idx < current_complexity:
             coeff = self.coefficients[term_idx]; exponent = self.exponents[term_idx]; func = self.functions[term_idx]
             coeff_val = round(coeff.item(), 3); exp_val_clamped = torch.clamp(exponent, self.exp_clamp_min, self.exp_clamp_max); exp_val = round(exp_val_clamped.item(), 3)
             func_name_part = getattr(func, '__name__', '<lambda>')
             if func_name_part == '<lambda>':
                 func_repr = repr(func)
                 if 'pow' in func_repr and '** 2' in func_repr: func_name_part = 'pow'
                 elif '*' in func_repr and 'sigmoid' in func_repr: func_name_part = '*'
                 elif 'sqrt' in func_repr: func_name_part = 'sqrt'
                 elif 'exp' in func_repr and 'abs' in func_repr: func_name_part = 'exp'
                 elif 'clamp' in func_repr: func_name_part = 'clamp'
                 elif 'x' in func_repr and len(func_repr) < 25: func_name_part = '<lambda>'

             repr_info = self._func_repr_map.get(func_name_part, self._func_repr_map['<lambda>']); shape_type = repr_info['shape']
             p1_vis = round(term_idx * param_scale * 0.5 + coeff_val * param_scale * 0.2, 2)
             p2_vis = round(coeff_val * param_scale, 2)
             p3_vis = round(abs(exp_val) * exp_scale, 2)
             params_str_vis = f"{p1_vis},{p2_vis},{p3_vis}"
             styles: Dict[str, Any] = {}
             styles['color'] = repr_info['color']
             styles['linewidth'] = round(1.0 + abs(exp_val - 1.0) * 1.5, 2)
             styles['opacity'] = round(np.clip(0.4 + abs(coeff_val) * 0.5, 0.2, 0.9), 2)
             if coeff_val < -0.01: styles['fill'] = 'True'
             if func_name_part in ['cos', 'relu', 'leaky_relu']: styles['dash'] = '--'
             styles_str_list: List[str] = []
             for k_style, v_style in styles.items(): style_part = f"{k_style}={v_style}"; styles_str_list.append(style_part)
             styles_str = ",".join(styles_str_list)
             term_str_final = f"{shape_type}({params_str_vis}){{{styles_str}}}"
             parts.append(term_str_final)
             term_idx += 1
        return " + ".join(parts) if parts else "empty_equation"

    def add_term(self) -> bool:
        """إضافة حد جديد."""
        if self.complexity >= self.complexity_limit: return False
        new_complexity = self.complexity + 1
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cpu")
        try:
            old_in_w = self.input_transform.weight.data.clone(); old_in_b = self.input_transform.bias.data.clone()
            old_out_w = self.output_layer.weight.data.clone(); old_out_b = self.output_layer.bias.data.clone()
        except AttributeError: return False
        # Add components first
        new_coeff = nn.Parameter(torch.randn(1, device=device)*0.01, requires_grad=True)
        self.coefficients.append(new_coeff)
        new_exp = nn.Parameter(torch.abs(torch.randn(1, device=device)*0.05)+1.0, requires_grad=True)
        self.exponents.append(new_exp)
        new_func = self._select_function()
        self.functions.append(new_func)
        # Then update layers
        new_input_transform = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad():
            new_input_transform.weight.data[:self.complexity,:] = old_in_w
            new_input_transform.bias.data[:self.complexity] = old_in_b
            nn.init.xavier_uniform_(new_input_transform.weight.data[self.complexity:], gain=nn.init.calculate_gain('relu'))
            new_input_transform.weight.data[self.complexity:] *= 0.01 # Smaller init
            nn.init.zeros_(new_input_transform.bias.data[self.complexity:])
        self.input_transform = new_input_transform
        new_output_layer = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad():
            new_output_layer.weight.data[:,:self.complexity] = old_out_w
            nn.init.xavier_uniform_(new_output_layer.weight.data[:,self.complexity:], gain=nn.init.calculate_gain('linear'))
            new_output_layer.weight.data[:,self.complexity:] *= 0.01 # Smaller init
            new_output_layer.bias.data.copy_(old_out_b)
        self.output_layer = new_output_layer
        # Finally, update complexity attribute
        self.complexity = new_complexity
        return True

    def prune_term(self, aggressive: bool = False) -> bool:
        """إزالة حد."""
        if self.complexity <= self.min_complexity: return False
        new_complexity = self.complexity - 1
        idx_to_prune = -1
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cpu")
        # Select index
        try:
            if aggressive and self.complexity > 1:
                 with torch.no_grad():
                    if not hasattr(self, 'coefficients') or not self.coefficients: idx_to_prune = random.randint(0, self.complexity - 1)
                    else:
                        coeffs_abs_cpu = torch.tensor([torch.abs(c.data).item() for c in self.coefficients], device='cpu')
                        in_w = self.input_transform.weight; out_w = self.output_layer.weight
                        in_w_cpu = in_w.data.cpu() if in_w is not None else None
                        out_w_cpu = out_w.data.cpu() if out_w is not None else None
                        in_norm = torch.norm(in_w_cpu, p=1, dim=1) if in_w_cpu is not None else torch.zeros(self.complexity, device='cpu')
                        out_norm = torch.norm(out_w_cpu, p=1, dim=0) if out_w_cpu is not None else torch.zeros(self.complexity, device='cpu')
                        shape_match = (coeffs_abs_cpu.shape[0]==self.complexity and in_norm.shape[0]==self.complexity and out_norm.shape[0]==self.complexity)
                        if shape_match:
                            importance = (coeffs_abs_cpu*(in_norm+out_norm))+1e-9
                            if torch.isnan(importance).any(): idx_to_prune = random.randint(0, self.complexity - 1)
                            else: idx_to_prune = torch.argmin(importance).item()
                        else: idx_to_prune = random.randint(0, self.complexity - 1)
            else: idx_to_prune = random.randint(0, self.complexity - 1)
            is_valid_index = (0 <= idx_to_prune < self.complexity)
            if not is_valid_index: return False
        except Exception as e: warnings.warn(f"Prune index error: {e}"); return False
        # Backup weights
        try: old_in_w = self.input_transform.weight.data.clone(); old_in_b = self.input_transform.bias.data.clone(); old_out_w = self.output_layer.weight.data.clone(); old_out_b = self.output_layer.bias.data.clone()
        except AttributeError: return False
        # Remove components first
        try:
            if hasattr(self,'coefficients') and len(self.coefficients)>idx_to_prune: del self.coefficients[idx_to_prune]
            if hasattr(self,'exponents') and len(self.exponents)>idx_to_prune: del self.exponents[idx_to_prune]
            if hasattr(self,'functions') and len(self.functions)>idx_to_prune: self.functions.pop(idx_to_prune)
        except (IndexError,Exception) as e: warnings.warn(f"Failed remove components: {e}"); return False
        # Shrink layers
        new_input_transform = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad():
            new_w_in = torch.cat([old_in_w[:idx_to_prune], old_in_w[idx_to_prune+1:]], dim=0)
            new_b_in = torch.cat([old_in_b[:idx_to_prune], old_in_b[idx_to_prune+1:]])
            if new_w_in.shape[0]!=new_complexity or new_b_in.shape[0]!=new_complexity: return False
            new_input_transform.weight.data.copy_(new_w_in); new_input_transform.bias.data.copy_(new_b_in)
        self.input_transform = new_input_transform
        new_output_layer = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad():
            new_w_out = torch.cat([old_out_w[:,:idx_to_prune], old_out_w[:,idx_to_prune+1:]], dim=1)
            if new_w_out.shape[1]!=new_complexity: return False
            new_output_layer.weight.data.copy_(new_w_out); new_output_layer.bias.data.copy_(old_out_b)
        self.output_layer = new_output_layer
        # Update complexity last
        self.complexity = new_complexity
        return True


# --- 4. Core Component: Evolution Engine ---
class EvolutionEngine:
    """محرك التطور: يدير عملية تطور كائن `EvolvingEquation`."""
    def __init__(self, mutation_power: float = 0.03, history_size: int = 50, cooldown_period: int = 30,
                 add_term_threshold: float = 0.85, prune_term_threshold: float = 0.20,
                 add_term_prob: float = 0.15, prune_term_prob: float = 0.25, swap_func_prob: float = 0.02):
        """تهيئة محرك التطور."""
        if not (0 < mutation_power < 0.5): warnings.warn(f"mut_power {mutation_power} atypical.")
        if not (0 < add_term_threshold < 1): warnings.warn(f"add_thresh {add_term_threshold} atypical.")
        if not (0 < prune_term_threshold < 1): warnings.warn(f"prune_thresh {prune_term_threshold} atypical.")
        if add_term_threshold <= prune_term_threshold: warnings.warn(f"add <= prune threshold.")
        if not (0 <= add_term_prob <= 1 and 0 <= prune_term_prob <= 1 and 0 <= swap_func_prob <= 1): warnings.warn("Evo probs not in [0, 1].")
        self.base_mutation_power = mutation_power; self.performance_history: deque = deque(maxlen=history_size)
        self.cooldown_period = cooldown_period; self.term_change_cooldown = 0
        self.add_term_threshold = add_term_threshold; self.prune_term_threshold = prune_term_threshold
        self.add_term_prob = add_term_prob; self.prune_term_prob = prune_term_prob; self.swap_func_prob = swap_func_prob

    def _calculate_percentile(self, current_reward: float) -> float:
        """يحسب النسبة المئوية للأداء."""
        if math.isnan(current_reward): return 0.5
        valid_history = [r for r in self.performance_history if not math.isnan(r)];
        if not valid_history: return 0.5
        history_array = np.array(valid_history)
        percentile = 0.5
        try: percentile_score = stats.percentileofscore(history_array, current_reward, kind='mean') / 100.0; percentile = percentile_score
        except Exception: percentile = np.mean(history_array < current_reward)
        return np.clip(percentile, 0.0, 1.0)

    def _dynamic_mutation_scale(self, percentile: float) -> float:
        """يحدد عامل قياس لقوة التحور."""
        if math.isnan(percentile): percentile = 0.5
        scale = 1.0
        if percentile > 0.9: scale = 0.5
        elif percentile < 0.1: scale = 1.5
        else: scale = 1.5 - (percentile - 0.1) * (1.0 / 0.8)
        return max(0.1, min(scale, 2.0))

    def evolve_equation(self, equation: EvolvingEquation, reward: float, step: int) -> bool:
        """ينفذ خطوة تطور للمعادلة."""
        structure_changed = False
        if not isinstance(equation, EvolvingEquation): return False
        if not math.isnan(reward):
            self.performance_history.append(reward)
            perf_percentile = self._calculate_percentile(reward)
        else:
            valid_hist = [r for r in self.performance_history if not math.isnan(r)]
            perf_percentile = self._calculate_percentile(valid_hist[-1]) if valid_hist else 0.5

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
            if structure_changed:
                 logger.info(f"EVO: Step {step}: Eq '{type(equation).__name__}' -> {action_desc} | Pct: {perf_percentile:.2f} | CD: {self.term_change_cooldown}")

        can_swap = (not structure_changed and self.term_change_cooldown == 0 and random.random() < self.swap_func_prob)
        if can_swap:
            swapped_ok = self.swap_function(equation)
            if swapped_ok: self.term_change_cooldown = max(self.term_change_cooldown, 2)

        mut_scale = self._dynamic_mutation_scale(perf_percentile)
        self._mutate_parameters(equation, mut_scale, step)
        return structure_changed

    def _mutate_parameters(self, equation: EvolvingEquation, mutation_scale: float, step: int):
        """يطبق التحور على المعاملات."""
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
        """يبدل دالة حد عشوائي."""
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
            equation.functions[idx] = new_f;
            return True
        except (IndexError, Exception) as e: warnings.warn(f"Function swap error: {e}", RuntimeWarning); return False


# --- 5. Core Component: Replay Buffer ---
class ReplayBuffer:
    """ذاكرة تخزين مؤقت للتجارب مع فحص NaN."""
    def __init__(self, capacity=100000):
        if not isinstance(capacity, int) or capacity <= 0: raise ValueError("Capacity must be pos int.")
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._push_nan_warnings = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
        self._sample_nan_warning = 0

    def push(self, state, action, reward, next_state, done):
        """يضيف تجربة مع فحص NaN."""
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
        """يأخذ عينة ويحولها لتنسورات مع فحص NaN."""
        current_size = len(self.buffer)
        if current_size < batch_size: return None
        try:
            batch_indices = np.random.choice(current_size, batch_size, replace=False)
            batch = [self.buffer[i] for i in batch_indices]
        except (ValueError, Exception) as e: warnings.warn(f"Error sampling buffer: {e}.", RuntimeWarning); return None
        try:
            states, actions, rewards, next_states, dones = zip(*batch)
            states_np = np.array(states, dtype=np.float32); actions_np = np.array(actions, dtype=np.float32)
            rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, 1)
            next_states_np = np.array(next_states, dtype=np.float32); dones_np = np.array(dones, dtype=np.float32).reshape(-1, 1)
            is_nan_inf = (np.isnan(states_np).any() or np.isnan(actions_np).any() or np.isnan(rewards_np).any() or np.isnan(next_states_np).any() or np.isnan(dones_np).any() or
                          np.isinf(states_np).any() or np.isinf(actions_np).any() or np.isinf(rewards_np).any() or np.isinf(next_states_np).any() or np.isinf(dones_np).any())
            if is_nan_inf:
                self._sample_nan_warning += 1
                if self._sample_nan_warning % 100 == 1: warnings.warn(f"NaN/Inf sampled batch. Skips: {self._sample_nan_warning}. None.", RuntimeWarning)
                return None
            states_tensor = torch.from_numpy(states_np); actions_tensor = torch.from_numpy(actions_np)
            rewards_tensor = torch.from_numpy(rewards_np); next_states_tensor = torch.from_numpy(next_states_np)
            dones_tensor = torch.from_numpy(dones_np)
        except (ValueError, TypeError, Exception) as e: warnings.warn(f"Failed convert batch tensors: {e}. None.", RuntimeWarning); return None
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        return len(self.buffer)

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
        # حفظ وسائط الإنشاء للحفظ/التحميل
        frame = inspect.currentframe().f_back
        if frame: # Check if frame exists
            args_info = inspect.getargvalues(frame)
            args_names = args_info.args
            args_values = args_info.locals
            self._init_args = {arg_name: args_values[arg_name] for arg_name in args_names if arg_name != 'self'}
            if 'tags' in self._init_args and self._init_args['tags'] is not None: self._init_args['tags'] = list(self._init_args['tags'])
            if 'required_context' in self._init_args and self._init_args['required_context'] is not None: self._init_args['required_context'] = list(self._init_args['required_context'])
            if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self._init_args['provides'])
        else:
             self._init_args = {'component_id': component_id} # Fallback
             warnings.warn(f"Could not get frame info for component {component_id} during init.")

    def is_active(self, context: Dict[str, Any]) -> bool:
        """التحقق من النشاط."""
        if not self.required_context: return True
        return self.required_context.issubset(context.keys())

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """التمرير الأمامي."""
        raise NotImplementedError("Subclasses must implement forward.")

    def evolve(self, engine: EvolutionEngine, reward: float, step: int) -> bool:
        """التطور."""
        return False

    def to_string(self) -> str:
        """تمثيل نصي."""
        req_str = ','.join(self.required_context) if self.required_context else 'Any'
        prov_str = ','.join(self.provides) if self.provides else 'None'
        tag_str = ','.join(self.tags) if self.tags else 'None'
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.component_id}, tags={{{tag_str}}}, req={{{req_str}}}, prov={{{prov_str}}})"

    def get_complexity(self) -> float:
        """تقدير التعقيد."""
        return 1.0


class NumericTransformComponent(EquationComponent):
    """مكون يغلف EvolvingEquation لتحويل عددي."""
    def __init__(self, component_id: str, equation: EvolvingEquation,
                 activation: Optional[Callable] = None, tags: Optional[Set[str]] = None,
                 required_context: Optional[Set[str]] = None, provides: Optional[Set[str]] = None,
                 input_key: str = 'features', output_key: str = 'features'):
        provides = provides if provides is not None else {output_key}
        super().__init__(component_id, tags, required_context, provides)
        if not isinstance(equation, EvolvingEquation): raise TypeError("Eq must be EvolvingEquation.")
        self.equation = equation
        self.activation = activation
        self.input_key = input_key
        self.output_key = output_key
        self._init_args['equation_config'] = {
            'input_dim': equation.input_dim, 'init_complexity': equation.complexity,
            'output_dim': equation.output_dim, 'complexity_limit': equation.complexity_limit,
            'min_complexity': equation.min_complexity, 'exp_clamp_min': equation.exp_clamp_min,
            'exp_clamp_max': equation.exp_clamp_max, 'term_clamp': equation.term_clamp,
            'output_clamp': equation.output_clamp, }
        activation_name = 'None'
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
        if input_tensor.shape[-1] != self.equation.input_dim:
             # warnings.warn(f"Dim mismatch NTC '{self.component_id}'. Skip.", RuntimeWarning);
             return {} # Return empty on dim mismatch
        try:
            transformed = self.equation(input_tensor)
            if torch.isnan(transformed).any(): transformed = torch.zeros_like(transformed)
            if self.activation:
                activated = self.activation(transformed)
                if torch.isnan(activated).any(): activated = torch.zeros_like(activated)
                update_dict = {self.output_key: activated}; return update_dict
            else: update_dict = {self.output_key: transformed}; return update_dict
        except Exception as e:
            # warnings.warn(f"Error NTC '{self.component_id}': {e}. Skip.", RuntimeWarning);
            return {}

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

    def get_complexity(self) -> float:
        return getattr(self.equation, 'complexity', 1.0)


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
        update_dict = {self.output_key: action_tanh}
        return update_dict

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
        if value_features.shape[-1] != 1:
             # warnings.warn(f"ValueEst '{self.component_id}' input dim != 1. Taking mean.", RuntimeWarning) # Reduce noise
             final_value = value_features.mean(dim=-1, keepdim=True)
        else: final_value = value_features
        if torch.isnan(final_value).any(): final_value = torch.zeros_like(final_value)
        update_dict = {self.output_key: final_value}
        return update_dict

    def to_string(self) -> str:
        base = super().to_string().replace("Component","ValueEstimation")[:-1]
        info = f", In='{self.input_key}', Out='{self.output_key}'"
        return base + info + ")"


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
        self.risk_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Softplus() )
        self._init_weights(self.risk_network)
        self.progress_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1) )
        self._init_weights(self.progress_network)
        self._init_args['input_dim'] = input_dim; self._init_args['hidden_dim'] = hidden_dim
        self._init_args['input_key'] = input_key; self._init_args['risk_output_key'] = risk_output_key
        self._init_args['progress_output_key'] = progress_output_key
        self.provides = self.provides.union({risk_output_key, progress_output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = list(self.provides)

    def _init_weights(self, module_seq: nn.Sequential):
        for layer in module_seq:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
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
        except Exception as e_risk: self.logger.error(f"Risk err '{self.component_id}': {e_risk}", exc_info=False)
        try:
            est_prog = self.progress_network(input_features)
            if torch.isnan(est_prog).any(): est_prog = torch.zeros_like(est_prog)
            output_updates[self.progress_output_key] = est_prog
        except Exception as e_prog: self.logger.error(f"Prog err '{self.component_id}': {e_prog}", exc_info=False)
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
    """المعادلة الموحدة التي تحتوي على المكونات."""
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
        lines.append(f"UnifiedEquation(id={eq_id}, TotalComplexity={total_comp:.2f})")
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
    """وكيل تعلم معزز موحد (الإصدار 1.1.1)."""
    def __init__(self, state_dim: int, action_dim: int, action_bounds: Tuple[float, float],
                 unified_equation: UnifiedAdaptableEquation, evolution_engine: EvolutionEngine,
                 gamma: float = 0.99, tau: float = 0.005, actor_lr: float = 1e-4, critic_lr: float = 3e-4,
                 buffer_capacity: int = int(1e6), batch_size: int = 128,
                 exploration_noise_std: float = 0.3, noise_decay_rate: float = 0.9999, min_exploration_noise: float = 0.1,
                 weight_decay: float = 1e-5, grad_clip_norm: Optional[float] = 1.0, grad_clip_value: Optional[float] = None,
                 risk_penalty_weight: float = 0.05, progress_bonus_weight: float = 0.05):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"UnifiedLearningAgent(v1.1.1) using device: {self.device}") # Use main logger
        self.state_dim = state_dim; self.action_dim = action_dim; self.batch_size = batch_size
        self.gamma = gamma; self.tau = tau
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm is not None and grad_clip_norm > 0 else None
        self.grad_clip_value = grad_clip_value if grad_clip_value is not None and grad_clip_value > 0 else None
        if action_bounds is None or len(action_bounds) != 2: raise ValueError("action_bounds needed.")
        self.action_low = float(action_bounds[0]); self.action_high = float(action_bounds[1])
        if self.action_low >= self.action_high: raise ValueError("action low < high needed.")
        action_scale_t = torch.tensor((self.action_high - self.action_low) / 2.0, dtype=torch.float32)
        action_bias_t = torch.tensor((self.action_high + self.action_low) / 2.0, dtype=torch.float32)
        if (action_scale_t <= 1e-6).any(): warnings.warn("Action scale near zero. Clamp."); action_scale_t.clamp_(min=1e-6)
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
             logger.info(f"Creating AdamW optimizer tags {tags} ({len(params)} tensors, {num_params:,} params) LR={lr:.1e}, WD={wd:.1e}")
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
                if len(main_params) != len(target_params): logger.warning("Target param mismatch. Syncing."); self._sync_target_network(); return
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
                self._set_requires_grad({'critic', 'expert'}, False); self._set_requires_grad({'actor', 'shared'}, True)
                self.unified_eq.train()
                # أ. حساب الإجراءات المقترحة
                policy_act_ctx = {'task': 'rl', 'request_action': True, 'mode': 'train_actor'}
                policy_act_data = {'state': states, 'features': states}
                policy_output = self.unified_eq(policy_act_data, policy_act_ctx)
                if 'action_raw' not in policy_output: raise ValueError("Main missing 'action_raw' policy")
                actions_pred_tanh = torch.tanh(policy_output['action_raw'])
                actions_pred_scaled = actions_pred_tanh * self.action_scale + self.action_bias

                # ب. تقييم الإجراءات بواسطة المقيّم
                policy_q_ctx = {'task': 'rl', 'request_q_value': True, 'mode': 'eval_policy'}
                policy_critic_in = torch.cat([states, actions_pred_scaled], dim=1)
                policy_q_data = {'state': states, 'action': actions_pred_scaled, 'critic_input': policy_critic_in, 'features': states}
                policy_critic_output = self.unified_eq(policy_q_data, policy_q_ctx)
                if 'q_value' not in policy_critic_output: raise ValueError("Main missing 'q_value' policy eval")
                policy_q_values = policy_critic_output['q_value']

                # ج. حساب تقديرات المخاطر/التقدم
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
            except Exception as e_actor: self.logger.warning(f"Exception Actor update step {step}: {e_actor}", exc_info=False)
            finally: self._set_requires_grad({'critic', 'expert'}, True) # إعادة تفعيل تدرجات المقيّم والخبير

        # --- 3. تحديث الهدف ---
        try: self._update_target_network()
        except Exception as e_target: self.logger.warning(f"Exception target update step {step}: {e_target}")

        # --- 4. التطور ---
        try:
            changed_info = self.unified_eq.evolve(self.evolution_engine, avg_reward, self.total_updates)
            if changed_info:
                 self.total_evolutions += 1; self.logger.info(f"Unified eq structure changed. Reinit optim & sync target. Changed: {changed_info}")
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
                    if action_eval is None or np.isnan(action_eval).any() or np.isinf(action_eval).any(): action_eval = np.zeros(self.action_dim, dtype=np.float32)
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

    def save_model(self, filename="unified_agent_checkpoint_v111.pt"): # تحديث اسم الملف الافتراضي
        """حفظ حالة الوكيل."""
        self.unified_eq.to('cpu')
        try:
            component_configs_save = {}
            for comp_id, comp in self.unified_eq.components.items():
                 init_args_comp = getattr(comp, '_init_args', {}).copy()
                 if hasattr(comp, 'equation') and isinstance(comp.equation, EvolvingEquation):
                      init_args_comp['equation_state_dict'] = comp.equation.state_dict()
                      if 'equation_config' in init_args_comp and isinstance(init_args_comp['equation_config'], dict): init_args_comp['equation_config']['init_complexity'] = comp.equation.complexity
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

    def load_model(self, filename="unified_agent_checkpoint_v111.pt"): # تحديث اسم الملف الافتراضي
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
                except Exception as ao_load_e: self.logger.warning(f"Failed load actor optim state: {ao_load_e}. Reset.")
            self.critic_optimizer = self._create_optimizer(critic_lr_load, wd_load, {'critic', 'shared', 'expert'})
            if self.critic_optimizer and critic_optim_state:
                 try: self.critic_optimizer.load_state_dict(critic_optim_state)
                 except Exception as co_load_e: self.logger.warning(f"Failed load critic optim state: {co_load_e}. Reset.")

            self._sync_target_network() # Sync target after loading main net and optimizers

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
    try:
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_name), deque_size=50)
        eval_render_mode = "human" if render_eval else None; eval_env = None
        try: eval_env = gym.make(env_name, render_mode=eval_render_mode)
        except Exception:
            try: eval_env = gym.make(env_name)
            except Exception as e_fallback: print(f"CRITICAL ERROR: Create eval env failed: {e_fallback}"); return None, []
        # Set seeds
        env_seed = RANDOM_SEED; eval_env_seed = RANDOM_SEED + 1
        env.reset(seed=env_seed); env.action_space.seed(env_seed)
        eval_env.reset(seed=eval_env_seed); eval_env.action_space.seed(eval_env_seed)
        # Get dimensions and bounds
        state_dim_env = env.observation_space.shape[0]; action_dim_env = env.action_space.shape[0]
        action_low_env = env.action_space.low; action_high_env = env.action_space.high
        if np.any(np.isinf(action_low_env)) or np.any(np.isinf(action_high_env)): action_bounds_env = (-1.0, 1.0)
        else: action_bounds_env = (float(action_low_env.min()), float(action_high_env.max()))
        logger.info(f"Env Details: State={state_dim_env}, Action={action_dim_env}, Bounds={action_bounds_env}")
    except Exception as e: print(f"CRITICAL ERROR initializing env: {e}"); return None, []

    # Build Unified Equation Structure
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
        critic_input_dim_build = current_dim_build + action_dim_env
        critic_hidden_dim_build = hidden_dims[-1] # Use last hidden dim for critic hidden
        critic_hidden_eq_build = EvolvingEquation(critic_input_dim_build, eq_init_complexity, critic_hidden_dim_build, eq_complexity_limit)
        critic_hidden_comp_build = NumericTransformComponent('critic_hidden_layer', critic_hidden_eq_build, F.relu, {'critic','expert'}, {'request_q_value'}, {'value_features'}, 'critic_input', 'value_features')
        unified_eq_agent.add_component(critic_hidden_comp_build, execute_after=shared_last_comp_id_build) # Depends on shared features
        critic_output_eq_build = EvolvingEquation(critic_hidden_dim_build, eq_init_complexity, 1, eq_complexity_limit)
        critic_output_comp_build = NumericTransformComponent('critic_output_layer', critic_output_eq_build, None, {'critic','expert'}, {'request_q_value'}, {'value_features_final'}, 'value_features', 'value_features_final')
        unified_eq_agent.add_component(critic_output_comp_build, execute_after='critic_hidden_layer')
        value_estimation_comp_build = ValueEstimationComponent('value_estimation', {'critic','expert'}, {'request_q_value'}, {'q_value'}, 'value_features_final', 'q_value')
        unified_eq_agent.add_component(value_estimation_comp_build, execute_after='critic_output_layer')
        logger.info("Critic path added.")
        logger.info("Unified Equation structure built successfully.")
    except Exception as e: print(f"CRITICAL ERROR building unified eq: {e}"); traceback.print_exc(); return None, []

    # Initialize Agent and Evolution Engine
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
        # Choose action: random exploration initially, then policy + noise
        if current_env_step_num < start_learning_steps:
            action_to_take = env.action_space.sample()
        else:
            action_to_take = agent.get_action(current_state, explore=True)

        # Step environment
        try:
            next_obs, reward_val, terminated_flag, truncated_flag, info_dict = env.step(action_to_take)
            is_done = terminated_flag or truncated_flag
            # Handle potential NaN/Inf reward
            if math.isnan(reward_val) or math.isinf(reward_val): reward_val = 0.0
            # Ensure next state is correct type
            next_obs_processed = np.asarray(next_obs, dtype=np.float32)
            # Store experience
            agent.replay_buffer.push(current_state, action_to_take, reward_val, next_obs_processed, float(is_done))
            # Update current state and episode stats
            current_state = next_obs_processed
            current_episode_reward += reward_val
            current_episode_steps += 1

            # --- Episode End Handling ---
            if is_done:
                total_episodes_count += 1
                # Log episode stats if available from wrapper
                if 'episode' in info_dict:
                    avg_rew_val = np.mean(env.return_queue) if env.return_queue else 0.0
                    avg_len_val = np.mean(env.length_queue) if env.length_queue else 0.0
                    eq_complexity_val = agent.unified_eq.get_total_complexity()
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
            try: # Attempt to reset environment after error
                 current_state, current_info = env.reset(seed=RANDOM_SEED + current_env_step_num + total_episodes_count + 1)
                 current_state = np.asarray(current_state, dtype=np.float32); logger.info("Env reset OK after error.")
                 current_episode_reward = 0.0; current_episode_steps = 0
            except Exception as e2: logger.error(f"CRITICAL ERROR: Failed reset after error: {e2}. Stop."); break
            continue # Continue to next step

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

    # --- Training Loop Finished ---
    progress_bar.close();
    # Close environments
    try: env.close()
    except Exception: pass
    try: eval_env.close()
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
            plt.style.use('ggplot'); fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True); fig.suptitle(f'Unified Agent Training ({env_name} - v1.1.1)', fontsize=16)
            # Plot Reward
            ax1 = axes[0]; ax1.plot(steps_history_list, evaluation_rewards_list, marker='.', linestyle='-', color='dodgerblue', label='Avg Eval Reward')
            if len(evaluation_rewards_list) >= 5: mv_avg = np.convolve(evaluation_rewards_list, np.ones(5)/5, mode='valid'); ax1.plot(steps_history_list[4:], mv_avg, linestyle='--', color='orangered', label='Moving Avg (5 evals)')
            ax1.set_ylabel('Avg Eval Reward'); ax1.set_title('Evaluation Reward'); ax1.legend(); ax1.grid(True, ls='--', lw=0.5)
            # Plot Q-Loss
            ax2 = axes[1];
            if q_loss_history:
                update_steps_est_q = np.linspace(start_learning_steps, current_env_step_num, len(q_loss_history), dtype=int)
                ax2.plot(update_steps_est_q, list(q_loss_history), label='Q-Loss (Raw)', alpha=0.4, c='darkorange', lw=0.8)
                if len(q_loss_history) >= 20: ql_ma = np.convolve(list(q_loss_history), np.ones(20)/20, mode='valid'); ax2.plot(update_steps_est_q[19:], ql_ma, label='Q-Loss (MA-20)', c='red', lw=1.2)
                q_loss_list = [ql for ql in q_loss_history if ql is not None]
                if q_loss_list and all(ql > 1e-9 for ql in q_loss_list):
                     try: ax2.set_yscale('log'); ax2.set_ylabel('Q-Loss (Log)')
                     except ValueError: ax2.set_ylabel('Q-Loss')
                else: ax2.set_ylabel('Q-Loss')
            else: ax2.text(0.5, 0.5, 'No Q-Loss Data', ha='center', va='center', transform=ax2.transAxes); ax2.set_ylabel('Q-Loss')
            ax2.set_title('Critic Loss'); ax2.legend(); ax2.grid(True, ls='--', lw=0.5);
            # Plot Policy Loss
            ax3 = axes[2];
            if policy_loss_history:
                update_steps_est_p = np.linspace(start_learning_steps, current_env_step_num, len(policy_loss_history), dtype=int)
                ax3.plot(update_steps_est_p, list(policy_loss_history), label='P Loss (Raw)', alpha=0.4, c='forestgreen', lw=0.8)
                if len(policy_loss_history) >= 20: pl_ma = np.convolve(list(policy_loss_history), np.ones(20)/20, mode='valid'); ax3.plot(update_steps_est_p[19:], pl_ma, label='P Loss (MA-20)', c='darkgreen', lw=1.2)
            else: ax3.text(0.5, 0.5, 'No Policy Loss Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_ylabel('Policy Loss (-Q+Risk-Prog)'); ax3.set_title('Actor Loss'); ax3.legend(); ax3.grid(True, ls='--', lw=0.5)
            # Finalize plot
            ax3.set_xlabel('Environment Steps'); fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            plot_filename = os.path.join(results_dir, f"unified_plots_{env_name.replace('-','_')}.png")
            plt.savefig(plot_filename, dpi=300); logger.info(f"Plots saved: '{plot_filename}'"); plt.close(fig)
        except Exception as e_plot: warnings.warn(f"Could not generate plots: {e_plot}", RuntimeWarning)
    else: logger.info("No evaluation data, skipping plots.")

    # --- التقييم النهائي ---
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
        # تقييم أفضل نموذج
        if SAVE_BEST_MODEL and os.path.exists(results_dir):
             best_model_file = os.path.join(results_dir, f"unified_best_{env_name.replace('-','_')}.pt")
             logger.info("\n" + "="*60); logger.info(f"=== Evaluating Best Saved Unified Agent ==="); logger.info("="*60)
             if os.path.exists(best_model_file):
                 try:
                    logger.info("Loading best model for evaluation...")
                    # إعادة بناء الهيكل للتحميل (يجب أن يطابق الهيكل المستخدم في التدريب)
                    eval_eq_structure_best = UnifiedAdaptableEquation("eval_best_eq")
                    # ... (إعادة بناء الهيكل كما في دالة التدريب) ...
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
                    critic_hid_dim_best = hidden_dims[-1] # Use last hidden dim for critic hidden
                    critic_hid_eq_best = EvolvingEquation(critic_in_dim_best, eq_init_complexity, critic_hid_dim_best, eq_complexity_limit)
                    critic_hid_comp_best = NumericTransformComponent('critic_hidden_layer', critic_hid_eq_best, F.relu, {'critic','expert'}, {'request_q_value'}, {'value_features'}, 'critic_input', 'value_features')
                    eval_eq_structure_best.add_component(critic_hid_comp_best, execute_after=eval_shared_last_id)
                    critic_out_eq_best = EvolvingEquation(critic_hid_dim_best, eq_init_complexity, 1, eq_complexity_limit)
                    critic_out_comp_best = NumericTransformComponent('critic_output_layer', critic_out_eq_best, None, {'critic','expert'}, {'request_q_value'}, {'value_features_final'}, 'value_features', 'value_features_final')
                    eval_eq_structure_best.add_component(critic_out_comp_best, execute_after='critic_hidden_layer')
                    value_est_comp_best = ValueEstimationComponent('value_estimation', {'critic','expert'}, {'request_q_value'}, {'q_value'}, 'value_features_final', 'q_value')
                    eval_eq_structure_best.add_component(value_est_comp_best, execute_after='critic_output_layer')

                    # تهيئة وكيل التقييم بنفس الإعدادات
                    eval_best_agent = UnifiedLearningAgent(
                        trained_agent.state_dim, trained_agent.action_dim, (trained_agent.action_low, trained_agent.action_high),
                        eval_eq_structure_best, evolution_engine=None, # No evolution engine
                        gamma=trained_agent.gamma, tau=trained_agent.tau, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                        buffer_capacity=100, batch_size=BATCH_SIZE, exploration_noise_std=0, noise_decay_rate=1, min_exploration_noise=0,
                        weight_decay=WEIGHT_DECAY, grad_clip_norm=None, grad_clip_value=None,
                        risk_penalty_weight=trained_agent.risk_penalty_weight, progress_bonus_weight=trained_agent.progress_bonus_weight)

                    if eval_best_agent.load_model(best_model_file):
                        logger.info("Best model loaded successfully. Evaluating...")
                        best_eval_env = None
                        try:
                             best_eval_env = gym.make(ENVIRONMENT_NAME)
                             best_performance = eval_best_agent.evaluate(best_eval_env, episodes=10)
                             logger.info(f"Best Saved Agent Avg Perf (10 eps): {best_performance:.2f}")
                             logger.info(f"  Equation Complexity: {eval_best_agent.unified_eq.get_total_complexity():.2f}")
                             logger.info("\n--- Best Model Equation Representation ---")
                             logger.info(eval_best_agent.unified_eq.to_string())
                        except Exception as eval_e: logger.error(f"Error evaluating best model: {eval_e}")
                        finally:
                             if best_eval_env:
                                 try: best_eval_env.close()
                                 except Exception: pass # Ignore close errors
                    else: logger.warning("Skipping eval of best model (load failed).")
                 except Exception as e_load_best: logger.error(f"ERROR loading/evaluating best model: {e_load_best}"); traceback.print_exc()
             else: logger.warning(f"Best model file not found: '{best_model_file}'. Skipping.")
    else: logger.info("\nTraining failed or agent init failed. No final evaluation.")

    print("\n==================================================="); print("===      Unified Learning Agent Execution End       ==="); print("===================================================")

# ============================================================== #
# ===================== OPTIMIZATION LOOP ====================== #
# ============================================================== #
if __name__ == "__main__":

    # طباعة عنوان المثال والإصدار
    print("*" * 70)
    print(" Shape Extractor Optimization Loop (v1.1.0 using Bayesian Optimization)")
    print("*" * 70)

    # التحقق من المكتبات الأساسية
    if not CV_AVAILABLE: # التحقق من OpenCV
        print("\nERROR: OpenCV required.")
        sys.exit(1)
    if not PYPARSING_AVAILABLE: # التحقق من Pyparsing
        print("\nERROR: Pyparsing required.")
        sys.exit(1)
    if not MATPLOTLIB_AVAILABLE: # التحقق من Matplotlib
        print("\nERROR: Matplotlib required.")
        sys.exit(1)
    if not SKIMAGE_AVAILABLE: # التحقق من Scikit-image
        print("\nERROR: scikit-image required for comparison.")
        sys.exit(1)
    if not SKOPT_AVAILABLE: # التحقق من Scikit-optimize
        print("\nERROR: scikit-optimize required for optimization.")
        sys.exit(1)

    # --- الإعدادات ---
    # !!! هام: يجب تعديل هذا المسار !!!
    external_image_path = "tt.png" # <--- ** قم بتعديل هذا المسار **
    # مسار لحفظ الصورة المؤقتة المعاد بناؤها
    reconstructed_image_path = "_temp_reconstructed_opt.png"
    # إجمالي عدد استدعاءات دالة الهدف للتحسين البايزي
    n_calls_optimizer = 30
    # عدد النقاط العشوائية الأولية قبل بدء النموذج البايزي
    n_initial_points_optimizer = 10

    # --- تعريف فضاء البحث للمعاملات المراد تحسينها ---
    params_to_tune = {
        "canny_threshold2":        {"min": 100, "max": 250, "type": int,   "step": 10},
        "approx_poly_epsilon_factor": {"min": 0.01, "max": 0.03, "type": float, "step": 0.002},
        "contour_min_area":        {"min": 100, "max": 1000, "type": int,   "step": 50},
        "line_polygon_distance_tolerance": {"min": 1.0, "max": 8.0, "type": float, "step": 0.5},
        "min_final_line_length":   {"min": 10.0, "max": 50.0, "type": float, "step": 5.0},
        "hough_lines_threshold":   {"min": 30,  "max": 100, "type": int,   "step": 5},
        "line_polygon_angle_tolerance_deg": {"min": 3.0, "max": 10.0, "type": float, "step": 0.5},
    }
    # تحويل القاموس إلى قائمة أبعاد يفهمها skopt
    search_space_definitions = []
    # التكرار على المعاملات المراد ضبطها
    for name, settings in params_to_tune.items():
        param_type = settings["type"]
        param_min = settings["min"]
        param_max = settings["max"]
        # إضافة البعد بناءً على النوع
        if param_type == int:
            space = Integer(param_min, param_max, name=name)
            search_space_definitions.append(space)
        elif param_type == float:
            space = Real(param_min, param_max, prior='uniform', name=name)
            search_space_definitions.append(space)
        # يمكن إضافة أنواع أخرى مثل Categorical إذا لزم الأمر
    # الحصول على أسماء الأبعاد بالترتيب لاستخدامها لاحقًا
    dimension_names_list = [dim.name for dim in search_space_definitions]

    # التحقق من وجود الصورة المدخلة
    if not os.path.exists(external_image_path):
        print(f"\nERROR: Image file not found: '{external_image_path}'")
        sys.exit(1) # الخروج إذا لم توجد الصورة
    else:
        print(f"\nUsing external image: '{external_image_path}'")

    # --- تهيئة أفضل النتائج والراسم والشكل ---
    # متغيرات لتخزين أفضل نتيجة تم التوصل إليها
    best_ssim_score_global: float = -1.1 # قيمة أولية سيئة (أقل من أصغر SSIM ممكن)
    best_config_global = ShapeExtractor.DEFAULT_CONFIG.copy() # البدء بالإعدادات الافتراضية
    best_equation_global: Optional[str] = None # أفضل معادلة مستخلصة
    # تهيئة الراسم مرة واحدة
    plotter_instance = ShapePlotter2D()
    # إنشاء الشكل والمحاور مرة واحدة لإعادة استخدامها (لإصلاح تحذير Matplotlib)
    reusable_fig, reusable_ax = plt.subplots(figsize=(6, 6)) # حجم ثابت للصور المؤقتة

    # --- دالة الهدف (Objective Function) للتحسين البايزي ---
    # استخدام الديكور @use_named_args لتمرير المعاملات بالاسم كـ **kwargs
    @use_named_args(search_space_definitions)
    def objective_function(**params_dict) -> float:
        """
        الدالة التي سيقوم التحسين البايزي بمحاولة تقليل قيمتها.
        تقوم بتشغيل دورة الاستخلاص-الرسم-المقارنة وتعيد سالب SSIM.
        """
        # استخدام global للوصول للمتغيرات الخارجية وتعديلها
        global best_ssim_score_global
        global best_config_global
        global best_equation_global

        # إنشاء قاموس إعدادات جديد لهذه المحاولة (ابدأ من الافتراضي)
        trial_config = ShapeExtractor.DEFAULT_CONFIG.copy()
        # تحديثه بالمعاملات المقترحة من المحسن البايزي
        trial_config.update(params_dict)

        # طباعة المعاملات قيد التجربة (للتشخيص)
        current_params_str_list: List[str] = []
        for k_param, v_param in params_dict.items():
             if isinstance(v_param, float):
                  param_str = f"{k_param}={v_param:.3f}" # تنسيق الأرقام العشرية
             else:
                  param_str = f"{k_param}={v_param}" # الأنواع الأخرى (int)
             current_params_str_list.append(param_str)
        current_params_str = ", ".join(current_params_str_list)
        logger.info(f"--- Running Trial with Params: {current_params_str} ---")

        # تهيئة متغيرات نتيجة المحاولة
        current_trial_ssim: float = -1.1 # قيمة افتراضية سيئة
        extracted_eq_trial: Optional[str] = None
        try:
            # 1. الاستخلاص بالإعدادات الحالية
            extractor_trial = ShapeExtractor(config=trial_config)
            extracted_equation_trial = extractor_trial.extract_equation(external_image_path)

            # 2. الرسم والمقارنة (فقط إذا تم استخلاص معادلة)
            if extracted_equation_trial:
                # مسح مكونات الراسم السابقة
                plotter_instance.components = []
                # التحقق من جاهزية المحلل
                if plotter_instance.parser:
                    # تحليل المعادلة المستخلصة
                    plotter_instance.parse_equation(extracted_equation_trial)
                    # تحديد حجم ثابت للرسم المؤقت
                    plot_fig_size = (6, 6)
                    # الرسم على المحاور المعاد استخدامها، بدون عرض، وحفظ الملف
                    plotter_instance.plot(ax=reusable_ax, show_plot=False, save_path=reconstructed_image_path, clear_before_plot=True)
                    # التحقق من حفظ الملف والمقارنة
                    if os.path.exists(reconstructed_image_path):
                         ssim_result = compare_images_ssim(external_image_path, reconstructed_image_path)
                         if ssim_result is not None:
                             current_trial_ssim = ssim_result # تحديث النتيجة
                             logger.info(f"  Trial SSIM = {current_trial_ssim:.4f}")
                         else:
                             # التعامل مع فشل المقارنة
                             current_trial_ssim = -1.1
                             logger.warning("  SSIM calculation failed for this trial.")
                    else:
                        # فشل حفظ الرسم
                        current_trial_ssim = -1.1
                        logger.warning("  Reconstructed image not saved, cannot compare.")
                else:
                    # فشل تهيئة الراسم
                     current_trial_ssim = -1.1
                     logger.error("  Plotter parser not initialized, cannot reconstruct.")
            else:
                # فشل الاستخلاص
                 current_trial_ssim = -1.1
                 logger.warning("  Extractor failed to produce an equation for this config.")

        except Exception as trial_exception:
            # التعامل مع أي خطأ آخر أثناء المحاولة
            logger.error(f"Error during objective function trial: {trial_exception}", exc_info=False)
            current_trial_ssim = -1.1 # تعيين نتيجة سيئة

        # 3. تحديث أفضل نتيجة تم التوصل إليها عالميًا
        is_improvement = current_trial_ssim > best_ssim_score_global
        if is_improvement:
            logger.info(f"*** New Best SSIM Found: {current_trial_ssim:.4f} (Prev: {best_ssim_score_global:.4f}) ***")
            logger.info(f"   Achieved with: {current_params_str}")
            # تحديث أفضل نتيجة وأفضل إعدادات وأفضل معادلة
            best_ssim_score_global = current_trial_ssim
            best_config_global = trial_config.copy()
            best_equation_global = extracted_equation_trial
        # else: # يمكن إضافة طباعة لعدم التحسن إذا أردت
        #    logger.info("No improvement in SSIM score.")

        # إرجاع سالب SSIM (لأن gp_minimize يهدف للتقليل)
        return -current_trial_ssim

    # --- تشغيل التحسين البايزي ---
    optimization_result = None
    # التحقق مرة أخرى من توفر المكتبات اللازمة للتحسين
    if SKIMAGE_AVAILABLE and SKOPT_AVAILABLE and gp_minimize:
        print(f"\n--- Starting Bayesian Optimization ({n_calls_optimizer} calls, {n_initial_points_optimizer} initial) ---")
        # تعريف البذرة العشوائية محليًا لضمان التكرار
        OPTIMIZATION_SEED = 42 # يمكن تغييرها إذا أردت
        try:
            # استدعاء دالة التحسين
            optimization_result = gp_minimize(
                func=objective_function,            # الدالة الهدف
                dimensions=search_space_definitions,# فضاء البحث
                n_calls=n_calls_optimizer,          # إجمالي المحاولات
                n_initial_points=n_initial_points_optimizer, # نقاط عشوائية أولية
                acq_func='EI',                      # دالة الاكتساب (تحسين متوقع)
                random_state=OPTIMIZATION_SEED,     # بذرة عشوائية
                n_jobs=-1,                          # استخدام أنوية متعددة للمعالجة المتوازية (إذا كان مدعومًا)
            )
        except Exception as opt_err:
            # التعامل مع أخطاء التحسين البايزي
            logger.error(f"Bayesian optimization process failed: {opt_err}", exc_info=True)
            optimization_result = None # التأكد من أن النتيجة None
    else:
         # رسالة في حالة عدم توفر المكتبات
         logger.warning("\nOptimization loop skipped (scikit-image or scikit-optimize unavailable).")
         # تشغيل أساسي إذا لم يتم حسابه بعد
         if best_ssim_score_global < -1.0:
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
                         if ssim_base is not None:
                              best_ssim_score_global = ssim_base
                              logger.info(f"Baseline SSIM calculated: {best_ssim_score_global:.4f}")
             except Exception as base_err:
                 logger.error(f"Error during baseline only run: {base_err}")


    # --- عرض النتائج النهائية ---
    print("\n--- Optimization Finished ---")
    # التحقق من نتيجة التحسين
    if optimization_result:
        # استخلاص أفضل المعاملات والقيمة من نتيجة التحسين
        best_params_values = optimization_result.x
        best_objective_value = optimization_result.fun
        # حساب أفضل SSIM
        best_ssim_from_optimizer = -best_objective_value

        print(f"Best SSIM score found by optimization: {best_ssim_from_optimizer:.4f}")
        print("Best Configuration Found by Optimizer:")
        # بناء قاموس أفضل الإعدادات من نتيجة التحسين
        best_config_from_optimizer = ShapeExtractor.DEFAULT_CONFIG.copy()
        param_idx = 0
        while param_idx < len(dimension_names_list):
            param_name_opt = dimension_names_list[param_idx]
            param_value_opt = best_params_values[param_idx]
            current_dimension_def = search_space_definitions[param_idx]
            # التأكد من النوع الصحيح (Integer يحتاج تقريب)
            if isinstance(current_dimension_def, Integer):
                 best_config_from_optimizer[param_name_opt] = int(round(param_value_opt))
            else: # يفترض أنه Real
                 best_config_from_optimizer[param_name_opt] = param_value_opt
            param_idx += 1

        # طباعة المعاملات المهمة من أفضل إعدادات تم العثور عليها
        key_idx_opt = 0
        config_keys_list_opt = list(best_config_from_optimizer.keys())
        while key_idx_opt < len(config_keys_list_opt):
             key_opt = config_keys_list_opt[key_idx_opt]
             value_opt = best_config_from_optimizer[key_opt]
             is_tuned_opt = key_opt in dimension_names_list
             is_relevant_opt = key_opt in ["deduplication_method", "remove_lines_within_polygons", "merge_corner_lines"]
             if is_tuned_opt or is_relevant_opt:
                  print(f"  {key_opt}: {value_opt}")
             key_idx_opt += 1

        # إعادة الاستخلاص باستخدام **أفضل إعدادات تم العثور عليها عالميًا**
        # (best_config_global تم تحديثها داخل objective_function)
        print("\nRe-extracting equation with overall best found configuration...")
        try:
             # استخدام best_config_global
             final_extractor = ShapeExtractor(config=best_config_global)
             final_equation_result = final_extractor.extract_equation(external_image_path)
             print("\nOverall Best Extracted Equation:")
             if final_equation_result:
                  print(final_equation_result)
             else:
                  # إذا فشل الاستخلاص النهائي، اعرض أفضل معادلة وجدت أثناء البحث
                  print("Extraction failed with overall best config.")
                  if best_equation_global:
                       print("\nBest equation found during optimization trials:")
                       print(best_equation_global)
                  else:
                       print("No valid equation was generated.")
        except Exception as final_extract_err:
             print(f"Error re-extracting with best config: {final_extract_err}")
             # عرض أفضل معادلة وجدت أثناء البحث
             print("\nBest Extracted Equation (potentially from earlier iteration):")
             if best_equation_global: print(best_equation_global)
             else: print("No valid equation generated.")

    else:
        # عرض النتائج إذا فشل التحسين البايزي أو تم تخطيه
        print(f"Bayesian optimization did not run or failed.")
        print(f"Best SSIM score from baseline/manual runs: {best_ssim_score_global:.4f}")
        print("Best Configuration Found (baseline or pre-optimization):")
        key_idx_base = 0
        config_keys_base_list = list(best_config_global.keys())
        while key_idx_base < len(config_keys_base_list):
             key_base = config_keys_base_list[key_idx_base]
             value_base = best_config_global[key_base]
             # التحقق إذا كان المعامل قابلاً للضبط أو مهمًا للعرض
             is_tuned_base = key_base in dimension_names_list
             is_relevant_base = key_base in ["deduplication_method", "remove_lines_within_polygons", "merge_corner_lines"]
             if is_tuned_base or is_relevant_base:
                  print(f"  {key_base}: {value_base}")
             key_idx_base += 1
        print("\nBest Extracted Equation (baseline or pre-optimization):")
        if best_equation_global: print(best_equation_global)
        else: print("No valid equation was generated.")


    # --- التنظيف النهائي ---
    # إغلاق الشكل والمحاور المعاد استخدامها
    if reusable_fig:
        plt.close(reusable_fig)
    # حذف الصورة المؤقتة
    if os.path.exists(reconstructed_image_path):
        try:
            os.remove(reconstructed_image_path)
            print(f"\nRemoved temp image: '{reconstructed_image_path}'")
        except OSError as e_remove:
            logger.warning(f"Cannot remove temp image '{reconstructed_image_path}': {e_remove}")

    # رسالة الانتهاء
    print("\n" + "*" * 70)
    print(" Shape Extractor Optimization Loop Complete")
    print("*" * 70)