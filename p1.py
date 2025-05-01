# -*- coding: utf-8 -*-

"""
كل الأفكار المبتكرة تعود بمطلقها للمطور: باسل يحيى عبدالله
مع أكواد أولية قدمها للنماذج الذكاء الصطناعي لانتاج اكواد أكبر وأنضج.
هذا العمل (وكثير غيره) لازال في مرحلة التطوير وأجعله مفتوح المصدر بترخيص MIT
بشرط ذكر المصدر الأساس عند تطويره والاستفادة منه، ومع احتفاظ المطور (باسل يحيى عبدالله) بحقوق الملكية الفكرية
وهو غير مسؤول عن أي مخاطر نتيجة استخدام وتجريب هذا الكود أو غيره مما أنتجه
"""

"""
==============================================================================
نظام الخبير/المستكشف المحسن 
==============================================================================
- المستكشف: يستخدم خسارة Policy Gradient بسيطة تعتمد على (1 - risk).
- الخبير: يستخدم EvolvingEquation القوية و EvolutionEngine لتطوير نموذج المخاطر.
- الإصلاح: الالتزام الصارم بتعليمات عدم كتابة أكثر من تعليمة في السطر.
"""

# ========================================
# 0. الاستيرادات الضرورية
# ========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import gym
import random
from collections import deque
# from collections import OrderedDict # لم تعد مستخدمة
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import time
import os
import warnings
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple
from typing import Set
from scipy import stats
import inspect
import traceback

# --- 1. الإعدادات العامة والبذرة العشوائية ---
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
# نهاية if
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {DEVICE}")


# ========================================
# 2. مكونات الخبير المتطور (EvolvingEquation, EvolutionEngine)
# ========================================
class EvolvingEquation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        init_complexity: int = 3,
        output_dim: int = 1,
        complexity_limit: int = 10,
        min_complexity: int = 2,
        output_activation: Optional[Callable] = torch.sigmoid,
        exp_clamp_min: float = 0.1,
        exp_clamp_max: float = 3.0,
        term_clamp: float = 1e3,
        output_clamp: float = 1e4
    ):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        # نهاية if
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        # نهاية if
        if not isinstance(init_complexity, int) or init_complexity <= 0:
            raise ValueError("init_complexity must be a positive integer.")
        # نهاية if
        if not isinstance(min_complexity, int) or min_complexity <= 0:
            raise ValueError("min_complexity must be a positive integer.")
        # نهاية if
        if not isinstance(complexity_limit, int) or complexity_limit < min_complexity:
            raise ValueError(f"complexity_limit must be >= min_complexity.")
        # نهاية if
        self.input_dim = input_dim
        self.output_dim = output_dim
        effective_init_complexity = max(min_complexity, init_complexity)
        effective_init_complexity = min(effective_init_complexity, complexity_limit)
        self.complexity = effective_init_complexity
        self.complexity_limit = complexity_limit
        self.min_complexity = min_complexity
        self.output_activation = output_activation
        self.exp_clamp_min = exp_clamp_min
        self.exp_clamp_max = exp_clamp_max
        self.term_clamp = term_clamp
        self.output_clamp = output_clamp
        self._initialize_components()
    # نهاية __init__

    def _initialize_components(self):
        self.input_transform = nn.Linear(self.input_dim, self.complexity)
        nn.init.xavier_uniform_(self.input_transform.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.input_transform.bias)
        self.coefficients = nn.ParameterList(
            [nn.Parameter(torch.randn(1) * 0.05, requires_grad=True) for _ in range(self.complexity)]
        )
        self.exponents = nn.ParameterList(
            [nn.Parameter(torch.abs(torch.randn(1) * 0.1) + 1.0, requires_grad=True) for _ in range(self.complexity)]
        )
        self.function_library = [
            torch.sin, torch.cos, torch.tanh, torch.sigmoid,
            F.relu, F.leaky_relu, F.gelu,
            lambda x: x,
            lambda x: torch.pow(x, 2),
            lambda x: torch.exp(-torch.abs(x)),
            lambda x: torch.sqrt(torch.abs(x) + 1e-6),
            lambda x: torch.clamp(x, -2.0, 2.0),
            lambda x: x * torch.sigmoid(x),
        ]
        if not self.function_library:
            raise ValueError("Function library cannot be empty.")
        # نهاية if
        self.functions = [self._select_function() for _ in range(self.complexity)]
        self.output_layer = nn.Linear(self.complexity, self.output_dim)
        gain_out = nn.init.calculate_gain('linear')
        if self.output_activation is not None:
            if self.output_activation == torch.sigmoid or self.output_activation == torch.tanh:
                 gain_out = nn.init.calculate_gain('sigmoid')
            # نهاية if
        # نهاية if
        nn.init.xavier_uniform_(self.output_layer.weight, gain=gain_out)
        nn.init.zeros_(self.output_layer.bias)
    # نهاية _initialize_components

    def _select_function(self) -> Callable:
        return random.choice(self.function_library)
    # نهاية _select_function

    def _safe_pow(self, base: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(base)
        base_abs_safe = torch.abs(base) + 1e-8
        powered = torch.pow(base_abs_safe, exp)
        result = sign * powered
        return result
    # نهاية _safe_pow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size_out = 1
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            except Exception as e:
                warnings.warn(f"Input 'x' not tensor convertible: {e}. Returning zeros.", RuntimeWarning)
                return torch.zeros(batch_size_out, self.output_dim, device=DEVICE)
            # نهاية try-except
        # نهاية if
        if x.dim() == 1:
            batch_size_out = 1
            if x.shape[0] == self.input_dim:
                x = x.unsqueeze(0)
            else:
                warnings.warn(f"Input dim {x.shape[0]} != expected {self.input_dim}. Returning zeros.", RuntimeWarning)
                return torch.zeros(batch_size_out, self.output_dim, device=x.device)
            # نهاية if-else
        elif x.dim() > 2:
            original_shape = x.shape
            batch_size_out = x.shape[0]
            x = x.view(batch_size_out, -1)
            if x.shape[1] != self.input_dim:
                 warnings.warn(f"Flattened dim {x.shape[1]} != expected {self.input_dim}. Returning zeros.", RuntimeWarning)
                 return torch.zeros(batch_size_out, self.output_dim, device=x.device)
            # نهاية if
        elif x.dim() == 2:
            batch_size_out = x.shape[0]
            if x.shape[1] != self.input_dim:
                warnings.warn(f"Input dim {x.shape[1]} != expected {self.input_dim}. Returning zeros.", RuntimeWarning)
                return torch.zeros(batch_size_out, self.output_dim, device=x.device)
            # نهاية if
        elif x.dim() == 0:
             x = x.unsqueeze(0).unsqueeze(0)
             batch_size_out = 1
             if self.input_dim != 1:
                 warnings.warn("Scalar input received but input_dim != 1. Returning zeros.", RuntimeWarning)
                 return torch.zeros(batch_size_out, self.output_dim, device=x.device)
             # نهاية if
        else:
             warnings.warn(f"Invalid input tensor dim: {x.dim()}. Returning zeros.", RuntimeWarning)
             return torch.zeros(batch_size_out, self.output_dim, device=DEVICE)
        # نهاية if-elif-else
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = DEVICE
        # نهاية try-except
        if x.device != model_device:
            x = x.to(model_device)
        # نهاية if
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # نهاية if
        output = torch.zeros(batch_size_out, self.output_dim, device=model_device)
        try:
            transformed_features = self.input_transform(x)
            if torch.isnan(transformed_features).any() or torch.isinf(transformed_features).any():
                transformed_features = torch.nan_to_num(transformed_features, nan=0.0, posinf=self.term_clamp, neginf=-self.term_clamp)
            # نهاية if
            transformed_features = torch.clamp(transformed_features, -self.term_clamp, self.term_clamp)
            term_results = torch.zeros(x.shape[0], self.complexity, device=model_device)
            if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')):
                 warnings.warn("Eq components missing in forward. Zeros.", RuntimeWarning)
                 return output
            # نهاية if
            for i in range(self.complexity):
                if i>=len(self.coefficients) or i>=len(self.exponents) or i>=len(self.functions):
                     continue
                # نهاية if
                try:
                    feature_i = transformed_features[:, i]
                    exp_val = torch.clamp(self.exponents[i], self.exp_clamp_min, self.exp_clamp_max)
                    term_powered = self._safe_pow(feature_i, exp_val)
                    if torch.isnan(term_powered).any() or torch.isinf(term_powered).any():
                         term_powered = torch.zeros_like(feature_i)
                    # نهاية if
                    term_powered = torch.clamp(term_powered, -self.term_clamp, self.term_clamp)
                    term_activated = self.functions[i](term_powered)
                    if torch.isnan(term_activated).any() or torch.isinf(term_activated).any():
                         term_activated = torch.zeros_like(feature_i)
                    # نهاية if
                    term_activated = torch.clamp(term_activated, -self.term_clamp, self.term_clamp)
                    term_value = self.coefficients[i] * term_activated
                    term_results[:, i] = torch.clamp(term_value, -self.term_clamp, self.term_clamp)
                except Exception as term_e:
                    term_results[:, i] = 0.0
                # نهاية try-except (term)
            # نهاية for
            if torch.isnan(term_results).any() or torch.isinf(term_results).any():
                term_results = torch.nan_to_num(term_results, nan=0.0, posinf=self.term_clamp, neginf=-self.term_clamp)
            # نهاية if
            output = self.output_layer(term_results)
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            # نهاية if
            if self.output_activation:
                output = self.output_activation(output)
                if torch.isnan(output).any() or torch.isinf(output).any():
                    output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
                # نهاية if
            # نهاية if (output_activation)
        except Exception as e:
            warnings.warn(f"Error during EvolvingEq forward pass: {e}. Zeros.", RuntimeWarning)
            traceback.print_exc()
            output = torch.zeros(batch_size_out, self.output_dim, device=model_device)
        # نهاية try-except (main forward)
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.zeros_like(output)
        # نهاية if
        return output
    # نهاية forward

    def to_string(self) -> str:
        parts = []
        func_names = {
            torch.sin: 'sin', torch.cos: 'cos', torch.tanh: 'tanh', torch.sigmoid: 'sigmoid',
            F.relu: 'relu', F.leaky_relu: 'lrelu', F.gelu: 'gelu',
        }
        lambda_counter = 0
        for i, func in enumerate(self.function_library):
            if getattr(func, '__name__', '') == '<lambda>':
                 func_names[func] = f'lambda_{lambda_counter}'
                 lambda_counter += 1
            # نهاية if
        # نهاية for
        if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')):
             return "equation_not_fully_initialized"
        # نهاية if
        for i in range(self.complexity):
            if i >= len(self.coefficients) or i >= len(self.exponents) or i >= len(self.functions):
                 continue
            # نهاية if
            try:
                coeff_val = self.coefficients[i].item()
                exp_val = torch.clamp(self.exponents[i], self.exp_clamp_min, self.exp_clamp_max).item()
                func = self.functions[i]
                func_name = func_names.get(func, getattr(func, '__name__', 'unknown_lambda'))
                part = f"{coeff_val:.2f}*{func_name}(F_{i}^{exp_val:.2f})"
                parts.append(part)
            except Exception as e:
                 parts.append(f"[Err T{i}]")
            # نهاية try-except
        # نهاية for
        eq_str = " + ".join(parts) if parts else "empty"
        act_str = getattr(self.output_activation, '__name__', 'None') if self.output_activation else 'None'
        return f"Out={act_str}(Lin({eq_str}))[C={self.complexity}]"
    # نهاية to_string

    def add_term(self) -> bool:
        if self.complexity >= self.complexity_limit:
            return False
        # نهاية if
        new_complexity = self.complexity + 1
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = DEVICE
        # نهاية try-except
        try:
            old_in_w=self.input_transform.weight.data.clone()
            old_in_b=self.input_transform.bias.data.clone()
            old_out_w=self.output_layer.weight.data.clone()
            old_out_b=self.output_layer.bias.data.clone()
        except AttributeError:
            warnings.warn("Missing layers during add_term.", RuntimeWarning)
            return False
        # نهاية try-except
        try:
            self.coefficients.append(nn.Parameter(torch.randn(1,device=device)*0.01, requires_grad=True))
            self.exponents.append(nn.Parameter(torch.abs(torch.randn(1,device=device)*0.05)+1.0, requires_grad=True))
            self.functions.append(self._select_function())
        except Exception as e:
            warnings.warn(f"Error appending new params/func: {e}", RuntimeWarning)
            return False
        # نهاية try-except
        new_in_t = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad():
            new_in_t.weight.data[:self.complexity,:] = old_in_w
            new_in_t.bias.data[:self.complexity] = old_in_b
            nn.init.xavier_uniform_(new_in_t.weight.data[self.complexity:], gain=nn.init.calculate_gain('relu'))
            new_in_t.weight.data[self.complexity:] *= 0.01
            nn.init.zeros_(new_in_t.bias.data[self.complexity:])
        # نهاية with
        self.input_transform = new_in_t
        new_out_l = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad():
            new_out_l.weight.data[:,:self.complexity] = old_out_w
            gain_out = nn.init.calculate_gain('linear')
            if self.output_activation is not None:
                 if self.output_activation == torch.sigmoid or self.output_activation == torch.tanh:
                      gain_out = nn.init.calculate_gain('sigmoid')
                 # نهاية if
            # نهاية if
            nn.init.xavier_uniform_(new_out_l.weight.data[:,self.complexity:], gain=gain_out)
            new_out_l.weight.data[:,self.complexity:] *= 0.01
            new_out_l.bias.data.copy_(old_out_b)
        # نهاية with
        self.output_layer = new_out_l
        self.complexity = new_complexity
        return True
    # نهاية add_term

    # --- !!! إصلاح دالة prune_term هنا !!! ---
    def prune_term(self, aggressive: bool = False) -> bool:
        """إزالة حد من المعادلة إذا كان التعقيد أكبر من الحد الأدنى."""
        if self.complexity <= self.min_complexity:
            return False
        # نهاية if
        new_complexity = self.complexity - 1
        idx_to_prune = -1
        # --- فصل try...except للحصول على الجهاز ---
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = DEVICE
        # نهاية try-except

        # --- اختيار الحد المراد تقليمه ---
        try:
            if self.complexity<=0:
                 return False
            # نهاية if
            if aggressive:
                with torch.no_grad():
                    has_coeffs = hasattr(self,'coefficients') and self.coefficients
                    has_in_t = hasattr(self,'input_transform') and self.input_transform
                    has_out_l = hasattr(self,'output_layer') and self.output_layer
                    if not (has_coeffs and has_in_t and has_out_l):
                        idx_to_prune=random.randint(0,self.complexity-1)
                    else:
                        coeffs_abs=torch.tensor([torch.abs(c.data).item() for c in self.coefficients],device='cpu')
                        in_w=self.input_transform.weight
                        out_w=self.output_layer.weight
                        expected_shape_in=(self.complexity,self.input_dim)
                        expected_shape_out=(self.output_dim,self.complexity)
                        # --- التأكد من المسافة البادئة الصحيحة ---
                        if in_w is None or out_w is None or in_w.shape!=expected_shape_in or out_w.shape!=expected_shape_out or coeffs_abs.shape[0]!=self.complexity:
                            warnings.warn(f"Shape mismatch prune calc. Random prune.", RuntimeWarning)
                            idx_to_prune = random.randint(0, self.complexity - 1)
                        else:
                            in_norm=torch.norm(in_w.data.cpu(),p=1,dim=1)
                            out_norm=torch.norm(out_w.data.cpu(),p=1,dim=0)
                            importance=(coeffs_abs*(in_norm+out_norm))+1e-9
                            if torch.isnan(importance).any():
                                warnings.warn("NaN importance. Random prune.", RuntimeWarning)
                                idx_to_prune=random.randint(0,self.complexity-1)
                            else:
                                idx_to_prune=torch.argmin(importance).item()
                            # نهاية if-else (isnan)
                        # نهاية if-else (shape mismatch)
                    # نهاية if-else (components exist)
                # نهاية with
            else:
                idx_to_prune = random.randint(0, self.complexity - 1)
            # نهاية if-else (aggressive)

            if not (0 <= idx_to_prune < self.complexity):
                 warnings.warn(f"Invalid index {idx_to_prune} for pruning.", RuntimeWarning)
                 return False
            # نهاية if
        except Exception as e:
            warnings.warn(f"Error selecting term to prune: {e}. Aborting.", RuntimeWarning)
            return False
        # نهاية try-except (selection)

        # --- إزالة المعلمات والدالة ---
        try:
            if hasattr(self,'coefficients') and len(self.coefficients)>idx_to_prune:
                 del self.coefficients[idx_to_prune]
            else:
                 raise IndexError("Coeff index")
            # نهاية if-else
            if hasattr(self,'exponents') and len(self.exponents)>idx_to_prune:
                 del self.exponents[idx_to_prune]
            else:
                 raise IndexError("Exp index")
            # نهاية if-else
            if hasattr(self,'functions') and len(self.functions)>idx_to_prune:
                 self.functions.pop(idx_to_prune)
            else:
                 raise IndexError("Func index")
            # نهاية if-else
        except (IndexError, Exception) as e:
            warnings.warn(f"Error removing params/func: {e}. Prune failed.", RuntimeWarning)
            return False
        # نهاية try-except (removal)

        # --- نسخ الأوزان القديمة ---
        try:
            old_in_w=self.input_transform.weight.data.clone()
            old_in_b=self.input_transform.bias.data.clone()
            old_out_w=self.output_layer.weight.data.clone()
            old_out_b=self.output_layer.bias.data.clone()
        except AttributeError:
             warnings.warn("Missing layers during prune update.", RuntimeWarning)
             return False
        # نهاية try-except (cloning)

        # --- تحديث الطبقات ---
        new_in_t = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad():
            new_w_in=torch.cat([old_in_w[:idx_to_prune], old_in_w[idx_to_prune+1:]], dim=0)
            new_b_in=torch.cat([old_in_b[:idx_to_prune], old_in_b[idx_to_prune+1:]])
            if new_w_in.shape[0]!=new_complexity or new_b_in.shape[0]!=new_complexity:
                 warnings.warn("In shape mismatch after prune.", RuntimeWarning)
                 return False
            # نهاية if
            new_in_t.weight.data.copy_(new_w_in)
            new_in_t.bias.data.copy_(new_b_in)
        # نهاية with
        self.input_transform = new_in_t

        new_out_l = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad():
            new_w_out = torch.cat([old_out_w[:, :idx_to_prune], old_out_w[:, idx_to_prune+1:]], dim=1)
            if new_w_out.shape[1]!=new_complexity:
                 warnings.warn("Out shape mismatch after prune.", RuntimeWarning)
                 return False
            # نهاية if
            new_out_l.weight.data.copy_(new_w_out)
            new_out_l.bias.data.copy_(old_out_b)
        # نهاية with
        self.output_layer = new_out_l

        self.complexity = new_complexity
        return True
    # نهاية prune_term
# نهاية فئة EvolvingEquation

class EvolutionEngine: # (كما هي)
    def __init__(self, mutation_power: float = 0.02, history_size: int = 100, cooldown_period: int = 50, add_term_threshold: float = 0.90, prune_term_threshold: float = 0.15, add_term_prob: float = 0.10, prune_term_prob: float = 0.20, swap_func_prob: float = 0.05):
        if not (0 < mutation_power < 0.5): warnings.warn(f"mutation_power ({mutation_power}) outside typical range.")
        if not (0 < add_term_threshold < 1): warnings.warn(f"add_term_threshold ({add_term_threshold}) outside (0, 1).")
        if not (0 < prune_term_threshold < 1): warnings.warn(f"prune_term_threshold ({prune_term_threshold}) outside (0, 1).")
        if add_term_threshold <= prune_term_threshold: warnings.warn(f"add_term_threshold <= prune_term_threshold.")
        if not (0 <= add_term_prob <= 1) or not (0 <= prune_term_prob <= 1) or not (0 <= swap_func_prob <= 1): warnings.warn("Evolution probs should be between 0 and 1.")
        self.base_mutation_power = mutation_power; self.performance_history = deque(maxlen=history_size); self.cooldown_period = cooldown_period; self.term_change_cooldown = 0; self.add_term_threshold = add_term_threshold; self.prune_term_threshold = prune_term_threshold; self.add_term_prob = add_term_prob; self.prune_term_prob = prune_term_prob; self.swap_func_prob = swap_func_prob
    # نهاية __init__
    def _calculate_percentile(self, current_metric: float) -> float: # (كما هي)
        if math.isnan(current_metric):
             return 0.5
        # نهاية if
        valid_history = [r for r in self.performance_history if not math.isnan(r)]
        if not valid_history:
             return 0.5
        # نهاية if
        history_array = np.array(valid_history)
        try:
            percentile = stats.percentileofscore(history_array, current_metric, kind='mean') / 100.0
        except Exception:
            percentile = np.mean(history_array < current_metric) if len(history_array) > 0 else 0.5
        # نهاية try-except
        return np.clip(percentile, 0.0, 1.0)
    # نهاية _calculate_percentile
    def _dynamic_mutation_scale(self, percentile: float) -> float: # (كما هي مع الإصلاح)
        if math.isnan(percentile):
            percentile = 0.5
        # نهاية if
        scale = 1.0
        if percentile > 0.9:
            scale = 0.5
        elif percentile < 0.1:
            scale = 1.5
        else:
            scale = 1.5 - (percentile - 0.1) * (1.5 - 0.5) / (0.9 - 0.1)
        # نهاية if-elif-else
        return max(0.1, min(scale, 2.0))
    # نهاية _dynamic_mutation_scale
    def evolve_equation(self, equation: EvolvingEquation, metric: float, step: int) -> bool: # (كما هي)
        structure_changed = False
        if not isinstance(equation, EvolvingEquation):
             return False
        # نهاية if
        if not math.isnan(metric):
            self.performance_history.append(metric)
            percentile = self._calculate_percentile(metric)
        else:
            valid_history = [r for r in self.performance_history if not math.isnan(r)]
            last_valid = valid_history[-1] if valid_history else np.nan
            percentile = self._calculate_percentile(last_valid) if not math.isnan(last_valid) else 0.5
        # نهاية if-else
        if self.term_change_cooldown > 0:
            self.term_change_cooldown -= 1
        # نهاية if
        if self.term_change_cooldown == 0 and len(self.performance_history) >= max(10, self.performance_history.maxlen // 4):
            rand_roll = random.random()
            action_taken = "None"
            if percentile > self.add_term_threshold and rand_roll < self.add_term_prob:
                if equation.add_term():
                    self.term_change_cooldown = self.cooldown_period
                    structure_changed = True
                    action_taken = f"Add(C:{equation.complexity})"
                # نهاية if
            elif percentile < self.prune_term_threshold and rand_roll < self.prune_term_prob:
                if equation.prune_term(aggressive=True):
                    self.term_change_cooldown = self.cooldown_period
                    structure_changed = True
                    action_taken = f"Prune(C:{equation.complexity})"
                # نهاية if
            # نهاية if-elif
            if structure_changed:
                 print(f"\nEVO [{time.strftime('%H:%M:%S')}]: Struct Upd @ S {step}. Act: {action_taken}. Pct: {percentile:.2f}. CD: {self.term_change_cooldown}")
            # نهاية if
        # نهاية if
        if not structure_changed and self.term_change_cooldown == 0 and random.random() < self.swap_func_prob:
            if self.swap_function(equation):
                 self.term_change_cooldown = max(self.term_change_cooldown, 5)
            # نهاية if
        # نهاية if
        mutation_scale = self._dynamic_mutation_scale(percentile)
        self._mutate_parameters(equation, mutation_scale, step)
        return structure_changed
    # نهاية evolve_equation
    def _mutate_parameters(self, equation: EvolvingEquation, mutation_scale: float, step: int): # (كما هي)
        if not isinstance(equation, EvolvingEquation):
             return
        # نهاية if
        cooling_factor = max(0.3, 1.0 - step / 1000000.0)
        effective_power = self.base_mutation_power * mutation_scale * cooling_factor
        if effective_power < 1e-9:
             return
        # نهاية if
        try:
            with torch.no_grad():
                try:
                     device = next(equation.parameters()).device
                except StopIteration:
                     return
                # نهاية try-except
                if hasattr(equation,'coefficients') and equation.coefficients:
                    for c in equation.coefficients:
                         c.data.add_(torch.randn_like(c.data)*effective_power)
                    # نهاية for
                # نهاية if
                if hasattr(equation,'exponents') and equation.exponents:
                    for e in equation.exponents:
                         e.data.add_(torch.randn_like(e.data)*effective_power*0.1)
                         e.data.clamp_(min=equation.exp_clamp_min,max=equation.exp_clamp_max)
                    # نهاية for
                # نهاية if
                if hasattr(equation,'input_transform') and isinstance(equation.input_transform,nn.Linear):
                    ps=0.5
                    equation.input_transform.weight.data.add_(torch.randn_like(equation.input_transform.weight.data)*effective_power*ps)
                    if equation.input_transform.bias is not None:
                         equation.input_transform.bias.data.add_(torch.randn_like(equation.input_transform.bias.data)*effective_power*ps*0.5)
                    # نهاية if
                # نهاية if
                if hasattr(equation,'output_layer') and isinstance(equation.output_layer,nn.Linear):
                    ps=0.5
                    equation.output_layer.weight.data.add_(torch.randn_like(equation.output_layer.weight.data)*effective_power*ps)
                    if equation.output_layer.bias is not None:
                         equation.output_layer.bias.data.add_(torch.randn_like(equation.output_layer.bias.data)*effective_power*ps*0.5)
                    # نهاية if
                # نهاية if
            # نهاية with
        except Exception as e:
             pass
        # نهاية try-except
    # نهاية _mutate_parameters
    def swap_function(self, equation: EvolvingEquation) -> bool: # (كما هي)
        if not isinstance(equation,EvolvingEquation) or equation.complexity<=0 or not hasattr(equation,'functions') or not equation.functions or not equation.function_library:
             return False
        # نهاية if
        try:
             idx=random.randint(0,equation.complexity-1)
             old_f=equation.functions[idx]
             attempts=0
             max_att=len(equation.function_library)*2
             new_f=old_f
             while attempts<max_att:
                  cand_f=equation._select_function()
                  if cand_f is not old_f:
                       new_f=cand_f
                       break
                  # نهاية if
                  attempts+=1
             # نهاية while
             equation.functions[idx]=new_f
             return True
        except (IndexError, Exception) as e:
             return False
        # نهاية try-except
    # نهاية swap_function
# نهاية فئة EvolutionEngine


# ========================================
# 3. البيئة (GridWorld - كما هي)
# ========================================
class AdvancedGridWorld: # (كما هي)
    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.max_distance = self._distance_to_goal((0, 0))
        self.obstacles = [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)]
        if self.goal in self.obstacles:
             self.obstacles.remove(self.goal)
        # نهاية if
        start_pos = (0, 0)
        if start_pos in self.obstacles:
             start_pos = (0, 1)
        # نهاية if
        self.agent_pos = start_pos
        self.start_pos = start_pos
        self.action_space_n = 4
    # نهاية __init__
    def reset(self):
        self.agent_pos = self.start_pos
        state = self._get_state()
        return state
    # نهاية reset
    def _distance_to_goal(self, pos):
        dist = math.sqrt((pos[0]-self.goal[0])**2 + (pos[1]-self.goal[1])**2)
        return dist
    # نهاية _distance_to_goal
    def _get_state(self):
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        grid[self.agent_pos] = 1.0
        goal_x, goal_y = self.goal
        grid[goal_x, goal_y] = 0.5
        for r,c in self.obstacles:
             if 0<=r<self.size and 0<=c<self.size:
                  grid[r,c] = -1.0
             # نهاية if
        # نهاية for
        flat_grid = grid.flatten()
        return flat_grid
    # نهاية _get_state
    def step(self, action):
        curr_pos = self.agent_pos
        dist_before = self._distance_to_goal(curr_pos)
        r, c = curr_pos
        nr, nc = r, c
        if action==0:
             nr=max(0,r-1)
        elif action==1:
             nc=min(self.size-1,c+1)
        elif action==2:
             nr=min(self.size-1,r+1)
        elif action==3:
             nc=max(0,c-1)
        # نهاية if-elif
        new_pos = (nr, nc)
        reward = 0.0
        done = False
        if new_pos == self.goal:
            reward = 100.0
            done = True
            dist_after = 0.0
        elif new_pos in self.obstacles:
            reward = -5.0
            new_pos = curr_pos
            dist_after = dist_before
        else:
            reward -= 0.05
            dist_after = self._distance_to_goal(new_pos)
            reward += (dist_before - dist_after) * 0.5
        # نهاية if-elif-else
        self.agent_pos = new_pos
        next_state = self._get_state()
        info = {'distance_before': dist_before, 'distance_after': dist_after}
        return next_state, reward, done, info
    # نهاية step
    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.goal[0]][self.goal[1]] = 'G'
        for r,c in self.obstacles:
             if 0<=r<self.size and 0<=c<self.size:
                  grid[r][c] = 'X'
             # نهاية if
        # نهاية for
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        print("-"*(self.size*2+1))
        for row in grid:
             print("|" + " ".join(row) + "|")
        # نهاية for
        print("-"*(self.size*2+1))
        print()
    # نهاية render
# نهاية فئة AdvancedGridWorld


# ========================================
# 4. مكونات المستكشف (كما هي)
# ========================================
class DynamicCorrelationLayer(nn.Module): # (كما هي)
    def __init__(self, input_dim):
        super().__init__()
        self.cov_matrix = nn.Parameter(torch.randn(input_dim, input_dim))
        self.phase_mod = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh())
    # نهاية __init__
    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if not isinstance(x, torch.Tensor):
                 return None
            # نهاية if
            if torch.isnan(x).any() or torch.isinf(x).any():
                 x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            # نهاية if
            original_dim = x.dim()
            if original_dim == 1:
                 x = x.unsqueeze(0)
            elif original_dim != 2:
                 return None
            # نهاية if-elif
            input_dim = x.shape[1]
            expected_cov_shape = (input_dim, input_dim)
            if self.cov_matrix.shape != expected_cov_shape:
                 return None
            # نهاية if
            correlated = torch.matmul(x, self.cov_matrix)
            if torch.isnan(correlated).any() or torch.isinf(correlated).any():
                 correlated = torch.nan_to_num(correlated, nan=0.0, posinf=1e4, neginf=-1e4)
            # نهاية if
            phased = self.phase_mod(correlated)
            if torch.isnan(phased).any() or torch.isinf(phased).any():
                 phased = torch.nan_to_num(phased, nan=0.0, posinf=1.0, neginf=-1.0)
            # نهاية if
            if phased.shape != x.shape:
                 return None
            # نهاية if
            output = phased + x
            if torch.isnan(output).any() or torch.isinf(output).any():
                 output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
            # نهاية if
            if original_dim == 1:
                 output = output.squeeze(0)
            # نهاية if
            return output
        except Exception as e:
             return None
        # نهاية try-except
    # نهاية forward
# نهاية فئة DynamicCorrelationLayer

class HyperbolicAttention(nn.Module): # (كما هي)
    def __init__(self, dim):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.05))
        self.epsilon = 1e-7
    # نهاية __init__
    def _poincare_distance(self, u, v):
        c_clamped = torch.abs(self.c) + self.epsilon
        c_clamped = torch.clamp(c_clamped, min=self.epsilon, max=1.0 - self.epsilon)
        c_sqrt = torch.sqrt(c_clamped)
        uu_norm_sq = torch.sum(u*u, dim=-1, keepdim=True)
        vv_norm_sq = torch.sum(v*v, dim=-1, keepdim=True)
        uu_norm_sq_c = torch.clamp(c_clamped * uu_norm_sq, max=1.0 - self.epsilon)
        vv_norm_sq_c = torch.clamp(c_clamped * vv_norm_sq, max=1.0 - self.epsilon)
        uv_diff_norm_sq = torch.sum((u - v)**2, dim=-1, keepdim=True)
        denominator = torch.clamp((1.0 - uu_norm_sq_c) * (1.0 - vv_norm_sq_c), min=self.epsilon)
        argument = torch.clamp(1.0 + (2.0 * c_clamped * uv_diff_norm_sq) / denominator, min=1.0 + self.epsilon)
        distance = (1.0 / c_sqrt) * torch.acosh(argument)
        return distance
    # نهاية _poincare_distance
    def forward(self, query, keys, values):
        if query is None or keys is None or values is None:
             return None
        # نهاية if
        is_single = query.dim()==1
        query, keys, values = [t.unsqueeze(0) if is_single else t for t in [query, keys, values]]
        B, D = query.shape
        query_ex_i = query.unsqueeze(1).expand(B, B, D)
        keys_ex_j = keys.unsqueeze(0).expand(B, B, D)
        dist = self._poincare_distance(query_ex_i, keys_ex_j).squeeze(-1)
        dist = torch.nan_to_num(dist, nan=100., posinf=100., neginf=-100.)
        weights = torch.softmax(-dist, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.)
        output = torch.matmul(weights, values)
        if is_single:
             output = output.squeeze(0)
        # نهاية if
        return output
    # نهاية forward
# نهاية فئة HyperbolicAttention

class ExplorerPolicyNetwork(nn.Module): # (كما هي)
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.correlator = DynamicCorrelationLayer(hidden_dim)
        self.attention = HyperbolicAttention(hidden_dim)
        self.post_attention_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.risk_predictor = nn.Linear(hidden_dim, 1)
        self.epsilon = 1e-8
    # نهاية __init__
    def forward(self, state):
        batch_size = 1
        try:
            if not isinstance(state, torch.Tensor):
                 state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            # نهاية if
            if state.dim() == 1:
                 state = state.unsqueeze(0)
            # نهاية if
            batch_size = state.shape[0]
            state = state.float().to(DEVICE)
            encoded = self.encoder(state)
            if encoded is None or (isinstance(encoded, torch.Tensor) and (torch.isnan(encoded).any() or torch.isinf(encoded).any())):
                 encoded = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
            # نهاية if
            correlated = self.correlator(encoded)
            if correlated is None:
                 warnings.warn("Correlator None", RuntimeWarning)
                 correlated = encoded
            elif isinstance(correlated, torch.Tensor) and (torch.isnan(correlated).any() or torch.isinf(correlated).any()):
                 correlated = torch.nan_to_num(correlated, nan=0.0, posinf=1e4, neginf=-1e4)
            # نهاية if-elif
            attn_out = self.attention(correlated, correlated, correlated)
            if attn_out is None:
                 warnings.warn("Attention None", RuntimeWarning)
                 attn_out = correlated
            elif isinstance(attn_out, torch.Tensor) and (torch.isnan(attn_out).any() or torch.isinf(attn_out).any()):
                 attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
            # نهاية if-elif
            processed = self.post_attention_layer(attn_out)
            if torch.isnan(processed).any() or torch.isinf(processed).any():
                 processed = torch.zeros_like(processed)
            # نهاية if
            action_logits = self.actor(processed)
            risk_logits = self.risk_predictor(processed)
            stable_logits = action_logits - torch.max(action_logits, dim=-1, keepdim=True)[0]
            probs = torch.softmax(stable_logits, dim=-1)
            risk = torch.sigmoid(risk_logits)
            probs = torch.nan_to_num(probs, nan=(1./self.action_dim), posinf=(1./self.action_dim))
            probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)
            risk = torch.nan_to_num(risk, nan=0.5, posinf=1.0, neginf=0.0)
            return probs, risk
        except Exception as e:
             warnings.warn(f"Explorer forward error: {e}. Default.", RuntimeWarning)
             traceback.print_exc()
             default_probs = torch.ones(batch_size, self.action_dim, device=DEVICE)/self.action_dim
             default_risk = torch.ones(batch_size, 1, device=DEVICE)*0.5
             return default_probs, default_risk
        # نهاية try-except
    # نهاية forward
# نهاية فئة ExplorerPolicyNetwork


# ========================================
# 5. الوكيل الهجين (كما هي)
# ========================================
class ExpertExplorerAgentWithEvoExpert: # (كما هي)
    def __init__(self, env: AdvancedGridWorld, explorer_hidden_dim: int = 128, explorer_lr: float = 5e-4, explorer_grad_clip: float = 1.0, expert_init_complexity: int = 4, expert_complexity_limit: int = 12, expert_lr: float = 1e-3, evo_mutation_power: float = 0.02, evo_cooldown: int = 50, evo_add_thresh: float = 0.9, evo_prune_thresh: float = 0.15, gamma: float = 0.99, memory_size: int = 100000, batch_size: int = 256, expert_update_freq: int = 10, expert_evolve_freq: int = 100, initial_epsilon: float = 1.0, final_epsilon: float = 0.05, epsilon_decay_steps: int = 80000, tau_clamp_min: float = 0.0, tau_clamp_max: float = 30.0):
        self.env = env
        self.input_dim = env.size * env.size
        self.action_dim = env.action_space_n
        self.batch_size = batch_size
        self.gamma = gamma
        self.expert_update_freq = expert_update_freq
        self.expert_evolve_freq = expert_evolve_freq
        self.tau_epsilon = 1e-8
        self.explorer_grad_clip = explorer_grad_clip
        self.tau_clamp_min = tau_clamp_min
        self.tau_clamp_max = tau_clamp_max
        self.max_distance = env.max_distance if hasattr(env, 'max_distance') else 10.0
        self.explorer_net = ExplorerPolicyNetwork(self.input_dim, explorer_hidden_dim, self.action_dim).to(DEVICE)
        self.explorer_optimizer = optim.Adam(self.explorer_net.parameters(), lr=explorer_lr)
        self.memory = deque(maxlen=memory_size)
        self.expert_evolving_eq = EvolvingEquation(input_dim=self.input_dim, init_complexity=expert_init_complexity, output_dim=1, complexity_limit=expert_complexity_limit, min_complexity=2, output_activation=torch.sigmoid).to(DEVICE)
        self.expert_optimizer = optim.Adam(self.expert_evolving_eq.parameters(), lr=expert_lr)
        self.expert_evolution_engine = EvolutionEngine(mutation_power=evo_mutation_power, cooldown_period=evo_cooldown, add_term_threshold=evo_add_thresh, prune_term_threshold=evo_prune_thresh)
        self.expert_performance_metric = deque(maxlen=self.expert_evolution_engine.performance_history.maxlen)
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_rate = (initial_epsilon - final_epsilon) / epsilon_decay_steps
        if epsilon_decay_steps <= 0:
             self.epsilon_decay_rate = 0
        # نهاية if
        self.step_counter = 0
    # نهاية __init__
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]: # (كما هي)
        if self.step_counter < self.epsilon_decay_steps:
             self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay_rate)
        else:
             self.epsilon = self.final_epsilon
        # نهاية if-else
        action = 0
        explorer_risk_item = 0.5
        expert_risk_item = 0.5
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            self.explorer_net.eval()
            self.expert_evolving_eq.eval()
            action_probs, explorer_risk = self.explorer_net(state_tensor)
            expert_risk = self.expert_evolving_eq(state_tensor)
            self.explorer_net.train()
            self.expert_evolving_eq.train()
        # نهاية with
        explorer_risk_item = explorer_risk.item()
        expert_risk_item = expert_risk.item()
        if random.random() < self.epsilon:
             action = random.randrange(self.action_dim)
        else:
            probs_to_sample = action_probs.squeeze(0)
            if torch.isnan(probs_to_sample).any() or (probs_to_sample < 0).any() or probs_to_sample.sum() <= 1e-8:
                 action = random.randrange(self.action_dim)
            else:
                 try:
                      action = torch.multinomial(probs_to_sample, num_samples=1).item()
                 except RuntimeError:
                      action = random.randrange(self.action_dim)
                 # نهاية try-except
            # نهاية if-else
        # نهاية if-else
        return action, explorer_risk_item, expert_risk_item
    # نهاية select_action
    def store_experience(self, state, action, reward, next_state, done, explorer_risk, expert_risk, info): # (كما هي)
        distance_before = info.get('distance_before', self.max_distance)
        distance_after = info.get('distance_after', self.max_distance)
        distance_progress = max(0.0, distance_before - distance_after)
        normalized_progress = distance_progress
        progress_term = normalized_progress + 0.1
        tau_raw = progress_term / (explorer_risk + 0.1 + self.tau_epsilon)
        tau = np.clip(tau_raw, self.tau_clamp_min, self.tau_clamp_max)
        experience = (state, action, tau, next_state, done, explorer_risk)
        self.memory.append(experience)
    # نهاية store_experience
    def _update_explorer(self): # (كما هي - خسارة PG)
        if len(self.memory) < self.batch_size:
             return None
        # نهاية if
        batch = random.sample(self.memory, self.batch_size)
        states, actions, _, _, _, stored_explorer_risks = zip(*batch)
        state_batch = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        current_probs, current_explorer_risks = self.explorer_net(state_batch)
        log_probs = torch.log(current_probs.gather(1, action_batch) + self.tau_epsilon)
        advantage = (1.0 - current_explorer_risks).detach()
        explorer_loss = -(log_probs * advantage).mean()
        if torch.isnan(explorer_loss):
             warnings.warn("NaN in Explorer PG Loss", RuntimeWarning)
             return None
        # نهاية if
        self.explorer_optimizer.zero_grad()
        explorer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.explorer_net.parameters(), max_norm=self.explorer_grad_clip)
        self.explorer_optimizer.step()
        return explorer_loss.item()
    # نهاية _update_explorer
    def _update_expert_parameters(self): # (كما هي)
        if len(self.memory) < self.batch_size:
             return None
        # نهاية if
        batch = random.sample(self.memory, self.batch_size)
        states, _, _, _, _, stored_explorer_risks = zip(*batch)
        state_batch = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        target_risk_batch = torch.tensor(stored_explorer_risks, dtype=torch.float32, device=DEVICE).unsqueeze(1).detach()
        expert_predictions = self.expert_evolving_eq(state_batch)
        loss_fn = nn.MSELoss()
        expert_loss = loss_fn(expert_predictions, target_risk_batch)
        if torch.isnan(expert_loss):
             warnings.warn("NaN in Expert Param Loss", RuntimeWarning)
             return None
        # نهاية if
        self.expert_optimizer.zero_grad()
        expert_loss.backward()
        self.expert_optimizer.step()
        self.expert_performance_metric.append(expert_loss.item())
        return expert_loss.item()
    # نهاية _update_expert_parameters
    def _evolve_expert_structure(self, step: int): # (كما هي)
        if not self.expert_performance_metric:
             return False
        # نهاية if
        avg_loss_metric = np.mean(self.expert_performance_metric)
        performance_for_evolution = -avg_loss_metric
        structure_changed = self.expert_evolution_engine.evolve_equation(
            self.expert_evolving_eq, performance_for_evolution, step
        )
        if structure_changed:
             print(f"    Expert struct changed @ S {step}. Reinit optim.")
             self.expert_optimizer = optim.Adam(self.expert_evolving_eq.parameters(), lr=self.expert_optimizer.param_groups[0]['lr'])
        # نهاية if
        return structure_changed
    # نهاية _evolve_expert_structure
    def train_step(self): # (كما هي)
        state = self._current_state
        action, explorer_risk, expert_risk = self.select_action(state)
        next_state, reward, done, info = self.env.step(action)
        self.store_experience(state, action, reward, next_state, done, explorer_risk, expert_risk, info)
        self._current_state = next_state
        explorer_loss = self._update_explorer()
        expert_param_loss = None
        if self.step_counter % self.expert_update_freq == 0:
             expert_param_loss = self._update_expert_parameters()
        # نهاية if
        expert_structure_changed = False
        if self.step_counter > 0 and self.step_counter % self.expert_evolve_freq == 0 :
             expert_structure_changed = self._evolve_expert_structure(self.step_counter)
        # نهاية if
        reset_episode = False
        if done:
             self._current_state = self.env.reset()
             reset_episode = True
        # نهاية if
        self.step_counter += 1
        return explorer_loss, expert_param_loss, reward, reset_episode, info
    # نهاية train_step
    def train(self, num_total_steps=150000, print_every=10000): # (كما هي)
        print("Starting Training: Explorer (PG Loss) + Evolving Expert (Adam + EvolutionEngine)...")
        self._current_state = self.env.reset()
        explorer_losses = deque(maxlen=print_every//2)
        expert_param_losses = deque(maxlen=print_every//2)
        episode_rewards = []
        episode_avg_taus = []
        episode_steps = []
        current_episode_reward = 0.0
        current_episode_tau_sum = 0.0
        current_episode_steps = 0
        progress_bar = tqdm(range(num_total_steps), desc="Training", unit="step", ncols=120)
        for step in progress_bar:
            expl_loss, exp_param_loss, reward, episode_reset, info = self.train_step()
            current_episode_reward += reward
            current_episode_steps += 1
            if self.memory:
                 current_episode_tau_sum += self.memory[-1][2]
            # نهاية if
            if expl_loss is not None:
                 explorer_losses.append(expl_loss)
            # نهاية if
            if exp_param_loss is not None:
                 expert_param_losses.append(exp_param_loss)
            # نهاية if
            if (step + 1) % 100 == 0:
                 postfix_dict = {"Eps": f"{self.epsilon:.2f}"}
                 if explorer_losses:
                      postfix_dict["ExpL(PG)"] = f"{np.mean(explorer_losses):.3f}"
                 # نهاية if
                 if expert_param_losses:
                      postfix_dict["EvExpL"] = f"{np.mean(expert_param_losses):.3f}"
                 # نهاية if
                 if episode_rewards:
                      postfix_dict["AvgR"] = f"{np.mean(episode_rewards[-20:]):.1f}"
                 # نهاية if
                 if episode_steps:
                      postfix_dict["AvgS"] = f"{np.mean(episode_steps[-20:]):.0f}"
                 # نهاية if
                 progress_bar.set_postfix(postfix_dict)
            # نهاية if
            if (step + 1) % print_every == 0:
                print(f"\n--- Step [{step + 1}/{num_total_steps}], Epsilon: {self.epsilon:.3f} ---")
                if explorer_losses:
                     print(f"  Avg Explorer Loss (PG): {np.mean(explorer_losses):.4f}")
                # نهاية if
                if expert_param_losses:
                     print(f"  Avg Evolving Expert Loss (Adam): {np.mean(expert_param_losses):.4f}")
                # نهاية if
                if episode_rewards:
                     print(f"  Avg Reward (last ~20 ep): {np.mean(episode_rewards[-20:]):.2f}")
                # نهاية if
                if episode_steps:
                     print(f"  Avg Steps (last ~20 ep) : {np.mean(episode_steps[-20:]):.1f}")
                # نهاية if
                if episode_avg_taus:
                     print(f"  Avg Tau/Step (last ~20 ep): {np.mean(episode_avg_taus[-20:]):.3f}")
                # نهاية if
                print(f"  Expert Complexity: {self.expert_evolving_eq.complexity}")
                self.print_expert_equation()
            # نهاية if
            if episode_reset:
                 episode_rewards.append(current_episode_reward)
                 episode_steps.append(current_episode_steps)
                 avg_tau_this_episode = 0.0
                 if current_episode_steps > 0:
                      avg_tau_this_episode = current_episode_tau_sum / current_episode_steps
                 # نهاية if
                 episode_avg_taus.append(avg_tau_this_episode)
                 if (len(episode_rewards)) % 20 == 0:
                      print(f"  Ep {len(episode_rewards)} End. Steps:{current_episode_steps}. Reward:{current_episode_reward:.1f}.")
                 # نهاية if
                 current_episode_reward = 0.0
                 current_episode_tau_sum = 0.0
                 current_episode_steps = 0
            # نهاية if
        # نهاية for
        progress_bar.close()
        print("Training finished.")
        return episode_rewards, episode_avg_taus, episode_steps
    # نهاية train
    def print_expert_equation(self):
         print(f"  Expert Equation: {self.expert_evolving_eq.to_string()}")
    # نهاية print_expert_equation
# نهاية فئة ExpertExplorerAgentWithEvoExpert


# ========================================
# 6. التشغيل والاختبار (كما هي)
# ========================================
if __name__ == "__main__":
    # --- إعدادات التجربة (كما هي) ---
    GRID_SIZE = 5
    EXPLORER_HIDDEN = 128
    EXPERT_LR_ADAM = 5e-4 # اسم معدل تعلم المستكشف
    EXPLORER_GRAD_CLIP = 1.0
    EXPERT_INIT_COMP = 5
    EXPERT_COMP_LIMIT = 15
    EXPERT_LR_EVO = 2e-3 # اسم معدل تعلم الخبير
    EVO_MUT_POWER = 0.015
    EVO_COOLDOWN = 60
    EVO_ADD_THRESH = 0.9
    EVO_PRUNE_THRESH = 0.1
    GAMMA = 0.99
    MEMORY_SIZE = 150000
    BATCH_SIZE = 256
    EXPERT_UPDATE_FREQ = 5
    EXPERT_EVOLVE_FREQ = 250
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.02
    EPSILON_DECAY_STEPS = 120000
    TAU_CLAMP_MAX = 35.0
    TOTAL_STEPS = 200000
    PRINT_EVERY = 10000

    # --- إنشاء البيئة والوكيل ---
    env = AdvancedGridWorld(size=GRID_SIZE)
    agent = ExpertExplorerAgentWithEvoExpert(
        env,
        explorer_hidden_dim=EXPLORER_HIDDEN,
        explorer_lr=EXPERT_LR_ADAM, # استخدام الاسم الصحيح
        explorer_grad_clip=EXPLORER_GRAD_CLIP,
        expert_init_complexity=EXPERT_INIT_COMP,
        expert_complexity_limit=EXPERT_COMP_LIMIT,
        expert_lr=EXPERT_LR_EVO, # استخدام الاسم الصحيح
        evo_mutation_power=EVO_MUT_POWER,
        evo_cooldown=EVO_COOLDOWN,
        evo_add_thresh=EVO_ADD_THRESH,
        evo_prune_thresh=EVO_PRUNE_THRESH,
        gamma=GAMMA,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        expert_update_freq=EXPERT_UPDATE_FREQ,
        expert_evolve_freq=EXPERT_EVOLVE_FREQ,
        initial_epsilon=INITIAL_EPSILON,
        final_epsilon=FINAL_EPSILON,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        tau_clamp_max=TAU_CLAMP_MAX
    )

    # --- التدريب ---
    reward_history, tau_history, steps_history = agent.train(num_total_steps=TOTAL_STEPS, print_every=PRINT_EVERY)

    # --- قسم الاختبار ---
    print("\n--- Testing the Trained Agent ---")
    num_test_episodes = 15
    test_rewards = []
    test_steps = []
    agent.explorer_net.eval()
    agent.expert_evolving_eq.eval()
    for i in range(num_test_episodes):
        state = env.reset()
        done = False
        if i < 3:
             print(f"\nTest Episode {i+1}")
        # نهاية if
        step = 0
        total_reward = 0.0
        while not done and step < env.size * env.size * 5:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action_probs, explorer_risk = agent.explorer_net(state_tensor)
                expert_risk = agent.expert_evolving_eq(state_tensor)
            # نهاية with
            action = torch.argmax(action_probs.squeeze(0)).item()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            step += 1
        # نهاية while
        test_rewards.append(total_reward)
        test_steps.append(step)
        goal_reached = False
        if isinstance(state, np.ndarray) and hasattr(env, 'goal'):
            try:
                 agent_pos_flat_index = state.argmax()
                 agent_pos_coords = np.unravel_index(agent_pos_flat_index, (env.size, env.size))
                 if agent_pos_coords == env.goal:
                     goal_reached = True
                 # نهاية if
            except Exception:
                 pass
            # نهاية try-except
        # نهاية if
        print(f"  Test Ep {i+1}: {'Goal!' if goal_reached else 'Ended.'} Steps: {step}. Reward: {total_reward:.2f}")
    # نهاية for
    print(f"\nAverage Test Reward over {num_test_episodes} episodes: {np.mean(test_rewards):.2f}")
    print(f"Average Test Steps over {num_test_episodes} episodes: {np.mean(test_steps):.1f}")
    print("\n--- Final Expert Equation ---")
    agent.print_expert_equation()

    # --- رسم بياني ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    if reward_history:
        axs[0].plot(reward_history, label='Ep Reward', alpha=0.5, lw=1)
        if len(reward_history)>=30:
             moving_avg_reward = np.convolve(reward_history, np.ones(30)/30, mode='valid')
             axs[0].plot(np.arange(29,len(reward_history)), moving_avg_reward, label='MA(30)', c='r', lw=2)
        # نهاية if
        axs[0].set_ylabel('Total Reward')
        axs[0].set_title('Episode Reward')
        axs[0].grid(True, ls='--', alpha=0.6)
        axs[0].legend()
    # نهاية if
    if tau_history:
        axs[1].plot(tau_history, label='Avg Tau/Step (Monitor)', alpha=0.5, lw=1)
        if len(tau_history)>=30:
             moving_avg_tau = np.convolve(tau_history, np.ones(30)/30, mode='valid')
             axs[1].plot(np.arange(29,len(tau_history)), moving_avg_tau, label='MA(30)', c='orange', lw=2)
        # نهاية if
        axs[1].set_ylabel('Average Tau')
        axs[1].set_title('Average Tau per Step (Monitoring)')
        axs[1].grid(True, ls='--', alpha=0.6)
        axs[1].legend()
    # نهاية if
    if steps_history:
        axs[2].plot(steps_history, label='Steps/Ep', alpha=0.5, lw=1)
        if len(steps_history)>=30:
             moving_avg_steps = np.convolve(steps_history, np.ones(30)/30, mode='valid')
             axs[2].plot(np.arange(29,len(steps_history)), moving_avg_steps, label='MA(30)', c='g', lw=2)
        # نهاية if
        axs[2].set_ylabel('Steps')
        axs[2].set_title('Steps per Episode')
        axs[2].grid(True, ls='--', alpha=0.6)
        axs[2].legend()
        axs[2].set_xlabel('Completed Episode')
    # نهاية if
    plt.tight_layout(rect=[0,0.03,1,0.97])
    plot_filename = f"training_plot_expert_evo_pgloss_{time.strftime('%Y%m%d_%H%M%S')}.png"
    try:
        plt.savefig(plot_filename, dpi=300)
        print(f"\nTraining plot saved to: {plot_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    # نهاية try-except
    # plt.show()
# نهاية if __name__ == "__main__"