# -*- coding: utf-8 -*-
"""
==============================================================================
نظام OmniMind-UL / RL-E⁴: وكيل تعلم موحد مع معادلة قابلة للتكيف (v1.5.15 - إصلاح SyntaxError نهائي)
==============================================================================

نظام الخبير/ والمستكشف، هذا بخبرته وهذا بتجاربه، هذا يوجه وذاك يعطي نتائجه، وغير ذلك من تفاعل بينهما
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque, OrderedDict
import matplotlib
matplotlib.use('Agg')
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

# --- 2. Global Configuration & Seed ---
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# --- 3. Core Component: Evolving Equation ---
class EvolvingEquation(nn.Module):
    """
    يمثل معادلة رياضية مرنة تتطور بنيتها ومعاملاتها ديناميكيًا.
    """
    def __init__(self, input_dim, init_complexity=3, output_dim=1, complexity_limit=15, min_complexity=2, output_activation=None, exp_clamp_min=0.1, exp_clamp_max=4.0, term_clamp=1e4, output_clamp=1e5):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0: raise ValueError("input_dim must be a positive integer.")
        if not isinstance(output_dim, int) or output_dim <= 0: raise ValueError("output_dim must be a positive integer.")
        if not isinstance(init_complexity, int) or init_complexity <= 0: raise ValueError("init_complexity must be a positive integer.")
        if not isinstance(min_complexity, int) or min_complexity <= 0: raise ValueError("min_complexity must be a positive integer.")
        if not isinstance(complexity_limit, int) or complexity_limit < min_complexity: raise ValueError(f"complexity_limit ({complexity_limit}) must be an integer >= min_complexity ({min_complexity}).")
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
        self._initialize_components()
        self._func_repr_map = OrderedDict([
            ('sin', {'shape': 'sine', 'color': '#FF6347'}), ('cos', {'shape': 'cosine', 'color': '#4682B4'}), ('tanh', {'shape': 'wave', 'color': '#32CD32'}),
            ('sigmoid', {'shape': 'step', 'color': '#FFD700'}), ('relu', {'shape': 'ramp', 'color': '#DC143C'}), ('leaky_relu', {'shape': 'ramp', 'color': '#8A2BE2'}),
            ('gelu', {'shape': 'smoothramp', 'color': '#00CED1'}), ('<lambda>', {'shape': 'line', 'color': '#A9A9A9'}), ('pow', {'shape': 'parabola', 'color': '#6A5ACD'}),
            ('exp', {'shape': 'decay', 'color': '#FF8C00'}), ('sqrt', {'shape': 'root', 'color': '#2E8B57'}), ('clamp', {'shape': 'plateau', 'color': '#DAA520'}),
            ('*', {'shape': 'swishlike', 'color': '#D2691E'})])

    def _initialize_components(self):
        self.input_transform = nn.Linear(self.input_dim, self.complexity)
        nn.init.xavier_uniform_(self.input_transform.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.input_transform.bias)
        self.coefficients = nn.ParameterList([nn.Parameter(torch.randn(1) * 0.05, requires_grad=True) for _ in range(self.complexity)])
        self.exponents = nn.ParameterList([nn.Parameter(torch.abs(torch.randn(1) * 0.1) + 1.0, requires_grad=True) for _ in range(self.complexity)])
        self.function_library = [
            torch.sin, torch.cos, torch.tanh, torch.sigmoid, F.relu, F.leaky_relu, F.gelu,
            lambda x: x, lambda x: torch.pow(x, 2), lambda x: torch.exp(-torch.abs(x)),
            lambda x: torch.sqrt(torch.abs(x) + 1e-6), lambda x: torch.clamp(x, -3.0, 3.0),
            lambda x: x * torch.sigmoid(x),]
        if not self.function_library: raise ValueError("Function library cannot be empty.")
        self.functions = [self._select_function() for _ in range(self.complexity)]
        self.output_layer = nn.Linear(self.complexity, self.output_dim)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.output_layer.bias)

    def _select_function(self): return random.choice(self.function_library)
    def _safe_pow(self, base, exp): sign = torch.sign(base); base_abs_safe = torch.abs(base) + 1e-8; powered = torch.pow(base_abs_safe, exp); return sign * powered

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
             try: x = torch.tensor(x, dtype=torch.float32)
             except Exception as e: raise TypeError(f"Input 'x' must be tensor: {e}")
        if x.dim() == 1:
            if x.shape[0] == self.input_dim: x = x.unsqueeze(0)
            else: raise ValueError(f"1D Input dim {x.shape[0]} != expected {self.input_dim}")
        elif x.dim() > 2:
             original_shape = x.shape; x = x.view(x.shape[0], -1)
             if x.shape[1] != self.input_dim: raise ValueError(f"Flattened dim {x.shape[1]} != expected {self.input_dim}")
             warnings.warn(f"Input > 2D ({original_shape}). Flattened to ({x.shape}).", RuntimeWarning)
        elif x.dim() == 2:
            if x.shape[1] != self.input_dim: raise ValueError(f"2D Input dim {x.shape[1]} != expected {self.input_dim}")
        else: raise ValueError("Input tensor must be 1D/2D/Flattenable.")
        try: model_device = next(self.parameters()).device
        except StopIteration: model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x.device != model_device: x = x.to(model_device)
        if torch.isnan(x).any(): x = torch.nan_to_num(x, nan=0.0)
        try:
            transformed_features = self.input_transform(x)
            if torch.isnan(transformed_features).any(): transformed_features = torch.nan_to_num(transformed_features, nan=0.0)
            transformed_features = torch.clamp(transformed_features, -self.term_clamp, self.term_clamp)
        except Exception as e: return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        term_results = torch.zeros(x.shape[0], self.complexity, device=x.device)
        if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')): return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        for i in range(self.complexity):
            try:
                if i >= len(self.coefficients) or i >= len(self.exponents) or i >= len(self.functions): continue
                feature_i = transformed_features[:, i]
                exp_val = torch.clamp(self.exponents[i], self.exp_clamp_min, self.exp_clamp_max)
                term_powered = self._safe_pow(feature_i, exp_val)
                if torch.isnan(term_powered).any() or torch.isinf(term_powered).any(): term_powered = torch.zeros_like(feature_i)
                term_powered = torch.clamp(term_powered, -self.term_clamp, self.term_clamp)
                term_activated = self.functions[i](term_powered)
                if torch.isnan(term_activated).any() or torch.isinf(term_activated).any(): term_activated = torch.zeros_like(feature_i)
                term_activated = torch.clamp(term_activated, -self.term_clamp, self.term_clamp)
                term_value = self.coefficients[i] * term_activated
                term_results[:, i] = torch.clamp(term_value, -self.term_clamp, self.term_clamp)
            except Exception as e: term_results[:, i] = 0.0
        if torch.isnan(term_results).any() or torch.isinf(term_results).any(): term_results = torch.nan_to_num(term_results, nan=0.0, posinf=self.term_clamp, neginf=-self.term_clamp)
        try:
            output = self.output_layer(term_results)
            output = torch.clamp(output, -self.output_clamp, self.output_clamp)
            if torch.isnan(output).any() or torch.isinf(output).any(): output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            if self.output_activation:
                output = self.output_activation(output)
                if torch.isnan(output).any() or torch.isinf(output).any(): output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e: return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        if torch.isnan(output).any() or torch.isinf(output).any(): output = torch.zeros_like(output)
        return output

    def to_shape_engine_string(self) -> str:
        parts = []; param_scale = 5.0; exp_scale = 2.0
        if not (hasattr(self, 'coefficients') and hasattr(self, 'exponents') and hasattr(self, 'functions')): return "equation_not_fully_initialized"
        for i in range(self.complexity):
             if i >= len(self.coefficients) or i >= len(self.exponents) or i >= len(self.functions): continue
             coeff_val = round(self.coefficients[i].item(), 3); exp_val = round(torch.clamp(self.exponents[i], self.exp_clamp_min, self.exp_clamp_max).item(), 3)
             func = self.functions[i]; func_name_part = getattr(func, '__name__', '<lambda>')
             if func_name_part == '<lambda>':
                 if 'pow' in repr(func): func_name_part = 'pow'
                 elif '*' in repr(func) and 'sigmoid' in repr(func): func_name_part = '*'
             repr_info = self._func_repr_map.get(func_name_part, self._func_repr_map['<lambda>']); shape_type = repr_info['shape']
             p1=round(i*param_scale*0.5+coeff_val*param_scale*0.2,2); p2=round(coeff_val*param_scale,2); p3=round(abs(exp_val)*exp_scale,2); params_str = f"{p1},{p2},{p3}"
             styles = {'color': repr_info['color'], 'linewidth': round(1.0 + abs(exp_val - 1.0) * 1.5, 2), 'opacity': round(np.clip(0.4 + abs(coeff_val) * 0.5, 0.2, 0.9), 2)}
             if coeff_val < -0.01: styles['fill'] = 'True'
             if func_name_part in ['cos', 'relu', 'leaky_relu']: styles['dash'] = '--'
             styles_str = ",".join([f"{k}={v}" for k, v in styles.items()])
             term_str = f"{shape_type}({params_str}){{{styles_str}}}"
             parts.append(term_str)
        return " + ".join(parts) if parts else "empty_equation"

    def add_term(self):
        if self.complexity >= self.complexity_limit: return False
        new_complexity = self.complexity + 1;
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try: old_in_w = self.input_transform.weight.data.clone(); old_in_b = self.input_transform.bias.data.clone(); old_out_w = self.output_layer.weight.data.clone(); old_out_b = self.output_layer.bias.data.clone()
        except AttributeError: return False
        self.coefficients.append(nn.Parameter(torch.randn(1, device=device)*0.01, requires_grad=True))
        self.exponents.append(nn.Parameter(torch.abs(torch.randn(1, device=device)*0.05)+1.0, requires_grad=True))
        self.functions.append(self._select_function())
        new_in_t = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad(): new_in_t.weight.data[:self.complexity,:] = old_in_w; new_in_t.bias.data[:self.complexity] = old_in_b; nn.init.xavier_uniform_(new_in_t.weight.data[self.complexity:], gain=nn.init.calculate_gain('relu')); new_in_t.weight.data[self.complexity:] *= 0.01; nn.init.zeros_(new_in_t.bias.data[self.complexity:])
        self.input_transform = new_in_t
        new_out_l = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad(): new_out_l.weight.data[:,:self.complexity] = old_out_w; nn.init.xavier_uniform_(new_out_l.weight.data[:,self.complexity:], gain=nn.init.calculate_gain('linear')); new_out_l.weight.data[:,self.complexity:] *= 0.01; new_out_l.bias.data.copy_(old_out_b)
        self.output_layer = new_out_l
        self.complexity = new_complexity; return True

    def prune_term(self, aggressive=False):
        if self.complexity <= self.min_complexity: return False
        new_complexity = self.complexity - 1; idx_to_prune = -1
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if aggressive and self.complexity > 1:
                 with torch.no_grad():
                    if not hasattr(self, 'coefficients') or not self.coefficients: idx_to_prune = random.randint(0, self.complexity - 1)
                    else:
                        coeffs_abs = torch.tensor([torch.abs(c.data).item() for c in self.coefficients], device='cpu')
                        in_w = self.input_transform.weight; out_w = self.output_layer.weight
                        in_norm = torch.norm(in_w.data.cpu(), p=1, dim=1) if in_w is not None else torch.zeros(self.complexity, device='cpu')
                        out_norm = torch.norm(out_w.data.cpu(), p=1, dim=0) if out_w is not None else torch.zeros(self.complexity, device='cpu')
                        if coeffs_abs.shape[0]==self.complexity and in_norm.shape[0]==self.complexity and out_norm.shape[0]==self.complexity:
                            importance = (coeffs_abs*(in_norm+out_norm))+1e-9
                            if torch.isnan(importance).any(): idx_to_prune = random.randint(0, self.complexity - 1)
                            else: idx_to_prune = torch.argmin(importance).item()
                        else: idx_to_prune = random.randint(0, self.complexity - 1)
            else: idx_to_prune = random.randint(0, self.complexity - 1)
            if not (0 <= idx_to_prune < self.complexity): return False
        except Exception as e: return False
        try: old_in_w = self.input_transform.weight.data.clone(); old_in_b = self.input_transform.bias.data.clone(); old_out_w = self.output_layer.weight.data.clone(); old_out_b = self.output_layer.bias.data.clone()
        except AttributeError: return False
        try:
            if hasattr(self,'coefficients') and len(self.coefficients)>idx_to_prune: del self.coefficients[idx_to_prune]
            if hasattr(self,'exponents') and len(self.exponents)>idx_to_prune: del self.exponents[idx_to_prune]
            if hasattr(self,'functions') and len(self.functions)>idx_to_prune: self.functions.pop(idx_to_prune)
        except (IndexError,Exception) as e: return False
        new_in_t = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad():
            new_w_in = torch.cat([old_in_w[:idx_to_prune], old_in_w[idx_to_prune+1:]], dim=0); new_b_in = torch.cat([old_in_b[:idx_to_prune], old_in_b[idx_to_prune+1:]])
            if new_w_in.shape[0]!=new_complexity or new_b_in.shape[0]!=new_complexity: return False
            new_in_t.weight.data.copy_(new_w_in); new_in_t.bias.data.copy_(new_b_in)
        self.input_transform = new_in_t
        new_out_l = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad():
            new_w_out = torch.cat([old_out_w[:,:idx_to_prune], old_out_w[:,idx_to_prune+1:]], dim=1)
            if new_w_out.shape[1]!=new_complexity: return False
            new_out_l.weight.data.copy_(new_w_out); new_out_l.bias.data.copy_(old_out_b)
        self.output_layer = new_out_l
        self.complexity = new_complexity; return True


# --- 4. Core Component: Evolution Engine ---
class EvolutionEngine:
    """
    محرك التطور: يدير عملية تطور كائن `EvolvingEquation`.
    """
    def __init__(self, mutation_power=0.03, history_size=50, cooldown_period=30, add_term_threshold=0.85, prune_term_threshold=0.20, add_term_prob=0.15, prune_term_prob=0.25, swap_func_prob=0.02):
        if not (0 < mutation_power < 0.5): warnings.warn(f"mutation_power ({mutation_power}) outside typical range.")
        if not (0 < add_term_threshold < 1): warnings.warn(f"add_term_threshold ({add_term_threshold}) outside (0, 1).")
        if not (0 < prune_term_threshold < 1): warnings.warn(f"prune_term_threshold ({prune_term_threshold}) outside (0, 1).")
        if add_term_threshold <= prune_term_threshold: warnings.warn(f"add_term_threshold <= prune_term_threshold.")
        if not (0 <= add_term_prob <= 1) or not (0 <= prune_term_prob <= 1) or not (0 <= swap_func_prob <= 1): warnings.warn("Evolution probs should be between 0 and 1.")
        self.base_mutation_power = mutation_power; self.performance_history = deque(maxlen=history_size)
        self.cooldown_period = cooldown_period; self.term_change_cooldown = 0
        self.add_term_threshold = add_term_threshold; self.prune_term_threshold = prune_term_threshold
        self.add_term_prob = add_term_prob; self.prune_term_prob = prune_term_prob; self.swap_func_prob = swap_func_prob

    def _calculate_percentile(self, current_reward):
        if math.isnan(current_reward): return 0.5
        valid_history = [r for r in self.performance_history if not math.isnan(r)];
        if not valid_history: return 0.5
        history_array = np.array(valid_history)
        try: percentile = stats.percentileofscore(history_array, current_reward, kind='mean') / 100.0
        except Exception: percentile = np.mean(history_array < current_reward)
        return np.clip(percentile, 0.0, 1.0)

    def _dynamic_mutation_scale(self, percentile):
        if math.isnan(percentile): percentile = 0.5
        if percentile > 0.9: scale = 0.5
        elif percentile < 0.1: scale = 1.5
        else: scale = 1.5 - (percentile - 0.1) * (1.5 - 0.5) / (0.9 - 0.1)
        return max(0.1, min(scale, 2.0))

    def evolve_equation(self, equation, reward, step):
        structure_changed = False
        if not isinstance(equation, EvolvingEquation): return False
        if not math.isnan(reward): self.performance_history.append(reward); percentile = self._calculate_percentile(reward)
        else: valid_history = [r for r in self.performance_history if not math.isnan(r)]; percentile = self._calculate_percentile(valid_history[-1]) if valid_history else 0.5
        if self.term_change_cooldown > 0: self.term_change_cooldown -= 1
        if self.term_change_cooldown == 0 and len(self.performance_history) >= max(10, self.performance_history.maxlen // 4):
            rand_roll = random.random(); action_taken = "None"
            if percentile > self.add_term_threshold and rand_roll < self.add_term_prob:
                if equation.add_term(): self.term_change_cooldown = self.cooldown_period; structure_changed = True; action_taken = f"Added Term (Comp:{equation.complexity})"
            elif percentile < self.prune_term_threshold and rand_roll < self.prune_term_prob:
                if equation.prune_term(aggressive=True): self.term_change_cooldown = self.cooldown_period; structure_changed = True; action_taken = f"Pruned Term (Comp:{equation.complexity})"
            if structure_changed: print(f"\nEVO [{time.strftime('%H:%M:%S')}]: Upd {step}: Eq {type(equation).__name__} -> {action_taken} | Pct: {percentile:.2f} | CD: {self.term_change_cooldown}")
        if not structure_changed and self.term_change_cooldown == 0 and random.random() < self.swap_func_prob:
            if self.swap_function(equation): self.term_change_cooldown = max(self.term_change_cooldown, 2)
        mutation_scale = self._dynamic_mutation_scale(percentile); self._mutate_parameters(equation, mutation_scale, step)
        return structure_changed

    def _mutate_parameters(self, equation, mutation_scale, step):
        if not isinstance(equation, EvolvingEquation): return
        cooling_factor = max(0.5, 1.0 - step / 2000000.0); effective_power = self.base_mutation_power * mutation_scale * cooling_factor
        if effective_power < 1e-9: return
        try:
            with torch.no_grad():
                try: device = next(equation.parameters()).device
                except StopIteration: return
                if hasattr(equation, 'coefficients') and equation.coefficients:
                    for coeff in equation.coefficients: noise = torch.randn_like(coeff.data) * effective_power; coeff.data.add_(noise)
                if hasattr(equation, 'exponents') and equation.exponents:
                    for exp in equation.exponents: noise = torch.randn_like(exp.data) * effective_power * 0.1; exp.data.add_(noise); exp.data.clamp_(min=equation.exp_clamp_min, max=equation.exp_clamp_max)
                if hasattr(equation, 'input_transform') and isinstance(equation.input_transform, nn.Linear):
                    ps = 0.3; noise_w = torch.randn_like(equation.input_transform.weight.data)*effective_power*ps; equation.input_transform.weight.data.add_(noise_w)
                    if equation.input_transform.bias is not None: noise_b = torch.randn_like(equation.input_transform.bias.data)*effective_power*ps*0.5; equation.input_transform.bias.data.add_(noise_b)
                if hasattr(equation, 'output_layer') and isinstance(equation.output_layer, nn.Linear):
                    ps = 0.3; noise_w = torch.randn_like(equation.output_layer.weight.data)*effective_power*ps; equation.output_layer.weight.data.add_(noise_w)
                    if equation.output_layer.bias is not None: noise_b = torch.randn_like(equation.output_layer.bias.data)*effective_power*ps*0.5; equation.output_layer.bias.data.add_(noise_b)
        except Exception as e: warnings.warn(f"Exception during param mutation: {e}", RuntimeWarning)

    def swap_function(self, equation):
        if not isinstance(equation, EvolvingEquation) or equation.complexity <= 0 or not hasattr(equation, 'functions') or not equation.functions or not equation.function_library: return False
        try:
            idx = random.randint(0, equation.complexity - 1); old_f = equation.functions[idx]
            attempts=0; max_att=len(equation.function_library)*2; new_f=old_f
            while attempts < max_att:
                cand_f = equation._select_function(); cand_repr = getattr(cand_f, '__name__', repr(cand_f)); old_repr = getattr(old_f, '__name__', repr(old_f))
                is_diff = (cand_repr != old_repr) or (cand_repr == 'lambda' and cand_f is not old_f) or (len(equation.function_library) == 1)
                if is_diff or attempts == max_att-1: new_f=cand_f; break
                attempts += 1
            equation.functions[idx] = new_f; return True
        except (IndexError, Exception) as e: warnings.warn(f"Exception function swap: {e}", RuntimeWarning); return False


# --- 5. Core Component: Replay Buffer ---
class ReplayBuffer:
    """
    ذاكرة تخزين مؤقت للتجارب مع فحص NaN.
    """
    def __init__(self, capacity=100000):
        if not isinstance(capacity, int) or capacity <= 0: raise ValueError("Replay buffer capacity must be positive.")
        self.capacity = capacity; self.buffer = deque(maxlen=capacity)
        self._push_nan_warnings = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}; self._sample_nan_warning = 0
    def push(self, state, action, reward, next_state, done):
        skip = False; nan_src = None
        try:
            if isinstance(reward, (int, float)) and (math.isnan(reward) or math.isinf(reward)): reward = 0.0; self._push_nan_warnings['reward'] += 1; nan_src = 'reward (fixed)'
            s_arr = np.asarray(state, dtype=np.float32); ns_arr = np.asarray(next_state, dtype=np.float32); a_arr = np.asarray(action, dtype=np.float32)
            if np.any(np.isnan(s_arr)) or np.any(np.isinf(s_arr)): skip = True; self._push_nan_warnings['state'] += 1; nan_src = 'state'
            if not skip and (np.any(np.isnan(ns_arr)) or np.any(np.isinf(ns_arr))): skip = True; self._push_nan_warnings['next_state'] += 1; nan_src = 'next_state'
            if not skip and (np.any(np.isnan(a_arr)) or np.any(np.isinf(a_arr))): skip = True; self._push_nan_warnings['action'] += 1; nan_src = 'action'
            if skip:
                tot_warn = self._push_nan_warnings['state'] + self._push_nan_warnings['next_state'] + self._push_nan_warnings['action']
                if tot_warn % 500 == 1: warnings.warn(f"Skip exp NaN/Inf in '{nan_src}'. Skips:S{self._push_nan_warnings['state']},A{self._push_nan_warnings['action']},S'{self._push_nan_warnings['next_state']}", RuntimeWarning)
                return
            self.buffer.append((s_arr, a_arr, float(reward), ns_arr, float(done)))
        except (TypeError, ValueError) as e: warnings.warn(f"Could not process/store experience: {e}. Skip.", RuntimeWarning)
    def sample(self, batch_size):
        if len(self.buffer) < batch_size: return None
        try: indices = np.random.choice(len(self.buffer), batch_size, replace=False); batch = [self.buffer[i] for i in indices]
        except (ValueError, Exception) as e: warnings.warn(f"Error sampling buffer: {e}.", RuntimeWarning); return None
        try:
            s, a, r, ns, d = zip(*batch)
            s_np=np.array(s,dtype=np.float32); a_np=np.array(a,dtype=np.float32); r_np=np.array(r,dtype=np.float32).reshape(-1,1)
            ns_np=np.array(ns,dtype=np.float32); d_np=np.array(d,dtype=np.float32).reshape(-1,1)
            if np.any(np.isnan(s_np)) or np.any(np.isnan(a_np)) or np.any(np.isnan(r_np)) or np.any(np.isnan(ns_np)) or np.any(np.isnan(d_np)) or \
               np.any(np.isinf(s_np)) or np.any(np.isinf(a_np)) or np.any(np.isinf(r_np)) or np.any(np.isinf(ns_np)) or np.any(np.isinf(d_np)):
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
    """الفئة الأساسية المجردة لمكونات المعادلة الموحدة."""
    def __init__(self, component_id: str, tags: Set[str] = None, required_context: Set[str] = None, provides: Set[str] = None): # Use Set for req_context/provides
        super().__init__()
        self.component_id = component_id
        self.tags = tags if tags else set()
        self.required_context = required_context if required_context else set()
        self.provides = provides if provides else set()
        frame = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame)
        self._init_args = {arg: values[arg] for arg in args if arg != 'self'}
        # Convert sets to lists only for saving in _init_args
        if 'tags' in self._init_args and self._init_args['tags'] is not None: self._init_args['tags'] = list(self._init_args['tags'])
        if 'required_context' in self._init_args and self._init_args['required_context'] is not None: self._init_args['required_context'] = list(self._init_args['required_context'])
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self._init_args['provides'])

    def is_active(self, context: Dict[str, Any]) -> bool:
        if not self.required_context: return True
        return self.required_context.issubset(context.keys())

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Returns update dictionary
        raise NotImplementedError("Subclasses must implement forward returning update dict.")

    def evolve(self, engine: EvolutionEngine, reward: float, step: int) -> bool: return False
    def to_string(self) -> str: req_str = ','.join(self.required_context) if self.required_context else 'Any'; prov_str = ','.join(self.provides) if self.provides else 'None'; tag_str = ','.join(self.tags) if self.tags else 'None'; return f"Component(id={self.component_id}, tags={{{tag_str}}}, req={{{req_str}}}, prov={{{prov_str}}})"
    def get_complexity(self) -> float: return 1.0


class NumericTransformComponent(EquationComponent):
    """مكون يغلف EvolvingEquation للتحويلات العددية."""
    def __init__(self, component_id: str, equation: EvolvingEquation, activation: Optional[Callable] = None, tags: Set[str] = None, required_context: Set[str] = None, provides: Set[str] = None, input_key: str = 'features', output_key: str = 'features'):
        super().__init__(component_id, tags, required_context, provides)
        self.equation = equation; self.activation = activation
        self.input_key = input_key; self.output_key = output_key
        self._init_args['equation_config'] = {'input_dim': equation.input_dim, 'init_complexity': equation.complexity, 'output_dim': equation.output_dim, 'complexity_limit': equation.complexity_limit, 'min_complexity': equation.min_complexity, 'exp_clamp_min': equation.exp_clamp_min, 'exp_clamp_max': equation.exp_clamp_max, 'term_clamp': equation.term_clamp, 'output_clamp': equation.output_clamp,}
        self._init_args['activation_name'] = activation.__name__ if activation and hasattr(activation, '__name__') else 'None'
        self._init_args.pop('equation', None); self._init_args.pop('activation', None)
        self._init_args['input_key'] = input_key; self._init_args['output_key'] = output_key
        self.provides = self.provides.union({output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Returns update dictionary
        if self.input_key not in data: warnings.warn(f"In key '{self.input_key}' missing comp '{self.component_id}'. Skip.", RuntimeWarning); return {}
        x = data[self.input_key]
        if not isinstance(x, torch.Tensor): warnings.warn(f"In '{self.input_key}' comp '{self.component_id}' not tensor. Skip.", RuntimeWarning); return {}
        # Dimension checks
        if x.dim() == 0: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
             if x.shape[0] == self.equation.input_dim: x = x.unsqueeze(0)
             else: warnings.warn(f"Dim mismatch 1D '{self.component_id}'. Exp {self.equation.input_dim}, got {x.shape[0]}. Skip.", RuntimeWarning); return {}
        elif x.dim() == 2:
            if x.shape[1] != self.equation.input_dim: warnings.warn(f"Dim mismatch 2D '{self.component_id}'. Exp {self.equation.input_dim}, got {x.shape[1]}. Skip.", RuntimeWarning); return {}
        else:
             batch_size = x.shape[0]; x = x.view(batch_size, -1)
             if x.shape[1] != self.equation.input_dim: warnings.warn(f"Flattened dim mismatch >2D '{self.component_id}'. Exp {self.equation.input_dim}, got {x.shape[1]}. Skip.", RuntimeWarning); return {}
        try:
            transformed_x = self.equation(x)
            if torch.isnan(transformed_x).any() or torch.isinf(transformed_x).any(): warnings.warn(f"NaN/Inf eq '{self.component_id}'. Clamp.", RuntimeWarning); transformed_x = torch.nan_to_num(transformed_x, nan=0.0, posinf=1.0, neginf=-1.0)
            if self.activation:
                transformed_x = self.activation(transformed_x)
                if torch.isnan(transformed_x).any() or torch.isinf(transformed_x).any(): warnings.warn(f"NaN/Inf act '{self.component_id}'. Clamp.", RuntimeWarning); transformed_x = torch.nan_to_num(transformed_x, nan=0.0, posinf=1.0, neginf=-1.0)
            return {self.output_key: transformed_x}
        except Exception as e: warnings.warn(f"Error eq/act '{self.component_id}': {e}. Skip.", RuntimeWarning); traceback.print_exc()
        return {}

    def evolve(self, engine: EvolutionEngine, reward: float, step: int) -> bool:
        structure_changed = engine.evolve_equation(self.equation, reward, step)
        if structure_changed: self._init_args['equation_config']['init_complexity'] = self.equation.complexity
        return structure_changed

    def to_string(self) -> str:
        act_name = self._init_args.get('activation_name', 'None')
        eq_str = self.equation.to_shape_engine_string() if hasattr(self.equation, 'to_shape_engine_string') else "N/A"
        comp = self.get_complexity()
        base_str = super().to_string().replace("Component","NumericTransform")
        return f"{base_str[:-1]}, EqComp={comp:.1f}, Act={act_name}, In='{self.input_key}', Out='{self.output_key}', Eq='{eq_str}')"

    def get_complexity(self) -> float: return self.equation.complexity if hasattr(self.equation, 'complexity') else 1.0


class PolicyDecisionComponent(EquationComponent):
    """مكون لاتخاذ قرار السياسة (يطبق tanh)."""
    def __init__(self, component_id: str, action_dim: int, tags: Set[str] = None, required_context: Set[str] = None, provides: Set[str] = None, input_key: str = 'policy_features', output_key: str = 'action_raw'):
        tags = tags or {'actor'}; required_context = required_context or {'request_action'}; provides = provides or {output_key}
        super().__init__(component_id, tags, required_context, provides)
        self.action_dim = action_dim; self.input_key = input_key; self.output_key = output_key
        self._init_args['action_dim'] = action_dim; self._init_args['input_key'] = input_key; self._init_args['output_key'] = output_key
        self.provides = self.provides.union({output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Returns update dictionary
        if self.input_key not in data: warnings.warn(f"In key '{self.input_key}' missing comp '{self.component_id}'. Skip.", RuntimeWarning); return {}
        policy_features = data[self.input_key]
        if not isinstance(policy_features, torch.Tensor): warnings.warn(f"In '{self.input_key}' comp '{self.component_id}' not tensor. Skip.", RuntimeWarning); return {}
        action_tanh = torch.tanh(policy_features)
        if torch.isnan(action_tanh).any() or torch.isinf(action_tanh).any(): warnings.warn(f"NaN/Inf tanh '{self.component_id}'. Clamp.", RuntimeWarning); action_tanh = torch.nan_to_num(action_tanh, nan=0.0, posinf=1.0, neginf=-1.0)
        return {self.output_key: action_tanh}

    def to_string(self) -> str:
        base_str = super().to_string().replace("Component","PolicyDecision")
        return f"{base_str[:-1]}, Dim={self.action_dim}, In='{self.input_key}', Out='{self.output_key}')"


class ValueEstimationComponent(EquationComponent):
    """مكون لتقدير القيمة (يضمن مخرجًا واحدًا)."""
    def __init__(self, component_id: str, tags: Set[str] = None, required_context: Set[str] = None, provides: Set[str] = None, input_key: str = 'value_features', output_key: str = 'q_value'):
        tags = tags or {'critic'}; required_context = required_context or {'request_q_value'}; provides = provides or {output_key}
        super().__init__(component_id, tags, required_context, provides)
        self.input_key = input_key; self.output_key = output_key
        self._init_args['input_key'] = input_key; self._init_args['output_key'] = output_key
        self.provides = self.provides.union({output_key})
        if 'provides' in self._init_args and self._init_args['provides'] is not None: self._init_args['provides'] = list(self.provides)
        else: self._init_args['provides'] = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Returns update dictionary
        if self.input_key not in data: warnings.warn(f"In key '{self.input_key}' missing comp '{self.component_id}'. Skip.", RuntimeWarning); return {}
        value_features = data[self.input_key]
        if not isinstance(value_features, torch.Tensor): warnings.warn(f"In '{self.input_key}' comp '{self.component_id}' not tensor. Skip.", RuntimeWarning); return {}
        if value_features.shape[-1] != 1:
           warnings.warn(f"ValueEst '{self.component_id}' input dim {value_features.shape[-1]} != 1. Taking mean.", RuntimeWarning)
           value_features = value_features.mean(dim=-1, keepdim=True)
        if torch.isnan(value_features).any() or torch.isinf(value_features).any(): warnings.warn(f"NaN/Inf ValueEst '{self.component_id}'. Clamp 0.", RuntimeWarning); value_features = torch.nan_to_num(value_features, nan=0.0)
        return {self.output_key: value_features}

    def to_string(self) -> str:
        base_str = super().to_string().replace("Component","ValueEstimation")
        return f"{base_str[:-1]}, In='{self.input_key}', Out='{self.output_key}')"


# --- 7. المكون الجديد: المعادلة الموحدة القابلة للتكيف ---
class UnifiedAdaptableEquation(nn.Module):
    """
    معادلة موحدة تحتوي على مكونات اختيارية قابلة للتطور.
    تدير التنفيذ السياقي والتطور المتكامل.
    (v1.5.14 - Forward Pass v3, return update_dict)
    """
    def __init__(self, equation_id: str):
        super().__init__()
        self.equation_id = equation_id
        self.components = nn.ModuleDict()
        self._execution_order = []

    def add_component(self, component: EquationComponent, execute_after: Optional[str] = None):
        comp_id = component.component_id
        if comp_id in self.components: raise ValueError(f"Component id '{comp_id}' already exists.")
        self.components[comp_id] = component
        if execute_after is None or not self._execution_order: self._execution_order.append(comp_id)
        else:
            try: idx = self._execution_order.index(execute_after); self._execution_order.insert(idx + 1, comp_id)
            except ValueError: warnings.warn(f"Exec after ID '{execute_after}' not found. Append '{comp_id}'.", RuntimeWarning); self._execution_order.append(comp_id)

    def forward(self, initial_data: Dict[str, torch.Tensor], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        التمرير الأمامي (v1.5.14):
        - يعتمد على إعادة المكونات الفرعية لقاموس التحديث.
        - يحدث البيانات بعد كل مكون ناجح.
        """
        current_data = initial_data.copy()
        active_components_ids = {cid for cid, c in self.components.items() if c.is_active(context)}
        if not active_components_ids: return current_data

        executed_components = set()
        max_passes = len(self.components) + 2
        execution_queue = [cid for cid in self._execution_order if cid in active_components_ids]
        pending_components = set(execution_queue)

        for _pass in range(max_passes):
            if not pending_components: break
            made_progress_this_pass = False
            next_execution_queue = []
            current_pass_queue = execution_queue[:]
            execution_queue = []

            for component_id in current_pass_queue:
                if component_id in pending_components:
                    component = self.components[component_id]
                    input_key = getattr(component, 'input_key', None)
                    if input_key is None or input_key in current_data:
                        try:
                            update_dict = component(current_data, context)
                            if not isinstance(update_dict, dict):
                                warnings.warn(f"Component '{component_id}' forward did not return a dict. Skipping update.", RuntimeWarning)
                                update_dict = {}

                            current_data.update(update_dict)
                            executed_components.add(component_id)
                            pending_components.remove(component_id)
                            made_progress_this_pass = True

                            for key, tensor in update_dict.items():
                                if isinstance(tensor, torch.Tensor) and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                                    warnings.warn(f"NaN/Inf key '{key}' after comp '{component_id}'. Clamp.", RuntimeWarning)
                                    current_data[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                        except Exception as e:
                            warnings.warn(f"Error executing component '{component_id}': {e}", RuntimeWarning)
                            traceback.print_exc()
                            if component_id in pending_components: pending_components.remove(component_id)
                    else:
                        next_execution_queue.append(component_id)

            execution_queue = next_execution_queue
            if not made_progress_this_pass and pending_components:
                 warnings.warn(f"Execution stalled pass {_pass+1}. Pending:{pending_components}, Available:{list(current_data.keys())}. Cycle/MissingKey?", RuntimeWarning)
                 break

        final_needed = active_components_ids - executed_components
        if final_needed:
            warnings.warn(f"Not all active components executed after {_pass+1} passes. Remaining: {final_needed}", RuntimeWarning)

        return current_data

    def evolve(self, evolution_engine: EvolutionEngine, reward: float, step: int) -> List[Tuple[str, Set[str]]]:
        changed_components_info = []
        for component_id, component in self.components.items():
            if hasattr(component, 'evolve') and callable(component.evolve):
                try:
                    structure_changed = component.evolve(evolution_engine, reward, step)
                    if structure_changed:
                        print(f"EVO-COMP [{time.strftime('%H:%M:%S')}]: Struct change comp '{component_id}' eq '{self.equation_id}'")
                        changed_components_info.append((component_id, getattr(component, 'tags', set())))
                except Exception as e: warnings.warn(f"Error evolving comp '{component_id}': {e}", RuntimeWarning)
        return changed_components_info
    def get_total_complexity(self) -> float: return sum(comp.get_complexity() for comp in self.components.values())
    def to_string(self) -> str:
        lines = [f"UnifiedEquation(id={self.equation_id}, Complexity={self.get_total_complexity():.2f})"]
        lines.append(" Execution Order & Components:")
        component_details = {cid: comp.to_string() for cid, comp in self.components.items()}
        for i, comp_id in enumerate(self._execution_order):
            detail = component_details.get(comp_id, f"ERROR: ID '{comp_id}' not found in components")
            lines.append(f"  {i+1}. {detail}")
        ordered_ids = set(self._execution_order)
        unordered = [f"  - {comp.to_string()}" for cid, comp in self.components.items() if cid not in ordered_ids]
        if unordered: lines.append(" Components NOT in Execution Order (Check add_component calls):"); lines.extend(unordered)
        return "\n".join(lines)
    def get_parameters(self): return self.parameters()
    def get_tagged_parameters(self, required_tags: Set[str]):
        tagged_params = []; processed_params = set()
        for component in self.components.values():
             comp_tags = getattr(component, 'tags', set())
             if 'shared' in comp_tags or not required_tags.isdisjoint(comp_tags):
                  try:
                       for param in component.parameters():
                            if id(param) not in processed_params: tagged_params.append(param); processed_params.add(id(param))
                  except Exception as e: warnings.warn(f"Could not get params comp '{getattr(component,'component_id','Unknown')}': {e}", RuntimeWarning)
        return tagged_params


# --- 8. Core Component: Unified Learning Agent ---
class UnifiedLearningAgent(nn.Module):
    """
    وكيل تعلم معزز موحد يستخدم UnifiedAdaptableEquation،
    مع محسنات منفصلة وتحديث DDPG-like.
    """
    def __init__(self,
                 state_dim: int, action_dim: int, action_bounds: Tuple[float, float],
                 unified_equation: UnifiedAdaptableEquation, evolution_engine: EvolutionEngine,
                 gamma: float = 0.99, tau: float = 0.005,
                 actor_lr: float = 1e-4, critic_lr: float = 3e-4,
                 buffer_capacity: int = int(1e6), batch_size: int = 128,
                 exploration_noise_std: float = 0.3, noise_decay_rate: float = 0.9999, min_exploration_noise: float = 0.1,
                 weight_decay: float = 1e-5, grad_clip_norm: Optional[float] = 1.0, grad_clip_value: Optional[float] = None):
        super().__init__() # *** تم التأكد من أن super().__init__() في البداية ***

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: UnifiedLearningAgent(v1.5.14) using device: {self.device}") # تحديث الإصدار

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm is not None and grad_clip_norm > 0 else None
        self.grad_clip_value = grad_clip_value if grad_clip_value is not None and grad_clip_value > 0 else None

        if action_bounds is None or len(action_bounds) != 2:
            raise ValueError("action_bounds must be (min, max).")
        self.action_low = float(action_bounds[0])
        self.action_high = float(action_bounds[1])
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be < action_high.")

        self.register_buffer('action_scale', torch.tensor((self.action_high - self.action_low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((self.action_high + self.action_low) / 2.0, dtype=torch.float32))
        if (self.action_scale <= 1e-6).any():
            warnings.warn(f"Action scale near zero: {self.action_scale}. Check.", RuntimeWarning)
            with torch.no_grad():
                 self.action_scale.clamp_(min=1e-6)

        if not isinstance(unified_equation, nn.Module):
             raise TypeError("unified_equation must be an nn.Module.")
        self.unified_eq = unified_equation.to(self.device)

        if hasattr(self.unified_eq, '_execution_order'):
            self._execution_order = self.unified_eq._execution_order[:]
        else:
            warnings.warn("Unified eq missing '_execution_order'. Using empty.", RuntimeWarning)
            self._execution_order = []

        # --- إنشاء الشبكة الهدف ---
        self.target_unified_eq = None # تهيئة أولية
        self._initialize_target_network() # استدعاء الدالة المساعدة

        # --- طباعة معلومات المعادلة ---
        print("\n--- Initial Unified Adaptable Equation ---")
        print(self.unified_eq.to_string())
        print("-" * 50 + "\n")

        # --- تهيئة بقية المتغيرات ---
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay

        self.actor_optimizer = self._create_optimizer(actor_lr, weight_decay, {'actor', 'shared'})
        self.critic_optimizer = self._create_optimizer(critic_lr, weight_decay, {'critic', 'shared'})

        if not isinstance(evolution_engine, EvolutionEngine):
             raise TypeError("evolution_engine must be an EvolutionEngine instance.")
        self.evolution_engine = evolution_engine
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.exploration_noise_std = exploration_noise_std
        self.min_exploration_noise = min_exploration_noise
        self.noise_decay_rate = noise_decay_rate
        self.current_noise_std = exploration_noise_std
        self.total_updates = 0
        self.total_evolutions = 0

    def _initialize_target_network(self):
        """ينشئ الشبكة الهدف ويجمد معلماتها."""
        try:
            self.target_unified_eq = copy.deepcopy(self.unified_eq)
            if self.target_unified_eq is None:
                raise RuntimeError("Failed to create target_unified_eq via deepcopy.")
            # تجميد المعلمات
            for param in self.target_unified_eq.parameters():
                param.requires_grad = False
        except Exception as e_init_tgt:
            print(f"CRITICAL ERROR during target network initialization: {e_init_tgt}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize target network") from e_init_tgt

    def _get_tagged_parameters(self, required_tags: Set[str]): return self.unified_eq.get_tagged_parameters(required_tags)
    def _create_optimizer(self, lr: float, wd: float, tags: Set[str]) -> Optional[optim.Optimizer]:
        params = self._get_tagged_parameters(tags)
        if params: num_params = sum(p.numel() for p in params); print(f"Creating optimizer tags {tags}: {len(params)} tensors ({num_params:,} params)."); return optim.AdamW(params, lr=lr, weight_decay=wd)
        else: warnings.warn(f"No parameters found tags {tags}. Optimizer not created.", RuntimeWarning); return None
    def _set_requires_grad(self, tags: Set[str], requires_grad: bool):
        params = self._get_tagged_parameters(tags)
        if not params and requires_grad: warnings.warn(f"Attempted set requires_grad={requires_grad} tags {tags}, but no parameters found.", RuntimeWarning)
        for param in params: param.requires_grad = requires_grad
    def _update_target_network(self):
        with torch.no_grad():
            try:
                params = list(self.unified_eq.parameters()); target_params = list(self.target_unified_eq.parameters())
                if len(params) != len(target_params): warnings.warn(f"Target param count mismatch ({len(target_params)} vs {len(params)}). Syncing instead.", RuntimeWarning); self._sync_target_network()
                else:
                    for target_param, main_param in zip(target_params, params): target_param.data.mul_(1.0 - self.tau); target_param.data.add_(self.tau * main_param.data)
            except Exception as e: warnings.warn(f"Error during soft target update: {e}. Syncing.", RuntimeWarning); self._sync_target_network()
    def _sync_target_network(self):
        try:
            if self.target_unified_eq is None: warnings.warn("Target network is None, cannot sync.", RuntimeWarning); return
            self.target_unified_eq.load_state_dict(self.unified_eq.state_dict());
            for param in self.target_unified_eq.parameters(): param.requires_grad = False
        except RuntimeError as e: warnings.warn(f"Failed sync target: {e}. Target may be out of sync.", RuntimeWarning)
        except AttributeError: warnings.warn("Target network might not be initialized yet for sync.", RuntimeWarning)

    def get_action(self, state, explore=True) -> np.ndarray:
        try: state_np = np.asarray(state, dtype=np.float32);
        if np.isnan(state_np).any() or np.isinf(state_np).any(): state_np = np.zeros_like(state_np, dtype=np.float32);
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        except Exception as e: return np.zeros(self.action_dim, dtype=np.float32)
        self.unified_eq.eval(); final_action = np.zeros(self.action_dim, dtype=np.float32)
        try:
            with torch.no_grad():
                context = {'task': 'rl', 'request_action': True, 'mode': 'eval' if not explore else 'explore'}
                initial_data = {'state': state_tensor, 'features': state_tensor}
                output_data = self.unified_eq(initial_data, context)
                if 'action_raw' not in output_data: warnings.warn(f"Key 'action_raw' missing get_action. Output keys: {list(output_data.keys())}. Zero.", RuntimeWarning)
                else:
                    action_tanh = output_data['action_raw']
                    if explore: noise = torch.randn_like(action_tanh) * self.current_noise_std; action_noisy_tanh = torch.clamp(action_tanh + noise, -1.0, 1.0)
                    else: action_noisy_tanh = action_tanh
                    action_scaled = action_noisy_tanh * self.action_scale + self.action_bias
                    action_clipped = torch.clamp(action_scaled, self.action_low, self.action_high)
                    if torch.isnan(action_clipped).any() or torch.isinf(action_clipped).any(): action_clipped = torch.full_like(action_clipped, (self.action_high + self.action_low) / 2.0)
                    final_action = action_clipped.squeeze(0).cpu().numpy()
        except Exception as e: warnings.warn(f"Exception get_action: {e}. Zero.", RuntimeWarning); traceback.print_exc()
        finally: self.unified_eq.train()
        return final_action.reshape(self.action_dim)

    def update(self, step):
        if len(self.replay_buffer) < self.batch_size: return None, None, 0.0
        self.total_updates += 1; sample = self.replay_buffer.sample(self.batch_size)
        if sample is None: return None, None, 0.0
        try: states, actions, rewards, next_states, dones = [t.to(self.device) for t in sample]
        except Exception as e: warnings.warn(f"Failed move batch device: {e}.", RuntimeWarning); return None, None, 0.0
        avg_reward_in_batch = rewards.mean().item();
        if math.isnan(avg_reward_in_batch) or math.isinf(avg_reward_in_batch): avg_reward_in_batch = 0.0
        q_loss_val, policy_loss_val = None, None; q_target = None
        try:
            with torch.no_grad():
                if self.target_unified_eq is None: raise RuntimeError("Target network not initialized in update step.")
                self.target_unified_eq.eval()
                tgt_act_ctx = {'task': 'rl', 'request_action': True, 'mode': 'target'}
                tgt_act_data = {'state': next_states, 'features': next_states}
                tgt_policy_out = self.target_unified_eq(tgt_act_data, tgt_act_ctx)
                if 'action_raw' not in tgt_policy_out: raise ValueError(f"Tgt eq missing 'action_raw'. Output keys: {list(tgt_policy_out.keys())}")
                next_actions_tanh = tgt_policy_out['action_raw']
                next_actions_scaled = torch.clamp(next_actions_tanh * self.action_scale + self.action_bias, self.action_low, self.action_high)
                if torch.isnan(next_actions_scaled).any(): next_actions_scaled = torch.zeros_like(next_actions_scaled)
                tgt_q_ctx = {'task': 'rl', 'request_q_value': True, 'mode': 'target'}
                tgt_critic_in = torch.cat([next_states, next_actions_scaled], dim=1)
                tgt_q_data = {'state': next_states, 'action': next_actions_scaled, 'critic_input': tgt_critic_in, 'features': next_states}
                tgt_critic_out = self.target_unified_eq(tgt_q_data, tgt_q_ctx)
                if 'q_value' not in tgt_critic_out: raise ValueError(f"Tgt eq missing 'q_value'. Output keys: {list(tgt_critic_out.keys())}")
                target_q_values = tgt_critic_out['q_value']
                if torch.isnan(target_q_values).any(): target_q_values = torch.zeros_like(target_q_values)
                q_target = rewards + self.gamma * target_q_values * (1.0 - dones)
                if torch.isnan(q_target).any(): raise ValueError("NaN/Inf q_target")
                self.target_unified_eq.train()
        except Exception as e: warnings.warn(f"Exception target Q step {step}: {e}", RuntimeWarning); traceback.print_exc(); return q_loss_val, policy_loss_val, avg_reward_in_batch
        if self.critic_optimizer and q_target is not None:
            try:
                self.unified_eq.train(); self._set_requires_grad({'actor'}, False); self._set_requires_grad({'critic', 'shared'}, True)
                curr_q_ctx = {'task': 'rl', 'request_q_value': True, 'mode': 'train_critic'}
                curr_critic_in = torch.cat([states, actions], dim=1)
                curr_q_data = {'state': states, 'action': actions, 'critic_input': curr_critic_in, 'features': states}
                critic_output = self.unified_eq(curr_q_data, curr_q_ctx)
                if 'q_value' not in critic_output: raise ValueError(f"Main eq missing 'q_value' critic loss. Output keys: {list(critic_output.keys())}")
                current_q_values = critic_output['q_value']
                if torch.isnan(current_q_values).any(): raise ValueError("NaN/Inf current_q_values")
                q_loss = F.mse_loss(current_q_values, q_target.detach())
                if torch.isnan(q_loss): raise ValueError(f"NaN Q-loss: {q_loss.item()}")
                self.critic_optimizer.zero_grad(); q_loss.backward()
                critic_params = self._get_tagged_parameters({'critic', 'shared'})
                if critic_params:
                    if self.grad_clip_norm: torch.nn.utils.clip_grad_norm_(critic_params, self.grad_clip_norm)
                    if self.grad_clip_value: torch.nn.utils.clip_grad_value_(critic_params, self.grad_clip_value)
                self.critic_optimizer.step(); q_loss_val = q_loss.item()
            except Exception as e: warnings.warn(f"Exception Critic update step {step}: {e}", RuntimeWarning); traceback.print_exc()
            finally: self._set_requires_grad({'actor'}, True)
        if self.actor_optimizer:
            try:
                self.unified_eq.train(); self._set_requires_grad({'critic'}, False); self._set_requires_grad({'actor', 'shared'}, True)
                policy_act_ctx = {'task': 'rl', 'request_action': True, 'mode': 'train_actor'}
                policy_act_data = {'state': states, 'features': states}
                policy_output = self.unified_eq(policy_act_data, policy_act_ctx)
                if 'action_raw' not in policy_output: raise ValueError(f"Main eq missing 'action_raw' policy loss. Output keys: {list(policy_output.keys())}")
                actions_pred_tanh = policy_output['action_raw']; actions_pred_scaled = actions_pred_tanh * self.action_scale + self.action_bias
                policy_q_ctx = {'task': 'rl', 'request_q_value': True, 'mode': 'eval_policy'}
                policy_critic_in = torch.cat([states, actions_pred_scaled], dim=1)
                policy_q_data = {'state': states, 'action': actions_pred_scaled, 'critic_input': policy_critic_in, 'features': states}
                policy_critic_output = self.unified_eq(policy_q_data, policy_q_ctx)
                if 'q_value' not in policy_critic_output: raise ValueError(f"Main eq missing 'q_value' policy loss. Output keys: {list(policy_critic_output.keys())}")
                policy_q_values = policy_critic_output['q_value']
                policy_loss = -policy_q_values.mean()
                if torch.isnan(policy_loss): raise ValueError(f"NaN Policy loss: {policy_loss.item()}")
                self.actor_optimizer.zero_grad(); policy_loss.backward()
                actor_params = self._get_tagged_parameters({'actor', 'shared'})
                if actor_params:
                    if self.grad_clip_norm: torch.nn.utils.clip_grad_norm_(actor_params, self.grad_clip_norm)
                    if self.grad_clip_value: torch.nn.utils.clip_grad_value_(actor_params, self.grad_clip_value)
                self.actor_optimizer.step(); policy_loss_val = policy_loss.item()
            except Exception as e: warnings.warn(f"Exception Actor update step {step}: {e}", RuntimeWarning); traceback.print_exc()
            finally: self._set_requires_grad({'critic'}, True)
        try: self._update_target_network()
        except Exception as e: warnings.warn(f"Exception target update step {step}: {e}", RuntimeWarning)
        try:
            changed_components_info = self.unified_eq.evolve(self.evolution_engine, avg_reward_in_batch, self.total_updates)
            if changed_components_info:
                 self.total_evolutions += 1; print(f"    Unified eq structure changed. Reinit optimizers & sync target.")
                 actor_needs_reset = False; critic_needs_reset = False
                 for _, tags in changed_components_info:
                     if 'shared' in tags or 'actor' in tags: actor_needs_reset = True
                     if 'shared' in tags or 'critic' in tags: critic_needs_reset = True
                     if actor_needs_reset and critic_needs_reset: break
                 wd = self.weight_decay; actor_optim_state = self.actor_optimizer.state_dict() if self.actor_optimizer else None; critic_optim_state = self.critic_optimizer.state_dict() if self.critic_optimizer else None
                 if actor_needs_reset:
                      self.actor_optimizer = self._create_optimizer(self.actor_lr, wd, {'actor', 'shared'})
                      if self.actor_optimizer and actor_optim_state:
                          try: self.actor_optimizer.load_state_dict(actor_optim_state); print("      Actor Optimizer State Restored (partially maybe).")
                          except: print("      Actor Optimizer Reinitialized (state mismatch).")
                 if critic_needs_reset:
                      self.critic_optimizer = self._create_optimizer(self.critic_lr, wd, {'critic', 'shared'})
                      if self.critic_optimizer and critic_optim_state:
                          try: self.critic_optimizer.load_state_dict(critic_optim_state); print("      Critic Optimizer State Restored (partially maybe).")
                          except: print("      Critic Optimizer Reinitialized (state mismatch).")
                 self._sync_target_network()
        except Exception as e: warnings.warn(f"Exception Unified Eq evolution step {self.total_updates}: {e}", RuntimeWarning)
        self.current_noise_std = max(self.min_exploration_noise, self.current_noise_std * self.noise_decay_rate)
        return q_loss_val, policy_loss_val, avg_reward_in_batch

    def evaluate(self, env, episodes=5):
        total_rewards = []; max_steps = getattr(env.spec, 'max_episode_steps', 1000)
        for i in range(episodes):
            ep_reward = 0.0; steps = 0; state, info = env.reset(seed=RANDOM_SEED + 1000 + i + self.total_updates)
            state = np.asarray(state, dtype=np.float32); terminated, truncated = False, False
            while not (terminated or truncated):
                if steps >= max_steps: truncated = True; break
                try:
                    action = self.get_action(state, explore=False)
                    if action is None or np.isnan(action).any() or np.isinf(action).any(): action = np.zeros(self.action_dim, dtype=np.float32)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    if math.isnan(reward) or math.isinf(reward): reward = 0.0
                    ep_reward += reward; state = np.asarray(next_state, dtype=np.float32); steps += 1
                except (gym.error.Error, Exception) as e: warnings.warn(f"Exception eval step ep {i+1}: {e}. End.", RuntimeWarning); terminated = True
            total_rewards.append(ep_reward)
        return np.mean(total_rewards) if total_rewards else -np.inf

    def save_model(self, filename="unified_agent_checkpoint_v1514.pt"): # تحديث الإصدار
        self.unified_eq.to('cpu')
        try:
            component_configs = {}
            for comp_id, comp in self.unified_eq.components.items():
                 init_args_comp = getattr(comp, '_init_args', {}).copy()
                 if isinstance(comp, NumericTransformComponent) and hasattr(comp, 'equation') and comp.equation: init_args_comp['equation_state_dict'] = comp.equation.state_dict()
                 else: init_args_comp['equation_state_dict'] = None
                 component_configs[comp_id] = {'class_name': type(comp).__name__, 'init_args': init_args_comp, 'state_dict': comp.state_dict()}
            equation_structure = {'equation_id': self.unified_eq.equation_id, 'execution_order': self._execution_order, 'component_configs': component_configs}
            save_data = {'metadata': {'description': 'Unified Learning Agent (v1.5.14)', 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}, # تحديث الإصدار
                         'agent_config': {'state_dim': self.state_dim, 'action_dim': self.action_dim, 'action_bounds': (self.action_low, self.action_high),'gamma': self.gamma, 'tau': self.tau, 'batch_size': self.batch_size,'actor_lr': self.actor_lr, 'critic_lr': self.critic_lr, 'weight_decay': self.weight_decay},
                         'unified_equation_structure': equation_structure, 'actor_optimizer_state_dict': self.actor_optimizer.state_dict() if self.actor_optimizer else None,
                         'critic_optimizer_state_dict': self.critic_optimizer.state_dict() if self.critic_optimizer else None,
                         'training_state': {'total_updates': self.total_updates, 'current_noise_std': self.current_noise_std, 'total_evolutions': self.total_evolutions},}
            os.makedirs(os.path.dirname(filename), exist_ok=True); torch.save(save_data, filename); print(f"INFO: Unified Agent state saved to '{filename}'")
        except Exception as e: warnings.warn(f"Failed save Unified agent state: {e}", RuntimeWarning); traceback.print_exc()
        finally: self.unified_eq.to(self.device)

    def load_model(self, filename="unified_agent_checkpoint_v1514.pt"): # تحديث الإصدار
        try:
            print(f"INFO: Attempting load Unified agent state from '{filename}'...")
            if not os.path.exists(filename): warnings.warn(f"Checkpoint not found: '{filename}'. Load failed.", RuntimeWarning); return False
            checkpoint = torch.load(filename, map_location=self.device)
            required = ['agent_config', 'unified_equation_structure', 'actor_optimizer_state_dict', 'critic_optimizer_state_dict', 'training_state']
            if not all(k in checkpoint for k in required): missing = [k for k in required if k not in checkpoint]; warnings.warn(f"Checkpoint incomplete. Missing: {missing}. Load failed.", RuntimeWarning); return False
            cfg = checkpoint['agent_config']; train_state = checkpoint['training_state']; eq_struct = checkpoint['unified_equation_structure']
            if cfg.get('state_dim')!=self.state_dim or cfg.get('action_dim')!=self.action_dim: warnings.warn("Dim mismatch. Load aborted.", RuntimeWarning); return False
            print("  Rebuilding Unified Equation...")
            new_unified_eq = UnifiedAdaptableEquation(eq_struct.get('equation_id', 'loaded_eq'))
            activation_map = {'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'None': None, 'identity': lambda x: x, 'gelu': F.gelu, '<lambda>': None}
            component_classes = {'NumericTransformComponent': NumericTransformComponent, 'PolicyDecisionComponent': PolicyDecisionComponent, 'ValueEstimationComponent': ValueEstimationComponent}
            rebuilt_components = {}; component_state_dicts = {}
            for comp_id, comp_config in eq_struct.get('component_configs', {}).items():
                comp_class_name = comp_config.get('class_name')
                if comp_class_name in component_classes:
                    CompClass = component_classes[comp_class_name]; init_args = comp_config.get('init_args', {}).copy(); comp_state_dict = comp_config.get('state_dict')
                    if comp_class_name == 'NumericTransformComponent':
                         eq_config = init_args.pop('equation_config', None); eq_state_dict = init_args.pop('equation_state_dict', None); activation_name = init_args.pop('activation_name', 'None'); init_args['activation'] = activation_map.get(activation_name)
                         if eq_config and eq_state_dict is not None:
                              try: evol_eq = EvolvingEquation(**eq_config); evol_eq.load_state_dict(eq_state_dict); init_args['equation'] = evol_eq
                              except Exception as eq_rebuild_e: warnings.warn(f"Failed rebuild EvolvingEq '{comp_id}': {eq_rebuild_e}. Skip comp.", RuntimeWarning); continue
                         elif eq_config:
                             try: init_args['equation'] = EvolvingEquation(**eq_config); warnings.warn(f"Rebuilt '{comp_id}' without state.", RuntimeWarning)
                             except Exception as eq_rebuild_e: warnings.warn(f"Failed rebuild EvolvingEq (no state) '{comp_id}': {eq_rebuild_e}. Skip comp.", RuntimeWarning); continue
                         else: warnings.warn(f"Eq config or state missing '{comp_id}'. Skip comp.", RuntimeWarning); continue
                    if 'tags' in init_args and isinstance(init_args['tags'], list): init_args['tags'] = set(init_args['tags'])
                    if 'required_context' in init_args and isinstance(init_args['required_context'], list): init_args['required_context'] = set(init_args['required_context'])
                    if 'provides' in init_args and isinstance(init_args['provides'], list): init_args['provides'] = set(init_args['provides'])
                    try:
                        expected_args = list(inspect.signature(CompClass.__init__).parameters.keys());
                        if 'self' in expected_args: expected_args.remove('self')
                        filtered_args = {k: v for k, v in init_args.items() if k in expected_args}
                        component = CompClass(**filtered_args); rebuilt_components[comp_id] = component; component_state_dicts[comp_id] = comp_state_dict
                    except Exception as build_e: warnings.warn(f"Failed init comp '{comp_id}' class '{comp_class_name}': {build_e}", RuntimeWarning)
            saved_exec_order = eq_struct.get('execution_order', [])
            new_unified_eq._execution_order = saved_exec_order
            for comp_id in saved_exec_order:
                if comp_id in rebuilt_components: new_unified_eq.components[comp_id] = rebuilt_components[comp_id]
                else: warnings.warn(f"Comp '{comp_id}' from saved exec order missing.", RuntimeWarning)
            for comp_id, comp in rebuilt_components.items():
                if comp_id not in new_unified_eq.components: warnings.warn(f"Comp '{comp_id}' rebuilt but not in saved exec order. Appending.", RuntimeWarning); new_unified_eq.components[comp_id] = comp; new_unified_eq._execution_order.append(comp_id)
            for comp_id, state_dict in component_state_dicts.items():
                if comp_id in new_unified_eq.components and state_dict:
                    try: new_unified_eq.components[comp_id].load_state_dict(state_dict, strict=False)
                    except Exception as load_comp_e: warnings.warn(f"Failed load state comp '{comp_id}': {load_comp_e}", RuntimeWarning)
            self.unified_eq = new_unified_eq.to(self.device); self._execution_order = self.unified_eq._execution_order[:]
            actor_lr_load = cfg.get('actor_lr', self.actor_lr); critic_lr_load = cfg.get('critic_lr', self.critic_lr); wd_load = cfg.get('weight_decay', self.weight_decay)
            actor_optim_state = checkpoint.get('actor_optimizer_state_dict'); critic_optim_state = checkpoint.get('critic_optimizer_state_dict')
            self.actor_optimizer = self._create_optimizer(actor_lr_load, wd_load, {'actor', 'shared'})
            if self.actor_optimizer and actor_optim_state:
                try: self.actor_optimizer.load_state_dict(actor_optim_state)
                except Exception as ao_load_e: warnings.warn(f"Failed load actor optim state: {ao_load_e}. Reset.", RuntimeWarning); self.actor_optimizer = self._create_optimizer(actor_lr_load, wd_load, {'actor', 'shared'})
            self.critic_optimizer = self._create_optimizer(critic_lr_load, wd_load, {'critic', 'shared'})
            if self.critic_optimizer and critic_optim_state:
                 try: self.critic_optimizer.load_state_dict(critic_optim_state)
                 except Exception as co_load_e: warnings.warn(f"Failed load critic optim state: {co_load_e}. Reset.", RuntimeWarning); self.critic_optimizer = self._create_optimizer(critic_lr_load, wd_load, {'critic', 'shared'})
            self.target_unified_eq = copy.deepcopy(self.unified_eq);
            for param in self.target_unified_eq.parameters(): param.requires_grad = False
            self.total_updates = train_state.get('total_updates', 0); self.current_noise_std = train_state.get('current_noise_std', self.exploration_noise_std); self.total_evolutions = train_state.get('total_evolutions', 0)
            print(f"INFO: Unified Agent state loaded successfully from '{filename}'."); print(f"  Updates: {self.total_updates}, Evols: {self.total_evolutions}"); print(f"  Rebuilt Eq:\n{self.unified_eq.to_string()}"); return True
        except Exception as e: warnings.warn(f"Error loading Unified agent state: {e}", RuntimeWarning); traceback.print_exc(); return False


# --- 9. Main Training Function ---
def train_unified_agent(env_name="Pendulum-v1", max_steps=100000, batch_size=128,
                        eval_frequency=5000, start_learning_steps=5000, update_frequency=1,
                        hidden_dims=[64, 64], eq_init_complexity=4, eq_complexity_limit=10,
                        actor_lr=1e-4, critic_lr=3e-4, tau=0.005, weight_decay=1e-5,
                        evolution_mutation_power=0.02, evolution_cooldown=60,
                        exploration_noise=0.3, noise_decay=0.9999, min_noise=0.1,
                        save_best=True, save_periodically=True, save_interval=25000,
                        render_eval=False, eval_episodes=10, grad_clip_norm=1.0, grad_clip_value=None,
                        resume_from_checkpoint=None, results_dir="unified_agent_results_v1514"): # تحديث اسم المجلد
    """
    الدالة الرئيسية لتدريب وكيل التعلم الموحد (RL-E⁴).
    """
    start_time = time.time()
    print("\n" + "="*60); print(f"=== Starting Unified Agent Training (v1.5.14) for {env_name} ==="); print(f"=== Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ==="); print("="*60) # تحديث الإصدار
    print(f"\n--- Hyperparameters ---"); print(f"  Env: {env_name}, Max Steps: {max_steps:,}"); print(f"  Unified Eq: Hidden={hidden_dims}, InitComp={eq_init_complexity}, Limit={eq_complexity_limit}"); print(f"  Batch: {batch_size}, StartLearn: {start_learning_steps:,}, UpdateFreq: {update_frequency}"); print(f"  Eval: Freq={eval_frequency:,}, Eps={eval_episodes}"); print(f"  Learning: ActorLR={actor_lr:.1e}, CriticLR={critic_lr:.1e}, Tau={tau:.3f}, WD={weight_decay:.1e}, ClipNorm={grad_clip_norm}, ClipVal={grad_clip_value}"); print(f"  Evolution: Mut={evolution_mutation_power:.3f}, CD={evolution_cooldown}"); print(f"  Exploration: Start={exploration_noise:.2f}, Decay={noise_decay:.4f}, Min={min_noise:.2f}"); print(f"  Saving: Best={save_best}, Periodic={save_periodically} (Interval={save_interval:,})"); print(f"  Resume: {resume_from_checkpoint if resume_from_checkpoint else 'None'}"); print(f"  Results Dir: {results_dir}"); print("-" * 50 + "\n")
    try:
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_name), deque_size=50)
        eval_render_mode = "human" if render_eval else None; eval_env = None
        try: eval_env = gym.make(env_name, render_mode=eval_render_mode)
        except Exception:
            try: eval_env = gym.make(env_name)
            except Exception as e_fallback: print(f"CRITICAL ERROR: Create eval env failed: {e_fallback}"); return None, []
        env.reset(seed=RANDOM_SEED); env.action_space.seed(RANDOM_SEED); eval_env.reset(seed=RANDOM_SEED+1); eval_env.action_space.seed(RANDOM_SEED+1)
        state_dim = env.observation_space.shape[0]; action_dim = env.action_space.shape[0]
        action_low = env.action_space.low; action_high = env.action_space.high
        if np.any(np.isinf(action_low)) or np.any(np.isinf(action_high)): action_bounds = (-1.0, 1.0)
        else: action_bounds = (float(action_low.min()), float(action_high.max()))
        print(f"Env Details: State={state_dim}, Action={action_dim}, Bounds={action_bounds}")
    except Exception as e: print(f"CRITICAL ERROR initializing env: {e}"); return None, []

    try:
        print("Building example RL Unified Equation structure...")
        unified_eq = UnifiedAdaptableEquation(equation_id="rl_policy_value_net_v1514") # تحديث الإصدار
        current_dim = state_dim; last_comp_id = None; features_key = 'state'; shared_last_comp_id = last_comp_id
        for i, h_dim in enumerate(hidden_dims):
             comp_id = f'shared_hidden_{i}'; eq = EvolvingEquation(current_dim, eq_init_complexity, h_dim, eq_complexity_limit)
             comp = NumericTransformComponent(comp_id, eq, activation=F.relu, tags={'shared'}, input_key=features_key, output_key='features', required_context={'task'})
             unified_eq.add_component(comp, execute_after=shared_last_comp_id); current_dim = h_dim; shared_last_comp_id = comp_id; features_key = 'features'
        print(f"  Shared layers added. Last shared output dim: {current_dim}, key: '{features_key}'")
        actor_output_eq = EvolvingEquation(current_dim, eq_init_complexity, action_dim, eq_complexity_limit)
        actor_output_comp = NumericTransformComponent('actor_output_layer', actor_output_eq, activation=None, tags={'actor'}, input_key='features', output_key='policy_features', required_context={'request_action'})
        unified_eq.add_component(actor_output_comp, execute_after=shared_last_comp_id)
        policy_decision_comp = PolicyDecisionComponent('policy_decision', action_dim, tags={'actor'}, input_key='policy_features', output_key='action_raw', required_context={'request_action'})
        unified_eq.add_component(policy_decision_comp, execute_after='actor_output_layer')
        print(f"  Actor path added.")
        critic_input_dim = current_dim + action_dim
        critic_hidden_layer_eq = EvolvingEquation(critic_input_dim, eq_init_complexity, hidden_dims[-1], eq_complexity_limit)
        critic_hidden_comp = NumericTransformComponent('critic_hidden_layer', critic_hidden_layer_eq, activation=F.relu, tags={'critic'}, input_key='critic_input', output_key='value_features', required_context={'request_q_value'})
        unified_eq.add_component(critic_hidden_comp)
        critic_output_eq = EvolvingEquation(hidden_dims[-1], eq_init_complexity, 1, eq_complexity_limit)
        critic_output_comp = NumericTransformComponent('critic_output_layer', critic_output_eq, activation=None, tags={'critic'}, input_key='value_features', output_key='value_features_final', required_context={'request_q_value'})
        unified_eq.add_component(critic_output_comp, execute_after='critic_hidden_layer')
        value_estimation_comp = ValueEstimationComponent('value_estimation', tags={'critic'}, input_key='value_features_final', output_key='q_value', required_context={'request_q_value'})
        unified_eq.add_component(value_estimation_comp, execute_after='critic_output_layer')
        print(f"  Critic path added.")
        print("Unified Equation structure built.")
    except Exception as e: print(f"CRITICAL ERROR building unified eq: {e}"); traceback.print_exc(); return None, []

    try:
        evo_engine = EvolutionEngine(mutation_power=evolution_mutation_power, cooldown_period=evolution_cooldown)
        agent = UnifiedLearningAgent(state_dim, action_dim, action_bounds, unified_equation=unified_eq, evolution_engine=evo_engine, gamma=0.99, tau=tau, actor_lr=actor_lr, critic_lr=critic_lr, buffer_capacity=int(1e6), batch_size=batch_size, exploration_noise_std=exploration_noise, noise_decay_rate=noise_decay, min_exploration_noise=min_noise, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm, grad_clip_value=grad_clip_value)
    except Exception as e: print(f"CRITICAL ERROR initializing agent: {e}"); return None, []

    evaluation_rewards = []; steps_history = []
    loss_buffer_size = max(500, eval_frequency * 2 // update_frequency); all_q_losses=deque(maxlen=loss_buffer_size); all_policy_losses=deque(maxlen=loss_buffer_size)
    best_eval_metric_value = -np.inf; start_step = 0; total_episodes = 0
    try: os.makedirs(results_dir, exist_ok=True)
    except OSError as e: save_best=False; save_periodically=False

    if resume_from_checkpoint:
        print(f"\n--- Resuming Training from: {resume_from_checkpoint} ---")
        if agent.load_model(resume_from_checkpoint): start_step = agent.total_updates; print(f"Resumed approx env step {start_step} (from {agent.total_updates} updates)")
        else: warnings.warn("Failed load checkpoint. Start scratch."); start_step = 0; agent.total_updates = 0

    try: state, info = env.reset(seed=RANDOM_SEED + start_step); state = np.asarray(state, dtype=np.float32)
    except Exception as e: print(f"CRITICAL ERROR resetting env: {e}"); env.close(); eval_env.close(); return None, []

    print("\n--- Starting Training Loop (Unified Agent v1.5.14) ---") # تحديث الإصدار
    progress_bar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="Training", unit="step", ncols=120)
    episode_reward = 0.0; episode_steps = 0

    for current_env_step in progress_bar:
        if current_env_step < start_learning_steps: action = env.action_space.sample()
        else: action = agent.get_action(state, explore=True)
        try:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if math.isnan(reward) or math.isinf(reward): reward = 0.0
            next_state = np.asarray(next_state, dtype=np.float32)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state; episode_reward += reward; episode_steps += 1
            if done:
                total_episodes += 1
                if 'episode' in info:
                    avg_rew = np.mean(env.return_queue) if env.return_queue else 0.0; avg_len = np.mean(env.length_queue) if env.length_queue else 0.0
                    eq_comp = agent.unified_eq.get_total_complexity()
                    postfix = {"Ep": total_episodes, "Rew(r)": f"{avg_rew:.1f}", "Len(r)": f"{avg_len:.0f}", "Noise": f"{agent.current_noise_std:.3f}", "Eq_C": f"{eq_comp:.1f}"}
                    if all_q_losses: postfix["QL(r)"] = f"{np.mean(all_q_losses):.2f}"
                    if all_policy_losses: postfix["PL(r)"] = f"{np.mean(all_policy_losses):.2f}"
                    progress_bar.set_postfix(postfix)
                state, info = env.reset(seed=RANDOM_SEED + current_env_step); state = np.asarray(state, dtype=np.float32)
                episode_reward = 0.0; episode_steps = 0
        except (gym.error.Error, Exception) as e:
            warnings.warn(f"\nException env step {current_env_step}: {e}", RuntimeWarning)
            try: state, info = env.reset(seed=RANDOM_SEED + current_env_step + 1); state = np.asarray(state, dtype=np.float32); print("Env reset OK.")
            except Exception as e2: print(f"CRITICAL ERROR: Failed reset: {e2}. Stop."); break
            continue
        if current_env_step >= start_learning_steps and current_env_step % update_frequency == 0:
            q_loss, policy_loss, batch_avg_reward = agent.update(step=agent.total_updates)
            if q_loss is not None: all_q_losses.append(q_loss)
            if policy_loss is not None: all_policy_losses.append(policy_loss)
        if current_env_step > 0 and current_env_step % eval_frequency == 0 and current_env_step >= start_learning_steps:
            progress_bar.write("\n" + "-"*40 + f" Evaluating at Env Step {current_env_step:,} (Agent Updates: {agent.total_updates:,}) " + "-"*40)
            eval_avg_reward = agent.evaluate(eval_env, episodes=eval_episodes)
            evaluation_rewards.append(eval_avg_reward); steps_history.append(current_env_step)
            progress_bar.write(f"  Avg Eval Reward ({eval_episodes} eps): {eval_avg_reward:.2f}"); progress_bar.write(f"  Evolutions: {agent.total_evolutions}")
            avg_q = f"{np.mean(all_q_losses):.4f}" if all_q_losses else "N/A"; avg_p = f"{np.mean(all_policy_losses):.4f}" if all_policy_losses else "N/A"
            progress_bar.write(f"  Avg Losses (Q): {avg_q} | (P): {avg_p}")
            try: progress_bar.write(f"--- Current Unified Equation ---"); progress_bar.write(agent.unified_eq.to_string())
            except Exception as repr_e: progress_bar.write(f"  Error repr eq: {repr_e}")
            progress_bar.write("-" * 100)
            if save_best:
                 if not math.isinf(eval_avg_reward) and not math.isnan(eval_avg_reward) and eval_avg_reward > best_eval_metric_value:
                     old_best = f"{best_eval_metric_value:.2f}" if not math.isinf(best_eval_metric_value) else "-inf"
                     progress_bar.write(f"  ** New best reward ({eval_avg_reward:.2f} > {old_best})! Saving... **")
                     best_eval_metric_value = eval_avg_reward; best_file = os.path.join(results_dir, f"unified_best_{env_name.replace('-','_')}.pt")
                     agent.save_model(filename=best_file)
        if save_periodically and current_env_step > 0 and current_env_step % save_interval == 0 and current_env_step > start_step:
             periodic_file = os.path.join(results_dir, f"unified_step_{current_env_step}_{env_name.replace('-','_')}.pt")
             progress_bar.write(f"\n--- Saving Periodic Checkpoint at Env Step {current_env_step:,} ---"); agent.save_model(filename=periodic_file)

    progress_bar.close();
    try: env.close()
    except Exception: pass
    try: eval_env.close()
    except Exception: pass
    final_file = os.path.join(results_dir, f"unified_final_step_{max_steps}_{env_name.replace('-','_')}.pt")
    print(f"\n--- Saving Final Model at Env Step {max_steps:,} ---"); agent.save_model(filename=final_file)
    end_time = time.time(); total_time = end_time - start_time
    print("\n" + "="*60); print(f"=== Training Finished (Unified Agent v1.5.14) ==="); print(f"=== Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ==="); print("="*60) # تحديث الإصدار
    print(f"Total Env Steps: {max_steps:,}"); print(f"Total Episodes: {total_episodes:,}"); print(f"Total Agent Updates: {agent.total_updates:,}"); print(f"Total Evolutions: {agent.total_evolutions}")
    print(f"Total Training Time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    if env.return_queue: print(f"Avg reward last {len(env.return_queue)} train eps: {np.mean(env.return_queue):.2f}")
    print(f"Final Equation Complexity: {agent.unified_eq.get_total_complexity():.2f}")
    best_str = f"{best_eval_metric_value:.2f}" if not math.isinf(best_eval_metric_value) else "N/A"; print(f"Best Eval Reward: {best_str}")

    if steps_history and evaluation_rewards:
        print("\n--- Generating Training Plots ---")
        try:
            plt.style.use('ggplot'); fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True); fig.suptitle(f'Unified Agent Training ({env_name} - v1.5.14)', fontsize=16) # تحديث الإصدار
            ax1 = axes[0]; ax1.plot(steps_history, evaluation_rewards, marker='.', linestyle='-', color='dodgerblue', label='Avg Eval Reward')
            if len(evaluation_rewards) >= 5: mv_avg = np.convolve(evaluation_rewards, np.ones(5)/5, mode='valid'); ax1.plot(steps_history[4:], mv_avg, linestyle='--', color='orangered', label='Moving Avg (5 evals)')
            ax1.set_ylabel('Avg Eval Reward'); ax1.set_title('Evaluation Reward'); ax1.legend(); ax1.grid(True, ls='--', lw=0.5)
            ax2 = axes[1];
            if all_q_losses: upd_est = np.linspace(start_learning_steps, current_env_step, len(all_q_losses), dtype=int); ax2.plot(upd_est, list(all_q_losses), label='Q-Loss (Raw)', alpha=0.4, c='darkorange', lw=0.8)
            if len(all_q_losses) >= 20: ql_ma = np.convolve(list(all_q_losses), np.ones(20)/20, mode='valid'); ax2.plot(upd_est[19:], ql_ma, label='Q-Loss (MA-20)', c='red', lw=1.2)
            ax2.set_ylabel('Q-Loss'); ax2.set_title('Critic Loss'); ax2.legend(); ax2.grid(True, ls='--', lw=0.5);
            if all_q_losses and all(ql > 0 for ql in all_q_losses):
                 try: ax2.set_yscale('log'); ax2.set_ylabel('Q-Loss (Log)')
                 except ValueError: pass
            ax3 = axes[2];
            if all_policy_losses: upd_est_p = np.linspace(start_learning_steps, current_env_step, len(all_policy_losses), dtype=int); ax3.plot(upd_est_p, list(all_policy_losses), label='P Loss (Raw)', alpha=0.4, c='forestgreen', lw=0.8)
            if len(all_policy_losses) >= 20: pl_ma = np.convolve(list(all_policy_losses), np.ones(20)/20, mode='valid'); ax3.plot(upd_est_p[19:], pl_ma, label='P Loss (MA-20)', c='darkgreen', lw=1.2)
            ax3.set_ylabel('Policy Loss (-Avg Q)'); ax3.set_title('Actor Loss'); ax3.legend(); ax3.grid(True, ls='--', lw=0.5)
            ax3.set_xlabel('Environment Steps'); fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            plot_file = os.path.join(results_dir, f"unified_plots_{env_name.replace('-','_')}.png")
            plt.savefig(plot_file, dpi=300); print(f"Plots saved: '{plot_file}'"); plt.close(fig)
        except Exception as e: warnings.warn(f"Could not generate plots: {e}", RuntimeWarning)
    else: print("No evaluation data, skipping plots.")

    return agent, evaluation_rewards


# --- 10. Execution Block ---
if __name__ == "__main__":
    print("\n" + "="*60); print("===     Unified Learning Agent Main Execution (v1.5.14)     ==="); print("="*60) # تحديث الإصدار
    ENVIRONMENT_NAME = "Pendulum-v1"
    MAX_TRAINING_STEPS = 100000
    EVALUATION_FREQUENCY = 5000
    START_LEARNING = 5000
    BATCH_SIZE = 128
    HIDDEN_DIMS = [64, 64]
    EQ_INIT_COMP = 3
    EQ_COMP_LIMIT = 8
    ACTOR_LR = 1e-4
    CRITIC_LR = 3e-4
    TAU = 0.005
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0
    GRAD_CLIP_VALUE = None
    EVO_MUT_POWER = 0.015
    EVO_COOLDOWN = 60
    EXPLORATION_NOISE = 0.3
    NOISE_DECAY = 0.9999
    MIN_NOISE = 0.1
    SAVE_BEST_MODEL = True
    SAVE_PERIODICALLY = True
    SAVE_INTERVAL = 25000
    RESULTS_DIRECTORY = f"unified_agent_results_{ENVIRONMENT_NAME.replace('-','_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    RESUME_CHECKPOINT = None

    trained_agent, eval_history = train_unified_agent(
        env_name=ENVIRONMENT_NAME, max_steps=MAX_TRAINING_STEPS, batch_size=BATCH_SIZE,
        eval_frequency=EVALUATION_FREQUENCY, start_learning_steps=START_LEARNING, update_frequency=1,
        hidden_dims=HIDDEN_DIMS, eq_init_complexity=EQ_INIT_COMP, eq_complexity_limit=EQ_COMP_LIMIT,
        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, tau=TAU, weight_decay=WEIGHT_DECAY,
        evolution_mutation_power=EVO_MUT_POWER, evolution_cooldown=EVO_COOLDOWN,
        exploration_noise=EXPLORATION_NOISE, noise_decay=NOISE_DECAY, min_noise=MIN_NOISE,
        save_best=SAVE_BEST_MODEL, save_periodically=SAVE_PERIODICALLY, save_interval=SAVE_INTERVAL,
        render_eval=False, eval_episodes=10, grad_clip_norm=GRAD_CLIP_NORM, grad_clip_value=GRAD_CLIP_VALUE,
        results_dir=RESULTS_DIRECTORY, resume_from_checkpoint=RESUME_CHECKPOINT
    )

    if trained_agent:
        print("\n" + "="*60); print("=== Final Evaluation of Trained Unified Agent ==="); print("="*60)
        final_eval_env = None
        try:
            print("Evaluating final agent (rendering if possible)...")
            final_eval_env = gym.make(ENVIRONMENT_NAME, render_mode="human")
            final_performance = trained_agent.evaluate(final_eval_env, episodes=5)
            print(f"Final Agent Avg Perf (5 eps, rendered): {final_performance:.2f}")
        except Exception as e:
            print(f"  * Render eval failed: {e}. Evaluating without render...")
            try:
                final_eval_env = gym.make(ENVIRONMENT_NAME)
                final_performance = trained_agent.evaluate(final_eval_env, episodes=10)
                print(f"  * Final Agent Avg Perf (10 eps, no render): {final_performance:.2f}")
            except Exception as e2: print(f"  * ERROR: Final evaluation failed: {e2}")
        finally:
             if final_eval_env:
                 try: final_eval_env.close()
                 except Exception: pass

        if SAVE_BEST_MODEL and os.path.exists(RESULTS_DIRECTORY):
             best_model_file = os.path.join(RESULTS_DIRECTORY, f"unified_best_{ENVIRONMENT_NAME.replace('-','_')}.pt")
             print("\n" + "="*60); print(f"=== Evaluating Best Saved Unified Agent ==="); print("="*60)
             if os.path.exists(best_model_file):
                 try:
                    print("Loading best model for evaluation...")
                    eval_evo_engine = EvolutionEngine(mutation_power=EVO_MUT_POWER, cooldown_period=EVO_COOLDOWN)
                    eval_eq_structure = UnifiedAdaptableEquation("dummy_for_best_eval_v1514") # تحديث الإصدار
                    eval_current_dim = trained_agent.state_dim
                    eval_last_comp_id = None; eval_features_key = 'state'; eval_shared_last_comp_id = None
                    for i, h_dim in enumerate(HIDDEN_DIMS):
                        comp_id = f'shared_hidden_{i}'; eq = EvolvingEquation(eval_current_dim, EQ_INIT_COMP, h_dim, EQ_COMP_LIMIT)
                        comp = NumericTransformComponent(comp_id, eq, activation=F.relu, tags={'shared'}, input_key=eval_features_key, output_key='features', required_context={'task'})
                        eval_eq_structure.add_component(comp, execute_after=eval_shared_last_comp_id); eval_current_dim = h_dim; eval_shared_last_comp_id = comp_id; eval_features_key = 'features'
                    actor_output_eq = EvolvingEquation(eval_current_dim, EQ_INIT_COMP, trained_agent.action_dim, EQ_COMP_LIMIT)
                    actor_output_comp = NumericTransformComponent('actor_output_layer', actor_output_eq, activation=None, tags={'actor'}, input_key='features', output_key='policy_features', required_context={'request_action'})
                    eval_eq_structure.add_component(actor_output_comp, execute_after=eval_shared_last_comp_id)
                    policy_decision_comp = PolicyDecisionComponent('policy_decision', trained_agent.action_dim, tags={'actor'}, input_key='policy_features', output_key='action_raw', required_context={'request_action'})
                    eval_eq_structure.add_component(policy_decision_comp, execute_after='actor_output_layer')
                    critic_input_dim = eval_current_dim + trained_agent.action_dim
                    critic_hidden_layer_eq = EvolvingEquation(critic_input_dim, EQ_INIT_COMP, HIDDEN_DIMS[-1], EQ_COMP_LIMIT)
                    critic_hidden_comp = NumericTransformComponent('critic_hidden_layer', critic_hidden_layer_eq, activation=F.relu, tags={'critic'}, input_key='critic_input', output_key='value_features', required_context={'request_q_value'})
                    eval_eq_structure.add_component(critic_hidden_comp)
                    critic_output_eq = EvolvingEquation(HIDDEN_DIMS[-1], EQ_INIT_COMP, 1, EQ_COMP_LIMIT)
                    critic_output_comp = NumericTransformComponent('critic_output_layer', critic_output_eq, activation=None, tags={'critic'}, input_key='value_features', output_key='value_features_final', required_context={'request_q_value'})
                    eval_eq_structure.add_component(critic_output_comp, execute_after='critic_hidden_layer')
                    value_estimation_comp = ValueEstimationComponent('value_estimation', tags={'critic'}, input_key='value_features_final', output_key='q_value', required_context={'request_q_value'})
                    eval_eq_structure.add_component(value_estimation_comp, execute_after='critic_output_layer')

                    eval_best_agent = UnifiedLearningAgent(
                        trained_agent.state_dim, trained_agent.action_dim, (trained_agent.action_low, trained_agent.action_high),
                        eval_eq_structure, evolution_engine=eval_evo_engine, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                        weight_decay=WEIGHT_DECAY, tau=TAU, grad_clip_norm=GRAD_CLIP_NORM, grad_clip_value=GRAD_CLIP_VALUE)

                    if eval_best_agent.load_model(best_model_file):
                        print("Best model loaded successfully. Evaluating...")
                        best_eval_env = None
                        try:
                             best_eval_env = gym.make(ENVIRONMENT_NAME)
                             best_performance = eval_best_agent.evaluate(best_eval_env, episodes=10)
                             print(f"Best Saved Agent Avg Perf (10 eps): {best_performance:.2f}")
                             print(f"  Equation Complexity: {eval_best_agent.unified_eq.get_total_complexity():.2f}")
                             print("\n--- Best Model Equation Representation ---")
                             print(eval_best_agent.unified_eq.to_string())
                        except Exception as eval_e: print(f"Error evaluating best model: {eval_e}")
                        finally:
                             if best_eval_env:
                                 try: best_eval_env.close()
                                 except Exception: pass
                    else: print("Skipping eval of best model (load failed).")
                 except Exception as e: print(f"ERROR during loading/evaluating best model: {e}"); traceback.print_exc()
             else: print(f"Best model file not found: '{best_model_file}'. Skipping.")
    else: print("\nTraining failed or agent init failed.")

    print("\n==================================================="); print("===      Unified Learning Agent Execution End       ==="); print("===================================================") # تحديث الإصدار