import os
import re
import io
import time
import math
import base64
import pickle
import warnings
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Suppress minor TensorFlow/Sklearn console warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid


class AirfoilOptimizerEngine:
    """
    Backend engine for inverse airfoil design:
    1. Compares all dataset airfoils AS THEY ARE to pick the single best baseline.
    2. Modifies that single baseline's 19 CST weights 10 times and delivers the best try.
    3. Exports a self-contained single-document report (Optimization_XXXX.html) and opens it.
    """

    def __init__(self,
                 model_path='data/airfoil_surrogate_model.keras',
                 dataset_path='data/complete_dataset_aerodinamico.csv',
                 dict_path='data/airfoil_tensor_dict.pkl'):
        print("Loading Core Systems (Surrogate Model, Dataset, and Scaler)...")
        self.model = load_model(model_path, compile=False)

        self.df = pd.read_csv(dataset_path)
        self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.df.iloc[:, 1:].values)

        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                self.airfoil_dict = pickle.load(f)
        else:
            self.airfoil_dict = None

        # Standard 25-point Angle of Attack grid (-4° to 20° in 1° steps)
        self.alpha_sweep = np.linspace(-4.0, 20.0, 25)

        # Build fast lookup table of existing dataset airfoils and their true CFD polars
        self._build_as_is_dataset_catalog()

    def _build_as_is_dataset_catalog(self):
        """
        Indexes all airfoils in complete_dataset_aerodinamico.csv AS THEY ARE,
        storing their true physical 19 CST weights and true 25-point Cl, Cd, Cm curves per Re.
        """
        self.dataset_catalog = {}
        id_col = self.df.columns[0]
        re_col = self.df.columns[1]
        aoa_col = self.df.columns[2]
        cst_cols = list(self.df.columns[3:22])
        cl_col, cd_col, cm_col = self.df.columns[22], self.df.columns[23], self.df.columns[24]

        for (airfoil_id, re_val), group in self.df.groupby([id_col, re_col]):
            group_sorted = group.sort_values(aoa_col)
            if len(group_sorted) == 25:
                cst_vec = group_sorted[cst_cols].iloc[0].values.astype(float)
                cl_vec = group_sorted[cl_col].values.astype(float)
                cd_vec = group_sorted[cd_col].values.astype(float)
                cm_vec = group_sorted[cm_col].values.astype(float)
                self.dataset_catalog[(str(airfoil_id), float(re_val))] = {
                    'id': str(airfoil_id),
                    're': float(re_val),
                    'cst': cst_vec,
                    'cl': cl_vec,
                    'cd': cd_vec,
                    'cm': cm_vec
                }

    # =========================================================================
    # 1. CST GEOMETRY RECONSTRUCTION & PHYSICAL CONSTRAINTS
    # =========================================================================
    @staticmethod
    def _bernstein_poly(n, i, x):
        k = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
        return k * (x ** i) * ((1.0 - x) ** (n - i))

    def calculate_cst_surface(self, x_array, weights, te_thickness, is_upper):
        """Reconstructs upper or lower airfoil surface using Class-Shape Transformation (CST)."""
        C = (x_array ** 0.5) * ((1.0 - x_array) ** 1.0)
        n = len(weights) - 1
        S = np.zeros_like(x_array)
        for i, w in enumerate(weights):
            S += w * self._bernstein_poly(n, i, x_array)

        y = C * S
        if is_upper:
            y += x_array * (te_thickness / 2.0)
        else:
            y -= x_array * (te_thickness / 2.0)
        return y

    def get_airfoil_coordinates(self, cst_weights, n_points_per_surface=200):
        """Generates 400 (x, y) coordinates for CAD import and plotting."""
        lower_w = cst_weights[0:9]
        upper_w = cst_weights[9:18]
        te_gap = cst_weights[18]

        x_up = np.linspace(1.0, 0.0, n_points_per_surface)
        y_up = self.calculate_cst_surface(x_up, upper_w, te_gap, is_upper=True)

        x_lo = np.linspace(0.0, 1.0, n_points_per_surface)[1:]
        y_lo = self.calculate_cst_surface(x_lo, lower_w, te_gap, is_upper=False)

        return np.concatenate((x_up, x_lo)), np.concatenate((y_up, y_lo))

    def constraint_no_intersection(self, cst_weights):
        """Full-chord scan (x=0.001 to 0.99) preventing upper/lower surface cross-over."""
        x_test = np.linspace(0.001, 0.99, 100)
        y_up = self.calculate_cst_surface(x_test, cst_weights[9:18], cst_weights[18], is_upper=True)
        y_lo = self.calculate_cst_surface(x_test, cst_weights[0:9], cst_weights[18], is_upper=False)
        return np.min(y_up - y_lo) - 0.001

    @staticmethod
    def constraint_rounded_leading_edge(cst_weights):
        """Forces upper nose weight > lower nose weight to guarantee a rounded leading edge."""
        return cst_weights[9] - cst_weights[0] - 0.005

    def constraint_min_thickness(self, cst_weights, min_tc):
        """Enforces minimum structural thickness-to-chord ratio (t/c)_max."""
        x_test = np.linspace(0.05, 0.95, 60)
        y_up = self.calculate_cst_surface(x_test, cst_weights[9:18], cst_weights[18], is_upper=True)
        y_lo = self.calculate_cst_surface(x_test, cst_weights[0:9], cst_weights[18], is_upper=False)
        return np.max(y_up - y_lo) - min_tc

    # =========================================================================
    # 2. SURROGATE BATCH EVALUATION (WITH DENORMALIZATION) & CURVE SCORING
    # =========================================================================
    def evaluate_surrogate_polar(self, cst_weights, reynolds):
        """
        Evaluates all 25 AoAs (-4° to 20°) in a single tensor pass and
        denormalizes [Cl, Cd, Cm] from [-1, 1] back to true physical units.
        """
        batch = np.zeros((25, 24), dtype=np.float32)
        batch[:, 0] = reynolds
        batch[:, 1] = self.alpha_sweep
        batch[:, 2:21] = cst_weights

        # Scale inputs to [-1, 1] using the fitted 24-column MinMaxScaler
        norm_input = (batch[:, 0:21] * self.scaler.scale_[0:21]) + self.scaler.min_[0:21]
        raw_preds = self.model(norm_input, training=False)

        if isinstance(raw_preds, list):
            norm_cl = raw_preds[0].numpy().flatten()
            norm_cd = raw_preds[1].numpy().flatten()
            norm_cm = raw_preds[2].numpy().flatten()
            norm_out = np.column_stack([norm_cl, norm_cd, norm_cm])
        else:
            norm_out = raw_preds.numpy()

        # Reverse [-1, 1] normalization to physical aerodynamic coefficients
        phys_out = (norm_out - self.scaler.min_[21:24]) / self.scaler.scale_[21:24]
        cl = phys_out[:, 0]
        cd = np.maximum(phys_out[:, 1], 1e-4)
        cm = phys_out[:, 2]
        return cl, cd, cm

    def extract_curve_metrics(self, cl, cd, cm):
        """Computes aerodynamic curve features with sub-degree parabolic peak interpolation."""
        dcl_dalpha = np.gradient(cl, self.alpha_sweep)
        d2cl_dalpha2 = np.gradient(dcl_dalpha, self.alpha_sweep)
        dcd_dalpha = np.gradient(cd, self.alpha_sweep)
        d2cd_dalpha2 = np.gradient(dcd_dalpha, self.alpha_sweep)
        dcm_dalpha = np.gradient(cm, self.alpha_sweep)

        cl0 = float(np.interp(0.0, self.alpha_sweep, cl))
        cm0 = float(np.interp(0.0, self.alpha_sweep, cm))
        cd0 = float(np.interp(0.0, self.alpha_sweep, cd))
        cl_alpha = float(np.mean(dcl_dalpha[2:11]))

        # 3-point parabolic vertex interpolation around discrete argmax(cl)
        max_idx = int(np.argmax(cl))
        if 0 < max_idx < len(self.alpha_sweep) - 1:
            y0, y1, y2 = cl[max_idx - 1], cl[max_idx], cl[max_idx + 1]
            denom = (y0 - 2.0 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            alpha_max = float(self.alpha_sweep[max_idx] + delta)
            cl_max = float(y1 - 0.25 * (y0 - y2) * delta)
        else:
            alpha_max = float(self.alpha_sweep[max_idx])
            cl_max = float(cl[max_idx])

        return {
            'cl0': cl0, 'cm0': cm0, 'cd0': cd0,
            'cl_alpha': cl_alpha, 'cl_max': cl_max, 'alpha_max': alpha_max,
            'max_abs_cm': float(np.max(np.abs(cm))),
            'dcl_dalpha': dcl_dalpha, 'd2cl_dalpha2': d2cl_dalpha2,
            'dcd_dalpha': dcd_dalpha, 'd2cd_dalpha2': d2cd_dalpha2,
            'dcm_dalpha': dcm_dalpha
        }


    def score_polar_curves(self, cl, cd, cm, req):
        """
        Upgraded objective scoring with sub-degree interpolation at cruise_aoa
        and high-priority target penalty weights so SLSQP locks onto user targets.
        """
        m = self.extract_curve_metrics(cl, cd, cm)
        penalty = 0.0

        if req['option'] == 'A':
            cl_cruise = float(np.interp(req['cruise_aoa'], self.alpha_sweep, cl))
            cd_cruise = float(np.interp(req['cruise_aoa'], self.alpha_sweep, cd))
            cl_err = (cl_cruise - req['target_cl']) ** 2
            primary_score = cd_cruise + (cl_err * 800.0)

            if req.get('flatten_bucket', False):
                c_idx = int(np.argmin(np.abs(self.alpha_sweep - req['cruise_aoa'])))
                hw = int(round(req.get('bucket_half_width', 2.0)))
                i_min, i_max = max(0, c_idx - hw), min(25, c_idx + hw + 1)
                slope_pen = np.mean(m['dcd_dalpha'][i_min:i_max] ** 2) * 15.0
                curv_pen = np.mean(np.abs(m['d2cd_dalpha2'][i_min:i_max])) * 5.0
                penalty += slope_pen + curv_pen

        elif req['option'] == 'B':
            cl_cruise = float(np.interp(req['cruise_aoa'], self.alpha_sweep, cl))
            cl_anchor_pen = ((cl_cruise - req['target_cl']) ** 2) * 800.0

            i_start = int(np.argmin(np.abs(self.alpha_sweep - req['alpha_min'])))
            i_end = int(np.argmin(np.abs(self.alpha_sweep - req['alpha_max'])))
            if i_end <= i_start:
                i_end = min(24, i_start + 1)

            ld_curve = cl / np.maximum(cd, 1e-4)
            mean_ld = np.trapz(ld_curve[i_start:i_end + 1], self.alpha_sweep[i_start:i_end + 1]) / (
                self.alpha_sweep[i_end] - self.alpha_sweep[i_start]
            )
            d_ld = np.gradient(ld_curve[i_start:i_end + 1], self.alpha_sweep[i_start:i_end + 1])
            smooth_ld_pen = np.mean(d_ld ** 2) * 0.02

            primary_score = (-mean_ld / 120.0) + cl_anchor_pen + smooth_ld_pen

        elif req['option'] == 'C':
            primary_score = np.mean(cd[2:15])

            if req.get('target_cl0') is not None:
                penalty += ((m['cl0'] - req['target_cl0']) ** 2) * 600.0
            if req.get('target_cl_alpha') is not None:
                penalty += ((m['cl_alpha'] - req['target_cl_alpha']) ** 2) * 4000.0
            if req.get('target_alpha_max') is not None:
                # Smooth differentiable soft-argmax combined with parabolic peak error
                cl_shifted = cl - np.max(cl)
                weights_soft = np.exp(cl_shifted * 25.0)
                soft_alpha_max = float(np.sum(self.alpha_sweep * weights_soft) / np.sum(weights_soft))
                blend_alpha_max = 0.5 * soft_alpha_max + 0.5 * m['alpha_max']
                penalty += (((blend_alpha_max - req['target_alpha_max']) / 10.0) ** 2) * 400.0

        # Secondary Modifiers
        if req.get('constrain_cm', False):
            if req.get('target_cm0') is not None:
                penalty += ((m['cm0'] - req['target_cm0']) ** 2) * 500.0
            if req.get('max_abs_cm') is not None:
                excess_cm = max(0.0, m['max_abs_cm'] - req['max_abs_cm'])
                penalty += (excess_cm ** 2) * 800.0
            penalty += np.mean(m['dcm_dalpha'][4:13] ** 2) * 10.0

        if req['option'] in ['A', 'C'] and req.get('stall_type') in ['open', 'closed']:
            max_idx = int(np.argmax(cl))
            cl_max_val = float(cl[max_idx])
            post_drop = float(cl_max_val - cl[-1])

            if max_idx >= len(self.alpha_sweep) - 2:
                # Differentiable penalty pushing the peak earlier and requiring a post-peak descent
                late_drop_deficit = max(0.0, 0.025 - float(np.max(cl[14:23]) - cl[-1]))
                penalty += 0.50 + (late_drop_deficit ** 2) * 100.0
            else:
                post_stall_dcl = m['dcl_dalpha'][max_idx + 1:]
                peak_d2cl = float(m['d2cl_dalpha2'][max_idx])

                if req['stall_type'] == 'open':
                    # 1. Require a genuine but gentle post-stall drop (0.020 <= Cl_max - Cl(20°) <= 0.180)
                    lack_of_drop = max(0.0, 0.020 - post_drop)
                    excessive_drop = max(0.0, post_drop - 0.180)
                    penalty += (lack_of_drop ** 2) * 80.0 + (excessive_drop ** 2) * 25.0

                    # 2. Post-stall slope window: gentle descent in [-0.030, -0.003] /deg (prevents flat table-tops)
                    too_steep = np.minimum(0.0, post_stall_dcl + 0.030)
                    too_flat = np.maximum(0.0, post_stall_dcl + 0.003)
                    penalty += float(np.sum(too_steep ** 2) * 35.0 + np.sum(too_flat ** 2) * 40.0)

                    # 3. Peak curvature dead-band: natural rounded arch has d2Cl/dAlpha2 in [-0.025, -0.004] /deg^2
                    too_sharp_curv = max(0.0, -0.025 - peak_d2cl)
                    too_flat_curv = max(0.0, peak_d2cl - (-0.004))
                    penalty += (too_sharp_curv ** 2) * 30.0 + (too_flat_curv ** 2) * 40.0

                    # 4. Prevent pre-peak table-top flattening before max_idx
                    if max_idx >= 16:
                        pre_peak_dcl = m['dcl_dalpha'][max_idx - 2:max_idx]
                        pre_flat = np.maximum(0.0, 0.004 - pre_peak_dcl)
                        penalty += float(np.sum(pre_flat ** 2) * 40.0)

                elif req['stall_type'] == 'closed':
                    # 1. Require a distinct post-stall drop (Cl_max - Cl(20°) >= 0.100)
                    lack_of_sharp_drop = max(0.0, 0.100 - post_drop)
                    penalty += (lack_of_sharp_drop ** 2) * 30.0

                    # 2. Require a sharp negative post-stall slope break (min dCl/dAlpha <= -0.035 /deg)
                    min_post_slope = float(np.min(post_stall_dcl))
                    gentle_slope_excess = max(0.0, min_post_slope - (-0.035))
                    penalty += (gentle_slope_excess ** 2) * 40.0

                    # 3. Require a sharp peak vertex curvature (d2Cl/dAlpha2 <= -0.020 /deg^2)
                    flat_peak_excess = max(0.0, peak_d2cl - (-0.020))
                    penalty += (flat_peak_excess ** 2) * 30.0

                    # 4. Mild tie-breaker reward for higher Cl_max (scaled so it never overrides cruise Cd)
                    penalty -= (m['cl_max'] * 0.0015)

        return float(primary_score + penalty)

    def objective_function(self, cst_weights, req):
        """SLSQP Objective Function: Predicts curves via NN and scores them."""
        cl, cd, cm = self.evaluate_surrogate_polar(cst_weights, req['Re'])
        return self.score_polar_curves(cl, cd, cm, req)

    # =========================================================================
    # 3. STAGE 1 (PICK 1 BEST AS-IS BASELINE) & STAGE 2 (ADAPTIVE WEIGHT MODIFICATIONS)
    # =========================================================================
    def find_best_dataset_baseline(self, req):
        """
        Stage 1: Scans all dataset airfoils AS-IS, and also checks top-5 candidates
        through the neural network at off-design fractional Reynolds numbers.
        """
        available_res = np.array([3000000.0, 6000000.0, 9000000.0])
        closest_re = float(available_res[np.argmin(np.abs(available_res - req['Re']))])
        min_tc = req.get('min_tc', 0.08)

        print(f"\n[Stage 1] Comparing dataset airfoils AS THEY ARE at Re = {closest_re/1e6:.1f}M...")

        candidates = []

        for (airfoil_id, re_val), entry in self.dataset_catalog.items():
            if not math.isclose(re_val, closest_re, rel_tol=0.05):
                continue
            cst = entry['cst']
            if (self.constraint_no_intersection(cst) >= 0 and
                self.constraint_rounded_leading_edge(cst) >= 0 and
                self.constraint_min_thickness(cst, min_tc) >= 0 and
                np.all(entry['cd'] > 0)):
                score = self.score_polar_curves(entry['cl'], entry['cd'], entry['cm'], req)
                candidates.append((score, entry))

        if not candidates:
            raise RuntimeError("No valid baseline airfoil found matching the minimum thickness constraint.")

        candidates.sort(key=lambda x: x[0])
        best_entry = candidates[0][1]
        best_score = candidates[0][0]
        best_nn_acc = -1.0

        for as_is_score, cand_entry in candidates[:5]:
            cst_c = cand_entry['cst'].copy()
            cst_c[18] = np.clip(cst_c[18], 0.001, 0.005)
            _, cand_acc, _, _ = self.calculate_accuracy_breakdown(cst_c, cst_c, req)
            if cand_acc > best_nn_acc:
                best_nn_acc = cand_acc
                best_entry = cand_entry
                best_score = as_is_score

        print(f"--- Best Dataset Airfoil Selected: {best_entry['id']} (As-Is Score: {best_score:.5f} | Starting NN Acc: {best_nn_acc:.2f}%) ---")
        return best_entry, best_score

    def run_optimization(self, req, max_tries=50, target_acc_threshold=95.0, patience=10):
        """
        Stage 2: Adaptive SLSQP multi-start loop:
          - Evaluates physical User-Needs Accuracy (%) after each try.
          - Once accuracy >= 95.0%, stops early if 10 consecutive tries do not improve.
          - Caps at 50 tries maximum.
        """
        t0 = time.perf_counter()

        # STAGE 1: Find the single best baseline airfoil in the dataset as-is
        best_base_entry, as_is_dataset_score = self.find_best_dataset_baseline(req)
        base_id = best_base_entry['id']
        base_cst = best_base_entry['cst'].copy()
        base_cst[18] = np.clip(base_cst[18], 0.001, 0.005)

        cl_b, cd_b, cm_b = self.evaluate_surrogate_polar(base_cst, req['Re'])
        initial_nn_score = self.score_polar_curves(cl_b, cd_b, cm_b, req)
        _, base_acc_pct, _, _ = self.calculate_accuracy_breakdown(base_cst, base_cst, req)

        # Define the +/- 25% bounding box anchored around this single winning baseline
        bounds = []
        for i, w in enumerate(base_cst):
            if i == 18:
                bounds.append((0.001, 0.005))
            else:
                margin = max(abs(w) * 0.25, 0.03)
                bounds.append((w - margin, w + margin))

        min_tc = req.get('min_tc', 0.08)
        constraints = [
            {'type': 'ineq', 'fun': self.constraint_no_intersection},
            {'type': 'ineq', 'fun': self.constraint_rounded_leading_edge},
            {'type': 'ineq', 'fun': lambda c, tc=min_tc: self.constraint_min_thickness(c, tc)}
        ]

        print(f"\n[Stage 2] Modifying CST weights of {base_id} (Target >= {target_acc_threshold:.1f}% Acc | "
              f"Patience: {patience} tries | Max: {max_tries} tries)...")
        rng = np.random.default_rng(seed=42)
        
        best_opt_cst = base_cst.copy()
        best_opt_score = initial_nn_score
        best_opt_acc = base_acc_pct
        winning_try = 1
        tries_since_best = 0
        try_logs = []

        # STAGE 2: Modify the chosen baseline's weights 
        for try_idx in range(1, max_tries + 1):
            try_t0 = time.perf_counter()

            if try_idx == 1:
                # Try 1 starts directly from the exact baseline CST weights
                x0_try = base_cst.copy()
                seed_type = "Exact Baseline"
            else:
                # Perturbations
                pct = 0.02 + (0.23 * ((try_idx - 2) / max(1, max_tries - 2)))
                x0_try = base_cst.copy()
                for i in range(18):
                    delta = max(abs(base_cst[i]) * pct, 0.006)
                    x0_try[i] = np.clip(base_cst[i] + rng.uniform(-delta, delta), bounds[i][0], bounds[i][1])
                x0_try[18] = np.clip(base_cst[18] + rng.uniform(-0.0005, 0.0005), 0.001, 0.005)
                seed_type = f"Perturbed (+/-{pct*100:.1f}%)"

            start_try_score = self.objective_function(x0_try, req)

            res = minimize(
                self.objective_function,
                x0=x0_try,
                args=(req,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 45, 'ftol': 1e-5, 'eps': 1e-4, 'disp': False}
            )

            try_time = time.perf_counter() - try_t0
            final_score = float(res.fun)
            is_valid = (
                self.constraint_no_intersection(res.x) >= -1e-4 and
                self.constraint_rounded_leading_edge(res.x) >= -1e-4 and
                self.constraint_min_thickness(res.x, min_tc) >= -1e-4
            )

            _, try_acc_pct, _, _ = self.calculate_accuracy_breakdown(res.x, base_cst, req)

            try_logs.append({
                'try_num': try_idx,
                'seed_type': seed_type,
                'initial_score': start_try_score,
                'final_score': final_score,
                'iterations': int(res.nit),
                'time_s': try_time,
                'valid': is_valid,
                'accuracy_pct': try_acc_pct
            })
            
            improved = False
            if is_valid:
                # Accept as new best if accuracy increases by >0.05%, OR if accuracy is tied/higher and score is lower
                if (try_acc_pct > best_opt_acc + 0.05) or (
                    try_acc_pct >= best_opt_acc - 0.05 and final_score < best_opt_score - 1e-5
                ):
                    best_opt_acc = try_acc_pct
                    best_opt_score = final_score
                    best_opt_cst = res.x.copy()
                    winning_try = try_idx
                    tries_since_best = 0
                    improved = True
                else:
                    if best_opt_acc >= target_acc_threshold:
                        tries_since_best += 1
            else:
                if best_opt_acc >= target_acc_threshold:
                    tries_since_best += 1

            status_tag = "NEW BEST" if improved else (
                f"Patience ({tries_since_best}/{patience})" if best_opt_acc >= target_acc_threshold else "Searching <95%"
            )
            print(f"  [Try {try_idx:02d}/{max_tries:02d}] {seed_type:<20} | "
                  f"Acc: {try_acc_pct:6.2f}% (Best: {best_opt_acc:6.2f}%) | "
                  f"Score: {start_try_score:8.4f} -> {final_score:8.4f} | "
                  f"Iter: {res.nit:02d} | {try_time:4.2f}s ({status_tag})")

            if best_opt_acc >= target_acc_threshold and tries_since_best >= patience:
                print(f"  >>> [EARLY STOP] {best_opt_acc:.2f}% accuracy achieved on Try #{winning_try} "
                      f"and {patience} subsequent tries found no better result. Stopping at Try #{try_idx}! <<<")
                break


        total_time = time.perf_counter() - t0

        result = {
            'winning_try': winning_try if winning_try > 0 else 1,
            'baseline_id': base_id,
            'base_cst': base_cst,
            'opt_cst': best_opt_cst,
            'as_is_dataset_score': as_is_dataset_score,
            'initial_nn_score': initial_nn_score,
            'final_score': best_opt_score,
            'best_accuracy_pct': best_opt_acc,
            'total_tries_executed': len(try_logs),
            'total_time_s': total_time,
            'try_logs': try_logs,
            'trust_degree_pct': self.calculate_trust_degree(best_opt_cst),
            'bounds': bounds
        }

        self.trigger_openfoam_pipeline_hook(best_opt_cst, req)

        cl_check, _, _ = self.evaluate_surrogate_polar(best_opt_cst, req['Re'])
        if req.get('stall_type') in ['open', 'closed']:
            max_i = int(np.argmax(cl_check))
            post_stall_drop = float(cl_check[max_i] - cl_check[-1])
            if max_i >= len(self.alpha_sweep) - 2 or post_stall_drop < 0.015:
                print("\n[WARNING] Based on the strict criteria provided, no airfoil was found that reaches full stall between -4° and 20°.")
                result['stall_warning'] = True
            else:
                result['stall_warning'] = False

        print(f"\n>>> Optimization Complete! Best Result from Try #{result['winning_try']} "
              f"(Base: {base_id} | Final Acc: {best_opt_acc:.2f}% | Tries: {len(try_logs)}) in {total_time:.2f}s <<<")
        return result

    # =========================================================================
    # 4. ACCURACY RATE CALCULATOR & FUTURE EXPANSION HOOKS
    # =========================================================================
    def calculate_accuracy_breakdown(self, opt_cst, base_cst, req):
        """Computes exact percentage accuracy using sub-degree interpolation."""
        cl_opt, cd_opt, cm_opt = self.evaluate_surrogate_polar(opt_cst, req['Re'])
        cl_base, cd_base, cm_base = self.evaluate_surrogate_polar(base_cst, req['Re'])

        m_opt = self.extract_curve_metrics(cl_opt, cd_opt, cm_opt)
        m_base = self.extract_curve_metrics(cl_base, cd_base, cm_base)

        rows = []

        def add_metric(label, target_val, achieved_val, base_val, scale_floor=0.1):
            denom = max(abs(target_val), scale_floor)
            acc = max(0.0, (1.0 - abs(achieved_val - target_val) / denom) * 100.0)
            rows.append({
                'parameter': label,
                'target': f"{target_val:.4f}",
                'baseline': f"{base_val:.4f}",
                'achieved': f"{achieved_val:.4f}",
                'abs_error': f"{abs(achieved_val - target_val):.4f}",
                'accuracy_pct': acc
            })

        if req['option'] in ['A', 'B']:
            cl_opt_c = float(np.interp(req['cruise_aoa'], self.alpha_sweep, cl_opt))
            cl_base_c = float(np.interp(req['cruise_aoa'], self.alpha_sweep, cl_base))
            add_metric(f"Target Cl @ AoA={req['cruise_aoa']}°", req['target_cl'], cl_opt_c, cl_base_c, 0.1)

        if req['option'] == 'C':
            if req.get('target_cl0') is not None:
                add_metric("Zero-Angle Lift (Cl0)", req['target_cl0'], m_opt['cl0'], m_base['cl0'], 0.1)
            if req.get('target_cl_alpha') is not None:
                add_metric("Lift Slope (Cl_alpha [/deg])", req['target_cl_alpha'], m_opt['cl_alpha'], m_base['cl_alpha'], 0.05)
            if req.get('target_alpha_max') is not None:
                add_metric("Stall Angle (Alpha_max [deg])", req['target_alpha_max'], m_opt['alpha_max'], m_base['alpha_max'], 10.0)

        if req.get('constrain_cm', False):
            if req.get('target_cm0') is not None:
                add_metric("Zero-Angle Moment (Cm0)", req['target_cm0'], m_opt['cm0'], m_base['cm0'], 0.05)
            if req.get('max_abs_cm') is not None:
                acc_cm = 100.0 if m_opt['max_abs_cm'] <= req['max_abs_cm'] else max(
                    0.0, (1.0 - (m_opt['max_abs_cm'] - req['max_abs_cm']) / max(req['max_abs_cm'], 0.05)) * 100.0
                )
                rows.append({
                    'parameter': "Max Pitching Moment |Cm|_max",
                    'target': f"<= {req['max_abs_cm']:.4f}",
                    'baseline': f"{m_base['max_abs_cm']:.4f}",
                    'achieved': f"{m_opt['max_abs_cm']:.4f}",
                    'abs_error': f"{max(0.0, m_opt['max_abs_cm'] - req['max_abs_cm']):.4f}",
                    'accuracy_pct': acc_cm
                })

        overall_acc = float(np.mean([r['accuracy_pct'] for r in rows])) if rows else 100.0
        return rows, overall_acc, (cl_opt, cd_opt, cm_opt), (cl_base, cd_base, cm_base)

    def calculate_trust_degree(self, opt_cst):
        """
        Sensitive, non-saturating Trust Degree (%) based on the standardized Z-distance
        to the nearest real physical airfoil in the dataset catalog.
        """
        if not hasattr(self, '_unique_dataset_csts'):
            unique_map = {}
            for (af_id, _), entry in self.dataset_catalog.items():
                if af_id not in unique_map:
                    unique_map[af_id] = entry['cst']
            self._unique_dataset_csts = np.array(list(unique_map.values()))
            self._cst_std = np.std(self._unique_dataset_csts, axis=0) + 1e-5

        # Standardized distance from opt_cst to every physical airfoil in the dataset
        z_diffs = np.abs(self._unique_dataset_csts - opt_cst) / self._cst_std
        nearest_mean_z = float(np.min(np.mean(z_diffs, axis=1)))
        nearest_max_z = float(np.min(np.max(z_diffs, axis=1)))

        # Smooth exponential decay starting immediately from 0.0 deviation
        trust = 100.0 * np.exp(-0.35 * nearest_mean_z - 0.08 * max(0.0, nearest_max_z - 0.5))
        return float(np.clip(trust, 0.0, 100.0))

    def trigger_openfoam_pipeline_hook(self, opt_cst, req):
        """Hook for future automated WSL OpenFOAM verification."""
        pass

    # =========================================================================
    # 5. SINGLE-DOCUMENT REPORT GENERATOR + AUTOMATIC BROWSER VIEWER
    # =========================================================================
    @staticmethod
    def _fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    @staticmethod
    def _get_next_run_id(out_dir="optimization"):
        os.makedirs(out_dir, exist_ok=True)
        existing = [f for f in os.listdir(out_dir) if re.match(r'^Optimization_\d{4}\.html$', f)]
        if not existing:
            return 1
        nums = [int(re.findall(r'\d{4}', f)[0]) for f in existing]
        return max(nums) + 1

    def export_single_document_report(self, result, req, auto_open_browser=True, run_id=None):
        """
        Compiles all 5 pictures and all numerical tables into optimization/Optimization_XXXX.html
        and automatically opens the rendered visual report in the user's default web browser.
        """
        out_dir = "optimization"
        if run_id is not None:
            run_num = int(run_id)
        else:
            run_num = self._get_next_run_id(out_dir)
        run_name = f"Optimization_{run_num:04d}"
        report_path = os.path.join(out_dir, f"{run_name}.html")

        opt_cst = result['opt_cst']
        base_cst = result['base_cst']
        x_opt, y_opt = self.get_airfoil_coordinates(opt_cst, 200)
        x_base, y_base = self.get_airfoil_coordinates(base_cst, 200)

        acc_rows, overall_acc, (cl_opt, cd_opt, cm_opt), (cl_base, cd_base, cm_base) = \
            self.calculate_accuracy_breakdown(opt_cst, base_cst, req)
        m_opt = self.extract_curve_metrics(cl_opt, cd_opt, cm_opt)

        # Build User Input Checklist HTML
        req_html_items = []
        req_html_items.append(f"<li><b>Flight Regime:</b> Re = {req['Re']/1e6:.2f}M</li>")
        req_html_items.append(f"<li><b>Minimum t/c:</b> {req.get('min_tc', 0.08):.3f}</li>")

        opt_names = {'A': 'Point Optimization (Cruise)', 'B': 'Broad Envelope Optimization', 'C': 'Lift Curve Architecture'}
        opt = req.get('option', 'A')
        req_html_items.append(f"<li><b>Primary Target (Option {opt}):</b> {opt_names.get(opt, opt)}</li>")

        if opt == 'A':
            req_html_items.append(f"<li><ul style='list-style-type: circle;'><li>Target Cl: {req.get('target_cl')} at AoA = {req.get('cruise_aoa')}°</li></ul></li>")
            fb = req.get('flatten_bucket', False)
            checked = "checked" if fb else ""
            req_html_items.append(f"<li><input type='checkbox' disabled {checked}> Flatten Drag Bucket (±{req.get('bucket_half_width', 2.0)}°)</li>")
        elif opt == 'B':
            req_html_items.append(f"<li><ul style='list-style-type: circle;'><li>Maximize L/D from {req.get('alpha_min')}° to {req.get('alpha_max')}°</li>")
            req_html_items.append(f"<li>Anchor: Target Cl = {req.get('target_cl')} at AoA = {req.get('cruise_aoa')}°</li></ul></li>")
        elif opt == 'C':
            req_html_items.append("<li><ul style='list-style-type: circle;'>")
            if req.get('target_cl0') is not None:
                req_html_items.append(f"<li>Target Cl0: {req['target_cl0']}</li>")
            if req.get('target_cl_alpha') is not None:
                req_html_items.append(f"<li>Target Cl_alpha: {req['target_cl_alpha']} /deg</li>")
            if req.get('target_alpha_max') is not None:
                req_html_items.append(f"<li>Target Alpha_max: {req['target_alpha_max']}°</li>")
            req_html_items.append("</ul></li>")

        ccm = req.get('constrain_cm', False)
        checked_cm = "checked" if ccm else ""
        req_html_items.append(f"<li><input type='checkbox' disabled {checked_cm}> Constrain Pitching Moment (Cm)</li>")
        if ccm:
            req_html_items.append("<li><ul style='list-style-type: circle;'>")
            if req.get('target_cm0') is not None:
                req_html_items.append(f"<li>Target Cm0: {req['target_cm0']}</li>")
            if req.get('max_abs_cm') is not None:
                req_html_items.append(f"<li>Max |Cm|: {req['max_abs_cm']}</li>")
            req_html_items.append("</ul></li>")

        st = req.get('stall_type')
        checked_st = "checked" if st else ""
        req_html_items.append(f"<li><input type='checkbox' disabled {checked_st}> Shape Stall Behavior {f'({st})' if st else ''}</li>")

        req_html_block = "<ul style='list-style-type: none; padding-left: 0; font-size: 1.05em; line-height: 1.8;'>" + "".join(req_html_items) + "</ul>"
        if result.get('stall_warning'):
            req_html_block = "<div style='background: #ffe6e6; border-left: 4px solid #d62728; padding: 10px; margin-bottom: 15px; color: #b30000; font-weight: bold;'>⚠️ WARNING: Based on the strict criteria provided, no airfoil was found that reaches full stall between -4° and 20°.</div>" + req_html_block

        # Picture 1: Baseline Airfoil
        fig1, ax1 = plt.subplots(figsize=(8, 2.8))
        ax1.plot(x_base, y_base, color='#1f77b4', linewidth=2.0, label=f"Baseline ({result['baseline_id']})")
        ax1.fill(x_base, y_base, color='#1f77b4', alpha=0.12)
        ax1.set_title(f"Selected Baseline Airfoil: {result['baseline_id']}")
        ax1.set_xlabel("x/c"); ax1.set_ylabel("y/c"); ax1.grid(True, linestyle='--', alpha=0.6); ax1.axis('equal')
        ax1.legend(loc='upper right')
        img_base_airfoil = self._fig_to_base64(fig1)

        # Picture 2: Final Optimized Airfoil
        fig2, ax2 = plt.subplots(figsize=(8, 2.8))
        ax2.plot(x_base, y_base, 'k--', linewidth=1.2, alpha=0.5, label=f"Baseline ({result['baseline_id']})")
        ax2.plot(x_opt, y_opt, color='#2ca02c', linewidth=2.2, label=f"Optimized ({run_name})")
        ax2.fill(x_opt, y_opt, color='#2ca02c', alpha=0.15)
        ax2.set_title(f"Final Optimized Airfoil ({run_name} - Try #{result['winning_try']})")
        ax2.set_xlabel("x/c"); ax2.set_ylabel("y/c"); ax2.grid(True, linestyle='--', alpha=0.6); ax2.axis('equal')
        ax2.legend(loc='upper right')
        img_opt_airfoil = self._fig_to_base64(fig2)

        # Picture 3: Lift Curve (Cl vs AoA) + Optimizer Target Overlays
        fig3, ax3 = plt.subplots(figsize=(6, 4.2))
        ax3.plot(self.alpha_sweep, cl_base, 'k--', alpha=0.6, label="Baseline")
        ax3.plot(self.alpha_sweep, cl_opt, color='#2ca02c', linewidth=2.2, marker='o', markersize=3, label="Optimized")

        if opt in ['A', 'B']:
            t_aoa = req.get('cruise_aoa')
            t_cl = req.get('target_cl')
            if t_aoa is not None and t_cl is not None:
                cl_ach = float(np.interp(t_aoa, self.alpha_sweep, cl_opt))
                ax3.plot(t_aoa, t_cl, marker='*', color='#d62728', markersize=11, linestyle='none', zorder=6,
                         label=f"Target Cl ({t_cl:.3f} @ {t_aoa}°)")
                ax3.plot(t_aoa, cl_ach, marker='D', color='#1b365d', markersize=6, linestyle='none', zorder=5,
                         label=f"Achieved Cl ({cl_ach:.3f})")
                ax3.annotate(f"Target Cl={t_cl:.2f}", (t_aoa, t_cl), textcoords="offset points",
                             xytext=(0, 8), ha='center', fontsize=8, fontweight='bold', color='#d62728')

        if opt == 'B':
            a_min, a_max = req.get('alpha_min'), req.get('alpha_max')
            if a_min is not None and a_max is not None:
                ax3.axvspan(a_min, a_max, color='#ffd700', alpha=0.18, label=f"Max L/D Env [{a_min}°, {a_max}°]")

        if opt == 'C':
            t_cl0 = req.get('target_cl0')
            if t_cl0 is not None:
                ax3.plot(0.0, t_cl0, marker='*', color='#d62728', markersize=11, linestyle='none', zorder=6,
                         label=f"Target Cl0 ({t_cl0:.3f})")
                ax3.plot(0.0, m_opt['cl0'], marker='D', color='#1b365d', markersize=6, linestyle='none', zorder=5,
                         label=f"Achieved Cl0 ({m_opt['cl0']:.3f})")
                ax3.annotate(f"Cl0={t_cl0:.2f}", (0.0, t_cl0), textcoords="offset points",
                             xytext=(6, -10), ha='left', fontsize=8, fontweight='bold', color='#d62728')

            t_cla = req.get('target_cl_alpha')
            if t_cla is not None:
                alpha_lin = np.linspace(-2.0, 6.0, 20)
                cl0_ref = t_cl0 if t_cl0 is not None else m_opt['cl0']
                cl_Target_line = cl0_ref + t_cla * alpha_lin
                ax3.plot(alpha_lin, cl_Target_line, color='#e377c2', linestyle='-.', linewidth=2.0, zorder=5,
                         label=f"Target Cl_α ({t_cla:.4f}/° | Ach: {m_opt['cl_alpha']:.4f}/°)")

            t_amax = req.get('target_alpha_max')
            if t_amax is not None:
                ax3.axvline(t_amax, color='#ff7f0e', linestyle='--', linewidth=1.8,
                            label=f"Target α_max ({t_amax:.1f}°)")
                ax3.plot(m_opt['alpha_max'], m_opt['cl_max'], marker='^', color='#ff7f0e', markersize=8,
                         linestyle='none', zorder=6, label=f"Stall Peak ({m_opt['alpha_max']:.1f}°)")

        if st in ['open', 'closed']:
            ax3.axvspan(10.0, 20.0, color='#9467bd', alpha=0.10, label=f"Stall Zone ({st.capitalize()})")
            ax3.plot(m_opt['alpha_max'], m_opt['cl_max'], marker='^', color='#9467bd', markersize=8, zorder=6)
            ax3.text(0.04, 0.94, f"Stall Style: {st.upper()}\nCl_max={m_opt['cl_max']:.3f} @ {m_opt['alpha_max']:.1f}°",
                     transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f0ff', edgecolor='#9467bd', alpha=0.85))

        ax3.set_title("Lift Coefficient Curve (Cl vs. AoA)"); ax3.set_xlabel("Angle of Attack α (°)"); ax3.set_ylabel("Cl")
        ax3.grid(True, linestyle='--', alpha=0.6); ax3.legend(loc='lower right', fontsize=7.5)
        img_cl = self._fig_to_base64(fig3)

        # Picture 4: Drag Curve (Cd vs AoA) + Optimizer Target Overlays
        fig4, ax4 = plt.subplots(figsize=(6, 4.2))
        ax4.plot(self.alpha_sweep, cd_base, 'k--', alpha=0.6, label="Baseline")
        ax4.plot(self.alpha_sweep, cd_opt, color='#d62728', linewidth=2.2, marker='o', markersize=3, label="Optimized")

        if opt in ['A', 'B'] and req.get('cruise_aoa') is not None:
            t_aoa = req['cruise_aoa']
            cd_opt_c = float(np.interp(t_aoa, self.alpha_sweep, cd_opt))
            cd_base_c = float(np.interp(t_aoa, self.alpha_sweep, cd_base))
            ax4.plot(t_aoa, cd_base_c, marker='o', color='black', markersize=6, linestyle='none', zorder=5)
            ax4.plot(t_aoa, cd_opt_c, marker='*', color='#1b365d', markersize=10, linestyle='none', zorder=6,
                     label=f"Cruise Cd ({cd_opt_c:.5f} @ {t_aoa}°)")
            ax4.annotate(f"Cd={cd_opt_c:.4f}", (t_aoa, cd_opt_c), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8, fontweight='bold', color='#1b365d')

        if opt == 'A' and req.get('flatten_bucket', False) and req.get('cruise_aoa') is not None:
            t_aoa = req['cruise_aoa']
            hw = req.get('bucket_half_width', 2.0)
            ax4.axvspan(t_aoa - hw, t_aoa + hw, color='#00bcd4', alpha=0.18,
                        label=f"Flat Bucket Zone ({t_aoa-hw:.1f}° to {t_aoa+hw:.1f}°)")

        if opt == 'B':
            a_min, a_max = req.get('alpha_min'), req.get('alpha_max')
            if a_min is not None and a_max is not None:
                ax4.axvspan(a_min, a_max, color='#ffd700', alpha=0.18, label=f"Max L/D Env [{a_min}°, {a_max}°]")

        if opt == 'C':
            min_cd_idx = int(np.argmin(cd_opt))
            ax4.plot(self.alpha_sweep[min_cd_idx], cd_opt[min_cd_idx], marker='*', color='#1b365d', markersize=10,
                     linestyle='none', zorder=6, label=f"Min Cd ({cd_opt[min_cd_idx]:.5f} @ {self.alpha_sweep[min_cd_idx]:.1f}°)")

        ax4.set_title("Drag Coefficient Curve (Cd vs. AoA)"); ax4.set_xlabel("Angle of Attack α (°)"); ax4.set_ylabel("Cd")
        ax4.grid(True, linestyle='--', alpha=0.6); ax4.legend(loc='upper left', fontsize=7.5)
        img_cd = self._fig_to_base64(fig4)

        # Picture 5: Pitching Moment Curve (Cm vs AoA) + Optimizer Target Overlays
        fig5, ax5 = plt.subplots(figsize=(6, 4.2))
        ax5.plot(self.alpha_sweep, cm_base, 'k--', alpha=0.6, label="Baseline")
        ax5.plot(self.alpha_sweep, cm_opt, color='#9467bd', linewidth=2.2, marker='o', markersize=3, label="Optimized")

        if req.get('constrain_cm', False):
            t_cm0 = req.get('target_cm0')
            if t_cm0 is not None:
                ax5.plot(0.0, t_cm0, marker='*', color='#d62728', markersize=11, linestyle='none', zorder=6,
                         label=f"Target Cm0 ({t_cm0:.3f})")
                ax5.plot(0.0, m_opt['cm0'], marker='D', color='#1b365d', markersize=6, linestyle='none', zorder=5,
                         label=f"Achieved Cm0 ({m_opt['cm0']:.3f})")
                ax5.annotate(f"Cm0={t_cm0:.3f}", (0.0, t_cm0), textcoords="offset points",
                             xytext=(6, 6), ha='left', fontsize=8, fontweight='bold', color='#d62728')

            max_cm = req.get('max_abs_cm')
            if max_cm is not None:
                ax5.axhline(max_cm, color='#d62728', linestyle='--', linewidth=1.5,
                            label=f"Max |Cm| Limit (±{max_cm:.3f})")
                ax5.axhline(-max_cm, color='#d62728', linestyle='--', linewidth=1.5)
                peak_cm_idx = int(np.argmax(np.abs(cm_opt)))
                ax5.plot(self.alpha_sweep[peak_cm_idx], cm_opt[peak_cm_idx], marker='s', color='#d62728',
                         markersize=6, linestyle='none', zorder=5,
                         label=f"Peak |Cm| ({m_opt['max_abs_cm']:.3f})")

        ax5.set_title("Pitching Moment Curve (Cm vs. AoA)"); ax5.set_xlabel("Angle of Attack α (°)"); ax5.set_ylabel("Cm")
        ax5.grid(True, linestyle='--', alpha=0.6); ax5.legend(loc='best', fontsize=7.5)
        img_cm = self._fig_to_base64(fig5)

        cst_table_rows = "".join(
            f"<tr><td>CST_{i:02d} ({'Lower' if i<9 else ('Upper' if i<18 else 'TE_Gap')})</td>"
            f"<td>{base_cst[i]:+.6f}</td><td>{opt_cst[i]:+.6f}</td>"
            f"<td>{opt_cst[i]-base_cst[i]:+.6f}</td></tr>"
            for i in range(19)
        )

        polar_table_rows = "".join(
            f"<tr><td>{self.alpha_sweep[i]:+5.1f}°</td>"
            f"<td>{cl_opt[i]:+.5f}</td><td>{cd_opt[i]:.5f}</td><td>{cm_opt[i]:+.5f}</td>"
            f"<td>{(cl_opt[i]/cd_opt[i]):.2f}</td></tr>"
            for i in range(25)
        )

        acc_table_rows = "".join(
            f"<tr><td>{r['parameter']}</td><td>{r['target']}</td><td>{r['baseline']}</td>"
            f"<td><b>{r['achieved']}</b></td><td>{r['abs_error']}</td>"
            f"<td><b>{r['accuracy_pct']:.2f}%</b></td></tr>"
            for r in acc_rows
        )

        try_table_rows = "".join(
            f"<tr style='{'background:#e8f8e8;font-weight:bold;' if t['try_num']==result['winning_try'] else ''}'>"
            f"<td>Try #{t['try_num']:02d}</td><td>{t['seed_type']}</td>"
            f"<td>{t['initial_score']:.5f}</td><td>{t['final_score']:.5f}</td>"
            f"<td>{t['iterations']}</td><td>{t['time_s']:.2f}s</td></tr>"
            for t in result['try_logs']
        )

        cad_coords_text = "\n".join(f"{x:.6f}\t{y:.6f}\t0.000000" for x, y in zip(x_opt, y_opt))
        cst_raw_text = ", ".join(f"{w:.6f}" for w in opt_cst)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{run_name} - Airfoil Optimization Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 28px; background: #f4f6f9; color: #222; }}
  .container {{ max-width: 1150px; margin: auto; background: #fff; padding: 28px; border-radius: 10px; box-shadow: 0 3px 12px rgba(0,0,0,0.08); }}
  h1, h2 {{ color: #1b365d; border-bottom: 2px solid #e3e8ef; padding-bottom: 6px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0; }}
  .kpi-card {{ background: #f8fafd; border-left: 4px solid #1b365d; padding: 12px 16px; border-radius: 6px; }}
  .kpi-val {{ font-size: 1.4em; font-weight: bold; color: #1b365d; }}
  .img-row {{ display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; margin: 16px 0; }}
  .img-box {{ flex: 1 1 340px; text-align: center; background: #fff; border: 1px solid #e1e6eb; border-radius: 8px; padding: 10px; }}
  .img-box img {{ max-width: 100%; height: auto; }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #d8e0e8; padding: 7px 10px; text-align: center; }}
  th {{ background: #1b365d; color: #fff; }}
  pre {{ background: #1e242b; color: #e6edf3; padding: 14px; border-radius: 6px; max-height: 300px; overflow-y: auto; font-size: 0.85em; }}
  button {{ background: #1b365d; color: white; border: none; padding: 8px 14px; border-radius: 5px; cursor: pointer; margin-bottom: 6px; }}
  button:hover {{ background: #2c5282; }}
</style>
<script>
function copyText(elementId) {{
  const text = document.getElementById(elementId).innerText;
  navigator.clipboard.writeText(text);
  alert("Copied to clipboard!");
}}
</script>
</head>
<body>
<div class="container">
  <h1>Airfoil Inverse Design Optimization Report — {run_name}</h1>

  <div class="kpi-grid">
    <div class="kpi-card"><div>Rate of Accuracy</div><div class="kpi-val">{overall_acc:.2f}%</div></div>
    <div class="kpi-card"><div>Total Time Taken</div><div class="kpi-val">{result['total_time_s']:.2f} s</div></div>
    <div class="kpi-card"><div>Selected Base Model</div><div class="kpi-val">{result['baseline_id']} (Try #{result['winning_try']})</div></div>
    <div class="kpi-card"><div>Dataset Trust Degree</div><div class="kpi-val">{result['trust_degree_pct']:.1f}%</div></div>
  </div>

  <h2>1. User Inputs & Optimization Configuration</h2>
  <div style='background: #fff; padding: 10px 20px; border: 1px solid #e1e6eb; border-radius: 8px;'>
    {req_html_block}
  </div>

  <h2>2. Rate of Accuracy Compared to Original User Requirements</h2>
  <table>
    <tr><th>Requested Parameter</th><th>User Target</th><th>Baseline Value</th><th>Optimized Value</th><th>Absolute Error</th><th>Accuracy Rate</th></tr>
    {acc_table_rows}
  </table>

  <h2>3. Baseline Airfoil vs. Final Optimized Airfoil</h2>
  <div class="img-row">
    <div class="img-box"><img src="data:image/png;base64,{img_base_airfoil}"/></div>
    <div class="img-box"><img src="data:image/png;base64,{img_opt_airfoil}"/></div>
  </div>

  <h2>4. Aerodynamic Polar Curves (-4° to 20° AoA)</h2>
  <div class="img-row">
    <div class="img-box"><img src="data:image/png;base64,{img_cl}"/></div>
    <div class="img-box"><img src="data:image/png;base64,{img_cd}"/></div>
    <div class="img-box"><img src="data:image/png;base64,{img_cm}"/></div>
  </div>

  <h2>5. Optimization Trajectory ({result['total_tries_executed']} Tries Evaluated)</h2>
  <table>
    <tr><th>Try</th><th>Starting Weight Modification</th><th>Initial Score</th><th>Final Score</th><th>Iterations</th><th>Time Taken</th></tr>
    {try_table_rows}
  </table>

  <h2>6. 19 CST Weights (Baseline vs. Optimized)</h2>
  <button onclick="copyText('cst-block')">Copy 19 CST Weights</button>
  <pre id="cst-block">{cst_raw_text}</pre>
  <table>
    <tr><th>CST Weight</th><th>Baseline ({result['baseline_id']})</th><th>Optimized ({run_name})</th><th>Delta (Δ)</th></tr>
    {cst_table_rows}
  </table>

  <h2>7. Numerical Aerodynamic Coefficients (-4.0° to 20.0° AoA)</h2>
  <table>
    <tr><th>AoA (°)</th><th>Lift Coeff (Cl)</th><th>Drag Coeff (Cd)</th><th>Moment Coeff (Cm)</th><th>Glide Ratio (L/D)</th></tr>
    {polar_table_rows}
  </table>

  <h2>8. 400 CAD Coordinates for SolidWorks (X, Y, Z=0)</h2>
  <button onclick="copyText('cad-block')">Copy 400 SolidWorks Coordinates</button>
  <pre id="cad-block">{cad_coords_text}</pre>
</div>
</body>
</html>"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Append or Overwrite in master CSV log
        ledger_path = os.path.join(out_dir, "optimization_master_log.csv")
        ledger_row = pd.DataFrame([{
            'Run_ID': run_name,
            'Option': req['option'],
            'Re': req['Re'],
            'Baseline_ID': result['baseline_id'],
            'Winning_Try': result['winning_try'],
            'Overall_Accuracy_Pct': round(overall_acc, 3),
            'Trust_Degree_Pct': round(result['trust_degree_pct'], 3),
            'Total_Time_s': round(result['total_time_s'], 3),
            **{f'CST_{i}': opt_cst[i] for i in range(19)}
        }])
        
        if os.path.exists(ledger_path):
            existing_ledger = pd.read_csv(ledger_path)
            if run_name in existing_ledger['Run_ID'].values:
                # Overwrite existing run
                existing_ledger = existing_ledger[existing_ledger['Run_ID'] != run_name]
            combined_ledger = pd.concat([existing_ledger, ledger_row], ignore_index=True)
            # Sort by Run_ID nicely
            combined_ledger.sort_values(by='Run_ID', inplace=True)
            combined_ledger.to_csv(ledger_path, index=False)
        else:
            ledger_row.to_csv(ledger_path, index=False)

        abs_report_path = os.path.realpath(report_path)
        print(f"\n[SUCCESS] Single-document report saved to: {abs_report_path}")

        if auto_open_browser:
            print("Opening rendered report in your default web browser...")
            webbrowser.open('file://' + abs_report_path)

        return report_path


# =============================================================================
# INTERACTIVE CLI COLLECTOR (MUTUALLY EXCLUSIVE OPTIONS A, B, C)
# =============================================================================
def collect_user_inputs_cli():
    print("\n========================================================")
    print("   INTELLIGENT AIRFOIL INVERSE DESIGN OPTIMIZER (v2.2)  ")
    print("========================================================")
    print("Max Thickness / c range : 0.0542 to 0.6199")
    print("Max Camber / c range    : 0.0000 to 0.2290")
    print("Trailing Edge Gap       : Locked to [0.0010, 0.0050] (Sharp TE)")
    print("Valid Flight Regime     : Re in [3.0M, 9.0M] | AoA in [-4.0°, 20.0°]")
    print("========================================================\n")

    re_m = float(input("Enter Flight Reynolds Number in Millions (e.g., 3.0, 6.0, 9.0): ").strip())
    min_tc_str = input("Enter Minimum Thickness ratio t/c [Press Enter for default 0.08]: ").strip()
    min_tc = float(min_tc_str) if min_tc_str else 0.08

    print("\nSelect Primary Aerodynamic Target (Mutually Exclusive):")
    print("  [A] Point Optimization (Minimize Drag at Target Cruise Cl & AoA)")
    print("  [B] Broad Envelope Optimization (Maximize L/D across AoA Range + Cruise Cl Anchor)")
    print("  [C] Lift Curve Architecture (Target up to 2 of: Cl0, Cl_alpha, Alpha_max)")
    option = input("Choose Option (A, B, or C): ").strip().upper()

    req = {'Re': re_m * 1e6, 'min_tc': min_tc, 'option': option}

    if option == 'A':
        req['cruise_aoa'] = float(input("  Enter Cruise Angle of Attack (-4.0 to 20.0°): ").strip())
        req['target_cl'] = float(input("  Enter Target Lift Coefficient (Cl): ").strip())

        if input("  Modifier - Flatten Drag Bucket for gust tolerance? (y/n): ").strip().lower() == 'y':
            req['flatten_bucket'] = True
            hw = input("    Enter Bucket Half-Width in degrees [Enter for default 2°]: ").strip()
            req['bucket_half_width'] = float(hw) if hw else 2.0

    elif option == 'B':
        req['alpha_min'] = float(input("  Enter Envelope Start AoA (e.g., 2.0°): ").strip())
        req['alpha_max'] = float(input("  Enter Envelope End AoA (e.g., 10.0°): ").strip())
        req['cruise_aoa'] = float(input("  Enter Cruise Anchor AoA inside range (e.g., 5.0°): ").strip())
        req['target_cl'] = float(input("  Enter Required Cruise Cl at Anchor AoA: ").strip())
        print("  [UI Rule] Drag Bucket and Stall modifiers automatically disabled for Option B.")

    elif option == 'C':
        print("  [UI Rule] Specify at least 1 and AT MOST 2 targets (leave blank to ignore):")
        cl0_s = input("  1. Target Zero-Angle Lift Cl0 (e.g., 0.25) [Enter to skip]: ").strip()
        cla_s = input("  2. Target Lift Slope Cl_alpha in /deg (e.g., 0.105) [Enter to skip]: ").strip()

        req['target_cl0'] = float(cl0_s) if cl0_s else None
        req['target_cl_alpha'] = float(cla_s) if cla_s else None

        if req['target_cl0'] is not None and req['target_cl_alpha'] is not None:
            print("  [UI Auto-Exclusion] 2 of 3 inputs filled. Target Stall Angle (Alpha_max) locked to float naturally.")
            req['target_alpha_max'] = None
        else:
            amax_s = input("  3. Target Stall Angle Alpha_max in deg (e.g., 15.0) [Enter to skip]: ").strip()
            req['target_alpha_max'] = float(amax_s) if amax_s else None

    if input("\nModifier - Constrain Pitching Moment (Cm)? (y/n): ").strip().lower() == 'y':
        req['constrain_cm'] = True
        cm0_s = input("  Target Zero-Angle Moment Cm0 [Enter to skip]: ").strip()
        cm_max_s = input("  Max Allowable |Cm|_max across polar [Enter to skip]: ").strip()
        req['target_cm0'] = float(cm0_s) if cm0_s else None
        req['max_abs_cm'] = float(cm_max_s) if cm_max_s else None

    if option in ['A', 'C']:
        if input("Modifier - Shape High-AoA Stall Curve Behavior? (y/n): ").strip().lower() == 'y':
            st = input("  Choose Stall Type ('open' for docile/flat, 'closed' for sharp/max-peak): ").strip().lower()
            req['stall_type'] = st if st in ['open', 'closed'] else None

    return req


if __name__ == "__main__":
    engine = AirfoilOptimizerEngine(
        model_path='data/airfoil_surrogate_model.keras',
        dataset_path='data/complete_dataset_aerodinamico.csv',
        dict_path='data/airfoil_tensor_dict.pkl'
    )
    user_request = collect_user_inputs_cli()
    optimization_result = engine.run_optimization(user_request, max_tries=50, target_acc_threshold=95.0, patience=10)
    engine.export_single_document_report(optimization_result, user_request, auto_open_browser=True)