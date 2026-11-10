import os
import re
import time
import threading
import webbrowser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, jsonify, send_file

# Import only AirfoilOptimizerEngine (since _fig_to_base64 is a staticmethod inside it)
from intelligent_optimizer import AirfoilOptimizerEngine

app = Flask(__name__)

# Initialize the engine once at startup
print("Initializing AirfoilOptimizerEngine for Web UI...")
engine = AirfoilOptimizerEngine()

# Global job state tracker for asynchronous multi-stage UI updates
job_state = {
    'status': 'idle',          # 'idle', 'finding_baseline', 'baseline_found', 'optimizing', 'done', 'error'
    'generic_img': None,
    'baseline_img': None,
    'baseline_id': None,
    'final_images': [],        # [Airfoil Overlay, Cl Curve, Cd Curve, Cm Curve]
    'html_path': None,
    'error': None
}


def render_single_airfoil_b64(cst_weights, title, color='#1e5bb0', fill_color='#dce8f8'):
    """Generates a clean Base64 PNG for the initial Airfoil_0000 and the found Baseline."""
    x, y = engine.get_airfoil_coordinates(cst_weights, 200)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x, y, color=color, linewidth=2.2, label=title)
    ax.fill(x, y, color=fill_color, alpha=0.45)
    ax.set_title(title, fontsize=12, fontweight='bold', color='#1b365d')
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    ax.legend(loc='upper right')
    return engine._fig_to_base64(fig)


def get_initial_airfoil_0000():
    """Extracts Airfoil_0000 from engine.dataset_catalog (or engine.dataset_polars) on startup."""
    if hasattr(engine, 'dataset_catalog') and engine.dataset_catalog:
        for (af_id, _), entry in engine.dataset_catalog.items():
            if af_id == 'Airfoil_0000' or entry.get('id') == 'Airfoil_0000':
                return render_single_airfoil_b64(entry['cst'], "Generic Baseline: Airfoil_0000")
        first_entry = next(iter(engine.dataset_catalog.values()))
        return render_single_airfoil_b64(first_entry['cst'], f"Generic Baseline: {first_entry.get('id', 'Airfoil_0000')}")
    elif hasattr(engine, 'dataset_polars') and engine.dataset_polars:
        for entry in engine.dataset_polars:
            af_id = entry.get('airfoil_id') or entry.get('id')
            if af_id == 'Airfoil_0000':
                return render_single_airfoil_b64(entry['cst'], "Generic Baseline: Airfoil_0000")
        first_entry = engine.dataset_polars[0]
        af_id = first_entry.get('airfoil_id') or first_entry.get('id', 'Airfoil_0000')
        return render_single_airfoil_b64(first_entry['cst'], f"Generic Baseline: {af_id}")
    raise RuntimeError("Dataset catalog is empty.")


# Pre-render Airfoil_0000 on startup
job_state['generic_img'] = get_initial_airfoil_0000()


def extract_carousel_images_from_html(html_path):
    """
    Extracts the Base64 PNGs generated inside Optimization_XXXX.html:
    Returns 4 images for the swipe carousel:
      [0] Optimized vs. Baseline Airfoil Geometry
      [1] Lift Curve (Cl vs AoA)
      [2] Drag Curve (Cd vs AoA)
      [3] Pitching Moment Curve (Cm vs AoA)
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    b64_matches = re.findall(r'src="data:image/png;base64,([^"]+)"', content)
    if len(b64_matches) >= 5:
        # b64_matches[0] is base, [1] is final overlay, [2] is Cl, [3] is Cd, [4] is Cm
        return [b64_matches[1], b64_matches[2], b64_matches[3], b64_matches[4]]
    return b64_matches


def background_optimization_worker(req):
    """Executes Stage 1 (Baseline Search) -> 3s Pause -> Stage 2 (Optimization & HTML Export)."""
    global job_state
    try:
        # --- STAGE 1: FINDING BASELINE ---
        job_state['status'] = 'finding_baseline'
        job_state['error'] = None

        best_entry, as_is_score = engine.find_best_dataset_baseline(req)
        base_cst = best_entry['cst'].copy()
        base_id = best_entry.get('id') or best_entry.get('airfoil_id', 'Baseline')

        base_img = render_single_airfoil_b64(
            base_cst,
            f"Selected Dataset Baseline: {base_id}",
            color='#4a5568',
            fill_color='#e2e8f0'
        )
        job_state['baseline_img'] = base_img
        job_state['baseline_id'] = base_id
        job_state['status'] = 'baseline_found'

        # Display the baseline clearly for 3 seconds before blurring
        time.sleep(3.2)

        # --- STAGE 2: OPTIMIZING AIRFOIL ---
        job_state['status'] = 'optimizing'

        if hasattr(engine, 'run_optimization'):
            result = engine.run_optimization(req)
        elif hasattr(engine, 'run_optimization_adaptive'):
            result = engine.run_optimization_adaptive(req)
        else:
            result = engine.run_10_weight_modifications(req)

        # Generate the single-document HTML report without popping a second browser tab
        report_path = engine.export_single_document_report(result, req, auto_open_browser=False)

        carousel_imgs = extract_carousel_images_from_html(report_path)

        job_state['final_images'] = carousel_imgs
        job_state['html_path'] = os.path.abspath(report_path)
        job_state['status'] = 'done'

    except Exception as e:
        job_state['error'] = str(e)
        job_state['status'] = 'error'


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Airfoil Optimization</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
    body { background: #f4f7fb; color: #1e293b; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

    /* Top Blue Fringe */
    .top-fringe {
      background: linear-gradient(90deg, #1b4f9c 0%, #2563eb 100%);
      color: #ffffff;
      padding: 16px 32px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.5px;
      box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15);
      z-index: 10;
    }

    /* Main Split Container */
    .main-container {
      display: flex;
      flex: 1;
      height: calc(100vh - 64px);
    }

    /* Left Interactive Area (50%) */
    .left-panel {
      width: 50%;
      padding: 26px 34px;
      overflow-y: auto;
      background: #ffffff;
      border-right: 1px solid #dbe3ef;
    }

    .welcome-msg {
      font-size: 16px;
      font-weight: 600;
      color: #1b365d;
      margin-bottom: 20px;
      padding-bottom: 10px;
      border-bottom: 2px solid #e2e8f0;
    }

    .control-group {
      margin-bottom: 18px;
      background: #f8fafc;
      padding: 14px 16px;
      border-radius: 8px;
      border: 1px solid #e2e8f0;
    }

    .control-label {
      display: block;
      font-weight: 600;
      font-size: 14px;
      color: #1e3a8a;
      margin-bottom: 8px;
    }

    .slider-row {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .slider-row input[type="range"] {
      flex: 1;
      accent-color: #2563eb;
      cursor: pointer;
    }

    .num-box {
      width: 85px;
      padding: 6px 8px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 600;
      text-align: center;
      color: #0f172a;
    }

    /* Sliding Accordion Menu for Primary Target */
    .sliding-menu {
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
      margin-bottom: 18px;
    }

    .menu-header {
      background: #eff6ff;
      color: #1e3a8a;
      padding: 12px 16px;
      font-weight: 700;
      font-size: 14px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }

    .menu-options {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.3s ease-out;
      background: #ffffff;
    }

    .menu-options.open {
      max-height: 180px;
      border-top: 1px solid #dbe3ef;
    }

    .menu-item {
      padding: 11px 16px;
      font-size: 14px;
      cursor: pointer;
      border-bottom: 1px solid #f1f5f9;
      transition: background 0.15s;
    }

    .menu-item:hover { background: #e0ecff; }
    .menu-item.active { background: #dbeafe; font-weight: 700; color: #1d4ed8; }

    /* Dynamic Option Sub-boxes */
    .target-subpanel {
      display: none;
      background: #f8fafc;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      margin-bottom: 18px;
    }
    .target-subpanel.active { display: block; }

    .field-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }

    .field-item label {
      display: block;
      font-size: 12.5px;
      font-weight: 600;
      color: #334155;
      margin-bottom: 4px;
    }

    .field-item input, .field-item select {
      width: 100%;
      padding: 7px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      font-size: 13.5px;
    }

    .field-item input:disabled {
      background: #e2e8f0;
      color: #94a3b8;
      cursor: not-allowed;
    }

    .modifier-box {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px dashed #cbd5e1;
      font-size: 13.5px;
    }

    .modifier-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      font-weight: 600;
      color: #1e3a8a;
    }

    /* Stronger Blue Optimize Button */
    .optimize-btn {
      width: 100%;
      background: #0d3b8c;
      color: #ffffff;
      border: none;
      padding: 14px;
      font-size: 16px;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(13, 59, 140, 0.3);
      transition: background 0.2s, transform 0.1s;
    }
    .optimize-btn:hover { background: #092c6b; }
    .optimize-btn:active { transform: scale(0.99); }
    .optimize-btn:disabled { background: #94a3b8; cursor: not-allowed; box-shadow: none; }

    /* Right Visualization Area (50%) */
    .right-panel {
      width: 50%;
      padding: 26px 34px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
      background: #f4f7fb;
    }

    .viewer-card {
      width: 100%;
      height: 78%;
      background: #ffffff;
      border-radius: 12px;
      border: 1px solid #dbe3ef;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 20px;
      overflow: hidden;
    }

    .image-stage {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: filter 0.4s ease;
    }

    .image-stage.blurred {
      filter: blur(8px) brightness(0.92);
    }

    .image-stage img {
      max-width: 96%;
      max-height: 92%;
      object-fit: contain;
      border-radius: 6px;
    }

    /* Throbber Overlay */
    .throbber-overlay {
      position: absolute;
      inset: 0;
      background: rgba(255, 255, 255, 0.65);
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 20;
    }
    .throbber-overlay.active { display: flex; }

    .spinner {
      width: 54px;
      height: 54px;
      border: 6px solid #dbeafe;
      border-top: 6px solid #0d3b8c;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
      margin-bottom: 14px;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }

    .throbber-text {
      font-size: 17px;
      font-weight: 700;
      color: #0d3b8c;
      background: rgba(255, 255, 255, 0.9);
      padding: 8px 18px;
      border-radius: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Carousel Controls */
    .carousel-nav {
      display: none;
      width: 100%;
      justify-content: space-between;
      align-items: center;
      margin-top: 12px;
      padding: 0 10px;
    }
    .carousel-nav.active { display: flex; }

    .nav-btn {
      background: #1e40af;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
    }
    .nav-btn:hover { background: #1e3a8a; }

    .slide-caption {
      font-size: 14px;
      font-weight: 700;
      color: #1e3a8a;
    }

    /* Download HTML Button */
    .download-btn {
      margin-top: 18px;
      background: #15803d;
      color: #ffffff;
      text-decoration: none;
      padding: 12px 28px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 15px;
      display: none;
      box-shadow: 0 4px 10px rgba(21, 128, 61, 0.25);
      transition: background 0.2s;
    }
    .download-btn.active { display: inline-block; }
    .download-btn:hover { background: #166534; }
  </style>
</head>
<body>

  <!-- Top Fringe Blue -->
  <div class="top-fringe">Airfoil Optimization</div>

  <div class="main-container">
    <!-- Left Interactive Half -->
    <div class="left-panel">
      <div class="welcome-msg">
        Welcome to the airfoil optimization, please specify your airfoil needs.
      </div>

      <!-- Reynolds Number (x.xx format, 3.00 to 9.00 M) -->
      <div class="control-group">
        <label class="control-label">Reynold’s number (Millions, x.xx format)</label>
        <div class="slider-row">
          <input type="range" id="re_slider" min="3.00" max="9.00" step="0.01" value="6.00">
          <input type="text" id="re_box" class="num-box" value="6.00">
        </div>
      </div>

      <!-- Minimum t/c (0.xx format) -->
      <div class="control-group">
        <label class="control-label">Minimum t/c (0.xx format)</label>
        <div class="slider-row">
          <input type="range" id="tc_slider" min="0.05" max="0.40" step="0.01" value="0.08">
          <input type="text" id="tc_box" class="num-box" value="0.08">
        </div>
      </div>

      <!-- Primary Target Sliding Menu -->
      <div class="sliding-menu">
        <div class="menu-header" onclick="toggleTargetMenu()">
          <span id="selected_target_label">Primary target: Option A (Point Optimization)</span>
          <span>▼</span>
        </div>
        <div class="menu-options" id="target_menu_options">
          <div class="menu-item active" onclick="selectTarget('A', 'Option A (Point Optimization)')">
            Option A — Point Optimization (Cruise Cl & AoA)
          </div>
          <div class="menu-item" onclick="selectTarget('B', 'Option B (Broad Envelope L/D)')">
            Option B — Broad Envelope Optimization (Maximize L/D Range)
          </div>
          <div class="menu-item" onclick="selectTarget('C', 'Option C (Lift Curve Architecture)')">
            Option C — Lift Curve Architecture (Cl0, Cl_alpha, Alpha_max)
          </div>
        </div>
      </div>

      <!-- Option A Inputs -->
      <div id="panel_A" class="target-subpanel active">
        <div class="field-grid">
          <div class="field-item">
            <label>Cruise AoA (-4 to 20°)</label>
            <input type="text" id="a_cruise_aoa" class="max2dec" value="5.0">
          </div>
          <div class="field-item">
            <label>Target Lift Coefficient (Cl)</label>
            <input type="text" id="a_target_cl" class="max2dec" value="1.10">
          </div>
        </div>
        <div class="modifier-box">
          <label class="modifier-row">
            <input type="checkbox" id="a_flatten_bucket" onchange="toggleSubBox('a_bucket_box', this.checked)">
            Flatten Drag Bucket (Gust Tolerance)
          </label>
          <div id="a_bucket_box" style="display:none; margin-top:6px;">
            <div class="field-item">
              <label>Bucket Half-Width (± deg)</label>
              <input type="text" id="a_bucket_hw" class="max2dec" value="2.0">
            </div>
          </div>
        </div>
      </div>

      <!-- Option B Inputs -->
      <div id="panel_B" class="target-subpanel">
        <div class="field-grid">
          <div class="field-item">
            <label>Envelope Start AoA (°)</label>
            <input type="text" id="b_alpha_min" class="max2dec" value="2.0">
          </div>
          <div class="field-item">
            <label>Envelope End AoA (°)</label>
            <input type="text" id="b_alpha_max" class="max2dec" value="10.0">
          </div>
          <div class="field-item">
            <label>Cruise Anchor AoA (°)</label>
            <input type="text" id="b_cruise_aoa" class="max2dec" value="5.0">
          </div>
          <div class="field-item">
            <label>Required Cruise Cl</label>
            <input type="text" id="b_target_cl" class="max2dec" value="1.05">
          </div>
        </div>
      </div>

      <!-- Option C Inputs (At most 2 of 3 allowed) -->
      <div id="panel_C" class="target-subpanel">
        <p style="font-size:12px; color:#475569; margin-bottom:10px;">
          Fill up to 2 targets (filling any 2 automatically locks the 3rd):
        </p>
        <div class="field-grid">
          <div class="field-item">
            <label>Target Zero-Angle Lift (Cl0)</label>
            <input type="text" id="c_cl0" class="max2dec c-limit" value="0.25" oninput="enforceOptionCLimit()">
          </div>
          <div class="field-item">
            <label>Target Lift Slope (Cl_alpha /deg)</label>
            <input type="text" id="c_cla" class="max2dec c-limit" value="0.11" oninput="enforceOptionCLimit()">
          </div>
          <div class="field-item">
            <label>Target Stall Angle (Alpha_max °)</label>
            <input type="text" id="c_amax" class="max2dec c-limit" value="" disabled oninput="enforceOptionCLimit()">
          </div>
        </div>
      </div>

      <!-- Shared Modifiers -->
      <div class="control-group">
        <label class="modifier-row">
          <input type="checkbox" id="constrain_cm" onchange="toggleSubBox('cm_subbox', this.checked)">
          Constrain Pitching Moment (Cm)
        </label>
        <div id="cm_subbox" class="field-grid" style="display:none; margin-top:8px;">
          <div class="field-item">
            <label>Target Cm0 (Optional)</label>
            <input type="text" id="target_cm0" class="max2dec" placeholder="e.g. -0.05">
          </div>
          <div class="field-item">
            <label>Max Allowable |Cm|</label>
            <input type="text" id="max_abs_cm" class="max2dec" placeholder="e.g. 0.12">
          </div>
        </div>

        <div id="stall_modifier_wrapper" style="margin-top:8px;">
          <label class="modifier-row">
            <input type="checkbox" id="shape_stall" onchange="toggleSubBox('stall_subbox', this.checked)">
            Shape High-AoA Stall Behavior
          </label>
          <div id="stall_subbox" style="display:none; margin-top:6px;">
            <div class="field-item">
              <label>Stall Profile Type</label>
              <select id="stall_type">
                <option value="open">Open / Docile (Smooth Top Plateau)</option>
                <option value="closed">Closed / Sharp (Maximum Peak Lift)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <!-- Stronger Blue Optimize Button -->
      <button id="optimize_btn" class="optimize-btn" onclick="startOptimization()">Optimize</button>
    </div>

    <!-- Right Visualization Half -->
    <div class="right-panel">
      <div class="viewer-card" id="swipe_area">
        <div class="image-stage" id="image_stage">
          <img id="display_img" src="data:image/png;base64,{{ generic_img }}" alt="Airfoil View">
        </div>

        <!-- Animated Throbber Overlay -->
        <div class="throbber-overlay" id="throbber_overlay">
          <div class="spinner"></div>
          <div class="throbber-text" id="throbber_text">Finding Baseline</div>
        </div>
      </div>

      <!-- Swipe / Carousel Navigation (Visible after optimization) -->
      <div class="carousel-nav" id="carousel_nav">
        <button class="nav-btn" onclick="changeSlide(-1)">&#8592; Prev</button>
        <span class="slide-caption" id="slide_caption">1 / 4 — Optimized Airfoil Geometry</span>
        <button class="nav-btn" onclick="changeSlide(1)">Next &#8594;</button>
      </div>

      <!-- Download HTML Report Button (Only available after optimization) -->
      <a href="/download_report" id="download_btn" class="download-btn" download>
        Download Full HTML Optimization Report
      </a>
    </div>
  </div>

  <script>
    let currentOption = 'A';
    let carouselImages = [];
    let currentSlide = 0;
    let pollTimer = null;
    const slideTitles = [
      "1 / 4 — Optimized vs. Baseline Airfoil",
      "2 / 4 — Lift Coefficient Curve (Cl vs. AoA)",
      "3 / 4 — Drag Coefficient Curve (Cd vs. AoA)",
      "4 / 4 — Pitching Moment Curve (Cm vs. AoA)"
    ];

    // --- 1. Synchronize Reynolds Slider & Box (x.xx format) ---
    const reSlider = document.getElementById('re_slider');
    const reBox = document.getElementById('re_box');
    reSlider.addEventListener('input', () => { reBox.value = parseFloat(reSlider.value).toFixed(2); });
    reBox.addEventListener('change', () => {
      let val = parseFloat(reBox.value);
      if (isNaN(val)) val = 6.00;
      val = Math.min(9.00, Math.max(3.00, val));
      reBox.value = val.toFixed(2);
      reSlider.value = val.toFixed(2);
    });

    // --- 2. Synchronize Minimum t/c Slider & Box (0.xx format) ---
    const tcSlider = document.getElementById('tc_slider');
    const tcBox = document.getElementById('tc_box');
    tcSlider.addEventListener('input', () => { tcBox.value = parseFloat(tcSlider.value).toFixed(2); });
    tcBox.addEventListener('change', () => {
      let val = parseFloat(tcBox.value);
      if (isNaN(val)) val = 0.08;
      val = Math.min(0.40, Math.max(0.05, val));
      tcBox.value = val.toFixed(2);
      tcSlider.value = val.toFixed(2);
    });

    // --- 3. Strict Format Enforcement: Allow x, x.x, x.xx but NEVER x.xxx ---
    function attachMax2DecValidation() {
      document.querySelectorAll('.max2dec').forEach(input => {
        input.addEventListener('input', function() {
          let v = this.value;
          if (v.includes('.')) {
            const parts = v.split('.');
            if (parts[1] && parts[1].length > 2) {
              this.value = parts[0] + '.' + parts[1].substring(0, 2);
            }
          }
        });
      });
    }
    attachMax2DecValidation();

    // --- 4. Sliding Menu & Mutual Exclusion Routing ---
    function toggleTargetMenu() {
      document.getElementById('target_menu_options').classList.toggle('open');
    }

    function selectTarget(opt, labelText) {
      currentOption = opt;
      document.getElementById('selected_target_label').innerText = "Primary target: " + labelText;
      document.getElementById('target_menu_options').classList.remove('open');

      document.querySelectorAll('.menu-item').forEach((el, idx) => {
        el.classList.toggle('active', ['A','B','C'][idx] === opt);
      });

      ['A', 'B', 'C'].forEach(k => {
        document.getElementById('panel_' + k).classList.toggle('active', k === opt);
      });

      // Option B excludes Stall Behavior modifier
      document.getElementById('stall_modifier_wrapper').style.display = (opt === 'B') ? 'none' : 'block';
    }

    function toggleSubBox(id, show) {
      document.getElementById(id).style.display = show ? 'grid' : 'none';
    }

    function enforceOptionCLimit() {
      const inputs = Array.from(document.querySelectorAll('.c-limit'));
      const filled = inputs.filter(i => i.value.trim() !== '');
      inputs.forEach(i => {
        if (filled.length >= 2 && i.value.trim() === '') {
          i.disabled = true;
        } else {
          i.disabled = false;
        }
      });
    }

    // --- 5. Start Optimization & Poll Multi-Stage Throbber States ---
    function startOptimization() {
      const payload = {
        Re: parseFloat(reBox.value) * 1e6,
        min_tc: parseFloat(tcBox.value),
        option: currentOption,
        constrain_cm: document.getElementById('constrain_cm').checked,
        target_cm0: document.getElementById('target_cm0').value,
        max_abs_cm: document.getElementById('max_abs_cm').value,
        stall_type: (currentOption !== 'B' && document.getElementById('shape_stall').checked)
                    ? document.getElementById('stall_type').value : null
      };

      if (currentOption === 'A') {
        payload.cruise_aoa = parseFloat(document.getElementById('a_cruise_aoa').value);
        payload.target_cl = parseFloat(document.getElementById('a_target_cl').value);
        payload.flatten_bucket = document.getElementById('a_flatten_bucket').checked;
        payload.bucket_half_width = parseFloat(document.getElementById('a_bucket_hw').value || 2.0);
      } else if (currentOption === 'B') {
        payload.alpha_min = parseFloat(document.getElementById('b_alpha_min').value);
        payload.alpha_max = parseFloat(document.getElementById('b_alpha_max').value);
        payload.cruise_aoa = parseFloat(document.getElementById('b_cruise_aoa').value);
        payload.target_cl = parseFloat(document.getElementById('b_target_cl').value);
      } else if (currentOption === 'C') {
        payload.target_cl0 = document.getElementById('c_cl0').value;
        payload.target_cl_alpha = document.getElementById('c_cla').value;
        payload.target_alpha_max = document.getElementById('c_amax').value;
      }

      // Reset UI for Stage 1 ("Finding Baseline")
      document.getElementById('optimize_btn').disabled = true;
      document.getElementById('carousel_nav').classList.remove('active');
      document.getElementById('download_btn').classList.remove('active');
      document.getElementById('image_stage').classList.remove('blurred');
      document.getElementById('throbber_text').innerText = "Finding Baseline";
      document.getElementById('throbber_overlay').classList.add('active');

      fetch('/start_optimization', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(() => {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollOptimizationStatus, 400);
      });
    }

    function pollOptimizationStatus() {
      fetch('/status')
        .then(r => r.json())
        .then(data => {
          const overlay = document.getElementById('throbber_overlay');
          const stage = document.getElementById('image_stage');
          const img = document.getElementById('display_img');
          const text = document.getElementById('throbber_text');

          if (data.status === 'finding_baseline') {
            overlay.classList.add('active');
            stage.classList.remove('blurred');
            text.innerText = "Finding Baseline";
          }
          else if (data.status === 'baseline_found') {
            overlay.classList.remove('active');
            stage.classList.remove('blurred');
            if (data.baseline_img) {
              img.src = "data:image/png;base64," + data.baseline_img;
            }
          }
          else if (data.status === 'optimizing') {
            if (data.baseline_img) {
              img.src = "data:image/png;base64," + data.baseline_img;
            }
            stage.classList.add('blurred');
            text.innerText = "Optimizing airfoil, please wait";
            overlay.classList.add('active');
          }
          else if (data.status === 'done') {
            clearInterval(pollTimer);
            overlay.classList.remove('active');
            stage.classList.remove('blurred');
            document.getElementById('optimize_btn').disabled = false;

            carouselImages = data.final_images || [];
            if (carouselImages.length > 0) {
              currentSlide = 0;
              updateCarouselView();
              document.getElementById('carousel_nav').classList.add('active');
            }
            document.getElementById('download_btn').classList.add('active');
          }
          else if (data.status === 'error') {
            clearInterval(pollTimer);
            overlay.classList.remove('active');
            stage.classList.remove('blurred');
            document.getElementById('optimize_btn').disabled = false;
            alert("Optimization Error: " + data.error);
          }
        });
    }

    // --- 6. Swipe & Button Carousel Navigation ---
    function updateCarouselView() {
      if (!carouselImages.length) return;
      document.getElementById('display_img').src = "data:image/png;base64," + carouselImages[currentSlide];
      document.getElementById('slide_caption').innerText = slideTitles[currentSlide] || `Slide ${currentSlide + 1}`;
    }

    function changeSlide(dir) {
      if (!carouselImages.length) return;
      currentSlide = (currentSlide + dir + carouselImages.length) % carouselImages.length;
      updateCarouselView();
    }

    // Support mouse/touch drag swiping on the right viewer card
    let dragStartX = null;
    const swipeArea = document.getElementById('swipe_area');
    swipeArea.addEventListener('pointerdown', e => { dragStartX = e.clientX; });
    swipeArea.addEventListener('pointerup', e => {
      if (dragStartX === null || !carouselImages.length) return;
      const diff = e.clientX - dragStartX;
      if (Math.abs(diff) > 40) {
        changeSlide(diff < 0 ? 1 : -1);
      }
      dragStartX = null;
    });
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, generic_img=job_state['generic_img'])


@app.route('/start_optimization', methods=['POST'])
def start_optimization():
    data = request.get_json()

    def parse_opt_float(val):
        if val is None or str(val).strip() == '':
            return None
        return float(val)

    req = {
        'Re': float(data.get('Re', 6000000.0)),
        'min_tc': float(data.get('min_tc', 0.08)),
        'option': data.get('option', 'A'),
        'constrain_cm': bool(data.get('constrain_cm', False)),
        'target_cm0': parse_opt_float(data.get('target_cm0')),
        'max_abs_cm': parse_opt_float(data.get('max_abs_cm')),
        'stall_type': data.get('stall_type')
    }

    if req['option'] == 'A':
        req['cruise_aoa'] = float(data.get('cruise_aoa', 5.0))
        req['target_cl'] = float(data.get('target_cl', 1.1))
        req['flatten_bucket'] = bool(data.get('flatten_bucket', False))
        req['bucket_half_width'] = float(data.get('bucket_half_width', 2.0))
    elif req['option'] == 'B':
        req['alpha_min'] = float(data.get('alpha_min', 2.0))
        req['alpha_max'] = float(data.get('alpha_max', 10.0))
        req['cruise_aoa'] = float(data.get('cruise_aoa', 5.0))
        req['target_cl'] = float(data.get('target_cl', 1.05))
    elif req['option'] == 'C':
        req['target_cl0'] = parse_opt_float(data.get('target_cl0'))
        req['target_cl_alpha'] = parse_opt_float(data.get('target_cl_alpha'))
        req['target_alpha_max'] = parse_opt_float(data.get('target_alpha_max'))

    thread = threading.Thread(target=background_optimization_worker, args=(req,), daemon=True)
    thread.start()
    return jsonify({'started': True})


@app.route('/status')
def get_status():
    return jsonify({
        'status': job_state['status'],
        'baseline_img': job_state['baseline_img'],
        'baseline_id': job_state['baseline_id'],
        'final_images': job_state['final_images'] if job_state['status'] == 'done' else [],
        'error': job_state['error']
    })


@app.route('/download_report')
def download_report():
    if job_state['html_path'] and os.path.exists(job_state['html_path']):
        return send_file(
            job_state['html_path'],
            as_attachment=True,
            download_name=os.path.basename(job_state['html_path'])
        )
    return "Report not available yet.", 404


if __name__ == '__main__':
    port = 5000
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False)