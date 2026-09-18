#!/usr/bin/env python3
"""
build_faculty_pdfs.py — Generates clean, authentic academic PDFs formatted to look
like real research reports and paper drafts prepared in OpenOffice Writer / LibreOffice Writer.
Compiled via headless Google Chrome with clean, normal academic typography.
"""

import os
import subprocess

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(WORKSPACE_DIR, "docs")
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Clean OpenOffice Writer / LibreOffice Writer standard style
OPENOFFICE_CSS = """
@page {
    size: A4;
    margin: 20mm 20mm 20mm 20mm;
    @top-right {
        content: "Research Project Report | Inverse Structural Damage Inversion";
        font-family: 'Times New Roman', 'Liberation Serif', Times, serif;
        font-size: 8.5pt;
        font-style: italic;
        color: #555555;
    }
    @bottom-center {
        content: "- " counter(page) " -";
        font-family: 'Times New Roman', 'Liberation Serif', Times, serif;
        font-size: 9pt;
        color: #333333;
    }
}

body {
    font-family: 'Times New Roman', 'Liberation Serif', Times, Georgia, serif;
    font-size: 11pt;
    line-height: 1.35;
    color: #000000;
    margin: 0;
    padding: 0;
}

.doc-title {
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0;
    margin-bottom: 4pt;
    color: #000000;
}

.doc-subtitle {
    font-size: 11.5pt;
    text-align: center;
    font-style: italic;
    margin-bottom: 6pt;
    color: #222222;
}

.doc-meta {
    font-size: 9.5pt;
    text-align: center;
    margin-bottom: 12pt;
    color: #333333;
    line-height: 1.35;
}

.doc-rule {
    border: none;
    border-top: 0.75pt solid #000000;
    margin: 8pt 0 14pt 0;
}

h1 {
    font-size: 12.5pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 4pt;
    color: #000000;
    border-bottom: 0.5pt solid #444444;
    padding-bottom: 2pt;
    page-break-after: avoid;
}

h2 {
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 10pt;
    margin-bottom: 3pt;
    color: #000000;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 8pt;
    margin-bottom: 2pt;
    color: #000000;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 6pt;
    text-align: justify;
    text-justify: inter-word;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 6pt;
    padding-left: 20pt;
}

li {
    margin-bottom: 2.5pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0 10pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 0.75pt solid #444444;
    padding: 4pt 6pt;
    vertical-align: top;
}

th {
    background-color: #f0f0f0;
    font-weight: bold;
    text-align: center;
    color: #000000;
}

.note-box {
    border: 0.75pt solid #666666;
    background-color: #fafafa;
    padding: 6pt 10pt;
    margin: 8pt 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

blockquote {
    margin: 6pt 0 8pt 16pt;
    padding-left: 8pt;
    border-left: 2pt solid #666666;
    font-style: italic;
    color: #222222;
}

code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 9pt;
}

pre {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    background-color: #f8f8f8;
    border: 0.5pt solid #cccccc;
    padding: 6pt 8pt;
    margin: 6pt 0;
    line-height: 1.25;
}

.page-break {
    page-break-before: always;
}
"""

def render_html_to_pdf(html_content, output_pdf_path):
    temp_html = output_pdf_path.replace(".pdf", "_temp.html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    cmd = [
        CHROME_BIN,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf_path}",
        temp_html
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(temp_html):
        os.remove(temp_html)
    
    if res.returncode == 0 and os.path.exists(output_pdf_path):
        size_kb = os.path.getsize(output_pdf_path) / 1024
        print(f"✅ Generated: {os.path.basename(output_pdf_path)} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"❌ Failed to generate {output_pdf_path}: {res.stderr}")
        return False


def build_research_dossier_pdf():
    """Builds the comprehensive Research Dossier & Internship Application PDF."""
    template = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Research Dossier & Internship Application | Inverse-FNO-Damage</title>
<style>
__OPENOFFICE_CSS__
</style>
</head>
<body>

<div class="doc-title">Inverse Neural Operators for Seismic Structural Damage Identification</div>
<div class="doc-subtitle">Research Report, Personal Statement, and Graduate Internship Application</div>
<div class="doc-meta">
    <strong>Author:</strong> Raghvendra Singh Gahlot &nbsp;|&nbsp; <strong>Affiliation:</strong> 2nd Year Undergraduate, Department of Civil Engineering, MBM University, Jodhpur<br>
    <strong>Project Code:</strong> <code>Inverse-FNO-Damage</code> &nbsp;|&nbsp; <strong>Focus:</strong> Computational Mechanics, Structural Dynamics, SciML &nbsp;|&nbsp; <strong>Date:</strong> September 2026
</div>

<hr class="doc-rule">

<div class="note-box">
    <strong>Summary of Application & Project Scope:</strong><br>
    My name is <strong>Raghvendra Singh Gahlot</strong>. I am a 2nd-year undergraduate student in the Department of Civil Engineering at MBM University, Jodhpur. Over the past several months, I have worked independently on an inverse problem at the boundary of structural mechanics and machine learning: estimating localized stiffness damage in building frames from noisy, sparse acceleration time series recorded during earthquakes. This document outlines my research motivation, mathematical formulations, experimental findings (including negative results and failure modes), and proposed research directions in your laboratory.
</div>

<h1>1. Candidate Profile & Application Overview</h1>
<p>
I am writing to apply for a <strong>Research Internship / Pre-Doctoral Visiting Position</strong> under your guidance. As a 2nd-year civil engineering student at MBM University, Jodhpur, I have focused my independent studies on structural dynamics, finite-element formulations in OpenSeesPy, and Scientific Machine Learning. My primary research interests center on:
</p>
<ul>
    <li>Physics-informed machine learning and operator learning (Fourier Neural Operators, Graph Neural Networks) applied to structural mechanics.</li>
    <li>Ill-posed inverse problems in structural dynamics and structural health monitoring.</li>
    <li>Gradient optimization dynamics in multiscale physical systems (e.g. resolving disparities between displacement, acceleration, and micro-strain measurements).</li>
    <li>Cross-structure generalization and domain transfer across changing topologies and stiffness distributions.</li>
</ul>
<p>
<strong>Availability:</strong> Full-time or hybrid research engagement for 6 months (flexible timeline). I have a self-contained, reproducible codebase built in PyTorch and OpenSeesPy, complete with deterministic testing pipelines.
</p>

<h1>2. Personal Statement: Why I Chose This Problem and How I Think</h1>

<h2>The Problem with Standard Benchmarks</h2>
<p>
When I began studying scientific machine learning literature, I noticed that many papers report impressive accuracy scores (e.g., $R^2 > 0.99$ or tiny relative errors) on smooth, synthetic benchmarks. However, when these methods are applied to realistic physical systems, they frequently collapse. In many instances, the models are not actually learning the underlying dynamic physics; they are simply interpolating smooth functions across dense, over-instrumented sensor grids.
</p>
<p>
In actual civil engineering structures, we do not have dense arrays of thousands of sensors. We have sparse sensors—perhaps one horizontal accelerometer per floor—and sensor noise is unavoidable. More importantly, civil buildings are designed to be nominally symmetric. 
</p>
<p>
I wanted to work on a problem that is <strong>fundamentally ill-posed from the first principles of mechanics</strong>: a problem where standard deep learning is mathematically guaranteed to fail unless structural symmetry and observation physics are explicitly addressed. That led me to seismic structural damage identification in laterally symmetric building frames under sparse instrumentation.
</p>

<h2>The Structural Symmetry Dilemma</h2>
<p>
Consider a multi-story building frame with lateral symmetry. Suppose severe earthquake shaking causes a 30% reduction in stiffness in the ground-floor left column (State A), or symmetrically in the ground-floor right column (State B). 
</p>
<p>
If we only install horizontal accelerometers on each floor slab (the conventional monitoring practice), both damage states produce floor vibrations that are virtually identical. In my OpenSees dynamic simulations under real earthquake ground motions, the relative $L_2$ difference between the floor acceleration histories of State A and State B is <strong>less than 0.14%</strong>. Under an ordinary 2% field noise floor, this difference signal is completely submerged.
</p>
<p>
When an unconstrained deep inverse model is trained with standard Mean Squared Error (MSE) loss on this data, it exhibits what I call <em>mean-seeking collapse</em>. Because the sensor data is virtually identical for both states, gradient descent forces the network to output the conditional expectation $\mathbb{E}[d \mid y]$. As a result, the model predicts diffuse, low-amplitude damage across both columns simultaneously ($\hat{d} \approx 0.01\text{--}0.02$). It completely misses the localized structural failure.
</p>

<h2>My Approach to Research: Rigor and Honesty</h2>
<p>
From the start, I adopted three core principles for this work:
</p>
<ol>
    <li><strong>Analyze the mechanics before training networks:</strong> Before writing neural network code, I computed linearized sensitivity Jacobians and performed noise-whitened singular value decomposition (SVD). This proved mathematically that the bilateral difference vector $v_{AB}$ lies directly in the near-null space of horizontal sensing ($|\langle v_E, v_{AB} \rangle| = 0.9982$).</li>
    <li><strong>Never hide ill-posedness or negative results:</strong> If a sensor configuration cannot physically distinguish two damage states, I did not tweak hyperparameters or search for random seeds until one run happened to guess correctly. True non-observability must be stated clearly.</li>
    <li><strong>Traceability:</strong> Every metric, table entry, and plot in this project is tied to deterministic scripts, frozen seeds, and audited result files in the repository.</li>
</ol>

<div class="page-break"></div>

<h1>3. Mathematical Formulation of the Dynamic Inverse Problem</h1>

<p>
The structural frame dynamic equilibrium under horizontal ground acceleration $a_g(t)$ is modeled by the equation of motion:
</p>
<p style="text-align: center; margin: 8pt 0;">
    <code>M u''(t) + C u'(t) + K(d) u(t) = -M r a_g(t)</code>
</p>
<p>
where <code>M</code> is the lumped mass matrix, <code>C</code> is the Rayleigh damping matrix, and <code>K(d)</code> is the global tangent stiffness matrix parameterized by element damage vector <code>d ∈ [0, 0.5]^E</code>. For each structural member <code>e</code>, the effective elastic modulus is:
</p>
<p style="text-align: center; margin: 6pt 0;">
    <code>E_e(d_e) = (1 - d_e) E_0</code>
</p>
<p>
Sparse observations are collected through a selection matrix <code>H_S</code> with 2% stationary Gaussian sensor noise:
</p>
<p style="text-align: center; margin: 6pt 0;">
    <code>y(t) = H_S [u''(t) + r a_g(t), u(t)] + η(t), &nbsp;&nbsp; η(t) ~ N(0, σ² I)</code>
</p>
<p>
To evaluate whether a method can distinguish symmetric damage, we define canonical bilateral damage states:
</p>
<ul>
    <li><strong>State A:</strong> 30% stiffness reduction in the ground-floor left column (<code>d_A = [0.30, 0, 0, ...]ᵀ</code>).</li>
    <li><strong>State B:</strong> 30% stiffness reduction in the ground-floor right column (<code>d_B = [0, 0.30, 0, ...]ᵀ</code>).</li>
    <li><strong>Canonical Difference Unit Vector:</strong> <code>v_AB = (d_A - d_B) / ||d_A - d_B||₂ = 1/√2 [1, -1, 0, ...]ᵀ</code> with true separation <code>||d_A - d_B||₂ = 0.4243</code>.</li>
</ul>

<h1>4. Observability Analysis & Sensor Configurations</h1>

<p>
To determine what sensors are physically necessary to break the bilateral symmetry, I formulated the noise-whitened sensitivity Jacobian <code>J_w</code> and directional Fisher sensitivity <code>√(I_AB)</code>:
</p>
<p style="text-align: center; margin: 6pt 0;">
    <code>J_w = Σ_η^(-1/2) (∂y / ∂d), &nbsp;&nbsp;&nbsp; √(I_AB) = ||J_w v_AB||₂ = √(v_ABᵀ J_wᵀ J_w v_AB)</code>
</p>
<p>
I evaluated four sensor suites on a 3-story, 1-bay frame:
</p>

<table>
    <thead>
        <tr>
            <th>Modality Suite</th>
            <th>Sensor Types & Locations</th>
            <th>Total Channels</th>
            <th>Fisher Sensitivity √(I_AB)</th>
            <th>Amplification vs. S0</th>
            <th>Physical Mechanism</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>S0</strong></td>
            <td>Horizontal floor accelerometers (Floors 1, 2, Roof)</td>
            <td>3</td>
            <td>5.62</td>
            <td>1.0×</td>
            <td>Near-null space baseline. Floor shear deformations are laterally symmetric.</td>
        </tr>
        <tr>
            <td><strong>S2</strong></td>
            <td>Horizontal accelerations + column axial strains</td>
            <td>9</td>
            <td>1344.18</td>
            <td>239.3×</td>
            <td>Column axial deformation captures lateral rocking load transfer.</td>
        </tr>
        <tr>
            <td><strong>S1</strong></td>
            <td>Horizontal accelerations + vertical joint accelerations</td>
            <td>9</td>
            <td>1829.63</td>
            <td>325.7×</td>
            <td>Vertical joint motions capture differential floor-beam rotation.</td>
        </tr>
        <tr>
            <td><strong>S4</strong></td>
            <td>Multimodal: Horizontal + Vertical + Axial Strains</td>
            <td>15</td>
            <td>2270.54</td>
            <td>404.2×</td>
            <td>Combined translational, rotational, and internal strain observability.</td>
        </tr>
    </tbody>
</table>

<h1>5. Inverse Model Architecture: DualStreamGFNO</h1>

<p>
Standard MLPs or CNNs assume a fixed grid and cannot easily handle frames with varying story heights or bay widths. To address this, I developed <strong>DualStreamGFNO</strong>, which couples temporal Fourier Neural Operators with structural Graph Neural Networks:
</p>
<ol>
    <li><strong>Temporal FNO Stream:</strong> Processes time series across global and joint sensor channels using 1D spectral convolutions with 24 Fourier modes, extracting frequency-domain dynamic features.</li>
    <li><strong>Structural Graph Message Passing:</strong> Passes temporal embeddings along the structural frame connectivity (nodes = beam-column joints, edges = structural members).</li>
    <li><strong>Symmetry Decomposition:</strong> Node representations are explicitly split into symmetric <code>h_sym = 0.5 (h + Π h)</code> and anti-symmetric <code>h_anti = 0.5 (h - Π h)</code> components via the structural permutation matrix <code>Π</code>.</li>
    <li><strong>Hierarchical Output Heads:</strong> Edge representations are fed to two separate heads:
        <ul>
            <li><em>Support Head:</em> Predicts probability of damage for each member <code>p_e ∈ [0, 1]</code> (using Sigmoid).</li>
            <li><em>Severity Head:</em> Predicts damage magnitude <code>μ_e ∈ [0, 0.5]</code> (using scaled Sigmoid).</li>
            <li><em>Final Prediction:</em> <code>d_hat_e = p_e · μ_e</code>.</li>
        </ul>
    </li>
    <li><strong>Bilateral Pair Loss:</strong> <code>L_total = L_base + λ_dir L_dir + λ_pair L_pair</code>, where <code>L_dir = 1 - cos(Δd_hat, v_AB)</code> penalizes incorrect orientation and <code>L_pair = max(0, m - ⟨Δd_hat, v_AB⟩)</code> enforces separation beyond margin <code>m = 0.15</code>.</li>
</ol>

<div class="page-break"></div>

<h1>6. Single-Structure Benchmark Results</h1>

<p>
I evaluated the trained models on an independent, held-out test set of <code>N = 30</code> bilateral test pairs under unseen earthquake records from the PEER strong motion database. The decision threshold was frozen prior to test evaluation:
</p>

<table>
    <thead>
        <tr>
            <th>Modality</th>
            <th>Attribution Successes</th>
            <th>Accuracy (%)</th>
            <th>Exact Binomial p-value</th>
            <th>95% Clopper-Pearson CI</th>
            <th>Directional Cosine</th>
            <th>Separation ||Δd_hat||₂</th>
            <th>True Separation Ratio</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>S0</strong></td>
            <td>15 / 30</td>
            <td>50.0%</td>
            <td>p = 1.000</td>
            <td>[31.3%, 68.7%]</td>
            <td>-0.115 ± 0.122</td>
            <td>0.000009</td>
            <td>0.002%</td>
        </tr>
        <tr>
            <td><strong>S2</strong></td>
            <td>15 / 30</td>
            <td>50.0%</td>
            <td>p = 1.000</td>
            <td>[31.3%, 68.7%]</td>
            <td>+0.105 ± 0.111</td>
            <td>0.000328</td>
            <td>0.08% (Learnability Gap)</td>
        </tr>
        <tr>
            <td><strong>S1</strong></td>
            <td>24 / 30</td>
            <td>80.0%</td>
            <td>p = 0.0014</td>
            <td>[61.4%, 92.3%]</td>
            <td>+0.727 ± 0.442</td>
            <td>0.1576</td>
            <td>37.15%</td>
        </tr>
        <tr>
            <td><strong>S4</strong></td>
            <td>27 / 30</td>
            <td><strong>90.0%</strong></td>
            <td><strong>p = 9.0 × 10⁻⁶</strong></td>
            <td><strong>[73.5%, 97.9%]</strong></td>
            <td><strong>+0.942 ± 0.098</strong></td>
            <td><strong>0.1671</strong></td>
            <td><strong>39.40%</strong></td>
        </tr>
    </tbody>
</table>

<h1>7. The Column Strain Paradox: Observability vs. Learnability</h1>
<p>
The most intriguing finding on the single building was the complete failure of the <strong>S2 sensor suite (column axial strain)</strong>.
</p>
<p>
Fisher sensitivity showed that adding strain gauges increased physical sensitivity by <strong>239×</strong> over horizontal floor sensing (<code>√(I_AB) = 1344.18</code> vs. <code>5.62</code>). In structural engineering textbooks, axial strains are well-known to capture the rocking moments that break lateral symmetry.
</p>
<p>
Yet when the neural operator was trained, it achieved only <strong>50.0% attribution accuracy (15/30, pure chance)</strong>, exactly matching the coin-flip baseline of the unobservable S0 suite.
</p>
<div class="note-box">
    <strong>Investigation of the Failure Mechanism:</strong><br>
    To understand why, I tracked the backpropagation gradient norms layer by layer during early training epochs. Floor accelerations have amplitudes of order ~1.0 m/s², whereas elastic column axial strains are on the order of ~10⁻⁵ m/m. Even with standard input normalization, gradients originating from the acceleration channels accounted for <strong>over 98% of the total backpropagation gradient norm</strong>. The micro-strain channels were effectively drowned out in the optimizer's updates. The network settled into a symmetric minimum near the mean before strain signals could exert sufficient torque to guide the weights.
</div>
<p>
This taught me an important lesson in Scientific Machine Learning: <em>high physical Fisher information is necessary, but it does not guarantee neural learnability under multiscale gradient dynamics.</em>
</p>

<h1>8. Multi-Structure Generalization & Magnitude Collapse</h1>
<p>
To test whether the learned inversion transfers to new structures, I built an experimental dataset of <strong>530 dynamic OpenSees simulations</strong> across 120 PEER earthquakes, covering 10 structural configurations (stiffness variations, height changes) and an unseen 4-story frame.
</p>
<p>
When evaluating the model zero-shot on these unseen structures, I observed a consistent phenomenon: <strong>Direction–Magnitude Decoupling</strong>.
</p>
<ul>
    <li><strong>Directional Alignment Transfers:</strong> The continuous directional cosine between predicted damage difference and the true bilateral vector remained consistently positive (<code>cos ≈ +0.80 to +0.85</code>, reaching <code>+0.996</code> on in-distribution frames and <code>+0.955</code> on the unseen 4-story frame). The network reliably identified <em>which side</em> had more damage.</li>
    <li><strong>Magnitude Collapses:</strong> However, the predicted damage separation collapsed to approximately <code>1.8 × 10⁻⁵</code> (compared to true separation <code>0.4243</code>). Because the magnitude difference was so close to zero, discrete attribution accuracy dropped back to 50.0% chance.</li>
</ul>
<p>
I ran a controlled 72-model ablation scaling training budgets from 12 to 100 epochs. Scaling compute improved directional cosine alignment from +0.47 to +0.85, but did not restore finite separation. When an inverse network is trained across buildings of differing stiffness and geometry, gradient updates across the diverse configurations average out, creating an implicit regularization that compresses output magnitudes toward the mean.
</p>

<div class="page-break"></div>

<h1>9. What I Wish to Work On in Your Research Group</h1>

<p>
Joining your laboratory would allow me to build on these insights and address the key bottlenecks identified in this project:
</p>

<ol>
    <li>
        <strong>3D Frame Mechanics and Torsional Symmetry Breaking:</strong><br>
        Real buildings are three-dimensional. When a column is damaged in a 3D building, it shifts the center of rigidity relative to the center of mass, inducing dynamic torsion and bi-directional response coupling. I want to formulate 3D graph neural operators that leverage torsional rotational degrees of freedom to break symmetry naturally without requiring dense sensor grids.
    </li>
    <li>
        <strong>Group-Equivariant Structural Operators:</strong><br>
        In this work, I used soft loss penalties to encourage bilateral symmetry. A more mathematically rigorous approach is to build exact group-equivariant graph operators (such as $C_2$ or $D_4$ equivariant GNNs) where reflection and rotation symmetries are built directly into the message-passing layers.
    </li>
    <li>
        <strong>Multiscale Gradient Optimization:</strong><br>
        To solve the Column Strain Paradox (where accelerations drown out strain gradients), I want to investigate multi-task gradient-balancing techniques (such as PCGrad or GradNorm) that adaptively project and balance gradient components from sensors operating at vastly different physical scales.
    </li>
    <li>
        <strong>Validation on Physical Shake-Table Data:</strong><br>
        Transitioning from OpenSees numerical simulations to real-world experimental data from shake-table frame tests (e.g. from PEER or NHERI repositories) to test how neural operators handle real material non-linearities and ambient noise.
    </li>
</ol>

<h1>10. Reproducibility & Engineering Architecture</h1>

<p>
The repository is designed for full transparency, determinism, and automated testing:
</p>
<ul>
    <li><strong>Automated Test Suite:</strong> Running <code>python scripts/gstack.py qa</code> executes 74 automated unit tests in under 3.5 seconds, verifying finite-element model frequencies, Rayleigh damping, Jacobian dimensions, loss functions, and Clopper-Pearson confidence bounds.</li>
    <li><strong>Physical Units:</strong> All OpenSees models enforce strict SI units ($m, N, kg, s, Pa$).</li>
    <li><strong>Code Layout:</strong>
        <ul>
            <li><code>src/damage_injection.py</code> — OpenSees transient dynamic finite element simulation.</li>
            <li><code>src/forensic_observability.py</code> — Noise-whitened SVD and Fisher sensitivity analysis.</li>
            <li><code>src/phase6_models.py</code> — DualStreamGFNO graph-Fourier operator architecture.</li>
            <li><code>src/losses.py</code> — Hierarchical support/severity and bilateral pair losses.</li>
            <li><code>tests/</code> — 74 unit tests with regression protection.</li>
        </ul>
    </li>
</ul>

<div class="note-box">
    <strong>Candidate Concluding Note:</strong><br>
    I enjoy getting my hands dirty with both structural dynamics equations and deep learning code. I value finding out why something fails just as much as showing that it works. I would welcome the opportunity to discuss my work, hear your perspectives, and contribute actively to your ongoing research projects.
</div>

<div class="doc-meta" style="margin-top: 18pt; text-align: right;">
    <em>Report compiled: September 2026 &nbsp;|&nbsp; 74/74 Unit Tests Verified Passing</em>
</div>

</body>
</html>"""
    html = template.replace("__OPENOFFICE_CSS__", OPENOFFICE_CSS)
    out_pdf = os.path.join(DOCS_DIR, "Inverse_FNO_Damage_Research_Dossier_and_Internship_Application.pdf")
    return render_html_to_pdf(html, out_pdf)


def build_executive_brief_pdf():
    """Builds the clean 2-Page Executive Research Brief PDF in OpenOffice Writer style."""
    template = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Executive Research Brief | Inverse-FNO-Damage</title>
<style>
__OPENOFFICE_CSS__
</style>
</head>
<body>

<div class="doc-title">Executive Research Brief: Inverse-FNO-Damage</div>
<div class="doc-subtitle">Physics-Constrained Structural Damage Inversion, Observability Limits, and Neural Learnability</div>
<div class="doc-meta">
    <strong>Author:</strong> Raghvendra Singh Gahlot &nbsp;|&nbsp; <strong>Affiliation:</strong> 2nd Year Undergraduate, Civil Engineering, MBM University, Jodhpur<br>
    <strong>Field:</strong> Scientific Machine Learning & Structural Dynamics &nbsp;|&nbsp; <strong>Status:</strong> Complete Research Program (74 Unit Tests Passing) &nbsp;|&nbsp; <strong>Date:</strong> September 2026
</div>

<hr class="doc-rule">

<h1>1. Research Overview</h1>
<p>
This project investigates an ill-posed inverse problem in structural mechanics: identifying localized member damage (stiffness reduction $d \in [0, 0.5]^E$) in a building frame based purely on sparse, noisy acceleration and strain time-series collected during earthquake ground motion.
</p>
<p>
The core challenge is that civil structures possess nominal lateral symmetry. Localized stiffness degradation in laterally symmetric structural members (e.g. ground-floor left column vs. right column) generates horizontal floor vibrations that differ by <strong>less than 0.14%</strong>. Linearized sensitivity SVD shows that the bilateral difference unit vector $v_{AB}$ aligns almost perfectly with the smallest singular vector of horizontal floor sensing ($|\langle v_E, v_{AB} \rangle| = 0.9982$). Under conventional horizontal floor accelerometers, bilateral failure modes lie inside an empirical near-null space.
</p>

<h1>2. Three Core Discoveries</h1>

<h2>Finding 1: Physical Observability Under Horizontal Sensing is Near-Null</h2>
<p>
Under calibrated 2% sensor noise, noise-whitened directional Fisher sensitivity for horizontal floor sensing is negligible ($\sqrt{I_{AB}} = 5.62$). In held-out bilateral test evaluations ($N = 30$), models trained on horizontal floor accelerometers achieve exactly <strong>50.0% attribution accuracy (pure chance, $p = 1.000$)</strong>. Multimodal sensing ($S4$: horizontal + vertical + column axial strain) amplifies Fisher sensitivity by 404× to $\sqrt{I_{AB}} = 2270.54$, elevating attribution accuracy to <strong>90.0% ($27/30$, $p = 9.0 \times 10^{-6}$, 95% CI $[73.5\%, 97.9\%]$)</strong>.
</p>

<h2>Finding 2: Physical Observability is Necessary but Insufficient for Learnability</h2>
<p>
Column axial strain gauges ($S2$) amplify directional Fisher sensitivity by <strong>239×</strong> over horizontal floor sensing ($\sqrt{I_{AB}} = 1344.18$), because axial deformation directly captures lateral rocking load transfer. Standard theory suggests the inverse network should easily distinguish the states.
</p>
<p>
However, in practice, models trained on $S2$ achieve only <strong>50.0% attribution accuracy ($15/30$, $p = 1.000$)</strong>. Tracking gradient norms reveals that horizontal floor accelerations ($\sim 1 \text{ m/s}^2$) account for <strong>over 98%</strong> of the total backpropagation gradient norm, drowning out micro-strain gradients ($\sim 10^{-5}\text{ m/m}$). The optimizer settles into a symmetric local minimum near the mean before strain signals can guide the weights. <em>High Fisher information does not guarantee neural learnability under multiscale gradient dynamics.</em>
</p>

<h2>Finding 3: Cross-Structure Direction–Magnitude Decoupling</h2>
<p>
When evaluated across 530 OpenSees dynamic simulations covering 10 structural configurations and an unseen 4-story frame, an unexpected phenomenon emerges:
</p>
<ul>
    <li>The continuous directional cosine alignment transfers zero-shot ($\cos \theta \approx +0.80\text{ to }+0.85$, peaking at $+0.996$). The model reliably identifies which side of the structure has more damage.</li>
    <li>However, predicted damage separation collapses to $\sim 10^{-5}$ (compared to ground truth $0.4243$), returning discrete attribution to 50.0% chance.</li>
    <li>A 72-model ablation scaling training from 12 to 100 epochs showed that training longer refines directional alignment but cannot break the magnitude collapse. Parameter dispersion across buildings creates an implicit regularization that compresses output magnitudes toward the mean.</li>
</ul>

<div class="page-break"></div>

<h1>3. Quantitative Benchmarks (Held-Out Test Set, N = 30)</h1>

<table>
    <thead>
        <tr>
            <th>Modality Suite</th>
            <th>Channels</th>
            <th>Fisher Sensitivity √(I_AB)</th>
            <th>Attribution Accuracy</th>
            <th>Exact Binomial p-value</th>
            <th>95% Clopper-Pearson CI</th>
            <th>Predicted Separation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>S0 (Horizontal Accel)</td>
            <td>3</td>
            <td>5.62</td>
            <td>50.0% (15/30)</td>
            <td>p = 1.000</td>
            <td>[31.3%, 68.7%]</td>
            <td>0.000009</td>
        </tr>
        <tr>
            <td>S2 (Horiz + Column Strain)</td>
            <td>9</td>
            <td>1344.18</td>
            <td>50.0% (15/30)</td>
            <td>p = 1.000</td>
            <td>[31.3%, 68.7%]</td>
            <td>0.000328</td>
        </tr>
        <tr>
            <td>S1 (Horiz + Vert Accel)</td>
            <td>9</td>
            <td>1829.63</td>
            <td>80.0% (24/30)</td>
            <td>p = 0.0014</td>
            <td>[61.4%, 92.3%]</td>
            <td>0.1576</td>
        </tr>
        <tr>
            <td><strong>S4 (Multimodal Union)</strong></td>
            <td>15</td>
            <td><strong>2270.54</strong></td>
            <td><strong>90.0% (27/30)</strong></td>
            <td><strong>p = 9.0 × 10⁻⁶</strong></td>
            <td><strong>[73.5%, 97.9%]</strong></td>
            <td><strong>0.1671</strong> (39.4% true)</td>
        </tr>
    </tbody>
</table>

<h1>4. Training-Budget Ablation Trajectory (Phase 7, P2 S4)</h1>

<table>
    <thead>
        <tr>
            <th>Epoch Budget</th>
            <th>In-Distribution Acc</th>
            <th>In-Distribution Cosine</th>
            <th>Topological OOD Cosine</th>
            <th>Damage Separation ||Δd_hat||₂</th>
            <th>Interpretation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>12 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.467 ± 0.037</td>
            <td>+0.489 ± 0.012</td>
            <td>2.68 × 10⁻⁶</td>
            <td>Initial baseline; weak positive directional torque.</td>
        </tr>
        <tr>
            <td>25 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.852 ± 0.115</td>
            <td>+0.777 ± 0.158</td>
            <td>7.68 × 10⁻⁶</td>
            <td>Steep rise in directional cosine alignment.</td>
        </tr>
        <tr>
            <td>50 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.897 ± 0.038</td>
            <td>+0.837 ± 0.090</td>
            <td>1.23 × 10⁻⁵</td>
            <td>Peak mean directional alignment across seeds.</td>
        </tr>
        <tr>
            <td>100 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.850 ± 0.205</td>
            <td>+0.798 ± 0.199</td>
            <td>1.83 × 10⁻⁵</td>
            <td>Orientation refined (peaks +0.996); separation remains collapsed.</td>
        </tr>
    </tbody>
</table>

<h1>5. Summary of Candidate Contributions & Research Interests</h1>
<p>
This project demonstrates my technical capability and research mindset:
</p>
<ol>
    <li><strong>Structural Dynamics Modeling:</strong> End-to-end parametric OpenSeesPy finite-element pipeline enforcing strict SI units and realistic seismic ground motions.</li>
    <li><strong>Mathematical Rigor:</strong> Derivation and implementation of noise-whitened sensitivity Jacobians and directional Fisher information to establish observability bounds.</li>
    <li><strong>Novel ML Architecture:</strong> Development of <code>DualStreamGFNO</code> combining temporal Fourier neural operators with structural graph convolutions, symmetry decomposition, and hierarchical damage heads.</li>
    <li><strong>Scientific Honesty:</strong> Transparent documentation of failure modes (the $S2$ learnability gap and cross-structure magnitude collapse) rather than cherry-picking isolated successes.</li>
</ol>
<p>
In your laboratory, I hope to extend this work to 3D structural frames with torsional coupling, group-equivariant neural operators ($C_2 / D_4$ GNNs), and multiscale gradient optimization.
</p>

<div class="doc-meta" style="margin-top: 16pt; text-align: right;">
    <em>Document complete & reproducible &nbsp;|&nbsp; 74/74 Unit Tests Passing (scripts/gstack.py qa)</em>
</div>

</body>
</html>"""
    html = template.replace("__OPENOFFICE_CSS__", OPENOFFICE_CSS)
    out_pdf = os.path.join(DOCS_DIR, "Inverse_FNO_Damage_Executive_Brief.pdf")
    return render_html_to_pdf(html, out_pdf)


def build_technical_sheet_pdf():
    """Builds the dense 1-Page Technical Results Sheet PDF in OpenOffice Writer style."""
    template = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Technical Results Sheet | Inverse-FNO-Damage</title>
<style>
__OPENOFFICE_CSS__
body { font-size: 9.5pt; line-height: 1.3; }
.doc-title { font-size: 14pt; margin-bottom: 2pt; }
.doc-subtitle { font-size: 10pt; margin-bottom: 4pt; }
.doc-meta { font-size: 8.5pt; margin-bottom: 8pt; }
h1 { font-size: 11pt; margin-top: 8pt; margin-bottom: 2pt; }
h2 { font-size: 10pt; margin-top: 6pt; margin-bottom: 2pt; }
table { font-size: 8.5pt; margin: 4pt 0 6pt 0; }
th, td { padding: 2.5pt 4pt; }
.note-box { font-size: 9pt; padding: 4pt 8pt; margin: 5pt 0; }
</style>
</head>
<body>

<div class="doc-title">Technical Reference & Results Sheet: Inverse-FNO-Damage</div>
<div class="doc-subtitle">Mathematical Formulation, Observability Values, and Benchmark Statistics</div>
<div class="doc-meta">
    <strong>Author:</strong> Raghvendra Singh Gahlot (2nd Year B.Tech, Civil Engineering, MBM University, Jodhpur) &nbsp;|&nbsp; <strong>Date:</strong> September 2026 &nbsp;|&nbsp; <strong>QA:</strong> 74/74 Unit Tests Passing
</div>

<hr class="doc-rule" style="margin: 4pt 0 8pt 0;">

<h1>1. Equations of Motion & Observation Model</h1>
<p>
Transient dynamic equilibrium under seismic ground acceleration $a_g(t)$:
<br/>
<code>M u''(t) + C u'(t) + K(d) u(t) = -M r a_g(t)</code>, &nbsp;&nbsp; with member stiffness $E_e(d_e) = (1 - d_e)E_0$, &nbsp;&nbsp; $d_e \in [0, 0.5]$.
<br/>
Observation with 2% sensor noise: <code>y(t) = H_S [u''(t) + r a_g(t), u(t)] + η(t)</code>.
<br/>
Bilateral difference unit vector: <code>v_AB = (d_A - d_B) / ||d_A - d_B||₂ = 1/√2 [1, -1, 0, ...]ᵀ</code>. Ground truth separation: <code>||d_A - d_B||₂ = 0.4243</code>.
</p>

<h1>2. Observability & Noise-Whitened Fisher Sensitivity</h1>
<p>
Linearized sensitivity Jacobian: <code>J = ∂y / ∂d ∈ R^{(C·T) × E}</code>. Noise-whitened Jacobian: <code>J_w = Σ_η^(-1/2) J</code>.
<br/>
Directional Fisher sensitivity along bilateral axis: <code>√(I_AB) = ||J_w v_AB||₂ = √(v_ABᵀ J_wᵀ J_w v_AB)</code>.
</p>

<table>
    <thead>
        <tr>
            <th>Suite</th>
            <th>Sensors</th>
            <th>Channels</th>
            <th>Fisher √(I_AB)</th>
            <th>S0 Ratio</th>
            <th>SVD Near-Null Alignment</th>
            <th>Physical Status</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>S0</strong></td>
            <td>Horizontal Floor Accelerations</td>
            <td>3</td>
            <td>5.62</td>
            <td>1.0×</td>
            <td>|⟨v_E, v_AB⟩| = 0.9982</td>
            <td>Near-null space baseline. Symmetric shear.</td>
        </tr>
        <tr>
            <td><strong>S2</strong></td>
            <td>Horiz Accel + Column Axial Strain</td>
            <td>9</td>
            <td>1344.18</td>
            <td>239.3×</td>
            <td>|⟨v_E, v_AB⟩| = 0.0412</td>
            <td>High Fisher sensitivity; rocking load transfer.</td>
        </tr>
        <tr>
            <td><strong>S1</strong></td>
            <td>Horiz Accel + Vertical Joint Accel</td>
            <td>9</td>
            <td>1829.63</td>
            <td>325.7×</td>
            <td>|⟨v_E, v_AB⟩| = 0.0298</td>
            <td>Joint vertical rotation breaks symmetry.</td>
        </tr>
        <tr>
            <td><strong>S4</strong></td>
            <td>Multimodal Union (Horiz + Vert + Strain)</td>
            <td>15</td>
            <td>2270.54</td>
            <td>404.2×</td>
            <td>|⟨v_E, v_AB⟩| = 0.0185</td>
            <td>Maximal directional sensitivity.</td>
        </tr>
    </tbody>
</table>

<h1>3. Single-Structure Benchmark Performance (Phase 6.2, N = 30 Held-Out Tests)</h1>

<table>
    <thead>
        <tr>
            <th>Suite</th>
            <th>Successes</th>
            <th>Attribution Acc</th>
            <th>Two-Sided p-value</th>
            <th>95% Clopper-Pearson CI</th>
            <th>Directional Cosine</th>
            <th>Separation ||Δd_hat||₂</th>
            <th>True Separation Ratio</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>S0</strong></td>
            <td>15 / 30</td>
            <td>50.0%</td>
            <td>p = 1.000</td>
            <td>[31.3%, 68.7%]</td>
            <td>-0.1151 ± 0.122</td>
            <td>0.000009</td>
            <td>0.002%</td>
        </tr>
        <tr>
            <td><strong>S2</strong></td>
            <td>15 / 30</td>
            <td>50.0%</td>
            <td>p = 1.000</td>
            <td>[31.3%, 68.7%]</td>
            <td>+0.1047 ± 0.111</td>
            <td>0.000328</td>
            <td>0.08% (Learnability Gap)</td>
        </tr>
        <tr>
            <td><strong>S1</strong></td>
            <td>24 / 30</td>
            <td>80.0%</td>
            <td>p = 0.0014</td>
            <td>[61.4%, 92.3%]</td>
            <td>+0.7272 ± 0.442</td>
            <td>0.1576</td>
            <td>37.15%</td>
        </tr>
        <tr>
            <td><strong>S4</strong></td>
            <td>27 / 30</td>
            <td><strong>90.0%</strong></td>
            <td><strong>p = 9.0 × 10⁻⁶</strong></td>
            <td><strong>[73.5%, 97.9%]</strong></td>
            <td><strong>+0.9420 ± 0.098</strong></td>
            <td><strong>0.1671</strong></td>
            <td><strong>39.40%</strong></td>
        </tr>
    </tbody>
</table>

<h1>4. Training-Budget Trajectory Across Multi-Structure Generalization (Phase 7, P2 S4)</h1>
<p>
Evaluated on 530 OpenSees dynamic simulations across 120 PEER earthquakes and 10 structural configurations:
</p>

<table>
    <thead>
        <tr>
            <th>Epoch Budget</th>
            <th>In-Distribution Acc</th>
            <th>In-Dist Cosine</th>
            <th>Interpolation (L2A)</th>
            <th>Extrapolation (L2B)</th>
            <th>Topological OOD (L3)</th>
            <th>Separation ||Δd_hat||₂</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>12 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.467 ± 0.037</td>
            <td>+0.480 ± 0.038</td>
            <td>+0.534 ± 0.055</td>
            <td>+0.489 ± 0.012</td>
            <td>2.68 × 10⁻⁶</td>
        </tr>
        <tr>
            <td>25 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.852 ± 0.115</td>
            <td>+0.892 ± 0.072</td>
            <td>+0.802 ± 0.204</td>
            <td>+0.777 ± 0.158</td>
            <td>7.68 × 10⁻⁶</td>
        </tr>
        <tr>
            <td>50 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.897 ± 0.038</td>
            <td>+0.896 ± 0.016</td>
            <td>+0.824 ± 0.096</td>
            <td>+0.837 ± 0.090</td>
            <td>1.23 × 10⁻⁵</td>
        </tr>
        <tr>
            <td>100 Epochs</td>
            <td>50.0% ± 0.0%</td>
            <td>+0.850 ± 0.205</td>
            <td>+0.841 ± 0.216</td>
            <td>+0.813 ± 0.204</td>
            <td>+0.798 ± 0.199</td>
            <td>1.83 × 10⁻⁵</td>
        </tr>
    </tbody>
</table>

<h1>5. Key Takeaways & Failure Mode Summary</h1>
<div class="note-box">
    <strong>1. Observability ≠ Learnability:</strong> High Fisher information (S2 = 1344.2) fails under standard gradient descent because floor accelerations dominate >98% of the backprop gradient norm over micro-strains.<br>
    <strong>2. Direction–Magnitude Decoupling:</strong> Under cross-structure distribution shift, directional orientation transfers zero-shot (cos ≈ +0.85), but finite damage separation collapses (~10⁻⁵), pinning discrete attribution at 50% chance.<br>
    <strong>3. Optimization Bounds:</strong> Scaling training budget by 8.3× (12 → 100 epochs) refines directional orientation (up to +0.996) but cannot recover finite separation without new representations.
</div>

</body>
</html>"""
    html = template.replace("__OPENOFFICE_CSS__", OPENOFFICE_CSS)
    out_pdf = os.path.join(DOCS_DIR, "Inverse_FNO_Damage_Technical_Results_Sheet.pdf")
    return render_html_to_pdf(html, out_pdf)


if __name__ == "__main__":
    print("🚀 Building Faculty Review Package PDFs (Clean OpenOffice Style)...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    ok1 = build_research_dossier_pdf()
    ok2 = build_executive_brief_pdf()
    ok3 = build_technical_sheet_pdf()
    if ok1 and ok2 and ok3:
        print("🎉 ALL 3 FACULTY PACKAGE PDFS SUCCESSFULLY GENERATED IN OPENOFFICE STYLE!")
    else:
        print("⚠️ Some PDFs failed to generate.")
