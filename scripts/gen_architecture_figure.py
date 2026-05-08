"""Generate EFDO architecture diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.6, 'EFDO: Sequential Eigenfunction Learning with Metric Adaptation',
        fontsize=14, fontweight='bold', ha='center')

# === INPUT ===
input_box = FancyBboxPatch((0.5, 5.5), 1.5, 1, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='#e8f4f8', linewidth=2)
ax.add_patch(input_box)
ax.text(1.25, 6, 'Input\n$X$', fontsize=11, ha='center', va='center', fontweight='bold')

# === LEFT BRANCH: BASISSET (Sequential Eigenfunction Learning) ===
# Arrow from input to left branch
arrow1 = FancyArrowPatch((2.1, 6), (2.8, 6),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#2E86AB')
ax.add_patch(arrow1)

# BasisSet header
ax.text(3.5, 7.1, 'BasisSet: Sequential Eigenfunction Learning', 
        fontsize=11, fontweight='bold', color='#2E86AB')

# Three eigenfunction boxes (representing sequential training)
ei_ys = [5.3, 4.0, 2.7]
ei_labels = [r'$\phi_1(X)$', r'$\phi_2(X)$', r'$\phi_K(X)$']
phi_boxes = []

for i, (y, label) in enumerate(zip(ei_ys, ei_labels)):
    # Status indicator: first one active
    color = '#4CAF50' if i == 0 else '#CCCCCC'
    alpha = 1.0 if i == 0 else 0.5
    edge = 'green' if i == 0 else '#CCCCCC'
    linewidth = 3 if i == 0 else 1.5
    
    box = FancyBboxPatch((2.8, y), 1.4, 0.6,
                         boxstyle="round,pad=0.05",
                         edgecolor=edge, facecolor=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    ax.text(3.5, y+0.3, label, fontsize=10, ha='center', va='center', 
            fontweight='bold', color='white' if i==0 else 'gray')
    phi_boxes.append((3.5, y+0.3))

# Arrow pattern: active func updates → others freeze
for i in range(len(ei_ys)-1):
    ax.annotate('', xy=(3.5, ei_ys[i+1]+0.6), xytext=(3.5, ei_ys[i]),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green', linestyle='dashed'))
    ax.text(3.9, (ei_ys[i]+ei_ys[i+1])/2, 'freeze', fontsize=8, style='italic', color='green')

# === RIGHT BRANCH: METRICNET (Metric Adaptation) ===
# Arrow from input to right branch
arrow2 = FancyArrowPatch((2.1, 6), (9.2, 6),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#D62828')
ax.add_patch(arrow2)

# MetricNet header
ax.text(10, 7.1, 'MetricNet: A(x) = Λ(x)·U(ω(x))', 
        fontsize=11, fontweight='bold', color='#D62828')

# Λ(x) network
lambda_box = FancyBboxPatch((9.2, 5.2), 1.6, 0.7,
                            boxstyle="round,pad=0.05",
                            edgecolor='#D62828', facecolor='#FFE5E5', linewidth=2)
ax.add_patch(lambda_box)
ax.text(10, 5.55, r'$\Lambda(x)$ network', fontsize=9, ha='center', va='center', fontweight='bold')

# U(ω(x)) network  
u_box = FancyBboxPatch((9.2, 3.8), 1.6, 0.7,
                       boxstyle="round,pad=0.05",
                       edgecolor='#D62828', facecolor='#FFE5E5', linewidth=2)
ax.add_patch(u_box)
ax.text(10, 4.15, r'$U_{\mathrm{Trotter}}(\omega(x))$', fontsize=9, ha='center', va='center', fontweight='bold')

# Description of U_trotter
ax.text(10, 3.3, r'even/odd Givens rotations', fontsize=8, ha='center', style='italic', color='#555')

# === INTERACTION: MDE LOSS ===
ax.text(6.2, 4.8, r'Modified Dirichlet Energy Loss', fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='#FF9800', linewidth=2, pad=0.4))

ax.text(6.2, 4.3, r'$\mathcal{L}_{\mathrm{MDE}} = \mathbb{E}_x[\|A(x)\nabla\phi_k(x)\|^2]$',
        fontsize=9, ha='center', style='italic')

# Bidirectional arrows showing interaction
arrow_left = FancyArrowPatch((4.3, 4.5), (5.5, 4.5),
                            arrowstyle='<->', mutation_scale=15, linewidth=2, color='#FF9800')
ax.add_patch(arrow_left)

arrow_right = FancyArrowPatch((6.9, 4.5), (8.1, 4.5),
                             arrowstyle='<->', mutation_scale=15, linewidth=2, color='#FF9800')
ax.add_patch(arrow_right)

# === CLASSIFICATION HEAD ===
head_arrow = FancyArrowPatch((4.7, 2.0), (6.0, 1.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#555')
ax.add_patch(head_arrow)

metric_arrow = FancyArrowPatch((10, 3.8), (6.5, 1.5),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='#555')
ax.add_patch(metric_arrow)

head_box = FancyBboxPatch((5.2, 0.8), 1.8, 0.9,
                          boxstyle="round,pad=0.1",
                          edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(head_box)
ax.text(6.1, 1.25, r'Classification Head', fontsize=10, ha='center', va='center', fontweight='bold', color='#1565C0')

# === OUTPUT ===
output_box = FancyBboxPatch((6.0, -0.7), 1.2, 0.6,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#C8E6C9', linewidth=2)
ax.add_patch(output_box)
ax.text(6.6, -0.4, r'$\hat{y}$', fontsize=12, ha='center', va='center', fontweight='bold')

arrow_out = FancyArrowPatch((6.1, 0.8), (6.6, 0),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow_out)

# === LOSS COMPONENTS ===
loss_y = 2.3
ax.text(0.5, loss_y, 'Loss Components:', fontsize=10, fontweight='bold')

loss_components = [
    (r'$\mathcal{L}_{\mathrm{task}}$: classification', 0.5, loss_y - 0.5, '#E8F5E9'),
    (r'$\mathcal{L}_{\mathrm{MDE}}$: orthogonality (eq. 2)', 0.5, loss_y - 1.0, '#FFF3E0'),
    (r'$\mathcal{L}_{\mathrm{gram}}$: Gram matrix constraint', 0.5, loss_y - 1.5, '#F3E5F5'),
]

for text, x, y, color in loss_components:
    box = FancyBboxPatch((x, y-0.2), 3, 0.35,
                         boxstyle="round,pad=0.03",
                         edgecolor='#999', facecolor=color, linewidth=1, alpha=0.7)
    ax.add_patch(box)
    ax.text(x + 0.05, y, text, fontsize=8, va='center')

# === LEGEND / KEY INSIGHT ===
key_insight = (
    '• φ₁, φ₂, ..., φₖ learn sequentially (one active at each step)\n'
    '• A(x) adapts to improve each eigenfunction via MDE loss\n'
    '• Gram constraint enforces orthogonality: C = ΦᵀΦ/N ≈ I\n'
    '• Sequential nature enables stability and interpretability'
)
ax.text(7.5, 0.3, key_insight, fontsize=8, 
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#999', linewidth=1, pad=0.5),
        verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('/Users/varvaranazarenko/materials/EFDO/paper_0/figures/fig_architecture.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved fig_architecture.png")
plt.close()

# === CREATE: ALGORITHM PSEUDOCODE AS FIGURE (IMPROVED) ===
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title box
title_box = FancyBboxPatch((0.2, 9.3), 9.6, 0.6,
                          boxstyle="round,pad=0.1",
                          edgecolor='#2E86AB', facecolor='#2E86AB', linewidth=2.5)
ax.add_patch(title_box)
ax.text(5, 9.6, 'Algorithm: Sequential Eigenfunction Dirichlet Optimization (EFDO)',
        fontsize=13, ha='center', va='center', fontweight='bold', color='white')

# Main algorithm box
algo_text = (
    "Input: Dataset {(xᵢ, yᵢ)}, K ∈ ℕ (number of eigenfunctions), w_task, w_mde, w_gram (loss weights)\n"
    "\n"
    "INITIALIZATION:\n"
    "  • Initialize BasisSet with K learnable functions φ₁, φ₂, …, φₖ\n"
    "  • Initialize MetricNet: A(x) = Λ(x) · U_Trotter(ω(x))  [exact orthogonal rotation]\n"
    "\n"
    "FOR step t ← 1 to T_max DO:\n"
    "\n"
    "  k ← ((t - 1) mod K) + 1    // Cycle through active eigenfunction: 1 → 2 → ⋯ → K → 1 → ⋯\n"
    "\n"
    "  FOR each batch (X, y):\n"
    "    \n"
    "    FORWARD PASS:\n"
    "      Φ(X) ← [φ₁(X), φ₂(X), …, φₖ(X)]ᵀ  // Evaluate all K eigenfunctions  (N × K)\n"
    "      A(X) ← MetricNet(X)                 // Get adaptive metric             (N × d × d)\n"
    "      ∇φₖ ← autodiff(φₖ(X))              // Gradient of ACTIVE eigenfunction (N × d)\n"
    "      ŷ ← Head(Φ(X))                     // Linear classifier output        (N,)\n"
    "\n"
    "    LOSS COMPUTATION:\n"
    "      L_task  ← CrossEntropy(ŷ, y)                    // Classification loss\n"
    "      L_mde   ← 𝔼ₓ[‖A(x)·∇φₖ(x)‖²]                   // Modified Dirichlet Energy\n"
    "      L_gram  ← ‖Φ(X)ᵀΦ(X)/N - I‖²_F              // Gram matrix constraint\n"
    "      L_total ← w_task·L_task + w_mde·L_mde + w_gram·L_gram\n"
    "\n"
    "    OPTIMIZATION:\n"
    "      Backward: ∇_θₖ L_total, ∇_θₐ L_total          // Only active function & metric get gradients\n"
    "      Update:   θₖ ← θₖ - α·∇_θₖ L_total             // Update active φₖ\n"
    "                 θₐ ← θₐ - α·∇_θₐ L_total             // Update metric A(x)\n"
    "                 // φ₁, …, φₖ₋₁, φₖ₊₁, …, φₖ stay FROZEN\n"
)

ax.text(0.3, 8.9, algo_text, fontsize=8, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F9F9F9', edgecolor='#333', 
        linewidth=1.5, pad=0.7, alpha=0.95))

# Key properties box at bottom
key_box = FancyBboxPatch((0.2, 0.05), 9.6, 0.85,
                         boxstyle="round,pad=0.08",
                         edgecolor='#FF9800', facecolor='#FFF8E1', linewidth=2)
ax.add_patch(key_box)

key_text = (
    "Key Properties:  • Sequential activation (k cycles through {1,…,K}) prevents basis collapse\n"
    "• MDE loss couples metric A(x) with gradients ∇φₖ, creating synergy  • Gram constraint enforces orthogonality C ≈ I"
)
ax.text(5, 0.45, key_text, fontsize=8.5, ha='center', va='center', style='italic',
        family='sans-serif')

plt.tight_layout()
plt.savefig('/Users/varvaranazarenko/materials/EFDO/paper_0/figures/fig_algorithm.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved fig_algorithm.png (improved)")
plt.close()

print("Done. Both figures saved.")
