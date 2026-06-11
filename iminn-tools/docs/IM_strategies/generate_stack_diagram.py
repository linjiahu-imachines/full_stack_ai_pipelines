#!/usr/bin/env python3
"""
Generate RISC-V AI Stack Architecture Diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Set up the figure with more space
fig, ax = plt.subplots(figsize=(16, 14))
ax.set_xlim(0, 11)
ax.set_ylim(-2.5, 12)  # Extended bottom to accommodate legend and hardware layer
ax.axis('off')

# Color scheme
COLOR_MY_SCOPE = '#4CAF50'  # Green
COLOR_COMPILER = '#FF5722'  # Red/Orange
COLOR_SHARED = '#FFC107'    # Amber
COLOR_HARDWARE = '#2196F3'  # Blue
COLOR_TEXT = '#212121'      # Dark text

# Layer heights and positions
layer_height = 1.8
spacing = 0.4
start_y = 10.5

layers = [
    {
        'name': 'Application/Orchestration Layer',
        'components': ['LangChain', 'LlamaIndex', 'Robotics\nPipeline', 'RAG\nSystems', 'AI Agent\nMCP/SKILL\netc.'],
        'color': COLOR_MY_SCOPE,
        'owner': '← SW Focus',
        'y': start_y - 0 * (layer_height + spacing)
    },
    {
        'name': 'ML Framework Layer',
        'components': ['PyTorch', 'TensorFlow', 'JAX', 'HuggingFace\nTransformers', 'ONNX'],
        'color': COLOR_MY_SCOPE,
        'owner': '← SW Focus',
        'y': start_y - 1 * (layer_height + spacing)
    },
    {
        'name': 'Serving/Runtime Layer',
        'components': ['vLLM', 'llama.cpp', 'Ray Serve', 'Triton\nServer', 'FastAPI'],
        'color': COLOR_MY_SCOPE,
        'owner': '← SW Focus',
        'y': start_y - 2 * (layer_height + spacing)
    },
    {
        'name': 'Compiler/Toolchain Layer',
        'components': ['IREE\n(MXFP4)', 'Triton\nBackend', 'MLIR\nDialect', 'StableHLO', 'RISC-V\nGCC/CLANG'],
        'color': COLOR_COMPILER,
        'owner': '← Compiler Team\n(Dependency)',
        'y': start_y - 3 * (layer_height + spacing)
    },
    {
        'name': 'Pre-Silicon Validation Layer',
        'components': ['QEMU\n(Functional)', 'PSim\n(Performance)', 'FPGA\n(4-6 core)', 'Synopsys\nCloud', 'Validation\nInfra'],
        'color': COLOR_SHARED,
        'owner': '← Shared',
        'y': start_y - 4 * (layer_height + spacing)
    },
    {
        'name': 'Hardware Layer',
        'components': ['RISC-V Processor + Custom MXFP4 Extensions', '(Tape-out: September 2026)'],
        'color': COLOR_HARDWARE,
        'owner': '← HW Team',
        'y': start_y - 5 * (layer_height + spacing)
    }
]

# Draw layers
for layer in layers:
    y_pos = layer['y']
    
    # Draw layer background
    layer_box = FancyBboxPatch((0.3, y_pos - layer_height), 8.5, layer_height,
                               boxstyle="round,pad=0.05", 
                               edgecolor='black', 
                               facecolor=layer['color'],
                               alpha=0.3,
                               linewidth=2)
    ax.add_patch(layer_box)
    
    # Draw layer name (positioned above the boxes)
    ax.text(0.4, y_pos - 0.15, layer['name'], 
            fontsize=11, fontweight='bold', 
            verticalalignment='top',
            color=COLOR_TEXT)
    
    # Draw components
    if layer['name'] == 'Hardware Layer':
        # Special handling for hardware layer
        ax.text(5.0, y_pos - 0.85, layer['components'][0],
                fontsize=10, ha='center', va='center',
                color=COLOR_TEXT, fontweight='bold')
        ax.text(5.0, y_pos - 1.25, layer['components'][1],
                fontsize=9, ha='center', va='center',
                color=COLOR_TEXT, style='italic')
    else:
        # Draw component boxes (positioned lower to avoid overlap)
        num_components = len(layer['components'])
        box_width = 7.8 / num_components
        box_height = 1.0
        box_top = y_pos - 0.5  # Start boxes below the layer name
        
        for i, component in enumerate(layer['components']):
            x_pos = 0.6 + i * box_width
            
            # Component box
            comp_box = FancyBboxPatch((x_pos, box_top - box_height), box_width - 0.2, box_height,
                                     boxstyle="round,pad=0.04",
                                     edgecolor='black',
                                     facecolor='white',
                                     linewidth=1.5)
            ax.add_patch(comp_box)
            
            # Component text (centered in box)
            ax.text(x_pos + (box_width - 0.2)/2, box_top - box_height/2,
                   component,
                   fontsize=9, ha='center', va='center',
                   color=COLOR_TEXT, fontweight='bold')
    
    # Draw owner label
    ax.text(9.0, y_pos - 0.9, layer['owner'],
           fontsize=10, ha='left', va='center',
           color=COLOR_TEXT, fontweight='bold')

# Add title
ax.text(5, 11.7, 'IM RISC-V AI Stack Architecture', 
        fontsize=18, fontweight='bold', ha='center',
        color=COLOR_TEXT)

# Add legend (positioned well below all layers)
# Calculate the bottom of the last layer
last_layer_bottom = start_y - 5 * (layer_height + spacing) - layer_height
legend_y = last_layer_bottom - 0.8  # Position legend below the last layer

ax.text(0.5, legend_y, 'LEGEND:', fontsize=11, fontweight='bold', color=COLOR_TEXT)

legend_items = [
    (COLOR_MY_SCOPE, 'SW Focus: Application, Framework, Serving + SDK + Benchmarking + QEMU'),
    (COLOR_COMPILER, 'Dependency: Compiler Team (IREE/Triton backends)'),
    (COLOR_SHARED, 'Shared: PSim Team, Hardware Team collaboration'),
    (COLOR_HARDWARE, 'Hardware: HW Team, Tape-out September 2026')
]

for i, (color, label) in enumerate(legend_items):
    rect = Rectangle((0.5, legend_y - 0.5 - i*0.35), 0.3, 0.2, 
                     facecolor=color, edgecolor='black', alpha=0.3, linewidth=1)
    ax.add_patch(rect)
    ax.text(0.9, legend_y - 0.4 - i*0.35, label, 
           fontsize=9, va='center', color=COLOR_TEXT)

plt.tight_layout(pad=0.5)

# Save the figure
output_file = '/home/linhu/repo/iminn-tools/docs/IM_strategies/stack_architecture_diagram.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
print(f"✅ Diagram saved to: {output_file}")

# Also save as PDF for better quality in presentations
output_pdf = '/home/linhu/repo/iminn-tools/docs/IM_strategies/stack_architecture_diagram.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3, facecolor='white')
print(f"✅ PDF saved to: {output_pdf}")

plt.close()
