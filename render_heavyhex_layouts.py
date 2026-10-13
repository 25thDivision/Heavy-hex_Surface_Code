#!/usr/bin/env python3
"""
Heavy-hex patch layout figures, styled after KCS fig1_codes.png (panels b/c/e).
===============================================================================
Outputs (repo root, png + pdf each):

  layout_heavyhex_18q_verificated      the depth-7 18-qubit KCS patch actually
                                       used by run_ibm_experiment.py (constants
                                       from KCS/ibm_experiment/circuits/
                                       heavyhex_surface_code_depth7.py;
                                       KCS itself untouched)
  layout_heavyhex_18q_37q_verificated  18q patch (left) + 37q patch (right)
  (fig_37q / fig_general remain available for the standalone 37q and the
  general diamond builder at (3,3), (3,5), (5,3))

Style (palette, markers, idle lattice) follows KCS/final_results/plot_fig1.py.
Dual-use rung ancillas are bicolor: left half = Z, right half = X.

Run from the repo root:  python3 render_heavyhex_layouts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from circuits.heavyhex.heavyhex_37q import DATA_PHYS, BRIDGE_PHYS
from circuits.heavyhex.heavyhex_general import assign

# ---- IEEE style (fig_common.py) ----
mpl.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size':        8,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})

# ---- palette (plot_fig1.py) ----
C = {
    'Zanc':   '#4477AA',
    'Xanc':   '#CC6677',
    'bridge': '#BBBBBB',
    'data':   '#666666',
    'idle':   '#E8E8E8',
    'idleE':  '#888888',
    'title':  '#333333',
    'edge':   '#2B3A4A',
}

# Marker radii in lattice units (adjacent-site distance = 1); same
# radius / site-spacing ratio as panel (e) (0.92-pitch grid, half-links).
DATA_R   = 0.24
ANC_R    = 0.22
BRIDGE_R = 0.22
IDLE_R   = 0.11


# ============================================================
# drawing helpers (plot_fig1.py, rescaled)
# ============================================================
def add_data(ax, x, y):
    ax.add_patch(Circle((x, y), DATA_R, facecolor=C['data'],
                        edgecolor='black', linewidth=0.5, zorder=5))


def add_anc(ax, x, y, kinds):
    """kinds = set of check types measured by this rung ancilla.
    {'Z','X'} -> bicolor (left Z, right X); single type -> solid."""
    if len(kinds) == 2:
        ax.add_patch(Wedge((x, y), ANC_R, 90, 270, facecolor=C['Zanc'],
                           alpha=0.55, edgecolor='none', zorder=5))
        ax.add_patch(Wedge((x, y), ANC_R, -90, 90, facecolor=C['Xanc'],
                           alpha=0.55, edgecolor='none', zorder=5))
        ax.add_patch(Circle((x, y), ANC_R, facecolor='none',
                            edgecolor='black', linewidth=0.4, zorder=5))
    else:
        color = C['Zanc' if 'Z' in kinds else 'Xanc']
        ax.add_patch(Circle((x, y), ANC_R, facecolor=color, alpha=0.55,
                            edgecolor='black', linewidth=0.4, zorder=5))


def add_bridge(ax, x, y):
    s = BRIDGE_R
    ax.add_patch(Polygon([(x, y + s), (x + s, y), (x, y - s), (x - s, y)],
                         closed=True, facecolor=C['bridge'], alpha=0.55,
                         edgecolor='black', linewidth=0.4, zorder=5))


def add_idle(ax, x, y):
    ax.add_patch(Circle((x, y), IDLE_R, facecolor=C['idle'],
                        edgecolor=C['idleE'], linewidth=0.3,
                        alpha=0.6, zorder=2))


def add_edge(ax, p1, p2, alpha=0.35, lw=0.4):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color='black', alpha=alpha, linewidth=lw, zorder=2)


def add_patch_edge(ax, p1, p2):
    """Dark coupling edge between two patch qubits."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=C['edge'], alpha=0.85, linewidth=1.1, zorder=3)


def draw_background(ax, row_lo, row_hi, col_lo, col_hi, yoff=0.0, ytrans=None):
    """Heron heavy-hex lattice in chip coordinates: every (row, col) site
    is a qubit (x = col, y = -2*row); rung qubits between rows r, r+1 sit
    at cols c%4 == 1 (r odd) / c%4 == 3 (r even) -- the parity of the
    q25-q107 numbering used by the 37q patch. `ytrans` remaps chip y
    before yoff (used for vertical flips)."""
    t = ytrans or (lambda y: y)
    P = lambda r, c: (c, t(-2 * r) + yoff)
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            add_idle(ax, *P(r, c))
            if c < col_hi:
                add_edge(ax, P(r, c), P(r, c + 1))
    for r in range(row_lo, row_hi):
        par = 1 if r % 2 == 1 else 3
        for c in range(col_lo, col_hi + 1):
            if c % 4 == par:
                pos = (c, t(-2 * r - 1) + yoff)
                add_idle(ax, *pos)
                add_edge(ax, P(r, c), pos)
                add_edge(ax, pos, P(r + 1, c))


def dashed_patch_box(ax, xs, ys, pad=0.45):
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor='none', edgecolor='black',
                           linewidth=0.7, linestyle='--', zorder=3))


def draw_legend(ax, x0, y, fontsize=10):
    """Manual legend row starting at data coords (x0, y); markers drawn
    in data units so their sizes match the lattice markers everywhere."""
    r = 0.15

    def circ(x, fc, rr=r, alpha=1.0, ec='black', lw=0.4):
        ax.add_patch(Circle((x, y), rr, facecolor=fc, alpha=alpha,
                            edgecolor=ec, linewidth=lw, zorder=6))

    def diamond(x):
        ax.add_patch(Polygon([(x, y + r), (x + r, y), (x, y - r), (x - r, y)],
                             closed=True, facecolor=C['bridge'], alpha=0.55,
                             edgecolor='black', linewidth=0.4, zorder=6))

    def dual(x):
        ax.add_patch(Wedge((x, y), r, 90, 270, facecolor=C['Zanc'],
                           alpha=0.55, edgecolor='none', zorder=6))
        ax.add_patch(Wedge((x, y), r, -90, 90, facecolor=C['Xanc'],
                           alpha=0.55, edgecolor='none', zorder=6))
        circ(x, 'none')

    items = [
        ('data',    lambda x: circ(x, C['data'], lw=0.5)),
        ('bridge',  diamond),
        ('Z-check', lambda x: circ(x, C['Zanc'], alpha=0.55)),
        ('X-check', lambda x: circ(x, C['Xanc'], alpha=0.55)),
        ('Z/X',     dual),
        ('idle',    lambda x: circ(x, C['idle'], rr=0.10, alpha=0.6,
                                   ec=C['idleE'], lw=0.3)),
    ]
    x = x0
    for label, drawfn in items:
        drawfn(x)
        ax.text(x + 0.28, y, label, va='center',
                fontsize=fontsize, color=C['title'])
        x += 0.62 + 0.24 * len(label)


def caption(ax, x, y, text, fontsize=11):
    ax.text(x, y, text, ha='center', va='top', fontsize=fontsize,
            color=C['title'], style='italic')


def save(fig, out_stem, dpi=220):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=dpi,
                    bbox_inches='tight', pad_inches=0.05,
                    facecolor='white')
        print(f'Saved: {out_stem}.{ext}')
    plt.close(fig)


# ============================================================
# 37-qubit (3,3) patch (heavyhex_37q / heavyhex_depth7_opt_for_37q)
# ============================================================
# rung ancilla -> its two data endpoints (heavyhex_37q.required_edges)
RUNG_37Q = {37: (25, 45), 56: (43, 63), 57: (47, 67), 76: (61, 81),
            77: (65, 85), 78: (69, 89), 96: (83, 103), 97: (87, 107)}
ROWS_37Q = [[25],
            list(range(43, 48)),
            list(range(61, 70)),
            list(range(81, 90)),
            list(range(103, 108))]


def draw_37q(ax, yoff=0.0):
    P = lambda p: (p % 20, -2 * (p // 20) + yoff)   # chip label -> (x, y)
    draw_background(ax, 1, 5, 0, 10, yoff=yoff)

    # dark coupling edges of the patch
    for row in ROWS_37Q:
        for a, b in zip(row, row[1:]):
            add_patch_edge(ax, P(a), P(b))
    anc_xy = {}
    for a, (u, v) in RUNG_37Q.items():
        pos = (u % 20, -(2 * (u // 20) + 1) + yoff)
        anc_xy[a] = pos
        add_patch_edge(ax, P(u), pos)
        add_patch_edge(ax, pos, P(v))

    for p in DATA_PHYS:
        add_data(ax, *P(p))
    for p in BRIDGE_PHYS:
        add_bridge(ax, *P(p))
    for a, pos in anc_xy.items():
        add_anc(ax, *pos, {'Z', 'X'})   # every 37q ancilla is dual-use


def fig_37q(out_stem):
    fig, ax = plt.subplots(figsize=(6.6, 7.2))
    ax.set_aspect('equal')
    ax.axis('off')
    draw_37q(ax)
    draw_legend(ax, -0.4, -0.8)
    caption(ax, 5, -11.2,
            '17 data + 12 bridges + 8 dual-use ancillas = 37 qubits')
    ax.set_xlim(-0.9, 10.9)
    ax.set_ylim(-12.2, -0.1)
    save(fig, out_stem)


# ============================================================
# KCS d=3 heavy-hex surface code patch (depth-7, 18 qubits)
# ============================================================
# Constants transcribed from KCS/ibm_experiment/circuits/
# heavyhex_surface_code_depth7.py (the file actually imported by
# run_ibm_experiment.py; read-only, KCS untouched):
#   DATA_QUBITS = 43,45,47,63,65,67,83,85,87
#   bridges     = 44,46,64,66,84,86
#   ancillas    = 56 (Z), 57 (Z), 77 (X)   [rung sites]
# 56/57 also serve as boundary X-checks and 44/86 as boundary Z-checks
# (ROUND_DEFS 'bnd'); that dual role is deliberately not shown here.
D7_DATA    = [43, 45, 47, 63, 65, 67, 83, 85, 87]
D7_BRIDGES = [44, 46, 64, 66, 84, 86]
D7_ANC     = {56: {'Z'}, 57: {'Z'}, 77: {'X'}}
D7_RUNGS   = {56: (43, 63), 57: (47, 67), 77: (65, 85)}
D7_ROWS    = [list(range(43, 48)),
              list(range(63, 68)),
              list(range(83, 88))]
D7_ALL     = sorted(set(D7_DATA + D7_BRIDGES + list(D7_ANC)))


def draw_18q(ax, yoff=0.0):
    """Depth-7 18-qubit patch in the same Heron chip mapping as draw_37q
    (qubit p -> (p % 20, -2 * (p // 20))); no flip/offset needed, the
    rung-column parity already matches the shared background."""
    P = lambda p: (p % 20, -2 * (p // 20) + yoff)
    draw_background(ax, 1, 5, 0, 10, yoff=yoff)

    for row in D7_ROWS:
        for a, b in zip(row, row[1:]):
            add_patch_edge(ax, P(a), P(b))
    anc_xy = {}
    for a, (u, v) in D7_RUNGS.items():
        pos = (u % 20, -(2 * (u // 20) + 1) + yoff)
        anc_xy[a] = pos
        add_patch_edge(ax, P(u), pos)
        add_patch_edge(ax, pos, P(v))

    for p in D7_DATA:
        add_data(ax, *P(p))
    for p in D7_BRIDGES:
        add_bridge(ax, *P(p))
    for a, pos in anc_xy.items():
        add_anc(ax, *pos, D7_ANC[a])


def fig_18q(out_stem, dpi=220):
    fig, ax = plt.subplots(figsize=(6.6, 7.2))
    ax.set_aspect('equal')
    ax.axis('off')
    draw_18q(ax)
    draw_legend(ax, -0.4, -0.8)
    caption(ax, 5, -11.2,
            '9 data + 6 bridges + 3 ancillas = 18 qubits')
    ax.set_xlim(-0.9, 10.9)
    ax.set_ylim(-12.2, -0.1)
    save(fig, out_stem, dpi=dpi)


# ============================================================
# combined: depth-7 18q patch (left) + 37q patch (right)
# ============================================================
def fig_18q_37q(out_stem, dpi=220):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01,
                        wspace=0.06)

    ax_leg = fig.add_axes([0.28, 0.895, 0.44, 0.06])
    ax_leg.set_aspect('equal')
    ax_leg.axis('off')
    ax_leg.set_xlim(-0.5, 10.9)
    ax_leg.set_ylim(-0.5, 0.5)
    draw_legend(ax_leg, -0.2, 0)

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-0.9, 10.9)
        ax.set_ylim(-12.0, -1.3)

    draw_18q(axes[0])
    caption(axes[0], 5, -11.2,
            '9 data + 6 bridges + 3 ancillas = 18 qubits')
    draw_37q(axes[1])
    caption(axes[1], 5, -11.2,
            '17 data + 12 bridges + 8 dual-use ancillas = 37 qubits')

    # panel-scale check: px per lattice unit and marker radius px at `dpi`
    for name, ax in zip(('left(18q)', 'right(37q)'), axes):
        sx = (ax.transData.transform((1, 0))
              - ax.transData.transform((0, 0)))[0] * dpi / fig.dpi
        print(f'  {name}: grid spacing = {sx:.2f}px, '
              f'data marker r = {DATA_R * sx:.2f}px, '
              f'anc/bridge r = {ANC_R * sx:.2f}px')
    save(fig, out_stem, dpi=dpi)


# ============================================================
# general diamond builder at (3,3), (3,5), (5,3)
# ============================================================
def general_panel_data(dx, dz):
    """code coords -> chip coords (code (r,c) -> chip (r+1, c+1))."""
    code, _sol, plan = assign(dx, dz)
    P = lambda r, c: (c + 1, -2 * (r + 1))
    data = [P(r, c) for (r, c) in code.data]
    bridges = [(c + 2, -2 * (r + 1)) for (r, c) in code.data
               if (r, c + 2) in code.dset]
    rung_types = {}
    for p in plan.values():
        rung_types.setdefault(p['anc'], set()).add(p['t'])
    ancs = []
    for rung in code.rungs:
        if rung in rung_types:
            (r, c) = rung[0]
            ancs.append(((c + 1, -(2 * (r + 1) + 1)), rung_types[rung]))
    return code, data, bridges, ancs


def fig_general(out_stem):
    sizes = [(3, 3), (3, 5), (5, 3)]
    panels = {s: general_panel_data(*s) for s in sizes}

    # chip-coordinate extents per panel (background = patch rows / + 2 cols)
    lims = {}
    for s, (code, data, bridges, ancs) in panels.items():
        rows = [r for r, _ in code.data]
        cols = [c for _, c in code.data]
        r0, r1 = min(rows) + 1, max(rows) + 1             # chip rows
        c0, c1 = min(cols) + 1 - 2, max(cols) + 1 + 2     # chip cols
        lims[s] = (r0, r1, c0, c1)
    widths = [lims[s][3] - lims[s][2] + 1.4 for s in sizes]
    r1max = max(lims[s][1] for s in sizes)
    y_top, y_bot = 0.1, -2 * r1max - 2.6

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.2),
                             gridspec_kw={'width_ratios': widths})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.05)

    for label, s, ax in zip('abc', sizes, axes):
        code, data, bridges, ancs = panels[s]
        r0, r1, c0, c1 = lims[s]
        yoff = -(2 * r1max - 2 * r1) / 2     # center smaller lattices
        ax.set_aspect('equal')
        ax.axis('off')
        draw_background(ax, r0, r1, c0, c1, yoff=yoff)
        for x, y in data:
            add_data(ax, x, y + yoff)
        for x, y in bridges:
            add_bridge(ax, x, y + yoff)
        for (x, y), kinds in ancs:
            add_anc(ax, x, y + yoff, kinds)
        patch_xy = data + bridges + [xy for xy, _ in ancs]
        dashed_patch_box(ax, [p[0] for p in patch_xy],
                         [p[1] + yoff for p in patch_xy])

        xc = (c0 + c1) / 2
        total = code.n + len(bridges) + len(ancs)
        caption(ax, xc, y_bot + 1.4,
                f'$(d_x, d_z)=({s[0]},{s[1]})$: '
                f'{code.n} data + {len(bridges)} bridges + {len(ancs)} '
                f'ancillas = {total} qubits')
        ax.set_xlim(c0 - 0.7, c1 + 0.7)
        ax.set_ylim(y_bot, y_top)
        if label == 'a':
            draw_legend(ax, c0 - 0.4, -0.6)

    save(fig, out_stem)


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    # verified renders: left panel = depth-7 18q patch actually used by
    # KCS/ibm_experiment/run_ibm_experiment.py (heavyhex_surface_code_depth7)
    fig_18q(os.path.join(here, 'layout_heavyhex_18q_verificated'), dpi=300)
    fig_18q_37q(os.path.join(here, 'layout_heavyhex_18q_37q_verificated'),
                dpi=300)
