#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D

from DecentralizedCBO import DecentralizedCBO, ring, star, ring_2hop
from problems import RastriginProblem
#%%
SEED = 16
d, num_nodes, n, max_it = 2, 20, 20, 500
lam, sigma, dt = 0.5, .01, 0.05

P = RastriginProblem(d=d)
f = P.generate()['objective']
f.b = 2
W = ring(num_nodes, lam)#ring_2hop(num_nodes) #star(num_nodes) # ring(num_nodes, lam)
#W = star(num_nodes)
np.random.seed(SEED)
dyn = DecentralizedCBO(
    f, W=W, x=P.sample(num_nodes, n),
    alpha=100., sigma=sigma, dt=dt, noise='anisotropic', f_dim='3D',
    verbosity=0, max_it=max_it, post_process=P.post_process,
    track_args={'names': [], 'save_int': 10 ** 9},
    noise_drift_mode = 'z',
    seed = SEED + 1
)

X, C, Z, E = [], [], [], []
while not dyn.terminate():
    dyn.step()
    X.append(np.array(dyn.x))
    C.append(np.array(dyn.consensus[:, 0, :]))
    Z.append(np.array(dyn.z))
    E.append(float(dyn.best_cur_energy.min()))   # current ensemble, not the running best
X, C, Z, E = np.array(X), np.array(C), np.array(Z), np.array(E)
T = len(X)


def diam(A):                       # (T, P, d) -> (T,)
    return (A.max(axis=1) - A.min(axis=1)).max(axis=-1)


dia_pool = diam(X.reshape(T, -1, d))                          # all particles
dia_node = (X.max(axis=2) - X.min(axis=2)).max(axis=-1).max(axis=-1)   # worst node
print('{} iterations; pooled diameter {:.3g} -> {:.3g}'.format(T, dia_pool[0], dia_pool[-1]))

#%% objective background
lim = P.domain[1]
gs_ = np.linspace(-lim, lim, 250)
G1, G2 = np.meshgrid(gs_, gs_)
BG = np.asarray(f(np.stack([G1, G2], -1).reshape(-1, d))).reshape(250, 250)

#%%
%matplotlib widget
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, 'text.color': 'black',
    'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
})
colors = plt.cm.tab20(np.linspace(0, 1, num_nodes))
fig = plt.figure(figsize=(8.5, 4.2))
grid = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, .05],
                        hspace=.45, wspace=.22)
ax2d = fig.add_subplot(grid[0, 0])
axd = fig.add_subplot(grid[0, 1])
axsl = fig.add_subplot(grid[1, :])

ZOOM, MARGIN, MIN_HALF, RES = True, .25, 1e-6, 200
ZOOM_HOLD = 1      # window is fixed over blocks of this many iterations
ZOOM_RANGE = None#(209, 275)    # or (k0, k1): pin one window for every frame, e.g. a cycle
_zoom_key = [None]

bg = ax2d.imshow(BG, extent=[-lim, lim, -lim, lim], origin='lower', aspect='auto',
                 cmap='magma_r', alpha=.75, zorder=0)
ax2d.set_xlim(-lim, lim); ax2d.set_ylim(-lim, lim); ax2d.set_aspect('equal')
ax2d.set_xlabel(r'$x_1$'); ax2d.set_ylabel(r'$x_2$')


def zoom_to(k):
    key = ZOOM_RANGE if ZOOM_RANGE is not None else k // ZOOM_HOLD
    if key == _zoom_key[0]:
        return
    _zoom_key[0] = key
    lo, hi = ZOOM_RANGE if ZOOM_RANGE is not None else (
        int(key) * ZOOM_HOLD, min(int(key) * ZOOM_HOLD + ZOOM_HOLD, T))

    pts_all = np.vstack([X[lo:hi].reshape(-1, d),
                         C[lo:hi].reshape(-1, d), Z[lo:hi].reshape(-1, d)])
    ctr = (pts_all.min(0) + pts_all.max(0)) / 2
    half = max((pts_all.max(0) - pts_all.min(0)).max() / 2, MIN_HALF) * (1 + MARGIN)
    g1, g2 = np.meshgrid(np.linspace(ctr[0] - half, ctr[0] + half, RES),
                         np.linspace(ctr[1] - half, ctr[1] + half, RES))
    zz = np.asarray(f(np.stack([g1, g2], -1).reshape(-1, d))).reshape(RES, RES)
    bg.set_data(zz)
    bg.set_extent([ctr[0] - half, ctr[0] + half, ctr[1] - half, ctr[1] + half])
    bg.set_clim(zz.min(), zz.max())
    ax2d.set_xlim(ctr[0] - half, ctr[0] + half)
    ax2d.set_ylim(ctr[1] - half, ctr[1] + half)

pts = [ax2d.scatter([], [], color=colors[m], s=12, alpha=.8, zorder=2)
       for m in range(num_nodes)]
con = [ax2d.scatter([], [], color=colors[m], marker='*', s=170, zorder=4,
                    edgecolors='k', linewidths=.5) for m in range(num_nodes)]
zed = [ax2d.scatter([], [], color=colors[m], marker='D', s=150, zorder=4,
                    edgecolors='k', linewidths=.5) for m in range(num_nodes)]
link = [ax2d.plot([], [], color=colors[m], lw=1., alpha=.6, zorder=3)[0]
        for m in range(num_nodes)]
# node index next to each tracking point; offset in points so it survives zoom
zlab = [ax2d.annotate(str(m), (0, 0), textcoords='offset points', xytext=(0, 0),
                      ha='center', va='center', fontsize=6, color='black',
                      zorder=6) for m in range(num_nodes)]
LEG_2D = [
    Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=5,
           alpha=.6, label='particles'),
    Line2D([0], [0], marker='*', color='gray', linestyle='None', markersize=10,
           label=r'consensus $c_m$'),
    Line2D([0], [0], marker='D', color='gray', linestyle='None', markersize=6,
           label=r'tracking $z_m$'),
]
ax2d.legend(handles=LEG_2D, loc='upper center', bbox_to_anchor=(.5, -.13),
            ncol=3, fontsize=8, frameon=True)

axd.semilogy(dia_node, color='#2a78d6', lw=1.4, label='within-node diameter (max)')
axd.semilogy(dia_pool, color='#eb6834', lw=1.4, label='pooled diameter')
axd.semilogy(E, color='#1baf7a', lw=1.4, alpha=.7, label='best energy (current ensemble)')
vline = axd.axvline(0, color='k', lw=1., zorder=5)
axd.set_xlabel('iteration'); axd.grid(True, lw=.4, alpha=.5)
axd.legend(fontsize=8, frameon=False)
an = ax2d.annotate('$k={}$'.format(0), 
             (0.8, 0.9), xycoords='axes fraction',
             xytext=(0, 0), 
             textcoords='offset points',
             fontsize=8, color='w', va='center', ha='left',
             bbox=dict(boxstyle='round', fc='k', alpha=.5, lw=0))


def draw_frame(k):
    for m in range(num_nodes):
        pts[m].set_offsets(X[k, m])
        con[m].set_offsets(C[k, m][None])
        zed[m].set_offsets(Z[k, m][None])
        link[m].set_data([C[k, m, 0], Z[k, m, 0]], [C[k, m, 1], Z[k, m, 1]])
        zlab[m].xy = (Z[k, m, 0], Z[k, m, 1])
    vline.set_xdata([k, k])
    if ZOOM:
        zoom_to(k)
    ax2d.set_title('iteration {}   within-node {:.3g}   pooled {:.3g}'.format(
        k, dia_node[k], dia_pool[k]), fontsize=10)
    an.set_text('$k={}$'.format(k))

def update(val):
    draw_frame(int(slider.val))


slider = Slider(axsl, 'iteration', 0, T - 1, valinit=0, valstep=1)
slider.on_changed(update)
update(0)
plt.show()


#%%
def save_gif(path=None, stride=None, fps=15, dpi=160, panel='all', colors=128):
    from io import BytesIO
    from PIL import Image
    path = Path(path) if path else (Path(__file__).resolve().parent.parent / 'results'
                                    / 'consensus_2d{}.gif'.format('' if panel == 'all' else '_2d'))
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = stride or max(1, T // 150)
    ks = range(0, T, stride)

    imgs = []
    for k in ks:
        slider.set_val(k)              # moves the handle too, so it reads as a progress bar
        fig.canvas.draw()
        bbox = None
        if panel == '2d':
            bbox = ax2d.get_tightbbox(fig.canvas.get_renderer())
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.03, 1.03)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches=bbox, facecolor='white')
        buf.seek(0)
        imgs.append(Image.open(buf).convert('RGB'))

    # a tight bbox shifts with the tick labels, so pad every frame to one size
    w, h = max(i.width for i in imgs), max(i.height for i in imgs)
    out = []
    for im in imgs:
        if im.size == (w, h):
            out.append(im)
        else:
            canvas = Image.new('RGB', (w, h), 'white')
            canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
            out.append(canvas)

    # one palette shared by every frame: per-frame palettes blow the file up by
    # an order of magnitude and make the colours flicker
    pal = out[0].quantize(colors=colors, method=Image.MEDIANCUT)
    out = [im.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for im in out]
    out[0].save(path, save_all=True, append_images=out[1:],
                duration=int(round(1000 / fps)), loop=0, optimize=True)
    print('wrote {} ({} frames, {}x{}, {:.1f} MB)'.format(
        path.name, len(out), w, h, path.stat().st_size / 1e6))
    return path


save_gif(fps=5, panel='2d')


#%%
def _save_axes(ax, path, pad=1.03):
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer())
    bb = bb.transformed(fig.dpi_scale_trans.inverted()).expanded(pad, pad)
    fig.savefig(path, bbox_inches=bb, transparent=True)


def save_legend(which='2d', out=None, ext='pdf', ncol=None, fontsize=8, frameon=False):
    if which == 'curves':
        handles, labels = axd.get_legend_handles_labels()
    else:
        handles, labels = LEG_2D, [h.get_label() for h in LEG_2D]
    out = Path(out) if out else (Path(__file__).resolve().parent.parent / 'results')
    out.mkdir(parents=True, exist_ok=True)

    lf, la = plt.subplots(figsize=(3, .4))
    la.axis('off')
    la.legend(handles, labels, loc='center', ncol=ncol or len(handles),
              fontsize=fontsize, frameon=frameon)
    p = out / 'legend_{}.{}'.format(which, ext)
    lf.savefig(p, bbox_inches='tight', transparent=True)
    plt.close(lf)
    print('wrote', p.name)
    return p


def save_frames(iters, panel='2d', out=None, stem='frame', ext='pdf', curves=True):
    """Write chosen iterations as individual files.

    panel='2d'   only the 2D axes (what you want for a figure)
    panel='all'  the whole figure, slider included
    curves=True  also writes <stem>_curves.<ext>: the diameter/energy panel
                 with the chosen iterations marked, so the frames can be read
                 off against the trajectory
    """
    out = Path(out) if out else (Path(__file__).resolve().parent.parent / 'results')
    out.mkdir(parents=True, exist_ok=True)
    iters = [int(k) for k in np.atleast_1d(iters)]

    paths = []
    for k in iters:
        draw_frame(k)
        p = out / '{}_it{:05d}.{}'.format(stem, k, ext)
        if panel == '2d':
            _save_axes(ax2d, p)
        else:
            fig.canvas.draw()
            fig.savefig(p, transparent=True)
        paths.append(p)

    if curves:
        marks, vis = [], vline.get_visible()
        vline.set_visible(False)          # the scrub line would read as a frame
        for k in iters:
            marks.append(axd.axvline(k, color='k', lw=.6, ls=':', alpha=.6, zorder=1))
            for series, col in ((dia_node, '#2a78d6'), (dia_pool, '#eb6834'), (E, '#1baf7a')):
                marks.append(axd.plot(k, series[k], 'o', color=col, ms=4.5, mec='k',
                                      mew=.4, zorder=6)[0])
        p = out / '{}_curves.{}'.format(stem, ext)
        _save_axes(axd, p)
        paths.append(p)
        for m in marks:
            m.remove()
        vline.set_visible(vis)
# 
    print('wrote {}: {}'.format(len(paths), ', '.join(q.name for q in paths)))
    return paths


save_frames([192, 220, 245, 265, 290])
save_legend('2d')
# %%
