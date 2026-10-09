"""CPU-only vector figures and table spreads from certified summaries, no private fields."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.set(xlim=(0, 8.5), ylim=(0, 4.3)); ax.axis('off')
    boxes = [(0.2, 2.45, 'Inputs\nConductivity K\nTransport setting'),
             (3.05, 2.45, 'Predictors\nCTA-UNet / FiLM\nBounded log-setting\nmean'),
             (5.9, 2.45, 'Outputs\n25 concentration fields\nFixed source / domain'),
             (0.2, 0.25, 'Generator audit\nHashes / correlations\nDistinct-stream\nscreening'),
             (3.05, 0.25, 'Evaluation pools\nGrouped / twinned\ngeology; fresh T and T2'),
             (5.9, 0.25, 'Evidence and limits\nSSIM / well decisions\nStream-clustered\ninference')]
    artists = []
    for i, (x, y, label) in enumerate(boxes):
        patch = FancyBboxPatch((x, y), 2.4, 1.5, boxstyle='round,pad=0.04',
                                   facecolor='#eef3f6' if i < 3 else '#f4f0e8',
                                   edgecolor='#42586b', linewidth=1.1)
        ax.add_patch(patch)
        text = ax.text(x + 1.2, y + .75, label, ha='center', va='center', linespacing=1.7, fontsize=11)
        artists.append((patch, text))
    for y in (3.2, 1.0):
        for x in (2.7, 5.55):
            ax.annotate('', xy=(x+.29, y), xytext=(x, y),
                        arrowprops={'arrowstyle': '->', 'color': '#42586b', 'lw': 1.5})
    ax.annotate('', xy=(1.4, 1.86), xytext=(1.4, 2.36),
                arrowprops={'arrowstyle': '->', 'color': '#42586b', 'lw': 1.5})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, text in artists:
        border, words = patch.get_window_extent(renderer), text.get_window_extent(renderer)
        if not (border.x0 + 4 < words.x0 < words.x1 < border.x1 - 4 and
                border.y0 + 4 < words.y0 < words.y1 < border.y1 - 4):
            raise ValueError('Schematic text exceeds padding: ' + text.get_text())
    fig.savefig(output/'fig_v9_workflow.pdf', bbox_inches='tight'); plt.close(fig)
    path = ROOT/'results/v8/review_diagnostics.json'
    d = json.loads(path.read_text())
    names = [('log_setting_mean', 'Bounded log-setting mean'), ('matched', 'CTA-UNet, matched'),
             ('k_only', 'CTA-UNet, K only')]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True)
    for ax, metric, title in zip(axes, ['false_negative_rate', 'false_positive_rate', 'balanced_accuracy'],
                                ['Missed exceedances (lower better)', 'False alarms (lower better)',
                                 'Balanced accuracy (higher better)']):
        for y, (name, _) in enumerate(names):
            r = d['wells'][name]['pooled'][metric]; value = r['estimate']; low, high = r['ci95']
            ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt='o', color='#24617d', capsize=4)
        ax.set(xlim=(0, 1), xticks=[0, .25, .5, .75, 1], title=title, xlabel='Rate')
        ax.grid(axis='x', color='#dddddd', lw=.6); ax.set_axisbelow(True)
    axes[0].set_yticks(range(3), [v[1] for v in names]); axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(output/'fig_v9_well_tradeoffs.pdf', bbox_inches='tight'); plt.close(fig)
    matrix_path = ROOT/'figures/v7/fig_v7_counterfactual.json'
    counter = json.loads(matrix_path.read_text())
    from build_counterfactual_v7_figure import CODES, LABEL, tag
    codes = [tag(c) for c in CODES]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['global_ssim', 'plume_ssim'],
                                ['(a) Global SSIM', '(b) Plume SSIM (true region)']):
        values = np.array([[counter['matrix'][metric][r][c] for c in codes] for r in codes])
        im = ax.imshow(values, vmin=.45, vmax=1., cmap='viridis')
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', fontsize=8,
                        color='black' if values[i,j] > .8 else 'white', fontweight='bold' if i==j else 'normal')
        ax.set_xticks(range(6), [f'z{i+1}' for i in range(6)], fontsize=10)
        ax.set_yticks(range(6), [f'z{i+1}' for i in range(6)], fontsize=10)
        ax.set(xlabel='Supplied code (categorical)', ylabel='Simulated setting', title=title)
    fig.colorbar(im, ax=axes, shrink=.75, label='SSIM (display range 0.45 to 1.00)')
    fig.savefig(output/'fig_v9_counterfactual.pdf', bbox_inches='tight'); plt.close(fig)
    lines = ['% Generated from certified absolute-score summaries; existing point macros are unchanged.']
    files = ['v7_poolT_best_complete_2026-10-08.json', 'v7_e2_poolT_results.json']
    summaries = [json.loads((ROOT/'results/v7/analysis'/f).read_text()) for f in files]
    entries = [(summaries[0]['summary'], {'cta_ffl_matched':'Matched', 'cta_ffl_k_only':'Konly',
                'cta_ffl_permuted':'Cyclic', 'cta_ffl_constant_placebo':'Constant',
                'cta_ffl_independent_nuisance':'Nuisance'}, 'reviewT'),
               (summaries[1]['best']['summary'], {'k_only':'Konly', 'v0_raw_channels':'VZero',
                'v1_factorized_levels':'VOne', 'v4_joint_continuous':'VFour', 'fno_v0_raw_channels':'Fno'}, 'reviewE')]
    for summary, keys, prefix in entries:
        for key, label in keys.items():
            for metric, suffix in [('global_ssim', 'Global'), ('plume_ssim', 'Plume')]:
                v = summary[key][metric]
                lines.append('\\newcommand{\\'+prefix+label+suffix+'PM}{'+f"{v['mean']:.4f} $\\pm$ {v['sd']:.4f}"+'}')
    (output/'numbers_review.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    provenance = {'scope':'Post-review descriptive diagnostics; no new model scores',
                  'figure_1':'Original vector schematic, revised padding; renderer checks all six text boxes; no empirical plume pixels',
                  'figure_3':'Case-weighted well rates; 10000 random-stream/seed block resamples; seven wells kept together',
                  'inputs': {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [path, matrix_path]+[ROOT/'results/v7/analysis'/f for f in files]}}
    (output/'review_figure_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
