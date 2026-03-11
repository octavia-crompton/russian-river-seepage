import json, pathlib, textwrap

nb_path = pathlib.Path('notebooks/2 Russian River  \u2013 analysis.ipynb')
nb = json.loads(nb_path.read_text())

# Cell source as a raw triple-quoted string.
# NOTE: no nested """ anywhere to avoid quote-termination bugs.
CELL_SOURCE = r"""from scipy.optimize import minimize

# -- model helpers -------------------------------------------------------------
_eps = 1e-6

def _model_linear(C, dh):
    return C * dh

def _model_offset_power_vis(C, offset, b, dh, hv, hv_ref):
    # S = C * ((hv + offset)/(hv_ref + offset))^b * dh
    base = np.maximum(hv + offset, _eps)
    xref = max(hv_ref + offset, _eps)
    return C * dh * np.power(base / xref, b)

def _r2(y, yhat):
    sst = np.sum((y - y.mean())**2)
    return float(1 - np.sum((y - yhat)**2) / max(sst, _eps))

def _fit_linear(dh, s):
    dh = np.asarray(dh, float)
    s  = np.asarray(s,  float)
    C  = float(np.dot(dh, s) / max(np.dot(dh, dh), _eps))
    C  = max(C, 1e-12)
    return C, _r2(s, _model_linear(C, dh))

def _fit_offset_power_vis(dh, hv, s, hv_ref):
    dh = np.asarray(dh, float)
    hv = np.asarray(hv, float)
    s  = np.asarray(s,  float)
    off_lb = float(-np.nanmin(hv) + 1e-3)
    off_ub = max(off_lb + 0.1, 0.3 * float(np.nanstd(hv)))
    C0 = float(np.dot(dh, s) / max(np.dot(dh, dh), _eps))
    def sse(theta):
        C, off, b = theta
        if C <= 0 or b < 0 or off < off_lb or off > off_ub:
            return 1e99
        yhat = _model_offset_power_vis(C, off, b, dh, hv, hv_ref)
        return float(np.sum((s - yhat)**2))
    res = minimize(sse, x0=[max(C0, 1e-12), (off_lb + off_ub) / 2, 0.5],
                   bounds=[(1e-12, None), (off_lb, off_ub), (1e-12, None)],
                   method='L-BFGS-B')
    C, off, b = map(float, res.x)
    yhat = _model_offset_power_vis(C, off, b, dh, hv, hv_ref)
    return C, off, b, _r2(s, yhat)

# global h_V reference (median across all pooled events)
_hv_ref_global = float(np.nanmedian(pooled_event_df['visitor_h']))

def plot_two_models(merged_case, ind, ax):
    subset     = get_subset(merged_case, ind, 0, 0)
    daily_sub  = get_daily_subset(subset, periods=1)
    print(daily_sub.index[0].date(),
          'duration = {0} days'.format(len(daily_sub)))
    daily_sub  = daily_sub.query('seepage_visitor < 6')
    daily_sub['seepage']     = daily_sub['seepage_visitor']
    daily_sub['State_visit'] = daily_sub['State_visit'].apply(np.floor)
    daily_sub['delta_h']     = daily_sub.visitor_filled - daily_sub.v
    daily_sub  = daily_sub[['seepage', 'delta_h', 'visitor_h']].dropna()
    daily_sub['date'] = daily_sub.index.date

    lose  = filter_overtop(subset, minval=0.0)
    pos   = daily_sub[~daily_sub.date.isin(lose)].query('seepage > 0')

    if len(pos) < 3:
        return

    dh   = pos['delta_h'].to_numpy(float)
    hv   = pos['visitor_h'].to_numpy(float)
    s    = pos['seepage'].to_numpy(float)
    ord_ = np.argsort(dh)

    ax.scatter(dh, s, c='grey', alpha=0.6, s=15, label='')

    # linear
    C_lin, r2_lin = _fit_linear(dh, s)
    ax.plot(dh[ord_], _model_linear(C_lin, dh[ord_]), color='C0', ls='--',
            label=rf'$S = {C_lin:.2f}\,\Delta h$;  $R^2={r2_lin:.2f}$')

    # offset-power (h_V)
    C_vis, off_vis, b_vis, r2_vis = _fit_offset_power_vis(dh, hv, s, _hv_ref_global)
    ys_vis = _model_offset_power_vis(C_vis, off_vis, b_vis, dh[ord_], hv[ord_], _hv_ref_global)
    ax.plot(dh[ord_], ys_vis, color='C1', ls='--',
            label=(rf'$S = {C_vis:.2f}\,\Delta h\,\zeta_V^{{{b_vis:.2f}}}$;'
                   rf'  $R^2={r2_vis:.2f}$'))

    ax.set_xlabel(r'$\Delta h$')
    ax.axhline(0, c='k', lw=1)
    ax.legend(loc='upper left', handlelength=1, fontsize=12)


# --- build figure -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.ravel()

plot_two_models(merged_case, 0, axes[0])
plot_two_models(merged_case, 1, axes[1])
plot_two_models(merged_case, 2, axes[2])

axes[0].set_xticks([1, 2, 3])
axes[0].set_yticks([1, 2, 3])
axes[0].set_xlim(0.5, 3)
axes[0].set_ylim(0, 4)
axes[0].set_ylabel(r'$S = Q - dV/dt$')
axes[2].set_ylabel(r'$S = Q - dV/dt$')

for i, lab in enumerate('ABCD'):
    axes[i].text(-0.1, 1.05, lab, transform=axes[i].transAxes,
                 ha='left', va='top', fontsize=16, style='italic')

# --- pooled panel (D) ---------------------------------------------------------
ax      = axes[3]
df_pool = pooled_event_df.dropna(subset=['delta_h', 'seepage', 'visitor_h']).sort_values('delta_h')
dh  = df_pool['delta_h'].to_numpy(float)
hv  = df_pool['visitor_h'].to_numpy(float)
s   = df_pool['seepage'].to_numpy(float)

ax.scatter(dh, s, c='grey', alpha=0.5, s=15, label='')

# linear
C_lin, r2_lin = _fit_linear(dh, s)
ax.plot(dh, _model_linear(C_lin, dh), '--',
        label=rf'$S = {C_lin:.2f}\,\Delta h$;  $R^2={r2_lin:.2f}$')

# offset-power (h_V)
C_vis, off_vis, b_vis, r2_vis = _fit_offset_power_vis(dh, hv, s, _hv_ref_global)
ys_vis = _model_offset_power_vis(C_vis, off_vis, b_vis, dh, hv, _hv_ref_global)
ax.plot(dh, ys_vis, '--',
        label=(rf'$S = {C_vis:.2f}\,\Delta h\,\zeta_V^{{{b_vis:.2f}}}$;'
               rf'  $R^2={r2_vis:.2f}$'))

ax.set_xlabel(r'$\Delta h$')
ax.legend(loc='upper left', fontsize=12)
ax.axhline(0, c='k', lw=1)
ax.set_xlim(0.2,)
ax.set_ylim(0., 5)
ax.set_yticks([1, 2, 3, 4])
ax.set_xticks([1, 2, 3])
"""

new_source = CELL_SOURCE.splitlines(keepends=True)

# Find and replace the target cell
found = False
for cell in nb['cells']:
    if cell.get('id') == '57c35077':
        cell['source'] = new_source
        found = True
        break

if found:
    nb_path.write_text(json.dumps(nb, indent=1))
    print("SUCCESS: patched cell 57c35077; notebook written.")
else:
    print("FAILED: cell 57c35077 not found.")
    print("Code cell IDs present:")
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            print(" ", repr(cell.get('id')))
