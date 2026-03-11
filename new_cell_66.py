def fit_linear_visitor(pos_subset):
    """
    Fit linear model: S = C_L * Δh
    """
    X = pos_subset[['delta_h']]
    y = pos_subset['seepage'].values
    predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)
    return {
        'C_L': slope,
        'R2': r_squared,
        'predictions': predictions,
        'residuals': residuals,
        'CI_low': CI_low,
        'CI_high': CI_high,
    }

def fit_offset_power_visitor(pos_subset, delta_offset_range=None, b_range=None):
    """
    Fit offset-power model: S = C_V * ζ_V^b * Δh
    where ζ_V = (h_V + δ_V) / m_V and m_V = median(h_V) + δ_V

    Uses grid search to find best δ_V and b_V, then linear regression for C_V
    """
    if delta_offset_range is None:
        delta_offset_range = [0, 1.5]
    if b_range is None:
        b_range = [0.1, 2.0]

    h_V = pos_subset['visitor_h'].values
    h_V_med = np.median(h_V)
    y = pos_subset['seepage'].values
    delta_h = pos_subset['delta_h'].values

    best_R2 = -np.inf
    best_params = None
    best_predictions = None

    delta_offsets = np.linspace(delta_offset_range[0], delta_offset_range[1], 15)
    b_values = np.linspace(b_range[0], b_range[1], 15)

    for delta_V in delta_offsets:
        m_V = h_V_med + delta_V
        zeta_V = (h_V + delta_V) / m_V

        for b_V in b_values:
            zeta_power = zeta_V ** b_V
            X = (zeta_power * delta_h).reshape(-1, 1)
            try:
                predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)
                if r_squared > best_R2:
                    best_R2 = r_squared
                    best_params = {
                        'delta_V': delta_V,
                        'b_V': b_V,
                        'm_V': m_V,
                        'C_V': slope,
                        'R2': r_squared,
                        'CI_low': CI_low,
                        'CI_high': CI_high,
                        'residuals': residuals,
                    }
                    best_predictions = predictions
            except:
                continue

    if best_params is not None:
        best_params['predictions'] = best_predictions

    return best_params

def plot_visitor_linear_vs_offset(merged_case, ind, ax):
    """
    Plot linear vs offset-power model fits for visitor center data
    """
    subset = get_subset(merged_case, ind, 0, 0)

    daily_subset = get_daily_subset(subset, periods=1)
    print(daily_subset.index[0].date(), "duration = {0} days".format(len(daily_subset)))
    daily_subset = daily_subset.query("seepage_visitor < 6")

    daily_subset['seepage'] = daily_subset['seepage_visitor']
    daily_subset['State_visit'] = daily_subset['State_visit'].apply(np.floor)

    daily_subset['delta_h'] = daily_subset.visitor_filled - daily_subset.v
    daily_subset = daily_subset[['seepage', 'delta_h', 'waveHs', 'visitor_h']].dropna()
    daily_subset['date'] = daily_subset.index.date

    lose = filter_overtop(subset, minval=0.0)
    pos_subset = daily_subset[~(daily_subset.date).isin(lose)]
    pos_subset = pos_subset.query("seepage > 0")

    if len(pos_subset) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        return None, None

    linear_fit = fit_linear_visitor(pos_subset)
    offset_power_fit = fit_offset_power_visitor(pos_subset)

    ax.scatter(pos_subset['delta_h'], pos_subset['seepage'], c='grey',
               label='data', alpha=0.6, s=15)

    sorted_x = np.sort(pos_subset['delta_h'].values)
    ax.plot(sorted_x, linear_fit['C_L'] * sorted_x, 'C0--', linewidth=2,
            label=f"Linear: $C_L$ = {linear_fit['C_L']:.2f}; $R^2$ = {linear_fit['R2']:.2f}")

    if offset_power_fit is not None:
        h_V = pos_subset['visitor_h'].values
        delta_V = offset_power_fit['delta_V']
        b_V = offset_power_fit['b_V']
        m_V = offset_power_fit['m_V']
        C_V = offset_power_fit['C_V']

        sorted_idx = np.argsort(pos_subset['delta_h'].values)
        sorted_x = pos_subset['delta_h'].values[sorted_idx]
        sorted_h_V = h_V[sorted_idx]
        zeta_V = (sorted_h_V + delta_V) / m_V
        fitted_y = C_V * (zeta_V ** b_V) * sorted_x

        ax.plot(sorted_x, fitted_y, 'C1--', linewidth=2,
                label=f"Power: $C_V$ = {C_V:.2f}, $\\delta_V$ = {delta_V:.2f}, $b_V$ = {b_V:.2f}; $R^2$ = {offset_power_fit['R2']:.2f}")

    ax.set_xlabel('$\\Delta h$ (m)', fontsize=12)
    ax.set_ylabel('$S$ (m³/day)', fontsize=12)
    ax.legend(loc='upper left', handlelength=1, fontsize=9)
    ax.axhline(0, c='k', lw=1)
    ax.grid(True, alpha=0.3)

    return linear_fit, offset_power_fit

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for i in range(3):
    lin_fit, off_fit = plot_visitor_linear_vs_offset(merged_case, i, axes[i])
    axes[i].text(-0.1, 1.05, chr(65+i), transform=axes[i].transAxes, ha="left", va="top",
                 fontsize=16, style='italic')

axes[0].set_xlim(0.2, 3)
axes[0].set_ylim(-0.5, 5)

plt.tight_layout()
plt.show()
