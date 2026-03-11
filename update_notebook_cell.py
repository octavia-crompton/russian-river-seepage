#!/usr/bin/env python3
"""
Script to update cell #VSC-3e5fb8a9 in the notebook with new code
"""
import json

notebook_path = "/Users/octaviacrompton/Google_Drive_quatratavia/estuaries_jenner/russian-river-seepage/notebooks/2 Russian River  – analysis.ipynb"

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# New cell content
new_code = [
    "def fit_linear_visitor(pos_subset):\n",
    "    \"\"\"\n",
    "    Fit linear model: S = C_L * \u0394h\n",
    "    \"\"\"\n",
    "    X = pos_subset[['delta_h']]\n",
    "    y = pos_subset['seepage'].values\n",
    "    predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)\n",
    "    return {\n",
    "        'C_L': slope,\n",
    "        'R2': r_squared,\n",
    "        'predictions': predictions,\n",
    "        'residuals': residuals,\n",
    "        'CI_low': CI_low,\n",
    "        'CI_high': CI_high,\n",
    "    }\n",
    "\n",
    "def fit_offset_power_visitor(pos_subset, delta_offset_range=None, b_range=None):\n",
    "    \"\"\"\n",
    "    Fit offset-power model: S = C_V * \u03b6_V^b * \u0394h\n",
    "    where \u03b6_V = (h_V + \u03b4_V) / m_V and m_V = median(h_V) + \u03b4_V\n",
    "\n",
    "    Uses grid search to find best \u03b4_V and b_V, then linear regression for C_V\n",
    "    \"\"\"\n",
    "    if delta_offset_range is None:\n",
    "        delta_offset_range = [0, 1.5]\n",
    "    if b_range is None:\n",
    "        b_range = [0.1, 2.0]\n",
    "\n",
    "    h_V = pos_subset['visitor_h'].values\n",
    "    h_V_med = np.median(h_V)\n",
    "    y = pos_subset['seepage'].values\n",
    "    delta_h = pos_subset['delta_h'].values\n",
    "\n",
    "    best_R2 = -np.inf\n",
    "    best_params = None\n",
    "    best_predictions = None\n",
    "\n",
    "    delta_offsets = np.linspace(delta_offset_range[0], delta_offset_range[1], 15)\n",
    "    b_values = np.linspace(b_range[0], b_range[1], 15)\n",
    "\n",
    "    for delta_V in delta_offsets:\n",
    "        m_V = h_V_med + delta_V\n",
    "        zeta_V = (h_V + delta_V) / m_V\n",
    "\n",
    "        for b_V in b_values:\n",
    "            zeta_power = zeta_V ** b_V\n",
    "            X = (zeta_power * delta_h).reshape(-1, 1)\n",
    "            try:\n",
    "                predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)\n",
    "                if r_squared > best_R2:\n",
    "                    best_R2 = r_squared\n",
    "                    best_params = {\n",
    "                        'delta_V': delta_V,\n",
    "                        'b_V': b_V,\n",
    "                        'm_V': m_V,\n",
    "                        'C_V': slope,\n",
    "                        'R2': r_squared,\n",
    "                        'CI_low': CI_low,\n",
    "                        'CI_high': CI_high,\n",
    "                        'residuals': residuals,\n",
    "                    }\n",
    "                    best_predictions = predictions\n",
    "            except:\n",
    "                continue\n",
    "\n",
    "    if best_params is not None:\n",
    "        best_params['predictions'] = best_predictions\n",
    "\n",
    "    return best_params\n",
    "\n",
    "def plot_visitor_linear_vs_offset(merged_case, ind, ax):\n",
    "    \"\"\"\n",
    "    Plot linear vs offset-power model fits for visitor center data\n",
    "    \"\"\"\n",
    "    subset = get_subset(merged_case, ind, 0, 0)\n",
    "\n",
    "    daily_subset = get_daily_subset(subset, periods=1)\n",
    "    print(daily_subset.index[0].date(), \"duration = {0} days\".format(len(daily_subset)))\n",
    "    daily_subset = daily_subset.query(\"seepage_visitor < 6\")\n",
    "\n",
    "    daily_subset['seepage'] = daily_subset['seepage_visitor']\n",
    "    daily_subset['State_visit'] = daily_subset['State_visit'].apply(np.floor)\n",
    "\n",
    "    daily_subset['delta_h'] = daily_subset.visitor_filled - daily_subset.v\n",
    "    daily_subset = daily_subset[['seepage', 'delta_h', 'waveHs', 'visitor_h']].dropna()\n",
    "    daily_subset['date'] = daily_subset.index.date\n",
    "\n",
    "    lose = filter_overtop(subset, minval=0.0)\n",
    "    pos_subset = daily_subset[~(daily_subset.date).isin(lose)]\n",
    "    pos_subset = pos_subset.query(\"seepage > 0\")\n",
    "\n",
    "    if len(pos_subset) < 5:\n",
    "        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',\n",
    "                transform=ax.transAxes)\n",
    "        return None, None\n",
    "\n",
    "    linear_fit = fit_linear_visitor(pos_subset)\n",
    "    offset_power_fit = fit_offset_power_visitor(pos_subset)\n",
    "\n",
    "    ax.scatter(pos_subset['delta_h'], pos_subset['seepage'], c='grey',\n",
    "               label='data', alpha=0.6, s=15)\n",
    "\n",
    "    sorted_x = np.sort(pos_subset['delta_h'].values)\n",
    "    ax.plot(sorted_x, linear_fit['C_L'] * sorted_x, 'C0--', linewidth=2,\n",
    "            label=f\"Linear: $C_L$ = {linear_fit['C_L']:.2f}; $R^2$ = {linear_fit['R2']:.2f}\")\n",
    "\n",
    "    if offset_power_fit is not None:\n",
    "        h_V = pos_subset['visitor_h'].values\n",
    "        delta_V = offset_power_fit['delta_V']\n",
    "        b_V = offset_power_fit['b_V']\n",
    "        m_V = offset_power_fit['m_V']\n",
    "        C_V = offset_power_fit['C_V']\n",
    "\n",
    "        sorted_idx = np.argsort(pos_subset['delta_h'].values)\n",
    "        sorted_x = pos_subset['delta_h'].values[sorted_idx]\n",
    "        sorted_h_V = h_V[sorted_idx]\n",
    "        zeta_V = (sorted_h_V + delta_V) / m_V\n",
    "        fitted_y = C_V * (zeta_V ** b_V) * sorted_x\n",
    "\n",
    "        ax.plot(sorted_x, fitted_y, 'C1--', linewidth=2,\n",
    "                label=f\"Power: $C_V$ = {C_V:.2f}, $\\\\delta_V$ = {delta_V:.2f}, $b_V$ = {b_V:.2f}; $R^2$ = {offset_power_fit['R2']:.2f}\")\n",
    "\n",
    "    ax.set_xlabel('$\\\\Delta h$ (m)', fontsize=12)\n",
    "    ax.set_ylabel('$S$ (m\u00b3/day)', fontsize=12)\n",
    "    ax.legend(loc='upper left', handlelength=1, fontsize=9)\n",
    "    ax.axhline(0, c='k', lw=1)\n",
    "    ax.grid(True, alpha=0.3)\n",
    "\n",
    "    return linear_fit, offset_power_fit\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
    "\n",
    "for i in range(3):\n",
    "    lin_fit, off_fit = plot_visitor_linear_vs_offset(merged_case, i, axes[i])\n",
    "    axes[i].text(-0.1, 1.05, chr(65+i), transform=axes[i].transAxes, ha=\"left\", va=\"top\",\n",
    "                 fontsize=16, style='italic')\n",
    "\n",
    "axes[0].set_xlim(0.2, 3)\n",
    "axes[0].set_ylim(-0.5, 5)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]

old_code = """def fit_linear_visitor(pos_subset):
    \"\"\"
    Fit linear model: S = C_L * Δh
    \"\"\"
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
    \"\"\"
    Fit offset-power model: S = C_V * ζ_V^b * Δh
    where ζ_V = (h_V + δ_V) / m_V and m_V = median(h_V) + δ_V
    
    Uses grid search to find best δ_V and b_V, then linear regression for C_V
    \"\"\"
    if delta_offset_range is None:
        delta_offset_range = [0, 1.5]  # Reasonable range for vertical offset
    if b_range is None:
        b_range = [0.1, 2.0]  # Range for exponent
    
    h_V = pos_subset['visitor_h'].values
    h_V_med = np.median(h_V)
    y = pos_subset['seepage'].values
    delta_h = pos_subset['delta_h'].values
    
    best_R2 = -np.inf
    best_params = None
    best_predictions = None
    
    # Grid search over δ_V and b_V
    delta_offsets = np.linspace(delta_offset_range[0], delta_offset_range[1], 15)
    b_values = np.linspace(b_range[0], b_range[1], 15)
    
    for delta_V in delta_offsets:
        m_V = h_V_med + delta_V
        zeta_V = (h_V + delta_V) / m_V
        
        for b_V in b_values:
            zeta_power = zeta_V ** b_V
            # Fit: S = C_V * ζ_V^b * Δh
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
    \"\"\"
    Plot linear vs offset-power model fits for visitor center data
    \"\"\"
    subset = get_subset(merged_case, ind, 0, 0)
    
    daily_subset = get_daily_subset(subset, periods=1)
    print (daily_subset.index[0].date(), "duration = {0} days" .format( len(daily_subset)))
    daily_subset = daily_subset.query("seepage_visitor < 6")
    
    daily_subset['seepage'] = daily_subset['seepage_visitor']
    daily_subset['State_visit'] = daily_subset['State_visit'].apply(np.floor)
    
    # Calculate Δh
    daily_subset['delta_h'] = daily_subset.visitor_filled - daily_subset.v
    daily_subset = daily_subset[['seepage', 'delta_h', 'waveHs', 'visitor_h']].dropna()
    daily_subset['date'] = daily_subset.index.date
    
    # Filter out overtopping days
    lose = filter_overtop(subset, minval=0.0)
    pos_subset = daily_subset[~(daily_subset.date).isin(lose)]
    pos_subset = pos_subset.query("seepage > 0")
    
    if len(pos_subset) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        return None, None
    
    # Fit linear model
    linear_fit = fit_linear_visitor(pos_subset)
    
    # Fit offset-power model
    offset_power_fit = fit_offset_power_visitor(pos_subset)
    
    # Plot data
    ax.scatter(pos_subset['delta_h'], pos_subset['seepage'], c='grey', 
               label='data', alpha=0.6, s=15)
    
    # Plot linear model
    sorted_x = np.sort(pos_subset['delta_h'].values)
    ax.plot(sorted_x, linear_fit['C_L'] * sorted_x, 'C0--', linewidth=2,
            label=f"Linear: $C_L$ = {linear_fit['C_L']:.2f}; $R^2$ = {linear_fit['R2']:.2f}")
    
    # Plot offset-power model if fit succeeded
    if offset_power_fit is not None:
        h_V = pos_subset['visitor_h'].values
        h_V_med = np.median(h_V)
        delta_V = offset_power_fit['delta_V']
        b_V = offset_power_fit['b_V']
        m_V = offset_power_fit['m_V']
        C_V = offset_power_fit['C_V']
        
        # Generate fitted curve
        sorted_idx = np.argsort(pos_subset['delta_h'].values)
        sorted_x = pos_subset['delta_h'].values[sorted_idx]
        sorted_h_V = h_V[sorted_idx]
        zeta_V = (sorted_h_V + delta_V) / m_V
        fitted_y = C_V * (zeta_V ** b_V) * sorted_x
        
        ax.plot(sorted_x, fitted_y, 'C1--', linewidth=2,
                label=f"Power: $C_V$ = {C_V:.2f}, $\\\\delta_V$ = {delta_V:.2f}, $b_V$ = {b_V:.2f}; $R^2$ = {offset_power_fit['R2']:.2f}")
    
    ax.set_xlabel('$\\\\Delta h$ (m)', fontsize=12)
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
plt.show()"""

# Find the cell with id VSC-3e5fb8a9
found = False
for i, cell in enumerate(notebook['cells']):
    if 'id' in cell and cell['id'] == 'VSC-3e5fb8a9':
        print(f"Found cell at index {i}")
        print(f"Current cell type: {cell.get('cell_type')}")
        print(f"Current source length: {len(cell.get('source', []))} lines")
        
        # Update the cell source
        cell['source'] = new_code.split('\n')
        # Add newlines back
        cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                         for i, line in enumerate(cell['source'])]
        found = True
        print("Cell updated")
        break

if not found:
    print("Cell VSC-3e5fb8a9 not found!")
    print("Available cell IDs:")
    for cell in notebook['cells']:
        if 'id' in cell:
            print(f"  - {cell['id']}")
else:
    # Write back to file
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook saved to {notebook_path}")
