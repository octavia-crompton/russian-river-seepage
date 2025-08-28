import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from datetime import timedelta

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


from pathlib import Path
import pandas as pd

# One level up from *this* file/notebook
try:
    # Works in .py files
    here = Path(__file__).resolve().parent
except NameError:
    # Works in notebooks (no __file__)
    here = Path.cwd().resolve()

root = here.parent  # ← exactly one level up

hypso_path = root / "data" / "RR_stage_storage.xlsx"
hypso = pd.read_excel(hypso_path, header=3)
hypso.columns = ["h_m", "vol_m3", "A_m2"]
hypso["h_m"] = hypso["h_m"] + 2.73 * 0.3048



def get_closures(patches ):

    closures = pd.DataFrame()

    for ind in range(len(patches.keys())):

        patch = get_subset(ind, 0, 0, patches)[1][['State_visit']]
        try:
            start = patch.index[0].date()
        except: 
            continue
        end = patch.index[-1].date()
        duration = len(patch)
        closures = pd.concat([closures,
                             pd.Series({'start' : start, 'end' : end, 'duration' : duration})
                            ], axis = 1)
    closures = closures.T.reset_index()    
    
    return closures




## Add hypso data
def add_vol(h_name, subset, name = "vol_m3"):
    """
    add hypsometry...
    """
    for i, ind in enumerate(subset[h_name].dropna().index):
        diff = (subset.loc[ind][h_name] - hypso.h_m).dropna().abs()
        subset.at[ind, name] = hypso.iloc[diff.idxmin()].vol_m3
    return subset



def get_daily_subset(subset, include_USGS = 0, periods = 1):
    """
    compute daily data from merged daily data
    """
    # groupby date and compute mean
    daily_subset = subset.groupby(subset.date).mean()
    daily_subset.index = pd.to_datetime(daily_subset.index)

    # fill visitor gaps up to 1 day
    result = sequence_lengths(daily_subset.visitor_h.isna())
    daily_subset['visitor_gap'] = result
    daily_subset['visitor_filled'] = daily_subset['visitor_h'].interpolate(
        method='polynomial', order=1)
    daily_subset.loc[daily_subset.query('visitor_gap > 1').index,  'visitor_filled'] = np.nan

    daily_subset['dh/dt_visitor'] = daily_subset.visitor_filled.diff(periods = periods)/periods

    # add estuary volume
    daily_subset = add_vol("visitor_h", daily_subset, name = "vol_visitor")
    daily_subset = add_vol("visitor_filled", daily_subset, name = "vol_visitor_filled")

    # change in estuary volume (m3/day)
    daily_subset['dvol/dt_visitor'] = daily_subset.vol_visitor.diff(
        periods = periods)/3600/24/periods
    daily_subset['dvol/dt_visitor_filled'] = daily_subset.vol_visitor_filled.diff(
        periods = periods)/3600/24/periods

    # seepage estimate (m3/day)
    daily_subset['seepage_visitor'] = (daily_subset['Q'] - daily_subset['dvol/dt_visitor_filled']
                                      ).where(daily_subset['State_visit'] == 1)

    # predict volume in one day, using discharge
    daily_subset['vol_p_1hr'] = daily_subset.vol_visitor_filled+daily_subset.Q*3600*24  # m3

    for i, ind in enumerate(daily_subset['vol_p_1hr'].dropna().index[:-1]):

        # convert predicted volume to predicted heigh, using hypose
        diff = (daily_subset.loc[ind]['vol_p_1hr'] - hypso.vol_m3 ).dropna().abs()

        # look 24 hours into the future
        advance = daily_subset.iloc[np.where(daily_subset.index == ind)[0][0] + 1].name
        # add predicted depth to daily subset
        daily_subset.loc[advance, 'visitor_h_Q'] = hypso.iloc[diff.idxmin()].h_m

    # compute predicted minus actual visitor h.
    if 'visitor_h_Q' in daily_subset.columns:
        daily_subset['h_hat_minus_h'] = daily_subset[ 'visitor_h_Q'] - daily_subset[ 'visitor_h']

    if include_USGS == 1:
        # fill USGS gaps up to 1 day
        result = sequence_lengths(daily_subset.USGS_h.isna())
        daily_subset['USGS_gap'] = result
        daily_subset['USGS_filled'] = daily_subset['USGS_h'].interpolate(
            method='polynomial', order=1)
        daily_subset.loc[daily_subset.query('USGS_gap > 1').index,
                         'USGS_filled'] = np.nan

        daily_subset['dh/dt_USGS'] = daily_subset.USGS_filled.diff(periods = periods)/periods

        daily_subset = add_vol("USGS_h", daily_subset, name = "vol_USGS")
        daily_subset = add_vol("USGS_filled", daily_subset, name = "vol_USGS_filled")

        daily_subset['dvol/dt_USGS'] = daily_subset.vol_USGS.diff(periods = periods)/3600/24/periods
        daily_subset['dvol/dt_USGS_filled'] = daily_subset.vol_USGS_filled.diff(
            periods = periods)/3600/24/periods

        # seepage estimate (m3/day)
        daily_subset['seepage_USGS'] = (daily_subset['Q'] - daily_subset['dvol/dt_USGS_filled']
                                      ).where(daily_subset['State_visit'] == 1)


        daily_subset['USGS-visitor'] = daily_subset['USGS_h'] - daily_subset['visitor_h']

    daily_subset.index = daily_subset.index.tz_localize('UCT')

    return daily_subset


def get_daily_subset_USGS(subset, periods = 1):
    """
    compute daily data from merged daily data
    """
    # groupby date and compute mean
    daily_subset = subset.groupby(subset.date).mean()
    daily_subset.index = pd.to_datetime(daily_subset.index)

    # fill USGS gaps up to 1 day
    result = sequence_lengths(daily_subset.USGS_h.isna())
    daily_subset['USGS_gap'] = result
    daily_subset['USGS_filled'] = daily_subset['USGS_h'].interpolate(
        method='polynomial', order=1)
    daily_subset.loc[daily_subset.query('USGS_gap > 1').index,
                     'USGS_filled'] = np.nan

    daily_subset['dh/dt_USGS'] = daily_subset.USGS_filled.diff(periods = periods)/periods

    daily_subset = add_vol("USGS_h", daily_subset, name = "vol_USGS")
    daily_subset = add_vol("USGS_filled", daily_subset, name = "vol_USGS_filled")

    daily_subset['dvol/dt_USGS'] = daily_subset.vol_USGS.diff(periods = periods)/3600/24/periods
    daily_subset['dvol/dt_USGS_filled'] = daily_subset.vol_USGS_filled.diff(
        periods = periods)/3600/24/periods

    # seepage estimate (m3/day)
    daily_subset['seepage_USGS'] = (daily_subset['Q'] - daily_subset['dvol/dt_USGS_filled']
                                  ).where(daily_subset['State_visit'] == 1)


    daily_subset['USGS-visitor'] = daily_subset['USGS_h'] - daily_subset['visitor_h']

    daily_subset.index = daily_subset.index.tz_localize('UCT')

    return daily_subset

def get_daily_subset_USGS(subset, periods = 1):
    """
    compute daily data from merged daily data
    """
    # groupby date and compute mean
    daily_subset = subset.groupby(subset.date).mean()
    daily_subset.index = pd.to_datetime(daily_subset.index)

    # fill USGS gaps up to 1 day
    result = sequence_lengths(daily_subset.USGS_h.isna())
    daily_subset['USGS_gap'] = result
    daily_subset['USGS_filled'] = daily_subset['USGS_h'].interpolate(
        method='polynomial', order=1)
    daily_subset.loc[daily_subset.query('USGS_gap > 1').index,
                     'USGS_filled'] = np.nan

    daily_subset['dh/dt_USGS'] = daily_subset.USGS_filled.diff(periods = periods)/periods

    daily_subset = add_vol("USGS_h", daily_subset, name = "vol_USGS")
    daily_subset = add_vol("USGS_filled", daily_subset, name = "vol_USGS_filled")

    daily_subset['dvol/dt_USGS'] = daily_subset.vol_USGS.diff(periods = periods)/3600/24/periods
    daily_subset['dvol/dt_USGS_filled'] = daily_subset.vol_USGS_filled.diff(
        periods = periods)/3600/24/periods

    # seepage estimate (m3/day)
    daily_subset['seepage_USGS'] = (daily_subset['Q'] - daily_subset['dvol/dt_USGS_filled']
                                  ).where(daily_subset['State_visit'] == 1)


    daily_subset['USGS-visitor'] = daily_subset['USGS_h'] - daily_subset['visitor_h']

    daily_subset.index = daily_subset.index.tz_localize('UCT')

    return daily_subset



def find_patches(nums):
    """
    identify closures by duration
    """
    if not nums:
        return {}

    patches = {}
    in_patch = False
    start_index = -1

    for i, num in enumerate(nums):
        if num == 1 and not in_patch:
            start_index = i
            patches[start_index] = 1
            in_patch = True
        elif num == 1 and in_patch:
            patches[start_index] += 1
        elif num == 0 and in_patch:
            in_patch = False

    # Sort patches by length in descending order and return as a dictionary
    sorted_patches = {k: v for k, v in sorted(patches.items(), key=lambda item: item[1], reverse=True)}

    return sorted_patches



def get_subset(merged, ind, prev, post):

    patches = find_patches(list(merged.State_visit))

    duration = patches[list(patches.keys())[ind]]

    start = list(patches.keys())[ind] - prev*24*4
    end = list(patches.keys())[ind] + duration + post*24*4

    start_date = merged.iloc[start].name
    end_date = merged.iloc[end].name
    subset = merged.iloc[start:end]

    return subset

def get_subset_USGS(merged, ind, prev, post):

    patches = find_patches(list(merged.State_USGS))

    duration = patches[list(patches.keys())[ind]]

    start = list(patches.keys())[ind] - prev*24*4
    end = list(patches.keys())[ind] + duration + post*24*4

    start_date = merged.iloc[start].name
    end_date = merged.iloc[end].name
    subset = merged.iloc[start:end]

    return subset

def get_visitor_closures(merged):

    patches = find_patches(list(merged.State_visit))
    closures = pd.DataFrame()

    for ind in range(92):

        subset = get_subset(merged, ind, 0, 0)
        daily_subset = get_daily_subset(subset, include_USGS = False, periods = 1)

        patch = daily_subset[['State']]
        try:
            start = patch.index[0].date()
        except:
            continue
        end = patch.index[-1].date()
        duration = len(patch)
        closures = pd.concat([closures,
                             pd.Series({
                                 'start' : start, 'end' : end,
                                 'year' : daily_subset.index[0].year,
                                 'duration' : duration})
                            ], axis = 1)
    closures = closures.T.reset_index()

    return closures.drop("index", axis = 1)


def filter_overtop_hourly(subset):

    hourly_subset = get_hourly_visitor(subset)
    lose = hourly_subset[(hourly_subset['visitor_h_Q']- hourly_subset['visitor_h']) <  .0].index
    lose = np.unique(lose)
    one_day = timedelta(days=1)
    dates_plus_one = [date + one_day for date in lose]

    return list(lose)

def get_hourly_visitor(subset):
    """
    compute hourly from 15 minute data
    """
    hourly_subset = subset.groupby(pd.Grouper(freq='H'))[['State_visit', 'visitor_h', 'Q', 'visitor_12_hr']].median()
    hourly_subset.index = pd.to_datetime(hourly_subset.index)

    # fill gaps less than 2 hours
    result = sequence_lengths(hourly_subset.visitor_h.isna())
    hourly_subset['visitor_gap'] = result
    hourly_subset['visitor_filled'] = hourly_subset['visitor_h'].interpolate(method='polynomial', order=1)
    hourly_subset.loc[hourly_subset.query('visitor_gap > 3').index, 'visitor_filled'] = np.nan

    # dh/dt in m/day
    hourly_subset['dh_dt_visitor'] = hourly_subset.visitor_h.diff(periods = 24)

    hourly_subset = add_vol("visitor_h", hourly_subset, name = "vol_visitor")
    hourly_subset = add_vol("visitor_filled", hourly_subset, name = "vol_visitor_filled")

    # predict volume in one day, using discharge
    hourly_subset['vol_p_1day'] = hourly_subset.vol_visitor_filled+hourly_subset.Q*3600*24

    for i, ind in enumerate(hourly_subset['vol_p_1day'][:-24].dropna().index):
        # look 24 hours into the future
        advance = hourly_subset.iloc[np.where(hourly_subset.index == ind)[0][0] + 24].name

        # convert predicted volume to predicted height, using hypsometry
        # first find nearest point in hypsometry data
        diff = (hourly_subset.loc[ind]['vol_p_1day'] - hypso.vol_m3 ).dropna().abs()

        # add predicted depth to hourly subset
        hourly_subset.loc[advance, 'visitor_h_Q'] = hypso.iloc[diff.idxmin()].h_m

    # compute d(volume)/dt in m3/day and convert to m3/s
    hourly_subset['dvol/dt_visitor'] = hourly_subset.vol_visitor.diff(periods = 24)/3600/24   # m3/s
    hourly_subset['dvol/dt_visitor_filled'] = hourly_subset.vol_visitor_filled.diff(periods = 24)/3600/24

    if  'visitor_h_Q' in hourly_subset.columns:
        hourly_subset['h_hat_minus_h'] = hourly_subset['visitor_h_Q'] - hourly_subset[ 'visitor_h']

    hourly_subset['seepage_visitor'] = (hourly_subset['Q'] - hourly_subset['dvol/dt_visitor_filled'])

    return hourly_subset

def get_hourly_USGS(subset):
    """
    compute hourly from 15 minute data

    """
    hourly_subset = subset.groupby(pd.Grouper(freq='H'))[['State_visit', 'USGS_h', 'Q', 'USGS_12_hr']].median()
    hourly_subset.index = pd.to_datetime(hourly_subset.index)

    # fill gaps less than 2 hours
    result = sequence_lengths(hourly_subset.USGS_h.isna())
    hourly_subset['USGS_gap'] = result
    hourly_subset['USGS_filled'] = hourly_subset['USGS_h'].interpolate(method='polynomial', order=1)
    hourly_subset.loc[hourly_subset.query('USGS_gap > 3').index, 'USGS_filled'] = np.nan

    # dh/dt in m/day
    hourly_subset['dh_dt_USGS'] = hourly_subset.USGS_h.diff(periods = 24)

    hourly_subset = add_vol("USGS_h", hourly_subset, name = "vol_USGS")
    hourly_subset = add_vol("USGS_filled", hourly_subset, name = "vol_USGS_filled")

    # predict volume in one hour, using discharge
    hourly_subset['vol_p_1hr'] = hourly_subset.vol_USGS_filled+hourly_subset.Q*3600*24

    for i, ind in enumerate(hourly_subset['vol_p_1hr'][:-24].dropna().index):

        # convert predicted volume to predicted heigh, using hypose
        diff = (hourly_subset.loc[ind]['vol_p_1hr'] - hypso.vol_m3 ).dropna().abs()
        # look 24 hours into the future
        advance = hourly_subset.iloc[np.where(hourly_subset.index == ind)[0][0] + 24].name
        # add predicted depth to hourly subset
        hourly_subset.loc[advance, 'USGS_h_Q'] = hypso.iloc[diff.idxmin()].h_m

    # compute d(volume)/dt in m3/day and convert to m3/s
    hourly_subset['dvol/dt_USGS'] = hourly_subset.vol_USGS.diff(periods = 24)/3600/24   # m3/s
    hourly_subset['dvol/dt_USGS_filled'] = hourly_subset.vol_USGS_filled.diff(periods = 24)/3600/24

    if  'USGS_h_Q' in hourly_subset.columns:
        hourly_subset['h_hat_minus_h'] = hourly_subset[ 'USGS_h_Q'] - hourly_subset[ 'USGS_h']

    hourly_subset['seepage_USGS'] = (hourly_subset['Q'] - hourly_subset['dvol/dt_USGS_filled'])

    return hourly_subset

### Plotting functions

def scale_series(series, maxval = 0.5):
    # Scale each series (column) from -1 to 1
    return series/(series.max())*maxval



def get_shading_regions(series, threshold):
    """
    """
    regions = []
    start = None
    for i, val in enumerate(series):
        if val < threshold and start is None:
            start = series.index[max(i-1, 0)]
        elif val >= threshold and start is not None:
            regions.append((start, series.index[i]))
            start = None
    if start is not None:
        regions.append((start, series.index[-1]))
    return regions

def get_ylim(daily_subset):
    """
    ylimits for plot_visitor_scatter
    """
    Qmax = np.percentile(daily_subset['Q'].dropna(), 99)
    dvol_max = np.percentile(daily_subset['dvol/dt_visitor_filled'].dropna(), 99)
    seep_max =np.percentile(daily_subset['seepage_visitor'].dropna(), 100)

    ymax = max(Qmax, dvol_max, seep_max )*1.1

    Qmin =np.percentile(daily_subset['Q'].dropna(), 0)
    dvol_min = np.percentile(daily_subset['dvol/dt_visitor_filled'].dropna(), 0)
    seep_min =np.percentile(daily_subset['seepage_visitor'].dropna(), 0)

    ymin = min(Qmin, dvol_min, seep_min )*1.1
    ymin = min(ymin, 0)

    return ymin, ymax



def get_ylim_visitor(hourly_subset):

    Qmax = np.percentile(hourly_subset['Q'].dropna(), 99)
    dvol_max = np.percentile(hourly_subset['dvol/dt_visitor_filled'].dropna(), 100)
    seep_max =np.percentile(hourly_subset['seepage_visitor'].dropna(), 100)

    ymax = max(Qmax, dvol_max, seep_max )*1.1

    if (Qmax > dvol_max) and (Qmax > seep_max):
        if Qmax > 12:
            ymax = 12

    Qmin =np.percentile(hourly_subset['Q'].dropna(), 1)
    dvol_min = np.percentile(hourly_subset['dvol/dt_visitor_filled'].dropna(), 1)
    seep_min =np.percentile(hourly_subset['seepage_visitor'].dropna(), 0)

    ymin = min(Qmin, dvol_min, seep_min )

    return ymin, ymax

def plot_mass_bal_hourly(merged, ind, ax, padding = 10, logscale = 1, power = 1):
    """
    """

    subset =  get_subset(merged, ind, padding, padding)
    closure_subset =  get_subset(merged, ind, 0, 0)
    closure_subset = closure_subset

    hourly_subset = get_hourly_visitor(closure_subset.query("State_visit == 1"))
    hourly_subset = hourly_subset.query("seepage_visitor < 3")

    hourly_subset['date'] = hourly_subset.index.date
    hourly_subset['day'] = np.arange(len(hourly_subset))/24
    lose = filter_overtop_hourly(hourly_subset)

    # fixes x-axis limits
    subset[[ 'v']].rename({'v' : ''}, axis= 1).plot(ax = ax, visible = False)

    # Create a boolean mask where 'State_visit' > 0
    closed = hourly_subset['State_visit'] > 0
    closed = hourly_subset[closed].index

    #hourly_subset = hourly_subset.groupby(hourly_subset.date).mean().reset_index()
    hourly_subset[['Q']].plot(secondary_y=False, xlabel='Time', style='C0.--',
                        legend=True, ax = ax, label = '$Q$')

    (hourly_subset['dvol/dt_visitor_filled']).plot(xlabel='Time', ylabel='m3/s',
                            style = 'C1.--', ax = ax, label = '$dV/dt$')

    (hourly_subset['seepage_visitor']).plot(xlabel='Time', ylabel='m3/s',
                                    style = 'C2:', ax = ax, label = '')

    pos_subset = hourly_subset[~(hourly_subset.index).isin(lose)].query("seepage_visitor < 8")
    pos_subset = pos_subset[['day', 'date', 'seepage_visitor']].dropna()

    (pos_subset[['seepage_visitor']]).plot(xlabel='Time', ylabel='m3/s',
                                    style = 'C2o', ax = ax, label = '$S = Q - dV/dt$')

    # plot regression
    X = pos_subset[['day']]
    y =  pos_subset['seepage_visitor']
    predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y,
                                                                                fit_intercept = True)
    pos_subset['predictions'] = predictions

    (pos_subset['predictions']).plot(style = 'C2--', ax = ax,
        label=f'S = {slope:.2f}$t$ + {intercept:.2f};  $R^2$ = {r_squared:.2f}')


    # Add shading wherever power_12_hour is less than threshold
    for start, end in get_shading_regions(subset['State_visit'], 0.2)[1:]:
        ax.axvspan(start, end, color='C0', alpha=0.1)

    for start, end in get_shading_regions(subset['State_visit'], 0.2)[:1]:
        ax.axvspan(start, end, color='C0', alpha=0.1, label = "Open")

    ax.legend(title = '', loc='center left', bbox_to_anchor=(1, 0.5));


    # label axes
    ax.axhline(0, c = 'grey', lw = 1)

    ymin, ymax = get_ylim_visitor(hourly_subset.loc[closed])
    ax.set_ylim(ymin,ymax)

    ax.set_ylabel("m3/s")
    ax.set_title("{0:.0f} day closure from {1} to {2}".format(
                 len(hourly_subset)/24,
                 closed[0].date().strftime("%b %d"),
                closed[-1].date().strftime("%b %d, %Y")))

    return ax


## Summary plots

    
def plot_seepage_ratio(summary_subset, ax, label = None, alpha = 1):

    ax.errorbar(summary_subset.date, summary_subset['SQ'],  alpha = alpha, label = label,
                yerr = (summary_subset['SQ_CI_high'] - summary_subset['SQ_CI_low'])/4,
                linestyle = '', marker = 'o')

    # Identify the range of years
    start_year =   min(summary_subset.year.min(), summary_subset.year.min())
    end_year = max(summary_subset.year.max(), summary_subset.year.max())

    # Add vertical lines for every May
    for year in range(start_year, end_year + 1):
        ax.axvline(pd.to_datetime(f'{year}-01-01'), color='grey', 
            linestyle='--', linewidth=0.8)

    ax.set_xlabel("date")
    ax.set_ylabel("$S/Q_r$")
    start = pd.to_datetime('2012-01-01').tz_localize("UCT")
    end = pd.to_datetime('2023-01-30').tz_localize("UCT")
    ax.set_xlim(start, end)



def plot_Q(summary_subset, ax):

    ax.errorbar(summary_subset.date, summary_subset['Q'],  color = 'C0',
                yerr = (summary_subset['Q_max'] - summary_subset['Q_min'])/4,
                linestyle = '', marker = 'o')

    # Identify the range of years
    start_year =   min(summary_subset.year.min(), summary_subset.year.min())
    end_year = max(summary_subset.year.max(), summary_subset.year.max())

    # Add vertical lines for every May
    for year in range(start_year, end_year + 1):
        ax.axvline(pd.to_datetime(f'{year}-01-01'), color='grey', 
            linestyle='--', linewidth=0.8)

    #ax.set_xlabel("date")
    ax.set_ylabel("$Q_r$")
    start = pd.to_datetime('2012-01-01').tz_localize("UCT")
    end = pd.to_datetime('2023-01-30').tz_localize("UCT")
    ax.set_xlim(start, end)

def plot_delta(summary_subset, ax):

    ax.errorbar(summary_subset.date, summary_subset['delta_h'],  color = 'C0',
                yerr = (summary_subset['delta_h_max'] - summary_subset['delta_h_min'])/4,
                linestyle = '', marker = 'o')

    # Identify the range of years
    start_year =   min(summary_subset.year.min(), summary_subset.year.min())
    end_year = max(summary_subset.year.max(), summary_subset.year.max())

    # Add vertical lines for every May
    for year in range(start_year, end_year + 1):
        ax.axvline(pd.to_datetime(f'{year}-01-01'), color='grey', 
            linestyle='--', linewidth=0.8)

    ax.set_xlabel("date")
    ax.set_ylabel("$\Delta h$")
    start = pd.to_datetime('2012-01-01').tz_localize("UCT")
    end = pd.to_datetime('2023-01-30').tz_localize("UCT")
    ax.set_xlim(start, end)




def plot_seepage(summary_subset, ax, label = None, alpha = 1):

    ax.errorbar(summary_subset.date, summary_subset['S'],  alpha = alpha, label = label,
                 yerr = (summary_subset['S_CI_high'] - summary_subset['S_CI_low'])/4,
                 linestyle = '', marker = 'o')

    # Identify the range of years
    start_year =   min(summary_subset.year.min(), summary_subset.year.min())
    end_year = max(summary_subset.year.max(), summary_subset.year.max())

    # Add vertical lines for every May
    for year in range(start_year, end_year + 1):
        ax.axvline(pd.to_datetime(f'{year}-01-01'), color='grey', linestyle='--', linewidth=0.8)

    #ax.set_xlabel("date")
    ax.set_ylabel("Seepage $S$ (m$^3$/s)")
    start = pd.to_datetime('2012-01-01').tz_localize("UCT")
    end = pd.to_datetime('2023-01-30').tz_localize("UCT")
    ax.set_xlim(start, end)


def plot_Ks(summary_subset, ax, label = None, alpha = 1):

    ax.errorbar(summary_subset.date, summary_subset['Ks'], alpha = alpha, label = label,
                 yerr = (summary_subset['CI_high'] - summary_subset['CI_low'])/4,
                 linestyle = '', marker = 'o')

    # Identify the range of years
    start_year =   min(summary_subset.year.min(), summary_subset.year.min())
    end_year = max(summary_subset.year.max(), summary_subset.year.max())

    # Add vertical lines for every May
    for year in range(start_year, end_year + 1):
        ax.axvline(pd.to_datetime(f'{year}-01-01'), color='grey', linestyle='--', linewidth=0.8)

    # ax.set_xlabel("date")
    ax.set_ylabel(r"$\tilde K$ (m$^2$/s)")
    start = pd.to_datetime('2012-01-01').tz_localize("UCT")
    end = pd.to_datetime('2023-01-30').tz_localize("UCT")
    ax.set_xlim(start, end)


###########  Regression functions ###############

def fit_Xy_power(X,y,power = 1, fit_intercept=False):
    """
    Not using this version
    """
    model = LinearRegression(fit_intercept=fit_intercept)

    X = X**power
    model.fit(X, y)

    predictions = model.predict(X)

    # Parameter (slope)
    slope = model.coef_[0]
    intercept = model.intercept_
    # R2 value
    r_squared = model.score(X, y)

    # Calculate standard errors of the slope
    n = len(y)
    MSE = np.sum((y - predictions) ** 2) / (n - 1)
    SE_slope = np.sqrt(MSE / (np.sum(X ** 2)))

    # 95% confidence intervals for the slope
    t_value = t.ppf(0.975, n - 1)  # t-value for 95% CI, two-tailed test
    CI_slope = [slope - t_value * SE_slope, slope + t_value * SE_slope]
    CI_low = float(CI_slope[0])
    CI_high = float(CI_slope[1])
    
    return predictions, slope, intercept, r_squared, t_value, CI_low, CI_high

from sklearn.linear_model import LinearRegression
from scipy.stats import t
import numpy as np

def fit_Xy( X,y, fit_intercept=False):

    model = LinearRegression(fit_intercept=fit_intercept)

    model.fit(X, y)

    predictions = model.predict(X)
    residuals = y - predictions
    n = len(y)
    p = X.shape[1]  # number of predictors
    df = n - p - 1

    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    # # standard error and t value
    # se = np.sqrt(np.sum(residuals**2) / df) / np.sqrt(np.sum((X.iloc[:, 0] - X.iloc[:, 0].mean())**2))
    # t_value = slope / se
    # CI = t.interval(0.95, df, loc=slope, scale=se)

    # Calculate standard errors of the slope
    n = len(y)
    MSE = np.sum((y - predictions) ** 2) / (n - 1)
    SE_slope = np.sqrt(MSE / (np.sum(X ** 2)))

    # 95% confidence intervals for the slope
    t_value = t.ppf(0.975, n - 1)  # t-value for 95% CI, two-tailed test
    CI_slope = [slope - t_value * SE_slope, slope + t_value * SE_slope]
    CI_low = float(CI_slope[0])
    CI_high = float(CI_slope[1])
    
    return predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals


def calculate_AIC(n, residuals, k):
    RSS = np.sum(residuals**2)
    return n * np.log(RSS / n) + 2 * k



def filter_overtop(subset, minval = 0):

    hourly_subset = get_hourly_visitor(subset)
    if 'visitor_h_Q' in hourly_subset.columns:
        lose = hourly_subset[(hourly_subset['visitor_h_Q']- hourly_subset['visitor_h']) < minval].index
        lose = np.unique(lose.date)
        return lose
    else:
        return

def optimize_exponent(merged_case, ind):

    subset = get_subset(merged_case, ind, 0, 0)
    daily_subset = get_daily_subset(subset, periods=1)
    date = daily_subset.index[0].date()
    duration = len(daily_subset)

    daily_subset['seepage'] = daily_subset['seepage_visitor']
    daily_subset['State_visit'] = daily_subset['State_visit'].apply(np.floor)

    # Calculate Δh
    daily_subset['delta_h'] = daily_subset.visitor_filled - daily_subset.v
    daily_subset = daily_subset[['seepage', 'delta_h', 'visitor_h', 'waveHs', 'Q']].dropna()
    daily_subset['date'] = daily_subset.index.date

    lose = filter_overtop(subset, minval=0.0)
    pos_subset = daily_subset[~(daily_subset.date).isin(lose)]
    pos_subset = pos_subset.query("seepage < 6 and seepage > 0")

    N = len(pos_subset)

    if len(pos_subset) < 5:
        return

    R2s = []
    slopes = []
    residuals_list = []
    ps = np.arange(0, 0.4, 0.5)

    for p in ps:
        #X = (pos_subset['delta_h']**p * pos_subset['visitor_h']**0.).values.reshape(-1, 1)
        X = (pos_subset['delta_h'] * (pos_subset['delta_h'] + p) ).values.reshape(-1, 1)
        y = pos_subset['seepage'].values
        predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)
        R2s.append(r_squared)
        slopes.append(slope)
        residuals_list.append(residuals)

    ind_best = np.argmax(R2s)
    best_exponent = np.round(ps[ind_best], 2)
    best_R2 = R2s[ind_best]
    best_slope = slopes[ind_best]
    best_residuals = residuals_list[ind_best]

    # AICs
    n = len(pos_subset)
    AIC_best = calculate_AIC(n, best_residuals, k=2)

    # Now do linear model
    X_lin = (pos_subset['delta_h'] ).values.reshape(-1, 1)
    y = pos_subset['seepage'].values
    predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals_lin = fit_Xy(X_lin, y)
    AIC_lin = calculate_AIC(n, residuals_lin, k=1)

    # Permutation test
    r2_diff_obs = best_R2 - r_squared
    r2_diffs = []
    for _ in range(100):
        y_perm = np.random.permutation(y)
        # R² for permuted linear
        _, _, _, r2_perm_lin, _, _, _, _ = fit_Xy(X_lin, y_perm)
        # R² for permuted exponent
        X_best = ((pos_subset['delta_h'] + best_exponent) * pos_subset['delta_h'] ).values.reshape(-1, 1)
        _, _, _, r2_perm_best, _, _, _, _ = fit_Xy(X_best, y_perm)
        r2_diffs.append(r2_perm_best - r2_perm_lin)
    p_val = np.mean(np.array(r2_diffs) >= r2_diff_obs)

    best = pd.Series({
        "K_tilde_best": best_slope.round(3),
        "b_best": '{0:.2f}'.format(best_exponent),
        "K_tilde": slope.round(3),
        "CI_low": CI_low,
        "CI_high": CI_high,
        "R2": '{0:.2f}'.format(r_squared),
        "best_R2": '{0:.2f}'.format(best_R2),
        "delta_R2": round(best_R2 - r_squared, 3),
        "AIC_linear": round(AIC_lin, 1),
        "AIC_best": round(AIC_best, 1),
        "AIC_diff": round(AIC_lin - AIC_best, 1),
        "p_val": round(p_val, 3),
        "date": date,
        "K_tilde_fmt": '{0:.2f} [{1:.2f}-{2:.2f}]'.format(slope, CI_low, CI_high),
        "N": len(pos_subset),
        "Q" : pos_subset['Q'].median(),
        "duration": duration
    })
    return best


def optimize_exponent_double(merged_case, ind):

    subset = get_subset(merged_case, ind, 0, 0)
    daily_subset = get_daily_subset(subset, periods=1)
    date = daily_subset.index[0].date()
    duration = len(daily_subset)

    daily_subset['seepage'] = daily_subset['seepage_visitor']
    daily_subset['State_visit'] = daily_subset['State_visit'].apply(np.floor)

    # Calculate Δh
    daily_subset['delta_h'] = daily_subset.visitor_filled - daily_subset.v
    daily_subset = daily_subset[['seepage', 'delta_h', 'visitor_h', 'waveHs', 'Q']].dropna()
    daily_subset['date'] = daily_subset.index.date

    lose = filter_overtop(subset, minval=0.0)
    pos_subset = daily_subset[~(daily_subset.date).isin(lose)]
    pos_subset = pos_subset.query("seepage < 6 and seepage > 0")

    N = len(pos_subset)

    if len(pos_subset) < 5:
        return

    R2s = []
    slopes = []
    residuals_list = []
    offsets = np.arange(0, 2.5, 0.5)
    ps = np.arange(0, 3, 0.5)

    # Preallocate result containers shaped (len(ps), len(offsets))
    R2s = np.full((len(ps), len(offsets)), np.nan)
    slopes = np.full_like(R2s, np.nan, dtype=float)
    intercepts = np.full_like(R2s, np.nan, dtype=float)
    # store residuals in a Python grid since lengths may vary
    residuals_grid = [[None for _ in range(len(offsets))] for _ in range(len(ps))]


    for p_idx, p in enumerate(ps):
        for o_idx, o in enumerate(offsets):
            # choose your X; leaving the "p" expression commented in case you want it
            X = (pos_subset['delta_h'] * (pos_subset['delta_h'] + o)**p).values.reshape(-1, 1)
            y = pos_subset['seepage'].values
            print(o, p)
            predictions, slope, intercept, r2, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)
            R2s[p_idx, o_idx] = r2
            print(o, p, r2)
            slopes[p_idx, o_idx] = slope
            intercepts[p_idx, o_idx] = intercept
            residuals_grid[p_idx][o_idx] = residuals

    # Find best (p, offset)
    best_flat = np.nanargmax(R2s)
    best_p_idx, best_o_idx = np.unravel_index(best_flat, R2s.shape)

    best_exponent = float(np.round(ps[best_p_idx], 2))
    best_offset = float(np.round(offsets[best_o_idx], 2))
    best_R2 = float(R2s[best_p_idx, best_o_idx])
    best_slope = float(slopes[best_p_idx, best_o_idx])
    best_intercept = float(intercepts[best_p_idx, best_o_idx])
    best_residuals = residuals_grid[best_p_idx][best_o_idx]

    # AICs
    n = len(pos_subset)
    AIC_best = calculate_AIC(n, best_residuals, k=2)

    # Now do linear model
    # X_lin = pos_subset['delta_h'].values.reshape(-1, 1)
    #X_lin = (pos_subset['delta_h'] **2 * pos_subset['visitor_h']**0. ).values.reshape(-1, 1)
    X_lin = (pos_subset['delta_h']  * pos_subset['delta_h'] ).values.reshape(-1, 1)
    y = pos_subset['seepage'].values
    predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals_lin = fit_Xy(X_lin, y)
    AIC_lin = calculate_AIC(n, residuals_lin, k=1)

    # Permutation test
    r2_diff_obs = best_R2 - r_squared
    r2_diffs = []
    for _ in range(100):
        y_perm = np.random.permutation(y)
        # R² for permuted linear
        _, _, _, r2_perm_lin, _, _, _, _ = fit_Xy(X_lin, y_perm)
        # R² for permuted exponent
        X_best = ((pos_subset['delta_h'] + best_exponent) * pos_subset['delta_h'] ).values.reshape(-1, 1)
        _, _, _, r2_perm_best, _, _, _, _ = fit_Xy(X_best, y_perm)
        r2_diffs.append(r2_perm_best - r2_perm_lin)
    p_val = np.mean(np.array(r2_diffs) >= r2_diff_obs)  

    best = pd.Series({
        "K_tilde_best": best_slope,
        "b_best": best_exponent,
        "offset_best": best_offset,
        "K_tilde": slope.round(3),
        "CI_low": CI_low,
        "CI_high": CI_high,
        "R2": '{0:.2f}'.format(r_squared),
        "best_R2": '{0:.2f}'.format(best_R2),
        "delta_R2": round(best_R2 - r_squared, 3),
        "AIC_linear": round(AIC_lin, 1),
        "AIC_best": round(AIC_best, 1),
        "AIC_diff": round(AIC_lin - AIC_best, 1),
        "p_val": round(p_val, 3),
        "date": date,
        "K_tilde_fmt": '{0:.2f} [{1:.2f}-{2:.2f}]'.format(slope, CI_low, CI_high),
        "N": len(pos_subset),
        "Q" : pos_subset['Q'].median(), 
        "duration": duration        
    })
    return best


def optimize_exponent_prev(merged_case, ind):
    """
    we can delete this. 
    """
    subset =  get_subset(merged_case, ind, 0, 0)
    daily_subset = get_daily_subset(subset, periods = 1)
    date = daily_subset.index[0].date()
    duration = len(daily_subset)

    daily_subset['seepage'] =  daily_subset['seepage_visitor']
    daily_subset['State_visit'] = daily_subset['State_visit'].apply(np.floor)

    # Calculate Δh
    daily_subset['delta_h'] = daily_subset.visitor_filled - daily_subset.v
    daily_subset = daily_subset[['seepage', 'delta_h', 'visitor_h', 'waveHs']].dropna()
    daily_subset['date'] = daily_subset.index.date

    lose = filter_overtop(subset, minval=0.0)
    pos_subset = daily_subset[~(daily_subset.date).isin(lose)]
    pos_subset = pos_subset.query("seepage < 6 and seepage > 0")

    N = len(pos_subset)

    if len(pos_subset) >= 5:

        X = (pos_subset['delta_h'] ).values.reshape(-1, 1)
        y =  pos_subset['seepage']

    else:
        return

    R2s = []
    ps= np.arange(0, 3, 0.1)
    slopes = []
    for p in ps:
        
        X = (pos_subset['delta_h'] * pos_subset['visitor_h']**p).values.reshape(-1, 1)
        y =  pos_subset['seepage']

        predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)
        R2s.append(r_squared)
        slopes.append(slope)

    ind = np.where(R2s == np.max(R2s))[0][0]
    best_exponent = np.round(ps[ind], 2)
    best_R2 = R2s[ind]
    best_slope = slopes[ind]

    X = (pos_subset['delta_h']).values.reshape(-1, 1)
    y =  pos_subset['seepage']
    predictions, slope, intercept, r_squared, t_value, CI_low, CI_high, residuals = fit_Xy(X, y)

    best = pd.Series({'K_tilde_best' : best_slope.round(3), 
                      'b_best' :  '{0:.2f}'.format(best_exponent),
                      'K_tilde' : slope.round(3), 
                      'CI_low' : CI_low, 'CI_high' : CI_high,
                      'R2' : '{0:.2f}'.format(r_squared),
                      'best_R2' : '{0:.2f}'.format(best_R2),                      
                      'date' : date,
                      'K_tilde_fmt' : '{0:.2f} [{1:.2f}-{2:.2f}]'.format(slope, CI_low, CI_high),
                      'N' : len(pos_subset),
                      'duration' : duration})
    return best

############################ Summarize the closures

###### Timeseries functions

from itertools import groupby

import numpy as np
import pandas as pd

def sequence_lengths(data):
    """
    identify lengths of inlet closures
    """
    lengths = []
    for key, group in groupby(data):
        group_list = list(group)
        if key == 1:
            lengths.extend([len(group_list)] * len(group_list))
        else:
            lengths.extend([0] * len(group_list))
    return lengths

# ###### ###### ###### ###### ###### ###### #####

# Functions to compute the Short Time Fourier Transform (STFT)
from scipy.signal import stft

def add_power_USGS(subset, period, data_freq='15T', label='USGS_24_hr'):
    """
    Add power at user-specified period for different data frequencies.
    
    Input:
        subset: dataset to modify, frequency should be either '15T' for 15-minute or 'H' for hourly.
        period: in hours
        data_freq: data frequency, '15T' for 15
         minutes or 'H' for hourly
        label: label for new column

    Note: returns power^(1/4)
    """
    if data_freq == '15T':
        samples_per_hour = 4
    elif data_freq == 'H':
        samples_per_hour = 1
    else:
        raise ValueError("data_freq must be '15T' for 15-minute or 'H' for hourly data")
    
    # Calculate the sampling frequency and nperseg for STFT
    fs = samples_per_hour / 3600  # samples per second
    nperseg = 4*samples_per_hour * period
    
    # Apply STFT Short Time Fourier Transform (STFT)
    f, t, Zxx = stft(subset['USGS_filled'], fs=fs, nperseg=nperseg, noverlap=nperseg-1) 

    # Find the index corresponding to the period-hour frequency
    freq_index = np.argmin(np.abs(f - 1/(period*3600)))
    
    # Extract the power at the period-hour frequency
    USGS_period_hr = np.abs(Zxx[freq_index, :])**0.25
    
    # Convert the time array from seconds to timedelta, then to the original timestamp
    time_deltas = pd.to_timedelta(t, unit='s')
    time_stamps = subset.index[0] + time_deltas
    
    # Create a DataFrame for the interpolated power data
    power_subset = pd.DataFrame(data={label: USGS_period_hr}, index=time_stamps)
    
    # Interpolate the power data to match the original DataFrame's time index
    power_subset = power_subset.reindex(power_subset.index.union(subset.index)).interpolate('index').loc[subset.index]
    
    # Add the interpolated power data to the original DataFrame
    subset[label] = power_subset[label]/power_subset[label].max()
    
    return subset



def add_power_visitor(subset, period, data_freq='15T', label = 'visitor_24_hr' ):
    """
    Add power at user-specified period
    
    Input:
        period : in hours
        subset : dataset to modify, frequency should by 15 minute
        label : label new column

    Note: returns power^(1/4)
    """

    if data_freq == '15T':
        samples_per_hour = 4
    elif data_freq == 'H':
        samples_per_hour = 1
    else:
        raise ValueError("data_freq must be '15T' for 15-minute or 'H' for hourly data")
    
    # Calculate the sampling frequency and nperseg for STFT
    fs = samples_per_hour / 3600  # samples per second
    nperseg = 4*samples_per_hour * period
    
    # Apply STFT Short Time Fourier Transform (STFT)
    f, t, Zxx = stft(subset['visitor_filled'], fs=fs, nperseg=nperseg, noverlap=nperseg-1) 
    # Apply STFT  Short Time Fourier Transform (STFT)
    #f, t, Zxx = stft(subset['visitor_filled'],fs=1/(3600), nperseg=4*period, noverlap=4*period-1) 

    # Find the index corresponding to the period-hour frequency
    # f is an array of frequencies. 1/(period*3600) is the period-hour frequency in Hz
    freq_index = np.argmin(np.abs(f - 1/(period*3600)))
    # Extract the power at the period-hour frequency
    visitor_period_hr = np.abs(Zxx[freq_index, :])**0.5#**2
    # Convert the time array from seconds to timedelta, then to the original timestamp
    time_deltas = pd.to_timedelta(t, unit='s')
    time_stamps = subset.index[0] + time_deltas

    # Create a DataFrame for the interpolated power data
    power_subset = pd.DataFrame(data={label: visitor_period_hr}, index=time_stamps)
    # Interpolate the power data to match the original DataFrame's time index
    power_subset = power_subset.reindex(power_subset.index.union(subset.index)).interpolate('index').loc[subset.index]

    # Add the interpolated power data to the original DataFrame
    subset[label] = power_subset[label]/power_subset[label].max()

    return subset
