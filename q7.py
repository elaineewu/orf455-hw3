import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('out')

# Same settings as the vatic-det command-line defaults, so every scenario
# differs from its baseline only in the input data we change.
SIM_DEFAULTS = dict(
    run_lmps=True, mipgap=0.01, load_shed_penalty=1e4,
    reserve_shortfall_penalty=1e3, reserve_factor=0.05, output_detail=2,
    prescient_sced_forecasts=False, ruc_prescience_hour=0,
    ruc_execution_hour=16, ruc_every_hours=24, ruc_horizon=48,
    sced_horizon=4, lmp_shortfall_costs=False,
    enforce_sced_shutdown_ramprate=False, no_startup_shutdown_curves=False,
    verbosity=0, output_max_decimals=4, create_plots=False, renew_costs=None,
    save_to_csv=True, last_conditions_file=None)

GRIDS = {'RTS-GMLC': dt.date(2020, 1, 19), 'Texas-7k': dt.date(2018, 1, 19)}
FUEL_NAMES = {'RTS-GMLC': {'N': 'Nuclear', 'C': 'Coal', 'G': 'Gas', 'O': 'Oil'},
              'Texas-7k': {'N': 'Nuclear', 'C': 'Coal', 'O': 'Lignite',
                           'G': 'Gas', 'A': 'Other'}}

_cache = {}


def base_inputs(grid):
    """Load (template, gen_data, load_data) once per grid and return copies.

    gen_data: columns are ('fcst'|'actl', generator name), one row per hour.
    load_data: columns are ('fcst'|'actl', bus name).
    'fcst' is what the day-ahead commitment sees; 'actl' is what real-time
    dispatch actually faces.
    """
    from vatic.data import load_input
    if grid not in _cache:
        _cache[grid] = load_input(grid, GRIDS[grid], 1, None)
    t, g, l = _cache[grid]
    return t, g.copy(), l.copy()


def run(label, grid, gen, load, args, ruc_file=None):
    """Run one Vatic simulation and save its output to out/<label>/."""
    from vatic.engines import Simulator
    out = OUT / label
    if (out / 'hourly_summary.csv').exists():
        print(f'[skip] {label} already done')
        return
    out.mkdir(parents=True, exist_ok=True)
    template, _, _ = base_inputs(grid)
    # save which fuel each thermal generator burns, for the summaries later
    pd.Series({g: FUEL_NAMES[grid].get(f, f) for g, f in
               template['ThermalGeneratorType'].items()}, name='fuel'
              ).rename_axis('Generator').to_csv(out / 'gen_fuel.csv')
    print(f'[run ] {label} ...', flush=True)
    t0 = dt.datetime.now()
    Simulator(template, gen, load, out_dir=out, start_date=GRIDS[grid],
              num_days=1, solver=args.solver,
              solver_options={'Threads': args.threads},
              init_ruc_file=Path(ruc_file) if ruc_file else None,
              **SIM_DEFAULTS).simulate()
    print(f'[done] {label} in {(dt.datetime.now() - t0).seconds / 60:.1f} min')


# ---------------------------------------------------------------- part (a)
def part_a(args):
    grid = 'RTS-GMLC'
    # The day-ahead unit commitment only sees forecasts. Demand scenarios change
    # actual demand only, so they share the baseline's commitment (saved here
    # by the baseline run). This isolates the real-time response and saves time.
    ruc = OUT / 'a_ruc.p'

    _, g, l = base_inputs(grid)
    run('a_base', grid, g, l, args, ruc_file=ruc)

    for k in [0.90, 0.95, 1.05, 1.10]:
        _, g, l = base_inputs(grid)
        l['actl'] = l['actl'] * k                    # surprise in actual demand
        run(f'a_demand_{k:.2f}', grid, g, l, args, ruc_file=ruc)

    for kind in ['RTPV', 'HYDRO']:
        _, g, l = base_inputs(grid)
        cols = [c for c in g['fcst'].columns if c.split('_')[1] == kind]
        print(f'{kind}: shutting down {len(cols)} generators')
        for side in ['fcst', 'actl']:                # outage known day-ahead
            for c in cols:
                g[(side, c)] = 0.0
        run(f'a_no_{kind}', grid, g, l, args)        # new commitment


# ---------------------------------------------------------------- part (b)
ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
WARMUP_HOURS = 1     # hours skipped at the start of Texas-7k runs in the summaries


def part_b(args):
    grid = 'Texas-7k'
    for a in ALPHAS:
        _, g, l = base_inputs(grid)
        g = g * a       # scale every wind and solar generator, forecast and actual
        run('b_base' if a == 1.0 else f'b_alpha_{a:.2f}', grid, g, l, args)


# ---------------------------------------------------------------- part (c)
def wind_errors(n_hours, n_gens, mu, sigma, rho, seed):
    """Proportional deviations, same N(mu, sigma^2) marginal for every generator.

    independent: eps[h, g] = mu + sigma * e[h, g]
    correlated:  eps[h, g] = mu + sigma * (sqrt(rho) Z[h] + sqrt(1-rho) e[h, g])
    Z[h] is a common weather shock shared by all generators in hour h, so
    corr(eps[h, g], eps[h, k]) = rho for g != k. Draws are independent across
    hours. Both cases use the same Z and e for a given seed (common random
    numbers), so the only difference between them is the correlation.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_hours)
    e = rng.standard_normal((n_hours, n_gens))
    ind = mu + sigma * e
    cor = mu + sigma * (np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * e)
    return ind, cor


def part_c(args):
    from vatic.data.loaders import T7kLoader
    grid = 'Texas-7k'
    template, g0, l0 = base_inputs(grid)
    wind = [c for c in g0['actl'].columns
            if template['NondispatchableGeneratorType'].get(c) == 'W']
    pmax = T7kLoader(None).gen_df.set_index('GEN UID')['PMax MW'].reindex(wind)
    pmax = pmax.fillna(g0['actl'][wind].max() * 2)
    print(f'{len(wind)} wind generators; mu={args.mu} sigma={args.sigma} rho={args.rho}')

    # the baseline (no extra errors) is the same run as part (b)'s baseline
    _, g, l = base_inputs(grid)
    run('b_base', grid, g, l, args)

    for seed in range(args.seeds):
        ind, cor = wind_errors(len(g0), len(wind), args.mu, args.sigma, args.rho, seed)
        for name, eps in [('ind', ind), ('cor', cor)]:
            _, g, l = base_inputs(grid)
            # only actual wind changes; the day-ahead forecast stays the same,
            # so this is extra forecast error the operator didn't plan for
            new = g['actl'][wind].values * (1 + eps)
            new = np.clip(new, 0, pmax.values[None, :])
            g.loc[:, [('actl', c) for c in wind]] = new
            run(f'c_{name}_s{seed}', grid, g, l, args)


# ---------------------------------------------------------------- summaries
def summarize_run(path):
    """One row of summary measures for one run folder."""
    hs = pd.read_csv(path / 'hourly_summary.csv')
    th = pd.read_csv(path / 'thermal_detail.csv')
    # Texas-7k runs (parts b, c): hour 0 is a start-up artifact. Thermal output
    # in the first hour is locked to the initial state from the input files, so
    # any shortfall there is shed regardless of the scenario. Shortage and price
    # measures below therefore use hours 1-23 for those runs (costs use all hours).
    warmup = WARMUP_HOURS if path.name.startswith(('b_', 'c_')) else 0
    hs_e = hs[hs.Hour >= warmup]
    fuel = pd.read_csv(path / 'gen_fuel.csv', index_col=0)['fuel']
    th['fuel'] = th.Generator.map(fuel)
    row = {
        'fixed cost ($)': hs.FixedCosts.sum(),
        'variable cost ($)': hs.VariableCosts.sum(),
        'total cost ($)': hs.FixedCosts.sum() + hs.VariableCosts.sum(),
        'demand (MWh)': hs.Demand.sum(),
        'load shed (MWh)': hs_e.LoadShedding.sum(),
        'reserve shortfall (MWh)': hs_e.ReserveShortfall.sum(),
        'renewables used (MWh)': hs.RenewablesUsed.sum(),
        'curtailment (MWh)': hs.RenewablesCurtailment.sum(),
        'hours with load shed': int((hs_e.LoadShedding > 1e-3).sum()),
        'hours with reserve shortfall': int((hs_e.ReserveShortfall > 1e-3).sum()),
        'starts/stops': hs['Number on/offs'].sum(),
        'avg units on': th.groupby('Hour')['Unit State'].sum().mean(),
        'first hour evaluated': warmup,
    }
    row['cost per MWh ($)'] = row['total cost ($)'] / row['demand (MWh)']
    bus = path / 'bus_detail.csv'
    if bus.exists():                                   # load-weighted LMP by hour
        b = pd.read_csv(bus)
        lw = (b.LMP * b.Demand).groupby(b.Hour).sum() / b.Demand.groupby(b.Hour).sum()
        lw = lw[lw.index >= warmup]
        shed_hrs = hs_e.groupby('Hour').LoadShedding.sum() > 1e-3
        row.update({'avg LMP ($/MWh)': lw.mean(), 'max LMP ($/MWh)': lw.max(),
                    'std LMP ($/MWh)': lw.std(),
                    'median LMP ($/MWh)': lw.median(),
                    'avg LMP excl. shed hours ($/MWh)': lw[~shed_hrs.reindex(lw.index, fill_value=False)].mean(),
                    'LMP spread across buses ($/MWh)':
                        b.groupby('Hour').LMP.agg(lambda x: x.quantile(.95) - x.quantile(.05)).mean()})
    for f, mwh in th.groupby('fuel').Dispatch.sum().items():
        row[f'{f} (MWh)'] = mwh
    return row


def hourly(path):
    hs = pd.read_csv(path / 'hourly_summary.csv').set_index('Hour')
    b = pd.read_csv(path / 'bus_detail.csv')
    hs['LMP'] = (b.LMP * b.Demand).groupby(b.Hour).sum() / b.Demand.groupby(b.Hour).sum()
    return hs


def summarize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e5e1', 'legend.frameon': False,
                         'figure.dpi': 150, 'savefig.bbox': 'tight'})
    runs = sorted(p for p in OUT.iterdir() if (p / 'hourly_summary.csv').exists())
    S = pd.DataFrame({p.name: summarize_run(p) for p in runs}).T
    fuel_cols = [c for c in S.columns if c.split(' (')[0] in
                 {'Nuclear', 'Coal', 'Gas', 'Oil', 'Lignite', 'Other'}]
    S[fuel_cols] = S[fuel_cols].fillna(0.0)
    S.round(2).to_csv('summary_all_runs.csv')
    def fuels_for(idx):
        return [c for c in fuel_cols if S.loc[idx, c].abs().sum() > 0]

    # ---- part (a)
    a = [r for r in ['a_base', 'a_demand_0.90', 'a_demand_0.95', 'a_demand_1.05',
                     'a_demand_1.10', 'a_no_RTPV', 'a_no_HYDRO'] if r in S.index]
    if 'a_base' in a:
        A = S.loc[a]; fuels = fuels_for(a)
        D = (A - A.loc['a_base']).drop('a_base')
        A.round(1).T.to_csv('table_a_levels.csv')
        D.round(1).T.to_csv('table_a_vs_baseline.csv')
        print('\n=== Part (a): levels (baseline first; table_a_vs_baseline.csv has differences) ===')
        print(A[['total cost ($)', 'avg LMP ($/MWh)', 'median LMP ($/MWh)', 'avg LMP excl. shed hours ($/MWh)',
                 'max LMP ($/MWh)', 'load shed (MWh)', 'hours with load shed', 'reserve shortfall (MWh)',
                 'hours with reserve shortfall', 'curtailment (MWh)', 'starts/stops'] + fuels].round(1).T.to_string())
        NAMES = {'a_base': 'Baseline', 'a_demand_0.90': 'Demand -10%', 'a_demand_0.95': 'Demand -5%',
                 'a_demand_1.05': 'Demand +5%', 'a_demand_1.10': 'Demand +10%',
                 'a_no_RTPV': 'No RTPV', 'a_no_HYDRO': 'No hydro'}
        fig = plt.figure(figsize=(7.2, 5.0))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
        ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        for i, r in enumerate(a):
            h = hourly(OUT / r)
            base = r == 'a_base'
            ax[0].plot(h.index, h.LMP, color='#222' if base else C[(i - 1) % 8], lw=2.2 if base else 1.1,
                       ls='--' if base else '-', zorder=5 if base else 2, label=NAMES[r])
        ax[0].set_yscale('symlog', linthresh=100)
        ax[0].set_yticks([0, 20, 50, 100, 1000, 10000])
        ax[0].set_yticklabels(['0', '20', '50', '100', '1k', '10k'])
        ax[0].set(xlabel='Hour', ylabel='$/MWh (log above 100)', title='Load-weighted LMP')
        ax[0].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        rel = A[['load shed (MWh)', 'reserve shortfall (MWh)', 'curtailment (MWh)']].copy()
        rel.columns = ['Load shed', 'Reserve shortfall', 'Curtailment']
        rel.index = [NAMES[r] for r in rel.index]
        rel.plot.bar(ax=ax[1], color=[C[7], C[3], C[2]], width=0.8)
        ax[1].set_ylabel('MWh'); ax[1].set_title('Shortages and curtailment', pad=16)
        ax[1].legend(fontsize=6, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
        big = [f for f in fuels if D[f].abs().max() > 50]
        dd = D[big].copy(); dd.columns = [f.split(' (')[0] for f in big]; dd.index = [NAMES[r] for r in dd.index]
        dd.plot.bar(ax=ax[2], color=[C[0], C[1], C[6], C[4]][:len(big)], width=0.8)
        ax[2].axhline(0, color='#999', lw=0.7)
        ax[2].set_ylabel('MWh vs baseline'); ax[2].set_title('Change in thermal output', pad=16)
        ax[2].legend(fontsize=6, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)
        for a_ in ax[1:]:
            a_.set_xticklabels(a_.get_xticklabels(), rotation=40, ha='right')
        plt.tight_layout(); plt.savefig('fig_a.png')

    # ---- part (b)
    b = {1.0: 'b_base', **{a_: f'b_alpha_{a_:.2f}' for a_ in ALPHAS[1:]}}
    b = {k: v for k, v in b.items() if v in S.index}
    if len(b) >= 2:
        fuels = fuels_for(list(b.values()))
        B = S.loc[list(b.values())]; B.index = list(b.keys()); B = B.sort_index(ascending=False)
        B.round(1).T.to_csv('table_b.csv')
        ren = B['renewables used (MWh)']
        # cost of each step down in alpha, per MWh of renewables lost in that step
        mc = (B['total cost ($)'].diff() / -ren.diff()).rename('step cost per renewable MWh lost ($/MWh)')
        print('\n=== Part (b) ===')
        print(pd.concat([B[['total cost ($)', 'avg LMP ($/MWh)', 'median LMP ($/MWh)',
                            'avg LMP excl. shed hours ($/MWh)', 'load shed (MWh)', 'hours with load shed',
                            'reserve shortfall (MWh)', 'renewables used (MWh)', 'curtailment (MWh)',
                            'avg units on'] + fuels], mc], axis=1).round(1).T.to_string())
        fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
        for a_, col, lab in [(ax[0], 'total cost ($)', 'Total cost ($M)'),
                             (ax[1], 'avg LMP ($/MWh)', 'Avg load-weighted LMP ($/MWh)\n(hours 1-23)')]:
            y = B[col] / (1e6 if col == 'total cost ($)' else 1)
            a_.plot(B.index, y, 'o-', color=C[0], label='simulated')
            a_.plot([1.0, 0.0], [y.loc[1.0], y.loc[0.0]], '--', color='#999', label='straight line')
            a_.set(xlabel='wind & solar scale factor α', title=lab)
            a_.set_xlim(1.05, -0.05)
        ax[0].legend(fontsize=6)
        fb = B[fuels] / 1e3; fb.columns = [f.split(' (')[0] for f in fuels]
        fb.plot.bar(ax=ax[2], stacked=True, color=C[:len(fuels)], width=0.7)
        ax[2].set(xlabel='α', ylabel='GWh', title='Thermal output by fuel')
        ax[2].legend(fontsize=5.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax[2].tick_params(axis='x', rotation=0)
        plt.tight_layout(); plt.savefig('fig_b.png')

    # ---- part (c)
    ci = [r for r in S.index if r.startswith('c_ind')]
    cc = [r for r in S.index if r.startswith('c_cor')]
    if ci and cc and 'b_base' in S.index:
        fuels = fuels_for(['b_base'] + ci + cc)
        cols = ['total cost ($)', 'avg LMP ($/MWh)', 'median LMP ($/MWh)', 'avg LMP excl. shed hours ($/MWh)',
                'std LMP ($/MWh)', 'curtailment (MWh)', 'reserve shortfall (MWh)', 'hours with reserve shortfall',
                'load shed (MWh)', 'hours with load shed', 'starts/stops', 'renewables used (MWh)'] + fuels
        T = pd.DataFrame({
            'baseline': S.loc['b_base', cols],
            'independent mean': S.loc[ci, cols].mean(),
            'independent s.e.': S.loc[ci, cols].std() / np.sqrt(len(ci)),
            'correlated mean': S.loc[cc, cols].mean(),
            'correlated s.e.': S.loc[cc, cols].std() / np.sqrt(len(cc))})
        # paired difference (same seeds in both cases)
        diff = S.loc[cc, cols].values - S.loc[ci, cols].values
        T['corr - ind (paired mean)'] = diff.mean(0)
        T['paired s.e.'] = diff.std(0, ddof=1) / np.sqrt(len(cc)) if len(cc) > 1 else np.nan
        T.round(1).to_csv('table_c.csv')
        per_seed = S.loc[ci + cc, ['total cost ($)', 'avg LMP ($/MWh)', 'load shed (MWh)', 'hours with load shed',
                                   'reserve shortfall (MWh)', 'curtailment (MWh)']]
        per_seed.round(1).to_csv('table_c_per_seed.csv')
        print('\n--- Part (c) per seed ---'); print(per_seed.round(1).to_string())
        print(f'\n=== Part (c): {len(ci)} seeds per case ===')
        print(T.round(1).to_string())
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
        hb = hourly(OUT / 'b_base'); hb = hb[hb.index >= WARMUP_HOURS]   # skip start-up hour
        for runs_, col, lab in [(ci, C[0], 'independent'), (cc, C[7], 'correlated')]:
            hs = pd.concat({r: hourly(OUT / r) for r in runs_})
            hs = hs[hs.index.get_level_values(1) >= WARMUP_HOURS]
            dev = (hs.RenewablesAvailable - hb.RenewablesAvailable.reindex(hs.index, level=1))
            m = dev.groupby(level=1).mean() / 1e3; s_ = dev.groupby(level=1).std() / 1e3
            ax[0].plot(m.index, m, color=col, label=lab); ax[0].fill_between(m.index, m - s_, m + s_, color=col, alpha=.2, lw=0)
            lm = hs.LMP.groupby(level=1); ax[1].plot(lm.mean().index, lm.mean(), color=col, label=lab)
            ax[1].fill_between(lm.mean().index, lm.min(), lm.max(), color=col, alpha=.15, lw=0)
        ax[1].plot(hb.index, hb.LMP, color='#222', lw=1.6, ls='--', label='baseline', zorder=5)
        ax[1].set_yscale('symlog', linthresh=100)
        ax[1].set_yticks([0, 20, 50, 100, 1000, 10000])
        ax[1].set_yticklabels(['0', '20', '50', '100', '1k', '10k'])
        ax[0].axhline(0, color='#999', lw=0.7)
        ax[0].set(xlabel='Hour', ylabel='GW', title='Available wind vs baseline\n(mean ± 1 sd across seeds)')
        ax[1].set(xlabel='Hour', ylabel='$/MWh (log above 100)', title='Load-weighted LMP\n(mean; band = min to max across seeds)')
        for a_ in ax: a_.legend(fontsize=6); a_.set_xlim(WARMUP_HOURS - 0.5, 23.5)
        plt.tight_layout(); plt.savefig('fig_c.png')
    print('\nSaved summary_all_runs.csv, table_*.csv and fig_*.png')


def diagnose(args):
    """Print the hours with load shedding or reserve shortfalls in key runs."""
    for r in ['a_base', 'a_demand_1.10', 'a_no_RTPV', 'b_base', 'b_alpha_0.00', 'c_cor_s0', 'c_ind_s0']:
        p = OUT / r
        if not (p / 'hourly_summary.csv').exists():
            continue
        h = hourly(p)
        bad = h[(h.LoadShedding > 1e-3) | (h.ReserveShortfall > 1e-3)]
        print(f'\n=== {r}: {len(bad)} problem hours ===')
        if len(bad):
            print(bad[['Demand', 'LoadShedding', 'ReserveShortfall', 'AvailableReserves',
                       'RenewablesUsed', 'LMP']].round(1).to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('part', choices=['a', 'b', 'c', 'summarize', 'diagnose'])
    p.add_argument('--solver', default='cbc')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--mu', type=float, default=0.05)
    p.add_argument('--sigma', type=float, default=0.15)
    p.add_argument('--rho', type=float, default=0.8)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    {'a': part_a, 'b': part_b, 'c': part_c, 'summarize': summarize,
     'diagnose': diagnose}[args.part](args)
