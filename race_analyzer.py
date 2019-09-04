#! /usr/local/bin/python3

import argparse
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def read_data(csv_path):
   def to_ms(timestr):
      lap_time = datetime.datetime.strptime(timestr, '%M:%S.%f').time()
      lap_time_ms = ((lap_time.hour * 60 + lap_time.minute) * 60 + lap_time.second) * 1000 + int(lap_time.microsecond / 1000)
      return lap_time_ms

   data = pd.read_csv(csv_path)
   data.loc[:, 'ms'] = data['TIME'].apply(to_ms)
   data.loc[:, 's'] = data['ms'] / 1000

   return data


def calculate_statistics(data):
   data.loc[:, 'best3'] = data.loc[:, 's'].rolling(3).sum()
   lap = 0
   count = 1
   for row in data.itertuples():
      if row.LAP == lap + 1:
         count += 1
      else:
         count = 1

         if row.LAP != lap + 1 or count < 3:
            data.at[row.Index, 'best3'] = np.nan
            lap = row.LAP

   data.loc[:, 'best3avg'] = data['best3'] / 3.0


def filter_data(data, min_lap_time = 5.0, max_lap_time=60.0, pilot=None):
    return data.loc[(data['s'] < max_lap_time) & (data['s'] > min_lap_time)].copy()


def find_hist_bounds(data):
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25

    minb = q25 - (iqr * 1.5)
    maxb = q75 + (iqr * 1.5)

    minb = max(minb, data.min())
    maxb = min(maxb, data.max())

    return (minb, maxb)


def plot_hist(ax, data, labels=None, bin_width=1.0, bounds=None):
   concatenated = pd.Series()
   for d in data:
      concatenated = concatenated.append(d.dropna(), ignore_index=True)

   if bounds is None:
      bounds = find_hist_bounds(concatenated)

   bounds = (max(bounds[0], concatenated.min()) - 2, min(bounds[1], concatenated.max()) + 2)
   bins = np.unique(np.concatenate(([math.floor(concatenated.min())], np.arange(math.floor(bounds[0]), math.ceil(bounds[1]), bin_width), [math.ceil(concatenated.max())])))
   ax.set_xlim(bounds)
   ax.minorticks_on()
   ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
   ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
   ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
   ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
   ax.grid(which='both')
   (n, bins, patches) = ax.hist(
      data,
      bins=bins,
      histtype='stepfilled',
      alpha=0.7,
      label=labels
   )

   return (n, bins, patches)


def plot_cumbest(ax, data):
   cummin = data.cummin().fillna(method='ffill')
   x = []
   y = []
   for _, values in cummin.iteritems():
      series = values.squeeze().reset_index(drop=True)
      y.append(series.iloc[-1])
      x.append(series.idxmin(y[-1]) + 1)

   ax.minorticks_on()
   ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
   ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
   ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
   ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
   ax.grid(which='both')
   ax.plot(range(1, len(cummin) + 1), cummin)
   ax.plot(x, y, 'ro')


def plot_pilot(axes, data, hist_bounds=None):
    plot_hist(axes[0], [data['s'], data['best3avg']], bounds=hist_bounds)
    plot_cumbest(axes[1], data[['s', 'best3avg']])

def plot(data, statistics, bounds, bin_width=1.0, max_lap_time=60.0):
   grouped = data.groupby('PILOT')
   print(grouped['s'].describe())
   print(grouped['best3avg'].describe())
   comparision_figure = plt.figure()
   comparision_figure.canvas.set_window_title('Pilots Comparision')
   comparision_axes = comparision_figure.subplots(len(grouped), 2)
   comparision_idx = 0

   sns_figure = plt.figure()
   LAP_TYPE = 'Lap Type'
   data_long = pd.melt(data, id_vars=('PILOT'), value_vars=('s', 'best3avg'), var_name=LAP_TYPE, value_name='time[s]')
   sns.violinplot(x='PILOT', hue=LAP_TYPE, y='time[s]', data=data_long, ax=sns_figure.gca(), split=True, scale='count')
   sns_figure.gca().grid(True)

   figures = [comparision_figure, sns_figure]
   hist_bounds = find_hist_bounds(data['s'])
   (minb, maxb) = hist_bounds
   for pilot, group in grouped:
      figure = plt.figure()
      figure.canvas.set_window_title('Pilot ' + pilot)
      figures.append(figure)
      axes = figure.subplots(1, 2)
      plot_pilot(comparision_axes[comparision_idx], group, hist_bounds=hist_bounds)
      plot_pilot(axes, group)
      for ax in (comparision_axes[comparision_idx][0], axes[0]):
         ax.set_title(pilot)

      comparision_idx += 1

      xlims = [[minb, maxb], [0, -math.inf]]
      ylims = [[0, -math.inf], [math.inf, -math.inf]]
      for i, axes in enumerate(comparision_axes):
         for j, ax in enumerate(axes):
            limx = ax.get_xlim()
            limy = ax.get_ylim()
            if j != 0:
               xlims[j] = [min(xlims[j][0], limx[0]), max(xlims[j][1], limx[1])]
            ylims[j] = [min(ylims[j][0], limy[0]), max(ylims[j][1], limy[1])]


      for i, axes in enumerate(comparision_axes):
         for j, ax in enumerate(axes):
            ax.set_xlim(xlims[j])
            ax.set_ylim(ylims[j])


   plt.show()


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--csv')
   parser.add_argument('--hist-left-bound', type=float)
   parser.add_argument('--hist-right-bound', type=float)
   parser.add_argument('--only-pilot')
   args = parser.parse_args()

   data = read_data(args.csv)
   filtered = filter_data(data, pilot=args.only_pilot)
   statistics = calculate_statistics(filtered)
   hist_bounds = [args.hist_left_bound, args.hist_right_bound]
   plot(filtered, statistics, bounds=hist_bounds)


if __name__ == "__main__":
   main()


