import pandas as pd
import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import cascaded_union
from descartes import PolygonPatch, patch
from operator import sub
from sklearn.preprocessing import minmax_scale
from matplotlib.collections import PatchCollection
from sklearn.utils import shuffle


def get_data(ii=5, data_path='../data/RESULTS_EUROVIS2015.csv', folder_path="../data/EUROVIS_new/"):
    df = pd.read_csv(data_path)
    for i, file in enumerate(df.fileName):
        file_name = folder_path + file.split('.csv')[0] + '_cls' + str(df.classNum[i]) + '.csv'
        try:
            sample_df = pd.read_csv(file_name, names = ['x', 'y', 'class'])
            if i == ii:
                break
            # print(sample_df.head(1))
        except FileNotFoundError:
            print('File \'' + file + '\' does not exist.')

    # sample_df.head()
    return sample_df


def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def scatter_points(ddf):
    f = plt.figure()
    ax0 = f.add_subplot(1, 1, 1)
    ax0.scatter(ddf.x, ddf.y, alpha = 0.8, c = 'gray', edgecolors = 'none')
    ax0.autoscale()
    ax0.set_aspect('auto', 'datalim')

    factor = get_aspect(ax0)
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim() / (1 / factor)

    return (factor, xlim, ylim)


def get_circles(df, size=2, factor=1):
    circles = []

    # df.x = minmax_scale(df.x)
    # df.y = minmax_scale(df.y)
    # buff = (max(df.x) - min(df.x)) / (np.sqrt(len(df.x))*size)
    buff = size

    df.y = df.y / factor
    for i, row in df.iterrows():
        circles.append(Point(row['x'], row['y']).buffer(buff))
    return pd.Series(circles)


def plot_circles(ax, circle_sets, colors=['red', 'blue'], alphas=[1, .3]):
    # ax = fig.add_subplot(1,1,1)
    for j, circles in enumerate(circle_sets):
        #print(len(circles))
        for i, circle in circles.items():
            #print(type(circle))
            if colors[j] =='red':
                ax.add_patch(PolygonPatch(circle, fc = colors[j], ec = 'none', alpha = alphas[j], label=str('aaaaaaaaaaaaaaaaaaaa')))
            else:
                ax.add_patch(PolygonPatch(circle, fc = colors[j], ec = 'none', alpha = alphas[j]))

    ax.autoscale()
    ax.set_aspect('equal', 'datalim')

    # fig.savefig(name + '.pdf')
    return ax


def remove_circles(circles, percent=0):
    fc = circles.copy()
    ri = []

    for i, circle in circles.items():
        temp_circ = fc[i:]
        #print(temp_circ.index)


        initial_area = cascaded_union(list(temp_circ)).area
        new_area = cascaded_union(list(temp_circ.drop([i]))).area
        # print('Initial Area: %6.2f; New Area: %6.2f' %(initial_area,new_area))


        if (initial_area-new_area)/circle.area <= percent:
            #print(i)
            ri.append(i)
            #fc.drop([i], inplace = True)

    fc.drop(ri, inplace = True)
    return fc, ri


def remove_circles_by_partition(df, circle_series, n=2, percent = 0):
    # fig= plt.figure()
    to_remove = []
    final_circles = circle_series.copy()

    for i, row in df.iterrows():

        # print(i)
        # get all circles within bound
        mini_df = df[df['x'].between(row['x'] - n, row['x'] + n) &
                     df['y'].between(row['y'] - n, row['y'] + n)].copy()

        #print('i: {}; Index: {}'.format(i, mini_df.index))
        #print('Length changing from {} to {}.'.format(len(mini_df), len(mini_df[mini_df.index >= i])))

        mini_df = mini_df[mini_df.index >= i]
        temp_circles = final_circles[mini_df.index]
        # get all circles that still exist in the series with these indexes

        initial_area = cascaded_union(list(temp_circles)).area
        new_area = cascaded_union(list(temp_circles.drop([i]))).area

        #print((initial_area - new_area) / temp_circles[i].area)
        if (initial_area-new_area)/circle_series[i].area <= percent:

            # print('Initial Area: %6.2f; New Area: %6.2f' %(initial_area,new_area))
            to_remove.append(i)

    final_circles.drop(to_remove, inplace = True)
    return final_circles, to_remove


def get_plotting_order(circle_series, rem_indexes):
    rem_indexes = rem_indexes
    plot_series = [circle_series[circle_series.index < rem_indexes[0]]]
    colors = ['blue']
    alphas = [0.3]
    for i, idx in enumerate(rem_indexes):
        plot_series.append(pd.Series(circle_series[idx]))
        colors.append('red')
        alphas.append(1)
        if i + 1 != len(rem_indexes):

            # print('[{}, {}]'.format(rem_indexes[i],rem_indexes[i+1]))
            s = circle_series[
                pd.Series(circle_series.index).between(rem_indexes[i], rem_indexes[i + 1], inclusive = False)]
            # print(len(s))

            plot_series.append(s)
            colors.append('blue')
            alphas.append(.3)


        else:
            s = circle_series[pd.Series(circle_series.index) > rem_indexes[i]]
            plot_series.append(s)
            colors.append('blue')
            alphas.append(.3)

    # print(len(plot_series))
    # print(len(colors))
    # print(len(alphas))

    return plot_series, colors, alphas