import pandas as pd
import numpy as np
import numba
import itertools
import statsmodels.api as sm
from numpy.lib.stride_tricks import sliding_window_view
# import tsfresh.feature_extraction.feature_calculators as tsfe


def ttest(ts, new_west=True):
    """
    使用Newey-West Adjustment 计算t值
    """
    ts = ts.dropna()
    T = len(ts)
    J = int(np.floor(4 * (T/100)**(2/9)))
    const = np.ones_like(ts)

    rst1 = sm.OLS(ts, const).fit(cov_type='HAC',
                                 cov_kwds={"maxlags": J}, use_t=True)
    return rst1.tvalues[0]


def multi_col_rolling(df, func, window):
    """
    df.columns and func's input must be the same
    """
    slide = sliding_window_view(df.values, window_shape=(window, df.shape[1])).squeeze(1)
    rst = [func(s) for s in slide]
    rst = pd.Series(np.pad(rst, (df.shape[0]-len(rst), 0), constant_values=np.nan))
    rst.index = df.index

    return rst


class Factor:

    @staticmethod
    def ts_diff(df, group_col, fcol, n=1):
        df[f'ts_diff({fcol},{n})'] = df.groupby(group_col)[fcol].diff(n)

        return df

    def ts_pct_change(df, group_col, fcol, n=1):
        df[f'ts_pct_change({fcol},{n})'] = df.groupby(group_col)[fcol].pct_change(n)

        return df

    @staticmethod
    def ts_std(df, group_col, fcol, n=20, clip=None, qclip=None, min_periods=None):
        """时序标准差

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            clip (tuple, optional): clip range, e.g.(-3,3). Defaults to None.
            qclip (tuple, optional): quantile clip range,e.g. (0.01,0.99). Defaults to None.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_std(fcol,n)']
        """
        df['ts_std(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).std().reset_index(level=0, drop=True)
        if clip:
            df['ts_std(%s,%s)' % (fcol, n)] = df['ts_std(%s,%s)' %
                                                 (fcol, n)].clip(clip[0], clip[1])
        if qclip:
            low = df['ts_std(%s,%s)' % (fcol, n)].quantile(qclip[0])
            high = df['ts_std(%s,%s)' % (fcol, n)].quantile(qclip[1])
            df['ts_std(%s,%s)' % (fcol, n)] = df['ts_std(%s,%s)' %
                                                 (fcol, n)].clip(low, high)

        return df

    @staticmethod
    def ts_mean(df, group_col, fcol, n=20, min_periods=None):
        """时序均值

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_mean(fcol,n)']
        """
        df['ts_mean(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).mean().reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_sum(df, group_col, fcol, n=20, min_periods=None):
        """时序求和

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_sum(fcol,n)']
        """
        df['ts_sum(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).sum().reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_skew(df, group_col, fcol, n=20, min_periods=None):
        """时序偏度

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_skew(fcol,n)']
        """
        df['ts_skew(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).skew().reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_kurt(df, group_col, fcol, n=20, min_periods=None):
        """时序峰度

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_kurt(fcol,n)']
        """
        df['ts_kurt(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).kurt().reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_median_abs_deviation(df, group_col, fcol, n=20, min_periods=None):
        """
        median(abs(x - x.mean()))
        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_median_abs_deviation(fcol,n)']
        """
        @numba.jit(nopython=True)
        def median_abs_deviation(x):
            return np.nanmedian(np.abs(x - np.mean(x)))

        df['ts_median_abs_deviation(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(median_abs_deviation, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_rms(df, group_col, fcol, n=20, min_periods=None):
        """
        np.sqrt(np.mean(x**2))
        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_rms(fcol,n)']
        """
        @numba.jit(nopython=True)
        def rms(x):
            return np.sqrt(np.nanmean(x**2))

        df['ts_rms(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(rms, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_norm_mean(df, group_col, fcol, n=20, min_periods=None):
        """
        mean / rms
        """
        @numba.jit(nopython=True)
        def norm_mean(x):
            rms = np.sqrt(np.nanmean(x**2))

            return (np.nanmean(x) / rms) if (rms != 0) else np.nan

        df['ts_norm_mean(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(norm_mean, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_norm_max(df, group_col, fcol, n=20, min_periods=None):
        """
        max / rms
        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_norm_max(fcol,n)']
        """
        @numba.jit(nopython=True)
        def norm_max(x):
            rms = np.sqrt(np.nanmean(x**2))

            return (np.nanmax(x) / rms) if (rms != 0) else np.nan

        df['ts_norm_max(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(norm_max, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_norm_min(df, group_col, fcol, n=20, min_periods=None):
        """
        min / rms
        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_norm_min(fcol,n)']
        """
        @numba.jit(nopython=True)
        def norm_min(x):
            rms = np.sqrt(np.nanmean(x**2))

            return (np.nanmin(x) / rms) if (rms != 0) else np.nan

        df['ts_norm_min(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(norm_min, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_norm_min_max(df, group_col, fcol, n=20, min_periods=None):
        """
        (max - min) / rms
        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_norm_min_max(fcol,n)']
        """
        @numba.jit(nopython=True)
        def norm_min_max(x: np.array):
            rms = np.sqrt(np.nanmean(x**2))

            return ((np.nanmax(x) - np.nanmin(x)) / rms) if (rms != 0) else np.nan

        df['ts_norm_min_max(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(norm_min_max, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_longest_strike_above_mean(df, group_col, fcol, n=20):
        """
        高于均值的最长时间
        """

        @numba.jit(nopython=True)
        def longest_steike_above_mean(x):
            abv_mean = x > np.mean(x)
            max_count = 0
            cur_count = 0
            for i in abv_mean:
                if i:
                    cur_count += 1
                else:
                    if cur_count > max_count:
                        max_count = cur_count
                    cur_count = 0

            return max_count

        df['ts_longest_strike_above_mean(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n).apply(longest_steike_above_mean, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_longest_strike_below_mean(df, group_col, fcol, n=20):
        """
        低于均值的最长时间
        """

        @numba.jit(nopython=True)
        def longest_steike_below_mean(x):
            abv_mean = x < np.mean(x)
            max_count = 0
            cur_count = 0
            for i in abv_mean:
                if i:
                    cur_count += 1
                else:
                    if cur_count > max_count:
                        max_count = cur_count
                    cur_count = 0

            return max_count

        df['ts_longest_strike_below_mean(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n).apply(longest_steike_below_mean, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_mean_over_1norm(df, group_col, fcol, n=20, min_periods=None):
        """mean(x) / mean(abs(x))

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_mean_over_1norm(fcol,n)']
        """
        df['mean'] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        df['abs'] = df[fcol].abs()
        df['abs_mean'] = df.groupby(group_col)['abs'].rolling(
            n, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        df['ts_mean_over_1norm(%s,%s)' % (
            fcol, n)] = df['mean'] / df['abs_mean']
        df.drop(columns=['mean', 'abs', 'abs_mean'], inplace=True)

        return df

    @staticmethod
    def ts_norm(df, group_col, fcol, n=20, clip=True, min_periods=None):
        df['mean'] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        df['std'] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).std().reset_index(level=0, drop=True)
        if clip:
            df['ts_norm(%s,%s)' % (fcol, n)] = (
                (df[fcol] - df['mean']) / df['std']).clip(-3, 3)
        else:
            df['ts_norm(%s,%s)' % (fcol, n)] = (
                (df[fcol] - df['mean']) / df['std'])

        df['ts_norm(%s,%s)' % (fcol, n)] = df['ts_norm(%s,%s)' %
                                              (fcol, n)].replace(-np.inf, np.nan)
        df = df.drop(columns=['mean', 'std'])

        return df

    @staticmethod
    def ts_zscore(df, group_col, fcol, n=20, clip=True, min_periods=None):

        return Factor.ts_norm(df, group_col, fcol, n=20, clip=True, min_periods=None)

    @staticmethod
    def ts_up_count_ratio(df, group_col, fcol, n=20, min_periods=None):
        """上涨比例

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['up_count_ratio(fcol,n)']
        """

        df['count'] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).count().reset_index(level=0, drop=True)
        df['up_count'] = df[fcol] > 0
        df['up_count'] = df.groupby(group_col)['up_count'].rolling(
            n, min_periods=min_periods).sum().reset_index(level=0, drop=True)
        df['ts_up_count_ratio(%s,%s)' % (
            fcol, n)] = df['up_count'] / df['count']
        df.drop(columns=['count', 'up_count'], inplace=True)

        return df

    @staticmethod
    def ts_up_ratio(df, group_col, fcol, n=20, min_periods=None):
        """sum(x * dirac(x>0)) / sum(abs(x))

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['up_count_ratio(fcol,n)']
        """
        df['abs'] = df[fcol].abs()
        df['up'] = df[fcol].mask(df[fcol] < 0, 0)
        df['abs'] = df.groupby(group_col)['abs'].rolling(
            n, min_periods=min_periods).sum().reset_index(level=0, drop=True)
        df['up_sum'] = df.groupby(group_col)['up'].rolling(
            n, min_periods=min_periods).sum().reset_index(level=0, drop=True)
        df['ts_up_ratio(%s,%s)' % (fcol, n)] = df['up_sum'] / df['abs']
        df.drop(columns=['abs', 'up', 'up_sum'], inplace=True)

        return df

    @staticmethod
    def ts_rank(df, group_col, fcol, n=20, min_periods=None):
        """

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_rank(fcol,n)']
        """
        @numba.jit(nopython=True)
        def ts_rank(x):
            last = x[-1]
            pct_score = (x < last).sum() / (~np.isnan(x)).sum()
            return pct_score
        df['ts_rank(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(ts_rank, raw=True).reset_index(level=0, drop=True)

        return df

    def ts_min_max_quantile(df, group_col, fcol, n=20, min_periods=None):
        """ (x-x.min()) / (x.max()-x.min())

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 20.
            min_periods (int, optional): rolling min_periods. Defaults to None.

        Returns:
           df (pd.DataFrame): columns=['ts_min_max_quantile(fcol,n)']
        """

        df['max'] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).max().reset_index(level=0, drop=True)
        df['min'] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).min().reset_index(level=0, drop=True)
        df['ts_min_max_quantile(%s,%s)' % (fcol, n)] = (
            df[fcol] - df['min']) / (df['max'] - df['min'])
        df = df.drop(columns=['min', 'max'])

        return df

    @staticmethod
    def ts_raw_SD(df, group_col, fcol, n=5):
        """二阶差分 [f(t) - f(t-n)] - [f(t-n) - f(t-2n)]

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 5.
        Returns:
            "ts_raw_SD(fcol,n-2n)"
        """

        df['short'] = df.groupby(group_col)[fcol].diff(n)
        df['long'] = df.groupby(group_col)[fcol].diff(2*n)
        df['ts_raw_SD(%s,%s-%s)' % (fcol, n, 2*n)] = df['short'] - (df['long'] - df['short'])

        df.drop(columns=['short', 'long'], inplace=True)

        return df

    @staticmethod
    def ts_pct_SD(df, group_col, fcol, n=5):
        """二阶导数 [f(t) / f(t-n)] - [f(t-n) / f(t-2n)]

        Args:
            df (pd.DataFrame): columns=[group_col, fcol]
            group_col (str, List[str]): groupby
            fcol (str): factor column
            n (int, optional): window size. Defaults to 5.
        Returns:
            "ts_pct_SD(fcol,n-2n)"
        """

        df['short'] = df.groupby(group_col)[fcol].pct_change(n) + 1
        df['long'] = df.groupby(group_col)[fcol].pct_change(2*n) + 1
        df['ts_pct_SD(%s,%s-%s)' % (fcol, n, 2*n)] = df['short'] - \
            (df['long'] / df['short'])

        df.drop(columns=['short', 'long'], inplace=True)

        return df

    @staticmethod
    def ts_linear_trend(df, group_col, fcol, return_var='e', n=20, clip=None, qclip=None):
        """
        y= bt ＋ e,计算slope or residual or ic

        return_var:
            - 'e': residual, 去掉trend 之后的结果，保留了fcol 的scaling
            - 'b': slope, 相当于判断trend 强度，该结果保留了fcol 的scaling
            - 'ic': np.sqrt(r2)，trend 强度，与fcol scaling 无关
            - 'std_e': standardised residual，与fcol scaling无关
        """
        @numba.jit(nopython=True)
        def calc_slope(y):
            y = y-y.mean()
            X = np.arange(y.shape[0]) - np.arange(y.shape[0]).mean()
            beta = X.T.dot(y) / (X.T.dot(X))

            return beta

        @numba.jit(nopython=True)
        def calc_stddev_e(y):
            y = y-y.mean()
            X = np.arange(y.shape[0]) - np.arange(y.shape[0]).mean()
            beta = X.T.dot(y)/(X.T.dot(X))
            e = y - beta*X
            try:
                e = e[-1]/np.nanstd(e)
            except:
                e = np.nan
            return e

        @numba.jit(nopython=True)
        def calc_e(y):
            y = y-y.mean()
            X = np.arange(y.shape[0]) - np.arange(y.shape[0]).mean()
            beta = X.T.dot(y)/(X.T.dot(X))
            e = y - beta*X

            return e[-1]

        if return_var == 'b':
            df['ts_slope(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
                n).apply(calc_slope, raw=True).reset_index(level=0, drop=True)
            col = 'ts_slope(%s,%s)' % (fcol, n)
        elif return_var == 'ic':
            x = np.arange(n)
            df['ts_corr(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
                n).apply(calc_slope, raw=True).reset_index(level=0, drop=True)
            df['ts_corr(%s,%s)' % (fcol, n)] = df['ts_corr(%s,%s)' % (fcol, n)] * x.std() / df.groupby(
                group_col)[fcol].rolling(n).std().reset_index(level=0, drop=True)  # b=ic＊std(y)／std(x)反推出ic
            col = 'ts_corr(%s,%s)' % (fcol, n)
        elif return_var == 'std_e':
            df['ts_std_resid(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
                n).apply(calc_stddev_e, raw=True).reset_index(level=0, drop=True)
            col = 'ts_std_resid(%s,%s)' % (fcol, n)
        else:

            df['ts_resid(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
                n).apply(calc_e, raw=True).reset_index(level=0, drop=True)
            col = 'ts_resid(%s,%s)' % (fcol, n)

        if clip:
            df[col] = df[col].clip([0], clip[1])
        if qclip:
            low = df[col].quantile(qclip[0])
            high = df[col].quantile(qclip[1])
            df[col] = df[col].clip(low, high)

        return df

    @staticmethod
    def ts_xar_resid(df, group_col, ycol, xcol, lag=5, n=20, return_var='e'):
        """yt = xt_1 + xt_2

        Args:
            return_var 
                - 'e': residual, 去掉trend 之后的结果，保留了fcol 的scaling
                - 'b': slope, 相当于判断trend 强度，该结果保留了fcol 的scaling
                - 'ic': np.sqrt(r2)，trend 强度，与fcol scaling 无关
                - 'std_e': standardised residual，与fcol scaling无关
        """

        name_list = []
        grouped = df.groupby(group_col, sort=False)
        for lag in range(1, lag+1):
            name = "lag(%s,%s)" % (xcol, lag)
            df[name] = grouped[xcol].shift(lag)
            name_list.append(name)

        df = Factor.groupby_ts_regression(
            df, group_col, ycol, name_list, n, add_conststant=True, return_var=return_var)
        drop_col = set(name_list).intersection(set(df.columns.tolist()))
        df.drop(columns=list(drop_col), inplace=True)

        return df

    @staticmethod
    def ts_max(df, group_col, fcol, n=20, min_periods=None):
        """

        """

        df['ts_max(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).max().reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_min(df, group_col, fcol, n=20, min_periods=None):
        """

        """

        df['ts_min(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).min().reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_last_location_maximum(df, group_col, fcol, n=20, min_periods=None):
        """

        """
        @numba.jit(nopython=True)
        def last_location_maximum(x):

            return 1 - (np.argmax(x[::-1]) / len(x)) if len(x) > 0 else np.nan

        df['ts_last_location_maximum(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(last_location_maximum, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_last_location_minimum(df, group_col, fcol, n=20, min_periods=None):
        """

        """
        @numba.jit(nopython=True)
        def last_location_minimum(x):

            return 1 - (np.argmin(x[::-1]) / len(x)) if len(x) > 0 else np.nan

        df['ts_last_location_minimum(%s,%s)' % (fcol, n)] = df.groupby(group_col)[fcol].rolling(
            n, min_periods=min_periods).apply(last_location_minimum, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_coef_of_variation(df, group_col, fcol, n=20):
        """
        sigma/mu
        """
        df['std'] = df.groupby(group_col)[fcol].rolling(
            n).std().reset_index(level=0, drop=True)
        df['mean'] = df.groupby(group_col)[fcol].rolling(
            n).mean().reset_index(level=0, drop=True)
        df['ts_coef_of_variation(%s)' % (n,)] = df['std']/df['mean']

        df.drop(columns=['std', 'mean'], inplace=True)

        return df

    @staticmethod
    def ts_longest_up_strike(df, group_col, fcol, n=20):
        """
        ts_longest_up_strike
        """
        @numba.jit(nopython=True)
        def ts_longest_up_strike(x):
            max_length = 0
            current_length = 0
            for i in range(1, len(x)):
                if x[i] > x[i - 1]:
                    current_length += 1
                else:
                    current_length = 0

                if current_length > max_length:
                    max_length = current_length

            return max_length/(len(x))

        df[f'ts_longest_up_strike({fcol},{n})'] = df.groupby(group_col)[fcol].rolling(
            n).apply(ts_longest_up_strike, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_longest_down_strike(df, group_col, fcol, n=20):
        """
        ts_longest_down_strike
        """
        @numba.jit(nopython=True)
        def ts_longest_down_strike(x):
            max_length = 0
            current_length = 0
            for i in range(1, len(x)):
                if x[i] < x[i - 1]:
                    current_length += 1
                else:
                    current_length = 0

                if current_length > max_length:
                    max_length = current_length

            return max_length/(len(x))

        df[f'ts_longest_down_strike({fcol},{n})'] = df.groupby(group_col)[fcol].rolling(
            n).apply(ts_longest_down_strike, raw=True).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def ts_corr(df, group_col, fcol1, fcol2, n=10, corr_method='pearson'):
        """
        rolling correlation of fcol1 and fcol2
        """
        tmp = df.groupby(group_col).rolling(
            n)[[fcol1, fcol2]].corr(method=corr_method).droplevel([0, 2])
        df[f'ts_corr({fcol1},{fcol2},{n})'] = - tmp.reset_index().drop_duplicates(
            subset=['index'], keep='last').set_index('index').iloc[:, 0]

        return df

    @staticmethod
    def cs_rank(df, fcol, date_col='TradingDay'):
        """
        """
        df[f'cs_rank({fcol})'] = df.groupby(date_col)[fcol].rank(pct=True)

        return df

    @staticmethod
    def cs_neutralize(df, fcol, desize_col, date_col='TradingDay', rst_col_name=None, category=False):
        """
        category: desize_col 是否是类别变量
        """

        rst_col_name = f'cs_neutralize({fcol},{desize_col})' if rst_col_name is None else rst_col_name
        if category:
            df['__mean'] = df.groupby([date_col, desize_col])[
                fcol].transform('mean')
            df['__std'] = df.groupby([date_col, desize_col])[
                fcol].transform('std')
            df[rst_col_name] = (df[fcol] - df['__mean']) / df['__std']

            df.drop(columns=['__mean', '__std'], errors='ignore', inplace=True)

        else:
            df[rst_col_name] = df.groupby(date_col).apply(
                lambda x: sm.OLS(x[fcol], sm.add_constant(
                    x[[desize_col]])).fit().resid
            ).reset_index(level=0, drop=True)

        return df

    @staticmethod
    def comb_sum(df, col1, col2, comparable=True):
        """
        因子相加，如果不可比则取截面rank
        """
        if not comparable:
            df = Factor.cs_rank(df, col1)
            df = Factor.cs_rank(df, col2)
            col1, col2 = f"cs_rank({col1})", f"cs_rank({col2})"

        df[f'comb_sum({col1},{col2})'] = df[col1] + df[col2]
        if not comparable:
            df.drop(columns=[col1, col2], inplace=True, errors='ignore')

        return df

    @staticmethod
    def comb_mul(df, col1, col2, comparable=True):
        """
        因子相乘，如果不可比则取截面rank
        """
        if not comparable:
            df = Factor.cs_rank(df, col1)
            df = Factor.cs_rank(df, col2)
            col1, col2 = f"cs_rank({col1})", f"cs_rank({col2})"

        df[f'comb_mul({col1},{col2})'] = df[col1] * df[col2]
        if not comparable:
            df.drop(columns=[col1, col2], inplace=True, errors='ignore')

        return df

    @staticmethod
    def create_lag(df, date_col, secu_col, lag_col, lag=10, inplace=True):
        """
        df: pd.DataFrame, columns = [date_col, entity_col, x_col], 
        lag: int, the number of lags to created for df
        return:
            df:pd.DataFrame, the dataframe with created lags
            name_list: list of str, the list of created lag column names 

        """
        df_list = []
        tmp_df = df.pivot(index=date_col, columns=secu_col, values=lag_col)
        if not isinstance(lag, tuple):
            lag = (1, lag)
        for la in range(lag[0], lag[1]+1):
            la_str = '+%s' % (la) if la >= 0 else '%s' % (la)
            xname = "%s(t%s)" % (lag_col, la_str)
            ts = tmp_df.shift(-la).unstack().rename(xname)
            df_list.append(ts)
        df_lag = pd.concat(df_list, axis=1).reset_index()
        if inplace:
            return df.merge(df_lag, on=[date_col, secu_col], how='left')
        else:
            return df_lag

    # ==================== Optiver 2023 专用因子 ====================
    # 针对 Trading at Close 竞赛的 Auction Order Book 特性

    @staticmethod
    def auction_price_pressure(df, group_col='stock_id'):
        """
        拍卖价格压力 - near_price 和 far_price 的差异

        near_price: 包含连续市场订单的清算价
        far_price: 仅包含拍卖订单的清算价
        差异反映了连续市场订单对价格的影响

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了 auction_price_pressure 列
        """
        # 价格压力（绝对值）
        df['auction_price_pressure'] = df['near_price'] - df['far_price']

        # 价格压力（相对值，基于 wap）
        df['auction_price_pressure_pct'] = df['auction_price_pressure'] / df['wap']

        # 价格压力方向
        df['auction_price_pressure_sign'] = np.sign(df['auction_price_pressure'])

        return df

    @staticmethod
    def auction_imbalance_intensity(df, group_col='stock_id'):
        """
        拍卖不平衡强度

        imbalance_size: 未匹配的金额
        matched_size: 可匹配的金额
        比率反映了市场失衡的严重程度

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了不平衡强度相关列
        """
        # 不平衡比率
        df['imbalance_ratio'] = df['imbalance_size'] / (df['matched_size'] + df['imbalance_size']).replace(0, np.nan)

        # 带符号的不平衡比率
        df['signed_imbalance_ratio'] = df['imbalance_ratio'] * df['imbalance_buy_sell_flag']

        # 不平衡金额占总成交量的比例
        total_volume = df['bid_size'] + df['ask_size']
        df['imbalance_volume_ratio'] = df['imbalance_size'] / total_volume.replace(0, np.nan)

        return df

    @staticmethod
    def auction_reference_price_quality(df, group_col='stock_id'):
        """
        参考价格质量指标

        reference_price 是使配对最大化、不平衡最小化的价格
        分析其与市场价格的关系

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了参考价格质量相关列
        """
        # 参考价格与 WAP 的偏离
        df['ref_price_deviation'] = df['reference_price'] - df['wap']
        df['ref_price_deviation_pct'] = df['ref_price_deviation'] / df['wap']

        # 参考价格与买卖中间价的偏离
        df['bid_ask_mid'] = (df['bid_price'] + df['ask_price']) / 2
        df['ref_price_mid_deviation'] = df['reference_price'] - df['bid_ask_mid']

        # 参考价格在买卖价差中的位置 (0-1, 0=bid, 1=ask)
        bid_ask_spread = df['ask_price'] - df['bid_price']
        df['ref_price_position'] = (df['reference_price'] - df['bid_price']) / bid_ask_spread.replace(0, np.nan)

        return df

    @staticmethod
    def auction_liquidity_profile(df, group_col='stock_id'):
        """
        拍卖流动性特征

        分析订单簿的流动性状况

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了流动性相关列
        """
        # 总流动性
        df['total_liquidity'] = df['matched_size'] + df['imbalance_size']

        # 买卖方流动性不平衡
        df['bid_ask_liquidity_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size']).replace(0, np.nan)

        # 拍卖订单占比（相对于连续市场）
        continuous_liquidity = df['bid_size'] + df['ask_size']
        df['auction_continuous_ratio'] = df['total_liquidity'] / continuous_liquidity.replace(0, np.nan)

        # 匹配效率（能够匹配的比例）
        df['matching_efficiency'] = df['matched_size'] / df['total_liquidity'].replace(0, np.nan)

        return df

    @staticmethod
    def auction_price_discovery(df, group_col='stock_id'):
        """
        价格发现过程特征

        分析 near_price, far_price, reference_price 之间的关系

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了价格发现相关列
        """
        # near 和 far 价格的平均值
        df['near_far_mid'] = (df['near_price'] + df['far_price']) / 2

        # reference_price 与 near_far_mid 的偏离
        df['ref_nearfar_deviation'] = df['reference_price'] - df['near_far_mid']

        # near_price 与 reference_price 的偏离
        df['near_ref_deviation'] = df['near_price'] - df['reference_price']
        df['near_ref_deviation_pct'] = df['near_ref_deviation'] / df['reference_price']

        # far_price 与 reference_price 的偏离
        df['far_ref_deviation'] = df['far_price'] - df['reference_price']
        df['far_ref_deviation_pct'] = df['far_ref_deviation'] / df['reference_price']

        # 价格一致性（三个价格的标准差）
        df['price_consistency'] = df[['near_price', 'far_price', 'reference_price']].std(axis=1)

        return df

    @staticmethod
    def auction_order_book_pressure(df, group_col='stock_id'):
        """
        订单簿压力特征

        分析买卖双方的压力

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了订单簿压力相关列
        """
        # 买卖价差
        df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
        df['bid_ask_spread_pct'] = df['bid_ask_spread'] / df['wap']

        # 买卖量比
        df['bid_ask_size_ratio'] = df['bid_size'] / df['ask_size'].replace(0, np.nan)
        df['bid_ask_size_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size']).replace(0, np.nan)

        # WAP 与买卖中间价的偏离
        df['wap_mid_deviation'] = df['wap'] - df['bid_ask_mid']
        df['wap_mid_deviation_pct'] = df['wap_mid_deviation'] / df['bid_ask_mid']

        # WAP 在买卖价差中的位置
        df['wap_position'] = (df['wap'] - df['bid_price']) / df['bid_ask_spread'].replace(0, np.nan)

        return df

    @staticmethod
    def auction_time_dynamics(df, group_col='stock_id', n=5):
        """
        拍卖时间动态特征

        分析随着时间推移，拍卖特征的变化

        Args:
            df: DataFrame
            group_col: 分组列
            n: 时间窗口

        Returns:
            df: 添加了时间动态相关列
        """
        # 不平衡的时序变化
        df[f'imbalance_size_change_{n}'] = df.groupby(group_col)['imbalance_size'].diff(n)
        df[f'imbalance_size_pct_change_{n}'] = df.groupby(group_col)['imbalance_size'].pct_change(n)

        # 不平衡方向的变化（翻转次数）
        df[f'imbalance_flag_change_{n}'] = df.groupby(group_col)['imbalance_buy_sell_flag'].diff(n)

        # 参考价格的变化
        df[f'reference_price_change_{n}'] = df.groupby(group_col)['reference_price'].diff(n)
        df[f'reference_price_pct_change_{n}'] = df.groupby(group_col)['reference_price'].pct_change(n)

        # near_price 的变化
        df[f'near_price_change_{n}'] = df.groupby(group_col)['near_price'].diff(n)
        df[f'near_price_pct_change_{n}'] = df.groupby(group_col)['near_price'].pct_change(n)

        # 匹配量的变化
        df[f'matched_size_change_{n}'] = df.groupby(group_col)['matched_size'].diff(n)
        df[f'matched_size_pct_change_{n}'] = df.groupby(group_col)['matched_size'].pct_change(n)

        return df

    @staticmethod
    def auction_imbalance_momentum(df, group_col='stock_id', windows=[3, 5, 10]):
        """
        不平衡动量特征

        分析不平衡的趋势和动量

        Args:
            df: DataFrame
            group_col: 分组列
            windows: 时间窗口列表

        Returns:
            df: 添加了不平衡动量相关列
        """
        # 带符号的不平衡
        df['signed_imbalance'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']

        for window in windows:
            # 不平衡的移动平均
            df[f'signed_imbalance_ma_{window}'] = df.groupby(group_col)['signed_imbalance'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            # 不平衡的标准差
            df[f'signed_imbalance_std_{window}'] = df.groupby(group_col)['signed_imbalance'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

            # 不平衡的动量（当前值 - 移动平均）
            df[f'signed_imbalance_momentum_{window}'] = df['signed_imbalance'] - df[f'signed_imbalance_ma_{window}']

            # 不平衡方向的持续性（同方向的次数）
            df[f'imbalance_direction_persistence_{window}'] = df.groupby(group_col)['imbalance_buy_sell_flag'].transform(
                lambda x: x.rolling(window=window, min_periods=1).apply(lambda y: (y == y.iloc[-1]).sum() if len(y) > 0 else 0)
            )

        return df

    @staticmethod
    def auction_price_volatility(df, group_col='stock_id', windows=[5, 10, 20]):
        """
        拍卖价格波动特征

        分析价格的波动性

        Args:
            df: DataFrame
            group_col: 分组列
            windows: 时间窗口列表

        Returns:
            df: 添加了价格波动相关列
        """
        for window in windows:
            # WAP 的波动率
            df[f'wap_volatility_{window}'] = df.groupby(group_col)['wap'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

            # reference_price 的波动率
            df[f'reference_price_volatility_{window}'] = df.groupby(group_col)['reference_price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

            # near_price 的波动率
            df[f'near_price_volatility_{window}'] = df.groupby(group_col)['near_price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

            # 买卖价差的波动率
            df[f'spread_volatility_{window}'] = df.groupby(group_col)['bid_ask_spread'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

        return df

    @staticmethod
    def auction_cross_features(df, group_col='stock_id'):
        """
        拍卖交叉特征

        组合不同字段创建交叉特征

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了交叉特征列
        """
        # 不平衡金额（带符号）
        df['signed_imbalance_amount'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']

        # 不平衡金额与价格的交叉
        df['imbalance_price_impact'] = df['signed_imbalance_amount'] * df['auction_price_pressure']

        # 不平衡与买卖价差的交叉
        df['imbalance_spread_interaction'] = df['imbalance_ratio'] * df['bid_ask_spread_pct']

        # 流动性与价格压力的交叉
        df['liquidity_pressure_interaction'] = df['total_liquidity'] * df['auction_price_pressure_pct']

        # 匹配效率与不平衡的交叉
        df['efficiency_imbalance_interaction'] = df['matching_efficiency'] * df['imbalance_ratio']

        # 时间因素（距离收盘的时间）
        df['time_to_close'] = 600 - df['seconds_in_bucket']  # 假设总共600秒
        df['time_urgency'] = df['imbalance_size'] / (df['time_to_close'] + 1)  # 避免除以0

        return df

    @staticmethod
    def auction_all_features(df, group_col='stock_id', windows=[3, 5, 10]):
        """
        一键生成所有 Optiver 拍卖相关特征

        Args:
            df: DataFrame
            group_col: 分组列
            windows: 时间窗口列表

        Returns:
            df: 添加了所有拍卖特征的 DataFrame
        """
        print("生成拍卖价格压力特征...")
        df = Factor.auction_price_pressure(df, group_col)

        print("生成拍卖不平衡强度特征...")
        df = Factor.auction_imbalance_intensity(df, group_col)

        print("生成参考价格质量特征...")
        df = Factor.auction_reference_price_quality(df, group_col)

        print("生成流动性特征...")
        df = Factor.auction_liquidity_profile(df, group_col)

        print("生成价格发现特征...")
        df = Factor.auction_price_discovery(df, group_col)

        print("生成订单簿压力特征...")
        df = Factor.auction_order_book_pressure(df, group_col)

        print("生成时间动态特征...")
        for n in [1, 3, 5]:
            df = Factor.auction_time_dynamics(df, group_col, n)

        print("生成不平衡动量特征...")
        df = Factor.auction_imbalance_momentum(df, group_col, windows)

        print("生成价格波动特征...")
        df = Factor.auction_price_volatility(df, group_col, windows)

        print("生成交叉特征...")
        df = Factor.auction_cross_features(df, group_col)

        print(f"完成！共生成 {len(df.columns)} 列特征")

        return df
