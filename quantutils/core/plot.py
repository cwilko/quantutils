

# Highcharts-based visualisation class
from highcharts import Highstock


class OHLCChart:

    options = {
        'yAxis': [{
            'type': 'logarithmic',
            'crosshair': {
                    'snap': False
            },
            'labels': {
                'align': 'right',
                'x': -3
            },
            'title': {
                'text': 'OHLC'
            },
            'height': '80%',
            'lineWidth': 2,
        }, {
            'height': '20%',
            'lineWidth': 2,
        }],
        'xAxis': [{
            'crosshair': True,
            'ordinal': False
        }],
    }

    modules = [
        "https://code.highcharts.com/stock/indicators/indicators-all.js",
        "https://code.highcharts.com/stock/modules/drag-panes.js",
        "https://code.highcharts.com/modules/annotations-advanced.js",
        "https://code.highcharts.com/modules/price-indicator.js",
        "https://code.highcharts.com/modules/full-screen.js",
        "https://code.highcharts.com/modules/stock-tools.js"
    ]

    css = [
        "https://code.highcharts.com/css/stocktools/gui.css",
        "https://code.highcharts.com/css/annotations/popup.css"
    ]

    rangeSelector = {
        'rangeSelector': {
            'dropdown': 'always',
            'buttons': [
                {
                    'type': 'hour',
                    'count': 120,
                    'text': 'Hourly',
                    'dataGrouping': {
                        'units': [['hour', [1]]]
                    }
                },
                {
                    'type': 'day',
                    'count': 30 * 12,
                    'text': 'Daily',
                    'dataGrouping': {
                        'units': [['day', [1]]]
                    }
                },
                {
                    'type': 'week',
                    'count': 4 * 52,
                    'text': 'Weekly',
                    'dataGrouping': {
                        'units': [['week', [1]]]
                    }
                },
                {
                    'type': 'month',
                    'count': 12 * 5,
                    'text': 'Monthly',
                    'dataGrouping': {
                        'units': [['month', [1]]]
                    }
                }
            ]
        }
    }

    def __init__(self, options=None):
        self.chart = Highstock()
        self.chart.add_JSsource(self.modules)
        self.chart.add_CSSsource(self.css)
        self.chart.set_dict_options(self.rangeSelector)
        if options:
            self.option = options
        self.chart.set_dict_options(self.options)
        self.indicatorCount = 0

    def addSeries(self, name, ohlc, type='candlestick', yAxis=0):
        self.chart.add_data_set(ohlc, type, name=name, id=name, yAxis=yAxis)

    def addIndicator(self, name, type, linkedTo, params, yAxis=1):
        self.chart.add_data_set([], type, name, linkedTo=linkedTo, params=params, dataGrouping={"enabled": False}, yAxis=yAxis)

    def getChart(self):
        return self.chart


##
# Visualise - taken from dataset.pipeline
##


def visualise(data, periods, count):

    # Plotly
    import plotly.offline as py
    import plotly.figure_factory as ff

    py.init_notebook_mode()  # run at the start of every ipython notebook

    csticks = data.values[0:count:, :periods * 4].ravel().reshape(-1, 4)

    fig = ff.create_candlestick(csticks[:, 0], csticks[:, 1], csticks[:, 2], csticks[:, 3])

    py.iplot(fig, filename='jupyter/simple-candlestick', validate=True)


##
# taken from core.statistics
##

def plot(ts, pnl, f=1):

    # Matplotlib basic charts
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num
    from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
    from mpl_finance import candlestick_ohlc

    plt.figure(0)
    plt.title("Strategy Returns")
    plt.plot(np.cumsum(np.log(1 + (pnl * f))))
    plt.show()

    # DATA PLOT
    fig = plt.figure(1)
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    # dayFormatter = DateFormatter('%d')      # e.g., 12

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    # ax.xaxis.set_minor_formatter(dayFormatter)

    #plot_day_summary(ax, quotes, ticksize=3)
    ts['d'] = ts.index.map(date2num)

    candlestick_ohlc(ax, ts[['d', 'Open', 'High', 'Low', 'Close']].astype('float32').values, width=0.6)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title("Underlying Security Prices")
    plt.show()
