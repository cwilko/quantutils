

# Highcharts-based visualisation class
from highcharts import Highstock


class Chart:

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

    groupingUnits = [
        [
            'minute',
            [1, 5, 15, 30]
        ], [
            'hour',
            [1, 4]
        ], [
            'day',
            [1]
        ], [
            'week',
            [1]
        ], [
            'month',
            [1]
        ]
    ]

    def __init__(self, options):
        self.chart = Highstock()
        self.chart.add_JSsource(self.modules)
        self.chart.add_CSSsource(self.css)
        self.chart.set_dict_options(options)

    def addSeries(self, name, ohlc, type='candlestick', yAxis=0):
        self.chart.add_data_set(ohlc, type, name=name, id=name + "_id", yAxis=yAxis, dataGrouping={
            'units': self.groupingUnits
        })

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
