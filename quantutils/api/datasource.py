import pandas
import json
import os
import quantutils.dataset.pipeline as ppl


class MarketDataStore:

    def __init__(self, root, sources_file="datasources.json"):
        # TODO read root from quantutils properties file
        self.root = root
        self.sources_file = sources_file

    def getDatasources(self):
        return json.load(open("".join([self.root, "/", self.sources_file])))

    def loadMarketData(self, market_info, sample_unit):

        marketData = dict()

        # Loop over datasources...
        for datasource in self.getDatasources():

            DS_path = self.root + "/" + datasource["name"] + "/"

            # Get HDFStore
            hdfFile = DS_path + datasource["name"] + ".hdf"
            hdfStore = pandas.HDFStore(hdfFile, 'r')

            for market in market_info["markets"]:

                mktSrcs = [mkt["sources"] for mkt in datasource["markets"] if mkt["name"] == market][0]

                for source in mktSrcs:

                    print("Loading {} data from {} in {}.hdf".format(market, source["name"], datasource["name"]))
                    # Load Dataframe from store
                    tsData = hdfStore[source["name"]]

                    # Crop selected data set to desired ranges

                    tsData = ppl.cropDate(tsData, market_info["start"], market_info["end"])

                    # 28/6/21 Move this to before data is saved for performance reasons
                    # Resample all to dataset sample unit (to introduce nans in all missing periods)
                    # tsData = ppl.resample(tsData, source["sample_unit"])

                    tsData = ppl.resample(tsData, sample_unit)

                    # 06/06/18
                    # Remove NaNs and resample again, to remove partial NaN entries before merging
                    tsData = ppl.removeNaNs(tsData)
                    tsData = ppl.resample(tsData, sample_unit)

                    if market not in marketData:
                        marketData[market] = pandas.DataFrame()

                    marketData[market] = ppl.merge(tsData, marketData[market])

            hdfStore.close()

        return marketData

    def appendHDF(self, hdfFile, bucket, data, sample_unit, update=False):

        # Get HDFStore
        hdfStore = pandas.HDFStore(hdfFile, 'a')
        append = True
        # TODO Sort incoming data

        try:
            if '/' + bucket in hdfStore.keys():

                # Get first,last row
                nrows = hdfStore.get_storer(bucket).nrows
                last = hdfStore.select(bucket, start=nrows - 1, stop=nrows)

                # If this is entirely beyond the last element in the file... append
                # If not... update (incurring a full file re-write and performance hit), or throw exception
                if not data[data.index <= last.index[0]].empty:
                    # Update table with overlapped data
                    storedData = hdfStore.get(bucket)
                    data = ppl.merge(data, storedData)
                    append = False

                    if not update:
                        raise ValueError('Error: Entry already exists for data starting at index ' + str(data.index[0]))
                else:
                    data = ppl.merge(last, data)

            data = ppl.resample(data, sample_unit)
            if append:
                print("Appending data...")
                hdfStore.append(bucket, data, format='table', append=True)
            else:
                print("Re-writing table data for update...")
                hdfStore.put(bucket, data, format='table')

        finally:
            hdfStore.close()

    def refreshMarketData(self):

        # Loop over datasources...
        # TODO: In chronological order

        for datasource in self.getDatasources():

            DS_path = self.root + "/" + datasource["name"] + "/"
            SRC_path = DS_path + "raw/"

            # Get HDFStore
            hdfFile = DS_path + datasource["name"] + ".hdf"

            for market in datasource["markets"]:

                for source in market["sources"]:

                    # Loop over any source files...
                    for infile in os.listdir(SRC_path):

                        if infile.lower().startswith(source["name"].lower()):

                            print("Adding " + infile + " to " + market["name"] + " table")

                            # Load RAW data (assume CSV)
                            newData = pandas.read_csv(SRC_path + infile,
                                                      index_col=datasource["index_col"],
                                                      parse_dates=datasource["parse_dates"],
                                                      header=None,
                                                      names=["Date", "Time", "Open", "High", "Low", "Close"],
                                                      usecols=range(0, 6),
                                                      skiprows=datasource["skiprows"],
                                                      dayfirst=datasource["dayfirst"]
                                                      )

                            if newData is not None:

                                newData = ppl.localize(newData, datasource["timezone"], "UTC")

                                self.appendHDF(hdfFile, source["name"], newData, source["sample_unit"], update=True)

    def getHDF(self, hdfFile, bucket):

        # Get HDFStore
        hdfStore = pandas.HDFStore(hdfFile, 'r')
        data = None
        try:
            data = hdfStore.get(bucket)
        finally:
            hdfStore.close()
        return data

    # Remove a bucket from a hdfFile
    def deleteHDF(self, hdfFile, bucket):

        # Get HDFStore
        hdfStore = pandas.HDFStore(hdfFile, 'a')
        try:
            hdfStore.remove(bucket)
        finally:
            hdfStore.close()
