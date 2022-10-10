import pandas
import quantutils.dataset.pipeline as ppl
from quantutils.core.decorators import synchronized
from quantutils.api.marketinsights import PriceStore
import json
import sys


class MarketDataStore:

    def __new__(cls, location, remote=False, hdfFile="data.hdf"):

        if (remote):
            return MarketDataStoreRemote(location)
        else:
            return super().__new__(cls)

    def __init__(self, location, hdfFile="data.hdf"):
        self.hdfFile = location + "/" + hdfFile

    # Load data from an ordered list of sources
    def aggregate(self, sources, sample_unit, start="1979-01-01", end="2050-01-01", debug=False):

        hdfStore = pandas.HDFStore(self.hdfFile, 'r')

        try:
            marketData = None

            for source in sources:

                datasource = ''.join(['/', source])

                if datasource in hdfStore.keys():

                    print("Loading data from {} in {}".format(source, self.hdfFile))

                    # Load Dataframe from store
                    select_stmt = ''.join(["index>'", start, "' and index<='", end, "'"])
                    tsData = hdfStore.select(datasource, where=select_stmt)

                    if not tsData.empty:

                        # 28/6/21 Move this to before data is saved for performance reasons
                        # Resample all to dataset sample unit (to introduce nans in all missing periods)
                        # tsData = ppl.resample(tsData, source["sample_unit"])

                        # Resample to the requested unit
                        tsData = ppl.resample(tsData, sample_unit)

                        # 06/06/18
                        # Remove NaNs and resample again, to remove partial NaN entries before merging
                        tsData = ppl.removeNaNs(tsData)
                        tsData = ppl.resample(tsData, sample_unit)

                        if marketData is None:
                            marketData = pandas.DataFrame()

                        marketData = ppl.merge(tsData, marketData)
                else:
                    raise ValueError('Error: Cannot find datasource: ' + datasource)
        finally:
            hdfStore.close()

        return marketData

    @synchronized
    def append(self, source_id, data, source_sample_unit, update=False, debug=False):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'a')
        append = True
        # Sort incoming data
        data = data.sort_index()

        if debug:
            print("Request to add data to table: " + source_id, flush=True)
        #print(data, flush=True)

        try:
            if '/' + source_id in hdfStore.keys():

                if debug:
                    print("Table found: " + source_id, flush=True)

                # Get first,last row
                nrows = hdfStore.get_storer(source_id).nrows
                last = hdfStore.select(source_id, start=nrows - 1, stop=nrows)

                # If this is entirely beyond the last element in the file... append
                # If not... update (incurring a full file re-write and performance hit), or throw exception
                if not data[data.index.get_level_values(0) <= last.index.get_level_values(0)[0]].empty:
                    # Update table with overlapped data
                    storedData = hdfStore.get(source_id)
                    # Oct 22 - Switch order of new vs old, i.e. Purposely do an update if update=True
                    data = ppl.merge(storedData, data)
                    append = False

                    if not update:
                        raise ValueError('Error: Entry already exists for data starting at index ' + str(data.index[0]))
                # Oct 22 - Moving away from NaN filled tables to sparse
                # Mainly due to technical limitation on sampling for multiindex dataframes
                #
                # else:
                #    data = ppl.merge(last, data)
                #
                #data = ppl.resample(data, source_sample_unit)
                if append:
                    if debug:
                        print("Appending data...", flush=True)
                    hdfStore.append(source_id, data, format='table', append=True)
                else:
                    if debug:
                        print("Re-writing table data for update...", flush=True)
                    hdfStore.put(source_id, data, format='table')
            else:
                # Oct 22 - see above comment
                #data = ppl.resample(data, source_sample_unit)
                if debug:
                    print("Creating new table for data...", flush=True)
                hdfStore.put(source_id, data, format='table')

        finally:
            hdfStore.close()

        if debug:
            print("Update complete", flush=True)
        sys.stdout.flush()

    def get(self, source_id):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'r')
        data = None
        try:
            data = hdfStore.get(source_id)
        finally:
            hdfStore.close()
        return data

    # Vanilla put of any data
    @synchronized
    def put(self, source_id, data, update=False):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'a')
        print("Request to add data to table: " + source_id, flush=True)
        #print(data, flush=True)
        try:
            if '/' + source_id in hdfStore.keys():
                storedData = hdfStore.get(source_id)
                if update:
                    data = ppl.merge(storedData, data)
                else:
                    data = ppl.merge(data, storedData)

            hdfStore.put(source_id, data, format='table')
        finally:
            hdfStore.close()
        return data

    # Remove a node from a hdfFile
    @synchronized
    def delete(self, source_id):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'a')
        try:
            hdfStore.remove(source_id)
        finally:
            hdfStore.close()

    def getKeys(self):
        hdfStore = pandas.HDFStore(self.hdfFile, 'r')
        data = None
        try:
            data = [x[1:] for x in hdfStore.keys()]
        finally:
            hdfStore.close()
        return data


class MarketDataStoreRemote():

    def __init__(self, endpoint):
        self.mdsRemote = PriceStore(endpoint)

    def aggregate(self, sources, sample_unit, start, end, debug=False):
        results = self.mdsRemote.aggregate(start, end, sources, sample_unit, debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return pandas.read_json(results["body"], orient="split").set_index(["Date_Time", "ID"])
        return pandas.DataFrame()

    def get(self, source_id, debug=False):
        results = self.mdsRemote.get(source_id, debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return pandas.read_json(results["body"], orient="split").set_index(["Date_Time", "ID"])
        return pandas.DataFrame()

    def append(self, source_id, data, source_sample_unit, update=False, debug=False):
        if update:
            return self.mdsRemote.put(source_id, data.reset_index().to_json(orient='split', date_format="iso"), source_sample_unit, debug)
        else:
            return self.mdsRemote.post(source_id, data.reset_index().to_json(orient='split', date_format="iso"), source_sample_unit, debug)

    def delete(self, source_id, debug=False):
        return self.mdsRemote.delete(source_id, debug)

    def getKeys(self, debug=False):
        results = self.mdsRemote.getKeys(debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return results["body"]
        return None
