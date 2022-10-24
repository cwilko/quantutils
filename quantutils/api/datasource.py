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
    def aggregate(self, table_id, sources, start="1979-01-01", end="2050-01-01", debug=False):

        hdfStore = pandas.HDFStore(self.hdfFile, 'r')

        try:
            marketData = None

            for source in sources:

                datasource = ''.join(['/', table_id])

                if datasource in hdfStore.keys():

                    print("Loading data from {} in {}".format(source, self.hdfFile))

                    # Load Dataframe from store
                    select_stmt = ''.join(["ID='", source, "' and Date_Time>'", start, "' and Date_Time<='", end, "'"])
                    tsData = hdfStore.select(datasource, where=select_stmt)

                    if not tsData.empty:

                        # Oct '22
                        # Moving to multi-id tables. All resampling to be performed by client.
                        # The following no longer applies...

                            # 28/6/21 Move this to before data is saved for performance reasons
                            # Resample all to dataset sample unit (to introduce nans in all missing periods)
                            # tsData = ppl.resample(tsData, source["sample_unit"])

                            # Resample to the requested unit
                            #tsData = ppl.resample(tsData, sample_unit)

                            # 06/06/18
                            # Remove NaNs and resample again, to remove partial NaN entries before merging
                            #tsData = ppl.removeNaNs(tsData)
                            #tsData = ppl.resample(tsData, sample_unit)

                        if marketData is None:
                            marketData = pandas.DataFrame(index=pandas.MultiIndex(levels=[[], []], codes=[[], []], names=[u'Date_Time', u'ID']))

                        marketData = ppl.merge(tsData, marketData)
                else:
                    raise ValueError('Error: Cannot find datasource: ' + datasource)
        finally:
            hdfStore.close()

        return marketData

    @synchronized
    def append(self, table_id, data, update=False, debug=False):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'a')
        append = True
        # Sort incoming data
        data = data.sort_index()

        if debug:
            print("Request to add data to table: " + table_id, flush=True)
        #print(data, flush=True)

        try:
            if '/' + table_id in hdfStore.keys():

                if debug:
                    print("Table found: " + table_id, flush=True)

                # Get first,last row
                nrows = hdfStore.get_storer(table_id).nrows
                last = hdfStore.select(table_id, start=nrows - 1, stop=nrows)

                # If this is entirely beyond the last element in the file... append
                # If not... update (incurring a full file re-write and performance hit), or throw exception
                if not data[data.index.get_level_values(0) <= last.index.get_level_values(0)[0]].empty:
                    # Update table with overlapped data
                    storedData = hdfStore.get(table_id)
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
                    hdfStore.append(table_id, data, min_itemsize={"values": 50}, format='table', append=True)
                else:
                    if debug:
                        print("Re-writing table data for update...", flush=True)
                    hdfStore.put(table_id, data, min_itemsize={"values": 50}, format='table')
            else:
                # Oct 22 - see above comment
                #data = ppl.resample(data, source_sample_unit)
                if debug:
                    print("Creating new table for data...", flush=True)
                hdfStore.put(table_id, data, min_itemsize={"values": 50}, format='table')

        finally:
            hdfStore.close()

        if debug:
            print("Update complete", flush=True)
        sys.stdout.flush()

    def get(self, table_id):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'r')
        data = None
        try:
            data = hdfStore.get(table_id)
        finally:
            hdfStore.close()
        return data

    # Vanilla put of any data
    @synchronized
    def put(self, table_id, data, update=False):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'a')
        print("Request to add data to table: " + table_id, flush=True)
        #print(data, flush=True)
        try:
            if '/' + table_id in hdfStore.keys():
                storedData = hdfStore.get(table_id)
                if update:
                    data = ppl.merge(storedData, data)
                else:
                    data = ppl.merge(data, storedData)

            hdfStore.put(table_id, data, min_itemsize={"values": 50}, format='table')
        finally:
            hdfStore.close()
        return data

    # Remove a node from a hdfFile
    @synchronized
    def delete(self, table_id):

        # Get HDFStore
        hdfStore = pandas.HDFStore(self.hdfFile, 'a')
        try:
            hdfStore.remove(table_id)
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

    def aggregate(self, table_id, sources, start="1979-01-01", end="2050-01-01", debug=False):
        results = self.mdsRemote.aggregate(table_id, sources, start, end, debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return pandas.read_json(results["body"], orient="split", dtype=False).set_index(["Date_Time", "ID"])
        return pandas.DataFrame()

    def get(self, table_id, debug=False):
        results = self.mdsRemote.get(table_id, debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return pandas.read_json(results["body"], orient="split", dtype=False).set_index(["Date_Time", "ID"])
        return pandas.DataFrame()

    def append(self, table_id, data, update=False, debug=False):
        if update:
            return self.mdsRemote.put(table_id, data.reset_index().to_json(orient='split', date_format="iso"), debug)
        else:
            return self.mdsRemote.post(table_id, data.reset_index().to_json(orient='split', date_format="iso"), debug)

    def delete(self, table_id, debug=False):
        return self.mdsRemote.delete(table_id, debug)

    def getKeys(self, debug=False):
        results = self.mdsRemote.getKeys(debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return results["body"]
        return None
