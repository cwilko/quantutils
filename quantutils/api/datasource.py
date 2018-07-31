import pandas
import json
import os
import quantutils.dataset.pipeline as ppl

class MarketDataStore:

    def __init__(self, root):
        # TODO read root from quantutils properties file
        self.root = root

        # Load datasources
        self.datasources = json.load(open("".join([root,"/datasources.json"])))

    def getDatasources(self):
        return self.datasources

    def loadMarketData(self, market_info, sample_unit):
    
        marketData = dict()

        ## Loop over datasources...
        for datasource in self.getDatasources():

            DS_path = self.root + "/" + datasource["name"] + "/"

            # Get HDFStore
            hdfFile = DS_path + datasource["name"] + ".hdf"
            hdfStore = pandas.HDFStore(hdfFile)

            for market in market_info["markets"]:   

                mktSrcs = [mkt["sources"] for mkt in datasource["markets"] if mkt["name"]==market][0]  

                for source in mktSrcs:

                    print("Loading {} data from {} in {}.hdf".format(market, source["name"], datasource["name"]))
                    # Load Dataframe from store
                    tsData = hdfStore[source["name"]]                        

                    ## Crop selected data set to desired ranges

                    tsData = ppl.cropDate(tsData, market_info["start"], market_info["end"])

                    ## Resample all to dataset sample unit (to introduce nans in all missing periods)

                    tsData = ppl.resample(tsData, source["sample_unit"])
                    
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

    def refreshMarketData(self):
        
        markets = dict()

        ## Loop over datasources...

        for datasource in self.getDatasources():
            
            DS_path = self.root + "/" + datasource["name"] + "/"
            SRC_path = DS_path + "raw/"
                
            # Get HDFStore
            hdfFile = DS_path + datasource["name"] + ".hdf"
            hdfStore = pandas.HDFStore(hdfFile)
            
            for market in datasource["markets"]:

                for source in market["sources"]:
                
                    # Load Dataframe from store
                    if source["name"] in hdfStore:
                        tsData = hdfStore[source["name"]]
                    else:
                        tsData = pandas.DataFrame()
                                    
                    ## Loop over any source files...
                    for infile in os.listdir(SRC_path):          

                        if infile.lower().startswith(source["name"].lower()):

                            print("Adding " + infile + " to " + market["name"] + " table")

                            ## Load RAW data (assume CSV)
                            newData = pandas.read_csv(SRC_path + infile, 
                                                      index_col=datasource["index_col"], 
                                                      parse_dates=datasource["parse_dates"], 
                                                      header=None,
                                                      names=["Date", "Time", "Open", "High", "Low", "Close"],
                                                      usecols=range(0,6),
                                                      skiprows=datasource["skiprows"],
                                                      dayfirst=datasource["dayfirst"]
                                                     ) 

                            if newData is not None:

                                newData = ppl.localize(newData, datasource["timezone"], "UTC")
                                
                                tsData = ppl.merge(newData, tsData)                
                            
                    
                    #ppl.save_hdf(tsData, timeseries["name"], hdfStore)
                    # TODO : Back up to object storage
                 

            hdfStore.close()
