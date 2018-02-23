
## 
## Merge data
##

def merge(newData, existingData):
    print "Merging data..."
    return existingData.combine_first(newData)
##
## Shuffle data
##
def shuffle(data):
    return data.sample(frac=1).reset_index(drop=True)