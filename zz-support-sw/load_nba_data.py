import json

datadir = '/data/Armand/NBA/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs/'
# Opening JSON file
f = open(datadir + '0021500499.json')

# returns JSON object as
# a dictionary
data = json.load(f)

# Closing file
f.close()