from Game import Game
import argparse

datadir = '/data/Armand/NBA/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs/'

parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')
parser.add_argument('--path', default=datadir + '0021500499.json', type=str,
                    help='a path to json file to read the events from')
parser.add_argument('--event', type=int, default=0,
                    help="""an index of the event to create the animation to
                            (the indexing start with zero, if you index goes beyond out
                            the total number of events (plays), it will show you the last
                            one of the game)""")

args = parser.parse_args()

game = Game(path_to_json=args.path, event_index=args.event)
game.read_json()

game.start()
