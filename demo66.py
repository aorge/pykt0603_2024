from keras import utils

origin = [4, 5, 6]
NUM_DIGITS = 10

for o in origin:
    converted = utils.to_categorical(o, NUM_DIGITS)
    print("{} becomes {}".format(o, converted))
