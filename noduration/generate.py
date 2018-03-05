from music21 import *
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle
import glob

with open('data/beethoven', 'rb') as path:
        notes = pickle.load(path)
chars = sorted(list(set(notes)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(notes)
n_vocab = len(chars)
print ("Total Characters: ")
print(n_chars)
print ("Total Vocab: ")
print(n_vocab)


# prepare the dataset of input to output pairs encoded as integers
seq_length = 40
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = notes[i:i + seq_length]
    seq_out = notes[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print( "Total Patterns: ") 
print(n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model


model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


filename = "w06-6.8584-beethoven.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print(pattern)

offset = 0.5

stream2 = stream.Stream()
for i in range(500):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    if(result[-6:len(result)]=='|chord'):
        result = result.replace("|chord","")
        pnote, dnote = result.split('|')
        pnote = pnote.split(',')
        mychord =''
        for p in pnote:
            p1 = pitch.Pitch()
            p1.midi=float(p)
            mychord +=p1.nameWithOctave.replace("-","")
            mychord +=' '
            if '/' in dnote:
                dnote=0.2
                
            firstchord = chord.Chord(mychord, quarterLength=float(dnote))##perfect
        stream2.append(firstchord)
    else:
        npitch, nduration = result.split('|')
        if '/' in nduration:
                nduration=0.3
        nduration = float(nduration)
        notez = note.Note(npitch)
        notez.duration.quarterLength= nduration
        notez.offset=offset
        stream2.append(notez)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    offset += 0.5


stream2.write('midi', fp='beethoven001.mid')