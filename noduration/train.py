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



notes = []
duration = []

for midfile in glob.glob('midi_source/beethoven/elise.mid'):

    mid = converter.parse(midfile)
    
    notes_to_parse = None

    parts = instrument.partitionByInstrument(mid)

    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()

    else:  # file has notes in a flat structure
        notes_to_parse = mid.flat.notes
        print('teste')

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            a = str(element.pitch)
            duration.append(str(element.duration.quarterLength))
            a += ("|")
            a += str(element.duration.quarterLength)
            notes.append(a)


        elif isinstance(element, chord.Chord):
            chord_pitches=[]
            

            for p in element.pitches:
                chord_pitches.append(p.midi)

            
            chord_pitches = str(chord_pitches).strip('[]')
            
            
           
            chord_pitches += ("|")
            
            chord_duration = str(element.duration.quarterLength)
            
            
            chord_pitches += chord_duration
            chord_pitches += ("|chord")

        
            notes.append(chord_pitches)
            with open('data/elise', 'wb') as path:
                pickle.dump(notes, path)
    

    print(notes)



with open('data/elise', 'rb') as path:
        notes = pickle.load(path)
        
chars = sorted(list(set(notes)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(notes)
n_vocab = len(chars)
print ("Total Characters: ")
print(n_chars)
print ("Total Vocab: ")
print(n_vocab)
print(notes)

seq_length = 100
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
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# define the checkpoint
# define the checkpoint
filepath="w{epoch:02d}-{loss:.4f}-sonata14.1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=200, batch_size=60, callbacks=callbacks_list)
