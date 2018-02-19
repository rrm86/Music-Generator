from music21 import *
import pickle
import glob


def parse_mid():
    notes = []
    duration = []

    for midifile in glob.glob('*.mid'):


        notes_to_parse = None

        parts = instrument.partitionByInstrument(midifile)

        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()

        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            print('teste')

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                a = str(element.pitch)
                duration.append(str(element.duration.quarterLength))
                a += ("|")
                a += str(element.duration.quarterLength)
                #notes.append(a)


            elif isinstance(element, chord.Chord):
                chordline = str(element.normalOrder)
                chordline = chordline[1:(len(chordline) - 1)]
                chordline += ("|")
                chordduration = str(element.duration.quarterLength)
                chordline += chordduration
                chordline += ("|chord")
                notes.append(chordline)

        print(notes)

if __name__=='__main__':
    parse_mid()