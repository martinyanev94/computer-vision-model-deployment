from music21 import stream, note

# Assume we have preprocessed our MIDI files into sequences of pitches
def generate_music(model, start_sequence, num_notes):
    generated_sequence = start_sequence
    for _ in range(num_notes):
        input_sequence = torch.tensor(generated_sequence).unsqueeze(0)  # make the input a batch
        output = model(input_sequence)
        next_note = output.argmax(dim=1).item()  # get the predicted note
        generated_sequence.append(next_note)
        
        # If you want to add the note to a stream to create a MIDI file
        stream_out = stream.Stream()
        stream_out.append(note.Note(next_note))
    
    return generated_sequence
