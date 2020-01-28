''' Fondamental Frequency:

This file defines a loop that gets the sigmoid of the score from the OSC server running locally.
It modifies the value with the "offset" and "weight" parameters; then plays the note with a low cutoff.

'''

update_per_second = 60
update_time = 1.to_f/update_per_second

# This is the synchronisation loop, or tick
live_loop :foo do
  sleep update_time
end

# Values to adapt to the user ?
# Was tested so that note was around 35-40
offset = 15
weight = 30

# This is the eeg loop
live_loop :eeg do
  sync "/live_loop/foo" # synchronize with the tick 
  sig_score = get "/osc/fund_freq/sig_score" # read the OSC message
  ssf = sig_score[0].to_f # get the value
  note = (1-ssf)*weight+offset # modification of the value so that the note plays
  play note, cutoff: 30, amp:1, sustain: 5 # play the note with a very low cutoff 
  # Long sustain (3-5) is important for the subpack.
end

