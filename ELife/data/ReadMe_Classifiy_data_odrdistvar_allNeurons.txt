For all the matrice data : each row is one trial, each column is one neuron. 
For the same row the trial data may not be collected on the same trial since data might are collected on different days.

cuerate: firing rate during 1st visual stimuli(cue) presentation.

cuerate_stim: firing rate during 1st cue presentation with NB stimulation. 

cdrate: firing rate during the 1st delay after 1st cue presentation. 

cdrate_stim: firing rate during the 1st delay after cue presentation  with NB stimulation.  
 
samplerate: firing rate during 2nd visual stimuli(sample) presentation. 

samplerate_stim: firing rate during sample presentation with NB stimulation. 

sdrate: firing rate during the delay after sample presentation. 

sdrate_stim: firing rate during the delay after sample presentation with NB stimulation. 

g1: location of 1st visual stimuli(cue)
 value -1 means no visual stimuli presented.
 value [1 2 3 4 5] are  relative locaions, such as [0 45 90 135 180], or [ 45 90 135 180 225], etc.
 The real locations depended on neurons' receptive field.
 According to our behavioral paradigm, the cue will only be presented on locaion [1 5]. that is two opposite locations, such as [0 180], [45 225], etc.
 For each neuron, thoes value([1:5]) refer to the same locations.

g2: location of 2nd visual stimuli(sample)
  Sample can show up in all five possible locations([1:5]) or not presented(-1).

g3: task type for each trial, eg. saccade to 1st (value 1), or saccade to 2nd(value 2).
