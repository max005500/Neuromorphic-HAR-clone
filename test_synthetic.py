import sys, tempfile, csv
from pathlib import Path
import numpy as np
import pandas as pd
from preprocess_kfall import load_record,load_fall_labels
from kfall_event_loader import signal_to_events,read_task_map

def test():
 with tempfile.TemporaryDirectory() as td:
  r=Path(td);(r/'sensor'/'SA06').mkdir(parents=True);(r/'labels').mkdir()
  f=np.arange(1,302)
  df=pd.DataFrame({'TimeStamp(s)':(f-1)/100,'FrameCounter':f})
  for col in ['AccX','AccY','AccZ','GyrX','GyrY','GyrZ']:df[col]=np.linspace(0,1,len(f))
  for col in ['EulerX','EulerY','EulerZ']:df[col]=0
  df.to_csv(r/'sensor'/'SA06'/'SA06T20R01.csv',index=False)
  pd.DataFrame([['F01 (20)','simulated',1,130,208]],columns=['Task Code (Task ID)','Description','Trial ID','Fall_onset_frame','Fall_impact_frame']).to_excel(r/'labels'/'SA06_label.xlsx',index=False)
  x,frames,meta=load_record(r/'sensor'/'SA06'/'SA06T20R01.csv')
  lab=load_fall_labels(r/'labels')
  assert x.shape==(301,6) and lab[('SA06',20,1)]==(130,208)
  e,o=signal_to_events(x[:205],(frames[:205]-frames[0])/100,2048,np.full(6,.01,dtype=np.float32),.5)
  assert e.shape==(2048,6,2) and o.shape==(6,)
  print('PASS: parsing, units, labels and event tensor',e.shape)
if __name__=='__main__':test()
