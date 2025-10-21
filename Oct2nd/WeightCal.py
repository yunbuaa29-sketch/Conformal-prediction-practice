import numpy as np
import DataGen as dg
#### future improvement, pass dynamic pmf function to the class
class WeightCal():
    def __init__(self, y_sample, y_hypo, label = [0, 1, 2], probsp = [0.3,0.3,0.4], probsq = [0.3,0.3,0.4]):
        self.y_dataset = np.append(y_sample,y_hypo)
        self.label = label
        self.probsp = probsp
        self.probsq = probsq
        self.wsum = 0

        for i in range(len(self.y_dataset)): 
            dp = self.pmf(self.y_dataset[i], self.probsp)
            dq = self.pmf(self.y_dataset[i], self.probsq)
            self.wsum = self.wsum+dq/dp
    

    def pmf(self, y, probs): 
        ###this function servs as pmf   
        ### y is a np array
        p = 0.0
        for i in range(len(self.label)):
            if y == self.label[i]:
                p = float(probs[i])
        
        return p  


    def Weight_i(self):
        #### return a numpy array
        #print(len(self.y_dataset))
        w = np.zeros(len(self.y_dataset))
   
        for i in range(len(self.y_dataset)):
            # dp = np.array([self.pmf(y, self.probsp)/self.pmf(y, self.probsp) for y in self.y_dataset])
            # dq =  np.array([self.pmf(y, self.probsq)/self.pmf(y, self.probsp) for y in self.y_dataset])
        ##########################################################################   
            dp = self.pmf(self.y_dataset[i], self.probsp)
            dq = self.pmf(self.y_dataset[i], self.probsq)
            w[i] = (dq/dp)/(self.wsum)

        return w
