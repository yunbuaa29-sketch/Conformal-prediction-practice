import numpy as np
#### future improvement, pass pmf function
class WeightCal():
    def __init__(self, y_sample, y_hypo, label = [0, 1, 2], probsp = [0.1,0.3,0.6], probsq = [0.5,0.2,0.3]):
        # self.y_sample = y_sample
        # self.y_hypo = y_hypo
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
        p = 0
        for i in range(len(self.label)):
            if y == self.label[i]:
                p = probs[i]
                break
        return p

    def Weight_i(self):
        #### return a numpy array
        w = np.zeros(len(self.y_dataset))

        for i in range(len(self.y_dataset)):
            dp = self.pmf(self.y_dataset[i], self.probsp)
            dq = self.pmf(self.y_dataset[i], self.probsq)
            w[i] = (dq/dp)/(self.wsum+dq/dp)

        return w

    # def Weight_test(self, y_hypo):
    #     dp = self.pmf(y_hypo, self.probsp)
    #     dq = self.pmf(y_hypo, self.probsq)
    #     wt = (dq/dp)/(self.wsum+dq/dp)
    #     return wt