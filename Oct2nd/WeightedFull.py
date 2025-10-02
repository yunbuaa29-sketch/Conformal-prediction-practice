import numpy as np
import pandas as pd
import DataGen as dg
import WeightCal as wc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class WeightedFull():
    def __init__(self, x_sample, y_sample, x_test,label = [0, 1, 2], probsp = [0.1,0.3,0.6], probsq = [0.5,0.2,0.3], alpha=0.05):
        ##Step 1 prepare data and respective weights
        self.x_dataset =np.append(x_sample,x_test)
        self.y_sample =y_sample

        self.label = label
        self.probp = probsp
        self.probq = probsq

        self.alpha = alpha

        # self.wt = wc_gen.Weight_test()
    
    def train_classifier(self,x_dataset, y_dataset):
        """
    Compute conformal scores: s(x, y) = -p_hat(y|x)

    Parameters
    ----------
    x_dataset : array-like of shape (n_samples, n_features)
        Feature dataset (includes n calibration + 1 hypothesis point).
    y_dataset : array-like of shape (n_samples,)
        Corresponding labels including hypothesis point

    Returns
    -------
    estimated probability for each x with each label of y
    """
        X = x_dataset.reshape(-1, 1) 
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        clf.fit(X,y_dataset)
        return clf.predict_proba(X)
    
    def score(self, y_hypo):
        """
    Compute conformal scores: s(x, y) = -p_hat(y|x)

    Parameters
    ----------
    y_hypo: label of htpothesis point
    Returns
    -------
    scores : np.ndarray of shape (n_samples,)
        Conformal nonconformity scores.
        """
        
        y_dataset = np.append(self.y_sample,y_hypo) # with n+1 instances
        probs_calib = self.train_classifier(self.x_dataset,y_dataset) 

        #p = clf.predict_proba(x_dataset)  # shape (n_calib, K)
        scores_calib = -probs_calib[np.arange(len(y_dataset)), y_dataset]#label starts from 0
        
        #size = len(scores_calib)-1
        return scores_calib
    
    #######Find Weighted Quantile
    def quantile(self, score_w):
        sorted_arr = score_w.sort_values(by="score",ascending=True)
        quant = 0
        i = 0
        while quant<=1-self.alpha:
            quant  = quant+sorted_arr["weight"][i]
            i=i+1
       
        return sorted_arr["score"][i]

    ####### perform the 7.1 algorithm
    def alg(self):
        prediction_set = []
        for i in range(len(self.label)):
            y_hypo = self.label[i]
            wc_gen = wc.WeightCal(self.y_sample, y_hypo, self.label,self.probp, self.probq) 
            weight = wc_gen.Weight_i() #n+1 weights
            scoreset = self.score(y_hypo)
            s_hypo = scoreset[-1]

            score_w = pd.DataFrame({"score": scoreset,"weight": weight})        
            if s_hypo <= self.quantile(score_w):
                prediction_set.append(y_hypo)

        return prediction_set



if __name__ == "__main__":
    
    coverage = 0
    n=100 #number of simulations
    # calculate the coverage rate for n times simulations
    for i in range(n):
        # generate data
        x_sample, y_sample, x_test, y_test = dg.Simple_generater()
        # run algorithm
        wf = WeightedFull(x_sample, y_sample, x_test)
        CI = wf.alg()
        if y_test in CI:
            coverage = coverage +1

    coverage_rate = coverage/n
    print(coverage_rate)
    
