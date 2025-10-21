import numpy as np
import pandas as pd
import DataGen as dg
import WeightCal as wc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class WeightedFull():
    def __init__(self, x_sample, y_sample, x_test,label = [0, 1, 2], probsp = [0.3,0.3,0.4], probsq = [0.6,0.2,0.2], alpha=0.05):
        ##Step 1 prepare data and respective weights
        self.x_dataset =np.append(x_sample,x_test)
        self.y_sample =y_sample

        self.label = label
        self.probp = probsp
        self.probq = probsq

        self.alpha = alpha
    
    def train_classifier(self,x_dataset, y_dataset):
        """
    train the pre - trained model over sample+hypothesis point
    return predict_proba  
    """
        X = x_dataset.reshape(-1, 1) 
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        clf.fit(X,y_dataset)
        
        p = clf.predict_proba(X) 
        
        return p 
    
    def score(self, y_hypo):
        """
    Compute conformal scores: s(x, y) = -p_hat(y|x) based on the pre trained model
        """       
        y_dataset = np.append(self.y_sample,y_hypo) # with n+1 instances
        probs_calib = self.train_classifier(self.x_dataset,y_dataset)  

        scores_true=-np.array([probs_calib[i, y_dataset[i]] for i in range(len(y_dataset))])
        return  scores_true 


    def weighted_quantile(self, score_w):
    # Sort by score ascending
        sorted_df = score_w.sort_values(by="score", ascending = True).reset_index(drop=True)
    # Cumulative sum of normalized weights
        w = sorted_df["weight"].to_numpy()
        s = sorted_df["score"].to_numpy()
        w_cum = np.cumsum(w / np.sum(w))
    # Find first score where cum prob >= 1 - alpha
        target = 1 - self.alpha
        idx = np.searchsorted(w_cum, target, side="right")
        idx = min(idx, len(s) - 1)  # clamp to last index
        return s[idx]

    #######Find Weighted Quantile
    # def quantile(self, score_w):
    #     sorted_arr = score_w.sort_values(by="score",ascending=True)
    #     i = 0
    #     quant = sorted_arr["weight"][i]
    #     q = 1-self.alpha
        
    #     while quant<=q and i<len(sorted_arr["weight"]):
            
    #         i=i+1
    #         quant  = quant+sorted_arr["weight"][i]
 
    #     return sorted_arr.loc[i, "score"]

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
            qa = self.weighted_quantile(score_w)

            if s_hypo <= qa:
                prediction_set.append(y_hypo)

        return prediction_set


if __name__ == "__main__":
    
    coverage = 0
    n=200 #number of simulations
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
    
