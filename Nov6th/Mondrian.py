import numpy as np
import pandas as pd
import DataGen as dg


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class Mondrian():
    def __init__(self, x_sample, y_sample, x_test,label = [0, 1, 2], alpha=0.05):
        ##Step 1 prepare data and respective weights
        self.x_dataset =np.append(x_sample,x_test)
        self.y_sample =y_sample

        self.label = label


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
        df_Score = pd.DataFrame(scores_true, columns=["score"])
        df_Score["y"] = y_dataset
        
        return  df_Score


    def con_quantile(self, groups):
        quant_set = dict()
        for label in groups:
            # get sub dataset
            scorei = groups[label]
            # Sort by score ascending
            sorted_df = scorei.sort_values(by="score", ascending = True).reset_index(drop=True)
            
            ni = len(scorei)
            weight = np.full(ni, 1/ni)
            sorted_df["weight"] = weight

            w = sorted_df["weight"].to_numpy()
            s = sorted_df["score"].to_numpy()
            w_cum = np.cumsum(w / np.sum(w))
        # Find first score where cum prob >= 1 - alpha
            target = 1 - self.alpha
            idx = np.searchsorted(w_cum, target, side="right")
            idx = min(idx, len(s) - 1)  # clamp to last index
            quant_set[label]=s[idx]


            
            # # Quantile 
            # quantile = np.quantile(sorted_df["score"], (1 - self.alpha) * (1 + 1/len(scorei)))
            # quant_set[label]=quantile
      
        return quant_set

    def alg(self):
        prediction_set = []
        for i in range(len(self.label)):
            y_hypo = self.label[i]
           
            df = self.score(y_hypo)
            
            
        # Split into dict of DataFrames keyed by label
            groups = {label: sub_df for label, sub_df in df.groupby("y")}
            s_hypo = df["score"].iloc[-1]

            qa_set = self.con_quantile(groups)

            if s_hypo <= qa_set[y_hypo]:
                prediction_set.append(y_hypo)

        return prediction_set


if __name__ == "__main__":
    
    coverage = 0
    n=300 #number of simulations
    # calculate the coverage rate for n times simulations
    for i in range(n):
        # generate data
        x_sample, y_sample, x_test, y_test = dg.Simple_generater()
        # run algorithm
        mcp = Mondrian(x_sample, y_sample, x_test)
        CI = mcp.alg()
        if y_test in CI:
            coverage = coverage +1

    coverage_rate = coverage/n
    print(coverage_rate)
    
