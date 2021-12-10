import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import util

# Regression Tree Implementation
class GBNode:
    def __init__(self, X, tree_indices, hess, grad, depth_threshold, weight_constr, leaf_constr):
        self.X = X
        self.tree_indices = tree_indices
        self.hess = hess
        self.grad = grad
        self.depth_threshold = depth_threshold
        
        self.leaf_weight = -1 * sum(grad[tree_indices]) / sum(hess[tree_indices])

        self.structure_score = -math.inf
        self.split(leaf_constr, weight_constr)
    
    def split(self, leaf_constr, weight_constr):
        for c in np.array(np.arange(self.X.shape[1])):
            X = self.X.iloc[self.tree_indices,c]
            
            for r in range(len(self.tree_indices)):
                lhs = [ i for i in X <= X.iloc[r]]
                rhs = [ i for i in X > X.iloc[r]]

                if( sum(rhs) < leaf_constr or sum(lhs) < leaf_constr): continue

                if( sum(self.hess[np.where(lhs)[0]]) >= weight_constr and
                    sum(self.hess[np.where(rhs)[0]]) >= weight_constr):

                    curr_score = util.gain(self.hess[self.tree_indices], self.grad[self.tree_indices], lhs, rhs)
                    
                    if curr_score > self.structure_score: 
                        self.split_col = c
                        self.structure_score = curr_score
                        self.split_point = X.iloc[r]
                
        if( self.structure_score != -math.inf): self.createChildren(self.X.iloc[self.tree_indices, self.split_col], leaf_constr, weight_constr)
            
    def predict(self, X):
        if self.structure_score == -math.inf or self.depth_threshold <= 0:
            return(self.leaf_weight)

        node = self.lhs if X.iloc[self.split_col] <= self.split_point else self.rhs
        return node.predict(X)
    
    def createChildren(self, X, leaf_constr, weight_constr):
        if( self.depth_threshold > 0):

            lhs_i = self.tree_indices[np.where(X <= self.split_point)[0]]
            rhs_i = self.tree_indices[np.where(X > self.split_point)[0]]
            
            self.lhs = GBNode(self.X, lhs_i, self.hess, self.grad, self.depth_threshold-1, weight_constr, leaf_constr)
            self.rhs = GBNode(self.X, rhs_i, self.hess, self.grad, self.depth_threshold-1, weight_constr, leaf_constr)
 
class GBTree:
    def __init__(self, X, grad, hess, leaf_constr, weight_constr,depth_threshold):
        self.leaf_constr = leaf_constr
        self.weight_constr = weight_constr
        self.row_indices = np.arange(len(X))
        self.grad = grad
        self.hess = hess
        self.depth_threshold = depth_threshold

        self.root = GBNode(X, self.row_indices, hess, grad, depth_threshold, weight_constr, leaf_constr)
        
    def predict(self, X):
        predictions = np.asarray([self.root.predict( row[1] ) for row in X.iterrows()])
        
        return predictions
      
# XGBoost Implementation (Regression)
class XGBoost:
        
    def fit(self, X, y, depth_threshold = 6, weight_constr = 2, leaf_constr = 8, iterations = 10, lr = 0.2, init_pred = 'mean'):
        self.weak_learners = []
        self.depth_threshold = depth_threshold
        self.weight_constr = weight_constr 
        self.leaf_constr = leaf_constr
        self.lr = lr
        self.iterations = iterations 

        # research extension: initial prediction is done by a linear model
        if (init_pred == 'linreg'):
            self.initial_prediction, X_, y_ = util.predictInitialPreds(X, y)
            self.X = X_
            self.y = y_

        else:
            # initial prediction is the mean
            self.initial_prediction = np.empty(X.shape[0])
            self.initial_prediction.fill(np.mean(y))
            self.X = X
            self.y = y
            
        self.boost()
  
    def predict(self, X):
        pred = np.zeros(X.shape[0])
        pred += sum([learner.predict(X) * self.lr for learner in self.weak_learners])
        
        labels = np.empty(X.shape[0])
        labels.fill(np.mean(self.y))

        return labels + pred
        
    def boost(self):
        accuracies = []
        rmse_store = []

        for i in range(self.iterations):
            print('ITERATION ',i)
            pseudo_residual = self.initial_prediction - self.y

            grad_boost_tree = GBTree(self.X, (pseudo_residual)*2, util.getHessian(self.initial_prediction), self.depth_threshold, self.leaf_constr, self.weight_constr)
            self.weak_learners.append(grad_boost_tree)

            pred_of_learner = self.lr * grad_boost_tree.predict(self.X)
            self.initial_prediction += pred_of_learner

            accuracies, rmse_store = util.evaluate(self.y, self.initial_prediction, accuracies, rmse_store)
        
        # plotting RMSE and Accuracy for Training
        plt.plot(range(self.iterations),accuracies,'b-',label='Accuracy over '+ str(self.iterations) + ' Iterations')
        plt.plot(range(self.iterations),rmse_store,'r-',label='RMSE over '+ str(self.iterations) + ' Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy and RMSE')
        plt.title('Accuracy and RMSE Plot for Training')
        plt.legend()
        plt.show()


            