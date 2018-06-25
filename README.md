# Information-Weighted-Consensus
Implementation of Informaiton Weighted Consensus using MATLAB
According to [this paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.310.4837&rep=rep1&type=pdf) _(Information Weighted Consensus)_

## Abstract
Information weighted consesus (ICF) is a distributed version of Kalman Filter. By distributed, it means that there are multiple nodes in a system, each having a limited observability of the target it is trying to track. Information is passed through nodes for the state update and information update of each node to arrive at a consensus between nodes.

## Implementation
### Tracking of a moving object in a 500x500 space
![alt text](https://github.com/eugenelet/Information-Weighted-Consensus/blob/master/ICF_CKF.gif)

The *blue triangles* corresponds to the *field of view (FOV)* or different cameras. As we can see, there are 5 cameras in total trying to track a moving object. The *gray circle* corresponds to the *inverse of the information matrix* and it implies the uncertainty of an estimation at that particular instance.

**Green**: Ground truth trace of moving object

**Red**: ICF

**Black**: Central Kalman Filter (CKF)


### Error vs Different Simulation Runs
![Error vs Different Simulation Runs](https://github.com/eugenelet/Information-Weighted-Consensus/blob/master/Error%20vs%20Sim%20Runs.png)

### Error vs Consensus Iterations
![Error vs Consensus Iterations](https://github.com/eugenelet/Information-Weighted-Consensus/blob/master/Error%20vs%20Consensus%20Iterations.png)
