#!/usr/bin/env python3

"""
"""
#Third Party
from IPython import embed
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
import tensorflow as tf
from sklearn.utils.fixes import logsumexp
import numpy as np

#first party
from sensor_config import friendly, hostile


class TFNaiveBayesClassifier:
    """Creates and makes predictions with a niave bayesian classifier"""
    dists = None

    def defineClasses(self, mean, sD):
        """ Takes in matrix of shape (class x features) for mean and 
            standard deviation and ads the resulting nodes to the model"""

        mean = tf.convert_to_tensor(mean, dtype = 'float64')
        sD   = tf.convert_to_tensor(sD,   dtype = 'float64')

        self.dists = tf.distributions.Normal(loc=mean, scale=sD) 


    def predict(self, X):
        """ Retrun class predictions for datta X of size 
            (objects X Features)."""

        assert self.dists is not None

        nb_classes, nb_features = map(int, self.dists.scale.shape)

        # Create grid for all values in figure
        spaceVals = tf.reshape(
                tf.tile(X, [1, nb_classes]), [-1, nb_classes, nb_features])

        # Conditional probabilities log P(x|c) with shape
        # (nb_samples, nb_classes)
        cond_probs = tf.reduce_sum(
            self.dists.log_prob(spaceVals), axis=2)

        # Uniform priors
        priors = np.log(np.array([1. / nb_classes] * nb_classes))

        # Posterior log probability, log P(c) + log P(x|c)
        joint_likelihood = tf.add(priors, cond_probs)

        # Normalize to get (log)-probabilities
        norm_factor = tf.reduce_logsumexp(
            joint_likelihood, axis=1, keep_dims=True)
        log_prob = joint_likelihood - norm_factor

        return tf.exp(log_prob) # exp to get the actual probabilities


class StatMethods:
    """Contains useful statistical methods for prob fusion"""

    @staticmethod
    def inverseVarMean(obs1, obs2, sD1, sD2):
        """Takes two arrays of observations to combine into a new array
            of measurements
            
            obs1 --array of measurements on attribute 1
            obs2 --array of measurements on attrivute 2
            sD1  --float representing standard devation of measurements 1
            sD2  --float representing standard devation of measurements 2
            """
        #sD1, sD2 are single floats
        var1 = sD1**2.0
        var2 = sD2**2.0

        weight1 = (1/var1)/((1/var1)+(1/var2))
        weight2 = (1/var2)/((1/var1)+(1/var2))
        newObs =[]
        for ob1, ob2 in zip(obs1, obs2):
            newObs.append(ob1*weight1 +ob2*weight2)
        return newObs

    @staticmethod
    def inverseVarSD(sD1, sD2):
        """Create MAP measurement standard deviation on sD1 and sD2."""
        fusedSD = 1/(sD1**(-1) + sD2**(-1))
        return fusedSD


class GenSigs:
    """Creates artificial sensor returns"""

    @staticmethod
    def getSenReturns(numSamples, hostile, friendly, seed = 1234):
        """Generate artificial return based for IR and Radar sensors
        based on sensor charcteristics and object being measured.
        """

        # Create simulated radar returns and formate them for tensor flow
        hSizesRadar = GenSigs.drawFromNormDist(
                numSamples, mean = hostile ['size'],
                stddev = hostile ['sizeRadarSD'], seed = seed)
        hTempRadar = GenSigs.drawFromNormDist(
                numSamples, mean = hostile ['temp'], 
                stddev = hostile ['tempRadarSD'], seed = seed + 1)
        radarReturnsHostile = np.array(
                [[size, temp] for size, temp in zip(hSizesRadar, hTempRadar)], 
                dtype = 'float64')
        fSizesRadar = GenSigs.drawFromNormDist(
                numSamples, mean = friendly['size'],
                stddev = friendly['sizeRadarSD'], seed = seed + 2)
        fTempRadar = GenSigs.drawFromNormDist(
                numSamples, mean = friendly['temp'], 
                stddev = friendly['tempRadarSD'], seed = seed + 3)
        radarReturnsFriendly = np.array(
                [[size, temp] for size, temp in zip(fSizesRadar, fTempRadar)], 
                dtype = 'float64')
        xRadar = np.append(radarReturnsHostile, radarReturnsFriendly, axis = 0)
        
        # Create simulated ir returns and formate them for tensor flow
        hSizesIR = GenSigs.drawFromNormDist(
                numSamples, mean = hostile ['size'], 
                stddev = hostile ['sizeIRSD'], seed = seed + 4)
        hTempIR  = GenSigs.drawFromNormDist(
                numSamples, mean = hostile ['temp'],
                stddev = hostile ['tempIRSD'],seed = seed + 5)
        irReturnsHostile = np.array(
                [[size, temp] for size, temp in zip(hSizesIR, hTempIR)], 
                dtype = 'float64')
        fSizesIR = GenSigs.drawFromNormDist(
                numSamples, mean = friendly['size'], 
                stddev = friendly['sizeIRSD'], seed = seed + 6)
        fTempIR  = GenSigs.drawFromNormDist(
                numSamples, mean = friendly['temp'], 
                stddev = friendly['tempIRSD'],seed = seed + 7)
        irReturnsFriendly = np.array(
                [[size, temp] for size, temp in zip(fSizesIR, fTempIR)], 
                dtype = 'float64')
        xIR = np.append(irReturnsHostile, irReturnsFriendly, axis = 0)
        
        return xRadar, xIR

    @staticmethod
    def drawFromNormDist(numSamples, mean, stddev, seed):
        """ Draw numSamples samples from a normal distrobution with params mean 
            and stddev.
        """
        sess = tf.Session()
        draws = sess.run(tf.random_normal([1, numSamples], mean=mean,
                         stddev=stddev, seed = seed))[0]
        return draws


if __name__ == '__main__':

    #params
    fuseTemp = True
    fuseSize = None #True
    
    #options sizeSensor = radar, ir, fuse
    sizeSensor = 'fuse'
    sizeSensor = sizeSensor.lower()
    tempSensor = 'radar'
    tempSensor = tempSensor.lower()
    
    numSamples = 10
    showPopMeans = True

    #Gets radar and ir returns
    xRadar, xIR = GenSigs.getSenReturns(numSamples, hostile, friendly)

    #Fuse attributes using inverseVar result is MAP
    if sizeSensor == 'fuse':
            #Fuse attributes using inverseVar result is MAP
        sizeMeasurements = StatMethods.inverseVarMean(xRadar[:,0], xIR[:,0], 
                friendly['sizeRadarSD'], friendly['sizeIRSD'])
    elif sizeSensor == 'radar':
        sizeMeasurements = xRadar[:,0]
    elif sizeSensor == 'ir':
        sizeMeasurements = xIR[:,0]
    else:
        raise "Size sensor must be: radar, ir, or fuse."

    if tempSensor == 'fuse':
            #Fuse attributes using inverseVar result is MAP
        tempMeasurements = StatMethods.inverseVarMean(xRadar[:,1], xIR[:,1], 
                friendly['tempRadarSD'], friendly['tempIRSD'])
    elif tempSensor == 'radar':
        tempMeasurements = xRadar[:,1]
    elif tempSensor == 'ir':
        tempMeasurements = xIR[:,1]
    else:
        raise "Temperature sensor must be: radar, ir, or fuse."

    X = np.array([[att1, att2] for att1, att2 in 
                zip(sizeMeasurements, tempMeasurements)])

    fuseSizeSD = StatMethods.inverseVarSD(hostile['sizeRadarSD'], 
                                          hostile['sizeIRSD'])
    fuseTempSD = StatMethods.inverseVarSD(hostile['tempRadarSD'], 
                                          hostile['tempIRSD'])

    mean = np.array(
            [[hostile['size'], hostile['temp']],
            [friendly['size'], friendly['temp']]])
    
    sD =  np.array([[fuseSizeSD, fuseTempSD],[fuseSizeSD, fuseTempSD]])

    tf_nb = TFNaiveBayesClassifier()
    tf_nb.defineClasses(mean, sD)

    # Create a regular grid and classify each point
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    s = tf.Session()
    Z = s.run(tf_nb.predict(np.c_[xx.ravel(), yy.ravel()]))

    # Extract probabilities of class 1 and 2
    Z1 = Z[:, 1].reshape(xx.shape)

    # Plot
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)

    points = ax.scatter(x = X[:10,0], y=X[:10, 1], c='red', edgecolor='k', 
                        label = 'Hostile')
    
    points = ax.scatter(x = X[10:, 0], y=X[10:, 1], c='blue', edgecolor='k', 
                        label = 'Friendly')

    if showPopMeans == True:
        points = ax.scatter(
                x = [friendly['size'], hostile['size']], y=[friendly['temp'], 
                hostile['temp']], c='black', s=200, edgecolor='k', 
                label = 'Population Means')

    # Swap signs to make the contour dashed (MPL default)
    ax.contour(xx, yy, -Z1, [-0.5], colors='k')

    ax.set_xlabel('Size (m)')
    ax.set_ylabel('Temp (C)')
    ax.set_title('Radar and IR Attribute Decision Boundary')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend()

    plt.tight_layout()
    plt.show()
    fig.savefig('____', bbox_inches='tight')
