# Experiment B: Learning Distance from a Classifier

    
- Until now, DNNs have been trained in a manner so that they learn the probability distribution P(y | X; \theta)

- We have seen in Stutz et.al. the relationship between adversarial examples and their distance from the learned data manifold. Specifically, that adversarial examples do not lie on the data manifold and lie off of it, evidently in a direction perpendicular to the data manifold.

- This observation raises an important question: "Is it feasible to train neural networks to learn distance of a data point from the data manifold?

- This will not only help us answer if learning distance of data instances from a learned manifold is feasible or not, it will also help us analyze how we can apply this idea to improving robustness and OOD detection.

- Note that this makes the learning setup significantly different from a typical pipeline where we are learning just the probability distribution of the data.


## Experiment: Learning distance of samples from the manifold

1. Data Generation: For this experiment, we first generate synthetic data, which intuitively captures the idea we want to investigate. This synthetic data will take the form of points sampled from an k-dimensional sphere, embedded in n-dimensional space (n >> k). Here is a proposed way to generate this data:
	
    a. Finding a center: Randomly sample a k-dimensional vector. This will be our center, x_c
	
    b. Sampling points: Randomly sample N k-dimensional vectors.
	
    c. Normalize: Say we want to sample points from a sphere of radius R. We normalize the vectors sampled in (b) as follows: 		
    
    x := x_c + r \cdot \frac{ x - x_c }{ || x - x_c || } 

	d. Embedding in n-dimensional space: Now to embed these points in the n-dimensional space, we take the following steps:

        i. First, consider the trivial embedding of k-dimensional vectors in n-dimensional space by concatenating (n - k) 0's to each vector.
			
        ii. Now, randomly apply a translation and/or rotation transformation on the n-dimensional spaces.
        
        iii. Finally, normalize the vectors using x := x_c + r \cdot \frac{ x - x_c }{ || x - x_c || } (although it is probably not needed)
	
    e. "Negative" examples: Take a subset of these points, and scale them by some constant factor to generate points that are not on the sphere. Additionally, clamp the distance of each sample from the center of the sphere at some maximum distance D, i.e., if d(x, x_c) > D + r, set d(x, x_c) = D + r, where r is the radius of the sphere.

2. Learning:
    
    a. Train a network that takes the generated data as input and tries to match the distance of the sample from the manifold.
    
    b. Use L2 loss as the loss function.


Remarks:

1. The Learning part can be tried with different architectures and loss functions.

2. Once we have seen some results with the synthetic spherical data, we can then try out some variations in terms of the underlying data manifold, such as say a Torus, or a Cube.

3. In a multi-class setup, the way that such a training procedure would function is that we train a network for each class separately, and finally to decide class of a sample, we take the class whose manifold is closest to the sample.
