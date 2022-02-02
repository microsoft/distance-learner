import os
from abc import ABC, abstractmethod

import numpy as np

class GeneralManifoldAttrs(object):
    """
    this class encapsulates the general attributes that any instance of `Manifold`
    or its children classes must have for sampling (k-1)-dimensional manifold
    embedded in an n-dimensional space

    in other words, these attributes should be in any class which inherits the from `Manifold`

    used as a property in the `Manifold` class

    non-data variables that only specify settings only have getters but no setters, since it
    does not make sense for them to change while the data remains the same.

    the data variables on the other hand can be changed. This difference is because during 
    debugging, or diagnostics, such behavior might be helpful, as data can be changed 
    quickly and directly passed to the model for re-training.

    """

    def __init__(self, N=1000, num_neg=None, n=100, k=3, D=50.0,\
                 max_norm=2.0, mu=10, sigma=5, seed=42, normalize=True,\
                 norm_factor=1, gamma=0.5, anchor=None, rotation=None, translation=None,\
                 online=False, off_online=False, augment=1, **kwargs):
        """
        :param N: total number of samples
        :type N: int
        :param num_neg: (static/dynamic) number of off-manifold points in the (static/dynamic) dataset (each epoch in dynamic)
        :type num_neg: int
        :param n: dimension of space in which manifold is embedded
        :type n: int
        :param k: low (k-1)-dimensional manifold, embedded in k dims 
        :type k: int
        :param D: clamping limit for off-manifold examples
        :type D: float
        :param max_norm: maximum possible distance of point from manifold
        :type max_norm: float
        :param mu: mean of normal distribution from which we sample
        :type mu: float
        :param sigma: standard deviation of normal distribution from which we sample
        :type sigma: float
        :param seed: random seed (default is the answer to the ultimate question!)
        :type seed: int
        :param normalize: whether to normalize the dataset or not
        :type normalize: bool
        :param norm_factor: factor by which to normalise (use for inverting)
        :type norm_factor: float
        :param gamma: conservative factor used in normalization
        :type gamma: int
        :param anchor: anchor point used during normalization
        :type anchor: torch.Tensor
        :param rotation: rotation matrix to be used
        :type numpy.ndarray:
        :param translation: translation vector to be used
        :type numpy.array:
        :param online: create points online on the fly
        :type online: bool
        :param off_online: bool
        :type off_online: only off-manifold points are created on the fly
        :param augment: float
        :type augment: off-manifold points created from on-manifold on the fly (like augmentations) 
        """

        self._N = N
        self._num_neg = np.floor(self._N / 2).astype(np.int64).item()
        if num_neg is not None:
            self._num_neg = num_neg
        self._n = n
        self._k = k
        self._D = D
        self._max_norm = max_norm
        self._mu = mu
        self._sigma = sigma
        self._seed = seed
        self._gamma = gamma
        self._normalize = normalize
        self._norm_factor = norm_factor
        self._rotation = None
        self._translation = None
        self._online = online
        self._off_online = off_online
        self._augment = augment

        self._anchor = anchor

        if translation is None:
            self._translation = np.random.normal(mu, sigma, n)
        else:
            self._translation = translation

        if rotation is None:
            self._rotation = np.random.normal(self._mu, self._sigma, (self._n, self._n))
            self._rotation = np.linalg.qr(self._rotation)[0]
        else:
            self._rotation = rotation

        self.points_k = None
        """points sampled from the sphere in k-dim"""

        self.pre_images_k = None
        """k-dimensional on-manifold pre-images of the off-manifold points"""
        
        self.points_n_trivial_ = None
        """sampled points in higher dimension after trivial embedding"""
        
        self.points_n_tr_ = None
        """sampled points in higher dimension after translation"""
        
        self.points_n_rot_ = None
        """sampled points in higher dimension after translation and rotation"""
        self.points_n = None
        """embedding of `self.points_k` in n-dim"""
        
        self.actual_distances = None
        """actual distance of points from the sphere's surface"""
        self.distances = None
        """clamped distance of the point from the sphere's surface"""

        self.fix_center = None
        """point used for translation close to the origin during normalization"""

        self.normed_points_n = self.points_n
        """points normalized so that they lie in a unit cube"""

        self.normed_distances = self.distances
        """normalised distances"""
        self.normed_actual_distances = self.actual_distances
        """actual normalised distances"""

    @property
    def N(self):
        return self._N

    @property
    def num_neg(self):
        return self._num_neg

    @property
    def n(self):
        return self._n 
    
    @property
    def k(self):
        return self._k

    @property
    def D(self):
        return self._D

    @property
    def max_norm(self):
        return self._max_norm

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @property
    def seed(self):
        return self._seed

    @property
    def gamma(self):
        return self._gamma

    @property
    def normalize(self):
        return self._normalize

    @property
    def norm_factor(self):
        return self._norm_factor

    @property
    def anchor(self):
        return self._anchor

    @property
    def rotation(self):
        return self._rotation

    @property
    def translation(self):
        return self._translation

    @property
    def online(self):
        return self._online

    @online.setter
    def online(self, x):
        raise RuntimeError("cannot set `online` after instantiation!")

    @property
    def off_online(self):
        return self._off_online

    @off_online.setter
    def off_online(self, x):
        raise RuntimeError("cannot set `off_online` after instantiation!")

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, x):
        raise RuntimeError("cannot set `augment` after instantiation!")


class SpecificManifoldAttrs(ABC):
    """
    placeholder class encapsulating the attributes for sampling a 
    (k-1)-dimensional manifold embedded in an n-dimensional space specific to the
    type of manifold

    classes which implement specific manifolds (say, a sphere), might
    require special attributes (say, center, radius); all such attributes should
    be encapsulated in a special class which inherits this class

    used as a property in the `Manifold` class

    NOTE: non-data variables that only specify settings *should* only have getters
    but no setters, since it does not make sense for them to change while the data remains the same.

    the data variables on the other hand can be changed and should have setters. 
    This difference is because during debugging, or diagnostics, such behavior might
    be helpful, as data can be changed quickly and directly passed to the model for re-training.

    """
    
    def __init__(self, mu=10, sigma=5, n=100, seed=42, **kwargs):
        """
        :param mu: mean of normal distribution from which we sample
        :type mu: float
        :param sigma: standard deviation of normal distribution from which we sample
        :type sigma: float
        :param n: dimension of higher dimensional space where manifold is embedded
        :type n: int
        :param seed: random seed (default is the answer to the ultimate question!)
        :type seed: int
        
        `mu`, `sigma`, `seed`, `n` should all come from corresponding `GeneralManifoldAttrs`
        object

        re-implement this but do add `**kwargs` as argument in the
        end in the function signature
        """
        
        self._mu = mu
        self._sigma = sigma
        self._n = n
        self._seed = seed

    @property
    @abstractmethod
    def mu(self):
        return self._mu

    @mu.setter
    @abstractmethod
    def mu(self, m):
        raise RuntimeError("cannot set `mu` after instantiation!")

    @property
    @abstractmethod
    def sigma(self):
        return self._sigma

    @sigma.setter
    @abstractmethod
    def sigma(self, m):
        raise RuntimeError("cannot set `sigma` after instantiation!")

    @property
    @abstractmethod
    def seed(self):
        return self._seed

    @seed.setter
    @abstractmethod
    def seed(self, m):
        raise RuntimeError("cannot set `seed` after instantiation!")

    @property
    @abstractmethod
    def n(self):
        return self._n

    @n.setter
    @abstractmethod
    def n(self, n):
        raise RuntimeError("cannot set `n` after instantiation!")



class Manifold(ABC):
    """
    Abstract class for a manifold

    Contains methods that any class representing a manifold should 
    contain
    """


    def __init__(self, genattrs=None, specattrs=None, **kwargs):
        """should be re-implemented!"""

        self._genattrs = genattrs
        self._specattrs = specattrs

        if genattrs is not None:
            self._genattrs = genattrs
        else:
            self._genattrs = GeneralManifoldAttrs(**kwargs)

        if specattrs is not None:
            self._specattrs = specattrs
        else:
            self._specattrs = SpecificManifoldAttrs(**kwargs)

        self.compute_points()

    @property
    @abstractmethod
    def genattrs(self):
        """
        instance of `GeneralManifoldAttrs`
        """
        return self._genattrs

    @genattrs.setter
    @abstractmethod
    def genattrs(self, *args, **kwargs):
        """
        why immutable?: because changing specs without re-computing 
        data is wrong! and recomputing with same object is pointless
        and confusing.
        """
        raise RuntimeError("Cannot set `genattrs` after instantiation!")
        
    @property
    @abstractmethod
    def specattrs(self):
        """
        instance of `SpecificManifoldAttrs`
        """
        return self._specattrs

    @specattrs.setter
    @abstractmethod
    def specattrs(self, *args, **kwargs):
        """
        why immutable?: because changing specs without re-computing 
        data is wrong! and recomputing with same object is pointless
        and confusing.
        """
        raise RuntimeError("Cannot set `specattrs` after instantiation!")

    @abstractmethod
    def gen_points(self):
        """
        generate random points on the (k-1)-dim manifold in the
        canonical k-dimensional embedding
        """
        raise NotImplementedError

    @abstractmethod
    def compute_normals(self):
        """
        compute the normal at each point in the trivial embedding
        of the (k-1)-dimensional manifold in n-dimensions
        """
        raise NotImplementedError

    @abstractmethod
    def online_compute_normals(self):
        """
        compute the normal at each point in the trivial embedding
        of the (k-1)-dimensional manifold in n-dimensions
        """
        raise NotImplementedError

    @abstractmethod
    def make_off_mfld_eg(self):
        """
        using normals for computing the normal space and
        the off-manifold examples
        """

        embedded_normal_vectors_to_mfld_at_p = self.compute_normals()

        # canonical basis $e_i$ over leftover dimensions
        remaining_dims = self._genattrs.n - self._genattrs.k
        leftover_basis = np.eye(remaining_dims)

        # variable storing the remaining spanning set apart from the normal
        remaining_span_set = np.zeros((remaining_dims, self._genattrs.n))
        remaining_span_set[:, self._genattrs.k:] = leftover_basis

        # coefficients for the remaining basis vectors
        remaining_coefficients = np.random.normal(self._genattrs.mu,\
             self._genattrs.sigma, size=(self._genattrs.num_neg, self._genattrs.n))
        # sum of the remaning span set
        sum_span_set = np.sum(remaining_span_set, axis=0)
        # taking advantage of the standard basis, we can form convert the sum to a linear combination
        remaining_linear_combination = remaining_coefficients * sum_span_set

        # coefficients to multiply the normals
        first_coefficients = np.random.normal(self._genattrs.mu, self._genattrs.sigma,\
             size=(self._genattrs.num_neg, 1))
        weighted_normals = first_coefficients * embedded_normal_vectors_to_mfld_at_p
    
        neg_examples = weighted_normals + remaining_linear_combination

        # re-scale with random norms, sampled from U[\epsilon, self.max_norm]
        neg_norms = np.random.uniform(low=1e-6 + np.finfo(np.float).eps,\
             high=self._genattrs.max_norm, size=self._genattrs.num_neg)            
        neg_examples = (neg_norms.reshape(-1, 1) / np.linalg.norm(neg_examples, axis=1, ord=2).reshape(-1, 1)) * neg_examples

        # add position vector of $p$ to get origin centered coordinates
        neg_examples[:, :self._genattrs.k] = neg_examples[:, :self._genattrs.k] + np.array(self._genattrs.pre_images_k)

        # distances from the manifold will be the norms the samples were rescaled by
        neg_distances = neg_norms
        
        return neg_examples, neg_distances

    @abstractmethod
    def online_make_off_mfld_eg(self, online_pt):
        """
        using normals for computing the normal space and
        the off-manifold examples
        """

        embedded_normal_vectors_to_mfld_at_p = self.online_compute_normals(online_pt)

        # canonical basis $e_i$ over leftover dimensions
        remaining_dims = self._genattrs.n - self._genattrs.k
        leftover_basis = np.eye(remaining_dims)

        # variable storing the remaining spanning set apart from the normal
        remaining_span_set = np.zeros((remaining_dims, self._genattrs.n))
        remaining_span_set[:, self._genattrs.k:] = leftover_basis

        # coefficients for the remaining basis vectors
        remaining_coefficients = np.random.normal(self._genattrs.mu,\
             self._genattrs.sigma, size=self._genattrs.n)
        # sum of the remaning span set
        sum_span_set = np.sum(remaining_span_set, axis=0)
        # taking advantage of the standard basis, we can form convert the sum to a linear combination
        remaining_linear_combination = remaining_coefficients * sum_span_set

        # coefficients to multiply the normals
        first_coefficients = np.random.normal(self._genattrs.mu, self._genattrs.sigma,\
             size=1)
        weighted_normals = first_coefficients * embedded_normal_vectors_to_mfld_at_p
    
        online_neg_examples = weighted_normals + remaining_linear_combination

        # re-scale with random norms, sampled from U[\epsilon, self.max_norm]
        online_neg_norms = np.random.uniform(low=1e-6 + np.finfo(np.float).eps,\
             high=self._genattrs.max_norm, size=1)            
        online_neg_examples = (online_neg_norms / np.linalg.norm(online_neg_examples, ord=2)) * online_neg_examples

        # add position vector of $p$ to get origin centered coordinates
        online_neg_examples[:self._genattrs.k] = online_neg_examples[:self._genattrs.k] + online_pt.numpy()

        # distances from the manifold will be the norms the samples were rescaled by
        online_neg_distances = online_neg_norms
        
        return online_neg_examples, online_neg_distances

    @abstractmethod
    def embed_in_n(self):
        """
        embedding sampled points, including any global descriptors
        of the manifold (like center, focii, etc.) in `self.n`-dims
        """
        raise NotImplementedError

    @abstractmethod
    def norm(self):
        """
        normalizing the data points to lie in a unit hyper-sphere
        """
        raise NotImplementedError

    @abstractmethod
    def invert_points(self, points):
        """invert normalised points to unnormalised values"""
        raise NotImplementedError

    @abstractmethod
    def invert_distances(self, distances):
        """invert normalised distances to unnormalised values"""
        raise NotImplementedError

    @abstractmethod
    def compute_points(self):
        """
        method to put it all together!

        collate all methods together to compute the
        points here
        """
        raise NotImplementedError

    @abstractmethod
    def viz_test(self):
        """
        specify some visualization tests to check correctess

        this method will differ based on type of manifold
        """
        raise NotImplementedError

    @abstractmethod
    def save_data(self, save_dir):
        """
        save data without pickling to avoid version issues
        """
        raise NotImplementedError

    @abstractmethod
    def load_data(self, dump_dir):
        """
        load data from the dump

        should be capable of loading both from a pickle as well
        as a raw dump
        """
        raise NotImplementedError


    @classmethod
    @abstractmethod
    def save_splits(cls, train_set, val_set, test_set, save_dir=None):
        train_dir = os.path.join(save_dir, "train")
        val_dir = os.path.join(save_dir, "val")
        test_dir = os.path.join(save_dir, "test")
    
        if save_dir is not None:
            os.makedirs(train_dir, exist_ok=True)
            train_set.save_data(train_dir)
            os.makedirs(val_dir, exist_ok=True)
            val_set.save_data(val_dir)
            os.makedirs(test_dir, exist_ok=True)
            test_set.save_data(test_dir)

        return train_set, val_set, test_set


    @classmethod
    @abstractmethod
    def load_splits(cls, dump_dir):

        train_dir = os.path.join(dump_dir, "train")
        train_set = cls()
        train_set.load_data(train_dir)

        val_dir = os.path.join(dump_dir, "val")
        val_set = cls()
        val_set.load_data(val_dir)

        test_dir = os.path.join(dump_dir, "test")
        test_set = cls()
        test_set.load_data(test_dir)

        return train_set, val_set, test_set



if __name__ == '__main__':

    class Test(Manifold):
        
        def __init__(self):
            pass

        @property
        def genattrs(self):
            return super().genattrs

        @genattrs.setter
        def genattrs(self):
            return super().genattrs

        @property
        def specattrs(self):
            return super().specattrs

        @specattrs.setter
        def specattrs(self):
            return super().specattrs

        def gen_points(self):
            return super().gen_points()

        def compute_normals(self):
            return super().compute_normals()

        def make_off_mfld_eg(self):
            return super().make_off_mfld_eg()

        def embed_in_n(self):
            return super().embed_in_n()

        def norm(self):
            return super().norm()

        def invert_distances(self, distances):
            return super().invert_distances(distances)

        def invert_points(self, points):
            return super().invert_points(points)

        def compute_points(self):
            return super().compute_points()

        def load_data(self):
            return super().load_data()
        
        def save_data(self, save_dir):
            return super().save_data(save_dir)

        def viz_test(self):
            return super().viz_test()

    test = Test()
