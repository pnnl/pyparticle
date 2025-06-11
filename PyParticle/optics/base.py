from abc import ABC, abstractmethod

class OpticalParticle(ABC):
    """
    Abstract base class for all optical particle morphologies.
    """

    @abstractmethod
    def compute_optics(self):
        """
        Compute all needed optical properties for this particle.
        """
        pass

    @abstractmethod
    def get_cross_sections(self):
        """
        Return a dictionary of cross-section arrays.
        """
        pass

    @abstractmethod
    def get_refractive_indices(self):
        """
        Return a dictionary or array with relevant refractive indices.
        """
        pass

    @abstractmethod
    def get_cross_section(self, optics_type, rh_idx=None, wvl_idx=None):
        """
        Return the specific cross-section array or value according to optics_type.
        """
        pass
