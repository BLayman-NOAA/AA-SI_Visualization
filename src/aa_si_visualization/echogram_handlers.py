"""Strategy-pattern handlers for different echogram data structures.

Each handler knows how to detect its dataset layout, calculate depth/ping
ranges, and slice data for a single frequency or feature.  The public
:func:`create_handler` factory inspects an :class:`xarray.Dataset` and
returns the appropriate handler subclass.
"""

import numpy as np
from abc import ABC, abstractmethod
from aa_si_utils import utils


class EchogramDataHandler(ABC):
    """Abstract base class for handling different echogram data structures."""
    
    def __init__(self, dataset, sv_variable_name):
        """
        Initialize handler with dataset and variable name.
        
        Args:
            dataset: xarray.Dataset containing the echogram data
            sv_variable_name: Name of the Sv variable to plot
        """
        self.dataset = dataset
        self.sv_variable_name = sv_variable_name
        
    @abstractmethod
    def detect_structure(self):
        """
        Analyze dataset structure and store relevant coordinates.
        
        Returns:
            dict: Structure information including type and dimensions
        """
        raise NotImplementedError("Subclasses must implement detect_structure()")
    
    @abstractmethod
    def calculate_depth_indices(self, min_depth, max_depth):
        """
        Calculate depth indices for the given depth range.
        
        Args:
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters
            
        Returns:
            tuple: (min_depth_index, max_depth_index)
        """
        raise NotImplementedError("Subclasses must implement calculate_depth_indices()")
    
    @abstractmethod
    def calculate_ping_range(self, ping_min, ping_max, original_ds=None):
        """
        Calculate actual ping range to use (may need conversion for MVBS).
        
        Args:
            ping_min: Minimum ping index
            ping_max: Maximum ping index
            original_ds: Original Sv dataset (needed for MVBS conversion)
            
        Returns:
            tuple: (ping_min_actual, ping_max_actual)
        """
        raise NotImplementedError("Subclasses must implement calculate_ping_range()")
    
    @abstractmethod
    def get_depth_extent(self, min_idx, max_idx):
        """
        Get actual depth values for display extent.
        
        Args:
            min_idx: Minimum depth index
            max_idx: Maximum depth index
            
        Returns:
            tuple: (min_depth_shown, max_depth_shown) in meters
        """
        raise NotImplementedError("Subclasses must implement get_depth_extent()")
    
    @abstractmethod
    def get_ping_times(self):
        """
        Get ping time array for x-axis calculations.
        
        Returns:
            numpy.ndarray: Array of ping times
        """
        raise NotImplementedError("Subclasses must implement get_ping_times()")
    
    @abstractmethod
    def slice_data_for_frequency(self, freq_idx, ping_range, depth_range):
        """
        Slice data for a specific frequency/channel.
        
        Args:
            freq_idx: Frequency/channel index
            ping_range: Tuple of (ping_min, ping_max)
            depth_range: Tuple of (depth_min_idx, depth_max_idx)
            
        Returns:
            xarray.DataArray: Sliced data for plotting
        """
        raise NotImplementedError("Subclasses must implement slice_data_for_frequency()")
    
    @abstractmethod
    def is_mvbs_structured(self):
        """
        Check if this handler represents MVBS-structured data.
        
        Returns:
            bool: True if MVBS structure, False otherwise
        """
        raise NotImplementedError("Subclasses must implement is_mvbs_structured()")


class SvDataHandler(EchogramDataHandler):
    """Handler for regular Sv data."""
    
    def detect_structure(self):
        """Analyze regular Sv dataset structure."""
        self.echo_range_coord = self.dataset['echo_range']
        self.is_mvbs = False
        self.ping_times = self.dataset['ping_time'].values
        
        return {
            'type': 'Sv',
            'dimensions': ['channel', 'ping_time', 'range_sample']
        }
    
    def calculate_depth_indices(self, min_depth, max_depth):
        """Calculate depth indices using helper function."""
        self.min_depth_index = utils.get_closest_index_for_depth(self.dataset, min_depth)
        self.max_depth_index = utils.get_closest_index_for_depth(self.dataset, max_depth)
        
        return (self.min_depth_index, self.max_depth_index)
    
    def calculate_ping_range(self, ping_min, ping_max, original_ds=None):
        """Store ping range directly (no conversion needed)."""
        self.ping_min = ping_min
        self.ping_max = ping_max
        
        return (ping_min, ping_max)
    
    def get_depth_extent(self, min_idx, max_idx):
        """Extract actual depth values from echo_range."""
        actual_depths = self.echo_range_coord.isel(channel=0, ping_time=0).values
        min_shown = actual_depths[min_idx]
        max_shown = actual_depths[min(max_idx, len(actual_depths)-1)]
        
        return (min_shown, max_shown)
    
    def get_ping_times(self):
        """Return stored ping times."""
        return self.ping_times
    
    def slice_data_for_frequency(self, freq_idx, ping_range, depth_range):
        """Slice regular Sv data using channel dimension."""
        ping_min, ping_max = ping_range
        depth_min, depth_max = depth_range
        
        return self.dataset[self.sv_variable_name].isel(
            channel=freq_idx,
            ping_time=slice(ping_min, ping_max),
            range_sample=slice(depth_min, depth_max)
        )
    
    def is_mvbs_structured(self):
        """Regular Sv is not MVBS structured."""
        return False


class MvbsDataHandler(EchogramDataHandler):
    """Handler for MVBS (gridded) data."""
    
    def detect_structure(self):
        """Analyze MVBS dataset structure."""
        self.echo_range_coord = self.dataset['echo_range']
        self.echo_range_values = self.echo_range_coord.values  # 1D array
        self.is_mvbs = True
        self.ping_times = self.dataset['ping_time'].values
        
        return {
            'type': 'MVBS',
            'dimensions': ['channel', 'ping_time', 'echo_range']
        }
    
    def calculate_depth_indices(self, min_depth, max_depth):
        """Calculate depth indices using direct array search."""
        self.min_depth_index = np.argmin(np.abs(self.echo_range_values - min_depth))
        self.max_depth_index = np.argmin(np.abs(self.echo_range_values - max_depth))
        
        return (self.min_depth_index, self.max_depth_index)
    
    def calculate_ping_range(self, ping_min, ping_max, original_ds=None):
        """
        Convert ping indices from original Sv to MVBS grid.
        Critical for aligning MVBS bins with original ping range.
        """
        # Store original ping indices for GPS calculations
        self.ping_min = ping_min
        self.ping_max = ping_max
        
        if original_ds is not None:
            # Get target times from original dataset
            original_ping_times = original_ds['ping_time'].values
            target_start_time = original_ping_times[ping_min]
            target_end_time = original_ping_times[ping_max]
            
            # Find closest indices in MVBS grid
            ping_min_mvbs = np.argmin(np.abs(self.ping_times - target_start_time))
            ping_max_mvbs = np.argmin(np.abs(self.ping_times - target_end_time))
            
            self.ping_min_converted = ping_min_mvbs
            self.ping_max_converted = ping_max_mvbs
        else:
            # Use full range if no conversion info available
            self.ping_min_converted = 0
            self.ping_max_converted = len(self.ping_times) - 1
        
        return (self.ping_min_converted, self.ping_max_converted)
    
    def get_depth_extent(self, min_idx, max_idx):
        """Get depth values directly from 1D echo_range array."""
        return (self.echo_range_values[min_idx], self.echo_range_values[max_idx])
    
    def get_ping_times(self):
        """Return stored ping times."""
        return self.ping_times
    
    def slice_data_for_frequency(self, freq_idx, ping_range, depth_range):
        """
        Slice MVBS data with dynamically discovered feature dimension.
        Uses converted ping range for MVBS bins.
        """
        ping_min, ping_max = ping_range
        depth_min, depth_max = depth_range
        
        # Discover feature dimension dynamically - exclude MVBS dimensions
        feature_dim = [d for d in self.dataset[self.sv_variable_name].dims 
                      if d not in ['ping_time', 'echo_range']][0]
        
        # Build slice dictionary with dynamic dimension name
        slice_dict = {
            feature_dim: freq_idx,
            'ping_time': slice(ping_min, ping_max),
            'echo_range': slice(depth_min, depth_max)
        }
        
        return self.dataset[self.sv_variable_name].isel(**slice_dict)
    
    def is_mvbs_structured(self):
        """Return ``True`` — MVBS data is always MVBS structured."""
        return True


class MlSvDataHandler(SvDataHandler):
    """Handler for ML data derived from regular Sv.

    Extends :class:`SvDataHandler` with automatic discovery of
    grid coordinates and feature dimensions used by the ML pipeline.
    """
    
    def detect_structure(self):
        """Analyze ML-Sv structure and discover grid coordinates."""
        # Get base structure from parent
        structure_info = super().detect_structure()
        
        # Discover grid coordinates using ML library
        from aa_si_ml import ml
        self.grid_coords = ml.get_grid_coordinates(self.dataset, self.sv_variable_name)
        
        # Discover feature dimension (could be 'channel' or 'feature')
        regridded_data = self.dataset[self.sv_variable_name]
        self.feature_dim = [dim for dim in regridded_data.dims 
                           if dim not in self.grid_coords][0]
        
        # Update structure info with ML-specific details
        structure_info['type'] = 'ML-Sv'
        structure_info['grid_coords'] = self.grid_coords
        structure_info['feature_dim'] = self.feature_dim
        
        return structure_info
    
    def slice_data_for_frequency(self, freq_idx, ping_range, depth_range):
        """
        Slice ML-Sv data using discovered grid coordinates and feature dimension.
        """
        ping_min, ping_max = ping_range
        depth_min, depth_max = depth_range
        
        # Build slice dictionary using discovered dimensions
        slice_dict = {
            self.feature_dim: freq_idx,
            self.grid_coords[0]: slice(ping_min, ping_max),
            self.grid_coords[1]: slice(depth_min, depth_max)
        }
        
        return self.dataset[self.sv_variable_name].isel(**slice_dict)


class MlMvbsDataHandler(MvbsDataHandler):
    """Handler for ML data derived from MVBS.

    Extends :class:`MvbsDataHandler` with automatic discovery of
    grid coordinates and feature dimensions used by the ML pipeline.
    """
    
    def detect_structure(self):
        """Analyze ML-MVBS structure and discover grid coordinates."""
        # Get base MVBS structure from parent
        structure_info = super().detect_structure()
        
        # Discover grid coordinates and feature dimension using ML library
        from aa_si_ml import ml
        self.grid_coords = ml.get_grid_coordinates(self.dataset, self.sv_variable_name)
        
        regridded_data = self.dataset[self.sv_variable_name]
        self.feature_dim = [dim for dim in regridded_data.dims 
                           if dim not in self.grid_coords][0]
        
        # Update structure info with ML-specific details
        structure_info['type'] = 'ML-MVBS'
        structure_info['grid_coords'] = self.grid_coords
        structure_info['feature_dim'] = self.feature_dim
        
        return structure_info
    
    def slice_data_for_frequency(self, freq_idx, ping_range, depth_range):
        """
        Slice ML-MVBS data combining MVBS logic with ML feature discovery.
        Uses converted ping range and discovered grid coordinates.
        """
        ping_min, ping_max = ping_range
        depth_min, depth_max = depth_range
        
        # Build slice dictionary using discovered dimensions
        slice_dict = {
            self.feature_dim: freq_idx,
            self.grid_coords[0]: slice(ping_min, ping_max),
            self.grid_coords[1]: slice(depth_min, depth_max)
        }
        
        return self.dataset[self.sv_variable_name].isel(**slice_dict)


class ClusterDataHandler(EchogramDataHandler):
    """Handler for cluster result data (single-channel, no frequency dimension)."""
    
    def detect_structure(self):
        """
        Analyze cluster data structure - can be derived from Sv or MVBS.
        Cluster data has shape (ping_time, range_sample) or (ping_time, echo_range).
        """
        self.echo_range_coord = self.dataset['echo_range']
        self.ping_times = self.dataset['ping_time'].values
        
        # Determine if MVBS-derived (1D echo_range) or Sv-derived (2D echo_range)
        if len(self.echo_range_coord.dims) == 1:
            self.is_mvbs = True
            self.echo_range_values = self.echo_range_coord.values
            data_type = 'Cluster-MVBS'
            dimensions = ['ping_time', 'echo_range']
        else:
            self.is_mvbs = False
            data_type = 'Cluster-Sv'
            dimensions = ['ping_time', 'range_sample']
        
        return {
            'type': data_type,
            'dimensions': dimensions
        }
    
    def calculate_depth_indices(self, min_depth, max_depth):
        """Calculate depth indices based on structure type."""
        if self.is_mvbs:
            # MVBS logic - direct array search on 1D echo_range
            self.min_depth_index = np.argmin(np.abs(self.echo_range_values - min_depth))
            self.max_depth_index = np.argmin(np.abs(self.echo_range_values - max_depth))
        else:
            # Sv logic - use helper function
            self.min_depth_index = utils.get_closest_index_for_depth(self.dataset, min_depth)
            self.max_depth_index = utils.get_closest_index_for_depth(self.dataset, max_depth)
        
        return (self.min_depth_index, self.max_depth_index)
    
    def calculate_ping_range(self, ping_min, ping_max, original_ds=None):
        """
        Calculate ping range - convert from original Sv indices if MVBS-structured.
        """
        # Store original ping indices for GPS calculations
        self.ping_min = ping_min
        self.ping_max = ping_max
        
        if self.is_mvbs and original_ds is not None:
            # Convert from original Sv ping indices to MVBS grid
            original_ping_times = original_ds['ping_time'].values
            target_start_time = original_ping_times[ping_min]
            target_end_time = original_ping_times[ping_max]
            
            # Find closest indices in MVBS grid
            ping_min_mvbs = np.argmin(np.abs(self.ping_times - target_start_time))
            ping_max_mvbs = np.argmin(np.abs(self.ping_times - target_end_time))
            
            self.ping_min_converted = ping_min_mvbs
            self.ping_max_converted = ping_max_mvbs
        else:
            # Use indices directly for Sv-structured or when no conversion available
            self.ping_min_converted = ping_min
            self.ping_max_converted = ping_max
        
        return (self.ping_min_converted, self.ping_max_converted)
    
    def get_depth_extent(self, min_idx, max_idx):
        """Get depth extent based on structure type."""
        if self.is_mvbs:
            # Get values from 1D echo_range array
            return (self.echo_range_values[min_idx], self.echo_range_values[max_idx])
        else:
            # Extract from 2D echo_range coordinate
            actual_depths = self.echo_range_coord.isel(channel=0, ping_time=0).values
            min_shown = actual_depths[min_idx]
            max_shown = actual_depths[min(max_idx, len(actual_depths)-1)]
            return (min_shown, max_shown)
    
    def get_ping_times(self):
        """Return ping time array."""
        return self.ping_times
    
    def slice_data_for_frequency(self, freq_idx, ping_range, depth_range):
        """
        Slice cluster data - ignores freq_idx since cluster data has no frequency dimension.
        Returns slice using only ping_range and depth_range.
        """
        ping_min, ping_max = ping_range
        depth_min, depth_max = depth_range
        
        if self.is_mvbs:
            # MVBS structure - use echo_range
            return self.dataset[self.sv_variable_name].isel(
                ping_time=slice(ping_min, ping_max),
                echo_range=slice(depth_min, depth_max)
            )
        else:
            # Sv structure - use range_sample
            return self.dataset[self.sv_variable_name].isel(
                ping_time=slice(ping_min, ping_max),
                range_sample=slice(depth_min, depth_max)
            )
    
    def is_mvbs_structured(self):
        """Return whether this is MVBS-structured cluster data."""
        return self.is_mvbs


def create_handler(ds_Sv, sv_variable_name, ml_data_variable=None):
    """
    Factory function to create appropriate handler based on data structure.
    
    Args:
        ds_Sv: xarray.Dataset containing echogram data
        sv_variable_name: Name of the Sv variable to plot
        ml_data_variable: Name of ML data variable (None for regular Sv/MVBS)
        
    Returns:
        EchogramDataHandler: Appropriate handler instance with structure detected
    """
    
    # Check if this is cluster data (single-channel, no feature dimension)
    if sv_variable_name in ds_Sv:
        data_var = ds_Sv[sv_variable_name]
        data_dims = set(data_var.dims)
        
        # Cluster data has only ping_time and range dimension (no channel/feature dim)
        is_cluster_data = (
            'ping_time' in data_dims and 
            len(data_dims) == 2 and
            ('range_sample' in data_dims or 'echo_range' in data_dims)
        )
        
        if is_cluster_data:
            print("Creating Cluster handler")
            handler = ClusterDataHandler(ds_Sv, sv_variable_name)
            structure_info = handler.detect_structure()
            print(f"Detected structure: {structure_info['type']}")
            return handler
    
    # Determine handler type based on data characteristics
    if ml_data_variable is None:
        # Regular Sv or MVBS data
        if 'echo_range' in ds_Sv[sv_variable_name].coords:
            # Check if echo_range is 1D (MVBS) or multi-dimensional (Sv)
            if len(ds_Sv['echo_range'].dims) == 1:
                print("Creating MVBS handler")
                handler = MvbsDataHandler(ds_Sv, sv_variable_name)
            else:
                print("Creating regular Sv handler")
                handler = SvDataHandler(ds_Sv, sv_variable_name)
        else:
            print("Creating regular Sv handler (no echo_range check)")
            handler = SvDataHandler(ds_Sv, sv_variable_name)
    else:
        # ML data - determine if derived from MVBS or regular Sv
        if len(ds_Sv['echo_range'].dims) == 1:
            print("Creating ML-MVBS handler")
            handler = MlMvbsDataHandler(ds_Sv, sv_variable_name)
        else:
            print("Creating ML-Sv handler")
            handler = MlSvDataHandler(ds_Sv, sv_variable_name)
    
    # Detect and store structure information
    structure_info = handler.detect_structure()
    print(f"Detected structure: {structure_info['type']}")
    
    return handler


