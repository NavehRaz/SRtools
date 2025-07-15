import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian2DKernel

##THIS IS THE OLD FILTER FUNCTION, USE filter_params INSTEAD DELETE THIS FUNCTION AFTER TESTING
def filter(data, sigma):
    """
    Apply a Gaussian filter to data, ignoring NaN values, using astropy.convolution.
    First mirrors the data using the upper triangle to fill the lower triangle,
    then applies filtering, and finally sets the lower triangle to zero.
    """
    try:
        from astropy.convolution import convolve, Gaussian2DKernel
    except ImportError:
        raise ImportError("astropy is required for astropy_nan_gaussian_filter. Install it with 'pip install astropy'.")
    
    # Mirror the data: use upper triangle to fill lower triangle (only for square part)
    data_mirrored = data.copy()
    min_dim = min(data.shape[0], data.shape[1])
    data_square = data[:min_dim, :min_dim]
    
    # Fill lower triangle with upper triangle values
    data_mirrored[:min_dim, :min_dim] = np.triu(data_square) + np.triu(data_square, k=1).T
    
    # Apply Gaussian filter to the full data
    kernel = Gaussian2DKernel(sigma)
    filtered = convolve(data_mirrored, kernel, boundary='fill', fill_value=0, nan_treatment='interpolate', preserve_nan=True)
    
    # Set lower triangle to zero (only for square part)
    filtered[:min_dim, :min_dim] = np.triu(filtered[:min_dim, :min_dim])
    
    # Restore diagonal values from original data (only for square part)
    for i in range(min_dim):
        filtered[i, i] = data[i, i]
        if i < min_dim-1:
            filtered[i+1, i] = data[i+1, i]
        
    nrows, ncols = filtered.shape

    # Upper triangle (all i < j)
    upper_indices = np.triu_indices(nrows, k=1, m=ncols)
    upper_vals = filtered[upper_indices]
    min_nonzero_upper = np.min(upper_vals[upper_vals > 0]) if np.any(upper_vals > 0) else 0
    zero_upper = (filtered[upper_indices] == 0)
    if min_nonzero_upper > 0:
        filtered[upper_indices[0][zero_upper], upper_indices[1][zero_upper]] = min_nonzero_upper / 10

    # Diagonal (i == j)
    diag_len = min(nrows, ncols)
    diag_indices = (np.arange(diag_len), np.arange(diag_len))
    diag_vals = filtered[diag_indices]
    min_nonzero_diag = np.min(diag_vals[diag_vals > 0]) if np.any(diag_vals > 0) else 0
    zero_diag = (filtered[diag_indices] == 0)
    if min_nonzero_diag > 0:
        filtered[diag_indices[0][zero_diag], diag_indices[1][zero_diag]] = min_nonzero_diag / 10

    # Sub-diagonal (i == j + 1)
    sub_diag_len = min(nrows - 1, ncols)
    sub_diag_indices = (np.arange(1, sub_diag_len + 1), np.arange(sub_diag_len))
    sub_diag_vals = filtered[sub_diag_indices]
    min_nonzero_sub_diag = np.min(sub_diag_vals[sub_diag_vals > 0]) if np.any(sub_diag_vals > 0) else 0
    zero_sub_diag = (filtered[sub_diag_indices] == 0)
    if min_nonzero_sub_diag > 0:
        filtered[sub_diag_indices[0][zero_sub_diag], sub_diag_indices[1][zero_sub_diag]] = min_nonzero_sub_diag / 10

    return filtered

def _treat_zeros_1d(arr, smooth=True, damper=10):
    """
    Helper function to treat zeros in a 1D array.
    
    Parameters:
    arr: 1D numpy array
    smooth: bool, if True apply smoothing to filled values (default: True)
    
    Returns:
    1D array with zeros replaced and optionally smoothed
    """
    arr = arr.copy()
    arr = np.array(arr)
    
    # Find minimum nonzero value
    nonzero_vals = arr[arr > 0]
    if len(nonzero_vals) == 0:
        return arr
    
    min_nonzero = np.min(nonzero_vals)
    
    # Replace zeros with min_nonzero / damper
    zero_mask = (arr == 0)
    arr[zero_mask] = min_nonzero / damper
    
    # Smooth filled values with their adjacent cells (optional)
    if smooth:
        for i in np.where(zero_mask)[0]:
            if i == 0:
                # First element: average with next element
                arr[i] = (arr[i] + arr[i + 1]) / 2
            elif i == len(arr) - 1:
                # Last element: average with previous element
                arr[i] = (arr[i - 1] + arr[i]) / 2
            else:
                # Middle element: average with both adjacent elements
                arr[i] = (arr[i - 1] + arr[i] + arr[i + 1]) / 3
    
    return arr

def filter_params(data, sigma=1, **kwargs):
    """
    Apply a Gaussian filter to data with customizable parameters.
    
    Parameters:
    data: 2D array to filter
    sigma: Standard deviation for Gaussian kernel (default: 1)
    **kwargs: Filtering parameters:
        - sigma: Standard deviation for Gaussian kernel (can override the sigma parameter)
        - ut/UT/upper_triangle: bool, if True mirror upper triangle before filtering (default: False)
        - lt/LT/lower_triangle: bool, if True mirror lower triangle before filtering (default: False)
        - kd/KD/keep_diagonal: bool, int, 'auto', or 'auto_log', keep and filter diagonal/subdiagonals (default: False)
        - kc/KC/keep_columns: bool, int, 'auto', or 'auto_log', keep and filter columns (default: False)
        - kr/KR/keep_rows: bool, int, 'auto', or 'auto_log', keep and filter rows (default: False)
        
        Auto options detect sharp boundaries using gradient analysis:
        - 'auto': Detect boundary using regular values
        - 'auto_log': Detect boundary using log values (handles zeros)
        - Maximum preserved elements limited to 25% of total size
    
    Returns:
    Filtered 2D array
    """
    try:
        from astropy.convolution import convolve, Gaussian2DKernel
    except ImportError:
        raise ImportError("astropy is required for filter_params. Install it with 'pip install astropy'.")
    
    # Get sigma from kwargs if provided, otherwise use the parameter
    sigma = kwargs.get('sigma', sigma)
    
    # Parse parameters with aliases
    upper_triangle = kwargs.get('upper_triangle', kwargs.get('ut', kwargs.get('UT', False)))
    lower_triangle = kwargs.get('lower_triangle', kwargs.get('lt', kwargs.get('LT', False)))
    keep_diagonal = kwargs.get('keep_diagonal', kwargs.get('kd', kwargs.get('KD', False)))
    keep_columns = kwargs.get('keep_columns', kwargs.get('kc', kwargs.get('KC', False)))
    keep_rows = kwargs.get('keep_rows', kwargs.get('kr', kwargs.get('KR', False)))
    
    # Handle lower triangle by 90-degree counter-clockwise rotation (convert to upper triangle problem)
    if lower_triangle:
        data = np.rot90(data, k=3)   # 90-degree clockwise rotation (3 * 90 = 270 degrees counter-clockwise)
        # Swap rows and columns for preservation parameters
        keep_columns, keep_rows = keep_rows, keep_columns
        upper_triangle = not upper_triangle
    
    def _detect_boundary(data, axis=0, use_log=False, max_fraction=0.25):
        """
        Detect the sharpest boundary along a given axis.
        
        Parameters:
        data: 2D array
        axis: 0 for rows, 1 for columns
        use_log: if True, use log values for detection
        max_fraction: maximum fraction of total size to return
        
        Returns:
        Number of elements to preserve
        """
        if use_log:
            # Use log values, handling zeros
            data_work = np.log10(np.where(data > 0, data, np.nan))
            # Remove NaN values
            # data_work = data_work[~np.isnan(data_work)]
            if len(data_work) == 0:
                return 0
        else:
            data_work = data
        
        # Calculate gradients along the specified axis
        if axis == 0:  # Rows
            gradients = np.abs(np.diff(np.nanmean(data_work, axis=1)))
        else:  # Columns
            gradients = np.abs(np.diff(np.nanmean(data_work, axis=0)))
        
        if len(gradients) == 0:
            return 0
        
        # Find the position of maximum gradient
        max_grad_idx = np.argmax(gradients)
        
        # Convert to number of elements to preserve
        if axis == 0:  # Rows (preserve from end)
            preserve_count = len(gradients) - max_grad_idx
        else:  # Columns (preserve from start)
            preserve_count = max_grad_idx + 1
        
        # Apply maximum fraction limit
        max_allowed = int(data.shape[axis] * max_fraction)
        preserve_count = min(preserve_count, max_allowed)
        
        return preserve_count
    
    def _detect_boundary_diagonal(data, use_log=False, max_fraction=0.25):
        """
        Detect the sharpest boundary along diagonals.
        
        Parameters:
        data: 2D array
        use_log: if True, use log values for detection
        max_fraction: maximum fraction of total size to return
        
        Returns:
        Number of diagonals to preserve
        """
        min_dim = min(data.shape[0], data.shape[1])
        if min_dim <= 1:
            return 0
        
        # Calculate mean of each diagonal (main diagonal and subdiagonals)
        diagonal_means = []
        for k in range(min_dim):
            # Get the k-th diagonal (main diagonal is k=0)
            diag_elements = np.diag(data[:min_dim, :min_dim], k=k)
            # Remove zeros and calculate mean
            nonzero_elements = diag_elements[diag_elements > 0]
            if len(nonzero_elements) > 0:
                if use_log:
                    diag_mean = np.mean(np.log10(nonzero_elements))
                else:
                    diag_mean = np.mean(nonzero_elements)
            else:
                diag_mean = 0
            diagonal_means.append(diag_mean)
        
        # Convert to numpy array
        diagonal_means = np.array(diagonal_means)
        
        # Calculate gradients between adjacent diagonals
        if len(diagonal_means) > 1:
            gradients = np.abs(np.diff(diagonal_means))
            if len(gradients) > 0:
                # Find the position of maximum gradient
                max_grad_idx = np.argmax(gradients)
                # Convert to number of diagonals to preserve (from main diagonal)
                preserve_count = max_grad_idx + 1
                # Apply maximum fraction limit
                max_allowed = int(min_dim * max_fraction)
                preserve_count = min(preserve_count, max_allowed)
                return preserve_count
        
        return 0
    
    # Handle auto detection for keep_columns
    if keep_columns in ['auto', 'auto_log']:
        use_log = (keep_columns == 'auto_log')
        keep_columns = _detect_boundary(data, axis=1, use_log=use_log, max_fraction=0.25)
        # print('keep_columns: ', keep_columns)
    
    # Handle auto detection for keep_rows
    if keep_rows in ['auto', 'auto_log']:
        use_log = (keep_rows == 'auto_log')
        keep_rows = _detect_boundary(data, axis=0, use_log=use_log, max_fraction=0.25)
        # print('keep_rows: ', keep_rows)
    # Handle auto detection for keep_diagonal
    if keep_diagonal in ['auto', 'auto_log']:
        use_log = (keep_diagonal == 'auto_log')
        keep_diagonal = _detect_boundary_diagonal(data, use_log=use_log, max_fraction=0.25)
        # print('keep_diagonal: ', keep_diagonal)
    # Convert boolean to int for diagonal/columns/rows
    if isinstance(keep_diagonal, bool):
        keep_diagonal = 2 if keep_diagonal else 0  # Default behavior: keep diagonal + 1 subdiagonal
    if isinstance(keep_columns, bool):
        keep_columns = 2 if keep_columns else 0
    if isinstance(keep_rows, bool):
        keep_rows = 2 if keep_rows else 0
    
    data_filtered = data.copy()
    nrows, ncols = data.shape
    
    # Handle upper triangle mirroring
    if upper_triangle:
        min_dim = min(nrows, ncols)
        data_square = data[:min_dim, :min_dim]
        # Fill lower triangle with upper triangle values
        data_filtered[:min_dim, :min_dim] = np.triu(data_square) + np.triu(data_square, k=1).T
    
    # Apply Gaussian filter
    kernel = Gaussian2DKernel(sigma)
    filtered = convolve(data_filtered, kernel, boundary='fill', fill_value=0, 
                       nan_treatment='interpolate', preserve_nan=True)
    
    # Handle diagonal preservation with zero treatment
    if keep_diagonal > 0:
        min_dim = min(nrows, ncols)
        for i in range(min_dim):
            # Main diagonal
            diag_val = data[i, i]
            filtered[i, i] = diag_val
            
            # Subdiagonals
            for k in range(1, min(keep_diagonal, min_dim - i)):
                if i + k < min_dim:
                    subdiag_val = data[i + k, i]
                    filtered[i + k, i] = subdiag_val
        
        # Apply zero treatment to all preserved diagonal elements at once
        diag_indices = []
        for i in range(min_dim):
            diag_indices.append((i, i))  # Main diagonal
            for k in range(1, min(keep_diagonal, min_dim - i)):
                if i + k < min_dim:
                    diag_indices.append((i + k, i))  # Subdiagonals
        
        if diag_indices:
            diag_vals = [filtered[idx] for idx in diag_indices]
            diag_vals_treated = _treat_zeros_1d(diag_vals, smooth=True)
            for idx, val in zip(diag_indices, diag_vals_treated):
                filtered[idx] = val
    
    # Handle column preservation with zero treatment
    if keep_columns > 0:
        for j in range(min(keep_columns, ncols)):
            if upper_triangle and j < min(nrows, ncols):
                # When upper triangle is enabled, only preserve upper triangle part of the column
                # For column j, preserve elements from row j onwards (upper triangle)
                col_part = data[j:, j]
                col_part_treated = _treat_zeros_1d(col_part, smooth=True)
                filtered[j:, j] = col_part_treated
            else:
                # Preserve entire column
                col_full = data[:, j]
                col_full_treated = _treat_zeros_1d(col_full, smooth=True)
                filtered[:, j] = col_full_treated
    
    # Handle row preservation with zero treatment
    if keep_rows > 0:
        for i in range(min(keep_rows, nrows)):
            row_idx = nrows - 1 - i  # Start from last row
            if upper_triangle and row_idx < min(nrows, ncols):
                # When upper triangle is enabled, only preserve upper triangle part of the row
                # For row row_idx, preserve elements from column row_idx onwards (upper triangle)
                row_part = data[row_idx, row_idx:]
                row_part_treated = _treat_zeros_1d(row_part, smooth=True)
                filtered[row_idx, row_idx:] = row_part_treated
            else:
                # Preserve entire row
                row_full = data[row_idx, :]
                row_full_treated = _treat_zeros_1d(row_full, smooth=True)
                filtered[row_idx, :] = row_full_treated
    
    # Handle upper triangle zeroing AFTER all preservation operations
    if upper_triangle:
        min_dim = min(nrows, ncols)
        # Zero the lower triangle in the square part
        filtered[:min_dim, :min_dim] = np.triu(filtered[:min_dim, :min_dim])
    
    # Handle zero value replacement for all preserved regions
    nrows, ncols = filtered.shape
    
    # Upper triangle (all i < j) - no smoothing needed
    upper_indices = np.triu_indices(nrows, k=1, m=ncols)
    upper_vals = filtered[upper_indices]
    if np.any(upper_vals > 0):
        # Extract upper triangle as 1D array, treat zeros without smoothing, then restore
        upper_1d = filtered[upper_indices]
        upper_1d_treated = _treat_zeros_1d(upper_1d, smooth=False)
        filtered[upper_indices] = upper_1d_treated
    # End of Selection

    # Note: Zero treatment for preserved regions is now done during preservation operations above

    # Rotate back if we used lower triangle
    if lower_triangle:
        filtered = np.rot90(filtered, k=1) # 90-degree counter-clockwise rotation

    return filtered

class Prob2d:
    def __init__(self, data, x_points, y_points, sigma=1, smooth=True, filter_params_dict=None):
        """
        Initialize histogramCDF using np.histogram2d to calculate PDF
        
        Parameters:
        data: 2D array of shape (n_points, 2) containing (x, y) coordinates
        x_points: array of x grid points
        y_points: array of y grid points
        sigma: Standard deviation for Gaussian kernel (default: 1)
        smooth: Boolean, if True apply filtering (default: True)
        filter_params_dict: Dictionary of filter parameters for filter_params function (default: None)
        """
        self.data = data
        self.x_points = x_points
        self.y_points = y_points
        self.n_points = len(data)
        
        # Calculate PDF using histogram2d
        self.pdf_, self.x_edges, self.y_edges = np.histogram2d(
            data[:, 0], data[:, 1], 
            bins=[x_points, y_points], 
            density=True
        )

        self.prob_,_,_ = np.histogram2d(
            data[:, 0], data[:, 1], 
            bins=[x_points, y_points], 
            density=False
        )
        self.prob_ = self.prob_/len(data)

        if smooth:
            # Always use filter_params function with default values if no filter_params_dict provided
            if filter_params_dict is not None:
                # Get sigma from filter_params_dict if provided, otherwise use the parameter
                filter_sigma = filter_params_dict.get('sigma', sigma)
                self.pdf_ = filter_params(self.pdf_, sigma=filter_sigma, **filter_params_dict)
                self.prob_ = filter_params(self.prob_, sigma=filter_sigma, **filter_params_dict)
            else:
                # Use filter_params with default values and the provided sigma
                self.pdf_ = filter_params(self.pdf_, sigma=sigma)
                self.prob_ = filter_params(self.prob_, sigma=sigma)

            
        
        # Calculate CDF by cumulative sum of PDF
        # Note: We need to handle the cumulative sum properly for 2D
        # Use cumulative sum to avoid redundant calculations
        # First, compute cumulative sum along rows
        row_cumsum = np.cumsum(self.prob_, axis=1)
        # Then compute cumulative sum along columns
        self.cdf = np.cumsum(row_cumsum, axis=0)
        
        # Normalize CDF to ensure it goes from 0 to 1
        if self.cdf[-1, -1] > 0:
            # print('cdf max: ', self.cdf[-1, -1])
            # print('normalizing cdf ')
            self.cdf = self.cdf / self.cdf[-1, -1]

    def _get_normalized_matrix(self, matrix, normalize):
        """
        Helper function to get normalized matrix based on normalize parameter
        
        Parameters:
        matrix: 2D array to normalize
        normalize: None, 'x', or 'y'
        
        Returns:
        Normalized matrix
        """
        if normalize is None:
            return matrix
        
        if normalize == 'x':
            # Normalize over rows (each row sums to 1)
            row_sums = np.sum(matrix, axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return matrix / row_sums
        elif normalize == 'y':
            # Normalize over columns (each column sums to 1)
            col_sums = np.sum(matrix, axis=0, keepdims=True)
            # Avoid division by zero
            col_sums = np.where(col_sums == 0, 1, col_sums)
            return matrix / col_sums
        else:
            raise ValueError("normalize must be None, 'x', or 'y'")

    def _get_normalized_cdf_matrix(self, normalize):
        """
        Helper function to get normalized CDF matrix using last cell in row/column
        
        Parameters:
        normalize: None, 'x', or 'y'
        
        Returns:
        Normalized CDF matrix
        """
        if normalize is None:
            return self.cdf
        
        if normalize == 'x':
            # Normalize by last cell in each row
            row_maxs = self.cdf[:, -1:].copy()  # Last column of each row
            # Avoid division by zero
            row_maxs = np.where(row_maxs == 0, 1, row_maxs)
            return self.cdf / row_maxs
        elif normalize == 'y':
            # Normalize by last cell in each column
            col_maxs = self.cdf[-1:, :].copy()  # Last row of each column
            # Avoid division by zero
            col_maxs = np.where(col_maxs == 0, 1, col_maxs)
            return self.cdf / col_maxs
        else:
            raise ValueError("normalize must be None, 'x', or 'y'")

    def CDF(self, x, y, normalize=None):
        """
        Calculate CDF values at given points
        
        Parameters:
        x: scalar or array of x coordinates
        y: scalar or array of y coordinates
        normalize: None, 'x', or 'y' for conditional CDF normalization
        
        Returns:
        CDF values at the specified points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Get normalized CDF matrix
        cdf_matrix = self._get_normalized_cdf_matrix(normalize)
        
        # Handle scalar inputs
        if x.ndim == 0 and y.ndim == 0:
            # Find nearest grid points
            x_idx = np.argmin(np.abs(self.x_points - x))
            y_idx = np.argmin(np.abs(self.y_points - y))
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return cdf_matrix[x_idx, y_idx]
        else:
            # Vectorized computation
            assert len(x) == len(y), "x and y must have the same length"
            x_idx = np.searchsorted(self.x_points, x)
            y_idx = np.searchsorted(self.y_points, y)
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return cdf_matrix[x_idx, y_idx]

    def prob(self, x, y, normalize=None):
        """
        Calculate probability at given points using the histogram PDF multiplied by bin volume
        
        Parameters:
        x: scalar or array of x coordinates
        y: scalar or array of y coordinates
        normalize: None, 'x', or 'y' for conditional probability normalization
        
        Returns:
        Probability values at the specified points (not density)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Get normalized probability matrix
        prob_matrix = self._get_normalized_matrix(self.prob_, normalize)
        
        # Handle scalar inputs
        if x.ndim == 0 and y.ndim == 0:
            # Find nearest grid points
            x_idx = np.argmin(np.abs(self.x_points - x))
            y_idx = np.argmin(np.abs(self.y_points - y))
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return prob_matrix[x_idx, y_idx]
        else:
            # Vectorized computation
            assert len(x) == len(y), "x and y must have the same length"
            x_idx = np.searchsorted(self.x_points, x)
            y_idx = np.searchsorted(self.y_points, y)
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return prob_matrix[x_idx, y_idx]

    def pdf(self, x, y, normalize=None):
        """
        Returns probability density (PDF) at given points
        
        Parameters:
        x: scalar or array of x coordinates
        y: scalar or array of y coordinates
        normalize: None, 'x', or 'y' for conditional PDF normalization
        
        Returns:
        PDF values at the specified points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Get normalized PDF matrix
        if normalize is not None:
            # For PDF normalization, we need to use the normalization factor from prob matrix
            prob_matrix = self._get_normalized_matrix(self.prob_, normalize)
            # Calculate the normalization factor for each row/column
            if normalize == 'x':
                # Normalize over rows
                row_sums = np.sum(self.prob_, axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                pdf_matrix = self.pdf_ / row_sums
            elif normalize == 'y':
                # Normalize over columns
                col_sums = np.sum(self.prob_, axis=0, keepdims=True)
                col_sums = np.where(col_sums == 0, 1, col_sums)
                pdf_matrix = self.pdf_ / col_sums
        else:
            pdf_matrix = self.pdf_
        
        # Handle scalar inputs
        if x.ndim == 0 and y.ndim == 0:
            # Find nearest grid points
            x_idx = np.argmin(np.abs(self.x_points - x))
            y_idx = np.argmin(np.abs(self.y_points - y))
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return pdf_matrix[x_idx, y_idx]
        else:
            # Vectorized computation
            assert len(x) == len(y), "x and y must have the same length"
            x_idx = np.searchsorted(self.x_points, x)
            y_idx = np.searchsorted(self.y_points, y)
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return pdf_matrix[x_idx, y_idx]

    def prob_diff(self, diff_bin):
        """
        PURE CDF APPROACH: Calculate the probability using only the CDF
        This is what you would use for new data points not in the original dataset.
        Note: This may have discretization errors due to the grid resolution.
        """
        prob = 0.0
        # For each grid point (x_i, y_j), check if it contributes to the difference bin
        for i, x in enumerate(self.x_points[:-2]):
            for j, y in enumerate(self.y_points[:-2]):
                # Check if this point satisfies the difference constraint
                diff = y - x
                if diff_bin[0] <= diff < diff_bin[1]:
                    # Calculate the probability mass at this point using mixed second difference
                    if i == 0 and j == 0:
                        contribution = self.cdf[i, j]
                    elif i == 0:
                        contribution = self.cdf[i, j] - self.cdf[i, j-1]
                    elif j == 0:
                        contribution = self.cdf[i, j] - self.cdf[i-1, j]
                    else:
                        contribution = (self.cdf[i, j] - self.cdf[i-1, j] - 
                                      self.cdf[i, j-1] + self.cdf[i-1, j-1])
                    
                    prob += contribution
        
        return prob
    
    def plotPDF(self, log=True, x_label='Smurf Time', y_label='Death Time', title='Probability Density Function', ax=None, normalize=None):
        if ax is None:
            ax = plt.gca()
        # Check if there are zeros in the upper triangle of self.pdf_
        nrows, ncols = self.pdf_.shape
        upper_triangle = np.triu(self.pdf_, k=1)  # k=1 excludes the diagonal
        has_zeros = np.any(upper_triangle == 0)
        
        # Get normalized PDF matrix for plotting
        if normalize is not None:
            if normalize == 'x':
                row_sums = np.sum(self.prob_, axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                pdf_matrix = self.pdf_ / row_sums
            elif normalize == 'y':
                col_sums = np.sum(self.prob_, axis=0, keepdims=True)
                col_sums = np.where(col_sums == 0, 1, col_sums)
                pdf_matrix = self.pdf_ / col_sums
        else:
            pdf_matrix = self.pdf_
        
        if log:
            im = ax.imshow(np.log10(pdf_matrix.T), extent=[self.x_points[0], self.x_points[-1], self.y_points[0], self.y_points[-1]], origin='lower')
        else:
            im = ax.imshow(pdf_matrix.T, extent=[self.x_points[0], self.x_points[-1], self.y_points[0], self.y_points[-1]], origin='lower')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return ax
    
    def plotCDF(self, x_label='Smurf Time', y_label='Death Time', title='Cumulative Distribution Function', ax=None, normalize=None):
        if ax is None:
            ax = plt.gca()
        
        # Get normalized CDF matrix for plotting
        cdf_matrix = self._get_normalized_cdf_matrix(normalize)
        
        im = ax.imshow(cdf_matrix.T, extent=[self.x_points[0], self.x_points[-1], self.y_points[0], self.y_points[-1]], origin='lower')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return ax

