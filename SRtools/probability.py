import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian2DKernel


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

class Prob2d:
    def __init__(self, data, x_points, y_points, smooth=True):
        """
        Initialize histogramCDF using np.histogram2d to calculate PDF
        
        Parameters:
        data: 2D array of shape (n_points, 2) containing (x, y) coordinates
        x_points: array of x grid points
        y_points: array of y grid points
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

            sigma = 1
            self.pdf_ = filter(self.pdf_, sigma=sigma)
            self.prob_ = filter(self.prob_, sigma=sigma)

            
        
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

    def CDF(self, x, y):
        """
        Calculate CDF values at given points
        
        Parameters:
        x: scalar or array of x coordinates
        y: scalar or array of y coordinates
        
        Returns:
        CDF values at the specified points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Handle scalar inputs
        if x.ndim == 0 and y.ndim == 0:
            # Find nearest grid points
            x_idx = np.argmin(np.abs(self.x_points - x))
            y_idx = np.argmin(np.abs(self.y_points - y))
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return self.cdf[x_idx, y_idx]
        else:
            # Vectorized computation
            assert len(x) == len(y), "x and y must have the same length"
            x_idx = np.searchsorted(self.x_points, x)
            y_idx = np.searchsorted(self.y_points, y)
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return self.cdf[x_idx, y_idx]

    def prob(self, x, y):
        """
        Calculate probability at given points using the histogram PDF multiplied by bin volume
        
        Parameters:
        x: scalar or array of x coordinates
        y: scalar or array of y coordinates
        
        Returns:
        Probability values at the specified points (not density)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Handle scalar inputs
        if x.ndim == 0 and y.ndim == 0:
            # Find nearest grid points
            x_idx = np.argmin(np.abs(self.x_points - x))
            y_idx = np.argmin(np.abs(self.y_points - y))
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return self.prob_[x_idx, y_idx]
        else:
            # Vectorized computation
            assert len(x) == len(y), "x and y must have the same length"
            x_idx = np.searchsorted(self.x_points, x)
            y_idx = np.searchsorted(self.y_points, y)
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return self.prob_[x_idx, y_idx]

    def pdf(self, x, y):
        """
        Returns probability density (PDF) at given points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Handle scalar inputs
        if x.ndim == 0 and y.ndim == 0:
            # Find nearest grid points
            x_idx = np.argmin(np.abs(self.x_points - x))
            y_idx = np.argmin(np.abs(self.y_points - y))
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return self.pdf_[x_idx, y_idx]
        else:
            # Vectorized computation
            assert len(x) == len(y), "x and y must have the same length"
            x_idx = np.searchsorted(self.x_points, x)
            y_idx = np.searchsorted(self.y_points, y)
            x_idx = np.clip(x_idx, 0, len(self.x_points) - 2)
            y_idx = np.clip(y_idx, 0, len(self.y_points) - 2)
            return self.pdf_[x_idx, y_idx]


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
    
    def plotPDF(self, log=True, x_label='Smurf Time', y_label='Death Time', title='Probability Density Function', ax=None):
        if ax is None:
            ax = plt.gca()
        # Check if there are zeros in the upper triangle of self.pdf_
        nrows, ncols = self.pdf_.shape
        upper_triangle = np.triu(self.pdf_, k=1)  # k=1 excludes the diagonal
        has_zeros = np.any(upper_triangle == 0)
        print(has_zeros)
        if log:
            im = ax.imshow(np.log10(self.pdf_.T), extent=[self.x_points[0], self.x_points[-1], self.y_points[0], self.y_points[-1]], origin='lower')
        else:
            im = ax.imshow(self.pdf_.T, extent=[self.x_points[0], self.x_points[-1], self.y_points[0], self.y_points[-1]], origin='lower')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return ax
    
    def plotCDF(self, x_label='Smurf Time', y_label='Death Time', title='Cumulative Distribution Function', ax=None):
        if ax is None:
            ax = plt.gca()
        
        im = ax.imshow(self.cdf.T, extent=[self.x_points[0], self.x_points[-1], self.y_points[0], self.y_points[-1]], origin='lower')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return ax

