import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import deathTimesDataSet as dtds


class Life_table(dtds.Dataset):
    """
    Life table class for working with aggregate survival/hazard data.
    
    This class inherits from Dataset but works with life table data (age bins with 
    number alive at each age) instead of individual death times. Methods that require 
    individual data either sample from the survival distribution or raise NotImplementedError.
    
    Attributes:
        ages (array): Age bins for the life table
        n_alive (array): Number alive at each age (will derive survival from this)
        hazard_values (array, optional): Hazard values at each age
        tail_bin (dict, optional): Single bin for individuals who survived beyond the last age
                                   Format: {'age': float, 'n_alive': int}
        warnings_enabled (bool): Class variable to control warning messages
    
    Examples:
        >>> ages = np.array([0, 10, 20, 30, 40, 50])
        >>> n_alive = np.array([1000, 950, 900, 800, 600, 300])
        >>> lt = Life_table(ages, n_alive)
        >>> 
        >>> # With tail bin (100 people alive past age 50)
        >>> lt = Life_table(ages, n_alive, tail_bin={'age': 50, 'n_alive': 100})
    """
    
    warnings_enabled = True  # Class variable
    
    def __init__(self, ages, n_alive, hazard_values=None, tail_bin=None, 
                 npeople=None, bandwidth=3, properties=None, data_dt=None):
        """
        Initialize a Life_table object.
        
        Parameters:
        -----------
        ages : array-like
            Age values (bin centers or edges) for the life table
        n_alive : array-like
            Number of individuals alive at each age
        hazard_values : array-like, optional
            Hazard values at each age. If None, computed from survival curve
        tail_bin : dict, optional
            Single bin for right-censored individuals beyond last age.
            Format: {'age': float, 'n_alive': int}
        npeople : int, optional
            Total cohort size. If None, inferred from n_alive[0]
        bandwidth : int, optional
            Bandwidth for hazard smoothing (kept for compatibility). Default: 3
        properties : dict, optional
            Dictionary of properties (kept for compatibility)
        data_dt : float, optional
            Time step size. If None, inferred from np.diff(ages)
        """
        # Convert to numpy arrays
        self.ages = np.array(ages)
        self.n_alive = np.array(n_alive)
        
        # Validate inputs
        self._validate_inputs()
        
        # Determine npeople
        if npeople is None:
            npeople = self.n_alive[0]
        
        # Store life table specific attributes
        self.tail_bin = tail_bin
        self._hazard_values = hazard_values
        
        # Infer data_dt if not provided
        if data_dt is None and len(self.ages) > 1:
            data_dt = np.median(np.diff(self.ages))
        
        # Create placeholder death_times and events
        # These are not used directly but needed for Dataset compatibility
        death_times = np.array([])
        events = np.array([])
        
        # Store attributes before calling parent __init__
        self.bandwidth = bandwidth
        self.n = int(npeople)
        self.t_end = self.ages[-1] if tail_bin is None else tail_bin['age']
        self.events = events
        self.death_times = death_times
        self.external_hazard = np.inf
        self.properties = properties
        self.data_dt = data_dt
        
        # Initialize parent class attributes that won't be set by parent __init__
        self.kmf = None
        self.naf = None
        self.median_lifetime = None
        self.survival = None
        self.hazard = None
        self.kmf_confidence_interval = None
        
        # Calculate survival and hazard from life table
        self.calc_survival_and_hazard()
    
    def _validate_inputs(self):
        """Validate that ages and n_alive are properly formatted."""
        if len(self.ages) != len(self.n_alive):
            raise ValueError("ages and n_alive must have the same length")
        
        if len(self.ages) == 0:
            raise ValueError("ages and n_alive cannot be empty")
        
        # Check ages are monotonic increasing
        if not np.all(np.diff(self.ages) > 0):
            raise ValueError("ages must be monotonically increasing")
        
        # Check n_alive is non-increasing
        if not np.all(np.diff(self.n_alive) <= 0):
            raise ValueError("n_alive must be non-increasing")
        
        # Check non-negative values
        if np.any(self.ages < 0):
            raise ValueError("ages must be non-negative")
        
        if np.any(self.n_alive < 0):
            raise ValueError("n_alive must be non-negative")
        
        if self.n_alive[0] == 0:
            raise ValueError("n_alive[0] must be positive to determine cohort size")
    
    def calc_survival_and_hazard(self, events=None):
        """
        Calculate survival and hazard functions from life table data.
        
        Overrides parent method to work with aggregate n_alive data instead of 
        individual death times.
        
        Parameters:
        -----------
        events : array-like, optional
            Not used for life table (kept for compatibility with parent signature)
        """
        # Calculate survival from n_alive
        survival_values = self.n_alive / self.n_alive[0]
        self.survival = (self.ages.copy(), survival_values)
        
        # Calculate or use provided hazard
        if self._hazard_values is not None:
            self.hazard = (self.ages.copy(), np.array(self._hazard_values))
        else:
            hazard_values = self._compute_hazard_from_survival()
            self.hazard = (self.ages.copy(), hazard_values)
        
        # Calculate median lifetime
        self.median_lifetime = self._calculate_median_from_survival()
        
        # Set confidence intervals to None (not available for life tables)
        self.kmf_confidence_interval = [np.zeros_like(self.ages), np.zeros_like(self.ages)]
    
    def _compute_hazard_from_survival(self):
        """
        Compute hazard function from survival curve.
        
        Uses the formula: h(t) ≈ -(S(t+dt) - S(t)) / (S(t) * dt)
        
        Returns:
        --------
        hazard_values : ndarray
            Hazard values at each age
        """
        if Life_table.warnings_enabled:
            print("Warning: Computing hazard from survival curve. " +
                  "Provide hazard_values explicitly for more accurate results.")
        
        t, s = self.survival
        dt = np.diff(t)
        ds = np.diff(s)
        
        # Hazard: h(t) = -dS/dt / S(t)
        # Use forward differences for all but the last point
        hazard = np.zeros(len(t))
        hazard[:-1] = -ds / (s[:-1] * dt)
        
        # For the last point, use the previous hazard value
        hazard[-1] = hazard[-2] if len(hazard) > 1 else 0
        
        # Ensure non-negative hazard
        hazard = np.maximum(hazard, 0)
        
        return hazard
    
    def _calculate_median_from_survival(self):
        """Calculate median lifetime from survival curve."""
        t, s = self.survival
        # Find where survival crosses 0.5
        if s[-1] > 0.5:
            # More than half survived to the end
            return t[-1]
        idx = np.where(s <= 0.5)[0]
        if len(idx) == 0:
            return t[-1]
        # Interpolate for more precise median
        i = idx[0]
        if i == 0:
            return t[0]
        # Linear interpolation
        t1, t2 = t[i-1], t[i]
        s1, s2 = s[i-1], s[i]
        median = t1 + (0.5 - s1) * (t2 - t1) / (s2 - s1)
        return median
    
    def _sample_death_times_from_survival(self, n):
        """
        Sample death times from the survival curve using inverse transform sampling.
        
        Parameters:
        -----------
        n : int
            Number of samples to generate
        
        Returns:
        --------
        death_times : ndarray
            Sampled death times
        events : ndarray
            Event indicators (1 for death, 0 for censored/tail_bin)
        """
        t, s = self.survival
        
        # Generate uniform random numbers
        u = np.random.uniform(0, 1, n)
        
        # For inverse transform sampling, we need to invert the survival function
        # S(t) gives probability of surviving past t
        # We want to find t such that S(t) = u, where u ~ Uniform(0,1)
        
        # Handle tail bin if present
        if self.tail_bin is not None:
            tail_proportion = self.tail_bin['n_alive'] / self.n_alive[0]
            # Survival at tail age
            s_tail = tail_proportion
            # Samples that fall below this survival value are censored at tail age
            tail_mask = u < s_tail
        else:
            tail_mask = np.zeros(n, dtype=bool)
            s_tail = 0
        
        death_times = np.zeros(n)
        events = np.ones(n)
        
        # For tail bin samples, assign to tail age with event=0
        if np.any(tail_mask):
            death_times[tail_mask] = self.tail_bin['age']
            events[tail_mask] = 0
        
        # For non-tail samples, use inverse transform sampling
        non_tail_mask = ~tail_mask
        if np.any(non_tail_mask):
            # Interpolate to find ages corresponding to survival probabilities
            # Note: survival is decreasing, so we need to reverse for interpolation
            death_times[non_tail_mask] = np.interp(u[non_tail_mask], s[::-1], t[::-1])
        
        return death_times, events
    
    def _death_times_distribution_from_survival(self, bins):
        """
        Calculate death times distribution from survival curve.
        
        Parameters:
        -----------
        bins : array-like
            Bin edges for the distribution
        
        Returns:
        --------
        prob_death : ndarray
            Probability of death in each bin
        """
        t, s = self.survival
        
        # For each bin, calculate the probability of death
        prob_death = np.zeros(len(bins) - 1)
        
        for i in range(len(bins) - 1):
            t1, t2 = bins[i], bins[i+1]
            # Interpolate survival at bin edges
            s1 = np.interp(t1, t, s)
            s2 = np.interp(t2, t, s)
            # Probability of death in this interval
            prob_death[i] = s1 - s2
        
        return prob_death
    
    # =========================================================================
    # Override methods with custom implementations
    # =========================================================================
    
    def getDeathTimes(self, n=None):
        """
        Sample death times from the survival distribution.
        
        Parameters:
        -----------
        n : int, optional
            Number of death times to sample. If None, uses self.npeople
        
        Returns:
        --------
        death_times : ndarray
            Sampled death times
        """
        if n is None:
            n = self.npeople
        death_times, _ = self._sample_death_times_from_survival(n)
        return death_times
    
    def sample(self, n):
        """
        Sample n individuals from the life table distribution.
        
        Creates a new Dataset object with sampled death times and events.
        
        Parameters:
        -----------
        n : int
            Number of samples to draw
        
        Returns:
        --------
        sampled_dataset : Dataset
            New Dataset object with sampled data
        """
        death_times, events = self._sample_death_times_from_survival(n)
        
        # Create properties if they exist
        if self.properties is not None:
            # For sampled data, we can't meaningfully sample properties
            s_properties = None
        else:
            s_properties = None
        
        # Return a regular Dataset (not Life_table) since we now have individual data
        return dtds.Dataset(death_times, events, external_hazard=self.external_hazard, 
                           bandwidth=self.bandwidth, properties=s_properties, data_dt=self.data_dt)
    
    def get_death_times_distribution(self, bins=None, dt=1, time_range=None):
        """
        Calculate death times distribution from the survival curve.
        
        Parameters:
        -----------
        bins : array-like, optional
            Bin edges for the distribution. If None, created from time_range or ages
        dt : float, optional
            Bin width if bins is None. Default: 1
        time_range : tuple, optional
            (min, max) time range for bins
        
        Returns:
        --------
        prob_death : ndarray
            Probability of death in each bin
        bin_edges : ndarray
            Edges of the bins
        """
        if bins is None:
            if time_range is not None:
                bins = np.arange(time_range[0], time_range[1] + dt, dt)
            else:
                # Use ages as reference
                max_age = self.ages[-1] if self.tail_bin is None else self.tail_bin['age']
                bins = np.arange(0, max_age + dt, dt)
        
        bins = np.array(bins)
        prob_death = self._death_times_distribution_from_survival(bins)
        
        # Handle tail bin if present and if last bin extends beyond last age
        if self.tail_bin is not None and bins[-1] >= self.tail_bin['age']:
            # Add tail bin probability to the last bin
            tail_prob = self.tail_bin['n_alive'] / self.n_alive[0]
            # This represents censored observations, might want to handle differently
            # For now, we'll note that these are censored at the tail age
        
        return prob_death, bins
    
    def plotDeathTimesDistribution(self, ax=None, bins=None, use_kde=False, dt=1, 
                                   time_range=None, **kwargs):
        """
        Plot the death times distribution.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        bins : array-like, optional
            Bin edges for the distribution
        use_kde : bool, optional
            Not used for life tables (kept for compatibility). Default: False
        dt : float, optional
            Bin width. Default: 1
        time_range : tuple, optional
            (min, max) time range for plot
        **kwargs : dict
            Additional plotting arguments
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes object
        bins : ndarray
            The bin edges used
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        prob_death, bin_edges = self.get_death_times_distribution(bins=bins, dt=dt, 
                                                                   time_range=time_range)
        
        ax.step(bin_edges[:-1], prob_death, where='post', **kwargs)
        ax.set_xlabel('Death time')
        ax.set_ylabel('Probability of death')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        return ax, bin_edges
    
    def remaining_lifetime_at_age(self, age, types='median'):
        """
        Calculate remaining lifetime statistics for individuals at a given age.
        
        Uses the conditional survival distribution: S(t|age) = S(age+t) / S(age)
        
        Parameters:
        -----------
        age : float
            Age at which to calculate remaining lifetime
        types : str or list
            Statistics to calculate: 'median', 'mean', 'std', or list of these
        
        Returns:
        --------
        stat or dict
            Requested statistic or dictionary of statistics
        """
        t, s = self.survival
        
        # Get survival at the given age
        s_age = np.interp(age, t, s)
        
        if s_age <= 0:
            # No one survived to this age
            if isinstance(types, str):
                return np.nan
            else:
                return {stat: np.nan for stat in types}
        
        # Create conditional survival: S(t|age) = S(age+t) / S(age)
        # Only consider times after age
        t_remaining = t[t >= age] - age
        s_conditional = s[t >= age] / s_age
        
        def compute(stat):
            if stat == 'median':
                # Find where conditional survival crosses 0.5
                if s_conditional[-1] > 0.5:
                    return t_remaining[-1]
                idx = np.where(s_conditional <= 0.5)[0]
                if len(idx) == 0:
                    return t_remaining[-1]
                i = idx[0]
                if i == 0:
                    return t_remaining[0]
                # Interpolate
                t1, t2 = t_remaining[i-1], t_remaining[i]
                s1, s2 = s_conditional[i-1], s_conditional[i]
                median = t1 + (0.5 - s1) * (t2 - t1) / (s2 - s1)
                return median
            
            elif stat == 'mean':
                # Mean remaining lifetime: integral of S(t|age) dt
                # Use trapezoidal rule
                mean = np.trapz(s_conditional, t_remaining)
                return mean
            
            elif stat == 'std':
                # For std, we need E[T^2] - E[T]^2
                # E[T] is the mean calculated above
                mean = np.trapz(s_conditional, t_remaining)
                # E[T^2] = 2 * integral(t * S(t) dt)
                # We can approximate this
                t2_times_s = 2 * t_remaining * s_conditional
                e_t2 = np.trapz(t2_times_s, t_remaining)
                variance = e_t2 - mean**2
                return np.sqrt(max(0, variance))
            
            elif stat == 'distribution':
                return (t_remaining, s_conditional)
            
            else:
                raise ValueError(f"Unknown type '{stat}' for remaining_lifetime_at_age")
        
        if isinstance(types, str):
            return compute(types)
        else:
            return {stat: compute(stat) for stat in types}
    
    def getSteepness(self, method='IQR'):
        """
        Calculate steepness of the survival function.
        
        Only IQR method is supported for life tables.
        
        Parameters:
        -----------
        method : str
            Method to use. Only 'IQR' is supported
        
        Returns:
        --------
        steepness : float
            Steepness measure: -median / (Q3 - Q1)
        """
        if method != 'IQR':
            raise NotImplementedError(
                f"Method '{method}' not implemented for Life_table. Only 'IQR' is supported.")
        
        t, s = self.survival
        
        # Find Q1 (where survival = 0.75)
        q1 = np.interp(0.75, s[::-1], t[::-1])
        
        # Find Q3 (where survival = 0.25)
        q3 = np.interp(0.25, s[::-1], t[::-1])
        
        # Median
        median = self.median_lifetime
        
        return -median / (q3 - q1)
    
    def getMaxLifetime(self):
        """
        Get maximum lifetime from the life table.
        
        Returns:
        --------
        max_lifetime : float
            Maximum age in the table, or tail_bin age if present
        """
        if self.tail_bin is not None:
            return self.tail_bin['age']
        else:
            return self.ages[-1]
    
    def toCsv(self, file_name, properties=False, as_life_table=False, dt=0.5):
        """
        Save life table data to CSV file.

        Parameters:
        -----------
        file_name : str
            Name of the CSV file
        properties : bool
            Whether to include properties (only used when as_life_table=False)
        as_life_table : bool
            If True, exports as life table (ages, n_alive, hazard)
            If False, generates and exports sampled death times
        dt : float
            Bin size for exported death times (as regular dataset); times are rounded down to nearest multiple of dt
        """
        if as_life_table:
            # Export as life table
            data = {
                'age': self.ages,
                'n_alive': self.n_alive,
                'survival': self.n_alive / self.n_alive[0],
                'hazard': self.hazard[1]
            }

            # Add tail bin as a final row if present
            if self.tail_bin is not None:
                tail_data = {
                    'age': [self.tail_bin['age']],
                    'n_alive': [self.tail_bin['n_alive']],
                    'survival': [self.tail_bin['n_alive'] / self.n_alive[0]],
                    'hazard': [np.nan]  # No hazard for censored tail
                }
                df_main = pd.DataFrame(data)
                df_tail = pd.DataFrame(tail_data)
                df = pd.concat([df_main, df_tail], ignore_index=True)
            else:
                df = pd.DataFrame(data)

            df.to_csv(file_name, index=False)
        else:
            # Generate death times and export as regular dataset
            death_times, events = self._sample_death_times_from_survival(self.npeople)
            
            # Bin all death times to nearest lower multiple of dt (e.g. 3.23→3.0 if dt=1)
            if dt > 0:
                binned_death_times = np.floor(death_times / dt) * dt
            else:
                binned_death_times = death_times  # If dt=0 or negative, skip binning

            data = {'death times': binned_death_times, 'events': events}

            if properties and self.properties is not None:
                data.update(self.properties)

            df = pd.DataFrame(data)
            df.to_csv(file_name, index=False)
    
    # =========================================================================
    # Class methods for loading from files
    # =========================================================================
    
    @classmethod
    def from_file(cls, filepath, year=None, age_col='Age', lx_col='lx', 
                  mx_col='mx', year_col='Year', npeople=None, delimiter=None,
                  skip_rows=None, properties=None):
        """
        Load Life_table from a file (CSV or TXT format).
        
        Supports both single-cohort and multi-cohort life table files.
        Auto-detects file format, delimiter, and header structure.
        
        Parameters:
        -----------
        filepath : str
            Path to life table file
        year : int, str, or None
            - int: Load specific year/cohort from multi-cohort file
            - 'all': Load all cohorts (returns list of Life_table objects)
            - None: Load single cohort (raises error if multi-cohort)
        age_col : str
            Column name for age values (default: 'Age')
        lx_col : str
            Column name for survivors (lx) (default: 'lx')
        mx_col : str or None
            Column name for mortality rate (default: 'mx')
            If None or not found, hazard will be derived from lx
        year_col : str
            Column name for year/cohort identifier (default: 'Year')
        npeople : int or None
            Override cohort size (if None, uses lx[0])
        delimiter : str or None
            Column delimiter (auto-detect if None)
        skip_rows : int or None
            Number of metadata rows to skip before column names
            (auto-detect if None)
        properties : dict or None
            Additional metadata to store
            
        Returns:
        --------
        Life_table or list of Life_table
            Single Life_table object, or list if year='all'
            
        Examples:
        ---------
        >>> # Load single cohort from CSV
        >>> lt = Life_table.from_file('cats.csv')
        
        >>> # Load specific year from multi-cohort file
        >>> lt = Life_table.from_file('fltcoh_1x1_denmark.txt', year=1835)
        
        >>> # Load all cohorts
        >>> lt_list = Life_table.from_file('fltcoh_1x1_denmark.txt', year='all')
        
        >>> # Custom column names
        >>> lt = Life_table.from_file('data.txt', age_col='age', lx_col='l(x)')
        """
        # Auto-detect skip_rows if not provided
        if skip_rows is None:
            skip_rows = cls._detect_skip_rows(filepath)
        
        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = cls._detect_delimiter(filepath)
        
        # Read the file with pandas
        if delimiter == ',':
            df = pd.read_csv(filepath, skiprows=skip_rows)
        else:
            df = pd.read_csv(filepath, sep=r'\s+', skiprows=skip_rows)
        
        # Clean column names (strip whitespace and quotes)
        df.columns = df.columns.str.strip().str.strip('"\'')
        
        # Find columns using case-insensitive matching
        age_col_actual = cls._find_column(df, age_col, required=True)
        lx_col_actual = cls._find_column(df, lx_col, required=True)
        mx_col_actual = cls._find_column(df, mx_col, required=False) if mx_col else None
        year_col_actual = cls._find_column(df, year_col, required=False)
        
        # Check if multi-cohort file
        is_multi_cohort = year_col_actual is not None
        
        # Handle year parameter
        if is_multi_cohort:
            if year is None:
                raise ValueError(
                    f"File contains multiple cohorts (years). "
                    f"Please specify year=<year> or year='all'"
                )
            elif year == 'all':
                # Return list of Life_table objects for all years
                years = sorted(df[year_col_actual].unique())
                return [
                    cls._create_from_dataframe(
                        df[df[year_col_actual] == y],
                        age_col_actual, lx_col_actual, mx_col_actual,
                        npeople, properties={'year': y, **(properties or {})}
                    )
                    for y in years
                ]
            else:
                # Filter for specific year
                df_year = df[df[year_col_actual] == year]
                if len(df_year) == 0:
                    available_years = sorted(df[year_col_actual].unique())
                    raise ValueError(
                        f"Year {year} not found in file. "
                        f"Available years: {available_years}"
                    )
                df = df_year
                if properties is None:
                    properties = {}
                properties['year'] = year
        else:
            if year is not None and year != 'all':
                raise ValueError(
                    "Year parameter specified but file contains single cohort"
                )
        
        # Create Life_table from dataframe
        return cls._create_from_dataframe(
            df, age_col_actual, lx_col_actual, mx_col_actual, npeople, properties
        )
    
    @classmethod
    def _detect_skip_rows(cls, filepath):
        """Auto-detect how many rows to skip before column names."""
        with open(filepath, 'r') as f:
            lines = [f.readline() for _ in range(5)]
        
        if not lines:
            return 0
        
        # Check if first line contains metadata keywords
        first_line_lower = lines[0].lower()
        has_metadata = any(
            keyword in first_line_lower 
            for keyword in ['life table', 'mortality', 'country', 
                          'denmark', 'sweden', 'females', 'males', 'cohort']
        )
        
        if has_metadata:
            # Check if second line is blank
            if len(lines) > 1 and lines[1].strip() == '':
                return 2  # Skip metadata + blank line
            else:
                return 1  # Skip just metadata line
        
        # First line should be column names
        return 0
    
    @classmethod
    def _detect_delimiter(cls, filepath):
        """Auto-detect delimiter (comma for CSV, whitespace for TXT)."""
        ext = filepath.lower().split('.')[-1]
        if ext == 'csv':
            return ','
        else:
            # For TXT files, use whitespace
            return None  # pandas will use delim_whitespace
    
    @classmethod
    def _find_column(cls, df, col_name, required=True):
        """
        Find column in dataframe using case-insensitive matching.
        
        Handles variations like 'lx', 'l(x)', '"lx"', etc.
        """
        if col_name is None:
            return None
        
        # Try exact match first
        if col_name in df.columns:
            return col_name
        
        # Try case-insensitive match
        col_name_lower = col_name.lower()
        for col in df.columns:
            if col.lower() == col_name_lower:
                return col
        
        # Try without parentheses and spaces
        col_name_stripped = col_name.lower().replace('(', '').replace(')', '').replace(' ', '')
        for col in df.columns:
            col_stripped = col.lower().replace('(', '').replace(')', '').replace(' ', '')
            if col_stripped == col_name_stripped:
                return col
        
        if required:
            raise ValueError(
                f"Required column '{col_name}' not found in file. "
                f"Available columns: {list(df.columns)}"
            )
        
        return None
    
    @classmethod
    def _create_from_dataframe(cls, df, age_col, lx_col, mx_col, npeople, properties):
        """Create Life_table object from a dataframe."""
        # Extract data
        ages = df[age_col].values
        n_alive = df[lx_col].values
        
        # Extract hazard/mortality if available
        hazard_values = None
        if mx_col is not None:
            hazard_values = df[mx_col].values
        
        # Handle special age bins (e.g., "17+")
        # Check if last age is a range or has '+'
        tail_bin = None
        if isinstance(ages[-1], str) and ('+' in str(ages[-1]) or '-' in str(ages[-1])):
            # Parse tail bin
            tail_age_str = str(ages[-1])
            if '+' in tail_age_str:
                tail_age = float(tail_age_str.replace('+', ''))
            else:
                # Range like "17-20", take the start
                tail_age = float(tail_age_str.split('-')[0])
            
            tail_bin = {'age': tail_age, 'n_alive': int(n_alive[-1])}
            
            # Remove tail from main arrays
            ages = ages[:-1]
            n_alive = n_alive[:-1]
            if hazard_values is not None:
                hazard_values = hazard_values[:-1]
        
        # Convert ages to numeric
        ages = np.array([float(a) for a in ages])
        n_alive = np.array([float(na) for na in n_alive])
        if hazard_values is not None:
            # Handle missing values (., NA, nan, etc.)
            clean_hazard = []
            for h in hazard_values:
                try:
                    clean_hazard.append(float(h))
                except (ValueError, TypeError):
                    # If conversion fails, use NaN
                    clean_hazard.append(np.nan)
            hazard_values = np.array(clean_hazard)
            
            # If all values are NaN or too many missing, don't use hazard
            if np.all(np.isnan(hazard_values)) or np.sum(np.isnan(hazard_values)) > len(hazard_values) * 0.5:
                hazard_values = None
        
        # Create Life_table object
        return cls(
            ages=ages,
            n_alive=n_alive,
            hazard_values=hazard_values,
            tail_bin=tail_bin,
            npeople=npeople,
            properties=properties
        )
    
    # =========================================================================
    # Overridden plotting methods (parent uses kmf/naf which are None here)
    # =========================================================================
    
    def plotSurvival(self, ax=None, time_range=None, **kwargs):
        """
        Plot the survival function for life table data.
        
        Overrides parent method to work without kmf object.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates new figure
        time_range : tuple, optional
            (min, max) time range for the plot
        **kwargs : dict
            Additional keyword arguments for the plot
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        t, s = self.getSurvival(time_range=time_range)
        ax.plot(t, s, **kwargs)
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Survival')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def plotScaledSurvival(self, ax=None, time_range=None, CI=True, **kwargs):
        """
        Plot the scaled survival function (normalized by median lifetime).
        
        Overrides parent method to work without kmf object.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates new figure
        time_range : tuple, optional
            (min, max) time range for the plot
        CI : bool
            If True, plot confidence interval (note: zeros for life table)
        **kwargs : dict
            Additional keyword arguments for the plot
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        t, s = self.getSurvival(time_range=time_range)
        
        # Get median time
        if time_range is None:
            median_time = self.median_lifetime
        else:
            median_time = t[np.argmin(np.abs(s - 0.5))]
        
        ax.plot(t / median_time, s, **kwargs)
        
        # Add confidence interval (zeros for life table, but included for compatibility)
        if CI:
            bottom, top = self.getSurvivalCI(time_range)
            if 'color' in kwargs:
                ax.fill_between(t / median_time, bottom, top, alpha=0.3, color=kwargs['color'])
            else:
                ax.fill_between(t / median_time, bottom, top, alpha=0.3)
        
        ax.set_xlabel('Scaled Age (Age / Median Lifetime)')
        ax.set_ylabel('Survival')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def plotHazard(self, ax=None, **kwargs):
        """
        Plot the hazard function for life table data.
        
        Overrides parent method to work without naf object.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates new figure
        **kwargs : dict
            Additional keyword arguments for the plot
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        t, h = self.hazard
        ax.plot(t, h, **kwargs)
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Hazard')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def plotScaledHazard(self, ax=None, scale=None, clean_plot=True, clean_at=1e-6, **kwargs):
        """
        Plot the scaled hazard function (normalized by median lifetime).
        
        Overrides parent method to work without kmf object.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates new figure
        scale : float, optional
            Scale factor for the time axis. If None, uses median_lifetime
        clean_plot : bool
            If True, removes very small hazard values
        clean_at : float
            Threshold for removing small hazard values
        **kwargs : dict
            Additional keyword arguments for the plot
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        t, h = self.hazard
        
        # Use median lifetime as scale if not provided
        if scale is None:
            scale = self.median_lifetime
        
        # Clean very small values if requested
        if clean_plot:
            mask = h > clean_at
            t = t[mask]
            h = h[mask]
        
        ax.plot(t / scale, h, **kwargs)
        ax.set_yscale('log')
        ax.set_xlabel('Scaled Age (Age / Median Lifetime)')
        ax.set_ylabel('Hazard')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def getSurvivalCI(self, time_range=None):
        """
        Return 95% confidence interval for survival function.
        
        Note: Life tables don't have confidence intervals, returns zeros.
        
        Parameters:
        -----------
        time_range : tuple, optional
            (min, max) time range to trim to
        
        Returns:
        --------
        bottom, top : ndarray
            Lower and upper confidence bounds (zeros for life table)
        """
        bottom, top = self.kmf_confidence_interval
        if time_range is not None:
            t, s = self.getSurvival()
            # Trim to range
            mask = (t >= time_range[0]) & (t <= time_range[1])
            bottom = bottom[mask]
            top = top[mask]
        return bottom, top
    
    def getConfidenceInterval(self):
        """
        Return the 95% confidence interval for the Kaplan-Meier estimator.
        
        Note: Life tables don't have confidence intervals, returns zeros.
        
        Returns:
        --------
        list of ndarray
            [lower_bound, upper_bound] (zeros for life table)
        """
        return self.kmf_confidence_interval
    
    def getMedianLifetimeCI(self):
        """
        Return confidence interval for median lifetime.
        
        Note: Life tables don't have confidence intervals.
        
        Returns:
        --------
        None
        """
        if self.warnings_enabled:
            print("Warning: Confidence intervals not available for life table data")
        return None
    
    def survivalAtTimes(self, times):
        """
        Return the survival probability at given time points.
        
        Overrides parent method to work without kmf object.
        
        Parameters:
        -----------
        times : array-like or float
            Time point(s) at which to evaluate survival
        
        Returns:
        --------
        survival_prob : ndarray or float
            Survival probabilities at the given times
        """
        # Convert to array if single value
        times = np.atleast_1d(times)
        
        t, s = self.survival
        
        # Interpolate survival at requested times
        survival_values = np.interp(times, t, s, left=s[0], right=s[-1])
        
        # Return scalar if input was scalar
        if len(survival_values) == 1:
            return survival_values[0]
        else:
            return survival_values
    
    
    # =========================================================================
    # Methods that raise NotImplementedError
    # =========================================================================
    
    def plot_survival_bootstrap(self, ax=None, **kwargs):
        """Bootstrap not available for life table data."""
        raise NotImplementedError("Bootstrap not available for life table data")
    
    def subSetByProperty(self, property, value):
        """Cannot subset life table by properties."""
        raise NotImplementedError("Cannot subset life table by properties")
    
    def subsetByProperties(self, properties, values):
        """Cannot subset life table by properties."""
        raise NotImplementedError("Cannot subset life table by properties")
    
    def removeNans(self):
        """Cannot remove NaNs from aggregate life table."""
        raise NotImplementedError("Cannot remove NaNs from aggregate life table")

