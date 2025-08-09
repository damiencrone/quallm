"""
DataFrame utility functions for QualLM.

This module provides common DataFrame operations that are shared primarily between Dataset and Prediction classes.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Type


# ============================================================================
# Type Inference and Validation
# ============================================================================

def infer_column_type(series: pd.Series) -> Tuple[str, Union[Type, np.dtype, None]]:
    """
    Infer the actual type of a pandas Series, handling object dtypes.
    
    Args:
        series: A pandas Series to analyze
    
    Returns:
        A tuple of (type_name, type_class) where:
        - type_name is a string description ('int64', 'str', 'list', 'mixed', 'unknown')
        - type_class is the actual type or dtype object, or None for mixed/unknown
    
    Examples:
        >>> pd.Series([1, 2, 3]) → ('int64', dtype('int64'))
        >>> pd.Series(['a', 'b', 'c']) → ('str', <class 'str'>)
        >>> pd.Series([[1, 2], [3, 4]]) → ('list', <class 'list'>)
        >>> pd.Series([None, None]) → ('unknown', None)
        >>> pd.Series([1, 'a', 2]) → ('mixed', None)
    """
    if series.dtype != 'object':
        return str(series.dtype), series.dtype
    
    # For object dtype, check the actual type of non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'unknown', None
    
    # Check for mixed types
    types = set(type(x) for x in non_null)
    if len(types) > 1:
        return 'mixed', None
    
    # Single type found
    sample_type = types.pop()
    return sample_type.__name__, sample_type


def get_dtype_category(dtype) -> str:
    """
    Categorize a numpy/pandas dtype into broad categories.
    
    Args:
        dtype: A numpy or pandas dtype object
    
    Returns:
        A string category: 'numeric', 'temporal', 'boolean', 'categorical', 'object', or 'other'
    
    Examples:
        >>> get_dtype_category(np.dtype('int64')) → 'numeric'
        >>> get_dtype_category(np.dtype('datetime64[ns]')) → 'temporal'
        >>> get_dtype_category(pd.CategoricalDtype()) → 'categorical'
    """
    dtype_str = str(dtype)
    if 'int' in dtype_str or 'float' in dtype_str:
        return 'numeric'
    elif 'datetime' in dtype_str or 'timedelta' in dtype_str:
        return 'temporal'
    elif 'bool' in dtype_str:
        return 'boolean'
    elif dtype == 'object':
        return 'object'
    elif 'category' in dtype_str:
        return 'categorical'
    return 'other'


def is_nan_or_none(value: Any) -> bool:
    """
    Check if a value is None or NaN.
    
    Args:
        value: Any value to check
    
    Returns:
        True if the value is None or NaN, False otherwise
    
    Examples:
        >>> is_nan_or_none(None) → True
        >>> is_nan_or_none(np.nan) → True
        >>> is_nan_or_none(float('nan')) → True
        >>> is_nan_or_none(0) → False
        >>> is_nan_or_none('') → False
    """
    if value is None:
        return True
    if isinstance(value, float):
        return np.isnan(value)
    return False


def validate_series_no_nulls(series: pd.Series, field_name: str) -> None:
    """
    Validate that a series contains no null values.
    
    Args:
        series: A pandas Series to validate
        field_name: The name of the field for error messages
    
    Raises:
        ValueError: If the series contains any None or NaN values
    
    Examples:
        >>> validate_series_no_nulls(pd.Series([1, 2, 3]), 'test') → None
        >>> validate_series_no_nulls(pd.Series([1, None, 3]), 'test') → raises ValueError
    """
    null_mask = series.isna()
    if null_mask.any():
        null_count = null_mask.sum()
        first_null_idx = null_mask.idxmax()
        raise ValueError(
            f"Field '{field_name}' contains {null_count} null values. "
            f"First null at index {first_null_idx}"
        )


# ============================================================================
# Statistical Computations
# ============================================================================

def safe_describe(series: pd.Series) -> Dict[str, Optional[float]]:
    """
    Safely compute numeric statistics, handling edge cases.
    
    Args:
        series: A pandas Series to compute statistics for
    
    Returns:
        A dictionary with min, max, mean, and std values (or None if not computable)
    
    Examples:
        >>> pd.Series([1, 2, 3, 4, 5]) → {'min': 1.0, 'max': 5.0, 'mean': 3.0, 'std': 1.58}
        >>> pd.Series([None, None]) → {'min': None, 'max': None, 'mean': None, 'std': None}
        >>> pd.Series(['a', 'b']) → {'min': None, 'max': None, 'mean': None, 'std': None}
    """
    try:
        desc = series.describe()
        return {
            'min': desc.get('min'),
            'max': desc.get('max'),
            'mean': desc.get('mean'),
            'std': desc.get('std')
        }
    except:
        return {'min': None, 'max': None, 'mean': None, 'std': None}


def compute_string_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Compute statistics for string columns.
    
    Args:
        series: A pandas Series containing string values
    
    Returns:
        A dictionary with empty_count, min_length, max_length, mean_length, and median_length
    
    Example:
        >>> pd.Series(['hello', 'world', '', 'foo', None])
        → {'empty_count': 1, 'min_length': 0, 'max_length': 5, 'mean_length': 3.0, 'median_length': 3.0}
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return {
            'empty_count': 0,
            'min_length': None,
            'max_length': None,
            'mean_length': None,
            'median_length': None
        }
    
    # Convert to string type if not already (handles mixed types)
    non_null = non_null.astype(str)
    lengths = non_null.str.len()
    empty_count = (non_null == '').sum()
    
    return {
        'empty_count': int(empty_count),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'mean_length': float(lengths.mean()),
        'median_length': float(lengths.median())
    }


def compute_list_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Compute statistics for list columns.
    
    Args:
        series: A pandas Series containing list values
    
    Returns:
        A dictionary with min_length, max_length, and mean_length
    
    Example:
        >>> pd.Series([[1, 2], [3, 4, 5], [], None])
        → {'min_length': 0, 'max_length': 3, 'mean_length': 1.67}
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return {'min_length': None, 'max_length': None, 'mean_length': None}
    
    lengths = non_null.apply(len)
    return {
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'mean_length': float(lengths.mean())
    }


def safe_value_counts(series: pd.Series, dropna: bool = False) -> pd.Series:
    """
    Safely get value counts, handling empty series.
    
    Args:
        series: A pandas Series to count values for
        dropna: Whether to exclude NaN values from the counts
    
    Returns:
        A pandas Series with value counts, or empty Series if input is empty
    
    Examples:
        >>> safe_value_counts(pd.Series(['a', 'b', 'a'])) → Series({'a': 2, 'b': 1})
        >>> safe_value_counts(pd.Series([])) → Series([], dtype='int64')
    """
    if len(series) == 0:
        return pd.Series([], dtype='int64')
    try:
        return series.value_counts(dropna=dropna)
    except Exception:
        return pd.Series([], dtype='int64')


# ============================================================================
# Binning and Categorization
# ============================================================================

def safe_binning(series: pd.Series, n_bins: int, include_lowest: bool = True) -> pd.Series:
    """
    Safely bin numeric data with edge case handling.
    
    Args:
        series: A pandas Series of numeric values to bin
        n_bins: Number of bins to create
        include_lowest: Whether to include the lowest value in the first bin
    
    Returns:
        A pandas Series of categorical intervals or labels
    
    Examples:
        >>> safe_binning(pd.Series([1, 2, 3, 4, 5]), 2)
        → Series with intervals like (0.996, 3.0], (3.0, 5.0]
        >>> safe_binning(pd.Series([5, 5, 5]), 3)
        → Series with all values '[5]'
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return pd.Series([], dtype='category')
    
    min_val, max_val = non_null.min(), non_null.max()
    if min_val == max_val:
        # All values are the same
        label = f"[{min_val}]"
        return pd.Series([label if not pd.isna(v) else np.nan for v in series], dtype='category')
    
    try:
        return pd.cut(series, bins=n_bins, include_lowest=include_lowest, duplicates='drop')
    except ValueError:
        # Not enough unique values for requested bins
        unique_vals = sorted(non_null.unique())
        if len(unique_vals) <= 2:
            # Just use the actual values as categories
            return series.astype('category')
        return pd.cut(series, bins=len(unique_vals)-1, include_lowest=include_lowest, duplicates='drop')


def format_bin_intervals(intervals: pd.Series) -> List[str]:
    """
    Format pandas intervals as readable strings.
    
    Args:
        intervals: A pandas Series of Interval objects
    
    Returns:
        A list of formatted interval strings
    
    Examples:
        >>> intervals = pd.cut(pd.Series([1, 2, 3, 4, 5]), bins=2)
        >>> format_bin_intervals(intervals)
        → ['[1-3]', '[1-3]', '[1-3]', '[3-5]', '[3-5]']
    """
    formatted = []
    for interval in intervals:
        if pd.isna(interval):
            formatted.append("Missing")
        else:
            # Try to format as integers if possible
            left = interval.left
            right = interval.right
            if left == int(left):
                left = int(left)
            if right == int(right):
                right = int(right)
            formatted.append(f"[{left}-{right}]")
    return formatted


# ============================================================================
# DataFrame Operations
# ============================================================================

def safe_column_access(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Safely access DataFrame column with better error messages.
    
    Args:
        df: A pandas DataFrame
        column: The name of the column to access
    
    Returns:
        The requested column as a pandas Series
    
    Raises:
        KeyError: If the column doesn't exist, with helpful error message
    
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> safe_column_access(df, 'a') → Series([1, 2])
        >>> safe_column_access(df, 'c') → raises KeyError with available columns
    """
    if column not in df.columns:
        available = list(df.columns)[:5]
        suffix = "..." if len(df.columns) > 5 else ""
        raise KeyError(
            f"Column '{column}' not found. "
            f"Available columns: {available}{suffix}"
        )
    return df[column]


def combine_dataframe_rows(df: pd.DataFrame, separator: str = "\n-----\n") -> str:
    """
    Combine DataFrame rows into a formatted string.
    
    Args:
        df: A pandas DataFrame to format
        separator: String to separate rows
    
    Returns:
        A formatted string representation of all rows
    
    Example:
        >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>> combine_dataframe_rows(df)
        → "name: Alice\\nage: 25\\n-----\\nname: Bob\\nage: 30"
    """
    combined = []
    for _, row in df.iterrows():
        row_str = "\n".join(
            f"{col}: {val}"
            for col, val in row.items()
            if pd.notna(val)
        )
        combined.append(row_str)
    return separator.join(combined)


# ============================================================================
# Formatting Utilities
# ============================================================================

def format_float(value: Optional[float], decimals: int = 2) -> str:
    """
    Format float values consistently, handling None, NaN, and infinity.
    
    Args:
        value: A float value or None
        decimals: Number of decimal places to display
    
    Returns:
        A formatted string representation
    
    Examples:
        >>> format_float(3.14159) → '3.14'
        >>> format_float(None) → 'None'
        >>> format_float(np.nan) → 'NaN'
        >>> format_float(np.inf) → 'Inf'
        >>> format_float(-np.inf) → '-Inf'
        >>> format_float(0.0) → '0.00'
    """
    if value is None:
        return 'None'
    if np.isnan(value):
        return 'NaN'
    if np.isinf(value):
        return 'Inf' if value > 0 else '-Inf'
    return f"{value:.{decimals}f}"


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string and add character count if needed.
    
    Args:
        text: The string to truncate
        max_length: Maximum length before truncation
        suffix: String to append after truncation
    
    Returns:
        The original string if short enough, or truncated with character count
    
    Examples:
        >>> truncate_string("Hello", 10) → "Hello"
        >>> truncate_string("Hello World!", 5) → '"Hello..." (12 chars)'
    """
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    total_chars = len(text)
    return f'"{truncated}{suffix}" ({total_chars:,} chars)'


def format_list_preview(lst: List, max_items: int = 3) -> str:
    """
    Format list showing first few items and total count.
    
    Args:
        lst: A list to format
        max_items: Maximum number of items to show
    
    Returns:
        A string representation showing preview and count
    
    Examples:
        >>> format_list_preview([1, 2]) → "[1, 2]"
        >>> format_list_preview([1, 2, 3, 4, 5], max_items=3) → "[1, 2, 3, ...] (5 items)"
    """
    if len(lst) <= max_items:
        return repr(lst)
    
    preview_items = lst[:max_items]
    # Build the preview string manually to avoid repr quirks
    item_strs = [repr(item) for item in preview_items]
    preview_str = "[" + ", ".join(item_strs) + ", ...]"
    return f"{preview_str} ({len(lst)} items)"


# ============================================================================
# Export all utilities
# ============================================================================

__all__ = [
    # Type inference and validation
    'infer_column_type',
    'get_dtype_category',
    'is_nan_or_none',
    'validate_series_no_nulls',
    
    # Statistical computations
    'safe_describe',
    'compute_string_stats',
    'compute_list_stats',
    'safe_value_counts',
    
    # Binning and categorization
    'safe_binning',
    'format_bin_intervals',
    
    # DataFrame operations
    'safe_column_access',
    'combine_dataframe_rows',
    
    # Formatting utilities
    'format_float',
    'truncate_string',
    'format_list_preview',
]