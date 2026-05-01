import os
import io
import base64
import logging
from enum import Enum, auto
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter

# Configure logging
logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types"""
    AUTO = auto()
    LINE = auto()
    BAR = auto()
    BARH = auto()
    PIE = auto()
    AREA = auto()
    SCATTER = auto()
    HISTOGRAM = auto()
    BOXPLOT = auto()
    HEATMAP = auto()
    
    @classmethod
    def from_string(cls, chart_type: str) -> 'ChartType':
        """Convert string to ChartType enum"""
        try:
            return cls[chart_type.upper()]
        except (KeyError, AttributeError):
            return cls.AUTO

class ChartConfig:
    """Configuration for chart generation"""
    def __init__(
        self,
        width: int = 12,
        height: int = 6,
        style: str = 'default',
        dpi: int = 100,
        max_categories: int = 50,
        max_points: int = 1000,
        grid_alpha: float = 0.3,
        normalize_hist: bool = False,
        colors: Optional[List[str]] = None
    ):
        self.width = width
        self.height = height
        self.style = style
        self.dpi = dpi
        self.max_categories = max_categories
        self.max_points = max_points
        self.grid_alpha = grid_alpha
        self.normalize_hist = normalize_hist
        self.colors = colors or [
            '#2c7be5', '#1a85ff', '#00a1ff', '#00b9ff', '#00d0ff',  # Blues
            '#ff6b6b', '#ff8e8e', '#ffb3b3', '#ffd8d8',  # Reds
            '#6bff6b', '#8eff8e', '#b3ffb3', '#d8ffd8',  # Greens
            '#ffd700', '#ffdf40', '#ffe680', '#ffecb3'   # Yellows
        ]

    @property
    def figsize(self) -> Tuple[int, int]:
        return (self.width, self.height)

def generate_chart(
    data: Dict[str, Any], 
    chart_type: Union[str, ChartType] = ChartType.AUTO,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    output_dir: str = 'ui/static',
    output_format: Literal['file', 'base64', 'figure'] = 'file',
    config: Optional[ChartConfig] = None
) -> Dict[str, Any]:
    """
    Generate professional business visualizations with enhanced error handling and multiple output formats.
    
    Args:
        data: Dictionary containing 'data' (list of rows) and 'columns' (list of column names)
        chart_type: Type of chart to generate (auto-detected if not specified)
        title: Chart title (auto-generated if None)
        x_label: X-axis label (uses column name if None)
        y_label: Y-axis label (uses column name if None)
        output_dir: Directory to save chart files (used when output_format='file')
        output_format: One of 'file', 'base64', or 'figure'
        config: Chart configuration object
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating success/failure
        - path/base64/figure: Depending on output_format
        - error: Error message if success is False
    """
    config = config or ChartConfig()
    
    try:
        # Input validation
        if not isinstance(chart_type, ChartType):
            chart_type = ChartType.from_string(str(chart_type))
            
        if not data or 'data' not in data or 'columns' not in data:
            raise ValueError("Input data must contain 'data' and 'columns' keys")
            
        if not data['data']:
            raise ValueError("Empty dataset provided")
            
        # Prepare output directory if needed
        if output_format == 'file':
            os.makedirs(output_dir, exist_ok=True)
            clean_old_charts(output_dir)
        
        # Create and validate DataFrame
        df = create_dataframe(data)
        
        # Auto-detect chart type if needed
        if chart_type == ChartType.AUTO:
            chart_type = determine_chart_type(df)
            
        # Generate the chart
        fig, ax = create_chart_figure(
            df=df,
            chart_type=chart_type,
            title=title,
            x_label=x_label,
            y_label=y_label,
            config=config
        )
        
        # Handle output based on format
        if output_format == 'file':
            return save_chart_to_file(fig, output_dir)
        elif output_format == 'base64':
            return save_chart_to_base64(fig)
        else:  # 'figure'
            return {
                'success': True,
                'figure': fig,
                'chart_type': chart_type.name.lower()
            }
            
    except Exception as e:
        error_msg = f"Chart generation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'success': False,
            'error': error_msg,
            'chart_type': chart_type.name.lower() if isinstance(chart_type, ChartType) else 'unknown'
        }


def clean_old_charts(
    output_dir: str, 
    max_age_days: int = 1,
    file_pattern: str = 'chart_*.png'
) -> None:
    """
    Safely remove old chart files matching the specified pattern.
    
    Args:
        output_dir: Directory containing chart files
        max_age_days: Maximum age of files to keep (in days)
        file_pattern: Glob pattern to match chart files
    """
    try:
        if not os.path.isdir(output_dir):
            logger.warning(f"Output directory does not exist: {output_dir}")
            return
            
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for file_path in Path(output_dir).glob(file_pattern):
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff:
                    file_path.unlink()
                    logger.debug(f"Removed old chart: {file_path}")
            except (OSError, Exception) as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
                
    except Exception as e:
        logger.error(f"Chart cleanup failed: {e}", exc_info=True)


def create_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create and validate a pandas DataFrame from input data.
    
    Args:
        data: Dictionary with 'data' (list of rows) and 'columns' (list of column names)
        
    Returns:
        Processed DataFrame with proper data types
        
    Raises:
        ValueError: If data validation fails
    """
    try:
        # Basic validation
        if not isinstance(data, dict) or not all(k in data for k in ['data', 'columns']):
            raise ValueError("Input must be a dictionary with 'data' and 'columns' keys")
            
        if not data['data'] or not data['columns']:
            raise ValueError("Empty data or columns provided")
            
        if len(data['columns']) < 2:
            raise ValueError("At least 2 columns required for visualization")
            
        # Create DataFrame
        df = pd.DataFrame(data['data'], columns=data['columns'])
        
        # Convert first column to datetime if it looks like a date
        first_col = df.columns[0].lower()
        if any(x in first_col for x in ['date', 'time', 'year', 'month', 'day']):
            df[first_col] = pd.to_datetime(df[first_col], errors='ignore')
        
        # Convert numeric columns
        for col in df.columns[1:]:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Try to convert to numeric, but keep as string if not possible
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if not numeric_col.isna().all():
                df[col] = numeric_col
        
        # Drop rows with missing values in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            df = df.dropna(subset=numeric_cols)
        
        if df.empty:
            raise ValueError("No valid data points after processing")
            
        return df
        
    except Exception as e:
        logger.error(f"Failed to create DataFrame: {e}", exc_info=True)
        raise ValueError(f"Data processing error: {str(e)}") from e


def determine_chart_type(df: pd.DataFrame) -> ChartType:
    """
    Automatically determine the most appropriate chart type based on data characteristics.
    
    Args:
        df: Input DataFrame with at least 2 columns
        
    Returns:
        ChartType enum value representing the recommended chart type
    """
    # Get basic statistics about the data
    num_cols = len(df.columns)
    num_rows = len(df)
    first_col = df.columns[0].lower()
    
    # Check for time series data
    if any(term in first_col for term in ['date', 'time', 'year', 'month', 'day']):
        try:
            # If first column can be parsed as datetime
            if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) or \
               (isinstance(df.iloc[0, 0], str) and 
                any(c.isdigit() for c in str(df.iloc[0, 0]))):
                if num_rows > 7:
                    return ChartType.LINE
                else:
                    return ChartType.BAR
        except (ValueError, TypeError):
            pass
    
    # Check for categorical data with many unique values
    if df[df.columns[0]].nunique() > 10:
        return ChartType.BARH  # Horizontal bar for many categories
    
    # Check for numeric data distribution
    if num_cols >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
        if num_rows > 20:
            return ChartType.HISTOGRAM
        elif num_rows > 5:
            return ChartType.BAR
        else:
            return ChartType.PIE
    
    # Default to bar chart
    return ChartType.BAR


def create_chart_figure(
    df: pd.DataFrame,
    chart_type: ChartType,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    config: Optional[ChartConfig] = None
) -> Tuple[Figure, Axes]:
    """
    Create a matplotlib Figure and Axes with the specified chart type.
    
    Args:
        df: Input DataFrame with data to plot
        chart_type: Type of chart to create
        title: Chart title (auto-generated if None)
        x_label: X-axis label (uses column name if None)
        y_label: Y-axis label (uses column name if None)
        config: Chart configuration
        
    Returns:
        Tuple of (Figure, Axes) objects
    """
    config = config or ChartConfig()
    plt.style.use(config.style)
    
    # Set default labels if not provided
    x_label = x_label or str(df.columns[0])
    y_label = y_label or (str(df.columns[1]) if len(df.columns) > 1 else 'Value')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    try:
        # Generate the appropriate chart type
        if chart_type == ChartType.LINE:
            _create_line_chart(ax, df, x_label, y_label, config)
        elif chart_type == ChartType.BAR:
            _create_bar_chart(ax, df, x_label, y_label, False, config)
        elif chart_type == ChartType.BARH:
            _create_bar_chart(ax, df, x_label, y_label, True, config)
        elif chart_type == ChartType.PIE:
            _create_pie_chart(ax, df, x_label, y_label, config)
        elif chart_type == ChartType.AREA:
            _create_area_chart(ax, df, x_label, y_label, config)
        elif chart_type == ChartType.SCATTER:
            _create_scatter_plot(ax, df, x_label, y_label, config)
        elif chart_type == ChartType.HISTOGRAM:
            _create_histogram(ax, df, x_label, y_label, config)
        elif chart_type == ChartType.BOXPLOT:
            _create_box_plot(ax, df, x_label, y_label, config)
        elif chart_type == ChartType.HEATMAP:
            _create_heatmap(ax, df, config)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Set title and labels
        if title:
            ax.set_title(title, pad=15)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Improve layout
        plt.tight_layout()
        
        return fig, ax
        
    except Exception as e:
        plt.close(fig)
        raise ValueError(f"Failed to create {chart_type.name} chart: {str(e)}") from e


def _create_line_chart(
    ax: Axes,
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    config: ChartConfig
) -> None:
    """Create a line chart with optional markers."""
    x = df[x_label]
    
    # Plot each numeric column as a separate line
    for i, col in enumerate(df.select_dtypes(include=['number']).columns):
        if col == x_label:
            continue
            
        ax.plot(
            x,
            df[col],
            marker='o' if len(x) <= 20 else None,
            linestyle='-',
            color=config.colors[i % len(config.colors)],
            label=col,
            alpha=0.8,
            markersize=6,
            linewidth=2
        )
    
    # Format x-axis for dates
    if pd.api.types.is_datetime64_any_dtype(x):
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=config.grid_alpha)
    if len(df.select_dtypes(include=['number']).columns) > 1:
        ax.legend(loc='best')


def _create_bar_chart(
    ax: Axes,
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    horizontal: bool = False,
    config: Optional[ChartConfig] = None
) -> None:
    """Create a bar chart (vertical or horizontal)."""
    config = config or ChartConfig()
    x = df[x_label]
    
    # Limit number of categories for better readability
    if len(x) > config.max_categories:
        df = df.nlargest(config.max_categories, y_label, keep='first')
        x = df[x_label]
    
    # Get numeric columns for stacking
    numeric_cols = [col for col in df.columns if col != x_label and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for bar chart")
    
    # For single series, use simpler bar chart
    if len(numeric_cols) == 1:
        y = df[numeric_cols[0]]
        if horizontal:
            bars = ax.barh(x.astype(str), y, color=config.colors[0], alpha=0.8)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        else:
            bars = ax.bar(x.astype(str), y, color=config.colors[0], alpha=0.8)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        # For multiple series, use grouped bars
        bar_width = 0.8 / len(numeric_cols)
        x_pos = np.arange(len(x))
        
        for i, col in enumerate(numeric_cols):
            offset = (i - (len(numeric_cols) - 1) / 2) * bar_width
            if horizontal:
                bars = ax.barh(
                    x_pos + offset,
                    df[col],
                    height=bar_width * 0.9,
                    color=config.colors[i % len(config.colors)],
                    label=col,
                    alpha=0.8
                )
            else:
                bars = ax.bar(
                    x_pos + offset,
                    df[col],
                    width=bar_width * 0.9,
                    color=config.colors[i % len(config.colors)],
                    label=col,
                    alpha=0.8
                )
                ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=7)
        
        # Set x-ticks and labels
        if horizontal:
            ax.set_yticks(x_pos)
            ax.set_yticklabels(x.astype(str))
        else:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x.astype(str), rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, alpha=config.grid_alpha, axis='y' if horizontal else 'x')


def _create_pie_chart(
    ax: Axes,
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    config: ChartConfig
) -> None:
    """Create a pie chart with automatic grouping of small slices."""
    # Sort by value and take top N-1 categories
    df = df.sort_values(y_label, ascending=False).reset_index(drop=True)
    
    # Limit number of categories
    if len(df) > 10:
        other = pd.DataFrame({
            x_label: ['Other'],
            y_label: df[y_label].iloc[9:].sum()
        })
        df = pd.concat([df.head(9), other])
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        df[y_label],
        labels=df[x_label].astype(str),
        autopct='%1.1f%%',
        startangle=90,
        colors=config.colors[:len(df)],
        wedgeprops=dict(width=0.5, edgecolor='w'),
        textprops={'fontsize': 9}
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Adjust text properties
    for text in texts + autotexts:
        text.set_fontsize(9)
    
    # Add a title
    ax.set_title(f"{y_label} Distribution", pad=20, fontsize=12)


def _create_area_chart(
    ax: Axes,
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    config: ChartConfig
) -> None:
    """Create a stacked area chart."""
    # Plot each numeric column as a separate area
    numeric_cols = [col for col in df.columns if col != x_label and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for area chart")
    
    # Create stacked area plot
    x = df[x_label]
    bottom = None
    
    for i, col in enumerate(numeric_cols):
        ax.fill_between(
            x,
            bottom if bottom is not None else 0,
            df[col] + (bottom if bottom is not None else 0),
            label=col,
            color=config.colors[i % len(config.colors)],
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )
        bottom = df[col] + (bottom if bottom is not None else 0)
    
    # Format x-axis for dates
    if pd.api.types.is_datetime64_any_dtype(x):
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=config.grid_alpha)
    if len(numeric_cols) > 1:
        ax.legend(loc='upper left')


def _create_scatter_plot(
    ax: Axes,
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    config: ChartConfig
) -> None:
    """Create a scatter plot with optional regression line."""
    if len(df.columns) < 3 or not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns[1:3]):
        raise ValueError("Scatter plot requires at least 2 numeric columns")
    
    x = df[df.columns[1]]
    y = df[df.columns[2]]
    
    # Create scatter plot
    scatter = ax.scatter(
        x,
        y,
        c=df[df.columns[0]] if pd.api.types.is_numeric_dtype(df[df.columns[0]]) else None,
        cmap='viridis',
        alpha=0.7,
        edgecolors='w',
        s=100,
        linewidth=0.5
    )
    
    # Add colorbar if color-coded
    if pd.api.types.is_numeric_dtype(df[df.columns[0]]):
        plt.colorbar(scatter, ax=ax, label=df.columns[0])
    
    # Add grid
    ax.grid(True, alpha=config.grid_alpha)
    
    # Add regression line if enough points
    if len(x) > 2:
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", linewidth=1, alpha=0.7)
        except Exception as e:
            logger.debug(f"Could not add regression line: {e}")


def _create_heatmap(
    ax: Axes,
    df: pd.DataFrame,
    config: ChartConfig
) -> None:
    """Create a heatmap from the data."""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        raise ValueError("Heatmap requires at least 2 numeric columns")
    
    # Create correlation matrix
    corr = df[numeric_cols].corr()
    
    # Generate heatmap
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add title
    ax.set_title("Correlation Heatmap", pad=15, fontsize=12)


def _create_box_plot(
    ax: Axes,
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    config: ChartConfig
) -> None:
    """Create a box plot."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for box plot")
    
    # If only one numeric column, use it as y and x as categories
    if len(numeric_cols) == 1:
        y_col = numeric_cols[0]
        x = df[x_label].astype(str)
        
        # Limit number of categories
        if len(x) > config.max_categories:
            top_cats = df.groupby(x_label)[y_col].sum().nlargest(config.max_categories).index
            df = df[df[x_label].isin(top_cats)]
            x = df[x_label].astype(str)
        
        # Create boxplot
        df.boxplot(column=y_col, by=x_label, ax=ax, grid=False)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_col)
    else:
        # Multiple numeric columns, create boxplot for each
        df[numeric_cols].boxplot(ax=ax, grid=False)
    
    # Remove default title
    plt.suptitle('')
    
    # Add grid and rotate x-labels
    ax.grid(True, alpha=config.grid_alpha)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _create_histogram(
    ax: Axes,
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    config: ChartConfig
) -> None:
    """Create a histogram of the data."""
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for histogram")
    
    # Calculate optimal number of bins using Freedman-Diaconis rule
    def fd_bins(x):
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = np.std(x)  # Fallback to standard deviation if IQR is 0
            if iqr == 0:
                return 10  # Default if no variation
        bin_width = 2 * iqr / (len(x) ** (1/3))
        return int(np.ceil((x.max() - x.min()) / bin_width)) if bin_width > 0 else 10
    
    # Plot histogram for each numeric column
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        bins = min(fd_bins(data), 50)  # Cap at 50 bins
        
        ax.hist(
            data,
            bins=bins,
            alpha=0.7,
            label=col,
            color=config.colors[i % len(config.colors)],
            edgecolor='white',
            density=config.normalize_hist
        )
    
    # Add grid and legend
    ax.grid(True, alpha=config.grid_alpha)
    if len(numeric_cols) > 1:
        ax.legend(loc='best')
    
    # Add labels
    ax.set_xlabel('Value')
    ax.set_ylabel('Density' if config.normalize_hist else 'Frequency')
    ax.set_title('Distribution of Values', pad=15, fontsize=12)


def save_chart_to_file(
    fig: Figure,
    output_dir: str,
    filename: Optional[str] = None,
    format: str = 'png',
    dpi: int = 300
) -> Dict[str, Any]:
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure to save
        output_dir: Directory to save the file
        filename: Optional filename (without extension)
        format: Image format (png, jpg, svg, pdf)
        dpi: Resolution in dots per inch
        
    Returns:
        Dictionary with success status and file path
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}"
        
        # Ensure format is valid
        format = format.lower()
        if format not in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            format = 'png'
        
        # Construct full path
        filepath = os.path.join(output_dir, f"{filename}.{format}")
        
        # Save the figure (quality only for JPEG)
        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'facecolor': fig.get_facecolor(),
            'edgecolor': 'none'
        }

        # Add quality parameter only for JPEG
        if format in ['jpg', 'jpeg']:
            save_kwargs['quality'] = 95

        fig.savefig(filepath, **save_kwargs)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return {
            "success": True,
            "path": filepath,
            "filename": os.path.basename(filepath),
            "format": format,
            "dpi": dpi
        }
        
    except Exception as e:
        logger.error(f"Error saving chart to file: {str(e)}")
        plt.close(fig)
        return {
            "success": False,
            "error": str(e),
            "path": None
        }


def figure_to_base64(fig: Figure, format: str = 'png') -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure
        format: Output format (png, jpg, svg, pdf)
        
    Returns:
        Base64 encoded string of the figure
    """
    try:
        # Convert to bytes
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format=format,
            bbox_inches='tight',
            facecolor=fig.get_facecolor(),
            edgecolor='none',
            dpi=100
        )
        buf.seek(0)
        
        # Encode to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close buffer and figure
        buf.close()
        plt.close(fig)
        
        # Return data URL
        return f"data:image/{format};base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Error converting figure to base64: {str(e)}")
        plt.close(fig)
        return ""


def generate_chart(
    data: Union[pd.DataFrame, Dict, List[Dict]],
    chart_type: Union[str, ChartType] = ChartType.AUTO,
    output_format: str = 'file',
    output_dir: str = 'charts',
    filename: Optional[str] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    config: Optional[ChartConfig] = None
) -> Dict[str, Any]:
    """
    Generate a chart from the given data.
    
    Args:
        data: Input data as DataFrame, dict, or list of dicts
        chart_type: Type of chart to generate (auto-detected if not specified)
        output_format: Output format ('file', 'base64', 'figure')
        output_dir: Directory to save the chart (for 'file' output)
        filename: Optional filename (without extension)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        config: Chart configuration
        
    Returns:
        Dictionary with chart data and metadata
    """
    try:
        # Process input data
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict) and 'data' in data and 'columns' in data:
            df = pd.DataFrame(data['data'], columns=data['columns'])
        elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported data format. Expected DataFrame, dict with 'data' and 'columns', or list of dicts")
        
        # Auto-detect chart type if not specified
        if chart_type == ChartType.AUTO:
            chart_type = determine_chart_type(df)
        elif isinstance(chart_type, str):
            chart_type = ChartType[chart_type.upper()]
        
        # Create the chart figure
        fig, ax = create_chart_figure(df, chart_type, title, x_label, y_label, config)
        
        # Handle different output formats
        if output_format == 'figure':
            return {
                "success": True,
                "chart_type": chart_type.name,
                "figure": fig,
                "axes": ax
            }
        elif output_format == 'base64':
            img_data = figure_to_base64(fig)
            return {
                "success": True,
                "chart_type": chart_type.name,
                "data": img_data,
                "format": 'base64'
            }
        else:  # 'file' or default
            result = save_chart_to_file(fig, output_dir, filename)
            result["chart_type"] = chart_type.name
            return result
            
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "chart_type": str(chart_type) if chart_type else None
        }