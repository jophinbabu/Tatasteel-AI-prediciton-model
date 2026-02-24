"""
Chart Image Generator

Converts OHLCV data windows into candlestick chart images for CNN training.
Each image is a 224x224 pixel rendering of a candlestick chart with:
- Candlestick bodies and wicks
- Volume bars at the bottom
- 10-period and 20-period SMA overlays

Images are saved to labeled directories for use with tf.keras.utils.image_dataset_from_directory.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def render_candlestick_image(df_window, save_path, img_size=(224, 224)):
    """
    Renders a candlestick chart from a DataFrame window and saves as image.
    
    Args:
        df_window: DataFrame with Open, High, Low, Close, Volume columns
        save_path: Path to save the image
        img_size: Image dimensions (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.24, 2.24), dpi=100,
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     facecolor='black')
    
    n = len(df_window)
    x = np.arange(n)
    
    opens = df_window['Open'].values
    highs = df_window['High'].values
    lows = df_window['Low'].values
    closes = df_window['Close'].values
    volumes = df_window['Volume'].values
    
    # Normalize price to 0-1 for consistent visual patterns
    price_min = lows.min()
    price_max = highs.max()
    price_range = price_max - price_min if price_max > price_min else 1
    
    norm_opens = (opens - price_min) / price_range
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range
    norm_closes = (closes - price_min) / price_range
    
    # Draw candlesticks
    ax1.set_facecolor('black')
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(0, 1.05)
    
    for i in range(n):
        color = '#00ff00' if norm_closes[i] >= norm_opens[i] else '#ff0000'
        
        # Wick (high-low line)
        ax1.plot([i, i], [norm_lows[i], norm_highs[i]], color=color, linewidth=0.5)
        
        # Body
        body_bottom = min(norm_opens[i], norm_closes[i])
        body_height = abs(norm_closes[i] - norm_opens[i])
        if body_height < 0.003:
            body_height = 0.003  # Minimum visible body
        
        rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.3)
        ax1.add_patch(rect)
    
    # SMA overlays
    if n >= 10:
        sma10 = pd.Series(norm_closes).rolling(10).mean().values
        ax1.plot(x, sma10, color='#ffff00', linewidth=0.6, alpha=0.7)
    if n >= 20:
        sma20 = pd.Series(norm_closes).rolling(20).mean().values
        ax1.plot(x, sma20, color='#00ffff', linewidth=0.6, alpha=0.7)
    
    # Volume bars
    ax2.set_facecolor('black')
    ax2.set_xlim(-0.5, n - 0.5)
    vol_max = volumes.max() if volumes.max() > 0 else 1
    norm_vol = volumes / vol_max
    
    colors = ['#00ff00' if closes[i] >= opens[i] else '#ff0000' for i in range(n)]
    ax2.bar(x, norm_vol, color=colors, width=0.6, alpha=0.7)
    
    # Remove all axes, ticks, labels, borders
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.02)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='black')
    plt.close(fig)


def generate_chart_dataset(df, output_dir='data/chart_images', window_size=40,
                           label_threshold=0.005, forward_bars=4):
    """
    Generates a labeled chart image dataset from OHLCV data.
    
    For each position i in the data:
    1. Takes a window of `window_size` bars ending at i
    2. Looks at the return over the next `forward_bars`
    3. Labels: BUY (return > threshold), SELL (return < -threshold), HOLD (else)
    4. Saves the chart image to the appropriate label folder
    
    Args:
        df: DataFrame with OHLCV data
        output_dir: Root directory for images
        window_size: Number of bars per chart image
        label_threshold: Return threshold for labeling
        forward_bars: How many bars ahead to compute the return
    
    Returns:
        DataFrame with image paths and labels
    """
    label_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    # Create label directories
    for name in label_names.values():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    
    records = []
    total = len(df) - window_size - forward_bars
    
    print(f"Generating chart images: {total} images from {len(df)} bars...")
    print(f"  Window: {window_size} bars, Forward: {forward_bars} bars, Threshold: {label_threshold}")
    
    for i in range(window_size, len(df) - forward_bars):
        # Window of data
        window = df.iloc[i - window_size:i]
        
        # Forward return for label
        current_close = df['Close'].iloc[i - 1]
        future_close = df['Close'].iloc[i + forward_bars - 1]
        
        # Handle multi-index
        if hasattr(current_close, 'values'):
            current_close = current_close.values[0]
        if hasattr(future_close, 'values'):
            future_close = future_close.values[0]
        
        fwd_return = (future_close - current_close) / current_close
        
        # Label
        if fwd_return > label_threshold:
            label = 1  # BUY
        elif fwd_return < -label_threshold:
            label = 2  # SELL
        else:
            label = 0  # HOLD
        
        label_name = label_names[label]
        
        # Save image
        img_filename = f'chart_{i:05d}.png'
        img_path = os.path.join(output_dir, label_name, img_filename)
        
        render_candlestick_image(window, img_path)
        
        records.append({
            'image_path': img_path,
            'label': label,
            'label_name': label_name,
            'forward_return': fwd_return
        })
        
        # Progress
        done = i - window_size + 1
        if done % 200 == 0 or done == total:
            print(f"  {done}/{total} images generated ({done/total*100:.0f}%)")
    
    records_df = pd.DataFrame(records)
    
    # Summary
    print(f"\nDataset Summary:")
    for label_val, name in label_names.items():
        count = (records_df['label'] == label_val).sum()
        pct = count / len(records_df) * 100
        print(f"  {name}: {count} images ({pct:.1f}%)")
    
    # Save manifest
    manifest_path = os.path.join(output_dir, 'manifest.csv')
    records_df.to_csv(manifest_path, index=False)
    print(f"  Manifest saved to {manifest_path}")
    
    return records_df


if __name__ == "__main__":
    data_file = os.path.join("data", "TATASTEEL_NS_1h.csv")
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        generate_chart_dataset(df, window_size=40)
    else:
        print("Data not found. Run: python main.py --mode fetch")
