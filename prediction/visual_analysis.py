import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class VisualAnalyst:
    """
    Analyzes stock chart images using Computer Vision to detect:
    1. Trend Direction (Linear Regression on price pixels)
    2. Support/Resistance Levels
    3. Volatility (Pixel spread)
    """
    
    def __init__(self):
        self.image_size = (224, 224)
        
    def analyze_chart(self, image_path):
        """
        Analyzes a chart image and returns a sentiment score (-1.0 to 1.0).
        -1.0 = Strong Bearish
         1.0 = Strong Bullish
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return 0.0, {}
            
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return 0.0, {}
            
        # Preprocess: specific to standard candlestick charts (white/black background)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find the "line" or candles
        edges = cv2.Canny(gray, 50, 150)
        
        # Find coordinates of all edge pixels
        y_coords, x_coords = np.where(edges > 0)
        
        if len(x_coords) < 10:
            return 0.0, {"error": "No chart features detected"}
            
        # Invert y-coordinates because image origin is top-left, but charts are bottom-left
        h, w = gray.shape
        y_coords = h - y_coords
        
        # 1. Trend Detection (Linear Regression)
        # Fit a line: y = mx + c
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        
        # Normalize slope (m) to -1 to 1 range
        # A slope of 1.0 (45 degrees) is huge for pixel coordinates
        trend_strength = np.clip(m * 2, -1.0, 1.0)
        
        # 2. Volatility (Standard Deviation of residuals)
        residuals = y_coords - (m * x_coords + c)
        volatility = np.std(residuals)
        
        # 3. Recent Momentum (Last 20% of width)
        last_20_percent_idx = x_coords > (w * 0.8)
        if np.sum(last_20_percent_idx) > 10:
            x_recent = x_coords[last_20_percent_idx]
            y_recent = y_coords[last_20_percent_idx]
            A_recent = np.vstack([x_recent, np.ones(len(x_recent))]).T
            m_recent, _ = np.linalg.lstsq(A_recent, y_recent, rcond=None)[0]
            momentum = np.clip(m_recent * 2, -1.0, 1.0)
        else:
            momentum = trend_strength
            
        # Weighted Score: 60% Trend, 40% Recent Momentum
        sentiment_score = (trend_strength * 0.6) + (momentum * 0.4)
        
        details = {
            "trend_slope": round(m, 4),
            "trend_strength": round(trend_strength, 4),
            "momentum_slope": round(m_recent, 4) if 'm_recent' in locals() else 0.0,
            "volatility": round(volatility, 4),
            "sentiment": round(sentiment_score, 4)
        }
        
        return sentiment_score, details

if __name__ == "__main__":
    # Test with a dummy plot
    import matplotlib.pyplot as plt
    
    # Create valid directory if not exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    print("Generating test chart...")
    x = np.linspace(0, 100, 100)
    # Bullish pattern
    y = x * 0.5 + np.random.normal(0, 5, 100) 
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='black')
    plt.axis('off')
    path = "temp/test_chart.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    analyst = VisualAnalyst()
    score, info = analyst.analyze_chart(path)
    print(f"Test Chart (Bullish) Score: {score}")
    print(f"Details: {info}")
    
    # Bearish pattern
    y_bear = 100 - (x * 0.5) + np.random.normal(0, 5, 100)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_bear, color='black')
    plt.axis('off')
    path_bear = "temp/test_chart_bear.png"
    plt.savefig(path_bear, bbox_inches='tight')
    plt.close()
    
    score_bear, info_bear = analyst.analyze_chart(path_bear)
    print(f"\nTest Chart (Bearish) Score: {score_bear}")
    print(f"Details: {info_bear}")
    
    # Cleanup
    # os.remove(path)
    # os.remove(path_bear)
