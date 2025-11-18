"""
HYGO Analysis Report Script

Standalone script containing all analysis logic from the Jupyter notebook.
Can be run directly or imported into a notebook for modular usage.

Usage:
    Direct execution: python notebooks/report_script.py
    Import: from report_script import *
"""

import sys
from pathlib import Path
import json
import warnings

# Data analysis libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Image handling
from PIL import Image, ImageDraw, ImageFont

# Database and analysis
from sqlalchemy import func

# Configure matplotlib for non-interactive backend when run as script
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision_pricing import (
    build_vision_price_table,
    get_cheapest_vision_models,
    get_model_pricing
)
from src.database import DatabaseService, Dataset, OriginalImage, ModelTest, ImageAnalysis, CellResult
from src.config import load_config

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# Configuration and Initialization
# ============================================================================

def setup_environment():
    """
    Set up the analysis environment.
    
    Returns:
        tuple: (config dict, DatabaseService instance, Path to project root)
    """
    # Find .env file - prefer project root
    env_candidate = project_root / ".env"
    if not env_candidate.exists():
        env_candidate = Path.cwd() / ".env"
    
    # Load configuration
    try:
        config = load_config(env_path=str(env_candidate))
        print(f"✅ Configuration loaded from: {env_candidate}")
    except FileNotFoundError:
        print(f"⚠️ Configuration file not found at: {env_candidate}")
        print("Some features will be disabled. Please create .env file with OPENROUTER_API_KEY.")
        config = {
            "api_key": None,
            "model": None,
            "referer": "http://localhost",
            "app_title": "hygo-vision-agent",
            "test_all": False,
            "db_path": "hygo_results.db",
        }
    
    # Initialize database
    db_path = config.get('db_path', 'hygo_results.db')
    db_service = DatabaseService(db_path=str(project_root / db_path))
    print(f"✅ Database initialized: {project_root / db_path}")
    
    if config.get('api_key'):
        print("✅ OpenRouter API key found")
    else:
        print("⚠️ OPENROUTER_API_KEY not set - pricing features disabled")
    
    return config, db_service, env_candidate


# ============================================================================
# Part 1: Vision Model Pricing Analysis
# ============================================================================

def fetch_vision_models(config, env_candidate):
    """
    Fetch all vision-capable models from OpenRouter with pricing.
    
    Args:
        config: Configuration dictionary
        env_candidate: Path to .env file
        
    Returns:
        pandas.DataFrame: Model pricing data, or empty DataFrame if API key missing
    """
    if not config.get('api_key'):
        print("⚠️ Skipping vision model pricing - API key not configured")
        return pd.DataFrame()
    
    print("Fetching vision model pricing from OpenRouter...")
    vision_models = build_vision_price_table(use_user_filter=False, env_path=str(env_candidate))
    
    df_models = pd.DataFrame(vision_models)
    
    if not df_models.empty:
        print(f"✅ Found {len(df_models)} vision-capable models")
        print(f"Price range: ${df_models['image_price'].min():.10f} - ${df_models['image_price'].max():.10f} per image")
        
        print("\nPricing Statistics:")
        print(df_models[['image_price', 'prompt_price', 'completion_price']].describe())
    else:
        print("⚠️ No vision models found")
    
    return df_models


def get_top_n_models(df_models, n=10):
    """
    Get top N most cost-effective vision models.
    
    Args:
        df_models: DataFrame with model pricing data
        n: Number of top models to return
        
    Returns:
        pandas.DataFrame: Top N models with formatted pricing
    """
    if df_models.empty:
        return pd.DataFrame()
    
    top_n = df_models.head(n).copy()
    top_n['image_price_formatted'] = top_n['image_price'].apply(lambda x: f"${x:.10f}")
    top_n['prompt_price_formatted'] = top_n['prompt_price'].apply(lambda x: f"${x:.12f}")
    top_n['completion_price_formatted'] = top_n['completion_price'].apply(lambda x: f"${x:.12f}")
    
    display_df = top_n[['id', 'name', 'image_price_formatted', 'prompt_price_formatted', 
                         'completion_price_formatted', 'context_length']].copy()
    display_df.columns = ['Model ID', 'Name', 'Image Price', 'Prompt Price/Token', 
                          'Completion Price/Token', 'Context Length']
    
    return display_df


def plot_image_cost_distribution(df_models, save_path=None):
    """
    Create visualization of image processing costs across all models.
    
    Args:
        df_models: DataFrame with model pricing data
        save_path: Optional path to save the figure
        
    Returns:
        plotly.graph_objects.Figure or None
    """
    if df_models.empty:
        print("⚠️ No pricing data available for visualization")
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_models['id'],
        y=df_models['image_price'],
        name='Image Price',
        marker_color='lightblue',
        hovertemplate='<b>%{x}</b><br>Image: $%{y:.10f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Vision Model Image Processing Costs (Ascending Order)',
        xaxis_title='Model',
        yaxis_title='Price per Image (USD, log scale)',
        yaxis_type='log',
        height=500,
        showlegend=False,
        xaxis={'tickangle': -45}
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved plot to: {save_path}")
    
    return fig


def plot_top_models_detailed(df_models, n=20, save_path=None):
    """
    Create detailed cost comparison for top N models.
    
    Args:
        df_models: DataFrame with model pricing data
        n: Number of top models to display
        save_path: Optional path to save the figure
        
    Returns:
        plotly.graph_objects.Figure or None
    """
    if df_models.empty:
        print("⚠️ No pricing data available for visualization")
        return None
    
    top_n = df_models.head(n)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Image Price',
        x=top_n['id'],
        y=top_n['image_price'],
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        name='Prompt Price (×10000)',
        x=top_n['id'],
        y=top_n['prompt_price'] * 10000,
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.add_trace(go.Bar(
        name='Completion Price (×10000)',
        x=top_n['id'],
        y=top_n['completion_price'] * 10000,
        marker_color='rgb(50, 171, 96)'
    ))
    
    fig.update_layout(
        title=f'Top {n} Most Cost-Effective Vision Models - Detailed Pricing',
        xaxis_title='Model',
        yaxis_title='Price (USD)',
        barmode='group',
        height=600,
        xaxis={'tickangle': -45}
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved plot to: {save_path}")
    
    return fig


# ============================================================================
# Part 2: Database Analysis
# ============================================================================

def get_database_summary(db_service):
    """
    Get overall database statistics.
    
    Args:
        db_service: DatabaseService instance
        
    Returns:
        dict: Summary statistics
    """
    session = db_service.Session()
    
    try:
        num_datasets = session.query(func.count(Dataset.id)).scalar()
        num_images = session.query(func.count(OriginalImage.id)).scalar()
        num_model_tests = session.query(func.count(ModelTest.id)).scalar()
        num_analyses = session.query(func.count(ImageAnalysis.id)).scalar()
        avg_issues = session.query(func.avg(ImageAnalysis.total_issues)).scalar()
        
        summary = {
            'datasets': num_datasets,
            'images': num_images,
            'model_tests': num_model_tests,
            'analyses': num_analyses,
            'avg_issues': avg_issues
        }
        
        print("=" * 60)
        print("DATABASE SUMMARY")
        print("=" * 60)
        print(f"Total Datasets:      {num_datasets}")
        print(f"Total Images:        {num_images}")
        print(f"Model Tests Run:     {num_model_tests}")
        print(f"Total Analyses:      {num_analyses}")
        
        if avg_issues:
            print(f"\nAverage Issues per Image: {avg_issues:.2f}")
        else:
            print("\nNo analyses found")
        
        return summary
        
    finally:
        session.close()


def get_model_performance(db_service):
    """
    Get performance comparison across different models.
    
    Args:
        db_service: DatabaseService instance
        
    Returns:
        pandas.DataFrame: Model performance statistics
    """
    session = db_service.Session()
    
    try:
        model_stats = session.query(
            ModelTest.model_name,
            func.count(ImageAnalysis.id).label('num_images'),
            func.avg(ImageAnalysis.total_issues).label('avg_issues'),
            func.min(ImageAnalysis.total_issues).label('min_issues'),
            func.max(ImageAnalysis.total_issues).label('max_issues')
        ).join(ImageAnalysis).group_by(ModelTest.model_name).all()
        
        if model_stats:
            df_model_perf = pd.DataFrame(model_stats, columns=[
                'Model', 'Images Analyzed', 'Avg Issues', 'Min Issues', 'Max Issues'
            ])
            df_model_perf = df_model_perf.sort_values('Avg Issues')
            
            print("\n" + "=" * 60)
            print("MODEL PERFORMANCE COMPARISON")
            print("=" * 60)
            print(df_model_perf.to_string(index=False))
            
            return df_model_perf
        else:
            print("\nNo model performance data available")
            return pd.DataFrame()
            
    finally:
        session.close()


def get_cell_heatmap_data(db_service):
    """
    Get cell-level error distribution for heatmap.
    
    Args:
        db_service: DatabaseService instance
        
    Returns:
        numpy.ndarray: 3x3 heatmap data
    """
    session = db_service.Session()
    
    try:
        cell_stats = session.query(
            CellResult.row,
            CellResult.col,
            func.sum(CellResult.issue_count).label('total_issues')
        ).group_by(CellResult.row, CellResult.col).all()
        
        if cell_stats:
            heatmap_data = np.zeros((3, 3))
            for row, col, issues in cell_stats:
                heatmap_data[row][col] = issues
            return heatmap_data
        else:
            return None
            
    finally:
        session.close()


def plot_cell_heatmap(db_service, save_path=None):
    """
    Create heatmap showing error distribution across 3x3 grid.
    
    Args:
        db_service: DatabaseService instance
        save_path: Optional path to save the figure
    """
    heatmap_data = get_cell_heatmap_data(db_service)
    
    if heatmap_data is None:
        print("⚠️ No cell-level data available for heatmap")
        return None
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Total Issues Detected'},
                xticklabels=['Left', 'Center', 'Right'],
                yticklabels=['Top', 'Middle', 'Bottom'])
    plt.title('Error Detection Heatmap Across 3x3 Grid', fontsize=14, fontweight='bold')
    plt.xlabel('Horizontal Position', fontsize=12)
    plt.ylabel('Vertical Position', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved heatmap to: {save_path}")
    
    return plt.gcf()


# ============================================================================
# Part 3: Image Comparison
# ============================================================================

def display_image_analysis(db_service, image_filename, model_name):
    """
    Get detailed analysis for a specific image and model.
    
    Args:
        db_service: DatabaseService instance
        image_filename: Name of the image file
        model_name: Name of the model used
        
    Returns:
        dict or None: Analysis result dictionary
    """
    result = db_service.get_model_image_issues(model_name, image_filename)
    
    if not result:
        print(f"⚠️ No results found for {image_filename} with {model_name}")
        return None
    
    return result


def create_comparison_figure(results_list, image_filename, save_path=None):
    """
    Create side-by-side comparison of multiple model results.
    
    Args:
        results_list: List of result dictionaries
        image_filename: Name of the image
        save_path: Optional path to save the figure
    """
    num_models = len(results_list)
    
    fig, axes = plt.subplots(2, num_models, figsize=(6*num_models, 10))
    if num_models == 1:
        axes = axes.reshape(2, 1)
    
    for idx, result in enumerate(results_list):
        # Load images
        raw_path = Path(result['raw_image_path'])
        annotated_path = Path(result['annotated_image_path'])
        
        if raw_path.exists():
            raw_img = Image.open(raw_path)
            axes[0, idx].imshow(raw_img)
            axes[0, idx].set_title(f"Original Image\n{result['model_name']}", 
                                   fontweight='bold', fontsize=10)
            axes[0, idx].axis('off')
        
        if annotated_path.exists():
            annotated_img = Image.open(annotated_path)
            axes[1, idx].imshow(annotated_img)
            axes[1, idx].set_title(f"Annotated ({result['total_issues']} issues)\n{result['model_name']}", 
                                   fontweight='bold', fontsize=10)
            axes[1, idx].axis('off')
    
    plt.suptitle(f"Image Comparison: {image_filename}", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison to: {save_path}")
    
    # Print error details
    for result in results_list:
        print(f"\n{'='*80}")
        print(f"Model: {result['model_name']}")
        print(f"Total Issues: {result['total_issues']}")
        print(f"{'='*80}")
        
        if result['all_issues']:
            for i, issue in enumerate(result['all_issues'], 1):
                print(f"{i}. {issue}")
        else:
            print("No issues detected.")
    
    return fig


def get_available_images_and_models(db_service):
    """
    Get list of all analyzed image-model combinations.
    
    Args:
        db_service: DatabaseService instance
        
    Returns:
        tuple: (DataFrame of combinations, list of unique images, list of unique models)
    """
    session = db_service.Session()
    
    try:
        image_model_pairs = session.query(
            OriginalImage.filename,
            ModelTest.model_name
        ).join(
            ImageAnalysis, OriginalImage.id == ImageAnalysis.original_image_id
        ).join(
            ModelTest, ImageAnalysis.model_test_id == ModelTest.id
        ).distinct().all()
        
        if image_model_pairs:
            df_available = pd.DataFrame(image_model_pairs, columns=['Image', 'Model'])
            unique_images = df_available['Image'].unique()
            unique_models = df_available['Model'].unique()
            
            print("\nAvailable Image-Model Combinations:")
            print("=" * 60)
            print(df_available.to_string(index=False))
            print(f"\nUnique Images: {len(unique_images)}")
            print(f"Unique Models: {len(unique_models)}")
            
            return df_available, unique_images, unique_models
        else:
            print("\n⚠️ No analyzed images found in database")
            return pd.DataFrame(), [], []
            
    finally:
        session.close()


# ============================================================================
# Part 4: Cost-Benefit Analysis
# ============================================================================

def create_cost_benefit_analysis(df_model_perf, df_models, config, env_candidate):
    """
    Combine performance and pricing data for cost-benefit analysis.
    
    Args:
        df_model_perf: DataFrame with model performance stats
        df_models: DataFrame with model pricing data
        config: Configuration dictionary
        env_candidate: Path to .env file
        
    Returns:
        pandas.DataFrame: Cost-benefit analysis results
    """
    if df_model_perf.empty or df_models.empty:
        print("⚠️ Insufficient data for cost-benefit analysis")
        return pd.DataFrame()
    
    cost_benefit = df_model_perf.copy()
    
    # Add pricing information - get once per model to avoid duplicate API calls
    def get_price_safe(model_name):
        pricing = get_model_pricing(model_name, use_user_filter=False, env_path=str(env_candidate))
        return pricing['image_price'] if pricing else 0
    
    cost_benefit['Image Price'] = cost_benefit['Model'].apply(get_price_safe)
    
    # Calculate cost per detected issue
    cost_benefit['Cost per Issue'] = cost_benefit.apply(
        lambda row: row['Image Price'] / row['Avg Issues'] if row['Avg Issues'] > 0 else float('inf'),
        axis=1
    )
    
    # Sort by cost per issue
    cost_benefit = cost_benefit.sort_values('Cost per Issue')
    
    print("\n" + "=" * 80)
    print("COST-BENEFIT ANALYSIS")
    print("=" * 80)
    print("\nModels ranked by cost-effectiveness (cost per detected issue):\n")
    
    return cost_benefit


def plot_cost_benefit_scatter(cost_benefit, save_path=None):
    """
    Create scatter plot for cost-benefit analysis.
    
    Args:
        cost_benefit: DataFrame with cost-benefit data
        save_path: Optional path to save the figure
        
    Returns:
        plotly.graph_objects.Figure or None
    """
    if cost_benefit.empty:
        return None
    
    fig = px.scatter(
        cost_benefit,
        x='Image Price',
        y='Avg Issues',
        size='Images Analyzed',
        color='Cost per Issue',
        hover_data=['Model'],
        title='Model Cost-Benefit Analysis',
        labels={
            'Image Price': 'Cost per Image (USD)',
            'Avg Issues': 'Average Issues Detected',
            'Cost per Issue': 'Cost per Issue (USD)'
        },
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(height=600)
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved cost-benefit plot to: {save_path}")
    
    return fig


def calculate_efficiency_scores(cost_benefit, config, env_candidate):
    """
    Calculate efficiency scores for models.
    
    Score = (Avg Issues Detected) / (Image Price × 10000)
    
    Args:
        cost_benefit: DataFrame with cost-benefit data
        config: Configuration dictionary
        env_candidate: Path to .env file
        
    Returns:
        pandas.DataFrame: Efficiency scores
    """
    if cost_benefit.empty:
        return pd.DataFrame()
    
    efficiency = cost_benefit.copy()
    
    # Get numeric image price - get once per model to avoid duplicate API calls
    def get_price_numeric_safe(model_name):
        pricing = get_model_pricing(model_name, use_user_filter=False, env_path=str(env_candidate))
        return pricing['image_price'] if pricing else 0
    
    efficiency['Image Price Numeric'] = efficiency['Model'].apply(get_price_numeric_safe)
    
    # Calculate efficiency score
    efficiency['Efficiency Score'] = efficiency.apply(
        lambda row: row['Avg Issues'] / (row['Image Price Numeric'] * 10000)
        if row['Image Price Numeric'] > 0 else 0,
        axis=1
    )
    
    efficiency = efficiency.sort_values('Efficiency Score', ascending=False)
    
    return efficiency


def plot_efficiency_scores(efficiency, save_path=None):
    """
    Create bar chart of efficiency scores.
    
    Args:
        efficiency: DataFrame with efficiency scores
        save_path: Optional path to save the figure
        
    Returns:
        plotly.graph_objects.Figure or None
    """
    if efficiency.empty:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=efficiency['Model'],
            y=efficiency['Efficiency Score'],
            marker_color='lightgreen',
            text=efficiency['Efficiency Score'].round(2),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Model Efficiency Scores (Higher = Better Value)',
        xaxis_title='Model',
        yaxis_title='Efficiency Score (Issues / Cost)',
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved efficiency plot to: {save_path}")
    
    return fig


# ============================================================================
# Part 5: Dataset Distribution Analysis
# ============================================================================

def get_dataset_statistics(db_service):
    """
    Get statistics for correct vs faulty image datasets.
    
    Args:
        db_service: DatabaseService instance
        
    Returns:
        pandas.DataFrame: Dataset statistics
    """
    session = db_service.Session()
    
    try:
        dataset_stats = session.query(
            Dataset.name,
            Dataset.category,
            func.count(ImageAnalysis.id).label('num_analyses'),
            func.avg(ImageAnalysis.total_issues).label('avg_issues'),
            func.sum(ImageAnalysis.total_issues).label('total_issues')
        ).join(
            OriginalImage, Dataset.id == OriginalImage.dataset_id
        ).join(
            ImageAnalysis, OriginalImage.id == ImageAnalysis.original_image_id
        ).group_by(Dataset.name, Dataset.category).all()
        
        if dataset_stats:
            df_datasets = pd.DataFrame(dataset_stats, columns=[
                'Dataset', 'Category', 'Analyses', 'Avg Issues', 'Total Issues'
            ])
            
            print("\n" + "=" * 60)
            print("DATASET ANALYSIS")
            print("=" * 60)
            print(df_datasets.to_string(index=False))
            
            return df_datasets
        else:
            print("\n⚠️ No dataset statistics available")
            return pd.DataFrame()
            
    finally:
        session.close()


def plot_dataset_comparison(df_datasets, save_path=None):
    """
    Create comparison chart for dataset statistics.
    
    Args:
        df_datasets: DataFrame with dataset statistics
        save_path: Optional path to save the figure
        
    Returns:
        plotly.graph_objects.Figure or None
    """
    if df_datasets.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Average Issues',
        x=df_datasets['Dataset'],
        y=df_datasets['Avg Issues'],
        marker_color=['green' if cat == 'correct' else 'red'
                     for cat in df_datasets['Category']]
    ))
    
    fig.update_layout(
        title='Average Issues by Dataset (Green=Correct, Red=Faulty)',
        xaxis_title='Dataset',
        yaxis_title='Average Issues Detected',
        height=400
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved dataset comparison to: {save_path}")
    
    return fig


# ============================================================================
# Main Execution
# ============================================================================

def run_full_analysis(save_plots=False):
    """
    Run the complete analysis pipeline.
    
    Args:
        save_plots: If True, save all plots to files
        
    Returns:
        dict: Analysis results containing all dataframes and figures
    """
    print("\n" + "=" * 80)
    print("HYGO - AI Image Error Detection - Comprehensive Analysis Report")
    print("=" * 80 + "\n")
    
    # Setup
    config, db_service, env_candidate = setup_environment()
    
    results = {
        'config': config,
        'db_service': db_service,
    }
    
    # Part 1: Vision Model Pricing
    print("\n" + "=" * 80)
    print("PART 1: VISION MODEL PRICING ANALYSIS")
    print("=" * 80)
    
    df_models = fetch_vision_models(config, env_candidate)
    results['df_models'] = df_models
    
    if not df_models.empty:
        top_10 = get_top_n_models(df_models, n=10)
        results['top_10_models'] = top_10
        print("\nTop 10 Most Cost-Effective Models:")
        print(top_10.to_string(index=False))
        
        if save_plots:
            plot_image_cost_distribution(df_models, 'output/pricing_distribution.html')
            plot_top_models_detailed(df_models, n=20, save_path='output/top_models_detailed.html')
    
    # Part 2: Database Analysis
    print("\n" + "=" * 80)
    print("PART 2: DATABASE ANALYSIS")
    print("=" * 80)
    
    summary = get_database_summary(db_service)
    results['db_summary'] = summary
    
    df_model_perf = get_model_performance(db_service)
    results['df_model_perf'] = df_model_perf
    
    if save_plots:
        plot_cell_heatmap(db_service, save_path='output/cell_heatmap.png')
    
    # Part 3: Image Comparison
    print("\n" + "=" * 80)
    print("PART 3: IMAGE COMPARISON")
    print("=" * 80)
    
    df_available, unique_images, unique_models = get_available_images_and_models(db_service)
    results['df_available'] = df_available
    results['unique_images'] = unique_images
    results['unique_models'] = unique_models
    
    # Part 4: Cost-Benefit Analysis
    if not df_model_perf.empty and not df_models.empty:
        print("\n" + "=" * 80)
        print("PART 4: COST-BENEFIT ANALYSIS")
        print("=" * 80)
        
        cost_benefit = create_cost_benefit_analysis(df_model_perf, df_models, config, env_candidate)
        results['cost_benefit'] = cost_benefit
        
        if not cost_benefit.empty:
            if save_plots:
                plot_cost_benefit_scatter(cost_benefit, save_path='output/cost_benefit_scatter.html')
            
            efficiency = calculate_efficiency_scores(cost_benefit, config, env_candidate)
            results['efficiency'] = efficiency
            
            if not efficiency.empty:
                print("\nTop 5 Most Efficient Models:")
                print(efficiency[['Model', 'Avg Issues', 'Image Price', 'Efficiency Score']].head().to_string(index=False))
                
                if save_plots:
                    plot_efficiency_scores(efficiency, save_path='output/efficiency_scores.html')
    
    # Part 5: Dataset Distribution
    print("\n" + "=" * 80)
    print("PART 5: DATASET DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    df_datasets = get_dataset_statistics(db_service)
    results['df_datasets'] = df_datasets
    
    if not df_datasets.empty and save_plots:
        plot_dataset_comparison(df_datasets, save_path='output/dataset_comparison.html')
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Run analysis with plot saving enabled
    results = run_full_analysis(save_plots=True)
    print("\n✅ Full analysis complete!")
    print("📊 Check the output/ directory for saved plots")
