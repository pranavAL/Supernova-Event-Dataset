import pandas as pd
import numpy as np
import os
import sys
import argparse
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import colorsys
from collections import Counter

def load_processed_data(input_file):
    """Load and parse CSV data."""
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data with {len(df)} rows and columns: {list(df.columns)}")
        
        # Standardize column names if needed
        if 'Model_Personality' in df.columns and 'Personality' not in df.columns:
            df = df.rename(columns={'Model_Personality': 'Personality'})
        
        # Clean up personality values
        df['Personality'] = df['Personality'].str.replace(r'[^\x00-\x7F]+', '', regex=True)  # Remove non-ASCII
        df['Personality'] = df['Personality'].str.strip()
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def get_color_palette(n_colors, theme="vibrant"):
    """Generate a visually attractive color palette with n colors."""
    if theme == "vibrant":
        # Vibrant color palette
        base_colors = [
            '#FF5252',  # Vibrant Red
            '#4CAF50',  # Green
            '#2196F3',  # Blue
            '#FFC107',  # Amber
            '#9C27B0',  # Purple
            '#00BCD4',  # Cyan
            '#FF9800',  # Orange
            '#3F51B5',  # Indigo
            '#8BC34A',  # Light Green
            '#E91E63',  # Pink
            '#009688',  # Teal
            '#CDDC39',  # Lime
        ]
    elif theme == "pastel":
        # Pastel palette
        base_colors = [
            '#FFB5B5',  # Pastel Red
            '#B5FFB5',  # Pastel Green
            '#B5B5FF',  # Pastel Blue
            '#FFFFB5',  # Pastel Yellow
            '#FFB5FF',  # Pastel Magenta
            '#B5FFFF',  # Pastel Cyan
            '#FFD9B5',  # Pastel Orange
            '#D9B5FF',  # Pastel Purple
            '#B5FFD9',  # Pastel Sea Green
            '#FFB5D9',  # Pastel Pink
            '#B5D9FF',  # Pastel Light Blue
            '#D9FFB5',  # Pastel Lime
        ]
    elif theme == "professional":
        # More professional/corporate palette
        base_colors = [
            '#1F77B4',  # Dark Blue
            '#FF7F0E',  # Orange
            '#2CA02C',  # Green
            '#D62728',  # Red
            '#9467BD',  # Purple
            '#8C564B',  # Brown
            '#E377C2',  # Pink
            '#7F7F7F',  # Gray
            '#BCBD22',  # Olive
            '#17BECF',  # Cyan
            '#AA40FC',  # Bright Purple
            '#4B0082',  # Indigo
        ]
    else:
        # Default to a rainbow palette
        base_colors = []
        for i in range(n_colors):
            hue = i / n_colors
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
            base_colors.append(color)
        return base_colors
    
    # If we don't have enough colors, generate additional ones
    if n_colors > len(base_colors):
        additional_colors = []
        for i in range(n_colors - len(base_colors)):
            hue = i / (n_colors - len(base_colors))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            color = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
            additional_colors.append(color)
        return base_colors + additional_colors
    else:
        return base_colors[:n_colors]

def categorize_personalities(df):
    """Categorize personality traits into broader categories."""
    
    personality_categories = {
        "Ideological": [
            "Historically Focused", "Historically Minded", "Historian", "Historical",
            "Historically Informed", "Historically Inclined", "Historical Awareness",
            "Historically Conscious", "Historical Icon", "Patriotic", "Patriotic Resistance",
            "Patriotic Defensive", "Nationalist", "Nationalist and Divisive",
            "Patriotic Resister", "Patriotic and Foundational", "Patriotic Achievements",
            "Patriotic and Historical", "Patriotic and Achievement-Oriented",
            "Patriotic and Nationalistic", "Nationalist Independence Supporter",
            "Conservative Traditional", "Traditionalist", "Traditional Respectful"
        ],
        "Emotional": [
            "Pessimistic", "Speculative", "Catastrophe-Conscious", "Empathetic", "Worried",
            "Concerned", "Humanitarian", "Cautious Optimist", "Cautious/Stable", "Cautious Realist",
            "Cautious", "Cautious_Realist", "Nostalgic", "Memorialistic", "Emotional",
            "Fatalistic", "Dramatic", "Spiritual and Holistic", "Spiritual and Holistic Focus",
            "Tragic", "Incoherent", "Inconsistent", "Casual",
            # New Emotional additions:
            "Emotional Reflective", "Introspective", "Emotional and Reflective", "Optimistic",
            "Emotional and Trauma-Focused", "Desperate and Frustrated", "Emotional and Introspective",
            "Emotionally Reflective", "Melodramatic", "Emotional, Empathetic"
        ],
        "Strategic Achievers": [
            "Strategic", "Action-Oriented", "Visionary", "Warrior Strategist",
            "Strategic Challenger", "Pragmatic", "Practical", "Practicality-focused",
            "Determined", "Objective", "Ambitious", "Ambitious Career Focus", "Ambitious Innovator",
            "Ambitious-Conqueror", "Ambitious Achiever", "Ambitious Venturesome",
            "Achievement-Oriented", "Celebratory and Achievement-Focused", "Awards-Oriented",
            "Recognition-Oriented", "Fame-Driven", "Competitive Achievement Oriented",
            "Career-Oriented", "Career-Focused", "Fanatic Careerist", "Sports and Business Focused",
            "Celebratory", "Conqueror", "Manipulative", "Interventionist", "Aggressive",
            "Aggressive Expansionist", "Assertive Expansionist", "Assertive", "Warrior",
            "Assertive Challenger", "Confrontational", "War-focused", "Destructive Force Awareness",
            "Chaosgenic",
            # New Strategic Achievers additions:
            "Unyielding and Confident", "Manipulative and Obsessive",
            "Violent and Chaotic", "Obsessive Perfectionist", "Obsessive", "Competitive and Ambitious"
        ],
        "Creative Innovators": [
            "Pioneering", "Innovative", "Pioneering Innovator", "Visionary Innovator",
            "Creative Innovator", "Innovative Contributor", "Innovative Iconalist",
            "Innovative and Pioneering", "Entrepreneurial", "Independent Pioneer",
            "Creative", "Artistic", "Creative and Musical", "Creative and Cultural Influence",
            "Creative and Influential", "Collaborative Innovator", "Holistic Innovator",
            "Adventurous", "Adventurous Explorer", "Risk-Taking", "Risk-Taking and Reputation-Building",
            "CuriousExplorer", "Curious", "Inquisitive", "Rebellious", "Progressive",
            "Revolutionary", "Transformationalist", "Rebel", "Rebellious Innovator",
            "Rebellious and Controversial", "Rebellious and Iconic", "Transformative",
            "Controversial", "Controversial and Boundary-Pushing",
            # New Creative Innovators additions:
            "Adventure-seeking", "Adventure Seeker", "Dark Humor Enthusiast", "Suspenseful",
            "Mysterious", "Dark and Morbid", "Dark and Complex", "Rebellious and Chaotic",
            "Thrill-Seeking", "Thriller-Inclined", "Transformational", "Curious and Enquiring"
        ],
        "Observational": [
            "Realist", "Realistic", "Tragic Realist", "Observant", "Reflective",
            "Politically Aware", "Stable and Consistent", "Moralistic", "Respectful",
            "Conflict-Aware", "Conflict-Influenced",
            # New Observational additions:
            "Inattentive"
        ],
        "Influencer": [
            "Celebrity-Focused", "Pop Icon and Performer", "Pop-Culture Influencer",
            "Pop-Cultural Influencer", "Entertainment-Focused", "Celebrity-focused and Media-engaged",
            "Public-Focused", "Public Figure", "Public-Oriented and Influential",
            "Public Figure and Advocate", "Influential and Public-Facing", "Positive Influencer",
            "Influential and Legacy-Focused", "Influential and Pioneering",
            "Inspirational and Societally Impactful", "Memorialized Legend", "Revered and Revived",
            "Cultural Icon", "Culturally Influenced", "Controversial and Influential",
            "Controversial and Troubled Legacy-focused"
        ],
        "Community Support": [
            "Resilient", "Reconstructive", "Resilient and Influential", "Creative and Resilient",
            "Heroic", "Heroic Rescuer", "Heroic and Compassionate", "Inspiring Peace Advocate",
            "Defensive Supporter", "Rebirth-focused", "Reformative Growth", "Reinventive",
            "Community-Oriented", "Community-Focused", "Supportive", "Family-Oriented",
            "Familial Influence", "Familial Influence and Career Transition",
            "Romantic and Career-Focused", "Philanthropic and Legacy-Focused", "Event-Focused",
            "Activist", "Socially Conscious", "Socially Conscious Activist", "Empathetic Advocacy",
            "Destructive", "Violent and Sectarian", "Conflict-oriented",
            # New Community Support additions:
            "Healing and Growth", "Survivor-focused", "Survivalist", "Romantic Idealist",
            "Heroic Teamplayer", "Heroic Sacrifice", "Heroic/Internal Struggle",
            "Optimistic Growth", "Romantic", "Emotional Growth-focused", "Heroic Growth"
        ]
    }
    
    # Map personalities to categories
    def get_category(personality):
        for category, traits in personality_categories.items():
            if personality in traits:
                return category
        return 'Other'
    
    df['Category'] = df['Personality'].apply(get_category)
    return df, personality_categories



def create_personality_radar_chart(df, personality_categories, models, output_dir,
                                color_theme='vibrant', show_annotations=False,
                                chart_size=(1200, 1200), text_size_factor=1.5, 
                                line_width=6, dpi=300):
    """
    Create a publication-quality radar chart of personality categories.

    Parameters:
    df : pandas.DataFrame
        DataFrame with columns ['Model', 'Personality', 'Category'].
    personality_categories : dict
        Mapping of category names to descriptions.
    models : list
        List of model names to plot.
    output_dir : str
        Directory to save output files.
    color_theme : str
        Palette: 'vibrant', 'pastel', 'dark', 'colorblind'.
    show_annotations : bool
        Annotate max values (optional).
    chart_size : tuple
        Width, height in pixels.
    text_size_factor : float
        Scale factor for all text sizes.
    line_width : int
        Thickness of radar lines.
    dpi : int
        Resolution for static output.

    Returns:
    plotly.graph_objects.Figure
    """
    import os, numpy as np, colorsys
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Color palettes with high contrast
    color_palettes = {
        'vibrant': ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F'],
        'pastel': ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3'],
        'dark': ['#1A535C', '#4ECDC4', '#F7B801', '#FF6B6B', '#3A5A40', '#6B705C', '#B56576', '#6D597A'],
        'colorblind': ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00', '#999999']
    }

    def get_palette(n, theme):
        base = color_palettes.get(theme, [])
        if n <= len(base): return base[:n]
        # interpolate extra
        hsv = [(i/n, 0.8, 0.9) for i in range(n)]
        return ['#{:02x}{:02x}{:02x}'.format(*[int(c*255) for c in colorsys.hsv_to_rgb(*h)]) for h in hsv]

    categories = list(personality_categories.keys())
    categories_closed = categories + [categories[0]]
    palette = get_palette(len(models), color_theme)

    # compute normalized percentages
    data = {}
    for m in models:
        sub = df[df['Model']==m]
        sub = sub[sub['Personality']!='Error']
        if sub.empty: continue
        pct = sub['Category'].value_counts(normalize=True) * 100
        data[m] = [pct.get(c, 0) for c in categories]

    fig = make_subplots(specs=[[{'type':'polar'}]])
    max_val = max(max(vals) for vals in data.values()) + 5

    # add traces
    for i,(m,vals) in enumerate(data.items()):
        vals_closed = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=categories_closed,
            fill='toself',
            name=m,
            line=dict(color=palette[i], width=line_width),
            opacity=0.9,
            hovertemplate='%{text}',
            text=[f'{m}<br>{cat}: {v:.1f}%' for cat,v in zip(categories, vals)]+[''],
        ))
        if show_annotations:
            idx = int(np.argmax(vals))
            if vals[idx] > 10:
                angle = 2*np.pi*idx/len(categories)
                r = vals[idx] * 1.05
                fig.add_annotation(
                    x=r*np.cos(angle), y=r*np.sin(angle), text=f'{vals[idx]:.1f}%',
                    showarrow=True, arrowhead=2, arrowsize=1,
                    font=dict(size=12*text_size_factor, color=palette[i])
                )

    # layout: remove title, add axis lines
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showline=True,
                linecolor='#444444',
                gridcolor='#BBBBBB',
                gridwidth=1,
                tickfont=dict(size=14*text_size_factor),
                ticksuffix='%', range=[0,max_val]
            ),
            angularaxis=dict(
                showline=True,
                linecolor='#444444',
                gridcolor='#BBBBBB',
                gridwidth=1,
                tickfont=dict(size=16*text_size_factor),
                rotation=90, direction='clockwise'
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(font=dict(size=14*text_size_factor)),
        margin=dict(l=100, r=100, t=50, b=50),
        width=chart_size[0], height=chart_size[1], paper_bgcolor='white'
    )

    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f'{output_dir}/radar_chart.html', include_plotlyjs='cdn')
    fig.write_image(f'{output_dir}/radar_chart.png', scale=3, width=chart_size[0], height=chart_size[1], engine='kaleido')
    fig.write_image(f'{output_dir}/radar_chart.svg')
    fig.write_image(f'{output_dir}/radar_chart.pdf', width=chart_size[0], height=chart_size[1])

    return fig

def create_semantic_space_plot(df, model_embeddings, output_dir,
                               color_theme='vibrant', chart_size=(900, 800),
                               marker_size=24, text_size=16, dpi=300):
    """
    Create an enhanced 2D visualization of models in semantic space for publication.

    Legend groups by category rather than individual models and overall styling matches radar chart.
    """
    import os, numpy as np, colorsys, plotly.graph_objects as go

    # Prepare palettes
    color_palettes = {
        'vibrant': ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#8C564B','#E377C2','#7F7F7F'],
        'pastel': ['#66C2A5','#FC8D62','#8DA0CB','#E78AC3','#A6D854','#FFD92F','#E5C494','#B3B3B3'],
        'dark': ['#1A535C','#4ECDC4','#F7B801','#FF6B6B','#3A5A40','#6B705C','#B56576','#6D597A'],
        'colorblind': ['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#F0E442','#D55E00','#999999']
    }
    def get_palette(n,theme):
        base=color_palettes.get(theme,[])
        if n<=len(base): return base[:n]
        hsv=[(i/n,0.8,0.9) for i in range(n)]
        return ['#{:02x}{:02x}{:02x}'.format(*[int(c*255) for c in colorsys.hsv_to_rgb(*h)]) for h in hsv]

    # Coordinates with minimal jitter
    models = list(model_embeddings.keys())
    coords = np.array([[emb[0]+np.random.normal(0,0.003), emb[1]+np.random.normal(0,0.003)]
                       for emb in model_embeddings.values()])

    # Determine categories
    categories = []
    for m in models:
        sub = df[(df['Model']==m)&(df['Personality']!='Error')]
        categories.append(sub['Category'].mode().iloc[0] if not sub.empty else 'Unknown')
    unique_cats = list(dict.fromkeys(categories))

    # Color & symbol per category
    cat_colors = {c: get_palette(len(unique_cats), color_theme)[i] for i,c in enumerate(unique_cats)}
    symbols = ['circle','square','diamond','cross','x','triangle-up','triangle-down','hexagon']
    cat_symbols = {c: symbols[i%len(symbols)] for i,c in enumerate(unique_cats)}

    fig = go.Figure()
    # Plot models (no legend entries)
    for i,m in enumerate(models):
        fig.add_trace(go.Scatter(
            x=[coords[i,0]], y=[coords[i,1]], mode='markers+text',
            marker=dict(size=marker_size, color=cat_colors[categories[i]], symbol=cat_symbols[categories[i]],
                        line=dict(width=2,color='white')),
            text=[m], textposition='top center', textfont=dict(size=text_size,family='Arial'),
            hovertext=f"{m}<br>Category: {categories[i]}", hoverinfo='text',
            showlegend=False
        ))
    # Dummy traces for legend by category
    for c in unique_cats:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=marker_size, color=cat_colors[c], symbol=cat_symbols[c], line=dict(width=2,color='white')),
            name=c
        ))

    # Cluster ellipses & labels
    for c in unique_cats:
        idxs = [i for i,cat in enumerate(categories) if cat==c]
        if len(idxs)>1:
            xs, ys = coords[idxs,0], coords[idxs,1]
            xr = max(0.03, xs.std()*2)
            yr = max(0.03, ys.std()*2)
            theta = np.linspace(0,2*np.pi,100)
            fig.add_trace(go.Scatter(
                x=xs.mean()+xr*np.cos(theta), y=ys.mean()+yr*np.sin(theta), mode='lines',
                line=dict(color='rgba(150,150,150,0.5)',width=1), fill='toself',
                fillcolor='rgba(150,150,150,0.1)', hoverinfo='skip', showlegend=False
            ))
            fig.add_annotation(x=xs.mean(), y=ys.mean(), text=c, showarrow=False,
                               font=dict(size=text_size+2,family='Arial'),
                               bgcolor='rgba(255,255,255,0.8)', bordercolor='#DDD', borderwidth=1)

    # Layout styling
    fig.update_layout(
        xaxis=dict(title='Dimension 1', showgrid=True, zeroline=True, gridcolor='#DDD', zerolinecolor='#AAA'),
        yaxis=dict(title='Dimension 2', showgrid=True, zeroline=True, gridcolor='#DDD', zerolinecolor='#AAA'),
        legend=dict(title=dict(text='Category', font=dict(size=text_size+2)),
                    font=dict(size=text_size), orientation='h', x=0.5, y=-0.1, xanchor='center'),
        margin=dict(l=80, r=80, t=80, b=120), width=chart_size[0], height=chart_size[1],
        paper_bgcolor='white', plot_bgcolor='rgba(245,245,245,1)'
    )
    # Save
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir,'semantic_space.html'), include_plotlyjs='cdn')
    fig.write_image(os.path.join(output_dir,'semantic_space.png'), scale=3, width=chart_size[0], height=chart_size[1], engine='kaleido')
    fig.write_image(os.path.join(output_dir,'semantic_space.svg'))
    fig.write_image(os.path.join(output_dir,'semantic_space.pdf'), width=chart_size[0], height=chart_size[1])
    return fig

def create_similarity_heatmap(model_embeddings, output_dir, color_theme='vibrant'):
    """Create an enhanced heatmap of model similarities."""
    # Calculate similarity between models
    models = list(model_embeddings.keys())
    similarity_matrix = np.zeros((len(models), len(models)))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            # Calculate cosine similarity (1 - cosine distance)
            # Simple dot product similarity as a fallback
            emb1 = model_embeddings[model1]
            emb2 = model_embeddings[model2]
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = 0
            similarity_matrix[i, j] = similarity
    
    # Create the heatmap
    fig = go.Figure()
    
    # Choose colorscale based on theme
    if color_theme == 'vibrant':
        colorscale = 'Viridis'
    elif color_theme == 'pastel': 
        colorscale = 'RdBu'
    elif color_theme == 'professional':
        colorscale = 'Blues'
    else:
        colorscale = 'YlGnBu'
    
    fig.add_trace(
        go.Heatmap(
            z=similarity_matrix,
            x=models,
            y=models,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            text=np.round(similarity_matrix, 2),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Similarity",
                    font=dict(size=16, family="Roboto, Arial, sans-serif")
                ),
                thickness=15,
                len=0.7,
                tickfont=dict(size=14, family="Roboto, Arial, sans-serif")
            )
        )
    )
    
    # Customize the layout
    fig.update_layout(
        title=dict(
            text="<b>Model Similarity Matrix</b><br><span style='font-size:16px;color:#666666'>Based on personality trait embeddings</span>",
            font=dict(size=24, family="Roboto, Arial, sans-serif", color="#333333"),
            x=0.5,
            y=0.95,
        ),
        xaxis=dict(
            title=dict(
                text="Model",
                font=dict(size=16, family="Roboto, Arial, sans-serif", color="#333333")
            ),
            tickfont=dict(size=14, family="Roboto, Arial, sans-serif"),
            tickangle=45
        ),
        yaxis=dict(
            title=dict(
                text="Model",
                font=dict(size=16, family="Roboto, Arial, sans-serif", color="#333333")
            ),
            tickfont=dict(size=14, family="Roboto, Arial, sans-serif")
        ),
        margin=dict(l=80, r=80, t=120, b=100),
        paper_bgcolor="white",
        height=700,
        width=800
    )
    
    # Save the figure
    output_file = os.path.join(output_dir, "enhanced_similarity_heatmap.html")
    fig.write_html(output_file)
    
    try:
        # Try to save as image
        image_file = os.path.join(output_dir, "enhanced_similarity_heatmap.png")
        fig.write_image(image_file, scale=2)
        print(f"Similarity heatmap saved to {image_file}")
    except Exception as e:
        print(f"Could not save similarity heatmap as image: {str(e)}")
    
    print(f"Similarity heatmap saved to {output_file}")
    return fig

def create_distinctive_traits_plot(df, models, output_dir, color_theme='vibrant'):
    """Create an enhanced bar chart of distinctive personality traits."""
    # Get colors for the models
    colors = get_color_palette(len(models), theme=color_theme)
    color_map = {model: color for model, color in zip(models, colors)}
    
    # Find distinctive traits for each model
    model_distinctive_traits = {}
    for model_name in models:
        # Count traits for this model
        model_df = df[df['Model'] == model_name]
        # Skip error entries
        model_df = model_df[model_df['Personality'] != 'Error']
        if len(model_df) == 0:
            continue
            
        model_traits = Counter(model_df['Personality'])
        
        # Count traits across all models
        all_traits = Counter(df[df['Personality'] != 'Error']['Personality'])
        
        # Find traits that are more common in this model than others
        distinctive = {}
        for trait, count in model_traits.items():
            model_freq = count / len(model_df)
            overall_freq = all_traits[trait] / len(df[df['Personality'] != 'Error'])
            distinctive[trait] = model_freq / overall_freq if overall_freq > 0 else 0
            
        # Get top 3 distinctive traits
        top_distinctive = {trait: score for trait, score in sorted(distinctive.items(), key=lambda x: x[1], reverse=True)[:3]}
        model_distinctive_traits[model_name] = top_distinctive
    
    # Create the figure
    fig = go.Figure()
    
    # For each model, create a grouped bar chart
    for i, model_name in enumerate(model_distinctive_traits.keys()):
        distinctive_traits = model_distinctive_traits[model_name]
        traits = list(distinctive_traits.keys())
        scores = list(distinctive_traits.values())
        
        # For readability, truncate very long trait names
        traits_display = [trait if len(trait) < 25 else trait[:22] + '...' for trait in traits]
        
        # Create hover text with full trait names
        hover_text = [f"<b>{model_name}</b><br>Trait: {trait}<br>Distinctiveness: {score:.2f}" 
                     for trait, score in zip(traits, scores)]
        
        fig.add_trace(
            go.Bar(
                x=traits_display,
                y=scores,
                name=model_name,
                marker_color=color_map.get(model_name, colors[i % len(colors)]),
                text=[f"{score:.2f}" for score in scores],
                textposition='auto',
                hoverinfo='text',
                hovertext=hover_text
            )
        )
    
    # Customize the layout
    fig.update_layout(
        title=dict(
            text="<b>Distinctive Personality Traits by Model</b><br><span style='font-size:16px;color:#666666'>Higher scores indicate more distinctive traits</span>",
            font=dict(size=24, family="Roboto, Arial, sans-serif", color="#333333"),
            x=0.5,
            y=0.95,
        ),
        xaxis=dict(
            title=dict(
                text="Personality Trait",
                font=dict(size=16, family="Roboto, Arial, sans-serif", color="#333333")
            ),
            tickfont=dict(size=14, family="Roboto, Arial, sans-serif"),
            tickangle=30
        ),
        yaxis=dict(
            title=dict(
                text="Distinctiveness Score",
                font=dict(size=16, family="Roboto, Arial, sans-serif", color="#333333")
            ),
            tickfont=dict(size=14, family="Roboto, Arial, sans-serif"),
            gridcolor="#e0e0e0"
        ),
        barmode='group',
        bargap=0.2,
        bargroupgap=0.1,
        showlegend=True,
        legend=dict(
            title=dict(
                text="Models",
                font=dict(size=16, family="Roboto, Arial, sans-serif", color="#333333")
            ),
            font=dict(size=14, family="Roboto, Arial, sans-serif"),
            bordercolor="#e0e0e0",
            borderwidth=1,
            x=1.05,
            y=0.5
        ),
        margin=dict(l=80, r=120, t=120, b=100),
        paper_bgcolor="white",
        plot_bgcolor="rgba(250, 250, 250, 0.9)",
        height=700,
        width=900
    )
    
    # Save the figure
    output_file = os.path.join(output_dir, "enhanced_distinctive_traits.html")
    fig.write_html(output_file)
    
    try:
        # Try to save as image
        image_file = os.path.join(output_dir, "enhanced_distinctive_traits.png")
        fig.write_image(image_file, scale=2)
        print(f"Distinctive traits plot saved to {image_file}")
    except Exception as e:
        print(f"Could not save distinctive traits plot as image: {str(e)}")
    
    print(f"Distinctive traits plot saved to {output_file}")
    return fig

def create_dashboard(figures, output_dir):
    """Combine all visualizations into a single dashboard."""
    # Create a 2x2 subplot figure
    fig = make_subplots(
        rows=2, 
        cols=2,
        specs=[
            [{"type": "polar"}, {"type": "xy"}],
            [{"type": "heatmap"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "Personality Category Distribution", 
            "Models in Semantic Space",
            "Model Similarity Matrix",
            "Distinctive Personality Traits"
        ),
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )
    
    # Add traces from each figure
    # 1. Radar chart
    for trace in figures[0].data:
        fig.add_trace(trace, row=1, col=1)
    
    # 2. Semantic space
    for trace in figures[1].data:
        fig.add_trace(trace, row=1, col=2)
    
    # 3. Similarity heatmap
    for trace in figures[2].data:
        fig.add_trace(trace, row=2, col=1)
    
    # 4. Distinctive traits
    for trace in figures[3].data:
        fig.add_trace(trace, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>LLM Personality Analysis Dashboard</b>",
            font=dict(size=28, family="Roboto, Arial, sans-serif", color="#333333"),
            x=0.5,
            y=0.98,
        ),
        showlegend=True,
        legend=dict(
            title=dict(
                text="Models",
                font=dict(size=16, family="Roboto, Arial, sans-serif", color="#333333")
            ),
            font=dict(size=14, family="Roboto, Arial, sans-serif"),
            bordercolor="#e0e0e0",
            borderwidth=1,
            x=1.05,
            y=0.5
        ),
        margin=dict(l=60, r=120, t=120, b=60),
        paper_bgcolor="white",
        height=1200,
        width=1500
    )
    
    # Update axes
    fig.update_xaxes(title_text="Semantic Dimension 1", row=1, col=2)
    fig.update_yaxes(title_text="Semantic Dimension 2", row=1, col=2)
    
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Model", row=2, col=1)
    
    fig.update_xaxes(title_text="Personality Trait", row=2, col=2)
    fig.update_yaxes(title_text="Distinctiveness Score", row=2, col=2)
    
    # Save the figure
    output_file = os.path.join(output_dir, "llm_personality_dashboard.html")
    fig.write_html(output_file)
    
    try:
        # Try to save as image
        image_file = os.path.join(output_dir, "llm_personality_dashboard.png")
        fig.write_image(image_file, scale=2)
        print(f"Dashboard saved to {image_file}")
    except Exception as e:
        print(f"Could not save dashboard as image: {str(e)}")
    
    print(f"Dashboard saved to {output_file}")
    return fig

def generate_simple_model_embeddings(df, models):
    """Generate simple embeddings for models based on their personality traits."""
    # For demonstration purposes when we don't have pre-computed embeddings
    model_embeddings = {}
    
    # Get all unique personality traits
    unique_traits = df['Personality'].unique()
    trait_indices = {trait: i for i, trait in enumerate(unique_traits)}
    
    for model in models:
        model_df = df[df['Model'] == model]
        model_df = model_df[model_df['Personality'] != 'Error']
        
        if len(model_df) == 0:
            continue
        
        # Create a simple embedding based on trait frequency
        embedding = np.zeros(len(unique_traits))
        trait_counts = model_df['Personality'].value_counts(normalize=True)
        
        for trait, freq in trait_counts.items():
            if trait in trait_indices:
                embedding[trait_indices[trait]] = freq
        
        # Ensure the embedding has some minimal dimensionality
        if len(embedding) < 2:
            embedding = np.pad(embedding, (0, 2 - len(embedding)))
        
        # Add some noise to separate models with identical traits
        embedding += np.random.normal(0, 0.01, size=embedding.shape)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        model_embeddings[model] = embedding
    
    return model_embeddings

def main():
    import argparse
    import os
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze model personalities from events')
    parser.add_argument('--input', type=str, default='model_personalities.csv',
                        help='Input CSV file (default: model_personalities.csv)')
    parser.add_argument('--output', type=str, default='visualizations',
                        help='Output directory for visualizations (default: visualizations)')
    parser.add_argument('--theme', type=str, default='vibrant', choices=['vibrant', 'pastel', 'professional'],
                        help='Color theme for visualizations (default: vibrant)')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Load and process data
        df = load_processed_data(args.input)
        df, personality_categories = categorize_personalities(df)
        
        # Get list of models excluding those with only errors
        model_count = {}
        for model in df['Model'].unique():
            model_df = df[(df['Model'] == model) & (df['Personality'] != 'Error')]
            if len(model_df) > 0:
                model_count[model] = len(model_df)
        
        models = list(model_count.keys())
        
        # Generate simple embeddings (in a real scenario, these would come from your analysis)
        model_embeddings = generate_simple_model_embeddings(df, models)
        
        print(f"Generating enhanced visualizations for {len(models)} models with theme: {args.theme}")
        
        radar_fig = create_personality_radar_chart(df, personality_categories, models, output_dir=args.output, color_theme=args.theme, show_annotations=True, chart_size=(1000, 900),
                                                   text_size_factor=1.2)
        
        semantic_fig = create_semantic_space_plot(df, model_embeddings, args.output, args.theme)
        similarity_fig = create_similarity_heatmap(model_embeddings, args.output, args.theme)
        traits_fig = create_distinctive_traits_plot(df, models, args.output, args.theme)
        
        # Create the combined dashboard
        dashboard_fig = create_dashboard([radar_fig, semantic_fig, similarity_fig, traits_fig], args.output)
        
        print("\nAll visualizations complete!")
        print(f"Check the '{args.output}' directory for the results")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    