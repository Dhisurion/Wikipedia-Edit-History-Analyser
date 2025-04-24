import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import json
import difflib
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import os
import time
import base64

st.set_page_config(layout="wide", page_title="Wikipedia Edit Histories")

# Sidebar logo or title
st.sidebar.title("Wikipedia Analysis")

# Database path
DB_PATH = "wikipedia_analysis.db"

# Helper functions
@st.cache_data
def load_tagged_revisions():
    """Loads tagged revisions from the database"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM tagged_revisions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Convert JSON strings to lists
    df['tags_list'] = df['tags'].apply(json.loads)
    return df

@st.cache_data
def load_sentence_changes():
    """Loads sentence changes from the database"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM sentence_changes"
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Convert JSON strings to lists
    df['tags_list'] = df['tags'].apply(json.loads)
    return df

@st.cache_data
def get_tag_statistics():
    """Calculates tag statistics"""
    df = load_tagged_revisions()
    all_tags = []
    for tags in df['tags_list']:
        all_tags.extend(tags)
    
    tag_counts = pd.Series(all_tags).value_counts().reset_index()
    tag_counts.columns = ['tag', 'count']
    return tag_counts

# Function to create a download link
def get_download_link(df, filename, text):
    """Creates a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}"> {text}</a>'
    return href

# Main title
st.title("Wikipedia Edit History Analysis")
st.markdown("""
This application visualizes edit histories extracted from Wikipedia dumps,
with special focus on edit tags and NPOV (Neutral Point of View).
""")

# Load data
try:
    revisions_df = load_tagged_revisions()
    changes_df = load_sentence_changes()
    tag_stats = get_tag_statistics()
    
    # Data is loaded
    st.success(f"Data successfully loaded: {len(revisions_df)} revisions and {len(changes_df)} sentence changes")
    
    # Sidebar for filters
    st.sidebar.header("Filters")
    
    # Article filter
    all_articles = revisions_df['page_title'].unique()
    selected_article = st.sidebar.selectbox("Select Article", ["All Articles"] + list(all_articles))
    
    # Time period filter
    min_date = pd.to_datetime(revisions_df['timestamp']).min().date()
    max_date = pd.to_datetime(revisions_df['timestamp']).max().date()
    date_range = st.sidebar.date_input(
        "Time Period",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Tag filter
    top_tags = tag_stats.head(50)['tag'].tolist()
    selected_tags = st.sidebar.multiselect("Filter Tags", top_tags)
    
    # Filter data based on selection
    filtered_revisions = revisions_df.copy()
    filtered_changes = changes_df.copy()
    
    if selected_article != "All Articles":
        filtered_revisions = filtered_revisions[filtered_revisions['page_title'] == selected_article]
        filtered_changes = filtered_changes[filtered_changes['page_title'] == selected_article]
    
    if len(date_range) == 2:
        filtered_revisions = filtered_revisions[
            (pd.to_datetime(filtered_revisions['timestamp']).dt.date >= date_range[0]) &
            (pd.to_datetime(filtered_revisions['timestamp']).dt.date <= date_range[1])
        ]
        filtered_changes = filtered_changes[
            (pd.to_datetime(filtered_changes['timestamp']).dt.date >= date_range[0]) &
            (pd.to_datetime(filtered_changes['timestamp']).dt.date <= date_range[1])
        ]
    
    if selected_tags:
        # Filter for selected tags
        mask = filtered_revisions['tags_list'].apply(lambda x: any(tag in x for tag in selected_tags))
        filtered_revisions = filtered_revisions[mask]
        
        mask = filtered_changes['tags_list'].apply(lambda x: any(tag in x for tag in selected_tags))
        filtered_changes = filtered_changes[mask]
    
    # Tabs for different analyses
    # Add this before the tab definition:
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # First tab by default

    # Function to change the active tab
    def change_tab(tab_index):
        st.session_state.active_tab = tab_index

    tab_names = ["Tag Analysis", "Sentence Changes", "NPOV Analysis", "Text Explorer", "Advanced Analysis"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        if st.session_state.active_tab == 0:
            st.header("Analysis of Edit Tags")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top-Tags bar chart
                if not filtered_revisions.empty:
                    all_tags = []
                    for tags in filtered_revisions['tags_list']:
                        all_tags.extend(tags)
                    
                    if all_tags:
                        tag_counts = pd.Series(all_tags).value_counts().head(10)
                        
                        fig = px.bar(
                            x=tag_counts.values, 
                            y=tag_counts.index, 
                            orientation='h',
                            title="Top 10 Tags"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download link for tag statistics
                        tag_df = pd.DataFrame({'tag': tag_counts.index, 'count': tag_counts.values})
                        st.markdown(get_download_link(tag_df, "tag_counts.csv", "Download tag statistics"), unsafe_allow_html=True)
                    else:
                        st.info("No tags found in the filtered data.")
                else:
                    st.info("No revisions found with the selected filters.")
            
            with col2:
                # Tag network
                if not filtered_revisions.empty and len(all_tags) > 0:
                    # Create tag co-occurrence matrix
                    tag_pairs = []
                    for tags in filtered_revisions['tags_list']:
                        if len(tags) > 1:
                            for i, tag1 in enumerate(tags):
                                for tag2 in tags[i+1:]:
                                    tag_pairs.append((tag1, tag2))
                    
                    if tag_pairs:
                        # Count frequencies
                        pair_counts = pd.Series(tag_pairs).value_counts().head(20)
                        
                        # Create network
                        G = nx.Graph()
                        
                        # Add nodes and edges
                        for (tag1, tag2), count in pair_counts.items():
                            if tag1 not in G:
                                G.add_node(tag1)
                            if tag2 not in G:
                                G.add_node(tag2)
                            G.add_edge(tag1, tag2, weight=count)
                        
                        # Visualize with NetworkX
                        pos = nx.spring_layout(G, seed=42)
                        
                        # Create Plotly figure
                        edge_x = []
                        edge_y = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')
                        
                        node_x = []
                        node_y = []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                        
                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=[node for node in G.nodes()],
                            textposition="top center",
                            hoverinfo='text',
                            marker=dict(
                                showscale=True,
                                colorscale='YlGnBu',
                                size=10,
                                colorbar=dict(
                                    thickness=15,
                                    title=dict(text='Node Degree', side='right'),
                                    xanchor='left'
                                ),
                                line_width=2))
                        
                        node_adjacencies = []
                        for node, adjacencies in enumerate(G.adjacency()):
                            node_adjacencies.append(len(adjacencies[1]))
                        
                        node_trace.marker.color = node_adjacencies
                        
                        fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Tag Network (Co-Occurrence)',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0,l=0,r=0,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough tag pairs found for a network.")
                else:
                    st.info("Not enough data for a tag network.")
            
            # Tags over time
            if not filtered_revisions.empty:
                st.subheader("Tags Over Time")
                
                # Convert timestamps to date and count tags per day
                filtered_revisions = filtered_revisions.copy()  # Create explicit copy
                filtered_revisions.loc[:, 'date'] = pd.to_datetime(filtered_revisions['timestamp']).dt.date
                
                # Create DataFrame with tag counts per day
                daily_tags = []
                for date, group in filtered_revisions.groupby('date'):
                    tags_on_day = []
                    for tags in group['tags_list']:
                        tags_on_day.extend(tags)
                    
                    tag_counts = pd.Series(tags_on_day).value_counts()
                    for tag, count in tag_counts.items():
                        daily_tags.append({'date': date, 'tag': tag, 'count': count})
                
                if daily_tags:
                    daily_tags_df = pd.DataFrame(daily_tags)
                    
                    # Show only the top-5 tags for clarity
                    top_tags = daily_tags_df.groupby('tag')['count'].sum().nlargest(5).index.tolist()
                    plot_df = daily_tags_df[daily_tags_df['tag'].isin(top_tags)]
                    
                    if not plot_df.empty:
                        fig = px.line(
                            plot_df, 
                            x='date', 
                            y='count', 
                            color='tag',
                            title="Trend of Top-5 Tags Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download link for time series data
                        st.markdown(get_download_link(daily_tags_df, "tags_time_series.csv", "Download tag time series"), unsafe_allow_html=True)
                    else:
                        st.info("Not enough temporal data for the selected tags.")
                else:
                    st.info("No temporal data for tags found.")
    
    with tabs[1]:
        if st.session_state.active_tab == 0:
            st.header("Analysis of Sentence Changes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Change type distribution
                if not filtered_changes.empty:
                    change_type_counts = filtered_changes['change_type'].value_counts()
                    
                    fig = px.pie(
                        values=change_type_counts.values,
                        names=change_type_counts.index,
                        title="Distribution of Sentence Change Types"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentence changes found with the selected filters.")
            
            with col2:
                # Sentence changes per article
                if not filtered_changes.empty:
                    changes_per_article = filtered_changes.groupby('page_title').size().sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=changes_per_article.values,
                        y=changes_per_article.index,
                        orientation='h',
                        title="Sentence Changes per Article"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentence changes found with the selected filters.")
            
            # Sentence changes over time
            if not filtered_changes.empty:
                st.subheader("Sentence Changes Over Time")
                
                # Convert timestamps to date
                filtered_changes = filtered_changes.copy()  # Create explicit copy
                filtered_changes.loc[:, 'date'] = pd.to_datetime(filtered_changes['timestamp']).dt.date
                
                # Count change types per day
                daily_changes = filtered_changes.groupby(['date', 'change_type']).size().reset_index(name='count')
                
                if not daily_changes.empty:
                    fig = px.line(
                        daily_changes,
                        x='date',
                        y='count',
                        color='change_type',
                        title="Trend of Sentence Changes Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download link for time series data
                    st.markdown(get_download_link(daily_changes, "changes_time_series.csv", "Download change time series"), unsafe_allow_html=True)
                else:
                    st.info("No temporal data for sentence changes found.")
            
            # Similarity values for modifications
            modifications = filtered_changes[filtered_changes['change_type'] == 'modification']
            if not modifications.empty:
                st.subheader("Similarity Values of Sentence Modifications")
                
                fig = px.histogram(
                    modifications,
                    x='similarity',
                    nbins=20,
                    title="Distribution of Similarity Values in Sentence Modifications"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap: Similarity values by revision date
                try:
                    # Convert to month for clearer display
                    modifications = modifications.copy()  # Create explicit copy
                    modifications.loc[:, 'month'] = pd.to_datetime(modifications['timestamp']).dt.strftime('%Y-%m')
                    
                    # Calculate mean similarity per month
                    monthly_similarity = modifications.groupby('month')['similarity'].mean().reset_index()
                    if len(monthly_similarity) > 1:
                        fig = px.bar(
                            monthly_similarity,
                            x='month',
                            y='similarity',
                            title="Average Similarity of Modifications Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not perform temporal analysis of similarity values: {e}")
        
    with tabs[2]:
        if st.session_state.active_tab == 0:
            st.header("NPOV (Neutral Point of View) Analysis")
            
            # Filter NPOV-related changes
            npov_keywords = ['npov', 'neutral', 'bias', 'pov', 'point of view']
            
            def has_npov_tag(tags):
                return any(any(kw in tag.lower() for kw in npov_keywords) for tag in tags)
            
            npov_revisions = filtered_revisions[filtered_revisions['tags_list'].apply(has_npov_tag)]
            npov_changes = filtered_changes[filtered_changes['tags_list'].apply(has_npov_tag)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # NPOV change types
                if not npov_changes.empty:
                    npov_type_counts = npov_changes['change_type'].value_counts()
                    
                    fig = px.pie(
                        values=npov_type_counts.values,
                        names=npov_type_counts.index,
                        title="NPOV-related Change Types"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No NPOV-related sentence changes found.")
            
            with col2:
                # NPOV Tags WordCloud
                if not npov_revisions.empty:
                    all_npov_tags = []
                    for tags in npov_revisions['tags_list']:
                        all_npov_tags.extend(tags)
                    
                    if all_npov_tags:
                        # Create WordCloud
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_npov_tags))
                        
                        # Show WordCloud
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title('WordCloud of NPOV-related Tags')
                        st.pyplot(fig)
                    else:
                        st.info("No tags found in NPOV revisions.")
                else:
                    st.info("No NPOV-related revisions found.")
            
            # NPOV changes over time
            if not npov_revisions.empty:
                st.subheader("NPOV Changes Over Time")
                
                # Convert timestamps to date
                npov_revisions = npov_revisions.copy()  # Create explicit copy
                npov_revisions.loc[:, 'date'] = pd.to_datetime(npov_revisions['timestamp']).dt.date
                
                # Group by date
                npov_over_time = npov_revisions.groupby('date').size().reset_index(name='count')
                
                fig = px.line(
                    npov_over_time,
                    x='date',
                    y='count',
                    title="NPOV Changes Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Examples of NPOV changes
            if not npov_changes.empty:
                st.subheader("Examples of NPOV Changes")
                
                # Show only modifications where old and new sentence are present
                npov_mods = npov_changes[
                    (npov_changes['change_type'] == 'modification') & 
                    (npov_changes['old_sentence'].notna()) & 
                    (npov_changes['new_sentence'].notna())
                ]
                
                if not npov_mods.empty:
                    selected_mod = st.selectbox(
                        "Select an NPOV modification:",
                        options=range(len(npov_mods)),
                        format_func=lambda i: f"{npov_mods.iloc[i]['page_title']} - {npov_mods.iloc[i]['timestamp']}"
                    )
                    
                    old_sent = npov_mods.iloc[selected_mod]['old_sentence']
                    new_sent = npov_mods.iloc[selected_mod]['new_sentence']
                    sim = npov_mods.iloc[selected_mod]['similarity']
                    
                    st.markdown("**Old Sentence:**")
                    st.markdown(f"<div style='background-color:#e84444;color:#ffffff;padding:10px;border-radius:4px;'>{old_sent}</div>", unsafe_allow_html=True)
                    
                    st.markdown("**New Sentence:**")
                    st.markdown(f"<div style='background-color:#44aa44;color:#ffffff;padding:10px;border-radius:4px;'>{new_sent}</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Similarity:** {sim:.2f}")
                    
                    # Show differences
                    diff = difflib.ndiff(old_sent.split(), new_sent.split())
                    st.markdown("**Text Difference:**")
                    diff_text = []
                    for i, s in enumerate(diff):
                        if s.startswith('- '):
                            diff_text.append(f"<span style='color:red;'>{s[2:]}</span>")
                        elif s.startswith('+ '):
                            diff_text.append(f"<span style='color:green;'>{s[2:]}</span>")
                        elif s.startswith('  '):
                            diff_text.append(s[2:])
                    
                    st.markdown(f"<div style='background-color:#333333;color:#ffffff;padding:10px;border-radius:4px;'>{' '.join(diff_text)}</div>", unsafe_allow_html=True)
                    
                    # Show all tags
                    tags = json.loads(npov_mods.iloc[selected_mod]['tags'])
                    if tags:
                        st.markdown("**Tags:**")
                        st.write(tags)
                    
                    # Download link for selected change
                    single_mod_df = npov_mods.iloc[[selected_mod]]
                    st.markdown(get_download_link(single_mod_df, "npov_change.csv", "Download this NPOV change"), unsafe_allow_html=True)
                else:
                    st.info("No NPOV modifications with text found.")
    
    with tabs[3]:
        if st.session_state.active_tab == 0:
            st.header("Text Explorer")
            
            if not filtered_changes.empty:
                # Selection of an article
                article_options = ["All Articles"] + list(filtered_changes['page_title'].unique())
                explore_article = st.selectbox("Select article for Text Explorer:", article_options)
                
                explore_df = filtered_changes
                if explore_article != "All Articles":
                    explore_df = filtered_changes[filtered_changes['page_title'] == explore_article]
                
                # Selection of change type
                change_type_options = ["All Types"] + list(explore_df['change_type'].unique())
                explore_type = st.selectbox("Select change type:", change_type_options)
                
                if explore_type != "All Types":
                    explore_df = explore_df[explore_df['change_type'] == explore_type]
                
                # Search in text
                search_query = st.text_input("Search in text (leave empty for all):")
                
                if search_query:
                    mask = (
                        explore_df['old_sentence'].fillna('').str.contains(search_query, case=False) |
                        explore_df['new_sentence'].fillna('').str.contains(search_query, case=False)
                    )
                    explore_df = explore_df[mask]
                
                # Show sentence changes
                if not explore_df.empty:
                    st.success(f"{len(explore_df)} sentence changes found")
                    
                    # Download link for results
                    st.markdown(get_download_link(explore_df, "text_explorer_results.csv", "Download search results"), unsafe_allow_html=True)
                    
                    # Pagination
                    items_per_page = 5
                    total_pages = (len(explore_df) - 1) // items_per_page + 1
                    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                    
                    start_idx = (page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, len(explore_df))
                    
                    for i in range(start_idx, end_idx):
                        change = explore_df.iloc[i]
                        
                        st.markdown(f"### Change {i+1}")
                        st.markdown(f"**Article:** {change['page_title']}")
                        st.markdown(f"**Time:** {change['timestamp']}")
                        st.markdown(f"**Type:** {change['change_type']}")
                        
                        tags = json.loads(change['tags'])
                        if tags:
                            st.markdown(f"**Tags:** {', '.join(tags)}")
                        
                        if change['change_type'] == 'addition':
                            st.markdown(f"**Added Sentence:**")
                            st.markdown(f"<div style='background-color:#44aa44;padding:10px;'>{change['new_sentence']}</div>", unsafe_allow_html=True)
                        
                        elif change['change_type'] == 'deletion':
                            st.markdown(f"**Deleted Sentence:**")
                            st.markdown(f"<div style='background-color:#e84444;padding:10px;'>{change['old_sentence']}</div>", unsafe_allow_html=True)
                        
                        elif change['change_type'] == 'modification':
                            st.markdown(f"**Old Sentence:**")
                            st.markdown(f"<div style='background-color:#e84444;padding:10px;'>{change['old_sentence']}</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"**New Sentence:**")
                            st.markdown(f"<div style='background-color:#44aa44;padding:10px;'>{change['new_sentence']}</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"**Similarity:** {change['similarity']:.2f}")
                        
                        st.markdown("---")
                else:
                    st.info("No sentence changes match your criteria.")
            else:
                st.info("No sentence changes available to explore.")
    
    with tabs[4]:
        if st.session_state.active_tab == 0:
            st.header("Advanced Network Analysis")
            
            # Parameters for network analysis
            min_tag_count = st.slider("Minimum number of occurrences for a tag", 5, 50, 10)
            edge_threshold = st.slider("Minimum number of joint occurrences for a connection", 2, 10, 3)
            
            # Advanced parameters for community detection
            advanced_options = st.expander("Advanced Options")
            with advanced_options:
                community_detection = st.checkbox("Perform community detection", True)
                algorithm = st.selectbox("Community detection algorithm", 
                                        ["Louvain", "Girvan-Newman", "Label Propagation"])
                show_centrality = st.checkbox("Calculate centrality measures", True)
                centrality_type = st.selectbox("Centrality type", 
                                            ["Degree", "Betweenness", "Closeness", "Eigenvector"])
            
            if st.button("Perform Network Analysis"):
                if not filtered_revisions.empty:
                    all_tags = []
                    for tags in filtered_revisions['tags_list']:
                        all_tags.extend(tags)
                    
                    with st.spinner("Performing network analysis..."):
                        # Count tag frequencies
                        tag_counts = pd.Series(all_tags).value_counts()
                        
                        # Filter by minimum frequency
                        filtered_tags = tag_counts[tag_counts >= min_tag_count]
                        
                        if len(filtered_tags) > 2:  # At least 3 tags for a meaningful network
                            # Create tag co-occurrence matrix
                            tag_pairs = []
                            for tags in filtered_revisions['tags_list']:
                                filtered_tags_in_revision = [tag for tag in tags if tag in filtered_tags.index]
                                if len(filtered_tags_in_revision) > 1:
                                    for i, tag1 in enumerate(filtered_tags_in_revision):
                                        for tag2 in filtered_tags_in_revision[i+1:]:
                                            tag_pairs.append((tag1, tag2))
                            
                            if tag_pairs:
                                # Count co-occurrence
                                co_occurrence = pd.Series(tag_pairs).value_counts()
                                
                                # Create network
                                G = nx.Graph()
                                
                                # Add nodes with weighting based on frequency
                                for tag, count in filtered_tags.items():
                                    G.add_node(tag, weight=count)
                                
                                # Add edges with weighting based on co-occurrence
                                for (tag1, tag2), count in co_occurrence.items():
                                    if count >= edge_threshold:
                                        G.add_edge(tag1, tag2, weight=count)
                                
                                # Remove isolated nodes
                                G.remove_nodes_from(list(nx.isolates(G)))
                                
                                if len(G.nodes()) > 0:
                                    # Calculate network statistics
                                    network_stats = {
                                        'nodes': len(G.nodes()),
                                        'edges': len(G.edges()),
                                        'density': nx.density(G),
                                        'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes())
                                    }
                                    
                                    st.write("### Network Statistics")
                                    stats_col1, stats_col2 = st.columns(2)
                                    stats_col1.metric("Number of Nodes", network_stats['nodes'])
                                    stats_col2.metric("Number of Edges", network_stats['edges'])
                                    stats_col1.metric("Network Density", f"{network_stats['density']:.4f}")
                                    stats_col2.metric("Average Degree", f"{network_stats['avg_degree']:.2f}")
                                    
                                    # Community detection
                                    if community_detection:
                                        st.write("### Community Structure")
                                        
                                        try:
                                            if algorithm == "Louvain":
                                                # Requires additional package: python-louvain or community
                                                import community as community_louvain
                                                partition = community_louvain.best_partition(G)
                                                communities = {}
                                                for node, community_id in partition.items():
                                                    if community_id not in communities:
                                                        communities[community_id] = []
                                                    communities[community_id].append(node)
                                                
                                            elif algorithm == "Girvan-Newman":
                                                comp = nx.community.girvan_newman(G)
                                                communities_tuple = tuple(sorted(c) for c in next(comp))
                                                communities = {i: list(comm) for i, comm in enumerate(communities_tuple)}
                                                
                                            elif algorithm == "Label Propagation":
                                                communities_gen = nx.community.label_propagation_communities(G)
                                                communities = {i: list(comm) for i, comm in enumerate(communities_gen)}
                                            
                                            # Display communities
                                            st.write(f"Found communities: {len(communities)}")
                                            
                                            for i, (community_id, nodes) in enumerate(communities.items()):
                                                if i < 5:  # Show max. 5 communities
                                                    st.write(f"Community {community_id} ({len(nodes)} nodes):")
                                                    st.write(", ".join(nodes[:10]) + ("..." if len(nodes) > 10 else ""))
                                            
                                            # Visualize communities
                                            pos = nx.spring_layout(G, seed=42)
                                            
                                            plt.figure(figsize=(12, 8))
                                            
                                            # Draw edges
                                            nx.draw_networkx_edges(G, pos, alpha=0.2)
                                            
                                            # Draw nodes colored by communities
                                            if algorithm == "Louvain":
                                                cmap = plt.colormaps["rainbow"]
                                                nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=80,
                                                                    cmap=cmap, node_color=list(partition.values()))
                                            else:
                                                for i, community in enumerate(communities.values()):
                                                    nx.draw_networkx_nodes(G, pos, community, node_size=80,
                                                                        node_color=[f"C{i}"] * len(community))
                                            
                                            # Draw labels
                                            nx.draw_networkx_labels(G, pos, font_size=8)
                                            
                                            plt.title("Communities in Tag Network")
                                            plt.axis("off")
                                            st.pyplot(plt)
                                            
                                        except Exception as e:
                                            st.error(f"Error in community detection: {str(e)}")
                                    
                                    # Centrality measures
                                    if show_centrality:
                                        st.write("### Centrality Analysis")
                                        
                                        try:
                                            if centrality_type == "Degree":
                                                centrality = nx.degree_centrality(G)
                                            elif centrality_type == "Betweenness":
                                                centrality = nx.betweenness_centrality(G)
                                            elif centrality_type == "Closeness":
                                                centrality = nx.closeness_centrality(G)
                                            elif centrality_type == "Eigenvector":
                                                centrality = nx.eigenvector_centrality(G, max_iter=1000)
                                            
                                            # Sort by centrality
                                            sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                                            
                                            # Show top-10 nodes
                                            central_tags_df = pd.DataFrame(sorted_centrality[:10], 
                                                                        columns=['Tag', 'Centrality'])
                                            
                                            st.write(f"Top-10 Tags by {centrality_type} Centrality:")
                                            st.dataframe(central_tags_df)
                                            
                                            # Visualize centrality
                                            plt.figure(figsize=(12, 8))
                                            
                                            # Draw edges
                                            nx.draw_networkx_edges(G, pos, alpha=0.2)
                                            
                                            # Size and color based on centrality
                                            node_size = [v * 5000 for v in centrality.values()]
                                            nodes = nx.draw_networkx_nodes(G, pos, 
                                                                        node_size=node_size,
                                                                        node_color=list(centrality.values()),
                                                                        cmap=plt.cm.Blues)
                                            
                                            # Draw labels for important nodes
                                            top_nodes = dict(sorted_centrality[:10])
                                            nx.draw_networkx_labels(G, pos, labels={n: n for n in top_nodes},
                                                                font_size=8)
                                            
                                            plt.colorbar(nodes)
                                            plt.title(f"{centrality_type} Centrality in Tag Network")
                                            plt.axis("off")
                                            st.pyplot(plt)
                                            
                                        except Exception as e:
                                            st.error(f"Error in centrality analysis: {str(e)}")
                                    
                                    # Export options
                                    # Network export area 
                                    st.write("### Network Export")

                                    # Initialize session state if not already done
                                    if 'export_ready' not in st.session_state:
                                        st.session_state.export_ready = False
                                        st.session_state.export_files = {}

                                    # Area for export options
                                    export_options = st.expander("Export Options", expanded=True)

                                    with export_options:
                                        # Simple export options without radio buttons
                                        st.write("Select an export format:")
                                        
                                        # Columns for export buttons
                                        col1, col2, col3 = st.columns(3)
                                        
                                        # Define export functions
                                        def export_png():
                                            try:
                                                # Create temporary files in working directory
                                                export_dir = os.path.join(os.getcwd(), "exports")
                                                os.makedirs(export_dir, exist_ok=True)
                                                export_path = os.path.join(export_dir, f"network_export_{int(time.time())}.png")
                                                
                                                # Draw and save network
                                                plt.figure(figsize=(12, 12))
                                                pos = nx.spring_layout(G, seed=42)
                                                
                                                # Simple coloring
                                                nx.draw_networkx(
                                                    G, pos, 
                                                    node_size=80, 
                                                    font_size=8, 
                                                    node_color='skyblue',
                                                    edge_color='gray',
                                                    alpha=0.7
                                                )
                                                plt.title('Tag Network')
                                                plt.axis('off')
                                                
                                                # Save as PNG
                                                plt.savefig(export_path, dpi=300, bbox_inches='tight')
                                                plt.close()
                                                
                                                # Save in session state
                                                with open(export_path, "rb") as f:
                                                    st.session_state.export_files['png'] = {
                                                        'path': export_path,
                                                        'data': f.read(),
                                                        'time': time.time()
                                                    }
                                                
                                                st.session_state.export_ready = True
                                                
                                                # Inform the user
                                                st.success(f"PNG export successfully created: {export_path}")
                                                
                                            except Exception as e:
                                                st.error(f"Error during PNG export: {str(e)}")
                                        
                                        def export_gexf():
                                            try:
                                                # Create temporary files in working directory
                                                export_dir = os.path.join(os.getcwd(), "exports")
                                                os.makedirs(export_dir, exist_ok=True)
                                                export_path = os.path.join(export_dir, f"network_export_{int(time.time())}.gexf")
                                                
                                                # Save as GEXF
                                                nx.write_gexf(G, export_path)
                                                
                                                # Save in session state
                                                with open(export_path, "rb") as f:
                                                    st.session_state.export_files['gexf'] = {
                                                        'path': export_path,
                                                        'data': f.read(),
                                                        'time': time.time()
                                                    }
                                                
                                                st.session_state.export_ready = True
                                                
                                                # Inform the user
                                                st.success(f"GEXF export successfully created: {export_path}")
                                                
                                            except Exception as e:
                                                st.error(f"Error during GEXF export: {str(e)}")
                                        
                                        def export_csv():
                                            try:
                                                # Create temporary files in working directory
                                                export_dir = os.path.join(os.getcwd(), "exports")
                                                os.makedirs(export_dir, exist_ok=True)
                                                export_path = os.path.join(export_dir, f"network_export_{int(time.time())}.csv")
                                                
                                                # Create adjacency matrix as DataFrame
                                                adj_matrix = pd.DataFrame(
                                                    nx.to_numpy_array(G),
                                                    index=list(G.nodes()),
                                                    columns=list(G.nodes())
                                                )
                                                
                                                # Save as CSV
                                                adj_matrix.to_csv(export_path)
                                                
                                                # Save in session state
                                                with open(export_path, "rb") as f:
                                                    st.session_state.export_files['csv'] = {
                                                        'path': export_path,
                                                        'data': f.read(),
                                                        'time': time.time()
                                                    }
                                                
                                                st.session_state.export_ready = True
                                                
                                                # Inform the user
                                                st.success(f"CSV export successfully created: {export_path}")
                                                
                                            except Exception as e:
                                                st.error(f"Error during CSV export: {str(e)}")
                                        
                                        # Export buttons
                                        with col1:
                                            st.button("Export as PNG", on_click=export_png)
                                        
                                        with col2:
                                            st.button("Export as GEXF", on_click=export_gexf)
                                        
                                        with col3:
                                            st.button("Export as CSV", on_click=export_csv)

                                    # Download area (displayed when exports are available)
                                    if st.session_state.export_ready:
                                        st.write("### Available Exports")
                                        
                                        # Show available exports
                                        for format_type, file_info in st.session_state.export_files.items():
                                            if 'data' in file_info and 'path' in file_info:
                                                col1, col2 = st.columns([3, 1])
                                                
                                                with col1:
                                                    st.info(f"Export in {format_type.upper()} format: {os.path.basename(file_info['path'])}")
                                                    st.write(f"Created: {time.strftime('%H:%M:%S', time.localtime(file_info['time']))}")
                                                
                                                with col2:
                                                    if format_type == 'png':
                                                        st.download_button(
                                                            "Download",
                                                            data=file_info['data'],
                                                            file_name=f"tag_network.png",
                                                            mime="image/png",
                                                            key=f"download_{format_type}"
                                                        )
                                                    elif format_type == 'gexf':
                                                        st.download_button(
                                                            "Download",
                                                            data=file_info['data'],
                                                            file_name=f"tag_network.gexf",
                                                            mime="application/xml",
                                                            key=f"download_{format_type}"
                                                        )
                                                    elif format_type == 'csv':
                                                        st.download_button(
                                                            "Download",
                                                            data=file_info['data'],
                                                            file_name=f"tag_network.csv",
                                                            mime="text/csv",
                                                            key=f"download_{format_type}"
                                                        )
                                        
                                        # Additional information
                                        st.info("""
                                        ℹ**Note:** All export files are also saved in the 'exports' 
                                        directory in the dashboard's working directory and remain there.
                                        """)
                                        
                                        # Preview of PNG export, if available
                                        if 'png' in st.session_state.export_files and os.path.exists(st.session_state.export_files['png']['path']):
                                            with st.expander("PNG Preview", expanded=False):
                                                st.image(st.session_state.export_files['png']['path'], caption="Tag Network", use_column_width=True)
                                        
                                        # Preview of CSV file, if available
                                        if 'csv' in st.session_state.export_files and os.path.exists(st.session_state.export_files['csv']['path']):
                                            with st.expander("CSV Preview", expanded=False):
                                                csv_path = st.session_state.export_files['csv']['path']
                                                try:
                                                    csv_data = pd.read_csv(csv_path, index_col=0)
                                                    st.dataframe(csv_data.iloc[:10, :10])
                                                    st.write(f"Showing the first 10×10 entries out of {csv_data.shape[0]}×{csv_data.shape[1]} total")
                                                except Exception as e:
                                                    st.error(f"Error reading CSV file: {e}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure the file 'wikipedia_analysis.db' is in the same directory as this script.")