import streamlit as st
import numpy as np
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="centered")

def main():
    st.sidebar.header("Sample Parameters")
    conversion_a = st.sidebar.slider("Conversion Rate for Sample A (%)", 0, 100, 20, 1)
    conversion_b = st.sidebar.slider("Conversion Rate for Sample B (%)", 0, 100, 25, 1)
    perfect_squares = [i**2 for i in range(10, 40, 2)]
    sample_size_a = st.sidebar.selectbox("Sample Size A (square)", perfect_squares, index=0)
    sample_size_b = st.sidebar.selectbox("Sample Size B (square)", perfect_squares, index=0)
    pastel_blue = "#aec6cf"
    pastel_red = "#ffcccb"
    pastel_green = "#c1e1c1"

    def generate_samples(conversion, size):
        return np.random.binomial(1, conversion / 100, size)
    
    def bootstrap_sample(data):
        return np.random.choice(data, size=len(data), replace=True)
    
    def bootstrap_means(data_a, data_b, iterations, current_means_a, current_means_b):
        for _ in range(iterations):
            sample_a = bootstrap_sample(data_a)
            sample_b = bootstrap_sample(data_b)
            current_means_a.append(np.mean(sample_a))
            current_means_b.append(np.mean(sample_b))
        return current_means_a, current_means_b

    def plot_conversion_grid(sample, label, color):
        n = len(sample)
        side = int(math.sqrt(n))
        matrix = np.reshape(sample, (side, side))
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale=[[0, 'white'], [1, color]],
            zmin=0, zmax=1,
            showscale=False,
            hoverinfo='skip'
        ))
        shapes = []
        for i in range(1, side):
            shapes.append(dict(
                type="line",
                x0=-0.5, x1=side-0.5,
                y0=i-0.5, y1=i-0.5,
                line=dict(color="gray", width=1)
            ))
            shapes.append(dict(
                type="line",
                x0=i-0.5, x1=i-0.5,
                y0=-0.5, y1=side-0.5,
                line=dict(color="gray", width=1)
            ))
        shapes.append(dict(
            type="rect",
            x0=-0.5, y0=-0.5,
            x1=side-0.5, y1=side-0.5,
            line=dict(color="gray", width=2)
        ))
        fig.update_layout(
            title=label,
            xaxis=dict(showgrid=False, zeroline=False, tickvals=[], constrain='domain'),
            yaxis=dict(showgrid=False, zeroline=False, tickvals=[], scaleanchor="x", scaleratio=1, autorange="reversed"),
            margin=dict(l=10, r=10, t=40, b=50),
            shapes=shapes,
            width=300,
            height=300
        )
        conv_rate = np.mean(sample) * 100
        conv_count = int(np.sum(sample))
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            text=f"Real conversion: {conv_rate:.1f}% ({conv_count}/{n})",
            showarrow=False,
            font=dict(size=12)
        )
        return fig

    def plot_results(means_a, means_b):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution of Means (A and B)", "Distribution of Mean Differences"))
        fig.add_trace(
            go.Histogram(x=means_a, nbinsx=20, name="A", marker_color=pastel_blue, opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=means_b, nbinsx=20, name="B", marker_color=pastel_red, opacity=0.7),
            row=1, col=1
        )
        diff_means = np.array(means_b) - np.array(means_a)
        fig.add_trace(
            go.Histogram(x=diff_means, nbinsx=20, name="B - A", marker_color=pastel_green, opacity=0.7),
            row=1, col=2
        )
        fig.update_layout(barmode='overlay', showlegend=True, width=900, height=400)
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Average difference between samples: {np.mean(diff_means):.4f}")
        st.write(f"Confidence interval of the difference (2.5%, 97.5%): ({np.percentile(diff_means, 2.5):.4f}, {np.percentile(diff_means, 97.5):.4f})")

    st.title("A/B Test")
    if st.sidebar.button("Regenerate"):
        st.session_state['means_a'] = []
        st.session_state['means_b'] = []
        st.session_state['sample_a'] = generate_samples(conversion_a, sample_size_a)
        st.session_state['sample_b'] = generate_samples(conversion_b, sample_size_b)
    if 'sample_a' not in st.session_state or 'sample_b' not in st.session_state:
        st.session_state['sample_a'] = generate_samples(conversion_a, sample_size_a)
        st.session_state['sample_b'] = generate_samples(conversion_b, sample_size_b)
    sample_a = st.session_state['sample_a']
    sample_b = st.session_state['sample_b']
    if 'means_a' not in st.session_state:
        st.session_state['means_a'] = []
        st.session_state['means_b'] = []
    means_a = st.session_state['means_a']
    means_b = st.session_state['means_b']
    col1, col2 = st.columns(2)
    with col1:
        fig_a = plot_conversion_grid(sample_a, "Sample A", pastel_blue)
        st.plotly_chart(fig_a, use_container_width=False)
    with col2:
        fig_b = plot_conversion_grid(sample_b, "Sample B", pastel_red)
        st.plotly_chart(fig_b, use_container_width=False)
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.button("1\niteration"):
        means_a, means_b = bootstrap_means(sample_a, sample_b, 1, means_a, means_b)
        plot_results(means_a, means_b)
    if col2.button("10 iterations"):
        means_a, means_b = bootstrap_means(sample_a, sample_b, 10, means_a, means_b)
        plot_results(means_a, means_b)
    if col3.button("100 iterations"):
        means_a, means_b = bootstrap_means(sample_a, sample_b, 100, means_a, means_b)
        plot_results(means_a, means_b)
    if col4.button("1000 iterations"):
        means_a, means_b = bootstrap_means(sample_a, sample_b, 1000, means_a, means_b)
        plot_results(means_a, means_b)
    if col5.button("10k iterations"):
        means_a, means_b = bootstrap_means(sample_a, sample_b, 10000, means_a, means_b)
        plot_results(means_a, means_b)
    if st.button("Reset"):
        st.session_state['means_a'] = []
        st.session_state['means_b'] = []

if __name__ == "__main__":
    main()

