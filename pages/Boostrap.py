import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="centered")

def main():
    # Sidebar inputs
    st.sidebar.header("Sample Parameters")
    conversion_a = st.sidebar.slider("Conversion Rate for Sample A (%)", 0, 100, 20, 1)
    conversion_b = st.sidebar.slider("Conversion Rate for Sample B (%)", 0, 100, 25, 1)
    sample_size_a = st.sidebar.number_input("Sample Size A", min_value=1, value=100)
    sample_size_b = st.sidebar.number_input("Sample Size B", min_value=1, value=100)

    # Functions
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

    def plot_results(means_a, means_b):
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        axes[0].hist(means_a, bins=20, color="blue", alpha=0.7, label="A")
        axes[0].hist(means_b, bins=20, color="red", alpha=0.7, label="B")
        axes[0].set_title("Distribution of Means (A and B)")
        axes[0].set_xlabel("Mean Value")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        diff_means = np.array(means_b) - np.array(means_a)
        axes[1].hist(diff_means, bins=20, color="green", alpha=0.7, label="B - A")
        axes[1].set_title("Distribution of Mean Differences")
        axes[1].set_xlabel("Mean Difference")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        st.pyplot(fig)

        st.write(f"Average difference between samples: {np.mean(diff_means):.4f}")
        st.write(f"Confidence interval of the difference (2.5%, 97.5%): ({np.percentile(diff_means, 2.5):.4f}, {np.percentile(diff_means, 97.5):.4f})")

    # Main interface
    st.title("A/B Test")

    # Regenerate button in sidebar
    if st.sidebar.button("Regenerate"):
        st.session_state['means_a'] = []
        st.session_state['means_b'] = []

        st.session_state['sample_a'] = generate_samples(conversion_a, sample_size_a)
        st.session_state['sample_b'] = generate_samples(conversion_b, sample_size_b)

    # Generate initial samples if not already generated
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

    # Render first two smaller height graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))  # Adjusted height
    axes[0].bar(range(len(sample_a)), sample_a, alpha=0.6, label="A", color='blue')
    axes[0].set_title("Sample A")
    axes[0].set_xlabel("Sample Member Index")
    axes[0].set_ylabel("Value (1 or 0)")

    axes[1].bar(range(len(sample_b)), sample_b, alpha=0.6, label="B", color='red')
    axes[1].set_title("Sample B")
    axes[1].set_xlabel("Sample Member Index")
    axes[1].set_ylabel("Value (1 or 0)")

    st.pyplot(fig)

    # Iteration buttons
    col1, col2, col3, col4, col5 = st.columns(5, gap='small')
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

    # Reset button
    if st.button("Reset"):
        st.session_state['means_a'] = []
        st.session_state['means_b'] = []

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
