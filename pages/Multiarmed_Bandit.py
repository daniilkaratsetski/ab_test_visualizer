import streamlit as st
import numpy as np
from scipy.stats import beta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="centered")

class Bandit:
    def __init__(self, true_conversion_rate):
        self.true_conversion_rate = true_conversion_rate
        self.alpha = 1
        self.beta = 1
        self.successes = 0
        self.failures = 0

    def pull(self):
        return np.random.rand() < self.true_conversion_rate

    def update(self, result):
        self.alpha += result
        self.beta += 1 - result
        if result:
            self.successes += 1
        else:
            self.failures += 1

def main():
    st.title("Thompson Sampling: Bayesian Bandits")
    bandit_count = 4
    labels = ["A", "B", "C", "D"]
    st.sidebar.header("Settings")
    if "conversion_rates" not in st.session_state:
        st.session_state.conversion_rates = np.random.rand(bandit_count).tolist()
    conversion_rates = [
        st.sidebar.number_input(f"Bandit {i+1}", min_value=0.0, max_value=1.0, value=st.session_state.conversion_rates[i], step=0.01)
        for i in range(bandit_count)
    ]
    generate_bandits = st.sidebar.button("Generate Bandits")
    if 'bandits' not in st.session_state or generate_bandits:
        np.random.shuffle(conversion_rates)
        st.session_state.bandits = [Bandit(p) for p in conversion_rates]

    def plot_distributions(bandits):
        x = np.linspace(0, 1, 100)
        subplot_titles = [f"Bandit {labels[i]}: α={b.alpha}, β={b.beta}" for i, b in enumerate(bandits)]
        fig = make_subplots(rows=1, cols=bandit_count, subplot_titles=subplot_titles)
        for i, bandit in enumerate(bandits):
            y = beta.pdf(x, bandit.alpha, bandit.beta)
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", name=f"Bandit {labels[i]}"),
                row=1, col=i+1
            )
            fig.update_xaxes(range=[0, 1], row=1, col=i+1)
        fig.update_layout(showlegend=False, width=1000, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Probability Distributions")
    st.subheader("Manual Interaction")
    columns = st.columns(bandit_count)
    for i, bandit in enumerate(st.session_state.bandits):
        if columns[i].button(f"Select Bandit {labels[i]}"):
            result = bandit.pull()
            bandit.update(result)
            st.write(f"Result: {'Success' if result else 'Failure'}")
    st.subheader("Automatic Simulation")
    n_trials = st.number_input("Number of Trials", min_value=1, max_value=10000, value=10, step=1)
    if st.button("Run Trials"):
        for _ in range(n_trials):
            samples = [np.random.beta(b.alpha, b.beta) for b in st.session_state.bandits]
            chosen_bandit = np.argmax(samples)
            result = st.session_state.bandits[chosen_bandit].pull()
            st.session_state.bandits[chosen_bandit].update(result)
        st.success(f"Completed {n_trials} trials.")
    plot_distributions(st.session_state.bandits)
    st.subheader("Statistics")
    st.table({
        f"Bandit {labels[i]}": {
            "Successes": b.successes,
            "Failures": b.failures,
            "Conversion Rate": f"{b.successes / (b.successes + b.failures):.2f}" if b.successes + b.failures > 0 else "N/A"
        } for i, b in enumerate(st.session_state.bandits)
    })

if __name__ == "__main__":
    main()
