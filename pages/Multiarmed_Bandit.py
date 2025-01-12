import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Class to represent a bandit
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

# Define the main function
def main():
    # Initialize the app
    st.title("Thompson Sampling: Bayesian Bandits")
    bandit_count = 4
    labels = ["A", "B", "C", "D"]

    # User settings
    st.sidebar.header("Settings")
    if "conversion_rates" not in st.session_state:
        st.session_state.conversion_rates = np.random.rand(bandit_count).tolist()

    # Allow user to adjust conversion rates in the sidebar
    conversion_rates = [
        st.sidebar.number_input(f"Bandit {i+1}", min_value=0.0, max_value=1.0, value=st.session_state.conversion_rates[i], step=0.01)
        for i in range(bandit_count)
    ]

    generate_bandits = st.sidebar.button("Generate Bandits")

    # Bandit logic
    if 'bandits' not in st.session_state or generate_bandits:
        np.random.shuffle(conversion_rates)
        st.session_state.bandits = [Bandit(p) for p in conversion_rates]

    def plot_distributions(bandits):
        x = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for i, bandit in enumerate(bandits):
            y = beta.pdf(x, bandit.alpha, bandit.beta)
            ax[i].plot(x, y, label=f"Bandit {labels[i]}")
            ax[i].fill_between(x, y, alpha=0.2)
            ax[i].set_title(f"Bandit {labels[i]}: α={bandit.alpha}, β={bandit.beta}")
        st.pyplot(fig)

    # Display probability distributions
    st.subheader("Probability Distributions")

    # Manual bandit selection
    st.subheader("Manual Interaction")
    columns = st.columns(4)
    for i, bandit in enumerate(st.session_state.bandits):
        if columns[i].button(f"Select Bandit {labels[i]}"):
            result = bandit.pull()
            bandit.update(result)
            st.write(f"Result: {'Success' if result else 'Failure'}")

    # Automatic trials
    st.subheader("Automatic Simulation")
    n_trials = st.number_input("Number of Trials", min_value=1, max_value=10_000, value=10, step=1)
    if st.button("Run Trials"):
        for _ in range(n_trials):
            samples = [np.random.beta(b.alpha, b.beta) for b in st.session_state.bandits]
            chosen_bandit = np.argmax(samples)
            result = st.session_state.bandits[chosen_bandit].pull()
            st.session_state.bandits[chosen_bandit].update(result)
        st.success(f"Completed {n_trials} trials.")
    plot_distributions(st.session_state.bandits)

    # Statistics
    st.subheader("Statistics")
    st.table({
        f"Bandit {labels[i]}": {
            "Successes": b.successes,
            "Failures": b.failures,
            "Conversion Rate": f"{b.successes / (b.successes + b.failures):.2f}" if b.successes + b.failures > 0 else "N/A"
        } for i, b in enumerate(st.session_state.bandits)
    })

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
