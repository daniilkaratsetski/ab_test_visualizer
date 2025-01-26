import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def main():
    # -----------------------------------------------------
    # Title Section
    # -----------------------------------------------------
    st.title("Bayesian Updating: Coin Toss Experiment")

    # -----------------------------------------------------
    # Explanation Section
    # -----------------------------------------------------
    st.markdown(
        """
        ## Overview
        Bayesian updating is a method of updating our beliefs in light of new data. 
        In this app, we'll demonstrate Bayesian updating using a coin toss experiment.

        - **Prior**: What you believe about the probability of heads before seeing data.
        - **Likelihood**: The probability of observing the data given a hypothesis.
        - **Posterior**: Updated belief about the probability of heads after observing data.
        """
    )

    # -----------------------------------------------------
    # Sidebar Inputs
    # -----------------------------------------------------
    st.sidebar.header("Set Inputs")

    st.sidebar.subheader("Prior Beliefs (Beta Distribution Parameters)")
    alpha_prior = st.sidebar.slider("Alpha (History Heads)", min_value=1, max_value=100, value=2)
    beta_prior = st.sidebar.slider("Beta (History Tails)", min_value=1, max_value=100, value=2)

    st.sidebar.subheader("Observed Data")
    heads = st.sidebar.number_input("Number of Heads", min_value=0, value=9)
    tails = st.sidebar.number_input("Number of Tails", min_value=0, value=1)

    # -----------------------------------------------------
    # Bayesian Updating Calculations
    # -----------------------------------------------------
    x = np.linspace(0, 1, 500)
    prior = beta.pdf(x, alpha_prior, beta_prior)

    alpha_posterior = alpha_prior + heads
    beta_posterior = beta_prior + tails
    posterior = beta.pdf(x, alpha_posterior, beta_posterior)

    posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    credible_interval = beta.interval(0.95, alpha_posterior, beta_posterior)

    # -----------------------------------------------------
    # Visualization: Prior and Posterior Distributions
    # -----------------------------------------------------
    st.subheader("Prior and Posterior Distributions")
    fig, ax = plt.subplots()

    ax.plot(x, prior, label="Prior", linestyle="--", color="orange")
    ax.plot(x, posterior, label="Posterior", color="blue")
    ax.legend()
    ax.set_title("Prior vs Posterior")
    ax.set_xlabel("Probability of Heads")
    ax.set_ylabel("Density")

    ax.annotate(
        f"Posterior Mean (probability of heads): {posterior_mean:.3f}\n"
        xy=(posterior_mean, beta.pdf(posterior_mean, alpha_posterior, beta_posterior)),
        xytext=(0.5, 3),
        arrowprops=dict(facecolor='black', arrowstyle="->"),
        fontsize=10,
    )

    st.pyplot(fig)

    # -----------------------------------------------------
    # Display Metrics
    # -----------------------------------------------------
    st.subheader("Key Metrics")
    st.write(f"**Posterior Mean**: {posterior_mean:.3f}")

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
