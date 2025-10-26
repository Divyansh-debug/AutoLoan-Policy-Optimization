# Policy Optimization for Financial Decision-Making: LendingClub Loans

This project implements and compares two machine learning approaches for optimizing loan approval decisions using the LendingClub dataset. The primary objective is to develop a policy that maximizes financial return for a fintech company, moving beyond simple default prediction.

**Models Compared:**
1.  **Predictive Deep Learning (MLP):** A supervised model trained to predict the probability of loan default.
2.  **Offline Reinforcement Learning (Discrete-CQL):** A decision-making agent trained using `d3rlpy` to learn a policy that directly maximizes the expected profit (reward).

---

## 🚀 Key Findings: RL Outperforms Prediction

The final analysis clearly demonstrates the superiority of the Offline RL approach (Discrete-CQL) in maximizing financial return compared to both the historical strategy (always approving) and the predictive DL model (approving based on a default probability threshold).

**Policy Performance on Test Set:**

| Policy                         | Total Profit     | Avg. Profit per Loan | Approval Rate |
| :----------------------------- | :--------------- | :------------------- | :------------ |
| 1. Historical (Always Approve) | $-63,972,634.52  | $-1,816.53           | 100.00%       |
| 2. DL Model (Threshold=0.5)    | $-6,569,940.06   | $-186.56             | 59.00%        |
| **3. RL Agent (Profit-Max)** | **$-424,549.21** | **$-12.06** | **1.15%** |

**Highlights:**
* The RL agent reduced losses by **~99.3%** compared to the baseline and **~93.5%** compared to the DL model.
* The RL agent learned an extremely conservative policy (1.15% approval rate), focusing only on loans with high expected profit, successfully distinguishing *profitable* loans from merely *non-defaulting* loans.
* **Disagreement Analysis:** In cases where the DL model approved but the RL agent denied (20,681 loans), the *actual average outcome* was a loss of **$-318.98**, confirming the RL agent's superior financial decision-making.

---

## 🛠️ Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install Dependencies:**
    Using a virtual environment is recommended.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Kaggle API Key:**
    * This project uses the Kaggle API to download the dataset. Ensure you have your `kaggle.json` API key set up.
    * You can download it from your Kaggle account settings (`Account` -> `API` -> `Create New API Token`).
    * Place the `kaggle.json` file in the appropriate location (e.g., `~/.kaggle/kaggle.json` on Linux/Mac, `C:\Users\<Username>\.kaggle\kaggle.json` on Windows) or follow the instructions in the first cell of the notebook if running in Colab.

---

## 🏃‍♂️ How to Run the Code

The entire project is contained within the single Jupyter Notebook: `AutoLoan_Policy_Optimization(Complete).ipynb`.

**Execution Order is Crucial:**
The notebook is divided into sections corresponding to the project tasks. **You must run the cells sequentially from top to bottom.**

* **Initial Setup:** The first cell handles Kaggle API setup and dataset download.
* **Task 1 (EDA):** Performs data loading, cleaning, feature engineering, and exploratory analysis.
* **Task 2 (DL Model):** Trains the MLP model for default prediction and saves `dl_model.h5` and intermediate data (`task_3_data.npz`).
* **Task 3 (RL Agent):** Loads intermediate data, engineers the RL environment (states, actions, rewards), trains the Discrete-CQL agent, and saves the RL policy (`cql_policy.pt`) and final analysis data (`task_4_analysis_data.npz`).
* **Task 4 (Analysis):** Loads the final analysis data and performs the policy comparison, generating the results table and disagreement analysis.

**Output Artifacts:**
Running the notebook sequentially will generate the following key files in the same directory:
* `dl_model.h5`: The trained TensorFlow/Keras MLP model.
* `task_3_data.npz`: Processed data used for RL training.
* `cql_policy.pt`: The trained `d3rlpy` CQL policy model.
* `task_4_analysis_data.npz`: Data required for the final policy evaluation.

---
