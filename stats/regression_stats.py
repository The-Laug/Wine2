from scipy.stats import norm
import numpy as np
from scipy.stats import ttest_rel

E_test_ANN = [1.202, 0.875, 0.928, 0.988, 0.968, 0.756, 0.816, 0.906, 1.046, 1.645]  # Test error for ANN
E_test_LR = [0.358, 0.321, 0.372, 0.338, 0.374, 0.336, 0.376, 0.344, 0.334, 0.317]  # Test error for logistic regression
E_test_baseline = [1.199, 0.871, 0.926, 0.987, 0.966, 0.755, 0.814, 0.904, 1.047, 1.648]  # Test error for baseline


n_ANN = len(E_test_ANN)
n_LR = len(E_test_LR)
n_baseline = len(E_test_baseline)

alpha = 0.05  # Significance level for a 95% confidence interval

z_hat_ANN_LR = np.mean(E_test_ANN) - np.mean(E_test_LR)  # Mean of the differences
z_hat_ANN_baseline = np.mean(E_test_ANN) - np.mean(E_test_baseline)  # Mean of the differences
z_hat_LR_baseline = np.mean(E_test_LR) - np.mean(E_test_baseline)  # Mean of the differences

# z_hat = 0.2  # Example mean of the differences, adjust as needed

#Standard deviation of the differences
sigma_tilde_ANN_LR = np.std(E_test_ANN) - np.std(E_test_LR)
sigma_tilde_baseline_ANN = np.std(E_test_baseline) - np.std(E_test_ANN) 
sigma_tilde_baseline_LR =  np.std(E_test_baseline) - np.std(E_test_LR)

# Degrees of freedom (v = n - 1)
v = n_ANN - 1

# Calculate the lower and upper bounds for the CI
z_L_ANN_LR = norm.ppf(alpha / 2, loc=z_hat_ANN_LR, scale=sigma_tilde_ANN_LR / np.sqrt(n_ANN))
z_U_ANN_LR = norm.ppf(1 - alpha / 2, loc=z_hat_ANN_LR, scale=sigma_tilde_ANN_LR / np.sqrt(n_ANN))

z_L_ANN_baseline = norm.ppf(alpha / 2, loc=z_hat_ANN_baseline, scale=sigma_tilde_baseline_ANN / np.sqrt(n_ANN))
z_U_ANN_baseline = norm.ppf(1 - alpha / 2, loc=z_hat_ANN_baseline, scale=sigma_tilde_baseline_ANN / np.sqrt(n_ANN))

z_L_LR_baseline = norm.ppf(alpha / 2, loc=z_hat_LR_baseline, scale=sigma_tilde_baseline_LR / np.sqrt(n_ANN))
z_U_LR_baseline = norm.ppf(1 - alpha / 2, loc=z_hat_LR_baseline, scale=sigma_tilde_baseline_LR / np.sqrt(n_ANN))

print(z_L_ANN_LR, z_U_ANN_LR, z_L_ANN_baseline, z_U_ANN_baseline, z_L_LR_baseline, z_U_LR_baseline)



# Calculate the differences
diff_ANN_LR = np.array(E_test_ANN) - np.array(E_test_LR)
diff_ANN_baseline = np.array(E_test_ANN) - np.array(E_test_baseline)
diff_LR_baseline = np.array(E_test_LR) - np.array(E_test_baseline)

# Perform paired t-tests
t_stat_ANN_LR, p_value_ANN_LR = ttest_rel(E_test_ANN, E_test_LR)
t_stat_ANN_baseline, p_value_ANN_baseline = ttest_rel(E_test_ANN, E_test_baseline)
t_stat_LR_baseline, p_value_LR_baseline = ttest_rel(E_test_LR, E_test_baseline)
print("Paired t-test results:")
print(f"ANN vs LR: t-statistic = {t_stat_ANN_LR:.5f}, p-value = {p_value_ANN_LR:.5f}")
print(f"ANN vs Baseline: t-statistic = {t_stat_ANN_baseline:.5f}, p-value = {p_value_ANN_baseline:.5f}")
print(f"LR vs Baseline: t-statistic = {t_stat_LR_baseline:.5f}, p-value = {p_value_LR_baseline:.5f}")
