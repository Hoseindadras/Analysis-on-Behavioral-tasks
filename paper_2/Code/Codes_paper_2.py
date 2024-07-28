# -*- coding: utf-8 -*-
import scipy.io
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

mat_file = scipy.io.loadmat('/content/T.mat')

T_variable = mat_file['T']
T_details = T_variable[0, 0]

participants = T_details['participant']
sessions = T_details['session']
choices = T_details['chosen']
sure_rewards = T_details['sure_gain']
potential_rewards = T_details['risk_gain']
probabilities = T_details['risk_prob']

data_frame = pd.DataFrame({
    'participant': participants.flatten(),
    'session': sessions.flatten(),
    'sure_gain': sure_rewards.flatten(),
    'risk_gain': potential_rewards.flatten(),
    'risk_prob': probabilities.flatten(),
    'chosen': choices.flatten()
})

filtered_sessions = data_frame[data_frame['session'].isin([1, 3, 5])]

filtered_sessions['expected_value'] = filtered_sessions['sure_gain'] * (1 - filtered_sessions['risk_prob']) + filtered_sessions['risk_gain'] * filtered_sessions['risk_prob']
filtered_sessions['variance'] = filtered_sessions['risk_prob'] * (1 - filtered_sessions['risk_prob']) * (filtered_sessions['risk_gain'] - filtered_sessions['expected_value'])**2

filtered_sessions.head()

def utility_function_1(exp_val, var, alpha):
    return exp_val + alpha * var

X_values = filtered_sessions[['expected_value', 'variance']]
y_values = filtered_sessions['chosen']

variance_adjusted = X_values['variance'].values.reshape(-1, 1)

linear_model = LinearRegression()
linear_model.fit(variance_adjusted, y_values)

alpha_value = linear_model.coef_[0]

def complex_utility(risk, probability, rho):
    return probability * (risk ** rho)

risk_values = filtered_sessions['risk_gain'].values
probability_values = filtered_sessions['risk_prob'].values

params, covariance = curve_fit(lambda risk, rho: complex_utility(risk, probability_values, rho), risk_values, y_values, maxfev=10000)

rho_estimated = params[0]

(alpha_value, rho_estimated)

probabilities = filtered_sessions['risk_prob'].values
rewards = filtered_sessions['risk_gain'].values
sure_bets = filtered_sessions['sure_gain'].values
decisions = filtered_sessions['chosen'].values

expected_values = probabilities * rewards
variances = rewards**2 * probabilities * (1 - probabilities)

def logistic_prob(params, E, V, sure_values):
    alpha, beta = params
    utility_gamble = E + alpha * V
    utility_sure = sure_values
    return 1 / (1 + np.exp(-beta * (utility_gamble - utility_sure)))

def negative_log_likelihood(params, E, V, sure_values, choices):
    probs_gamble = logistic_prob(params, E, V, sure_values)
    likelihood = choices * np.log(probs_gamble) + (1 - choices) * np.log(1 - probs_gamble)
    return -np.sum(likelihood)

initial_guess = [0.1, 0.1]

optimization_result = minimize(negative_log_likelihood, initial_guess, args=(expected_values, variances, sure_bets, decisions), method='Nelder-Mead')

estimated_alpha, estimated_beta = optimization_result.x

(estimated_alpha, estimated_beta)

def exponential_logistic(params, probabilities, rewards, sure_outcomes):
    rho, beta = params
    utility_from_gamble = probabilities * (rewards ** rho)
    utility_from_sure = sure_outcomes
    odds_ratio = -beta * (utility_from_gamble - utility_from_sure)
    return 1 / (1 + np.exp(odds_ratio))

def exponential_neg_log_likelihood(params, probabilities, rewards, sure_outcomes, decisions):
    probabilities_of_gamble = exponential_logistic(params, probabilities, rewards, sure_outcomes)
    likelihood = decisions * np.log(probabilities_of_gamble) + (1 - decisions) * np.log(1 - probabilities_of_gamble)
    return -np.sum(likelihood)

initial_parameters = [1.0, 0.1]

optimization_result = minimize(exponential_neg_log_likelihood, initial_parameters, args=(probabilities, rewards, sure_bets, decisions), method='Nelder-Mead')

estimated_rho, estimated_beta = optimization_result.x

(estimated_rho, estimated_beta)

average_choice_probability = filtered_sessions['chosen'].mean()

risk_neutral_probability = 0.5

model_free_risk_preference = average_choice_probability - risk_neutral_probability

model_free_risk_preference
