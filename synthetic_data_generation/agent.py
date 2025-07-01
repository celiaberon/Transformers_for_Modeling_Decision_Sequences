import numpy as np


class RFLR_mouse:
    """Recursively formulated logistic regression agent for two-armed bandit task"""
    def __init__(self, alpha=0.5, beta=2, tau=1.2, policy="probability_matching"):
        """
        Initialize the agent with the provided parameters.

        Args:
            alpha (float): Weight for the most recent choice
            beta (float): Weight for the reward-choice interaction
            tau (float): Time constant for decaying influence of past choices/rewards
            policy (str): Strategy for action selection ('probability_matching' or 'greedy_policy')
        """
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.phi_t = 0.5
        self.last_choice = np.random.choice([0, 1])
        self.policy = policy

    def update_phi(self, c_t, r_t):
        """
        Update the recursive reward term (phi_t) based on the current choice and reward.

        Args:
            c_t (int): Current choice (0 for left, 1 for right)
            r_t (int): Current reward (1 for rewarded, 0 for no reward)
        """
        self.phi_t = self.beta * (2 * c_t - 1) * r_t + np.exp(-1 / self.tau) * self.phi_t

    def compute_log_odds(self, c_t):
        """
        Compute the log-odds for the next choice based on the recursive formula.

        Args:
            c_t (int): Current choice (0 for left, 1 for right)

        Returns:
            float: Log-odds of selecting the right choice on the next trial
        """
        log_odds = self.alpha * (2 * c_t - 1) + self.phi_t
        return log_odds

    def sigmoid(self, log_odds):
        """Convert log-odds to probability (P(choice==right))"""
        return 1 / (1 + np.exp(-log_odds))

    def make_choice(self):
        """Make choice using policy on current model estimate"""
        log_odds = self.compute_log_odds(self.last_choice)
        prob_right = self.sigmoid(log_odds)

        if self.policy == "probability_matching":
            # Make a choice with probability matching the calculated value
            choice = np.random.choice([0, 1], p=[1 - prob_right, prob_right])
        elif self.policy == "greedy_policy":
            # Always select the higher-value option
            choice = 1 if prob_right > 0.5 else 0
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        self.last_choice = choice
        return choice
    
    def passive_estimator(self, choices, rewards):
        """
        Passive estimator for a sequence of trials.
        Receives lists of choices and rewards for a sequence of trials and returns
        the phi, log odds, and probabilities for the entire sequence.
        
        Args:
            choices (array): Array of choices for the sequence of trials
            rewards (array): Array of rewards for the sequence of trials
            
        Returns:
            tuple: (phi_list, logodds_list, pright_list)
                phi_list (list): Phi values for each trial
                logodds_list (list): Log odds for each trial
                pright_list (list): Probability of choosing right for each trial
        """
        phis = np.zeros(len(choices))
        logodds = np.zeros(len(choices))
        prights = np.zeros(len(choices))
        
        # Set initial values for first trial
        phis[0] = self.phi_t
        logodds[0] = 0
        prights[0] = 0.5
        
        for t, (prev_c_t, prev_r_t) in enumerate(zip(choices[:-1], rewards[:-1]), start=1):
            # Update agent state with previous choice/reward
            self.update_phi(prev_c_t, prev_r_t)
            self.last_choice = prev_c_t
            
            # Compute the values that would be used for the current choice
            phis[t] = self.phi_t
            logodds[t] = self.compute_log_odds(prev_c_t)
            prights[t] = self.sigmoid(logodds[t])
        
        return phis, logodds, prights

    def get_qlearning_params(self):
        """
        Get the Q-learning parameters for the agent.
        """
        self.q_k = 1 - np.exp(-1 / self.tau)
        self.q_T = self.q_k / self.beta
        self.q_alpha = self.alpha

    def reset_qlearning_params(self):
        """
        Reset the Q-learning parameters for the agent.
        """
        self.q = np.array((0.5, 0.5))
        self.v = 0

    def update_q(self, c_t, r_t):
        """
        Update the Q-learning parameters for the agent.
        """
        self.q[c_t] = self.q[c_t] + (self.q_k * (r_t - self.q[c_t]))
        self.q[1-c_t] = (1 - self.q_k) * self.q[1-c_t]

    def compute_logodds_qlearning(self, c_t):
        """
        Compute the log-odds for the next choice based on the Q-learning parameters.
        """
        log_odds = (np.diff(self.q) / self.q_T) + (self.q_alpha * (2 * c_t - 1))
        return log_odds
    
    def apply_softmax_qlearning(self, c_t):
        """
        Apply the softmax function to the Q-learning parameters.
        """
        prob_right = 1 / (1 + np.exp(-((np.diff(self.q) / self.q_T) + (self.q_alpha * (2 * c_t - 1)))))
        return prob_right

    def passive_estimator_qlearning(self, choices, rewards):
        """
        Passive Q-learning estimator for a sequence of trials.
        Receives lists of choices and rewards for a sequence of trials and returns
        the RPEs, values, and Q-values for the entire sequence.
        
        Args:
            choices (array): Array of choices for the sequence of trials
            rewards (array): Array of rewards for the sequence of trials
            new_session (bool): Whether to reset Q-learning parameters
            
        Returns:
            tuple: (rpe_list, value_list, q_left_list, q_right_list)
                rpe_list (list): Reward prediction errors for each trial
                value_list (list): Value estimates for each trial
                q_left_list (list): Q-values for left action for each trial 
                q_right_list (list): Q-values for right action for each trial
        """
        self.get_qlearning_params()
        self.reset_qlearning_params()

        rpes = np.zeros(len(choices))
        qs = np.zeros((len(choices), 2))
        logodds = np.zeros(len(choices))
        pright = np.zeros(len(choices))

        # Set initial values for first trial
        rpes[0] = 0  # No RPE for first trial
        qs[0, 0] = self.q[0]
        qs[0, 1] = self.q[1]
        logodds[0] = 0
        pright[0] = 0.5

        for t, (prev_c_t, prev_r_t) in enumerate(zip(choices[:-1], rewards[:-1]), start=1):

            # Update Q-values and value
            self.update_q(prev_c_t, prev_r_t)
            
            # Store values for current trial (after update)
            rpes[t] = prev_r_t - self.q[prev_c_t]
            qs[t, 0] = self.q[0]  # Left Q-value
            qs[t, 1] = self.q[1]  # Right Q-value
            
            # Compute log odds using the choice that would be used for current decision
            log_odds = self.compute_logodds_qlearning(prev_c_t)  # Use previous choice
            prob_right = self.apply_softmax_qlearning(prev_c_t)
            logodds[t] = log_odds
            pright[t] = prob_right

        return rpes, qs, logodds, pright

def main():
    agent = RFLR_mouse(alpha=0.75, beta=2.1, tau=1.4)

main()