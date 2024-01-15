#%%
import numpy as np
from scipy.stats import beta, gamma
import matplotlib.pyplot as plt

def metropolis_hastings(num_iterations, dist_1, dist_2):

    # Initialize parameter values using mean
    current_1 = np.mean(dist_1.rvs(size=100))
    current_2 = np.mean(dist_2.rvs(size=100))

    # Initialize accepted samples
    posterior_value = []

    for _ in range(num_iterations):
        
        # Generate proposal values
        # for the value of the beta distribution
        proposal_1 = dist_1.rvs()
        # for the value of the gamma distribution
        proposal_2 = dist_2.rvs()

        # Calculate the joint distribution of the current and proposed parameters
        current_joint_dist = dist_1.pdf(current_1) * dist_2.pdf(current_2)
        proposal_joint_dist = dist_1.pdf(proposal_1) * dist_2.pdf(proposal_2)

        # Calculate the acceptance ratio
        acceptance_ratio = proposal_joint_dist / current_joint_dist

        # Accept or reject the proposal
        if np.random.uniform(0,1) < acceptance_ratio:
            current_1 = proposal_1
            current_2 = proposal_2
            #accepted_samples.append((current_p, current_mu))
            posterior_value.append(current_1*current_2)

    return posterior_value

class ProdDist:

    def __init__(self, dist_1, dist_2, num_iterations=5000) -> None:

        if callable(dist_1.pdf) & callable(dist_1.rvs) & callable(dist_2.pdf) & callable(dist_2.rvs):
            self.dist_1 = dist_1
            self.dist_2 = dist_2
        else:
            raise ValueError("The given distribution class must have a rvs and pdf method.")
        
        self.num_iterations = num_iterations
        self.post_prod = None
    
    def make_post(self):
        post = metropolis_hastings(self.num_iterations, self.dist_1, self.dist_2)
        self.post_prod = post
        return post

    def check_post(self):

        if self.post_prod is None:
            raise ValueError("First perform estimation of the posetrior distribution using the make_post method")

        samples = self.post_prod
        N_samples = len(samples)
        burnin = N_samples//10

        plt.title("Histogramm of the posterior distribution")
        plt.hist(samples[burnin:], bins=20, density=True)
        #plt.hist(beta.rvs(a, b, size=1000), density=True)
        #plt.hist(gamma.rvs(aa, scale=1/bb, size=1000), density=True)
        plt.show()

        plt.title("Samples")
        plt.plot(samples[burnin:])
        plt.show()

        plt.title("Cumulative mean of the samples")
        plt.plot(range(burnin, N_samples), np.cumsum(samples[burnin:])/range(1, N_samples-burnin+1))
        plt.show()


if __name__ == "__main__":

    # x = p*mu with p~beta(a, b) and mu~gamma(aa, 1/bb)
    a = 100
    b = 20
    aa = 4.0
    bb = 0.5

    num_iterations = 5000
    burnin = 1000
    post_class = ProdDist(beta(a=a, b=b), gamma(a=aa, scale=1/bb))

    post_class.make_post()
    post_class.check_post()
