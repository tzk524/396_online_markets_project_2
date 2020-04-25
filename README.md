In this project you will implement two online algorithms and compare their regrets at both the theoretical optimal learning rate and the empirically optimal learning rate.  You will also evaluate the effectiveness of online learning for bidding in a first price auction.
Complete the following parts and prepare a project report according to the Project Report Guidelines.  Submit your project reports on canvas.  Do not include your names on your submission.  For group projects, you must join a Project Group before submitting.
Part 1: Implement an Online Learning Algorithm
Implement the exponential weights and follow the perturbed leader algorithms.  Compare the regrets of these algorithms against each other using the theoretical learning rate identified in class of 𝜀 = √(ln k / n). In addition to using the theoretical optimal learning rate, you can compare them using the empirically optimal learning rate.  How do the theoretical and empirical optimal learning rates compare?  How do the regrets of the algorithms compare?

At a minimum you should compare the algorithm's regrets on stochastic inputs with the following data generating model:

The payoff of each action j in each round is an independent draw from a Bernoulli distribution, i.e., it is 1 with a given probability pj and and 0 with the remaining probability (1-pj).
The probabilities of the k actions are fixed in advance as some (p1,...,pk).
Find another reasonable data source and compare the algorithms regret on that data source.

Part 2: Empirical Study of Online Learning for Bidding in the First-Price Auction
For the second part of the project you will evaluate online learning as an approach for learning how to bid in a first-price auction. Assume you are bidding repeatedly in a first price auction and your value is v.

One approach to learning in a first-price auction: You can discretize bid space to k bids between 0 and v, for example with the jth bid as bj = v j/k.  Action j corresponds to bidding bj.  If the competing bid is greater than bj, you lose and your payoff is 0.  If the competing bid is less than bj, then you win and your payoff v-bj.   Note that in applying online learning to bidding in an auction, you both need to optimize the learning rate and optimize the level of discretization k.  Notice that the maximum payoff in the auction is your value v (which you should plug into your learning algorithm as h).   

Adapt an online learning algorithm to optimize bids in an auction.   You can use the above approach or another approach that you invent.  Consider optimizing your bid learning algorithm to minimize regret (versus the optimal bid in hindsight, over all bids, e.g., not just the bids in your discretization).  Empirically, how does regret drop with the number n of rounds as you optimally tune both the discretization and learning rate?

Evaluate your bid learning algorithm on the data from Exercise 1.2: Place Your Bids.  The bid data from this exercise in bid_data.csvPreview the document.  To test your algorithm on larger n than there are bids in this data you can at each time i draw a random bid from the bids in this data set.  You may also evaluate your algorithm on synthetic data.
