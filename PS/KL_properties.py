import math

probability = [[0.5, 0.5], [0.25, 0.75], [0.125, 0.875]]
events = ['A', 'B', 'C']

def KL_divergence(prob1, prob2):
    divergence = 0
    for p_x, q_x in zip(prob1, prob2):
        divergence += p_x * math.log(p_x / q_x, 2)
    return divergence

for event1, prob1 in zip(events, probability):
    for event2, prob2 in zip(events, probability):
        divergence = KL_divergence(prob1, prob2)
        print("KL({} || {}) = {:.3f}".format(event1, event2, divergence))
