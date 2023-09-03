import math
import random


# Scores the results the neural network produced
def score(results):
    points = 0
    for i in results:
        # This rewards the neural network for outputing a value closer to 1, with the point increase being calculated on a normal distribution with the value 1 being the highest reward
        points += math.e**(-(i-1)**2/(2*0.2**2))

    # Divided total points by the number of tests*100 resulting in a score that is out of 100  
    return points/len(results)*100


# Changes the weights and biases randomly
def evolve(input, intensity):
    output = input
    for i in range(len(input)):
        for j in range(len(input[i])):
            if (random.randrange(0, 1) == 0):
                output[i][j] = input[i][j]+((random.random()-0.5)*intensity)

    return output


# Sigmoid function/activation function
def sigmoid(x):
    return 2 * (1 + x / (1 + abs(x))) - 2


# The neural network to process the data
def network(nodeStructure, weights, biases, n):
    # Initiate nodes with values of 0
    nodes = [[0 for j in range(nodeStructure[i])]
             for i in range(len(nodeStructure))]

    # Setting input nodes to the inputs
    for i in range(nodeStructure[0]):
        nodes[0][i] = n[i]
        
    # Run through each node layer (not including input nodes)
    # and sum all nodes before them by multiplying them by the weights
    # adding the bias, and running it through the sigmoid function
    for i in range(len(nodeStructure)-1):
        for k in range(nodeStructure[i+1]):
            for j in range(nodeStructure[i]):
                nodes[i+1][k] += nodes[i][j]*weights[i][j]+biases[i][j]
            nodes[i+1][k] = sigmoid(nodes[i+1][k])
            sigmoid(nodes[i+1][k])

    # Only returning first output node (because I'm lazy to make it responsive to multiple output nodes)
    return nodes[len(nodeStructure)-1][0]


# Tests network on random inputs, and returns the score
def test(nodeStructure, w, b, testDepth):
    output = []
    for i in range(testDepth):
        output.append(network(nodeStructure, w, b, [random.random()*4-2 for n in range(nodeStructure[0])]))
    return score(output)


def trainNetwork(nodeStructure):
    # Code for random weights and biases:
    """
    w = [[random.random()*2-1 for j in range(nodeStructure[i] * nodeStructure[i+1])]
         for i in range(len(nodeStructure)-1)]
    b = [[random.random()*2-1 for j in range(nodeStructure[i] * nodeStructure[i+1])]
         for i in range(len(nodeStructure)-1)]
    """
    
    # Preset weights and biases
    w = [[0.7800651716911686, -0.3667746060774171, 0.10970330143754885, 0.5116041794956627, 0.0025043064805411954, 0.4281079221060522, 0.9792535137847369, 0.285192261333022], [-0.016408196126518428, 0.22335074436985372, 0.5077409113144514, -0.6670107955306451, -0.33646152282309993, 0.2798896995667015, 0.6604066611679474, 0.37587451882123496, -0.9946666880748698, 0.9180747363293634, -0.7908002653268714, 0.2410122525180272, 0.46569553993633894, 0.019436727392213304, -0.95086466332089, -0.4239839794735785, -0.6613971666019572, 0.052480136307858624, 0.5760695069365457, 0.8124183268009693, 0.6562560208196281, -0.6052000640113029, -0.44171276387274205, 0.4809162555506965], [-0.28070289952549454, -0.8207594162822108, 0.41690492462395423, 0.6057217267306184, -0.04611862722768595, 0.7818601192419021]]
    b = [[0.757469871567709, -0.10699409346981487, 0.2540351243338468, 0.6632540962638814, 0.23244657617236555, 0.765267566032636, 0.3801608416907131, -0.6043671671590312], [-0.0013497359282434475, 0.8163448556308195, 0.13104308748238983, -0.31929692675355376, 0.07128298297300392, -0.413176322191873, -0.3637221021927868, -0.07836167713007999, -0.8145419983860424, 0.11942463151876127, -0.6369087905396953, -0.3878913119689225, 0.5907915205077033, 0.5744399790655335, -0.9353936788534273, 0.23409980501091432, -0.026214516957733327, 1.0035301065039655, -0.6308665176629409, -0.6768370965923759, 0.4604842278402339, -0.9675513093661673, -0.31799602088858187, -0.8272194706146727], [-0.3743961772267939, 0.7302410174119278, -0.30759104355274236, 0.2603697645858115, 0.519888500287481, -0.09415705979758095]]

    # Log weights and biases to console (needed when generating random initial weights and biases)
    print("Weights " + str(w) + "\n")
    print("biases " + str(b) + "\n")

    training = True
    testDepth = 10000
    deepTestDepth = 100000
    generations = 1
    variations = 2000
    bestGeneration = [0, None, None, None]
    while training:
        for g in range(generations):
            # To change weights and biases by higher value when it is performing poorly, and a smaller value when it is performing well
            # intensity = min(max(100/bestGeneration[0], 1), 0.0001)
            # bestGeneration[0] cannot equal zero when using this however - use a value close to zero
            
            # Fixed intensity for evaluation
            intensity = 1
            
            # Create variation of weights and biases to evaluate
            wv = [evolve(w, intensity) for i in range(variations)]
            bv = [evolve(b, intensity) for i in range(variations)]
            curGenScores = []
            for v in range(variations):
                # Test neural network
                curScore = test(nodeStructure, wv[v], bv[v], testDepth)
                if (curScore > bestGeneration[0]):
                    print("Potential best found with score of " + str(curScore))
                    # Deeper neural network test
                    curScore = test(nodeStructure, wv[v], bv[v], deepTestDepth)
                    if (curScore > bestGeneration[0]):
                        bestGeneration[0] = curScore
                        bestGeneration[1] = "Gen: " + str(g) + " Var: " + str(v) + " Score: " + str(curScore)
                        bestGeneration[2] = wv[v]
                        bestGeneration[3] = bv[v]
                        print(bestGeneration[1] + " New best score!")
                    else:
                        print("Failed deep re-test")
                        print("Gen: " + str(g) + " Var: " + str(v) + " Score: " + str(curScore))
                else:
                    print("Gen: " + str(g) + " Var: " +
                          str(v) + " Score: " + str(curScore))
                curGenScores.append(curScore)
            bestCurGen = [0, None]
            for i in range(len(curGenScores)):
                if (curGenScores[i] > bestCurGen[0]):
                    bestCurGen = [curGenScores[i], i]
            if (bestCurGen[0] >= bestGeneration[0]):
                w, b = wv[bestCurGen[1]], bv[bestCurGen[1]]

        print("Final Weights " + str(w) + "\n")
        print("Final biases " + str(b) + "\n")

        print(bestGeneration[1])
        print("w = " + str(bestGeneration[2]))
        print("b = " + str(bestGeneration[3]))
        
        training = False

# Trains network with 2 input nodes, two hidden layers with 4 and 6 nodes respectively, and one exit/output node
x = trainNetwork([2, 4, 6, 1])
