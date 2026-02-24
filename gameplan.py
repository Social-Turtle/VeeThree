"""
This is a project to develop a neuromorphic convolutional neural network that uses ordered spikes to transmit information rather than analog signals.
This prototype will be developed to operate on the MNIST dataset, which is composed of images of handwritten numbers 0-9 in 28x28 resolution in grayscale.

I don't have a complete pipeline yet. Do not try to create one.
Here's the steps that I have so far:
1) Convolve a series of *N* 3x3 matrices (filters) with each of the 26x26 possible starting positions. (Just implement a function that does this for an input matrix [[a,b,c][d,e,f][g,h,i]] 
which correspond to top: left middle right, middle: left middle right, etc., convolving them with each position and returning a normalized value 0-1 in a 26x26 list of lists).
2) Once I've run *N* filters, I'll select the filter categories with the highest *X* values and "pass those along" to a corresponding layer map. Any pixel coordinate in the layer map that 
doesn't get a value passed to it is just treated as a zero.
3) Place a .jpg of each activation map in ./maps_layer_1 for visualization
4) Max pool each layer map for a 2x2 region.
5) Begin at step 1 again, but with a new set of filters for the next layer.

Implement these steps in simple, concise, modular code. Each step should be done in one or two major functions, with 2 or fewer helper functions.
"""