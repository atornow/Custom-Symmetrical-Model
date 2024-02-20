<h1>Symmetrical Number Identifier</h1>
The Symmetrical Number Identifier is a custom-built machine learning model designed to identify symmetrical numbers. This project showcases a handcrafted model, including custom functions for the forward and backward pass, and everything necessary for backpropagation. The model is set up with Matplotlib to optimize hyperparameters.

<h2>Project Structure</h2>
<ul>
<li><code>ModelFunctions.py</code>: Contains the core logic for the model, including data generation, the training loop, forward and backward pass implementations, and error plotting.</li>
<li>Main run file: Orchestrates the model training and validation processes.</li>
</ul>
<h2>Features</h2>
<ul>
<li>Custom model implementation from scratch.</li>
<li>Hyperparameter optimization using Matplotlib.</li>
<li>Support for generating training and validation datasets.</li>
<li>Implementation of forward and backward passes manually.</li>
<li>Utility for loading or generating model weights.</li>
</ul>
<h2>Setup</h2>
To run this project, you need Python 3 and the following libraries:

<ul>
<li>numpy</li>
<li>matplotlib</li>
</ul>
Install the dependencies using pip:

<pre><code>pip install numpy matplotlib</code></pre>
<h2>Usage</h2>
To start the model training and validation process, run the main Python file:

<pre><code>python main.py</code></pre>
Ensure you have the <code>ModelFunctions.py</code> file in the same directory as your main script.

<h2>Hyperparameters</h2>
The model's behavior can be adjusted using the following hyperparameters:

<ul>
<li><code>modelSize</code>: Size of the model.</li>
<li><code>learningRate</code>: Starting learning rate, which is incremented by 0.05 for each run in the series.</li>
<li><code>momentum</code>: Momentum for the update rule.</li>
<li><code>epochs</code>: Number of training epochs.</li>
<li><code>fullRunCount</code>: Number of full training runs with incremented learning rates.</li>
<li><code>genData</code>, <code>genValid</code>, <code>genWeights</code>: Booleans to control the generation of training/validation data and model weights.</li>
<li><code>train</code>: Boolean to toggle the training phase.</li>
</ul>
<h2>Contributing</h2>
Feel free to fork this project, submit pull requests, or report issues. Your contributions are highly appreciated!
