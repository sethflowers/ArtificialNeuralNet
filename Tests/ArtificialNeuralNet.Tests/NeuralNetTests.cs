//-----------------------------------------------------------------------
// <copyright file="NeuralNetTests.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// Tests for the NeuralNet class.
    /// </summary>
    [TestClass]
    public class NeuralNetTests
    {
        /// <summary>
        /// Validates that the Layers property is initialized by the constructor.
        /// </summary>
        [TestMethod]
        public void Constructor_LayersProperty_Initialized()
        {
            Assert.IsNotNull(new NeuralNet().Layers);
        }

        /// <summary>
        /// Validates that the constructor throws a meaningful exception if the number of neurons per layer argument is null.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Constructor_NeuronsPerLayerIsNull_ThrowsMeaningfulException()
        {
            try
            {
                new NeuralNet(neuronsPerLayer: null);
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("A neural net without layers is invalid.{0}Parameter name: neuronsPerLayer", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that the constructor throws a meaningful exception if the number of neurons per layer argument is empty.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Constructor_NeuronsPerLayerIsEmpty_ThrowsMeaningfulException()
        {
            try
            {
                new NeuralNet(neuronsPerLayer: new int[0]);
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("A neural net without layers is invalid.{0}Parameter name: neuronsPerLayer", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that the constructor throws a meaningful exception if the number of neurons per layer argument contains a value that is not positive.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Constructor_NeuronsPerLayerHasNonPositiveData_ThrowsMeaningfulException()
        {
            try
            {
                new NeuralNet(neuronsPerLayer: new[] { 1, 3, 0, 2 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("A neural net cannot have a layer with no inputs.{0}Parameter name: neuronsPerLayer", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that the constructor throws a meaningful exception if the weights and biases argument is null.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Constructor_WeightsAndBiasesInitializationIsNull_ThrowsMeaningfulException()
        {
            try
            {
                new NeuralNet(
                    neuronsPerLayer: new[] { 1, 3, 1 },
                    weightsAndBiases: null);
            }
            catch (ArgumentNullException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to initialize a neural net with null initialization data.{0}Parameter name: weightsAndBiases", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that the constructor throws a meaningful exception if the weights and biases argument is less than the correct length.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Constructor_WeightsAndBiasesInitializationDataNotCorrectLength_ThrowsMeaningfulException()
        {
            try
            {
                new NeuralNet(
                    neuronsPerLayer: new[] { 1, 3, 1 },
                    weightsAndBiases: new[] { 0d });
            }
            catch (ArgumentException exception)
            {
                // There should be 0 for the first layer.
                // There should be 6 for the second layer.
                // There should be 4 for the third layer.
                Assert.AreEqual(
                    string.Format("The total number of weights and biases to initialize the neural net is not the required amount of 10.{0}Parameter name: weightsAndBiases", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that the constructor creates the correct number of layers.
        /// </summary>
        [TestMethod]
        public void Constructor_CreatesCorrectNumberOfLayers()
        {
            NeuralNet net = new NeuralNet(neuronsPerLayer: new[] { 1, 3, 2 });

            Assert.AreEqual(3, net.Layers.Count);
            Assert.IsTrue(net.Layers.All(layer => layer != null));
        }

        /// <summary>
        /// Validates that each layer has the correct number of neurons.
        /// </summary>
        [TestMethod]
        public void Constructor_EachLayerHasCorrectNumberOfNeurons()
        {
            NeuralNet net = new NeuralNet(neuronsPerLayer: new[] { 3, 2, 4 });

            Assert.AreEqual(3, net.Layers[0].Neurons.Count, "First layer");
            Assert.AreEqual(2, net.Layers[1].Neurons.Count, "Second layer");
            Assert.AreEqual(4, net.Layers[2].Neurons.Count, "Third layer");
        }

        /// <summary>
        /// Validates that the first layer has no input synapses for each neuron.
        /// This is because the neurons in the input layer are just there to provide the input to the hidden layers.
        /// Input nodes do not run their inputs through an activation function,
        /// they just forward the input to their output. 
        /// </summary>
        [TestMethod]
        public void Constructor_EachNeuronInInputLayerHasNoInputs()
        {
            NeuralNet net = new NeuralNet(neuronsPerLayer: new[] { 3, 2, 4 });

            foreach (Neuron neuron in net.Layers.First().Neurons)
            {
                Assert.AreEqual(0, neuron.Inputs.Count);
            }
        }

        /// <summary>
        /// Validates that the first layer has an input synapse for each neuron.
        /// </summary>
        [TestMethod]
        public void Constructor_EachNeuronInOutputLayerHasSingleOutput()
        {
            NeuralNet net = new NeuralNet(neuronsPerLayer: new[] { 3, 2, 4 });

            foreach (Neuron neuron in net.Layers.Last().Neurons)
            {
                Assert.AreEqual(1, neuron.Outputs.Count);
                Assert.IsNotNull(neuron.Outputs[0]);
            }
        }

        /// <summary>
        /// Validates that every neuron in each layer is connected to every neuron in the adjacent layers.
        /// To test this, we can have a simple net with 2 input neurons, a hidden layer with 3 neurons and 2 output neurons.
        /// This scenario should have 16 total synapses, 2 into the first layer, 6 from the first to the second layer,
        /// 6 from the second to the third layer, and 2 out of the third layer.
        /// </summary>
        [TestMethod]
        public void Constructor_EachNeuronInEachLayerIsConnectedToEveryNeuronInAdjacentLayers()
        {
            NeuralNet net = new NeuralNet(neuronsPerLayer: new[] { 2, 3, 2 });

            IList<Neuron> inputLayerNeurons = net.Layers[0].Neurons;
            IList<Neuron> hiddenLayerNeurons = net.Layers[1].Neurons;
            IList<Neuron> outputLayerNeurons = net.Layers[2].Neurons;

            // Each neuron in the input layer should have 0 inputs, and three outputs.
            Assert.AreEqual(0, inputLayerNeurons[0].Inputs.Count);
            Assert.AreEqual(3, inputLayerNeurons[0].Outputs.Count);
            Assert.AreEqual(0, inputLayerNeurons[1].Inputs.Count);
            Assert.AreEqual(3, inputLayerNeurons[1].Outputs.Count);

            // Each neuron in the hidden layer should have two inputs, and two outputs.
            Assert.AreEqual(2, hiddenLayerNeurons[0].Inputs.Count);
            Assert.AreEqual(2, hiddenLayerNeurons[0].Outputs.Count);
            Assert.AreEqual(2, hiddenLayerNeurons[1].Inputs.Count);
            Assert.AreEqual(2, hiddenLayerNeurons[1].Outputs.Count);

            // Each neuron in the output layer should have three inputs, and one outputs.
            Assert.AreEqual(3, outputLayerNeurons[0].Inputs.Count);
            Assert.AreEqual(1, outputLayerNeurons[0].Outputs.Count);
            Assert.AreEqual(3, outputLayerNeurons[1].Inputs.Count);
            Assert.AreEqual(1, outputLayerNeurons[1].Outputs.Count);

            // All the first nodes outputs from the first layer should be the first inputs into each neuron in the hidden layer.
            Assert.AreEqual(inputLayerNeurons[0].Outputs[0], hiddenLayerNeurons[0].Inputs[0]);
            Assert.AreEqual(inputLayerNeurons[0].Outputs[1], hiddenLayerNeurons[1].Inputs[0]);
            Assert.AreEqual(inputLayerNeurons[0].Outputs[2], hiddenLayerNeurons[2].Inputs[0]);

            // All the second nodes outputs from the first layer should be the second inputs into each neuron in the hidden layer.
            Assert.AreEqual(inputLayerNeurons[1].Outputs[0], hiddenLayerNeurons[0].Inputs[1]);
            Assert.AreEqual(inputLayerNeurons[1].Outputs[1], hiddenLayerNeurons[1].Inputs[1]);
            Assert.AreEqual(inputLayerNeurons[1].Outputs[2], hiddenLayerNeurons[2].Inputs[1]);

            // All the first outputs from each node in the hidden layer should be the inputs to the first node in the output layer.
            Assert.AreEqual(hiddenLayerNeurons[0].Outputs[0], outputLayerNeurons[0].Inputs[0]);
            Assert.AreEqual(hiddenLayerNeurons[1].Outputs[0], outputLayerNeurons[0].Inputs[1]);
            Assert.AreEqual(hiddenLayerNeurons[2].Outputs[0], outputLayerNeurons[0].Inputs[2]);

            // All the second outputs from each node in the hidden layer should be the inputs to the second node in the output layer.
            Assert.AreEqual(hiddenLayerNeurons[0].Outputs[1], outputLayerNeurons[1].Inputs[0]);
            Assert.AreEqual(hiddenLayerNeurons[1].Outputs[1], outputLayerNeurons[1].Inputs[1]);
            Assert.AreEqual(hiddenLayerNeurons[2].Outputs[1], outputLayerNeurons[1].Inputs[2]);
        }

        /// <summary>
        /// Validates that the neurons are initialized with the correct bias and weights from the constructor.
        /// </summary>
        [TestMethod]
        public void Constructor_InitializationData_UsedCorrectlyForBiasesAndWeights()
        {
            // Create the initialization data for the biases and input weights 
            // for every node in every layer but the input layer.
            double[] initializationData = new[] 
            {
                0.3, 0.4, // The bias and weight for the first node in the hidden layer.
                0.5, 0.6, // The bias and weight for the second node in the hidden layer.
                0.7, 0.8, 0.9 // The bias and weight for the single node in the output layer.
            };

            NeuralNet neuralNet = new NeuralNet(
                neuronsPerLayer: new[] { 1, 2, 1 },
                weightsAndBiases: initializationData);

            // Validate the single node in the input has no inputs.
            Assert.AreEqual(0, neuralNet.Layers[0].Neurons[0].Inputs.Count, "Layer 1 Node 1 Input 1");

            // Validate the first node in the hidden layer is initialized correctly.
            Assert.AreEqual(0.3, neuralNet.Layers[1].Neurons[0].Bias, "Layer 2 Node 1 bias");
            Assert.AreEqual(0.4, neuralNet.Layers[1].Neurons[0].Inputs[0].Weight, "Layer 2 Node 1 Input 1 weight");

            // Validate the second node in the hidden layer is initialized correctly.
            Assert.AreEqual(0.5, neuralNet.Layers[1].Neurons[1].Bias, "Layer 2 Node 2 bias");
            Assert.AreEqual(0.6, neuralNet.Layers[1].Neurons[1].Inputs[0].Weight, "Layer 2 Node 2 Input 1 weight");

            // Validate the single node in the output layer is initialized correctly.
            Assert.AreEqual(0.7, neuralNet.Layers[2].Neurons[0].Bias, "Layer 3 Node 1 bias");
            Assert.AreEqual(0.8, neuralNet.Layers[2].Neurons[0].Inputs[0].Weight, "Layer 3 Node 1 Input 1 weight");
            Assert.AreEqual(0.9, neuralNet.Layers[2].Neurons[0].Inputs[1].Weight, "Layer 3 Node 1 Input 2 weight");
        }

        /// <summary>
        /// Validates that Think throws a meaningful exception if the inputs argument is null.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Think_NullInputs_ThrowsMeaningfulException()
        {
            try
            {
                new NeuralNet(neuronsPerLayer: new[] { 1 })
                    .Think(inputs: null);
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("The number of inputs to a neural net should match the number of neurons in the input layer.{0}Parameter name: inputs", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that Think throws a meaningful exception if the length of the 
        /// inputs argument does not match the number of nodes in the input layer.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Think_InputsHasDifferentLengthThanNumberOfNodesInInputLayer_ThrowsMeaningfulException()
        {
            try
            {
                new NeuralNet(neuronsPerLayer: new[] { 1 })
                    .Think(inputs: new[] { 2d, 3 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("The number of inputs to a neural net should match the number of neurons in the input layer.{0}Parameter name: inputs", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// In order to test that the net returns the correct output, we have to set up an extremely simple net.
        /// Our net will have 1 input node in an input layer, and 1 output node in an output layer, with no hidden layers.
        /// </summary>
        [TestMethod]
        public void Think_ReturnsTheCorrectOutput()
        {
            // Create the initialization data.
            // This data represents the biases and weights for every input synapse 
            // for every node in every layer but the first.
            // This means we need one bias and one weight for the only node in the second layer.
            double[] initializationData = new[] { 0.5, 0.25 };

            // Setup a net with two layers of one neuron each.
            NeuralNet net = new NeuralNet(new[] { 1, 1 }, initializationData);

            // Execute the code to test.
            IEnumerable<double> output = net.Think(inputs: new[] { 0.3 });

            // If the input is 0.3, then the output of the single node in the input layer is 0.3.
            // The weight of this synapse to the output layer is 0.25, and the bias is 0.5.
            // The sum of the inputs to the output node is 0.25 * 0.3 is .075.
            // Add this number to the bias to get .575.
            // Run this number through the sigmoid activation function.
            double expectedOutputOfNet = 1 / (1 + Math.Pow(Math.E, -0.575));

            // Validate that we got the correct output.
            Assert.IsNotNull(output);
            Assert.AreEqual(1, output.Count());
            Assert.AreEqual(expectedOutputOfNet, output.First());
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the inputs argument is null.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ThinkFast_InputsIsNull_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: null,
                    neuronsPerLayerAfterInputLayer: new[] { 1 },
                    biases: new[] { 0.1 },
                    weights: new[] { 0.2 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to run a neural net without any inputs.{0}Parameter name: inputs", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the inputs argument is empty.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ThinkFast_InputsIsEmpy_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new double[0],
                    neuronsPerLayerAfterInputLayer: new[] { 1 },
                    biases: new[] { 0.1 },
                    weights: new[] { 0.2 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to run a neural net without any inputs.{0}Parameter name: inputs", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the neurons per layer after the input layer argument is null.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ThinkFast_NeuronsPerLayerAfterInputLayerIsNull_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new[] { 0.5 },
                    neuronsPerLayerAfterInputLayer: null,
                    biases: new[] { 0.1 },
                    weights: new[] { 0.2 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to run a neural net without knowing how many neurons to create in each layer after the input layer.{0}Parameter name: neuronsPerLayerAfterInputLayer", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the neurons per layer after the input layer argument is empty.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ThinkFast_NeuronsPerLayerAfterInputLayerIsEmpty_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new[] { 0.5 },
                    neuronsPerLayerAfterInputLayer: new int[0],
                    biases: new[] { 0.1 },
                    weights: new[] { 0.2 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to run a neural net without knowing how many neurons to create in each layer after the input layer.{0}Parameter name: neuronsPerLayerAfterInputLayer", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the 
        /// neurons per layer after the input layer argument has non-positive values in it.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ThinkFast_NeuronsPerLayerAfterInputLayerHasNonPositiveValues_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new[] { 0.5 },
                    neuronsPerLayerAfterInputLayer: new[] { -4 },
                    biases: new[] { 0.1 },
                    weights: new[] { 0.2 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to create a neural net with a layer with a non-positive number of neurons.{0}Parameter name: neuronsPerLayerAfterInputLayer", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the biases argument is null.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ThinkFast_BiasesIsNull_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new[] { 0.5 },
                    neuronsPerLayerAfterInputLayer: new[] { 1 },
                    biases: null,
                    weights: new[] { 0.2 });
            }
            catch (ArgumentNullException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to run a neural net without any biases.{0}Parameter name: biases", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the biases argument is not the correct length.
        /// There should be as many biases in the array as there are neurons in all the layers after the input layer.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ThinkFast_BiasesIsNotCorrectLength_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new[] { 0.5 },
                    neuronsPerLayerAfterInputLayer: new[] { 1 },
                    biases: new[] { 0.1, 0.3 },
                    weights: new[] { 0.2 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("The total number of biases should be 1, but was 2.{0}Parameter name: biases", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the weights argument is null.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ThinkFast_WeightsIsNull_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new[] { 0.5 },
                    neuronsPerLayerAfterInputLayer: new[] { 1 },
                    biases: new[] { 0.1 },
                    weights: null);
            }
            catch (ArgumentNullException exception)
            {
                Assert.AreEqual(
                    string.Format("Unable to run a neural net without any weights.{0}Parameter name: weights", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that ThinkFast throws a meaningful exception if the biases argument is not the correct length.
        /// There should be as many weights in the array as there sum of all the nodes in each layer times all the nodes in the previous layer.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ThinkFast_WeightsIsNotCorrectLength_ThrowsMeaningfulException()
        {
            try
            {
                NeuralNet.ThinkFast(
                    inputs: new[] { 0.5 },
                    neuronsPerLayerAfterInputLayer: new[] { 1 },
                    biases: new[] { 0.1 },
                    weights: new[] { 0.2, 0.4, .03 });
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("The total number of weights should be 1, but was 3.{0}Parameter name: weights", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that the ThinkFast method returns the correct outputs for a simple neural network
        /// consisting of a single input and a single output, with no hidden nodes.
        /// </summary>
        [TestMethod]
        public void ThinkFast_SingleInputNoHiddenNodesSingleOutput_ReturnsCorrectOutputs()
        {
            // Execute the code to test.
            double[] output = NeuralNet.ThinkFast(
                inputs: new[] { 0.3 },
                neuronsPerLayerAfterInputLayer: new[] { 1 },
                biases: new[] { 0.5 },
                weights: new[] { 0.25 });

            // If the input is 0.3, then the output of the single node in the input layer is 0.3.
            // The weight of this synapse to the output layer is 0.25, and the bias is 0.5.
            // The sum of the inputs to the output node is 0.25 * 0.3 is .075.
            // Add this number to the bias to get .575.
            // Run this number through the sigmoid activation function.
            double expectedOutputOfNet = 1 / (1 + Math.Pow(Math.E, -0.575));

            // Validate that we got the correct output.
            Assert.IsNotNull(output);
            Assert.AreEqual(1, output.Length);
            Assert.AreEqual(expectedOutputOfNet, output[0]);
        }

        /// <summary>
        /// Validates that the ThinkFast method returns the correct outputs for another simple neural network
        /// consisting of two inputs, two hidden nodes in a hidden layer, and two outputs.
        /// </summary>
        [TestMethod]
        public void ThinkFast_TwoInputsTwoHiddenNodesTwoOutputs_ReturnsCorrectOutputs()
        {
            // Initialize the inputs for our tests.
            double[] inputs = new[] { 1.0, 2.0 };

            // Initialize the biases for our tests.
            double[] biases = new[] 
            {
                0.9, // This is the bias for the first node in the hidden layer. 
                0.8, // This is the bias for the second node in the hidden layer.
                0.7, // This is the bias for the second node in the output layer.
                0.6  // This is the bias for the second node in the output layer.
            };

            // Initialize the weights for our tests.
            double[] weights = new[] 
            { 
                0.02, 0.04, // This is the first and second weights for the inputs to the first node in the hidden layer. 
                0.06, 0.08, // This is the first and second weights for the inputs to the second node in the hidden layer.
                0.01, 0.03, // This is the first and second weights for the inputs to the first node in the output layer.
                0.05, 0.01  // This is the first and second weights for the inputs to the second node in the output layer.
            };

            // Execute the code to test.
            double[] output = NeuralNet.ThinkFast(
                inputs: inputs,
                neuronsPerLayerAfterInputLayer: new[] { 2, 2 },
                biases: biases,
                weights: weights);

            Func<double, double, double, double, double, double> activationFunction = 
                (firstInputWeight, firstInput, secondInputWeight, secondInput, bias) =>
                    1 / (1 + Math.Pow(Math.E, -((firstInputWeight * firstInput) + (secondInputWeight * secondInput) + bias)));

            double outputOfFirstNodeInHiddenLayer = activationFunction(weights[0], inputs[0], weights[1], inputs[1], biases[0]);
            double outputOfSecondNodeInHiddenLayer = activationFunction(weights[2], inputs[0], weights[3], inputs[1], biases[1]);
            double firstOutputOfNet = activationFunction(weights[4], outputOfFirstNodeInHiddenLayer, weights[5], outputOfSecondNodeInHiddenLayer, biases[2]);
            double secondOutputOfNet = activationFunction(weights[6], outputOfFirstNodeInHiddenLayer, weights[7], outputOfSecondNodeInHiddenLayer, biases[3]);

            // Validate that we got the correct output.
            Assert.IsNotNull(output);
            Assert.AreEqual(2, output.Length);
            Assert.AreEqual(firstOutputOfNet, output[0]);
            Assert.AreEqual(secondOutputOfNet, output[1]);
        }
    }
}
