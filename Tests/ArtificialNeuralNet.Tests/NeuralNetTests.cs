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
        /// Validates that the first layer has an input synapse for each neuron.
        /// </summary>
        [TestMethod]
        public void Constructor_EachNeuronInInputLayerHasSingleInput()
        {
            NeuralNet net = new NeuralNet(neuronsPerLayer: new[] { 3, 2, 4 });

            foreach (Neuron neuron in net.Layers.First().Neurons)
            {
                Assert.AreEqual(1, neuron.Inputs.Count);
                Assert.IsNotNull(neuron.Inputs[0]);
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

            // Each neuron in the input layer should have one input, and three outputs.
            Assert.AreEqual(1, inputLayerNeurons[0].Inputs.Count);
            Assert.AreEqual(3, inputLayerNeurons[0].Outputs.Count);
            Assert.AreEqual(1, inputLayerNeurons[1].Inputs.Count);
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
            // Setup a net with two layers of one neuron each.
            NeuralNet net = new NeuralNet(new[] { 1, 1 });

            // Reset the biases in all nodes so we can disregard them in the calculations.
            net.Layers[0].Neurons[0].Bias = net.Layers[1].Neurons[0].Bias = 0;

            // Set the input weights for all nodes, so we can verify the calculations
            net.Layers[0].Neurons[0].Inputs[0].Weight = 0.5;
            net.Layers[1].Neurons[0].Inputs[0].Weight = 0.25;

            // Execute the code to test.
            IEnumerable<double> output = net.Think(inputs: new[] { 0.3 });

            // If the input is 0.3, and no nodes have biases,
            double expectedOutputOfFirstNode = 1 / (1 + Math.Pow(Math.E, -(0.5 * 0.3)));
            double expectedOutputOfNet = 1 / (1 + Math.Pow(Math.E, -(0.25 * expectedOutputOfFirstNode)));
            
            // Validate that we got the correct output.
            Assert.IsNotNull(output);
            Assert.AreEqual(1, output.Count());
            Assert.AreEqual(expectedOutputOfNet, output.First());
        }
    }
}
