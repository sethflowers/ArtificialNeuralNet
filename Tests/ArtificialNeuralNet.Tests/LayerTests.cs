//-----------------------------------------------------------------------
// <copyright file="LayerTests.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet.Tests
{
    using System;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// Tests for the Layer class.
    /// </summary>
    [TestClass]
    public class LayerTests
    {
        /// <summary>
        /// Validates that the Neurons property is initialized by the constructor.
        /// </summary>
        [TestMethod]
        public void Constructor_NeuronsProperty_Initialized()
        {
            Assert.IsNotNull(new Layer().Neurons);
        }
        
        /// <summary>
        /// Validates that the constructor throws a meaningful exception if the number of neurons is less than one.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Constructor_ZeroNeuronsPerLayer_ThrowsMeaningfulException()
        {
            try
            {
                new Layer(numberOfNeurons: 0);
            }
            catch (ArgumentException exception)
            {
                Assert.AreEqual(
                    string.Format("A neural layer cannot have a negative number of neurons.{0}Parameter name: numberOfNeurons", Environment.NewLine),
                    exception.Message);

                throw;
            }
        }

        /// <summary>
        /// Validates that the correct number of neurons are created in the layer.
        /// </summary>
        [TestMethod]
        public void Constructor_CreatesCorrectNumberOfNeurons()
        {
            Layer layer = new Layer(numberOfNeurons: 5);

            Assert.AreEqual(5, layer.Neurons.Count);
            Assert.IsTrue(layer.Neurons.All(n => n != null));
        }

        /// <summary>
        /// Validates that the Think method just forwards the call to each neuron in the layer.
        /// This is a little difficult since each neuron sends its inputs through a sigmoid function,
        /// which we pretty much have to duplicate to assert against.
        /// </summary>
        [TestMethod]
        public void Think_CausesEachNeuronInLayerToThink()
        {
            Layer layer = new Layer(numberOfNeurons: 2);

            // Give each neuron one input and one output.
            layer.Neurons[0].Inputs.Add(new Synapse { Weight = 1, Value = 4 });
            layer.Neurons[1].Inputs.Add(new Synapse { Weight = 2, Value = 3 });
            layer.Neurons[0].Outputs.Add(new Synapse());
            layer.Neurons[1].Outputs.Add(new Synapse());

            // Reset the biases since we know they are randomized, and we want to disregard them.
            layer.Neurons[0].Bias = layer.Neurons[1].Bias = 0;

            // Execute the code to test.
            layer.Think();

            // Validate that each neuron sets the value for its output.
            // -4 is the sum of the neurons bias and all its inputs weights times their value.
            Assert.AreEqual(
                1 / (1 + Math.Pow(Math.E, -4)), 
                layer.Neurons[0].Outputs[0].Value, 
                "First neuron output");

            // -6 is the sum of the neurons bias and all its inputs weights times their value.
            Assert.AreEqual(
                1 / (1 + Math.Pow(Math.E, -6)), 
                layer.Neurons[1].Outputs[0].Value, 
                "Second neuron output");
        }
    }
}
